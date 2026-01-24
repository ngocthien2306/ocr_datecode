"""
Shared TensorRT Engine for SuperPoint+LightGlue inference.

This module provides:
- SuperPointEngineTRT: Singleton engine that handles TensorRT inference
- TemplateConfig: Lightweight template data container

Usage:
    # Get shared engine (singleton)
    engine = SuperPointEngineTRT.get_instance(engine_path)

    # Create lightweight template configs
    template1 = TemplateConfig.from_json(json_path1)
    template2 = TemplateConfig.from_json(json_path2)

    # Run batch inference
    results = engine.match_batch(
        target_imgs=[img1, img2],
        templates=[template1, template2]
    )
"""

import os
import cv2
import json
import numpy as np
import time
import threading
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TemplateConfig:
    """
    Lightweight container for template data.
    Does NOT load TensorRT engine - only stores template info.
    """
    template_path: str
    template_img: np.ndarray
    template_gray: np.ndarray
    template_bbox: Dict[str, Any]
    other_bboxes: List[Dict[str, Any]]
    scale: float = 1.0
    crop_area: Optional[Dict[str, int]] = None

    # Original annotations for reference
    annotations: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_json(cls, json_path: str, scale: float = 1.0) -> 'TemplateConfig':
        """
        Create TemplateConfig from annotation JSON file.

        Args:
            json_path: Path to annotation JSON file
            scale: Scale factor for template image

        Returns:
            TemplateConfig instance
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        template_path = data['_template_image']
        annotations = data[template_path]

        # Load template image
        template_img = cv2.imread(template_path)
        if scale != 1.0:
            template_img = cv2.resize(template_img, None, fx=scale, fy=scale)

        template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

        # Parse bboxes
        template_bbox = None
        other_bboxes = []

        for ann in annotations:
            if ann['type'] == 'template':
                template_bbox = ann
            elif ann['type'] not in ['crop_area']:
                other_bboxes.append(ann)

        if template_bbox is None:
            raise ValueError(f"No template bbox found in {json_path}")

        return cls(
            template_path=template_path,
            template_img=template_img,
            template_gray=template_gray,
            template_bbox=template_bbox,
            other_bboxes=other_bboxes,
            scale=scale,
            annotations=annotations
        )

    @classmethod
    def from_matcher(cls, matcher: 'SuperPointMatcherTRT') -> 'TemplateConfig':
        """
        Create TemplateConfig from existing SuperPointMatcherTRT instance.
        Useful for migration from old code.

        Args:
            matcher: Existing matcher instance

        Returns:
            TemplateConfig instance
        """
        return cls(
            template_path=matcher.template_path,
            template_img=matcher.template_img,
            template_gray=matcher.template_gray,
            template_bbox=matcher.template_bbox,
            other_bboxes=matcher.other_bboxes,
            scale=matcher.scale,
            crop_area=getattr(matcher, 'crop_area', None),
            annotations=matcher.annotations
        )


class SuperPointEngineTRT:
    """
    Singleton TensorRT engine for SuperPoint+LightGlue inference.

    Handles:
    - TensorRT engine loading (once)
    - CUDA memory allocation (once)
    - Batch inference for multiple template-target pairs

    Usage:
        engine = SuperPointEngineTRT.get_instance(engine_path)
        results = engine.match_batch(target_imgs, templates)
    """

    _instance: Optional['SuperPointEngineTRT'] = None
    _lock = threading.Lock()

    def __init__(self, engine_path: str, verbose: bool = False):
        """
        Initialize TensorRT engine.

        NOTE: Use get_instance() instead of direct instantiation.
        """
        self.engine_path = engine_path
        self.verbose = verbose

        t_start = time.time()

        # Load TensorRT engine
        t0 = time.time()
        self._load_engine(engine_path)
        if verbose:
            logger.info(f"⏱️  Load engine: {(time.time()-t0)*1000:.1f}ms")

        # Warm-up
        if verbose:
            logger.info("🔥 Warming up engine...")
            t0 = time.time()

        h, w = self.input_shape[2:]
        dummy_input = np.random.rand(2, 1, h, w).astype(np.float32)
        _ = self._infer(dummy_input)

        if verbose:
            logger.info(f"   Warm-up done: {(time.time()-t0)*1000:.1f}ms")

        logger.info(f"✅ SuperPointEngineTRT initialized ({(time.time()-t_start)*1000:.1f}ms)")
        logger.info(f"   Engine: {Path(engine_path).name}")
        logger.info(f"   Input shape: {self.input_shape}")

    @classmethod
    def get_instance(cls, engine_path: str, verbose: bool = False) -> 'SuperPointEngineTRT':
        """
        Get singleton instance of the engine.

        Args:
            engine_path: Path to TensorRT engine file
            verbose: Enable verbose logging

        Returns:
            Shared SuperPointEngineTRT instance
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(engine_path, verbose)
            elif cls._instance.engine_path != engine_path:
                # Different engine path - reinitialize
                logger.warning(f"Reinitializing engine with new path: {engine_path}")
                cls._instance = cls(engine_path, verbose)

            return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (for testing or cleanup)"""
        with cls._lock:
            cls._instance = None

    def _load_engine(self, engine_path: str):
        """Load TensorRT engine and allocate buffers"""
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        # Load engine
        with open(engine_path, 'rb') as f:
            engine_data = f.read()

        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        # Create stream
        self.stream = cuda.Stream()

        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            shape = self.engine.get_tensor_shape(tensor_name)

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                # Handle dynamic shapes (-1 in shape)
                if -1 in shape:
                    max_shape = list(shape)
                    max_shape[0] = 8  # Max batch size
                    size = trt.volume(max_shape)
                else:
                    size = trt.volume(shape)

                host_mem = np.empty(size, dtype=dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)

                self.bindings.append(int(device_mem))
                self.inputs.append({
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape,
                    'name': tensor_name,
                    'dtype': dtype
                })
                self.input_shape = shape
            else:
                # Output: handle dynamic shapes (100MB buffer)
                max_size = 100 * 1024 * 1024 // np.dtype(dtype).itemsize
                device_mem = cuda.mem_alloc(max_size * np.dtype(dtype).itemsize)

                self.bindings.append(int(device_mem))
                self.outputs.append({
                    'host': None,
                    'device': device_mem,
                    'shape': shape,
                    'name': tensor_name,
                    'dtype': dtype,
                    'max_size': max_size
                })

    def _infer(self, input_data: np.ndarray) -> List[np.ndarray]:
        """Run TensorRT inference"""
        batch_size = input_data.shape[0]
        if -1 in self.input_shape or self.input_shape[0] != batch_size:
            self.context.set_input_shape(self.inputs[0]['name'], input_data.shape)

        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            input_data.ravel(),
            self.stream
        )

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(tensor_name, self.bindings[i])

        success = self.context.execute_async_v3(stream_handle=self.stream.handle)
        if not success:
            raise RuntimeError("TensorRT inference failed")

        self.stream.synchronize()

        results = []
        for output in self.outputs:
            actual_shape = self.context.get_tensor_shape(output['name'])

            if -1 in actual_shape:
                host_mem = np.empty(output['max_size'], dtype=output['dtype'])
                cuda.memcpy_dtoh_async(host_mem, output['device'], self.stream)
                self.stream.synchronize()
                results.append(host_mem)
            else:
                actual_size = trt.volume(actual_shape)
                if actual_size <= 0:
                    raise ValueError(f"Invalid output size {actual_size}")

                host_mem = np.empty(actual_size, dtype=output['dtype'])
                cuda.memcpy_dtoh_async(host_mem, output['device'], self.stream)
                self.stream.synchronize()
                results.append(host_mem.reshape(actual_shape))

        return results

    def _resize_to_engine_size(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Resize image to match engine input size"""
        h, w = img.shape[:2]
        target_h, target_w = self.input_shape[2:]

        if h != target_h or w != target_w:
            resized = cv2.resize(img, (target_w, target_h))
            return resized, (w / target_w, h / target_h)
        return img, (1.0, 1.0)

    def match_single(
        self,
        target_img: np.ndarray,
        template: TemplateConfig,
        score_threshold: float = 0.3,
        ransac_threshold: float = 5.0
    ) -> Dict:
        """
        Match a single template against target image.

        Args:
            target_img: Target image (BGR numpy array)
            template: TemplateConfig instance
            score_threshold: Matching score threshold
            ransac_threshold: RANSAC threshold

        Returns:
            Match result dict
        """
        return self.match_batch(
            target_imgs=[target_img],
            templates=[template],
            score_threshold=score_threshold,
            ransac_threshold=ransac_threshold
        )['results'][0]

    def match_batch(
        self,
        target_imgs: List[np.ndarray],
        templates: List[TemplateConfig],
        score_threshold: float = 0.3,
        ransac_threshold: float = 5.0
    ) -> Dict:
        """
        Match multiple template-target pairs in a single batch inference.

        Args:
            target_imgs: List of target images (BGR numpy arrays)
            templates: List of TemplateConfig instances
            score_threshold: Matching score threshold
            ransac_threshold: RANSAC threshold

        Returns:
            Dict with batch results:
            {
                'success': bool,
                'batch_timings': {...},
                'results': [...]  # Per-pair results
            }
        """
        batch_timings = {}
        t_total = time.time()

        num_pairs = len(target_imgs)
        if num_pairs != len(templates):
            return {
                'success': False,
                'error': 'Number of targets and templates must match',
                'batch_timings': {},
                'results': []
            }

        if num_pairs == 0 or num_pairs > 4:
            return {
                'success': False,
                'error': f'Batch size must be 1-4, got {num_pairs}',
                'batch_timings': {},
                'results': []
            }

        # Preprocess all images
        t0 = time.time()
        batch_images = []
        per_pair_preprocess = []

        for idx, (target_img, template) in enumerate(zip(target_imgs, templates)):
            t_pre = time.time()

            # Scale target if needed
            if template.scale != 1.0:
                target_scaled = cv2.resize(target_img, None, fx=template.scale, fy=template.scale)
            else:
                target_scaled = target_img

            target_gray = cv2.cvtColor(target_scaled, cv2.COLOR_BGR2GRAY)

            # Resize to engine size
            template_resized, _ = self._resize_to_engine_size(template.template_gray)
            target_resized, _ = self._resize_to_engine_size(target_gray)

            # Convert to tensors
            template_tensor = template_resized.astype(np.float32)[None, None] / 255.0
            target_tensor = target_resized.astype(np.float32)[None, None] / 255.0

            batch_images.append(template_tensor)
            batch_images.append(target_tensor)

            per_pair_preprocess.append((time.time() - t_pre) * 1000)

        batch_timings['preprocess'] = (time.time() - t0) * 1000

        # Concatenate all images into single batch
        t0 = time.time()
        batch_input = np.concatenate(batch_images, axis=0)
        batch_timings['concat'] = (time.time() - t0) * 1000

        # Single TRT inference
        t0 = time.time()
        outputs = self._infer(batch_input)
        batch_timings['trt_inference'] = (time.time() - t0) * 1000

        # Post-process each pair
        t0 = time.time()
        results = []
        per_pair_postprocess = []

        kpts_raw = outputs[0]
        matches_raw = outputs[1]
        mscores_raw = outputs[2]

        for idx, template in enumerate(templates):
            t_post = time.time()

            template_idx = idx * 2
            target_idx = idx * 2 + 1

            result = self._postprocess_pair(
                kpts_raw=kpts_raw,
                matches_raw=matches_raw,
                mscores_raw=mscores_raw,
                template_idx=template_idx,
                target_idx=target_idx,
                template=template,
                target_img_full=target_imgs[idx],
                score_threshold=score_threshold,
                ransac_threshold=ransac_threshold
            )

            per_pair_postprocess.append((time.time() - t_post) * 1000)
            results.append(result)

        batch_timings['postprocess'] = (time.time() - t0) * 1000
        batch_timings['total'] = (time.time() - t_total) * 1000

        # Add per-pair timings
        for idx, result in enumerate(results):
            result['timings'] = {
                'total': batch_timings['total'],
                'trt_inference': batch_timings['trt_inference'],
                'preprocess': per_pair_preprocess[idx],
                'postprocess': per_pair_postprocess[idx],
            }

        return {
            'success': True,
            'batch_timings': batch_timings,
            'per_pair_preprocess': per_pair_preprocess,
            'per_pair_postprocess': per_pair_postprocess,
            'results': results
        }

    def _postprocess_pair(
        self,
        kpts_raw: np.ndarray,
        matches_raw: np.ndarray,
        mscores_raw: np.ndarray,
        template_idx: int,
        target_idx: int,
        template: TemplateConfig,
        target_img_full: np.ndarray,
        score_threshold: float,
        ransac_threshold: float
    ) -> Dict:
        """Post-process a single template-target pair from batch outputs"""

        # Reshape keypoints
        kpts_flat = kpts_raw.ravel()
        try:
            for num_kpts in [4096, 2048, 1024, 512]:
                try:
                    kpts = kpts_flat[:kpts_raw.shape[0]*num_kpts*2].reshape(kpts_raw.shape[0], num_kpts, 2)
                    break
                except:
                    continue
            else:
                kpts = kpts_flat[:kpts_raw.shape[0]*1024*2].reshape(kpts_raw.shape[0], 1024, 2)
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to reshape keypoints: {e}',
                'homography': None,
                'confidence': 0.0,
                'inliers': 0,
                'total_matches': 0,
                'transformed_bboxes': [],
                'target_img': target_img_full
            }

        kpts0 = kpts[template_idx].astype(np.float32)
        kpts1 = kpts[target_idx].astype(np.float32)

        # Find valid matches
        matches_flat = matches_raw.ravel()
        mscores_flat = mscores_raw.ravel()

        valid_mask = mscores_flat > 1e-6
        num_matches = np.sum(valid_mask)

        if num_matches == 0:
            num_matches = len(mscores_flat)

        try:
            matches = matches_flat[:num_matches*3].reshape(num_matches, 3).astype(np.int32)
            mscores = mscores_flat[:num_matches]
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to reshape matches: {e}',
                'homography': None,
                'confidence': 0.0,
                'inliers': 0,
                'total_matches': 0,
                'transformed_bboxes': [],
                'target_img': target_img_full
            }

        # Filter by batch index
        pair_idx = template_idx // 2
        batch_mask = matches[:, 0] == pair_idx
        batch_matches = matches[batch_mask]
        batch_mscores = mscores[batch_mask]

        # Filter by score
        valid_mask = batch_mscores > score_threshold
        valid_matches = batch_matches[valid_mask]

        if len(valid_matches) < 10:
            return {
                'success': False,
                'error': f'Too few matches: {len(valid_matches)}',
                'homography': None,
                'confidence': 0.0,
                'inliers': 0,
                'total_matches': len(valid_matches),
                'transformed_bboxes': [],
                'target_img': target_img_full
            }

        m_kpts0 = kpts0[valid_matches[:, 1]].copy()
        m_kpts1 = kpts1[valid_matches[:, 2]].copy()

        # Scale keypoints back
        template_h, template_w = template.template_gray.shape[:2]
        engine_h, engine_w = self.input_shape[2:]
        template_scale = (template_w / engine_w, template_h / engine_h)

        target_scaled = target_img_full
        if template.scale != 1.0:
            target_scaled = cv2.resize(target_img_full, None, fx=template.scale, fy=template.scale)
        target_gray = cv2.cvtColor(target_scaled, cv2.COLOR_BGR2GRAY)
        target_h, target_w = target_gray.shape[:2]
        target_scale = (target_w / engine_w, target_h / engine_h)

        m_kpts0[:, 0] *= template_scale[0]
        m_kpts0[:, 1] *= template_scale[1]
        m_kpts1[:, 0] *= target_scale[0]
        m_kpts1[:, 1] *= target_scale[1]

        # RANSAC homography
        H, mask = cv2.findHomography(m_kpts0, m_kpts1, cv2.RANSAC, ransac_threshold)

        if H is None:
            return {
                'success': False,
                'error': 'Homography estimation failed',
                'homography': None,
                'confidence': 0.0,
                'inliers': 0,
                'total_matches': len(valid_matches),
                'transformed_bboxes': [],
                'target_img': target_img_full
            }

        inliers = np.sum(mask)
        confidence = inliers / len(m_kpts0)

        # Transform bboxes
        scale_matrix = np.array([
            [1/template.scale, 0, 0],
            [0, 1/template.scale, 0],
            [0, 0, 1]
        ])

        H_full = scale_matrix @ H @ np.linalg.inv(scale_matrix)

        transformed_bboxes = []

        # Transform template bbox
        template_pts = np.array(template.template_bbox['points'], dtype=np.float32).reshape(-1, 1, 2)
        template_transformed = cv2.perspectiveTransform(template_pts, H_full)
        transformed_bboxes.append({
            'type': 'template',
            'points': template_transformed.reshape(-1, 2).tolist(),
            'conf': template.template_bbox.get('conf', 0.8)
        })

        # Transform other bboxes
        for bbox in template.other_bboxes:
            pts = np.array(bbox['points'], dtype=np.float32).reshape(-1, 1, 2)
            pts_transformed = cv2.perspectiveTransform(pts, H_full)
            transformed_bboxes.append({
                'type': bbox['type'],
                'points': pts_transformed.reshape(-1, 2).tolist(),
                'text': bbox.get('text', ''),
                'annotation_index': bbox.get('annotation_index'),
                'conf': bbox.get('conf', 0.8)
            })

        return {
            'success': True,
            'homography': H_full,
            'confidence': float(confidence),
            'inliers': int(inliers),
            'total_matches': len(valid_matches),
            'transformed_bboxes': transformed_bboxes,
            'target_img': target_img_full
        }


# Backward compatibility wrapper
class SuperPointMatcherTRTOptimized:
    """
    Drop-in replacement for SuperPointMatcherTRT that uses shared engine.

    This class provides the same interface as SuperPointMatcherTRT but
    uses the shared SuperPointEngineTRT singleton internally.
    """

    def __init__(self, json_path: str, engine_path: str, scale: float = 1.0, verbose: bool = False):
        self.verbose = verbose
        self.scale = scale
        self.engine_path = engine_path

        t_start = time.time()

        # Load template config (lightweight)
        t0 = time.time()
        self.template_config = TemplateConfig.from_json(json_path, scale)
        if verbose:
            logger.info(f"⏱️  Load template config: {(time.time()-t0)*1000:.1f}ms")

        # Get shared engine (singleton - only loads once)
        t0 = time.time()
        self.engine = SuperPointEngineTRT.get_instance(engine_path, verbose)
        if verbose:
            logger.info(f"⏱️  Get shared engine: {(time.time()-t0)*1000:.1f}ms")

        # Copy attributes for backward compatibility
        self.template_path = self.template_config.template_path
        self.template_img = self.template_config.template_img
        self.template_gray = self.template_config.template_gray
        self.template_bbox = self.template_config.template_bbox
        self.other_bboxes = self.template_config.other_bboxes
        self.annotations = self.template_config.annotations
        self.input_shape = self.engine.input_shape

        # crop_area attribute (set externally)
        self.crop_area = None

        logger.info(f"✅ SuperPointMatcherTRTOptimized initialized ({(time.time()-t_start)*1000:.1f}ms)")
        logger.info(f"   Template: {self.template_gray.shape[1]}x{self.template_gray.shape[0]}")
        logger.info(f"   Bboxes: template + {len(self.other_bboxes)} regions")
        logger.info(f"   Using shared engine: {Path(engine_path).name}")

    def match_array(self, target_img_array: np.ndarray, score_threshold: float = 0.3,
                    ransac_threshold: float = 5.0) -> Dict:
        """Match template against target image (numpy array)"""
        return self.engine.match_single(
            target_img=target_img_array,
            template=self.template_config,
            score_threshold=score_threshold,
            ransac_threshold=ransac_threshold
        )

    def match(self, target_path: str, score_threshold: float = 0.3,
              ransac_threshold: float = 5.0) -> Dict:
        """Match template against target image (file path)"""
        target_img = cv2.imread(target_path)
        return self.match_array(target_img, score_threshold, ransac_threshold)

    def match_batch(
        self,
        target_imgs: List[np.ndarray],
        templates: List['SuperPointMatcherTRTOptimized'],
        score_threshold: float = 0.3,
        ransac_threshold: float = 5.0
    ) -> Dict:
        """
        Match multiple template-target pairs in a single batch inference.

        Args:
            target_imgs: List of target images
            templates: List of SuperPointMatcherTRTOptimized instances
            score_threshold: Matching score threshold
            ransac_threshold: RANSAC threshold

        Returns:
            Batch result dict
        """
        # Extract TemplateConfig from each matcher
        template_configs = [t.template_config for t in templates]

        return self.engine.match_batch(
            target_imgs=target_imgs,
            templates=template_configs,
            score_threshold=score_threshold,
            ransac_threshold=ransac_threshold
        )
