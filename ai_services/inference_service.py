import os
import cv2
import json
import numpy as np
import requests
import time
import shutil
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

API_BASE = os.environ.get("API_BASE", "https://suntech-vision-api.ngrok.app")


class SuperPointMatcherTRT:
    """TensorRT-accelerated template matching using SuperPoint + LightGlue"""

    def __init__(self, json_path: str, engine_path: str, scale: float = 1.0, verbose: bool = False):
        self.verbose = verbose
        self.scale = scale
        t_start = time.time()

        # Load template and annotations
        with open(json_path, 'r') as f:
            data = json.load(f)

        self.template_path = data['_template_image']
        self.annotations = data[self.template_path]

        t0 = time.time()
        template_img = cv2.imread(self.template_path)
        if scale != 1.0:
            template_img = cv2.resize(template_img, None, fx=scale, fy=scale)

        self.template_img = template_img
        self.template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        if verbose:
            logger.info(f"⏱️  Load template: {(time.time()-t0)*1000:.1f}ms")

        # Parse bboxes
        self.template_bbox = None
        self.other_bboxes = []
        for ann in self.annotations:
            if ann['type'] == 'template':
                self.template_bbox = ann
            elif ann['type'] not in ['crop_area']:
                self.other_bboxes.append(ann)

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

        logger.info(f"✅ Initialized SuperPointMatcherTRT ({(time.time()-t_start)*1000:.1f}ms)")
        logger.info(f"   Engine: {Path(engine_path).name}")
        logger.info(f"   Input shape: {self.input_shape}")
        logger.info(f"   Scale: {scale}x")
        logger.info(f"   Template: {self.template_gray.shape[1]}x{self.template_gray.shape[0]}")
        logger.info(f"   Bboxes: template + {len(self.other_bboxes)} regions")

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
                    # Use max shape for buffer allocation
                    max_shape = list(shape)
                    max_shape[0] = 8  # Max batch size from build config
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
        h, w = img.shape
        target_h, target_w = self.input_shape[2:]
        # target_h, target_w = 1080, 1920

        if h != target_h or w != target_w:
            resized = cv2.resize(img, (target_w, target_h))
            return resized, (w / target_w, h / target_h)
        return img, (1.0, 1.0)

    def match(self, target_path: str, score_threshold: float = 0.3,
              ransac_threshold: float = 5.0) -> Dict:
        """Match template against target image from file path"""
        timings = {}
        t_total = time.time()

        # Load target
        t0 = time.time()
        target_img_full = cv2.imread(target_path)
        timings['load_target'] = (time.time() - t0) * 1000

        # Process with array
        return self._match_impl(target_img_full, score_threshold, ransac_threshold, timings, t_total)

    def match_array(self, target_img_array: np.ndarray, score_threshold: float = 0.3,
                    ransac_threshold: float = 5.0) -> Dict:
        """Match template against target image from numpy array (BGR format)"""
        timings = {}
        t_total = time.time()

        # Process directly without file I/O
        return self._match_impl(target_img_array, score_threshold, ransac_threshold, timings, t_total)

    def match_batch(
        self,
        target_imgs: List[np.ndarray],
        templates: List['SuperPointMatcherTRT'],
        score_threshold: float = 0.3,
        ransac_threshold: float = 5.0
    ) -> Dict:
        """
        Match multiple template-target pairs in a single batch inference

        Args:
            target_imgs: List of target images (numpy arrays in BGR format)
            templates: List of matcher instances (one per camera with its template)
            score_threshold: Matching score threshold
            ransac_threshold: RANSAC threshold for homography

        Returns:
            Dict with batch results:
            {
                'success': bool,
                'batch_timings': {...},  # Overall batch timing
                'results': [...]  # Per-camera results
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
        per_camera_preprocess = []

        for idx, (target_img, matcher) in enumerate(zip(target_imgs, templates)):
            t_pre = time.time()

            # Scale if needed
            if matcher.scale != 1.0:
                target_scaled = cv2.resize(target_img, None, fx=matcher.scale, fy=matcher.scale)
            else:
                target_scaled = target_img

            target_gray = cv2.cvtColor(target_scaled, cv2.COLOR_BGR2GRAY)

            # Resize to engine size
            template_resized, _ = matcher._resize_to_engine_size(matcher.template_gray)
            target_resized, _ = matcher._resize_to_engine_size(target_gray)

            # Convert to tensors
            template_tensor = template_resized.astype(np.float32)[None, None] / 255.0
            target_tensor = target_resized.astype(np.float32)[None, None] / 255.0

            batch_images.append(template_tensor)
            batch_images.append(target_tensor)

            per_camera_preprocess.append((time.time() - t_pre) * 1000)

        batch_timings['preprocess'] = (time.time() - t0) * 1000

        # Concatenate all images into single batch
        t0 = time.time()
        batch_input = np.concatenate(batch_images, axis=0)  # Shape: [num_pairs*2, 1, H, W]
        batch_timings['concat'] = (time.time() - t0) * 1000

        # Single TRT inference for all pairs
        t0 = time.time()
        outputs = self._infer(batch_input)
        batch_timings['trt_inference'] = (time.time() - t0) * 1000

        # Post-process each pair
        t0 = time.time()
        results = []
        per_camera_postprocess = []

        kpts_raw = outputs[0]
        matches_raw = outputs[1]
        mscores_raw = outputs[2]

        for idx, matcher in enumerate(templates):
            t_post = time.time()

            # Extract this pair's data from batch outputs
            template_idx = idx * 2
            target_idx = idx * 2 + 1

            result = self._postprocess_pair(
                kpts_raw=kpts_raw,
                matches_raw=matches_raw,
                mscores_raw=mscores_raw,
                template_idx=template_idx,
                target_idx=target_idx,
                matcher=matcher,
                target_img_full=target_imgs[idx],
                score_threshold=score_threshold,
                ransac_threshold=ransac_threshold
            )

            per_camera_postprocess.append((time.time() - t_post) * 1000)
            results.append(result)

        batch_timings['postprocess'] = (time.time() - t0) * 1000
        batch_timings['total'] = (time.time() - t_total) * 1000

        # Add per-camera timings to each result
        for idx, result in enumerate(results):
            result['timings'] = {
                'total': batch_timings['total'],  # Shared
                'trt_inference': batch_timings['trt_inference'],  # Shared (batch inference)
                'preprocess': per_camera_preprocess[idx],  # Per-camera
                'postprocess': per_camera_postprocess[idx],  # Per-camera
            }

        return {
            'success': True,
            'batch_timings': batch_timings,
            'per_camera_preprocess': per_camera_preprocess,
            'per_camera_postprocess': per_camera_postprocess,
            'results': results
        }

    def _postprocess_pair(
        self,
        kpts_raw: np.ndarray,
        matches_raw: np.ndarray,
        mscores_raw: np.ndarray,
        template_idx: int,
        target_idx: int,
        matcher: 'SuperPointMatcherTRT',
        target_img_full: np.ndarray,
        score_threshold: float,
        ransac_threshold: float
    ) -> Dict:
        """
        Post-process a single template-target pair from batch outputs

        Args:
            kpts_raw: Raw keypoints output from batch inference
            matches_raw: Raw matches output from batch inference
            mscores_raw: Raw match scores output from batch inference
            template_idx: Index of template in batch (e.g., 0, 2, 4, 6)
            target_idx: Index of target in batch (e.g., 1, 3, 5, 7)
            matcher: Matcher instance with template info
            target_img_full: Full-size target image
            score_threshold: Matching score threshold
            ransac_threshold: RANSAC threshold

        Returns:
            Result dict with success, homography, confidence, transformed_bboxes, etc.
        """
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

        # Get keypoints for this pair
        kpts0 = kpts[template_idx].astype(np.float32)
        kpts1 = kpts[target_idx].astype(np.float32)

        # Find valid matches for this pair
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

        # Filter by batch index (template_idx -> target_idx pair)
        # Match format: [batch_idx, kpt0_idx, kpt1_idx]
        # For pair (template_idx=0, target_idx=1): we want batch_idx == 0
        # For pair (template_idx=2, target_idx=3): we want batch_idx == 1
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

        # Scale keypoints back to original image size
        # Note: matcher stores template_gray which is already resized to engine size
        # Get scale factors
        template_h, template_w = matcher.template_gray.shape[:2]
        engine_h, engine_w = matcher.input_shape[2:]
        template_scale = (template_w / engine_w, template_h / engine_h)

        target_scaled = target_img_full
        if matcher.scale != 1.0:
            target_scaled = cv2.resize(target_img_full, None, fx=matcher.scale, fy=matcher.scale)
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

        # Transform bboxes to full-size image
        scale_matrix = np.array([
            [1/matcher.scale, 0, 0],
            [0, 1/matcher.scale, 0],
            [0, 0, 1]
        ])

        H_full = scale_matrix @ H @ np.linalg.inv(scale_matrix)

        transformed_bboxes = []

        # Transform template bbox
        template_pts = np.array(matcher.template_bbox['points'], dtype=np.float32).reshape(-1, 1, 2)
        template_transformed = cv2.perspectiveTransform(template_pts, H_full)
        transformed_bboxes.append({
            'type': 'template',
            'points': template_transformed.reshape(-1, 2).tolist(),
            'conf': matcher.template_bbox.get('conf', 0.8)

        })

        # Transform other bboxes
        for bbox in matcher.other_bboxes:
            pts = np.array(bbox['points'], dtype=np.float32).reshape(-1, 1, 2)
            pts_transformed = cv2.perspectiveTransform(pts, H_full)
            transformed_bboxes.append({
                'type': bbox['type'],
                'points': pts_transformed.reshape(-1, 2).tolist(),
                'text': bbox.get('text', ''),
                'annotation_index': bbox.get('annotation_index'),  # Preserve original annotation index
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

    def _match_impl(self, target_img_full: np.ndarray, score_threshold: float,
                    ransac_threshold: float, timings: Dict, t_total: float) -> Dict:
        """Internal implementation of template matching"""
        if self.scale != 1.0:
            target_img = cv2.resize(target_img_full, None, fx=self.scale, fy=self.scale)
        else:
            target_img = target_img_full

        target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

        # Resize to engine size
        t0 = time.time()
        template_resized, template_scale = self._resize_to_engine_size(self.template_gray)
        target_resized, target_scale = self._resize_to_engine_size(target_gray)
        timings['resize_to_engine'] = (time.time() - t0) * 1000

        # Prepare batch input
        t0 = time.time()
        template_tensor = template_resized.astype(np.float32)[None, None] / 255.0
        target_tensor = target_resized.astype(np.float32)[None, None] / 255.0
        batch_input = np.concatenate([template_tensor, target_tensor], axis=0)
        timings['to_tensor'] = (time.time() - t0) * 1000

        # TensorRT inference
        t0 = time.time()
        outputs = self._infer(batch_input)
        timings['trt_inference'] = (time.time() - t0) * 1000

        # Post-process outputs
        t0 = time.time()
        kpts_raw = outputs[0]
        matches_raw = outputs[1]
        mscores_raw = outputs[2]

        # Reshape keypoints
        kpts_flat = kpts_raw.ravel()
        try:
            for num_kpts in [4096, 2048, 1024, 512]:
                try:
                    kpts = kpts_flat[:2*num_kpts*2].reshape(2, num_kpts, 2)
                    break
                except:
                    continue
            else:
                kpts = kpts_flat[:2*1024*2].reshape(2, 1024, 2)
        except Exception as e:
            kpts = kpts_raw

        # Find valid matches
        matches_flat = matches_raw.ravel()
        mscores_flat = mscores_raw.ravel()

        valid_mask = mscores_flat > 1e-6
        num_matches = np.sum(valid_mask)

        if num_matches == 0:
            num_matches = len(mscores_flat)

        # Reshape matches
        try:
            matches = matches_flat[:num_matches*3].reshape(num_matches, 3).astype(np.int32)
            mscores = mscores_flat[:num_matches]
        except Exception as e:
            timings['total'] = (time.time() - t_total) * 1000
            return {
                'success': False,
                'error': f'Failed to reshape outputs: {e}',
                'homography': None,
                'confidence': 0.0,
                'transformed_bboxes': [],
                'target_img': target_img_full,
                'timings': timings
            }

        # Filter by batch and score
        batch_mask = matches[:, 0] == 0
        batch_matches = matches[batch_mask]
        batch_mscores = mscores[batch_mask]

        kpts0 = kpts[0].astype(np.float32)
        kpts1 = kpts[1].astype(np.float32)

        valid_mask = batch_mscores > score_threshold
        valid_matches = batch_matches[valid_mask]

        if len(valid_matches) < 10:
            timings['total'] = (time.time() - t_total) * 1000
            return {
                'success': False,
                'error': f'Too few matches: {len(valid_matches)}',
                'homography': None,
                'confidence': 0.0,
                'transformed_bboxes': [],
                'target_img': target_img_full,
                'timings': timings
            }

        m_kpts0 = kpts0[valid_matches[:, 1]].copy()
        m_kpts1 = kpts1[valid_matches[:, 2]].copy()

        # Scale keypoints back
        m_kpts0[:, 0] *= template_scale[0]
        m_kpts0[:, 1] *= template_scale[1]
        m_kpts1[:, 0] *= target_scale[0]
        m_kpts1[:, 1] *= target_scale[1]
        timings['postprocess_matches'] = (time.time() - t0) * 1000

        # RANSAC homography
        t0 = time.time()
        H, mask = cv2.findHomography(m_kpts0, m_kpts1, cv2.RANSAC, ransac_threshold)
        timings['ransac_homography'] = (time.time() - t0) * 1000

        if H is None:
            timings['total'] = (time.time() - t_total) * 1000
            return {
                'success': False,
                'error': 'Homography estimation failed',
                'homography': None,
                'confidence': 0.0,
                'transformed_bboxes': [],
                'target_img': target_img_full,
                'timings': timings
            }

        inliers = np.sum(mask)
        confidence = inliers / len(m_kpts0)

        # Transform bboxes
        t0 = time.time()
        scale_matrix = np.array([
            [1/self.scale, 0, 0],
            [0, 1/self.scale, 0],
            [0, 0, 1]
        ])

        H_full = scale_matrix @ H @ np.linalg.inv(scale_matrix)

        transformed_bboxes = []

        template_pts = np.array(self.template_bbox['points'], dtype=np.float32).reshape(-1, 1, 2)
        template_transformed = cv2.perspectiveTransform(template_pts, H_full)
        transformed_bboxes.append({
            'type': 'template',
            'points': template_transformed.reshape(-1, 2).tolist(),
            'conf': self.template_bbox.get('conf', 0.8)

        })

        for bbox in self.other_bboxes:
            pts = np.array(bbox['points'], dtype=np.float32).reshape(-1, 1, 2)
            pts_transformed = cv2.perspectiveTransform(pts, H_full)
            transformed_bboxes.append({
                'type': bbox['type'],
                'points': pts_transformed.reshape(-1, 2).tolist(),
                'text': bbox.get('text', ''),
                'annotation_index': bbox.get('annotation_index'),  # Preserve original annotation index
                'conf': bbox.get('conf', 0.8)

            })
        timings['transform_bboxes'] = (time.time() - t0) * 1000

        timings['total'] = (time.time() - t_total) * 1000

        return {
            'success': True,
            'homography': H_full,
            'confidence': confidence,
            'inliers': inliers,
            'total_matches': len(m_kpts0),
            'transformed_bboxes': transformed_bboxes,
            'target_img': target_img_full,
            'timings': timings
        }


@dataclass
class InferenceConfig:
    enabled: bool = False
    recipe_id: Optional[str] = None
    recipe_name: Optional[str] = None
    template_path: Optional[str] = None
    annotations: Optional[List[Dict]] = None
    matcher: Optional[SuperPointMatcherTRT] = None
    camera_serial: Optional[str] = None


class InferenceService:
    def __init__(self, engine_path: str = "weights/pipeline_fp16_small.engine"):
        self.engine_path = Path(engine_path)
        self.base_matcher = None
        self.config = InferenceConfig()
        self.temp_dir = Path("ocr_inference")
        self.temp_dir.mkdir(exist_ok=True)

        # State directory for IPC between backend and camera producer
        self.state_dir = Path(__file__).parent.parent / "backend" / "inference_state"
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Pre-initialize TensorRT engine with dummy template
        if self.engine_path.exists():
            logger.info(f"🔥 Warming up TensorRT engine: {self.engine_path}")
            try:
                # Create dummy template and annotations for engine warm-up
                dummy_template = self.temp_dir / "dummy_template.jpg"
                dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.imwrite(str(dummy_template), dummy_img)

                dummy_ann_path = self.temp_dir / "dummy_annotations.json"
                dummy_ann = {
                    '_template_image': str(dummy_template),
                    str(dummy_template): [{
                        'type': 'template',
                        'points': [[100, 100], [200, 100], [200, 200], [100, 200]]
                    }]
                }
                with open(dummy_ann_path, 'w') as f:
                    json.dump(dummy_ann, f)

                # Init matcher to warm up engine
                self.base_matcher = SuperPointMatcherTRT(
                    json_path=str(dummy_ann_path),
                    engine_path=str(self.engine_path),
                    scale=1.0,
                    verbose=False
                )
                logger.info("✅ TensorRT engine warmed up and ready")
            except Exception as e:
                logger.error(f"❌ Failed to warm up TensorRT engine: {e}")
                self.base_matcher = None
        else:
            logger.warning(f"⚠️ TensorRT engine not found: {self.engine_path}")

        logger.info("InferenceService initialized")

    def _save_state_to_file(self, camera_serial: str):
        """Save inference state to file for IPC"""
        state_file = self.state_dir / f"{camera_serial}.json"
        state_data = {
            'enabled': self.config.enabled,
            'recipe_id': self.config.recipe_id,
            'recipe_name': self.config.recipe_name,
            'camera_serial': self.config.camera_serial,
            'template_path': self.config.template_path,
            'annotations': self.config.annotations
        }
        with open(state_file, 'w') as f:
            json.dump(state_data, f, indent=2)
        logger.info(f"💾 [STATE FILE] Saved inference state to {state_file}")

    def _load_state_from_file(self, camera_serial: str) -> Optional[Dict]:
        """Load inference state from file for IPC"""
        state_file = self.state_dir / f"{camera_serial}.json"
        if not state_file.exists():
            return None

        try:
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            return state_data
        except Exception as e:
            logger.error(f"❌ Failed to load state file {state_file}: {e}")
            return None

    def init_matcher_from_file(self, camera_serial: str) -> bool:
        """Initialize matcher from state file (for camera producer IPC)

        Args:
            camera_serial: Camera serial number

        Returns:
            True if matcher initialized successfully, False otherwise
        """
        state_data = self._load_state_from_file(camera_serial)
        if not state_data or not state_data.get('enabled'):
            return False

        # Check if matcher already initialized for this camera
        if (self.config.enabled and
            self.config.camera_serial == camera_serial and
            self.config.matcher is not None):
            logger.info(f"Matcher already initialized for camera {camera_serial}")
            return True

        try:
            template_path = state_data.get('template_path')
            annotations = state_data.get('annotations', [])

            if not template_path or not Path(template_path).exists():
                logger.error(f"Template file not found: {template_path}")
                return False

            # Create annotations JSON
            template_bbox = None
            other_bboxes = []

            for ann in annotations:
                ann_type = ann.get('type', '')
                if ann_type == 'template':
                    x = ann.get('x', 0)
                    y = ann.get('y', 0)
                    w = ann.get('width', 0)
                    h = ann.get('height', 0)

                    template_img = cv2.imread(template_path)
                    img_h, img_w = template_img.shape[:2]

                    x1 = int(x * img_w)
                    y1 = int(y * img_h)
                    x2 = int((x + w) * img_w)
                    y2 = int((y + h) * img_h)

                    template_bbox = {
                        'type': 'template',
                        'points': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    }
                elif ann_type == 'text' and ann.get('points'):
                    points = ann.get('points', [])
                    template_img = cv2.imread(template_path)
                    img_h, img_w = template_img.shape[:2]

                    pixel_points = []
                    for pt in points:
                        px = int(pt[0] * img_w)
                        py = int(pt[1] * img_h)
                        pixel_points.append([px, py])

                    other_bboxes.append({
                        'type': ann_type,
                        'text': ann.get('text', ''),
                        'points': pixel_points
                    })

            if not template_bbox:
                logger.error("No template bbox found in annotations")
                return False

            # Create annotation file
            ann_json_path = self.temp_dir / f"annotations_{camera_serial}.json"
            ann_data = {
                '_template_image': template_path,
                template_path: [template_bbox] + other_bboxes
            }

            with open(ann_json_path, 'w') as f:
                json.dump(ann_data, f, indent=2)

            # Initialize matcher
            logger.info(f"Initializing matcher from state file for camera {camera_serial}")
            matcher = SuperPointMatcherTRT(
                json_path=str(ann_json_path),
                engine_path=str(self.engine_path),
                scale=1.0,
                verbose=True
            )

            # Update config
            self.config.enabled = state_data.get('enabled', False)
            self.config.recipe_id = state_data.get('recipe_id')
            self.config.recipe_name = state_data.get('recipe_name')
            self.config.template_path = template_path
            self.config.annotations = annotations
            self.config.matcher = matcher
            self.config.camera_serial = camera_serial

            logger.info(f"✅ Matcher initialized for camera {camera_serial}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize matcher from file: {e}", exc_info=True)
            return False

    def load_recipe(self, camera_serial: str, recipe_data: Optional[Dict] = None) -> Dict:
        """Load recipe and initialize matcher"""
        try:
            # Use provided recipe_data or fetch from API
            if recipe_data:
                data = recipe_data
                metadata = data
            else:
                url = f"{API_BASE}/api/recipes/loads/latest"
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                metadata = data.get('metadata', {})

            # Extract recipe metadata
            recipe_id = data.get('recipe_id') or data.get('id')
            recipe_name = metadata.get('name', 'Unknown')

            # Find camera config
            cameras = metadata.get('cameras', [])
            camera_config = None
            for cam in cameras:
                if cam.get('serial_number') == camera_serial:
                    camera_config = cam
                    break

            if not camera_config:
                logger.warning(f"Camera {camera_serial} not found in recipe")
                return {
                    'success': False,
                    'error': f'Camera {camera_serial} not configured in recipe'
                }

            # Find template for this camera
            camera_templates = metadata.get('camera_templates', [])
            template_data = None
            for ct in camera_templates:
                if ct.get('camera_id') == camera_config.get('camera_id'):
                    templates = ct.get('templates', [])
                    if templates:
                        template_data = templates[0]
                        break

            if not template_data:
                return {
                    'success': False,
                    'error': 'No template found for this camera'
                }

            # Get template image from local filesystem
            image_url = template_data.get('image_url')
            if not image_url:
                return {
                    'success': False,
                    'error': 'No template image URL'
                }

            # Extract filename from URL
            filename = image_url.split('/')[-1]

            # Read from local uploads directory
            backend_dir = Path(__file__).parent.parent / "backend"
            source_template_path = backend_dir / "uploads" / "templates" / filename

            if not source_template_path.exists():
                return {
                    'success': False,
                    'error': f'Template image not found: {source_template_path}'
                }

            logger.info(f"Reading template from {source_template_path}")

            # Copy to temp directory for processing
            template_path = self.temp_dir / f"template_{camera_serial}.jpg"
            shutil.copy(source_template_path, template_path)

            # Get annotations
            annotations = template_data.get('annotations', [])

            # Create annotations JSON for TensorRT matcher
            template_bbox = None
            other_bboxes = []

            for ann in annotations:
                ann_type = ann.get('type', '')
                if ann_type == 'template':
                    # Convert rectangle to polygon points
                    x = ann.get('x', 0)
                    y = ann.get('y', 0)
                    w = ann.get('width', 0)
                    h = ann.get('height', 0)

                    # Read template to get actual dimensions
                    template_img = cv2.imread(str(template_path))
                    img_h, img_w = template_img.shape[:2]

                    # Convert normalized coords to pixels
                    x1 = int(x * img_w)
                    y1 = int(y * img_h)
                    x2 = int((x + w) * img_w)
                    y2 = int((y + h) * img_h)

                    template_bbox = {
                        'type': 'template',
                        'points': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    }
                elif ann_type == 'text' and ann.get('points'):
                    # Already in polygon format
                    points = ann.get('points', [])

                    # Convert normalized to pixels
                    template_img = cv2.imread(str(template_path))
                    img_h, img_w = template_img.shape[:2]

                    pixel_points = []
                    for pt in points:
                        px = int(pt[0] * img_w)
                        py = int(pt[1] * img_h)
                        pixel_points.append([px, py])

                    other_bboxes.append({
                        'type': ann_type,
                        'text': ann.get('text', ''),
                        'points': pixel_points
                    })

            if not template_bbox:
                return {
                    'success': False,
                    'error': 'No template bbox found in annotations'
                }

            # Create annotation structure for matcher
            ann_json_path = self.temp_dir / f"annotations_{camera_serial}.json"
            ann_data = {
                '_template_image': str(template_path),
                str(template_path): [template_bbox] + other_bboxes
            }

            with open(ann_json_path, 'w') as f:
                json.dump(ann_data, f, indent=2)

            # Initialize TensorRT matcher
            logger.info(f"Initializing TensorRT matcher with {len(other_bboxes)} regions")
            matcher = SuperPointMatcherTRT(
                json_path=str(ann_json_path),
                engine_path=str(self.engine_path),
                scale=1.0,
                verbose=True
            )

            # Update config
            self.config.enabled = True
            self.config.recipe_id = recipe_id
            self.config.recipe_name = recipe_name
            self.config.template_path = str(template_path)
            self.config.annotations = annotations
            self.config.matcher = matcher
            self.config.camera_serial = camera_serial

            # Save state to file for IPC with camera producer
            self._save_state_to_file(camera_serial)

            logger.info(f"Recipe loaded successfully: {recipe_name}")

            return {
                'success': True,
                'recipe_id': recipe_id,
                'recipe_name': recipe_name,
                'camera_serial': camera_serial,
                'template_regions': len(other_bboxes),
                'message': f'Recipe "{recipe_name}" loaded successfully'
            }

        except Exception as e:
            logger.error(f"Error loading recipe: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    def process_trigger(self, frame: np.ndarray, frame_idx: int) -> Dict:
        """Process triggered frame with inference"""
        if not self.config.enabled or not self.config.matcher:
            return {
                'success': False,
                'error': 'Inference not enabled. Load a recipe first.'
            }

        try:
            # Save frame temporarily
            temp_frame_path = self.temp_dir / f"trigger_frame_{self.config.camera_serial}_{frame_idx}.jpg"
            cv2.imwrite(str(temp_frame_path), frame)

            logger.info(f"Processing trigger frame #{frame_idx}")

            # Run TensorRT matching
            result = self.config.matcher.match(
                target_path=str(temp_frame_path),
                score_threshold=0.3,
                ransac_threshold=5.0
            )

            if not result['success']:
                logger.warning(f"Matching failed: {result.get('error')}")
                return {
                    'success': False,
                    'error': result.get('error'),
                    'frame_idx': frame_idx
                }

            # Visualize result
            output_path = self.temp_dir / f"result_{self.config.camera_serial}_{frame_idx}.jpg"
            vis_img = self._visualize_result(result, frame_idx)
            cv2.imwrite(str(output_path), vis_img)

            logger.info(f"Inference complete: confidence={result['confidence']:.1%}, output={output_path}")

            return {
                'success': True,
                'frame_idx': frame_idx,
                'confidence': result['confidence'],
                'inliers': result['inliers'],
                'total_matches': result['total_matches'],
                'output_path': str(output_path),
                'timings': result['timings'],
                'transformed_bboxes': result['transformed_bboxes']
            }

        except Exception as e:
            logger.error(f"Error processing trigger: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'frame_idx': frame_idx
            }

    def _visualize_result(self, result: Dict, frame_idx: int) -> np.ndarray:
        """Visualize matching result on target image"""
        target_img = result['target_img'].copy()

        colors = {
            'template': (0, 255, 0),
            'text': (255, 165, 0),
            'barcode': (255, 0, 255),
            'datecode': (0, 255, 255)
        }

        # Draw transformed bboxes
        for bbox in result['transformed_bboxes']:
            pts = np.array(bbox['points'], dtype=np.int32)
            bbox_type = bbox['type']
            color = colors.get(bbox_type, (255, 255, 255))

            # Draw polygon
            cv2.polylines(target_img, [pts], True, color, 3)

            # Draw corner markers
            corner_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
            for j, pt in enumerate(pts):
                cv2.circle(target_img, tuple(pt), 8, corner_colors[j], -1)

            # Draw label
            center = pts.mean(axis=0).astype(int)
            cv2.putText(target_img, bbox_type, tuple(center),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Draw info overlay
        timings = result.get('timings', {})
        info_text = [
            f"Frame: #{frame_idx}",
            f"Recipe: {self.config.recipe_name}",
            f"Confidence: {result['confidence']:.1%}",
            f"Matches: {result['inliers']}/{result['total_matches']}",
            f"Time: {timings.get('total', 0):.0f}ms (TRT: {timings.get('trt_inference', 0):.0f}ms)"
        ]

        y_offset = 40
        for i, text in enumerate(info_text):
            y_pos = y_offset + i * 35
            cv2.putText(target_img, text, (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(target_img, text, (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        return target_img

    def disable(self):
        """Disable inference"""
        self.config.enabled = False
        self.config.matcher = None
        logger.info("Inference disabled")

    def get_status(self, camera_serial: Optional[str] = None) -> Dict:
        """Get current inference status

        Args:
            camera_serial: If provided, load status from file (for IPC).
                          If None, return current in-memory config.
        """
        # If camera_serial provided, try to load from file first (IPC)
        if camera_serial:
            state_data = self._load_state_from_file(camera_serial)
            if state_data:
                return {
                    'enabled': state_data.get('enabled', False),
                    'recipe_id': state_data.get('recipe_id'),
                    'recipe_name': state_data.get('recipe_name'),
                    'camera_serial': state_data.get('camera_serial'),
                    'template_path': state_data.get('template_path')
                }

        # Fallback to in-memory config
        return {
            'enabled': self.config.enabled,
            'recipe_id': self.config.recipe_id,
            'recipe_name': self.config.recipe_name,
            'camera_serial': self.config.camera_serial,
            'template_path': self.config.template_path
        }


# Singleton instance
home = os.environ.get('HOME')
inference_service = InferenceService(f"{home}/Source/ocr_datecode/weights/pipeline_fp16_dynamic_480x640.engine")
