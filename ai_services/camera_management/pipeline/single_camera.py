"""
Single Camera Pipeline

Handles inference for single camera scenarios:
- Single camera with single template
- Single camera with multiple templates (batch processing)
"""

import logging
import time
from typing import Dict, Any, List, Optional, TYPE_CHECKING

import cv2
import numpy as np

from .base import InferencePipelineTemplate, PipelineContext
from ..camera import Camera

logger = logging.getLogger(__name__)


class SingleCameraPipeline(InferencePipelineTemplate):
    """
    Pipeline for single camera inference.

    Supports both single template and multi-template scenarios.
    Uses batch inference when multiple templates are configured.
    """

    def __init__(
        self,
        crop_func=None,
        transform_func=None,
        result_builder=None,
        save_and_encode_func=None,
        encode_display_func=None
    ):
        """
        Initialize SingleCameraPipeline.

        Args:
            crop_func: Function to crop frame by crop_area
            transform_func: Function to transform results to full image coords
            result_builder: InferenceResultBuilder class for building results
            save_and_encode_func: Function to save and encode frames
            encode_display_func: Function to encode frames for display
        """
        self._crop_func = crop_func
        self._transform_func = transform_func
        self._result_builder = result_builder
        self._save_and_encode_func = save_and_encode_func
        self._encode_display_func = encode_display_func

    def prepare(self, context: PipelineContext) -> bool:
        """Validate single camera setup"""
        if not context.cameras:
            logger.warning(f"[Job #{context.job_id}] No cameras provided")
            return False

        if len(context.cameras) > 1:
            logger.warning(
                f"[Job #{context.job_id}] SingleCameraPipeline received "
                f"{len(context.cameras)} cameras, using first only"
            )

        camera = context.cameras[0]
        serial_number = camera.serial_number

        # Check if matcher exists
        if serial_number not in context.camera_matchers:
            logger.error(f"[Job #{context.job_id}] No matcher for camera {serial_number}")
            return False

        # Check if frames exist
        if serial_number not in context.results:
            logger.error(f"[Job #{context.job_id}] No results for camera {serial_number}")
            return False

        context.cameras_to_process = [camera]
        return True

    def preprocess(self, context: PipelineContext) -> Optional[Dict[str, Any]]:
        """Prepare frames for inference"""
        camera = context.cameras_to_process[0]
        serial_number = camera.serial_number
        frames = context.results[serial_number].get('frames', [])

        matcher_or_list = context.camera_matchers.get(serial_number)
        is_multi_template = isinstance(matcher_or_list, list)

        if is_multi_template:
            matchers = matcher_or_list
            num_templates = len(matchers)
        else:
            matchers = [matcher_or_list]
            num_templates = 1

        # For Check_Color: rotate FULL frames BEFORE crop (single YOLO pass per frame)
        # Rotated frames replace context.results so verify_results + result_builder
        # automatically use them for OCR crop and display output.
        use_obb_rotation = (
            camera.function_type == 'Check_Color'
            and context.obb_rotation_service
            and context.obb_rotation_service.available
        )
        frame_rotation_matrices = [None] * len(frames)

        if use_obb_rotation:
            from ..preprocessing.obb_rotator import transform_crop_area
            rotated_frames = []
            for idx, frame in enumerate(frames):
                frame_tag = f"{serial_number}/frame{idx}"
                rotated, M = context.obb_rotation_service.rotate_frame(
                    frame, frame_tag=frame_tag
                )
                rotated_frames.append(rotated)
                frame_rotation_matrices[idx] = M  # None means rotation failed → original used

            if context.obb_rotation_service.inverse_transform:
                # inverse_transform=True: output dùng ảnh gốc, bbox sẽ được inverse về tọa độ gốc
                # → KHÔNG replace frames trong context (giữ ảnh gốc cho display/OCR output)
                pass
            else:
                # inverse_transform=False: output dùng ảnh đã xoay
                # → replace frames để verify/display dùng ảnh xoay
                context.results[serial_number]['frames'] = rotated_frames

            frames = rotated_frames  # superpoint luôn dùng ảnh đã xoay

        # Prepare inputs: crop from (rotated or original) frame
        target_imgs = []
        crop_areas = []

        for idx, (frame, matcher) in enumerate(zip(frames[:num_templates], matchers)):
            crop_area = getattr(matcher, 'crop_area', None)

            # Transform crop_area coords into rotated frame space (if rotation succeeded)
            M = frame_rotation_matrices[idx] if idx < len(frame_rotation_matrices) else None
            if M is not None and crop_area:
                crop_area = transform_crop_area(crop_area, M)

            if self._crop_func and crop_area:
                frame_for_inference = self._crop_func(frame, crop_area)
            else:
                frame_for_inference = frame

            # Apply horizontal erosion to suppress variable text before SuperPoint matching
            if getattr(camera, 'match_erosion_enabled', False):
                kw = getattr(camera, 'match_erosion_kernel_w', 80)
                kh = getattr(camera, 'match_erosion_kernel_h', 1)
                iters = getattr(camera, 'match_erosion_iterations', 1)
                kernel = np.ones((kh, kw), np.uint8)
                frame_for_inference = cv2.erode(frame_for_inference, kernel, iterations=iters)

            target_imgs.append(frame_for_inference)
            crop_areas.append(crop_area)

        return {
            'camera': camera,
            'serial_number': serial_number,
            'frames': frames,
            'matchers': matchers,
            'target_imgs': target_imgs,
            'crop_areas': crop_areas,
            'rotation_matrices': frame_rotation_matrices,
            'is_multi_template': is_multi_template
        }

    def run_inference(
        self,
        context: PipelineContext,
        preprocessed: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Run batch inference"""
        matchers = preprocessed['matchers']
        target_imgs = preprocessed['target_imgs']
        crop_areas = preprocessed['crop_areas']
        rotation_matrices = preprocessed.get('rotation_matrices', [None] * len(target_imgs))

        try:
            # Per-recipe matching-confidence gate (default 0.20 = 20%)
            camera = context.cameras_to_process[0]
            matching_conf = float(getattr(camera, 'matching_conf', 0.20) or 0.20)

            # Use first matcher for batch inference
            batch_result = matchers[0].match_batch(
                target_imgs=target_imgs,
                templates=matchers,
                score_threshold=0.3,
                ransac_threshold=5.0,
                min_confidence=matching_conf,
            )

            if not batch_result.get('success', False):
                logger.error(f"Batch inference failed: {batch_result.get('error')}")
                return None

            # Transform results back to full frame coordinates:
            # 1. crop offset (cropped-rotated → full-rotated)
            # 2. inverse rotation (full-rotated → full-original)
            transformed_results = []
            for idx, result in enumerate(batch_result['results']):
                # Step 1: offset back from (rotated) crop area → full rotated frame coords
                crop_area = crop_areas[idx]
                if self._transform_func:
                    result = self._transform_func(result, crop_area)

                # Step 2: inverse rotation → original frame coords (if enabled)
                M = rotation_matrices[idx] if idx < len(rotation_matrices) else None
                if (M is not None
                        and context.obb_rotation_service
                        and context.obb_rotation_service.inverse_transform):
                    from ..preprocessing.obb_rotator import inverse_transform_bboxes
                    result = inverse_transform_bboxes(result, M)

                transformed_results.append(result)

            return {
                'batch_result': batch_result,
                'transformed_results': transformed_results,
                'timings': batch_result.get('batch_timings', {})
            }

        except Exception as e:
            logger.error(f"Inference error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def verify_results(
        self,
        context: PipelineContext,
        inference_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify text/char/template/product for each frame.

        Char ML verification is BATCHED across all eligible frames in a single
        embedder call to amortize the model's fixed setup cost (≈50-310ms).
        Template + product verify stay per-frame (cheap, independent state).
        """
        camera = context.cameras_to_process[0]
        serial_number = camera.serial_number
        frames = context.results[serial_number].get('frames', [])
        matchers = context.camera_matchers.get(serial_number)
        if not isinstance(matchers, list):
            matchers = [matchers]

        transformed_results = inference_results['transformed_results']

        # Batch product verification (existing optimization)
        product_verification_results = self._batch_verify_products(
            context, camera, frames, transformed_results
        )

        # ── Phase 0: initialize per-frame result skeletons ──
        verified_frames: List[Dict[str, Any]] = []
        for idx, result in enumerate(transformed_results):
            verified_frames.append({
                'frame_idx':            idx,
                'result':               'PASS' if result.get('success') else 'FAIL',
                'confidence':           float(result.get('confidence', 0.0)),
                'inliers':              int(result.get('inliers', 0)),
                'total_matches':        int(result.get('total_matches', 0)),
                'timings':              dict(result.get('timings', {})),
                'transformed_bboxes':   result.get('transformed_bboxes', []),
                'text_verification':    None,
                'char_verification':    None,
                'template_verification': None,
                'product_verification': None,
            })

        # ── Phase 1: collect frames eligible for text/char verification ──
        text_verify_indices: List[int] = []
        text_verify_data: List[Dict[str, Any]] = []
        if (camera.function_type in ('Check_Type_Product', 'Check_Color') and
                context.text_verification_service):
            for idx, result in enumerate(transformed_results):
                if not result.get('success'):
                    continue
                frame_expected_texts = camera.expected_texts.get(idx, {})
                if not frame_expected_texts:
                    continue
                matcher = matchers[idx] if idx < len(matchers) else matchers[0]
                sim_template_img = None
                sim_original_bboxes = None
                if hasattr(matcher, 'template_img') and hasattr(matcher, 'other_bboxes'):
                    sim_template_img = matcher.template_img
                    sim_original_bboxes = matcher.other_bboxes
                text_verify_indices.append(idx)
                text_verify_data.append({
                    'frame_img':          frames[idx],
                    'transformed_bboxes': verified_frames[idx]['transformed_bboxes'],
                    'expected_texts':     frame_expected_texts,
                    'camera':             camera,
                    'template_img':       sim_template_img,
                    'original_bboxes':    sim_original_bboxes,
                })

        # ── Phase 2: SINGLE batched verify across eligible frames ──
        per_frame_ocr_ms = {idx: 0.0 for idx in range(len(transformed_results))}
        if text_verify_data:
            t_batched_start = time.perf_counter()
            batched_verifs = context.text_verification_service.verify_text_regions_batched_frames(
                text_verify_data
            )
            t_batched_total_ms = (time.perf_counter() - t_batched_start) * 1000
            # Distribute the batched cost evenly across participating frames so per-frame
            # timing logs stay meaningful.
            per_frame_share = t_batched_total_ms / max(1, len(text_verify_data))

            for idx, verification in zip(text_verify_indices, batched_verifs):
                fr = verified_frames[idx]
                text_verification = verification.get('text') or {}
                char_verification = verification.get('char') or {}
                fr['text_verification'] = text_verification
                fr['char_verification'] = char_verification
                per_frame_ocr_ms[idx] = per_frame_share

                # Update bboxes with recognized text
                context.text_verification_service.update_bboxes_with_recognized_text(
                    fr['transformed_bboxes'], text_verification
                )

                # Mark failed text bboxes
                for text_result in text_verification.get('results', []):
                    annotation_idx = text_result.get('annotation_idx')
                    if annotation_idx is not None and not text_result.get('match', False):
                        for bbox in fr['transformed_bboxes']:
                            if (bbox.get('type') in ['text', 'datecode'] and
                                    bbox.get('annotation_index') == annotation_idx):
                                bbox['verification_status'] = 'fail'

                # Mark failed char bboxes
                for char_result in char_verification.get('results', []):
                    annotation_idx = char_result.get('annotation_idx')
                    if annotation_idx is not None and not char_result.get('match', False):
                        for bbox in fr['transformed_bboxes']:
                            if (bbox.get('type') == 'char' and
                                    bbox.get('annotation_index') == annotation_idx):
                                bbox['verification_status'] = 'fail'

        # ── Phase 3: per-frame template + product verify + final decision ──
        for idx, result in enumerate(transformed_results):
            frame_result = verified_frames[idx]
            frame = frames[idx]
            matcher = matchers[idx] if idx < len(matchers) else matchers[0]

            # Template verification
            if (result.get('success') and
                context.template_verification_service and
                hasattr(matcher, 'template_img') and
                hasattr(matcher, 'template_bbox')):

                template_verification = context.template_verification_service.verify_template_regions(
                    frame_img=frame,
                    template_img=matcher.template_img,
                    transformed_bboxes=frame_result['transformed_bboxes'],
                    original_template_bbox=matcher.template_bbox,
                    camera=camera
                )
                frame_result['template_verification'] = template_verification

                if template_verification and not template_verification.get('match', True):
                    for bbox in frame_result['transformed_bboxes']:
                        if bbox.get('type') == 'template':
                            bbox['verification_status'] = 'fail'

            # Product verification — skip for Check_Color
            if (result.get('success') and
                context.product_verification_service and
                camera.function_type != 'Check_Color' and
                idx < len(product_verification_results)):

                product_verification = product_verification_results[idx]
                frame_result['product_verification'] = product_verification

                if (product_verification and
                        not product_verification.get('skipped', True) and
                        not product_verification.get('match', True)):
                    for bbox in frame_result['transformed_bboxes']:
                        if bbox.get('type') == 'product':
                            bbox['verification_status'] = 'fail'

            # Merge verification timings
            timings = frame_result['timings']
            t_ocr_ms = per_frame_ocr_ms.get(idx, 0.0)
            if t_ocr_ms > 0:
                timings['ocr_ms'] = t_ocr_ms
            template_verif = frame_result.get('template_verification') or {}
            if template_verif.get('timing'):
                timings['template_verification_ms'] = template_verif['timing'].get('total_ms', 0.0)
            product_verif = frame_result.get('product_verification') or {}
            if product_verif.get('timing'):
                timings['product_verification_ms'] = product_verif['timing'].get('total', 0.0)
            timings['total'] = (
                timings.get('total', 0.0)
                + timings.get('ocr_ms', 0.0)
                + timings.get('template_verification_ms', 0.0)
                + timings.get('product_verification_ms', 0.0)
            )

            # Final pass/fail (text AND char AND template AND product)
            text_ok = (frame_result['text_verification'] is None or
                       frame_result['text_verification'].get('all_match', True))
            char_ok = (frame_result['char_verification'] is None or
                       frame_result['char_verification'].get('all_match', True))
            template_ok = (frame_result['template_verification'] is None or
                           frame_result['template_verification'].get('match', True))
            product_ok = (frame_result['product_verification'] is None or
                          frame_result['product_verification'].get('skipped', True) or
                          frame_result['product_verification'].get('match', True))

            if not (text_ok and char_ok and template_ok and product_ok):
                frame_result['result'] = 'FAIL'

        # Determine overall result
        overall = 'PASS'
        for fr in verified_frames:
            if fr['result'] in ['FAIL', 'ERROR']:
                overall = fr['result']
                break

        return {
            'serial_number': serial_number,
            'frames': verified_frames,
            'overall_result': overall
        }

    def _batch_verify_products(
        self,
        context: PipelineContext,
        camera: 'Camera',
        frames: List[np.ndarray],
        transformed_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Batch verify products for all frames.

        Args:
            context: Pipeline context
            camera: Camera object
            frames: List of frame images
            transformed_results: List of transformed inference results

        Returns:
            List of product verification results (one per frame)
        """
        if not context.product_verification_service:
            return [None] * len(frames)

        # Prepare frames data for batch verification
        frames_data = []
        for idx, result in enumerate(transformed_results):
            # Get center_offset_threshold from template config
            # In single camera scenario with multiple templates, idx corresponds to template_idx
            center_offset_threshold = None
            center_offset_threshold_left = 50.0
            center_offset_threshold_right = 50.0
            center_offset_unit = 'px'
            wrinkle_area = None
            wrinkle_min_area = 0.0
            wrinkle_max_area = 0.0
            if camera.templates and idx < len(camera.templates):
                template = camera.templates[idx]
                center_offset_threshold = template.get('center_offset_threshold', 50.0)
                center_offset_threshold_left = template.get('center_offset_threshold_left', 50.0)
                center_offset_threshold_right = template.get('center_offset_threshold_right', 50.0)
                center_offset_unit = template.get('center_offset_unit', 'px') or 'px'
                wrinkle_area = template.get('wrinkle_area', None)
                wrinkle_min_area = template.get('wrinkle_min_area', 0.0) or 0.0
                wrinkle_max_area = template.get('wrinkle_max_area', 0.0) or 0.0

            wrinkle_conf = getattr(camera, 'wrinkle_conf', 0.25)
            wrinkle_show_when_pass = getattr(camera, 'wrinkle_show_when_pass', True)
            mask_overlap_threshold = getattr(camera, 'mask_overlap_threshold', 0.6)

            if result.get('success') and idx < len(frames):
                frames_data.append({
                    'frame_img': frames[idx],
                    'transformed_bboxes': result.get('transformed_bboxes', []),
                    'camera': camera,
                    'center_offset_threshold': center_offset_threshold,
                    'center_offset_threshold_left': center_offset_threshold_left,
                    'center_offset_threshold_right': center_offset_threshold_right,
                    'center_offset_unit': center_offset_unit,
                    'wrinkle_area': wrinkle_area,
                    'wrinkle_min_area': wrinkle_min_area,
                    'wrinkle_max_area': wrinkle_max_area,
                    'wrinkle_conf': wrinkle_conf,
                    'wrinkle_show_when_pass': wrinkle_show_when_pass,
                    'mask_overlap_threshold': mask_overlap_threshold,
                })
            else:
                frames_data.append({
                    'frame_img': None,
                    'transformed_bboxes': [],
                    'camera': camera,
                    'center_offset_threshold': center_offset_threshold,
                    'center_offset_unit': center_offset_unit,
                    'wrinkle_area': wrinkle_area,
                    'wrinkle_min_area': wrinkle_min_area,
                    'wrinkle_max_area': wrinkle_max_area,
                    'wrinkle_conf': wrinkle_conf,
                    'wrinkle_show_when_pass': wrinkle_show_when_pass,
                    'mask_overlap_threshold': mask_overlap_threshold,
                })

        # Filter valid frames for batch processing
        valid_frames = [
            data for data in frames_data
            if data['frame_img'] is not None
        ]

        if not valid_frames:
            return [None] * len(frames)

        # Check how many frames need verification (have both product and label regions)
        frames_needing_verification = [
            data for data in valid_frames
            if context.product_verification_service.should_verify_frame(data['transformed_bboxes'])
        ]

        logger.debug(
            f"[{camera.serial_number}] Product verification: "
            f"{len(frames_needing_verification)}/{len(valid_frames)} frames need verification"
        )

        # Batch verify
        try:
            import time
            t_start = time.perf_counter()
            verification_results = context.product_verification_service.verify_batch(frames_data)
            t_elapsed = (time.perf_counter() - t_start) * 1000

            # Log timing details if available (find first valid timing)
            timing = None
            for result in verification_results:
                if result and result.get('timing') and result['timing'].get('total', 0) > 0:
                    timing = result['timing']
                    break

            if timing:
                logger.info(
                    f"[{camera.serial_number}] Product verification complete: "
                    f"total={timing.get('total', 0):.1f}ms, "
                    f"yolo={timing.get('yolo_inference', 0):.1f}ms, "
                    f"frames={timing.get('frames_checked', 0)}/{timing.get('frames_total', 0)}"
                )

            return verification_results
        except Exception as e:
            logger.error(f"[{camera.serial_number}] Batch product verification failed: {e}")
            import traceback
            traceback.print_exc()
            # Return error results
            return [{
                'match': False,
                'skipped': False,
                'error': f'Batch verification failed: {str(e)}',
                'timing': {'total': 0.0}
            } for _ in frames]

    def postprocess(
        self,
        context: PipelineContext,
        verified_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build final result structure"""
        serial_number = verified_results['serial_number']
        overall = verified_results['overall_result']

        # Store in context
        context.camera_inference_results[serial_number] = verified_results
        context.overall_pass_fail = overall

        # Build result using builder if available
        if self._result_builder:
            return self._result_builder.from_cameras(
                cameras=context.cameras,
                results=context.results,
                camera_inference_results=context.camera_inference_results,
                overall_pass_fail=overall,
                camera_matchers=context.camera_matchers,
                save_and_encode_func=self._save_and_encode_func,
                encode_display_func=self._encode_display_func,
                statistics=context.statistics
            )

        # Fallback to simple result
        camera = context.cameras[0]
        return {
            "recipe_id": camera.recipe_id,
            "recipe_name": camera.recipe_name,
            "product_pass_fail": overall,
            "camera_results": [{
                "serial_number": serial_number,
                "frames": verified_results['frames']
            }],
            "metadata": {
                "total_cameras": 1,
                "total_frames": len(verified_results['frames'])
            }
        }
