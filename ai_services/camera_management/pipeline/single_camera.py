"""
Single Camera Pipeline

Handles inference for single camera scenarios:
- Single camera with single template
- Single camera with multiple templates (batch processing)
"""

import logging
import time
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from .base import InferencePipelineTemplate, PipelineContext


from ..camera import Camera
import numpy as np

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
            # Use first matcher for batch inference
            batch_result = matchers[0].match_batch(
                target_imgs=target_imgs,
                templates=matchers,
                score_threshold=0.3,
                ransac_threshold=5.0
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
        """Verify text and template for each frame"""
        preprocessed = context.cameras_to_process[0]  # Actually stored in preprocess return
        # Re-get preprocessed data
        camera = context.cameras_to_process[0]
        serial_number = camera.serial_number
        frames = context.results[serial_number].get('frames', [])
        matchers = context.camera_matchers.get(serial_number)
        if not isinstance(matchers, list):
            matchers = [matchers]

        transformed_results = inference_results['transformed_results']

        # Batch product verification (NEW)
        # Collect all frames that need product verification and verify them in batch
        product_verification_results = self._batch_verify_products(
            context, camera, frames, transformed_results
        )

        verified_frames = []

        for idx, result in enumerate(transformed_results):
            frame = frames[idx]
            matcher = matchers[idx] if idx < len(matchers) else matchers[0]

            frame_result = {
                'frame_idx': idx,
                'result': 'PASS' if result.get('success') else 'FAIL',
                'confidence': float(result.get('confidence', 0.0)),
                'inliers': int(result.get('inliers', 0)),
                'total_matches': int(result.get('total_matches', 0)),
                'timings': result.get('timings', {}),
                'transformed_bboxes': result.get('transformed_bboxes', []),
                'text_verification': None,
                'template_verification': None
            }

            # Text verification
            t_ocr_ms = 0.0
            if (result.get('success') and
                camera.function_type in ('Check_Type_Product', 'Check_Color') and
                context.text_verification_service):

                frame_expected_texts = camera.expected_texts.get(idx, {})

                if frame_expected_texts:
                    # Get template_img + original_bboxes for sim check
                    sim_template_img = None
                    sim_original_bboxes = None
                    if hasattr(matcher, 'template_img') and hasattr(matcher, 'other_bboxes'):
                        sim_template_img = matcher.template_img
                        sim_original_bboxes = matcher.other_bboxes

                    t_ocr_start = time.perf_counter()
                    text_verification = context.text_verification_service.verify_text_regions(
                        frame_img=frame,
                        transformed_bboxes=frame_result['transformed_bboxes'],
                        expected_texts=frame_expected_texts,
                        camera=camera,
                        template_img=sim_template_img,
                        original_bboxes=sim_original_bboxes,
                    )
                    t_ocr_ms = (time.perf_counter() - t_ocr_start) * 1000
                    frame_result['text_verification'] = text_verification

                    # Update bboxes with recognized text
                    context.text_verification_service.update_bboxes_with_recognized_text(
                        frame_result['transformed_bboxes'],
                        text_verification
                    )

                    # Mark failed text bboxes with verification_status
                    if text_verification and text_verification.get('results'):
                        for text_result in text_verification['results']:
                            annotation_idx = text_result.get('annotation_idx')
                            if annotation_idx is not None and not text_result.get('match', False):
                                # Find bbox with this annotation_index and mark as fail
                                for bbox in frame_result['transformed_bboxes']:
                                    if (bbox.get('type')in ['text', 'datecode'] and
                                        bbox.get('annotation_index') == annotation_idx):
                                        bbox['verification_status'] = 'fail'

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

                # Mark failed template bbox with verification_status
                if template_verification and not template_verification.get('match', True):
                    for bbox in frame_result['transformed_bboxes']:
                        if bbox.get('type') == 'template':
                            bbox['verification_status'] = 'fail'

            # Product verification — skip for Check_Color
            frame_result['product_verification'] = None
            if (result.get('success') and
                context.product_verification_service and
                camera.function_type != 'Check_Color' and
                idx < len(product_verification_results)):

                product_verification = product_verification_results[idx]
                frame_result['product_verification'] = product_verification

                # Mark failed product bbox with verification_status
                if (product_verification and
                    not product_verification.get('skipped', True) and
                    not product_verification.get('match', True)):
                    for bbox in frame_result['transformed_bboxes']:
                        if bbox.get('type') == 'product':
                            bbox['verification_status'] = 'fail'

            # Merge verification timings into frame_result['timings']
            timings = dict(frame_result['timings'])
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
            frame_result['timings'] = timings

            # Determine final pass/fail
            text_ok = (frame_result['text_verification'] is None or
                      frame_result['text_verification'].get('all_match', True))
            template_ok = (frame_result['template_verification'] is None or
                         frame_result['template_verification'].get('match', True))
            product_ok = (frame_result['product_verification'] is None or
                         frame_result['product_verification'].get('skipped', True) or
                         frame_result['product_verification'].get('match', True))

            if not (text_ok and template_ok and product_ok):
                frame_result['result'] = 'FAIL'

            verified_frames.append(frame_result)

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
            wrinkle_area = None
            if camera.templates and idx < len(camera.templates):
                template = camera.templates[idx]
                center_offset_threshold = template.get('center_offset_threshold', 50.0)
                center_offset_threshold_left = template.get('center_offset_threshold_left', 50.0)
                center_offset_threshold_right = template.get('center_offset_threshold_right', 50.0)
                wrinkle_area = template.get('wrinkle_area', None)

            if result.get('success') and idx < len(frames):
                frames_data.append({
                    'frame_img': frames[idx],
                    'transformed_bboxes': result.get('transformed_bboxes', []),
                    'camera': camera,
                    'center_offset_threshold': center_offset_threshold,
                    'center_offset_threshold_left': center_offset_threshold_left,
                    'center_offset_threshold_right': center_offset_threshold_right,
                    'wrinkle_area': wrinkle_area
                })
            else:
                frames_data.append({
                    'frame_img': None,
                    'transformed_bboxes': [],
                    'camera': camera,
                    'center_offset_threshold': center_offset_threshold,
                    'wrinkle_area': wrinkle_area
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
