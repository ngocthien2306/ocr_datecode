"""
Multi Camera Pipeline

Handles inference for multiple cameras with batch processing.
All cameras are processed in a single batch inference call.
"""

import logging
import time
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from .base import InferencePipelineTemplate, PipelineContext

from ..camera import Camera
import numpy as np

logger = logging.getLogger(__name__)


class MultiCameraPipeline(InferencePipelineTemplate):
    """
    Pipeline for multi-camera batch inference.

    Processes multiple cameras in a single batch inference call
    for optimal performance.
    """

    MAX_CAMERAS_PER_BATCH = 4

    def __init__(
        self,
        crop_func=None,
        transform_func=None,
        result_builder=None,
        save_and_encode_func=None,
        encode_display_func=None
    ):
        """
        Initialize MultiCameraPipeline.

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
        """Filter cameras that have matchers"""
        cameras_with_matchers = [
            c for c in context.cameras
            if c.serial_number in context.camera_matchers
        ][:self.MAX_CAMERAS_PER_BATCH]

        if not cameras_with_matchers:
            logger.warning(f"[Job #{context.job_id}] No cameras with matchers")
            return False

        context.cameras_to_process = cameras_with_matchers
        logger.info(
            f"[Job #{context.job_id}] Prepared {len(cameras_with_matchers)} cameras for batch"
        )
        return True

    def preprocess(self, context: PipelineContext) -> Optional[Dict[str, Any]]:
        """Prepare all camera frames for batch inference"""
        target_imgs = []
        matchers = []
        serial_numbers = []
        crop_areas = []

        for camera in context.cameras_to_process:
            serial_number = camera.serial_number
            frames = context.results.get(serial_number, {}).get('frames', [])

            if not frames:
                logger.warning(f"[{serial_number}] No frames, skipping")
                continue

            matcher = context.camera_matchers.get(serial_number)
            if isinstance(matcher, list):
                matcher = matcher[0]  # Use first matcher for multi-camera

            crop_area = getattr(matcher, 'crop_area', None)
            frame = frames[0]  # First frame for each camera

            if self._crop_func and crop_area:
                frame_for_inference = self._crop_func(frame, crop_area)
            else:
                frame_for_inference = frame

            target_imgs.append(frame_for_inference)
            matchers.append(matcher)
            serial_numbers.append(serial_number)
            crop_areas.append(crop_area)

        if not target_imgs:
            logger.error(f"[Job #{context.job_id}] No valid frames to process")
            return None

        return {
            'target_imgs': target_imgs,
            'matchers': matchers,
            'serial_numbers': serial_numbers,
            'crop_areas': crop_areas
        }

    def run_inference(
        self,
        context: PipelineContext,
        preprocessed: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Run batch inference for all cameras"""
        target_imgs = preprocessed['target_imgs']
        matchers = preprocessed['matchers']
        serial_numbers = preprocessed['serial_numbers']
        crop_areas = preprocessed['crop_areas']

        try:
            # Per-recipe matching-confidence gate (shared across all cameras in batch).
            # Use cameras_to_process[0]; cameras under the same recipe carry identical matching_conf.
            cameras_in_batch = context.cameras_to_process or []
            matching_conf = 0.20
            if cameras_in_batch:
                matching_conf = float(getattr(cameras_in_batch[0], 'matching_conf', 0.20) or 0.20)

            # Single batch inference call
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

            batch_timings = batch_result.get('batch_timings', {})
            logger.info(
                f"Batch inference complete: "
                f"total={batch_timings.get('total', 0):.1f}ms, "
                f"trt={batch_timings.get('trt_inference', 0):.1f}ms, "
                f"cameras={len(serial_numbers)}"
            )

            # Transform results
            transformed_results = {}
            for idx, serial_number in enumerate(serial_numbers):
                result = batch_result['results'][idx]
                crop_area = crop_areas[idx]

                if self._transform_func:
                    result = self._transform_func(result, crop_area)

                transformed_results[serial_number] = result

            return {
                'batch_result': batch_result,
                'transformed_results': transformed_results,
                'timings': batch_timings
            }

        except Exception as e:
            logger.error(f"Batch inference error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def verify_results(
        self,
        context: PipelineContext,
        inference_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify results for all cameras"""
        transformed_results = inference_results['transformed_results']

        # Collect OCR tasks for batch processing
        ocr_tasks = []

        for camera in context.cameras_to_process:
            serial_number = camera.serial_number
            result = transformed_results.get(serial_number, {})
            frames = context.results.get(serial_number, {}).get('frames', [])

            if not result.get('success') or not frames:
                continue

            if camera.function_type == 'Check_Type_Product':
                frame_expected_texts = camera.expected_texts.get(0, {})
                if frame_expected_texts:
                    # Get matcher for sim check (template_img + original_bboxes)
                    matcher = context.camera_matchers.get(serial_number)
                    if isinstance(matcher, list):
                        matcher = matcher[0]

                    ocr_task = {
                        'serial_number': serial_number,
                        'frame_img': frames[0],
                        'transformed_bboxes': result.get('transformed_bboxes', []),
                        'expected_texts': frame_expected_texts,
                        'camera': camera,
                        'recognition_threshold': getattr(camera, 'recognition_threshold', 0.5),
                    }

                    # Attach template_img + original_bboxes for sim check
                    if matcher and hasattr(matcher, 'template_img') and hasattr(matcher, 'other_bboxes'):
                        ocr_task['template_img'] = matcher.template_img
                        ocr_task['original_bboxes'] = matcher.other_bboxes

                    ocr_tasks.append(ocr_task)

        # Batch OCR
        batch_ocr_results = {}
        t_ocr_ms = 0.0
        ocr_serial_numbers = set()
        if ocr_tasks and context.text_verification_service:
            ocr_serial_numbers = {task['serial_number'] for task in ocr_tasks}
            t_ocr_start = time.perf_counter()
            batch_ocr_results = context.text_verification_service.batch_verify_multi_camera(
                ocr_tasks
            )
            t_ocr_ms = (time.perf_counter() - t_ocr_start) * 1000

        # Batch Product Verification
        batch_product_results = self._batch_verify_products(
            context, transformed_results
        )

        # Batch Template Verification (PARALLEL)
        batch_template_results = self._batch_verify_templates(
            context, transformed_results
        )

        # Build verified results for each camera
        verified_results = {}
        overall = 'PASS'

        for camera in context.cameras_to_process:
            serial_number = camera.serial_number
            result = transformed_results.get(serial_number, {})
            matcher = context.camera_matchers.get(serial_number)
            if isinstance(matcher, list):
                matcher = matcher[0]
            frames = context.results.get(serial_number, {}).get('frames', [])

            # batch_verify_multi_camera now returns {serial: {text: {...}, char: {...}}}
            verification = batch_ocr_results.get(serial_number) or {}
            text_verification = verification.get('text')
            char_verification = verification.get('char')

            camera_result = {
                'result': 'PASS' if result.get('success') else 'FAIL',
                'confidence': float(result.get('confidence', 0.0)),
                'inliers': int(result.get('inliers', 0)),
                'total_matches': int(result.get('total_matches', 0)),
                'timings': result.get('timings', {}),
                'transformed_bboxes': result.get('transformed_bboxes', []),
                'text_verification': text_verification,
                'char_verification': char_verification,
                'template_verification': None
            }

            # Update bboxes with recognized text (text/datecode only)
            if text_verification and context.text_verification_service:
                context.text_verification_service.update_bboxes_with_recognized_text(
                    camera_result['transformed_bboxes'],
                    text_verification
                )

                for text_result in text_verification.get('results', []):
                    annotation_idx = text_result.get('annotation_idx')
                    if annotation_idx is not None and not text_result.get('match', False):
                        for bbox in camera_result['transformed_bboxes']:
                            if (bbox.get('type') in ['text', 'datecode'] and
                                bbox.get('annotation_index') == annotation_idx):
                                bbox['verification_status'] = 'fail'

            # Mark failed char bboxes with verification_status
            if char_verification:
                for char_result in char_verification.get('results', []):
                    annotation_idx = char_result.get('annotation_idx')
                    if annotation_idx is not None and not char_result.get('match', False):
                        for bbox in camera_result['transformed_bboxes']:
                            if (bbox.get('type') == 'char' and
                                bbox.get('annotation_index') == annotation_idx):
                                bbox['verification_status'] = 'fail'

            # Template verification (UPDATED - use batch results)
            camera_result['template_verification'] = None
            if result.get('success') and context.template_verification_service and frames:
                template_verification = batch_template_results.get(serial_number)
                camera_result['template_verification'] = template_verification

                # Mark failed template bbox with verification_status
                if template_verification and not template_verification.get('match', True):
                    for bbox in camera_result['transformed_bboxes']:
                        if bbox.get('type') == 'template':
                            bbox['verification_status'] = 'fail'

            # Product verification (UPDATED - use batch results)
            camera_result['product_verification'] = None
            if result.get('success') and context.product_verification_service and frames:
                product_verification = batch_product_results.get(serial_number)
                camera_result['product_verification'] = product_verification

                # Mark failed product bbox with verification_status
                if (product_verification and
                    not product_verification.get('skipped', True) and
                    not product_verification.get('match', True)):
                    for bbox in camera_result['transformed_bboxes']:
                        if bbox.get('type') == 'product':
                            bbox['verification_status'] = 'fail'

            # Merge verification timings into camera_result['timings']
            timings = dict(camera_result['timings'])
            if serial_number in ocr_serial_numbers:
                timings['ocr_ms'] = t_ocr_ms
            template_verif = camera_result.get('template_verification') or {}
            if template_verif.get('timing'):
                timings['template_verification_ms'] = template_verif['timing'].get('total_ms', 0.0)
            product_verif = camera_result.get('product_verification') or {}
            if product_verif.get('timing'):
                timings['product_verification_ms'] = product_verif['timing'].get('total', 0.0)
            timings['total'] = (
                timings.get('total', 0.0)
                + timings.get('ocr_ms', 0.0)
                + timings.get('template_verification_ms', 0.0)
                + timings.get('product_verification_ms', 0.0)
            )
            camera_result['timings'] = timings

            # Determine pass/fail (text AND char both must pass)
            text_ok = (camera_result['text_verification'] is None or
                      camera_result['text_verification'].get('all_match', True))
            char_ok = (camera_result.get('char_verification') is None or
                      camera_result['char_verification'].get('all_match', True))
            template_ok = (camera_result['template_verification'] is None or
                         camera_result['template_verification'].get('match', True))
            product_ok = (camera_result['product_verification'] is None or
                         camera_result['product_verification'].get('skipped', True) or
                         camera_result['product_verification'].get('match', True))

            if not (text_ok and char_ok and template_ok and product_ok) or not result.get('success'):
                camera_result['result'] = 'FAIL'
                overall = 'FAIL'

            verified_results[serial_number] = camera_result

        return {
            'camera_results': verified_results,
            'overall_result': overall
        }

    def _batch_verify_products(
        self,
        context: PipelineContext,
        transformed_results: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Batch verify products for all cameras.

        Args:
            context: Pipeline context
            transformed_results: Dict mapping serial_number -> inference result

        Returns:
            Dict mapping serial_number -> product verification result
        """
        if not context.product_verification_service:
            return {}

        # Collect all cameras/frames that need product verification
        frames_data = []
        serial_numbers = []

        for camera in context.cameras_to_process:
            serial_number = camera.serial_number
            result = transformed_results.get(serial_number, {})
            frames = context.results.get(serial_number, {}).get('frames', [])

            # Get center_offset_threshold from first template (multi-camera uses first template only)
            center_offset_threshold = None
            center_offset_threshold_left = 50.0
            center_offset_threshold_right = 50.0
            center_offset_unit = 'px'
            wrinkle_area = None
            wrinkle_min_area = 0.0
            wrinkle_max_area = 0.0
            if camera.templates and len(camera.templates) > 0:
                template = camera.templates[0]
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

            if result.get('success') and frames:
                frames_data.append({
                    'frame_img': frames[0],
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
                serial_numbers.append(serial_number)

        if not frames_data:
            return {}

        # Check how many frames need verification
        frames_needing_verification = [
            data for data in frames_data
            if context.product_verification_service.should_verify_frame(data['transformed_bboxes'])
        ]

        logger.debug(
            f"[Job #{context.job_id}] Product verification: "
            f"{len(frames_needing_verification)}/{len(frames_data)} cameras need verification"
        )

        # Batch verify
        try:
            import time
            t_start = time.perf_counter()
            verification_results = context.product_verification_service.verify_batch(frames_data)
            t_elapsed = (time.perf_counter() - t_start) * 1000

            # Map results back to serial numbers
            result_map = {}
            for idx, serial_number in enumerate(serial_numbers):
                if idx < len(verification_results):
                    result_map[serial_number] = verification_results[idx]

            # Log timing details if available (find first valid timing)
            timing = None
            for result in verification_results:
                if result and result.get('timing') and result['timing'].get('total', 0) > 0:
                    timing = result['timing']
                    break

            if timing:
                logger.info(
                    f"[Job #{context.job_id}] Product verification complete: "
                    f"total={timing.get('total', 0):.1f}ms, "
                    f"yolo={timing.get('yolo_inference', 0):.1f}ms, "
                    f"cameras={timing.get('frames_checked', 0)}/{timing.get('frames_total', 0)}"
                )

            return result_map

        except Exception as e:
            logger.error(f"[Job #{context.job_id}] Batch product verification failed: {e}")
            import traceback
            traceback.print_exc()
            # Return error results
            return {
                sn: {
                    'match': False,
                    'skipped': False,
                    'error': f'Batch verification failed: {str(e)}',
                    'timing': {'total': 0.0}
                }
                for sn in serial_numbers
            }

    def _batch_verify_templates(
        self,
        context: PipelineContext,
        transformed_results: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Batch verify templates for all cameras (PARALLEL).

        Args:
            context: Pipeline context
            transformed_results: Dict mapping serial_number -> inference result

        Returns:
            Dict mapping serial_number -> template verification result
        """
        if not context.template_verification_service:
            return {}

        # Collect all template verification tasks
        verification_tasks = []
        serial_numbers = []

        for camera in context.cameras_to_process:
            serial_number = camera.serial_number
            result = transformed_results.get(serial_number, {})
            frames = context.results.get(serial_number, {}).get('frames', [])

            # Get matcher
            matcher = context.camera_matchers.get(serial_number)
            if isinstance(matcher, list):
                matcher = matcher[0]

            # Check if template verification is needed
            if (result.get('success') and
                matcher and
                hasattr(matcher, 'template_img') and
                hasattr(matcher, 'template_bbox') and
                frames):

                verification_tasks.append({
                    'frame_img': frames[0],
                    'template_img': matcher.template_img,
                    'transformed_bboxes': result.get('transformed_bboxes', []),
                    'original_template_bbox': matcher.template_bbox,
                    'camera': camera
                })
                serial_numbers.append(serial_number)

        if not verification_tasks:
            return {}

        # Batch verify with PARALLEL execution
        try:
            verification_results = context.template_verification_service.batch_verify_templates(
                verification_tasks,
                parallel=True  # Enable parallel processing
            )

            # Map results back to serial numbers
            result_map = {}
            for idx, serial_number in enumerate(serial_numbers):
                if idx < len(verification_results):
                    result_map[serial_number] = verification_results[idx]

            return result_map

        except Exception as e:
            logger.error(f"[Job #{context.job_id}] Batch template verification failed: {e}")
            import traceback
            traceback.print_exc()
            # Return error results
            return {
                sn: {
                    'match': False,
                    'similarity': 0.0,
                    'threshold': 0.0,
                    'method': 'ERROR',
                    'error': f'Batch verification failed: {str(e)}'
                }
                for sn in serial_numbers
            }

    def postprocess(
        self,
        context: PipelineContext,
        verified_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build final result structure"""
        context.camera_inference_results = verified_results['camera_results']
        context.overall_pass_fail = verified_results['overall_result']

        # Build result using builder if available
        if self._result_builder:
            return self._result_builder.from_cameras(
                cameras=context.cameras,
                results=context.results,
                camera_inference_results=context.camera_inference_results,
                overall_pass_fail=context.overall_pass_fail,
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
            "product_pass_fail": context.overall_pass_fail,
            "camera_results": [
                {
                    "serial_number": sn,
                    "frames": [result]
                }
                for sn, result in verified_results['camera_results'].items()
            ],
            "metadata": {
                "total_cameras": len(context.cameras_to_process),
                "total_frames": len(context.cameras_to_process)
            }
        }
