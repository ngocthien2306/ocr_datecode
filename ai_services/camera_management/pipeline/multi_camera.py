"""
Multi Camera Pipeline

Handles inference for multiple cameras with batch processing.
All cameras are processed in a single batch inference call.
"""

import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from .base import InferencePipelineTemplate, PipelineContext

if TYPE_CHECKING:
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
            # Single batch inference call
            batch_result = matchers[0].match_batch(
                target_imgs=target_imgs,
                templates=matchers,
                score_threshold=0.3,
                ransac_threshold=5.0
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
                    ocr_tasks.append({
                        'serial_number': serial_number,
                        'frame_img': frames[0],
                        'transformed_bboxes': result.get('transformed_bboxes', []),
                        'expected_texts': frame_expected_texts,
                        'camera': camera,
                        'recognition_threshold': getattr(camera, 'recognition_threshold', 0.5)
                    })

        # Batch OCR
        batch_ocr_results = {}
        if ocr_tasks and context.text_verification_service:
            batch_ocr_results = context.text_verification_service.batch_verify_multi_camera(
                ocr_tasks
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

            camera_result = {
                'result': 'PASS' if result.get('success') else 'FAIL',
                'confidence': float(result.get('confidence', 0.0)),
                'inliers': int(result.get('inliers', 0)),
                'total_matches': int(result.get('total_matches', 0)),
                'timings': result.get('timings', {}),
                'transformed_bboxes': result.get('transformed_bboxes', []),
                'text_verification': batch_ocr_results.get(serial_number),
                'template_verification': None
            }

            # Update bboxes with recognized text
            if camera_result['text_verification'] and context.text_verification_service:
                context.text_verification_service.update_bboxes_with_recognized_text(
                    camera_result['transformed_bboxes'],
                    camera_result['text_verification']
                )

                # Mark failed text bboxes with verification_status
                text_verification = camera_result['text_verification']
                if text_verification and text_verification.get('results'):
                    for text_result in text_verification['results']:
                        annotation_idx = text_result.get('annotation_idx')
                        if annotation_idx is not None and not text_result.get('match', False):
                            # Find bbox with this annotation_index and mark as fail
                            for bbox in camera_result['transformed_bboxes']:
                                if (bbox.get('type') == 'text' and
                                    bbox.get('annotation_index') == annotation_idx):
                                    bbox['verification_status'] = 'fail'

            # Template verification
            if (result.get('success') and
                context.template_verification_service and
                hasattr(matcher, 'template_img') and
                hasattr(matcher, 'template_bbox') and
                frames):

                template_verification = context.template_verification_service.verify_template_regions(
                    frame_img=frames[0],
                    template_img=matcher.template_img,
                    transformed_bboxes=camera_result['transformed_bboxes'],
                    original_template_bbox=matcher.template_bbox,
                    camera=camera
                )
                camera_result['template_verification'] = template_verification

                # Mark failed template bbox with verification_status
                if template_verification and not template_verification.get('match', True):
                    for bbox in camera_result['transformed_bboxes']:
                        if bbox.get('type') == 'template':
                            bbox['verification_status'] = 'fail'

            # Product verification
            camera_result['product_verification'] = None
            if (result.get('success') and context.product_verification_service and frames):
                product_verification = context.product_verification_service.verify_product_alignment(
                    frame_img=frames[0],
                    transformed_bboxes=camera_result['transformed_bboxes'],
                    camera=camera
                )
                camera_result['product_verification'] = product_verification

                # Mark failed product bbox with verification_status
                if (product_verification and
                    not product_verification.get('skipped', True) and
                    not product_verification.get('match', True)):
                    for bbox in camera_result['transformed_bboxes']:
                        if bbox.get('type') == 'product1':
                            bbox['verification_status'] = 'fail'

            # Determine pass/fail
            text_ok = (camera_result['text_verification'] is None or
                      camera_result['text_verification'].get('all_match', True))
            template_ok = (camera_result['template_verification'] is None or
                         camera_result['template_verification'].get('match', True))
            product_ok = (camera_result['product_verification'] is None or
                         camera_result['product_verification'].get('skipped', True) or
                         camera_result['product_verification'].get('match', True))

            if not (text_ok and template_ok and product_ok) or not result.get('success'):
                camera_result['result'] = 'FAIL'
                overall = 'FAIL'

            verified_results[serial_number] = camera_result

        return {
            'camera_results': verified_results,
            'overall_result': overall
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
                encode_display_func=self._encode_display_func
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
