"""
Single Camera Pipeline

Handles inference for single camera scenarios:
- Single camera with single template
- Single camera with multiple templates (batch processing)
"""

import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from .base import InferencePipelineTemplate, PipelineContext

if TYPE_CHECKING:
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

        # Prepare inputs
        target_imgs = []
        crop_areas = []

        for idx, (frame, matcher) in enumerate(zip(frames[:num_templates], matchers)):
            crop_area = getattr(matcher, 'crop_area', None)

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

            # Transform results back to full image coordinates
            transformed_results = []
            for idx, result in enumerate(batch_result['results']):
                crop_area = crop_areas[idx]
                if self._transform_func:
                    result = self._transform_func(result, crop_area)
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
            if (result.get('success') and
                camera.function_type == 'Check_Type_Product' and
                context.text_verification_service):

                frame_expected_texts = camera.expected_texts.get(idx, {})

                if frame_expected_texts:
                    text_verification = context.text_verification_service.verify_text_regions(
                        frame_img=frame,
                        transformed_bboxes=frame_result['transformed_bboxes'],
                        expected_texts=frame_expected_texts,
                        camera=camera
                    )
                    frame_result['text_verification'] = text_verification

                    # Update bboxes with recognized text
                    context.text_verification_service.update_bboxes_with_recognized_text(
                        frame_result['transformed_bboxes'],
                        text_verification
                    )

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

            # Determine final pass/fail
            text_ok = (frame_result['text_verification'] is None or
                      frame_result['text_verification'].get('all_match', True))
            template_ok = (frame_result['template_verification'] is None or
                         frame_result['template_verification'].get('match', True))

            if not (text_ok and template_ok):
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
                encode_display_func=self._encode_display_func
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
