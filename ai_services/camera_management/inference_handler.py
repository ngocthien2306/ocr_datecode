"""
Inference Handler Module
Handles inference processing and result building
"""

import logging
import cv2
import json
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from pathlib import Path

from .utils import save_and_encode_frame

if TYPE_CHECKING:
    from .camera import Camera

logger = logging.getLogger(__name__)

# Import inference service
try:
    import sys
    ai_services_path = Path(__file__).parent.parent
    if str(ai_services_path) not in sys.path:
        sys.path.insert(0, str(ai_services_path))
    from inference_service import SuperPointMatcherTRT
    INFERENCE_AVAILABLE = True
except Exception as e:
    logging.warning(f"Inference service not available: {e}")
    INFERENCE_AVAILABLE = False
    SuperPointMatcherTRT = None


class InferenceHandler:
    """
    Handles inference processing and result building

    Features:
    - Initialize TensorRT matcher from camera templates
    - Run inference on captured frames
    - Build inference result structure for backend
    - Image encoding and compression
    """

    def __init__(self):
        """Initialize InferenceHandler"""
        self.camera_matchers: Dict[str, Any] = {}  # Map serial_number -> matcher
        self.engine_path = "/home/demo/Source/ocr_datecode/weights/pipeline_fp16_dynamic.engine"
        logger.info("InferenceHandler initialized with dynamic batch engine")

    def init_matchers(self, cameras: List['Camera']):
        """
        Initialize inference matchers for all cameras with templates
        Each camera gets its own matcher with its own template

        Args:
            cameras: List of Camera objects with templates loaded

        Returns:
            Number of matchers initialized
        """
        try:
            if not INFERENCE_AVAILABLE or not SuperPointMatcherTRT:
                logger.warning("Inference service not available")
                return 0

            temp_dir = Path("ocr_inference")
            temp_dir.mkdir(exist_ok=True)
            backend_dir = Path(__file__).parent.parent.parent / "backend"

            initialized_count = 0

            for camera in cameras:
                serial_number = camera.serial_number

                if not camera.templates:
                    logger.warning(f"No templates for camera {serial_number}, skipping")
                    continue

                # Get first template
                template_data = camera.templates[0]
                image_url = template_data.get("image_url")

                if not image_url:
                    logger.error(f"No template image URL for camera {serial_number}")
                    continue

                # Copy template to temp directory
                filename = image_url.split("/")[-1]
                source_path = backend_dir / "uploads" / "templates" / filename

                if not source_path.exists():
                    logger.error(f"Template not found: {source_path}")
                    continue

                template_path = temp_dir / f"template_{serial_number}.jpg"
                import shutil
                shutil.copy(source_path, template_path)

                # Parse annotations
                annotations = template_data.get("annotations", [])
                template_bbox = None
                other_bboxes = []

                template_img = cv2.imread(str(template_path))
                img_h, img_w = template_img.shape[:2]

                for ann in annotations:
                    ann_type = ann.get("type", "")

                    if ann_type == "template":
                        x, y = ann.get("x", 0), ann.get("y", 0)
                        w, h = ann.get("width", 0), ann.get("height", 0)

                        x1, y1 = int(x * img_w), int(y * img_h)
                        x2, y2 = int((x + w) * img_w), int((y + h) * img_h)

                        template_bbox = {
                            "type": "template",
                            "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        }

                    elif ann_type == "text" and ann.get("points"):
                        pixel_points = [
                            [int(pt[0] * img_w), int(pt[1] * img_h)]
                            for pt in ann.get("points", [])
                        ]
                        other_bboxes.append({
                            "type": ann_type,
                            "text": ann.get("text", ""),
                            "points": pixel_points
                        })

                if not template_bbox:
                    logger.error(f"No template bbox for camera {serial_number}")
                    continue

                # Create annotation file for this camera
                ann_json_path = temp_dir / f"annotations_{serial_number}.json"
                ann_data = {
                    "_template_image": str(template_path),
                    str(template_path): [template_bbox] + other_bboxes
                }

                with open(ann_json_path, "w") as f:
                    json.dump(ann_data, f, indent=2)

                # Initialize matcher for this camera
                matcher = SuperPointMatcherTRT(
                    json_path=str(ann_json_path),
                    engine_path=self.engine_path,
                    scale=1.0,
                    verbose=(initialized_count == 0)  # Only verbose for first matcher
                )

                self.camera_matchers[serial_number] = matcher
                initialized_count += 1
                logger.info(f"✅ Matcher {initialized_count} initialized for camera {serial_number}")

            logger.info(f"Initialized {initialized_count} matchers using dynamic batch engine")
            return initialized_count

        except Exception as e:
            logger.error(f"Error initializing inference matchers: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def clear_matchers(self):
        """Clear all inference matchers"""
        self.camera_matchers.clear()
        logger.info("All inference matchers cleared")

    def process_inference(
        self,
        cameras: List['Camera'],
        results: Dict[str, Any],
        emit_callback
    ):
        """
        Process inference on captured frames (runs in main thread - has CUDA context)

        Args:
            cameras: List of cameras that captured frames
            results: Capture results from trigger_cameras_group
            emit_callback: Callback to emit events

        This method is called from the event loop in main thread,
        where CUDA context is available for TensorRT inference.
        """
        try:
            # Process first frame of all cameras with matchers (up to 4)
            cameras_to_process = [c for c in cameras if c.serial_number in self.camera_matchers][:4]

            # Store inference results per camera
            camera_inference_results = {}
            overall_pass_fail = "PASS"  # Default

            logger.info(f"Running BATCH inference on {len(cameras_to_process)} cameras")

            if len(cameras_to_process) == 0:
                logger.warning("No cameras to process")
            elif len(cameras_to_process) == 1:
                # Single camera - use original match_array (no batch overhead)
                camera = cameras_to_process[0]
                serial_number = camera.serial_number
                first_frame = results[serial_number]['frames'][0]

                logger.info(f"Running single-camera inference on {serial_number}")

                inference_result_data = "PASS"
                confidence = 0.0
                inliers = 0
                total_matches = 0
                timings = {}
                transformed_bboxes = []

                matcher = self.camera_matchers.get(serial_number)
                if matcher:
                    try:
                        match_result = matcher.match_array(
                            target_img_array=first_frame,
                            score_threshold=0.3,
                            ransac_threshold=5.0
                        )

                        if match_result.get('success', False):
                            confidence = float(match_result.get('confidence', 0.0))
                            inliers = int(match_result.get('inliers', 0))
                            total_matches = int(match_result.get('total_matches', 0))
                            timings = match_result.get('timings', {})
                            transformed_bboxes = match_result.get('transformed_bboxes', [])

                            if confidence > 0.5 and inliers >= 10:
                                inference_result_data = "PASS"
                            else:
                                inference_result_data = "FAIL"
                        else:
                            inference_result_data = "FAIL"
                            timings = match_result.get('timings', {})

                        logger.info(
                            f"Camera {serial_number} inference: {inference_result_data}, "
                            f"confidence: {confidence:.2%}, "
                            f"inliers: {inliers}/{total_matches}, "
                            f"time: {timings.get('total', 0):.1f}ms"
                        )
                    except Exception as e:
                        logger.error(f"Error running inference: {e}")
                        import traceback
                        traceback.print_exc()
                        inference_result_data = "ERROR"

                camera_inference_results[serial_number] = {
                    "result": inference_result_data,
                    "confidence": confidence,
                    "inliers": inliers,
                    "total_matches": total_matches,
                    "timings": timings,
                    "transformed_bboxes": transformed_bboxes
                }

                if inference_result_data in ["FAIL", "ERROR"]:
                    overall_pass_fail = inference_result_data

            else:
                # Multiple cameras - use batch inference
                logger.info(f"Using batch inference for {len(cameras_to_process)} cameras")

                # Collect frames and matchers
                target_imgs = []
                matchers = []
                serial_numbers = []

                for camera in cameras_to_process:
                    serial_number = camera.serial_number
                    first_frame = results[serial_number]['frames'][0]
                    matcher = self.camera_matchers.get(serial_number)

                    if matcher:
                        target_imgs.append(first_frame)
                        matchers.append(matcher)
                        serial_numbers.append(serial_number)

                # Run batch inference (single TRT call for all cameras)
                try:
                    # Use first matcher to call batch inference
                    batch_result = matchers[0].match_batch(
                        target_imgs=target_imgs,
                        templates=matchers,
                        score_threshold=0.3,
                        ransac_threshold=5.0
                    )

                    if batch_result.get('success', False):
                        batch_timings = batch_result['batch_timings']
                        logger.info(
                            f"Batch inference complete: "
                            f"total={batch_timings['total']:.1f}ms, "
                            f"trt={batch_timings['trt_inference']:.1f}ms, "
                            f"cameras={len(serial_numbers)}"
                        )

                        # Process results for each camera
                        for idx, serial_number in enumerate(serial_numbers):
                            result = batch_result['results'][idx]

                            inference_result_data = "PASS"
                            confidence = 0.0
                            inliers = 0
                            total_matches = 0
                            timings = {}
                            transformed_bboxes = []

                            if result.get('success', False):
                                confidence = float(result.get('confidence', 0.0))
                                inliers = int(result.get('inliers', 0))
                                total_matches = int(result.get('total_matches', 0))
                                timings = result.get('timings', {})
                                transformed_bboxes = result.get('transformed_bboxes', [])

                                if confidence > 0.5 and inliers >= 10:
                                    inference_result_data = "PASS"
                                else:
                                    inference_result_data = "FAIL"
                            else:
                                inference_result_data = "FAIL"
                                timings = result.get('timings', {})

                            logger.info(
                                f"Camera {serial_number} result: {inference_result_data}, "
                                f"confidence: {confidence:.2%}, "
                                f"inliers: {inliers}/{total_matches}"
                            )

                            camera_inference_results[serial_number] = {
                                "result": inference_result_data,
                                "confidence": confidence,
                                "inliers": inliers,
                                "total_matches": total_matches,
                                "timings": timings,
                                "transformed_bboxes": transformed_bboxes
                            }

                            if inference_result_data in ["FAIL", "ERROR"]:
                                overall_pass_fail = inference_result_data

                    else:
                        logger.error(f"Batch inference failed: {batch_result.get('error')}")
                        # Fall back to sequential for all cameras
                        for serial_number in serial_numbers:
                            camera_inference_results[serial_number] = {
                                "result": "ERROR",
                                "confidence": 0.0,
                                "inliers": 0,
                                "total_matches": 0,
                                "timings": {},
                                "transformed_bboxes": []
                            }
                        overall_pass_fail = "ERROR"

                except Exception as e:
                    logger.error(f"Error in batch inference: {e}")
                    import traceback
                    traceback.print_exc()
                    # Mark all as ERROR
                    for serial_number in serial_numbers:
                        camera_inference_results[serial_number] = {
                            "result": "ERROR",
                            "confidence": 0.0,
                            "inliers": 0,
                            "total_matches": 0,
                            "timings": {},
                            "transformed_bboxes": []
                        }
                    overall_pass_fail = "ERROR"

            # Build inference result structure
            inference_result = self._build_inference_result(
                cameras=cameras,
                results=results,
                camera_inference_results=camera_inference_results,
                overall_pass_fail=overall_pass_fail
            )

            # Emit inference result to backend
            emit_callback("inference_result", inference_result)

            logger.info(f"Inference complete: {inference_result['product_pass_fail']}")

        except Exception as e:
            logger.error(f"Error in process_inference: {e}")
            import traceback
            traceback.print_exc()

    def _build_inference_result(
        self,
        cameras: List['Camera'],
        results: Dict[str, Any],
        camera_inference_results: Dict[str, Dict[str, Any]],
        overall_pass_fail: str
    ) -> Dict[str, Any]:
        """
        Build inference result structure for backend

        Args:
            cameras: List of cameras
            results: Capture results
            camera_inference_results: Dict mapping serial_number -> inference data
            overall_pass_fail: Overall product pass/fail status

        Returns:
            Inference result dictionary
        """
        # Build camera_results structure for backend
        camera_results = []
        for camera in cameras:
            serial_number = camera.serial_number
            camera_frames = results[serial_number]['frames']

            # Get inference results for this camera (if available)
            camera_inference = camera_inference_results.get(serial_number, {})
            has_inference = bool(camera_inference)

            # Build frame results
            frame_results = []
            for idx, frame_img in enumerate(camera_frames):
                # Determine if this is the first frame of a camera with inference results
                is_inference_frame = (has_inference and idx == 0)

                # Get inference data for this frame
                if is_inference_frame:
                    frame_pass_fail = camera_inference["result"]
                    frame_confidence = camera_inference["confidence"]
                    frame_inliers = camera_inference["inliers"]
                    frame_total_matches = camera_inference["total_matches"]
                    frame_timings = camera_inference["timings"]
                    frame_bboxes = camera_inference["transformed_bboxes"]
                else:
                    frame_pass_fail = "PASS"
                    frame_confidence = 0.0
                    frame_inliers = 0
                    frame_total_matches = 0
                    frame_timings = None
                    frame_bboxes = None

                # Only save/encode image if FAIL or ERROR (to save storage)
                image_path = None
                image_base64 = None

                if frame_pass_fail == "FAIL" or frame_pass_fail == "ERROR":
                    # Save directly to permanent storage with recipe_id
                    # Pass bbox info for inference frames to draw on image
                    image_path, image_base64 = save_and_encode_frame(
                        frame_img=frame_img,
                        serial_number=camera.serial_number,
                        recipe_id=camera.recipe_id,
                        pass_fail=frame_pass_fail,
                        frame_idx=idx,
                        transformed_bboxes=frame_bboxes,
                        confidence=frame_confidence,
                        inliers=frame_inliers,
                        total_matches=frame_total_matches
                    )

                # Build frame result
                frame_result = {
                    "template_name": camera.templates[idx].get('name', f'Template {idx+1}') if idx < len(camera.templates) else f'Template {idx+1}',
                    "frame_idx": idx,
                    "pass_fail": frame_pass_fail,
                    "confidence": frame_confidence,
                    "detected_regions": frame_bboxes,
                    "image_path": image_path,
                    "image_base64": image_base64,
                    "timings": frame_timings
                }

                frame_results.append(frame_result)

            # Build camera result
            camera_result = {
                "camera_id": camera.serial_number,  # TODO: Use actual camera_id from DB
                "serial_number": camera.serial_number,
                "frames": frame_results
            }
            camera_results.append(camera_result)

        # Aggregate inference stats from all cameras that ran inference
        all_confidences = []
        all_inliers = []
        all_total_matches = []
        all_timings = {}

        # Build detailed per-camera stats
        per_camera_detailed_stats = []
        for serial_number, camera_inference in camera_inference_results.items():
            all_confidences.append(camera_inference["confidence"])
            all_inliers.append(camera_inference["inliers"])
            all_total_matches.append(camera_inference["total_matches"])

            # Collect timing from each camera
            if not all_timings and camera_inference["timings"]:
                all_timings = camera_inference["timings"]

            # Find this camera's result to get frame stats
            camera_result = next((cr for cr in camera_results if cr["serial_number"] == serial_number), None)

            if camera_result:
                frames = camera_result.get("frames", [])
                pass_count = sum(1 for f in frames if f["pass_fail"] == "PASS")
                fail_count = sum(1 for f in frames if f["pass_fail"] == "FAIL")
                error_count = sum(1 for f in frames if f["pass_fail"] == "ERROR")

                # Calculate average confidence for all frames of this camera
                frame_confidences = [f["confidence"] for f in frames if f["confidence"] > 0]
                avg_frame_confidence = sum(frame_confidences) / len(frame_confidences) if frame_confidences else camera_inference["confidence"]

                per_camera_detailed_stats.append({
                    "serial_number": serial_number,
                    "confidence": camera_inference["confidence"],
                    "inliers": camera_inference["inliers"],
                    "total_matches": camera_inference["total_matches"],
                    "timings": camera_inference["timings"],  # Per-camera timing
                    "frame_stats": {
                        "total_frames": len(frames),
                        "pass_count": pass_count,
                        "fail_count": fail_count,
                        "error_count": error_count,
                        "avg_confidence": avg_frame_confidence
                    }
                })

        # Build inference result with correct structure for backend
        first_camera = cameras[0]  # For recipe info
        inference_result = {
            "recipe_id": first_camera.recipe_id,
            "recipe_name": first_camera.recipe_name,
            "product_pass_fail": overall_pass_fail,  # Overall result
            "camera_results": camera_results,
            "metadata": {
                "total_cameras": len(cameras),
                "total_frames": sum(len(r['frames']) for r in results.values()),
                "inference_stats": {
                    "avg_confidence": sum(all_confidences) / len(all_confidences) if all_confidences else 0.0,
                    "total_inliers": sum(all_inliers),
                    "total_matches": sum(all_total_matches),
                    "per_camera_stats": per_camera_detailed_stats,  # Enhanced stats
                    "overall_timings": all_timings  # Renamed for clarity
                }
            }
        }

        return inference_result
