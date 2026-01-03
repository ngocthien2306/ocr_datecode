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
        self.inference_matcher: Optional[Any] = None
        logger.info("InferenceHandler initialized")

    def init_matcher(self, camera: 'Camera'):
        """
        Initialize inference matcher using first camera's templates

        Args:
            camera: Camera object with templates loaded

        Returns:
            SuperPointMatcherTRT instance or None
        """
        try:
            if not INFERENCE_AVAILABLE or not SuperPointMatcherTRT:
                logger.warning("Inference service not available")
                return None

            if not camera.templates:
                logger.error(f"No templates found for camera {camera.serial_number}")
                return None

            # Get first template
            template_data = camera.templates[0]
            image_url = template_data.get("image_url")

            if not image_url:
                logger.error("No template image URL")
                return None

            # Copy template to temp directory
            filename = image_url.split("/")[-1]
            backend_dir = Path(__file__).parent.parent.parent / "backend"
            source_path = backend_dir / "uploads" / "templates" / filename

            if not source_path.exists():
                logger.error(f"Template not found: {source_path}")
                return None

            temp_dir = Path("ocr_inference")
            temp_dir.mkdir(exist_ok=True)
            template_path = temp_dir / f"template_{camera.serial_number}.jpg"

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
                logger.error("No template bbox in annotations")
                return None

            # Create annotation file
            ann_json_path = temp_dir / f"annotations_manager.json"
            ann_data = {
                "_template_image": str(template_path),
                str(template_path): [template_bbox] + other_bboxes
            }

            with open(ann_json_path, "w") as f:
                json.dump(ann_data, f, indent=2)

            # Initialize matcher
            engine_path = "/home/demo/Source/ocr_datecode/weights/pipeline_fp16_small.engine"
            matcher = SuperPointMatcherTRT(
                json_path=str(ann_json_path),
                engine_path=engine_path,
                scale=1.0,
                verbose=True
            )

            logger.info(f"Inference matcher initialized for camera {camera.serial_number}")
            self.inference_matcher = matcher
            return matcher

        except Exception as e:
            logger.error(f"Error initializing inference matcher: {e}")
            import traceback
            traceback.print_exc()
            return None

    def clear_matcher(self):
        """Clear inference matcher"""
        self.inference_matcher = None
        logger.info("Inference matcher cleared")

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
            # Use first camera's first frame only (Phase 1)
            first_camera = cameras[0]
            first_frame = results[first_camera.serial_number]['frames'][0]

            logger.info(f"Running inference on camera {first_camera.serial_number} frame 0")

            # Run inference using pre-initialized matcher
            inference_result_data = "PASS"  # Default
            confidence = 0.0
            inliers = 0
            total_matches = 0

            if self.inference_matcher:
                try:
                    # Run matcher inference using match_array (in main thread - CUDA context OK)
                    match_result = self.inference_matcher.match_array(
                        target_img_array=first_frame,
                        score_threshold=0.3,
                        ransac_threshold=5.0
                    )

                    # Check if matching succeeded
                    if match_result.get('success', False):
                        # Convert numpy types to Python native types for JSON serialization
                        confidence = float(match_result.get('confidence', 0.0))
                        inliers = int(match_result.get('inliers', 0))
                        total_matches = int(match_result.get('total_matches', 0))

                        # Simple pass/fail: confidence > 0.5 and enough inliers
                        if confidence > 0.5 and inliers >= 10:
                            inference_result_data = "PASS"
                        else:
                            inference_result_data = "FAIL"
                    else:
                        inference_result_data = "FAIL"

                    logger.info(
                        f"Inference result: {inference_result_data}, "
                        f"confidence: {confidence:.2%}, "
                        f"inliers: {inliers}/{total_matches}"
                    )
                except Exception as e:
                    logger.error(f"Error running inference: {e}")
                    import traceback
                    traceback.print_exc()
                    inference_result_data = "ERROR"
            else:
                logger.warning("Inference matcher not available, skipping inference")

            # Build inference result structure
            inference_result = self._build_inference_result(
                cameras=cameras,
                results=results,
                first_camera=first_camera,
                inference_data={
                    "result": inference_result_data,
                    "confidence": confidence,
                    "inliers": inliers,
                    "total_matches": total_matches
                }
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
        first_camera: 'Camera',
        inference_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build inference result structure for backend

        Args:
            cameras: List of cameras
            results: Capture results
            first_camera: First camera (for inference)
            inference_data: Inference results (result, confidence, inliers, total_matches)

        Returns:
            Inference result dictionary
        """
        inference_result_data = inference_data["result"]
        confidence = inference_data["confidence"]
        inliers = inference_data["inliers"]
        total_matches = inference_data["total_matches"]

        # Build camera_results structure for backend
        camera_results = []
        for camera in cameras:
            camera_frames = results[camera.serial_number]['frames']

            # Build frame results (Phase 1: only process first frame of first camera)
            frame_results = []
            for idx, frame_img in enumerate(camera_frames):
                # Determine pass/fail first
                is_first_frame = (camera == first_camera and idx == 0)
                frame_pass_fail = inference_result_data if is_first_frame else "PASS"
                frame_confidence = confidence if is_first_frame else 0.0

                # Only save/encode image if FAIL or ERROR (to save storage)
                image_path = None
                image_base64 = None

                if frame_pass_fail == "FAIL" or frame_pass_fail == "ERROR":
                    # Save directly to permanent storage with recipe_id
                    image_path, image_base64 = save_and_encode_frame(
                        frame_img=frame_img,
                        serial_number=camera.serial_number,
                        recipe_id=first_camera.recipe_id,
                        pass_fail=frame_pass_fail,
                        frame_idx=idx
                    )

                # Build frame result
                frame_result = {
                    "template_name": camera.templates[idx].get('name', f'Template {idx+1}') if idx < len(camera.templates) else f'Template {idx+1}',
                    "frame_idx": idx,
                    "pass_fail": frame_pass_fail,
                    "confidence": frame_confidence,
                    "detected_regions": None,  # TODO: Add detected regions from match_result
                    "image_path": image_path,  # Changed from temp_image_path
                    "image_base64": image_base64
                }

                frame_results.append(frame_result)

            # Build camera result
            camera_result = {
                "camera_id": camera.serial_number,  # TODO: Use actual camera_id from DB
                "serial_number": camera.serial_number,
                "frames": frame_results
            }
            camera_results.append(camera_result)

        # Build inference result with correct structure for backend
        inference_result = {
            "recipe_id": first_camera.recipe_id,
            "recipe_name": first_camera.recipe_name,
            "product_pass_fail": inference_result_data,  # Overall result
            "camera_results": camera_results,
            "metadata": {
                "total_cameras": len(cameras),
                "total_frames": sum(len(r['frames']) for r in results.values()),
                "inference_stats": {
                    "confidence": confidence,
                    "inliers": inliers,
                    "total_matches": total_matches
                }
            }
        }

        return inference_result
