import logging
import os
import cv2
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from pathlib import Path
import numpy as np
from .utils import save_and_encode_frame, encode_frame_for_display
from camera_management.ocr_utils import crop_text_region

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

# Import text recognizers (both TensorRT and ONNX)
try:
    from .text_recognizer_trt import get_text_recognizer as get_text_recognizer_trt, TENSORRT_AVAILABLE
    TRT_AVAILABLE = TENSORRT_AVAILABLE
except Exception as e:
    logging.warning(f"TensorRT text recognizer not available: {e}")
    TRT_AVAILABLE = False
    get_text_recognizer_trt = None

try:
    from .text_recognizer import TextRecognizer as TextRecognizerONNX
    ONNX_AVAILABLE = True
except Exception as e:
    logging.warning(f"ONNX text recognizer not available: {e}")
    ONNX_AVAILABLE = False
    TextRecognizerONNX = None

try:
    from .ocr_utils import crop_text_region, compare_texts
    OCR_UTILS_AVAILABLE = True
except Exception as e:
    logging.warning(f"OCR utils not available: {e}")
    OCR_UTILS_AVAILABLE = False

# OCR is available if either backend is available
OCR_AVAILABLE = TRT_AVAILABLE or ONNX_AVAILABLE

home = os.environ.get('HOME')

class InferenceHandler:

    def __init__(self, reject_scheduler=None):
        """
        Initialize InferenceHandler

        Args:
            reject_scheduler: RejectScheduler instance for scheduling reject actions
        """
        self.camera_matchers: Dict[str, Any] = {}  # Map serial_number -> matcher
        self.engine_path = f"{home}/Source/ocr_datecode/weights/pipeline_fp16_dynamic_480x640.engine"

        # Text recognizer for Check_Type_Product function
        self.text_recognizer = None
        self.ocr_backend = None

        # Reject scheduler reference
        self.reject_scheduler = reject_scheduler

        # Async inference executor
        self._inference_executor = ThreadPoolExecutor(
            max_workers=1,  # Max 2 concurrent inferences
            thread_name_prefix="InferenceWorker"
        )

        # Job tracking
        self._active_jobs = {}  # Map job_id -> Future
        self._job_counter = 0
        self._job_lock = threading.Lock()

        # Statistics
        self._inference_stats = {
            'total_submitted': 0,
            'total_completed': 0,
            'total_failed': 0,
            'max_queue_depth': 0
        }
        self._stats_lock = threading.Lock()

        if OCR_AVAILABLE:
            try:
                import os
                # Get OCR backend from environment: "tensorrt", "onnx", or "auto" (default)
                ocr_backend_choice = os.getenv("OCR_BACKEND", "auto").lower()

                logger.info(f"OCR Backend availability - TensorRT: {TRT_AVAILABLE}, ONNX: {ONNX_AVAILABLE}")


                ocr_backend_choice = "onnx"
                # Initialize selected backend
                if ocr_backend_choice == "tensorrt" and TRT_AVAILABLE:
                    # Use TensorRT backend

                    ocr_engine_path = f"{home}/Source/ocr_datecode/languages/english/rec.engine"
                    ocr_dict_path = f"{home}/Source/ocr_datecode/languages/english/dict.txt"
                    self.text_recognizer = get_text_recognizer_trt(
                        engine_path=ocr_engine_path,
                        dict_path=ocr_dict_path,
                        min_width=320,
                        max_width=2000
                    )
                    self.ocr_backend = "tensorrt"
                    logger.info("✅ Text recognizer initialized with TensorRT backend")

                elif ocr_backend_choice == "onnx" and ONNX_AVAILABLE:

                    ocr_model_path = f"{home}/Source/ocr_datecode/languages/english/rec.onnx"
                    ocr_dict_path = f"{home}/Source/ocr_datecode/languages/english/dict.txt"

                    # Check file existence
                    if not os.path.exists(ocr_model_path):
                        raise FileNotFoundError(f"ONNX model not found: {ocr_model_path}")
                    if not os.path.exists(ocr_dict_path):
                        raise FileNotFoundError(f"Dict file not found: {ocr_dict_path}")

                    logger.info(f"Loading ONNX model from: {ocr_model_path}")
                    self.text_recognizer = TextRecognizerONNX(
                        model_path=ocr_model_path,
                        dict_path=ocr_dict_path,
                        use_gpu=True
                    )
                    self.ocr_backend = "onnx"
                    logger.info("✅ Text recognizer initialized with ONNX backend")
                    logger.info(f"   Model: {ocr_model_path}")
                    logger.info(f"   Dict: {ocr_dict_path}")

                else:
                    logger.warning(f"Requested backend '{ocr_backend_choice}' not available")
                    self.text_recognizer = None

            except Exception as e:
                logger.warning(f"Failed to initialize text recognizer: {e}")
                import traceback
                traceback.print_exc()
                self.text_recognizer = None

        logger.info("InferenceHandler initialized with dynamic batch engine")

    def _create_single_matcher(
        self,
        camera: 'Camera',
        template_data: Dict[str, Any],
        template_idx: int,
        temp_dir: Path,
        backend_dir: Path,
        verbose: bool = False
    ) -> Optional[Any]:
        """
        Create a single matcher for one template

        Args:
            camera: Camera instance
            template_data: Template data dict (from camera.templates[idx])
            template_idx: Template index
            temp_dir: Temporary directory for matcher files
            backend_dir: Backend directory path
            verbose: Verbose logging

        Returns:
            SuperPointMatcherTRT instance or None if failed
        """
        try:
            serial_number = camera.serial_number

            if not template_data:
                logger.error(f"[{serial_number}] Template {template_idx}: No template data")
                return None

            image_url = template_data.get("image_url")

            if not image_url:
                logger.error(f"[{serial_number}] Template {template_idx}: No image_url")
                return None

            # Copy template to temp directory
            filename = image_url.split("/")[-1]
            source_path = backend_dir / "uploads" / "templates" / filename

            if not source_path.exists():
                logger.error(f"[{serial_number}] Template {template_idx}: Not found at {source_path}")
                return None

            template_path = temp_dir / f"template_{serial_number}_t{template_idx}.jpg"
            import shutil
            shutil.copy(source_path, template_path)

            # Parse annotations
            annotations = template_data.get("annotations", [])
            template_bbox = None
            other_bboxes = []
            crop_area = None

            template_img = cv2.imread(str(template_path))
            img_h, img_w = template_img.shape[:2]

            # First pass: find crop_area annotation
            for ann in annotations:
                if ann.get("type") == "crop_area":
                    x, y = ann.get("x", 0), ann.get("y", 0)
                    w, h = ann.get("width", 1), ann.get("height", 1)

                    # Convert to pixel coordinates
                    x1, y1 = int(x * img_w), int(y * img_h)
                    x2, y2 = int((x + w) * img_w), int((y + h) * img_h)

                    # Clamp to image boundaries
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_w, x2)
                    y2 = min(img_h, y2)

                    # Validate crop area size
                    crop_w = x2 - x1
                    crop_h = y2 - y1
                    if crop_w < 100 or crop_h < 100:
                        logger.warning(
                            f"[{serial_number}] Template {template_idx}: crop_area too small ({crop_w}x{crop_h}), ignoring"
                        )
                        break

                    crop_area = {
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2,
                        "width": crop_w,
                        "height": crop_h
                    }

                    logger.info(
                        f"[{serial_number}] Template {template_idx}: Found crop_area: "
                        f"({x1}, {y1}) → ({x2}, {y2}), size: {crop_w}x{crop_h}"
                    )
                    break

            # Second pass: parse other annotations (keep original index for expected_texts lookup)
            for ann_idx, ann in enumerate(annotations):
                ann_type = ann.get("type", "")
                shape = ann.get("shape", "rectangle")
                conf = ann.get("conf", 0.8)

                if ann_type == "template":
                    x, y = ann.get("x", 0), ann.get("y", 0)
                    w, h = ann.get("width", 0), ann.get("height", 0)

                    x1, y1 = int(x * img_w), int(y * img_h)
                    x2, y2 = int((x + w) * img_w), int((y + h) * img_h)

                    template_bbox = {
                        "type": "template",
                        "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        "annotation_index": ann_idx,  # Keep original index
                        "conf": conf
                    }

                elif ann_type in ["text", "barcode", "datecode"]:
                    pixel_points = []

                    if ann.get("points"):
                        pixel_points = [
                            [int(pt[0] * img_w), int(pt[1] * img_h)]
                            for pt in ann.get("points", [])
                        ]
                    elif shape == "rectangle" and ann.get("x") is not None:
                        x, y = ann.get("x", 0), ann.get("y", 0)
                        w, h = ann.get("width", 0), ann.get("height", 0)

                        x1, y1 = int(x * img_w), int(y * img_h)
                        x2, y2 = int((x + w) * img_w), int((y + h) * img_h)

                        pixel_points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

                    if pixel_points:
                        other_bboxes.append({
                            "type": ann_type,
                            "text": ann.get("text", ""),
                            "points": pixel_points,
                            "annotation_index": ann_idx, # Keep original index for expected_texts lookup
                            "conf": conf
                        })

            if not template_bbox:
                logger.error(f"[{serial_number}] Template {template_idx}: No template bbox")
                return None

            # Process crop_area if exists
            final_template_path = template_path
            if crop_area:
                logger.info(f"[{serial_number}] Template {template_idx}: Applying crop_area")

                # Crop template image
                template_img_cropped = template_img[
                    crop_area["y1"]:crop_area["y2"],
                    crop_area["x1"]:crop_area["x2"]
                ]

                # Save cropped template
                cropped_template_path = temp_dir / f"template_{serial_number}_t{template_idx}_cropped.jpg"
                cv2.imwrite(str(cropped_template_path), template_img_cropped)
                final_template_path = cropped_template_path

                logger.info(
                    f"[{serial_number}] Template {template_idx}: Cropped template saved: "
                    f"{template_img_cropped.shape[1]}x{template_img_cropped.shape[0]}"
                )

                # Adjust bboxes to cropped coordinates
                offset_x = crop_area["x1"]
                offset_y = crop_area["y1"]

                template_bbox["points"] = [
                    [pt[0] - offset_x, pt[1] - offset_y]
                    for pt in template_bbox["points"]
                ]

                for bbox in other_bboxes:
                    bbox["points"] = [
                        [pt[0] - offset_x, pt[1] - offset_y]
                        for pt in bbox["points"]
                    ]

                logger.info(
                    f"[{serial_number}] Template {template_idx}: "
                    f"Adjusted {len(other_bboxes) + 1} bbox(es) to cropped coordinates"
                )

            # Create annotation file for this template
            ann_json_path = temp_dir / f"annotations_{serial_number}_t{template_idx}.json"
            ann_data = {
                "_template_image": str(final_template_path),
                str(final_template_path): [template_bbox] + other_bboxes
            }

            with open(ann_json_path, "w") as f:
                json.dump(ann_data, f, indent=2)

            # Initialize matcher
            matcher = SuperPointMatcherTRT(
                json_path=str(ann_json_path),
                engine_path=self.engine_path,
                scale=1.0,
                verbose=verbose
            )

            # Store crop_area metadata
            matcher.crop_area = crop_area

            return matcher

        except Exception as e:
            logger.error(f"Error creating matcher for template {template_idx}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def init_matchers(self, cameras: List['Camera']):
        try:
            if not INFERENCE_AVAILABLE or not SuperPointMatcherTRT:
                logger.warning("Inference service not available")
                return 0

            temp_dir = Path("ocr_inference")
            temp_dir.mkdir(exist_ok=True)
            backend_dir = Path(__file__).parent.parent.parent / "backend"

            initialized_count = 0
            is_single_camera = len(cameras) == 1

            for camera in cameras:
                serial_number = camera.serial_number
                num_templates = len(camera.templates)

                if not camera.templates:
                    logger.warning(f"No templates for camera {serial_number}, skipping")
                    continue

                # Detect scenario: single camera with multiple templates
                if is_single_camera and num_templates > 1:
                    # Single camera, multiple templates → create matcher for EACH template
                    logger.info(
                        f"[{serial_number}] Single camera scenario: "
                        f"{num_templates} templates → creating {num_templates} matchers"
                    )

                    matchers_list = []
                    for template_idx, template_data in enumerate(camera.templates):
                        verbose = (template_idx == 0)  # Only verbose for first template
                        matcher = self._create_single_matcher(
                            camera=camera,
                            template_data=template_data,
                            template_idx=template_idx,
                            temp_dir=temp_dir,
                            backend_dir=backend_dir,
                            verbose=verbose
                        )

                        if matcher:
                            matchers_list.append(matcher)
                            logger.info(
                                f"✅ [{serial_number}] Matcher {template_idx + 1}/{num_templates} initialized"
                            )
                        else:
                            logger.error(
                                f"❌ [{serial_number}] Failed to create matcher {template_idx + 1}/{num_templates}"
                            )

                    if matchers_list:
                        # Store as list for single camera multi-template
                        self.camera_matchers[serial_number] = matchers_list
                        initialized_count += len(matchers_list)
                        logger.info(
                            f"[{serial_number}] ✅ {len(matchers_list)}/{num_templates} matchers initialized"
                        )
                    else:
                        logger.error(f"[{serial_number}] Failed to initialize any matchers")

                else:
                    # Multiple cameras OR single camera with single template
                    # Use first template only (existing logic)
                    logger.info(
                        f"[{serial_number}] Standard scenario: using first template only"
                    )

                    template_data = camera.templates[0]
                    verbose = (initialized_count == 0)  # Only verbose for first matcher

                    matcher = self._create_single_matcher(
                        camera=camera,
                        template_data=template_data,
                        template_idx=0,
                        temp_dir=temp_dir,
                        backend_dir=backend_dir,
                        verbose=verbose
                    )

                    if matcher:
                        # Store as single matcher (backward compatible)
                        self.camera_matchers[serial_number] = matcher
                        initialized_count += 1
                        logger.info(f"✅ [{serial_number}] Matcher initialized (first template)")

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

    def shutdown(self):
        """Shutdown inference handler and cleanup resources"""
        logger.info("Shutting down InferenceHandler...")

        # Shutdown executor (wait for pending jobs)
        self._inference_executor.shutdown(wait=True, cancel_futures=False)

        # Clear matchers
        self.clear_matchers()

        # Log final stats
        with self._stats_lock:
            logger.info(
                f"📊 [INFERENCE STATS] Total: {self._inference_stats['total_submitted']}, "
                f"Completed: {self._inference_stats['total_completed']}, "
                f"Failed: {self._inference_stats['total_failed']}, "
                f"Max Queue: {self._inference_stats['max_queue_depth']}"
            )

        logger.info("InferenceHandler shutdown complete")

    def _crop_frame_if_needed(
        self,
        frame: 'np.ndarray',
        crop_area: Optional[Dict[str, int]]
    ) -> 'np.ndarray':
        """
        Crop frame if crop_area exists

        Args:
            frame: Input frame (full image)
            crop_area: Crop area dict with x1, y1, x2, y2

        Returns:
            Cropped frame or original frame if no crop_area
        """
        if crop_area is None:
            return frame

        return frame[
            crop_area["y1"]:crop_area["y2"],
            crop_area["x1"]:crop_area["x2"]
        ]

    def _transform_results_to_full_image(
        self,
        match_result: Dict[str, Any],
        crop_area: Optional[Dict[str, int]]
    ) -> Dict[str, Any]:
        """
        Transform inference results from cropped coordinates to full image coordinates

        Args:
            match_result: Result from matcher (with cropped coordinates)
            crop_area: Crop area dict with x1, y1 offsets

        Returns:
            Transformed result with full image coordinates
        """
        if crop_area is None:
            return match_result

        offset_x = crop_area["x1"]
        offset_y = crop_area["y1"]

        # Transform matched_bbox if exists
        if "matched_bbox" in match_result and match_result["matched_bbox"]:
            match_result["matched_bbox"]["points"] = [
                [pt[0] + offset_x, pt[1] + offset_y]
                for pt in match_result["matched_bbox"]["points"]
            ]

        # Transform transformed_bboxes (text regions, etc.)
        if "transformed_bboxes" in match_result:
            for bbox in match_result["transformed_bboxes"]:
                if "points" in bbox and bbox["points"]:
                    bbox["points"] = [
                        [pt[0] + offset_x, pt[1] + offset_y]
                        for pt in bbox["points"]
                    ]

        return match_result

    def verify_text_regions(
        self,
        frame_img: 'np.ndarray',
        transformed_bboxes: List[Dict[str, Any]],
        expected_texts: Dict[int, str],
        camera: 'Camera',
        recognition_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Verify text in transformed regions match expected texts

        Args:
            frame_img: Captured frame (numpy array)
            transformed_bboxes: List of transformed bbox dicts from matcher
            expected_texts: Dict mapping region_idx -> expected_text
            camera: Camera object with function_type
            recognition_threshold: Minimum OCR confidence threshold (default: 0.5)

        Returns:
            {
                'all_match': bool,
                'results': [
                    {
                        'region_idx': 0,
                        'expected': '123',
                        'recognized': '123',
                        'match': True,
                        'confidence': 0.95
                    },
                    ...
                ]
            }
        """
        if not self.text_recognizer or not OCR_AVAILABLE:
            logger.warning("OCR model not available, skipping text verification")
            return {'all_match': False, 'results': []}

        verification_results = []
        all_match = True

        # Filter only text type bboxes
        # Use annotation_index from bbox metadata (preserves original annotation index)
        text_bboxes = [
            bbox for bbox in transformed_bboxes
            if bbox.get('type') == 'text'
        ]
        
        
        logger.info(f"[{camera.serial_number}] Verifying {len(text_bboxes)} text regions")
        logger.info(f"[{camera.serial_number}] Expected texts dict: {expected_texts}")
        logger.info(f"[{camera.serial_number}] Text bbox annotation indices: {[bbox.get('annotation_index') for bbox in text_bboxes]}")

        for bbox in text_bboxes:
            try:
                # Get annotation_index from bbox (this is the original index in annotations list)
                annotation_idx = bbox.get('annotation_index')
                conf_threshold = bbox.get('conf', 0.8)
                if annotation_idx is None:
                    logger.warning(f"[{camera.serial_number}] Bbox missing annotation_index, skipping")
                    continue

                # Get expected text using annotation_index
                expected_text = expected_texts.get(annotation_idx, '')
                logger.info(
                    f"[{camera.serial_number}] Processing annotation {annotation_idx}: expected_text='{expected_text}'"
                )

                # Crop text region from frame
                points = bbox.get('points', [])
                if len(points) < 4:
                    logger.warning(f"[{camera.serial_number}] Invalid points for annotation {annotation_idx}")
                    all_match = False
                    verification_results.append({
                        'annotation_idx': annotation_idx,
                        'expected': expected_text,
                        'recognized': '',
                        'match': False,
                        'confidence': 0.0,
                        'threshold': conf_threshold,
                        'error': 'Invalid bbox points'
                    })
                    continue

                # Crop using perspective transform
                cropped_region = crop_text_region(frame_img, points)
                path_save = f"{home}/Source/ocr_datecode/ai_services/test_result"
                cv2.imwrite(f"{path_save}/cropped_region_{camera.serial_number}_{annotation_idx}.png", cropped_region)

                # Run OCR on cropped region
                logger.debug(f"[{camera.serial_number}] Running OCR with {self.ocr_backend} backend...")
                text, confidence = self.text_recognizer.recognize(cropped_region, return_confidence=True)

                recognized_text = text.strip()
                logger.debug(f"[{camera.serial_number}] OCR result: '{recognized_text}' (conf: {confidence:.2%})")

                # Check if confidence meets threshold first
                if confidence < conf_threshold:
                    logger.warning(
                        f"[{camera.serial_number}] Annotation {annotation_idx}: "
                        f"Low confidence {confidence:.2%} < threshold {conf_threshold:.2%}, treating as NO MATCH"
                    )
                    match = False
                else:
                    # Compare texts (case-insensitive, strip whitespace)
                    match = compare_texts(recognized_text, expected_text, case_sensitive=False, strip=True)

                if not match:
                    all_match = False

                verification_results.append({
                    'annotation_idx': annotation_idx,
                    'expected': expected_text,
                    'recognized': recognized_text,
                    'match': match,
                    'confidence': confidence,
                    'threshold': conf_threshold,

                })

                logger.info(
                    f"[{camera.serial_number}] Annotation {annotation_idx}: "
                    f"expected='{expected_text}', recognized='{recognized_text}', "
                    f"match={match}, conf={confidence:.2%}"
                )

            except Exception as e:
                logger.error(f"[{camera.serial_number}] Error verifying annotation {annotation_idx}: {e}")
                import traceback
                traceback.print_exc()
                all_match = False
                verification_results.append({
                    'annotation_idx': annotation_idx,
                    'expected': expected_text,
                    'recognized': '',
                    'match': False,
                    'confidence': 0.0,
                    'threshold': conf_threshold,
                    'error': str(e)
                })

        return {
            'all_match': all_match,
            'results': verification_results
        }

    def _update_bboxes_with_recognized_text(
        self,
        transformed_bboxes: List[Dict[str, Any]],
        text_verification: Dict[str, Any]
    ) -> None:
        """
        Update transformed_bboxes with recognized text from text_verification.
        This modifies transformed_bboxes in-place to replace expected text with OCR result.

        Args:
            transformed_bboxes: List of bbox dicts (modified in-place)
            text_verification: Result from verify_text_regions containing recognized texts
        """
        if not text_verification or not text_verification.get('results'):
            return

        # Create lookup map: annotation_idx -> recognized_text
        recognized_map = {}
        for text_result in text_verification['results']:
            annotation_idx = text_result.get('annotation_idx')
            recognized_text = text_result.get('recognized', '')
            if annotation_idx is not None:
                recognized_map[annotation_idx] = recognized_text

        # Update text bboxes with recognized text
        for bbox in transformed_bboxes:
            if bbox.get('type') == 'text':
                annotation_idx = bbox.get('annotation_index')
                if annotation_idx is not None and annotation_idx in recognized_map:
                    # Replace expected text with recognized text
                    bbox['text'] = recognized_map[annotation_idx]

    def verify_template_regions(
        self,
        frame_img: 'np.ndarray',
        template_img: 'np.ndarray',
        transformed_bboxes: List[Dict[str, Any]],
        original_template_bbox: Dict[str, Any],
        camera: 'Camera',
        similarity_threshold: float = 0.85,
        method: int = cv2.TM_CCOEFF_NORMED
    ) -> Dict[str, Any]:

        try:
            # Filter only template type bboxes
            template_bboxes = [
                bbox for bbox in transformed_bboxes
                if bbox.get('type') == 'template'
            ]

            print("TRANSFORMED_BOXES: ", template_bboxes)
            try:
                similarity_threshold = template_bboxes[0]['conf'] if template_bboxes else similarity_threshold
            except Exception as e:
                similarity_threshold = similarity_threshold
                
            logger.info(f"[{camera.serial_number}] Verifying {len(template_bboxes)} template regions")

            if not template_bboxes:
                logger.warning(f"[{camera.serial_number}] No template bbox found in transformed_bboxes")
                return {
                    'match': False,
                    'similarity': 0.0,
                    'threshold': similarity_threshold,
                    'method': 'TM_CCOEFF_NORMED',
                    'template_bbox': None,
                    'error': 'No template bbox found'
                }

            # Get the transformed template bbox (for target crop)
            transformed_template_bbox = template_bboxes[0]
            transformed_points = transformed_template_bbox.get('points', [])

            if len(transformed_points) < 4:
                logger.warning(f"[{camera.serial_number}] Invalid transformed template bbox points")
                return {
                    'match': False,
                    'similarity': 0.0,
                    'threshold': similarity_threshold,
                    'method': 'TM_CCOEFF_NORMED',
                    'template_bbox': transformed_template_bbox,
                    'error': 'Invalid transformed bbox points'
                }

            # Get original template bbox points (for template crop)
            original_points = original_template_bbox.get('points', [])
            if len(original_points) < 4:
                logger.warning(f"[{camera.serial_number}] Invalid original template bbox points")
                return {
                    'match': False,
                    'similarity': 0.0,
                    'threshold': similarity_threshold,
                    'method': 'TM_CCOEFF_NORMED',
                    'template_bbox': transformed_template_bbox,
                    'error': 'Invalid original bbox points'
                }

            # Crop template region from both images using perspective transform
            # IMPORTANT:
            # - Template crop: Use ORIGINAL bbox points from template image
            # - Target crop: Use TRANSFORMED bbox points (after homography) from target image

            cropped_template = crop_text_region(template_img, original_points)
            cropped_target = crop_text_region(frame_img, transformed_points)

            path_save = f"{home}/Source/ocr_datecode/ai_services/test_result"
            cv2.imwrite(f"{path_save}/template_crop_{camera.serial_number}.png", cropped_template)
            cv2.imwrite(f"{path_save}/target_crop_{camera.serial_number}.png", cropped_target)

            # Ensure both crops are the same size
            if cropped_template.shape != cropped_target.shape:
                logger.warning(
                    f"[{camera.serial_number}] Template and target crop size mismatch: "
                    f"{cropped_template.shape} vs {cropped_target.shape}"
                )
                # Resize target to match template
                cropped_target = cv2.resize(
                    cropped_target,
                    (cropped_template.shape[1], cropped_template.shape[0])
                )

            # Convert to grayscale for template matching (more robust)
            if len(cropped_template.shape) == 3:
                template_gray = cv2.cvtColor(cropped_template, cv2.COLOR_BGR2GRAY)
            else:
                template_gray = cropped_template

            if len(cropped_target.shape) == 3:
                target_gray = cv2.cvtColor(cropped_target, cv2.COLOR_BGR2GRAY)
            else:
                target_gray = cropped_target

            # Use cv2.matchTemplate for pixel-level comparison
            # Since both images are the same size, result will be a single value
            result = cv2.matchTemplate(target_gray, template_gray, method)

            # Get similarity score
            if method == cv2.TM_SQDIFF_NORMED:
                # For SQDIFF, lower is better, so invert: similarity = 1 - score
                similarity_score = float(1.0 - result[0, 0])
            else:
                # For CCOEFF_NORMED and CCORR_NORMED, higher is better
                similarity_score = float(result[0, 0])

            # Determine if it matches threshold
            match = bool(similarity_score >= similarity_threshold)

            method_name = {
                cv2.TM_CCOEFF_NORMED: 'TM_CCOEFF_NORMED',
                cv2.TM_CCORR_NORMED: 'TM_CCORR_NORMED',
                cv2.TM_SQDIFF_NORMED: 'TM_SQDIFF_NORMED'
            }.get(method, 'UNKNOWN')

            logger.info(
                f"[{camera.serial_number}] Template verification: "
                f"similarity={similarity_score:.4f}, threshold={similarity_threshold}, "
                f"match={match}, method={method_name}"
            )

            return {
                'match': match,
                'similarity': similarity_score,
                'threshold': similarity_threshold,
                'method': method_name,
                'template_bbox': transformed_template_bbox
            }

        except Exception as e:
            logger.error(f"[{camera.serial_number}] Error verifying template region: {e}")
            import traceback
            traceback.print_exc()
            return {
                'match': False,
                'similarity': 0.0,
                'threshold': similarity_threshold,
                'method': 'FAIL',
                'template_bbox': None,
                'error': str(e)
            }

    def process_inference_async(
        self,
        cameras: List['Camera'],
        results: Dict[str, Any],
        emit_callback,
        group_id: int = None,
        T_capture_complete: float = None
    ):
        """
        Submit inference job to thread pool (non-blocking)

        Args:
            cameras: List of cameras that captured frames
            results: Capture results from trigger
            emit_callback: Callback to emit inference result
            group_id: Optional group ID for tracking
            T_capture_complete: Time when cameras finished capturing (for reject timing)

        Returns:
            job_id: Unique job identifier
        """
        # Generate job ID
        with self._job_lock:
            self._job_counter += 1
            job_id = self._job_counter

        # Update stats
        with self._stats_lock:
            self._inference_stats['total_submitted'] += 1
            queue_depth = len(self._active_jobs)
            if queue_depth > self._inference_stats['max_queue_depth']:
                self._inference_stats['max_queue_depth'] = queue_depth

        logger.info(
            f"[Job #{job_id}] Submitting inference for {len(cameras)} camera(s) "
            f"(queue depth: {queue_depth})"
        )

        # Submit job to executor
        future = self._inference_executor.submit(
            self._do_inference_sync,
            job_id=job_id,
            cameras=cameras,
            results=results,
            emit_callback=emit_callback,
            group_id=group_id,
            T_capture_complete=T_capture_complete
        )

        # Track active job
        with self._job_lock:
            self._active_jobs[job_id] = future

        # Add callback to cleanup when done
        future.add_done_callback(lambda f: self._cleanup_job(job_id, f))

        return job_id

    def _cleanup_job(self, job_id: int, future):
        """Cleanup completed job"""
        with self._job_lock:
            if job_id in self._active_jobs:
                del self._active_jobs[job_id]

        # Update stats
        with self._stats_lock:
            if future.exception():
                self._inference_stats['total_failed'] += 1
                logger.error(f"[Job #{job_id}] Failed with exception: {future.exception()}")
            else:
                self._inference_stats['total_completed'] += 1
                logger.info(f"[Job #{job_id}] Completed successfully")

    def _do_inference_sync(
        self,
        job_id: int,
        cameras: List['Camera'],
        results: Dict[str, Any],
        emit_callback,
        group_id: int = None,
        T_capture_complete: float = None
    ):
        """
        Synchronous inference worker (runs in thread pool)

        This is the blocking function that does actual inference work.
        """
        import time

        thread_name = threading.current_thread().name
        logger.info(
            f"[Job #{job_id}] Starting inference in {thread_name} "
            f"(group_id: {group_id})"
        )

        # Record inference start time
        T_inference_start = time.time()

        try:
            # Process first frame of all cameras with matchers (up to 4)
            cameras_to_process = [c for c in cameras if c.serial_number in self.camera_matchers][:4]

            # Store inference results per camera
            camera_inference_results = {}
            overall_pass_fail = "PASS"  # Default

            logger.info(f"[Job #{job_id}] Running BATCH inference on {len(cameras_to_process)} cameras")

            if len(cameras_to_process) == 0:
                logger.warning("No cameras to process")
            elif len(cameras_to_process) == 1:
                # Single camera - check if multi-template or single template
                camera = cameras_to_process[0]
                serial_number = camera.serial_number
                camera_frames = results[serial_number]['frames']

                matcher_or_list = self.camera_matchers.get(serial_number)

                if isinstance(matcher_or_list, list):
                    # ✅ SINGLE CAMERA, MULTIPLE TEMPLATES - USE BATCH INFERENCE
                    matchers_list = matcher_or_list
                    num_frames = len(camera_frames)
                    num_matchers = len(matchers_list)

                    logger.info(
                        f"[Job #{job_id}] Single camera BATCH inference: "
                        f"{num_frames} frames with {num_matchers} templates"
                    )

                    if num_frames != num_matchers:
                        logger.error(
                            f"[Job #{job_id}] Frame/template mismatch: "
                            f"{num_frames} frames vs {num_matchers} templates"
                        )
                        overall_pass_fail = "FAIL"
                        camera_inference_results[serial_number] = {
                            'overall_result': 'FAIL',
                            'frames': [{
                                'frame_idx': idx,
                                'result': 'FAIL',
                                'error': 'Frame/template count mismatch',
                                'confidence': 0.0,
                                'inliers': 0,
                                'total_matches': 0,
                                'timings': {},
                                'transformed_bboxes': [],
                                'text_verification': None,
                                'template_verification': None
                            } for idx in range(num_frames)]
                        }
                    else:
                        # Prepare batch inputs
                        target_imgs = []
                        crop_areas = []

                        for frame_idx, (frame, matcher) in enumerate(zip(camera_frames, matchers_list)):
                            crop_area = getattr(matcher, 'crop_area', None)
                            frame_for_inference = self._crop_frame_if_needed(frame, crop_area)
                            target_imgs.append(frame_for_inference)
                            crop_areas.append(crop_area)

                            if crop_area:
                                logger.debug(
                                    f"[Frame {frame_idx}] Using cropped frame: "
                                    f"{frame_for_inference.shape[1]}x{frame_for_inference.shape[0]}"
                                )

                        try:
                            # ⭐ RUN BATCH INFERENCE (SINGLE TRT CALL!)
                            batch_result = matchers_list[0].match_batch(
                                target_imgs=target_imgs,
                                templates=matchers_list,
                                score_threshold=0.3,
                                ransac_threshold=5.0
                            )

                            if not batch_result.get('success', False):
                                logger.error(f"Batch inference failed: {batch_result.get('error')}")
                                overall_pass_fail = "FAIL"
                                camera_inference_results[serial_number] = {
                                    'overall_result': 'FAIL',
                                    'frames': [{
                                        'frame_idx': idx,
                                        'result': 'FAIL',
                                        'error': batch_result.get('error', 'Unknown error'),
                                        'confidence': 0.0,
                                        'inliers': 0,
                                        'total_matches': 0,
                                        'timings': {},
                                        'transformed_bboxes': [],
                                        'text_verification': None,
                                        'template_verification': None
                                    } for idx in range(num_frames)]
                                }
                            else:
                                logger.info(
                                    f"[Job #{job_id}] Batch inference complete: "
                                    f"total={batch_result['batch_timings']['total']:.1f}ms, "
                                    f"trt={batch_result['batch_timings']['trt_inference']:.1f}ms"
                                )

                                # Process results for each frame
                                frame_results = []
                                overall_pass_fail = "PASS"  # Default

                                for frame_idx, result in enumerate(batch_result['results']):
                                    frame = camera_frames[frame_idx]
                                    crop_area = crop_areas[frame_idx]

                                    # Transform back to full image coordinates
                                    result = self._transform_results_to_full_image(result, crop_area)

                                    # Determine pass/fail for this frame
                                    inference_result_data = "PASS"
                                    confidence = 0.0
                                    inliers = 0
                                    total_matches = 0
                                    timings = {}
                                    transformed_bboxes = []
                                    text_verification = None
                                    template_verification = None

                                    if result.get('success', False):
                                        confidence = float(result.get('confidence', 0.0))
                                        inliers = int(result.get('inliers', 0))
                                        total_matches = int(result.get('total_matches', 0))
                                        timings = result.get('timings', {})
                                        transformed_bboxes = result.get('transformed_bboxes', [])

                                        # Check function_type for text verification
                                        if camera.function_type == 'Check_Type_Product':
                                            logger.info(
                                                f"[Frame {frame_idx}] Running text verification (Check_Type_Product)"
                                            )

                                            # Get expected_texts for this specific template
                                            logger.info(
                                                f"[Frame {frame_idx}] camera.expected_texts structure: {camera.expected_texts}"
                                            )
                                            frame_expected_texts = camera.expected_texts.get(frame_idx, {})
                                            logger.info(
                                                f"[Frame {frame_idx}] Retrieved expected_texts: {frame_expected_texts}"
                                            )

                                            if not frame_expected_texts:
                                                logger.warning(
                                                    f"[Frame {frame_idx}] No expected_texts found for template {frame_idx}"
                                                )

                                            # Verify text with recognition threshold
                                            text_verification = self.verify_text_regions(
                                                frame_img=frame,
                                                transformed_bboxes=transformed_bboxes,
                                                expected_texts=frame_expected_texts,
                                                camera=camera,
                                                recognition_threshold=getattr(camera, 'recognition_threshold', 0.5)
                                            )

                                            # Update bboxes with recognized text for visualization
                                            self._update_bboxes_with_recognized_text(
                                                transformed_bboxes, text_verification
                                            )

                                            # Also verify template similarity
                                            matcher = matchers_list[frame_idx]
                                            if hasattr(matcher, 'template_img') and hasattr(matcher, 'template_bbox'):
                                                logger.info(
                                                    f"[Frame {frame_idx}] Running template verification"
                                                )
                                                template_verification = self.verify_template_regions(
                                                    frame_img=frame,
                                                    template_img=matcher.template_img,
                                                    transformed_bboxes=transformed_bboxes,
                                                    original_template_bbox=matcher.template_bbox,
                                                    camera=camera,
                                                    similarity_threshold=getattr(camera, 'matching_threshold', 0.85)
                                                )
                                            else:
                                                logger.warning(
                                                    f"[Frame {frame_idx}] Matcher missing template_img or template_bbox attribute"
                                                )

                                            # PASS/FAIL based on BOTH text match AND template similarity
                                            if text_verification['all_match'] and (template_verification is None or template_verification['match']):
                                                inference_result_data = "PASS"
                                            else:
                                                inference_result_data = "FAIL"
                                                if not text_verification['all_match']:
                                                    logger.info(f"[Frame {frame_idx}] FAIL reason: text verification failed")
                                                if template_verification and not template_verification['match']:
                                                    logger.info(
                                                        f"[Frame {frame_idx}] FAIL reason: template similarity "
                                                        f"({template_verification['similarity']:.4f}) below threshold "
                                                        f"({template_verification['threshold']})"
                                                    )

                                        else:
                                            # Default: template matching
                                            if confidence > 0.5 and inliers >= 10:
                                                inference_result_data = "PASS"
                                            else:
                                                inference_result_data = "FAIL"

                                    else:
                                        inference_result_data = "FAIL"
                                        timings = result.get('timings', {})

                                    logger.info(
                                        f"[Frame {frame_idx}] result: {inference_result_data}, "
                                        f"confidence: {confidence:.2%}, inliers: {inliers}/{total_matches}"
                                    )

                                    # Store frame result
                                    frame_results.append({
                                        'frame_idx': frame_idx,
                                        'result': inference_result_data,
                                        'confidence': confidence,
                                        'inliers': inliers,
                                        'total_matches': total_matches,
                                        'timings': timings,
                                        'transformed_bboxes': transformed_bboxes,
                                        'text_verification': text_verification,
                                        'template_verification': template_verification
                                    })

                                    # Update overall result (Option A: All must PASS)
                                    if inference_result_data in ["FAIL", "ERROR"]:
                                        overall_pass_fail = inference_result_data

                                # Store structured results
                                camera_inference_results[serial_number] = {
                                    'overall_result': overall_pass_fail,
                                    'frames': frame_results
                                }

                                logger.info(
                                    f"[Job #{job_id}] Single camera inference complete: {overall_pass_fail}"
                                )

                        except Exception as e:
                            logger.error(f"Error running batch inference: {e}")
                            import traceback
                            traceback.print_exc()
                            overall_pass_fail = "FAIL"
                            camera_inference_results[serial_number] = {
                                'overall_result': 'FAIL',
                                'frames': [{
                                    'frame_idx': idx,
                                    'result': 'FAIL',
                                    'error': str(e),
                                    'confidence': 0.0,
                                    'inliers': 0,
                                    'total_matches': 0,
                                    'timings': {},
                                    'transformed_bboxes': [],
                                    'text_verification': None,
                                    'template_verification': None
                                } for idx in range(num_frames)]
                            }

                else:
                    # ✅ SINGLE CAMERA, SINGLE TEMPLATE (EXISTING LOGIC)
                    matcher = matcher_or_list
                    first_frame = camera_frames[0]

                    logger.info(f"[Job #{job_id}] Running single-camera inference on {serial_number}")

                    inference_result_data = "PASS"
                    confidence = 0.0
                    inliers = 0
                    total_matches = 0
                    timings = {}
                    transformed_bboxes = []
                    text_verification = None

                    if matcher:
                        try:
                            # Get crop_area from matcher metadata
                            crop_area = getattr(matcher, 'crop_area', None)

                            # Crop frame if needed
                            frame_for_inference = self._crop_frame_if_needed(first_frame, crop_area)

                            if crop_area:
                                logger.info(
                                    f"[{serial_number}] Using cropped frame: "
                                    f"{frame_for_inference.shape[1]}x{frame_for_inference.shape[0]}"
                                )

                            # Run inference
                            match_result = matcher.match_array(
                                target_img_array=frame_for_inference,
                                score_threshold=0.3,
                                ransac_threshold=5.0
                            )

                            # Transform results back to full image coordinates
                            match_result = self._transform_results_to_full_image(match_result, crop_area)

                            if match_result.get('success', False):
                                confidence = float(match_result.get('confidence', 0.0))
                                inliers = int(match_result.get('inliers', 0))
                                total_matches = int(match_result.get('total_matches', 0))
                                timings = match_result.get('timings', {})
                                transformed_bboxes = match_result.get('transformed_bboxes', [])

                                # Check function_type
                                template_verification = None
                                if camera.function_type == 'Check_Type_Product':
                                    logger.info(f"Running text verification for {serial_number}")

                                    # Get expected_texts for first template (template 0)
                                    logger.info(
                                        f"[{serial_number}] camera.expected_texts structure: {camera.expected_texts}"
                                    )
                                    frame_expected_texts = camera.expected_texts.get(0, {})
                                    logger.info(
                                        f"[{serial_number}] Retrieved expected_texts for template 0: {frame_expected_texts}"
                                    )

                                    if not frame_expected_texts:
                                        logger.warning(
                                            f"[{serial_number}] No expected_texts found for template 0"
                                        )

                                    text_verification = self.verify_text_regions(
                                        frame_img=first_frame,
                                        transformed_bboxes=transformed_bboxes,
                                        expected_texts=frame_expected_texts,
                                        camera=camera,
                                        recognition_threshold=getattr(camera, 'recognition_threshold', 0.5)
                                    )

                                    # Update bboxes with recognized text for visualization
                                    self._update_bboxes_with_recognized_text(
                                        transformed_bboxes, text_verification
                                    )

                                    # Also verify template similarity
                                    if hasattr(matcher, 'template_img') and hasattr(matcher, 'template_bbox'):
                                        logger.info(f"[{serial_number}] Running template verification")
                                        template_verification = self.verify_template_regions(
                                            frame_img=first_frame,
                                            template_img=matcher.template_img,
                                            transformed_bboxes=transformed_bboxes,
                                            original_template_bbox=matcher.template_bbox,
                                            camera=camera,
                                            similarity_threshold=getattr(camera, 'matching_threshold', 0.85)
                                        )
                                    else:
                                        logger.warning(f"[{serial_number}] Matcher missing template_img or template_bbox attribute")

                                    # PASS/FAIL based on BOTH text match AND template similarity
                                    if text_verification['all_match'] and (template_verification is None or template_verification['match']):
                                        inference_result_data = "PASS"
                                    else:
                                        inference_result_data = "FAIL"
                                        if not text_verification['all_match']:
                                            logger.info(f"[{serial_number}] FAIL reason: text verification failed")
                                        if template_verification and not template_verification['match']:
                                            logger.info(
                                                f"[{serial_number}] FAIL reason: template similarity "
                                                f"({template_verification['similarity']:.4f}) below threshold "
                                                f"({template_verification['threshold']})"
                                            )

                                else:
                                    # Default: template matching
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
                            inference_result_data = "FAIL"

                        # Store result (backward compatible structure)
                        camera_inference_results[serial_number] = {
                            "result": inference_result_data,
                            "confidence": confidence,
                            "inliers": inliers,
                            "total_matches": total_matches,
                            "timings": timings,
                            "transformed_bboxes": transformed_bboxes,
                            "text_verification": text_verification,
                            "template_verification": template_verification
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
                crop_areas = []  # Store crop_area for each camera

                for camera in cameras_to_process:
                    serial_number = camera.serial_number
                    first_frame = results[serial_number]['frames'][0]
                    matcher = self.camera_matchers.get(serial_number)

                    if matcher:
                        # Get crop_area from matcher
                        crop_area = getattr(matcher, 'crop_area', None)

                        # Crop frame if needed
                        frame_for_inference = self._crop_frame_if_needed(first_frame, crop_area)

                        if crop_area:
                            logger.info(
                                f"[{serial_number}] Using cropped frame: "
                                f"{frame_for_inference.shape[1]}x{frame_for_inference.shape[0]}"
                            )

                        target_imgs.append(frame_for_inference)
                        matchers.append(matcher)
                        serial_numbers.append(serial_number)
                        crop_areas.append(crop_area)

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
                            camera = cameras_to_process[idx]
                            result = batch_result['results'][idx]
                            first_frame = results[serial_number]['frames'][0]
                            crop_area = crop_areas[idx]  # Get corresponding crop_area

                            # Transform result back to full image coordinates
                            result = self._transform_results_to_full_image(result, crop_area)

                            inference_result_data = "PASS"
                            confidence = 0.0
                            inliers = 0
                            total_matches = 0
                            timings = {}
                            transformed_bboxes = []
                            text_verification = None
                            template_verification = None

                            if result.get('success', False):
                                confidence = float(result.get('confidence', 0.0))
                                inliers = int(result.get('inliers', 0))
                                total_matches = int(result.get('total_matches', 0))
                                timings = result.get('timings', {})
                                transformed_bboxes = result.get('transformed_bboxes', [])

                                # Check function_type for specialized logic
                                if camera.function_type == 'Check_Type_Product':
                                    logger.info(f"Running text verification for {serial_number} (Check_Type_Product)")

                                    # Get expected_texts for first template (template 0)
                                    # In multi-camera batch, each camera uses first template only
                                    logger.info(
                                        f"[{serial_number}] camera.expected_texts structure: {camera.expected_texts}"
                                    )
                                    frame_expected_texts = camera.expected_texts.get(0, {})
                                    logger.info(
                                        f"[{serial_number}] Retrieved expected_texts for template 0: {frame_expected_texts}"
                                    )

                                    if not frame_expected_texts:
                                        logger.warning(
                                            f"[{serial_number}] No expected_texts found for template 0"
                                        )

                                    # Verify text regions
                                    text_verification = self.verify_text_regions(
                                        frame_img=first_frame,
                                        transformed_bboxes=transformed_bboxes,
                                        expected_texts=frame_expected_texts,
                                        camera=camera,
                                        recognition_threshold=getattr(camera, 'recognition_threshold', 0.5)
                                    )

                                    # Update bboxes with recognized text for visualization
                                    self._update_bboxes_with_recognized_text(
                                        transformed_bboxes, text_verification
                                    )

                                    # Also verify template similarity
                                    matcher = matchers[idx]
                                    if hasattr(matcher, 'template_img') and hasattr(matcher, 'template_bbox'):
                                        logger.info(f"[{serial_number}] Running template verification")
                                        template_verification = self.verify_template_regions(
                                            frame_img=first_frame,
                                            template_img=matcher.template_img,
                                            transformed_bboxes=transformed_bboxes,
                                            original_template_bbox=matcher.template_bbox,
                                            camera=camera,
                                            similarity_threshold=getattr(camera, 'matching_threshold', 0.85)
                                        )
                                    else:
                                        logger.warning(f"[{serial_number}] Matcher missing template_img or template_bbox attribute")

                                    # Determine PASS/FAIL based on BOTH text match AND template similarity
                                    if text_verification['all_match'] and (template_verification is None or template_verification['match']):
                                        inference_result_data = "PASS"
                                    else:
                                        inference_result_data = "FAIL"
                                        if not text_verification['all_match']:
                                            logger.info(f"[{serial_number}] FAIL reason: text verification failed")
                                        if template_verification and not template_verification['match']:
                                            logger.info(
                                                f"[{serial_number}] FAIL reason: template similarity "
                                                f"({template_verification['similarity']:.4f}) below threshold "
                                                f"({template_verification['threshold']})"
                                            )
                                        logger.warning(
                                            f"Text verification FAILED for {serial_number}: "
                                            f"{len([r for r in text_verification['results'] if not r['match']])} mismatches"
                                        )

                                elif camera.function_type == 'OCR':
                                    # TODO: Implement OCR logic
                                    if confidence > 0.5 and inliers >= 2000:
                                        inference_result_data = "PASS"
                                    else:
                                        inference_result_data = "PASS"

                                else:
                                    # Default: template matching
                                    if confidence > 0.5 and inliers >= 2000:
                                        inference_result_data = "PASS"
                                    else:
                                        inference_result_data = "PASS"

                            else:
                                inference_result_data = "FAIL"
                                timings = result.get('timings', {})

                            logger.info(
                                f"Camera {serial_number} result: {inference_result_data}, "
                                f"confidence: {confidence:.2%}, "
                                f"inliers: {inliers}/{total_matches}"
                            )

                            # Store results
                            camera_inference_results[serial_number] = {
                                "result": inference_result_data,
                                "confidence": confidence,
                                "inliers": inliers,
                                "total_matches": total_matches,
                                "timings": timings,
                                "transformed_bboxes": transformed_bboxes
                            }

                            # Add text verification if available
                            if text_verification:
                                camera_inference_results[serial_number]["text_verification"] = text_verification

                            # Add template verification if available
                            if template_verification:
                                camera_inference_results[serial_number]["template_verification"] = template_verification

                            if inference_result_data in ["FAIL", "ERROR"]:
                                overall_pass_fail = inference_result_data

                    else:
                        logger.error(f"Batch inference failed: {batch_result.get('error')}")
                        # Fall back to sequential for all cameras
                        for serial_number in serial_numbers:
                            camera_inference_results[serial_number] = {
                                "result": "FAIL",
                                "confidence": 0.0,
                                "inliers": 0,
                                "total_matches": 0,
                                "timings": {},
                                "transformed_bboxes": []
                            }
                        overall_pass_fail = "FAIL"

                except Exception as e:
                    logger.error(f"Error in batch inference: {e}")
                    import traceback
                    traceback.print_exc()
                    # Mark all as ERROR
                    for serial_number in serial_numbers:
                        camera_inference_results[serial_number] = {
                            "result": "FAIL",
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

            # Calculate inference time
            T_inference_done = time.time()
            inference_time = T_inference_done - T_inference_start

            logger.info(
                f"[Job #{job_id}] Inference complete: {inference_result['product_pass_fail']}, "
                f"time: {inference_time:.3f}s"
            )

            # Schedule or cancel reject based on result
            if group_id and T_capture_complete and self.reject_scheduler:
                self._handle_reject_decision(
                    group_id=group_id,
                    overall_pass_fail=overall_pass_fail,
                    T_capture_complete=T_capture_complete,
                    inference_time=inference_time,
                    cameras=cameras
                )

            # Emit inference result to backend
            emit_callback("inference_result", inference_result)

        except Exception as e:
            logger.error(f"[Job #{job_id}] Error in inference worker: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _handle_reject_decision(
        self,
        group_id: int,
        overall_pass_fail: str,
        T_capture_complete: float,
        inference_time: float,
        cameras: List['Camera']
    ):
        """
        Handle reject scheduling/cancellation based on inference result

        Args:
            group_id: Group ID for tracking
            overall_pass_fail: Overall inference result (PASS/FAIL/ERROR)
            T_capture_complete: Time when cameras finished capturing
            inference_time: Inference duration in seconds
            cameras: List of cameras (to get reject config)
        """
        try:
            if overall_pass_fail == "FAIL" or overall_pass_fail == "ERROR":
                # Get reject config from first camera (recipe-level config)
                if not cameras:
                    logger.warning(f"[Group #{group_id}] No cameras available for reject config")
                    return

                camera = cameras[0]
                delay_reject = camera.delay_reject  # ms
                do_reject_number = camera.do_reject_number

                # Schedule reject
                success = self.reject_scheduler.schedule_reject(
                    group_id=group_id,
                    T_capture_complete=T_capture_complete,
                    inference_time=inference_time,
                    delay_reject=delay_reject,
                    do_number=do_reject_number
                )

                if success:
                    logger.info(
                        f"[Group #{group_id}] Reject scheduled "
                        f"(delay_reject={delay_reject}ms, DO{do_reject_number})"
                    )
                else:
                    logger.error(
                        f"[Group #{group_id}] Failed to schedule reject "
                        f"(inference too slow: {inference_time:.3f}s)"
                    )

            else:  # PASS
                # Cancel reject if scheduled
                cancelled = self.reject_scheduler.cancel_reject(group_id)
                if cancelled:
                    logger.info(f"[Group #{group_id}] Reject cancelled (result: PASS)")

        except Exception as e:
            logger.error(f"[Group #{group_id}] Error handling reject decision: {e}")
            import traceback
            traceback.print_exc()

    def _build_inference_result(
        self,
        cameras: List['Camera'],
        results: Dict[str, Any],
        camera_inference_results: Dict[str, Dict[str, Any]],
        overall_pass_fail: str
    ) -> Dict[str, Any]:
        # Build camera_results structure for backend
        camera_results = []
        for camera in cameras:
            serial_number = camera.serial_number
            camera_frames = results[serial_number]['frames']

            # Get inference results for this camera (if available)
            camera_inference = camera_inference_results.get(serial_number, {})
            has_inference = bool(camera_inference)

            # Detect structure type
            is_multi_frame = 'frames' in camera_inference  # NEW structure for multi-template

            # Get crop_area from matcher if available
            matcher = self.camera_matchers.get(serial_number)
            if isinstance(matcher, list):
                # Multi-template: no single crop_area
                crop_area = None
            else:
                crop_area = getattr(matcher, 'crop_area', None) if matcher else None

            # Build frame results
            frame_results = []

            # Determine how many frames to process
            if is_multi_frame:
                # Multi-template: Only process frames that have inference results
                num_frames_to_process = len(camera_inference['frames'])
            else:
                # Single template: Process only the number of templates (usually 1)
                num_frames_to_process = len(camera.templates)

            # Ensure we don't exceed available frames
            num_frames_to_process = min(num_frames_to_process, len(camera_frames))

            logger.debug(
                f"[{serial_number}] Processing {num_frames_to_process} frames "
                f"(total captured: {len(camera_frames)}, templates: {len(camera.templates)})"
            )

            for idx in range(num_frames_to_process):
                frame_img = camera_frames[idx]

                # Get inference data for this frame
                if is_multi_frame:
                    # ✅ NEW: Multi-template scenario
                    frame_data = camera_inference['frames'][idx]
                    frame_pass_fail = frame_data['result']
                    frame_confidence = frame_data['confidence']
                    frame_inliers = frame_data['inliers']
                    frame_total_matches = frame_data['total_matches']
                    frame_timings = frame_data['timings']
                    frame_bboxes = frame_data['transformed_bboxes']
                    frame_text_verification = frame_data.get('text_verification')
                    frame_template_verification = frame_data.get('template_verification')

                    # Get crop_area for this frame
                    if isinstance(matcher, list) and idx < len(matcher):
                        frame_crop_area = getattr(matcher[idx], 'crop_area', None)
                    else:
                        frame_crop_area = None

                else:
                    # ✅ EXISTING: Single template scenario
                    is_inference_frame = (has_inference and idx == 0)

                    if is_inference_frame:
                        frame_pass_fail = camera_inference["result"]
                        frame_confidence = camera_inference["confidence"]
                        frame_inliers = camera_inference["inliers"]
                        frame_total_matches = camera_inference["total_matches"]
                        frame_timings = camera_inference["timings"]
                        frame_bboxes = camera_inference["transformed_bboxes"]
                        frame_text_verification = camera_inference.get("text_verification")
                        frame_template_verification = camera_inference.get("template_verification")
                    else:
                        frame_pass_fail = "PASS"
                        frame_confidence = 0.0
                        frame_inliers = 0
                        frame_total_matches = 0
                        frame_timings = None
                        frame_bboxes = None
                        frame_text_verification = None
                        frame_template_verification = None

                    frame_crop_area = crop_area

                # Encode image for display
                image_path = None
                image_base64 = None

                if frame_pass_fail == "FAIL" or frame_pass_fail == "ERROR":
                    # FAIL/ERROR: Save to disk + encode base64 for display
                    base_dir: str = f"{home}/Source/ocr_datecode/backend/uploads/inference_results"

                    image_path, image_base64 = save_and_encode_frame(
                        frame_img=frame_img,
                        serial_number=camera.serial_number,
                        recipe_id=camera.recipe_id,
                        pass_fail=frame_pass_fail,
                        base_dir=base_dir,
                        frame_idx=idx,
                        transformed_bboxes=frame_bboxes,
                        confidence=frame_confidence,
                        inliers=frame_inliers,
                        total_matches=frame_total_matches,
                        crop_area=frame_crop_area
                    )
                else:
                    # PASS: Only encode base64 for display (no disk storage)
                    image_base64 = encode_frame_for_display(
                        frame_img=frame_img,
                        transformed_bboxes=frame_bboxes,
                        confidence=frame_confidence,
                        inliers=frame_inliers,
                        total_matches=frame_total_matches,
                        crop_area=frame_crop_area
                    )
                    # image_path remains None (not saved to disk)

                # Build frame result
                frame_result = {
                    "template_name": camera.templates[idx].get('name', f'Template {idx+1}') if idx < len(camera.templates) else f'Template {idx+1}',
                    "frame_idx": idx,
                    "pass_fail": frame_pass_fail,
                    "confidence": frame_confidence,
                    "detected_regions": frame_bboxes,
                    "image_path": image_path,
                    "image_base64": image_base64,
                    "timings": frame_timings,
                    "text_verification": frame_text_verification,
                    "template_verification": frame_template_verification
                }

                frame_results.append(frame_result)

            # Build camera result
            camera_result = {
                "camera_id": camera.serial_number,  # TODO: Use actual camera_id from DB
                "serial_number": camera.serial_number,
                "delay_trigger": camera.delay_trigger,  # Add delay_trigger for frontend
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
            # Check if multi-frame structure
            if 'frames' in camera_inference:
                # Multi-template: aggregate from frames
                frame_results = camera_inference['frames']
                frame_confidences = [f['confidence'] for f in frame_results]
                frame_inliers = [f['inliers'] for f in frame_results]
                frame_total_matches = [f['total_matches'] for f in frame_results]

                avg_confidence = sum(frame_confidences) / len(frame_confidences) if frame_confidences else 0.0
                total_inliers = sum(frame_inliers)
                total_matches = sum(frame_total_matches)

                all_confidences.extend(frame_confidences)
                all_inliers.extend(frame_inliers)
                all_total_matches.extend(frame_total_matches)

                # Collect timing from first frame
                if not all_timings and frame_results and frame_results[0]["timings"]:
                    all_timings = frame_results[0]["timings"]

            else:
                # Single template: use values directly
                all_confidences.append(camera_inference["confidence"])
                all_inliers.append(camera_inference["inliers"])
                all_total_matches.append(camera_inference["total_matches"])

                avg_confidence = camera_inference["confidence"]
                total_inliers = camera_inference["inliers"]
                total_matches = camera_inference["total_matches"]

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
                frame_confidences_list = [f["confidence"] for f in frames if f["confidence"] > 0]
                avg_frame_confidence = sum(frame_confidences_list) / len(frame_confidences_list) if frame_confidences_list else avg_confidence

                per_camera_detailed_stats.append({
                    "serial_number": serial_number,
                    "confidence": avg_confidence,
                    "inliers": total_inliers,
                    "total_matches": total_matches,
                    "timings": all_timings if all_timings else {},  # Per-camera timing
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
                "total_frames": sum(
                    len(r['frames']) for r in results.values()
                    if isinstance(r, dict) and 'frames' in r
                ),
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
