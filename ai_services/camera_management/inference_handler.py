import logging
import cv2
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from pathlib import Path
import numpy as np
from .utils import save_and_encode_frame, encode_frame_for_display

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


class InferenceHandler:

    def __init__(self):
        """Initialize InferenceHandler"""
        self.camera_matchers: Dict[str, Any] = {}  # Map serial_number -> matcher
        self.engine_path = "/home/demo/Source/ocr_datecode/weights/pipeline_fp16_dynamic_480x640.engine"

        # Text recognizer for Check_Type_Product function
        self.text_recognizer = None
        self.ocr_backend = None

        # Async inference executor
        self._inference_executor = ThreadPoolExecutor(
            max_workers=2,  # Max 2 concurrent inferences
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
                    ocr_engine_path = "/home/demo/Source/ocr_datecode/languages/english/rec.engine"
                    ocr_dict_path = "/home/demo/Source/ocr_datecode/languages/english/dict.txt"
                    self.text_recognizer = get_text_recognizer_trt(
                        engine_path=ocr_engine_path,
                        dict_path=ocr_dict_path,
                        min_width=320,
                        max_width=2000
                    )
                    self.ocr_backend = "tensorrt"
                    logger.info("✅ Text recognizer initialized with TensorRT backend")

                elif ocr_backend_choice == "onnx" and ONNX_AVAILABLE:
                    # Use ONNX backend
                    import os
                    ocr_model_path = "/home/demo/Source/ocr_datecode/languages/english/rec.onnx"
                    ocr_dict_path = "/home/demo/Source/ocr_datecode/languages/english/dict.txt"

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

    def init_matchers(self, cameras: List['Camera']):
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
                # print("TEMPLATE DATA: ", template_data)
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
                crop_area = None  # Will store crop_area info if exists

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
                                f"[{serial_number}] crop_area too small ({crop_w}x{crop_h}), ignoring"
                            )
                            break

                        crop_area = {
                            "x1": x1, "y1": y1,
                            "x2": x2, "y2": y2,
                            "width": crop_w,
                            "height": crop_h
                        }

                        logger.info(
                            f"[{serial_number}] Found crop_area: ({x1}, {y1}) → ({x2}, {y2}), "
                            f"size: {crop_w}x{crop_h}"
                        )
                        break  # Only use first crop_area

                # Second pass: parse other annotations
                for ann in annotations:
                    ann_type = ann.get("type", "")
                    shape = ann.get("shape", "rectangle")

                    if ann_type == "template":
                        x, y = ann.get("x", 0), ann.get("y", 0)
                        w, h = ann.get("width", 0), ann.get("height", 0)

                        x1, y1 = int(x * img_w), int(y * img_h)
                        x2, y2 = int((x + w) * img_w), int((y + h) * img_h)

                        template_bbox = {
                            "type": "template",
                            "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        }

                    elif ann_type in ["text", "barcode", "datecode"]:
                        # Handle text, barcode, datecode with same logic
                        # Support both polygon (with points) and rectangle (with x,y,width,height)
                        pixel_points = []

                        if ann.get("points"):
                            # Polygon format
                            pixel_points = [
                                [int(pt[0] * img_w), int(pt[1] * img_h)]
                                for pt in ann.get("points", [])
                            ]
                        elif shape == "rectangle" and ann.get("x") is not None:
                            # Rectangle format - convert to polygon points
                            x, y = ann.get("x", 0), ann.get("y", 0)
                            w, h = ann.get("width", 0), ann.get("height", 0)

                            x1, y1 = int(x * img_w), int(y * img_h)
                            x2, y2 = int((x + w) * img_w), int((y + h) * img_h)

                            pixel_points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

                        if pixel_points:
                            other_bboxes.append({
                                "type": ann_type,
                                "text": ann.get("text", ""),
                                "points": pixel_points
                            })

                if not template_bbox:
                    logger.error(f"No template bbox for camera {serial_number}")
                    continue

                # Process crop_area if exists
                final_template_path = template_path
                if crop_area:
                    logger.info(f"[{serial_number}] Applying crop_area to template and annotations")

                    # Crop template image
                    template_img_cropped = template_img[
                        crop_area["y1"]:crop_area["y2"],
                        crop_area["x1"]:crop_area["x2"]
                    ]

                    # Save cropped template
                    cropped_template_path = temp_dir / f"template_{serial_number}_cropped.jpg"
                    cv2.imwrite(str(cropped_template_path), template_img_cropped)
                    final_template_path = cropped_template_path

                    logger.info(
                        f"[{serial_number}] Cropped template saved: "
                        f"{template_img_cropped.shape[1]}x{template_img_cropped.shape[0]}"
                    )

                    # Adjust template_bbox to cropped coordinates
                    offset_x = crop_area["x1"]
                    offset_y = crop_area["y1"]

                    template_bbox["points"] = [
                        [pt[0] - offset_x, pt[1] - offset_y]
                        for pt in template_bbox["points"]
                    ]

                    # Adjust other_bboxes to cropped coordinates
                    for bbox in other_bboxes:
                        bbox["points"] = [
                            [pt[0] - offset_x, pt[1] - offset_y]
                            for pt in bbox["points"]
                        ]

                    logger.info(
                        f"[{serial_number}] Adjusted {len(other_bboxes) + 1} bbox(es) to cropped coordinates"
                    )

                # Create annotation file for this camera
                ann_json_path = temp_dir / f"annotations_{serial_number}.json"
                ann_data = {
                    "_template_image": str(final_template_path),
                    str(final_template_path): [template_bbox] + other_bboxes
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

                # Store crop_area metadata in matcher for use during inference
                matcher.crop_area = crop_area

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
        camera: 'Camera'
    ) -> Dict[str, Any]:
        """
        Verify text in transformed regions match expected texts

        Args:
            frame_img: Captured frame (numpy array)
            transformed_bboxes: List of transformed bbox dicts from matcher
            expected_texts: Dict mapping region_idx -> expected_text
            camera: Camera object with function_type

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
        text_bboxes = [
            (idx, bbox) for idx, bbox in enumerate(transformed_bboxes)
            if bbox.get('type') == 'text'
        ]

        logger.info(f"[{camera.serial_number}] Verifying {len(text_bboxes)} text regions")
        # print("TEST BOXXXXXXXXXXXXXXX: ",text_bboxes)
        # print("EXPECTED TEXTS: ", expected_texts)
        for region_idx, bbox in text_bboxes:
            try:


                # Crop text region from frame
                points = bbox.get('points', [])
                if len(points) < 4:
                    logger.warning(f"[{camera.serial_number}] Invalid points for region {region_idx}")
                    all_match = False
                    verification_results.append({
                        'region_idx': region_idx,
                        'expected': bbox.get('text', ''),
                        'recognized': '',
                        'match': False,
                        'confidence': 0.0,
                        'error': 'Invalid bbox points'
                    })
                    continue

                # Crop using perspective transform
                cropped_region = crop_text_region(frame_img, points)
                path_save = "/home/demo/Source/ocr_datecode/ai_services/test_result"
                cv2.imwrite(f"{path_save}/cropped_region_{camera.serial_number}_{region_idx}.png", cropped_region)

                # Run OCR on cropped region
                logger.debug(f"[{camera.serial_number}] Running OCR with {self.ocr_backend} backend...")
                text, confidence = self.text_recognizer.recognize(cropped_region, return_confidence=True)
                recognized_text = text.strip()
                logger.debug(f"[{camera.serial_number}] OCR result: '{recognized_text}' (conf: {confidence:.2%})")


                # Compare texts (case-insensitive, strip whitespace)
                match = compare_texts(recognized_text, bbox.get('text', ''), case_sensitive=False, strip=True)

                if not match:
                    all_match = False

                verification_results.append({
                    'region_idx': region_idx,
                    'expected': bbox.get('text', ''),
                    'recognized': recognized_text,
                    'match': match,
                    'confidence': confidence
                })

                logger.info(
                    f"[{camera.serial_number}] Region {region_idx}: "
                    f"expected='{bbox.get('text', '')}', recognized='{recognized_text}', "
                    f"match={match}, conf={confidence:.2%}"
                )

            except Exception as e:
                logger.error(f"[{camera.serial_number}] Error verifying region {region_idx}: {e}")
                import traceback
                traceback.print_exc()
                all_match = False
                verification_results.append({
                    'region_idx': region_idx,
                    'expected': expected_texts.get(region_idx, ''),
                    'recognized': '',
                    'match': False,
                    'confidence': 0.0,
                    'error': str(e)
                })

        return {
            'all_match': all_match,
            'results': verification_results
        }

    def process_inference_async(
        self,
        cameras: List['Camera'],
        results: Dict[str, Any],
        emit_callback,
        group_id: int = None
    ):
        """
        Submit inference job to thread pool (non-blocking)

        Args:
            cameras: List of cameras that captured frames
            results: Capture results from trigger
            emit_callback: Callback to emit inference result
            group_id: Optional group ID for tracking

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
            group_id=group_id
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
        group_id: int = None
    ):
        """
        Synchronous inference worker (runs in thread pool)

        This is the blocking function that does actual inference work.
        """
        thread_name = threading.current_thread().name
        logger.info(
            f"[Job #{job_id}] Starting inference in {thread_name} "
            f"(group_id: {group_id})"
        )

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
                        # Get crop_area from matcher metadata
                        crop_area = getattr(matcher, 'crop_area', None)

                        # Crop frame if needed
                        frame_for_inference = self._crop_frame_if_needed(first_frame, crop_area)

                        if crop_area:
                            logger.info(
                                f"[{serial_number}] Using cropped frame: "
                                f"{frame_for_inference.shape[1]}x{frame_for_inference.shape[0]}"
                            )

                        # Run inference on (possibly cropped) frame
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

                            # Check function_type for specialized logic
                            text_verification = None
                            if camera.function_type == 'Check_Type_Product':
                                logger.info(f"Running text verification for {serial_number} (Check_Type_Product)")

                                # Verify text regions
                                text_verification = self.verify_text_regions(
                                    frame_img=first_frame,
                                    transformed_bboxes=transformed_bboxes,
                                    expected_texts=camera.expected_texts,
                                    camera=camera
                                )

                                # Determine PASS/FAIL based on text match
                                if text_verification['all_match']:
                                    inference_result_data = "PASS"
                                else:
                                    inference_result_data = "FAIL"
                                    logger.warning(
                                        f"Text verification FAILED for {serial_number}: "
                                        f"{len([r for r in text_verification['results'] if not r['match']])} mismatches"
                                    )

                                # Store verification results
                                camera_inference_results[serial_number] = {
                                    "result": inference_result_data,
                                    "confidence": confidence,
                                    "inliers": inliers,
                                    "total_matches": total_matches,
                                    "timings": timings,
                                    "transformed_bboxes": transformed_bboxes,
                                    "text_verification": text_verification
                                }

                            elif camera.function_type == 'OCR':
                                # TODO: Implement OCR logic later
                                # For now, use standard template matching
                                if confidence > 0.5 and inliers >= 10:
                                    inference_result_data = "PASS"
                                else:
                                    inference_result_data = "FAIL"

                                camera_inference_results[serial_number] = {
                                    "result": inference_result_data,
                                    "confidence": confidence,
                                    "inliers": inliers,
                                    "total_matches": total_matches,
                                    "timings": timings,
                                    "transformed_bboxes": transformed_bboxes
                                }

                            else:
                                # Default: template matching logic
                                if confidence > 0.5 and inliers >= 10:
                                    inference_result_data = "PASS"
                                else:
                                    inference_result_data = "FAIL"

                                camera_inference_results[serial_number] = {
                                    "result": inference_result_data,
                                    "confidence": confidence,
                                    "inliers": inliers,
                                    "total_matches": total_matches,
                                    "timings": timings,
                                    "transformed_bboxes": transformed_bboxes
                                }

                        else:
                            inference_result_data = "FAIL"
                            timings = match_result.get('timings', {})
                            camera_inference_results[serial_number] = {
                                "result": inference_result_data,
                                "confidence": 0.0,
                                "inliers": 0,
                                "total_matches": 0,
                                "timings": timings,
                                "transformed_bboxes": []
                            }

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
                            "result": "ERROR",
                            "confidence": 0.0,
                            "inliers": 0,
                            "total_matches": 0,
                            "timings": {},
                            "transformed_bboxes": []
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

                            if result.get('success', False):
                                confidence = float(result.get('confidence', 0.0))
                                inliers = int(result.get('inliers', 0))
                                total_matches = int(result.get('total_matches', 0))
                                timings = result.get('timings', {})
                                transformed_bboxes = result.get('transformed_bboxes', [])

                                # Check function_type for specialized logic
                                if camera.function_type == 'Check_Type_Product':
                                    logger.info(f"Running text verification for {serial_number} (Check_Type_Product)")

                                    # Verify text regions
                                    text_verification = self.verify_text_regions(
                                        frame_img=first_frame,
                                        transformed_bboxes=transformed_bboxes,
                                        expected_texts=camera.expected_texts,
                                        camera=camera
                                    )

                                    # Determine PASS/FAIL based on text match
                                    if text_verification['all_match']:
                                        inference_result_data = "PASS"
                                    else:
                                        inference_result_data = "FAIL"
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

        except Exception as e:
            logger.error(f"[Job #{job_id}] Error in inference worker: {e}")
            import traceback
            traceback.print_exc()
            raise

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

            # Get crop_area from matcher if available
            matcher = self.camera_matchers.get(serial_number)
            crop_area = getattr(matcher, 'crop_area', None) if matcher else None

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
                    frame_text_verification = camera_inference.get("text_verification")
                else:
                    frame_pass_fail = "PASS"
                    frame_confidence = 0.0
                    frame_inliers = 0
                    frame_total_matches = 0
                    frame_timings = None
                    frame_bboxes = None
                    frame_text_verification = None

                # Encode image for display
                image_path = None
                image_base64 = None

                if frame_pass_fail == "FAIL" or frame_pass_fail == "ERROR":
                    # FAIL/ERROR: Save to disk + encode base64 for display
                    image_path, image_base64 = save_and_encode_frame(
                        frame_img=frame_img,
                        serial_number=camera.serial_number,
                        recipe_id=camera.recipe_id,
                        pass_fail=frame_pass_fail,
                        frame_idx=idx,
                        transformed_bboxes=frame_bboxes,
                        confidence=frame_confidence,
                        inliers=frame_inliers,
                        total_matches=frame_total_matches,
                        crop_area=crop_area
                    )
                else:
                    # PASS: Only encode base64 for display (no disk storage)
                    image_base64 = encode_frame_for_display(
                        frame_img=frame_img,
                        transformed_bboxes=frame_bboxes,
                        confidence=frame_confidence,
                        inliers=frame_inliers,
                        total_matches=frame_total_matches,
                        crop_area=crop_area
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
                    "text_verification": frame_text_verification
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
