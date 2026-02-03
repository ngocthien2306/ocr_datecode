import logging
import os
import cv2
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from pathlib import Path
import numpy as np
from .utils import save_and_encode_frame, encode_frame_for_display

# Import verification services
from .verification import TextVerificationService, TemplateVerificationService, ProductVerificationService

# Import OCR backend factory (Strategy Pattern)
from .ocr import OCRBackendFactory, OCRBackendType

# Import matcher factory
from .matchers import MatcherFactory

# Import result builder
from .result_builder import InferenceResultBuilder

# Import pipeline (Template Method Pattern)
from .pipeline import PipelineContext, SingleCameraPipeline, MultiCameraPipeline

if TYPE_CHECKING:
    from .camera import Camera

logger = logging.getLogger(__name__)

# Import inference service (optimized version with shared engine)
try:
    import sys
    ai_services_path = Path(__file__).parent.parent
    if str(ai_services_path) not in sys.path:
        sys.path.insert(0, str(ai_services_path))

    # Try optimized version first (shared engine - saves memory)
    try:
        from inference_engine_shared import SuperPointMatcherTRTOptimized as SuperPointMatcherTRT
        from inference_engine_shared import SuperPointEngineTRT, TemplateConfig
        USING_OPTIMIZED_ENGINE = True
        logger.info("Using optimized shared TensorRT engine")
    except ImportError:
        # Fall back to original implementation
        from inference_service import SuperPointMatcherTRT
        USING_OPTIMIZED_ENGINE = False
        logger.info("Using legacy TensorRT engine (not optimized)")

    INFERENCE_AVAILABLE = True
except Exception as e:
    logging.warning(f"Inference service not available: {e}")
    INFERENCE_AVAILABLE = False
    USING_OPTIMIZED_ENGINE = False
    SuperPointMatcherTRT = None

home = os.environ.get('HOME')

class InferenceHandler:

    def __init__(self, reject_scheduler=None):
        """
        Initialize InferenceHandler

        Args:
            reject_scheduler: RejectScheduler instance for scheduling reject actions
        """
        self.camera_matchers: Dict[str, Any] = {}  # Map serial_number -> matcher
        self.engine_path = f"{home}/Source/ocr_datecode/weights/pipeline_fp16_dynamic_480_640.engine"

        # Initialize matcher factory (for future use)
        self._matcher_factory = MatcherFactory(
            engine_path=self.engine_path,
            temp_dir=Path("ocr_inference"),
            backend_dir=Path(__file__).parent.parent.parent / "backend"
        )

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

        # Initialize OCR backend using Factory Pattern
        self._init_ocr_backend()

        # Initialize verification services
        self._init_verification_services()

        logger.info("InferenceHandler initialized with dynamic batch engine")

    def _init_ocr_backend(self):
        """
        Initialize OCR backend using Factory Pattern.

        Uses OCRBackendFactory to create the appropriate backend based on:
        - Environment variable OCR_BACKEND (tensorrt/onnx/auto)
        - Available backends on the system
        """
        try:
            # Use factory to create backend (respects OCR_BACKEND env var)
            # Force ONNX for now (can be changed via env var later)
            self._ocr_backend_instance = OCRBackendFactory.create(OCRBackendType.ONNX)

            if self._ocr_backend_instance is not None:
                self.text_recognizer = self._ocr_backend_instance
                self.ocr_backend = self._ocr_backend_instance.backend_name
                logger.info(f"OCR backend initialized: {self._ocr_backend_instance}")
            else:
                self.text_recognizer = None
                self.ocr_backend = None
                logger.warning("No OCR backend available")

        except Exception as e:
            logger.error(f"Failed to initialize OCR backend: {e}")
            import traceback
            traceback.print_exc()
            self.text_recognizer = None
            self.ocr_backend = None
            self._ocr_backend_instance = None

    def _init_verification_services(self):
        """Initialize text and template verification services"""
        # Text Verification Service
        if self.text_recognizer is not None:
            self.text_verification_service = TextVerificationService(
                text_recognizer=self.text_recognizer,
                ocr_backend=self.ocr_backend or "unknown",
                save_debug_images=True,
                debug_path=f"{home}/Source/ocr_datecode/ai_services/test_result"
            )
            logger.info(f"TextVerificationService initialized with {self.ocr_backend} backend")
        else:
            self.text_verification_service = None
            logger.warning("TextVerificationService not available (no OCR backend)")

        # Template Verification Service
        self.template_verification_service = TemplateVerificationService(
            save_debug_images=True,
            debug_path=f"{home}/Source/ocr_datecode/ai_services/test_result",
            default_threshold=0.85
        )
        logger.info("TemplateVerificationService initialized")

        # Product Verification Service with YOLO OBB
        yolo_obb_engine_path = f"{home}/Source/ocr_datecode/weights/yolo26n-ultralight-obb_fp16_dynamic.engine"

        # Configuration: save_debug_images can be "never", "on_fail", or "always"
        # Use "never" for production (fastest), "on_fail" for debugging failures
        product_debug_mode = os.environ.get('PRODUCT_VERIFICATION_DEBUG', 'never')

        self.product_verification_service = ProductVerificationService(
            engine_path=yolo_obb_engine_path,
            save_debug_images=product_debug_mode,  # "never", "on_fail", or "always"
            debug_path=f"{home}/Source/ocr_datecode/ai_services/test_result",
            angle_threshold=3.0,  # Rotation threshold in degrees
            margin_pixels=30,     # Misalignment margin in pixels
            conf_threshold=0.25   # YOLO confidence threshold
        )
        logger.info(
            f"ProductVerificationService initialized with YOLO OBB: {yolo_obb_engine_path}, "
            f"debug_mode={product_debug_mode}"
        )

    def init_matchers(self, cameras: List['Camera']):
        """
        Initialize matchers for all cameras using MatcherFactory.

        Args:
            cameras: List of Camera objects with templates

        Returns:
            Number of matchers initialized
        """
        try:
            if not self._matcher_factory.is_available:
                logger.warning("Matcher factory not available (inference service missing)")
                return 0

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
                    # Single camera, multiple templates -> create matcher for EACH template
                    logger.info(
                        f"[{serial_number}] Single camera scenario: "
                        f"{num_templates} templates -> creating {num_templates} matchers"
                    )

                    matchers_list = self._matcher_factory.create_matchers_for_camera(
                        camera=camera,
                        verbose_first=True
                    )

                    if matchers_list:
                        self.camera_matchers[serial_number] = matchers_list
                        initialized_count += len(matchers_list)
                        logger.info(
                            f"[{serial_number}] {len(matchers_list)}/{num_templates} matchers initialized"
                        )
                    else:
                        logger.error(f"[{serial_number}] Failed to initialize any matchers")

                else:
                    # Multiple cameras OR single camera with single template
                    # Use first template only
                    logger.info(
                        f"[{serial_number}] Standard scenario: using first template only"
                    )

                    verbose = (initialized_count == 0)
                    matcher = self._matcher_factory.create_matcher(
                        camera=camera,
                        template_data=camera.templates[0],
                        template_idx=0,
                        verbose=verbose
                    )

                    if matcher:
                        self.camera_matchers[serial_number] = matcher
                        initialized_count += 1
                        logger.info(f"[{serial_number}] Matcher initialized (first template)")

            if USING_OPTIMIZED_ENGINE:
                logger.info(
                    f"Initialized {initialized_count} matchers using SHARED TensorRT engine "
                    f"(1 engine, {initialized_count} template configs)"
                )
            else:
                logger.info(
                    f"Initialized {initialized_count} matchers using LEGACY mode "
                    f"({initialized_count} separate engines - high memory usage)"
                )
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
        Verify text in transformed regions match expected texts.
        Delegates to TextVerificationService.

        Args:
            frame_img: Captured frame (numpy array)
            transformed_bboxes: List of transformed bbox dicts from matcher
            expected_texts: Dict mapping region_idx -> expected_text
            camera: Camera object with function_type
            recognition_threshold: Minimum OCR confidence threshold (default: 0.5)

        Returns:
            {
                'all_match': bool,
                'results': [...]
            }
        """
        if self.text_verification_service is None:
            logger.warning("TextVerificationService not available, skipping text verification")
            return {'all_match': False, 'results': []}

        return self.text_verification_service.verify_text_regions(
            frame_img=frame_img,
            transformed_bboxes=transformed_bboxes,
            expected_texts=expected_texts,
            camera=camera,
            recognition_threshold=recognition_threshold
        )

    def _update_bboxes_with_recognized_text(
        self,
        transformed_bboxes: List[Dict[str, Any]],
        text_verification: Dict[str, Any]
    ) -> None:
        """
        Update transformed_bboxes with recognized text from text_verification.
        Delegates to TextVerificationService static method.

        Args:
            transformed_bboxes: List of bbox dicts (modified in-place)
            text_verification: Result from verify_text_regions containing recognized texts
        """
        TextVerificationService.update_bboxes_with_recognized_text(
            transformed_bboxes, text_verification
        )

    def batch_verify_text_regions_multi_camera(
        self,
        ocr_tasks: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Batch OCR verification for ALL cameras at once.
        Delegates to TextVerificationService.

        Args:
            ocr_tasks: List of tasks, each containing:
                {
                    'serial_number': str,
                    'frame_img': np.ndarray,
                    'transformed_bboxes': list,
                    'expected_texts': dict,
                    'camera': Camera,
                    'recognition_threshold': float
                }

        Returns:
            Dict mapping serial_number -> verification_result
        """
        if self.text_verification_service is None:
            logger.warning("TextVerificationService not available, skipping batch text verification")
            return {task['serial_number']: {'all_match': False, 'results': []} for task in ocr_tasks}

        return self.text_verification_service.batch_verify_multi_camera(ocr_tasks)

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
        """
        Verify template region similarity between frame and reference template.
        Delegates to TemplateVerificationService.

        Args:
            frame_img: Captured frame (numpy array)
            template_img: Reference template image (numpy array)
            transformed_bboxes: List of transformed bbox dicts from matcher
            original_template_bbox: Original template bbox from reference
            camera: Camera object for logging
            similarity_threshold: Minimum similarity score (0-1)
            method: OpenCV template matching method

        Returns:
            {
                'match': bool,
                'similarity': float,
                'threshold': float,
                'method': str,
                'template_bbox': dict,
                'error': str (optional)
            }
        """
        return self.template_verification_service.verify_template_regions(
            frame_img=frame_img,
            template_img=template_img,
            transformed_bboxes=transformed_bboxes,
            original_template_bbox=original_template_bbox,
            camera=camera,
            similarity_threshold=similarity_threshold,
            method=method
        )

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
        Synchronous inference worker using Pipeline pattern.
        Delegates to SingleCameraPipeline or MultiCameraPipeline.
        """
        import time

        thread_name = threading.current_thread().name
        logger.info(
            f"[Job #{job_id}] Starting inference in {thread_name} "
            f"(group_id: {group_id})"
        )

        try:
            # Create pipeline context
            context = PipelineContext(
                job_id=job_id,
                cameras=cameras,
                results=results,
                camera_matchers=self.camera_matchers,
                group_id=group_id,
                T_capture_complete=T_capture_complete,
                text_verification_service=self.text_verification_service,
                template_verification_service=self.template_verification_service,
                product_verification_service=self.product_verification_service,
                emit_callback=emit_callback
            )

            # Filter cameras with matchers (up to 4)
            cameras_to_process = [
                c for c in cameras
                if c.serial_number in self.camera_matchers
            ][:4]

            if not cameras_to_process:
                logger.warning(f"[Job #{job_id}] No cameras with matchers to process")
                # Build and emit error result
                if cameras:
                    first_camera = cameras[0]
                    error_result = {
                        "recipe_id": first_camera.recipe_id,
                        "recipe_name": first_camera.recipe_name,
                        "product_pass_fail": "ERROR",
                        "error": "No cameras with matchers",
                        "camera_results": [],
                        "metadata": {}
                    }
                else:
                    error_result = {
                        "recipe_id": 0,
                        "recipe_name": "Unknown",
                        "product_pass_fail": "ERROR",
                        "error": "No cameras provided",
                        "camera_results": [],
                        "metadata": {}
                    }
                emit_callback("inference_result", error_result)
                return

            # Select appropriate pipeline
            if len(cameras_to_process) == 1:
                logger.info(f"[Job #{job_id}] Using SingleCameraPipeline")
                pipeline = SingleCameraPipeline(
                    crop_func=self._crop_frame_if_needed,
                    transform_func=self._transform_results_to_full_image,
                    result_builder=InferenceResultBuilder,
                    save_and_encode_func=save_and_encode_frame,
                    encode_display_func=encode_frame_for_display
                )
            else:
                logger.info(f"[Job #{job_id}] Using MultiCameraPipeline for {len(cameras_to_process)} cameras")
                pipeline = MultiCameraPipeline(
                    crop_func=self._crop_frame_if_needed,
                    transform_func=self._transform_results_to_full_image,
                    result_builder=InferenceResultBuilder,
                    save_and_encode_func=save_and_encode_frame,
                    encode_display_func=encode_frame_for_display
                )

            # Record inference start time
            T_inference_start = time.time()

            # Execute pipeline (will emit result via finalize)
            inference_result = pipeline.process(context)

            # Calculate inference time
            inference_time = time.time() - T_inference_start

            # Handle reject decision
            if group_id and T_capture_complete and self.reject_scheduler:
                self._handle_reject_decision(
                    group_id=group_id,
                    overall_pass_fail=context.overall_pass_fail,
                    T_capture_complete=T_capture_complete,
                    inference_time=inference_time,
                    cameras=cameras
                )

        except Exception as e:
            logger.error(f"[Job #{job_id}] Error in inference worker: {e}")
            import traceback
            traceback.print_exc()

            # Build and emit error result
            if cameras:
                first_camera = cameras[0]
                error_result = {
                    "recipe_id": first_camera.recipe_id,
                    "recipe_name": first_camera.recipe_name,
                    "product_pass_fail": "ERROR",
                    "error": str(e),
                    "camera_results": [],
                    "metadata": {}
                }
            else:
                error_result = {
                    "recipe_id": 0,
                    "recipe_name": "Unknown",
                    "product_pass_fail": "ERROR",
                    "error": str(e),
                    "camera_results": [],
                    "metadata": {}
                }
            emit_callback("inference_result", error_result)
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
        """
        Build inference result using InferenceResultBuilder.
        Delegates to builder pattern for cleaner code organization.
        """
        return InferenceResultBuilder.from_cameras(
            cameras=cameras,
            results=results,
            camera_inference_results=camera_inference_results,
            overall_pass_fail=overall_pass_fail,
            camera_matchers=self.camera_matchers,
            save_and_encode_func=save_and_encode_frame,
            encode_display_func=encode_frame_for_display
        )
