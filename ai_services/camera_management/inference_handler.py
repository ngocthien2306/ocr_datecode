import logging
import os
import cv2
import json
import time
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from pathlib import Path
import numpy as np
from .utils import save_and_encode_frame, encode_frame_for_display

# Import verification services
from .verification import (
    TextVerificationService,
    TemplateVerificationService,
    ProductVerificationService,
    MLClassifierService,
    EmbeddingClassifierService,
)

# Import OCR backend factory (Strategy Pattern)
from .ocr import OCRBackendFactory, OCRBackendType
from .ocr.factory import OCRModelType, OCRConfig

# ── Default OCR model (used when recipe doesn't specify one) ───────────────
# OCRModelType.SMTR            → SMTR dual-head GTC+CTC (dynamic width) ← MỚI
# OCRModelType.SVTRV2_CTC      → SVTRv2 CTC (6625 classes, width=320)
# OCRModelType.OPENOCR_REPSVTR → OpenOCR RepSVTR
# OCRModelType.PADDLEV5        → PaddleV5 (legacy)
DEFAULT_OCR_MODEL_TYPE = OCRModelType.OPENOCR_REPSVTR

# Map recipe string → enum (used by set_ocr_model_type)
_OCR_MODEL_MAP: Dict[str, OCRModelType] = {
    'SMTR': OCRModelType.SMTR,
    'SVTRV2_CTC': OCRModelType.SVTRV2_CTC,
    'OPENOCR_REPSVTR': OCRModelType.OPENOCR_REPSVTR,
    'PADDLEV5': OCRModelType.PADDLEV5,
}

# OCRBackendType.AUTO      → tự chọn TRT nếu có engine, fallback ONNX
# OCRBackendType.TENSORRT  → bắt buộc dùng .engine (pycuda)
# OCRBackendType.ONNX      → dùng ONNX Runtime (device trong OCRConfig)
OCR_BACKEND_TYPE = OCRBackendType.AUTO

# Import matcher factory
from .matchers import MatcherFactory

# Import result builder
from .result_builder import InferenceResultBuilder

# Import pipeline (Template Method Pattern)
from .pipeline import PipelineContext, SingleCameraPipeline, MultiCameraPipeline

# Import OBB rotation service for Check_Color
from .preprocessing import OBBRotationService

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

    def __init__(self, reject_scheduler=None, trigger_handler=None):
        """
        Initialize InferenceHandler

        Args:
            reject_scheduler: RejectScheduler instance for scheduling reject actions
            trigger_handler: TriggerHandler instance for statistics
        """
        self.camera_matchers: Dict[str, Any] = {}  # Map serial_number -> matcher
        # self.engine_path = f"{home}/Source/ocr_datecode/weights/pipeline_fp16_dynamic_480_640.engine"
        self.engine_path = f"{home}/Source/ocr_datecode/weights/pipeline_fp16_dynamic_315_560.engine"

        # Initialize matcher factory (for future use)
        self._matcher_factory = MatcherFactory(
            engine_path=self.engine_path,
            temp_dir=Path("ocr_inference"),
            backend_dir=Path(__file__).parent.parent.parent / "backend"
        )

        # OCR model type (dynamic, set from recipe)
        self._ocr_model_type: OCRModelType = DEFAULT_OCR_MODEL_TYPE

        # Text recognizer for Check_Type_Product function
        self.text_recognizer = None
        self.ocr_backend = None

        # OBB rotation service for Check_Color function
        self.obb_rotation_service = None

        # Reject scheduler reference
        self.reject_scheduler = reject_scheduler

        # Trigger handler reference for statistics
        self.trigger_handler = trigger_handler

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
            'total_timeout': 0,  # NEW: Track timeout jobs
            'max_queue_depth': 0
        }
        self._stats_lock = threading.Lock()

        # Per-camera cumulative statistics
        self._per_camera_stats: Dict[str, Dict[str, int]] = {}
        # Structure: {serial_number: {'pass_count': N, 'fail_count': M, 'error_count': K}}
        self._camera_stats_lock = threading.Lock()

        # LAZY INITIALIZATION FLAG
        # Models will be initialized on WORKER THREAD to avoid CUDA context issues
        self._models_initialized = False
        self._models_init_lock = threading.Lock()

        logger.info("InferenceHandler initialized (models will be lazy-loaded on worker thread)")

    def set_ocr_model_type(self, model_type_str: Optional[str]) -> bool:
        """Set OCR model type from recipe string. Returns True if changed (needs reinit)."""
        if not model_type_str:
            return False  # keep current
        new_type = _OCR_MODEL_MAP.get(model_type_str, DEFAULT_OCR_MODEL_TYPE)
        if new_type == self._ocr_model_type:
            return False
        logger.info(f"🔄 OCR model type changed: {self._ocr_model_type.name} → {new_type.name}")
        self._ocr_model_type = new_type
        return True

    def _reinit_ocr_backend(self):
        """Reinit OCR backend when recipe changes model type (called on worker thread)."""
        logger.info(f"🔄 Reinitializing OCR backend → {self._ocr_model_type.name}")
        old = getattr(self, '_ocr_backend_instance', None)
        self._init_ocr_backend()
        # Update text_verification_service with new recognizer
        if hasattr(self, 'text_verification_service') and self.text_verification_service and self.text_recognizer:
            self.text_verification_service.text_recognizer = self.text_recognizer
            self.text_verification_service.ocr_backend = self.ocr_backend or "unknown"
            logger.info(f"✅ TextVerificationService updated with new OCR backend: {self.ocr_backend}")
        del old  # free VRAM

    def _init_ocr_backend(self):
        """
        Initialize OCR backend using Factory Pattern.

        Uses OCRBackendFactory to create the appropriate backend based on:
        - Environment variable OCR_MODEL (openocr/paddlev5) - default: openocr
        - Environment variable OCR_BACKEND (tensorrt/onnx/auto) - default: auto
        - Available backends on the system

        Default: OpenOCR RepSVTR with TensorRT (127 imgs/sec, batch=4)
        Fallback chain:
            1. OpenOCR TensorRT (fastest)
            2. OpenOCR ONNX (13 imgs/sec)
            3. PaddleV5 TensorRT (legacy, ~40 imgs/sec)
            4. PaddleV5 ONNX (legacy, ~10 imgs/sec)
        """
        try:
            # AUTO selection with fixed TensorRT (no more CUDA context conflict!)
            # Priority: OpenOCR TensorRT > OpenOCR ONNX > PaddleV5 TensorRT > PaddleV5 ONNX
            # Note: TensorRT OpenOCR now uses pre-allocated buffers (like YOLO OBB) - NO CONFLICT!
            self._ocr_backend_instance = OCRBackendFactory.create(
                backend_type=OCR_BACKEND_TYPE,
                model_type=self._ocr_model_type,
            )

            if self._ocr_backend_instance is not None:
                self.text_recognizer = self._ocr_backend_instance
                self.ocr_backend = self._ocr_backend_instance.backend_name
                logger.info(f"✅ OCR backend initialized: {self.ocr_backend}")
            else:
                self.text_recognizer = None
                self.ocr_backend = None
                logger.warning("⚠️ No OCR backend available")

        except Exception as e:
            logger.error(f"❌ Failed to initialize OCR backend: {e}")
            import traceback
            traceback.print_exc()
            self.text_recognizer = None
            self.ocr_backend = None
            self._ocr_backend_instance = None

    def _ensure_models_initialized(self, thread_name: str = None):
        """
        Ensure OCR and YOLO models are initialized (lazy initialization).
        This is called from worker thread to ensure CUDA context consistency.

        Args:
            thread_name: Optional thread name for logging
        """
        if not self._models_initialized:
            with self._models_init_lock:
                if not self._models_initialized:  # Double-check locking
                    thread_name = thread_name or threading.current_thread().name
                    logger.info(f"🔧 Lazy-initializing OCR/YOLO models on {thread_name}...")

                    try:
                        # CRITICAL: Initialize CUDA context on worker thread
                        # pycuda.autoinit only runs once (on main thread import)
                        # Worker thread needs explicit context creation
                        self._initialize_cuda_context_for_thread(thread_name)

                        # Initialize OCR backend
                        self._init_ocr_backend()

                        # Initialize verification services (YOLO OBB)
                        self._init_verification_services()

                        self._models_initialized = True
                        logger.info(f"✅ OCR/YOLO models initialized on {thread_name}")

                    except Exception as e:
                        logger.error(f"❌ Failed to initialize models on {thread_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        # Don't set flag = True to allow retry
                        raise

    def _initialize_cuda_context_for_thread(self, thread_name: str):
        """
        Initialize CUDA context for the current thread.

        CRITICAL: pycuda.autoinit only works on the thread that imports it first.
        Worker threads need explicit context initialization.

        Args:
            thread_name: Thread name for logging
        """
        current_thread = threading.current_thread()

        # Check if context already initialized for this thread
        if hasattr(current_thread, '_cuda_context_initialized'):
            logger.debug(f"CUDA context already initialized on {thread_name}")
            return

        try:
            import pycuda.driver as cuda

            # Initialize CUDA driver
            cuda.init()

            # Get device 0
            device = cuda.Device(0)

            # Create a context for this thread
            # Use retain_primary_context() which creates/reuses the primary context
            ctx = device.retain_primary_context()
            ctx.push()

            # Mark thread as initialized
            current_thread._cuda_context_initialized = True
            current_thread._cuda_context = ctx

            logger.info(f"✅ CUDA context created on {thread_name} (handle: {ctx.handle})")

        except Exception as e:
            logger.error(f"❌ Failed to initialize CUDA context on {thread_name}: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _init_verification_services(self):
        """Initialize text and template verification services"""
        # Debug image writes are disk-I/O heavy (~50-150ms each) and hurt
        # realtime inference latency. Gate on env var for troubleshooting.
        save_debug = True # os.environ.get('VERIFY_DEBUG_IMAGES', '0').lower() in ('1', 'true', 'yes')

        # ML Classifier Service — loads sklearn models from shared filesystem.
        # BE saves models to {PROJECT_ROOT}/public/ml_projects/{project_id}/models/{model_id}.joblib
        ml_base_dir = Path(f"{home}/Source/ocr_datecode/public/ml_projects")
        self.ml_classifier_service = MLClassifierService(
            ml_base_dir=ml_base_dir,
            save_debug_images=save_debug,
            debug_path=f"{home}/Source/ocr_datecode/ai_services/test_result",
        )
        logger.info(f"MLClassifierService initialized: base_dir={ml_base_dir}, debug={save_debug}")

        # Embedding Classifier Services — registry keyed by defect_model name.
        # Recipe.defect_model picks which service to use at runtime.
        # `embedding_classifier_service` keeps the legacy attr pointing at the
        # default (arcface) for backward compat with paths that still read it.
        _embedding_weights_map = {
            'arcface': f"{home}/Source/ocr_datecode/weights/arcface_128_b3_20260428-190614",
            'supcon':  f"{home}/Source/ocr_datecode/weights/supcon_128_efficientnet_b2_20260429-073504",
        }
        self.embedding_classifier_services: Dict[str, EmbeddingClassifierService] = {}
        for name, weights_dir in _embedding_weights_map.items():
            onnx_p = Path(weights_dir) / "model.onnx"
            cfg_p  = Path(weights_dir) / "config.yaml"
            if not (onnx_p.exists() and cfg_p.exists()):
                logger.warning(f"Embedding model '{name}' not found at {weights_dir}, skipping")
                continue
            try:
                self.embedding_classifier_services[name] = EmbeddingClassifierService(
                    onnx_path=str(onnx_p),
                    config_path=str(cfg_p),
                    save_debug_images=save_debug,
                    debug_path=f"{home}/Source/ocr_datecode/ai_services/test_result",
                )
                logger.info(f"EmbeddingClassifierService '{name}' initialized: {Path(weights_dir).name}")
            except Exception as e:
                logger.warning(f"EmbeddingClassifierService '{name}' failed to init: {e}")

        # Default → arcface; fall back to first available if arcface missing
        self.embedding_classifier_service = (
            self.embedding_classifier_services.get('arcface')
            or next(iter(self.embedding_classifier_services.values()), None)
        )
        if self.embedding_classifier_service is None:
            logger.warning("No embedding classifier service initialized — char embedding disabled")

        # Text Verification Service
        if self.text_recognizer is not None:
            self.text_verification_service = TextVerificationService(
                text_recognizer=self.text_recognizer,
                ocr_backend=self.ocr_backend or "unknown",
                save_debug_images=save_debug,
                debug_path=f"{home}/Source/ocr_datecode/ai_services/test_result",
                use_char_conf_check=False,
                use_sim_check=False,
                ml_classifier_service=self.ml_classifier_service,
                embedding_classifier_service=self.embedding_classifier_service,
                embedding_classifier_services=self.embedding_classifier_services,
            )
            logger.info(f"TextVerificationService initialized with {self.ocr_backend} backend (debug={save_debug})")
        else:
            self.text_verification_service = None
            logger.warning("TextVerificationService not available (no OCR backend)")

        # Template Verification Service
        self.template_verification_service = TemplateVerificationService(
            save_debug_images=save_debug,
            debug_path=f"{home}/Source/ocr_datecode/ai_services/test_result",
            default_threshold=0.85
        )
        logger.info(f"TemplateVerificationService initialized (debug={save_debug})")

        # Product Verification Service with YOLO OBB
        yolo_obb_engine_path = f"{home}/Source/ocr_datecode/weights/best_bottle_m_320_new_1.engine"

        # Configuration: save_debug_images can be "never", "on_fail", or "always"
        # Use "never" for production (fastest), "on_fail" for debugging failures
        product_debug_mode = os.environ.get('PRODUCT_VERIFICATION_DEBUG', 'never')

        # Configuration: check_label_boundary can be enabled/disabled via env var
        # Set to "0", "false", or "no" to disable the label region boundary check
        check_label_boundary = os.environ.get('PRODUCT_CHECK_LABEL_BOUNDARY', 'true').lower()
        check_label_boundary = check_label_boundary not in ['0', 'false', 'no']

        # Configuration: check_misalignment can be enabled/disabled via env var
        # Set to "0", "false", or "no" to disable the misalignment check
        check_misalignment = os.environ.get('PRODUCT_CHECK_MISALIGNMENT', 'true').lower()
        check_misalignment = check_misalignment not in ['0', 'false', 'no']

        self.product_verification_service = ProductVerificationService(
            engine_path=yolo_obb_engine_path,
            save_debug_images=product_debug_mode,  # "never", "on_fail", or "always"
            debug_path=f"{home}/Source/ocr_datecode/ai_services/test_result",
            angle_threshold=3.0,  # Rotation threshold in degrees
            margin_pixels=15,     # Misalignment margin in pixels
            conf_threshold=0.001,  # YOLO confidence threshold
            check_label_boundary=check_label_boundary,  # Enable/disable label boundary check
            check_misalignment=check_misalignment  # Enable/disable misalignment check
        )
        logger.info(
            f"ProductVerificationService initialized with YOLO OBB: {yolo_obb_engine_path}, "
            f"debug_mode={product_debug_mode}, check_misalignment={check_misalignment}, "
            f"check_label_boundary={check_label_boundary}"
        )

        # OBB Rotation Service for Check_Color function type
        obb_rotation_engine = f"{home}/Source/ocr_datecode/weights/best_bottle_m.engine"
        self.obb_rotation_service = OBBRotationService(
            engine_path=obb_rotation_engine,
            conf_threshold=0.25,
            inverse_transform=False
        )
        logger.info(f"OBBRotationService initialized: {obb_rotation_engine}")
        logger.info(f"OBBRotationService available: {self.obb_rotation_service.available}")

        # Pure-CV alternative — selected per-recipe via cap_rotation_method='yolo_segment'
        from .preprocessing.cv_rotator import CVRotationService
        self.cv_rotation_service = CVRotationService(inverse_transform=False)
        logger.info(f"CVRotationService initialized (always available)")

    def init_matchers(self, cameras: List['Camera']):
        """
        Initialize matchers for all cameras using MatcherFactory.

        IMPORTANT: This runs on WORKER THREAD to avoid CUDA context issues.

        Args:
            cameras: List of Camera objects with templates

        Returns:
            Number of matchers initialized
        """
        # Submit to worker thread and wait for completion
        logger.info("Submitting matcher initialization to worker thread...")
        future = self._inference_executor.submit(self._init_matchers_on_worker, cameras)

        try:
            num_initialized = future.result(timeout=30.0)  # 30s timeout for init
            logger.info(f"✅ Matcher initialization completed: {num_initialized} matchers")
            return num_initialized
        except Exception as e:
            logger.error(f"❌ Matcher initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def _init_matchers_on_worker(self, cameras: List['Camera']):
        """
        Internal method that runs on WORKER THREAD to initialize matchers.
        This ensures TensorRT engine is created on the same thread as inference.
        """
        thread_name = threading.current_thread().name
        logger.info(f"🔧 Initializing matchers on {thread_name}...")

        # IMPORTANT: Initialize OCR/YOLO first to establish CUDA context
        # This ensures pycuda.autoinit runs before SuperPoint TensorRT init
        self._ensure_models_initialized(thread_name)

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
            logger.info(f"✅ Initialized {initialized_count} matchers on {thread_name}")

            # Pre-compute template embeddings for embedding classifier
            svc = getattr(self, 'text_verification_service', None)
            return initialized_count

        except Exception as e:
            logger.error(f"❌ Error initializing inference matchers on {thread_name}: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def clear_matchers(self):
        """Clear all inference matchers and reset per-camera stats"""
        self.camera_matchers.clear()
        with self._camera_stats_lock:
            self._per_camera_stats.clear()
        if getattr(self, 'ml_classifier_service', None) is not None:
            self.ml_classifier_service.clear_cache()
        logger.info("All inference matchers, camera stats, and ML cache cleared")

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
                f"Timeout: {self._inference_stats['total_timeout']}, "
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
        Verify text + char regions for one camera. Delegates to
        TextVerificationService.

        Returns:
            {
                'text': {'all_match': bool, 'results': [...]},
                'char': {'all_match': bool, 'results': [...]},
            }
        """
        if self.text_verification_service is None:
            logger.warning("TextVerificationService not available, skipping verification")
            empty = {'all_match': False, 'results': []}
            return {'text': empty, 'char': dict(empty)}

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
        T_capture_complete: float = None,
        stuck_count: int = 1,
        pulse_width_ms: float = 0.0
    ):
        """
        Submit inference job to thread pool (non-blocking)

        Args:
            cameras: List of cameras that captured frames
            results: Capture results from trigger
            emit_callback: Callback to emit inference result
            group_id: Optional group ID for tracking
            T_capture_complete: Time when cameras finished capturing (for reject timing)
            stuck_count: Number of stuck bottles detected (>1 = stuck, force FAIL)
            pulse_width_ms: DI pulse width in ms (used to calculate per-bottle reject offset)

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
            T_capture_complete=T_capture_complete,
            stuck_count=stuck_count,
            pulse_width_ms=pulse_width_ms
        )

        # Track active job with timeout flags
        with self._job_lock:
            self._active_jobs[job_id] = {
                'future': future,
                'timeout': False,      # Has timeout occurred?
                'emitted': False,      # Has result been emitted?
                'handled': False,      # Has reject been handled?
                'group_id': group_id,
                'cameras': cameras,
                'T_capture_complete': T_capture_complete
            }

        # Add callback to cleanup when done
        future.add_done_callback(lambda f: self._cleanup_job(job_id, f))

        # Start timeout checker thread
        timeout_thread = threading.Thread(
            target=self._check_timeout_and_handle,
            args=(job_id, cameras, emit_callback),
            daemon=True,
            name=f"TimeoutChecker-{job_id}"
        )
        timeout_thread.start()

        return job_id

    def _cleanup_job(self, job_id: int, future):
        """Cleanup completed job"""
        is_timeout = False
        with self._job_lock:
            if job_id in self._active_jobs:
                is_timeout = self._active_jobs[job_id]['timeout']
                del self._active_jobs[job_id]

        # Update stats
        with self._stats_lock:
            if is_timeout:
                # Already counted in _check_timeout_and_handle
                pass
            elif future.exception():
                self._inference_stats['total_failed'] += 1
                logger.error(f"[Job #{job_id}] Failed with exception: {future.exception()}")
            else:
                self._inference_stats['total_completed'] += 1
                logger.info(f"[Job #{job_id}] Completed successfully")

    def _check_timeout_and_handle(
        self,
        job_id: int,
        cameras: List['Camera'],
        emit_callback
    ):
        """
        Check for inference timeout and handle accordingly.

        This runs in a separate thread to monitor each job's execution time.
        If timeout occurs:
        1. Emit FAIL result (if not already emitted)
        2. Schedule reject (if not already handled)
        3. Mark job as timeout to prevent double processing

        Args:
            job_id: Job identifier
            cameras: List of cameras for this job
            emit_callback: Callback to emit results
        """
        TIMEOUT_SECONDS = 10.0  # 1 second timeout

        try:
            # Get future from active jobs
            with self._job_lock:
                if job_id not in self._active_jobs:
                    return  # Job already cleaned up
                future = self._active_jobs[job_id]['future']
                group_id = self._active_jobs[job_id]['group_id']
                T_capture_complete = self._active_jobs[job_id]['T_capture_complete']

            # Wait for result with timeout
            try:
                future.result(timeout=TIMEOUT_SECONDS)
                # Inference completed within timeout - nothing to do
                return

            except concurrent.futures.TimeoutError:
                # Timeout occurred!
                logger.warning(
                    f"[Job #{job_id}] ⏱️ TIMEOUT after {TIMEOUT_SECONDS}s "
                    f"(group_id: {group_id})"
                )

                # Mark job as timeout
                with self._job_lock:
                    if job_id not in self._active_jobs:
                        return  # Already cleaned up

                    job_info = self._active_jobs[job_id]

                    # Check if already handled
                    if job_info['timeout']:
                        return  # Already handled by another thread

                    # Mark as timeout
                    job_info['timeout'] = True

                    # Emit FAIL result (if not already emitted)
                    if not job_info['emitted']:
                        self._emit_timeout_result(
                            job_id, cameras, emit_callback, TIMEOUT_SECONDS
                        )
                        job_info['emitted'] = True

                    # Schedule reject (if not already handled)
                    if not job_info['handled'] and group_id and T_capture_complete:
                        self._schedule_timeout_reject(
                            job_id, group_id, cameras,
                            T_capture_complete, TIMEOUT_SECONDS
                        )
                        job_info['handled'] = True

                # Update statistics
                with self._stats_lock:
                    self._inference_stats['total_timeout'] += 1

        except Exception as e:
            logger.error(f"[Job #{job_id}] Error in timeout checker: {e}")
            import traceback
            traceback.print_exc()

    def _emit_timeout_result(
        self,
        job_id: int,
        cameras: List['Camera'],
        emit_callback,
        timeout_seconds: float
    ):
        """Emit FAIL result for timeout"""
        try:
            if not cameras:
                return

            first_camera = cameras[0]

            # Build timeout error result
            error_result = {
                "recipe_id": first_camera.recipe_id,
                "recipe_name": first_camera.recipe_name,
                "product_pass_fail": "FAIL",
                "error": f"Inference timeout ({timeout_seconds}s)",
                "timeout": True,
                "camera_results": [],
                "metadata": {
                    "job_id": job_id,
                    "timeout_seconds": timeout_seconds,
                    "inference_time": timeout_seconds
                }
            }

            emit_callback("inference_result", error_result)
            logger.warning(
                f"[Job #{job_id}] Emitted FAIL result due to timeout"
            )

        except Exception as e:
            logger.error(f"[Job #{job_id}] Error emitting timeout result: {e}")

    def _schedule_timeout_reject(
        self,
        job_id: int,
        group_id: int,
        cameras: List['Camera'],
        T_capture_complete: float,
        timeout_seconds: float
    ):
        """Schedule reject for timeout case"""
        try:
            if not cameras or not self.reject_scheduler:
                return

            camera = cameras[0]
            delay_reject = camera.delay_reject  # ms
            do_reject_number = camera.do_reject_number
            do_alarm_number = camera.do_alarm_number
            allow_late_reject = getattr(camera, 'allow_late_reject', False)

            # Use timeout as inference_time for reject timing calculation
            success = self.reject_scheduler.schedule_reject(
                group_id=group_id,
                T_capture_complete=T_capture_complete,
                inference_time=timeout_seconds,
                delay_reject=delay_reject,
                do_number=do_reject_number,
                alarm_number=do_alarm_number,
                allow_late_reject=allow_late_reject
            )

            if success:
                logger.warning(
                    f"[Job #{job_id}] Reject scheduled for timeout "
                    f"(delay_reject={delay_reject}ms, DO{do_reject_number})"
                )
            else:
                logger.error(
                    f"[Job #{job_id}] Failed to schedule reject for timeout"
                )

        except Exception as e:
            logger.error(f"[Job #{job_id}] Error scheduling timeout reject: {e}")

    def _update_camera_stats(self, serial_number: str, pass_count: int, fail_count: int, error_count: int):
        """
        Update cumulative per-camera statistics

        Args:
            serial_number: Camera serial number
            pass_count: Number of pass frames in this inference
            fail_count: Number of fail frames in this inference
            error_count: Number of error frames in this inference
        """
        with self._camera_stats_lock:
            if serial_number not in self._per_camera_stats:
                self._per_camera_stats[serial_number] = {
                    'pass_count': 0,
                    'fail_count': 0,
                    'error_count': 0
                }

            self._per_camera_stats[serial_number]['pass_count'] += pass_count
            self._per_camera_stats[serial_number]['fail_count'] += fail_count
            self._per_camera_stats[serial_number]['error_count'] += error_count

    def _get_camera_stats_summary(self) -> List[Dict[str, Any]]:
        """
        Get per-camera statistics summary with both session and cumulative counts

        Returns:
            List of dicts with serial_number, session counts, and total counts
        """
        # Get cumulative stats from TriggerHandler (persistent across recipe reload)
        cumulative_stats = {}
        if self.trigger_handler:
            cumulative_stats = self.trigger_handler.get_camera_cumulative_stats()

        with self._camera_stats_lock:
            summary = []
            # Get all serial numbers (union of session and cumulative)
            all_serial_numbers = set(self._per_camera_stats.keys()) | set(cumulative_stats.keys())

            for serial_number in all_serial_numbers:
                # Session stats (current recipe run)
                session_stats = self._per_camera_stats.get(serial_number, {
                    'pass_count': 0,
                    'fail_count': 0,
                    'error_count': 0
                })
                pass_count = session_stats['pass_count']
                fail_count = session_stats['fail_count']
                error_count = session_stats['error_count']

                # Cumulative stats (across all recipe runs)
                cum_stats = cumulative_stats.get(serial_number, {
                    'total_pass': 0,
                    'total_fail': 0,
                    'total_error': 0
                })
                total_pass_count = cum_stats['total_pass']
                total_fail_count = cum_stats['total_fail']
                total_error_count = cum_stats['total_error']

                # Determine camera_pass_fail based on cumulative stats
                camera_pass_fail = "PASS" if (total_fail_count == 0 and total_error_count == 0) else "FAIL"

                summary.append({
                    'serial_number': serial_number,
                    # Session counts (current recipe run)
                    'pass_count': pass_count,
                    'fail_count': fail_count,
                    'error_count': error_count,
                    # Cumulative counts (persistent across recipe reload)
                    'total_pass_count': total_pass_count,
                    'total_fail_count': total_fail_count,
                    'total_error_count': total_error_count,
                    # Overall status based on cumulative
                    'camera_pass_fail': camera_pass_fail
                })
            return summary

    def _do_inference_sync(
        self,
        job_id: int,
        cameras: List['Camera'],
        results: Dict[str, Any],
        emit_callback,
        group_id: int = None,
        T_capture_complete: float = None,
        stuck_count: int = 1,
        pulse_width_ms: float = 0.0
    ):
        """
        Synchronous inference worker using Pipeline pattern.
        Delegates to SingleCameraPipeline or MultiCameraPipeline.
        """
        import time

        thread_name = threading.current_thread().name

        # LAZY INITIALIZATION: Initialize OCR/YOLO models on WORKER THREAD (once)
        # Note: SuperPoint matchers are initialized separately via init_matchers()
        # This ensures ALL CUDA operations happen on the same thread
        self._ensure_models_initialized(thread_name)

        logger.info(
            f"[Job #{job_id}] Starting inference in {thread_name} "
            f"(group_id: {group_id})"
        )

        # Check if timeout already occurred
        with self._job_lock:
            if job_id in self._active_jobs and self._active_jobs[job_id]['timeout']:
                logger.info(
                    f"[Job #{job_id}] Skipping inference (timeout already handled)"
                )
                return  # Exit early, timeout handler already emitted result

        try:
            # Get statistics from trigger handler if available
            statistics = {
                'total_triggers': 0,
                'total_inferences': 0,
                'total_pass': 0,
                'total_fail': 0,
                'total_reject': 0
            }
            if self.trigger_handler:
                with self.trigger_handler._stats_lock:
                    stats = self.trigger_handler._stats
                    statistics['total_triggers'] = stats.get('total_triggers', 0)
                    statistics['total_inferences'] = stats.get('total_inferences', 0)
                    # Pass/Fail based on completed vs timeout groups
                    statistics['total_pass'] = stats.get('total_groups_completed', 0)
                    statistics['total_fail'] = stats.get('total_groups_timeout', 0)

            # Get reject count from reject scheduler if available
            if self.reject_scheduler:
                reject_stats = self.reject_scheduler.get_stats()
                statistics['total_reject'] = reject_stats.get('total_rejected', 0)

            # Get cumulative per-camera statistics
            statistics['per_camera'] = self._get_camera_stats_summary()

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
                obb_rotation_service=self.obb_rotation_service,
                cv_rotation_service=self.cv_rotation_service,
                emit_callback=emit_callback,
                statistics=statistics
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
            # Note: If timeout occurs during pipeline.process(), the timeout handler
            # will emit FAIL result. The pipeline will continue to completion, but
            # we'll skip the result emit below.
            inference_result = pipeline.process(context)

            # Calculate inference time
            inference_time = time.time() - T_inference_start

            # Check if timeout occurred during inference
            with self._job_lock:
                if job_id in self._active_jobs and self._active_jobs[job_id]['timeout']:
                    logger.info(
                        f"[Job #{job_id}] Inference completed but timeout already handled "
                        f"(actual time: {inference_time:.3f}s). Skipping result emit and reject."
                    )
                    return  # Exit early, timeout handler already handled everything

            # Update per-camera cumulative statistics
            if context.camera_inference_results:
                for serial_number, camera_result in context.camera_inference_results.items():
                    # Handle different structures for single vs multi-camera
                    if 'frames' in camera_result:
                        # Single-camera mode: has 'frames' list
                        frames_list = camera_result['frames']
                        logger.debug(
                            f"[Job #{job_id}] Camera {serial_number}: "
                            f"Single-camera mode, {len(frames_list)} frames"
                        )
                    else:
                        # Multi-camera mode: camera_result is the frame result itself
                        frames_list = [camera_result]
                        logger.debug(
                            f"[Job #{job_id}] Camera {serial_number}: "
                            f"Multi-camera mode, single frame"
                        )

                    # Count pass/fail/error
                    pass_count = sum(1 for f in frames_list if f.get('result') == 'PASS')
                    fail_count = sum(1 for f in frames_list if f.get('result') == 'FAIL')
                    error_count = sum(1 for f in frames_list if f.get('result') == 'ERROR')

                    logger.info(
                        f"[Job #{job_id}] Camera {serial_number}: "
                        f"pass={pass_count}, fail={fail_count}, error={error_count}"
                    )

                    if pass_count > 0 or fail_count > 0 or error_count > 0:
                        # Update session stats (InferenceHandler)
                        self._update_camera_stats(serial_number, pass_count, fail_count, error_count)

                        # Update cumulative stats (TriggerHandler - persistent across recipe reload)
                        if self.trigger_handler:
                            self.trigger_handler.update_camera_cumulative_stats(
                                serial_number, pass_count, fail_count, error_count
                            )
                    else:
                        logger.warning(
                            f"[Job #{job_id}] Camera {serial_number}: No valid results to count. "
                            f"Frames: {frames_list}"
                        )
            else:
                logger.warning(f"[Job #{job_id}] No camera_inference_results to update stats")

            # Handle reject decision
            if group_id and T_capture_complete and self.reject_scheduler:
                self._handle_reject_decision(
                    group_id=group_id,
                    overall_pass_fail=context.overall_pass_fail,
                    T_capture_complete=T_capture_complete,
                    inference_time=inference_time,
                    cameras=cameras,
                    stuck_count=stuck_count,
                    pulse_width_ms=pulse_width_ms
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
        cameras: List['Camera'],
        stuck_count: int = 1,
        pulse_width_ms: float = 0.0
    ):
        """
        Handle reject scheduling/cancellation based on inference result.

        If stuck_count > 1 (bottles stuck together), force FAIL regardless of
        inference result and schedule N reject pulses with per-bottle timing offset.

        Args:
            group_id: Group ID for tracking
            overall_pass_fail: Overall inference result (PASS/FAIL/ERROR)
            T_capture_complete: Time when cameras finished capturing
            inference_time: Inference duration in seconds
            cameras: List of cameras (to get reject config)
            stuck_count: Number of stuck bottles detected (1 = normal)
            pulse_width_ms: DI pulse width in ms, used to compute per-bottle offset
        """
        try:
            # --- Stuck bottle override ---
            if stuck_count > 1:
                ratio = pulse_width_ms / 250.0
                logger.warning(
                    f"[Group #{group_id}] ⚠️ STUCK BOTTLES: "
                    f"N={stuck_count}, pulse={pulse_width_ms:.1f}ms, ratio={ratio:.2f}. "
                    f"Inference result '{overall_pass_fail}' overridden → FAIL. "
                    f"Scheduling {stuck_count} rejects."
                )
                overall_pass_fail = "FAIL"

            if overall_pass_fail == "FAIL" or overall_pass_fail == "ERROR":
                # Get reject config from first camera (recipe-level config)
                if not cameras:
                    logger.warning(f"[Group #{group_id}] No cameras available for reject config")
                    return

                camera = cameras[0]
                delay_reject = camera.delay_reject  # ms
                do_reject_number = camera.do_reject_number
                do_alarm_number = camera.do_alarm_number
                allow_late_reject = getattr(camera, 'allow_late_reject', False)

                if stuck_count > 1:
                    # Schedule N rejects with per-bottle timing offset
                    # Bottle i arrives at rejector (i * delta_t) seconds after bottle 0
                    delta_t = pulse_width_ms / stuck_count / 1000.0  # seconds per bottle
                    for i in range(stuck_count):
                        bottle_group_id = group_id * 1000 + i
                        fake_T_capture = T_capture_complete + i * delta_t
                        success = self.reject_scheduler.schedule_reject(
                            group_id=bottle_group_id,
                            T_capture_complete=fake_T_capture,
                            inference_time=inference_time,
                            delay_reject=delay_reject,
                            do_number=do_reject_number,
                            # Alarm only on first bottle to avoid duplicate alarms
                            alarm_number=do_alarm_number if i == 0 else -1,
                            allow_late_reject=allow_late_reject
                        )
                        if success:
                            logger.info(
                                f"[Group #{group_id}] Stuck reject {i+1}/{stuck_count} scheduled "
                                f"(offset +{i * delta_t * 1000:.0f}ms, "
                                f"virtual_group={bottle_group_id})"
                            )
                        else:
                            logger.error(
                                f"[Group #{group_id}] Stuck reject {i+1}/{stuck_count} "
                                f"FAILED to schedule (inference too slow)"
                            )
                else:
                    # Normal single-bottle reject
                    success = self.reject_scheduler.schedule_reject(
                        group_id=group_id,
                        T_capture_complete=T_capture_complete,
                        inference_time=inference_time,
                        delay_reject=delay_reject,
                        do_number=do_reject_number,
                        alarm_number=do_alarm_number,
                        allow_late_reject=allow_late_reject
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

            else:  # PASS (and not stuck)
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
