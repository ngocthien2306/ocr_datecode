"""
OCR Backend Factory

Factory for creating OCR backend instances.
Supports automatic backend selection and configuration.
"""

import os
import logging
from enum import Enum
from typing import Optional, List, Tuple
from dataclasses import dataclass
import numpy as np

from .base import OCRBackendStrategy

logger = logging.getLogger(__name__)

home = os.environ.get('HOME')


class OCRBackendType(Enum):
    """Available OCR backend types"""
    TENSORRT = "tensorrt"
    ONNX = "onnx"
    AUTO = "auto"


class OCRModelType(Enum):
    """Available OCR model architectures"""
    PADDLEV5 = "paddlev5"           # Legacy model (old)
    OPENOCR_REPSVTR = "openocr"     # New model (default)


@dataclass
class OCRConfig:
    """Configuration for OCR backend"""
    model_path: str
    dict_path: str
    model_type: OCRModelType = OCRModelType.OPENOCR_REPSVTR  # Default to new model
    use_gpu: bool = True
    min_width: int = 50
    max_width: int = 2000
    batch_size: int = 4  # Optimal batch size from benchmark


# ============================================================================
# Backend Adapters - Wrap existing implementations to match Strategy interface
# ============================================================================

class TensorRTOCRBackend(OCRBackendStrategy):
    """
    TensorRT OCR Backend adapter.

    Wraps the existing TextRecognizerTRT class to implement OCRBackendStrategy.
    """

    def __init__(self, config: OCRConfig):
        """
        Initialize TensorRT backend.

        Args:
            config: OCR configuration with engine_path and dict_path
        """
        self._config = config
        self._recognizer = None
        self._available = False

        try:
            from ..text_recognizer_trt import TextRecognizerTRT, TENSORRT_AVAILABLE

            if not TENSORRT_AVAILABLE:
                logger.warning("TensorRT not available")
                return

            if not os.path.exists(config.model_path):
                raise FileNotFoundError(f"TensorRT engine not found: {config.model_path}")
            if not os.path.exists(config.dict_path):
                raise FileNotFoundError(f"Dict file not found: {config.dict_path}")

            self._recognizer = TextRecognizerTRT(
                engine_path=config.model_path,
                dict_path=config.dict_path,
                min_width=config.min_width,
                max_width=config.max_width
            )
            self._available = True
            logger.info(f"TensorRTOCRBackend initialized: {config.model_path}")

        except Exception as e:
            logger.error(f"Failed to initialize TensorRT backend: {e}")
            self._available = False

    @property
    def backend_name(self) -> str:
        return "tensorrt"

    @property
    def is_available(self) -> bool:
        return self._available and self._recognizer is not None

    def recognize(
        self,
        image: np.ndarray,
        return_confidence: bool = True
    ) -> Tuple[str, float]:
        if not self.is_available:
            return ("", 0.0)
        return self._recognizer.recognize(image, return_confidence=return_confidence)

    def recognize_batch(
        self,
        images: List[np.ndarray]
    ) -> List[Tuple[str, float]]:
        if not self.is_available:
            return [("", 0.0) for _ in images]

        if hasattr(self._recognizer, 'recognize_batch'):
            return self._recognizer.recognize_batch(images)
        else:
            # Fallback to sequential
            return [self.recognize(img) for img in images]


class ONNXOCRBackend(OCRBackendStrategy):
    """
    ONNX OCR Backend adapter (PaddleV5 legacy model).

    Wraps the existing TextRecognizer (ONNX) class to implement OCRBackendStrategy.
    """

    def __init__(self, config: OCRConfig):
        """
        Initialize ONNX backend.

        Args:
            config: OCR configuration with model_path and dict_path
        """
        self._config = config
        self._recognizer = None
        self._available = False

        try:
            from ..text_recognizer import TextRecognizer

            if not os.path.exists(config.model_path):
                raise FileNotFoundError(f"ONNX model not found: {config.model_path}")
            if not os.path.exists(config.dict_path):
                raise FileNotFoundError(f"Dict file not found: {config.dict_path}")

            self._recognizer = TextRecognizer(
                model_path=config.model_path,
                dict_path=config.dict_path,
                use_gpu=config.use_gpu
            )
            self._available = True
            logger.info(f"ONNXOCRBackend (PaddleV5) initialized: {config.model_path}")

        except Exception as e:
            logger.error(f"Failed to initialize ONNX backend: {e}")
            self._available = False

    @property
    def backend_name(self) -> str:
        return "onnx_paddlev5"

    @property
    def is_available(self) -> bool:
        return self._available and self._recognizer is not None

    def recognize(
        self,
        image: np.ndarray,
        return_confidence: bool = True
    ) -> Tuple[str, float]:
        if not self.is_available:
            return ("", 0.0)
        return self._recognizer.recognize(image, return_confidence=return_confidence)

    def recognize_batch(
        self,
        images: List[np.ndarray]
    ) -> List[Tuple[str, float]]:
        if not self.is_available:
            return [("", 0.0) for _ in images]

        if hasattr(self._recognizer, 'recognize_batch'):
            return self._recognizer.recognize_batch(images)
        else:
            # Fallback to sequential
            return [self.recognize(img) for img in images]


class TensorRTOpenOCRBackend(OCRBackendStrategy):
    """
    TensorRT OCR Backend for OpenOCR RepSVTR model.

    Uses the optimized batch-enabled engine with dynamic batch support (1-16).
    Recommended batch_size=4 for optimal performance (127 imgs/sec).
    """

    def __init__(self, config: OCRConfig):
        """
        Initialize TensorRT OpenOCR backend.

        Args:
            config: OCR configuration with engine_path and dict_path
        """
        self._config = config
        self._recognizer = None
        self._available = False
        self._batch_size = config.batch_size

        try:
            # Import from camera_management directory (new implementation)
            from ..text_recognizer_openocr_trt import TextRecognizerOpenOCRTRT

            if not os.path.exists(config.model_path):
                raise FileNotFoundError(f"TensorRT engine not found: {config.model_path}")
            if not os.path.exists(config.dict_path):
                raise FileNotFoundError(f"Dict file not found: {config.dict_path}")

            self._recognizer = TextRecognizerOpenOCRTRT(
                engine_path=config.model_path,
                dict_path=config.dict_path,
                use_space_char=True
            )
            self._available = True
            logger.info(
                f"TensorRTOpenOCRBackend initialized: {config.model_path} "
                f"(batch_size={self._batch_size})"
            )

        except Exception as e:
            logger.error(f"Failed to initialize TensorRT OpenOCR backend: {e}")
            import traceback
            traceback.print_exc()
            self._available = False

    @property
    def backend_name(self) -> str:
        return "tensorrt_openocr"

    @property
    def is_available(self) -> bool:
        return self._available and self._recognizer is not None

    def recognize(
        self,
        image: np.ndarray,
        return_confidence: bool = True
    ) -> Tuple[str, float]:
        if not self.is_available:
            return ("", 0.0)

        # Direct numpy array processing (no temp files needed!)
        result = self._recognizer.recognize(image)
        return (result['text'], result['confidence'])

    def recognize_batch(
        self,
        images: List[np.ndarray]
    ) -> List[Tuple[str, float]]:
        if not self.is_available:
            return [("", 0.0) for _ in images]

        # Direct numpy array batch processing
        results = self._recognizer.recognize_batch(
            images,
            use_true_batch=True
        )
        return [(r['text'], r['confidence']) for r in results]


class ONNXOpenOCRBackend(OCRBackendStrategy):
    """
    ONNX OCR Backend for OpenOCR RepSVTR model.

    Fallback option when TensorRT is not available.
    Performance: ~13 imgs/sec (batch=4) vs 127 imgs/sec (TensorRT).
    """

    def __init__(self, config: OCRConfig):
        """
        Initialize ONNX OpenOCR backend.

        Args:
            config: OCR configuration with model_path and dict_path
        """
        self._config = config
        self._recognizer = None
        self._available = False
        self._batch_size = config.batch_size

        try:
            # Import from camera_management directory (new implementation)
            from ..text_recognizer_openocr_onnx import TextRecognizerOpenOCRONNX

            if not os.path.exists(config.model_path):
                raise FileNotFoundError(f"ONNX model not found: {config.model_path}")
            if not os.path.exists(config.dict_path):
                raise FileNotFoundError(f"Dict file not found: {config.dict_path}")

            self._recognizer = TextRecognizerOpenOCRONNX(
                onnx_path=config.model_path,
                dict_path=config.dict_path,
                use_space_char=True
            )
            self._available = True
            logger.info(
                f"ONNXOpenOCRBackend initialized: {config.model_path} "
                f"(batch_size={self._batch_size})"
            )

        except Exception as e:
            logger.error(f"Failed to initialize ONNX OpenOCR backend: {e}")
            import traceback
            traceback.print_exc()
            self._available = False

    @property
    def backend_name(self) -> str:
        return "onnx_openocr"

    @property
    def is_available(self) -> bool:
        return self._available and self._recognizer is not None

    def recognize(
        self,
        image: np.ndarray,
        return_confidence: bool = True
    ) -> Tuple[str, float]:
        if not self.is_available:
            return ("", 0.0)

        # Direct numpy array processing (no temp files!)
        result = self._recognizer.recognize(image)
        return (result['text'], result['confidence'])

    def recognize_batch(
        self,
        images: List[np.ndarray]
    ) -> List[Tuple[str, float]]:
        if not self.is_available:
            return [("", 0.0) for _ in images]

        # Direct numpy array batch processing
        results = self._recognizer.recognize_batch(images, batch_size=self._batch_size)
        return [(r['text'], r['confidence']) for r in results]


# ============================================================================
# Factory
# ============================================================================

class OCRBackendFactory:
    """
    Factory for creating OCR backend instances.

    Supports:
    - Automatic backend selection based on availability
    - Manual backend selection
    - Default configuration paths
    - Multiple model architectures (PaddleV5, OpenOCR RepSVTR)

    Default: OpenOCR RepSVTR (9.77x faster than ONNX, batch_size=4 optimal)
    """

    # Default paths - OpenOCR RepSVTR (NEW MODEL)
    DEFAULT_TRT_ENGINE = f"{home}/Source/ocr_datecode/languages/english/openocr_rec_model_batch.engine"
    DEFAULT_ONNX_MODEL = f"{home}/Source/ocr_datecode/languages/english/openocr_rec_model.onnx"
    DEFAULT_DICT_PATH = f"{home}/Source/ocr_datecode/languages/english/ppocr_keys_v1.txt"

    # Legacy paths - PaddleV5 (OLD MODEL - for backward compatibility)
    LEGACY_TRT_ENGINE = f"{home}/Source/ocr_datecode/languages/english/rec.engine"
    LEGACY_ONNX_MODEL = f"{home}/Source/ocr_datecode/languages/english/rec.onnx"
    LEGACY_DICT_PATH = f"{home}/Source/ocr_datecode/languages/english/dict.txt"

    @classmethod
    def check_availability(cls) -> dict:
        """
        Check availability of all backends and models.

        Returns:
            Dict with backend availability status
        """
        result = {
            'tensorrt_openocr': False,
            'onnx_openocr': False,
            'tensorrt_paddlev5': False,  # Legacy
            'onnx_paddlev5': False,      # Legacy
        }

        # Check OpenOCR TensorRT (NEW - Priority 1)
        try:
            import tensorrt
            result['tensorrt_openocr'] = os.path.exists(cls.DEFAULT_TRT_ENGINE) and os.path.exists(cls.DEFAULT_DICT_PATH)
        except Exception:
            pass

        # Check OpenOCR ONNX (NEW - Priority 2)
        try:
            import onnxruntime
            result['onnx_openocr'] = os.path.exists(cls.DEFAULT_ONNX_MODEL) and os.path.exists(cls.DEFAULT_DICT_PATH)
        except Exception:
            pass

        # Check PaddleV5 TensorRT (LEGACY - Priority 3)
        try:
            from ..text_recognizer_trt import TENSORRT_AVAILABLE
            result['tensorrt_paddlev5'] = TENSORRT_AVAILABLE and os.path.exists(cls.LEGACY_TRT_ENGINE)
        except Exception:
            pass

        # Check PaddleV5 ONNX (LEGACY - Priority 4)
        try:
            import onnxruntime
            result['onnx_paddlev5'] = os.path.exists(cls.LEGACY_ONNX_MODEL)
        except Exception:
            pass

        return result

    @classmethod
    def create(
        cls,
        backend_type: OCRBackendType = OCRBackendType.AUTO,
        config: Optional[OCRConfig] = None,
        model_type: Optional[OCRModelType] = None
    ) -> Optional[OCRBackendStrategy]:
        """
        Create an OCR backend instance.

        Args:
            backend_type: Type of backend to create (AUTO, TENSORRT, ONNX)
            config: Optional configuration (uses defaults if not provided)
            model_type: Optional model type (defaults to OpenOCR RepSVTR)

        Returns:
            OCRBackendStrategy instance or None if no backend available

        Selection Priority (AUTO mode):
            1. TensorRT OpenOCR (127 imgs/sec, batch=4)
            2. ONNX OpenOCR (13 imgs/sec, batch=4)
            3. TensorRT PaddleV5 (legacy, ~40 imgs/sec)
            4. ONNX PaddleV5 (legacy, ~10 imgs/sec)
        """
        availability = cls.check_availability()
        logger.info(f"OCR Backend availability: {availability}")

        # Get model preference from environment or parameter
        if model_type is None:
            env_model = os.getenv("OCR_MODEL", "openocr").lower()
            model_type = OCRModelType.OPENOCR_REPSVTR if env_model == "openocr" else OCRModelType.PADDLEV5

        # Determine which backend to use
        selected_backend = None

        if backend_type == OCRBackendType.AUTO:
            # Priority: OpenOCR TensorRT > OpenOCR ONNX > PaddleV5 TensorRT > PaddleV5 ONNX
            if model_type == OCRModelType.OPENOCR_REPSVTR:
                if availability['tensorrt_openocr']:
                    selected_backend = 'tensorrt_openocr'
                elif availability['onnx_openocr']:
                    selected_backend = 'onnx_openocr'
                elif availability['tensorrt_paddlev5']:
                    logger.warning("OpenOCR not available, falling back to PaddleV5 TensorRT")
                    selected_backend = 'tensorrt_paddlev5'
                elif availability['onnx_paddlev5']:
                    logger.warning("OpenOCR not available, falling back to PaddleV5 ONNX")
                    selected_backend = 'onnx_paddlev5'
            else:
                # Legacy PaddleV5 requested
                if availability['tensorrt_paddlev5']:
                    selected_backend = 'tensorrt_paddlev5'
                elif availability['onnx_paddlev5']:
                    selected_backend = 'onnx_paddlev5'
                elif availability['tensorrt_openocr']:
                    logger.warning("PaddleV5 not available, falling back to OpenOCR TensorRT")
                    selected_backend = 'tensorrt_openocr'
                elif availability['onnx_openocr']:
                    logger.warning("PaddleV5 not available, falling back to OpenOCR ONNX")
                    selected_backend = 'onnx_openocr'

            if selected_backend is None:
                logger.error("No OCR backend available")
                return None

        elif backend_type == OCRBackendType.TENSORRT:
            # Manual TensorRT selection
            if model_type == OCRModelType.OPENOCR_REPSVTR and availability['tensorrt_openocr']:
                selected_backend = 'tensorrt_openocr'
            elif availability['tensorrt_paddlev5']:
                selected_backend = 'tensorrt_paddlev5'
            else:
                logger.error("TensorRT backend not available")
                return None

        elif backend_type == OCRBackendType.ONNX:
            # Manual ONNX selection
            if model_type == OCRModelType.OPENOCR_REPSVTR and availability['onnx_openocr']:
                selected_backend = 'onnx_openocr'
            elif availability['onnx_paddlev5']:
                selected_backend = 'onnx_paddlev5'
            else:
                logger.error("ONNX backend not available")
                return None

        # Create configuration if not provided
        if config is None:
            if selected_backend == 'tensorrt_openocr':
                config = OCRConfig(
                    model_path=cls.DEFAULT_TRT_ENGINE,
                    dict_path=cls.DEFAULT_DICT_PATH,
                    model_type=OCRModelType.OPENOCR_REPSVTR,
                    use_gpu=True,
                    batch_size=4  # Optimal from benchmark
                )
            elif selected_backend == 'onnx_openocr':
                config = OCRConfig(
                    model_path=cls.DEFAULT_ONNX_MODEL,
                    dict_path=cls.DEFAULT_DICT_PATH,
                    model_type=OCRModelType.OPENOCR_REPSVTR,
                    use_gpu=True,
                    batch_size=4
                )
            elif selected_backend == 'tensorrt_paddlev5':
                config = OCRConfig(
                    model_path=cls.LEGACY_TRT_ENGINE,
                    dict_path=cls.LEGACY_DICT_PATH,
                    model_type=OCRModelType.PADDLEV5,
                    use_gpu=True,
                    batch_size=1
                )
            else:  # onnx_paddlev5
                config = OCRConfig(
                    model_path=cls.LEGACY_ONNX_MODEL,
                    dict_path=cls.LEGACY_DICT_PATH,
                    model_type=OCRModelType.PADDLEV5,
                    use_gpu=True,
                    batch_size=1
                )

        # Create backend instance
        if selected_backend == 'tensorrt_openocr':
            backend = TensorRTOpenOCRBackend(config)
        elif selected_backend == 'onnx_openocr':
            backend = ONNXOpenOCRBackend(config)
        elif selected_backend == 'tensorrt_paddlev5':
            backend = TensorRTOCRBackend(config)
        elif selected_backend == 'onnx_paddlev5':
            backend = ONNXOCRBackend(config)
        else:
            logger.error(f"Unknown backend: {selected_backend}")
            return None

        if backend.is_available:
            logger.info(f"✅ Created OCR backend: {backend.backend_name} ({config.model_type.value})")
            return backend
        else:
            logger.error(f"❌ Failed to initialize {selected_backend}")
            return None

    @classmethod
    def create_from_env(cls) -> Optional[OCRBackendStrategy]:
        """
        Create OCR backend based on environment variable OCR_BACKEND.

        Environment variable values:
        - "tensorrt": Use TensorRT backend
        - "onnx": Use ONNX backend
        - "auto" (default): Auto-select best available

        Returns:
            OCRBackendStrategy instance or None if no backend available
        """
        env_backend = os.getenv("OCR_BACKEND", "auto").lower()

        backend_map = {
            "tensorrt": OCRBackendType.TENSORRT,
            "onnx": OCRBackendType.ONNX,
            "auto": OCRBackendType.AUTO
        }

        backend_type = backend_map.get(env_backend, OCRBackendType.AUTO)
        logger.info(f"Creating OCR backend from env: {env_backend} -> {backend_type}")

        return cls.create(backend_type)
