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


@dataclass
class OCRConfig:
    """Configuration for OCR backend"""
    model_path: str
    dict_path: str
    use_gpu: bool = True
    min_width: int = 50
    max_width: int = 2000


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
    ONNX OCR Backend adapter.

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
            logger.info(f"ONNXOCRBackend initialized: {config.model_path}")

        except Exception as e:
            logger.error(f"Failed to initialize ONNX backend: {e}")
            self._available = False

    @property
    def backend_name(self) -> str:
        return "onnx"

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
    """

    # Default paths
    DEFAULT_TRT_ENGINE = f"{home}/Source/ocr_datecode/languages/english/rec.engine"
    DEFAULT_ONNX_MODEL = f"{home}/Source/ocr_datecode/languages/english/rec.onnx"
    DEFAULT_DICT_PATH = f"{home}/Source/ocr_datecode/languages/english/dict.txt"

    @classmethod
    def check_availability(cls) -> dict:
        """
        Check availability of all backends.

        Returns:
            Dict with backend availability status
        """
        result = {
            'tensorrt': False,
            'onnx': False
        }

        # Check TensorRT
        try:
            from ..text_recognizer_trt import TENSORRT_AVAILABLE
            result['tensorrt'] = TENSORRT_AVAILABLE and os.path.exists(cls.DEFAULT_TRT_ENGINE)
        except Exception:
            pass

        # Check ONNX
        try:
            import onnxruntime
            result['onnx'] = os.path.exists(cls.DEFAULT_ONNX_MODEL)
        except Exception:
            pass

        return result

    @classmethod
    def create(
        cls,
        backend_type: OCRBackendType = OCRBackendType.AUTO,
        config: Optional[OCRConfig] = None
    ) -> Optional[OCRBackendStrategy]:
        """
        Create an OCR backend instance.

        Args:
            backend_type: Type of backend to create (AUTO, TENSORRT, ONNX)
            config: Optional configuration (uses defaults if not provided)

        Returns:
            OCRBackendStrategy instance or None if no backend available
        """
        availability = cls.check_availability()
        logger.info(f"OCR Backend availability: {availability}")

        # Determine which backend to use
        if backend_type == OCRBackendType.AUTO:
            # Priority: TensorRT > ONNX
            if availability['tensorrt']:
                backend_type = OCRBackendType.TENSORRT
            elif availability['onnx']:
                backend_type = OCRBackendType.ONNX
            else:
                logger.error("No OCR backend available")
                return None

        # Create configuration if not provided
        if config is None:
            if backend_type == OCRBackendType.TENSORRT:
                config = OCRConfig(
                    model_path=cls.DEFAULT_TRT_ENGINE,
                    dict_path=cls.DEFAULT_DICT_PATH,
                    use_gpu=True
                )
            else:
                config = OCRConfig(
                    model_path=cls.DEFAULT_ONNX_MODEL,
                    dict_path=cls.DEFAULT_DICT_PATH,
                    use_gpu=True
                )

        # Create backend
        if backend_type == OCRBackendType.TENSORRT:
            backend = TensorRTOCRBackend(config)
        elif backend_type == OCRBackendType.ONNX:
            backend = ONNXOCRBackend(config)
        else:
            logger.error(f"Unknown backend type: {backend_type}")
            return None

        if backend.is_available:
            logger.info(f"Created OCR backend: {backend}")
            return backend
        else:
            logger.warning(f"Backend {backend_type} not available, trying fallback...")
            # Try fallback
            if backend_type == OCRBackendType.TENSORRT and availability['onnx']:
                return cls.create(OCRBackendType.ONNX)
            elif backend_type == OCRBackendType.ONNX and availability['tensorrt']:
                return cls.create(OCRBackendType.TENSORRT)

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
