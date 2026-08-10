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
    OPENOCR_REPSVTR = "openocr"     # OpenOCR RepSVTR
    SVTRV2_CTC = "svtrv2"           # SVTRv2 CTC (6625 classes, width=320)
    SMTR = "smtr"                   # SMTR dual-head (GTC + CTC, dynamic width)
    # A model fine-tuned in the OCR Training Studio (ocr_service). Same SMTR
    # architecture and the same backends as SMTR — only the weights and the dict
    # come from somewhere else, so there are no default paths for it: the recipe
    # supplies engine_path + dict_path and create() REQUIRES a config.
    CUSTOM = "custom"


@dataclass
class OCRConfig:
    """Configuration for OCR backend"""
    model_path: str
    dict_path: str
    model_type: OCRModelType = OCRModelType.OPENOCR_REPSVTR
    use_gpu: bool = True
    min_width: int = 50
    max_width: int = 2000
    batch_size: int = 4
    device: str = "cuda"    # For ONNX backends: 'cpu' | 'cuda' | 'trt'


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
            from .backends.default_trt import TextRecognizerTRT, TENSORRT_AVAILABLE

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

    def recognize_with_char_conf(self, image: np.ndarray) -> tuple:
        if not self.is_available:
            return ("", 0.0, [])
        return self._recognizer.recognize_with_char_conf(image)


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

    def recognize_with_char_conf(self, image: np.ndarray) -> tuple:
        if not self.is_available:
            return ("", 0.0, [])
        return self._recognizer.recognize_with_char_conf(image)


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
            from .backends.openocr_trt import TextRecognizerOpenOCRTRT

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
        return self._recognizer.recognize(image, return_confidence=return_confidence)

    def recognize_batch(
        self,
        images: List[np.ndarray]
    ) -> List[Tuple[str, float]]:
        if not self.is_available:
            return [("", 0.0) for _ in images]

        # Direct numpy array batch processing
        return self._recognizer.recognize_batch(images, use_true_batch=True)

    def recognize_with_char_conf(self, image: np.ndarray) -> tuple:
        if not self.is_available:
            return ("", 0.0, [])
        return self._recognizer.recognize_with_char_conf(image)


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
            from .backends.openocr_onnx import TextRecognizerOpenOCRONNX

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
        return self._recognizer.recognize(image, return_confidence=return_confidence)

    def recognize_batch(
        self,
        images: List[np.ndarray]
    ) -> List[Tuple[str, float]]:
        if not self.is_available:
            return [("", 0.0) for _ in images]

        # Direct numpy array batch processing
        return self._recognizer.recognize_batch(images, batch_size=self._batch_size)

    def recognize_with_char_conf(self, image: np.ndarray) -> tuple:
        if not self.is_available:
            return ("", 0.0, [])
        return self._recognizer.recognize_with_char_conf(image)


class SVTRv2ONNXBackend(OCRBackendStrategy):
    """ONNX Runtime backend for SVTRv2 CTC model (device: cpu/cuda/trt EP)."""

    def __init__(self, config: OCRConfig):
        self._config = config
        self._recognizer = None
        self._available = False
        try:
            from .backends.svtrv2_onnx import TextRecognizerSVTRv2ONNX
            if not os.path.exists(config.model_path):
                raise FileNotFoundError(f"ONNX model not found: {config.model_path}")
            if not os.path.exists(config.dict_path):
                raise FileNotFoundError(f"Dict not found: {config.dict_path}")
            self._recognizer = TextRecognizerSVTRv2ONNX(
                onnx_path=config.model_path,
                dict_path=config.dict_path,
                device=config.device,
            )
            self._available = True
            logger.info(f"SVTRv2ONNXBackend initialized: {config.model_path} (device={config.device})")
        except Exception as e:
            logger.error(f"Failed to initialize SVTRv2ONNXBackend: {e}")

    @property
    def backend_name(self) -> str:
        return f"svtrv2_onnx_{self._config.device}"

    @property
    def is_available(self) -> bool:
        return self._available and self._recognizer is not None

    def recognize(self, image: np.ndarray, return_confidence: bool = True) -> Tuple[str, float]:
        if not self.is_available:
            return ("", 0.0)
        return self._recognizer.recognize(image, return_confidence=return_confidence)

    def recognize_batch(self, images: List[np.ndarray]) -> List[Tuple[str, float]]:
        if not self.is_available:
            return [("", 0.0) for _ in images]
        return self._recognizer.recognize_batch(images, batch_size=self._config.batch_size)

    def recognize_with_char_conf(self, image: np.ndarray) -> tuple:
        if not self.is_available:
            return ("", 0.0, [])
        return self._recognizer.recognize_with_char_conf(image)


class SMTRONNXBackend(OCRBackendStrategy):
    def __init__(self, config: OCRConfig):
        self._config = config
        self._recognizer = None
        self._available = False
        try:
            from .backends.smtr_onnx import TextRecognizerSMTRONNX
            if not os.path.exists(config.model_path):
                raise FileNotFoundError(f"ONNX model not found: {config.model_path}")
            if not os.path.exists(config.dict_path):
                raise FileNotFoundError(f"Dict not found: {config.dict_path}")
            self._recognizer = TextRecognizerSMTRONNX(
                onnx_path=config.model_path,
                dict_path=config.dict_path,
                device=config.device,
            )
            self._available = True
            logger.info(f"SMTRONNXBackend initialized: {config.model_path}")
        except Exception as e:
            logger.error(f"Failed to init SMTRONNXBackend: {e}")

    @property
    def backend_name(self) -> str:
        return f"smtr_onnx_{self._config.device}"

    @property
    def is_available(self) -> bool:
        return self._available and self._recognizer is not None

    def recognize(self, image: np.ndarray, return_confidence: bool = True):
        if not self.is_available:
            return [("", 0.0, []), ("", 0.0, [])]
        return self._recognizer.recognize(image)

    def recognize_batch(self, images: List[np.ndarray]):
        if not self.is_available:
            return [[("", 0.0, []), ("", 0.0, [])] for _ in images]
        return self._recognizer.recognize_batch(images)

    def recognize_with_char_conf(self, image: np.ndarray):
        if not self.is_available:
            return [("", 0.0, []), ("", 0.0, [])]
        return self._recognizer.recognize_with_char_conf(image)


class SMTRTRTBackend(OCRBackendStrategy):
    def __init__(self, config: OCRConfig):
        self._config = config
        self._recognizer = None
        self._available = False
        try:
            from .backends.smtr_trt import TextRecognizerSMTRTRT, TENSORRT_AVAILABLE
            if not TENSORRT_AVAILABLE:
                logger.warning("TensorRT not available for SMTR")
                return
            if not os.path.exists(config.model_path):
                raise FileNotFoundError(f"Engine not found: {config.model_path}")
            if not os.path.exists(config.dict_path):
                raise FileNotFoundError(f"Dict not found: {config.dict_path}")
            self._recognizer = TextRecognizerSMTRTRT(
                engine_path=config.model_path,
                dict_path=config.dict_path,
            )
            self._available = True
            logger.info(f"SMTRTRTBackend initialized: {config.model_path}")
        except Exception as e:
            logger.error(f"Failed to init SMTRTRTBackend: {e}")

    @property
    def backend_name(self) -> str:
        return "smtr_trt"

    @property
    def is_available(self) -> bool:
        return self._available and self._recognizer is not None

    def recognize(self, image: np.ndarray, return_confidence: bool = True):
        if not self.is_available:
            return [("", 0.0, []), ("", 0.0, [])]
        return self._recognizer.recognize(image)

    def recognize_batch(self, images: List[np.ndarray]):
        if not self.is_available:
            return [[("", 0.0, []), ("", 0.0, [])] for _ in images]
        return self._recognizer.recognize_batch(images)

    def recognize_with_char_conf(self, image: np.ndarray):
        if not self.is_available:
            return [("", 0.0, []), ("", 0.0, [])]
        return self._recognizer.recognize_with_char_conf(image)


class SVTRv2TRTBackend(OCRBackendStrategy):
    """Pure TensorRT backend for SVTRv2 CTC model (.engine file, pycuda)."""

    def __init__(self, config: OCRConfig):
        self._config = config
        self._recognizer = None
        self._available = False
        try:
            from .backends.svtrv2_trt import TextRecognizerSVTRv2TRT, TENSORRT_AVAILABLE
            if not TENSORRT_AVAILABLE:
                logger.warning("TensorRT not available for SVTRv2")
                return
            if not os.path.exists(config.model_path):
                raise FileNotFoundError(f"Engine not found: {config.model_path}")
            if not os.path.exists(config.dict_path):
                raise FileNotFoundError(f"Dict not found: {config.dict_path}")
            self._recognizer = TextRecognizerSVTRv2TRT(
                engine_path=config.model_path,
                dict_path=config.dict_path,
            )
            self._available = True
            logger.info(f"SVTRv2TRTBackend initialized: {config.model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize SVTRv2TRTBackend: {e}")

    @property
    def backend_name(self) -> str:
        return "svtrv2_trt"

    @property
    def is_available(self) -> bool:
        return self._available and self._recognizer is not None

    def recognize(self, image: np.ndarray, return_confidence: bool = True) -> Tuple[str, float]:
        if not self.is_available:
            return ("", 0.0)
        return self._recognizer.recognize(image, return_confidence=return_confidence)

    def recognize_batch(self, images: List[np.ndarray]) -> List[Tuple[str, float]]:
        if not self.is_available:
            return [("", 0.0) for _ in images]
        return self._recognizer.recognize_batch(images, batch_size=self._config.batch_size)

    def recognize_with_char_conf(self, image: np.ndarray) -> tuple:
        if not self.is_available:
            return ("", 0.0, [])
        return self._recognizer.recognize_with_char_conf(image)


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

    # SVTRv2 CTC paths
    SVTRV2_TRT_ENGINE = f"{home}/Source/ocr_datecode/languages/english/rec_model_fp16.engine"
    SVTRV2_ONNX_MODEL = f"{home}/Source/ocr_datecode/languages/english/rec_model.onnx"
    SVTRV2_DICT_PATH  = f"{home}/Source/ocr_datecode/languages/english/ppocr_keys_v1.txt"

    # SMTR dual-head paths
    SMTR_TRT_ENGINE = f"{home}/Source/ocr_datecode/languages/english/rec_smtr_attn_fp16.engine"
    SMTR_ONNX_MODEL = f"{home}/Source/ocr_datecode/languages/english/rec_smtr_fp16.onnx"
    SMTR_DICT_PATH  = f"{home}/Source/ocr_datecode/languages/english/EN_symbol_dict.txt"

    # -----------------------------------------------------------------------
    # Registry: (model_type, backend_type) → (AdapterClass, model_path, dict_path, extra_kwargs)
    # To add a new model: add 2 entries here + paths above.
    # -----------------------------------------------------------------------
    _REGISTRY = None   # built lazily to avoid circular imports

    @classmethod
    def _get_registry(cls) -> dict:
        if cls._REGISTRY is None:
            cls._REGISTRY = {
                (OCRModelType.SMTR,            OCRBackendType.TENSORRT): (SMTRTRTBackend,    cls.SMTR_TRT_ENGINE,    cls.SMTR_DICT_PATH,    {}),
                (OCRModelType.SMTR,            OCRBackendType.ONNX):     (SMTRONNXBackend,   cls.SMTR_ONNX_MODEL,    cls.SMTR_DICT_PATH,    {"device": "cuda"}),
                # CUSTOM reuses the SMTR adapters. The paths here are only
                # placeholders — create() rejects CUSTOM without a config, so
                # they are never the ones actually loaded.
                (OCRModelType.CUSTOM,          OCRBackendType.TENSORRT): (SMTRTRTBackend,    cls.SMTR_TRT_ENGINE,    cls.SMTR_DICT_PATH,    {}),
                (OCRModelType.CUSTOM,          OCRBackendType.ONNX):     (SMTRONNXBackend,   cls.SMTR_ONNX_MODEL,    cls.SMTR_DICT_PATH,    {"device": "cuda"}),
                (OCRModelType.SVTRV2_CTC,      OCRBackendType.TENSORRT): (SVTRv2TRTBackend,  cls.SVTRV2_TRT_ENGINE,  cls.SVTRV2_DICT_PATH,  {}),
                (OCRModelType.SVTRV2_CTC,      OCRBackendType.ONNX):     (SVTRv2ONNXBackend, cls.SVTRV2_ONNX_MODEL,  cls.SVTRV2_DICT_PATH,  {"device": "cuda"}),
                (OCRModelType.OPENOCR_REPSVTR, OCRBackendType.TENSORRT): (TensorRTOpenOCRBackend, cls.DEFAULT_TRT_ENGINE, cls.DEFAULT_DICT_PATH, {}),
                (OCRModelType.OPENOCR_REPSVTR, OCRBackendType.ONNX):     (ONNXOpenOCRBackend,     cls.DEFAULT_ONNX_MODEL, cls.DEFAULT_DICT_PATH, {}),
                (OCRModelType.PADDLEV5,        OCRBackendType.TENSORRT): (TensorRTOCRBackend, cls.LEGACY_TRT_ENGINE,  cls.LEGACY_DICT_PATH,  {}),
                (OCRModelType.PADDLEV5,        OCRBackendType.ONNX):     (ONNXOCRBackend,     cls.LEGACY_ONNX_MODEL,  cls.LEGACY_DICT_PATH,  {}),
            }
        return cls._REGISTRY

    @classmethod
    def check_availability(cls) -> dict:
        """
        Check availability of all backends and models.

        Returns:
            Dict with backend availability status
        """
        result = {
            'tensorrt_openocr':  False,
            'onnx_openocr':      False,
            'tensorrt_paddlev5': False,
            'onnx_paddlev5':     False,
            'tensorrt_svtrv2':   False,
            'onnx_svtrv2':       False,
            'tensorrt_smtr':     False,
            'onnx_smtr':         False,
        }

        trt_ok, onnx_ok = False, False
        try:
            import tensorrt  # noqa: F401
            trt_ok = True
        except Exception:
            pass
        try:
            import onnxruntime  # noqa: F401
            onnx_ok = True
        except Exception:
            pass

        result['tensorrt_openocr']  = trt_ok  and os.path.exists(cls.DEFAULT_TRT_ENGINE) and os.path.exists(cls.DEFAULT_DICT_PATH)
        result['onnx_openocr']      = onnx_ok and os.path.exists(cls.DEFAULT_ONNX_MODEL) and os.path.exists(cls.DEFAULT_DICT_PATH)
        result['tensorrt_paddlev5'] = trt_ok  and os.path.exists(cls.LEGACY_TRT_ENGINE)  and os.path.exists(cls.LEGACY_DICT_PATH)
        result['onnx_paddlev5']     = onnx_ok and os.path.exists(cls.LEGACY_ONNX_MODEL)  and os.path.exists(cls.LEGACY_DICT_PATH)
        result['tensorrt_svtrv2']   = trt_ok  and os.path.exists(cls.SVTRV2_TRT_ENGINE)  and os.path.exists(cls.SVTRV2_DICT_PATH)
        result['onnx_svtrv2']       = onnx_ok and os.path.exists(cls.SVTRV2_ONNX_MODEL)  and os.path.exists(cls.SVTRV2_DICT_PATH)
        result['tensorrt_smtr']     = trt_ok  and os.path.exists(cls.SMTR_TRT_ENGINE)    and os.path.exists(cls.SMTR_DICT_PATH)
        result['onnx_smtr']         = onnx_ok and os.path.exists(cls.SMTR_ONNX_MODEL)    and os.path.exists(cls.SMTR_DICT_PATH)

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
        # Default model type if not specified
        if model_type is None:
            model_type = OCRModelType.PADDLEV5

        registry = cls._get_registry()

        # ── CUSTOM: paths come from the recipe, so a config is mandatory ─────
        # It also bypasses AUTO: check_availability() only knows the built-in
        # paths, so it cannot tell whether a per-recipe engine exists. The
        # extension decides the backend instead, which is unambiguous.
        if model_type == OCRModelType.CUSTOM:
            if config is None or not config.model_path:
                logger.error(
                    "OCRModelType.CUSTOM requires an OCRConfig with model_path "
                    "(the recipe's engine/onnx) — refusing to fall back to the "
                    "built-in SMTR weights, which would silently read with the "
                    "wrong model"
                )
                return None
            if not os.path.exists(config.model_path):
                logger.error(f"Custom OCR model not found on disk: {config.model_path}")
                return None
            if not os.path.exists(config.dict_path):
                # The dict decides every character index; a wrong one decodes
                # into garbage rather than failing, so this must not be lenient.
                logger.error(f"Custom OCR dict not found on disk: {config.dict_path}")
                return None
            resolved = (OCRBackendType.ONNX if config.model_path.endswith(".onnx")
                        else OCRBackendType.TENSORRT)
            AdapterCls = registry[(OCRModelType.CUSTOM, resolved)][0]
            backend = AdapterCls(config)
            if backend.is_available:
                logger.info(f"✅ OCR backend (custom): {backend.backend_name} "
                            f"← {os.path.basename(config.model_path)}")
                return backend
            logger.error(f"❌ Failed to init custom OCR backend from {config.model_path}")
            return None

        # ── Manual selection ────────────────────────────────────────────────
        if backend_type != OCRBackendType.AUTO:
            key = (model_type, backend_type)
            if key not in registry:
                logger.error(f"No backend registered for {model_type} + {backend_type}")
                return None
            AdapterCls, default_model, default_dict, extra = registry[key]
            if config is None:
                config = OCRConfig(
                    model_path=default_model,
                    dict_path=default_dict,
                    model_type=model_type,
                    **{k: v for k, v in extra.items() if k in OCRConfig.__dataclass_fields__},
                )
            backend = AdapterCls(config)
            if backend.is_available:
                logger.info(f"✅ OCR backend: {backend.backend_name}")
                return backend
            logger.error(f"❌ Failed to init {AdapterCls.__name__}")
            return None

        # ── AUTO: prefer TensorRT over ONNX for the requested model ────────
        availability = cls.check_availability()
        logger.info(f"OCR availability: {availability}")

        # Priority order per model type
        _auto_priority = {
            OCRModelType.SMTR:            [OCRBackendType.TENSORRT, OCRBackendType.ONNX],
            # CUSTOM never reaches here — it returns above, before AUTO.
            OCRModelType.SVTRV2_CTC:      [OCRBackendType.TENSORRT, OCRBackendType.ONNX],
            OCRModelType.OPENOCR_REPSVTR: [OCRBackendType.TENSORRT, OCRBackendType.ONNX],
            OCRModelType.PADDLEV5:        [OCRBackendType.TENSORRT, OCRBackendType.ONNX],
        }
        _avail_key = {
            (OCRModelType.SMTR,            OCRBackendType.TENSORRT): 'tensorrt_smtr',
            (OCRModelType.SMTR,            OCRBackendType.ONNX):     'onnx_smtr',
            (OCRModelType.SVTRV2_CTC,      OCRBackendType.TENSORRT): 'tensorrt_svtrv2',
            (OCRModelType.SVTRV2_CTC,      OCRBackendType.ONNX):     'onnx_svtrv2',
            (OCRModelType.OPENOCR_REPSVTR, OCRBackendType.TENSORRT): 'tensorrt_openocr',
            (OCRModelType.OPENOCR_REPSVTR, OCRBackendType.ONNX):     'onnx_openocr',
            (OCRModelType.PADDLEV5,        OCRBackendType.TENSORRT): 'tensorrt_paddlev5',
            (OCRModelType.PADDLEV5,        OCRBackendType.ONNX):     'onnx_paddlev5',
        }

        for bt in _auto_priority.get(model_type, []):
            key = (model_type, bt)
            avail_key = _avail_key.get(key)
            if avail_key and availability.get(avail_key):
                AdapterCls, default_model, default_dict, extra = registry[key]
                if config is None:
                    config = OCRConfig(
                        model_path=default_model,
                        dict_path=default_dict,
                        model_type=model_type,
                        **{k: v for k, v in extra.items() if k in OCRConfig.__dataclass_fields__},
                    )
                backend = AdapterCls(config)
                if backend.is_available:
                    logger.info(f"✅ OCR backend (auto): {backend.backend_name}")
                    return backend

        logger.error(f"No available OCR backend for model_type={model_type}")
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
