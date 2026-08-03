"""
SVTRv2 CTC — Pure TensorRT backend (pycuda, TensorRT 10.x)

Build engine once with trtexec:
    /usr/src/tensorrt/bin/trtexec \
        --onnx=languages/english/rec_model.onnx \
        --saveEngine=languages/english/rec_model_fp32.engine \
        --minShapes=input:1x3x48x320 \
        --optShapes=input:4x3x48x320 \
        --maxShapes=input:8x3x48x320 \
        --memPoolSize=workspace:4096
"""
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT/PyCUDA not available. SVTRv2 TRT will be disabled.")

# Reuse preprocessing and decode from ONNX module — no duplication
from .svtrv2_onnx import (
    _load_charset,
    _preprocess_image,
    _ctc_greedy_decode,
)


class TextRecognizerSVTRv2TRT:
    """
    SVTRv2 CTC text recognizer — Pure TensorRT engine backend.

    Pre-allocates CUDA buffers once in __init__ and reuses them
    for every inference call (same pattern as TextRecognizerOpenOCRTRT).

    Model input  : [N, 3, 48, 320]  — built with dynamic batch
    Model output : [N, T, 6625]     — CTC logits
    """

    def __init__(
        self,
        engine_path: str,
        dict_path: str,
        use_space_char: bool = True,
        imgH: int = 48,
        imgW: int = 320,
    ):
        if not TENSORRT_AVAILABLE:
            raise ImportError("tensorrt / pycuda not installed.")

        self.imgH = imgH
        self.imgW = imgW
        self.charset = _load_charset(dict_path, use_space_char)

        # Load engine
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(trt_logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream  = cuda.Stream()

        # Resolve I/O tensor names
        self.input_name = self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name
        assert self.input_name and self.output_name, "Could not resolve I/O tensor names"

        # Pre-allocate CUDA buffers at max profile shape
        in_max = list(self.engine.get_tensor_profile_shape(self.input_name, 0)[2])
        self.context.set_input_shape(self.input_name, in_max)
        out_max = list(self.context.get_tensor_shape(self.output_name))
        self._d_input  = cuda.mem_alloc(int(np.prod(in_max))  * 4)
        self._d_output = cuda.mem_alloc(int(np.prod(out_max)) * 4)
        self.context.set_tensor_address(self.input_name,  int(self._d_input))
        self.context.set_tensor_address(self.output_name, int(self._d_output))
        self._max_batch = in_max[0]

        # Warm-up
        self._infer_raw(np.zeros(in_max, dtype=np.float32))

        logger.info(f"TextRecognizerSVTRv2TRT | {Path(engine_path).name} | "
                    f"max_batch={self._max_batch} | {len(self.charset)} classes")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        return _preprocess_image(img, self.imgH, self.imgW)

    def _infer_raw(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        self.context.set_input_shape(self.input_name, x.shape)
        out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        cuda.memcpy_htod_async(self._d_input, x, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        output = np.empty(out_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output, self._d_output, self.stream)
        self.stream.synchronize()
        return output

    # ------------------------------------------------------------------
    # Public API  (matches TextRecognizerSVTRv2ONNX interface)
    # ------------------------------------------------------------------

    def recognize(self, image: np.ndarray, return_confidence: bool = True) -> Tuple[str, float]:
        """Single image → (text, confidence)"""
        x = self._preprocess(image)
        logits = self._infer_raw(x)
        texts, confs, _ = _ctc_greedy_decode(logits, self.charset)
        return (texts[0], confs[0]) if return_confidence else (texts[0], 0.0)

    def recognize_batch(self, images: List[np.ndarray], batch_size: int = 8) -> List[dict]:
        """
        List of images → list of {'text', 'confidence', 'char_confs'}
        batch_size must not exceed max_batch from engine profile.
        """
        if batch_size > self._max_batch:
            batch_size = self._max_batch
            logger.warning(f"batch_size clamped to engine max_batch={self._max_batch}")

        results = []
        for i in range(0, len(images), batch_size):
            chunk = images[i: i + batch_size]
            x = np.concatenate([self._preprocess(img) for img in chunk], axis=0)
            logits = self._infer_raw(x)
            texts, confs, char_confs = _ctc_greedy_decode(logits, self.charset)
            for text, conf, ccs in zip(texts, confs, char_confs):
                results.append({"text": text, "confidence": conf, "char_confs": ccs})
        return results

    def recognize_with_char_conf(self, image: np.ndarray) -> Tuple[str, float, list]:
        """
        Single image → (text, avg_conf, char_confs)
        char_confs: [(char, conf), ...] — chỉ alnum, bỏ ký tự đặc biệt và blank
        """
        x = self._preprocess(image)
        logits = self._infer_raw(x)
        texts, confs, char_confs = _ctc_greedy_decode(logits, self.charset)
        alnum_confs = [(c, cf) for c, cf in char_confs[0] if c.isalnum()]
        return texts[0], confs[0], alnum_confs

    def __del__(self):
        try:
            if hasattr(self, "_d_input"):  self._d_input.free()
            if hasattr(self, "_d_output"): self._d_output.free()
        except Exception:
            pass
