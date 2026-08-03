import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from ..smtr_utils import build_postprocessors, preprocess, preprocess_batch, _decode_dual


class TextRecognizerSMTRONNX:
    def __init__(self, onnx_path: str, dict_path: str, device: str = 'cuda'):
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime not installed")

        self._gtc, self._ctc = build_postprocessors(dict_path)

        if device == 'trt':
            providers = [
                ("TensorrtExecutionProvider", {
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "./trt_cache",
                }),
                "CUDAExecutionProvider", "CPUExecutionProvider",
            ]
        elif device == 'cuda':
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self._sess = ort.InferenceSession(onnx_path, providers=providers)
        logger.info(f"TextRecognizerSMTRONNX | {Path(onnx_path).name} | device={device}")

    def _infer(self, inp: np.ndarray):
        return self._sess.run(None, {'image': inp.astype(np.float32)})

    def recognize(self, image: np.ndarray):
        gtc, ctc = self._infer(preprocess(image))
        return _decode_dual(gtc, ctc, self._gtc, self._ctc)[0]

    def recognize_with_char_conf(self, image: np.ndarray):
        return self.recognize(image)

    def recognize_batch(self, images: list):
        if not images:
            return []
        gtc, ctc = self._infer(preprocess_batch(images))
        return _decode_dual(gtc, ctc, self._gtc, self._ctc)
