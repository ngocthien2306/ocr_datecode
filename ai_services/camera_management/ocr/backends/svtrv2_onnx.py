"""
SVTRv2 CTC — ONNX Runtime backend
Supports device: 'cpu' | 'cuda' | 'trt' (ORT-TensorRT EP)
"""
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available. SVTRv2 ONNX will be disabled.")


# ---------------------------------------------------------------------------
# Shared helpers (charset / preprocess / decode)
# ---------------------------------------------------------------------------

def _load_charset(dict_path: str, use_space_char: bool = True) -> List[str]:
    chars = []
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            ch = line.rstrip("\n")
            if ch:
                chars.append(ch)
    if use_space_char and " " not in chars:
        chars.append(" ")
    return [""] + chars   # index 0 = CTC blank


def _resize_keep_ratio_pad(img: np.ndarray, imgH: int = 48, imgW: int = 320) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"Invalid image shape: {img.shape}")
    ratio = w / float(h)
    new_w = min(int(np.ceil(imgH * ratio)), imgW)
    resized = cv2.resize(img, (new_w, imgH), interpolation=cv2.INTER_CUBIC)
    if new_w < imgW:
        pad = np.zeros((imgH, imgW - new_w, 3), dtype=resized.dtype)
        resized = np.concatenate([resized, pad], axis=1)
    else:
        resized = resized[:, :imgW, :]
    return resized


def _preprocess_image(img_bgr: np.ndarray, imgH: int = 48, imgW: int = 320) -> np.ndarray:
    """BGR → RGB → resize+pad to (imgH, imgW) → normalize → [1, 3, H, W]"""
    if len(img_bgr.shape) == 2:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = _resize_keep_ratio_pad(img, imgH=imgH, imgW=imgW)
    img = img.astype("float32") / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, 0).astype("float32")


def _ctc_greedy_decode(logits: np.ndarray, charset: List[str]):
    """
    logits: [batch, time_steps, num_classes]  (or transposed — auto-detected)
    Returns: texts, confs, char_confs
      char_confs: list of [(char, conf), ...]  per image
    """
    if logits.ndim != 3:
        raise ValueError(f"Unexpected logits shape: {logits.shape}")
    _, d1, _ = logits.shape
    if d1 == len(charset):          # shape is [B, C, T] — transpose
        logits = np.transpose(logits, (0, 2, 1))

    pred = np.argmax(logits, axis=2)
    texts, confs, char_confs = [], [], []
    blank_id = 0
    for b in range(pred.shape[0]):
        out_chars, out_confs, last = [], [], None
        for t, p in enumerate(pred[b].tolist()):
            if p != blank_id and p != last and 0 <= p < len(charset):
                out_chars.append(charset[p])
                out_confs.append(float(np.max(logits[b, t])))
            last = p
        texts.append("".join(out_chars))
        confs.append(float(np.mean(out_confs)) if out_confs else 0.0)
        char_confs.append(list(zip(out_chars, out_confs)))
    return texts, confs, char_confs


# ---------------------------------------------------------------------------
# ONNX session builder
# ---------------------------------------------------------------------------

def _build_session(model_path: str, device: str) -> "ort.InferenceSession":
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    if device == "cpu":
        providers = ["CPUExecutionProvider"]
    elif device == "cuda":
        providers = [
            ("CUDAExecutionProvider", {
                "device_id": 0,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            }),
            "CPUExecutionProvider",
        ]
    elif device == "trt":
        providers = [
            ("TensorrtExecutionProvider", {
                "trt_fp16_enable": False,
                "trt_max_workspace_size": 4 * 1024 * 1024 * 1024,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "./trt_cache",
                "trt_min_subgraph_size": 3,
            }),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    else:
        raise ValueError(f"Unknown device: {device!r}. Choose 'cpu' | 'cuda' | 'trt'")

    return ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)


# ---------------------------------------------------------------------------
# Main recognizer class
# ---------------------------------------------------------------------------

class TextRecognizerSVTRv2ONNX:
    """
    SVTRv2 CTC text recognizer — ONNX Runtime backend.

    device options:
      'cuda' — CUDAExecutionProvider + IO Binding  (recommended, ~60-80ms/batch)
      'trt'  — ORT-TensorRT EP (builds & caches engine on first run, ~60ms/batch)
      'cpu'  — CPUExecutionProvider (no GPU required)

    Model input  : [N, 3, 48, 320]  fixed width=320, padded
    Model output : [N, T, 6625]     CTC logits, 6625 character classes
    """

    def __init__(
        self,
        onnx_path: str,
        dict_path: str,
        device: str = "cuda",
        use_space_char: bool = True,
        imgH: int = 48,
        imgW: int = 320,
    ):
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime not installed.")

        self.imgH = imgH
        self.imgW = imgW
        self._use_iobinding = device in ("cuda", "trt")

        self.charset = _load_charset(dict_path, use_space_char)
        self.sess    = _build_session(onnx_path, device)
        self.in_name  = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

        active = self.sess.get_providers()[0]
        logger.info(f"TextRecognizerSVTRv2ONNX | {Path(onnx_path).name} | "
                    f"provider={active} | {len(self.charset)} classes")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        return _preprocess_image(img, self.imgH, self.imgW)

    def _infer(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if not self._use_iobinding:
            return self.sess.run([self.out_name], {self.in_name: x})[0]
        # IO binding — keeps tensors on GPU, one D2H copy at the end
        io = self.sess.io_binding()
        x_ort = ort.OrtValue.ortvalue_from_numpy(x, "cuda", 0)
        io.bind_ortvalue_input(self.in_name, x_ort)
        io.bind_output(self.out_name, device_type="cuda", device_id=0,
                       element_type=np.float32)
        self.sess.run_with_iobinding(io)
        return io.get_outputs()[0].numpy()

    # ------------------------------------------------------------------
    # Public API  (matches TextRecognizerOpenOCRONNX interface)
    # ------------------------------------------------------------------

    def recognize(self, image: np.ndarray, return_confidence: bool = True) -> Tuple[str, float]:
        """Single image → (text, confidence)"""
        x = self._preprocess(image)
        logits = self._infer(x)
        texts, confs, _ = _ctc_greedy_decode(logits, self.charset)
        return (texts[0], confs[0]) if return_confidence else (texts[0], 0.0)

    def recognize_batch(self, images: List[np.ndarray], batch_size: int = 8) -> List[dict]:
        """
        List of images → list of {'text', 'confidence', 'char_confs'}
        Images are processed in chunks of batch_size.
        All images are padded to imgW=320, so no dynamic padding needed.
        """
        results = []
        for i in range(0, len(images), batch_size):
            chunk = images[i: i + batch_size]
            x = np.concatenate([self._preprocess(img) for img in chunk], axis=0)
            logits = self._infer(x)
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
        logits = self._infer(x)
        texts, confs, char_confs = _ctc_greedy_decode(logits, self.charset)
        alnum_confs = [(c, cf) for c, cf in char_confs[0] if c.isalnum()]
        return texts[0], confs[0], alnum_confs
