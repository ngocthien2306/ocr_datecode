"""
ML Char OCR Service (minimal port of ai_services TextRecognizer).

Loads rec.onnx once (lazy singleton) to predict the character identity of
each char-level crop during ML labeling. Narrow chars like I/l/1// are
padded before resize to avoid the model rejecting tiny widths.

Only used by the ML training studio — NOT by production inference.
"""
import logging
from pathlib import Path
from threading import Lock
from typing import List

import cv2 as cv
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

# ── Paths (backend → repo root → languages/english) ────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_MODEL_PATH = _PROJECT_ROOT / "languages" / "english" / "rec.onnx"
_DICT_PATH = _PROJECT_ROOT / "languages" / "english" / "dict.txt"

# ── Config ──────────────────────────────────────────────────────────────────
_TARGET_H = 48
_MIN_W = 32       # pad narrow chars to this width before inference

# ── Singleton state ─────────────────────────────────────────────────────────
_session: "ort.InferenceSession | None" = None
_char_list: "List[str] | None" = None
_input_name: "str | None" = None
_output_name: "str | None" = None
_init_lock = Lock()


def _load_dict(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def _get_session():
    """Lazy-initialize ONNX session (CUDA → CPU fallback)."""
    global _session, _char_list, _input_name, _output_name
    if _session is not None:
        return _session
    with _init_lock:
        if _session is not None:
            return _session

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess = ort.InferenceSession(str(_MODEL_PATH), providers=providers)

        char_dict = _load_dict(_DICT_PATH)
        char_list = ["blank"] + char_dict

        # Pad char_list if model output is larger than dict (legacy rec.onnx)
        out_shape = sess.get_outputs()[0].shape
        expected_classes = out_shape[-1] if isinstance(out_shape[-1], int) else None
        if expected_classes and len(char_list) < expected_classes:
            char_list.extend([" "] * (expected_classes - len(char_list)))

        _input_name = sess.get_inputs()[0].name
        _output_name = sess.get_outputs()[0].name
        _char_list = char_list
        _session = sess

        logger.info(
            f"[ml_char_ocr] loaded rec.onnx — provider={sess.get_providers()[0]}, "
            f"dict={len(char_dict)}, classes={expected_classes}"
        )
        return _session


def _preprocess(image: np.ndarray) -> np.ndarray:
    """
    BGR/gray → resize (h=48, keep aspect) → pad right to MIN_W → normalize to
    [-1, 1] 3-channel CHW float32. Narrow chars get padded so the ONNX model
    doesn't reject them.
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        gray = np.zeros((_TARGET_H, _MIN_W), dtype=np.uint8)
    else:
        ratio = _TARGET_H / float(h)
        new_w = max(int(round(w * ratio)), 1)
        gray = cv.resize(gray, (new_w, _TARGET_H), interpolation=cv.INTER_LINEAR)

    # Pad right to reach MIN_W (replicate edge so narrow strokes don't
    # produce a hard black bar that the model interprets as a glyph boundary)
    if gray.shape[1] < _MIN_W:
        pad = _MIN_W - gray.shape[1]
        gray = cv.copyMakeBorder(gray, 0, 0, 0, pad, cv.BORDER_REPLICATE)

    normalized = gray.astype(np.float32) / 255.0
    normalized = (normalized - 0.5) / 0.5
    chw = np.stack([normalized, normalized, normalized], axis=0)
    return chw.astype(np.float32)


def _decode_ctc(pred: np.ndarray) -> str:
    """CTC greedy decode on a single prediction map [T, C]."""
    best_path = np.argmax(pred, axis=-1)
    chars: List[str] = []
    prev = -1
    for idx in best_path:
        idx_i = int(idx)
        if idx_i != 0 and idx_i != prev and idx_i < len(_char_list):
            chars.append(_char_list[idx_i])
        prev = idx_i
    return "".join(chars)


def _first_char(text: str) -> str:
    """Constrain to 1 character. Strip whitespace first, then take first."""
    text = (text or "").strip()
    return text[0] if text else ""


def recognize_char(image: np.ndarray) -> str:
    """Return the first recognized character for a single crop, or '' on failure."""
    if image is None or image.size == 0:
        return ""
    try:
        sess = _get_session()
        tensor = _preprocess(image)
        batch = np.expand_dims(tensor, axis=0)
        preds = sess.run([_output_name], {_input_name: batch})[0]
        return _first_char(_decode_ctc(preds[0]))
    except Exception as e:
        logger.warning(f"[ml_char_ocr] recognize_char failed: {e}")
        return ""


def recognize_chars(images: List[np.ndarray]) -> List[str]:
    """
    Batch OCR over N crops. Output list is parallel to input; failed items
    become ''. Crops of different widths are right-padded to max_width.
    """
    if not images:
        return []
    try:
        sess = _get_session()
        tensors = [_preprocess(img) if img is not None and img.size > 0 else None for img in images]
        valid_idx = [i for i, t in enumerate(tensors) if t is not None]
        if not valid_idx:
            return ["" for _ in images]

        # Pad to common max width (last dim of CHW tensor = shape[2])
        max_w = max(tensors[i].shape[2] for i in valid_idx)
        padded = []
        for i in valid_idx:
            t = tensors[i]
            if t.shape[2] < max_w:
                pad_w = max_w - t.shape[2]
                t = np.pad(t, ((0, 0), (0, 0), (0, pad_w)), mode="constant")
            padded.append(t)
        batch = np.stack(padded, axis=0)

        preds = sess.run([_output_name], {_input_name: batch})[0]
        out = ["" for _ in images]
        for out_idx, in_idx in enumerate(valid_idx):
            out[in_idx] = _first_char(_decode_ctc(preds[out_idx]))
        return out
    except Exception as e:
        logger.warning(f"[ml_char_ocr] recognize_chars batch failed: {e}")
        return ["" for _ in images]
