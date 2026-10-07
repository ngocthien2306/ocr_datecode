"""
CPU-only ONNX Runtime models for live template realignment.

Two models, both loaded lazily as process-wide singletons:

  * OnnxCpuYOLOOBB      — YOLO OBB (bottle_cap + text_box). Drop-in for
                           ai_services/yolo_obb_tensorrt.YOLOOBBTensorRT:
                           same constructor keywords, same
                           `predict(images, conf_threshold, return_timing)`
                           return shape, same letterbox preprocessing — so the
                           copied OBBRotationService body runs unchanged.
  * SuperPointLightGlue — SuperPoint + LightGlue pipeline (template, target)
                           → (keypoints, matches, mscores). Same I/O as the
                           `sp_lg_pipeline_fp16_dynamic_480_640.engine` the
                           AI service runs; resolution fixed at 480x640 to
                           match it.

WHY CPU, AND WHY NOTHING HERE MAY EVER TOUCH CUDA
-------------------------------------------------
This module runs inside the BACKEND (uvicorn) process. The backend must not
hold a CUDA context — docs/DEBUG_SERVICE_CRASH_RUNBOOK.md §5 / §6 rule 2: a
CUDA runtime in a process with a dynamic threadpool is what SIGABRT-ed the
backend on 2026-08-05 (PyCUDA TLS destructor on a retiring anyio thread), and
the rule is explicitly extended to onnxruntime's CUDAExecutionProvider. The
`vision` conda env the backend runs in ships onnxruntime-gpu, so the provider
list below is the ONLY thing standing between this code and a CUDA context.
Keep it `["CPUExecutionProvider"]`. Do not add a GPU branch "for speed".

Measured on this machine (CPU, 4 intra-op threads):
    OBB 320x320 ............................ ~38 ms / image
    SuperPoint+LightGlue, one pair 480x640 .. ~480-510 ms
    session load (SP+LG) .................... ~175 ms, once
which is fine for an operator clicking a button.

Nothing in this package may import `ai_services.*`, `camera_management.*`,
`inference_engine_shared`, `yolo_obb_tensorrt`, `tensorrt` or `pycuda`.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

# backend/app/services/live_template/ → parents[4] == repo root
_REPO_ROOT = Path(__file__).resolve().parents[4]
WEIGHTS_DIR = _REPO_ROOT / "weights"

OBB_MODEL_PATH = WEIGHTS_DIR / "best_bottle_m.onnx"
SP_LG_MODEL_PATH = WEIGHTS_DIR / "superpoint_lightglue_pipeline.onnx"

# The one provider this module is allowed to use. See module docstring.
_PROVIDERS = ["CPUExecutionProvider"]
# Bounded so a realign click can't starve the camera producer / other
# services of CPU. 4 was the measured sweet spot above.
_INTRA_OP_THREADS = 4

# SuperPoint engine input the AI service uses: (H, W) = (480, 640).
SP_LG_INPUT_HW: Tuple[int, int] = (480, 640)


def _make_session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = _INTRA_OP_THREADS
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), sess_options=so, providers=_PROVIDERS)
    active = sess.get_providers()
    # Belt and braces: fail loudly rather than silently run on a GPU provider.
    if any(p != "CPUExecutionProvider" for p in active):
        raise RuntimeError(f"live_template: refusing non-CPU providers {active} for {path.name}")
    return sess


# ─── YOLO OBB ────────────────────────────────────────────────────────────────

class OnnxCpuYOLOOBB:
    """CPU twin of ai_services/yolo_obb_tensorrt.YOLOOBBTensorRT.

    Pre/post-processing mirrors that file line for line (letterbox to
    img_size with 114 padding, BGR→RGB, /255; built-in-NMS output
    [cx, cy, w, h, conf, cls, angle]) so detections — and therefore rotation
    angles and cap circles — agree with the AI service's.
    """

    def __init__(self, engine_path: str = str(OBB_MODEL_PATH),
                 class_names: Optional[List[str]] = None, img_size: int = 320):
        # `engine_path` keeps the TRT class's keyword so OBBRotationService's
        # constructor call is untouched; here it is an .onnx path. A .engine
        # path (what OBBRotationService's callers pass in ai_services) is
        # mapped to the ONNX sibling rather than ever being opened.
        p = Path(engine_path)
        if p.suffix != ".onnx":
            p = OBB_MODEL_PATH
        self.session = _get_cached_session(p)
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = class_names or []
        self.img_size = int(img_size)

    def preprocess(self, images: List[np.ndarray]):
        batch, scales, pads = [], [], []
        for image in images:
            h, w = image.shape[:2]
            scale = min(self.img_size / w, self.img_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(image, (new_w, new_h))
            padded = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
            pad_top = (self.img_size - new_h) // 2
            pad_left = (self.img_size - new_w) // 2
            padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            batch.append(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
            scales.append(scale)
            pads.append((pad_top, pad_left))
        return np.array(batch, dtype=np.float32), scales, pads

    @staticmethod
    def postprocess(output: np.ndarray, conf_threshold: float = 0.7):
        boxes = np.concatenate([output[:, :4], output[:, 6:7]], axis=1)
        scores = output[:, 4]
        class_ids = output[:, 5].astype(int)
        mask = (scores >= conf_threshold) & (scores > 0)
        return boxes[mask], scores[mask], class_ids[mask]

    def predict(self, images: List[np.ndarray], conf_threshold: float = 0.7,
                return_timing: bool = False):
        if not isinstance(images, list):
            images = [images]
        t0 = time.perf_counter()
        batch, scales, pads = self.preprocess(images)
        out = self.session.run(None, {self.input_name: batch})[0]
        results = []
        for i in range(len(images)):
            boxes, scores, class_ids = self.postprocess(out[i], conf_threshold)
            if len(boxes) > 0:
                pad_top, pad_left = pads[i]
                scale = scales[i]
                boxes = boxes.copy()
                boxes[:, 0] = (boxes[:, 0] - pad_left) / scale
                boxes[:, 1] = (boxes[:, 1] - pad_top) / scale
                boxes[:, 2] = boxes[:, 2] / scale
                boxes[:, 3] = boxes[:, 3] / scale
            results.append((boxes, scores, class_ids))
        if return_timing:
            return results, {"total": (time.perf_counter() - t0) * 1000.0}
        return results


# ─── SuperPoint + LightGlue ──────────────────────────────────────────────────

class SuperPointLightGlue:
    """Runs the (template, target) pair through the SP+LG ONNX pipeline.

    Mirrors the preprocessing of SuperPointMatcherTRTOptimized.match_batch:
    grayscale, plain cv2.resize to the engine size (no letterbox), /255,
    stacked as [template, target] in one batch.
    """

    def __init__(self, model_path: Path = SP_LG_MODEL_PATH,
                 input_hw: Tuple[int, int] = SP_LG_INPUT_HW):
        self.session = _get_cached_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        # Mimic the TRT matcher's `input_shape` attribute: (B, C, H, W).
        self.input_shape = (2, 1, int(input_hw[0]), int(input_hw[1]))

    def _resize_to_engine_size(self, img: np.ndarray) -> np.ndarray:
        target_h, target_w = self.input_shape[2:]
        h, w = img.shape[:2]
        if h != target_h or w != target_w:
            return cv2.resize(img, (target_w, target_h))
        return img

    def run_pair(self, template_gray: np.ndarray, target_gray: np.ndarray):
        t = self._resize_to_engine_size(template_gray).astype(np.float32)[None, None] / 255.0
        g = self._resize_to_engine_size(target_gray).astype(np.float32)[None, None] / 255.0
        batch = np.concatenate([t, g], axis=0)
        t0 = time.perf_counter()
        kpts, matches, mscores = self.session.run(None, {self.input_name: batch})
        return kpts, matches, mscores, (time.perf_counter() - t0) * 1000.0


# ─── Session cache ───────────────────────────────────────────────────────────

_sessions: Dict[str, ort.InferenceSession] = {}
_sessions_lock = threading.Lock()


def _get_cached_session(path: Path) -> ort.InferenceSession:
    key = str(path)
    sess = _sessions.get(key)
    if sess is not None:
        return sess
    with _sessions_lock:
        sess = _sessions.get(key)
        if sess is None:
            if not path.exists():
                raise FileNotFoundError(f"live_template model not found: {path}")
            t0 = time.perf_counter()
            sess = _make_session(path)
            _sessions[key] = sess
            logger.info(f"[live_template] loaded {path.name} on CPU in "
                        f"{(time.perf_counter() - t0) * 1000:.0f} ms")
    return sess


_sp_lg: Optional[SuperPointLightGlue] = None
_sp_lg_lock = threading.Lock()


def get_sp_lg() -> SuperPointLightGlue:
    global _sp_lg
    if _sp_lg is None:
        with _sp_lg_lock:
            if _sp_lg is None:
                _sp_lg = SuperPointLightGlue()
    return _sp_lg
