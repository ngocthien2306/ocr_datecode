"""
AnomalyInference — runs a PatchCore/Padim ONNX model (trained + exported by
anomaly_service, see docs/anomaly_training_plan.md) against the same
`label` crop WrinkledSegmenterTRT uses today, and produces a same-shaped
result dict so it can drop into product_verifier.py as a swap-in
replacement — no changes needed to the pass/fail aggregation or
visualization code downstream.

Loader mirrors backend/app/services/ml_training_service.py's
`_get_supcon_session` (onnxruntime, TensorRT → CUDA → CPU providers, cached
TRT engine on disk). Cropping reuses WrinkledSegmenterTRT.crop_from_obb
(a @staticmethod, no model instance required) so training-time crops
(anomaly_service) and inference-time crops (here) match exactly.

Preprocessing note: anomalib's exported graph traces its own PreProcessor
(Resize + ImageNet Normalize) as part of the model, verified against
Patchcore.configure_pre_processor in anomalib==2.5.0 — so this file must
NOT apply ImageNet normalization itself, only resize to the export
image_size, BGR→RGB, and scale to [0,1].
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Sessions keyed by onnx_path. A recipe change that swaps the anomaly model
# requires restarting ai_services to take effect — same as a newly trained
# char-classifier model today (see backend's _trigger_restart_all) — so in
# practice this holds at most one entry per process lifetime. Keying by path
# just makes that explicit rather than assuming a single global model.
_sessions: Dict[str, Any] = {}

_TRT_CACHE_ROOT = Path(os.environ.get("ANOMALY_TRT_CACHE_DIR", "/tmp/anomaly_trt_cache"))


def _get_session(onnx_path: str):
    sess = _sessions.get(onnx_path)
    if sess is not None:
        return sess

    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.enable_mem_pattern = False
    sess_options.enable_cpu_mem_arena = False
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.log_severity_level = 3

    available = set(ort.get_available_providers())
    providers: list = []
    if "TensorrtExecutionProvider" in available:
        cache_dir = _TRT_CACHE_ROOT / Path(onnx_path).stem
        cache_dir.mkdir(parents=True, exist_ok=True)
        providers.append((
            "TensorrtExecutionProvider",
            {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache_dir),
                "trt_fp16_enable": True,
                "trt_max_workspace_size": 512 * 1024 * 1024,
            },
        ))
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    sess = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
    active = sess.get_providers()[0]
    logger.info(f"[AnomalyInference] loaded {Path(onnx_path).name} on {active}")
    _sessions[onnx_path] = sess
    return sess


def crop_label(frame: np.ndarray, cx: float, cy: float, w: float, h: float, angle: float) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Same OBB crop WrinkledSegmenterTRT.predict_batch's caller uses —
    calling the shared @staticmethod directly so the two checks are always
    pixel-identical on the same product_box, whichever one is active."""
    from .wrinkle_segmenter import WrinkledSegmenterTRT
    return WrinkledSegmenterTRT.crop_from_obb(frame, cx, cy, w, h, angle)


def _preprocess(crop_bgr: np.ndarray, image_size: int) -> np.ndarray:
    resized = cv2.resize(crop_bgr, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...])  # (1, 3, H, W)


def predict(onnx_path: str, crop_bgr: Optional[np.ndarray], image_size: int = 256) -> Dict[str, Any]:
    """Run one anomaly-detection inference on a label crop.

    Never raises — a broken model must not take down the whole inspection
    pipeline. On any failure returns skipped=True with the error recorded,
    so the caller can decide whether to fail-open (skip, like a disabled
    check) or fail-closed (treat as abnormal); build_anomaly_check()
    fails open, matching how WrinkledSegmenterTRT.build_wrinkled_check
    treats a missing/failed detection today.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return {"skipped": True, "reason": "empty crop", "pred_score": 0.0}
    try:
        sess = _get_session(onnx_path)
        x = _preprocess(crop_bgr, image_size)
        outputs = sess.run(None, {sess.get_inputs()[0].name: x})
        out_names = [o.name for o in sess.get_outputs()]
        score = float(np.asarray(outputs[out_names.index("pred_score")]).reshape(-1)[0])
        return {"skipped": False, "pred_score": score}
    except Exception as e:
        logger.exception(f"[AnomalyInference] inference failed for {onnx_path}")
        return {"skipped": True, "reason": f"inference error: {e}", "pred_score": 0.0, "error": str(e)}


def build_anomaly_check(pred: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    """Convert predict()'s raw result into the same shape
    WrinkledSegmenterTRT.build_wrinkled_check() returns (ok / has_wrinkled /
    wrinkled_count / wrinkled_boxes / min_area) so this drops straight into
    product_verifier.py's existing aggregation + visualization code as a
    swap-in replacement — no downstream changes needed.
    """
    if pred.get("skipped"):
        return {
            "ok": True, "skipped": True, "reason": pred.get("reason", "skipped"),
            "has_wrinkled": False, "wrinkled_count": 0, "wrinkled_boxes": [],
            "anomaly_score": pred.get("pred_score", 0.0), "anomaly_threshold": threshold,
        }
    score = pred["pred_score"]
    is_abnormal = score >= threshold
    return {
        "ok": not is_abnormal,
        "skipped": False,
        "has_wrinkled": is_abnormal,   # reused key name — see module docstring
        "wrinkled_count": 1 if is_abnormal else 0,
        "wrinkled_boxes": [],          # image-level model — no per-region boxes to draw
        "anomaly_score": score,
        "anomaly_threshold": threshold,
    }


def release_session(onnx_path: str) -> bool:
    """Drop a cached session (e.g. on graceful shutdown). Returns whether
    one was actually held."""
    return _sessions.pop(onnx_path, None) is not None
