"""
ML Training Service
Handles feature extraction, augmentation, model training and prediction.
Algorithms: Random Forest, SVM, MLP.
"""
import base64
import io
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2 as cv
import joblib
import numpy as np

from app.models.ml_training import MLAnnotationInDB, TrainRequest
from app.services.ml_segment_service import crop_segment, segment_region

logger = logging.getLogger(__name__)

# ──────────────────────────────────────── SupCon embedding ──
#
# Pipeline: crop → ONNX SupCon (efficientnet_b2, 128-dim L2-normalized) → sklearn clf.
# Replaces the old handcrafted 1016-dim v2 pipeline + per-char goldens entirely.

EMBED_DIM = 128
EMBED_SIZE = 64                              # input image size for embedder (matches eval_svm.py)
_EMB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_EMB_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_BACKEND_DIR = Path(__file__).parent.parent.parent.parent
_SUPCON_PATH = _BACKEND_DIR / "weights/supcon_128_efficientnet_b2_20260429-073504"

_supcon_session = None  # singleton ONNX session (lazy-loaded on first call)


def _to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


def _get_supcon_session():
    """Lazy-init singleton ONNX session for SupCon embedder."""
    global _supcon_session
    if _supcon_session is None:
        import onnxruntime as ort
        onnx_path = _SUPCON_PATH / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"SupCon model not found at {onnx_path}")
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "CUDAExecutionProvider" in ort.get_available_providers()
                     else ["CPUExecutionProvider"])
        _supcon_session = ort.InferenceSession(str(onnx_path), providers=providers)
        logger.info(f"[SupCon] loaded {onnx_path.name} on {_supcon_session.get_providers()[0]}")
    return _supcon_session


def _preprocess_for_supcon(bgr: np.ndarray) -> np.ndarray:
    """Resize keep-aspect → pad-255 to 64×64 → ImageNet normalize → CHW float32.
    Accepts grayscale (2D) or BGR (3D) input — grayscale is widened to 3 channels.
    """
    if bgr.ndim == 2:
        bgr = cv.cvtColor(bgr, cv.COLOR_GRAY2BGR)
    elif bgr.ndim == 3 and bgr.shape[2] == 1:
        bgr = cv.cvtColor(bgr, cv.COLOR_GRAY2BGR)
    h, w = bgr.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((3, EMBED_SIZE, EMBED_SIZE), dtype=np.float32)
    s = EMBED_SIZE / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    img = cv.resize(bgr, (nw, nh), interpolation=cv.INTER_LINEAR)
    canvas = np.full((EMBED_SIZE, EMBED_SIZE, 3), 255, dtype=np.uint8)
    canvas[(EMBED_SIZE - nh) // 2:(EMBED_SIZE - nh) // 2 + nh,
           (EMBED_SIZE - nw) // 2:(EMBED_SIZE - nw) // 2 + nw] = img
    rgb = cv.cvtColor(canvas, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - _EMB_MEAN) / _EMB_STD).transpose(2, 0, 1).astype(np.float32)


def embed_crops(
    crops: List[np.ndarray],
    batch_size: int = 128,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> np.ndarray:
    """
    Batch-embed crops via SupCon ONNX. Returns (N, 128) L2-normalized.

    `progress_cb(frac)` — optional, called after each batch with frac in [0, 1]
    so callers can stream fine-grained progress while embedding (the dominant
    cost during training).
    """
    if not crops:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    sess = _get_supcon_session()
    out_chunks: List[np.ndarray] = []
    total_batches = max(1, (len(crops) + batch_size - 1) // batch_size)
    for batch_idx, i in enumerate(range(0, len(crops), batch_size)):
        chunk = crops[i:i + batch_size]
        x = np.stack([_preprocess_for_supcon(c) for c in chunk]).astype(np.float32)
        out = sess.run(None, {"input": x})[0]   # output[0] = 128-dim L2-norm sẵn
        norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
        out_chunks.append(out / norms)
        if progress_cb is not None:
            try:
                progress_cb((batch_idx + 1) / total_batches)
            except Exception:
                logger.exception("[embed_crops] progress_cb failed")
    return np.vstack(out_chunks)


# ──────────────────────────────────────── NG augmentation ──
# Realistic mask-aware defects (cut/segment/dropout/crack/block/blob/erosion/
# tape/stroke_thinning) — see ml_ng_augment.py.
from app.services.ml_ng_augment import (  # noqa: E402
    NG_AUG_TYPES, augment_ng,
)


# ──────────────────────────────────────── Image encoding ──

def img_to_b64(img: np.ndarray, quality: int = 85) -> str:
    """Encode BGR numpy array to base64 JPEG string."""
    ok, buf = cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ──────────────────────────────────────── Training ──

def build_dataset(
    annotations: List[MLAnnotationInDB],
    images_dir: Path,
    augment_factor: int = 0,
    severity_dist: Optional[Dict[str, float]] = None,
    ok_synth_target: int = 0,
    embed_progress_cb: Optional[Callable[[float], None]] = None,
) -> Tuple[
    np.ndarray, np.ndarray, List[np.ndarray], List[Optional[str]],
    Dict[str, Dict[str, int]], int, int,
]:
    """
    Build training dataset using SupCon embedding features.

    Groups crops by char_id for char-balanced NG augmentation, then embeds all
    crops via SupCon ONNX (128-dim L2-normalized).

    Returns (all lists / arrays share order):
        X: feature matrix (N, 128)
        y: labels (1=OK, 0=NG)
        crops_raw: raw crops parallel to X rows
        char_ids_raw: char_id strings parallel to X rows (None for _unknown)
        char_stats: {char_id: {n_ok_train, n_ng_train, ...}} — for display
        n_ok_total, n_ng_total: counts after augmentation
    """
    from collections import defaultdict

    # Group OK / NG crops by char_id (key='_unknown' if char_id missing).
    ok_by_char: Dict[str, List[np.ndarray]] = defaultdict(list)
    ng_by_char: Dict[str, List[np.ndarray]] = defaultdict(list)

    for ann in annotations:
        img_path = images_dir / ann.filename
        for region in ann.regions:
            for seg in region.segments:
                if seg.label not in ("OK", "NG"):
                    continue
                crop = crop_segment(img_path, {
                    "x": seg.x, "y": seg.y, "w": seg.w, "h": seg.h,
                })
                if crop is None:
                    continue
                key = seg.char_id or "_unknown"
                (ok_by_char if seg.label == "OK" else ng_by_char)[key].append(crop)

    if not ok_by_char and not ng_by_char:
        raise ValueError("No labeled segments found. Please label images before training.")

    n_ok_real = sum(len(v) for v in ok_by_char.values())
    n_ng_real = sum(len(v) for v in ng_by_char.values())

    # --- OK synthesis (font-render) — top up chars below target_n ---
    # Generates plausible OK samples for chars with too few real labels. Only
    # runs when target > 0; failures are logged but non-fatal.
    n_ok_synth = 0
    if ok_synth_target and ok_synth_target > 0:
        try:
            from app.services.ml_ok_synthesize import synthesize_ok_from_annotations
            synth = synthesize_ok_from_annotations(
                annotations, images_dir,
                target_n_per_char=ok_synth_target,
                only_below_threshold=True,
            )
            for item in synth:
                cid = item['char_id'] or "_unknown"
                ok_by_char[cid].append(item['crop'])
            n_ok_synth = len(synth)
            logger.info(f"[build_dataset] OK synthesis: +{n_ok_synth} crops "
                        f"(target {ok_synth_target}/char)")
        except Exception as e:
            logger.warning(f"[build_dataset] OK synthesis failed (continuing): {e}")

    # --- NG augmentation — char-balanced (Option A) ---
    # Target per char = factor * max(n_ok_real across chars). Each char tops up
    # with synthetic NG to reach this target. Ensures every char has the same
    # total NG count regardless of its original OK/NG distribution.
    #
    # OK samples are NOT augmented. Only real OK + NG (real + synthetic) go
    # into training. The classifier learns what real OK looks like; synthetic
    # NG supplies the negative class.
    aug_ng_by_char: Dict[str, List[np.ndarray]] = defaultdict(list)
    aug_ng_type_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {t: 0 for t in NG_AUG_TYPES}
    )

    max_ok_per_char = max((len(v) for v in ok_by_char.values()), default=0)
    target_ng_per_char = augment_factor * max_ok_per_char if augment_factor >= 2 else 0

    if target_ng_per_char > 0:
        all_chars_with_ok = [c for c in ok_by_char.keys() if c != "_unknown"]
        for char_id in all_chars_with_ok:
            ok_crops = ok_by_char[char_id]
            if not ok_crops:
                continue
            n_ng_real_this = len(ng_by_char.get(char_id, []))
            deficit = max(0, target_ng_per_char - n_ng_real_this)
            if deficit == 0:
                continue
            # Distribute deficit across OK crops — each OK template generates
            # base (+1 for first `extra` crops) augmentations.
            base = deficit // len(ok_crops)
            extra = deficit - base * len(ok_crops)
            for i, crop in enumerate(ok_crops):
                n_this = base + (1 if i < extra else 0)
                if n_this == 0:
                    continue
                # Pass char_id so augment_ng can apply per-char boost rules
                for aug_img, tag in augment_ng(crop, n=n_this,
                                                char_id=char_id if char_id != "_unknown" else None,
                                                severity_dist=severity_dist):
                    aug_ng_by_char[char_id].append(aug_img)
                    if tag in aug_ng_type_counts[char_id]:
                        aug_ng_type_counts[char_id][tag] += 1

    # --- Flatten + embed all crops via SupCon (single batched pass) ---
    crops_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    char_ids_rows: List[Optional[str]] = []

    def _append_samples(samples_by_char, label_val):
        for char_id, crops in samples_by_char.items():
            for c in crops:
                crops_rows.append(c)
                y_rows.append(label_val)
                char_ids_rows.append(None if char_id == "_unknown" else char_id)

    _append_samples(ok_by_char, 1)
    _append_samples(ng_by_char, 0)
    _append_samples(aug_ng_by_char, 0)

    logger.info(f"[build_dataset] embedding {len(crops_rows)} crops via SupCon...")
    X = embed_crops(crops_rows, progress_cb=embed_progress_cb)  # (N, 128) L2-normalized
    y = np.asarray(y_rows, dtype=np.int32)

    n_aug_ng = sum(len(v) for v in aug_ng_by_char.values())
    total_ok = n_ok_real                      # no OK augmentation
    total_ng = n_ng_real + n_aug_ng

    # Per-char training sample counts (for FE display)
    char_stats: Dict[str, Dict[str, Any]] = {}
    all_chars = set(ok_by_char.keys()) | set(ng_by_char.keys())
    for c in all_chars:
        if c == "_unknown":
            continue
        n_ng_aug_c = len(aug_ng_by_char.get(c, []))
        char_stats[c] = {
            "n_ok_train":  len(ok_by_char.get(c, [])),
            "n_ng_train":  len(ng_by_char.get(c, [])) + n_ng_aug_c,
            "n_ok_real":   len(ok_by_char.get(c, [])),
            "n_ng_real":   len(ng_by_char.get(c, [])),
            "n_ng_aug":    n_ng_aug_c,
            "n_ng_aug_by_type": dict(aug_ng_type_counts.get(c, {})),
        }

    logger.info(
        f"[build_dataset supcon] "
        f"chars: {sorted(all_chars)} | "
        f"OK: {n_ok_real} (no aug) | "
        f"NG: {n_ng_real}+{n_aug_ng}={total_ng} | "
        f"target/char={target_ng_per_char} | factor={augment_factor}"
    )

    # Shuffle — keep crops + char_ids in sync
    idx = np.random.permutation(len(X))
    crops_shuffled = [crops_rows[i] for i in idx]
    char_ids_shuffled = [char_ids_rows[i] for i in idx]
    return (
        X[idx], y[idx], crops_shuffled, char_ids_shuffled,
        char_stats, total_ok, total_ng,
    )


def _save_bundle_and_testset(
    model_save_path: Path,
    bundle: Dict[str, Any],
    test_set_items: List[Dict[str, Any]],
) -> None:
    """Persist bundle joblib + test-set sidecar JSON."""
    import json
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, str(model_save_path))
    test_set_path = model_save_path.parent / f"{model_save_path.stem}_test_set.json"
    test_set_path.write_text(json.dumps(test_set_items))


class CentroidClassifier:
    """
    Sklearn-compatible global centroid classifier.

    fit():     compute mean OK + mean NG embedding (L2-normalized).
    predict_proba(X):
        sim_ok = X @ c_ok ;  sim_ng = X @ c_ng
        raw    = sim_ok - sim_ng       ∈ [-2, 2]
        p_ok   = sigmoid(raw * temperature)   ∈ [0, 1]
        Returns (N, 2) → columns [p_ng, p_ok] for sklearn API parity.
    """

    def __init__(self, temperature: float = 5.0):
        self.temperature = float(temperature)
        self.centroid_ok: Optional[np.ndarray] = None
        self.centroid_ng: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CentroidClassifier":
        ok_mask = (y == 1)
        ng_mask = (y == 0)
        if ok_mask.sum() < 1 or ng_mask.sum() < 1:
            raise ValueError(
                f"Centroid classifier needs ≥1 OK and ≥1 NG sample; "
                f"got OK={int(ok_mask.sum())}, NG={int(ng_mask.sum())}"
            )
        c_ok = X[ok_mask].mean(axis=0)
        c_ng = X[ng_mask].mean(axis=0)
        self.centroid_ok = c_ok / (np.linalg.norm(c_ok) + 1e-8)
        self.centroid_ng = c_ng / (np.linalg.norm(c_ng) + 1e-8)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.centroid_ok is None or self.centroid_ng is None:
            raise RuntimeError("Centroid classifier not fitted")
        sim_ok = X @ self.centroid_ok
        sim_ng = X @ self.centroid_ng
        raw = sim_ok - sim_ng
        p_ok = 1.0 / (1.0 + np.exp(-np.clip(raw * self.temperature, -30, 30)))
        return np.stack([1.0 - p_ok, p_ok], axis=1)        # (N, 2): [p_ng, p_ok]


def train_model(
    annotations: List[MLAnnotationInDB],
    images_dir: Path,
    request: TrainRequest,
    model_save_path: Path,
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """
    Train a binary classifier (rf / svm / mlp / centroid) on SupCon embedding
    features and save to disk. Always saves a sidecar test-set JSON with
    per-crop predictions.

    `progress_cb(phase, progress_pct)` — optional callback fired at each phase
    boundary so the API layer can stream live progress to the FE.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    def _emit(phase: str, pct: float) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(phase, float(pct))
        except Exception:
            logger.exception("[train_model] progress_cb failed")

    algo = (request.algorithm or "rf").lower()

    _emit("preparing", 5)
    severity_dist = getattr(request, 'severity_dist', None)
    ok_synth_target = int(getattr(request, 'ok_synth_target', 0) or 0)

    # Embedding is the dominant cost — stream per-batch sub-progress mapped to
    # the 10..55% global range so the UI bar moves smoothly through this phase
    # instead of sitting at 5% until embedding finishes.
    def _embed_progress(frac: float) -> None:
        if progress_cb is not None:
            try:
                progress_cb("embedding", 10.0 + 45.0 * float(frac))
            except Exception:
                logger.exception("[train_model] embed progress_cb failed")

    X, y, crops_raw, char_ids_raw, char_stats, n_ok, n_ng = build_dataset(
        annotations, images_dir, request.augment_factor,
        severity_dist=severity_dist,
        ok_synth_target=ok_synth_target,
        embed_progress_cb=_embed_progress,
    )
    _emit("training_classifier", 60)

    if len(X) < 4:
        raise ValueError(f"Need at least 4 samples, got {len(X)}.")

    # Split — pass crops_raw + char_ids_raw alongside X/y so indices stay in sync
    test_size = min(request.test_split, 0.4)
    if len(np.unique(y)) > 1:
        (X_train, X_test, y_train, y_test,
         _, crops_test, _, char_ids_test) = train_test_split(
            X, y, crops_raw, char_ids_raw,
            test_size=test_size, random_state=42, stratify=y,
        )
    else:
        # Only one class — no meaningful split; use all crops for display
        X_train, X_test, y_train, y_test = X, X, y, y
        crops_test = crops_raw
        char_ids_test = char_ids_raw

    clf = _build_classifier(request)
    clf.fit(X_train, y_train)
    _emit("evaluating", 80)

    threshold = float(getattr(request, "threshold", 0.5))

    def _apply_threshold(X: np.ndarray) -> np.ndarray:
        proba = clf.predict_proba(X)
        p_ok = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        return (p_ok >= threshold).astype(np.int32)

    y_pred_train = _apply_threshold(X_train)
    y_pred_test  = _apply_threshold(X_test)
    acc_train = float(accuracy_score(y_train, y_pred_train))
    acc_test  = float(accuracy_score(y_test,  y_pred_test))

    cm     = confusion_matrix(y_test, y_pred_test).tolist()
    report = classification_report(y_test, y_pred_test,
                                   target_names=["NG", "OK"], zero_division=0)

    proba_test = clf.predict_proba(X_test)
    _emit("encoding_testset", 90)
    test_set_items = []
    for crop_img, char_id, true_y, pred_y, proba in zip(
        crops_test, char_ids_test, y_test, y_pred_test, proba_test,
    ):
        p_ok = float(proba[1]) if len(proba) > 1 else float(proba[0])
        test_set_items.append({
            "crop_b64":   img_to_b64(crop_img),
            "char_id":    char_id,
            "true_label": "OK" if int(true_y) == 1 else "NG",
            "pred_label": "OK" if int(pred_y) == 1 else "NG",
            "prob_ok":    round(p_ok, 4),
            "correct":    bool(true_y == pred_y),
        })

    # Serialize bundle as pure data for centroid (avoid cross-module pickle
    # dependency on CentroidClassifier when loading from ai_services).
    if algo == 'centroid':
        bundle = {
            'algorithm':    'centroid',
            'centroid_ok':  np.asarray(clf.centroid_ok, dtype=np.float32),
            'centroid_ng':  np.asarray(clf.centroid_ng, dtype=np.float32),
            'temperature':  float(clf.temperature),
            'char_stats':   char_stats,
        }
    else:
        bundle = {
            'algorithm':  algo,
            'clf':        clf,
            'char_stats': char_stats,
        }
    _emit("saving", 97)
    _save_bundle_and_testset(model_save_path, bundle, test_set_items)

    logger.info(
        f"[train_model:{algo}] saved bundle to {model_save_path.name}: "
        f"chars={sorted(char_stats.keys())}"
    )
    _emit("completed", 100)

    return {
        "accuracy_train": acc_train,
        "accuracy_test":  acc_test,
        "n_ok":           n_ok,
        "n_ng":           n_ng,
        "n_total":        len(X),
        "confusion_matrix": cm,
        "report":         report,
        "trained_chars":  sorted(char_stats.keys()),
    }


def _build_classifier(request: TrainRequest):
    algo = request.algorithm.lower()
    if algo == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=request.n_estimators,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",   # safety net for imbalanced datasets
        )
    elif algo == "svm":
        from sklearn.svm import SVC
        return SVC(
            C=request.C,
            kernel="rbf",
            probability=True,
            max_iter=request.max_iter,
            random_state=42,
            class_weight="balanced",
        )
    elif algo == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(
            hidden_layer_sizes=tuple(request.hidden_layer_sizes),
            max_iter=request.max_iter,
            random_state=42,
            early_stopping=True,
        )
    elif algo == "centroid":
        temperature = float(getattr(request, "centroid_temperature", 5.0) or 5.0)
        return CentroidClassifier(temperature=temperature)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


# ──────────────────────────────────────── Prediction ──

def _load_model_bundle(model_path: Path):
    """
    Load a model bundle from disk.
    Accepted shapes:
      - sklearn  : {'algorithm': rf|svm|mlp, 'clf': ..., 'char_stats': ...}
      - centroid : {'algorithm': 'centroid', 'centroid_ok', 'centroid_ng',
                    'temperature', 'char_stats'}
    Legacy formats (raw classifier, v2 with goldens) are no longer supported.
    """
    data = joblib.load(str(model_path))
    if not isinstance(data, dict):
        raise ValueError(f"Unsupported model bundle at {model_path}: expected dict")
    algo = (data.get('algorithm') or '').lower()
    if algo == 'centroid':
        if 'centroid_ok' not in data or 'centroid_ng' not in data:
            raise ValueError(f"Centroid bundle missing centroid_ok/centroid_ng")
        return data
    if 'clf' in data:
        return data
    raise ValueError(
        f"Unsupported model bundle at {model_path}. "
        "Legacy formats (v1/v2 with goldens) are no longer supported — please retrain."
    )


def _centroid_predict_proba(X: np.ndarray, bundle: Dict[str, Any]) -> np.ndarray:
    """Apply centroid scoring → return p_ok array (N,) ∈ [0,1]."""
    c_ok = np.asarray(bundle['centroid_ok'])
    c_ng = np.asarray(bundle['centroid_ng'])
    T = float(bundle.get('temperature', 5.0))
    sim_ok = X @ c_ok
    sim_ng = X @ c_ng
    raw = sim_ok - sim_ng
    return 1.0 / (1.0 + np.exp(-np.clip(raw * T, -30, 30)))


def predict_on_image(
    model_path: Path,
    image_path: Path,
    region: Optional[Dict] = None,
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Segment an image (or a region of it) and predict OK/NG per character via
    SupCon embedding → sklearn classifier.

    Args:
        model_path: Path to saved joblib bundle.
        image_path: Image to predict on.
        region: Optional {x, y, w, h} normalized — segment only this area.
        threshold: Probability threshold for OK class.

    Returns:
        List of dicts — see LabeledCrop/PredictResult schema.
    """
    bundle = _load_model_bundle(model_path)
    algorithm = bundle.get('algorithm', 'rf').lower()

    if region:
        segments = segment_region(image_path, region)
    else:
        segments = segment_region(image_path, {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0})

    crops: List[np.ndarray] = []
    valid_segs: List[Dict] = []
    for seg in segments:
        crop = crop_segment(image_path, seg)
        if crop is None:
            continue
        crops.append(crop)
        valid_segs.append(seg)

    if not crops:
        return []

    X = embed_crops(crops)                        # (N, 128)
    if algorithm == 'centroid':
        p_ok_arr = _centroid_predict_proba(X, bundle)
    else:
        clf = bundle.get('clf')
        proba = clf.predict_proba(X)
        p_ok_arr = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

    results = []
    for crop, seg, p in zip(crops, valid_segs, p_ok_arr):
        p_ok = float(p)
        results.append({
            "id":       seg["id"],
            "x":        seg["x"],
            "y":        seg["y"],
            "w":        seg["w"],
            "h":        seg["h"],
            "prob_ok":  round(p_ok, 4),
            "label":    "OK" if p_ok >= threshold else "NG",
            "crop_b64": img_to_b64(crop),
            "algorithm": algorithm,
        })
    return results


def get_model_chars(model_path: Path) -> List[str]:
    """Return list of char_ids the model was trained on (from char_stats)."""
    try:
        bundle = _load_model_bundle(model_path)
        char_stats = bundle.get('char_stats') or {}
        return sorted(char_stats.keys())
    except Exception as e:
        logger.warning(f"[get_model_chars] Failed to load {model_path}: {e}")
        return []


def get_labeled_crops(
    annotations: List[MLAnnotationInDB],
    images_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Collect all labeled character crops for the Train tab preview grid.
    Returns list of {segment_id, region_id, filename, label, crop_b64}.
    """
    result = []
    for ann in annotations:
        img_path = images_dir / ann.filename
        for region in ann.regions:
            for seg in region.segments:
                if seg.label not in ("OK", "NG"):
                    continue
                crop = crop_segment(img_path, {
                    "x": seg.x, "y": seg.y, "w": seg.w, "h": seg.h,
                })
                if crop is None:
                    continue
                result.append({
                    "segment_id": seg.id,
                    "region_id": region.id,
                    "filename": ann.filename,
                    "label": seg.label,
                    "crop_b64": img_to_b64(crop),
                    "char_id": seg.char_id,
                })
    return result


def generate_synthetic_crops(
    annotations: List[MLAnnotationInDB],
    images_dir: Path,
    augment_factor: int,
    label: str = "NG",
    severity_dist: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate synthetic NG crops from OK samples for preview.

    Only NG augmentation is supported now (OK augmentation removed).
    Accepted labels: 'NG' or 'BOTH' → generate NG preview. 'OK' → empty.

    Args:
        annotations: project annotations.
        images_dir: project images directory.
        augment_factor: preview uses (augment_factor - 1) augments per OK sample.
        label: kept for API compatibility; anything except 'NG'/'BOTH' returns [].

    Returns list of {source_segment_id, filename, label, crop_b64, char_id, aug_type}.
    """
    if augment_factor < 2:
        return []
    n_per_sample = augment_factor - 1
    label = (label or "NG").upper()
    if label not in ("NG", "BOTH"):
        return []

    result = []
    for ann in annotations:
        img_path = images_dir / ann.filename
        for region in ann.regions:
            for seg in region.segments:
                if seg.label != "OK":
                    continue
                crop = crop_segment(img_path, {
                    "x": seg.x, "y": seg.y, "w": seg.w, "h": seg.h,
                })
                if crop is None:
                    continue
                for aug_img, aug_tag in augment_ng(crop, n=n_per_sample,
                                                    char_id=seg.char_id,
                                                    severity_dist=severity_dist):
                    result.append({
                        "source_segment_id": seg.id,
                        "filename": ann.filename,
                        "label": "NG",
                        "crop_b64": img_to_b64(aug_img),
                        "char_id": seg.char_id,
                        "aug_type": aug_tag,
                    })
    return result
