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
from typing import Any, Dict, List, Optional, Tuple

import cv2 as cv
import joblib
import numpy as np

from app.models.ml_training import MLAnnotationInDB, TrainRequest
from app.services.ml_segment_service import crop_segment, segment_region

logger = logging.getLogger(__name__)

FEAT_SIZE = (48, 48)   # resolution to catch smaller defects
GRID = 6               # 6×6 grid → 36 cells of 8×8 each
CELL_SIZE = FEAT_SIZE[0] // GRID
FEAT_DIM = 576 + 144 + 32 + 96 + 8   # 856


# ──────────────────────────────────────── Feature extraction ──

def _to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


def extract_features(char_img: np.ndarray) -> np.ndarray:
    """
    Extract 856-dim feature vector mixing global + LOCAL signals.

    Layout:
      - 24×24 downsampled pixels (576)        — global shape
      - 6×6 grid × 4 stats/cell (144)         — LOCAL defect signal
           mean, std, edge density, min-pixel
      - Sobel Gx/Gy histograms (32)           — stroke orientation
      - H/V projection profiles (96)          — row/column sums
      - 2×2 quadrant Sobel mag stats (8)      — coarse gradient energy

    Grid cell stats are the key upgrade: a tiny local defect (e.g. 6-12px
    ink blot / cut) concentrates in 1-2 cells → their mean/std/edge/min
    diverge sharply from neighbours, giving RF/MLP a strong local signal
    that the original global-only features washed out.
    """
    gray = _to_gray(char_img)
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return np.zeros(FEAT_DIM, dtype=np.float32)

    # Resize preserving aspect, pad to FEAT_SIZE
    scale = min(FEAT_SIZE[0] / w, FEAT_SIZE[1] / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    resized = cv.resize(gray, (nw, nh), interpolation=cv.INTER_AREA)
    canvas = np.zeros(FEAT_SIZE[::-1], dtype=np.uint8)
    yo = (FEAT_SIZE[1] - nh) // 2
    xo = (FEAT_SIZE[0] - nw) // 2
    canvas[yo:yo + nh, xo:xo + nw] = resized

    # --- 1) Global downsampled pixels (24×24 = 576) ---
    small = cv.resize(canvas, (24, 24), interpolation=cv.INTER_AREA)
    pixels = small.astype(np.float32).flatten() / 255.0

    # --- 2) 6×6 grid local stats (36 cells × 4 = 144) ---
    edges = cv.Canny(canvas, 50, 150)
    cell_stats: List[float] = []
    for i in range(GRID):
        for j in range(GRID):
            y0, y1 = i * CELL_SIZE, (i + 1) * CELL_SIZE
            x0, x1 = j * CELL_SIZE, (j + 1) * CELL_SIZE
            cell = canvas[y0:y1, x0:x1]
            edge_cell = edges[y0:y1, x0:x1]
            cell_stats.append(float(cell.mean()) / 255.0)
            cell_stats.append(float(cell.std()) / 128.0)
            cell_stats.append(float(np.count_nonzero(edge_cell)) / (CELL_SIZE * CELL_SIZE))
            cell_stats.append(float(cell.min()) / 255.0)
    cell_arr = np.asarray(cell_stats, dtype=np.float32)

    # --- 3) Sobel Gx/Gy histograms (16 + 16 = 32) ---
    gx = cv.Sobel(canvas, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(canvas, cv.CV_32F, 0, 1, ksize=3)
    hist_gx = np.histogram(gx, bins=16, range=(-255, 255))[0].astype(np.float32)
    hist_gy = np.histogram(gy, bins=16, range=(-255, 255))[0].astype(np.float32)
    hist_gx /= hist_gx.sum() + 1e-6
    hist_gy /= hist_gy.sum() + 1e-6

    # --- 4) H/V projection profiles (48 + 48 = 96) ---
    h_proj = canvas.astype(np.float32).sum(axis=1) / (FEAT_SIZE[0] * 255.0 + 1e-6)
    v_proj = canvas.astype(np.float32).sum(axis=0) / (FEAT_SIZE[1] * 255.0 + 1e-6)

    # --- 5) Sobel magnitude stats per 2×2 quadrant (4 × 2 = 8) ---
    sob = np.hypot(gx, gy)
    qsize = FEAT_SIZE[0] // 2
    quad_stats: List[float] = []
    for qi in range(2):
        for qj in range(2):
            q = sob[qi * qsize:(qi + 1) * qsize, qj * qsize:(qj + 1) * qsize]
            quad_stats.append(float(q.mean()) / 255.0)
            quad_stats.append(float(q.std()) / 255.0)
    quad_arr = np.asarray(quad_stats, dtype=np.float32)

    return np.concatenate([pixels, cell_arr, hist_gx, hist_gy, h_proj, v_proj, quad_arr])


# ──────────────────────────────────────── Golden template (v2) ──
#
# v2 pipeline: per-char golden reference + alignment + diff features.
# Requires char_id on each segment (either manually labeled or imported
# from recipe's expected_text).
#
# extract_features_v2 layout:
#   base_v1 (856) = extract_features(aligned_input)
#   diff features (160):
#     - 6×6 grid × 4 stats on diff_map (144): mean, max, std, count>thr
#     - 4×4 region × 1 stat (16): max
#   total 1016

FEAT_DIM_V2 = FEAT_DIM + 160   # 856 + 160 = 1016
GOLDEN_MIN_OK_SAMPLES = 5      # skip char's golden if fewer than this many OK


def preprocess_canonical(img: np.ndarray) -> np.ndarray:
    """
    Gray → CLAHE (contrast normalize) → resize preserve aspect → center-pad 48×48.
    Used by compute_golden and extract_features_v2 for consistent canonical form.
    """
    gray = _to_gray(img)
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return np.zeros(FEAT_SIZE[::-1], dtype=np.uint8)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normed = clahe.apply(gray)

    scale = min(FEAT_SIZE[0] / w, FEAT_SIZE[1] / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    resized = cv.resize(normed, (nw, nh), interpolation=cv.INTER_AREA)
    canvas = np.zeros(FEAT_SIZE[::-1], dtype=np.uint8)
    yo = (FEAT_SIZE[1] - nh) // 2
    xo = (FEAT_SIZE[0] - nw) // 2
    canvas[yo:yo + nh, xo:xo + nw] = resized
    return canvas


def compute_golden(ok_crops: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Average OK samples (after canonical preprocessing) → 48×48 uint8 reference.

    Returns None if fewer than GOLDEN_MIN_OK_SAMPLES samples provided —
    caller should fall back to non-golden (v1 features only) for this char.
    """
    if len(ok_crops) < GOLDEN_MIN_OK_SAMPLES:
        return None
    canonical = [preprocess_canonical(c).astype(np.float32) for c in ok_crops]
    return np.mean(canonical, axis=0).astype(np.uint8)


def align_to_golden(
    input_48: np.ndarray, golden_48: np.ndarray, search: int = 5,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Find best ±search-px offset that aligns input to golden; apply the shift.
    Uses cv2.matchTemplate for fast sub-pixel-accurate search.
    """
    padded = cv.copyMakeBorder(
        input_48, search, search, search, search, cv.BORDER_REPLICATE,
    )
    result = cv.matchTemplate(padded, golden_48, cv.TM_CCOEFF_NORMED)
    _, _, _, (mx, my) = cv.minMaxLoc(result)
    dx, dy = mx - search, my - search
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned = cv.warpAffine(
        input_48, M, FEAT_SIZE,
        flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE,
    )
    return aligned, (int(dx), int(dy))


def extract_diff_features(diff_48: np.ndarray) -> np.ndarray:
    """
    160-dim features from |input - golden| diff map.
      - 6×6 grid × 4 stats (144): mean, max, std, count(>30)
      - 4×4 region × 1 stat (16): max
    """
    feats: List[float] = []

    # 6×6 grid of 8×8 cells
    for i in range(GRID):
        for j in range(GRID):
            y0, y1 = i * CELL_SIZE, (i + 1) * CELL_SIZE
            x0, x1 = j * CELL_SIZE, (j + 1) * CELL_SIZE
            cell = diff_48[y0:y1, x0:x1]
            feats.append(float(cell.mean()) / 255.0)
            feats.append(float(cell.max()) / 255.0)
            feats.append(float(cell.std()) / 128.0)
            feats.append(float(np.count_nonzero(cell > 30)) / (CELL_SIZE * CELL_SIZE))

    # 4×4 coarse grid of 12×12 regions — captures broader local anomalies
    region_size = FEAT_SIZE[0] // 4
    for i in range(4):
        for j in range(4):
            y0, y1 = i * region_size, (i + 1) * region_size
            x0, x1 = j * region_size, (j + 1) * region_size
            region = diff_48[y0:y1, x0:x1]
            feats.append(float(region.max()) / 255.0)

    return np.asarray(feats, dtype=np.float32)


def _compute_diff_map(img: np.ndarray, char_id: Optional[str],
                     goldens: Optional[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Shared helper: canonical preprocess + align + diff.
    Returns (aligned_48, diff_48_or_None). diff is None when no golden available.
    """
    canvas = preprocess_canonical(img)
    if char_id and goldens and char_id in goldens:
        golden = goldens[char_id]
        aligned, _ = align_to_golden(canvas, golden)
        diff = np.abs(aligned.astype(np.int16) - golden.astype(np.int16)).astype(np.uint8)
        return aligned, diff
    return canvas, None


def extract_features_v2(
    char_img: np.ndarray,
    char_id: Optional[str] = None,
    goldens: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """
    1016-dim feature vector = 856 base + 160 diff.
    When char_id missing or no golden exists → diff features = zeros (model
    learns to discount them).
    """
    aligned, diff = _compute_diff_map(char_img, char_id, goldens)
    base = extract_features(aligned)
    if diff is None:
        diff_feats = np.zeros(160, dtype=np.float32)
    else:
        diff_feats = extract_diff_features(diff)
    return np.concatenate([base, diff_feats])


# ──────────────────────────────────────── Augmentation helpers ──

def _estimate_bg_color(img: np.ndarray):
    """Ước lượng màu background (chữ tối → bg là vùng sáng, lấy percentile 75)."""
    if img.ndim == 3:
        return np.percentile(img.reshape(-1, img.shape[2]), 75, axis=0)
    return np.percentile(img, 75)


def _estimate_fg_color(img: np.ndarray):
    """Ước lượng màu foreground (chữ) — vùng tối, percentile 25."""
    if img.ndim == 3:
        return np.percentile(img.reshape(-1, img.shape[2]), 25, axis=0)
    return np.percentile(img, 25)


# ──────────────────────────────────────── Augmentation (synthetic OK) ──

def augment_ok(char_img: np.ndarray, n: int = 5) -> List[np.ndarray]:
    """
    Generate n mildly-augmented OK samples.

    Biên độ NHỎ — giữ nguyên semantic OK. Mô phỏng variation thực tế:
    rotation nhẹ, dịch vị trí nhỏ, ánh sáng, noise sensor, focus drift nhẹ.
    """
    gray = _to_gray(char_img)
    h, w = gray.shape[:2]
    results: List[np.ndarray] = []
    num_aug_types = 5
    choices = np.random.choice(num_aug_types, size=n, replace=n > num_aug_types)

    for choice in choices:
        aug = gray.copy()
        if choice == 0:
            # Rotation ±5° — replicate border để không tạo viền đen giả
            angle = float(np.random.uniform(-5, 5))
            M = cv.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            aug = cv.warpAffine(
                aug, M, (w, h),
                flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE,
            )
        elif choice == 1:
            # Translation ±3px
            dx = int(np.random.randint(-3, 4))
            dy = int(np.random.randint(-3, 4))
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            aug = cv.warpAffine(
                aug, M, (w, h),
                flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE,
            )
        elif choice == 2:
            # Brightness/contrast jitter ±15%
            alpha = float(np.random.uniform(0.85, 1.15))
            beta = int(np.random.randint(-15, 15))
            aug = np.clip(aug.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        elif choice == 3:
            # Mild gaussian noise (σ=5-8) — sensor noise
            sigma = float(np.random.uniform(5, 8))
            noise = np.random.normal(0, sigma, aug.shape).astype(np.int16)
            aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif choice == 4:
            # Slight blur k=3/5 — focus drift nhẹ
            k = int(np.random.choice([3, 5]))
            aug = cv.GaussianBlur(aug, (k, k), 0)
        results.append(aug)
    return results


# ──────────────────────────────────────── Augmentation (synthetic NG) ──

# Augmentation type tags — emitted alongside each crop so char_stats can
# break down "what kind of synthetic NG" was generated per char.
NG_AUG_TYPES: Tuple[str, ...] = ("noise", "cut", "erode", "dilate", "translate")


def _augment_ng_one(gray: np.ndarray) -> Tuple[np.ndarray, str]:
    """Apply one random NG transform. Returns (crop, type_tag)."""
    h, w = gray.shape[:2]
    choice = int(np.random.randint(0, len(NG_AUG_TYPES)))
    aug = gray.copy()

    if choice == 0:  # noise
        noise = np.random.normal(0, 30, aug.shape).astype(np.int16)
        aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return aug, "noise"

    if choice == 1:  # random black-rect cuts
        num_cuts = int(np.random.randint(1, 4))
        for _ in range(num_cuts):
            rh = int(np.random.randint(max(2, h // 6), max(3, h // 3)))
            rw = int(np.random.randint(max(2, w // 6), max(3, w // 3)))
            ry = int(np.random.randint(0, max(1, h - rh)))
            rx = int(np.random.randint(0, max(1, w - rw)))
            aug[ry:ry + rh, rx:rx + rw] = 0
        return aug, "cut"

    if choice == 2:  # erode (stroke thinning)
        k = int(np.random.randint(4, 7))
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
        return cv.erode(aug, kernel, iterations=1), "erode"

    if choice == 3:  # dilate (stroke bleeding)
        k = int(np.random.randint(5, 8))
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
        return cv.dilate(aug, kernel, iterations=1), "dilate"

    # choice == 4 — translate (misalignment)
    dx = int(np.random.randint(-w // 3, w // 3 + 1))
    dy = int(np.random.randint(-h // 3, h // 3 + 1))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv.warpAffine(aug, M, (w, h), borderValue=0), "translate"


def augment_ng(char_img: np.ndarray, n: int = 5) -> List[Tuple[np.ndarray, str]]:
    """
    Generate n synthetic NG crops from an OK character.

    5 transform types (uniform random choice):
      noise · cut · erode · dilate · translate

    Returns list of (crop, type_tag) so callers can log what was generated.
    """
    gray = _to_gray(char_img)
    return [_augment_ng_one(gray) for _ in range(n)]


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
) -> Tuple[
    np.ndarray, np.ndarray, List[np.ndarray], List[Optional[str]],
    Dict[str, np.ndarray], Dict[str, Dict[str, int]], int, int,
]:
    """
    Build training dataset + per-char goldens.

    v2: groups OK samples by char_id → per-char golden templates.
    Char without char_id (or with <5 OK samples) → no golden, diff features zero.

    Returns (all lists / arrays share order):
        X: feature matrix (N, 1016)
        y: labels (1=OK, 0=NG)
        crops_raw: raw crops parallel to X rows
        char_ids_raw: char_id strings parallel to X rows (None for _unknown)
        goldens: {char_id: np.ndarray(48,48)} — skipped chars absent
        char_stats: {char_id: {n_ok_train, n_ng_train}} — counts used for training
        n_ok_total, n_ng_total: counts after augmentation
    """
    from collections import defaultdict

    # Group OK crops by char_id for golden computation.
    # Samples tagged with char_id but no label → ignored (unlabeled).
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

    # --- Compute goldens per char (skip char_id='_unknown' or <5 OK) ---
    goldens: Dict[str, np.ndarray] = {}
    for char_id, crops in ok_by_char.items():
        if char_id == "_unknown":
            continue
        g = compute_golden(crops)
        if g is not None:
            goldens[char_id] = g
        else:
            logger.warning(
                f"[build_dataset] char '{char_id}' has {len(crops)} OK samples "
                f"(< {GOLDEN_MIN_OK_SAMPLES}) — skipping golden, diff features will be zero"
            )

    n_ok_real = sum(len(v) for v in ok_by_char.values())
    n_ng_real = sum(len(v) for v in ng_by_char.values())

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
                for aug_img, tag in augment_ng(crop, n=n_this):
                    aug_ng_by_char[char_id].append(aug_img)
                    aug_ng_type_counts[char_id][tag] += 1

    # --- Flatten + extract features (with goldens for alignment) ---
    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    crops_rows: List[np.ndarray] = []
    char_ids_rows: List[Optional[str]] = []

    def _append_samples(samples_by_char, label_val):
        for char_id, crops in samples_by_char.items():
            for c in crops:
                X_rows.append(extract_features_v2(c, char_id, goldens))
                y_rows.append(label_val)
                crops_rows.append(c)
                char_ids_rows.append(None if char_id == "_unknown" else char_id)

    _append_samples(ok_by_char, 1)
    _append_samples(ng_by_char, 0)
    _append_samples(aug_ng_by_char, 0)

    X = np.asarray(X_rows, dtype=np.float32)
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
            "has_golden":  c in goldens,
        }

    logger.info(
        f"[build_dataset v3 char-balanced] "
        f"chars: {sorted(all_chars)} | "
        f"goldens: {sorted(goldens.keys())} | "
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
        goldens, char_stats, total_ok, total_ng,
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


def train_model(
    annotations: List[MLAnnotationInDB],
    images_dir: Path,
    request: TrainRequest,
    model_save_path: Path,
) -> Dict[str, Any]:
    """
    Train a binary sklearn classifier (rf / svm / mlp) on v2 features and
    save it to disk. Always saves a sidecar test-set JSON with per-crop
    predictions.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    X, y, crops_raw, char_ids_raw, goldens, char_stats, n_ok, n_ng = build_dataset(
        annotations, images_dir, request.augment_factor,
    )

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

    bundle = {
        'clf': clf,
        'goldens': goldens,
        'char_stats': char_stats,
        'feat_version': 'v2',
        'feat_dim': FEAT_DIM_V2,
        'algorithm': algo,
    }
    _save_bundle_and_testset(model_save_path, bundle, test_set_items)

    logger.info(
        f"[train_model:{algo}] saved bundle to {model_save_path.name}: "
        f"feat_version=v2, goldens={sorted(goldens.keys())}"
    )

    return {
        "accuracy_train": acc_train,
        "accuracy_test":  acc_test,
        "n_ok":           n_ok,
        "n_ng":           n_ng,
        "n_total":        len(X),
        "confusion_matrix": cm,
        "report":         report,
        "golden_chars":   sorted(goldens.keys()),
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
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


# ──────────────────────────────────────── Prediction ──

def _load_model_bundle(model_path: Path):
    """
    Load a model from disk. Returns full bundle dict (or synthesizes one for
    legacy joblibs that stored the raw classifier directly).
    """
    data = joblib.load(str(model_path))
    if isinstance(data, dict) and ('clf' in data or 'goldens' in data):
        return data
    # Legacy: raw classifier
    return {
        'clf': data,
        'goldens': {},
        'feat_version': 'v1',
        'algorithm': 'rf',
    }


def predict_on_image(
    model_path: Path,
    image_path: Path,
    region: Optional[Dict] = None,
    threshold: float = 0.5,
    char_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Segment an image (or a region of it) and predict OK/NG per character.

    When `char_id` is provided AND model has a golden for it, result includes
    aligned/golden/diff base64 images for FE heatmap preview.

    Args:
        model_path: Path to saved joblib bundle.
        image_path: Image to predict on.
        region: Optional {x, y, w, h} normalized — segment only this area.
        threshold: Probability threshold for OK class.
        char_id: Optional char identity (uses per-char golden for alignment).

    Returns:
        List of dicts — see LabeledCrop/PredictResult schema.
    """
    bundle = _load_model_bundle(model_path)
    clf = bundle.get('clf')
    goldens = bundle.get('goldens') or {}
    feat_version = bundle.get('feat_version', 'v2')
    algorithm = bundle.get('algorithm', 'rf').lower()

    if region:
        segments = segment_region(image_path, region)
    else:
        segments = segment_region(image_path, {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0})

    use_v2 = (feat_version == 'v2')

    # Binary sklearn classifier (rf / svm / mlp). Legacy bundles with other
    # algorithm tags fall through here as well since we only keep one path.
    def _predict(crop: np.ndarray) -> Tuple[float, str]:
        feat = (extract_features_v2(crop, char_id, goldens)
                if use_v2 else extract_features(crop)).reshape(1, -1)
        proba = clf.predict_proba(feat)[0]
        p_ok = float(proba[1]) if len(proba) > 1 else float(proba[0])
        return p_ok, ("OK" if p_ok >= threshold else "NG")

    results = []
    for seg in segments:
        crop = crop_segment(image_path, seg)
        if crop is None:
            continue

        p_ok, label = _predict(crop)

        result = {
            "id": seg["id"],
            "x": seg["x"],
            "y": seg["y"],
            "w": seg["w"],
            "h": seg["h"],
            "prob_ok": round(p_ok, 4),
            "label": label,
            "crop_b64": img_to_b64(crop),
            "char_id": char_id,
            "algorithm": algorithm,
        }

        # Diff heatmap preview (whenever golden exists for char)
        if use_v2 and char_id and char_id in goldens:
            aligned, diff = _compute_diff_map(crop, char_id, goldens)
            if diff is not None:
                heatmap = cv.applyColorMap(diff, cv.COLORMAP_JET)
                result["aligned_b64"] = img_to_b64(cv.cvtColor(aligned, cv.COLOR_GRAY2BGR))
                result["golden_b64"] = img_to_b64(cv.cvtColor(goldens[char_id], cv.COLOR_GRAY2BGR))
                result["diff_b64"] = img_to_b64(heatmap)

        results.append(result)

    return results


def get_model_chars(model_path: Path) -> List[str]:
    """Return list of char_ids the model has goldens for (empty if legacy)."""
    try:
        bundle = _load_model_bundle(model_path)
        goldens = bundle.get('goldens') or {}
        return sorted(goldens.keys())
    except Exception as e:
        logger.warning(f"[get_model_chars] Failed to load {model_path}: {e}")
        return []


def get_model_goldens(model_path: Path) -> List[Dict[str, Any]]:
    """
    Return list of goldens with per-char training stats.
    Each item: {char_id, golden_b64, n_ok_train, n_ng_train, n_ok_real, n_ng_real}.
    Legacy models (no bundle): returns [].
    """
    try:
        data = joblib.load(str(model_path))
    except Exception as e:
        logger.warning(f"[get_model_goldens] Failed to load {model_path}: {e}")
        return []
    if not isinstance(data, dict) or 'goldens' not in data:
        return []

    goldens = data.get('goldens') or {}
    char_stats = data.get('char_stats') or {}
    out: List[Dict[str, Any]] = []
    for char_id in sorted(goldens.keys()):
        g = goldens[char_id]
        # Convert to BGR for uniform encode, scale up 2× for readable preview
        g_bgr = cv.cvtColor(g, cv.COLOR_GRAY2BGR) if g.ndim == 2 else g
        g_up = cv.resize(g_bgr, (g_bgr.shape[1] * 2, g_bgr.shape[0] * 2),
                         interpolation=cv.INTER_NEAREST)
        stats = char_stats.get(char_id, {})
        out.append({
            "char_id": char_id,
            "golden_b64": img_to_b64(g_up),
            "n_ok_train": int(stats.get("n_ok_train", 0)),
            "n_ng_train": int(stats.get("n_ng_train", 0)),
            "n_ok_real":  int(stats.get("n_ok_real", 0)),
            "n_ng_real":  int(stats.get("n_ng_real", 0)),
        })
    return out


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
                for aug_img, aug_tag in augment_ng(crop, n=n_per_sample):
                    result.append({
                        "source_segment_id": seg.id,
                        "filename": ann.filename,
                        "label": "NG",
                        "crop_b64": img_to_b64(aug_img),
                        "char_id": seg.char_id,
                        "aug_type": aug_tag,
                    })
    return result
