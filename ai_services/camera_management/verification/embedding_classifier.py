"""
Template ↔ target OK/NG classifier (CV-based, no model).

Per-character similarity is measured via 3 metrics on binary character masks:
  - pixel_conf  : foreground pixel-count ratio
  - blur_tm     : multi-scale blurred template matching (TM_CCOEFF_NORMED)
  - iou         : IoU after centroid alignment + dilation

confidence = min(max(blur_tm, iou), pixel_conf)  ∈ [0, 1]

Class name `EmbeddingClassifierService` is preserved for caller compatibility,
but no embedding model is loaded — pure OpenCV pipeline.
"""

import base64
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Minimum grayscale std-dev for a crop to be considered a real character.
# A black/white uniform patch → std ≈ 0. Typical characters → std > 15.
MIN_CROP_STD: float = 8.0

# Template bank tuning (hardcoded — exposed via recipe.template_bank_size/enabled)
BANK_ADD_THRESHOLD: float        = 0.90  # p_ok ≥ this → eligible for bank add
BANK_DIVERSITY_THRESHOLD: float  = 0.98  # skip add if too similar to existing
BANK_SANITY_THRESHOLD: float     = 0.70  # skip add if too dissimilar to all (outlier)
BANK_VALIDATE_THRESHOLD: float   = 0.85  # min similarity vs seed to survive on-load validate


def _crop_std(bgr: np.ndarray) -> float:
    """Return grayscale std-dev of an image crop."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    return float(np.std(gray))



# --------------------------------------------------------------------------- #
# Character quality comparison — copied from tests/test_segment.py logic
# --------------------------------------------------------------------------- #

def _to_thresh_norm(raw: np.ndarray) -> np.ndarray:
    """GaussianBlur(5,5) + Otsu inverse + morphClose(ellipse 3×3).
    Normalizes stroke width so chars like B/D/O are more comparable."""
    blurred = cv2.GaussianBlur(raw, (5, 5), 0)
    _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.mean(th) > 127:
        th = cv2.bitwise_not(th)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)


def _tight_crop(thresh: np.ndarray) -> np.ndarray:
    """Crop sát foreground, loại bỏ padding thừa."""
    coords = np.where(thresh > 0)
    if len(coords[0]) == 0:
        return thresh
    y0, y1 = int(coords[0].min()), int(coords[0].max())
    x0, x1 = int(coords[1].min()), int(coords[1].max())
    return thresh[y0:y1 + 1, x0:x1 + 1]


def _deskew_char(thresh_char: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Xoay ký tự về đứng nếu góc lệch < max_angle. Bỏ qua nếu lớn —
    tránh xoay sai với chữ bất đối xứng như L, J, F."""
    coords = np.column_stack(np.where(thresh_char > 0))
    if len(coords) < 10:
        return thresh_char
    angle = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))[2]
    if angle < -45:
        angle += 90
    if abs(angle) > max_angle:
        return thresh_char
    h, w = thresh_char.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(thresh_char, M, (w, h),
                          flags=cv2.INTER_NEAREST, borderValue=0)


def _fit_to_square(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize giữ aspect ratio, pad đen → size×size.
    Quan trọng cho ký tự hẹp (I, l, 1, j) — tránh stretch méo."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size[1], size[0]), dtype=np.uint8)
    scale = min(size[0] / w, size[1] / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size[1], size[0]), dtype=np.uint8)
    yo = (size[1] - nh) // 2
    xo = (size[0] - nw) // 2
    canvas[yo:yo + nh, xo:xo + nw] = resized
    return canvas


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component. Filters out noise blobs
    (binarization specks, edge artifacts) before centroid alignment.

    Bench (ann_idx with noise differing between template ↔ target):
      centroid alone     IoU=0.556
      largest_cc + cent. IoU=0.786
    """
    if mask.size == 0:
        return mask
    try:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return mask
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return ((labels == biggest).astype(np.uint8)) * 255
    except Exception:
        return mask


def _center_by_centroid(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Đặt mask vào khung size×size với khối tâm foreground ở giữa."""
    H, W = size[1], size[0]
    canvas = np.zeros((H, W), dtype=np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return canvas
    cy, cx = float(ys.mean()), float(xs.mean())
    h, w = mask.shape
    yo = int(round(H / 2 - cy))
    xo = int(round(W / 2 - cx))
    y1s, y1e = max(0, yo), min(H, yo + h)
    x1s, x1e = max(0, xo), min(W, xo + w)
    y2s, y2e = max(0, -yo), max(0, -yo) + (y1e - y1s)
    x2s, x2e = max(0, -xo), max(0, -xo) + (x1e - x1s)
    if y1e > y1s and x1e > x1s:
        canvas[y1s:y1e, x1s:x1e] = mask[y2s:y2e, x2s:x2e]
    return canvas


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU của 2 mask nhị phân cùng kích thước."""
    inter = int(np.count_nonzero((a > 0) & (b > 0)))
    union = int(np.count_nonzero((a > 0) | (b > 0)))
    return inter / union if union > 0 else 0.0


def _compute_char_quality(
    tmpl_gray: np.ndarray,
    tgt_gray: np.ndarray,
    size: Tuple[int, int] = (64, 64),
    denoise: bool = False,   # deprecated — old centroid IoU is replaced by ECC IoU
) -> Dict[str, float]:
    """So sánh 2 ký tự bằng 3 metric, sau khi BỎ centroid alignment:
      1. pixel_conf  — ratio px_tgt/px_tmpl
      2. blur_tm     — multi-scale blurred TM_CCOEFF_NORMED
      3. iou         — IoU sau ECC TRANSLATION alignment (sub-pixel, principled)

    tm_conf    = max(blur_tm, iou)
    confidence = min(tm_conf, pixel_conf)

    Khác bản gốc: alignment trước IoU dùng ECC thay centroid. Centroid của
    asymmetric chars (T, L, F, P, b, p, q) lệch về thanh dày → 2 char tương tự
    bị align ở vị trí khác nhau → IoU drop oan. ECC tối ưu correlation
    pixel-level → defect cục bộ không kéo lệch alignment toàn cục.
    """
    tmpl_b = _to_thresh_norm(tmpl_gray)
    tgt_b  = _to_thresh_norm(tgt_gray)

    t1 = _fit_to_square(_deskew_char(_tight_crop(tmpl_b)), size)
    g2_deskewed = _deskew_char(_tight_crop(tgt_b))
    t2_base = _fit_to_square(g2_deskewed, size)

    # (1) Pixel-count confidence
    px1 = int(np.count_nonzero(t1))
    px2 = int(np.count_nonzero(t2_base))
    ratio = px2 / (px1 + 1e-6)
    deviation = abs(ratio - 1.0)
    pixel_conf = float(np.clip(1.0 - deviation * (1.0 / 1.4), 0.0, 1.0))

    # (2) Blurred multi-scale template matching — blur biến binary thành soft mask,
    # giúp TM chịu được lệch 1–2 px và stroke khác nhau. matchTemplate tự align.
    t1_blur = cv2.GaussianBlur(t1.astype(np.float32), (0, 0), sigmaX=1.2)
    best_tm = 0.0
    for scale in (0.85, 0.92, 1.0, 1.08, 1.15):
        s = (max(t1.shape[1], int(size[0] * scale)),
             max(t1.shape[0], int(size[1] * scale)))
        t2 = _fit_to_square(g2_deskewed, s)
        t2_blur = cv2.GaussianBlur(t2.astype(np.float32), (0, 0), sigmaX=1.2)
        result = cv2.matchTemplate(t2_blur, t1_blur, cv2.TM_CCOEFF_NORMED)
        best_tm = max(best_tm, float(result.max()))
    blur_tm = float(np.clip(best_tm, 0.0, 1.0))

    # (3) IoU sau ECC TRANSLATION alignment + dilate(ellipse 5×5)
    b_aligned = t2_base
    try:
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-3)
        t1f = t1.astype(np.float32)
        t2f = t2_base.astype(np.float32)
        _, warp = cv2.findTransformECC(t1f, t2f, warp, cv2.MOTION_TRANSLATION,
                                        criteria, None, 3)
        if np.isfinite(warp).all() and abs(warp[0, 2]) <= size[0] / 4 \
                                    and abs(warp[1, 2]) <= size[1] / 4:
            b_aligned = cv2.warpAffine(
                t2_base, warp, size,
                flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
    except cv2.error:
        pass  # ECC diverged → IoU on un-aligned masks (fallback)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    a = cv2.dilate(t1, k, iterations=1)
    b = cv2.dilate(b_aligned, k, iterations=1)
    iou = _iou(a, b)

    tm_conf = max(blur_tm, iou)
    confidence = min(tm_conf, pixel_conf)

    return {
        "confidence": float(confidence),
        "tm_conf":    float(tm_conf),
        "blur_tm":    float(blur_tm),
        "iou":        float(iou),
        "pixel_conf": float(pixel_conf),
        "px_tmpl":    px1,
        "px_tgt":     px2,
        # Diff-XOR masks for debug — template bbox-centered, target ECC-aligned
        "_mask_tmpl_aligned": t1,
        "_mask_tgt_aligned":  b_aligned,
    }


def _encode_diff_mask_b64(mask_tmpl: np.ndarray, mask_tgt: np.ndarray) -> Optional[str]:
    """XOR 2 aligned masks → PNG base64. Returns None if encode fails."""
    try:
        diff = cv2.bitwise_xor(mask_tmpl, mask_tgt)
        ok, buf = cv2.imencode('.png', diff)
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode('ascii')
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Template Bank — online adaptive multi-template per (recipe, camera, ann_idx)
# --------------------------------------------------------------------------- #

@dataclass
class TemplateRecord:
    """One template slot with pre-computed features for fast runtime compare."""
    raw_bgr: np.ndarray      # original BGR crop (for inspection / re-encode to disk)
    t1: np.ndarray           # (64,64) uint8 thresh+aligned (for px_count, blur_tm)
    t1_blur: np.ndarray      # (64,64) float32 pre-Gaussian σ=1.2 (for multi-scale TM)
    a_centroid: np.ndarray   # (64,64) uint8 centroid-aligned + dilated (for IoU)
    px_count: int
    added_at: float
    is_seed: bool = False
    hit_count: int = 0       # bumped each time this template is the best-match in compare()

    @classmethod
    def from_bgr(cls, bgr: np.ndarray, is_seed: bool = False, denoise: bool = False) -> Optional['TemplateRecord']:
        """Pre-compute all features from a raw BGR crop. Returns None on failure.
        If denoise=True, applies largest-CC noise filter before centroid alignment."""
        try:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
            tb = _to_thresh_norm(gray)
            t1 = _fit_to_square(_deskew_char(_tight_crop(tb)), (64, 64))
            if np.count_nonzero(t1) == 0:
                return None
            t1_blur = cv2.GaussianBlur(t1.astype(np.float32), (0, 0), sigmaX=1.2)
            t1_for_iou = _largest_cc(t1) if denoise else t1
            a = _center_by_centroid(_tight_crop(t1_for_iou), (64, 64))
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            a_dilated = cv2.dilate(a, k, iterations=1)
            return cls(
                raw_bgr=bgr.copy(),
                t1=t1, t1_blur=t1_blur, a_centroid=a_dilated,
                px_count=int(np.count_nonzero(t1)),
                added_at=time.time(),
                is_seed=is_seed,
            )
        except Exception as e:
            logger.warning(f"TemplateRecord.from_bgr failed: {e}")
            return None


def _compare_target_vs_record(
    target_t1: np.ndarray,
    target_t1_blur: np.ndarray,
    target_a_dilated: np.ndarray,
    target_px: int,
    tmpl: 'TemplateRecord',
) -> Dict[str, float]:
    """Single template ↔ pre-processed target comparison using cached tmpl features.
    All inputs are already 64×64 (template + target same shape after preprocessing)."""
    # (1) pixel confidence
    ratio = target_px / (tmpl.px_count + 1e-6)
    pixel_conf = float(np.clip(1.0 - abs(ratio - 1.0) * (1.0 / 1.4), 0.0, 1.0))

    # (2) multi-scale blurred TM
    best_tm = 0.0
    for scale in (0.85, 0.92, 1.0, 1.08, 1.15):
        s = (max(tmpl.t1.shape[1], int(64 * scale)),
             max(tmpl.t1.shape[0], int(64 * scale)))
        # Re-fit target at this scale (cheap — t1 is 64x64)
        if s == (64, 64):
            t2_blur = target_t1_blur
        else:
            t2 = _fit_to_square(target_t1, s)
            t2_blur = cv2.GaussianBlur(t2.astype(np.float32), (0, 0), sigmaX=1.2)
        result = cv2.matchTemplate(t2_blur, tmpl.t1_blur, cv2.TM_CCOEFF_NORMED)
        best_tm = max(best_tm, float(result.max()))
    blur_tm = float(np.clip(best_tm, 0.0, 1.0))

    # (3) IoU on centroid-aligned + dilated masks
    iou = _iou(tmpl.a_centroid, target_a_dilated)

    tm_conf = max(blur_tm, iou)
    confidence = min(tm_conf, pixel_conf)
    return {
        "confidence": float(confidence),
        "tm_conf": float(tm_conf),
        "blur_tm": float(blur_tm),
        "iou": float(iou),
        "pixel_conf": float(pixel_conf),
    }


def _preprocess_target_for_bank(target_bgr_or_gray: np.ndarray, denoise: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """Pre-compute target features once (shared across all template comparisons in bank).
    Returns (t1, t1_blur, a_dilated, px_count) or None on failure.
    If denoise=True, applies largest-CC noise filter before centroid alignment."""
    try:
        gray = (cv2.cvtColor(target_bgr_or_gray, cv2.COLOR_BGR2GRAY)
                if target_bgr_or_gray.ndim == 3 else target_bgr_or_gray)
        tb = _to_thresh_norm(gray)
        t1 = _fit_to_square(_deskew_char(_tight_crop(tb)), (64, 64))
        if np.count_nonzero(t1) == 0:
            return None
        t1_blur = cv2.GaussianBlur(t1.astype(np.float32), (0, 0), sigmaX=1.2)
        t1_for_iou = _largest_cc(t1) if denoise else t1
        a = _center_by_centroid(_tight_crop(t1_for_iou), (64, 64))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        a_dilated = cv2.dilate(a, k, iterations=1)
        return t1, t1_blur, a_dilated, int(np.count_nonzero(t1))
    except Exception as e:
        logger.warning(f"preprocess_target_for_bank failed: {e}")
        return None


class TemplateBank:
    """One bank per (recipe_id, camera_serial, annotation_idx).
    Seed templates locked from recipe; dynamic templates auto-collected at runtime."""

    def __init__(self, bank_dir: Path, size: int, denoise: bool = False):
        self.bank_dir = bank_dir
        self.size = max(1, int(size))
        self.denoise = bool(denoise)
        self.seed: List[TemplateRecord] = []
        self.dynamic: List[TemplateRecord] = []
        self._dynamic_filenames: List[str] = []  # parallel to self.dynamic
        # Cached pairwise similarity matrix over (seed + dynamic) in flat order.
        # Eliminates O(N²) recompute in _replace_weakest — incremental O(N) updates.
        # Indices: 0..len(seed)-1 = seed rows, len(seed).. = dynamic rows.
        self._pairwise: Optional[np.ndarray] = None  # (n, n) float32, symmetric, diag=1

    @classmethod
    def load_or_create(
        cls,
        bank_dir: Path,
        size: int,
        seed_bgr: np.ndarray,
        denoise: bool = False,
    ) -> 'TemplateBank':
        """Build bank: seed from recipe + load+validate dynamic from disk."""
        bank = cls(bank_dir, size, denoise=denoise)
        seed_rec = TemplateRecord.from_bgr(seed_bgr, is_seed=True, denoise=denoise)
        if seed_rec is not None:
            bank.seed.append(seed_rec)
        else:
            logger.warning(f"Bank {bank_dir}: seed template could not be encoded")
            return bank  # empty bank — caller should fallback

        if not bank_dir.exists():
            bank_dir.mkdir(parents=True, exist_ok=True)
            return bank

        # Validate each persisted dynamic vs seed
        for png in sorted(bank_dir.glob('dynamic_*.png')):
            try:
                img = cv2.imread(str(png))
                if img is None:
                    png.unlink()
                    continue
                # Compute similarity against seed
                target_pre = _preprocess_target_for_bank(img, denoise=denoise)
                if target_pre is None:
                    png.unlink()
                    continue
                t1, t1b, ad, pxc = target_pre
                metrics = _compare_target_vs_record(t1, t1b, ad, pxc, seed_rec)
                if metrics['confidence'] < BANK_VALIDATE_THRESHOLD:
                    logger.info(f"Bank {bank_dir.name}: drop stale {png.name} "
                                f"(sim vs seed={metrics['confidence']:.2f} < {BANK_VALIDATE_THRESHOLD})")
                    png.unlink()
                    continue
                rec = TemplateRecord.from_bgr(img, denoise=denoise)
                if rec is None:
                    png.unlink()
                    continue
                bank.dynamic.append(rec)
                bank._dynamic_filenames.append(png.name)
            except Exception as e:
                logger.warning(f"Bank {bank_dir}: failed to load {png}: {e}")

        # Cap to size (in case persisted files exceed configured size)
        if len(bank.dynamic) > bank.size:
            bank.dynamic = bank.dynamic[:bank.size]
            bank._dynamic_filenames = bank._dynamic_filenames[:bank.size]

        # Pre-compute pairwise similarity matrix once — eliminates O(N²) recompute
        # inside each future _replace_weakest call.
        bank._init_pairwise()

        logger.info(f"TemplateBank {bank_dir.name}: 1 seed + {len(bank.dynamic)}/{bank.size} dynamic loaded")
        return bank

    # ── Pairwise similarity matrix (cached) ────────────────────────────────

    def _flat_templates(self) -> List[TemplateRecord]:
        """Seed + dynamic in stable insertion order (NOT sorted)."""
        return self.seed + self.dynamic

    def _compute_sim(self, a: TemplateRecord, b: TemplateRecord) -> float:
        """Pairwise template ↔ template similarity using cached features."""
        return _compare_target_vs_record(b.t1, b.t1_blur, b.a_centroid, b.px_count, a)['confidence']

    def _init_pairwise(self) -> None:
        """Build full N×N pairwise matrix from scratch. Called once after load."""
        flat = self._flat_templates()
        n = len(flat)
        if n == 0:
            self._pairwise = None
            return
        mat = np.eye(n, dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                conf = self._compute_sim(flat[i], flat[j])
                mat[i, j] = mat[j, i] = conf
        self._pairwise = mat

    def _matrix_in_sync(self) -> bool:
        n = len(self._flat_templates())
        return self._pairwise is not None and self._pairwise.shape == (n, n)

    def _append_row_to_matrix(self, sims_to_existing: List[float]) -> None:
        """Grow matrix by 1 row+col using already-computed similarities of the
        new template to each EXISTING template (flat order, BEFORE append)."""
        if not self._matrix_in_sync():
            # Out of sync — defer to full rebuild after caller appends
            self._pairwise = None
            return
        n_old = self._pairwise.shape[0]
        if len(sims_to_existing) != n_old:
            self._pairwise = None
            return
        new_mat = np.eye(n_old + 1, dtype=np.float32)
        new_mat[:n_old, :n_old] = self._pairwise
        sims_arr = np.asarray(sims_to_existing, dtype=np.float32)
        new_mat[:n_old, n_old] = sims_arr
        new_mat[n_old, :n_old] = sims_arr
        self._pairwise = new_mat

    def _replace_row_in_matrix(self, flat_idx: int, sims_to_others: List[float]) -> None:
        """Overwrite row/col at flat_idx using pre-computed similarities of new
        template to each template in flat order (sims_to_others[flat_idx] ignored)."""
        if not self._matrix_in_sync():
            self._pairwise = None
            return
        n = self._pairwise.shape[0]
        if flat_idx < 0 or flat_idx >= n or len(sims_to_others) != n:
            self._pairwise = None
            return
        sims_arr = np.asarray(sims_to_others, dtype=np.float32)
        self._pairwise[flat_idx, :] = sims_arr
        self._pairwise[:, flat_idx] = sims_arr
        self._pairwise[flat_idx, flat_idx] = 1.0

    def _find_weakest_dynamic_idx(self) -> Optional[int]:
        """Index INTO self.dynamic of slot with lowest avg pairwise sim to OTHERS.
        Returns None if matrix unavailable or no dynamic slots."""
        if not self._matrix_in_sync() or not self.dynamic:
            return None
        n_seed = len(self.seed)
        n_total = n_seed + len(self.dynamic)
        # Average of each dynamic row, excluding self (diag = 1.0)
        rows = self._pairwise[n_seed:n_total]
        sums = rows.sum(axis=1) - 1.0  # subtract diag
        denom = max(1, n_total - 1)
        avgs = sums / denom
        return int(np.argmin(avgs))

    def all_templates(self) -> List[TemplateRecord]:
        """Seed + dynamic, ordered by hit_count desc (best-first for early termination)."""
        return sorted(self.seed + self.dynamic, key=lambda t: -t.hit_count)

    def compare(self, target_gray: np.ndarray, threshold: float) -> Tuple[float, Optional[Dict[str, float]], Optional[TemplateRecord]]:
        """Find best template ↔ target match. Early-terminate when conf ≥ threshold.
        Returns (best_conf, best_metrics, best_template)."""
        target_pre = _preprocess_target_for_bank(target_gray, denoise=self.denoise)
        if target_pre is None:
            return 0.0, None, None
        t1, t1b, ad, pxc = target_pre

        best_conf = 0.0
        best_metrics: Optional[Dict[str, float]] = None
        best_template: Optional[TemplateRecord] = None

        for tmpl in self.all_templates():
            # Cheap filter: pixel-count ratio sanity (skip clearly hopeless candidates)
            ratio = pxc / (tmpl.px_count + 1e-6)
            if abs(ratio - 1.0) > 0.4:
                continue
            metrics = _compare_target_vs_record(t1, t1b, ad, pxc, tmpl)
            if metrics['confidence'] > best_conf:
                best_conf = metrics['confidence']
                best_metrics = metrics
                best_template = tmpl
            if best_conf >= threshold:
                break  # early-terminate

        if best_template is not None:
            best_template.hit_count += 1

        # Embed aligned masks for caller (diff XOR rendering)
        if best_metrics is not None and best_template is not None:
            best_metrics['_mask_tmpl_aligned'] = best_template.a_centroid
            best_metrics['_mask_tgt_aligned']  = ad

        return best_conf, best_metrics, best_template

    def try_add(self, target_bgr: np.ndarray, p_ok: float) -> bool:
        """Add target as new dynamic template if conditions are met.
        Returns True if added.

        Side effect: the pairwise similarities computed for diversity/sanity
        checks are reused to update the cached pairwise matrix — avoiding any
        extra compute when the template is actually appended/replaced.
        """
        if p_ok < BANK_ADD_THRESHOLD:
            return False

        target_pre = _preprocess_target_for_bank(target_bgr, denoise=self.denoise)
        if target_pre is None:
            return False
        t1, t1b, ad, pxc = target_pre

        # Compute target ↔ each existing template similarity (flat order: seed + dynamic).
        # These same numbers will be reused as the new row/col in the pairwise matrix.
        flat = self._flat_templates()
        sims: List[float] = []
        for tmpl in flat:
            ratio = pxc / (tmpl.px_count + 1e-6)
            if abs(ratio - 1.0) > 0.4:
                sims.append(0.0)
                continue
            m = _compare_target_vs_record(t1, t1b, ad, pxc, tmpl)
            sims.append(m['confidence'])

        if not sims:
            return False
        if max(sims) > BANK_DIVERSITY_THRESHOLD:
            return False  # too similar — waste of slot
        if max(sims) < BANK_SANITY_THRESHOLD:
            return False  # outlier — suspicious

        new_rec = TemplateRecord.from_bgr(target_bgr, denoise=self.denoise)
        if new_rec is None:
            return False

        if len(self.dynamic) < self.size:
            self._append(new_rec, sims_to_existing=sims)
        else:
            self._replace_weakest(new_rec, sims_to_existing=sims)
        return True

    def _append(self, rec: TemplateRecord, sims_to_existing: Optional[List[float]] = None) -> None:
        """Add a new dynamic slot and persist to disk.
        sims_to_existing: pre-computed similarities of rec to each existing template
        (flat order). When provided, the pairwise matrix is updated in O(N) — no
        extra comparisons. When None, matrix is invalidated and lazily rebuilt."""
        try:
            filename = self._next_filename()

            # Update matrix BEFORE adding rec to dynamic — uses pre-computed sims.
            if sims_to_existing is not None and self._matrix_in_sync():
                self._append_row_to_matrix(sims_to_existing)
            else:
                # Matrix out of sync → trigger lazy rebuild after append below
                self._pairwise = None

            self.dynamic.append(rec)
            self._dynamic_filenames.append(filename)

            if self._pairwise is None:
                self._init_pairwise()  # full rebuild fallback

            self._save_record_to_disk(rec, filename)
            logger.info(f"Bank {self.bank_dir.name}: appended {filename} ({len(self.dynamic)}/{self.size})")
        except Exception as e:
            logger.warning(f"Bank {self.bank_dir.name}: failed to append: {e}")

    def _replace_weakest(self, rec: TemplateRecord, sims_to_existing: Optional[List[float]] = None) -> None:
        """Replace the dynamic slot with lowest avg pairwise similarity to others.
        sims_to_existing: same shape as _flat_templates() — used to update matrix
        row/col without recomputing. When None, matrix is rebuilt full."""
        try:
            if not self.dynamic:
                return

            # Find weakest using O(N) lookup on cached matrix.
            if not self._matrix_in_sync():
                self._init_pairwise()
            weakest_local = self._find_weakest_dynamic_idx()
            if weakest_local is None:
                logger.warning(f"Bank {self.bank_dir.name}: cannot find weakest (matrix missing)")
                return
            n_seed = len(self.seed)
            flat_idx = n_seed + weakest_local

            # Snapshot avg_sim of the slot being evicted (for log only)
            row_sum = float(self._pairwise[flat_idx].sum() - 1.0)
            denom = max(1, self._pairwise.shape[0] - 1)
            evicted_avg_sim = row_sum / denom

            old_filename = self._dynamic_filenames[weakest_local]
            try:
                (self.bank_dir / old_filename).unlink(missing_ok=True)
            except Exception:
                pass

            new_filename = self._next_filename()
            self.dynamic[weakest_local] = rec
            self._dynamic_filenames[weakest_local] = new_filename

            # Update matrix row/col with already-known sims (no extra compute).
            if sims_to_existing is not None and len(sims_to_existing) == self._pairwise.shape[0]:
                self._replace_row_in_matrix(flat_idx, sims_to_existing)
            else:
                # Fallback — sims missing or shape changed
                self._pairwise = None
                self._init_pairwise()

            self._save_record_to_disk(rec, new_filename)
            logger.info(f"Bank {self.bank_dir.name}: replaced {old_filename} → {new_filename} "
                        f"(slot {weakest_local}, avg_sim was {evicted_avg_sim:.2f})")
        except Exception as e:
            logger.warning(f"Bank {self.bank_dir.name}: failed to replace weakest: {e}")

    def _next_filename(self) -> str:
        """Next available dynamic_NNN.png that doesn't collide on disk."""
        existing = {p.name for p in self.bank_dir.glob('dynamic_*.png')}
        for i in range(10000):
            cand = f"dynamic_{i:03d}.png"
            if cand not in existing and cand not in self._dynamic_filenames:
                return cand
        # Fallback (very unlikely)
        return f"dynamic_{int(time.time())}.png"

    def _save_record_to_disk(self, rec: TemplateRecord, filename: str) -> None:
        try:
            self.bank_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.bank_dir / filename), rec.raw_bgr)
        except Exception as e:
            logger.warning(f"Bank {self.bank_dir}: imwrite failed for {filename}: {e}")

    def clear_dynamic(self) -> None:
        """Drop all dynamic templates (keep seed). Deletes files on disk."""
        self.dynamic.clear()
        self._dynamic_filenames.clear()
        if self.bank_dir.exists():
            for png in self.bank_dir.glob('dynamic_*.png'):
                try:
                    png.unlink()
                except Exception:
                    pass
        logger.info(f"Bank {self.bank_dir.name}: cleared all dynamic templates")


class TemplateBankRegistry:
    """Process-wide registry of TemplateBank instances keyed by
    (recipe_id, camera_serial, annotation_idx)."""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self._banks: Dict[Tuple[str, str, int], TemplateBank] = {}

    def _bank_path(self, recipe_id: str, camera_serial: str, ann_idx: int) -> Path:
        return self.base_dir / str(recipe_id) / str(camera_serial) / f"ann_{int(ann_idx):03d}"

    def get_or_create(
        self,
        recipe_id: str,
        camera_serial: str,
        ann_idx: int,
        seed_bgr: np.ndarray,
        size: int,
        seed_version_key: Optional[str] = None,
        denoise: bool = False,
    ) -> TemplateBank:
        """Get bank, lazy-loading from disk on first access.

        seed_version_key: if provided, stored in meta.json. On future loads, if
            current seed_version_key differs → dynamic templates are wiped (seed changed).
        denoise: when toggled, treated as part of version key so a flip wipes
            stale pre-computed features.
        """
        key = (str(recipe_id), str(camera_serial), int(ann_idx))
        bank = self._banks.get(key)
        if bank is not None and bank.size == size and bank.denoise == denoise:
            return bank

        bank_dir = self._bank_path(recipe_id, camera_serial, ann_idx)
        # Bake denoise flag into version_key so toggling forces a clean rebuild.
        effective_version = seed_version_key
        if effective_version is not None:
            effective_version = f"{effective_version}|denoise={int(bool(denoise))}"
        if effective_version is not None:
            self._invalidate_if_seed_changed(bank_dir, effective_version)

        bank = TemplateBank.load_or_create(bank_dir, size, seed_bgr, denoise=denoise)
        self._banks[key] = bank
        return bank

    def _invalidate_if_seed_changed(self, bank_dir: Path, seed_version_key: str) -> None:
        meta_path = bank_dir / 'meta.json'
        bank_dir.mkdir(parents=True, exist_ok=True)
        prev_key = None
        if meta_path.exists():
            try:
                prev_key = json.loads(meta_path.read_text()).get('seed_version_key')
            except Exception:
                prev_key = None

        if prev_key == seed_version_key:
            return  # no change

        # Only wipe when there was a PREVIOUS key (real seed change).
        # First-run case (prev_key is None) just stamps meta — does NOT wipe.
        if prev_key is not None:
            for png in bank_dir.glob('dynamic_*.png'):
                try:
                    png.unlink()
                except Exception:
                    pass
            logger.info(f"Bank {bank_dir.name}: seed changed ({prev_key} → {seed_version_key}), "
                        f"dynamic templates wiped")

        try:
            meta_path.write_text(json.dumps({
                'seed_version_key': seed_version_key,
                'updated_at': time.time(),
            }))
        except Exception:
            pass

    def get_existing(self, recipe_id: str, camera_serial: str, ann_idx: int) -> Optional[TemplateBank]:
        """Return bank if already loaded, else None (does NOT create)."""
        return self._banks.get((str(recipe_id), str(camera_serial), int(ann_idx)))

    def reset(
        self,
        recipe_id: Optional[str] = None,
        camera_serial: Optional[str] = None,
        ann_idx: Optional[int] = None,
    ) -> int:
        """Clear in-memory + on-disk dynamic templates matching criteria.
        Returns number of banks reset."""
        to_reset = []
        for key in list(self._banks.keys()):
            rid, cs, ai = key
            if recipe_id is not None and rid != str(recipe_id):
                continue
            if camera_serial is not None and cs != str(camera_serial):
                continue
            if ann_idx is not None and ai != int(ann_idx):
                continue
            to_reset.append(key)
        for key in to_reset:
            self._banks[key].clear_dynamic()
            del self._banks[key]
        return len(to_reset)


class EmbeddingClassifierService:
    """
    Per-character OK/NG classifier (CV-based, no model).

    classify_batch() compares each template/target pair using
    `_compute_char_quality` — multi-metric character similarity.

    `onnx_path` and `config_path` are kept in the signature for backward
    compatibility with existing callers but are NOT loaded or used.
    """

    def __init__(
        self,
        onnx_path: str,
        config_path: str,
        save_debug_images: bool = True,
        debug_path: Optional[str] = None,
    ):
        """
        OLD: ONNX MODEL LOADING (commented out — CV-based mode does not need a model)

        cfg = OmegaConf.load(config_path)
        self.head = cfg.model.head.type
        self.size = int(cfg.data.image_size)

        if self.head not in ("projection", "arcface"):
            raise ValueError(
                f"EmbeddingClassifierService requires projection/arcface head, got: {self.head}"
            )

        self.sess = ort.InferenceSession(
            str(onnx_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.input_name = self.sess.get_inputs()[0].name
        """

        # Params accepted but unused — kept for signature compatibility
        self._onnx_path = onnx_path
        self._config_path = config_path

        self.save_debug_images = save_debug_images
        self.debug_path = debug_path or f"{os.environ.get('HOME')}/Source/ocr_datecode/ai_services/test_result"
        if self.save_debug_images:
            os.makedirs(self.debug_path, exist_ok=True)

        # Template bank registry — persists to filesystem.
        # Filesystem path is sibling of debug_path so it shares the same root.
        bank_base = os.path.join(os.path.dirname(self.debug_path), 'template_banks')
        self.bank_registry = TemplateBankRegistry(bank_base)

        # Thread pool for parallel commit_bank_adds across N banks per frame.
        # max_workers tuned for Jetson Orin Nano (6× Cortex-A78AE).
        # cv2.matchTemplate releases the GIL so true parallelism is achievable.
        self._commit_pool: Optional[ThreadPoolExecutor] = None
        self._commit_pool_workers = int(os.environ.get('BANK_COMMIT_WORKERS', '6'))

        logger.info(
            f"EmbeddingClassifierService (CV-based mode): no model loaded, "
            f"debug={self.debug_path}, bank_base={bank_base}, "
            f"commit_workers={self._commit_pool_workers}"
        )

    def _get_commit_pool(self) -> ThreadPoolExecutor:
        if self._commit_pool is None:
            self._commit_pool = ThreadPoolExecutor(
                max_workers=self._commit_pool_workers,
                thread_name_prefix='bank-commit',
            )
        return self._commit_pool

    def classify_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Batch classify N character crops by template ↔ target CV similarity.

        Each item must supply both `region_img` (target) and `template_crop`.
        Per-item `conf_threshold` is compared against `p_ok` to decide OK/NG.

        Input item shape:
            {
                'region_img':     np.ndarray  — target crop (current frame)
                'template_crop':  np.ndarray  — template crop (required)
                'conf_threshold': float       — per-character OK threshold
                'serial_number':  str
                'annotation_idx': int
            }

        Returns list parallel to items:
            {'ml_pass', 'p_ok', 'label', 'threshold', 'time_ms', 'error'}
        """

        # ====================================================================
        # NEW: CV-based per-pair character quality (no model inference)
        # ====================================================================
        n = len(items)
        if n == 0:
            return []

        t0 = time.perf_counter()
        results: List[Optional[Dict[str, Any]]] = [None] * n
        valid_idxs: List[int] = []

        # ---- Stage 1: validate inputs ----
        for i, item in enumerate(items):
            region = item.get('region_img')
            template = item.get('template_crop')
            conf_thr = float(item.get('conf_threshold', 0.5))
            serial = item.get('serial_number', '')
            ann_idx = item.get('annotation_idx', -1)

            if region is None or region.size == 0:
                results[i] = {
                    'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                    'threshold': conf_thr, 'time_ms': 0.0, 'error': 'empty_region',
                }
                continue

            if template is None or template.size == 0:
                results[i] = {
                    'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                    'threshold': conf_thr, 'time_ms': 0.0, 'error': 'missing_template_crop',
                }
                continue

            tgt_std = _crop_std(region)
            if tgt_std < MIN_CROP_STD:
                logger.debug(
                    f"[{serial}] ann {ann_idx}: target crop too uniform "
                    f"(std={tgt_std:.1f} < {MIN_CROP_STD}) → NG"
                )
                results[i] = {
                    'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                    'threshold': conf_thr, 'time_ms': 0.0,
                    'error': f'low_variance_target:{tgt_std:.1f}',
                }
                continue

            tmpl_std = _crop_std(template)
            if tmpl_std < MIN_CROP_STD:
                logger.warning(
                    f"[{serial}] ann {ann_idx}: template crop too uniform "
                    f"(std={tmpl_std:.1f} < {MIN_CROP_STD}), check recipe"
                )
                results[i] = {
                    'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                    'threshold': conf_thr, 'time_ms': 0.0,
                    'error': f'low_variance_template:{tmpl_std:.1f}',
                }
                continue

            valid_idxs.append(i)

        if not valid_idxs:
            for i, r in enumerate(results):
                if r is None:
                    results[i] = {
                        'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                        'threshold': float(items[i].get('conf_threshold', 0.5)),
                        'time_ms': 0.0, 'error': 'all_invalid',
                    }
            return results  # type: ignore

        # ---- Stage 2: per-pair comparison ----
        m = len(valid_idxs)

        debug_dir = None
        if self.save_debug_images:
            try:
                serial = items[valid_idxs[0]].get('serial_number', 'unknown')
                ts = time.strftime("%Y%m%d_%H%M%S")
                debug_dir = os.path.join(self.debug_path, f"emb_{serial}_{ts}")
                os.makedirs(debug_dir, exist_ok=True)
            except Exception:
                debug_dir = None

        for i in valid_idxs:
            item = items[i]
            region = item['region_img']
            template = item['template_crop']
            conf_thr = float(item.get('conf_threshold', 0.5))

            # HARDCODED: template bank disabled (per recipe-system refactor).
            # `template_bank_enabled` recipe field is ignored — always single-template path.
            bank_enabled    = False
            bank_size       = int(item.get('template_bank_size', 10))
            denoise_enabled = bool(item.get('char_denoise_enabled', False))
            recipe_id       = item.get('recipe_id')
            serial          = item.get('serial_number', '')
            ann_idx         = item.get('annotation_idx', -1)
            seed_version_key = item.get('template_version_key')
            # Per-recipe CV method routing: 'legacy' (default) | 'v4' | 'v7'
            cv_method = str(item.get('cv_method', 'legacy')).lower()

            try:
                metrics: Optional[Dict[str, Any]] = None
                best_template = None
                p_ok = 0.0

                if bank_enabled and recipe_id and ann_idx >= 0:
                    bank = self.bank_registry.get_or_create(
                        recipe_id=str(recipe_id),
                        camera_serial=str(serial),
                        ann_idx=int(ann_idx),
                        seed_bgr=template,
                        size=bank_size,
                        seed_version_key=seed_version_key,
                        denoise=denoise_enabled,
                    )
                    if bank.seed:
                        tgt_gray = (cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                                    if region.ndim == 3 else region)
                        p_ok, metrics, best_template = bank.compare(tgt_gray, conf_thr)

                # Fallback to single-template path. Branch by cv_method.
                if metrics is None:
                    tmpl_gray = (cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                                 if template.ndim == 3 else template)
                    tgt_gray = (cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                                if region.ndim == 3 else region)

                    if cv_method == 'v3':
                        from .char_quality_v3 import compute_char_quality_v3
                        metrics = compute_char_quality_v3(tmpl_gray, tgt_gray)
                        p_ok = float(metrics['confidence'])
                        if metrics.get('defect_type'):
                            p_ok = min(p_ok, max(0.0, conf_thr - 0.01))
                    elif cv_method == 'v4':
                        from .char_quality_v4 import compute_char_quality_v4
                        metrics = compute_char_quality_v4(tmpl_gray, tgt_gray)
                        p_ok = float(metrics['confidence'])
                        # v4 defect_type → cap p_ok so NG verdict triggers
                        if metrics.get('defect_type'):
                            p_ok = min(p_ok, max(0.0, conf_thr - 0.01))
                    elif cv_method == 'v5':
                        from .char_quality_v5 import compute_char_quality_v5
                        metrics = compute_char_quality_v5(tmpl_gray, tgt_gray)
                        p_ok = float(metrics['confidence'])
                        if metrics.get('defect_type'):
                            p_ok = min(p_ok, max(0.0, conf_thr - 0.01))
                    elif cv_method in ('v7', 'shape_v7'):
                        from .char_quality_v7_shape import compute_char_quality_v7
                        metrics = compute_char_quality_v7(tmpl_gray, tgt_gray)
                        p_ok = float(metrics['confidence'])
                        if metrics.get('defect_type'):
                            p_ok = min(p_ok, max(0.0, conf_thr - 0.01))
                    else:  # 'legacy' or unknown → original CV pipeline
                        metrics = _compute_char_quality(tmpl_gray, tgt_gray, denoise=denoise_enabled)
                        p_ok = float(metrics['confidence'])

                label = "OK" if p_ok >= conf_thr else "NG"
                # Compute diff mask base64 — schema differs per cv_method
                if '_mask_tmpl_aligned' in metrics and '_mask_tgt_aligned' in metrics:
                    mask_b64 = _encode_diff_mask_b64(
                        metrics['_mask_tmpl_aligned'], metrics['_mask_tgt_aligned']
                    )
                elif '_t_bin' in metrics and '_g_bin' in metrics:
                    mask_b64 = _encode_diff_mask_b64(metrics['_t_bin'], metrics['_g_bin'])
                elif '_quant_t' in metrics and '_quant_g' in metrics:
                    t_b = ((metrics['_quant_t'] > 0).astype(np.uint8)) * 255
                    g_b = ((metrics['_quant_g'] > 0).astype(np.uint8)) * 255
                    mask_b64 = _encode_diff_mask_b64(t_b, g_b)
                else:
                    mask_b64 = None
                results[i] = {
                    'ml_pass': (label == "OK"),
                    'p_ok': round(p_ok, 4),
                    'label': label,
                    'threshold': conf_thr,
                    'time_ms': 0.0,  # filled after batch
                    'error': None,
                    'mask_diff_b64': mask_b64,
                }
                # Stash bank context for post-batch try_add (only set when bank was used)
                if bank_enabled and recipe_id and ann_idx >= 0 and best_template is not None:
                    results[i]['_bank_try_add'] = {
                        'recipe_id': str(recipe_id),
                        'serial': str(serial),
                        'ann_idx': int(ann_idx),
                        'target_bgr': region,
                        'p_ok': p_ok,
                    }

                if debug_dir is not None:
                    try:
                        ann = item.get('annotation_idx', i)
                        prefix = f"char{ann:02d}_{label}_p{p_ok:.2f}"
                        cv2.imwrite(os.path.join(debug_dir, f"{prefix}_template.png"),
                                    template)
                        cv2.imwrite(os.path.join(debug_dir, f"{prefix}_target.png"),
                                    region)
                    except Exception:
                        pass

                # Debug log — metric keys differ per cv_method
                if cv_method == 'legacy':
                    logger.debug(
                        f"[{item.get('serial_number', '')}] cv ann "
                        f"{item.get('annotation_idx', -1)}: "
                        f"conf={p_ok:.4f} tm={metrics.get('tm_conf', 0):.3f} "
                        f"blur_tm={metrics.get('blur_tm', 0):.3f} iou={metrics.get('iou', 0):.3f} "
                        f"px={metrics.get('pixel_conf', 0):.3f} {label} thr={conf_thr}"
                    )
                elif cv_method in ('v3', 'v4', 'v5'):
                    logger.debug(
                        f"[{item.get('serial_number', '')}] cv ann "
                        f"{item.get('annotation_idx', -1)}: {cv_method} conf={p_ok:.4f} "
                        f"ncc={metrics.get('ncc', 0):.3f} "
                        f"over={metrics.get('over_ink_score', 0):.3f} "
                        f"under={metrics.get('under_ink_score', 0):.3f} "
                        f"defect={metrics.get('defect_type')} {label} thr={conf_thr}"
                    )
                elif cv_method == 'v7':
                    logger.debug(
                        f"[{item.get('serial_number', '')}] cv ann "
                        f"{item.get('annotation_idx', -1)}: v7 conf={p_ok:.4f} "
                        f"match={100*metrics.get('orientation_match_pct', 0):.0f}% "
                        f"strong_px={metrics.get('n_strong_pixels', 0)} "
                        f"defect={metrics.get('defect_type')} {label} thr={conf_thr}"
                    )
            except Exception as e:
                results[i] = {
                    'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                    'threshold': conf_thr, 'time_ms': 0.0,
                    'error': f'compare_failed:{e}',
                }

        elapsed = (time.perf_counter() - t0) * 1000
        time_per_item = round(elapsed / m, 2)
        for i in valid_idxs:
            if results[i] is not None:
                results[i]['time_ms'] = time_per_item

        logger.info(
            f"CV classify batch: N={m}, elapsed={elapsed:.1f}ms ({time_per_item:.2f}ms/item)"
        )

        for i, r in enumerate(results):
            if r is None:
                results[i] = {
                    'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                    'threshold': float(items[i].get('conf_threshold', 0.5)),
                    'time_ms': 0.0, 'error': 'unknown',
                }

        return results  # type: ignore

    def commit_bank_adds(
        self,
        results: List[Dict[str, Any]],
        should_commit: bool,
    ) -> int:
        """Flush bank-add intents stashed in classify_batch results.

        - Always strips the private '_bank_try_add' key from each result dict
          (so it never leaks to FE/DB).
        - If `should_commit`, runs bank.try_add() for each eligible entry in
          parallel across banks (each (recipe, serial, ann_idx) → independent bank).
          cv2.matchTemplate releases the GIL → Python threads are truly parallel.

        Returns count of templates actually added.
        """
        entries: List[Dict[str, Any]] = []
        for r in results:
            if not isinstance(r, dict):
                continue
            entry = r.pop('_bank_try_add', None)
            if entry is None or not should_commit:
                continue
            entries.append(entry)

        if not entries:
            return 0

        registry = self.bank_registry

        def _do_one(entry: Dict[str, Any]) -> bool:
            try:
                bank = registry.get_existing(
                    entry['recipe_id'], entry['serial'], entry['ann_idx']
                )
                if bank is None:
                    return False
                return bank.try_add(entry['target_bgr'], entry['p_ok'])
            except Exception as e:
                logger.warning(f"commit_bank_add (parallel) failed: {e}")
                return False

        # Single-task fast path — avoid pool submit overhead
        if len(entries) == 1:
            n_added = int(_do_one(entries[0]))
        else:
            pool = self._get_commit_pool()
            n_added = sum(1 for r in pool.map(_do_one, entries) if r)

        if n_added:
            logger.info(f"Bank: committed {n_added} new template(s) (parallel, n_tasks={len(entries)})")
        return n_added

    def reset_template_bank(
        self,
        recipe_id: Optional[str] = None,
        camera_serial: Optional[str] = None,
        ann_idx: Optional[int] = None,
    ) -> int:
        """Public passthrough for external callers (e.g. recipe re-load).
        Returns number of banks reset."""
        return self.bank_registry.reset(recipe_id, camera_serial, ann_idx)
