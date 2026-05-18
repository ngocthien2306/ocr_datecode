"""char_quality_v7_shape — Gradient orientation similarity (LineMOD/Halcon insight).

Lấy core insight của Halcon shape-based matching / LineMOD: dùng gradient
ORIENTATION thay vì pixel intensity hay edge. Bỏ qua phần linear-memory /
sliding-window (vì bài toán ta là verify char đã crop sẵn, template ≈ target size).

Pipeline:
  1. Crop+resize template & target về cùng size (reuse v3._crop_resize).
  2. ECC translation alignment để khử shift sub-pixel.
  3. Sobel → magnitude + angle cho cả template & aligned target.
  4. Quantize angle thành 8 bin (paper-style: 360°/16 bin, & 7 → 8 bin
     orientation — bin i và i+8 đồng nghĩa, vì là edge orientation chứ
     không phải gradient direction).
  5. Hysteresis 3×3 voting (NEIGHBOR_THRESHOLD=5 trong 9 pixel) → filter
     pixel có local agreement → loại noise.
  6. Score = mean(cos(bin_diff × 22.5°)) tại pixel có strong gradient ở
     cả 2 ảnh.

Bỏ vs paper:
  - Spread + Response Map (không cần — same-size compare, không sliding)
  - Linear memory (không cần — direct pixel compare)
  - Multi-scale/rotation training (đã có v4 AFFINE)
  - Scattered feature selection (so sánh dense pixel-wise đủ với crop nhỏ)
"""

import sys
sys.path.insert(0, '/Users/ngocthien.ai/Source/Projects/ocr_datecode/tools')

from typing import Dict, Tuple

import cv2
import numpy as np

from char_quality_v3 import (
    _crop_resize,
    SIZE, PAD_Y_DEFAULT, PAD_X_DEFAULT,
)


GRADIENT_THRESHOLD = 30.0   # paper: weak=30, strong=60. Crop nhỏ → giá trị trung bình
GAUSS_KSIZE = 5             # Gaussian blur trước Sobel
SOBEL_KSIZE = 3
NEIGHBOR_THRESHOLD = 5      # paper: ≥5/9 pixel trong 3×3 agree → orientation OK
N_BINS = 8                  # 8 quantized orientation bins (0°, 22.5°, ..., 157.5°)


# Lookup bit→bin: paper encodes bin i as (1<<i). Reverse: log2 via table.
_LUT_BIT_TO_BIN = np.zeros(256, dtype=np.int8)
for _i in range(N_BINS):
    _LUT_BIT_TO_BIN[1 << _i] = _i


def _quantize_orientation(
    angle_deg: np.ndarray,
    magnitude_sq: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Quantize gradient angle → 8 bin. Hysteresis 3×3 majority voting.

    Returns uint8 array where pixel = (1<<bin) if strong + locally consistent,
    else 0. Border (1 px) luôn = 0 (theo paper).
    """
    H, W = angle_deg.shape
    if H < 3 or W < 3:
        return np.zeros((H, W), dtype=np.uint8)

    # Step 1: angle [0,360) → bin 0..15 → mask & 7 → 0..7
    quantized_16 = (angle_deg * 16.0 / 360.0).astype(np.int32) % 16
    quantized_8 = (quantized_16 & 7).astype(np.uint8)  # 0..7

    # Step 2: strong-gradient mask (paper uses squared magnitude with squared threshold)
    strong = magnitude_sq > (threshold * threshold)

    # Step 3: hysteresis voting trên 3×3 — pixel keep nếu ≥5/9 neighbor cùng bin
    # Dùng boxFilter (sum/area). normalize=False → trả về sum trực tiếp.
    out = np.zeros((H, W), dtype=np.uint8)
    for b in range(N_BINS):
        bin_mask_f = (quantized_8 == b).astype(np.float32)
        bin_count = cv2.boxFilter(bin_mask_f, ddepth=cv2.CV_32F, ksize=(3, 3),
                                   normalize=False, borderType=cv2.BORDER_CONSTANT)
        pass_mask = strong & (bin_count >= NEIGHBOR_THRESHOLD) & (quantized_8 == b)
        out[pass_mask] = (1 << b)

    # Border zero-out (paper)
    out[0, :] = 0; out[-1, :] = 0; out[:, 0] = 0; out[:, -1] = 0
    return out


def _compute_quant_orient(img: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (quantized_orientation, magnitude_squared) for image."""
    if img is None or img.size == 0:
        return np.zeros_like(img, dtype=np.uint8), np.zeros(img.shape, dtype=np.float32)
    blurred = cv2.GaussianBlur(img, (GAUSS_KSIZE, GAUSS_KSIZE), 0,
                                borderType=cv2.BORDER_REPLICATE)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=SOBEL_KSIZE, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=SOBEL_KSIZE, borderType=cv2.BORDER_REPLICATE)
    mag_sq = gx * gx + gy * gy
    angle = cv2.phase(gx, gy, angleInDegrees=True)
    quant = _quantize_orientation(angle, mag_sq, threshold)
    return quant, mag_sq


def _orientation_similarity(quant_t: np.ndarray, quant_g: np.ndarray) -> Dict:
    """Compare 2 quantized orientation maps pixel-wise.

    Score per pixel = cos(bin_diff × 22.5°) — smooth similarity in [0, 1].
      - bin diff 0 (cùng orientation) → 1.0
      - bin diff 1 (±22.5°)           → 0.92
      - bin diff 2 (±45°)             → 0.71
      - bin diff 3 (±67.5°)           → 0.38
      - bin diff 4 (±90°)             → 0.0
    Bin diff được clip vào [0, 4] (circular wrap, vì 8 bin orientation).

    Chỉ tính tại pixel có strong gradient ở CẢ template VÀ target.
    """
    valid = (quant_t > 0) & (quant_g > 0)
    n_valid = int(valid.sum())
    if n_valid < 10:
        return {'score': 0.0, 'n_valid': n_valid, 'n_match': 0, 'n_partial': 0}

    bin_t = _LUT_BIT_TO_BIN[quant_t[valid]].astype(np.int16)
    bin_g = _LUT_BIT_TO_BIN[quant_g[valid]].astype(np.int16)

    # Circular distance in 8-bin space (orientation, so max diff = 4)
    d = np.abs(bin_t - bin_g)
    d = np.minimum(d, 8 - d)  # circular wrap

    # cos similarity (smooth)
    cos_sim = np.cos(d.astype(np.float32) * (np.pi / 8.0))
    cos_sim = np.clip(cos_sim, 0.0, 1.0)
    score = float(cos_sim.mean())

    n_match   = int((d == 0).sum())
    n_partial = int((d <= 1).sum())
    return {
        'score': score,
        'n_valid': n_valid,
        'n_match': n_match,
        'n_partial': n_partial,
    }


def compute_char_quality_v7(
    tmpl_gray: np.ndarray,
    tgt_gray: np.ndarray,
    size: int = SIZE,
    pad_y: int = PAD_Y_DEFAULT,
    pad_x: int = PAD_X_DEFAULT,
    grad_threshold: float = GRADIENT_THRESHOLD,
    clean_fragments: bool = True,
) -> Dict:
    """Shape-based (orientation correlation) similarity.

    Returns dict with confidence ∈ [0, 1] + breakdown + visualization maps.
    """
    t_g, t_b = _crop_resize(tmpl_gray, size, pad_y=pad_y, pad_x=pad_x,
                            clean_fragments=clean_fragments)
    g_g, g_b = _crop_resize(tgt_gray, size, pad_y=pad_y, pad_x=pad_x,
                            clean_fragments=clean_fragments)

    # ECC TRANSLATION alignment (sub-pixel)
    cc = 0.0
    motion = 'identity'
    try:
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-3)
        cc, warp = cv2.findTransformECC(t_g, g_g, warp, cv2.MOTION_TRANSLATION,
                                          criteria, None, 3)
        if np.isfinite(warp).all() and abs(warp[0, 2]) <= size / 4 and abs(warp[1, 2]) <= size / 4:
            g_g = cv2.warpAffine(
                g_g, warp, (size, size),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
            motion = 'translation'
        else:
            cc = 0.0
    except cv2.error:
        pass

    quant_t, _ = _compute_quant_orient(t_g, grad_threshold)
    quant_g, _ = _compute_quant_orient(g_g, grad_threshold)

    sim = _orientation_similarity(quant_t, quant_g)
    confidence = sim['score']

    n_t = int((quant_t > 0).sum())
    n_g = int((quant_g > 0).sum())

    # Defect classification (shape-based khó tách lem vs mất nét — dùng coverage hint)
    defect_type = None
    if confidence < 0.60:
        # Strong gradient coverage hint
        if n_t > 20:
            ratio = n_g / n_t
            if ratio < 0.70:
                defect_type = 'under_ink'   # mất nét → ít gradient
            elif ratio > 1.30:
                defect_type = 'over_ink'    # lem → nhiều gradient (extra edges)
            else:
                defect_type = 'shape_mismatch'  # orientation khác mà coverage tương đương → distortion
        else:
            defect_type = 'shape_mismatch'

    return {
        'confidence': float(confidence),
        'orientation_match_pct': sim['n_match'] / max(1, sim['n_valid']),
        'partial_match_pct':     sim['n_partial'] / max(1, sim['n_valid']),
        'n_strong_pixels':       sim['n_valid'],
        'n_strong_template':     n_t,
        'n_strong_target':       n_g,
        'coverage_ratio':        n_g / max(1, n_t),
        'ecc_cc': float(cc),
        'motion': motion,
        'defect_type': defect_type,
        '_t_prep': t_g,
        '_g_aligned': g_g,
        '_quant_t': quant_t,
        '_quant_g': quant_g,
    }


def render_orientation_overlay(t_g: np.ndarray, quant_t: np.ndarray,
                                quant_g: np.ndarray) -> np.ndarray:
    """Color-overlay: green = orientation match, red = mismatch, gray = base."""
    base = cv2.cvtColor(t_g, cv2.COLOR_GRAY2BGR)
    base = (base // 2 + 30).astype(np.uint8)  # dim background

    valid = (quant_t > 0) & (quant_g > 0)
    if not valid.any():
        return base

    bin_t = _LUT_BIT_TO_BIN[quant_t].astype(np.int16)
    bin_g = _LUT_BIT_TO_BIN[quant_g].astype(np.int16)
    d = np.abs(bin_t - bin_g)
    d = np.minimum(d, 8 - d)

    # Green pixels: perfect match (d=0)
    match_mask = valid & (d == 0)
    base[match_mask] = (60, 220, 60)
    # Yellow: partial (d=1)
    partial_mask = valid & (d == 1)
    base[partial_mask] = (60, 200, 200)
    # Red: mismatch (d≥2)
    miss_mask = valid & (d >= 2)
    base[miss_mask] = (60, 60, 220)
    # Blue: only template has gradient, target doesn't (mất nét)
    only_t = (quant_t > 0) & (quant_g == 0)
    base[only_t] = (220, 60, 60)
    return base
