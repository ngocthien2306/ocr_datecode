"""char_quality_v4 — v3 + scale-invariant alignment.

Khác v3:
  1. ECC tries AFFINE (xử lý scale) → EUCLIDEAN → TRANSLATION → identity.
     AFFINE có scale + rotation + translation (6 params).
  2. Safety bounds: scale chỉ accept khi nằm trong [0.7, 1.4].
  3. Sau align, COMPENSATE scale: nếu target bị scale x1.3 để fit template,
     dilate/erode mask để bù stroke width — tránh false LEM cho chữ to.
  4. Track `scale_x, scale_y` trong output để debug.

Defect classification giữ nguyên:
  - over_ink_score > THR_LEM  → 'over_ink' (lem mực)
  - under_ink_score > THR_DROP → 'under_ink' (mất nét)
"""

from typing import Dict, Tuple, Optional

import cv2
import numpy as np

import sys
sys.path.insert(0, '/Users/ngocthien.ai/Source/Projects/ocr_datecode/tools')
from char_quality_v3 import _binarize_norm, _crop_resize, PAD_Y_DEFAULT, PAD_X_DEFAULT  # reuse

SIZE = 64
DILATE_PX = 3
NOISE_FLOOR_OVER  = 0.10
NOISE_FLOOR_UNDER = 0.08
MAX_PENALTY = 0.70
THR_LEM  = 0.25
THR_DROP = 0.20

# AFFINE safety bounds — reject if scale wildly off (likely bad fit)
SCALE_MIN, SCALE_MAX = 0.65, 1.50
SHEAR_MAX = 0.20            # |off-diagonal / diagonal| ratio
TRANS_MAX_RATIO = 0.33      # |tx|, |ty| ≤ size * this


def _decompose_affine(W: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Decompose 2×3 affine into (sx, sy, rot_rad, shear, tx, ty)."""
    a, b, tx = W[0, 0], W[0, 1], W[0, 2]
    c, d, ty = W[1, 0], W[1, 1], W[1, 2]
    sx = float(np.hypot(a, c))
    sy = float(np.hypot(b, d))
    return sx, sy, tx, ty


def _affine_ok(W: np.ndarray, size: int) -> bool:
    """Validate affine warp: scale + translation within sane bounds."""
    if W is None or not np.isfinite(W).all():
        return False
    sx, sy, tx, ty = _decompose_affine(W)
    if not (SCALE_MIN <= sx <= SCALE_MAX): return False
    if not (SCALE_MIN <= sy <= SCALE_MAX): return False
    if abs(tx) > size * TRANS_MAX_RATIO:   return False
    if abs(ty) > size * TRANS_MAX_RATIO:   return False
    # Reject extreme aspect-ratio change (sx vs sy)
    if max(sx, sy) / max(min(sx, sy), 1e-6) > 1.35:
        return False
    return True


def _try_align(t: np.ndarray, g: np.ndarray, size: int):
    """Try AFFINE → EUCLIDEAN → TRANSLATION. Return (warp_or_None, cc, motion_name)."""
    candidates = [
        (cv2.MOTION_AFFINE,      'affine'),
        (cv2.MOTION_EUCLIDEAN,   'euclidean'),
        (cv2.MOTION_TRANSLATION, 'translation'),
    ]
    for motion, name in candidates:
        try:
            warp = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
            cc, warp = cv2.findTransformECC(t, g, warp, motion, criteria, None, 3)
            if motion == cv2.MOTION_AFFINE:
                if not _affine_ok(warp, size):
                    continue
            else:
                if not np.isfinite(warp).all():
                    continue
                if abs(warp[0, 2]) > size * TRANS_MAX_RATIO or abs(warp[1, 2]) > size * TRANS_MAX_RATIO:
                    continue
            return warp, float(cc), name
        except cv2.error:
            continue
    return None, 0.0, 'identity'


def _stroke_width(mask: np.ndarray) -> float:
    """Mean stroke width via distance transform of foreground."""
    if int(np.count_nonzero(mask)) == 0:
        return 0.0
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    return float(dt[mask > 0].mean() * 2.0)


def compute_char_quality_v4(
    tmpl_gray: np.ndarray,
    tgt_gray: np.ndarray,
    size: int = SIZE,
    dilate_px: int = DILATE_PX,
    compensate_scale: bool = True,
    pad_y: int = PAD_Y_DEFAULT,
    pad_x: int = PAD_X_DEFAULT,
    clean_fragments: bool = True,
) -> Dict:
    t_g, t_b = _crop_resize(tmpl_gray, size, pad_y=pad_y, pad_x=pad_x, clean_fragments=clean_fragments)
    g_g, g_b = _crop_resize(tgt_gray,  size, pad_y=pad_y, pad_x=pad_x, clean_fragments=clean_fragments)

    warp, cc, motion = _try_align(t_g, g_g, size)
    sx, sy, tx, ty = 1.0, 1.0, 0.0, 0.0
    if warp is not None:
        sx, sy, tx, ty = _decompose_affine(warp) if motion == 'affine' else (1.0, 1.0, warp[0, 2], warp[1, 2])
        g_g = cv2.warpAffine(
            g_g, warp, (size, size),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        g_b = cv2.warpAffine(
            g_b, warp, (size, size),
            flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    # Stroke-width compensation:
    # Nếu target được scale up (sx,sy > 1) bởi AFFINE để fit template, stroke target
    # cũng "co" lại theo cùng tỉ lệ. Stroke width đo SAU align nên đã được scaled.
    # Nếu vẫn dày hơn → đó là lem mực thật. Nếu mỏng hơn → mất nét thật.
    # Ngược lại, nếu sx,sy ≈ 1 thì AFFINE không bù gì, diff đo trực tiếp.
    sw_t = _stroke_width(t_b)
    sw_g = _stroke_width(g_b)
    sw_ratio = sw_g / max(sw_t, 1e-6)

    # Optional: nếu chỉ size khác (sx far from 1) NHƯNG stroke width sau align gần match
    # → loose tolerance vì user chỉ care lem/mất nét, không care size.
    avg_scale = (sx + sy) / 2 if motion == 'affine' else 1.0
    big_scale_change = abs(avg_scale - 1.0) > 0.15

    # NCC primary similarity
    res = cv2.matchTemplate(g_g, t_g, cv2.TM_CCOEFF_NORMED)
    ncc = float(np.clip(res.max(), 0.0, 1.0))

    # Directional diff với dilate có thể tăng nếu compensate_scale=True và scale change
    eff_dilate = dilate_px + (2 if (compensate_scale and big_scale_change) else 0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * eff_dilate + 1, 2 * eff_dilate + 1))
    t_dil = cv2.dilate(t_b, k)
    g_dil = cv2.dilate(g_b, k)

    extra   = cv2.bitwise_and(g_b, cv2.bitwise_not(t_dil))
    missing = cv2.bitwise_and(t_b, cv2.bitwise_not(g_dil))

    t_px = max(1, int(np.count_nonzero(t_b)))
    over  = min(1.0, int(np.count_nonzero(extra))   / t_px)
    under = min(1.0, int(np.count_nonzero(missing)) / t_px)

    over_eff  = max(0.0, over  - NOISE_FLOOR_OVER)
    under_eff = max(0.0, under - NOISE_FLOOR_UNDER)
    defect_pen = min(MAX_PENALTY, over_eff + under_eff)
    confidence = ncc * (1.0 - defect_pen)

    # Defect classification — combined pixel diff + stroke-width ratio signal
    if over > THR_LEM and over > under:
        defect_type = 'over_ink'
    elif under > THR_DROP and under > over:
        defect_type = 'under_ink'
    elif sw_ratio > 1.45 and not big_scale_change:
        defect_type = 'over_ink'   # stroke dày hơn rõ rệt, không phải scale up
    elif sw_ratio < 0.65 and not big_scale_change:
        defect_type = 'under_ink'  # stroke mỏng hơn rõ rệt
    else:
        defect_type = None

    return {
        'confidence': float(confidence),
        'ncc': float(ncc),
        'ecc_cc': float(cc),
        'motion': motion,
        'scale_x': float(sx),
        'scale_y': float(sy),
        'translation_x': float(tx),
        'translation_y': float(ty),
        'sw_template': float(sw_t),
        'sw_target_aligned': float(sw_g),
        'sw_ratio': float(sw_ratio),
        'over_ink_score':  float(over),
        'under_ink_score': float(under),
        'eff_dilate': int(eff_dilate),
        'defect_type': defect_type,
        '_t_prep': t_g,
        '_g_aligned': g_g,
        '_t_bin': t_b,
        '_g_bin': g_b,
        '_extra_ink': extra,
        '_missing_ink': missing,
    }
