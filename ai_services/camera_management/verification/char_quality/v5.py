"""char_quality_v5 — v4 + tile-wise local defect detection.

Khác v4:
  - Sau khi AFFINE align, chia mask thành lưới n_tiles × n_tiles (default 3×3).
  - Compute per-tile over_ink / under_ink.
  - Aggregate:
      over_max   = max over toàn các tile (surface defect cục bộ)
      under_max  = max under toàn các tile
      over_avg   = weighted avg (theo template ink/tile)
      under_avg  = weighted avg
      n_bad_tiles = số tile có over hoặc under > local threshold
  - Defect flag:
      'over_ink'  nếu over_max > THR_LEM_TILE   (cục bộ) hoặc over_avg > THR_LEM_GLOBAL
      'under_ink' nếu under_max > THR_DROP_TILE hoặc under_avg > THR_DROP_GLOBAL
  - Confidence kết hợp NCC + (1 - global penalty) + (penalty hiệu chỉnh từ tile_max).

Output thêm:
  - `tile_over` / `tile_under` (n_tiles, n_tiles) — for heatmap viz
  - `defect_location` = (i, j) tile có defect nặng nhất, hoặc None
"""

from typing import Dict, Tuple, Optional

import cv2
import numpy as np

# import sys (not needed in package)
# (path injection removed — using relative imports below)
from .char_quality_v4 import (
    _crop_resize, _try_align, _decompose_affine, _stroke_width,
    SIZE, DILATE_PX, NOISE_FLOOR_OVER, NOISE_FLOOR_UNDER,
    MAX_PENALTY, TRANS_MAX_RATIO, PAD_Y_DEFAULT, PAD_X_DEFAULT,
)

N_TILES_DEFAULT = 3              # 3×3 = 9 tile
MIN_INK_PER_TILE = 30            # template ink threshold (skip corner tiles không stable)
TILE_DILATE_PX = 4               # dilate lớn hơn global → tha thứ alignment lệch trong tile
THR_LEM_TILE   = 0.45            # over tại 1 tile để flag lem cục bộ
THR_DROP_TILE  = 0.40            # under tại 1 tile để flag mất nét cục bộ
THR_LEM_GLOBAL  = 0.20           # over toàn cục
THR_DROP_GLOBAL = 0.15           # under toàn cục
BAD_TILE_COUNT_THR = 2           # ≥ này tile bad → flag


def _compute_tile_scores(
    t_b: np.ndarray, g_b: np.ndarray,
    n_tiles: int, min_ink: int, dilate_px: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chia thành n×n tile, compute over/under per-tile với dilate riêng (loose hơn global).
    Returns (tile_over, tile_under, valid_mask). Shape (n, n).

    Per-tile diff dùng dilate_px lớn hơn global vì tile boundary nhạy alignment.
    """
    H, W = t_b.shape
    # Per-tile diff with larger dilate to absorb alignment jitter at tile edges
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    t_dil_tile = cv2.dilate(t_b, k)
    g_dil_tile = cv2.dilate(g_b, k)
    extra_tile   = cv2.bitwise_and(g_b, cv2.bitwise_not(t_dil_tile))
    missing_tile = cv2.bitwise_and(t_b, cv2.bitwise_not(g_dil_tile))

    tile_over  = np.zeros((n_tiles, n_tiles), dtype=np.float32)
    tile_under = np.zeros((n_tiles, n_tiles), dtype=np.float32)
    valid      = np.zeros((n_tiles, n_tiles), dtype=bool)

    for i in range(n_tiles):
        y0 = i * H // n_tiles
        y1 = (i + 1) * H // n_tiles if i < n_tiles - 1 else H
        for j in range(n_tiles):
            x0 = j * W // n_tiles
            x1 = (j + 1) * W // n_tiles if j < n_tiles - 1 else W

            t_tile = t_b[y0:y1, x0:x1]
            g_tile = g_b[y0:y1, x0:x1]
            extra_t   = extra_tile[y0:y1, x0:x1]
            missing_t = missing_tile[y0:y1, x0:x1]

            t_px = int(np.count_nonzero(t_tile))
            g_px = int(np.count_nonzero(g_tile))

            if t_px < min_ink and g_px < min_ink:
                # Cả 2 đều rỗng → no signal
                continue
            if t_px < min_ink:
                if g_px >= min_ink * 2:  # target có nhiều ink ở tile mà template không có
                    tile_over[i, j] = 1.0
                    valid[i, j] = True
                # Else: target chỉ có vài pixel ở góc → noise, skip
                continue
            tile_over[i, j]  = min(1.0, int(np.count_nonzero(extra_t))   / t_px)
            tile_under[i, j] = min(1.0, int(np.count_nonzero(missing_t)) / t_px)
            valid[i, j] = True
    return tile_over, tile_under, valid


def _aggregate_tile(tile_over, tile_under, valid):
    """Aggregate tile arrays into scalar metrics."""
    if not valid.any():
        return {
            'over_max': 0.0, 'over_avg': 0.0,
            'under_max': 0.0, 'under_avg': 0.0,
            'n_bad_tiles': 0,
            'defect_location': None,
        }
    over_max  = float(tile_over[valid].max())
    under_max = float(tile_under[valid].max())
    over_avg  = float(tile_over[valid].mean())
    under_avg = float(tile_under[valid].mean())
    bad = (tile_over > THR_LEM_TILE) | (tile_under > THR_DROP_TILE)
    bad &= valid
    n_bad = int(bad.sum())

    # Defect location: chọn tile có (over+under) lớn nhất
    combined = tile_over + tile_under
    combined[~valid] = -1
    if combined.max() > 0:
        flat = int(combined.argmax())
        defect_loc = (flat // tile_over.shape[1], flat % tile_over.shape[1])
    else:
        defect_loc = None

    return {
        'over_max': over_max, 'over_avg': over_avg,
        'under_max': under_max, 'under_avg': under_avg,
        'n_bad_tiles': n_bad,
        'defect_location': defect_loc,
    }


def _render_tile_heatmap(t_b: np.ndarray, tile_over, tile_under, valid,
                         scale: int = 2) -> np.ndarray:
    """Render N×N heatmap overlay on template mask.
    Red intensity = over_ink, Blue intensity = under_ink. White=valid tile boundary."""
    H, W = t_b.shape
    base = cv2.cvtColor(t_b, cv2.COLOR_GRAY2BGR)
    base[t_b > 0] = (140, 140, 140)
    n = tile_over.shape[0]
    for i in range(n):
        y0 = i * H // n
        y1 = (i + 1) * H // n if i < n - 1 else H
        for j in range(n):
            x0 = j * W // n
            x1 = (j + 1) * W // n if j < n - 1 else W
            o = tile_over[i, j]
            u = tile_under[i, j]
            # Tile color: blend red(over) + blue(under) on the foreground area
            r = int(min(255, 255 * o))
            b = int(min(255, 255 * u))
            if valid[i, j]:
                # Tint background of tile
                tint = base[y0:y1, x0:x1].astype(np.int32)
                tint[..., 2] += int(r * 0.6)    # R
                tint[..., 0] += int(b * 0.6)    # B (BGR)
                base[y0:y1, x0:x1] = np.clip(tint, 0, 255).astype(np.uint8)
            # Tile grid lines
            cv2.rectangle(base, (x0, y0), (x1 - 1, y1 - 1), (60, 60, 60), 1)
            if valid[i, j] and (o > 0.05 or u > 0.05):
                txt = f"{o:.2f}/{u:.2f}"
                cv2.putText(base, txt, (x0 + 2, y0 + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
    return base


def compute_char_quality_v5(
    tmpl_gray: np.ndarray,
    tgt_gray: np.ndarray,
    size: int = SIZE,
    dilate_px: int = DILATE_PX,
    n_tiles: int = N_TILES_DEFAULT,
    pad_y: int = PAD_Y_DEFAULT,
    pad_x: int = PAD_X_DEFAULT,
    clean_fragments: bool = True,
) -> Dict:
    t_g, t_b = _crop_resize(tmpl_gray, size, pad_y=pad_y, pad_x=pad_x, clean_fragments=clean_fragments)
    g_g, g_b = _crop_resize(tgt_gray,  size, pad_y=pad_y, pad_x=pad_x, clean_fragments=clean_fragments)

    # AFFINE align (reuse v4 logic)
    warp, cc, motion = _try_align(t_g, g_g, size)
    sx, sy, tx, ty = 1.0, 1.0, 0.0, 0.0
    if warp is not None:
        sx, sy, tx, ty = _decompose_affine(warp) if motion == 'affine' else (1.0, 1.0, warp[0, 2], warp[1, 2])
        g_g = cv2.warpAffine(g_g, warp, (size, size),
                             flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REPLICATE)
        g_b = cv2.warpAffine(g_b, warp, (size, size),
                             flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # NCC primary similarity
    res = cv2.matchTemplate(g_g, t_g, cv2.TM_CCOEFF_NORMED)
    ncc = float(np.clip(res.max(), 0.0, 1.0))

    # Global directional diff (giống v3/v4)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    t_dil = cv2.dilate(t_b, k)
    g_dil = cv2.dilate(g_b, k)
    extra   = cv2.bitwise_and(g_b, cv2.bitwise_not(t_dil))
    missing = cv2.bitwise_and(t_b, cv2.bitwise_not(g_dil))

    t_px = max(1, int(np.count_nonzero(t_b)))
    over_g  = min(1.0, int(np.count_nonzero(extra))   / t_px)
    under_g = min(1.0, int(np.count_nonzero(missing)) / t_px)

    # Tile-wise — TÍN HIỆU PHỤ, không vào công thức confidence
    tile_over, tile_under, valid = _compute_tile_scores(
        t_b, g_b, n_tiles, MIN_INK_PER_TILE, TILE_DILATE_PX
    )
    agg = _aggregate_tile(tile_over, tile_under, valid)

    # Stroke width signal (giống v4)
    sw_t = _stroke_width(t_b)
    sw_g = _stroke_width(g_b)
    sw_ratio = sw_g / max(sw_t, 1e-6)

    # CONFIDENCE: chỉ dùng global penalty (như v4) → giữ ổn định.
    # Tile signal chỉ dùng để FLAG defect, không kéo conf xuống.
    over_eff  = max(0.0, over_g  - NOISE_FLOOR_OVER)
    under_eff = max(0.0, under_g - NOISE_FLOOR_UNDER)
    defect_pen = min(MAX_PENALTY, over_eff + under_eff)
    confidence = ncc * (1.0 - defect_pen)

    # Defect classification — tile-aware
    n_bad = agg['n_bad_tiles']
    if (agg['over_max'] > THR_LEM_TILE and agg['over_max'] > agg['under_max']) or \
       (over_g > THR_LEM_GLOBAL and over_g > under_g):
        defect_type = 'over_ink'
    elif (agg['under_max'] > THR_DROP_TILE and agg['under_max'] > agg['over_max']) or \
         (under_g > THR_DROP_GLOBAL and under_g > over_g):
        defect_type = 'under_ink'
    elif n_bad >= BAD_TILE_COUNT_THR:
        # nhiều tile bị bad nhưng max chưa đủ → dạng defect rải rác
        defect_type = 'over_ink' if agg['over_avg'] > agg['under_avg'] else 'under_ink'
    else:
        defect_type = None

    heatmap = _render_tile_heatmap(t_b, tile_over, tile_under, valid)

    return {
        'confidence': float(confidence),
        'ncc': float(ncc),
        'ecc_cc': float(cc),
        'motion': motion,
        'scale_x': float(sx), 'scale_y': float(sy),
        # Global diff
        'over_ink_score':  float(over_g),
        'under_ink_score': float(under_g),
        # Tile-wise
        'tile_over_max':  float(agg['over_max']),
        'tile_under_max': float(agg['under_max']),
        'tile_over_avg':  float(agg['over_avg']),
        'tile_under_avg': float(agg['under_avg']),
        'n_bad_tiles':    int(agg['n_bad_tiles']),
        'defect_location': agg['defect_location'],
        # Stroke width
        'sw_template': float(sw_t),
        'sw_target_aligned': float(sw_g),
        'sw_ratio': float(sw_ratio),
        'defect_type': defect_type,
        # Visualization
        '_t_prep': t_g,
        '_g_aligned': g_g,
        '_t_bin': t_b,
        '_g_bin': g_b,
        '_extra_ink': extra,
        '_missing_ink': missing,
        '_tile_heatmap': heatmap,
        '_tile_over_arr':  tile_over,
        '_tile_under_arr': tile_under,
    }
