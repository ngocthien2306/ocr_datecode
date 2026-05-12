"""
Realistic NG augmentation for ML training.

Mask-aware defects that operate on actual ink regions of a character crop.
Adapted from `ai_generate/generate_ng_samples.py`.

Public API:
    augment_ng(char_img, n, char_id=None, severity_dist=None)
        → List[Tuple[np.ndarray, str]]  parallel to n requested crops

Severity tiers (per crop):
    subtle  : 1 light defect (stroke_thinning / 1 cut / mild tape)
    light   : 1-2 defects
    medium  : 2-3 defects
    heavy   : 2-3 strong defects

Defect types (10):
    cut_horizontal, cut_vertical, segment_removal, dropout_dots, crack,
    block_overlay, local_blob, edge_erosion, tape_overlay, stroke_thinning
"""
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────── Public type tags

NG_AUG_TYPES: Tuple[str, ...] = (
    'cut_horizontal', 'cut_vertical', 'segment_removal',
    'dropout_dots', 'crack', 'block_overlay', 'local_blob',
    'edge_erosion', 'tape_overlay', 'stroke_thinning',
)

DEFAULT_SEVERITY_DIST: Dict[str, float] = {
    'subtle':  0.10,
    'light':   0.50,
    'medium':  0.35,
    'heavy':   0.05,
}

DEFAULT_MIN_CHANGE = 0.015            # reject augmentation if mean diff/255 < this
MAX_RETRY = 3                         # per-sample retry on min-change failure


# ─────────────────────────────────────────────── Mask helpers

def _get_char_mask(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.mean(th) > 127:
        th = cv2.bitwise_not(th)
    return th > 0


def _get_avg_color(img: np.ndarray, mask: np.ndarray):
    if not np.any(mask):
        return (128, 128, 128)
    if img.ndim == 2:
        m = float(img[mask].mean())
        return (int(m), int(m), int(m))
    return tuple(int(c) for c in img[mask].mean(axis=0))


def _get_char_bbox(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _sample_position_on_ink(mask: np.ndarray, rng: random.Random,
                            allow_near_px: int = 2):
    if not np.any(mask):
        return None
    if allow_near_px > 0:
        m = cv2.dilate(mask.astype(np.uint8) * 255,
                       np.ones((3, 3), np.uint8),
                       iterations=allow_near_px)
        ys, xs = np.where(m > 0)
    else:
        ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    idx = rng.randint(0, len(xs) - 1)
    return int(xs[idx]), int(ys[idx])


def _ink_zones_horizontal(mask: np.ndarray):
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) == 0:
        return None
    y_lo, y_hi = int(rows.min()), int(rows.max())
    h = max(2, y_hi - y_lo)
    return {
        'top_edge':  (y_lo,                 y_lo + int(h * 0.20)),
        'top_band':  (y_lo + int(h * 0.15), y_lo + int(h * 0.40)),
        'middle':    (y_lo + int(h * 0.40), y_lo + int(h * 0.60)),
        'bot_band':  (y_lo + int(h * 0.60), y_lo + int(h * 0.85)),
        'bot_edge':  (y_lo + int(h * 0.80), y_hi),
    }


def _ink_zones_vertical(mask: np.ndarray):
    cols = np.where(mask.any(axis=0))[0]
    if len(cols) == 0:
        return None
    x_lo, x_hi = int(cols.min()), int(cols.max())
    w = max(2, x_hi - x_lo)
    return {
        'left_edge':   (x_lo,                 x_lo + int(w * 0.20)),
        'left_band':   (x_lo + int(w * 0.15), x_lo + int(w * 0.40)),
        'middle':      (x_lo + int(w * 0.40), x_lo + int(w * 0.60)),
        'right_band':  (x_lo + int(w * 0.60), x_lo + int(w * 0.85)),
        'right_edge':  (x_lo + int(w * 0.80), x_hi),
    }


# ─────────────────────────────────────────────── Defect functions

def _defect_cut_horizontal(img, params, rng):
    h, w = img.shape[:2]
    mask = _get_char_mask(img)
    bg = _get_avg_color(img, ~mask)
    zones = _ink_zones_horizontal(mask)
    if zones is None:
        return img.copy()
    zy0, zy1 = zones.get(params.get('zone', 'middle'), zones['middle'])
    if zy1 <= zy0 + 1:
        zy1 = zy0 + 2
    band_h = max(2, params.get('band_h', rng.randint(3, 6)))
    cy = rng.randint(zy0, max(zy0, zy1 - 1))
    bp = max(0, cy - band_h // 2)
    out = img.copy()
    out[bp:min(h, bp + band_h), :] = bg
    return out


def _defect_cut_vertical(img, params, rng):
    h, w = img.shape[:2]
    mask = _get_char_mask(img)
    bg = _get_avg_color(img, ~mask)
    zones = _ink_zones_vertical(mask)
    if zones is None:
        return img.copy()
    zx0, zx1 = zones.get(params.get('zone', 'middle'), zones['middle'])
    if zx1 <= zx0 + 1:
        zx1 = zx0 + 2
    band_w = max(2, params.get('band_w', rng.randint(3, 6)))
    cx = rng.randint(zx0, max(zx0, zx1 - 1))
    bp = max(0, cx - band_w // 2)
    out = img.copy()
    out[:, bp:min(w, bp + band_w)] = bg
    return out


def _defect_segment_removal(img, params, rng):
    mask = _get_char_mask(img)
    bg = _get_avg_color(img, ~mask)
    bbox = _get_char_bbox(mask)
    if bbox is None:
        return img.copy()
    x0, y0, x1, y1 = bbox
    cw, ch = max(4, x1 - x0), max(4, y1 - y0)
    frac = params.get('frac', rng.uniform(0.15, 0.30))
    rw = max(3, int(cw * frac))
    rh = max(3, int(ch * frac))
    pos = _sample_position_on_ink(mask, rng, allow_near_px=0)
    if pos is None:
        return img.copy()
    cx, cy = pos
    out = img.copy()
    cv2.ellipse(out, (cx, cy), (rw // 2, rh // 2),
                rng.randint(0, 180), 0, 360, bg, -1)
    return out


def _defect_dropout_dots(img, params, rng):
    mask = _get_char_mask(img)
    bg = _get_avg_color(img, ~mask)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return img.copy()
    n = params.get('n', rng.randint(6, 18))
    out = img.copy()
    for _ in range(n):
        idx = rng.randint(0, len(ys) - 1)
        cv2.circle(out, (int(xs[idx]), int(ys[idx])),
                   rng.randint(1, 3), bg, -1)
    return out


def _defect_crack(img, params, rng):
    h, w = img.shape[:2]
    mask = _get_char_mask(img)
    bg = _get_avg_color(img, ~mask)
    bbox = _get_char_bbox(mask)
    if bbox is None:
        return img.copy()
    x0, y0, x1, y1 = bbox
    out = img.copy()
    thickness = params.get('thickness', rng.choice([1, 1, 2]))
    pad = 4
    n_joints = rng.randint(2, 4)
    pos = _sample_position_on_ink(mask, rng, allow_near_px=0)
    anchor = pos if pos else ((x0 + x1) // 2, (y0 + y1) // 2)

    if rng.random() < 0.5:
        xs_pts = ([max(0, x0 - pad)]
                  + sorted(rng.randint(x0, x1) for _ in range(n_joints - 1))
                  + [min(w - 1, x1 + pad)])
        ys_pts = [anchor[1] + rng.randint(-3, 3) for _ in xs_pts]
        ys_pts[len(ys_pts) // 2] = anchor[1]
    else:
        ys_pts = ([max(0, y0 - pad)]
                  + sorted(rng.randint(y0, y1) for _ in range(n_joints - 1))
                  + [min(h - 1, y1 + pad)])
        xs_pts = [anchor[0] + rng.randint(-3, 3) for _ in ys_pts]
        xs_pts[len(xs_pts) // 2] = anchor[0]

    pts = np.array(list(zip(xs_pts, ys_pts)), dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], False, bg, thickness)
    return out


def _defect_block_overlay(img, params, rng):
    h, w = img.shape[:2]
    mask = _get_char_mask(img)
    bg = _get_avg_color(img, ~mask)
    bbox = _get_char_bbox(mask)
    if bbox is None:
        return img.copy()
    x0, y0, x1, y1 = bbox
    cw = max(4, x1 - x0); ch = max(4, y1 - y0)
    frac = params.get('frac', rng.uniform(0.20, 0.45))
    bw = max(4, int(cw * frac)); bh = max(4, int(ch * frac))
    pos = _sample_position_on_ink(mask, rng, allow_near_px=0)
    if pos is None:
        bx = rng.randint(max(0, x0 - 4), max(x0, min(w - bw, x1 + 4 - bw)))
        by = rng.randint(max(0, y0 - 4), max(y0, min(h - bh, y1 + 4 - bh)))
    else:
        cx_b, cy_b = pos
        bx = max(0, min(w - bw, cx_b - bw // 2))
        by = max(0, min(h - bh, cy_b - bh // 2))
    color = params.get('color', rng.choice([(0, 0, 0), (255, 255, 255), bg]))
    out = img.copy()
    cv2.rectangle(out, (bx, by), (bx + bw, by + bh), color, -1)
    return out


def _defect_local_blob(img, params, rng):
    mask = _get_char_mask(img)
    ink = _get_avg_color(img, mask)
    n = params.get('n', rng.randint(1, 3))
    out = img.copy()
    for _ in range(n):
        pos = _sample_position_on_ink(mask, rng, allow_near_px=2)
        if pos is None:
            continue
        cx, cy = pos
        r = rng.randint(3, 6)
        cv2.circle(out, (cx, cy), r, ink, -1)
        out = cv2.GaussianBlur(out, (3, 3), 0.6)
    return out


def _defect_edge_erosion(img, params, rng):
    """Xoá pixel mực ở đỉnh/đáy ký tự (feathered)."""
    mask = _get_char_mask(img)
    rows = np.any(mask, axis=1)
    if not rows.any():
        return img.copy()

    y_top = int(np.argmax(rows))
    y_bottom = int(len(rows) - np.argmax(rows[::-1]) - 1)
    char_h = y_bottom - y_top + 1

    ratio = params.get('ratio', rng.uniform(0.05, 0.10))
    position = params.get('position', rng.choice(['top', 'bottom', 'both']))
    feather = params.get('feather', True)
    cut_px = max(1, int(char_h * ratio))

    out = img.copy()
    kill = np.zeros(mask.shape, dtype=bool)
    if position in ('top', 'both'):
        kill[y_top:y_top + cut_px, :] = True
    if position in ('bottom', 'both'):
        kill[y_bottom - cut_px + 1:y_bottom + 1, :] = True

    remove = kill & mask
    if (~mask).any():
        bg_color = img[~mask].mean(axis=0) if img.ndim == 3 else img[~mask].mean()
    else:
        bg_color = 255
    out[remove] = bg_color

    if feather:
        alpha = np.ones(mask.shape, dtype=np.float32)
        if position in ('top', 'both'):
            for i, row in enumerate(range(y_top, y_top + cut_px)):
                alpha[row, :] = i / cut_px
        if position in ('bottom', 'both'):
            for i, row in enumerate(range(y_bottom - cut_px + 1, y_bottom + 1)):
                alpha[row, :] = 1 - i / cut_px
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.5)

        if img.ndim == 3:
            bg_arr = np.full_like(img, fill_value=bg_color, dtype=np.float32)
            a = alpha[..., np.newaxis]
        else:
            bg_arr = np.full_like(img, fill_value=bg_color, dtype=np.float32)
            a = alpha
        out = (img.astype(np.float32) * a + bg_arr * (1 - a))
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def _defect_tape_overlay(img, params, rng):
    """Băng keo dán qua ký tự — dải ngang/dọc/chéo có độ trong suốt."""
    h, w = img.shape[:2]
    mask = _get_char_mask(img)
    bbox = _get_char_bbox(mask)
    if bbox is None:
        return img.copy()
    x0, y0, x1, y1 = bbox
    char_h = max(4, y1 - y0)
    char_w = max(4, x1 - x0)

    orientation = params.get('orientation',
                             rng.choice(['horizontal', 'horizontal',
                                         'vertical', 'diagonal']))
    band_frac = params.get('band_frac', rng.uniform(0.18, 0.35))
    alpha = params.get('alpha', rng.uniform(0.55, 0.85))
    color = params.get('color',
                       rng.choice([(230, 230, 230), (210, 210, 215),
                                   (200, 195, 180), (40, 40, 45),
                                   (180, 180, 180)]))

    overlay = img.copy()
    if orientation == 'horizontal':
        band_h = max(4, int(char_h * band_frac))
        pos = _sample_position_on_ink(mask, rng, allow_near_px=0)
        cy = pos[1] if pos else (y0 + y1) // 2
        ty = max(0, cy - band_h // 2)
        cv2.rectangle(overlay, (0, ty), (w, ty + band_h), color, -1)
    elif orientation == 'vertical':
        band_w = max(4, int(char_w * band_frac))
        pos = _sample_position_on_ink(mask, rng, allow_near_px=0)
        cx = pos[0] if pos else (x0 + x1) // 2
        tx = max(0, cx - band_w // 2)
        cv2.rectangle(overlay, (tx, 0), (tx + band_w, h), color, -1)
    else:  # diagonal
        band_h = max(4, int(char_h * band_frac))
        angle = rng.uniform(-35, 35)
        pos = _sample_position_on_ink(mask, rng, allow_near_px=0)
        cx, cy = pos if pos else ((x0 + x1) // 2, (y0 + y1) // 2)
        rect_pts = cv2.boxPoints(((cx, cy), (w * 2, band_h), angle))
        rect_pts = np.int32(rect_pts)
        cv2.fillPoly(overlay, [rect_pts], color)

    out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=rng.uniform(0.3, 0.8))
    return out


def _defect_stroke_thinning(img, params, rng):
    """1-2 đoạn đứt/mờ NHỎ trên nét. Background giữ nguyên."""
    mask = _get_char_mask(img)
    if not mask.any():
        return img.copy()
    if (~mask).any():
        bg_color = img[~mask].mean(axis=0) if img.ndim == 3 else img[~mask].mean()
    else:
        bg_color = 255

    n_breaks = params.get('n_breaks', rng.randint(1, 2))
    break_size = params.get('break_size', rng.randint(3, 7))
    fade_alpha = params.get('fade_alpha', rng.uniform(0.20, 0.50))

    ys, xs = np.where(mask)
    if len(ys) == 0:
        return img.copy()

    break_mask = np.zeros(mask.shape, dtype=np.float32)
    for _ in range(n_breaks):
        idx = rng.randint(0, len(ys) - 1)
        cx, cy = int(xs[idx]), int(ys[idx])
        r = rng.randint(max(2, break_size - 2), break_size + 1)
        cv2.circle(break_mask, (cx, cy), r, 1.0, -1)

    break_mask = cv2.GaussianBlur(break_mask, (0, 0), sigmaX=1.2)
    break_mask = break_mask * mask.astype(np.float32)   # only on ink

    out = img.copy().astype(np.float32)
    bg_arr = np.full_like(out, fill_value=bg_color, dtype=np.float32)
    a = break_mask[..., np.newaxis] if img.ndim == 3 else break_mask
    blend = a * (1 - fade_alpha)
    out = out * (1 - blend) + bg_arr * blend
    return np.clip(out, 0, 255).astype(np.uint8)


_DEFECT_FNS = {
    'cut_horizontal':   _defect_cut_horizontal,
    'cut_vertical':     _defect_cut_vertical,
    'segment_removal':  _defect_segment_removal,
    'dropout_dots':     _defect_dropout_dots,
    'crack':            _defect_crack,
    'block_overlay':    _defect_block_overlay,
    'local_blob':       _defect_local_blob,
    'edge_erosion':     _defect_edge_erosion,
    'tape_overlay':     _defect_tape_overlay,
    'stroke_thinning':  _defect_stroke_thinning,
}

_DEFAULT_WEIGHTS = {
    'cut_horizontal':   1.2, 'cut_vertical':    1.0, 'segment_removal': 0.9,
    'dropout_dots':     0.9, 'crack':           0.8, 'block_overlay':   0.4,
    'local_blob':       0.6, 'edge_erosion':    0.9, 'tape_overlay':    1.0,
    'stroke_thinning':  1.1,
}

# Subtle pool — only one light defect
_SUBTLE_POOL = ['stroke_thinning', 'cut_horizontal', 'cut_vertical', 'tape_overlay']
_SUBTLE_WEIGHTS = {
    'stroke_thinning': 2.0,
    'cut_horizontal':  1.5,
    'cut_vertical':    1.5,
    'tape_overlay':    1.0,
}


# ─────────────────────────────────────────────── Per-char rules

_CHAR_DEFECT_BOOSTS: Dict[str, List[Tuple[str, Optional[str], int]]] = {
    '0': [('cut_horizontal', 'top_edge', 4), ('cut_horizontal', 'bot_edge', 4),
          ('cut_vertical', 'left_edge', 2), ('cut_vertical', 'right_edge', 2),
          ('edge_erosion', None, 2)],
    'O': [('cut_horizontal', 'top_edge', 4), ('cut_horizontal', 'bot_edge', 4),
          ('cut_vertical', 'left_edge', 2), ('cut_vertical', 'right_edge', 2),
          ('edge_erosion', None, 2)],
    'o': [('cut_horizontal', 'top_edge', 3), ('cut_horizontal', 'bot_edge', 3),
          ('edge_erosion', None, 2)],
    'Q': [('cut_horizontal', 'top_edge', 3), ('cut_horizontal', 'bot_edge', 3)],
    '8': [('cut_horizontal', 'middle', 4),
          ('cut_horizontal', 'top_edge', 2), ('cut_horizontal', 'bot_edge', 2),
          ('edge_erosion', None, 2)],
    'B': [('cut_horizontal', 'middle', 4), ('cut_vertical', 'right_edge', 3)],
    'H': [('cut_horizontal', 'middle', 5)],
    'h': [('cut_horizontal', 'top_edge', 3)],
    '6': [('cut_horizontal', 'top_edge', 5), ('edge_erosion', None, 3)],
    '9': [('cut_horizontal', 'bot_edge', 5), ('edge_erosion', None, 3)],
    '5': [('cut_horizontal', 'top_edge', 4)],
    'D': [('cut_vertical', 'left_edge', 4)],
    'd': [('cut_vertical', 'right_edge', 4)],
    'b': [('cut_vertical', 'left_edge', 4)],
    'p': [('cut_vertical', 'left_edge', 4)],
    'q': [('cut_vertical', 'right_edge', 4)],
    'P': [('cut_horizontal', 'middle', 3), ('cut_horizontal', 'bot_edge', 3)],
    'R': [('cut_horizontal', 'middle', 3), ('cut_horizontal', 'bot_edge', 3)],
    'T': [('cut_horizontal', 'top_edge', 5), ('edge_erosion', None, 3)],
    't': [('cut_horizontal', 'top_edge', 4)],
    'L': [('cut_horizontal', 'bot_edge', 5), ('edge_erosion', None, 3)],
    'F': [('cut_horizontal', 'middle', 3)],
    'E': [('cut_horizontal', 'middle', 2)],
    'U': [('cut_horizontal', 'top_edge', 3), ('edge_erosion', None, 2)],
    'u': [('cut_horizontal', 'top_edge', 2), ('edge_erosion', None, 2)],
    'n': [('cut_horizontal', 'bot_edge', 3), ('edge_erosion', None, 2)],
    'C': [('cut_vertical', 'right_edge', 3)],
    'c': [('cut_vertical', 'right_edge', 2)],
    'S': [('cut_horizontal', 'top_edge', 3), ('cut_horizontal', 'bot_edge', 3)],
    's': [('cut_horizontal', 'top_edge', 2), ('cut_horizontal', 'bot_edge', 2)],
    'a': [('cut_horizontal', 'top_edge', 2), ('cut_vertical', 'right_edge', 2)],
    'g': [('cut_horizontal', 'bot_edge', 3)],
    'e': [('cut_horizontal', 'middle', 3)],
    '3': [('cut_horizontal', 'middle', 3)],
    'J': [('cut_horizontal', 'top_edge', 3)],
    'j': [('cut_horizontal', 'top_edge', 2)],
}

_BULLETPROOF = set('KMWXYAVNZ47kmwxz')   # less defect-prone in real prints
_FRAGILE     = set('Iil1.,;:!')          # only light defects make sense
_FRAGILE_LIGHT_ONLY = {'dropout_dots', 'crack', 'local_blob',
                       'edge_erosion', 'stroke_thinning'}


# ─────────────────────────────────────────────── Defect picker

def _pick_subtle_defect(rng: random.Random,
                        enabled: Optional[set] = None) -> List[Tuple[str, Dict[str, Any]]]:
    pool = _SUBTLE_POOL if enabled is None else [d for d in _SUBTLE_POOL if d in enabled]
    if not pool:
        # User disabled every subtle-eligible defect → fall back to the enabled
        # set so the caller still gets a single defect to apply.
        pool = list(enabled) if enabled else list(_SUBTLE_POOL)
    weights = [_SUBTLE_WEIGHTS.get(d, 1.0) for d in pool]
    defect = rng.choices(pool, weights=weights, k=1)[0]
    params: Dict[str, Any] = {}
    if defect == 'stroke_thinning':
        params['n_breaks'] = 1
        params['break_size'] = rng.randint(3, 5)
        params['fade_alpha'] = rng.uniform(0.30, 0.50)
    elif defect == 'cut_horizontal':
        params['zone'] = rng.choice(['top_edge', 'middle', 'bot_edge'])
        params['band_h'] = rng.randint(2, 4)
    elif defect == 'cut_vertical':
        params['zone'] = rng.choice(['left_edge', 'middle', 'right_edge'])
        params['band_w'] = rng.randint(2, 4)
    elif defect == 'tape_overlay':
        params['band_frac'] = rng.uniform(0.10, 0.18)
        params['alpha'] = rng.uniform(0.40, 0.55)
    return [(defect, params)]


def _build_defect_params(defect: str, severity: str,
                         rng: random.Random) -> Dict[str, Any]:
    """Generate per-severity params for a single defect type. Mirrors the
    inline blocks inside `_pick_defects` so forced-defect previews use the
    same parameter distributions as the regular path."""
    params: Dict[str, Any] = {}
    if defect in ('cut_horizontal', 'cut_vertical'):
        zones = (['top_edge', 'top_band', 'middle', 'bot_band', 'bot_edge']
                 if defect == 'cut_horizontal' else
                 ['left_edge', 'left_band', 'middle', 'right_band', 'right_edge'])
        params['zone'] = rng.choice(zones)

    if severity == 'subtle':
        if defect == 'stroke_thinning':
            params['n_breaks'] = 1
            params['break_size'] = rng.randint(3, 5)
            params['fade_alpha'] = rng.uniform(0.30, 0.50)
        elif defect == 'cut_horizontal':
            params['band_h'] = rng.randint(2, 4)
        elif defect == 'cut_vertical':
            params['band_w'] = rng.randint(2, 4)
        elif defect == 'tape_overlay':
            params['band_frac'] = rng.uniform(0.10, 0.18)
            params['alpha'] = rng.uniform(0.40, 0.55)
    elif severity == 'light':
        if defect.startswith('cut_'):
            key = 'band_h' if defect.endswith('horizontal') else 'band_w'
            params[key] = rng.randint(3, 4)
        elif defect == 'block_overlay':
            params['frac'] = rng.uniform(0.10, 0.18)
        elif defect == 'segment_removal':
            params['frac'] = rng.uniform(0.10, 0.18)
        elif defect == 'dropout_dots':
            params['n'] = rng.randint(4, 8)
        elif defect == 'local_blob':
            params['n'] = rng.randint(1, 2)
        elif defect == 'crack':
            params['thickness'] = 1
        elif defect == 'edge_erosion':
            params['ratio'] = rng.uniform(0.04, 0.07)
            params['position'] = rng.choice(['top', 'bottom'])
        elif defect == 'tape_overlay':
            params['band_frac'] = rng.uniform(0.12, 0.20)
            params['alpha'] = rng.uniform(0.45, 0.65)
        elif defect == 'stroke_thinning':
            params['n_breaks'] = 1
            params['break_size'] = rng.randint(3, 5)
            params['fade_alpha'] = rng.uniform(0.35, 0.55)
    elif severity == 'medium':
        if defect.startswith('cut_'):
            key = 'band_h' if defect.endswith('horizontal') else 'band_w'
            params[key] = rng.randint(4, 6)
        elif defect == 'block_overlay':
            params['frac'] = rng.uniform(0.20, 0.35)
        elif defect == 'segment_removal':
            params['frac'] = rng.uniform(0.18, 0.28)
        elif defect == 'dropout_dots':
            params['n'] = rng.randint(8, 14)
        elif defect == 'local_blob':
            params['n'] = rng.randint(2, 3)
        elif defect == 'edge_erosion':
            params['ratio'] = rng.uniform(0.07, 0.12)
            params['position'] = rng.choice(['top', 'bottom', 'both'])
        elif defect == 'tape_overlay':
            params['band_frac'] = rng.uniform(0.20, 0.30)
            params['alpha'] = rng.uniform(0.60, 0.80)
        elif defect == 'stroke_thinning':
            params['n_breaks'] = rng.randint(1, 2)
            params['break_size'] = rng.randint(4, 6)
            params['fade_alpha'] = rng.uniform(0.25, 0.45)
    else:  # heavy
        if defect.startswith('cut_'):
            key = 'band_h' if defect.endswith('horizontal') else 'band_w'
            params[key] = rng.randint(5, 8)
        elif defect == 'block_overlay':
            params['frac'] = rng.uniform(0.30, 0.45)
        elif defect == 'segment_removal':
            params['frac'] = rng.uniform(0.25, 0.38)
        elif defect == 'dropout_dots':
            params['n'] = rng.randint(14, 22)
        elif defect == 'local_blob':
            params['n'] = rng.randint(2, 4)
        elif defect == 'crack':
            params['thickness'] = rng.choice([1, 2])
        elif defect == 'edge_erosion':
            params['ratio'] = rng.uniform(0.12, 0.18)
            params['position'] = 'both'
        elif defect == 'tape_overlay':
            params['band_frac'] = rng.uniform(0.28, 0.40)
            params['alpha'] = rng.uniform(0.70, 0.88)
        elif defect == 'stroke_thinning':
            params['n_breaks'] = 2
            params['break_size'] = rng.randint(5, 7)
            params['fade_alpha'] = rng.uniform(0.15, 0.35)
    return params


def _pick_defects(char: Optional[str], severity: str, rng: random.Random,
                  edge_cut_prob: float = 0.40,
                  force_defect_type: Optional[str] = None,
                  enabled_defect_types: Optional[List[str]] = None) -> List[Tuple[str, Dict[str, Any]]]:
    if force_defect_type:
        if force_defect_type not in NG_AUG_TYPES:
            raise ValueError(f"Unknown defect type: {force_defect_type}")
        return [(force_defect_type, _build_defect_params(force_defect_type, severity, rng))]

    # Whitelist filter — None means "all enabled" (legacy behavior).
    enabled_set: Optional[set] = None
    if enabled_defect_types is not None:
        enabled_set = {d for d in enabled_defect_types if d in NG_AUG_TYPES}
        if not enabled_set:
            # Empty whitelist would cause an unrecoverable loop; treat as "no
            # restriction" so training still produces NG samples.
            enabled_set = None

    if severity == 'subtle':
        return _pick_subtle_defect(rng, enabled=enabled_set)

    weights = dict(_DEFAULT_WEIGHTS)
    boosts: List[Tuple[str, Optional[str], int]] = []
    if char and char in _CHAR_DEFECT_BOOSTS:
        boosts = list(_CHAR_DEFECT_BOOSTS[char])
    if char and char in _BULLETPROOF:
        weights['cut_horizontal'] *= 0.5
        weights['cut_vertical']   *= 0.5
        weights['segment_removal'] *= 0.7
    if char and char in _FRAGILE:
        weights = {k: v for k, v in weights.items() if k in _FRAGILE_LIGHT_ONLY}
    if enabled_set is not None:
        weights = {k: v for k, v in weights.items() if k in enabled_set}
        boosts = [b for b in boosts if b[0] in enabled_set]
        if not weights:
            # Char-specific filters wiped out the enabled set — fall back to
            # the enabled set with uniform weights so we still produce a defect.
            weights = {d: 1.0 for d in enabled_set}

    n_defects = rng.randint(1, 2) if severity == 'light' else rng.randint(2, 3)
    use_edge_cut = bool(boosts) and rng.random() < edge_cut_prob

    picks: List[Tuple[str, Dict[str, Any]]] = []
    if use_edge_cut:
        bws = [b[2] for b in boosts]
        chosen = rng.choices(range(len(boosts)), weights=bws, k=1)[0]
        defect, zone, _ = boosts[chosen]
        bp: Dict[str, Any] = {}
        if zone is not None:
            bp['zone'] = zone
        if defect.startswith('cut_'):
            key = 'band_h' if defect == 'cut_horizontal' else 'band_w'
            if severity == 'light':
                bp[key] = rng.randint(3, 4)
            elif severity == 'heavy':
                bp[key] = rng.randint(5, 9)
        elif defect == 'edge_erosion':
            if severity == 'light':
                bp['ratio'] = rng.uniform(0.04, 0.07)
            elif severity == 'heavy':
                bp['ratio'] = rng.uniform(0.12, 0.20)
                bp['position'] = 'both'
        picks.append((defect, bp))
        n_defects -= 1

    pool = [d for d in weights.keys() if d not in {p[0] for p in picks}]
    pool_w = [weights[d] for d in pool]
    while n_defects > 0 and pool:
        idx = rng.choices(range(len(pool)), weights=pool_w, k=1)[0]
        defect = pool.pop(idx); pool_w.pop(idx)
        params: Dict[str, Any] = {}
        if defect in ('cut_horizontal', 'cut_vertical'):
            zones = (['top_edge', 'top_band', 'middle', 'bot_band', 'bot_edge']
                     if defect == 'cut_horizontal' else
                     ['left_edge', 'left_band', 'middle', 'right_band', 'right_edge'])
            params['zone'] = rng.choice(zones)

        if severity == 'light':
            if defect.startswith('cut_'):
                key = 'band_h' if defect.endswith('horizontal') else 'band_w'
                params[key] = rng.randint(3, 4)
            elif defect == 'block_overlay':
                params['frac'] = rng.uniform(0.10, 0.18)
            elif defect == 'segment_removal':
                params['frac'] = rng.uniform(0.10, 0.18)
            elif defect == 'dropout_dots':
                params['n'] = rng.randint(4, 8)
            elif defect == 'local_blob':
                params['n'] = rng.randint(1, 2)
            elif defect == 'crack':
                params['thickness'] = 1
            elif defect == 'edge_erosion':
                params['ratio'] = rng.uniform(0.04, 0.07)
                params['position'] = rng.choice(['top', 'bottom'])
            elif defect == 'tape_overlay':
                params['band_frac'] = rng.uniform(0.12, 0.20)
                params['alpha'] = rng.uniform(0.45, 0.65)
            elif defect == 'stroke_thinning':
                params['n_breaks'] = 1
                params['break_size'] = rng.randint(3, 5)
                params['fade_alpha'] = rng.uniform(0.35, 0.55)
        elif severity == 'medium':
            if defect.startswith('cut_'):
                key = 'band_h' if defect.endswith('horizontal') else 'band_w'
                params[key] = rng.randint(4, 6)
            elif defect == 'block_overlay':
                params['frac'] = rng.uniform(0.20, 0.35)
            elif defect == 'segment_removal':
                params['frac'] = rng.uniform(0.18, 0.28)
            elif defect == 'dropout_dots':
                params['n'] = rng.randint(8, 14)
            elif defect == 'local_blob':
                params['n'] = rng.randint(2, 3)
            elif defect == 'edge_erosion':
                params['ratio'] = rng.uniform(0.07, 0.12)
                params['position'] = rng.choice(['top', 'bottom', 'both'])
            elif defect == 'tape_overlay':
                params['band_frac'] = rng.uniform(0.20, 0.30)
                params['alpha'] = rng.uniform(0.60, 0.80)
            elif defect == 'stroke_thinning':
                params['n_breaks'] = rng.randint(1, 2)
                params['break_size'] = rng.randint(4, 6)
                params['fade_alpha'] = rng.uniform(0.25, 0.45)
        else:  # heavy
            if defect.startswith('cut_'):
                key = 'band_h' if defect.endswith('horizontal') else 'band_w'
                params[key] = rng.randint(5, 8)
            elif defect == 'block_overlay':
                params['frac'] = rng.uniform(0.30, 0.45)
            elif defect == 'segment_removal':
                params['frac'] = rng.uniform(0.25, 0.38)
            elif defect == 'dropout_dots':
                params['n'] = rng.randint(14, 22)
            elif defect == 'local_blob':
                params['n'] = rng.randint(2, 4)
            elif defect == 'crack':
                params['thickness'] = rng.choice([1, 2])
            elif defect == 'edge_erosion':
                params['ratio'] = rng.uniform(0.12, 0.18)
                params['position'] = 'both'
            elif defect == 'tape_overlay':
                params['band_frac'] = rng.uniform(0.28, 0.40)
                params['alpha'] = rng.uniform(0.70, 0.88)
            elif defect == 'stroke_thinning':
                params['n_breaks'] = 2
                params['break_size'] = rng.randint(5, 7)
                params['fade_alpha'] = rng.uniform(0.15, 0.35)
        picks.append((defect, params))
        n_defects -= 1
    return picks


def _apply_defects(img: np.ndarray, defects, rng: random.Random):
    out = img.copy()
    primary = defects[0][0] if defects else 'unknown'
    for name, params in defects:
        fn = _DEFECT_FNS.get(name)
        if fn is None:
            continue
        try:
            out = fn(out, params, rng)
        except Exception as e:
            logger.warning(f"NG defect '{name}' failed: {e}")
            continue
    return out, primary


def _change_ratio(orig: np.ndarray, defected: np.ndarray) -> float:
    diff = np.abs(orig.astype(np.int16) - defected.astype(np.int16))
    return float(diff.mean()) / 255.0


# ─────────────────────────────────────────────── Public API

def _normalize_severity_dist(d: Optional[Dict[str, float]]) -> Dict[str, float]:
    if not d:
        return DEFAULT_SEVERITY_DIST.copy()
    out = {k: max(0.0, float(d.get(k, 0.0))) for k in ('subtle', 'light', 'medium', 'heavy')}
    s = sum(out.values())
    if s <= 0:
        return DEFAULT_SEVERITY_DIST.copy()
    return {k: v / s for k, v in out.items()}


def _pick_severity(rng: random.Random, dist: Dict[str, float]) -> str:
    r = rng.random()
    cum = 0.0
    for sev in ('subtle', 'light', 'medium', 'heavy'):
        cum += dist.get(sev, 0.0)
        if r <= cum:
            return sev
    return 'medium'


def augment_ng(
    char_img: np.ndarray,
    n: int = 5,
    char_id: Optional[str] = None,
    severity_dist: Optional[Dict[str, float]] = None,
    min_change: float = DEFAULT_MIN_CHANGE,
    rng: Optional[random.Random] = None,
    force_defect_type: Optional[str] = None,
    enabled_defect_types: Optional[List[str]] = None,
) -> List[Tuple[np.ndarray, str]]:
    """
    Generate `n` realistic NG variants of `char_img`.

    Returns a list of (aug_img, primary_defect_type) parallel to n requested.
    Defects pick severity from `severity_dist` (default: 10/50/35/5 for
    subtle/light/medium/heavy). Per-char boost rules apply when `char_id`
    matches an entry in CHAR_DEFECT_BOOSTS.

    Each augmentation is retried up to MAX_RETRY times if `_change_ratio` falls
    below `min_change` — guarantees defects are visually noticeable.
    """
    if rng is None:
        rng = random.Random()
    dist = _normalize_severity_dist(severity_dist)

    # Convert grayscale → BGR so colored defects (tape_overlay) work
    if char_img.ndim == 2:
        base = cv2.cvtColor(char_img, cv2.COLOR_GRAY2BGR)
    elif char_img.ndim == 3 and char_img.shape[2] == 1:
        base = cv2.cvtColor(char_img, cv2.COLOR_GRAY2BGR)
    else:
        base = char_img

    out: List[Tuple[np.ndarray, str]] = []
    for _ in range(n):
        for attempt in range(MAX_RETRY):
            severity = _pick_severity(rng, dist)
            defects = _pick_defects(char_id, severity, rng,
                                    force_defect_type=force_defect_type,
                                    enabled_defect_types=enabled_defect_types)
            aug, primary = _apply_defects(base, defects, rng)
            if _change_ratio(base, aug) >= min_change:
                out.append((aug, primary))
                break
        else:
            # All retries failed min-change → keep last attempt anyway
            out.append((aug, primary))
    return out
