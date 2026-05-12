"""
Project-derived glyph dictionary.

Builds an "average soft glyph mask" per char_id from the project's labeled OK
crops (+ optional imported OK). The output behaves like a font but its shapes
are derived from the actual chars in the user's images instead of a generic
TTF — synthetic OK rendered through this dict ends up visually close to the
real distribution the model will see at inference.

Public API:
    build_glyph_dict(ok_crops_by_char) -> dict[char_id, mask uint8 80x80]
    save_glyph_dict(project_dir, glyphs) -> meta dict
    load_glyph_dict(project_dir) -> dict | None
    load_meta(project_dir) -> dict | None
    load_thumbnails(project_dir) -> dict[char_id, base64 PNG]
"""
import base64
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# Normalized canvas the glyph is averaged on. 80×80 captures shape detail
# without exploding memory: 30 chars × 80×80 × uint8 ≈ 200 KB on disk.
GLYPH_CANVAS = 80
GLYPH_INK_PAD_FRAC = 0.10            # padding around ink bbox after fit
GLYPH_THUMB_SIZE = 48                 # FE preview thumbnail edge (px)
MIN_INK_PIXELS = 5                    # reject crops with too little ink


def _center_normalize_to_canvas(crop: np.ndarray) -> Optional[np.ndarray]:
    """
    Crop's ink bbox → resize keep-aspect → center on (CANVAS, CANVAS).
    Returns uint8 mask where ink ≈ 255, bg = 0. None if ink not detected.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if th.mean() > 127:
        th = cv2.bitwise_not(th)
    ys, xs = np.where(th > 0)
    if len(xs) < MIN_INK_PIXELS:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    ink = th[y0:y1, x0:x1]
    h, w = ink.shape
    if h == 0 or w == 0:
        return None
    pad = int(GLYPH_INK_PAD_FRAC * GLYPH_CANVAS)
    inner = GLYPH_CANVAS - 2 * pad
    scale = inner / max(h, w)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    resized = cv2.resize(ink, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((GLYPH_CANVAS, GLYPH_CANVAS), dtype=np.uint8)
    y_off = (GLYPH_CANVAS - nh) // 2
    x_off = (GLYPH_CANVAS - nw) // 2
    out[y_off:y_off + nh, x_off:x_off + nw] = resized
    return out


def build_glyph_dict(
    ok_crops_by_char: Dict[str, List[np.ndarray]],
    min_samples: int = 1,
) -> Dict[str, np.ndarray]:
    """
    For each char with ≥min_samples OK crops, return the averaged soft glyph
    mask (uint8). Char keys with too few usable samples are skipped — callers
    should fall back to a TTF font for them.
    """
    glyphs: Dict[str, np.ndarray] = {}
    for cid, crops in ok_crops_by_char.items():
        if not cid:
            continue
        normalized: List[np.ndarray] = []
        for crop in crops:
            m = _center_normalize_to_canvas(crop)
            if m is not None:
                normalized.append(m)
        if len(normalized) < min_samples:
            continue
        avg = np.mean(np.stack(normalized, axis=0).astype(np.float32), axis=0)
        glyphs[cid] = np.clip(avg, 0, 255).astype(np.uint8)
    return glyphs


# ─────────────────────────────────────────────── Persist layer

def _dir(project_dir: Path) -> Path:
    return project_dir / "glyphs"


def glyphs_path(project_dir: Path) -> Path:
    return _dir(project_dir) / "glyph_dict.npz"


def meta_path(project_dir: Path) -> Path:
    return _dir(project_dir) / "meta.json"


def thumbnails_path(project_dir: Path) -> Path:
    return _dir(project_dir) / "thumbnails.json"


def _glyph_to_thumb_b64(mask: np.ndarray) -> str:
    """Render glyph as a small black-on-white PNG for FE preview."""
    inv = 255 - mask                          # ink dark on white
    img = cv2.resize(inv, (GLYPH_THUMB_SIZE, GLYPH_THUMB_SIZE),
                     interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ok, png = cv2.imencode('.png', bgr)
    if not ok:
        return ''
    return base64.b64encode(png.tobytes()).decode('ascii')


def save_glyph_dict(project_dir: Path,
                    glyphs: Dict[str, np.ndarray],
                    sample_counts: Optional[Dict[str, int]] = None,
                    ) -> Dict[str, Any]:
    """Persist glyphs + metadata + per-char thumbnails. Returns meta dict."""
    out_dir = _dir(project_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if glyphs:
        np.savez_compressed(glyphs_path(project_dir), **glyphs)
    else:
        # Wipe stale file so reload returns None instead of bogus dict.
        gp = glyphs_path(project_dir)
        if gp.exists():
            gp.unlink()

    thumbs = {cid: _glyph_to_thumb_b64(m) for cid, m in glyphs.items()}
    with open(thumbnails_path(project_dir), 'w') as f:
        json.dump(thumbs, f)

    meta = {
        'built_at':       time.time(),
        'chars_covered':  sorted(glyphs.keys()),
        'count':          len(glyphs),
        'canvas':         GLYPH_CANVAS,
        'sample_counts':  sample_counts or {},
    }
    with open(meta_path(project_dir), 'w') as f:
        json.dump(meta, f)
    return meta


def load_glyph_dict(project_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    p = glyphs_path(project_dir)
    if not p.exists():
        return None
    try:
        npz = np.load(p)
        return {k: npz[k] for k in npz.files}
    except Exception as e:
        logger.warning(f"[project_glyphs] load failed at {p}: {e}")
        return None


def load_meta(project_dir: Path) -> Optional[Dict[str, Any]]:
    p = meta_path(project_dir)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def load_thumbnails(project_dir: Path) -> Dict[str, str]:
    p = thumbnails_path(project_dir)
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}
