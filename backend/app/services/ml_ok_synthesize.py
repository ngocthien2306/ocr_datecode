"""
Synthetic OK augmentation for ML training.

Pipeline (PIL render → composite on real BG → camera noise):
  1. Style fingerprint   — extract ink/bg color, blur σ, noise σ from REAL OK crops
  2. BG harvesting       — inpaint char out → clean BG (per-char stratified pool)
  3. Render char alpha   — PIL textbbox + binary search font size, optional rotation
  4. Composite           — bg * (1−α) + ink * α (soft alpha 0.6 sigma)
  5. Camera noise        — chromatic aberration + shot/read/edge/row noise + vignette
  6. Validate            — contrast ≥ 20, fill ∈ [10%, 65%], retry x4

Adapted from `ai_generate/generate_missing_chars.py`. Public API:

    synthesize_ok_from_annotations(annotations, images_dir, target_n_per_char,
                                    only_below_threshold=True, char_filter=None)
        → List[Dict[str, Any]]

Cache (process-local, keyed by project root path) for style + BG pool — first
training/preview is slow (~few seconds), subsequent calls reuse.
"""
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.models.ml_training import MLAnnotationInDB
from app.services.ml_segment_service import crop_segment

logger = logging.getLogger(__name__)


# Special font path that signals "use the project-derived glyph dictionary"
# instead of a real TTF. Callers pass this in `font_paths` to mix project
# glyphs with regular fonts (random pick per crop). See `ml_project_glyphs`.
PROJECT_GLYPHS_TOKEN = "__project_glyphs__"


# ─────────────────────────────────────────────── Font discovery

_BACKEND_DIR = Path(__file__).parent.parent.parent.parent
_FONT_DIR = _BACKEND_DIR / "weights/fonts"

_SYSTEM_FONT_FALLBACKS = [
    '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
    '/System/Library/Fonts/Supplemental/Tahoma Bold.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    'C:/Windows/Fonts/arialbd.ttf',
]

STROKE_RATIO_MIN = 0.08          # regular weights ≈ 0.10-0.13
STROKE_RATIO_MAX = 0.22          # extra-bold ≈ 0.20-0.22 (typical datecode prints)


def _measure_stroke_ratio(font_path: str, glyph: str = '8',
                          size: int = 80) -> Optional[float]:
    """Estimate stroke thickness ratio (≈ stroke / glyph height)."""
    try:
        font = ImageFont.truetype(font_path, size)
        l, t, r, b = ImageDraw.Draw(Image.new('L', (1, 1))).textbbox(
            (0, 0), glyph, font=font)
        gw, gh = max(1, r - l), max(1, b - t)
        canvas = Image.new('L', (gw + 8, gh + 8), 0)
        ImageDraw.Draw(canvas).text((-l + 4, -t + 4), glyph, font=font, fill=255)
        arr = np.array(canvas)
    except Exception:
        return None
    mask = (arr > 127).astype(np.uint8)
    if mask.sum() < 10:
        return None
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    half = float(np.percentile(dist[mask > 0], 90))
    return (2 * half) / float(gh)


def discover_fonts(stroke_min: float = STROKE_RATIO_MIN,
                   stroke_max: float = STROKE_RATIO_MAX) -> List[str]:
    """Scan weights/fonts/ first, fall back to system fonts. Filter by stroke."""
    paths: List[str] = []
    if _FONT_DIR.is_dir():
        for ext in ('*.ttf', '*.otf', '*.ttc'):
            paths.extend(str(p) for p in _FONT_DIR.glob(ext))
    if not paths:
        paths = [p for p in _SYSTEM_FONT_FALLBACKS if os.path.exists(p)]
    seen, uniq = set(), []
    for p in paths:
        if p not in seen and os.path.exists(p):
            seen.add(p); uniq.append(p)
    kept = []
    for p in uniq:
        r = _measure_stroke_ratio(p)
        if r is None:
            continue
        if stroke_min <= r <= stroke_max:
            kept.append(p)
    return kept


# ─────────────────────────────────────────────── Mask + style helpers

def _get_char_mask(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.mean(th) > 127:
        th = cv2.bitwise_not(th)
    return th > 0


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _extract_style_from_crops(crops: List[np.ndarray]) -> Dict[str, Any]:
    """Style fingerprint from real OK crops. Median-based, robust to outliers."""
    inks, bgs, hs, ws = [], [], [], []
    blur_vars, noise_stds = [], []
    for crop in crops:
        bgr = _ensure_bgr(crop)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if np.mean(th) > 127:
            th = cv2.bitwise_not(th)
        fg = th > 0
        bg = ~fg
        if fg.sum() < 10 or bg.sum() < 10:
            continue
        inks.append(np.median(bgr[fg], axis=0))
        bgs.append(np.median(bgr[bg], axis=0))
        ys, xs = np.where(fg)
        hs.append(ys.max() - ys.min() + 1)
        ws.append(xs.max() - xs.min() + 1)
        blur_vars.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        smooth = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_stds.append(float(np.std(gray.astype(np.float32) - smooth)))

    if not inks:
        raise ValueError("No usable real OK crops to extract style from")

    style = {
        'ink_bgr':   tuple(int(v) for v in np.median(inks, axis=0)),
        'bg_bgr':    tuple(int(v) for v in np.median(bgs, axis=0)),
        'mean_h':    int(np.median(hs)),
        'mean_w':    int(np.median(ws)),
        'blur_var':  float(np.median(blur_vars)),
        'noise_std': float(np.median(noise_stds)),
        'n_analyzed': len(inks),
    }
    if style['blur_var'] > 200:    style['blur_sigma'] = 0.4
    elif style['blur_var'] > 100:  style['blur_sigma'] = 0.7
    elif style['blur_var'] > 50:   style['blur_sigma'] = 1.0
    else:                          style['blur_sigma'] = 1.4
    ink_lum = 0.299 * style['ink_bgr'][2] + 0.587 * style['ink_bgr'][1] + 0.114 * style['ink_bgr'][0]
    bg_lum  = 0.299 * style['bg_bgr'][2]  + 0.587 * style['bg_bgr'][1]  + 0.114 * style['bg_bgr'][0]
    style['ink_bg_contrast'] = float(abs(ink_lum - bg_lum))
    return style


# ─────────────────────────────────────────────── BG harvesting

def _inpaint_clean(img: np.ndarray, dilate_iters: int = 5,
                   validate: bool = True) -> Tuple[np.ndarray, bool]:
    """Inpaint char out → clean BG. ok_flag=False if ghost ink remains."""
    img = _ensure_bgr(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.mean(mask) > 127:
        mask = cv2.bitwise_not(mask)
    kernel = np.ones((3, 3), np.uint8)
    mask_dil = cv2.dilate(mask, kernel, iterations=dilate_iters)
    clean = cv2.inpaint(img, mask_dil, 4, cv2.INPAINT_TELEA)
    if not validate:
        return clean, True
    clean_gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(clean_gray, 40, 120)
    inside = mask_dil > 0
    outside = ~inside
    if inside.sum() < 10 or outside.sum() < 10:
        return clean, True
    edge_in = edges[inside].mean() / 255.0
    edge_out = edges[outside].mean() / 255.0
    ok = (edge_in <= edge_out * 2.5 + 0.02)
    return clean, ok


def _collect_bgs(crops: List[np.ndarray], target_size: Tuple[int, int],
                 n: int = 24, rng: Optional[random.Random] = None,
                 dilate_iters: int = 5) -> List[np.ndarray]:
    rng = rng or random.Random()
    chosen = crops[:n] if len(crops) <= n else rng.sample(crops, n)
    bgs = []
    for crop in chosen:
        clean, ok = _inpaint_clean(crop, dilate_iters=dilate_iters, validate=True)
        if not ok:
            continue
        if clean.shape[1] != target_size[0] or clean.shape[0] != target_size[1]:
            clean = cv2.resize(clean, target_size)
        bgs.append(clean)
    return bgs


# ─────────────────────────────────────────────── Rendering

_CHAR_ASPECT_CACHE: Dict[Tuple[str, str], float] = {}


def _char_natural_aspect(font_path: str, ch: str) -> float:
    key = (font_path, ch)
    if key in _CHAR_ASPECT_CACHE:
        return _CHAR_ASPECT_CACHE[key]
    try:
        font = ImageFont.truetype(font_path, 80)
        l, t, r, b = ImageDraw.Draw(Image.new('L', (1, 1))).textbbox(
            (0, 0), ch, font=font)
        aspect = max(1, r - l) / max(1, b - t)
    except Exception:
        aspect = 0.55
    _CHAR_ASPECT_CACHE[key] = aspect
    return aspect


def _render_char_alpha(ch: str, canvas_size: Tuple[int, int], font_path: str,
                       target_h_frac: float = 0.85, jitter: bool = True,
                       rng: Optional[random.Random] = None,
                       rotation_deg: float = 0.0) -> np.ndarray:
    rng = rng or random.Random()
    W, H = canvas_size
    img = Image.new('L', (W, H), 0)
    draw = ImageDraw.Draw(img)
    target_h = int(H * target_h_frac)
    fs_lo, fs_hi = 8, max(10, int(H * 1.5))
    while fs_hi - fs_lo > 1:
        fs = (fs_lo + fs_hi) // 2
        font = ImageFont.truetype(font_path, fs)
        l, t, r, b = draw.textbbox((0, 0), ch, font=font)
        if (b - t) > target_h:
            fs_hi = fs
        else:
            fs_lo = fs
    font = ImageFont.truetype(font_path, fs_lo)
    l, t, r, b = draw.textbbox((0, 0), ch, font=font)
    cw, chh = r - l, b - t
    cx = (W - cw) // 2 - l
    cy = (H - chh) // 2 - t
    if jitter:
        cx += rng.randint(-2, 2)
        cy += rng.randint(-2, 2)
    draw.text((cx, cy), ch, font=font, fill=255)
    arr = np.array(img)
    if abs(rotation_deg) > 0.01:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rotation_deg, 1.0)
        arr = cv2.warpAffine(arr, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    return arr


def _render_glyph_alpha_from_dict(ch: str, canvas_size: Tuple[int, int],
                                  glyph_dict: Dict[str, np.ndarray],
                                  target_h_frac: float = 0.85,
                                  jitter: bool = True,
                                  rng: Optional[random.Random] = None,
                                  rotation_deg: float = 0.0) -> Optional[np.ndarray]:
    """
    Render alpha mask using a project-derived glyph mask instead of a TTF.
    Returns None when `ch` is not in the dict — caller falls back to a font.

    Adds stochastic threshold + ±1px morph so stroke thickness has variance
    even though the source is a single averaged mask.
    """
    if ch not in glyph_dict:
        return None
    rng = rng or random.Random()
    W, H = canvas_size
    src = glyph_dict[ch]
    ys, xs = np.where(src > 20)
    if len(xs) < 3:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    ink = src[y0:y1, x0:x1]

    th = max(8, int(H * target_h_frac))
    ih, iw = ink.shape
    scale = th / ih
    tw = max(1, int(round(iw * scale)))
    th_resized = max(1, int(round(ih * scale)))
    resized = cv2.resize(ink, (tw, th_resized), interpolation=cv2.INTER_AREA)

    # Stochastic threshold → variance in stroke thickness across crops.
    thr = rng.randint(110, 180)
    binary = (resized >= thr).astype(np.uint8) * 255
    if rng.random() < 0.3:
        k = np.ones((2, 2), np.uint8)
        binary = (cv2.dilate(binary, k) if rng.random() < 0.5
                  else cv2.erode(binary, k))

    out = np.zeros((H, W), dtype=np.uint8)
    cx = (W - tw) // 2
    cy = (H - th_resized) // 2
    if jitter:
        cx += rng.randint(-2, 2)
        cy += rng.randint(-2, 2)
    x0_o = max(0, cx); y0_o = max(0, cy)
    x1_o = min(W, cx + tw); y1_o = min(H, cy + th_resized)
    src_x0 = max(0, -cx); src_y0 = max(0, -cy)
    src_x1 = src_x0 + (x1_o - x0_o); src_y1 = src_y0 + (y1_o - y0_o)
    if x1_o > x0_o and y1_o > y0_o:
        out[y0_o:y1_o, x0_o:x1_o] = binary[src_y0:src_y1, src_x0:src_x1]

    if abs(rotation_deg) > 0.01:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rotation_deg, 1.0)
        out = cv2.warpAffine(out, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    return out


def _composite_on_bg(ch: str, real_bg: np.ndarray, ink_bgr: Tuple[int, int, int],
                     font_path: str, target_h_frac: float = 0.85,
                     jitter: bool = True, rng: Optional[random.Random] = None,
                     rotation_deg: float = 0.0,
                     project_glyphs: Optional[Dict[str, np.ndarray]] = None) -> Optional[np.ndarray]:
    H, W = real_bg.shape[:2]
    if font_path == PROJECT_GLYPHS_TOKEN:
        if not project_glyphs:
            return None
        mask = _render_glyph_alpha_from_dict(
            ch, (W, H), project_glyphs,
            target_h_frac=target_h_frac, jitter=jitter, rng=rng,
            rotation_deg=rotation_deg,
        )
        if mask is None:
            return None
    else:
        mask = _render_char_alpha(ch, (W, H), font_path, target_h_frac, jitter, rng, rotation_deg)
    soft = cv2.GaussianBlur(mask, (3, 3), 0.6).astype(np.float32) / 255.0
    bg = real_bg.astype(np.float32)
    ink = np.array([ink_bgr[0], ink_bgr[1], ink_bgr[2]],
                   dtype=np.float32).reshape(1, 1, 3)
    result = bg * (1 - soft[..., None]) + ink * soft[..., None]
    return result.clip(0, 255).astype(np.uint8)


def _render_solid_bg(ch: str, canvas_size: Tuple[int, int],
                     ink_bgr: Tuple[int, int, int], bg_bgr: Tuple[int, int, int],
                     font_path: str, target_h_frac: float = 0.85,
                     jitter: bool = True, rng: Optional[random.Random] = None,
                     rotation_deg: float = 0.0,
                     project_glyphs: Optional[Dict[str, np.ndarray]] = None) -> Optional[np.ndarray]:
    W, H = canvas_size
    bg_arr = np.full((H, W, 3),
                     (int(bg_bgr[0]), int(bg_bgr[1]), int(bg_bgr[2])), dtype=np.uint8)
    return _composite_on_bg(ch, bg_arr, ink_bgr, font_path,
                            target_h_frac=target_h_frac, jitter=jitter,
                            rng=rng, rotation_deg=rotation_deg,
                            project_glyphs=project_glyphs)


# ─────────────────────────────────────────────── Camera artifacts

def _apply_chromatic_aberration(bgr: np.ndarray, shift_px: int = 1,
                                rng: Optional[random.Random] = None) -> np.ndarray:
    rng = rng or random.Random()
    r, g, b = bgr[:, :, 2], bgr[:, :, 1], bgr[:, :, 0]
    sx = rng.randint(0, max(1, shift_px))
    return np.stack([np.roll(b, -sx, axis=1), g, np.roll(r, sx, axis=1)], axis=2)


def _apply_camera_noise(bgr: np.ndarray, style: Dict[str, Any],
                        rng: Optional[random.Random] = None,
                        edge_noise_std: float = 15.0, edge_band_px: int = 2,
                        edge_color_jitter: float = 8.0, shot_scale: float = 4.0,
                        row_wobble: float = 1.5, vignette_strength: float = 0.05) -> np.ndarray:
    rng = rng or random.Random()
    out = bgr.astype(np.float32)
    h, w = out.shape[:2]
    intensity = out.mean(axis=2, keepdims=True) / 255.0
    out += np.random.normal(0, 1, out.shape).astype(np.float32) * \
           (shot_scale * np.sqrt(intensity + 0.05))
    read_std = max(0.8, style.get('noise_std', 2.5) * 0.7)
    out += np.random.normal(0, read_std, out.shape).astype(np.float32)
    out += np.random.normal(0, row_wobble, (h, 1, 1)).astype(np.float32)
    if edge_noise_std > 0:
        gray = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        band = cv2.dilate(edges, np.ones((3, 3), np.uint8),
                          iterations=max(1, int(edge_band_px)))
        band = cv2.GaussianBlur(band.astype(np.float32), (5, 5), 0)
        band /= max(1.0, band.max())
        em = band[..., None]
        out += np.random.normal(0, edge_noise_std, out.shape).astype(np.float32) * em
        if edge_color_jitter > 0:
            for ch_idx in range(3):
                bias = rng.uniform(-1, 1) * edge_color_jitter
                out[..., ch_idx] += np.random.normal(0, edge_color_jitter, out.shape[:2]) \
                                       .astype(np.float32) * em[..., 0]
                out[..., ch_idx] += bias * em[..., 0]
    if vignette_strength > 0:
        yy, xx = np.mgrid[:h, :w]
        cy, cx = h / 2, w / 2
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r_norm = r / (np.sqrt(cy ** 2 + cx ** 2) + 1e-6)
        vig = (1.0 - vignette_strength * r_norm ** 2)[..., None].astype(np.float32)
        out *= vig
    return np.clip(out, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────── Validation

def _validate(img: np.ndarray, fill_min: float = 0.10, fill_max: float = 0.65,
              min_contrast: float = 20.0) -> Tuple[bool, str]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.mean(th) > 127:
        th = cv2.bitwise_not(th)
    fg = th > 0
    bg = ~fg
    if fg.sum() < 5 or bg.sum() < 5:
        return False, 'no-fg-or-bg'
    fill = fg.sum() / fg.size
    if fill < fill_min:
        return False, f'fill-low={fill:.3f}'
    if fill > fill_max:
        return False, f'fill-high={fill:.3f}'
    if abs(float(gray[fg].mean()) - float(gray[bg].mean())) < min_contrast:
        return False, 'contrast-low'
    return True, 'ok'


# ─────────────────────────────────────────────── One-shot synthesis

def _synth_one(ch: str, style: Dict[str, Any], fonts: List[str],
               size: Tuple[int, int], rng: random.Random,
               real_bgs: List[np.ndarray], rotation_max_deg: float,
               size_jitter: float, aspect_aware: bool,
               pad_frac_min: float, pad_frac_max: float,
               min_width: int, max_width: int,
               char_fill_min: float, char_fill_max: float,
               project_glyphs: Optional[Dict[str, np.ndarray]] = None,
               ) -> Tuple[np.ndarray, str, float]:
    # If user picks project glyphs but the dict misses this char, transparently
    # fall back to a real font for THIS crop so coverage gaps don't kill the
    # synthesis loop. Done before any rendering work.
    font_path = rng.choice(fonts)
    if font_path == PROJECT_GLYPHS_TOKEN:
        if not project_glyphs or ch not in project_glyphs:
            real_fonts = [f for f in fonts if f != PROJECT_GLYPHS_TOKEN]
            if not real_fonts:
                raise ValueError(
                    f"char '{ch}' has no project glyph and no TTF fallback in font_paths"
                )
            font_path = rng.choice(real_fonts)

    sh = (max(20, int(round(size[1] * rng.uniform(1 - size_jitter, 1 + size_jitter))))
          if size_jitter > 0 else size[1])
    if aspect_aware:
        if font_path == PROJECT_GLYPHS_TOKEN and project_glyphs and ch in project_glyphs:
            # Aspect from the actual averaged glyph mask
            g = project_glyphs[ch]
            ys, xs = np.where(g > 20)
            if len(xs) > 3 and len(ys) > 3:
                aspect = max(1, xs.max() - xs.min()) / max(1, ys.max() - ys.min())
            else:
                aspect = 0.55
        else:
            aspect = _char_natural_aspect(font_path, ch)
        pad = rng.uniform(pad_frac_min, pad_frac_max) * sh
        sw = max(min_width, min(max_width, int(round(aspect * sh + pad))))
    elif size_jitter > 0:
        sw = max(min_width, min(max_width,
                                int(round(size[0] * rng.uniform(1 - size_jitter, 1 + size_jitter)))))
    else:
        sw = max(min_width, min(max_width, size[0]))
    h_frac = rng.uniform(char_fill_min, char_fill_max)
    rot = rng.uniform(-rotation_max_deg, rotation_max_deg)

    if real_bgs:
        bg = rng.choice(real_bgs)
        if bg.shape[1] != sw or bg.shape[0] != sh:
            bg = cv2.resize(bg, (sw, sh))
        crop = _composite_on_bg(ch, bg, style['ink_bgr'], font_path,
                                target_h_frac=h_frac, jitter=True, rng=rng,
                                rotation_deg=rot, project_glyphs=project_glyphs)
    else:
        crop = _render_solid_bg(ch, (sw, sh), style['ink_bgr'], style['bg_bgr'],
                                font_path, target_h_frac=h_frac, jitter=True,
                                rng=rng, rotation_deg=rot,
                                project_glyphs=project_glyphs)
    # If project-glyph render returned None (race-y check above missed), retry
    # once with a real font instead of failing the whole synth attempt.
    if crop is None:
        real_fonts = [f for f in fonts if f != PROJECT_GLYPHS_TOKEN]
        if real_fonts:
            font_path = rng.choice(real_fonts)
            crop = (_composite_on_bg(ch, bg, style['ink_bgr'], font_path,
                                     target_h_frac=h_frac, jitter=True, rng=rng,
                                     rotation_deg=rot)
                    if real_bgs else
                    _render_solid_bg(ch, (sw, sh), style['ink_bgr'], style['bg_bgr'],
                                     font_path, target_h_frac=h_frac, jitter=True,
                                     rng=rng, rotation_deg=rot))
        if crop is None:
            raise ValueError(f"render failed for char '{ch}' even after font fallback")
    crop = _apply_chromatic_aberration(crop, shift_px=1, rng=rng)
    crop = _apply_camera_noise(crop, style, rng=rng)
    sigma = max(0.3, style['blur_sigma'] * rng.uniform(0.6, 1.0))
    crop = cv2.GaussianBlur(crop, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return crop, font_path, rot


def _synthesize_char(ch: str, n: int, style: Dict[str, Any], fonts: List[str],
                     size: Tuple[int, int], real_bgs: List[np.ndarray],
                     rng: random.Random,
                     rotation_max_deg: float = 5.0,
                     size_jitter: float = 0.30, aspect_aware: bool = True,
                     pad_frac_min: float = 0.05, pad_frac_max: float = 0.15,
                     min_width: int = 12, max_width: int = 90,
                     char_fill_min: float = 0.85, char_fill_max: float = 0.95,
                     max_retries: int = 4,
                     fill_min: float = 0.10, fill_max: float = 0.65,
                     min_contrast: float = 20.0,
                     project_glyphs: Optional[Dict[str, np.ndarray]] = None,
                     ) -> List[Tuple[np.ndarray, str, float]]:
    out = []
    for _ in range(n):
        for _ in range(max_retries):
            crop, fp, rot = _synth_one(
                ch, style, fonts, size, rng, real_bgs,
                rotation_max_deg, size_jitter, aspect_aware,
                pad_frac_min, pad_frac_max, min_width, max_width,
                char_fill_min, char_fill_max,
                project_glyphs=project_glyphs,
            )
            ok, _ = _validate(crop, fill_min=fill_min, fill_max=fill_max,
                              min_contrast=min_contrast)
            if ok:
                out.append((crop, fp, rot))
                break
        else:
            out.append((crop, fp, rot))
    return out


# ─────────────────────────────────────────────── Process-local cache

# Keyed by images_dir absolute path string; reset on process restart.
_cache: Dict[str, Dict[str, Any]] = {}


def _get_or_build_cache(annotations: List[MLAnnotationInDB],
                       images_dir: Path,
                       imported_ok_crops: Optional[List[Tuple[np.ndarray, str]]] = None,
                       style_sample_n: int = 64,
                       sample_strategy: str = "random",
                       ) -> Dict[str, Any]:
    key = f"{str(images_dir.resolve())}:imp{len(imported_ok_crops or [])}:n{style_sample_n}:{sample_strategy}"
    if key in _cache:
        return _cache[key]

    ok_crops_by_char: Dict[str, List[np.ndarray]] = {}
    for ann in annotations:
        img_path = images_dir / ann.filename
        for region in ann.regions:
            for seg in region.segments:
                if seg.label != "OK" or not seg.char_id:
                    continue
                crop = crop_segment(img_path, {
                    "x": seg.x, "y": seg.y, "w": seg.w, "h": seg.h,
                })
                if crop is None:
                    continue
                ok_crops_by_char.setdefault(seg.char_id, []).append(crop)

    if imported_ok_crops:
        for crop, cid in imported_ok_crops:
            if not cid or crop is None:
                continue
            ok_crops_by_char.setdefault(cid, []).append(crop)

    all_crops: List[np.ndarray] = []
    for v in ok_crops_by_char.values():
        all_crops.extend(v)
    if not all_crops:
        raise ValueError("No real OK crops with char_id in this project — cannot synthesize")

    rng = random.Random(42)
    n = max(1, min(int(style_sample_n), len(all_crops)))
    if sample_strategy == "first":
        sample = all_crops[:n]
    elif sample_strategy == "stratified":
        per_char = max(1, n // max(1, len(ok_crops_by_char)))
        sample = []
        for v in ok_crops_by_char.values():
            sample.extend(rng.sample(v, min(per_char, len(v))))
        sample = sample[:n]
        if len(sample) < n:
            extra = [c for c in all_crops if c not in sample]
            rng.shuffle(extra)
            sample.extend(extra[:n - len(sample)])
    else:
        sample = rng.sample(all_crops, n)
    style = _extract_style_from_crops(sample)
    bundle = {
        'style': style,
        'ok_crops_by_char': ok_crops_by_char,
        'bg_pool_by_char': {},
        'mean_size': (style['mean_w'] + 4, style['mean_h'] + 4),
        'sample_used': sample,
    }
    _cache[key] = bundle
    logger.info(f"[ok-synth cache] built for {key}: {len(all_crops)} crops, "
                f"{len(ok_crops_by_char)} chars, size={bundle['mean_size']}")
    return bundle


def _get_bgs_for_char(bundle: Dict[str, Any], char_id: str,
                      target_size: Tuple[int, int],
                      rng: random.Random, n: int = 24) -> List[np.ndarray]:
    cache = bundle['bg_pool_by_char']
    if char_id in cache:
        return cache[char_id]
    char_crops = bundle['ok_crops_by_char'].get(char_id, [])
    # Fallback to global pool if too few per-char
    if len(char_crops) < 4:
        char_crops = []
        for v in bundle['ok_crops_by_char'].values():
            char_crops.extend(v)
    bgs = _collect_bgs(char_crops, target_size, n=n, rng=rng)
    cache[char_id] = bgs
    return bgs


def clear_cache(images_dir: Optional[Path] = None) -> None:
    """Drop synthesis cache (e.g. after new annotations added)."""
    if images_dir is None:
        _cache.clear()
    else:
        _cache.pop(str(images_dir.resolve()), None)


# ─────────────────────────────────────────────── Public API

def _resolve_fonts(font_paths: Optional[List[str]]) -> Tuple[List[str], bool]:
    """
    Split caller-supplied font_paths into (real_font_files, want_project_glyphs).
    Strips the PROJECT_GLYPHS_TOKEN out so downstream callers see only files,
    then re-adds the token to the rng pool inside _synth_one if needed.
    """
    want_project = False
    real: List[str] = []
    for p in (font_paths or []):
        if p == PROJECT_GLYPHS_TOKEN:
            want_project = True
        elif os.path.exists(p):
            real.append(p)
    return real, want_project


def synthesize_ok_from_annotations(
    annotations: List[MLAnnotationInDB],
    images_dir: Path,
    target_n_per_char: int = 30,
    only_below_threshold: bool = True,
    char_filter: Optional[List[str]] = None,
    rotation_max_deg: float = 5.0,
    size_jitter: float = 0.30,
    imported_ok_crops: Optional[List[Tuple[np.ndarray, str]]] = None,
    font_paths: Optional[List[str]] = None,
    style_sample_n: int = 64,
    sample_strategy: str = "random",
    bg_per_char: int = 24,
    char_fill_min: float = 0.85,
    char_fill_max: float = 0.95,
    fill_min: float = 0.10,
    fill_max: float = 0.65,
    min_contrast: float = 20.0,
    max_retries: int = 4,
    project_glyphs: Optional[Dict[str, np.ndarray]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate synthetic OK crops to top-up chars below `target_n_per_char`.

    Args:
        annotations:        project annotations (provides real OK crops + char_ids).
        images_dir:         project images dir for crop_segment.
        target_n_per_char:  target N OK samples per char after synthesis.
        only_below_threshold: if True, only top-up chars below target. False → add
                              `target_n_per_char` to ALL chars regardless.
        char_filter:        optional list of char_ids to restrict to.
        rotation_max_deg:   max rotation per sample (±deg).
        size_jitter:        relative size jitter per sample.
        imported_ok_crops:  optional (crop_bgr, char_id) pairs from the
                            Imported Chars pool — merged into the OK pool used
                            for style fingerprint + BG sampling. Lets chars
                            only present in imports get covered.

    Returns list of {crop_b64, char_id, font_name, rotation_deg, source}.
    """
    if font_paths:
        real_fonts, want_project = _resolve_fonts(font_paths)
    else:
        real_fonts = discover_fonts()
        want_project = False

    # Build the final font pool — `_synth_one` rolls one entry per crop, so
    # adding PROJECT_GLYPHS_TOKEN once gives it a roughly equal probability
    # with each TTF the user selected. If the user selected ONLY project
    # glyphs, the pool will be just the token (single entry).
    fonts: List[str] = list(real_fonts)
    use_project_glyphs = bool(want_project and project_glyphs)
    if use_project_glyphs:
        fonts.append(PROJECT_GLYPHS_TOKEN)
    if not fonts:
        raise RuntimeError(
            f"No usable fonts found in {_FONT_DIR} (and no system fallback). "
            "Drop .ttf/.otf files in weights/fonts/, or build project glyphs."
        )
    rng = random.Random()
    np.random.seed(int(time.time()) % (2**31))

    bundle = _get_or_build_cache(
        annotations, images_dir, imported_ok_crops,
        style_sample_n=style_sample_n, sample_strategy=sample_strategy,
    )
    style = bundle['style']
    ok_by_char = bundle['ok_crops_by_char']
    target_size = bundle['mean_size']

    target_chars = list(ok_by_char.keys())
    if char_filter:
        wanted = set(char_filter)
        target_chars = [c for c in target_chars if c in wanted]

    out: List[Dict[str, Any]] = []
    for char_id in target_chars:
        existing = len(ok_by_char[char_id])
        n_need = (max(0, target_n_per_char - existing)
                  if only_below_threshold else target_n_per_char)
        if n_need <= 0:
            continue
        bgs = _get_bgs_for_char(bundle, char_id, target_size, rng, n=bg_per_char)
        results = _synthesize_char(
            char_id, n_need, style, fonts, target_size, bgs, rng,
            rotation_max_deg=rotation_max_deg, size_jitter=size_jitter,
            char_fill_min=char_fill_min, char_fill_max=char_fill_max,
            max_retries=max_retries, fill_min=fill_min, fill_max=fill_max,
            min_contrast=min_contrast,
            project_glyphs=project_glyphs if use_project_glyphs else None,
        )
        for crop, fp, rot in results:
            out.append({
                'crop': crop,
                'char_id': char_id,
                'font_name': ('project_glyph' if fp == PROJECT_GLYPHS_TOKEN
                              else os.path.basename(fp)),
                'rotation_deg': round(rot, 2),
                'source': 'synthetic_ok',
            })
    return out
