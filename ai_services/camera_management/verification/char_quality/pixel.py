"""
cv_method='pixel' — plain ink pixel count against the template (missing stroke).

Both crops are the char box exactly as the pipeline hands it over: the
template crop from the box drawn on the template image, the target crop from
that box mapped into the frame by SuperPoint (ocr_utils.crop_text_region).
No re-location, no neighbour-stroke removal, no fragment cleaning — the
operator chose the raw SuperPoint region over image-processing segmentation
(2026-09-29). Anything the box catches is counted, so a box that lands
partly on a neighbour or partly off its own char shows up in the numbers.

Steps:
  1. target resized to the template crop's size (same box, same geometry);
  2. both binarised with Otsu; ink polarity read off the TEMPLATE as the
     minority class (text never fills most of its own box) and applied to
     both — this line prints light ink on a dark cap;
  3. counts:
        ratio   = target ink px / template ink px
        missing = template ink with no target ink within 1 px / template ink
        extra   = target ink with no template ink within 1 px / template ink

The template must print the SAME characters as the product; when the
date/lot changes, re-set the template (Get Live Template → Set as Template),
or the digits that changed are compared against a different digit.

Pure cv2 + numpy: no CUDA, safe to load anywhere.
"""

from typing import Dict

import cv2
import numpy as np

# Defect thresholds — chosen for the fewest false NG on the line (operator's
# call for the first trial). Tune from the ink_* values stored with every char
# result once real defective bottles have been run.
UNDER_MISSING = 0.55   # > this share of template ink has no target ink nearby  → under_ink
UNDER_RATIO = 0.50     # target ink below this share of template ink            → under_ink
OVER_EXTRA = 0.60      # > this much ink where the template has none            → over_ink
OVER_RATIO = 1.70      # target ink above this multiple of template ink         → over_ink

_MIN_TEMPLATE_INK = 10  # fewer template ink px than this → the box holds no char (recipe problem)


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _otsu_bright(g: np.ndarray) -> np.ndarray:
    gb = cv2.GaussianBlur(g, (3, 3), 0)
    t, _ = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gb > t


def template_polarity(tmpl_gray: np.ndarray) -> bool:
    """True = ink brighter than background (ink = minority Otsu class)."""
    return bool(_otsu_bright(tmpl_gray).mean() < 0.5)


def compute_char_quality_pixel(tmpl_img: np.ndarray, tgt_img: np.ndarray) -> Dict:
    tg = _gray(tmpl_img)
    gg = _gray(tgt_img)
    if gg.shape != tg.shape:
        gg = cv2.resize(gg, (tg.shape[1], tg.shape[0]), interpolation=cv2.INTER_LINEAR)

    bright = template_polarity(tg)
    T = _otsu_bright(tg)
    G = _otsu_bright(gg)
    if not bright:
        T, G = ~T, ~G
    T = T.astype(np.uint8)
    G = G.astype(np.uint8)

    k = np.ones((3, 3), np.uint8)
    t_px, g_px = int(T.sum()), int(G.sum())
    denom = max(t_px, 1)
    ratio = g_px / denom
    missing = float((T & (1 - cv2.dilate(G, k))).sum()) / denom
    extra = float((G & (1 - cv2.dilate(T, k))).sum()) / denom

    defect_type = None
    if t_px < _MIN_TEMPLATE_INK:
        defect_type = 'template_empty'
    elif missing > UNDER_MISSING or ratio < UNDER_RATIO:
        defect_type = 'under_ink'
    elif extra > OVER_EXTRA or ratio > OVER_RATIO:
        defect_type = 'over_ink'

    confidence = float(np.clip(1.0 - max(missing, extra, abs(1.0 - ratio)), 0.0, 1.0))
    return {
        'confidence': confidence,
        'defect_type': defect_type,
        'ink_ratio': float(ratio),
        'ink_missing': float(missing),
        'ink_extra': float(extra),
        'px_tmpl': t_px,
        'px_tgt': g_px,
        'polarity_bright': bool(bright),
        # binaries for the diff overlay (embedding_classifier looks for these keys)
        '_t_bin': (T * 255).astype(np.uint8),
        '_g_bin': (G * 255).astype(np.uint8),
    }
