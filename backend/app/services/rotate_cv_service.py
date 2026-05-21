"""
Pure-CV cap rotation service — alternative to YOLO OBB.

Algorithm:
1. Detect cap (round white disc) using cv2.HoughCircles.
2. Extract dark text pixels inside cap via Otsu inverse threshold.
3. Find reading-direction angle by projection-profile search: for candidate
   angles in [-90°, 90°), rotate text mask and pick the angle that maximizes
   variance of horizontal projection (text lines collapse into bright bands).
4. Resolve 180° ambiguity by template-matching a rendered "BEST" reference
   word at 0° vs 180° — whichever has higher TM_CCOEFF_NORMED score wins.
5. Rotate cap region only (circular mask) and paste back into frame.

Returns the original frame unchanged when the cap can't be detected.

Logs go to logs/obb_rotation/{date}.log via setup_category_logger so both
the YOLO OBB and CV paths share one log file for easy diffing.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from app.utils.logging_config import setup_category_logger


logger = setup_category_logger(
    category="obb_rotation",
    level=logging.INFO,
    add_console=True,
    logger_name=__name__,
)


_FONT = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: cap detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_cap(gray: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """HoughCircles → return (cx, cy, r) for the cap, or None."""
    h, w = gray.shape
    blur = cv2.medianBlur(gray, 9)
    min_dim = min(h, w)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=1.5, minDist=int(min_dim * 0.4),
        param1=80, param2=50,
        minRadius=int(min_dim * 0.12),
        maxRadius=int(min_dim * 0.35),
    )
    if circles is None:
        return None
    cx_img, cy_img = w / 2.0, h / 2.0
    best = None
    best_score = -1.0
    for x, y, r in circles[0, :]:
        cx_i, cy_i, r_i = int(x), int(y), int(r)
        if not (0 <= cx_i < w and 0 <= cy_i < h):
            continue
        sub = gray[max(0, cy_i - r_i // 3): cy_i + r_i // 3,
                   max(0, cx_i - r_i // 3): cx_i + r_i // 3]
        if sub.size == 0:
            continue
        inner_mean = float(sub.mean())
        if inner_mean < 140:
            # Reject dark "circles" (these are usually background artifacts)
            continue
        cdist = float(np.hypot(x - cx_img, y - cy_img))
        score = inner_mean - 0.3 * cdist
        if score > best_score:
            best_score = score
            best = (float(x), float(y), float(r))
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: text mask inside cap
# ─────────────────────────────────────────────────────────────────────────────

def _text_mask_in_cap(gray: np.ndarray, cap: Tuple[float, float, float],
                       shrink: float = 0.88) -> np.ndarray:
    cx, cy, r = cap
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cap_mask = np.zeros_like(gray)
    cv2.circle(cap_mask, (int(cx), int(cy)), int(r * shrink), 255, -1)
    text = cv2.bitwise_and(dark, cap_mask)
    text = cv2.morphologyEx(text, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: text reading direction angle
# ─────────────────────────────────────────────────────────────────────────────

def _text_angle_in_cap(gray: np.ndarray, cap: Tuple[float, float, float]) -> Tuple[float, int, np.ndarray]:
    """
    Projection-profile search for the rotation angle that makes text horizontal.
    Returns (angle_deg, dark_pixel_count, text_mask_at_full_frame).
    """
    text = _text_mask_in_cap(gray, cap)
    n_dark = int((text > 0).sum())
    if n_dark < 50:
        return 0.0, n_dark, text

    cx, cy, r = cap
    H, W = gray.shape
    pad = int(r + 10)
    x1 = max(0, int(cx - pad)); y1 = max(0, int(cy - pad))
    x2 = min(W, int(cx + pad)); y2 = min(H, int(cy + pad))
    roi = text[y1:y2, x1:x2]
    lcx, lcy = float(cx - x1), float(cy - y1)

    def score(a: float) -> float:
        M = cv2.getRotationMatrix2D((lcx, lcy), a, 1.0)
        rot = cv2.warpAffine(roi, M, (roi.shape[1], roi.shape[0]),
                              flags=cv2.INTER_NEAREST)
        proj = rot.sum(axis=1).astype(np.float32)
        return float(proj.var())

    # Coarse 2°, fine 0.5°
    best_a, best_s = 0.0, -1.0
    for a in np.arange(-90.0, 90.0, 2.0):
        s = score(a)
        if s > best_s:
            best_s = s; best_a = a
    for a in np.arange(best_a - 2, best_a + 2.01, 0.5):
        s = score(a)
        if s > best_s:
            best_s = s; best_a = a
    return float(best_a), n_dark, text


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: 180° flip via shape match
# ─────────────────────────────────────────────────────────────────────────────

def _render_reference(text: str, scale: float, thickness: int) -> np.ndarray:
    (tw, th), baseline = cv2.getTextSize(text, _FONT, scale, thickness)
    pad = 8
    img = np.full((th + baseline + 2 * pad, tw + 2 * pad), 255, dtype=np.uint8)
    cv2.putText(img, text, (pad, pad + th), _FONT, scale, 0, thickness)
    return img


def _needs_flip_180_shape(gray: np.ndarray, cap: Tuple[float, float, float],
                           angle_deg: float, ref_word: str = "BEST") -> Tuple[bool, float, float]:
    """Rotate ROI by angle_deg, template-match `ref_word` at 0° and 180°."""
    cx, cy, r = cap
    H, W = gray.shape
    pad = int(r + 10)
    x1 = max(0, int(cx - pad)); y1 = max(0, int(cy - pad))
    x2 = min(W, int(cx + pad)); y2 = min(H, int(cy + pad))
    roi = gray[y1:y2, x1:x2].copy()
    lcx, lcy = float(cx - x1), float(cy - y1)

    M = cv2.getRotationMatrix2D((lcx, lcy), angle_deg, 1.0)
    rotated = cv2.warpAffine(roi, M, (roi.shape[1], roi.shape[0]),
                              flags=cv2.INTER_LINEAR, borderValue=255)
    flipped = cv2.rotate(rotated, cv2.ROTATE_180)

    best_s0, best_s180 = -2.0, -2.0
    for scale in (0.8, 1.0, 1.2, 1.5, 1.8, 2.2):
        tpl = _render_reference(ref_word, scale, 2)
        if tpl.shape[0] >= rotated.shape[0] or tpl.shape[1] >= rotated.shape[1]:
            continue
        try:
            s0 = float(cv2.matchTemplate(rotated, tpl, cv2.TM_CCOEFF_NORMED).max())
            s180 = float(cv2.matchTemplate(flipped, tpl, cv2.TM_CCOEFF_NORMED).max())
        except cv2.error:
            continue
        if max(s0, s180) > max(best_s0, best_s180):
            best_s0, best_s180 = s0, s180
    return best_s180 > best_s0, best_s0, best_s180


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: rotate cap region only
# ─────────────────────────────────────────────────────────────────────────────

def _rotate_cap_region(image: np.ndarray, cap: Tuple[float, float, float],
                        angle_deg: float, flip180: bool, margin: int = 30) -> np.ndarray:
    cx, cy, r = cap
    total = angle_deg + (180.0 if flip180 else 0.0)
    cr = int(r + margin)
    x1 = max(0, int(cx - cr)); y1 = max(0, int(cy - cr))
    x2 = min(image.shape[1], int(cx + cr)); y2 = min(image.shape[0], int(cy + cr))
    crop = image[y1:y2, x1:x2].copy()
    lcx, lcy = float(cx - x1), float(cy - y1)
    M = cv2.getRotationMatrix2D((lcx, lcy), total, 1.0)
    crop_rot = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]),
                               flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
    cmask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.circle(cmask, (int(lcx), int(lcy)), int(r), 255, -1)
    out_crop = crop.copy()
    out_crop[cmask > 0] = crop_rot[cmask > 0]
    out = image.copy()
    out[y1:y2, x1:x2] = out_crop
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public
# ─────────────────────────────────────────────────────────────────────────────

def rotate_frame_cv(image: np.ndarray) -> np.ndarray:
    """
    Pure-CV cap rotation. Returns the original frame if no cap is detected.
    Logs the angle, flip decision, and shape-match scores.
    """
    if image is None or image.size == 0:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    H, W = gray.shape
    cap = _detect_cap(gray)
    if cap is None:
        logger.info(f"[RotateCV] frame={W}x{H} → no cap detected, returning original")
        return image
    cx, cy, r = cap
    angle_deg, n_dark, _ = _text_angle_in_cap(gray, cap)
    flip, s0, s180 = _needs_flip_180_shape(gray, cap, angle_deg)
    final_angle = angle_deg + (180.0 if flip else 0.0)
    logger.info(
        f"[RotateCV] frame={W}x{H} cap=(cx={cx:.0f}, cy={cy:.0f}, r={r:.0f}) "
        f"dark_px={n_dark} angle={angle_deg:.1f}° "
        f"shape_match score_0={s0:.3f} score_180={s180:.3f} → flip={flip} "
        f"final_rotation={final_angle:.1f}°"
    )
    return _rotate_cap_region(image, cap, angle_deg, flip)
