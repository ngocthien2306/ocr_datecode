"""
CVRotationService — pure-CV alternative to OBBRotationService.

Same interface as OBBRotationService (rotate_frame returns (frame, None)) so
the pipeline can swap in/out based on the recipe's `cap_rotation_method`
field.

Algorithm:
  1. cv2.HoughCircles → find cap (white disc) in the full frame.
  2. Otsu inverse threshold inside cap → dark text pixel mask.
  3. Projection-profile search over angle → rotation that aligns text rows
     (text lines collapse into bright bands → max horizontal-projection var).
  4. cv2.matchTemplate against a rendered "BEST" reference at both 0° and
     180° → pick the higher correlation score to resolve 180° ambiguity.
  5. Rotate the cap region only (circular mask), background untouched.

Returns (rotated_frame, None) so single_camera's inverse-transform branch
is skipped (matches OBB service behavior).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np


logger = logging.getLogger(__name__)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────────────────────────────────────
# Cap detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_cap(gray: np.ndarray) -> Optional[Tuple[float, float, float]]:
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
    best, best_score = None, -1.0
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
            continue
        cdist = float(np.hypot(x - cx_img, y - cy_img))
        score = inner_mean - 0.3 * cdist
        if score > best_score:
            best_score = score
            best = (float(x), float(y), float(r))
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Text mask + angle
# ─────────────────────────────────────────────────────────────────────────────

def _text_mask(gray: np.ndarray, cap: Tuple[float, float, float],
                shrink: float = 0.88) -> np.ndarray:
    cx, cy, r = cap
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cap_mask = np.zeros_like(gray)
    cv2.circle(cap_mask, (int(cx), int(cy)), int(r * shrink), 255, -1)
    text = cv2.bitwise_and(dark, cap_mask)
    text = cv2.morphologyEx(text, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return text


def _text_angle(gray: np.ndarray, cap: Tuple[float, float, float]) -> Tuple[float, int]:
    text = _text_mask(gray, cap)
    n_dark = int((text > 0).sum())
    if n_dark < 50:
        return 0.0, n_dark
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
        return float(rot.sum(axis=1).astype(np.float32).var())

    best_a, best_s = 0.0, -1.0
    for a in np.arange(-90.0, 90.0, 2.0):
        s = score(a)
        if s > best_s: best_s = s; best_a = a
    for a in np.arange(best_a - 2, best_a + 2.01, 0.5):
        s = score(a)
        if s > best_s: best_s = s; best_a = a
    return float(best_a), n_dark


# ─────────────────────────────────────────────────────────────────────────────
# 180° flip via shape match
# ─────────────────────────────────────────────────────────────────────────────

def _render_ref(text: str, scale: float, thickness: int) -> np.ndarray:
    (tw, th), baseline = cv2.getTextSize(text, _FONT, scale, thickness)
    pad = 8
    img = np.full((th + baseline + 2 * pad, tw + 2 * pad), 255, dtype=np.uint8)
    cv2.putText(img, text, (pad, pad + th), _FONT, scale, 0, thickness)
    return img


def _need_flip(gray: np.ndarray, cap: Tuple[float, float, float],
                angle_deg: float) -> Tuple[bool, float, float]:
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
        tpl = _render_ref("BEST", scale, 2)
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
# Rotate cap region (background untouched)
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
# Public service (mirrors OBBRotationService interface)
# ─────────────────────────────────────────────────────────────────────────────

class CVRotationService:
    """Pure-CV alternative to OBBRotationService. Same `rotate_frame` signature."""

    # inverse_transform kept for interface parity with OBBRotationService — we
    # do cap-only rotation so the inverse-transform branch is never used.
    def __init__(self, inverse_transform: bool = True):
        self.inverse_transform = inverse_transform
        logger.info("CVRotationService initialized (pure CV — no model needed)")

    @property
    def available(self) -> bool:
        return True

    def rotate_frame(
        self,
        frame: np.ndarray,
        frame_tag: str = ""
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        tag = f"[{frame_tag}] " if frame_tag else ""
        if frame is None or frame.size == 0:
            return frame, None
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            cap = _detect_cap(gray)
            if cap is None:
                logger.info(f"{tag}[RotateCV] FAIL — no cap detected")
                return frame, None
            cx, cy, r = cap
            angle_deg, n_dark = _text_angle(gray, cap)
            flip, s0, s180 = _need_flip(gray, cap, angle_deg)
            final_angle = angle_deg + (180.0 if flip else 0.0)
            logger.info(
                f"{tag}[RotateCV] OK — cap=(cx={cx:.0f}, cy={cy:.0f}, r={r:.0f}) "
                f"angle={angle_deg:.1f}° score_0={s0:.3f} score_180={s180:.3f} "
                f"flip={flip} → total={final_angle:.1f}°"
            )
            result = _rotate_cap_region(frame, cap, angle_deg, flip)
            return result, None
        except Exception as e:
            logger.error(f"{tag}[RotateCV] ERROR — {e}", exc_info=True)
            return frame, None
