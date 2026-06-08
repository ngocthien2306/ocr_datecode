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
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


logger = logging.getLogger(__name__)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


# ─── File logger for rotation results ─────────────────────────────────────────

_rotation_file_logger: Optional[logging.Logger] = None


def _get_rotation_file_logger() -> logging.Logger:
    """Lazy-init a dedicated file logger for CV rotation results.

    Mirrors the OBB rotator's logger so both rotation methods produce
    daily files under logs/{category}/{date}.log with identical formatting.
    """
    global _rotation_file_logger
    if _rotation_file_logger is not None:
        return _rotation_file_logger

    from logging_config import make_handler

    _rotation_file_logger = logging.getLogger('cv_rotation')
    _rotation_file_logger.setLevel(logging.DEBUG)
    _rotation_file_logger.propagate = False  # don't bubble to root logger

    if not any(getattr(h, "_marker", None) == "daily-cv_rotation"
               for h in _rotation_file_logger.handlers):
        fh = make_handler(
            "cv_rotation",
            level=logging.DEBUG,
            fmt='%(asctime)s  %(levelname)-8s  %(message)s',
        )
        fh.formatter.datefmt = '%Y-%m-%d %H:%M:%S'
        setattr(fh, "_marker", "daily-cv_rotation")
        _rotation_file_logger.addHandler(fh)

    logger.info("CV rotation log: cv_rotation/{date}.log")
    return _rotation_file_logger

# Pre-rendered "BEST" templates at fixed scales — cached at import time so
# _need_flip doesn't re-render on every frame. Two scales cover the typical
# size variation of date-code text inside the cap region; more scales would
# re-introduce the matchTemplate cost dominating cap_rotation.
# Scales here are matched to the 2× downsampled ROI used in _need_flip
# (i.e. 0.5 == effective 1.0× on full ROI).
_REF_SCALES = (0.5, 0.75)
_REF_THICKNESS = 1
_REF_TEMPLATES: list = []  # filled lazily by _get_ref_templates()


# ─────────────────────────────────────────────────────────────────────────────
# Cap detection
# ─────────────────────────────────────────────────────────────────────────────

def _slice_by_crop_area(
    arr: np.ndarray,
    crop_area: Optional[Dict[str, int]],
) -> Tuple[int, int, Optional[np.ndarray]]:
    """
    Slice `arr` by user-provided crop_area dict {x1,y1,x2,y2}. Returns
    (offset_x, offset_y, sub_array). When crop_area is falsy or its bounds
    collapse, returns (0, 0, arr) or (0, 0, None) respectively.

    Coords are clamped to array bounds. Callers add (offset_x, offset_y)
    back onto sub-array detection results to land in full-frame coords.
    """
    if not crop_area:
        return 0, 0, arr
    H, W = arr.shape[:2]
    x1 = max(0, int(crop_area.get('x1', 0)))
    y1 = max(0, int(crop_area.get('y1', 0)))
    x2 = min(W, int(crop_area.get('x2', W)))
    y2 = min(H, int(crop_area.get('y2', H)))
    if x2 <= x1 or y2 <= y1:
        return x1, y1, None
    return x1, y1, arr[y1:y2, x1:x2]


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


def _get_ref_templates() -> list:
    """Render 'BEST' templates once at module load and reuse across frames."""
    global _REF_TEMPLATES
    if not _REF_TEMPLATES:
        _REF_TEMPLATES = [
            _render_ref("BEST", s, _REF_THICKNESS) for s in _REF_SCALES
        ]
    return _REF_TEMPLATES


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
    # Downsample 2× before matchTemplate — matchTemplate is O((W·H)·(w·h))
    # so halving both dimensions gives ~16× speedup with negligible accuracy
    # impact for shape-match at the scales we use.
    rotated_ds = cv2.resize(rotated, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_AREA)
    flipped_ds = cv2.rotate(rotated_ds, cv2.ROTATE_180)

    best_s0, best_s180 = -2.0, -2.0
    for tpl in _get_ref_templates():
        if (tpl.shape[0] >= rotated_ds.shape[0]
                or tpl.shape[1] >= rotated_ds.shape[1]):
            continue
        try:
            s0 = float(cv2.matchTemplate(rotated_ds, tpl, cv2.TM_CCOEFF_NORMED).max())
            s180 = float(cv2.matchTemplate(flipped_ds, tpl, cv2.TM_CCOEFF_NORMED).max())
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

def detect_cap_and_crop(
    image: np.ndarray,
    margin_ratio: float = 0.10,
    fill_value: int = 114,
) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Detect the bottle cap (white disc) via HoughCircles, crop a tight square
    around it, and fill pixels outside the circular cap region with `fill_value`
    (gray 114 by default — neutral for SuperPoint).

    Args:
        image:         BGR or grayscale frame.
        margin_ratio:  Extra padding around cap radius (fraction of radius).
        fill_value:    Pixel intensity for pixels outside the cap circle.

    Returns:
        (cropped_image, (x1, y1, x2, y2)) — cropped image with circular mask
        applied, and the bbox in original image coords. Returns None if cap not
        detected.
    """
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    cap = _detect_cap(gray)
    if cap is None:
        return None
    return _compose_cap_result(image, cap, margin_ratio, fill_value)


def _compose_cap_result(
    image: np.ndarray,
    cap: Tuple[float, float, float],
    margin_ratio: float,
    fill_value: int,
) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Inner helper: given a detected cap, crop+mask + return bbox."""
    cx, cy, r = cap
    H, W = image.shape[:2]
    margin = int(r * margin_ratio)
    sq = int(r + margin)
    x1 = max(0, int(cx) - sq)
    y1 = max(0, int(cy) - sq)
    x2 = min(W, int(cx) + sq)
    y2 = min(H, int(cy) + sq)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2].copy()
    ch, cw = crop.shape[:2]
    lcx = int(cx) - x1
    lcy = int(cy) - y1
    mask = np.zeros((ch, cw), dtype=np.uint8)
    cv2.circle(mask, (lcx, lcy), int(r), 255, -1)
    if image.ndim == 3:
        fill = np.full_like(crop, fill_value)
        crop = np.where(mask[..., None] > 0, crop, fill)
    else:
        crop[mask == 0] = fill_value
    return crop, (x1, y1, x2, y2)


def detect_cap_circle(
    image: np.ndarray,
    crop_area: Optional[Dict[str, int]] = None,
) -> Optional[Tuple[float, float, float]]:
    """
    Run HoughCircles ONLY (no crop/mask) and return the cap (cx, cy, r) in
    FULL-frame coordinates.

    If `crop_area` is provided, restrict the search to that sub-region and
    offset the result back to full-frame coords. Returns None if no cap is
    found within the sub-region (no fallback to full-frame — caller chose
    crop_area on purpose, trust it).
    """
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if crop_area:
        x1 = max(0, int(crop_area['x1']))
        y1 = max(0, int(crop_area['y1']))
        x2 = min(gray.shape[1], int(crop_area['x2']))
        y2 = min(gray.shape[0], int(crop_area['y2']))
        if x2 <= x1 or y2 <= y1:
            return None
        sub = gray[y1:y2, x1:x2]
        cap = _detect_cap(sub)
        if cap is None:
            return None
        return (cap[0] + float(x1), cap[1] + float(y1), cap[2])
    return _detect_cap(gray)


def apply_cap_crop(
    image: np.ndarray,
    cap: Tuple[float, float, float],
    margin_ratio: float = 0.10,
    fill_value: int = 114,
) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Apply the cap crop+mask to `image` using a PRE-DETECTED cap circle. Skips
    HoughCircles entirely — useful when the caller already detected the cap on
    a different (but spatially-aligned) frame.
    """
    if image is None or image.size == 0 or cap is None:
        return None
    return _compose_cap_result(image, cap, margin_ratio, fill_value)


class CVRotationService:
    """Pure-CV alternative to OBBRotationService. Same `rotate_frame` signature."""

    # inverse_transform kept for interface parity with OBBRotationService — we
    # do cap-only rotation so the inverse-transform branch is never used.
    def __init__(self, inverse_transform: bool = True):
        self.inverse_transform = inverse_transform
        self._rot_logger = _get_rotation_file_logger()
        logger.info("CVRotationService initialized (pure CV — no model needed)")
        self._rot_logger.info(
            f"CVRotationService initialized (inverse_transform={inverse_transform})"
        )

    @property
    def available(self) -> bool:
        return True

    def rotate_frame(
        self,
        frame: np.ndarray,
        frame_tag: str = "",
        crop_area: Optional[Dict[str, int]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Tuple[float, float, float]]]:
        """Returns (rotated_frame, M, cap_circle).
        cap_circle = cap đã detect bằng HoughCircles — downstream reuse được
        thay vì gọi detect_cap_circle 1 lần nữa.

        If `crop_area` is provided, cap detection + angle search are restricted
        to that sub-region (avoid picking up neighbouring bottles). The rotated
        full frame is returned; cap_circle is reported in FULL-frame coords.
        """
        import time as _time
        tag = f"[{frame_tag}] " if frame_tag else ""
        if frame is None or frame.size == 0:
            return frame, None, None
        try:
            _t0 = _time.perf_counter()
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            ox, oy, sub_gray = _slice_by_crop_area(gray_full, crop_area)
            if sub_gray is None:
                self._rot_logger.info(f"{tag}[RotateCV] FAIL — crop_area outside frame")
                return frame, None, None
            _t_gray = (_time.perf_counter() - _t0) * 1000

            _t1 = _time.perf_counter()
            cap_sub = _detect_cap(sub_gray)
            _t_detect = (_time.perf_counter() - _t1) * 1000
            if cap_sub is None:
                self._rot_logger.info(
                    f"{tag}[RotateCV] FAIL — no cap detected "
                    f"(crop_area={'on' if crop_area else 'off'}, "
                    f"gray={_t_gray:.1f}ms, detect={_t_detect:.1f}ms)"
                )
                return frame, None, None
            cap_full = (cap_sub[0] + ox, cap_sub[1] + oy, cap_sub[2])
            cx, cy, r = cap_full

            _t2 = _time.perf_counter()
            # _text_angle + _need_flip work on sub_gray + sub-coord cap so the
            # ROI computation stays inside the user-bounded region. Angle/flip
            # are coord-independent.
            angle_deg, n_dark = _text_angle(sub_gray, cap_sub)
            _t_angle = (_time.perf_counter() - _t2) * 1000

            _t3 = _time.perf_counter()
            flip, s0, s180 = _need_flip(sub_gray, cap_sub, angle_deg)
            _t_flip = (_time.perf_counter() - _t3) * 1000

            _t4 = _time.perf_counter()
            result = _rotate_cap_region(frame, cap_full, angle_deg, flip)
            _t_rot = (_time.perf_counter() - _t4) * 1000

            final_angle = angle_deg + (180.0 if flip else 0.0)
            _t_total = (_time.perf_counter() - _t0) * 1000
            self._rot_logger.info(
                f"{tag}[RotateCV] OK in {_t_total:.1f}ms — cap=(cx={cx:.0f}, cy={cy:.0f}, "
                f"r={r:.0f}) angle={angle_deg:.1f}° flip={flip} (s0={s0:.3f} s180={s180:.3f}) "
                f"→ total={final_angle:.1f}° | "
                f"gray={_t_gray:.1f}ms, detect_cap={_t_detect:.1f}ms, "
                f"text_angle={_t_angle:.1f}ms, need_flip={_t_flip:.1f}ms, "
                f"rotate={_t_rot:.1f}ms"
            )
            return result, None, (float(cx), float(cy), float(r))
        except Exception as e:
            self._rot_logger.error(f"{tag}[RotateCV] ERROR — {e}", exc_info=True)
            return frame, None, None

    def rotate_frame_dual(
        self,
        frame: np.ndarray,
        frame_tag: str = "",
        crop_area: Optional[Dict[str, int]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[float, float, float]]]:
        """
        Produce BOTH rotation candidates (no-flip + flip180) without running
        the shape-match `_need_flip` step. Caller picks via match confidence.
        Returns (candidate_no_flip, candidate_flipped180, cap_circle).

        If `crop_area` is provided, cap detection is restricted to that
        sub-region; cap_circle in the returned tuple is in FULL-frame coords.
        """
        tag = f"[{frame_tag}] " if frame_tag else ""
        if frame is None or frame.size == 0:
            return None, None, None
        try:
            import time as _time
            _t0 = _time.perf_counter()
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            ox, oy, sub_gray = _slice_by_crop_area(gray_full, crop_area)
            if sub_gray is None:
                self._rot_logger.info(f"{tag}[RotateCV-Dual] FAIL — crop_area outside frame")
                return None, None, None
            cap_sub = _detect_cap(sub_gray)
            if cap_sub is None:
                self._rot_logger.info(f"{tag}[RotateCV-Dual] FAIL — no cap detected")
                return None, None, None
            cap_full = (cap_sub[0] + ox, cap_sub[1] + oy, cap_sub[2])
            angle_deg, _n_dark = _text_angle(sub_gray, cap_sub)
            cand_a = _rotate_cap_region(frame, cap_full, angle_deg, False)
            cand_b = _rotate_cap_region(frame, cap_full, angle_deg, True)
            _t_total = (_time.perf_counter() - _t0) * 1000
            self._rot_logger.info(
                f"{tag}[RotateCV-Dual] OK in {_t_total:.1f}ms — angle={angle_deg:.1f}° "
                f"both candidates emitted"
            )
            return cand_a, cand_b, (float(cap_full[0]), float(cap_full[1]), float(cap_full[2]))
        except Exception as e:
            self._rot_logger.error(f"{tag}[RotateCV-Dual] ERROR — {e}", exc_info=True)
            return None, None, None
