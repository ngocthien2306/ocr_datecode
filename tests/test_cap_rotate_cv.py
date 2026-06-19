"""
Pure-CV cap rotation prototype — alternative to YOLO OBB model for the
'/api/cameras/frames/rotate' endpoint.

Pipeline:
1. Find cap (bright disc):
   - Otsu threshold → bright blob mask
   - Pick largest blob with circularity > 0.6 → cap mask
   - Fit min-enclosing circle → (cx, cy, r)
2. Find text orientation inside cap:
   - Invert + threshold inside cap mask → dark text pixels
   - PCA on dark pixel coords → principal axis (modulo 180°)
3. Resolve 180° ambiguity:
   - After rotating by -angle, compute vertical projection of dark pixels
     (sum per row). Top half should have more dark pixels than bottom half
     (because first text line "BEST if Used by" is longer than later line).
   - If bottom half > top half → flip 180°
4. Apply rotation to the cap region only (circular mask, keep background).
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

REPO = Path("/home/demo/Source/ocr_datecode")
IMAGES = [
    REPO / "backend/uploads/inference_results/6a0d53cae10e970b7a50c76d/2026-05-20/24026290/fail_f0_20260520_214911545842_org.jpg",
    REPO / "backend/uploads/inference_results/6a0d53cae10e970b7a50c76d/2026-05-20/24026290/fail_f0_20260520_214912008391_org.jpg",
    REPO / "backend/uploads/inference_results/6a0d53cae10e970b7a50c76d/2026-05-20/24026290/fail_f0_20260520_214912466939_org.jpg",
    REPO / "backend/uploads/inference_results/6a0d53cae10e970b7a50c76d/2026-05-20/24026290/fail_f0_20260520_214913064103_org.jpg",
]
OUT_DIR = REPO / "tests/crop_output/cap_rotate_cv"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def detect_cap(gray: np.ndarray):
    """
    Find the cap (round white disc) using HoughCircles. Returns (cx, cy, r) or None.

    Tuning notes (based on user's 1600x1200 frames):
    - Cap radius is ~260px → search 180-360.
    - Background zigzag has strong edges; medianBlur(9) suppresses high-freq
      texture while keeping the cap's outer rim intact.
    - minDist large (~400) so we only keep 1 dominant circle.
    - Higher param2 (50) → more selective.
    """
    h, w = gray.shape
    blur = cv2.medianBlur(gray, 9)
    # Estimate radius range from frame size (cap ~ 20-40% of min dim)
    min_dim = min(h, w)
    min_r = int(min_dim * 0.12)
    max_r = int(min_dim * 0.35)

    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=1.5, minDist=int(min_dim * 0.4),
        param1=80, param2=50,
        minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        return None
    circles = circles[0, :]
    # Pick circle closest to image center (cap is roughly framed in the scene)
    img_cx, img_cy = w / 2, h / 2
    best = None
    best_score = -1.0
    for x, y, r in circles:
        # Validate: interior must be brighter than the rim (cap white vs bg)
        cy_i, cx_i = int(y), int(x)
        if not (0 <= cx_i < w and 0 <= cy_i < h):
            continue
        r_i = int(r)
        # Sample mean interior brightness in a small inner disk
        inner = gray[max(0, cy_i - r_i // 3): cy_i + r_i // 3,
                     max(0, cx_i - r_i // 3): cx_i + r_i // 3]
        if inner.size == 0:
            continue
        inner_mean = float(inner.mean())
        # Cap interior should be quite bright
        if inner_mean < 140:
            continue
        center_dist = np.hypot(x - img_cx, y - img_cy)
        score = inner_mean - 0.3 * center_dist
        if score > best_score:
            best_score = score
            best = (float(x), float(y), float(r))
    return best


def _text_mask_in_cap(gray: np.ndarray, cap: tuple, shrink: float = 0.88) -> np.ndarray:
    """Return a binary mask (uint8) of dark text pixels inside the cap interior."""
    cx, cy, r = cap
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cap_mask = np.zeros_like(gray)
    cv2.circle(cap_mask, (int(cx), int(cy)), int(r * shrink), 255, -1)
    text = cv2.bitwise_and(dark, cap_mask)
    text = cv2.morphologyEx(text, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return text


def text_angle_in_cap(gray: np.ndarray, cap: tuple) -> tuple:
    """
    Estimate text reading-direction angle using projection-profile search.

    Method:
      For candidate angle θ ∈ [-90, 90) in 2° steps, rotate the text mask by -θ
      around the cap center. Compute horizontal projection (sum per row). When
      θ matches the text reading direction, the rows align with text lines →
      projection has sharp peaks (high variance + clear inter-line gaps).
      Pick θ that maximizes variance(projection).

    Then refine ±2° in 0.5° steps.

    Returns: (angle_deg, dark_pixel_count, text_mask_aligned_at_angle_0)
    """
    cx, cy, r = cap
    H, W = gray.shape
    text = _text_mask_in_cap(gray, cap)
    n_dark = int((text > 0).sum())
    if n_dark < 50:
        return 0.0, n_dark, text

    # Crop a square ROI around the cap to make rotation cheap and avoid edge wrap
    pad = int(r + 10)
    x1 = max(0, int(cx - pad)); y1 = max(0, int(cy - pad))
    x2 = min(W, int(cx + pad)); y2 = min(H, int(cy + pad))
    roi = text[y1:y2, x1:x2]
    lcx, lcy = float(cx - x1), float(cy - y1)

    def score(angle_deg: float) -> float:
        M = cv2.getRotationMatrix2D((lcx, lcy), angle_deg, 1.0)
        rot = cv2.warpAffine(roi, M, (roi.shape[1], roi.shape[0]),
                              flags=cv2.INTER_NEAREST)
        proj = rot.sum(axis=1).astype(np.float32)
        # Variance — text lines = bright rows separated by dark gaps.
        return float(proj.var())

    # Coarse search 2° steps
    best_angle = 0.0
    best_score = -1.0
    for a in np.arange(-90.0, 90.0, 2.0):
        s = score(a)
        if s > best_score:
            best_score = s; best_angle = a
    # Fine 0.5° around best
    for a in np.arange(best_angle - 2, best_angle + 2.01, 0.5):
        s = score(a)
        if s > best_score:
            best_score = s; best_angle = a

    return float(best_angle), n_dark, text


def _render_reference(text: str, scale: float, thickness: int) -> np.ndarray:
    """Render text as DARK pixels on WHITE background (same polarity as printed cap text)."""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = 8
    img = np.full((th + baseline + 2 * pad, tw + 2 * pad), 255, dtype=np.uint8)
    cv2.putText(img, text, (pad, pad + th), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, thickness)
    return img


def needs_flip_180_shape(gray: np.ndarray, cap: tuple, angle_deg: float,
                          reference_word: str = "BEST") -> tuple:
    """
    Resolve 180° flip via template matching against a rendered reference word.

    Pipeline:
      1. Rotate the gray cap-crop by angle_deg so text is horizontal (possibly
         upside-down).
      2. Render reference word at multiple font scales.
      3. Template-match the rotated crop vs reference (TM_CCOEFF_NORMED) — score_0.
      4. Flip the rotated crop 180°, repeat → score_180.
      5. need_flip = (score_180 > score_0).

    Returns: (need_flip, score_0, score_180, best_scale)
    """
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

    best_s0, best_s180, best_scale = -2.0, -2.0, 0
    for scale in (0.8, 1.0, 1.2, 1.5, 1.8, 2.2):
        tpl = _render_reference(reference_word, scale, 2)
        if tpl.shape[0] >= rotated.shape[0] or tpl.shape[1] >= rotated.shape[1]:
            continue
        try:
            r0 = cv2.matchTemplate(rotated, tpl, cv2.TM_CCOEFF_NORMED).max()
            r180 = cv2.matchTemplate(flipped, tpl, cv2.TM_CCOEFF_NORMED).max()
        except cv2.error:
            continue
        if max(r0, r180) > max(best_s0, best_s180):
            best_s0, best_s180, best_scale = float(r0), float(r180), scale
    return best_s180 > best_s0, best_s0, best_s180, best_scale


def needs_flip_180(text_mask_rotated: np.ndarray, cap_center: tuple) -> bool:
    """
    After rotation, decide if 180° flip is needed.

    Heuristic — line-length asymmetry: for the project's date-code caps the
    last line ("06140-13279-U 04:48") is the LONGEST (19 chars), the first
    line ("BEST if Used by") is medium (15 chars), middle is shortest. So
    width(last line) > width(first line) means the orientation is correct;
    width(first line) > width(last line) means it's upside-down.

    We compute the horizontal projection (rows), find the contiguous text
    bands (lines), then compare the widths of the first vs last bands.
    """
    ys, _xs = np.where(text_mask_rotated > 0)
    if len(ys) < 50:
        return False

    proj = text_mask_rotated.sum(axis=1).astype(np.int32)  # one value per row
    # Identify rows that contain text (above 10% of max projection)
    if proj.max() <= 0:
        return False
    thresh = max(1, int(proj.max() * 0.10))
    in_line = proj > thresh

    # Find contiguous line bands
    bands: list = []  # list of (start_row, end_row)
    start = None
    for r, on in enumerate(in_line):
        if on and start is None:
            start = r
        elif (not on) and start is not None:
            bands.append((start, r))
            start = None
    if start is not None:
        bands.append((start, len(in_line)))

    if len(bands) < 2:
        # Fall back: compare top-half vs bottom-half of the text bbox
        y_min, y_max = int(ys.min()), int(ys.max())
        mid = (y_min + y_max) // 2
        top = int((ys < mid).sum())
        bot = int((ys >= mid).sum())
        return top > bot

    # Width of a line = horizontal span of dark pixels in its row band
    def line_width(start_r: int, end_r: int) -> int:
        sub = text_mask_rotated[start_r:end_r]
        col_any = sub.any(axis=0)
        cols = np.where(col_any)[0]
        return int(cols.max() - cols.min()) if len(cols) else 0

    first_w = line_width(*bands[0])
    last_w  = line_width(*bands[-1])
    # Upside-down → first line wider than last line.
    return first_w > last_w


def rotate_cap_region(image: np.ndarray, cap: tuple, angle_deg: float,
                     flip180: bool, margin: int = 30) -> np.ndarray:
    """Rotate only the cap region (circular mask), background untouched."""
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


def rotate_frame_cv(image: np.ndarray):
    """Full pipeline. Returns (rotated_image, info_dict)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    cap = detect_cap(gray)
    info = {"cap": cap, "angle": None, "flip": None, "dark_pixels": 0}
    if cap is None:
        return image, info

    angle_deg, n_dark, text_mask = text_angle_in_cap(gray, cap)
    info["angle"] = angle_deg
    info["dark_pixels"] = n_dark

    # 180° resolution via shape-match against rendered "BEST" reference.
    flip, s0, s180, scale = needs_flip_180_shape(gray, cap, angle_deg)
    info["flip"] = flip
    info["shape_match"] = {"score_0": s0, "score_180": s180, "scale": scale}

    out = rotate_cap_region(image, cap, angle_deg, flip)
    return out, info


def main():
    for idx, path in enumerate(IMAGES, 1):
        img = cv2.imread(str(path))
        if img is None:
            print(f"[{idx}] read fail: {path}")
            continue
        rotated, info = rotate_frame_cv(img)
        # Side-by-side
        h, w = img.shape[:2]
        side = np.hstack([img, rotated])
        # Annotate
        title = (f"img{idx}  cap={info['cap']}  angle={info['angle']}  "
                 f"flip={info['flip']}  dark_px={info['dark_pixels']}")
        cv2.putText(side, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)
        if info["cap"]:
            cx, cy, r = info["cap"]
            cv2.circle(side, (int(cx), int(cy)), int(r), (0, 255, 255), 3)
            # mark same cap on rotated half
            cv2.circle(side, (int(cx + w), int(cy)), int(r), (255, 255, 0), 3)
        out_path = OUT_DIR / f"cap_rotate_{idx}.jpg"
        cv2.imwrite(str(out_path), side)
        print(f"[{idx}] {info}  →  {out_path}")


if __name__ == "__main__":
    main()
