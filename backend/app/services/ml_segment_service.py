"""
ML Segment Service
Character segmentation logic for ML training pipeline.
Ported from test_segment.py at project root.
"""
import cv2 as cv
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import uuid


# ──────────────────────────────────────── helpers ──

def _split_wide_box(thresh_roi: np.ndarray, box: Tuple, median_w: float) -> List[Tuple]:
    """Split a box that is too wide using vertical projection."""
    x, y, w, h = box
    if w < median_w * 1.5:
        return [box]

    roi = thresh_roi[y:y + h, x:x + w]
    v_proj = np.sum(roi, axis=0) / 255.0

    n_chars = max(2, round(w / median_w))
    split_points = []
    for i in range(1, n_chars):
        expected_x = int(i * w / n_chars)
        search_w = max(3, int(median_w * 0.3))
        left = max(1, expected_x - search_w)
        right = min(w - 1, expected_x + search_w)
        if left < right:
            window = v_proj[left:right]
            best = left + int(np.argmin(window))
            split_points.append(best)

    split_points = sorted(set(split_points))
    sub_boxes = []
    prev = 0
    for sp in split_points:
        if sp - prev > 3:
            sub_boxes.append((x + prev, y, sp - prev, h))
        prev = sp
    if w - prev > 3:
        sub_boxes.append((x + prev, y, w - prev, h))

    return sub_boxes if len(sub_boxes) > 1 else [box]


def _segment_image(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Segment characters in a BGR image.
    Returns list of (x, y, w, h) in pixel coordinates of the input image.
    """
    h_img, w_img = img.shape[:2]

    # Upscale for small text to improve segmentation
    MIN_PROC_H = 80
    scale = max(1.0, MIN_PROC_H / h_img) if h_img < MIN_PROC_H else 1.0
    proc_img = cv.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv.INTER_CUBIC) if scale > 1.0 else img

    gray = cv.cvtColor(proc_img, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Ensure foreground (characters) is white on black
    if np.mean(thresh) > 127:
        thresh = cv.bitwise_not(thresh)

    h_proc = proc_img.shape[0]
    close_k = max(1, int(h_proc * 0.025))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (close_k, close_k))
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    min_char_h = h_proc * 0.3
    min_char_w = 3

    boxes = []
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        if h >= min_char_h and w >= min_char_w:
            boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[0])

    # Merge overlapping boxes
    min_overlap_px = max(1, int(h_proc * 0.01))
    merged: List[Tuple] = []
    for box in boxes:
        if merged and box[0] < merged[-1][0] + merged[-1][2] - min_overlap_px:
            px, py, pw, ph = merged[-1]
            nx = min(px, box[0])
            ny = min(py, box[1])
            nx2 = max(px + pw, box[0] + box[2])
            ny2 = max(py + ph, box[1] + box[3])
            merged[-1] = (nx, ny, nx2 - nx, ny2 - ny)
        else:
            merged.append(box)

    # Split wide boxes
    if merged:
        widths = [b[2] for b in merged]
        median_w = float(np.median(widths))
        final_proc = []
        for box in merged:
            final_proc.extend(_split_wide_box(thresh, box, median_w))
    else:
        final_proc = merged

    # Scale boxes back to original image coordinates
    if scale > 1.0:
        final_boxes = [
            (int(x / scale), int(y / scale),
             max(1, int(w / scale)), max(1, int(h / scale)))
            for x, y, w, h in final_proc
        ]
    else:
        final_boxes = final_proc

    return final_boxes


# ──────────────────────────────────────── public API ──

def segment_region(
    image_path: Path,
    region: dict,
) -> List[dict]:
    """
    Segment characters inside a user-drawn region.

    Args:
        image_path: Absolute path to the image file.
        region: {x, y, w, h} all normalized 0-1 relative to full image.

    Returns:
        List of {id, x, y, w, h} normalized relative to full image.
    """
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img_h, img_w = img.shape[:2]

    # Convert region from normalized to pixel coords
    rx = int(region["x"] * img_w)
    ry = int(region["y"] * img_h)
    rw = int(region["w"] * img_w)
    rh = int(region["h"] * img_h)

    # Clamp to image bounds
    rx = max(0, min(rx, img_w - 1))
    ry = max(0, min(ry, img_h - 1))
    rw = min(rw, img_w - rx)
    rh = min(rh, img_h - ry)

    if rw <= 0 or rh <= 0:
        return []

    roi = img[ry:ry + rh, rx:rx + rw]
    boxes = _segment_image(roi)

    # Convert from ROI-local pixels to full-image normalized coords
    result = []
    for bx, by, bw, bh in boxes:
        full_x = (rx + bx) / img_w
        full_y = (ry + by) / img_h
        full_w = bw / img_w
        full_h = bh / img_h
        result.append({
            "id": str(uuid.uuid4()),
            "x": round(full_x, 6),
            "y": round(full_y, 6),
            "w": round(full_w, 6),
            "h": round(full_h, 6),
        })

    return result


def crop_segment(image_path: Path, seg: dict, padding: int = 2) -> Optional[np.ndarray]:
    """
    Crop a single character segment from an image.

    Args:
        image_path: Absolute path to image.
        seg: {x, y, w, h} normalized 0-1.
        padding: Extra pixels around crop.

    Returns:
        BGR numpy array or None if failed.
    """
    img = cv.imread(str(image_path))
    if img is None:
        return None

    img_h, img_w = img.shape[:2]
    px = max(0, int(seg["x"] * img_w) - padding)
    py = max(0, int(seg["y"] * img_h) - padding)
    pw = min(int(seg["w"] * img_w) + padding * 2, img_w - px)
    ph = min(int(seg["h"] * img_h) + padding * 2, img_h - py)

    if pw <= 0 or ph <= 0:
        return None

    return img[py:py + ph, px:px + pw]
