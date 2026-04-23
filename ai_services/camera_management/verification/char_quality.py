"""
Character-level Quality Analysis

Helpers for:
  - Segmenting characters from a text region crop
  - Per-character similarity / defect analysis against a template
  - Rendering the side-by-side comparison strip for debugging

Used by TextVerificationService's similarity check pipeline.
"""

import base64
from typing import List

import cv2
import numpy as np


# ── Character segmentation ──────────────────────────────────────────────────

def split_wide_box(thresh_roi: np.ndarray, box: tuple, median_w: float) -> list:
    """Split a box that is too wide using vertical projection."""
    x, y, w, h = box
    if w < median_w * 1.5:
        return [box]
    roi = thresh_roi[y:y + h, x:x + w]
    v_proj = np.sum(roi, axis=0) / 255
    n_chars = max(2, round(w / median_w))
    split_points = []
    for i in range(1, n_chars):
        expected_x = int(i * w / n_chars)
        search_w = max(3, int(median_w * 0.3))
        left = max(1, expected_x - search_w)
        right = min(w - 1, expected_x + search_w)
        if left < right:
            window = v_proj[left:right]
            best = left + np.argmin(window)
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


def segment_characters_from_image(img: np.ndarray) -> List[np.ndarray]:
    """
    Segment characters from a BGR image.
    Returns list of cropped char images (BGR), sorted left-to-right.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = img.shape[:2]
    min_char_height = h_img * 0.3
    min_char_width = 3

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= min_char_height and w >= min_char_width:
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[0])

    merged = []
    for box in boxes:
        if merged and box[0] < merged[-1][0] + merged[-1][2]:
            px, py, pw, ph = merged[-1]
            nx = min(px, box[0])
            ny = min(py, box[1])
            nx2 = max(px + pw, box[0] + box[2])
            ny2 = max(py + ph, box[1] + box[3])
            merged[-1] = (nx, ny, nx2 - nx, ny2 - ny)
        else:
            merged.append(box)

    if merged:
        widths = [b[2] for b in merged]
        median_w = float(np.median(widths))
        final_boxes = []
        for box in merged:
            sub = split_wide_box(thresh, box, median_w)
            final_boxes.extend(sub)
    else:
        final_boxes = merged

    padding = 2
    char_imgs = []
    for (x, y, w, h) in final_boxes:
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(w_img, x + w + padding), min(h_img, y + h + padding)
        char_imgs.append(img[y1:y2, x1:x2])

    return char_imgs


# ── Low-level shape ops ─────────────────────────────────────────────────────

def tight_crop(thresh: np.ndarray) -> np.ndarray:
    """Crop to foreground bounding box, removing excess padding."""
    coords = np.where(thresh > 0)
    if len(coords[0]) == 0:
        return thresh
    y0, y1 = int(coords[0].min()), int(coords[0].max())
    x0, x1 = int(coords[1].min()), int(coords[1].max())
    return thresh[y0:y1 + 1, x0:x1 + 1]


def deskew_char(thresh_char: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Rotate upright. Skip if angle > max_angle to avoid wrong rotation on L/J/F."""
    coords = np.column_stack(np.where(thresh_char > 0))
    if len(coords) < 10:
        return thresh_char
    angle = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))[2]
    if angle < -45:
        angle += 90
    if abs(angle) > max_angle:
        return thresh_char
    h, w = thresh_char.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(thresh_char, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)


def fit_to_square(img: np.ndarray, size: tuple) -> np.ndarray:
    """Resize keeping aspect ratio, pad with black. Prevents I/l/1 distortion."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size[1], size[0]), dtype=np.uint8)
    scale = min(size[0] / w, size[1] / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size[1], size[0]), dtype=np.uint8)
    yo = (size[1] - nh) // 2
    xo = (size[0] - nw) // 2
    canvas[yo:yo + nh, xo:xo + nw] = resized
    return canvas


def center_by_centroid(mask: np.ndarray, size: tuple) -> np.ndarray:
    """Place mask in canvas so its centroid is centered."""
    H, W = size[1], size[0]
    canvas = np.zeros((H, W), dtype=np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return canvas
    cy, cx = float(ys.mean()), float(xs.mean())
    h, w = mask.shape
    yo = int(round(H / 2 - cy))
    xo = int(round(W / 2 - cx))
    y1s, y1e = max(0, yo), min(H, yo + h)
    x1s, x1e = max(0, xo), min(W, xo + w)
    y2s = max(0, -yo)
    x2s = max(0, -xo)
    y2e = y2s + (y1e - y1s)
    x2e = x2s + (x1e - x1s)
    if y1e > y1s and x1e > x1s:
        canvas[y1s:y1e, x1s:x1e] = mask[y2s:y2e, x2s:x2e]
    return canvas


def iou_binary(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two same-size binary masks."""
    inter = int(np.count_nonzero((a > 0) & (b > 0)))
    union = int(np.count_nonzero((a > 0) | (b > 0)))
    return inter / union if union > 0 else 0.0


# ── Per-character quality scoring ───────────────────────────────────────────

def char_quality(tmpl_char: np.ndarray, tgt_char: np.ndarray, size: tuple = (64, 64)) -> dict:
    """
    Per-character quality analysis (ported from test_segment.py).

    Pipeline: BGR → thresh → tight_crop → deskew → fit_to_square
    Metrics:
      1. Blurred Template Matching (TM_CCOEFF_NORMED, σ=1.2, multi-scale)
      2. IoU after centroid alignment
      3. Pixel ratio confidence
    tm_conf = max(blur_TM, IoU). final confidence = min(tm_conf, pixel_conf).
    Defect detection retained for production use.
    """
    def _to_thresh(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        # Blur mạnh hơn để normalize stroke width trước Otsu
        # (3,3) quá nhỏ → chữ mờ và chữ nét cho mask khác nhau nhiều, đặc biệt B/D/O)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if np.mean(th) > 127:
            th = cv2.bitwise_not(th)
        # Morphological closing: lấp holes nhỏ do stroke mỏng, chuẩn hóa enclosed regions (B, D, O)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k_close, iterations=1)
        return th

    tmpl_thresh = _to_thresh(tmpl_char)
    tgt_thresh = _to_thresh(tgt_char)

    t1 = fit_to_square(deskew_char(tight_crop(tmpl_thresh)), size)
    g2_deskewed = deskew_char(tight_crop(tgt_thresh))
    t2_base = fit_to_square(g2_deskewed, size)

    # (1) Pixel confidence
    px1 = int(np.count_nonzero(t1))
    px2 = int(np.count_nonzero(t2_base))
    pixel_ratio = px2 / (px1 + 1e-6)
    deviation = abs(pixel_ratio - 1.0)
    pixel_conf = float(np.clip(1.0 - deviation * (1.0 / 1.4), 0.0, 1.0))

    # (2) Blurred multi-scale Template Matching
    t1_blur = cv2.GaussianBlur(t1.astype(np.float32), (0, 0), sigmaX=1.2)
    best_tm = 0.0
    for scale in [0.85, 0.92, 1.0, 1.08, 1.15]:
        s = (max(t1.shape[1], int(size[0] * scale)),
             max(t1.shape[0], int(size[1] * scale)))
        t2 = fit_to_square(g2_deskewed, s)
        t2_blur = cv2.GaussianBlur(t2.astype(np.float32), (0, 0), sigmaX=1.2)
        result = cv2.matchTemplate(t2_blur, t1_blur, cv2.TM_CCOEFF_NORMED)
        best_tm = max(best_tm, float(result.max()))
    blur_tm = float(np.clip(best_tm, 0.0, 1.0))

    # (3) IoU after centroid alignment
    # Dilation 5×5 để tha thứ stroke width khác nhau (B/D/O có enclosed regions nhạy cảm)
    a = center_by_centroid(tight_crop(t1), size)
    b = center_by_centroid(tight_crop(t2_base), size)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    a = cv2.dilate(a, k, iterations=1)
    b = cv2.dilate(b, k, iterations=1)
    iou = iou_binary(a, b)

    tm_conf = max(blur_tm, iou)
    confidence = min(tm_conf, pixel_conf)

    # Sharpness from BGR for LEM detection
    g1 = cv2.cvtColor(tmpl_char, cv2.COLOR_BGR2GRAY) if len(tmpl_char.shape) == 3 else tmpl_char
    g2 = cv2.cvtColor(tgt_char, cv2.COLOR_BGR2GRAY) if len(tgt_char.shape) == 3 else tgt_char
    g1 = cv2.resize(g1, size)
    g2 = cv2.resize(g2, size)
    sharp_t = float(np.var(cv2.Laplacian(g1, cv2.CV_64F)))
    sharp_g = float(np.var(cv2.Laplacian(g2, cv2.CV_64F)))
    sharp_ratio = sharp_g / (sharp_t + 1e-8)

    # Connected components for GAY_NET — tính trên tight_crop của thresh gốc,
    # KHÔNG trên t1/t2_base (đã qua fit_to_square INTER_NEAREST → stroke mỏng bị đứt giả tạo)
    cc_t = cv2.connectedComponents(tight_crop(tmpl_thresh))[0] - 1
    cc_g = cv2.connectedComponents(tight_crop(tgt_thresh))[0] - 1

    # Defect detection
    defects = []
    if confidence < 0.5:
        defects.append("SAI_KY_TU")
    if sharp_ratio < 0.25:
        defects.append("LEM")
    if pixel_ratio > 1.3 and iou > 0.6:
        defects.append("IN_CHONG")
    if cc_g > cc_t + 2:
        defects.append("GAY_NET")
    if pixel_ratio < 0.5:
        defects.append("MAT_NET")

    return {
        "confidence": round(confidence, 4),
        "tm_conf": round(tm_conf, 4),
        "blur_tm": round(blur_tm, 4),
        "iou": round(iou, 4),
        "pixel_conf": round(pixel_conf, 3),
        "px_tmpl": px1,
        "px_tgt": px2,
        "sharp_ratio": round(sharp_ratio, 4),
        "cc_tmpl": cc_t,
        "cc_tgt": cc_g,
        "defects": defects,
    }


# ── Debug image rendering ───────────────────────────────────────────────────

def save_char_comparison(
    tmpl_img, tgt_img, tmpl_chars, tgt_chars, char_results, output_path, conf_threshold: float = 0.75,
):
    """Save visual comparison strip — layout identical to test_segment.py:
      Row 1: ORIG TEMPLATE  |  ORIG TARGET        (side by side)
      Row 2: THRESH TEMPLATE | THRESH TARGET       (side by side)
      Row 3: template chars  (80×80 per char)
      Row 4: target chars    (80×80, green/red border)
      Row 5: confidence + PASS/FAIL per char
      Banner: overall PASS/FAIL
    """
    n = len(char_results)
    if n == 0:
        return

    cell_w, cell_h = 80, 80
    gap = 4
    score_h = 40
    strip_w = n * (cell_w + gap) + gap

    orig_h = 120
    orig_section_h = (orig_h + gap) * 2 + gap
    char_strip_h = 2 * (cell_h + gap) + score_h + gap * 2
    total_h = orig_section_h + char_strip_h
    total_w = max(strip_w, 200)

    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 50

    def place_image(src, y_start, x_start, w, h):
        img_bgr = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR) if len(src.shape) == 2 else src
        ih, iw = img_bgr.shape[:2]
        scale = w / iw
        nw = w
        nh = min(h, max(1, int(ih * scale)))
        resized = cv2.resize(img_bgr, (nw, nh))
        canvas[y_start:y_start + nh, x_start:x_start + nw] = resized

    def make_full_thresh(bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if len(bgr.shape) == 3 else bgr
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if np.mean(th) > 127:
            th = cv2.bitwise_not(th)
        return th

    def fit_char(src, w, h):
        img = src if len(src.shape) == 3 else cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        ih, iw = img.shape[:2]
        scale = min(w / iw, h / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = cv2.resize(img, (nw, nh))
        cell = np.ones((h, w, 3), dtype=np.uint8) * 50
        yo = (h - nh) // 2
        xo = (w - nw) // 2
        cell[yo:yo + nh, xo:xo + nw] = resized
        return cell

    tmpl_thresh = make_full_thresh(tmpl_img)
    tgt_thresh = make_full_thresh(tgt_img)
    half_w = total_w // 2

    # Row 1: original images side by side
    y0 = gap
    place_image(tmpl_img, y0, 0, half_w, orig_h)
    place_image(tgt_img, y0, half_w, half_w, orig_h)
    cv2.line(canvas, (half_w, y0), (half_w, y0 + orig_h), (120, 120, 120), 1)
    cv2.putText(canvas, "ORIG TEMPLATE", (4, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(canvas, "ORIG TARGET", (half_w + 4, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # Row 2: thresh images side by side
    y1 = y0 + orig_h + gap
    place_image(tmpl_thresh, y1, 0, half_w, orig_h)
    place_image(tgt_thresh, y1, half_w, half_w, orig_h)
    cv2.line(canvas, (half_w, y1), (half_w, y1 + orig_h), (120, 120, 120), 1)
    cv2.putText(canvas, "THRESH TEMPLATE", (4, y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(canvas, "THRESH TARGET", (half_w + 4, y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # Char rows
    y_base = orig_section_h
    cv2.putText(canvas, "TEMPLATE", (2, y_base + gap + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(canvas, "TARGET", (2, y_base + gap * 2 + cell_h + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    for idx, cr in enumerate(char_results):
        x_off = gap + idx * (cell_w + gap)
        if x_off + cell_w > total_w:
            break
        i = cr["idx"]
        is_pass = cr["confidence"] >= conf_threshold
        color = (0, 220, 0) if is_pass else (0, 0, 255)

        # Template char row
        canvas[y_base + gap:y_base + gap + cell_h, x_off:x_off + cell_w] = fit_char(tmpl_chars[i], cell_w, cell_h)

        # Target char row + border
        tgt_y0 = y_base + gap * 2 + cell_h
        tgt_y1 = tgt_y0 + cell_h
        canvas[tgt_y0:tgt_y1, x_off:x_off + cell_w] = fit_char(tgt_chars[i], cell_w, cell_h)
        cv2.rectangle(canvas, (x_off, tgt_y0), (x_off + cell_w, tgt_y1), color, 2)

        # Score row
        y_text = y_base + gap * 3 + cell_h * 2
        cv2.putText(canvas, f"{cr['confidence']:.2f}", (x_off + 2, y_text + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        label = "PASS" if is_pass else "FAIL"
        cv2.putText(canvas, label, (x_off + 2, y_text + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Overall PASS/FAIL banner
    overall_pass = all(cr["confidence"] >= conf_threshold for cr in char_results)
    banner_color = (0, 180, 0) if overall_pass else (0, 0, 220)
    banner_label = "PASS" if overall_pass else "FAIL"
    banner_h = 28
    banner = np.ones((banner_h, total_w, 3), dtype=np.uint8)
    banner[:] = banner_color
    text_size = cv2.getTextSize(banner_label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
    tx = (total_w - text_size[0]) // 2
    cv2.putText(banner, banner_label, (tx, 21), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    canvas = np.vstack([canvas, banner])

    cv2.imwrite(output_path, canvas)


def img_to_b64(img: np.ndarray) -> str:
    """Encode BGR image to base64 PNG string for FE display."""
    _, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf).decode('utf-8')
