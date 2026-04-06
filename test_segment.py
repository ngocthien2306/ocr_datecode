import cv2 as cv
import numpy as np
import sys
import os

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def split_wide_box(thresh_roi, box, median_w):
    """Split a box that is too wide using vertical projection."""
    x, y, w, h = box
    if w < median_w * 1.5:
        return [box]

    roi = thresh_roi[y:y+h, x:x+w]
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


def segment_characters(image_path, output_dir="output_chars", save=True):
    if save:
        os.makedirs(output_dir, exist_ok=True)

    img = cv.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image '{image_path}'")
        return None, None, None, None, None

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    # Chuẩn hóa hướng: foreground (chữ) luôn là trắng trên nền đen
    if np.mean(thresh) > 127:
        thresh = cv.bitwise_not(thresh)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    h_img, w_img = img.shape[:2]
    min_char_height = h_img * 0.3
    min_char_width = 3

    boxes = []
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
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

    if len(merged) > 0:
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
    thresh_char_imgs = []
    for i, (x, y, w, h) in enumerate(final_boxes):
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_img, x + w + padding)
        y2 = min(h_img, y + h + padding)
        char_img = img[y1:y2, x1:x2]
        char_imgs.append(char_img)
        thresh_char_imgs.append(thresh[y1:y2, x1:x2])
        if save:
            cv.imwrite(os.path.join(output_dir, f"char_{i}.png"), char_img)

    if save:
        result = img.copy()
        for i, (x, y, w, h) in enumerate(final_boxes):
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w_img, x + w + padding)
            y2 = min(h_img, y + h + padding)
            cv.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv.putText(result, str(i), (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv.imwrite(os.path.join(output_dir, "result.png"), result)
        cv.imwrite(os.path.join(output_dir, "thresh.png"), thresh)

    return final_boxes, char_imgs, thresh_char_imgs, img, thresh


def deskew_char(thresh_char, max_angle=15):
    """Xoay ký tự về đứng, chỉ sửa nếu góc lệch nhỏ (< max_angle độ).
    Bỏ qua nếu góc lớn — tránh xoay sai với chữ bất đối xứng như L, J, F."""
    coords = np.column_stack(np.where(thresh_char > 0))
    if len(coords) < 10:
        return thresh_char
    angle = cv.minAreaRect(coords[:, ::-1].astype(np.float32))[2]
    if angle < -45:
        angle += 90
    if abs(angle) > max_angle:
        return thresh_char  # góc quá lớn → không tin cậy, giữ nguyên
    h, w = thresh_char.shape
    M = cv.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv.warpAffine(thresh_char, M, (w, h),
                         flags=cv.INTER_NEAREST, borderValue=0)


def tight_crop(thresh):
    """Crop sát foreground pixels, loại bỏ padding thừa."""
    coords = np.where(thresh > 0)
    if len(coords[0]) == 0:
        return thresh
    y0, y1 = int(coords[0].min()), int(coords[0].max())
    x0, x1 = int(coords[1].min()), int(coords[1].max())
    return thresh[y0:y1 + 1, x0:x1 + 1]


def fit_to_square(img, size):
    """Resize giữ aspect ratio, pad đen để thành vuông size×size.
    Quan trọng cho ký tự hẹp như I, l, 1, j — tránh bị stretch méo."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size[1], size[0]), dtype=np.uint8)
    scale = min(size[0] / w, size[1] / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv.resize(img, (nw, nh), interpolation=cv.INTER_NEAREST)
    canvas = np.zeros((size[1], size[0]), dtype=np.uint8)
    yo = (size[1] - nh) // 2
    xo = (size[0] - nw) // 2
    canvas[yo:yo + nh, xo:xo + nw] = resized
    return canvas


def _center_by_centroid(mask, size):
    """Đặt mask vào khung size×size sao cho khối tâm foreground nằm giữa khung."""
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
    y2s, y2e = max(0, -yo), max(0, -yo) + (y1e - y1s)
    x2s, x2e = max(0, -xo), max(0, -xo) + (x1e - x1s)
    if y1e > y1s and x1e > x1s:
        canvas[y1s:y1e, x1s:x1e] = mask[y2s:y2e, x2s:x2e]
    return canvas


def _iou(a, b):
    """IoU của 2 mask nhị phân cùng kích thước."""
    inter = int(np.count_nonzero((a > 0) & (b > 0)))
    union = int(np.count_nonzero((a > 0) | (b > 0)))
    return inter / union if union > 0 else 0.0


def compute_char_quality(tmpl_char, tgt_char, size=(64, 64)):
    """
    So sánh chất lượng 2 ký tự bằng 3 metric, robust với binary shapes:
    1. Pixel confidence: ratio px_tgt/px_tmpl
    2. Blurred Template Matching: gaussian blur trước TM → tha thứ lệch sub-pixel & stroke width
    3. IoU sau centroid alignment: chuẩn shape comparison cho ảnh nhị phân
    tm_conf = max(blur_TM, IoU) → metric nào "giỏi" thắng, không bị kéo xuống.
    """
    def _to_thresh_norm(raw):
        """Blur(5,5) + Otsu + morphClose — normalize stroke width cho B/D/O."""
        blurred = cv.GaussianBlur(raw, (5, 5), 0)
        _, th = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        if np.mean(th) > 127:
            th = cv.bitwise_not(th)
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        return cv.morphologyEx(th, cv.MORPH_CLOSE, k, iterations=1)

    tmpl_char = _to_thresh_norm(tmpl_char)
    tgt_char  = _to_thresh_norm(tgt_char)

    t1 = fit_to_square(deskew_char(tight_crop(tmpl_char)), size)
    g2_deskewed = deskew_char(tight_crop(tgt_char))
    t2_base = fit_to_square(g2_deskewed, size)

    # --- (1) Pixel confidence ---
    px1 = int(np.count_nonzero(t1))
    px2 = int(np.count_nonzero(t2_base))
    ratio = px2 / (px1 + 1e-6)
    deviation = abs(ratio - 1.0)
    pixel_conf = float(np.clip(1.0 - deviation * (1.0 / 1.4), 0.0, 1.0))

    # --- (2) Blurred multi-scale Template Matching ---
    # Blur biến binary → soft mask, giúp TM chịu được lệch 1–2 px và stroke khác nhau.
    t1_blur = cv.GaussianBlur(t1.astype(np.float32), (0, 0), sigmaX=1.2)
    best_tm = 0.0
    for scale in [0.85, 0.92, 1.0, 1.08, 1.15]:
        s = (max(t1.shape[1], int(size[0] * scale)),
             max(t1.shape[0], int(size[1] * scale)))
        t2 = fit_to_square(g2_deskewed, s)
        t2_blur = cv.GaussianBlur(t2.astype(np.float32), (0, 0), sigmaX=1.2)
        result = cv.matchTemplate(t2_blur, t1_blur, cv.TM_CCOEFF_NORMED)
        best_tm = max(best_tm, float(result.max()))
    blur_tm = float(np.clip(best_tm, 0.0, 1.0))

    # --- (3) IoU sau centroid alignment ---
    a = _center_by_centroid(tight_crop(t1),       size)
    b = _center_by_centroid(tight_crop(t2_base),  size)
    # Dilation 5×5 để tha thứ stroke width khác nhau (B/D/O nhạy cảm với enclosed regions)
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    a = cv.dilate(a, k, iterations=1)
    b = cv.dilate(b, k, iterations=1)
    iou = _iou(a, b)

    tm_conf = max(blur_tm, iou)

    return {
        "confidence": min(tm_conf, pixel_conf),
        "tm_conf": round(tm_conf, 4),
        "blur_tm": round(blur_tm, 4),
        "iou": round(iou, 4),
        "pixel_conf": round(pixel_conf, 3),
        "px_tmpl": px1,
        "px_tgt": px2,
    }


def compare_template_target(template_path, target_path, output_dir="output_compare"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Segmenting template: {template_path} ===")
    tmpl_dir = os.path.join(output_dir, "template")
    tgt_dir = os.path.join(output_dir, "target")
    tmpl_boxes, tmpl_chars, tmpl_thresh_chars, tmpl_img, tmpl_thresh = segment_characters(template_path, tmpl_dir, save=True)
    print(f"  Found {len(tmpl_boxes)} chars")

    print(f"\n=== Segmenting target: {target_path} ===")
    tgt_boxes, tgt_chars, tgt_thresh_chars, tgt_img, tgt_thresh = segment_characters(target_path, tgt_dir, save=True)
    print(f"  Found {len(tgt_boxes)} chars")

    n_pairs = min(len(tmpl_chars), len(tgt_chars))
    if n_pairs == 0:
        print("No characters to compare.")
        return

    print(f"\n=== Character Quality Comparison ({n_pairs} pairs) ===")
    print(f"{'Char':<6} {'Final':>8} {'TM':>8} {'BlurTM':>8} {'IoU':>8} {'PxConf':>8} {'PxTmpl':>8} {'PxTgt':>8}  Quality")
    print("-" * 85)

    results = []
    for i in range(n_pairs):
        metrics = compute_char_quality(tmpl_thresh_chars[i], tgt_thresh_chars[i])
        conf       = metrics["confidence"]
        tm_conf    = metrics["tm_conf"]
        blur_tm    = metrics["blur_tm"]
        iou        = metrics["iou"]
        pixel_conf = metrics["pixel_conf"]

        quality = "PASS" if conf >= 0.75 else "FAIL"

        print(f"  {i:<4} {conf:>8.4f} {tm_conf:>8.4f} {blur_tm:>8.4f} {iou:>8.4f} {pixel_conf:>8.3f} {metrics['px_tmpl']:>8} {metrics['px_tgt']:>8}  {quality}")
        results.append((i, metrics, quality))

    overall = "PASS" if all(r[2] == "PASS" for r in results) else "FAIL"
    print(f"\n  Overall: {overall}")

    if len(tmpl_chars) != len(tgt_chars):
        print(f"\n  WARNING: template has {len(tmpl_chars)} chars, target has {len(tgt_chars)} chars")
        print(f"  Only first {n_pairs} pairs compared.")

    # Build visual comparison strip
    _save_comparison_strip(tmpl_chars, tgt_chars, results, output_dir,
                           tmpl_img, tmpl_thresh, tgt_img, tgt_thresh)
    print(f"\nComparison strip saved to: {output_dir}/comparison.png")


def _save_comparison_strip(tmpl_chars, tgt_chars, results, output_dir,
                           tmpl_img=None, tmpl_thresh=None, tgt_img=None, tgt_thresh=None):
    """Create a visual strip: orig row / thresh row / template chars / target chars / scores."""
    n = len(results)
    cell_w, cell_h = 80, 80
    gap = 4
    score_h = 40
    strip_w = n * (cell_w + gap) + gap

    orig_h = 120
    has_orig = tmpl_img is not None
    orig_section_h = (orig_h + gap) * 2 + gap if has_orig else 0
    char_strip_h = 2 * (cell_h + gap) + score_h + gap * 2
    total_h = orig_section_h + char_strip_h
    total_w = strip_w

    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 50

    def place_image(src, y_start, x_start, w, h):
        if src is None:
            return
        img_bgr = cv.cvtColor(src, cv.COLOR_GRAY2BGR) if len(src.shape) == 2 else src
        ih, iw = img_bgr.shape[:2]
        # Scale theo width, cap height → không bị khoảng trắng trên/dưới
        scale = w / iw
        nw = w
        nh = min(h, max(1, int(ih * scale)))
        resized = cv.resize(img_bgr, (nw, nh))
        canvas[y_start:y_start + nh, x_start:x_start + nw] = resized

    if has_orig:
        half_w = strip_w // 2

        # Row 1: original images
        y0 = gap
        place_image(tmpl_img,    y0, 0,      half_w, orig_h)
        place_image(tgt_img,     y0, half_w, half_w, orig_h)
        cv.line(canvas, (half_w, y0), (half_w, y0 + orig_h), (120, 120, 120), 1)
        cv.putText(canvas, "ORIG TEMPLATE", (4, y0 + 12),
                   cv.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv.putText(canvas, "ORIG TARGET", (half_w + 4, y0 + 12),
                   cv.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # Row 2: thresh images
        y1 = y0 + orig_h + gap
        place_image(tmpl_thresh, y1, 0,      half_w, orig_h)
        place_image(tgt_thresh,  y1, half_w, half_w, orig_h)
        cv.line(canvas, (half_w, y1), (half_w, y1 + orig_h), (120, 120, 120), 1)
        cv.putText(canvas, "THRESH TEMPLATE", (4, y1 + 12),
                   cv.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv.putText(canvas, "THRESH TARGET", (half_w + 4, y1 + 12),
                   cv.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    def fit_char(src, w, h):
        """Resize char keeping aspect ratio, pad remainder with canvas background (50)."""
        ih, iw = src.shape[:2]
        scale = min(w / iw, h / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = cv.resize(src, (nw, nh))
        cell = np.ones((h, w, 3), dtype=np.uint8) * 50
        yo = (h - nh) // 2
        xo = (w - nw) // 2
        cell[yo:yo + nh, xo:xo + nw] = resized
        return cell

    # Char comparison rows (offset below orig section)
    y_base = orig_section_h

    for idx, (i, metrics, quality) in enumerate(results):
        x_off = gap + idx * (cell_w + gap)

        # Template char
        t = fit_char(tmpl_chars[i], cell_w, cell_h)
        canvas[y_base + gap:y_base + gap + cell_h, x_off:x_off + cell_w] = t

        # Target char
        tgt = fit_char(tgt_chars[i], cell_w, cell_h)
        canvas[y_base + gap * 2 + cell_h:y_base + gap * 2 + cell_h * 2, x_off:x_off + cell_w] = tgt

        # Score text + PASS/FAIL border
        is_pass = quality == "PASS"
        color = (0, 220, 0) if is_pass else (0, 0, 255)
        y_text = y_base + gap * 3 + cell_h * 2

        # Vẽ border đỏ/xanh quanh target char khi FAIL/PASS
        tgt_y0 = y_base + gap * 2 + cell_h
        tgt_y1 = tgt_y0 + cell_h
        cv.rectangle(canvas, (x_off, tgt_y0), (x_off + cell_w, tgt_y1), color, 2)

        cv.putText(canvas, f"{metrics['confidence']:.2f}", (x_off + 2, y_text + 14),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv.putText(canvas, quality, (x_off + 2, y_text + 28),
                   cv.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Labels for char rows
    cv.putText(canvas, "TEMPLATE", (2, y_base + gap + 12),
               cv.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv.putText(canvas, "TARGET", (2, y_base + gap * 2 + cell_h + 12),
               cv.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # Overall PASS/FAIL banner
    overall_pass = all(r[2] == "PASS" for r in results)
    banner_color = (0, 180, 0) if overall_pass else (0, 0, 220)
    banner_label = "PASS" if overall_pass else "FAIL"
    banner_h = 28
    banner = np.ones((banner_h, total_w, 3), dtype=np.uint8)
    banner[:] = banner_color
    text_size = cv.getTextSize(banner_label, cv.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
    tx = (total_w - text_size[0]) // 2
    cv.putText(banner, banner_label, (tx, 21), cv.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    canvas = np.vstack([canvas, banner])

    cv.imwrite(os.path.join(output_dir, "comparison.png"), canvas)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Single image mode
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            sys.exit(1)
        boxes, chars, _, _, _ = segment_characters(image_path)
        if boxes:
            print(f"Found {len(boxes)} characters")
            for i, (x, y, w, h) in enumerate(boxes):
                print(f"  char_{i}: x={x}, y={y}, w={w}, h={h}")

    elif len(sys.argv) == 3:
        # Compare mode: template vs target
        template_path, target_path = sys.argv[1], sys.argv[2]
        for p in [template_path, target_path]:
            if not os.path.exists(p):
                print(f"File not found: {p}")
                sys.exit(1)
        compare_template_target(template_path, target_path)

    else:
        print("Usage:")
        print("  Segment only:  python test_segment.py <image>")
        print("  Compare:       python test_segment.py <template> <target>")
        sys.exit(1)

# /home/suntech/Source/ocr_datecode/ai_services/test_result/cropped_region_40767173_3_800_1775135412.png
# /home/suntech/Source/ocr_datecode/ai_services/test_result/cropped_region_40767173_3_815_1775135588.png
# data/test_result_1/template_crop_40767169_0.png data/test_result_1/cropped_region_40767169_2_435_1775466274.png