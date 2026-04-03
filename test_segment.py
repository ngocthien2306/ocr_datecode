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
        return None, None

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
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
    for i, (x, y, w, h) in enumerate(final_boxes):
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_img, x + w + padding)
        y2 = min(h_img, y + h + padding)
        char_img = img[y1:y2, x1:x2]
        char_imgs.append(char_img)
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

    return final_boxes, char_imgs


def compute_char_quality(template_char, target_char, size=(64, 64)):
    """
    So sánh chất lượng 2 ký tự bằng SSIM, MSE, PSNR.
    Cả hai ảnh được resize về cùng kích thước trước khi so sánh.
    Returns: dict với các metric
    """
    t1 = cv.resize(cv.cvtColor(template_char, cv.COLOR_BGR2GRAY), size)
    t2 = cv.resize(cv.cvtColor(target_char, cv.COLOR_BGR2GRAY), size)

    mse = float(np.mean((t1.astype(np.float32) - t2.astype(np.float32)) ** 2))
    psnr = float('inf') if mse == 0 else 10 * np.log10(255 ** 2 / mse)

    if HAS_SKIMAGE:
        score_ssim = float(ssim(t1, t2, data_range=255))
    else:
        # Fallback: normalized cross-correlation
        n1 = t1.astype(np.float32) - t1.mean()
        n2 = t2.astype(np.float32) - t2.mean()
        denom = (np.std(n1) * np.std(n2) * size[0] * size[1]) + 1e-8
        score_ssim = float(np.sum(n1 * n2) / denom)

    return {"ssim": score_ssim, "mse": mse, "psnr": psnr}


def compare_template_target(template_path, target_path, output_dir="output_compare"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Segmenting template: {template_path} ===")
    tmpl_dir = os.path.join(output_dir, "template")
    tgt_dir = os.path.join(output_dir, "target")
    tmpl_boxes, tmpl_chars = segment_characters(template_path, tmpl_dir, save=True)
    print(f"  Found {len(tmpl_boxes)} chars")

    print(f"\n=== Segmenting target: {target_path} ===")
    tgt_boxes, tgt_chars = segment_characters(target_path, tgt_dir, save=True)
    print(f"  Found {len(tgt_boxes)} chars")

    n_pairs = min(len(tmpl_chars), len(tgt_chars))
    if n_pairs == 0:
        print("No characters to compare.")
        return

    print(f"\n=== Character Quality Comparison ({n_pairs} pairs) ===")
    print(f"{'Char':<6} {'SSIM':>8} {'MSE':>10} {'PSNR (dB)':>10}  Quality")
    print("-" * 50)

    results = []
    for i in range(n_pairs):
        metrics = compute_char_quality(tmpl_chars[i], tgt_chars[i])
        ssim_val = metrics["ssim"]
        mse_val = metrics["mse"]
        psnr_val = metrics["psnr"]

        if ssim_val >= 0.85:
            quality = "GOOD"
        elif ssim_val >= 0.65:
            quality = "FAIR"
        else:
            quality = "POOR"

        psnr_str = f"{psnr_val:.2f}" if psnr_val != float('inf') else "  inf"
        print(f"  {i:<4} {ssim_val:>8.4f} {mse_val:>10.2f} {psnr_str:>10}  {quality}")
        results.append((i, metrics, quality))

    if len(tmpl_chars) != len(tgt_chars):
        print(f"\n  WARNING: template has {len(tmpl_chars)} chars, target has {len(tgt_chars)} chars")
        print(f"  Only first {n_pairs} pairs compared.")

    # Build visual comparison strip
    _save_comparison_strip(tmpl_chars, tgt_chars, results, output_dir)
    print(f"\nComparison strip saved to: {output_dir}/comparison.png")
    print(f"Metric used for SSIM: {'skimage' if HAS_SKIMAGE else 'NCC fallback (pip install scikit-image for SSIM)'}")


def _save_comparison_strip(tmpl_chars, tgt_chars, results, output_dir):
    """Create a side-by-side visual strip: template row / target row / score row."""
    n = len(results)
    cell_w, cell_h = 80, 80
    gap = 4
    score_h = 40
    total_w = n * (cell_w + gap) + gap
    total_h = 2 * (cell_h + gap) + score_h + gap * 2

    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 50

    for idx, (i, metrics, quality) in enumerate(results):
        x_off = gap + idx * (cell_w + gap)

        # Template char
        t = cv.resize(tmpl_chars[i], (cell_w, cell_h))
        canvas[gap:gap+cell_h, x_off:x_off+cell_w] = t

        # Target char — tint by quality
        tgt = cv.resize(tgt_chars[i], (cell_w, cell_h))
        tint = tgt.copy()
        if quality == "POOR":
            tint[:, :, 0] = (tint[:, :, 0].astype(np.int32) * 0.4).clip(0, 255).astype(np.uint8)
            tint[:, :, 1] = (tint[:, :, 1].astype(np.int32) * 0.4).clip(0, 255).astype(np.uint8)
        elif quality == "FAIR":
            tint[:, :, 2] = (tint[:, :, 2].astype(np.int32) * 0.4).clip(0, 255).astype(np.uint8)

        y_off = gap * 2 + cell_h
        canvas[y_off:y_off+cell_h, x_off:x_off+cell_w] = tint

        # Score text
        color = (0, 220, 0) if quality == "GOOD" else (0, 165, 255) if quality == "FAIR" else (0, 0, 255)
        y_text = gap * 3 + cell_h * 2
        cv.putText(canvas, f"{metrics['ssim']:.2f}", (x_off + 2, y_text + 14),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv.putText(canvas, quality[:4], (x_off + 2, y_text + 28),
                   cv.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Labels
    cv.putText(canvas, "TEMPLATE", (2, gap + 12), cv.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv.putText(canvas, "TARGET", (2, gap * 2 + cell_h + 12), cv.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    cv.imwrite(os.path.join(output_dir, "comparison.png"), canvas)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Single image mode
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            sys.exit(1)
        boxes, chars = segment_characters(image_path)
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
