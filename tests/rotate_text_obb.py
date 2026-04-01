import cv2
import numpy as np
import time
import threading
from test_label_onnx import YOLOOBBInference
from rapidocr_onnxruntime import RapidOCR
import matplotlib.pyplot as plt

def rotate_cap_region_only(image: np.ndarray, cap_box: np.ndarray, angle_deg: float,
                            need_flip: bool = False, margin: int = 20) -> tuple:
    """
    Crop vùng nắp chai (radius + margin), xoay nội dung bên trong vòng tròn.
    - Dùng circular mask thay OBB rectangle → tránh background góc bị xoay theo
    - text_box tự xoay theo vì nằm bên trong vùng tròn
    Returns:
        result_crop: ảnh crop (cap + margin) với cap đã xoay, background ngoài tròn giữ nguyên
        full_result: full image với cap đã xoay tại chỗ
    """
    cx, cy, w, h, _ = cap_box
    radius = int(min(w, h) / 2)
    total_angle = angle_deg + (180 if need_flip else 0)

    # Vùng crop: hình vuông bao quanh cap + margin
    crop_r = radius + margin
    x1 = max(0, int(cx - crop_r))
    y1 = max(0, int(cy - crop_r))
    x2 = min(image.shape[1], int(cx + crop_r))
    y2 = min(image.shape[0], int(cy + crop_r))

    crop = image[y1:y2, x1:x2].copy()
    local_cx = float(cx - x1)
    local_cy = float(cy - y1)

    # Xoay crop quanh tâm cap (tọa độ local)
    M = cv2.getRotationMatrix2D((local_cx, local_cy), total_angle, 1.0)
    crop_rotated = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]),
                                   flags=cv2.INTER_LINEAR,
                                   borderValue=(114, 114, 114))

    # Circular mask: chỉ lấy vùng tròn của cap, không lấy góc OBB
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(local_cx), int(local_cy)), radius, 255, -1)

    # Blend: background crop gốc + vùng tròn đã xoay (text_box xoay theo tự động)
    result_crop = crop.copy()
    result_crop[mask > 0] = crop_rotated[mask > 0]

    # Paste lại vào full image
    full_result = image.copy()
    full_result[y1:y2, x1:x2] = result_crop

    return result_crop, full_result


def crop_text_region(cap_crop: np.ndarray, cap_box: np.ndarray,
                     text_box: np.ndarray, total_angle: float, margin: int = 20) -> np.ndarray:
    """
    Crop chỉ vùng text từ cap_crop đã xoay.
    Tính vị trí tâm text sau khi xoay để không cần chạy OCR trên toàn cap.
    """
    cx, cy, cap_w, cap_h, _ = cap_box
    crop_r = int(min(cap_w, cap_h) / 2) + margin
    x1_cap = max(0, int(cx - crop_r))
    y1_cap = max(0, int(cy - crop_r))

    tx, ty, tw, th = text_box[0], text_box[1], text_box[2], text_box[3]
    local_tx = tx - x1_cap
    local_ty = ty - y1_cap
    local_cx = float(cx - x1_cap)
    local_cy = float(cy - y1_cap)

    M = cv2.getRotationMatrix2D((local_cx, local_cy), total_angle, 1.0)
    new_center = M @ np.array([local_tx, local_ty, 1.0])

    pad = int(max(tw, th) * 0.15)
    half_w = int(max(tw, th) / 2) + pad
    half_h = int(min(tw, th) / 2) + pad

    nx, ny = int(new_center[0]), int(new_center[1])
    h_c, w_c = cap_crop.shape[:2]
    return cap_crop[max(0, ny - half_h):min(h_c, ny + half_h),
                    max(0, nx - half_w):min(w_c, nx + half_w)]


def detect_flip_by_density(text_crop: np.ndarray, threshold: float = 1.15) -> tuple[bool, str]:
    """
    Cách 1 – Pixel density comparison (~0.1ms).
    Phù hợp với date code 2 dòng không đều nhau:
      Dòng 1 "BEST BY DD MMM YYYY" (dài) phải ở trên.
      Dòng 2 "PL XXXXXX" (ngắn) ở dưới.
    Nếu nửa dưới dày đặc hơn → dòng dài đang ở dưới → cần flip.
    Returns (need_flip, confidence: 'sure'/'unsure')
    """
    gray = cv2.cvtColor(text_crop, cv2.COLOR_BGR2GRAY) if text_crop.ndim == 3 else text_crop.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h = binary.shape[0]
    upper = float(np.sum(binary[:h // 2]))
    lower = float(np.sum(binary[h // 2:]))
    if lower > upper * threshold:
        return True, 'sure'
    if upper > lower * threshold:
        return False, 'sure'
    return False, 'unsure'   # 2 dòng gần bằng nhau → không chắc


def detect_flip_by_gradient(text_crop: np.ndarray) -> tuple[bool, str]:
    """
    Cách 2 – Horizontal gradient / baseline detection (~0.5ms).
    Baseline của text (cạnh dưới ký tự) tạo SobelY dương (dark→light đi xuống).
    Cap-height (cạnh trên ký tự) tạo SobelY âm.
    Text đúng chiều: baseline ở nửa dưới → tổng SobelY dương của nửa dưới lớn hơn nửa trên.
    Hoạt động tốt nhất với text 1 dòng.
    """
    gray = cv2.cvtColor(text_crop, cv2.COLOR_BGR2GRAY) if text_crop.ndim == 3 else text_crop.copy()
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    h = sobel_y.shape[0]
    upper_pos = float(np.sum(sobel_y[:h // 2][sobel_y[:h // 2] > 0]))
    lower_pos = float(np.sum(sobel_y[h // 2:][sobel_y[h // 2:] > 0]))
    # Baseline ở nửa trên → dương nhiều ở trên → text ngược → cần flip
    ratio = upper_pos / (lower_pos + 1e-6)
    if ratio > 1.2:
        return True, 'sure'
    if ratio < 0.8:
        return False, 'sure'
    return False, 'unsure'


def ocr_confidence(crop: np.ndarray, ocr_engine) -> float:
    """Trả về tổng confidence OCR trên ảnh crop."""
    result, _ = ocr_engine(crop)
    if result is None:
        return 0.0
    return sum(float(line[2]) for line in result)


def pick_flip_by_ocr(cap_crop_0: np.ndarray, cap_crop_180: np.ndarray,
                     cap_box: np.ndarray, text_box: np.ndarray,
                     angle_deg: float, margin: int, ocr_engine) -> bool:
    """
    Chạy OCR song song trên 2 text crop (0° và 180°).
    Trả về True nếu chiều 180° cho confidence cao hơn.
    """
    text_crop_0   = crop_text_region(cap_crop_0,   cap_box, text_box, angle_deg,         margin)
    text_crop_180 = crop_text_region(cap_crop_180, cap_box, text_box, angle_deg + 180.0, margin)

    results = [0.0, 0.0]

    def run_ocr(idx, crop):
        results[idx] = ocr_confidence(crop, ocr_engine)

    t0   = threading.Thread(target=run_ocr, args=(0, text_crop_0))
    t180 = threading.Thread(target=run_ocr, args=(1, text_crop_180))
    t0.start();   t180.start()
    t0.join();    t180.join()

    print(f"  OCR conf 0°: {results[0]:.3f} | 180°: {results[1]:.3f}")
    return results[1] > results[0]


def rotate_image_by_obb(image: np.ndarray, box: np.ndarray):
    """
    Xoay ảnh để vùng text nằm chính diện dựa vào OBB
    box: [cx, cy, w, h, angle]
    """
    cx, cy, w, h, angle = box
    angle_deg = angle * 180 / np.pi

    if h > w:
        angle_deg += 90

    h_img, w_img = image.shape[:2]
    M = cv2.getRotationMatrix2D((w_img/2, h_img/2), angle_deg, 1.0)
    rotated = cv2.warpAffine(image, M, (w_img, h_img), flags=cv2.INTER_LINEAR, borderValue=(114,114,114))

    return rotated, angle_deg, M

def transform_boxes(boxes: np.ndarray, M: np.ndarray):
    """
    Transform boxes theo ma trận xoay
    boxes: [[cx, cy, w, h, angle], ...]
    M: Ma trận affine transform 2x3
    """
    transformed_boxes = []
    for box in boxes:
        cx, cy, w, h, angle = box

        # Transform điểm trung tâm
        center = np.array([cx, cy, 1])
        new_center = M @ center

        # Tính toán các điểm góc của OBB gốc
        rect = ((cx, cy), (w, h), angle * 180 / np.pi)
        box_points = cv2.boxPoints(rect)

        # Transform từng điểm góc
        transformed_points = []
        for point in box_points:
            pt = np.array([point[0], point[1], 1])
            new_pt = M @ pt
            transformed_points.append(new_pt[:2])

        transformed_points = np.array(transformed_points)

        # Tính lại OBB từ các điểm đã transform
        new_rect = cv2.minAreaRect(transformed_points.astype(np.float32))
        new_cx, new_cy = new_rect[0]
        new_w, new_h = new_rect[1]
        new_angle = new_rect[2] * np.pi / 180

        transformed_boxes.append([new_cx, new_cy, new_w, new_h, new_angle])

    return np.array(transformed_boxes)

def draw_obb(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, class_names: list):
    """
    Vẽ OBB lên ảnh
    boxes: [[cx, cy, w, h, angle], ...]
    """
    # Màu sắc cho từng class
    colors = {
        'bottle_cap': (255, 0, 0),    # Xanh dương - nắp chai
        'text_box': (0, 255, 0)       # Xanh lá - vùng text
    }

    img_draw = image.copy()
    for box, score, class_id in zip(boxes, scores, class_ids):
        cx, cy, w, h, angle = box
        class_name = class_names[int(class_id)]
        color = colors.get(class_name, (0, 255, 0))

        # Tính các điểm góc của OBB
        rect = ((cx, cy), (w, h), angle * 180 / np.pi)
        box_points = cv2.boxPoints(rect)
        box_points = np.int0(box_points)

        # Vẽ OBB
        cv2.drawContours(img_draw, [box_points], 0, color, 2)

        # Vẽ 4 điểm góc và đánh số thứ tự (chỉ cho text_box)
        if class_name == 'text_box':
            for i, point in enumerate(box_points):
                cv2.circle(img_draw, tuple(point), 8, (0, 0, 255), -1)
                cv2.putText(img_draw, str(i), (point[0]+10, point[1]+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Vẽ điểm trung tâm
        cv2.circle(img_draw, (int(cx), int(cy)), 5, (255, 0, 255), -1)

        # Vẽ class name và score
        label = f'{class_name}: {score:.2f}'
        cv2.putText(img_draw, label, (int(cx)-50, int(cy)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img_draw

def main(top_k=10):
    class_names = ['bottle_cap', 'text_box']
    model = YOLOOBBInference("weights/best_bottle_text.onnx", class_names)
    ocr = RapidOCR()
    image = cv2.imread('test_image/bottle9.jpg')

    results = model.predict([image], conf_threshold=0.4)
    boxes, scores, class_ids = results[0]
    print(results)
    # Sắp xếp theo score và lấy top K
    if len(boxes) > top_k:
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        boxes = boxes[sorted_indices]
        scores = scores[sorted_indices]
        class_ids = class_ids[sorted_indices]

    print(f"Results: {len(boxes)} boxes (top {top_k})")
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        print(f"Box {i}: {class_names[int(class_id)]} - score: {score:.3f}")

    if len(boxes) > 0:
        # Tìm text_box và bottle_cap
        text_box_idx = None
        bottle_cap_idx = None
        for i, class_id in enumerate(class_ids):
            if class_names[int(class_id)] == 'text_box' and text_box_idx is None:
                text_box_idx = i
            if class_names[int(class_id)] == 'bottle_cap' and bottle_cap_idx is None:
                bottle_cap_idx = i

        if text_box_idx is not None and bottle_cap_idx is not None:
            text_box = boxes[text_box_idx]
            cap_box = boxes[bottle_cap_idx]
            cx, cy, w, h, angle = text_box
            angle_deg = angle * 180 / np.pi

            print(f"\n=== Phân tích Text Box ===")
            print(f"Center: ({cx:.1f}, {cy:.1f})")
            print(f"Width: {w:.1f}, Height: {h:.1f}")
            print(f"Angle (rad): {angle:.4f}")
            print(f"Angle (deg): {angle_deg:.2f}°")
            if h > w:
                angle_deg += 90
                print(f"h > w → điều chỉnh angle: {angle_deg:.2f}°")

            # Vẽ OBB lên ảnh gốc
            original_with_obb = draw_obb(image, boxes, scores, class_ids, class_names)

            MARGIN = 20

            MARGIN = 20

            # --- Bước 1: xoay cap (chỉ cần 1 lần cho flip=False) ---
            t0 = time.perf_counter()
            crop_0, full_0 = rotate_cap_region_only(image, cap_box, angle_deg, False, margin=MARGIN)
            t_rotate = time.perf_counter() - t0

            # Crop vùng text nhỏ (dùng chung cho cả 3 cách)
            text_crop = crop_text_region(crop_0, cap_box, text_box, angle_deg, MARGIN)

            # --- Cách 1: Density comparison ---
            t0 = time.perf_counter()
            flip_density, conf_density = detect_flip_by_density(text_crop)
            t_density = time.perf_counter() - t0

            # --- Cách 2: Gradient analysis ---
            t0 = time.perf_counter()
            flip_gradient, conf_gradient = detect_flip_by_gradient(text_crop)
            t_gradient = time.perf_counter() - t0

            # --- Cách 3: OCR parallel (fallback khi 2 cách trên unsure) ---
            t0 = time.perf_counter()
            crop_180, full_180 = rotate_cap_region_only(image, cap_box, angle_deg, True, margin=MARGIN)
            text_crop_180 = crop_text_region(crop_180, cap_box, text_box, angle_deg + 180.0, MARGIN)
            flip_ocr = pick_flip_by_ocr(crop_0, crop_180, cap_box, text_box, angle_deg, MARGIN, ocr)
            t_ocr = time.perf_counter() - t0

            # --- Quyết định cuối: ưu tiên density nếu sure, fallback OCR ---
            if conf_density == 'sure':
                need_flip = flip_density
                method_used = 'density'
            elif conf_gradient == 'sure':
                need_flip = flip_gradient
                method_used = 'gradient'
            else:
                need_flip = flip_ocr
                method_used = 'ocr'

            result_crop = crop_180 if need_flip else crop_0
            full_result = full_180 if need_flip else full_0

            print(f"\n=== Flip detection benchmark ===")
            print(f"  [density]  need_flip={flip_density} ({conf_density})  {t_density*1000:.2f}ms")
            print(f"  [gradient] need_flip={flip_gradient} ({conf_gradient})  {t_gradient*1000:.2f}ms")
            print(f"  [ocr]      need_flip={flip_ocr}  {t_ocr*1000:.1f}ms")
            print(f"  → Dùng: [{method_used}] need_flip={need_flip}")
            print(f"\n[Timing]  rotate: {t_rotate*1000:.1f}ms")
            print(f"Xoay tổng: {angle_deg:.1f}° + {180 if need_flip else 0}°")

            # Hiển thị 3 ảnh: gốc | crop cap đã xoay | full image với cap đã xoay
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Ảnh gốc với OBB
            axes[0].imshow(cv2.cvtColor(original_with_obb, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f'Original (angle: {angle_deg:.2f}°)', fontsize=12)
            axes[0].axis('off')

            # Crop cap đã xoay (vòng tròn, background margin giữ nguyên)
            axes[1].imshow(cv2.cvtColor(result_crop, cv2.COLOR_BGR2RGB))
            title_crop = f'Cap crop (r+{MARGIN}px, circular mask)'
            axes[1].set_title(title_crop, fontsize=12)
            axes[1].axis('off')

            # Full image với cap đã xoay tại chỗ
            full_with_obb = draw_obb(full_result, boxes, scores, class_ids, class_names)
            axes[2].imshow(cv2.cvtColor(full_with_obb, cv2.COLOR_BGR2RGB))
            title_full = f'Full image cap rotated ({angle_deg:.1f}°{"+ 180°" if need_flip else ""})'
            axes[2].set_title(title_full, fontsize=12)
            axes[2].axis('off')

            plt.tight_layout()
            # plt.show()
            plt.savefig('rotate_text_obb_result.png')
        else:
            print("Không tìm thấy text_box hoặc bottle_cap")
    else:
        print("Không detect được box nào")

if __name__ == "__main__":

    main()
