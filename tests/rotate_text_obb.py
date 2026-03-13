import cv2
import numpy as np
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


def compute_need_flip(cap_box: np.ndarray, text_box: np.ndarray, angle_deg: float) -> bool:
    """
    Kiểm tra xem có cần xoay thêm 180° không.
    Xoay quanh tâm cap → cap center giữ nguyên, chỉ text center dịch chuyển.
    """
    cx, cy, _, _, _ = cap_box
    tx, ty = text_box[0], text_box[1]

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    new_text = M @ np.array([tx, ty, 1])
    # Text nằm trên tâm cap (y nhỏ hơn) → ngược chiều → cần flip
    return new_text[1] < cy


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
    model = YOLOOBBInference("weights/yolo26_bottle_obb.onnx", class_names)
    image = cv2.imread('test_image/bottle6.jpg')

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

            # Kiểm tra need_flip (không cần xoay toàn ảnh, tính bằng ma trận)
            need_flip = compute_need_flip(cap_box, text_box, angle_deg)
            print(f"\n=== Kiểm tra chiều text ===")
            if need_flip:
                print("Text ngược chiều → Cần xoay thêm 180°")
            else:
                print("Text đúng chiều → Giữ nguyên")

            # Chỉ xoay vùng nắp chai (circular mask), background giữ nguyên
            MARGIN = 20  # pixel margin quanh cap
            result_crop, full_result = rotate_cap_region_only(
                image, cap_box, angle_deg, need_flip, margin=MARGIN
            )
            print(f"\nXoay tổng: {angle_deg:.1f}° + {180 if need_flip else 0}°")

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
