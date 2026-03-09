import cv2
import numpy as np
from test_label_onnx import YOLOOBBInference
from rapidocr_onnxruntime import RapidOCR
import matplotlib.pyplot as plt

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
    image = cv2.imread('test_image/bottle2.jpg')

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

        if text_box_idx is not None:
            box = boxes[text_box_idx]
            cx, cy, w, h, angle = box
            angle_deg = angle * 180 / np.pi

            print(f"\n=== Phân tích Text Box ===")
            print(f"Center: ({cx:.1f}, {cy:.1f})")
            print(f"Width: {w:.1f}, Height: {h:.1f}")
            print(f"Angle (rad): {angle:.4f}")
            print(f"Angle (deg): {angle_deg:.2f}°")
            print(f"w > h: {w > h}")
            print(f"h > w: {h > w}")

            # Vẽ OBB lên ảnh gốc
            original_with_obb = draw_obb(image, boxes, scores, class_ids, class_names)

            # Xoay lần 1
            rotated, angle, M = rotate_image_by_obb(image, boxes[text_box_idx])

            # Transform boxes theo ma trận xoay
            transformed_boxes = transform_boxes(boxes, M)

            # Kiểm tra vị trí text_box so với bottle_cap sau khi xoay
            need_flip = False
            if bottle_cap_idx is not None:
                text_cy = transformed_boxes[text_box_idx][1]
                cap_cy = transformed_boxes[bottle_cap_idx][1]

                print(f"\n=== Kiểm tra vị trí sau xoay ===")
                print(f"Text box center Y: {text_cy:.1f}")
                print(f"Bottle cap center Y: {cap_cy:.1f}")

                # Nếu text nằm trên bottle cap (y nhỏ hơn) → xoay thêm 180°
                if text_cy < cap_cy:
                    need_flip = True
                    print("Text nằm TRÊN bottle cap → Cần xoay thêm 180°")
                else:
                    print("Text nằm DƯỚI bottle cap → Giữ nguyên")

            # Nếu cần xoay thêm 180°
            if need_flip:
                h_img, w_img = rotated.shape[:2]
                M_flip = cv2.getRotationMatrix2D((w_img/2, h_img/2), 180, 1.0)
                rotated = cv2.warpAffine(rotated, M_flip, (w_img, h_img),
                                        flags=cv2.INTER_LINEAR, borderValue=(114,114,114))
                transformed_boxes = transform_boxes(transformed_boxes, M_flip)
                print("Đã xoay thêm 180°")

            # Vẽ OBB lên ảnh đã xoay
            rotated_with_obb = draw_obb(rotated, transformed_boxes, scores, class_ids, class_names)

            # Vẽ đường tham chiếu
            if bottle_cap_idx is not None:
                cap_cy = int(transformed_boxes[bottle_cap_idx][1])
                cv2.line(rotated_with_obb, (0, cap_cy), (rotated.shape[1], cap_cy), (255, 0, 255), 2)

            print(f"\nXoay tổng: {angle:.1f}° + {180 if need_flip else 0}° = {angle + (180 if need_flip else 0):.1f}°")

            # Hiển thị 2 ảnh cạnh nhau
            fig, axes = plt.subplots(1, 2, figsize=(15, 8))

            # Ảnh gốc
            axes[0].imshow(cv2.cvtColor(original_with_obb, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f'Original (angle: {angle_deg:.2f}°)', fontsize=14)
            axes[0].axis('off')

            # Ảnh đã xoay
            axes[1].imshow(cv2.cvtColor(rotated_with_obb, cv2.COLOR_BGR2RGB))
            title = f'Rotated ({angle:.1f}°{"+ 180°" if need_flip else ""})'
            axes[1].set_title(title, fontsize=14)
            axes[1].axis('off')

            plt.tight_layout()
            # plt.show()
            plt.savefig('rotate_text_obb_result.png')
        else:
            print("Không tìm thấy text_box")
    else:
        print("Không detect được box nào")

if __name__ == "__main__":

    main()
