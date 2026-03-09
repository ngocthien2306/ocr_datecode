"""
Rotate text OBB using TensorRT inference (320x320 model)
Reuses YOLOOBBTensorRT from test_tensorrt_obb.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from test_tensorrt_obb import YOLOOBBTensorRT


# ─── Helper functions (same as rotate_text_obb.py) ───────────────────────────

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
    M = cv2.getRotationMatrix2D((w_img / 2, h_img / 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(image, M, (w_img, h_img),
                             flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
    return rotated, angle_deg, M


def transform_boxes(boxes: np.ndarray, M: np.ndarray):
    """
    Transform boxes theo ma trận xoay
    boxes: [[cx, cy, w, h, angle], ...]
    """
    transformed_boxes = []
    for box in boxes:
        cx, cy, w, h, angle = box

        rect       = ((cx, cy), (w, h), angle * 180 / np.pi)
        box_points = cv2.boxPoints(rect)

        transformed_points = []
        for point in box_points:
            pt     = np.array([point[0], point[1], 1])
            new_pt = M @ pt
            transformed_points.append(new_pt[:2])

        transformed_points = np.array(transformed_points)
        new_rect  = cv2.minAreaRect(transformed_points.astype(np.float32))
        new_cx, new_cy = new_rect[0]
        new_w,  new_h  = new_rect[1]
        new_angle      = new_rect[2] * np.pi / 180

        transformed_boxes.append([new_cx, new_cy, new_w, new_h, new_angle])

    return np.array(transformed_boxes)


def draw_obb(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
             class_ids: np.ndarray, class_names: list):
    """Vẽ OBB lên ảnh"""
    colors = {
        'bottle_cap': (255, 0, 0),
        'text_box':   (0, 255, 0),
    }

    img_draw = image.copy()
    for box, score, class_id in zip(boxes, scores, class_ids):
        cx, cy, w, h, angle = box
        class_name = class_names[int(class_id)]
        color      = colors.get(class_name, (0, 255, 0))

        rect       = ((cx, cy), (w, h), angle * 180 / np.pi)
        box_points = cv2.boxPoints(rect)
        box_points = np.int0(box_points)

        cv2.drawContours(img_draw, [box_points], 0, color, 2)

        if class_name == 'text_box':
            for i, point in enumerate(box_points):
                cv2.circle(img_draw, tuple(point), 8, (0, 0, 255), -1)
                cv2.putText(img_draw, str(i), (point[0] + 10, point[1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.circle(img_draw, (int(cx), int(cy)), 5, (255, 0, 255), -1)

        label = f'{class_name}: {score:.2f}'
        cv2.putText(img_draw, label, (int(cx) - 50, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img_draw


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(top_k=10):
    class_names = ['bottle_cap', 'text_box']
    model = YOLOOBBTensorRT(
        engine_path="weights/yolo26_bottle_obb.engine",
        class_names=class_names,
    )

    image = cv2.imread('test_image/bottle1.jpg')

    results, timing = model.predict([image], conf_threshold=0.4, return_timing=True)
    boxes, scores, class_ids = results[0]

    print(f"Timing: preprocess={timing['preprocess']:.1f}ms  "
          f"inference={timing['inference']:.1f}ms  "
          f"total={timing['total']:.1f}ms  "
          f"({1000/timing['total']:.1f} FPS)")

    # Top-K
    if len(boxes) > top_k:
        sorted_idx = np.argsort(scores)[::-1][:top_k]
        boxes      = boxes[sorted_idx]
        scores     = scores[sorted_idx]
        class_ids  = class_ids[sorted_idx]

    print(f"Results: {len(boxes)} boxes (top {top_k})")
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        print(f"  Box {i}: {class_names[int(class_id)]} - score: {score:.3f}")

    if len(boxes) == 0:
        print("Không detect được box nào")
        return

    # Tìm text_box và bottle_cap
    text_box_idx   = next((i for i, c in enumerate(class_ids) if class_names[int(c)] == 'text_box'),   None)
    bottle_cap_idx = next((i for i, c in enumerate(class_ids) if class_names[int(c)] == 'bottle_cap'), None)

    if text_box_idx is None:
        print("Không tìm thấy text_box")
        return

    box = boxes[text_box_idx]
    cx, cy, w, h, angle = box
    angle_deg = angle * 180 / np.pi

    print(f"\n=== Phân tích Text Box ===")
    print(f"Center: ({cx:.1f}, {cy:.1f})")
    print(f"Width: {w:.1f}, Height: {h:.1f}")
    print(f"Angle (rad): {angle:.4f}")
    print(f"Angle (deg): {angle_deg:.2f}°")

    original_with_obb = draw_obb(image, boxes, scores, class_ids, class_names)

    # Xoay lần 1
    rotated, angle_used, M = rotate_image_by_obb(image, boxes[text_box_idx])
    transformed_boxes      = transform_boxes(boxes, M)

    # Kiểm tra vị trí text_box so với bottle_cap
    need_flip = False
    if bottle_cap_idx is not None:
        text_cy = transformed_boxes[text_box_idx][1]
        cap_cy  = transformed_boxes[bottle_cap_idx][1]

        print(f"\n=== Kiểm tra vị trí sau xoay ===")
        print(f"Text box center Y: {text_cy:.1f}")
        print(f"Bottle cap center Y: {cap_cy:.1f}")

        if text_cy < cap_cy:
            need_flip = True
            print("Text nằm TRÊN bottle cap → Cần xoay thêm 180°")
        else:
            print("Text nằm DƯỚI bottle cap → Giữ nguyên")

    if need_flip:
        h_img, w_img = rotated.shape[:2]
        M_flip  = cv2.getRotationMatrix2D((w_img / 2, h_img / 2), 180, 1.0)
        rotated = cv2.warpAffine(rotated, M_flip, (w_img, h_img),
                                 flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
        transformed_boxes = transform_boxes(transformed_boxes, M_flip)
        print("Đã xoay thêm 180°")

    rotated_with_obb = draw_obb(rotated, transformed_boxes, scores, class_ids, class_names)

    if bottle_cap_idx is not None:
        cap_cy_px = int(transformed_boxes[bottle_cap_idx][1])
        cv2.line(rotated_with_obb, (0, cap_cy_px), (rotated.shape[1], cap_cy_px), (255, 0, 255), 2)

    total_angle = angle_used + (180 if need_flip else 0)
    print(f"\nXoay tổng: {angle_used:.1f}° + {180 if need_flip else 0}° = {total_angle:.1f}°")

    # Hiển thị & lưu
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    axes[0].imshow(cv2.cvtColor(original_with_obb, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Original (angle: {angle_deg:.2f}°)', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(rotated_with_obb, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Rotated ({angle_used:.1f}°{"+ 180°" if need_flip else ""})', fontsize=14)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('rotate_text_obb_trt_result.png')
    print("Saved: rotate_text_obb_trt_result.png")


if __name__ == "__main__":
    main()
