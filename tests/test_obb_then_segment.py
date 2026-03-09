"""
Pipeline: OBB Bottle Detection -> Crop -> Wrinkled Label Segmentation

Step 1: Dùng model OBB (best_bottle_obb_m.onnx) để detect chai
Step 2: Crop từng chai ra (xoay thẳng theo góc OBB)
Step 3: Dùng model Segment (best_wrinkled_instance_segmentation_crop_bottle.onnx)
        để detect nhãn nhăn trên ảnh chai đã crop
"""

import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────
# Model 1: OBB Bottle Detector
# ─────────────────────────────────────────────

class OBBBottleDetector:
    """YOLO OBB model để detect chai (oriented bounding box)"""

    def __init__(self, model_path: str, class_names: List[str] = None):
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = class_names or ['product', 'label']

        output_shape = self.session.get_outputs()[0].shape
        # built-in NMS: output shape [batch, 300, 7]
        self.has_builtin_nms = (len(output_shape) == 3 and output_shape[1] == 300)
        print(f"[OBB] Loaded: built-in NMS = {self.has_builtin_nms}")

    def preprocess(self, image: np.ndarray, input_size=(640, 640)):
        h, w = image.shape[:2]
        scale = min(input_size[0] / w, input_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))
        padded = np.full((*input_size[::-1], 3), 114, dtype=np.uint8)
        pad_top  = (input_size[1] - new_h) // 2
        pad_left = (input_size[0] - new_w) // 2
        padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        rgb    = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(tensor, 0), scale, (pad_top, pad_left)

    def predict(self, image: np.ndarray, conf_threshold=0.5):
        """
        Returns:
            boxes   : np.ndarray [N, 5]  — (cx, cy, w, h, angle_rad) in original coords
            scores  : np.ndarray [N]
            class_ids: np.ndarray [N]
        """
        tensor, scale, (pad_top, pad_left) = self.preprocess(image)
        output = self.session.run(None, {self.input_name: tensor})[0][0]  # [300, 7]

        # Format: [cx, cy, w, h, conf, class_id, angle]
        scores    = output[:, 4]
        class_ids = output[:, 5].astype(int)
        mask      = scores >= conf_threshold

        raw_boxes = output[mask]  # [cx640, cy640, w640, h640, conf, cls, angle]
        scores    = scores[mask]
        class_ids = class_ids[mask]

        if len(raw_boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        # Scale từ 640-padded space về original image
        cx = (raw_boxes[:, 0] - pad_left) / scale
        cy = (raw_boxes[:, 1] - pad_top)  / scale
        w  =  raw_boxes[:, 2] / scale
        h  =  raw_boxes[:, 3] / scale
        angle = raw_boxes[:, 6]  # radians, giữ nguyên

        boxes = np.stack([cx, cy, w, h, angle], axis=1)
        return boxes, scores, class_ids

    def draw(self, image: np.ndarray, boxes, scores, class_ids, color=(0, 255, 0)):
        img = image.copy()
        for box, score, cls_id in zip(boxes, scores, class_ids):
            cx, cy, w, h, angle = box
            corners = self._box_corners(cx, cy, w, h, angle)
            cv2.polylines(img, [corners], True, color, 2)
            label = f"{self.class_names[cls_id] if cls_id < len(self.class_names) else cls_id}: {score:.2f}"
            cv2.putText(img, label, tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img

    @staticmethod
    def _box_corners(cx, cy, w, h, angle):
        """Trả về 4 góc của OBB dạng int32 array"""
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        hw, hh = w / 2, h / 2
        corners = np.array([
            [cx + (-hw * cos_a + hh * sin_a), cy + (-hw * sin_a - hh * cos_a)],
            [cx + ( hw * cos_a + hh * sin_a), cy + ( hw * sin_a - hh * cos_a)],
            [cx + ( hw * cos_a - hh * sin_a), cy + ( hw * sin_a + hh * cos_a)],
            [cx + (-hw * cos_a - hh * sin_a), cy + (-hw * sin_a + hh * cos_a)],
        ]).astype(np.int32)
        return corners


# ─────────────────────────────────────────────
# Model 2: Wrinkled Label Segmenter
# ─────────────────────────────────────────────

class WrinkledSegmenter:
    """YOLO Segment model để detect nhăn nhãn trên chai đã crop"""

    def __init__(self, model_path: str, class_names: List[str] = None):
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.class_names = class_names or ['label', 'wrinkled']
        print(f"[SEG] Loaded: {model_path}")

    def preprocess(self, image: np.ndarray, input_size=(640, 640)):
        h, w = image.shape[:2]
        scale = min(input_size[0] / w, input_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))
        padded  = np.full((*input_size[::-1], 3), 114, dtype=np.uint8)
        pad_top  = (input_size[1] - new_h) // 2
        pad_left = (input_size[0] - new_w) // 2
        padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        rgb    = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(tensor, 0), scale, (pad_top, pad_left)

    def predict(self, image: np.ndarray, conf_threshold=0.3):
        """
        Returns:
            boxes : [N, 6] — (cx, cy, w, h, conf, class_id) in image coords
            masks : [N, H, W] binary masks in image coords  |  None if no detection
        """
        input_tensor, scale, (pad_top, pad_left) = self.preprocess(image)
        outputs = self.session.run(None, {"images": input_tensor})

        # output0: [batch, num_dets, 38]  (x1y1x2y2 xyxy + conf + cls + 32 coeffs)
        # output1: [batch, 32, 160, 160]  prototypes
        boxes_out = outputs[0][0]
        protos    = outputs[1][0] if len(outputs) > 1 else None

        valid = boxes_out[:, 4] >= conf_threshold
        boxes = boxes_out[valid]

        if len(boxes) == 0:
            return np.array([]), None

        box_coords   = boxes[:, :6].copy()   # xyxy + conf + cls
        mask_coeffs  = boxes[:, 6:]          # 32 coefficients

        # Generate masks trước khi scale box
        masks = None
        if protos is not None:
            masks = self._process_masks(
                protos, mask_coeffs, box_coords,
                image.shape, scale, (pad_top, pad_left)
            )

        # Scale boxes từ 640-space về image coords
        box_coords[:, 0] = (box_coords[:, 0] - pad_left) / scale  # x1
        box_coords[:, 1] = (box_coords[:, 1] - pad_top)  / scale  # y1
        box_coords[:, 2] = (box_coords[:, 2] - pad_left) / scale  # x2
        box_coords[:, 3] = (box_coords[:, 3] - pad_top)  / scale  # y2

        # Convert xyxy -> cxcywh
        x1, y1, x2, y2 = box_coords[:, 0], box_coords[:, 1], box_coords[:, 2], box_coords[:, 3]
        box_coords[:, 0] = (x1 + x2) / 2
        box_coords[:, 1] = (y1 + y2) / 2
        box_coords[:, 2] = x2 - x1
        box_coords[:, 3] = y2 - y1

        return box_coords, masks

    def _process_masks(self, protos, mask_coeffs, boxes_xyxy_640,
                       orig_shape, scale, pad):
        pad_top, pad_left = pad
        c, mh, mw = protos.shape
        h_orig, w_orig = orig_shape[:2]

        masks = np.matmul(mask_coeffs, protos.reshape(c, -1)).reshape(-1, mh, mw)
        masks = 1 / (1 + np.exp(-masks))  # sigmoid

        masks_640 = np.array([
            cv2.resize(m, (640, 640), interpolation=cv2.INTER_LINEAR) for m in masks
        ])

        new_h, new_w = int(h_orig * scale), int(w_orig * scale)
        masks_unpad = masks_640[:, pad_top:pad_top + new_h, pad_left:pad_left + new_w]

        masks_orig = np.array([
            cv2.resize(m, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR) for m in masks_unpad
        ])

        # Crop mask theo bbox (convert từ 640-space về original)
        boxes_orig = []
        for box in boxes_xyxy_640:
            x1 = (box[0] - pad_left) / scale
            y1 = (box[1] - pad_top)  / scale
            x2 = (box[2] - pad_left) / scale
            y2 = (box[3] - pad_top)  / scale
            boxes_orig.append([x1, y1, x2, y2])
        boxes_orig = np.array(boxes_orig)

        masks_cropped = self._crop_mask(masks_orig, boxes_orig)
        return (masks_cropped > 0.5).astype(np.float32)

    @staticmethod
    def _crop_mask(masks, boxes):
        n, h, w = masks.shape
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
            masks[i, :y1,  :] = 0
            masks[i, y2:,  :] = 0
            masks[i, :, :x1]  = 0
            masks[i, :, x2:]  = 0
        return masks

    def draw(self, image: np.ndarray, boxes, masks,
             seg_colors=None):
        """Vẽ mask + bounding box + diện tích lên ảnh crop"""
        img = image.copy()
        default_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]
        crop_area = image.shape[0] * image.shape[1]

        if masks is not None and len(masks) > 0:
            overlay = np.zeros_like(img)
            for box, mask in zip(boxes, masks):
                cls_id = int(box[5])
                color  = (seg_colors or default_colors)[cls_id % len(default_colors)]
                overlay[mask > 0] = color
            img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

        for i, box in enumerate(boxes):
            cx, cy, bw, bh, conf, cls_id = box
            cls_id = int(cls_id)
            x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
            x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
            color = (seg_colors or default_colors)[cls_id % len(default_colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Tính diện tích mask (pixel thực)
            mask_area = int(np.sum(masks[i] > 0)) if masks is not None and i < len(masks) else int(bw * bh)
            pct = mask_area / crop_area * 100 if crop_area > 0 else 0

            cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            label = f"{cls_name} {conf:.2f} | {mask_area}px ({pct:.1f}%)"
            cv2.putText(img, label, (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return img


# ─────────────────────────────────────────────
# Crop chai từ OBB (xoay thẳng)
# ─────────────────────────────────────────────

def crop_bottle_from_obb(image: np.ndarray, cx, cy, w, h, angle,
                          padding: int = 10) -> np.ndarray:
    """
    Crop chai ra khỏi ảnh gốc bằng cách xoay ảnh theo góc OBB.

    Args:
        image  : ảnh BGR gốc
        cx, cy : tâm chai (pixel, original coords)
        w, h   : kích thước chai (pixel)
        angle  : góc xoay (radians)
        padding: thêm viền xung quanh crop

    Returns:
        crop: ảnh BGR đã thẳng và crop theo chai
    """
    img_h, img_w = image.shape[:2]
    angle_deg = np.degrees(angle)

    # Ma trận xoay quanh tâm chai
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rotated = cv2.warpAffine(image, M, (img_w, img_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(114, 114, 114))

    # Crop vùng axis-aligned sau khi xoay
    x1 = int(cx - w / 2) - padding
    y1 = int(cy - h / 2) - padding
    x2 = int(cx + w / 2) + padding
    y2 = int(cy + h / 2) + padding

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)

    crop = rotated[y1:y2, x1:x2]
    return crop


# ─────────────────────────────────────────────
# Pipeline chính
# ─────────────────────────────────────────────

def run_pipeline(
    image_path: str,
    obb_model_path: str  = "weights/best_bottle_obb_m.onnx",
    seg_model_path: str  = "weights/best_wrinkled_instance_segmentation_crop_bottle.onnx",
    obb_conf: float      = 0.5,
    seg_conf: float      = 0.3,
    crop_padding: int    = 15,
    show_plot: bool      = True,
    save_path: str       = None,
):
    """
    Chạy full pipeline: detect chai (OBB) -> crop -> detect nhăn nhãn (Segment)

    Returns:
        results: list of dict, mỗi phần tử ứng với 1 chai:
            {
              'bottle_idx': int,
              'obb_box'   : (cx, cy, w, h, angle),
              'obb_score' : float,
              'crop'      : np.ndarray (BGR),
              'seg_boxes' : np.ndarray [M, 6],
              'seg_masks' : np.ndarray [M, H_crop, W_crop],
              'has_wrinkle': bool,
            }
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

    # ── Step 1: Load models ──
    obb_detector = OBBBottleDetector(obb_model_path, class_names=['product', 'label'])
    seg_model    = WrinkledSegmenter(seg_model_path, class_names=['label', 'wrinkled'])

    # ── Step 2: Detect bằng OBB ──
    all_boxes, all_scores, all_class_ids = obb_detector.predict(image, conf_threshold=obb_conf)
    print(f"\n[OBB] Phát hiện {len(all_boxes)} object (conf >= {obb_conf})")

    # Vẽ OBB lên ảnh gốc để xem (tất cả class)
    annotated_orig = obb_detector.draw(image, all_boxes, all_scores, all_class_ids)

    PRODUCT_CLASS_ID = obb_detector.class_names.index('product')
    LABEL_CLASS_ID   = obb_detector.class_names.index('label')

    # Ưu tiên crop product; nếu không có thì fallback sang label
    product_mask = all_class_ids == PRODUCT_CLASS_ID
    if product_mask.any():
        obb_boxes     = all_boxes[product_mask]
        obb_scores    = all_scores[product_mask]
        obb_class_ids = all_class_ids[product_mask]
        crop_source   = "product"
    else:
        label_mask = all_class_ids == LABEL_CLASS_ID
        obb_boxes     = all_boxes[label_mask]
        obb_scores    = all_scores[label_mask]
        obb_class_ids = all_class_ids[label_mask]
        crop_source   = "label (fallback)"

    print(f"[OBB] Crop từ {crop_source}: {len(obb_boxes)} vùng")

    # ── Step 3: Crop từng chai và segment ──
    results = []
    crops_annotated = []

    for idx, (box, score, cls_id) in enumerate(zip(obb_boxes, obb_scores, obb_class_ids)):
        cx, cy, w, h, angle = box
        print(f"\n  Chai #{idx}: cx={cx:.1f} cy={cy:.1f} w={w:.1f} h={h:.1f} "
              f"angle={np.degrees(angle):.1f}° conf={score:.3f}")

        # Crop chai (xoay thẳng)
        crop = crop_bottle_from_obb(image, cx, cy, w, h, angle, padding=crop_padding)

        if crop.size == 0:
            print(f"    [WARN] Crop rỗng, bỏ qua!")
            continue

        # Segment nhăn nhãn trên crop
        seg_boxes, seg_masks = seg_model.predict(crop, conf_threshold=seg_conf)
        print(f"    [SEG] Phát hiện {len(seg_boxes)} vùng nhãn/nhăn")

        # Kiểm tra có nhăn không + tính tổng diện tích nhăn
        has_wrinkle   = False
        total_wrinkle_area = 0
        crop_area = crop.shape[0] * crop.shape[1]
        if len(seg_boxes) > 0:
            for i, b in enumerate(seg_boxes):
                cls  = int(b[5])
                name = seg_model.class_names[cls] if cls < len(seg_model.class_names) else str(cls)
                area = int(np.sum(seg_masks[i] > 0)) if seg_masks is not None and i < len(seg_masks) else 0
                pct  = area / crop_area * 100 if crop_area > 0 else 0
                print(f"      - {name}: conf={b[4]:.3f}  area={area}px ({pct:.1f}%)")
                if 'wrinkl' in name.lower():
                    has_wrinkle = True
                    total_wrinkle_area += area

        wrinkle_pct = total_wrinkle_area / crop_area * 100 if crop_area > 0 else 0
        print(f"    Kết quả: {'CÓ NHĂN' if has_wrinkle else 'KHÔNG NHĂN'}"
              + (f"  |  tổng nhăn={total_wrinkle_area}px ({wrinkle_pct:.1f}%)" if has_wrinkle else ""))

        # Vẽ kết quả segment lên crop
        crop_vis = seg_model.draw(crop, seg_boxes, seg_masks) if len(seg_boxes) > 0 else crop.copy()

        # Header: status + tổng diện tích nhăn
        status_text  = "FAIL" if has_wrinkle else "OK"
        status_color = (0, 0, 255) if has_wrinkle else (0, 255, 0)
        header = f"#{idx} {status_text}"
        if has_wrinkle:
            header += f"  {total_wrinkle_area}px ({wrinkle_pct:.1f}%)"
        cv2.putText(crop_vis, header, (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        crops_annotated.append(crop_vis)
        results.append({
            'bottle_idx' : idx,
            'obb_box'    : (cx, cy, w, h, angle),
            'obb_score'  : score,
            'crop'       : crop,
            'seg_boxes'  : seg_boxes,
            'seg_masks'  : seg_masks,
            'has_wrinkle': has_wrinkle,
        })

    # ── Step 4: Visualize ──
    if show_plot or save_path:
        _visualize(image_path, annotated_orig, crops_annotated, results,
                   show_plot=show_plot, save_path=save_path)

    return results


def _visualize(image_path, annotated_orig, crops_annotated, results,
               show_plot=True, save_path=None):
    n_crops = len(crops_annotated)
    n_cols  = max(1, n_crops)
    n_rows  = 2  # row 0: ảnh gốc; row 1: các crop

    fig_w = max(12, n_cols * 4)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, 8))

    # Flatten axes để dễ index
    if n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]

    # Row 0: ảnh gốc với OBB boxes
    ax_orig = axes[0, 0]
    ax_orig.imshow(cv2.cvtColor(annotated_orig, cv2.COLOR_BGR2RGB))
    ax_orig.set_title(f"Original + OBB ({len(results)} chai)", fontsize=11)
    ax_orig.axis('off')

    # Tắt các ô trống ở row 0
    for j in range(1, n_cols):
        axes[0, j].axis('off')

    # Row 1: từng crop với segmentation
    for j, (crop_vis, res) in enumerate(zip(crops_annotated, results)):
        ax = axes[1, j]
        ax.imshow(cv2.cvtColor(crop_vis, cv2.COLOR_BGR2RGB))
        status = "FAIL" if res['has_wrinkle'] else "OK"
        color  = 'red' if res['has_wrinkle'] else 'green'
        ax.set_title(f"Chai #{res['bottle_idx']} — {status}", color=color, fontsize=10)
        ax.axis('off')

    # Tắt ô trống ở row 1
    for j in range(len(crops_annotated), n_cols):
        axes[1, j].axis('off')

    plt.suptitle(f"OBB Detect → Crop → Wrinkle Segment\n{image_path}", fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n[Saved] {save_path}")

    if show_plot:
        plt.show()

    plt.close()


# ─────────────────────────────────────────────
# Batch: xử lý cả folder
# ─────────────────────────────────────────────

def run_folder(
    input_folder: str,
    output_folder: str,
    obb_model_path: str = "weights/best_bottle_obb_m.onnx",
    seg_model_path: str = "weights/best_wrinkled_instance_segmentation_crop_bottle.onnx",
    obb_conf: float     = 0.5,
    seg_conf: float     = 0.3,
    crop_padding: int   = 15,
):
    """
    Xử lý toàn bộ ảnh trong input_folder, lưu kết quả vào output_folder.

    Cấu trúc output:
        output_folder/
        ├── wrinkled/          ← chai có nhăn nhãn
        │   ├── <tên_ảnh>_result.jpg   (ảnh gốc + OBB)
        │   └── <tên_ảnh>_crop_<i>.jpg (từng crop + segment mask)
        ├── ok/                ← chai không nhăn
        │   └── ...
        ├── no_detection/      ← không phát hiện chai nào
        │   └── ...
        └── summary.txt        ← bảng tổng kết toàn bộ
    """
    import os, glob, time, traceback

    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    # Thu thập danh sách ảnh
    all_files = []
    for ext in IMG_EXTS:
        all_files += glob.glob(os.path.join(input_folder, f'*{ext}'))
        all_files += glob.glob(os.path.join(input_folder, f'*{ext.upper()}'))
    all_files = sorted(set(all_files))

    if not all_files:
        print(f"[WARN] Không tìm thấy ảnh nào trong: {input_folder}")
        return

    print(f"\n{'='*50}")
    print(f"Batch processing: {len(all_files)} ảnh")
    print(f"Input : {input_folder}")
    print(f"Output: {output_folder}")
    print(f"{'='*50}")

    # Tạo thư mục output
    dir_fail         = os.path.join(output_folder, "fail")
    dir_ok           = os.path.join(output_folder, "ok")
    dir_no_detection = os.path.join(output_folder, "no_detection")
    for d in [dir_fail, dir_ok, dir_no_detection]:
        os.makedirs(d, exist_ok=True)

    # Load model một lần duy nhất cho cả batch
    obb_detector = OBBBottleDetector(obb_model_path, class_names=['product', 'label'])
    seg_model    = WrinkledSegmenter(seg_model_path, class_names=['wrinkled'])
    PRODUCT_CLASS_ID = obb_detector.class_names.index('product')
    LABEL_CLASS_ID   = obb_detector.class_names.index('label')

    summary_rows = []
    total_fail   = 0
    total_ok     = 0
    total_no_det = 0
    t_start = time.time()

    for file_idx, img_path in enumerate(all_files):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n[{file_idx+1}/{len(all_files)}] {os.path.basename(img_path)}")

        image = cv2.imread(img_path)
        if image is None:
            print(f"  [SKIP] Không đọc được ảnh")
            summary_rows.append(f"{basename}\tERROR\t0\t0\t0")
            continue

        try:
            # ── OBB detect chai ──
            obb_boxes, obb_scores, obb_class_ids = obb_detector.predict(
                image, conf_threshold=obb_conf
            )

            # Vẽ OBB tất cả class lên ảnh gốc trước khi lọc
            annotated_orig = obb_detector.draw(image, obb_boxes, obb_scores, obb_class_ids)

            # Ưu tiên crop product; nếu không có thì fallback sang label
            product_mask = obb_class_ids == PRODUCT_CLASS_ID
            if product_mask.any():
                crop_boxes     = obb_boxes[product_mask]
                crop_scores    = obb_scores[product_mask]
                crop_class_ids = obb_class_ids[product_mask]
                crop_source    = "product"
            else:
                label_mask     = obb_class_ids == LABEL_CLASS_ID
                crop_boxes     = obb_boxes[label_mask]
                crop_scores    = obb_scores[label_mask]
                crop_class_ids = obb_class_ids[label_mask]
                crop_source    = "label(fallback)"

            n_bottles = len(crop_boxes)
            print(f"  [OBB] {crop_source}: {n_bottles} vùng crop")

            if n_bottles == 0:
                out_path = os.path.join(dir_no_detection, f"{basename}_result.jpg")
                _save_no_detection(image, out_path)
                summary_rows.append(f"{basename}\tNO_DETECTION\t0\t0\t0")
                total_no_det += 1
                continue

            obb_boxes, obb_scores, obb_class_ids = crop_boxes, crop_scores, crop_class_ids

            # ── Crop + Segment từng chai ──
            img_has_wrinkle = False
            n_wrinkle_this  = 0
            n_ok_this       = 0
            crops_info      = []  # list of (crop_vis, has_wrinkle, bottle_idx)

            for idx, (box, _score, _cls_id) in enumerate(
                zip(obb_boxes, obb_scores, obb_class_ids)
            ):
                cx, cy, w, h, angle = box
                crop = crop_bottle_from_obb(image, cx, cy, w, h, angle, padding=crop_padding)

                if crop.size == 0:
                    print(f"    Chai #{idx}: crop rỗng, bỏ qua")
                    continue

                seg_boxes, seg_masks = seg_model.predict(crop, conf_threshold=seg_conf)

                has_wrinkle        = False
                total_wrinkle_area = 0
                crop_area = crop.shape[0] * crop.shape[1]
                if len(seg_boxes) > 0:
                    for i, b in enumerate(seg_boxes):
                        cls  = int(b[5])
                        name = seg_model.class_names[cls] if cls < len(seg_model.class_names) else str(cls)
                        area = int(np.sum(seg_masks[i] > 0)) if seg_masks is not None and i < len(seg_masks) else 0
                        if 'wrinkl' in name.lower():
                            has_wrinkle = True
                            total_wrinkle_area += area

                wrinkle_pct  = total_wrinkle_area / crop_area * 100 if crop_area > 0 else 0
                status_text  = "FAIL" if has_wrinkle else "OK"
                status_color = (0, 0, 255) if has_wrinkle else (0, 255, 0)

                crop_vis = seg_model.draw(crop, seg_boxes, seg_masks) if len(seg_boxes) > 0 else crop.copy()
                header = f"#{idx} {status_text}"
                if has_wrinkle:
                    header += f"  {total_wrinkle_area}px ({wrinkle_pct:.1f}%)"
                cv2.putText(crop_vis, header, (5, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                crops_info.append((crop_vis, has_wrinkle, idx))

                if has_wrinkle:
                    n_wrinkle_this += 1
                    img_has_wrinkle = True
                else:
                    n_ok_this += 1

                print(f"    Chai #{idx}: {status_text}  ({len(seg_boxes)} det)")

            # ── Lưu output ──
            out_dir = dir_fail if img_has_wrinkle else dir_ok

            # Lưu ảnh gốc đã annotate
            orig_out = os.path.join(out_dir, f"{basename}_result.jpg")
            cv2.imwrite(orig_out, annotated_orig)

            # Lưu từng crop
            for crop_vis, has_wrinkle, cidx in crops_info:
                tag      = "fail" if has_wrinkle else "ok"
                crop_out = os.path.join(out_dir, f"{basename}_crop{cidx}_{tag}.jpg")
                cv2.imwrite(crop_out, crop_vis)

            # Lưu ảnh tổng hợp (grid)
            grid_out = os.path.join(out_dir, f"{basename}_grid.jpg")
            _save_grid(annotated_orig, [c[0] for c in crops_info], grid_out)

            status_str = "FAIL" if img_has_wrinkle else "OK"
            summary_rows.append(
                f"{basename}\t{status_str}\t{crop_source}\t{n_bottles}\t{n_wrinkle_this}\t{n_ok_this}"
            )
            if img_has_wrinkle:
                total_fail += 1
            else:
                total_ok += 1

        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            summary_rows.append(f"{basename}\tERROR\t0\t0\t0")

    elapsed = time.time() - t_start

    # ── Ghi summary.txt ──
    summary_path = os.path.join(output_folder, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Batch kết quả: {input_folder}\n")
        f.write(f"Thời gian    : {elapsed:.1f}s  ({elapsed/max(1,len(all_files))*1000:.0f} ms/ảnh)\n")
        f.write(f"Tổng ảnh     : {len(all_files)}\n")
        f.write(f"FAIL (nhăn)  : {total_fail}\n")
        f.write(f"OK           : {total_ok}\n")
        f.write(f"Không detect : {total_no_det}\n")
        f.write(f"\n{'─'*60}\n")
        f.write("Tên file\tKết quả\tNguồn crop\tSố vùng\tNhăn\tOK\n")
        f.write('\n'.join(summary_rows))

    print(f"\n{'='*50}")
    print(f"HOÀN TẤT — {elapsed:.1f}s")
    print(f"  FAIL (nhăn)  : {total_fail}")
    print(f"  OK           : {total_ok}")
    print(f"  Không detect : {total_no_det}")
    print(f"  Summary      : {summary_path}")
    print(f"{'='*50}")


def _save_no_detection(image: np.ndarray, out_path: str):
    img = image.copy()
    cv2.putText(img, "NO DETECTION", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
    cv2.imwrite(out_path, img)


def _save_grid(orig: np.ndarray, crops: list, out_path: str, thumb_w=320):
    """Ghép ảnh gốc (thu nhỏ) và các crop thành 1 file duy nhất để xem nhanh."""
    if not crops:
        cv2.imwrite(out_path, orig)
        return

    # Thu nhỏ ảnh gốc
    h0, w0 = orig.shape[:2]
    scale0  = thumb_w * 2 / w0
    orig_thumb = cv2.resize(orig, (int(w0 * scale0), int(h0 * scale0)))

    # Thu nhỏ từng crop về cùng chiều rộng
    crop_thumbs = []
    for c in crops:
        hc, wc = c.shape[:2]
        sc = thumb_w / max(wc, 1)
        crop_thumbs.append(cv2.resize(c, (int(wc * sc), int(hc * sc))))

    # Ghép các crop theo chiều ngang, rồi stack dọc với ảnh gốc
    max_h   = max(c.shape[0] for c in crop_thumbs)
    row_crops = []
    for c in crop_thumbs:
        pad_h = max_h - c.shape[0]
        padded = cv2.copyMakeBorder(c, 0, pad_h, 0, 0,
                                    cv2.BORDER_CONSTANT, value=(50, 50, 50))
        row_crops.append(padded)
    crops_row = np.hstack(row_crops)

    # Điều chỉnh width cho khớp
    target_w = max(orig_thumb.shape[1], crops_row.shape[1])

    def pad_width(img, w):
        diff = w - img.shape[1]
        if diff > 0:
            return cv2.copyMakeBorder(img, 0, 0, 0, diff,
                                      cv2.BORDER_CONSTANT, value=(50, 50, 50))
        return img

    orig_thumb = pad_width(orig_thumb, target_w)
    crops_row  = pad_width(crops_row,  target_w)

    grid = np.vstack([orig_thumb, crops_row])
    cv2.imwrite(out_path, grid)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os

    # Chuyển working dir về root project để path weights/data đúng
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    OBB_MODEL = "weights/best_bottle_obb_m.onnx"
    SEG_MODEL = "weights/best_wrinkled_instance_segmentation_crop_bottle.onnx"

    # ── Chế độ 1: xử lý cả folder ──
    if len(sys.argv) >= 3:
        input_folder  = sys.argv[1]
        output_folder = sys.argv[2]
        run_folder(
            input_folder  = input_folder,
            output_folder = output_folder,
            obb_model_path= OBB_MODEL,
            seg_model_path= SEG_MODEL,
            obb_conf      = 0.25,
            seg_conf      = 0.3,
            crop_padding  = 15,
        )

    # ── Chế độ mặc định: folder nhannhan ──
    elif len(sys.argv) == 1:
        run_folder(
            input_folder  = "/Users/ngocthien.ai/Downloads/nhannhan",
            output_folder = "/Users/ngocthien.ai/Downloads/nhannhan_output",
            obb_model_path= OBB_MODEL,
            seg_model_path= SEG_MODEL,
            obb_conf      = 0.5,
            seg_conf      = 0.3,
            crop_padding  = 15,
        )

    # ── Chế độ 2: 1 ảnh đơn lẻ ──
    else:
        test_image = sys.argv[1]
        results = run_pipeline(
            image_path    = test_image,
            obb_model_path= OBB_MODEL,
            seg_model_path= SEG_MODEL,
            obb_conf      = 0.5,
            seg_conf      = 0.3,
            crop_padding  = 15,
            show_plot     = True,
            save_path     = "output_pipeline.png",
        )
        n_fail = sum(r['has_wrinkle'] for r in results)
        print(f"\n{'='*40}")
        print(f"Tổng số chai: {len(results)}")
        print(f"FAIL (nhăn) : {n_fail}")
        print(f"OK          : {len(results) - n_fail}")
        print(f"{'='*40}")

# /usr/src/tensorrt/bin/trtexec --onnx=weights/best_wrinkled_instance_segmentation_crop_bottle.onnx --saveEngine=weights/best_wrinkled_instance_segmentation_crop_bottle.engine --fp16 --minShapes=images:1x3x320x320 --optShapes=images:4x3x320x320 --maxShapes=images:8x3x320x320