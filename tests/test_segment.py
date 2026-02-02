import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt


class YOLOSegmentInference:
    def __init__(self, model_path: str, class_names: list):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider', "CUDAExecutionProvider"])
        self.class_names = class_names

    def preprocess(self, image: np.ndarray, input_size=(640, 640)):
        """Preprocess image with letterbox padding"""
        h, w = image.shape[:2]
        scale = min(input_size[0] / w, input_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))
        padded = np.full((*input_size[::-1], 3), 114, dtype=np.uint8)
        pad_top = (input_size[1] - new_h) // 2
        pad_left = (input_size[0] - new_w) // 2
        padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0

        return np.expand_dims(tensor, 0), scale, (pad_top, pad_left)

    def predict(self, image: np.ndarray, conf_threshold=0.5):
        """Predict segmentation masks"""
        input_tensor, scale, (pad_top, pad_left) = self.preprocess(image)

        outputs = self.session.run(None, {"images": input_tensor})

        # print(f"Number of outputs: {len(outputs)}")
        # for i, out in enumerate(outputs):
        #     print(f"Output {i} shape: {out.shape}")

        # YOLOv11-seg with built-in NMS format:
        # output0: [batch, num_dets, 38] - [x, y, w, h, conf, class, mask_coeffs(32)]
        # output1: [batch, 32, mask_h, mask_w] - mask prototypes

        boxes_output = outputs[0][0]  # [num_dets, 38]
        protos = outputs[1][0] if len(outputs) > 1 else None  # [32, 160, 160]

        # print(f"Boxes output shape: {boxes_output.shape}")
        # if protos is not None:
        #     print(f"Protos shape: {protos.shape}")

        # Filter by confidence
        valid_mask = boxes_output[:, 4] >= conf_threshold
        boxes = boxes_output[valid_mask]

        # print(f"Valid detections: {len(boxes)}")

        # Extract box info and mask coefficients
        if len(boxes) > 0:
            box_coords = boxes[:, :6].copy()  # [x1, y1, x2, y2, conf, class] - xyxy format!
            mask_coeffs = boxes[:, 6:]  # [32 coefficients]

            # Generate masks BEFORE scaling boxes (masks need original 640x640 coords)
            if protos is not None:
                # Use original box_coords (in 640x640 xyxy space) for mask processing
                masks = self.process_masks(protos, mask_coeffs, box_coords, image.shape, scale, (pad_top, pad_left))
            else:
                masks = None

            # Scale boxes from xyxy 640x640 to xyxy original AFTER mask processing
            box_coords[:, 0] = (box_coords[:, 0] - pad_left) / scale  # x1
            box_coords[:, 1] = (box_coords[:, 1] - pad_top) / scale   # y1
            box_coords[:, 2] = (box_coords[:, 2] - pad_left) / scale  # x2
            box_coords[:, 3] = (box_coords[:, 3] - pad_top) / scale   # y2

            # Convert xyxy to xywh center for compatibility
            x1, y1, x2, y2 = box_coords[:, 0], box_coords[:, 1], box_coords[:, 2], box_coords[:, 3]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            box_coords[:, 0] = cx
            box_coords[:, 1] = cy
            box_coords[:, 2] = w
            box_coords[:, 3] = h

            return box_coords, masks
        else:
            return np.array([]), None

    def crop_mask(self, masks, boxes):
        """Crop masks to bounding box regions - Ultralytics style."""
        n, h, w = masks.shape

        # For each mask, zero out pixels outside its bbox
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Zero out regions outside bbox
            masks[i, :y1, :] = 0  # Above bbox
            masks[i, y2:, :] = 0  # Below bbox
            masks[i, :, :x1] = 0  # Left of bbox
            masks[i, :, x2:] = 0  # Right of bbox

        return masks

    def process_masks(self, protos, mask_coeffs, boxes_input, orig_shape, scale, pad):
        """Generate instance masks from prototypes and coefficients - Ultralytics method

        Flow: masks @ protos -> sigmoid -> upsample to 640x640 -> remove padding ->
              resize to original -> crop by bbox -> threshold
        """
        pad_top, pad_left = pad
        c, mh, mw = protos.shape
        h_orig, w_orig = orig_shape[:2]

        # print(f"\n=== DEBUG process_masks ===")
        # print(f"Original image shape: {orig_shape[:2]}")
        # print(f"Scale: {scale}, Padding: ({pad_top}, {pad_left})")

        # Step 1: Generate masks from prototypes
        masks = np.matmul(mask_coeffs, protos.reshape(c, -1)).reshape(-1, mh, mw)
        masks = 1 / (1 + np.exp(-masks))  # Sigmoid

        # Step 2: Upsample to 640x640
        masks_640 = np.array([
            cv2.resize(mask, (640, 640), interpolation=cv2.INTER_LINEAR)
            for mask in masks
        ])

        # Step 3: Remove padding and get masks in unpadded space
        new_h, new_w = int(h_orig * scale), int(w_orig * scale)
        masks_unpad = masks_640[:, pad_top:pad_top+new_h, pad_left:pad_left+new_w]

        # print(f"After unpad: mask shape = {masks_unpad.shape}, target = ({new_h}, {new_w})")

        # Step 4: Resize to original image size
        masks_orig = np.array([
            cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            for mask in masks_unpad
        ])

        # Step 5: Scale boxes to original image space for crop_mask
        # boxes_input are ALREADY in xyxy format in 640x640 padded space
        boxes_xyxy_orig = []
        for box in boxes_input:
            x1_640, y1_640, x2_640, y2_640 = box[:4]

            # Remove padding first
            x1_unpad = x1_640 - pad_left
            y1_unpad = y1_640 - pad_top
            x2_unpad = x2_640 - pad_left
            y2_unpad = y2_640 - pad_top

            # Scale to original size
            x1_orig = x1_unpad / scale
            y1_orig = y1_unpad / scale
            x2_orig = x2_unpad / scale
            y2_orig = y2_unpad / scale

            boxes_xyxy_orig.append([x1_orig, y1_orig, x2_orig, y2_orig])

        boxes_xyxy_orig = np.array(boxes_xyxy_orig)

        # print(f"Boxes in original space (first): [{boxes_xyxy_orig[0][0]:.1f}, {boxes_xyxy_orig[0][1]:.1f}, {boxes_xyxy_orig[0][2]:.1f}, {boxes_xyxy_orig[0][3]:.1f}]")

        # Step 6: Crop masks by bounding boxes
        masks_cropped = self.crop_mask(masks_orig, boxes_xyxy_orig)

        # Step 7: Apply threshold
        masks_binary = (masks_cropped > 0.5).astype(np.float32)

        # print(f"Final mask pixels>0: {masks_binary[0].sum():.0f}")

        return masks_binary

    def draw_results(self, image: np.ndarray, boxes: np.ndarray, masks: np.ndarray = None):
        """Draw boxes and masks on image"""
        img = image.copy()

        # Color map for each class
        colors = {
            0: (255, 0, 0),    # bottle - blue
            1: (0, 255, 0)     # label - green
        }

        if masks is not None and len(masks) > 0:
            h, w = image.shape[:2]
            mask_overlay = np.zeros_like(img)

            for i, (box, mask) in enumerate(zip(boxes, masks)):
                class_id = int(box[5])
                color = colors.get(class_id, (0, 255, 255))

                # Mask is already binary (0 or 1) from process_masks
                mask_binary = mask.astype(np.uint8)

                # Apply colored mask where mask > 0
                mask_overlay[mask_binary > 0] = color

            # Blend mask with original image
            img = cv2.addWeighted(img, 0.6, mask_overlay, 0.4, 0)

        # Draw boxes
        for box in boxes:
            x, y, w_box, h_box, conf, class_id = box
            class_id = int(class_id)
            x1, y1 = int(x - w_box/2), int(y - h_box/2)
            x2, y2 = int(x + w_box/2), int(y + h_box/2)

            color = colors.get(class_id, (0, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f'{self.class_names[class_id]}: {conf:.2f}'
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img


def analyze_dominant_color(image, mask):
    """Phân tích màu sắc chủ đạo trong vùng mask"""
    # Lấy pixels trong vùng mask
    mask_binary = mask > 0
    pixels = image[mask_binary]

    if len(pixels) == 0:
        return None, None

    # Convert BGR to HSV để phân tích màu dễ hơn
    pixels_hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)

    # Tính màu trung bình
    mean_bgr = pixels.mean(axis=0)
    mean_hsv = pixels_hsv.mean(axis=0)

    # Phân loại màu dựa trên HSV
    h, s, v = mean_hsv

    color_name = "Unknown"
    if s < 30:  # Low saturation = grayscale
        if v < 50:
            color_name = "Black/Dark"
        elif v > 200:
            color_name = "White/Light"
        else:
            color_name = "Gray"
    else:  # Colored
        if h < 10 or h > 170:
            color_name = "Red"
        elif h < 25:
            color_name = "Orange"
        elif h < 40:
            color_name = "Yellow"
        elif h < 80:
            color_name = "Green"
        elif h < 130:
            color_name = "Cyan/Blue"
        else:
            color_name = "Purple/Magenta"

    print(f"  Color analysis:")
    print(f"    Mean BGR: {mean_bgr}")
    print(f"    Mean HSV: H={h:.1f}, S={s:.1f}, V={v:.1f}")
    print(f"    Dominant color: {color_name}")
    print(f"    Pixels in mask: {len(pixels)}")

    return color_name, mean_bgr

def main():
    model = YOLOSegmentInference(
        "weights/yolo26_label_seg.onnx",
        class_names=['bottle', 'label']
    )

    image = cv2.imread('data/2026-01-31/22376896/fail_f0_20260131_102257559309_org.jpg')

    boxes, masks = model.predict(image, conf_threshold=0.3)

    print(f"\nDetected {len(boxes)} objects:")
    for i, box in enumerate(boxes):
        cx, cy, w, h, conf, class_id = box
        class_name = model.class_names[int(class_id)]
        print(f"\n{i}: {class_name} - conf: {conf:.3f}")

        # Phân tích màu sắc cho label
        if class_name == 'label' and masks is not None:
            analyze_dominant_color(image, masks[i])

    result = model.draw_results(image, boxes, masks)

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Segmentation ({len(boxes)} objects)')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
