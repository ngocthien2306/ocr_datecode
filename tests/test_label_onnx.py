"""
YOLO OBB ONNX Inference with Custom NMS
Supports both single and batch inference
Model output format: [cx, cy, w, h, conf, class_id, angle]
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
from typing import List, Tuple, Dict


class YOLOOBBInference:
    def __init__(self, model_path: str, class_names: List[str] = None):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider', "CUDAExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.class_names = class_names or []

        # Check if model has built-in NMS
        output_shape = self.session.get_outputs()[0].shape
        self.has_builtin_nms = (len(output_shape) == 3 and output_shape[1] == 300)

    def preprocess(self, images: List[np.ndarray], input_size: Tuple[int, int] = (640, 640)):
        """Preprocess images with letterbox padding"""
        batch_tensor = []
        scales = []
        pads = []

        for image in images:
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
            batch_tensor.append(tensor)
            scales.append(scale)
            pads.append((pad_top, pad_left))

        return np.array(batch_tensor), scales, pads

    def postprocess_builtin_nms(self, output: np.ndarray, conf_threshold: float = 0.7):
        """Postprocess output from model with built-in NMS"""
        # Format: [cx, cy, w, h, conf, class_id, angle]
        boxes_xywhr = np.concatenate([output[:, :4], output[:, 6:7]], axis=1)
        scores = output[:, 4]
        class_ids = output[:, 5].astype(int)

        mask = (scores >= conf_threshold) & (scores > 0)
        return boxes_xywhr[mask], scores[mask], class_ids[mask]

    def nms_rotated(self, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray,
                    iou_threshold: float = 0.45):
        """Apply custom NMS for rotated boxes"""
        keep_indices = []

        for cls in np.unique(class_ids):
            cls_mask = class_ids == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            cls_indices = np.where(cls_mask)[0]

            order = cls_scores.argsort()[::-1]
            keep = []

            while order.size > 0:
                i = order[0]
                keep.append(i)
                if order.size == 1:
                    break

                ious = np.array([self._rotated_iou(cls_boxes[i], cls_boxes[j])
                               for j in order[1:]])
                order = order[np.where(ious <= iou_threshold)[0] + 1]

            keep_indices.extend(cls_indices[keep])

        return keep_indices

    def _rotated_iou(self, box1: np.ndarray, box2: np.ndarray):
        """Calculate IoU between two rotated boxes"""
        rect1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4] * 180 / np.pi)
        rect2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4] * 180 / np.pi)

        result, pts = cv2.rotatedRectangleIntersection(rect1, rect2)

        if result == cv2.INTERSECT_NONE:
            return 0.0

        inter_area = min(box1[2] * box1[3], box2[2] * box2[3]) if result == cv2.INTERSECT_FULL \
                     else cv2.contourArea(pts) if pts is not None and len(pts) > 2 else 0.0

        union_area = box1[2] * box1[3] + box2[2] * box2[3] - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def predict(self, images: List[np.ndarray], conf_threshold: float = 0.7,
                iou_threshold: float = 0.45, return_timing: bool = False):
        """
        Predict on batch of images
        Args:
            images: List of BGR images
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS (only used if model has no built-in NMS)
            return_timing: If True, return timing info
        Returns:
            If return_timing=False: List of (boxes, scores, class_ids) for each image
            If return_timing=True: (results, timing_dict)
        """
        if not isinstance(images, list):
            images = [images]

        timing = {}

        # Preprocess
        t0 = time.perf_counter()
        batch_tensor, scales, pads = self.preprocess(images)
        timing['preprocess'] = (time.perf_counter() - t0) * 1000

        # Inference
        t0 = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: batch_tensor})[0]
        timing['inference'] = (time.perf_counter() - t0) * 1000

        # Postprocess each image
        t0 = time.perf_counter()
        results = []
        for i in range(len(images)):
            if self.has_builtin_nms:
                boxes, scores, class_ids = self.postprocess_builtin_nms(
                    outputs[i], conf_threshold
                )
            else:
                # TODO: Implement raw output postprocessing
                raise NotImplementedError("Model without built-in NMS not yet supported")

            # Scale boxes back to original coordinates
            if len(boxes) > 0:
                pad_top, pad_left = pads[i]
                scale = scales[i]
                boxes[:, 0] = (boxes[:, 0] - pad_left) / scale
                boxes[:, 1] = (boxes[:, 1] - pad_top) / scale
                boxes[:, 2] = boxes[:, 2] / scale
                boxes[:, 3] = boxes[:, 3] / scale

            results.append((boxes, scores, class_ids))

        timing['postprocess'] = (time.perf_counter() - t0) * 1000
        timing['total'] = sum(timing.values())
        timing['per_image'] = timing['total'] / len(images)

        return (results, timing) if return_timing else results

    def draw_boxes(self, image: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
                   class_ids: np.ndarray):
        """Draw rotated boxes on image"""
        img = image.copy()

        for box, score, cls_id in zip(boxes, scores, class_ids):
            # Convert to 4 corners
            cx, cy, w, h, angle = box
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            hw, hh = w / 2, h / 2

            corners = np.array([
                [cx + (-hw * cos_a + hh * sin_a), cy + (-hw * sin_a - hh * cos_a)],
                [cx + (hw * cos_a + hh * sin_a), cy + (hw * sin_a - hh * cos_a)],
                [cx + (hw * cos_a - hh * sin_a), cy + (hw * sin_a + hh * cos_a)],
                [cx + (-hw * cos_a - hh * sin_a), cy + (-hw * sin_a + hh * cos_a)]
            ]).astype(np.int32)

            # Draw
            cv2.polylines(img, [corners], True, (0, 255, 0), 2)

            label = f"{self.class_names[cls_id]}: {score:.2f}" if cls_id < len(self.class_names) \
                    else f"Class {cls_id}: {score:.2f}"
            cv2.putText(img, label, tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 2)

        return img


def main():
    # Initialize model
    model = YOLOOBBInference(
        model_path="weights/yolo26n-obb-label.onnx",
        class_names=['bottle', 'label', "wrinkled"]
    )

    print(f"Model loaded: built-in NMS = {model.has_builtin_nms}")

    # Single image inference with timing
    print("\n=== Single Image Inference ===")
    image = cv2.imread('test.jpg')
    results, timing = model.predict([image], conf_threshold=0.3, return_timing=True)

    boxes, scores, class_ids = results[0]
    print(f"Detected {len(boxes)} objects:")
    for i, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids)):
        print(f"  {i}: {model.class_names[cls_id]} "
              f"conf={score:.4f} cx={box[0]:.1f} cy={box[1]:.1f} "
              f"w={box[2]:.1f} h={box[3]:.1f} angle={box[4]:.4f}")

    print(f"\nTiming:")
    print(f"  Preprocess:  {timing['preprocess']:.2f} ms")
    print(f"  Inference:   {timing['inference']:.2f} ms")
    print(f"  Postprocess: {timing['postprocess']:.2f} ms")
    print(f"  Total:       {timing['total']:.2f} ms ({1000/timing['total']:.1f} FPS)")

    annotated = model.draw_boxes(image, boxes, scores, class_ids)
    cv2.imwrite("output.png", annotated)
    print("\nSaved: output.png")

    # Batch inference with timing
    print("\n=== Batch Inference (3 images) ===")
    images = [cv2.imread('test.jpg') for _ in range(3)]
    for _ in range(10):   
        batch_results, timing = model.predict(images, conf_threshold=0.6, return_timing=True)

        for i, (boxes, scores, class_ids) in enumerate(batch_results):
            print(f"Image {i}: {len(boxes)} detections")

        print(f"\nBatch Timing:")
        print(f"  Preprocess:  {timing['preprocess']:.2f} ms")
        print(f"  Inference:   {timing['inference']:.2f} ms")
        print(f"  Postprocess: {timing['postprocess']:.2f} ms")
        print(f"  Total:       {timing['total']:.2f} ms")
        print(f"  Per image:   {timing['per_image']:.2f} ms ({1000/timing['per_image']:.1f} FPS)")


if __name__ == "__main__":
    main()
 

# /usr/src/tensorrt/bin/trtexec --onnx=weights/best_obb.onnx --saveEngine=weights/best_obb.engine --fp16 --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640