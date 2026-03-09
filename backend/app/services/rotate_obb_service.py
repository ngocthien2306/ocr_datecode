"""
Rotate frame service using YOLO OBB ONNX inference.
Detects bottle_cap + text_box, then rotates the frame so text faces upright.
"""

import cv2
import numpy as np
import onnxruntime as ort
import logging
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Weights path relative to this file: backend/app/services/ → up 4 levels → project root
_WEIGHTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "weights"
_DEFAULT_MODEL_PATH = _WEIGHTS_DIR / "yolo26_bottle_obb.onnx"
_CLASS_NAMES = ["bottle_cap", "text_box"]


class YOLOOBBInference:
    """YOLO OBB ONNX Inference — model output format: [cx, cy, w, h, conf, class_id, angle]"""

    def __init__(self, model_path: str, class_names: List[str] = None):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.class_names = class_names or []

        output_shape = self.session.get_outputs()[0].shape
        self.has_builtin_nms = (len(output_shape) == 3 and output_shape[1] == 300)
        logger.info(f"YOLOOBBInference loaded: {model_path}, built-in NMS={self.has_builtin_nms}")

    def preprocess(self, images: List[np.ndarray], input_size: Tuple[int, int] = (320, 320)):
        batch_tensor, scales, pads = [], [], []
        for image in images:
            h, w = image.shape[:2]
            scale = min(input_size[0] / w, input_size[1] / h)
            new_w, new_h = int(w * scale), int(h * scale)

            resized = cv2.resize(image, (new_w, new_h))
            padded = np.full((*input_size[::-1], 3), 114, dtype=np.uint8)
            pad_top = (input_size[1] - new_h) // 2
            pad_left = (input_size[0] - new_w) // 2
            padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
            batch_tensor.append(tensor)
            scales.append(scale)
            pads.append((pad_top, pad_left))

        return np.array(batch_tensor), scales, pads

    def postprocess_builtin_nms(self, output: np.ndarray, conf_threshold: float = 0.4):
        # Format: [cx, cy, w, h, conf, class_id, angle]
        boxes = np.concatenate([output[:, :4], output[:, 6:7]], axis=1)
        scores = output[:, 4]
        class_ids = output[:, 5].astype(int)

        mask = (scores >= conf_threshold) & (scores > 0)
        return boxes[mask], scores[mask], class_ids[mask]

    def predict(self, images: List[np.ndarray], conf_threshold: float = 0.4):
        if not isinstance(images, list):
            images = [images]

        batch_tensor, scales, pads = self.preprocess(images)
        outputs = self.session.run(None, {self.input_name: batch_tensor})[0]

        results = []
        for i in range(len(images)):
            if self.has_builtin_nms:
                boxes, scores, class_ids = self.postprocess_builtin_nms(outputs[i], conf_threshold)
            else:
                raise NotImplementedError("Model without built-in NMS not supported")

            if len(boxes) > 0:
                pad_top, pad_left = pads[i]
                scale = scales[i]
                boxes[:, 0] = (boxes[:, 0] - pad_left) / scale
                boxes[:, 1] = (boxes[:, 1] - pad_top) / scale
                boxes[:, 2] = boxes[:, 2] / scale
                boxes[:, 3] = boxes[:, 3] / scale

            results.append((boxes, scores, class_ids))

        return results


def _rotate_image_by_obb(image: np.ndarray, box: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """Xoay ảnh để vùng text nằm chính diện dựa vào OBB. box: [cx, cy, w, h, angle]"""
    cx, cy, w, h, angle = box
    angle_deg = angle * 180 / np.pi

    if h > w:
        angle_deg += 90

    h_img, w_img = image.shape[:2]
    M = cv2.getRotationMatrix2D((w_img / 2, h_img / 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(image, M, (w_img, h_img),
                             flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
    return rotated, angle_deg, M


def _transform_center(cx: float, cy: float, M: np.ndarray) -> Tuple[float, float]:
    """Transform một điểm center theo ma trận affine."""
    pt = np.array([cx, cy, 1.0])
    new_pt = M @ pt
    return new_pt[0], new_pt[1]


def rotate_frame(image: np.ndarray, model: YOLOOBBInference) -> np.ndarray:
    """
    Detect OBB trên frame, xoay để text nằm chính diện.
    Nếu không detect được text_box → trả về ảnh gốc.
    Không vẽ OBB lên ảnh trả về.
    """
    results = model.predict([image], conf_threshold=0.2)
    boxes, scores, class_ids = results[0]

    if len(boxes) == 0:
        logger.debug("No detections, returning original frame")
        return image

    text_box_idx = next(
        (i for i, c in enumerate(class_ids) if model.class_names[int(c)] == "text_box"), None
    )
    bottle_cap_idx = next(
        (i for i, c in enumerate(class_ids) if model.class_names[int(c)] == "bottle_cap"), None
    )

    if text_box_idx is None:
        logger.debug("No text_box detected, returning original frame")
        return image

    # Xoay lần 1 theo angle của text_box
    rotated, angle_deg, M = _rotate_image_by_obb(image, boxes[text_box_idx])

    # Kiểm tra vị trí text_box vs bottle_cap sau xoay → flip 180° nếu cần
    if bottle_cap_idx is not None:
        text_cx, text_cy = _transform_center(boxes[text_box_idx][0], boxes[text_box_idx][1], M)
        cap_cx, cap_cy = _transform_center(boxes[bottle_cap_idx][0], boxes[bottle_cap_idx][1], M)

        if text_cy < cap_cy:
            h_img, w_img = rotated.shape[:2]
            M_flip = cv2.getRotationMatrix2D((w_img / 2, h_img / 2), 180, 1.0)
            rotated = cv2.warpAffine(rotated, M_flip, (w_img, h_img),
                                     flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
            logger.debug(f"Rotated {angle_deg:.1f}° + 180° flip")
        else:
            logger.debug(f"Rotated {angle_deg:.1f}°")

    return rotated


# ─── Singleton ────────────────────────────────────────────────────────────────

_model_instance: Optional[YOLOOBBInference] = None


def get_rotate_obb_model() -> Optional[YOLOOBBInference]:
    """Trả về singleton model, lazy load lần đầu."""
    global _model_instance
    if _model_instance is None:
        if not _DEFAULT_MODEL_PATH.exists():
            logger.error(f"OBB model not found: {_DEFAULT_MODEL_PATH}")
            return None
        try:
            _model_instance = YOLOOBBInference(
                model_path=str(_DEFAULT_MODEL_PATH),
                class_names=_CLASS_NAMES,
            )
        except Exception as e:
            logger.error(f"Failed to load OBB model: {e}")
            return None
    return _model_instance
