"""
Rotate frame service using YOLO OBB ONNX inference.
Detects bottle_cap + text_box, then rotates the frame so text faces upright.

Logs go to logs/obb_rotation/{YYYY-MM-DD}.log (in addition to console) via
the centralized daily-rotating file handler.
"""

import cv2
import numpy as np
import onnxruntime as ort
import logging
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional

from app.utils.logging_config import setup_category_logger

# Dedicated category logger — writes to logs/obb_rotation/{date}.log.
# `add_console=True` keeps it visible in uvicorn stdout too.
logger = setup_category_logger(
    category="obb_rotation",
    level=logging.INFO,
    add_console=True,
    logger_name=__name__,
)

# Weights path relative to this file: backend/app/services/ → up 4 levels → project root
_WEIGHTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "weights"
_DEFAULT_MODEL_PATH = _WEIGHTS_DIR / "best_bottle_m.onnx"
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
                # Pre-NMS view: log top-5 raw scores so user can see what the
                # model thinks even when nothing passes conf_threshold.
                raw = outputs[i]   # shape (300, 7): [cx, cy, w, h, conf, cls, angle]
                if raw.shape[0] > 0:
                    raw_scores = raw[:, 4]
                    top_n = min(5, int((raw_scores > 0).sum()))
                    if top_n > 0:
                        order = np.argsort(-raw_scores)[:top_n]
                        peek = ", ".join(
                            f"{self.class_names[int(raw[k, 5])]}={float(raw[k, 4]):.3f}"
                            for k in order
                        )
                        logger.info(
                            f"[RotateOBB] top-{top_n} raw detections (pre-threshold {conf_threshold}): {peek}"
                        )
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


def _compute_need_flip(cap_box: np.ndarray, text_box: np.ndarray, angle_deg: float) -> bool:
    """Kiểm tra cần flip 180° không bằng cách tính vị trí text sau xoay (không xoay thật)."""
    cx, cy = float(cap_box[0]), float(cap_box[1])
    tx, ty = float(text_box[0]), float(text_box[1])
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    new_text = M @ np.array([tx, ty, 1.0])
    return float(new_text[1]) < cy


def _rotate_cap_region_only(image: np.ndarray, cap_box: np.ndarray, angle_deg: float,
                             need_flip: bool = False, margin: int = 20) -> np.ndarray:
    """
    Chỉ xoay vùng nắp chai (circular mask), background giữ nguyên.
    Trả về full image với cap đã xoay tại chỗ.
    """
    cx, cy, w, h, _ = cap_box
    radius = int(min(w, h) / 2)
    total_angle = angle_deg + (180 if need_flip else 0)

    crop_r = radius + margin
    x1 = max(0, int(cx - crop_r))
    y1 = max(0, int(cy - crop_r))
    x2 = min(image.shape[1], int(cx + crop_r))
    y2 = min(image.shape[0], int(cy + crop_r))

    crop = image[y1:y2, x1:x2].copy()
    local_cx = float(cx - x1)
    local_cy = float(cy - y1)

    M = cv2.getRotationMatrix2D((local_cx, local_cy), total_angle, 1.0)
    crop_rotated = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]),
                                   flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(local_cx), int(local_cy)), radius, 255, -1)

    result_crop = crop.copy()
    result_crop[mask > 0] = crop_rotated[mask > 0]

    full_result = image.copy()
    full_result[y1:y2, x1:x2] = result_crop
    return full_result


_DEBUG_DIR = Path(__file__).resolve().parents[3] / "logs" / "obb_rotation" / "debug"


def _maybe_save_debug(image: np.ndarray, tag: str, suffix: str = "") -> Optional[str]:
    """
    Save debug image to logs/obb_rotation/debug/ if env ROTATE_OBB_DEBUG=1.
    Returns the saved path or None.
    """
    if os.environ.get("ROTATE_OBB_DEBUG", "0") not in ("1", "true", "TRUE", "yes"):
        return None
    try:
        _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        name = f"{ts}_{tag}{suffix}.jpg"
        path = _DEBUG_DIR / name
        cv2.imwrite(str(path), image)
        return str(path)
    except Exception as e:
        logger.warning(f"[RotateOBB] save debug image failed: {e}")
        return None


def _annotate(
    image: np.ndarray, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray,
    class_names: List[str],
) -> np.ndarray:
    """Draw OBB corners + label on a copy of the image for debug visualization."""
    out = image.copy()
    for box, score, cls in zip(boxes, scores, class_ids):
        cx, cy, w, h, angle = box
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        hw, hh = w / 2, h / 2
        corners = np.array([
            [cx + (-hw * cos_a + hh * sin_a), cy + (-hw * sin_a - hh * cos_a)],
            [cx + ( hw * cos_a + hh * sin_a), cy + ( hw * sin_a - hh * cos_a)],
            [cx + ( hw * cos_a - hh * sin_a), cy + ( hw * sin_a + hh * cos_a)],
            [cx + (-hw * cos_a - hh * sin_a), cy + (-hw * sin_a + hh * cos_a)],
        ], dtype=np.int32)
        name = class_names[int(cls)] if int(cls) < len(class_names) else f"cls{int(cls)}"
        color = (0, 255, 0) if name == "bottle_cap" else (0, 200, 255)
        cv2.polylines(out, [corners], True, color, 3)
        cv2.putText(out, f"{name} {float(score):.2f}",
                    tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return out


def rotate_frame(image: np.ndarray, model: YOLOOBBInference) -> np.ndarray:
    """
    Detect OBB trên frame, chỉ xoay vùng nắp chai (circular mask), background giữ nguyên.
    Nếu không detect được text_box + bottle_cap → trả về ảnh gốc.

    Set env `ROTATE_OBB_DEBUG=1` to dump input/annotated/output images to
    logs/obb_rotation/debug/.
    """
    h_img, w_img = image.shape[:2]
    in_path = _maybe_save_debug(image, "input")
    if in_path:
        logger.info(f"[RotateOBB] debug input saved → {in_path}")

    results = model.predict([image], conf_threshold=0.2)
    boxes, scores, class_ids = results[0]

    # Always save the annotated view in debug mode so user can SEE what (if
    # anything) the model picked up — even when boxes is empty we still want
    # the input to inspect.
    if len(boxes) > 0:
        annot_path = _maybe_save_debug(
            _annotate(image, boxes, scores, class_ids, model.class_names),
            "annotated"
        )
        if annot_path:
            logger.info(f"[RotateOBB] debug annotated saved → {annot_path}")

    # Summarize detections per class (for visibility in INFO logs)
    det_summary = {}
    for cls_id, sc in zip(class_ids, scores):
        name = model.class_names[int(cls_id)]
        det_summary.setdefault(name, []).append(float(sc))
    if det_summary:
        summary_str = ", ".join(
            f"{name}×{len(scs)} (conf={','.join(f'{s:.2f}' for s in scs)})"
            for name, scs in det_summary.items()
        )
    else:
        summary_str = "none"
    logger.info(
        f"[RotateOBB] frame={w_img}x{h_img} → detected {len(boxes)} box(es): {summary_str}"
    )

    if len(boxes) == 0:
        logger.info("[RotateOBB] No detections → returning original frame")
        return image

    text_box_idx = next(
        (i for i, c in enumerate(class_ids) if model.class_names[int(c)] == "text_box"), None
    )
    bottle_cap_idx = next(
        (i for i, c in enumerate(class_ids) if model.class_names[int(c)] == "bottle_cap"), None
    )

    if text_box_idx is None or bottle_cap_idx is None:
        missing = []
        if text_box_idx is None: missing.append("text_box")
        if bottle_cap_idx is None: missing.append("bottle_cap")
        logger.warning(
            f"[RotateOBB] Missing required class(es): {missing} → returning original frame"
        )
        return image

    text_box = boxes[text_box_idx]
    cap_box = boxes[bottle_cap_idx]
    text_conf = float(scores[text_box_idx])
    cap_conf = float(scores[bottle_cap_idx])

    tcx, tcy, tw, th, text_angle = text_box
    ccx, ccy, cw, ch, _ = cap_box
    angle_deg = text_angle * 180 / np.pi
    if th > tw:
        angle_deg += 90

    need_flip = _compute_need_flip(cap_box, text_box, angle_deg)
    final_angle = angle_deg + (180 if need_flip else 0)

    logger.info(
        f"[RotateOBB] bottle_cap conf={cap_conf:.2f} center=({ccx:.0f},{ccy:.0f}) "
        f"size={cw:.0f}x{ch:.0f}  |  text_box conf={text_conf:.2f} "
        f"center=({tcx:.0f},{tcy:.0f}) size={tw:.0f}x{th:.0f} raw_angle={text_angle*180/np.pi:.1f}°"
    )
    logger.info(
        f"[RotateOBB] computed angle={angle_deg:.1f}° "
        f"{'(+180° flip — text was above cap center)' if need_flip else '(no flip)'} "
        f"→ final rotation={final_angle:.1f}°"
    )

    result = _rotate_cap_region_only(image, cap_box, angle_deg, need_flip, margin=20)
    out_path = _maybe_save_debug(result, "rotated")
    if out_path:
        logger.info(f"[RotateOBB] debug rotated saved → {out_path}")
    return result


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
