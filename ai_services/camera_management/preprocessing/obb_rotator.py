"""
OBB Rotation Service for Check_Color function type.

Rotates the FULL camera frame upright (before crop) using YOLO OBB to detect
text_box and bottle_cap orientation.

If rotation succeeds  → rotated frame is used for superpoint, OCR crop, and output.
If rotation fails     → original frame is used as-is.
"""

import logging
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import sys

logger = logging.getLogger(__name__)

HOME = os.environ.get('HOME', '/home/demo')

# ─── File logger for rotation results ─────────────────────────────────────────

_rotation_file_logger: Optional[logging.Logger] = None


def _get_rotation_file_logger() -> logging.Logger:
    """Lazy-init a dedicated file logger for rotation results."""
    global _rotation_file_logger
    if _rotation_file_logger is not None:
        return _rotation_file_logger

    log_dir = Path(HOME) / 'Source' / 'ocr_datecode' / 'ai_services' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'obb_rotation.log'

    _rotation_file_logger = logging.getLogger('obb_rotation')
    _rotation_file_logger.setLevel(logging.DEBUG)
    _rotation_file_logger.propagate = False  # don't bubble to root logger

    if not _rotation_file_logger.handlers:
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s  %(levelname)-8s  %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        _rotation_file_logger.addHandler(fh)

    logger.info(f"OBB rotation log: {log_path}")
    return _rotation_file_logger


# ─── Import YOLO OBB (same as product_verifier) ───────────────────────────────

try:
    ai_services_path = Path(__file__).parent.parent.parent
    if str(ai_services_path) not in sys.path:
        sys.path.insert(0, str(ai_services_path))
    from yolo_obb_tensorrt import YOLOOBBTensorRT
    YOLO_OBB_AVAILABLE = True
except ImportError as e:
    logger.warning(f"YOLO OBB not available for OBBRotationService: {e}")
    YOLO_OBB_AVAILABLE = False
    YOLOOBBTensorRT = None


# ─── Helper functions (ported from tests/rotate_text_obb_trt.py) ──────────────

def rotate_image_by_obb(image: np.ndarray, box: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Xoay ảnh để vùng text nằm chính diện dựa vào OBB.

    Args:
        image: Input image (BGR)
        box: [cx, cy, w, h, angle] của text_box

    Returns:
        (rotated_image, angle_deg_used, rotation_matrix_M)
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


def transform_boxes(boxes: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Transform OBB boxes theo ma trận xoay.

    Args:
        boxes: [[cx, cy, w, h, angle], ...]
        M: 2x3 affine rotation matrix

    Returns:
        Transformed boxes array
    """
    transformed_boxes = []
    for box in boxes:
        cx, cy, w, h, angle = box

        rect = ((cx, cy), (w, h), angle * 180 / np.pi)
        box_points = cv2.boxPoints(rect)

        transformed_points = []
        for point in box_points:
            pt = np.array([point[0], point[1], 1])
            new_pt = M @ pt
            transformed_points.append(new_pt[:2])

        transformed_points = np.array(transformed_points)
        new_rect = cv2.minAreaRect(transformed_points.astype(np.float32))
        new_cx, new_cy = new_rect[0]
        new_w, new_h = new_rect[1]
        new_angle = new_rect[2] * np.pi / 180

        transformed_boxes.append([new_cx, new_cy, new_w, new_h, new_angle])

    return np.array(transformed_boxes)


def transform_crop_area(crop_area: Dict[str, int], M: np.ndarray) -> Dict[str, int]:
    """
    Transform crop_area {x1,y1,x2,y2} sang tọa độ trong rotated frame.

    Args:
        crop_area: {'x1': int, 'y1': int, 'x2': int, 'y2': int}
        M: 2x3 affine rotation matrix

    Returns:
        Transformed crop_area dict (clamped to >= 0)
    """
    x1, y1 = crop_area['x1'], crop_area['y1']
    x2, y2 = crop_area['x2'], crop_area['y2']

    corners = np.array([
        [x1, y1, 1],
        [x2, y1, 1],
        [x2, y2, 1],
        [x1, y2, 1],
    ], dtype=np.float32)

    transformed = (M @ corners.T).T

    new_x1 = max(0, int(np.floor(np.min(transformed[:, 0]))))
    new_y1 = max(0, int(np.floor(np.min(transformed[:, 1]))))
    new_x2 = int(np.ceil(np.max(transformed[:, 0])))
    new_y2 = int(np.ceil(np.max(transformed[:, 1])))

    return {'x1': new_x1, 'y1': new_y1, 'x2': new_x2, 'y2': new_y2}


def _combine_affine(M2: np.ndarray, M1: np.ndarray) -> np.ndarray:
    """Kết hợp 2 affine: áp dụng M1 trước rồi M2. M_combined(x) = M2(M1(x))"""
    A1 = np.vstack([M1, [0, 0, 1]])
    A2 = np.vstack([M2, [0, 0, 1]])
    return (A2 @ A1)[:2, :]


# ─── Service class ─────────────────────────────────────────────────────────────

class OBBRotationService:
    """
    Service xoay FULL frame cho Check_Color trước khi crop và đưa vào MatcherFactory.

    Logic:
      1. YOLO OBB detect text_box + bottle_cap trên full frame
      2. Xoay theo text_box OBB
      3. Flip 180° nếu text nằm trên bottle_cap
      4. Trả về (rotated_frame, M) nếu thành công; (original_frame, None) nếu thất bại
    """

    CLASS_NAMES = ['bottle_cap', 'text_box']

    def __init__(self, engine_path: str, conf_threshold: float = 0.4,
                 inverse_transform: bool = True):
        self.conf_threshold = conf_threshold
        self.inverse_transform = inverse_transform

        self._model: Optional['YOLOOBBTensorRT'] = None
        self._rot_logger = _get_rotation_file_logger()

        if not YOLO_OBB_AVAILABLE:
            logger.error("OBBRotationService: YOLOOBBTensorRT not available")
            self._rot_logger.error("INIT FAILED — YOLOOBBTensorRT not available")
            return

        try:
            self._model = YOLOOBBTensorRT(
                engine_path=engine_path,
                class_names=self.CLASS_NAMES,
            )
            logger.info(f"OBBRotationService initialized: {engine_path}, inverse_transform={self.inverse_transform}")
            self._rot_logger.info(f"OBBRotationService initialized: {engine_path}, inverse_transform={self.inverse_transform}")
        except Exception as e:
            logger.error(f"OBBRotationService: Failed to load engine: {e}")
            self._rot_logger.error(f"INIT FAILED — engine load error: {e}")
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def rotate_frame(
        self,
        frame: np.ndarray,
        frame_tag: str = ""
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Detect OBB trên full frame và xoay về hướng chuẩn.

        Args:
            frame: Full camera frame (BGR numpy array) — TRƯỚC khi crop
            frame_tag: Optional tag để log (vd. serial_number + frame_idx)

        Returns:
            (result_frame, M) trong đó:
            - result_frame: rotated frame nếu thành công, hoặc frame gốc nếu thất bại
            - M: combined affine matrix (None nếu không xoay)
        """
        tag = f"[{frame_tag}] " if frame_tag else ""

        if self._model is None:
            self._rot_logger.warning(f"{tag}SKIP — model not loaded")
            return frame, None

        try:
            results, timing = self._model.predict(
                [frame], conf_threshold=self.conf_threshold, return_timing=True
            )
            boxes, scores, class_ids = results[0]

            infer_ms = timing.get('total', 0)

            if len(boxes) == 0:
                self._rot_logger.info(
                    f"{tag}FAIL — no boxes detected  "
                    f"(infer={infer_ms:.1f}ms)"
                )
                return frame, None

            # Tìm text_box và bottle_cap
            text_box_idx = next(
                (i for i, c in enumerate(class_ids)
                 if self.CLASS_NAMES[int(c)] == 'text_box'), None
            )
            bottle_cap_idx = next(
                (i for i, c in enumerate(class_ids)
                 if self.CLASS_NAMES[int(c)] == 'bottle_cap'), None
            )

            if text_box_idx is None:
                self._rot_logger.info(
                    f"{tag}FAIL — text_box not detected "
                    f"(found {len(boxes)} boxes, no text_box, infer={infer_ms:.1f}ms)"
                )
                return frame, None

            # Bước 1: xoay theo text_box
            rotated, angle_used, M1 = rotate_image_by_obb(frame, boxes[text_box_idx])
            transformed_boxes = transform_boxes(boxes, M1)
            M_combined = M1
            flipped = False

            # Bước 2: flip 180° nếu text nằm trên bottle_cap
            if bottle_cap_idx is not None:
                text_cy = transformed_boxes[text_box_idx][1]
                cap_cy  = transformed_boxes[bottle_cap_idx][1]

                if text_cy < cap_cy:
                    h_img, w_img = rotated.shape[:2]
                    M_flip = cv2.getRotationMatrix2D((w_img / 2, h_img / 2), 180, 1.0)
                    rotated = cv2.warpAffine(rotated, M_flip, (w_img, h_img),
                                             flags=cv2.INTER_LINEAR,
                                             borderValue=(114, 114, 114))
                    M_combined = _combine_affine(M_flip, M1)
                    flipped = True

            total_angle = angle_used + (180.0 if flipped else 0.0)
            flip_str    = " + flip180" if flipped else ""
            cap_str     = f"cap_detected=True" if bottle_cap_idx is not None else "cap_detected=False"

            self._rot_logger.info(
                f"{tag}OK — angle={angle_used:.1f}°{flip_str} total={total_angle:.1f}°  "
                f"{cap_str}  infer={infer_ms:.1f}ms"
            )

            return rotated, M_combined

        except Exception as e:
            logger.error(f"OBBRotationService.rotate_frame error: {e}")
            self._rot_logger.error(f"{tag}ERROR — {e}")
            return frame, None


def inverse_transform_bboxes(match_result: dict, M: np.ndarray) -> dict:
    """
    Inverse transform bbox coords từ rotated frame → original frame.

    Transform chain (forward):
        original frame → rotate(M) → rotated frame → crop → match_batch → bboxes
        → _transform_func (crop offset) → full-rotated coords

    This function applies the inverse of M to bring bboxes back to original coords.

    Args:
        match_result: Dict chứa 'transformed_bboxes' và optionally 'matched_bbox'
        M: 2x3 affine rotation matrix đã dùng khi rotate frame

    Returns:
        match_result với bboxes đã được inverse transform (in-place modification)
    """
    M_inv = cv2.invertAffineTransform(M)

    for bbox in match_result.get('transformed_bboxes', []):
        _apply_affine_to_bbox(bbox, M_inv)

    if match_result.get('matched_bbox'):
        _apply_affine_to_bbox(match_result['matched_bbox'], M_inv)

    return match_result


def _apply_affine_to_bbox(bbox: dict, M: np.ndarray):
    """Apply 2x3 affine matrix to a single bbox dict (in-place)."""
    # Transform 'points' list if present
    if bbox.get('points'):
        pts = np.array(bbox['points'], dtype=np.float32)
        ones = np.ones((len(pts), 1), dtype=np.float32)
        transformed = (M @ np.hstack([pts, ones]).T).T
        bbox['points'] = transformed.tolist()

    # Transform center coordinates
    for x_key, y_key in [('x', 'y'), ('cx', 'cy'), ('center_x', 'center_y')]:
        if x_key in bbox and y_key in bbox:
            pt = np.array([bbox[x_key], bbox[y_key], 1.0], dtype=np.float32)
            new_pt = M @ pt
            bbox[x_key] = float(new_pt[0])
            bbox[y_key] = float(new_pt[1])
