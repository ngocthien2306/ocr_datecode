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

# crop_area sub-slicing helper — shared with CV rotator (search restricted
# to user-defined product region; results offset back to full-frame coords).
from .cv_rotator import _slice_by_crop_area

logger = logging.getLogger(__name__)

HOME = os.environ.get('HOME', '/home/demo')

# ─── File logger for rotation results ─────────────────────────────────────────

_rotation_file_logger: Optional[logging.Logger] = None


def _get_rotation_file_logger() -> logging.Logger:
    """Lazy-init a dedicated file logger for rotation results.

    Wraps the daily file handler in a MemoryHandler so INFO writes get buffered
    (~5-10ms saved per frame on sync disk I/O). WARNING+ auto-flush ngay để
    không mất diagnostic khi có lỗi.
    """
    global _rotation_file_logger
    if _rotation_file_logger is not None:
        return _rotation_file_logger

    from logging_config import make_handler
    from logging.handlers import MemoryHandler

    _rotation_file_logger = logging.getLogger('obb_rotation')
    _rotation_file_logger.setLevel(logging.DEBUG)
    _rotation_file_logger.propagate = False  # don't bubble to root logger

    if not any(getattr(h, "_marker", None) == "daily-obb_rotation" for h in _rotation_file_logger.handlers):
        fh = make_handler(
            "obb_rotation",
            level=logging.DEBUG,
            fmt='%(asctime)s  %(levelname)-8s  %(message)s',
        )
        fh.formatter.datefmt = '%Y-%m-%d %H:%M:%S'
        # Buffer up to 200 INFO records; auto-flush on WARNING+ hoặc khi handler đóng.
        buffered = MemoryHandler(capacity=200, flushLevel=logging.WARNING, target=fh)
        setattr(buffered, "_marker", "daily-obb_rotation")
        _rotation_file_logger.addHandler(buffered)

    logger.info("OBB rotation log: obb_rotation/{date}.log (buffered)")
    return _rotation_file_logger


# Helper: derive a generous bounding circle from a YOLO OBB cap_box.
# Replaces the need to run HoughCircles on the same frame — saves ~55ms.
def _obb_cap_box_to_circle(cap_box: np.ndarray, margin: float = 8.0) -> Tuple[float, float, float]:
    """
    Convert OBB cap_box (cx, cy, w, h, angle) → (cx, cy, radius) with margin.
    Use max(w, h)/2 so the circle bao trùm toàn bộ cap (an toàn cho downstream crop).
    """
    cx, cy, w, h, _ = cap_box
    r = max(float(w), float(h)) / 2.0 + float(margin)
    return (float(cx), float(cy), r)


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

def rotate_cap_region_only(image: np.ndarray, cap_box: np.ndarray, angle_deg: float,
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

    # View thay vì copy — warpAffine không modify input nên không cần copy.
    # Tiết kiệm ~2ms / call so với image[y1:y2, x1:x2].copy() trước đây.
    crop_view = image[y1:y2, x1:x2]
    local_cx = float(cx - x1)
    local_cy = float(cy - y1)

    M = cv2.getRotationMatrix2D((local_cx, local_cy), total_angle, 1.0)
    crop_rotated = cv2.warpAffine(
        crop_view, M, (crop_view.shape[1], crop_view.shape[0]),
        flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114),
    )

    mask = np.zeros(crop_view.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(local_cx), int(local_cy)), radius, 255, -1)

    # Copy full frame once cho output (cần thiết — không in-place vào input).
    # Bỏ bước trung gian `result_crop = crop.copy()` — apply mask-blend trực tiếp
    # vào sub-region của full_result. Tiết kiệm thêm ~2ms / call.
    full_result = image.copy()
    sub = full_result[y1:y2, x1:x2]
    sub[mask > 0] = crop_rotated[mask > 0]
    return full_result


def compute_need_flip(cap_box: np.ndarray, text_box: np.ndarray, angle_deg: float) -> bool:
    """Kiểm tra cần flip 180° không bằng cách tính vị trí text sau xoay (không xoay thật)."""
    cx, cy = float(cap_box[0]), float(cap_box[1])
    tx, ty = float(text_box[0]), float(text_box[1])
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    new_text = M @ np.array([tx, ty, 1.0])
    return float(new_text[1]) < cy


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
        frame_tag: str = "",
        crop_area: Optional[Dict[str, int]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Tuple[float, float, float]]]:
        """
        Detect OBB trên full frame và xoay về hướng chuẩn.

        Args:
            frame: Full camera frame (BGR numpy array) — TRƯỚC khi crop
            frame_tag: Optional tag để log (vd. serial_number + frame_idx)

        Returns:
            (result_frame, M, cap_circle) trong đó:
            - result_frame: rotated frame nếu thành công, hoặc frame gốc nếu thất bại
            - M: combined affine matrix (None nếu không xoay)
            - cap_circle: (cx, cy, r) derived from OBB cap_box, dùng cho cap_crop
              step bên downstream (tránh chạy HoughCircles lần 2). None nếu fail.
        """
        tag = f"[{frame_tag}] " if frame_tag else ""

        if self._model is None:
            self._rot_logger.warning(f"{tag}SKIP — model not loaded")
            return frame, None, None

        try:
            import time as _time
            _t_total_start = _time.perf_counter()
            # Restrict OBB inference to user-defined product crop_area so we
            # don't detect bottles outside that region (multi-bottle FOV).
            # Boxes returned from the sub-frame are offset back to full coords
            # before any downstream use.
            ox, oy, sub_frame = _slice_by_crop_area(frame, crop_area)
            if sub_frame is None:
                self._rot_logger.info(
                    f"{tag}FAIL — crop_area outside frame bounds"
                )
                return frame, None, None
            results, timing = self._model.predict(
                [sub_frame], conf_threshold=self.conf_threshold, return_timing=True
            )
            boxes, scores, class_ids = results[0]
            if crop_area and len(boxes) > 0:
                boxes = boxes.copy()
                boxes[:, 0] += ox
                boxes[:, 1] += oy

            infer_ms = timing.get('total', 0)

            if len(boxes) == 0:
                self._rot_logger.info(
                    f"{tag}FAIL — no boxes detected  "
                    f"(crop_area={'on' if crop_area else 'off'}, "
                    f"infer={infer_ms:.1f}ms)"
                )
                return frame, None, None

            # Tìm text_box và bottle_cap
            text_box_idx = next(
                (i for i, c in enumerate(class_ids)
                 if self.CLASS_NAMES[int(c)] == 'text_box'), None
            )
            bottle_cap_idx = next(
                (i for i, c in enumerate(class_ids)
                 if self.CLASS_NAMES[int(c)] == 'bottle_cap'), None
            )

            if text_box_idx is None or bottle_cap_idx is None:
                self._rot_logger.info(
                    f"{tag}FAIL — text_box or bottle_cap not detected "
                    f"(found {len(boxes)} boxes, infer={infer_ms:.1f}ms)"
                )
                return frame, None, None

            text_box = boxes[text_box_idx]
            cap_box  = boxes[bottle_cap_idx]
            # Derive cap_circle from OBB cap_box — downstream cap_crop step
            # sẽ dùng cái này thay vì gọi HoughCircles trên cùng frame nữa.
            cap_circle = _obb_cap_box_to_circle(cap_box)

            # Tính góc xoay từ text_box
            _, _, tw, th, text_angle = text_box
            angle_deg = text_angle * 180 / np.pi
            if th > tw:
                angle_deg += 90

            # Shape-match flip detection (100% accurate when text is near cap
            # center, unlike the legacy compute_need_flip heuristic which is
            # fragile in that case). Reuse the same logic as the pure-CV path.
            _t_flip_start = _time.perf_counter()
            try:
                from .cv_rotator import _need_flip as _cv_need_flip
                cap_cx, cap_cy, cap_w, cap_h, _ = cap_box
                cap_r = max(float(cap_w), float(cap_h)) / 2.0
                gray = (
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if frame.ndim == 3 else frame
                )
                flipped, s0, s180 = _cv_need_flip(
                    gray, (float(cap_cx), float(cap_cy), cap_r), angle_deg
                )
                flip_method = f"shape-match (s0={s0:.3f} s180={s180:.3f})"
            except Exception as e:
                # Fall back to old heuristic if shape-match fails for any reason
                self._rot_logger.warning(
                    f"{tag}shape-match failed ({e}), falling back to legacy heuristic"
                )
                flipped = compute_need_flip(cap_box, text_box, angle_deg)
                flip_method = "legacy heuristic"
            _t_flip = (_time.perf_counter() - _t_flip_start) * 1000

            # Chỉ xoay vùng cap (circular mask), background giữ nguyên
            _t_rot_start = _time.perf_counter()
            result = rotate_cap_region_only(frame, cap_box, angle_deg, flipped)
            _t_rot = (_time.perf_counter() - _t_rot_start) * 1000

            _t_total = (_time.perf_counter() - _t_total_start) * 1000
            total_angle = angle_deg + (180.0 if flipped else 0.0)
            flip_str    = " + flip180" if flipped else ""

            self._rot_logger.info(
                f"{tag}OK in {_t_total:.1f}ms — angle={angle_deg:.1f}°{flip_str} "
                f"total={total_angle:.1f}°  cap_only=True  "
                f"infer={infer_ms:.1f}ms  need_flip={_t_flip:.1f}ms  "
                f"rotate={_t_rot:.1f}ms  flip_via={flip_method}"
            )

            return result, None, cap_circle

        except Exception as e:
            logger.error(f"OBBRotationService.rotate_frame error: {e}")
            self._rot_logger.error(f"{tag}ERROR — {e}")
            return frame, None, None

    def rotate_frame_dual(
        self,
        frame: np.ndarray,
        frame_tag: str = "",
        crop_area: Optional[Dict[str, int]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[float, float, float]]]:
        """
        Produce BOTH rotation candidates (no-flip + flip180) without using the
        shape-match `_need_flip` heuristic. The caller picks the higher-scoring
        candidate via match confidence.

        Returns (candidate_no_flip, candidate_flipped180, cap_circle):
        - candidate_no_flip / candidate_flipped180: rotation candidates; either
          may be None on failure (no cap detected, no text_box, etc.).
        - cap_circle: (cx, cy, r) derived from OBB cap_box — downstream skip
          HoughCircles bằng cách dùng cái này. None khi rotation fail.
        """
        tag = f"[{frame_tag}] " if frame_tag else ""
        if self._model is None or frame is None or frame.size == 0:
            return None, None, None
        try:
            import time as _time
            _t0 = _time.perf_counter()
            # Restrict OBB inference + CV fallback to user-defined crop_area.
            # Boxes/cap detected on sub_frame are offset back to full coords
            # before any downstream rotation/return.
            ox, oy, sub_frame = _slice_by_crop_area(frame, crop_area)
            if sub_frame is None:
                self._rot_logger.info(
                    f"{tag}DUAL FAIL — crop_area outside frame bounds"
                )
                return None, None, None
            results, timing = self._model.predict(
                [sub_frame], conf_threshold=self.conf_threshold, return_timing=True
            )
            boxes, scores, class_ids = results[0]
            if crop_area and len(boxes) > 0:
                boxes = boxes.copy()
                boxes[:, 0] += ox
                boxes[:, 1] += oy
            infer_ms = timing.get('total', 0)

            if len(boxes) == 0:
                self._rot_logger.info(
                    f"{tag}DUAL FAIL — no boxes "
                    f"(crop_area={'on' if crop_area else 'off'}, "
                    f"infer={infer_ms:.1f}ms)"
                )
                return None, None, None
            text_box_idx = next(
                (i for i, c in enumerate(class_ids)
                 if self.CLASS_NAMES[int(c)] == 'text_box'), None
            )
            bottle_cap_idx = next(
                (i for i, c in enumerate(class_ids)
                 if self.CLASS_NAMES[int(c)] == 'bottle_cap'), None
            )
            if text_box_idx is None or bottle_cap_idx is None:
                # OBB thiếu 1 trong 2 box → fallback CV: HoughCircles tìm cap,
                # dark-pixel histogram (`_text_angle`) tìm góc text. OBB cap_box
                # angle vốn không align với text (label được vẽ axis-aligned),
                # nên không thể tận dụng được cái có sẵn từ OBB — cứ chạy CV
                # lại từ đầu cho đơn giản. CV cũng chạy trong sub-frame.
                self._rot_logger.info(
                    f"{tag}DUAL OBB miss → CV fallback "
                    f"(text_box={text_box_idx is not None}, "
                    f"bottle_cap={bottle_cap_idx is not None}, "
                    f"infer={infer_ms:.1f}ms)"
                )
                try:
                    from .cv_rotator import _detect_cap, _text_angle, _rotate_cap_region
                    gray = (
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        if frame.ndim == 3 else frame
                    )
                    sub_gray = (
                        gray[oy:oy + sub_frame.shape[0], ox:ox + sub_frame.shape[1]]
                        if crop_area else gray
                    )
                    cap_sub = _detect_cap(sub_gray)
                    if cap_sub is None:
                        self._rot_logger.info(
                            f"{tag}DUAL CV-fallback FAIL — no cap detected"
                        )
                        return None, None, None
                    cap_full = (cap_sub[0] + ox, cap_sub[1] + oy, cap_sub[2])
                    angle_deg, n_dark = _text_angle(sub_gray, cap_sub)
                    cand_a = _rotate_cap_region(frame, cap_full, angle_deg, False)
                    cand_b = _rotate_cap_region(frame, cap_full, angle_deg, True)
                    _t_total = (_time.perf_counter() - _t0) * 1000
                    self._rot_logger.info(
                        f"{tag}DUAL CV-fallback OK in {_t_total:.1f}ms — "
                        f"angle={angle_deg:.1f}° n_dark={n_dark} "
                        f"both candidates emitted"
                    )
                    return cand_a, cand_b, (float(cap_full[0]), float(cap_full[1]), float(cap_full[2]))
                except Exception as cv_err:
                    logger.error(f"OBB→CV fallback error: {cv_err}")
                    self._rot_logger.error(
                        f"{tag}DUAL CV-fallback ERROR — {cv_err}"
                    )
                    return None, None, None
            text_box = boxes[text_box_idx]
            cap_box  = boxes[bottle_cap_idx]
            # Derive cap_circle ngay sau khi có cap_box → return cùng candidates.
            cap_circle = _obb_cap_box_to_circle(cap_box)
            _, _, tw, th, text_angle = text_box
            angle_deg = text_angle * 180 / np.pi
            if th > tw:
                angle_deg += 90
            # Apply rotation WITHOUT flip and WITH flip — let downstream choose.
            # 2 warpAffine độc lập, cv2 release GIL → chạy song song 2 threads
            # tiết kiệm ~30ms so với serial.
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as _pool:
                _fa = _pool.submit(rotate_cap_region_only, frame, cap_box, angle_deg, False)
                _fb = _pool.submit(rotate_cap_region_only, frame, cap_box, angle_deg, True)
                candidate_a = _fa.result()
                candidate_b = _fb.result()
            _t_total = (_time.perf_counter() - _t0) * 1000
            self._rot_logger.info(
                f"{tag}DUAL OK in {_t_total:.1f}ms — angle={angle_deg:.1f}° "
                f"both candidates emitted (infer={infer_ms:.1f}ms)"
            )
            return candidate_a, candidate_b, cap_circle
        except Exception as e:
            logger.error(f"OBBRotationService.rotate_frame_dual error: {e}")
            self._rot_logger.error(f"{tag}DUAL ERROR — {e}")
            return None, None, None


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
