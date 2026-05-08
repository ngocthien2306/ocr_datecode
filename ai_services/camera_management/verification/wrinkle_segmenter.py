"""
WrinkledSegmenterTRT
Production TensorRT segmentation model for wrinkled label detection.
Input: cropped product images (từ OBB crop)
Output: wrinkled regions back-projected lên original frame space
"""

import logging
import cv2
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    logger.warning("TensorRT/PyCUDA not available — WrinkledSegmenterTRT disabled")


class WrinkledSegmenterTRT:
    """
    YOLO Instance Segmentation via TensorRT cho wrinkled label detection.

    Pipeline:
        crop (từ OBB product box) → TRT inference → masks crop space
        → back-project masks → original frame space → filter area > min_area
        → contour extraction (JSON serializable)

    Engine build command:
        trtexec --onnx=best_wrinkled_instance_segmentation_crop_bottle.onnx
                --saveEngine=...engine --fp16
                --minShapes=images:1x3x320x320
                --optShapes=images:4x3x320x320
                --maxShapes=images:8x3x320x320
    """

    CROP_PADDING = 15

    def __init__(
        self,
        engine_path: str,
        class_names: Optional[List[str]] = None,
        min_area: int = 2000,
    ):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")

        self.class_names = class_names or ['wrinkled']
        self.min_area    = min_area

        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream   = cuda.Stream()

        n = self.engine.num_io_tensors
        names = [self.engine.get_tensor_name(i) for i in range(n)]
        self.input_name   = names[0]
        self.output_names = names[1:]   # [det_name, proto_name]

        # Get img_size and max_batch from profile
        profile       = self.engine.get_tensor_profile_shape(self.input_name, 0)
        max_shape     = profile[2]                  # (max_batch, 3, H, W)
        self.max_batch = max_shape[0]
        self.img_size  = max_shape[2]               # H = W = 320

        # Query output shapes at batch=1
        self.context.set_input_shape(self.input_name, (1, 3, self.img_size, self.img_size))
        self._out_shapes: Dict[str, tuple] = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            self._out_shapes[name] = shape

        self._alloc_buffers()
        logger.info(
            f"WrinkledSegmenterTRT loaded: {engine_path} "
            f"img_size={self.img_size} max_batch={self.max_batch} min_area={min_area}px"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Buffer allocation
    # ──────────────────────────────────────────────────────────────────────────
    def _alloc_buffers(self):
        s, mb = self.img_size, self.max_batch
        self.h_input = cuda.pagelocked_empty(mb * 3 * s * s, np.float32)
        self.d_input  = cuda.mem_alloc(self.h_input.nbytes)
        self.h_outs, self.d_outs = {}, {}
        for name, shape in self._out_shapes.items():
            size = int(np.prod(shape)) * self.max_batch
            h = cuda.pagelocked_empty(size, np.float32)
            d = cuda.mem_alloc(h.nbytes)
            self.h_outs[name] = h
            self.d_outs[name] = d

    # ──────────────────────────────────────────────────────────────────────────
    # Preprocess (single image → tensor)
    # ──────────────────────────────────────────────────────────────────────────
    def _preprocess_one(self, image: np.ndarray):
        s = self.img_size
        h, w = image.shape[:2]
        scale = min(s / w, s / h)
        new_w, new_h = int(w * scale), int(h * scale)

        padded   = np.full((s, s, 3), 114, dtype=np.uint8)
        pad_top  = (s - new_h) // 2
        pad_left = (s - new_w) // 2
        padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = cv2.resize(image, (new_w, new_h))

        tensor = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
        return tensor, scale, (pad_top, pad_left)

    # ──────────────────────────────────────────────────────────────────────────
    # Predict batch — returns list of (boxes_cxcywh [N,6], masks [N,H_crop,W_crop])
    # ──────────────────────────────────────────────────────────────────────────
    def predict_batch(
        self,
        images: List[np.ndarray],
        conf_threshold: float = 0.3,
        return_timing: bool = False,
    ):
        timing = {}
        batch_size = len(images)
        if batch_size == 0:
            return ([], timing) if return_timing else []
        if batch_size > self.max_batch:
            raise ValueError(f"Batch size {batch_size} > max_batch {self.max_batch}")

        s = self.img_size

        # ── Preprocess ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        tensors, scales, pads = [], [], []
        for img in images:
            t, sc, pd = self._preprocess_one(img)
            tensors.append(t); scales.append(sc); pads.append(pd)
        batch_input = np.stack(tensors)                     # (B, 3, H, W)
        timing['preprocess'] = (time.perf_counter() - t0) * 1000

        # ── H2D ─────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        self.context.set_input_shape(self.input_name, (batch_size, 3, s, s))
        np.copyto(self.h_input[:batch_input.size], batch_input.ravel())
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        for name in self.output_names:
            self.context.set_tensor_address(name, int(self.d_outs[name]))
        self.stream.synchronize()
        timing['h2d'] = (time.perf_counter() - t0) * 1000

        # ── Inference ───────────────────────────────────────────────────────
        t0 = time.perf_counter()
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        self.stream.synchronize()
        timing['inference'] = (time.perf_counter() - t0) * 1000

        # ── D2H ─────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        for name in self.output_names:
            cuda.memcpy_dtoh_async(self.h_outs[name], self.d_outs[name], self.stream)
        self.stream.synchronize()
        timing['d2h'] = (time.perf_counter() - t0) * 1000

        # ── Postprocess ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        det_name   = self.output_names[0]
        proto_name = self.output_names[1] if len(self.output_names) > 1 else None

        det_shape  = tuple(self.context.get_tensor_shape(det_name))    # (B, N, 38)
        det_out    = self.h_outs[det_name][:int(np.prod(det_shape))].reshape(det_shape)

        protos_out = None
        if proto_name:
            p_shape    = tuple(self.context.get_tensor_shape(proto_name))  # (B, 32, mh, mw)
            protos_out = self.h_outs[proto_name][:int(np.prod(p_shape))].reshape(p_shape)

        results = []
        for i in range(batch_size):
            boxes_raw = det_out[i]
            valid = boxes_raw[:, 4] >= conf_threshold
            boxes = boxes_raw[valid]

            if len(boxes) == 0:
                results.append((np.array([]), None))
                continue

            box_coords  = boxes[:, :6].copy()
            mask_coeffs = boxes[:, 6:]
            protos      = protos_out[i] if protos_out is not None else None

            masks = None
            if protos is not None:
                masks = self._decode_masks(protos, mask_coeffs, box_coords,
                                           images[i].shape, scales[i], pads[i])

            # Scale boxes từ padded space → image coords (xyxy → cxcywh)
            pad_top, pad_left = pads[i]
            sc = scales[i]
            box_coords[:, 0] = (box_coords[:, 0] - pad_left) / sc
            box_coords[:, 1] = (box_coords[:, 1] - pad_top)  / sc
            box_coords[:, 2] = (box_coords[:, 2] - pad_left) / sc
            box_coords[:, 3] = (box_coords[:, 3] - pad_top)  / sc

            x1, y1, x2, y2 = box_coords[:, 0], box_coords[:, 1], box_coords[:, 2], box_coords[:, 3]
            box_coords[:, 0] = (x1 + x2) / 2
            box_coords[:, 1] = (y1 + y2) / 2
            box_coords[:, 2] = x2 - x1
            box_coords[:, 3] = y2 - y1

            results.append((box_coords, masks))

        timing['postprocess'] = (time.perf_counter() - t0) * 1000
        timing['total']       = sum(timing.values())
        timing['per_image']   = timing['total'] / batch_size

        return (results, timing) if return_timing else results

    # ──────────────────────────────────────────────────────────────────────────
    # Mask decoding: coeffs × protos → binary masks in crop image space
    # ──────────────────────────────────────────────────────────────────────────
    def _decode_masks(self, protos, mask_coeffs, boxes_xyxy_s, orig_shape, scale, pad):
        pad_top, pad_left = pad
        s = self.img_size
        c, mh, mw = protos.shape
        h_orig, w_orig = orig_shape[:2]

        # Sigmoid + matmul in proto space (small, fast)
        masks = 1 / (1 + np.exp(-np.matmul(mask_coeffs, protos.reshape(c, -1)).reshape(-1, mh, mw)))

        # Threshold early in proto space → uint8 → single resize NEAREST
        ratio    = mh / s
        pt       = int(pad_top  * ratio)
        pl       = int(pad_left * ratio)
        new_mh   = int(h_orig * scale * ratio)
        new_mw   = int(w_orig * scale * ratio)
        unpadded = (masks[:, pt:pt + new_mh, pl:pl + new_mw] > 0.5).astype(np.uint8)

        masks_orig = np.array([
            cv2.resize(m, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            for m in unpadded
        ], dtype=np.float32)

        # Crop mask theo bbox
        for i, b in enumerate(boxes_xyxy_s):
            x1 = max(0, int((b[0] - pad_left) / scale))
            y1 = max(0, int((b[1] - pad_top)  / scale))
            x2 = min(w_orig, int((b[2] - pad_left) / scale))
            y2 = min(h_orig, int((b[3] - pad_top)  / scale))
            masks_orig[i, :y1, :] = 0; masks_orig[i, y2:, :] = 0
            masks_orig[i, :, :x1] = 0; masks_orig[i, :, x2:] = 0

        return masks_orig

    # ──────────────────────────────────────────────────────────────────────────
    # Crop product region from frame theo OBB box
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def crop_from_obb(
        frame: np.ndarray,
        cx: float, cy: float,
        w: float,  h: float,
        angle: float,
        padding: int = CROP_PADDING,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Crop product region từ frame theo OBB (xoay thẳng đứng).

        Returns:
            crop        : BGR image (axis-aligned)
            crop_offset : (x1c, y1c) — vị trí thực trong rotated frame (sau clip)
        """
        img_h, img_w = frame.shape[:2]
        angle_deg = np.degrees(angle)

        M       = cv2.getRotationMatrix2D((float(cx), float(cy)), angle_deg, 1.0)
        rotated = cv2.warpAffine(frame, M, (img_w, img_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(114, 114, 114))

        x1c = max(0,      int(cx - w / 2) - padding)
        y1c = max(0,      int(cy - h / 2) - padding)
        x2c = min(img_w,  int(cx + w / 2) + padding)
        y2c = min(img_h,  int(cy + h / 2) + padding)

        return rotated[y1c:y2c, x1c:x2c], (x1c, y1c)

    # ──────────────────────────────────────────────────────────────────────────
    # Back-project masks từ crop space → original frame space
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def back_project_masks(
        masks_crop:   np.ndarray,          # (N, H_crop, W_crop) float32 binary
        cx: float, cy: float, angle: float,
        crop_offset:  Tuple[int, int],     # (x1c, y1c) actual clip position
        frame_shape:  Tuple[int, ...],
    ) -> List[np.ndarray]:
        """
        Đặt mask vào vị trí đúng trong rotated frame space,
        sau đó inverse-rotate về original frame space.
        """
        img_h, img_w = frame_shape[:2]
        x1c, y1c     = crop_offset
        angle_deg    = np.degrees(angle)
        M_inv        = cv2.getRotationMatrix2D((float(cx), float(cy)), -angle_deg, 1.0)

        frame_masks = []
        for mask in masks_crop:
            crop_h, crop_w = mask.shape
            canvas = np.zeros((img_h, img_w), dtype=np.float32)

            # Đặt mask đúng vị trí trong rotated frame
            dst_h = min(crop_h, img_h - y1c)
            dst_w = min(crop_w, img_w - x1c)
            if dst_h > 0 and dst_w > 0:
                canvas[y1c:y1c + dst_h, x1c:x1c + dst_w] = mask[:dst_h, :dst_w]

            # Inverse rotate về original frame
            mask_frame = cv2.warpAffine(
                canvas, M_inv, (img_w, img_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            frame_masks.append(mask_frame)

        return frame_masks

    # ──────────────────────────────────────────────────────────────────────────
    # Build wrinkled_check dict từ segmentation result
    # ──────────────────────────────────────────────────────────────────────────
    def build_wrinkled_check(
        self,
        seg_boxes:    np.ndarray,          # [N, 6] cxcywh + conf + cls  (crop space)
        seg_masks:    Optional[np.ndarray], # [N, H_crop, W_crop]
        cx: float, cy: float,
        w:  float, h:  float,
        angle: float,
        crop_offset:  Tuple[int, int],
        frame_shape:  Tuple[int, ...],
        min_area:     Optional[int] = None,    # per-frame total threshold (sum of valid regions)
        min_region_area: float = 0.0,          # per-region: ignore regions smaller than this
        max_region_area: float = 0.0,          # per-region: any region ≥ this → FAIL (0 = disabled)
        conf_threshold:  float = 0.0,          # per-frame post-filter on detection score
    ) -> Dict[str, Any]:
        """
        FAIL conditions (OR logic):
          - any region area ≥ max_region_area  (critical, when max_region_area > 0)
          - sum(valid regions) ≥ min_area      (total threshold)

        Per-region filters applied first:
          - score < conf_threshold       → drop
          - area  < min_region_area      → drop

        Returns wrinkled_check dict:
          ok            : bool
          has_wrinkled  : bool
          triggered_by  : 'critical' | 'total' | None
          wrinkled_count: int
          total_area    : int
          min_area      : int   (total threshold used)
          min_region_area, max_region_area, conf_threshold : echo back for debug
          wrinkled_boxes: list[{score, area, area_pct, contour, corners}]
        """
        effective_min_area = min_area if min_area is not None else self.min_area
        crop_area = int(w * h)

        empty_payload = {
            'ok': True, 'has_wrinkled': False, 'triggered_by': None,
            'wrinkled_count': 0, 'total_area': 0,
            'min_area': effective_min_area,
            'min_region_area': min_region_area,
            'max_region_area': max_region_area,
            'conf_threshold': conf_threshold,
            'wrinkled_boxes': []
        }

        if len(seg_boxes) == 0 or seg_masks is None:
            return empty_payload

        # Back-project tất cả masks về frame space
        frame_masks = self.back_project_masks(
            seg_masks, cx, cy, angle, crop_offset, frame_shape
        )

        wrinkled_boxes = []
        for i, (box, mask_frame) in enumerate(zip(seg_boxes, frame_masks)):
            score = float(box[4])
            # Per-frame conf filter (post-hoc — predict_batch may have used a lower batch-min conf)
            if score < conf_threshold:
                continue

            # Tính area trong crop space (consistent với min_area threshold)
            mask_crop = seg_masks[i]
            area = int(np.sum(mask_crop > 0.5))

            # Per-region min filter — drop noisy small regions
            if min_region_area > 0 and area < min_region_area:
                continue

            area_pct = area / max(crop_area, 1) * 100

            # Contour trong frame space (JSON serializable)
            contour = self._mask_to_contour(mask_frame)
            if contour is None:
                continue

            # Axis-aligned bbox corners cho backward compat với draw_detected_obb_boxes
            corners = self._contour_to_bbox_corners(contour)

            wrinkled_boxes.append({
                'score'   : score,
                'class'   : 'wrinkled',
                'area'    : area,
                'area_pct': round(area_pct, 2),
                'contour' : contour,   # list of [x,y] in frame space
                'corners' : corners,   # 4 corners of bounding rect in frame space
            })

        total_area = sum(b['area'] for b in wrinkled_boxes)

        # Critical: bất kỳ vùng đơn nào ≥ max_region_area → FAIL ngay (kể cả total chưa đạt)
        has_critical = max_region_area > 0 and any(b['area'] >= max_region_area for b in wrinkled_boxes)
        # Total: tổng các vùng hợp lệ ≥ ngưỡng → FAIL
        total_exceeded = total_area >= effective_min_area

        has_wrinkled = has_critical or total_exceeded
        triggered_by = 'critical' if has_critical else ('total' if total_exceeded else None)

        if has_wrinkled:
            logger.info(
                f"Wrinkled detected ({triggered_by}): {len(wrinkled_boxes)} region(s), "
                f"total_area={total_area}px areas={[b['area'] for b in wrinkled_boxes]}px "
                f"thresholds: total={effective_min_area} max={max_region_area} min={min_region_area} conf={conf_threshold:.2f}"
            )

        return {
            'ok'            : not has_wrinkled,
            'has_wrinkled'  : has_wrinkled,
            'triggered_by'  : triggered_by,
            'wrinkled_count': len(wrinkled_boxes),
            'total_area'    : total_area,
            'min_area'      : effective_min_area,
            'min_region_area': min_region_area,
            'max_region_area': max_region_area,
            'conf_threshold': conf_threshold,
            'wrinkled_boxes': wrinkled_boxes,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _mask_to_contour(mask_frame: np.ndarray) -> Optional[List[List[int]]]:
        """Largest external contour của mask (float32) trong frame space."""
        binary = (mask_frame > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        return largest.reshape(-1, 2).tolist()

    @staticmethod
    def _contour_to_bbox_corners(contour: List[List[int]]) -> List[List[int]]:
        """Trả về 4 góc axis-aligned bounding rect của contour (TL,TR,BR,BL)."""
        pts = np.array(contour, dtype=np.int32)
        x, y, bw, bh = cv2.boundingRect(pts)
        return [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]
