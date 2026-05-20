"""
Color verification for Check_Color templates with a 'product' annotation.

Pipeline per frame (decoupled from SuperPoint match):
1. Read raw template annotations (template coords). Product polygon = shape hint,
   crop_area (if present) = crop window on the frame.
2. Image-proc detect bottle in the cropped frame using local sharpness — the
   bottle is the only in-focus object so sharpness lights it up regardless of
   label color.
3. Convert detected bottle region to HSV, count pixels matching the template's
   color_config HSV range.
4. PASS if matching_pixels >= color_config.pixel_threshold.

Output dict matches ProductVerificationService.verify_batch shape so the
existing visualizer / result_builder / FE all keep working.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from ..camera import Camera


logger = logging.getLogger(__name__)


# Sharpness-mask tuning (kept conservative — empirically works on the 4 sample
# images user provided in tests/test_bottle_detect_compare.py).
_LAPLACIAN_KSIZE = 3
_SHARP_BOX_WINDOW = (31, 31)
_SHARP_NORM_THRESHOLD = 0.30
_CLOSE_KERNEL = (25, 5)
_OPEN_KERNEL = (9, 9)
_MIN_HEIGHT_RATIO = 0.20      # bottle height >= 20% of crop height
_MIN_ASPECT_H_OVER_W = 1.2    # bottle taller than wide


class ColorVerificationService:
    """HSV color check for Check_Color templates with a 'product' annotation."""

    def __init__(
        self,
        save_debug_images: str = "never",
        debug_path: Optional[str] = None,
    ):
        self.save_debug_images = (save_debug_images or "never").lower()
        if self.save_debug_images not in ("never", "on_fail", "always"):
            self.save_debug_images = "never"
        home = os.environ.get("HOME", "")
        self.debug_path = debug_path or f"{home}/Source/ocr_datecode/ai_services/test_result"

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def verify_batch(self, frames_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run color verification on a batch of frames.

        Caller (product_verifier) is responsible for filtering so only
        Check_Color frames with a product annotation reach this method.

        Each frames_data entry should contain:
            - frame_img:          BGR numpy array
            - camera:             Camera (for template lookup + serial logging)
            - template_idx:       int  — which template within camera.templates
                                  to use (default 0)
            - color_config:       Optional[Dict]  — pre-extracted; if missing,
                                  we fall back to camera.templates[idx].color_config

        Returns: list of dicts, same shape as ProductVerificationService.verify_batch.
        """
        t_start = time.perf_counter()
        results: List[Dict[str, Any]] = []

        for data in frames_data:
            t0 = time.perf_counter()
            try:
                result = self._verify_one(data)
            except Exception as e:
                cam = data.get("camera")
                serial = getattr(cam, "serial_number", "unknown") if cam else "unknown"
                logger.error(f"[{serial}] color verify failed: {e}", exc_info=True)
                result = {
                    "match": False,
                    "skipped": False,
                    "error": str(e),
                    "reason": "Color verifier exception",
                }
            result.setdefault("timing", {})
            result["timing"]["frame_ms"] = (time.perf_counter() - t0) * 1000
            result["timing"]["method"] = "color_check"
            results.append(result)

        total_ms = (time.perf_counter() - t_start) * 1000
        for r in results:
            r["timing"]["total"] = total_ms

        logger.info(
            f"Color verification batch: {len(frames_data)} frames, total={total_ms:.1f}ms"
        )
        return results

    # ──────────────────────────────────────────────────────────────────────
    # Per-frame
    # ──────────────────────────────────────────────────────────────────────

    def _verify_one(self, data: Dict[str, Any]) -> Dict[str, Any]:
        frame_img = data.get("frame_img")
        camera: Optional["Camera"] = data.get("camera")
        template_idx = int(data.get("template_idx", 0) or 0)
        serial = getattr(camera, "serial_number", "unknown") if camera else "unknown"

        if frame_img is None or not hasattr(frame_img, "shape"):
            # Frame wasn't included in frames_data (typically because SuperPoint
            # match failed AND color_config wasn't set on the template). Skip
            # rather than fail — color check is an opt-in feature.
            return {
                "match": True,
                "skipped": True,
                "reason": "No frame image (color check not active for this frame)",
            }

        template = self._get_template(camera, template_idx)
        if template is None:
            return {
                "match": True,
                "skipped": True,
                "reason": "No template for color verification",
            }

        color_config: Optional[Dict[str, Any]] = (
            data.get("color_config") or template.get("color_config")
        )
        if not color_config:
            return {
                "match": True,
                "skipped": True,
                "reason": "color_config not set on template",
            }

        annotations = template.get("annotations") or []
        # Annotations are stored in normalized [0, 1] coords. Denormalize to
        # FRAME pixel coords (assumes template was captured from the same
        # camera, so frame ≈ template image dimensions).
        H, W = frame_img.shape[:2]
        product_polygon = self._extract_product_polygon(annotations, W, H)
        if product_polygon is None:
            return {
                "match": True,
                "skipped": True,
                "reason": "No 'product' annotation in template",
            }

        crop_area = self._extract_crop_area(annotations)
        crop_img, crop_offset = self._apply_crop_area(frame_img, crop_area, W, H)

        bottle_box = self._detect_bottle(crop_img, product_polygon, crop_offset)
        threshold = int(color_config.get("pixel_threshold", 1000))

        if bottle_box is None:
            logger.debug(f"[{serial}] color: failed to detect bottle in crop")
            return {
                "match": False,
                "skipped": False,
                "reason": "Failed to detect bottle in frame",
                "color_check": {
                    "matching_pixels": 0,
                    "bottle_pixels": 0,
                    "pixel_threshold": threshold,
                    "detected": False,
                },
            }

        matching, bottle_pixels = self._count_hsv_match(
            frame_img, bottle_box, color_config
        )
        ok = matching >= threshold

        logger.info(
            f"[{serial}] color: matching={matching} / threshold={threshold} "
            f"(bottle_px={bottle_pixels}, ratio={matching/max(1,bottle_pixels)*100:.1f}%) "
            f"→ {'PASS' if ok else 'FAIL'}"
        )

        return {
            "match": bool(ok),
            "skipped": False,
            "color_check": {
                "matching_pixels": int(matching),
                "bottle_pixels": int(bottle_pixels),
                "pixel_threshold": threshold,
                "detected": True,
                "h_range": [int(color_config.get("h_min", 0)), int(color_config.get("h_max", 180))],
                "s_range": [int(color_config.get("s_min", 0)), int(color_config.get("s_max", 255))],
                "v_range": [int(color_config.get("v_min", 0)), int(color_config.get("v_max", 255))],
            },
            "detected_boxes": {
                "product": bottle_box,
            },
        }

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_template(camera: Optional["Camera"], idx: int) -> Optional[Dict[str, Any]]:
        if camera is None:
            return None
        templates = getattr(camera, "templates", None)
        if not templates:
            return None
        if idx < 0 or idx >= len(templates):
            idx = 0
        return templates[idx]

    @staticmethod
    def _extract_product_polygon(
        annotations: List[Dict[str, Any]],
        frame_w: int,
        frame_h: int,
    ) -> Optional[np.ndarray]:
        """
        Return Nx2 float32 polygon for the first 'product' annotation in FRAME
        pixel coords. Annotation values are NORMALIZED [0, 1] (relative coords),
        so we scale by (frame_w, frame_h).
        """
        for ann in annotations:
            if ann.get("type") != "product":
                continue
            pts = ann.get("points")
            if pts and len(pts) >= 3:
                arr = []
                for p in pts:
                    if isinstance(p, dict):
                        px = float(p.get("x", 0)); py = float(p.get("y", 0))
                    elif isinstance(p, (list, tuple)) and len(p) >= 2:
                        px = float(p[0]); py = float(p[1])
                    else:
                        continue
                    arr.append([px * frame_w, py * frame_h])
                if len(arr) >= 3:
                    return np.array(arr, dtype=np.float32)
            # Rectangle fallback
            x = ann.get("x")
            y = ann.get("y")
            w = ann.get("width")
            h = ann.get("height")
            if x is not None and y is not None and w and h:
                x1 = float(x) * frame_w
                y1 = float(y) * frame_h
                x2 = (float(x) + float(w)) * frame_w
                y2 = (float(y) + float(h)) * frame_h
                return np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    dtype=np.float32,
                )
        return None

    @staticmethod
    def _extract_crop_area(annotations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for ann in annotations:
            if ann.get("type") == "crop_area":
                return ann
        return None

    @staticmethod
    def _apply_crop_area(
        frame_img: np.ndarray,
        crop_area: Optional[Dict[str, Any]],
        frame_w: int,
        frame_h: int,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Crop the frame to the crop_area annotation. Annotation coords are
        normalized [0, 1]; we scale by (frame_w, frame_h).
        Returns (crop, (offset_x, offset_y)).
        """
        if crop_area is None:
            return frame_img, (0, 0)
        x1 = y1 = 0
        x2 = frame_w
        y2 = frame_h
        pts = crop_area.get("points")
        if pts and len(pts) >= 3:
            xs = []
            ys = []
            for p in pts:
                if isinstance(p, dict):
                    xs.append(float(p.get("x", 0)) * frame_w)
                    ys.append(float(p.get("y", 0)) * frame_h)
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    xs.append(float(p[0]) * frame_w)
                    ys.append(float(p[1]) * frame_h)
            if xs and ys:
                x1 = max(0, int(min(xs)))
                y1 = max(0, int(min(ys)))
                x2 = min(frame_w, int(max(xs)))
                y2 = min(frame_h, int(max(ys)))
        else:
            x = crop_area.get("x")
            y = crop_area.get("y")
            w = crop_area.get("width")
            h = crop_area.get("height")
            if x is not None and y is not None and w and h:
                x1 = max(0, int(float(x) * frame_w))
                y1 = max(0, int(float(y) * frame_h))
                x2 = min(frame_w, int((float(x) + float(w)) * frame_w))
                y2 = min(frame_h, int((float(y) + float(h)) * frame_h))
        if x2 <= x1 or y2 <= y1:
            return frame_img, (0, 0)
        return frame_img[y1:y2, x1:x2], (x1, y1)

    @staticmethod
    def _detect_bottle(
        crop_img: np.ndarray,
        product_polygon: np.ndarray,
        crop_offset: Tuple[int, int],
    ) -> Optional[Dict[str, Any]]:
        """
        Sharpness-based bottle detection. Returns a YOLO-OBB-compatible dict in
        FRAME coordinates (crop_offset applied) so the existing visualizer can
        draw it directly.
        """
        if crop_img is None or crop_img.size == 0:
            return None
        h, w = crop_img.shape[:2]
        if h < 8 or w < 8:
            return None

        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=_LAPLACIAN_KSIZE)
        sharp = cv2.boxFilter(lap * lap, ddepth=cv2.CV_32F, ksize=_SHARP_BOX_WINDOW)
        sharp = np.sqrt(np.maximum(sharp, 0))
        smax = float(sharp.max())
        if smax <= 1e-6:
            return None
        s_norm = sharp / (smax + 1e-6)
        mask = (s_norm > _SHARP_NORM_THRESHOLD).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones(_CLOSE_KERNEL, np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(_OPEN_KERNEL, np.uint8))

        nlabels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
        if nlabels <= 1:
            return None

        # Expected aspect from product polygon (template coords).
        poly_x = product_polygon[:, 0]
        poly_y = product_polygon[:, 1]
        expected_w = float(poly_x.max() - poly_x.min())
        expected_h = float(poly_y.max() - poly_y.min())
        expected_aspect = expected_h / max(1.0, expected_w)

        best = None
        best_score = 0.0
        for i in range(1, nlabels):
            x, y, bw, bh, area = stats[i]
            if bh < _MIN_HEIGHT_RATIO * h:
                continue
            if bh < _MIN_ASPECT_H_OVER_W * bw:
                continue
            aspect = bh / max(1.0, float(bw))
            aspect_match = 1.0 / (1.0 + abs(aspect - expected_aspect))
            cx_dist = abs((x + bw / 2.0) - w / 2.0) / max(1.0, float(w))
            score = float(area) * aspect_match * (1.0 - 0.5 * cx_dist)
            if score > best_score:
                best_score = score
                best = (int(x), int(y), int(bw), int(bh), int(area))

        if best is None:
            return None

        x, y, bw, bh, area = best
        ox, oy = crop_offset
        fx = x + ox
        fy = y + oy
        cx = fx + bw / 2.0
        cy = fy + bh / 2.0

        corners = np.array(
            [
                [fx, fy],
                [fx + bw, fy],
                [fx + bw, fy + bh],
                [fx, fy + bh],
            ],
            dtype=np.float32,
        )

        # Mask fill ratio as a confidence-ish score.
        fill_ratio = area / float(max(1, bw * bh))

        return {
            "box": [float(cx), float(cy), float(bw), float(bh), 0.0],
            "score": float(min(1.0, fill_ratio)),
            "class": "product",
            "corners": corners.tolist(),
            "source": "image_proc_color",
        }

    @staticmethod
    def _count_hsv_match(
        frame_img: np.ndarray,
        bottle_box: Dict[str, Any],
        color_config: Dict[str, Any],
    ) -> Tuple[int, int]:
        """Count pixels inside bottle bbox matching the HSV range. Returns (matching, total)."""
        corners = np.array(bottle_box.get("corners", []), dtype=np.int32)
        if corners.shape[0] < 3:
            return 0, 0
        H, W = frame_img.shape[:2]
        x1 = max(0, int(corners[:, 0].min()))
        y1 = max(0, int(corners[:, 1].min()))
        x2 = min(W, int(corners[:, 0].max()))
        y2 = min(H, int(corners[:, 1].max()))
        if x2 <= x1 or y2 <= y1:
            return 0, 0

        roi = frame_img[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        h_lo = int(color_config.get("h_min", 0))
        h_hi = int(color_config.get("h_max", 180))
        s_lo = int(color_config.get("s_min", 0))
        s_hi = int(color_config.get("s_max", 255))
        v_lo = int(color_config.get("v_min", 0))
        v_hi = int(color_config.get("v_max", 255))

        mask = cv2.inRange(hsv, (h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi))
        return int(np.count_nonzero(mask)), int(roi.shape[0] * roi.shape[1])
