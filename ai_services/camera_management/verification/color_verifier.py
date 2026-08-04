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
4. PASS if pixel_threshold <= matching_pixels <= pixel_max (pixel_max=0 disables
   the upper bound). The upper bound catches a MISSING label on products whose
   own colour matches the label's — a bare bottle of turmeric matches as much
   yellow as a yellow label would, so a lower bound alone cannot see it.

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

        localization_method = str(
            data.get("color_localization_method")
            or template.get("color_localization_method")
            or color_config.get("localization_method")
            or "image_proc"
        ).strip().lower()

        threshold = int(color_config.get("pixel_threshold", 1000))
        # Upper bound (0 = off): a bare product sharing the label's colour still
        # matches lots of pixels, so "too much match" means the label is missing.
        pixel_max = int(color_config.get("pixel_max", 0) or 0)

        def _ok(matching: int) -> bool:
            if matching < threshold:
                return False
            return pixel_max <= 0 or matching <= pixel_max

        def _why(matching: int) -> str:
            """Short tag for the log so an operator can tell the two failures
            apart without doing the arithmetic themselves."""
            if matching < threshold:
                return " [below min → wrong/absent colour]"
            if 0 < pixel_max < matching:
                return " [above max → label missing, bare product]"
            return ""

        if localization_method in ("superpoint", "edge_regions"):
            product_box = self._extract_transformed_product_box(
                data.get("transformed_bboxes") or []
            )
            if product_box is None:
                return {
                    "match": False,
                    "skipped": False,
                    "reason": "No transformed product region from SuperPoint",
                    "color_check": {
                        "matching_pixels": 0,
                        "bottle_pixels": 0,
                        "pixel_threshold": threshold,
                        "pixel_max": pixel_max,
                        "detected": False,
                        "localization_method": localization_method,
                    },
                }

            # ── Cap-axis regression (edge_regions) ───────────────────────────
            # SuperPoint anchors on the cap, which fixes rotation + vertical but
            # leaves the horizontal centre and apparent diameter jittery on a
            # round bottle. Detect both cap edges and slide/scale the product
            # polygon along the cap axis to match. Any failure degrades to the
            # plain SuperPoint polygon — never worse than 'superpoint' mode.
            axis_info: Optional[Dict[str, Any]] = None
            if localization_method == "edge_regions":
                product_box, axis_info = self._apply_cap_axis_correction(
                    frame_img, data, product_box, color_config, camera,
                    template_idx, serial,
                )

            matching, bottle_pixels = self._count_hsv_match(
                frame_img, product_box, color_config
            )
            ok = _ok(matching)

            _axis_txt = ""
            if localization_method == "edge_regions":
                _axis_txt = (
                    f" [cap shift={axis_info['shift_px']:+.1f}px scale={axis_info['scale']:.3f}]"
                    if axis_info else " [cap regression UNAVAILABLE → raw SuperPoint ROI]"
                )
            logger.info(
                f"[{serial}] color({localization_method}): matching={matching} / min={threshold}"
                f"{f' max={pixel_max}' if pixel_max > 0 else ''} "
                f"(roi_px={bottle_pixels}, ratio={matching/max(1,bottle_pixels)*100:.1f}%) "
                f"→ {'PASS' if ok else 'FAIL'}{_why(matching)}{_axis_txt}"
            )

            color_check: Dict[str, Any] = {
                "matching_pixels": int(matching),
                "bottle_pixels": int(bottle_pixels),
                "pixel_threshold": threshold,
                "pixel_max": pixel_max,
                "detected": True,
                "localization_method": localization_method,
                "h_range": [int(color_config.get("h_min", 0)), int(color_config.get("h_max", 180))],
                "s_range": [int(color_config.get("s_min", 0)), int(color_config.get("s_max", 255))],
                "v_range": [int(color_config.get("v_min", 0)), int(color_config.get("v_max", 255))],
            }
            if localization_method == "edge_regions":
                color_check["cap_axis"] = axis_info      # None when it fell back
            return {
                "match": bool(ok),
                "skipped": False,
                "color_check": color_check,
                "detected_boxes": {
                    "product": product_box,
                },
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

        bottle_box = self._detect_bottle(
            crop_img, product_polygon, crop_offset,
            sharp_threshold=float(color_config.get("bottle_sharp_threshold", 0.30) or 0.30),
            min_height_ratio=float(color_config.get("bottle_min_height_ratio", 0.20) or 0.20),
            min_aspect=float(color_config.get("bottle_min_aspect", 1.2) or 1.2),
        )
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
                    "pixel_max": pixel_max,
                    "detected": False,
                    "localization_method": "image_proc",
                },
            }

        matching, bottle_pixels = self._count_hsv_match(
            frame_img, bottle_box, color_config
        )
        ok = _ok(matching)

        logger.info(
            f"[{serial}] color: matching={matching} / min={threshold}"
            f"{f' max={pixel_max}' if pixel_max > 0 else ''} "
            f"(bottle_px={bottle_pixels}, ratio={matching/max(1,bottle_pixels)*100:.1f}%) "
            f"→ {'PASS' if ok else 'FAIL'}{_why(matching)}"
        )

        return {
            "match": bool(ok),
            "skipped": False,
            "color_check": {
                "matching_pixels": int(matching),
                "bottle_pixels": int(bottle_pixels),
                "pixel_threshold": threshold,
                "pixel_max": pixel_max,
                "detected": True,
                "localization_method": "image_proc",
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

    # ──────────────────────────────────────────────────────────────────────
    # Cap-axis regression (localization_method='edge_regions')
    # ──────────────────────────────────────────────────────────────────────
    def _load_cap_edge_ref(
        self, serial_number: str, template_idx: int
    ) -> Optional[Dict[str, float]]:
        """Template cap-edge reference written by MatcherFactory at build time.
        Cached per (serial, template_idx) — the miss is cached too, so a template
        without the file doesn't hit the disk on every frame."""
        if not hasattr(self, "_cap_ref_cache"):
            self._cap_ref_cache: Dict[Tuple[str, int], Optional[Dict[str, float]]] = {}
        key = (serial_number, template_idx)
        if key in self._cap_ref_cache:
            return self._cap_ref_cache[key]
        ref: Optional[Dict[str, float]] = None
        try:
            import json
            from pathlib import Path
            home = os.environ.get("HOME", "")
            suffix = f"_t{template_idx}" if template_idx > 0 else ""
            p = (Path(home) / "Source" / "ocr_datecode" / "crop_samples"
                 / serial_number / f"template{suffix}_cap_edges.json")
            if p.exists():
                ref = json.loads(p.read_text())
        except Exception as e:
            logger.error(f"[{serial_number}] Failed to load cap-edge reference: {e}")
        self._cap_ref_cache[key] = ref
        return ref

    def _apply_cap_axis_correction(
        self,
        frame_img: np.ndarray,
        data: Dict[str, Any],
        product_box: Dict[str, Any],
        color_config: Dict[str, Any],
        camera: Optional["Camera"],
        template_idx: int,
        serial: str,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, float]]]:
        """Slide + scale the SuperPoint product polygon onto the detected cap axis.

        Returns (product_box, axis_info). On ANY failure the ORIGINAL box comes
        back with axis_info=None: a colour check on a slightly-off ROI is far
        better than failing the frame, and it keeps this mode a strict
        improvement over plain 'superpoint'.
        """
        try:
            from .image_proc_detector import (
                EdgeParams, detect_cap_axis_from_regions, correct_polygon_along_axis,
            )
        except Exception as e:
            logger.warning(f"[{serial}] cap-axis import failed: {e}")
            return product_box, None

        bboxes = data.get("transformed_bboxes") or []
        q_left = self._edge_region_pts(bboxes, "left")
        q_right = self._edge_region_pts(bboxes, "right")
        if q_left is None or q_right is None:
            logger.warning(
                f"[{serial}] localization_method='edge_regions' but the template has "
                f"no edge_left/edge_right annotation — using the raw SuperPoint ROI"
            )
            return product_box, None

        ref = self._load_cap_edge_ref(serial, template_idx)
        if not ref:
            logger.warning(
                f"[{serial}] no cap-edge reference for template {template_idx} "
                f"(crop_samples/.../template_cap_edges.json) — using the raw SuperPoint ROI"
            )
            return product_box, None

        params = EdgeParams.from_config(color_config.get("edge_config") or {})
        axis = detect_cap_axis_from_regions(
            frame_img, q_left, q_right, ref_frac=ref,
            params=params, serial_number=serial,
        )
        if axis is None or "exp_left_mid" not in axis:
            return product_box, None

        corrected = correct_polygon_along_axis(
            product_box.get("corners") or [],
            axis["left_mid"], axis["right_mid"],
            axis["exp_left_mid"], axis["exp_right_mid"],
            max_shift_px=float(color_config.get("cap_max_shift_px", 0) or 0),
        )
        if corrected is None:
            logger.warning(
                f"[{serial}] cap-axis correction rejected (out of bounds) "
                f"— using the raw SuperPoint ROI"
            )
            return product_box, None

        pts, info = corrected
        new_box = dict(product_box)
        new_box["corners"] = pts.tolist()
        x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
        new_box["box"] = [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1, 0.0]
        new_box["source"] = "superpoint_product_cap_axis"
        info = dict(info)
        info.update({"col_L": axis["col_L"], "col_R": axis["col_R"]})
        return new_box, info

    @staticmethod
    def _edge_region_pts(
        transformed_bboxes: List[Dict[str, Any]], side: str
    ) -> Optional[List[List[float]]]:
        """SuperPoint-transformed 'edge_left'/'edge_right' quad, frame coords."""
        want = f"edge_{side}"
        for b in transformed_bboxes:
            if b.get("type") == want and b.get("points") and len(b["points"]) >= 4:
                return b["points"]
        return None

    @staticmethod
    def _extract_transformed_product_box(
        transformed_bboxes: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        for bbox in transformed_bboxes:
            if bbox.get("type") != "product":
                continue
            corners = bbox.get("points") or []
            if len(corners) < 3:
                continue
            arr = np.array(corners, dtype=np.float32)
            x1 = float(arr[:, 0].min())
            y1 = float(arr[:, 1].min())
            x2 = float(arr[:, 0].max())
            y2 = float(arr[:, 1].max())
            return {
                "box": [
                    float((x1 + x2) / 2.0),
                    float((y1 + y2) / 2.0),
                    float(x2 - x1),
                    float(y2 - y1),
                    0.0,
                ],
                "score": float(bbox.get("confidence", 1.0) or 1.0),
                "class": "product",
                "corners": arr.tolist(),
                "source": "superpoint_transformed_product",
            }
        return None

    @staticmethod
    def _detect_bottle(
        crop_img: np.ndarray,
        product_polygon: np.ndarray,
        crop_offset: Tuple[int, int],
        sharp_threshold: float = _SHARP_NORM_THRESHOLD,
        min_height_ratio: float = _MIN_HEIGHT_RATIO,
        min_aspect: float = _MIN_ASPECT_H_OVER_W,
    ) -> Optional[Dict[str, Any]]:
        """
        Sharpness-based bottle detection. Returns a YOLO-OBB-compatible dict in
        FRAME coordinates (crop_offset applied) so the existing visualizer can
        draw it directly.

        Tunable via color_config:
        - sharp_threshold:    sharpness mask threshold (fraction of max)
        - min_height_ratio:   min bottle height / crop height
        - min_aspect:         min h/w aspect (bottle taller than wide)
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
        mask = (s_norm > sharp_threshold).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones(_CLOSE_KERNEL, np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(_OPEN_KERNEL, np.uint8))

        nlabels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
        if nlabels <= 1:
            return None

        # Expected width/aspect from product polygon (template coords). The
        # bottle WIDTH is the most reliable shape constraint — bottle position
        # in frame varies a lot but width is consistent.
        poly_x = product_polygon[:, 0]
        poly_y = product_polygon[:, 1]
        expected_w = float(poly_x.max() - poly_x.min())
        expected_h = float(poly_y.max() - poly_y.min())
        expected_aspect = expected_h / max(1.0, expected_w)

        # ── Shape-match reference: vertical width profile of the product polygon ──
        # For a rectangle annotation this is a flat (uniform) profile; for a free
        # polygon it varies per row. Each candidate's mask profile is compared via
        # Pearson correlation — discriminates bottle silhouettes from non-bottle
        # blobs of similar bbox dimensions.
        template_profile_ref: Optional[np.ndarray] = None
        try:
            poly_h_int = max(2, int(round(expected_h)))
            poly_w_int = max(2, int(round(expected_w)))
            tpl_canvas = np.zeros((poly_h_int, poly_w_int), dtype=np.uint8)
            shifted = (product_polygon - np.array(
                [poly_x.min(), poly_y.min()], dtype=np.float32
            )).astype(np.int32)
            cv2.fillPoly(tpl_canvas, [shifted], 255)
            prof = (tpl_canvas > 0).sum(axis=1).astype(np.float32)
            pmax = float(prof.max())
            if pmax > 1.0:
                template_profile_ref = prof / pmax
        except Exception:
            template_profile_ref = None

        def _shape_match_score(cand_x: int, cand_y: int, cand_w: int, cand_h: int) -> float:
            """Pearson correlation between candidate mask profile and template
            profile. Returns value in [0, 1] (negative correlations clipped to 0).
            Returns 1.0 (no penalty) if profile can't be computed.
            """
            if template_profile_ref is None or cand_h < 8 or cand_w < 4:
                return 1.0
            cand_mask = mask[cand_y:cand_y + cand_h, cand_x:cand_x + cand_w]
            cand_prof = (cand_mask > 0).sum(axis=1).astype(np.float32)
            cmax = float(cand_prof.max())
            if cmax < 1.0:
                return 0.0
            cand_prof = cand_prof / cmax
            tlen = len(template_profile_ref)
            if len(cand_prof) != tlen:
                cand_prof = np.interp(
                    np.linspace(0, len(cand_prof) - 1, tlen),
                    np.arange(len(cand_prof)),
                    cand_prof,
                )
            tp = template_profile_ref
            t_std = float(tp.std())
            c_std = float(cand_prof.std())
            if t_std < 1e-6 or c_std < 1e-6:
                # Template/candidate is flat — fall back to RMSE-based similarity.
                rmse = float(np.sqrt(((tp - cand_prof) ** 2).mean()))
                return float(max(0.0, 1.0 - rmse))
            tp_c = tp - tp.mean()
            cp_c = cand_prof - cand_prof.mean()
            corr = float((tp_c * cp_c).mean() / (t_std * c_std))
            return float(max(0.0, corr))

        best = None
        best_score = 0.0
        best_shape = 0.0
        for i in range(1, nlabels):
            x, y, bw, bh, area = stats[i]
            if bh < min_height_ratio * h:
                continue
            if bh < min_aspect * bw:
                continue
            aspect = bh / max(1.0, float(bw))
            # Width match — penalize candidates whose width differs from the
            # product polygon's width (relative). Squared to be more selective.
            width_diff_rel = abs(float(bw) - expected_w) / max(1.0, expected_w)
            width_match = 1.0 / (1.0 + width_diff_rel * width_diff_rel * 4.0)
            aspect_match = 1.0 / (1.0 + abs(aspect - expected_aspect))
            # Shape match — vertical width profile vs template polygon's profile.
            # Map [0, 1] correlation → [0.5, 1.0] factor so candidates with bad
            # profile are penalized but not fully excluded (allow for noise).
            shape_corr = _shape_match_score(int(x), int(y), int(bw), int(bh))
            shape_factor = 0.5 + 0.5 * shape_corr
            # No centrality factor — bottle position can vary across frames.
            score = float(area) * width_match * aspect_match * shape_factor
            if score > best_score:
                best_score = score
                best_shape = shape_corr
                best = (int(x), int(y), int(bw), int(bh), int(area))

        # Fallback: best candidate's width OR height is abnormally off from the
        # product polygon → trust the user-drawn crop_area instead. Ranges below
        # define the "acceptable" band (relative to expected); outside →
        # fallback to crop_area bbox. Also fires when no candidate survived.
        # Height check catches sharpness mask only lighting up part of the
        # bottle (e.g. detected height ≈ 30% of the real bottle).
        ox, oy = crop_offset
        WIDTH_MIN_RATIO = 0.7    # bottle ≥ 70% of expected width
        WIDTH_MAX_RATIO = 2.0    # bottle ≤ 200% of expected width
        HEIGHT_MIN_RATIO = 0.9   # bottle ≥ 70% of expected height
        HEIGHT_MAX_RATIO = 1.5   # bottle ≤ 200% of expected height

        use_fallback = False
        fallback_reason = ""
        if best is None:
            use_fallback = True
            fallback_reason = "no candidate"
        else:
            bw_best = float(best[2])
            bh_best = float(best[3])
            w_ratio = bw_best / max(1.0, expected_w)
            h_ratio = bh_best / max(1.0, expected_h)
            if w_ratio < WIDTH_MIN_RATIO or w_ratio > WIDTH_MAX_RATIO:
                use_fallback = True
                fallback_reason = (
                    f"width {bw_best:.0f}px is {w_ratio:.2f}× expected ({expected_w:.0f}px) "
                    f"— outside [{WIDTH_MIN_RATIO}, {WIDTH_MAX_RATIO}]"
                )
            elif h_ratio < HEIGHT_MIN_RATIO or h_ratio > HEIGHT_MAX_RATIO:
                use_fallback = True
                fallback_reason = (
                    f"height {bh_best:.0f}px is {h_ratio:.2f}× expected ({expected_h:.0f}px) "
                    f"— outside [{HEIGHT_MIN_RATIO}, {HEIGHT_MAX_RATIO}]"
                )

        if use_fallback:
            # Use the full crop_area as the bottle bbox.
            fx = float(ox)
            fy = float(oy)
            bw = int(w)   # crop_img width
            bh = int(h)   # crop_img height
            area = bw * bh
            logger.info(
                f"_detect_bottle fallback to crop_area: {fallback_reason}; "
                f"bbox=({int(fx)},{int(fy)},{bw}×{bh})"
            )
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
            return {
                "box": [float(cx), float(cy), float(bw), float(bh), 0.0],
                "score": 0.3,   # lower score to flag this is a fallback
                "class": "product",
                "corners": corners.tolist(),
                "source": "image_proc_color_fallback_crop_area",
            }

        x, y, bw, bh, area = best
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

        logger.debug(
            f"_detect_bottle picked: bbox=({int(fx)},{int(fy)},{bw}×{bh}) "
            f"shape_match={best_shape:.2f} fill={fill_ratio:.2f}"
        )

        return {
            "box": [float(cx), float(cy), float(bw), float(bh), 0.0],
            "score": float(min(1.0, fill_ratio)),
            "class": "product",
            "corners": corners.tolist(),
            "source": "image_proc_color",
            "shape_match": float(best_shape),
        }

    @staticmethod
    def _count_hsv_match(
        frame_img: np.ndarray,
        bottle_box: Dict[str, Any],
        color_config: Dict[str, Any],
    ) -> Tuple[int, int]:
        """Count pixels inside bottle polygon matching the HSV range. Returns (matching, total)."""
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
        local_corners = corners.copy()
        local_corners[:, 0] -= x1
        local_corners[:, 1] -= y1
        poly_mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [local_corners], 255)

        h_lo = int(color_config.get("h_min", 0))
        h_hi = int(color_config.get("h_max", 180))
        s_lo = int(color_config.get("s_min", 0))
        s_hi = int(color_config.get("s_max", 255))
        v_lo = int(color_config.get("v_min", 0))
        v_hi = int(color_config.get("v_max", 255))

        mask = cv2.inRange(hsv, (h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi))
        masked = cv2.bitwise_and(mask, poly_mask)
        return int(np.count_nonzero(masked)), int(np.count_nonzero(poly_mask))
