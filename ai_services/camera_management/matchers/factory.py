"""
Matcher Factory

Factory for creating SuperPointMatcherTRT instances.
Encapsulates all the logic for template preparation and matcher initialization.
"""

import json
import shutil
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from .config import MatcherConfig, CropArea
from .annotation_parser import AnnotationParser

if TYPE_CHECKING:
    from ..camera import Camera

logger = logging.getLogger(__name__)


class ColorCheckStubMatcher:
    """
    Stub matcher for Check_Color cameras with a 'product' annotation.

    Color check uses image-proc bottle detection — no SuperPoint inference and
    no template-image alignment. We still need a *matcher object* so the camera
    survives the `serial in camera_matchers` gate in inference_handler. Only
    `crop_area` is consulted downstream; the other attributes exist for
    interface compatibility with the visualizer / `_batch_verify_templates`
    (which skips color cameras anyway).
    """

    def __init__(self, crop_area: Optional[Dict[str, Any]] = None):
        self.crop_area = crop_area
        self.template_img = None
        self.template_bbox = None
        self.other_bboxes = []
        self.is_color_check_stub = True

    def match_batch(self, *args, **kwargs):
        """Return an empty success result. Should not be called in practice — the
        pipeline filters color cameras out of the match batch before this gets
        invoked; this fallback exists only so a mis-routed call doesn't crash."""
        return {
            'success': True,
            'results': [],
            'batch_timings': {'total': 0.0, 'trt_inference': 0.0},
        }


class MatcherFactory:
    """
    Factory for creating template matchers.

    Responsibilities:
    - Parse template data and annotations
    - Prepare template images (copy, crop)
    - Create annotation JSON files
    - Initialize SuperPointMatcherTRT instances
    """

    def __init__(self, engine_path: str, temp_dir: Path, backend_dir: Path):
        """
        Initialize MatcherFactory.

        Args:
            engine_path: Path to TensorRT engine file
            temp_dir: Temporary directory for matcher files
            backend_dir: Backend directory for template sources
        """
        self.engine_path = engine_path
        self.temp_dir = temp_dir
        self.backend_dir = backend_dir

        # Ensure temp_dir exists
        self.temp_dir.mkdir(exist_ok=True)

        # Import matcher class
        self._matcher_class = None
        self._import_matcher_class()

    def _import_matcher_class(self):
        """Import SuperPointMatcherTRT class"""
        try:
            import sys
            ai_services_path = Path(__file__).parent.parent.parent
            if str(ai_services_path) not in sys.path:
                sys.path.insert(0, str(ai_services_path))

            try:
                from inference_engine_shared import SuperPointMatcherTRTOptimized as SuperPointMatcherTRT
                logger.info("MatcherFactory using optimized shared TensorRT engine")
            except ImportError:
                from inference_service import SuperPointMatcherTRT
                logger.info("MatcherFactory using legacy TensorRT engine")

            self._matcher_class = SuperPointMatcherTRT

        except Exception as e:
            logger.error(f"Failed to import matcher class: {e}")
            self._matcher_class = None

    @property
    def is_available(self) -> bool:
        """Check if matcher class is available"""
        return self._matcher_class is not None

    def create_matcher(
        self,
        camera: 'Camera',
        template_data: Dict[str, Any],
        template_idx: int,
        verbose: bool = False
    ) -> Optional[Any]:
        """
        Create a single matcher for one template.

        Args:
            camera: Camera instance
            template_data: Template data dict (from camera.templates[idx])
            template_idx: Template index
            verbose: Verbose logging

        Returns:
            SuperPointMatcherTRT instance or None if failed
        """
        if not self.is_available:
            logger.error("Matcher class not available")
            return None

        serial_number = camera.serial_number

        try:
            # Validate template data
            if not template_data:
                logger.error(f"[{serial_number}] Template {template_idx}: No template data")
                return None

            # ── Fast path: Check_Color + product annotation ──────────────────
            # These cameras don't need SuperPoint matching — image-proc handles
            # bottle localization in color_verifier. Skip the heavy template-bbox
            # requirement and TRT engine load; return a stub matcher carrying
            # only crop_area (used for optional frame pre-crop).
            annotations_raw = template_data.get("annotations") or []
            is_color_camera = (
                getattr(camera, 'function_type', '') == 'Check_Color'
                and any(a.get('type') == 'product' for a in annotations_raw)
            )
            if is_color_camera:
                # Best-effort parse of crop_area (optional, may be missing).
                # If template image dims unknown we skip crop_area; color_verifier
                # falls back to using the full frame.
                crop_area_dict: Optional[Dict[str, Any]] = None
                try:
                    img_w_t = int(template_data.get('image_width') or 0)
                    img_h_t = int(template_data.get('image_height') or 0)
                    if img_w_t > 0 and img_h_t > 0:
                        crop_area_obj = AnnotationParser._parse_crop_area(
                            annotations=annotations_raw,
                            img_width=img_w_t,
                            img_height=img_h_t,
                            serial_number=serial_number,
                            template_idx=template_idx,
                        )
                        if crop_area_obj and crop_area_obj.is_valid():
                            crop_area_dict = crop_area_obj.to_dict()
                except Exception as e:
                    logger.warning(
                        f"[{serial_number}] Template {template_idx}: "
                        f"crop_area parse failed (continuing without crop): {e}"
                    )
                stub = ColorCheckStubMatcher(crop_area=crop_area_dict)
                logger.info(
                    f"[{serial_number}] Matcher {template_idx}: ColorCheck stub "
                    f"(no SuperPoint), crop_area={'set' if crop_area_dict else 'none'}"
                )
                return stub

            image_url = template_data.get("image_url")
            if not image_url:
                logger.error(f"[{serial_number}] Template {template_idx}: No image_url")
                return None

            # Copy template to temp directory
            template_path = self._copy_template(
                image_url, serial_number, template_idx
            )
            if template_path is None:
                return None

            # Load template image to get dimensions
            template_img = cv2.imread(str(template_path))
            if template_img is None:
                logger.error(f"[{serial_number}] Template {template_idx}: Failed to load image")
                return None

            img_h, img_w = template_img.shape[:2]

            # Parse annotations
            annotations = template_data.get("annotations", [])
            template_bbox, other_bboxes, crop_area = AnnotationParser.parse_annotations(
                annotations=annotations,
                img_width=img_w,
                img_height=img_h,
                serial_number=serial_number,
                template_idx=template_idx
            )

            if not template_bbox:
                logger.error(f"[{serial_number}] Template {template_idx}: No template bbox found")
                return None

            # ── Cap-crop fast path (cap_crop_method='yolo_segment'/'yolo_obb') ──
            # Detect cap in template, crop tight square + circular mask, replace
            # the user's crop_area with the cap bbox. Adjust template_bbox +
            # other_bboxes to the new (cropped) coordinate space.
            #
            # We still preserve `user_crop_area_obj` so the pipeline can pass
            # it down to rotation / HoughCircles to bound the search to the
            # product region (otherwise OBB sees the full frame and picks up
            # neighbouring bottles).
            user_crop_area_obj = crop_area
            cap_crop_method = getattr(camera, 'cap_crop_method', 'none') or 'none'
            cap_crop_bbox: Optional[Tuple[int, int, int, int]] = None
            if cap_crop_method != 'none':
                try:
                    if cap_crop_method == 'yolo_segment':
                        from ..preprocessing.cv_rotator import detect_cap_and_crop
                        result = detect_cap_and_crop(template_img, margin_ratio=0.10)
                    else:
                        # 'yolo_obb' — for now reuse HoughCircles too (the OBB
                        # cap-detect engine isn't wired here yet; falls back to
                        # HoughCircles which has been validated on user data).
                        from ..preprocessing.cv_rotator import detect_cap_and_crop
                        result = detect_cap_and_crop(template_img, margin_ratio=0.10)

                    if result is None:
                        logger.warning(
                            f"[{serial_number}] Template {template_idx}: cap_crop_method="
                            f"{cap_crop_method} but no cap detected — falling back to "
                            f"normal crop_area flow"
                        )
                    else:
                        cap_crop_img, (cx1, cy1, cx2, cy2) = result
                        cap_crop_bbox = (cx1, cy1, cx2, cy2)
                        # Save cap-cropped template as new template image
                        cap_crop_path = template_path.parent / f"cap_crop_{template_path.name}"
                        cv2.imwrite(str(cap_crop_path), cap_crop_img)
                        # Adjust template_bbox + other_bboxes to cap-crop coords
                        # (BoundingBox stores corners in `points: List[List[int]]`).
                        # Use the dataclass `.offset()` helper to subtract (cx1, cy1).
                        try:
                            template_bbox = template_bbox.offset(cx1, cy1)
                            other_bboxes = [ob.offset(cx1, cy1) for ob in (other_bboxes or [])]
                        except Exception as e:
                            logger.warning(
                                f"[{serial_number}] cap_crop bbox adjust failed: {e}"
                            )
                        # Use cap-cropped template path for SuperPoint
                        template_path = cap_crop_path
                        template_img = cap_crop_img
                        # IMPORTANT: skip _apply_crop because template_img is
                        # already cropped. Pipeline preprocess will detect cap
                        # per-frame and crop frames the same way.
                        crop_area = None
                        logger.info(
                            f"[{serial_number}] Template {template_idx}: cap_crop_method="
                            f"{cap_crop_method} → cropped ({cx2-cx1}×{cy2-cy1}) at ({cx1},{cy1})"
                        )
                except Exception as e:
                    logger.warning(
                        f"[{serial_number}] Template {template_idx}: cap_crop failed: {e}; "
                        f"continuing with normal crop_area"
                    )

            # Apply crop if needed
            final_template_path = template_path
            if crop_area and crop_area.is_valid():
                final_template_path, template_bbox, other_bboxes = self._apply_crop(
                    template_img=template_img,
                    template_path=template_path,
                    template_bbox=template_bbox,
                    other_bboxes=other_bboxes,
                    crop_area=crop_area,
                    serial_number=serial_number,
                    template_idx=template_idx
                )

            # Apply horizontal erosion to suppress variable text (date codes) before matching
            if getattr(camera, 'match_erosion_enabled', False):
                final_template_path = self._apply_erosion(
                    template_path=final_template_path,
                    kernel_w=getattr(camera, 'match_erosion_kernel_w', 80),
                    kernel_h=getattr(camera, 'match_erosion_kernel_h', 1),
                    iterations=getattr(camera, 'match_erosion_iterations', 1),
                    serial_number=serial_number,
                    template_idx=template_idx
                )

            # Create annotation JSON file
            ann_json_path = self._create_annotation_file(
                template_path=final_template_path,
                template_bbox=template_bbox,
                other_bboxes=other_bboxes,
                serial_number=serial_number,
                template_idx=template_idx
            )

            # Create matcher instance
            matcher = self._matcher_class(
                json_path=str(ann_json_path),
                engine_path=self.engine_path,
                scale=1.0,
                verbose=verbose
            )

            # Store crop_area metadata — use the ORIGINAL user-drawn crop_area
            # (`user_crop_area_obj`), not the local `crop_area` which gets
            # zeroed-out in the cap_crop_method fast-path above. The pipeline
            # uses this to bound rotation / HoughCircles to the product region.
            if user_crop_area_obj:
                matcher.crop_area = user_crop_area_obj.to_dict()
            else:
                matcher.crop_area = None
            # Flag this matcher as cap-cropped (pipeline preprocess will detect
            # cap in frame per-inference and crop the same way, instead of using
            # crop_area).
            matcher.cap_crop_method = cap_crop_method if cap_crop_bbox else 'none'
            if cap_crop_bbox:
                matcher.template_cap_bbox = cap_crop_bbox

            # Save template crop + product bbox for analysis
            self._save_template_sample(
                final_template_path, other_bboxes, serial_number, template_idx
            )

            logger.info(f"[{serial_number}] Matcher {template_idx} created successfully")
            return matcher

        except Exception as e:
            logger.error(f"[{serial_number}] Error creating matcher {template_idx}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_matchers_for_camera(
        self,
        camera: 'Camera',
        verbose_first: bool = True
    ) -> List[Any]:
        """
        Create all matchers for a camera.

        Args:
            camera: Camera instance with templates
            verbose_first: Only verbose for first matcher

        Returns:
            List of matcher instances (may be empty if all fail)
        """
        matchers = []
        serial_number = camera.serial_number

        if not camera.templates:
            logger.warning(f"[{serial_number}] No templates to create matchers for")
            return matchers

        for idx, template_data in enumerate(camera.templates):
            verbose = verbose_first and (idx == 0)

            matcher = self.create_matcher(
                camera=camera,
                template_data=template_data,
                template_idx=idx,
                verbose=verbose
            )

            if matcher:
                matchers.append(matcher)
                logger.info(f"[{serial_number}] Matcher {idx + 1}/{len(camera.templates)} initialized")
            else:
                logger.error(f"[{serial_number}] Failed to create matcher {idx + 1}/{len(camera.templates)}")

        return matchers

    def _copy_template(
        self,
        image_url: str,
        serial_number: str,
        template_idx: int
    ) -> Optional[Path]:
        """Copy template from backend to temp directory"""
        filename = image_url.split("/")[-1]
        source_path = self.backend_dir / "uploads" / "templates" / filename

        if not source_path.exists():
            logger.error(f"[{serial_number}] Template {template_idx}: Not found at {source_path}")
            return None

        dest_path = self.temp_dir / f"template_{serial_number}_t{template_idx}.jpg"
        shutil.copy(source_path, dest_path)

        return dest_path

    def _apply_crop(
        self,
        template_img,
        template_path: Path,
        template_bbox,
        other_bboxes: List,
        crop_area: CropArea,
        serial_number: str,
        template_idx: int
    ):
        """
        Apply crop to template and adjust bboxes.

        Returns:
            Tuple of (cropped_template_path, adjusted_template_bbox, adjusted_other_bboxes)
        """
        logger.info(f"[{serial_number}] Template {template_idx}: Applying crop_area")

        # Crop template image
        cropped_img = template_img[
            crop_area.y1:crop_area.y2,
            crop_area.x1:crop_area.x2
        ]

        # Save cropped template
        cropped_path = self.temp_dir / f"template_{serial_number}_t{template_idx}_cropped.jpg"
        cv2.imwrite(str(cropped_path), cropped_img)

        logger.info(
            f"[{serial_number}] Template {template_idx}: Cropped template saved: "
            f"{cropped_img.shape[1]}x{cropped_img.shape[0]}"
        )

        # Adjust bboxes
        adjusted_template, adjusted_others = AnnotationParser.adjust_bboxes_for_crop(
            template_bbox, other_bboxes, crop_area
        )

        logger.info(
            f"[{serial_number}] Template {template_idx}: "
            f"Adjusted {len(adjusted_others) + 1} bbox(es) to cropped coordinates"
        )

        return cropped_path, adjusted_template, adjusted_others

    def _apply_erosion(
        self,
        template_path: Path,
        kernel_w: int,
        kernel_h: int,
        iterations: int,
        serial_number: str,
        template_idx: int
    ) -> Path:
        """Apply horizontal erosion to suppress variable text before SuperPoint matching."""
        img = cv2.imread(str(template_path))
        if img is None:
            logger.warning(f"[{serial_number}] Template {template_idx}: Erosion skipped — cannot read image")
            return template_path

        kernel = np.ones((kernel_h, kernel_w), np.uint8)
        eroded = cv2.erode(img, kernel, iterations=iterations)

        eroded_path = template_path.parent / (template_path.stem + "_eroded" + template_path.suffix)
        cv2.imwrite(str(eroded_path), eroded)
        logger.info(
            f"[{serial_number}] Template {template_idx}: Erosion applied "
            f"(kernel={kernel_w}x{kernel_h}, iter={iterations}) → {eroded_path.name}"
        )
        return eroded_path

    def _create_annotation_file(
        self,
        template_path: Path,
        template_bbox,
        other_bboxes: List,
        serial_number: str,
        template_idx: int
    ) -> Path:
        """Create annotation JSON file for matcher"""
        ann_json_path = self.temp_dir / f"annotations_{serial_number}_t{template_idx}.json"

        # Build annotation data
        all_bboxes = [template_bbox.to_dict()]
        all_bboxes.extend([bbox.to_dict() for bbox in other_bboxes])

        ann_data = {
            "_template_image": str(template_path),
            str(template_path): all_bboxes
        }

        with open(ann_json_path, "w") as f:
            json.dump(ann_data, f, indent=2)

        return ann_json_path

    def _save_template_sample(
        self,
        template_path: Path,
        other_bboxes: List,
        serial_number: str,
        template_idx: int,
    ):
        """Save template crop + product bbox coordinates to crop_samples dir for analysis."""
        import os
        home = os.environ.get('HOME', '')
        sample_dir = Path(home) / 'Source' / 'ocr_datecode' / 'crop_samples' / serial_number
        sample_dir.mkdir(parents=True, exist_ok=True)

        suffix = f"_t{template_idx}" if template_idx > 0 else ""

        # Save template image crop (copy as-is — already cropped by _apply_crop)
        img = cv2.imread(str(template_path))
        if img is not None:
            out_img = sample_dir / f"template{suffix}.jpg"
            cv2.imwrite(str(out_img), img)

        def _bbox_to_dict(bbox_type: str, pts):
            return {
                'type': bbox_type,
                'points': [list(p) for p in pts],
                'x_min': int(min(p[0] for p in pts)),
                'y_min': int(min(p[1] for p in pts)),
                'x_max': int(max(p[0] for p in pts)),
                'y_max': int(max(p[1] for p in pts)),
                'width': int(max(p[0] for p in pts) - min(p[0] for p in pts)),
                'height': int(max(p[1] for p in pts) - min(p[1] for p in pts)),
            }

        # Save product bbox coordinates as JSON
        product_bboxes = [b for b in other_bboxes if b.type == 'product']
        if product_bboxes:
            bbox_data = _bbox_to_dict('product', product_bboxes[0].points)
            out_json = sample_dir / f"template{suffix}_product_bbox.json"
            with open(str(out_json), 'w') as f:
                json.dump(bbox_data, f, indent=2)
            logger.info(
                f"[{serial_number}] Template sample saved: {out_img.name}, "
                f"product bbox {bbox_data['width']}x{bbox_data['height']}px"
            )

        # Save label bbox coordinates as JSON (used for alignment reference)
        label_bboxes = [b for b in other_bboxes if b.type == 'label']
        if label_bboxes:
            bbox_data = _bbox_to_dict('label', label_bboxes[0].points)
            out_json = sample_dir / f"template{suffix}_label_bbox.json"
            with open(str(out_json), 'w') as f:
                json.dump(bbox_data, f, indent=2)
            logger.info(
                f"[{serial_number}] Template label bbox saved: "
                f"{bbox_data['width']}x{bbox_data['height']}px"
            )

            # Detect template walls (inner + outer) bằng image processing
            # → used when product_detection_method='yolo_segment'
            try:
                from ..verification.image_proc_detector import detect_template_walls
                tmpl_img = cv2.imread(str(template_path))
                if tmpl_img is not None:
                    walls = detect_template_walls(tmpl_img, label_bboxes[0].points)
                    if walls is not None:
                        walls_path = sample_dir / f"template{suffix}_walls.json"
                        with open(str(walls_path), 'w') as f:
                            json.dump(walls, f, indent=2)
                        logger.info(
                            f"[{serial_number}] Template walls saved: "
                            f"inner_L={walls['inner_L']} inner_R={walls['inner_R']} "
                            f"outer_L={walls['outer_L']} outer_R={walls['outer_R']}"
                        )
            except Exception as e:
                logger.warning(
                    f"[{serial_number}] Template walls detection failed (yolo_segment "
                    f"mode may not work): {e}"
                )
