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

            # Store crop_area metadata
            if crop_area:
                matcher.crop_area = crop_area.to_dict()
            else:
                matcher.crop_area = None

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
