"""
Annotation Parser

Parses template annotations and converts normalized coordinates to pixel coordinates.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

from .config import BoundingBox, CropArea

logger = logging.getLogger(__name__)


class AnnotationParser:
    """
    Parse and convert template annotations.

    Handles:
    - Normalized to pixel coordinate conversion
    - Crop area extraction
    - Template and text bbox parsing
    - Coordinate offset adjustment
    """

    @staticmethod
    def parse_annotations(
        annotations: List[Dict[str, Any]],
        img_width: int,
        img_height: int,
        serial_number: str = "",
        template_idx: int = 0
    ) -> Tuple[Optional[BoundingBox], List[BoundingBox], Optional[CropArea]]:
        """
        Parse all annotations from template data.

        Args:
            annotations: List of annotation dicts from template
            img_width: Image width in pixels
            img_height: Image height in pixels
            serial_number: Camera serial for logging
            template_idx: Template index for logging

        Returns:
            Tuple of (template_bbox, other_bboxes, crop_area)
        """
        template_bbox = None
        other_bboxes = []
        crop_area = None

        # First pass: find crop_area
        crop_area = AnnotationParser._parse_crop_area(
            annotations, img_width, img_height, serial_number, template_idx
        )

        # Second pass: parse other annotations
        for ann_idx, ann in enumerate(annotations):
            ann_type = ann.get("type", "")

            if ann_type == "crop_area":
                continue  # Already processed

            if ann_type == "template":
                template_bbox = AnnotationParser._parse_flexible_bbox(
                    ann, ann_idx, img_width, img_height, "template"
                )

            elif ann_type in ["product", "label"]:
                bbox = AnnotationParser._parse_flexible_bbox(
                    ann, ann_idx, img_width, img_height, ann_type
                )
                if bbox:
                    other_bboxes.append(bbox)

            elif ann_type in ["text", "barcode", "datecode"]:
                bbox = AnnotationParser._parse_text_bbox(
                    ann, ann_idx, img_width, img_height, ann_type
                )
                if bbox:
                    other_bboxes.append(bbox)

        return template_bbox, other_bboxes, crop_area

    @staticmethod
    def _parse_crop_area(
        annotations: List[Dict[str, Any]],
        img_width: int,
        img_height: int,
        serial_number: str,
        template_idx: int
    ) -> Optional[CropArea]:
        """Parse crop_area annotation if exists"""
        for ann in annotations:
            if ann.get("type") != "crop_area":
                continue

            x, y = ann.get("x", 0), ann.get("y", 0)
            w, h = ann.get("width", 1), ann.get("height", 1)

            # Convert to pixel coordinates
            x1, y1 = int(x * img_width), int(y * img_height)
            x2, y2 = int((x + w) * img_width), int((y + h) * img_height)

            # Clamp to image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)

            crop_area = CropArea(x1=x1, y1=y1, x2=x2, y2=y2)

            if not crop_area.is_valid(min_size=100):
                logger.warning(
                    f"[{serial_number}] Template {template_idx}: "
                    f"crop_area too small ({crop_area.width}x{crop_area.height}), ignoring"
                )
                return None

            logger.info(
                f"[{serial_number}] Template {template_idx}: Found crop_area: "
                f"({x1}, {y1}) -> ({x2}, {y2}), size: {crop_area.width}x{crop_area.height}"
            )
            return crop_area

        return None

    @staticmethod
    def _parse_rectangle_bbox(
        ann: Dict[str, Any],
        ann_idx: int,
        img_width: int,
        img_height: int,
        bbox_type: str
    ) -> Optional[BoundingBox]:
        """Parse a rectangle-type annotation"""
        x, y = ann.get("x", 0), ann.get("y", 0)
        w, h = ann.get("width", 0), ann.get("height", 0)
        conf = ann.get("conf", 0.8)

        x1, y1 = int(x * img_width), int(y * img_height)
        x2, y2 = int((x + w) * img_width), int((y + h) * img_height)

        points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        return BoundingBox(
            type=bbox_type,
            points=points,
            annotation_index=ann_idx,
            conf=conf
        )

    @staticmethod
    def _parse_flexible_bbox(
        ann: Dict[str, Any],
        ann_idx: int,
        img_width: int,
        img_height: int,
        bbox_type: str
    ) -> Optional[BoundingBox]:
        """
        Parse annotation that can have either points or rectangle format.
        Supports both polygon points and rectangle coordinates.
        """
        conf = ann.get("conf", 0.8)
        text = ann.get("text", "")
        pixel_points = []

        if ann.get("points"):
            # Polygon points provided
            pixel_points = [
                [int(pt[0] * img_width), int(pt[1] * img_height)]
                for pt in ann.get("points", [])
            ]
        elif ann.get("x") is not None:
            # Rectangle coordinates
            x, y = ann.get("x", 0), ann.get("y", 0)
            w, h = ann.get("width", 0), ann.get("height", 0)

            x1, y1 = int(x * img_width), int(y * img_height)
            x2, y2 = int((x + w) * img_width), int((y + h) * img_height)

            pixel_points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        if not pixel_points:
            return None

        bbox_kwargs = {
            "type": bbox_type,
            "points": pixel_points,
            "annotation_index": ann_idx,
            "conf": conf
        }

        # Only add text if it's not empty
        if text:
            bbox_kwargs["text"] = text

        return BoundingBox(**bbox_kwargs)

    @staticmethod
    def _parse_text_bbox(
        ann: Dict[str, Any],
        ann_idx: int,
        img_width: int,
        img_height: int,
        bbox_type: str
    ) -> Optional[BoundingBox]:
        """Parse a text/barcode/datecode annotation"""
        shape = ann.get("shape", "rectangle")
        conf = ann.get("conf", 0.8)
        text = ann.get("text", "")
        pixel_points = []

        if ann.get("points"):
            # Polygon points provided
            pixel_points = [
                [int(pt[0] * img_width), int(pt[1] * img_height)]
                for pt in ann.get("points", [])
            ]
        elif shape == "rectangle" and ann.get("x") is not None:
            # Rectangle coordinates
            x, y = ann.get("x", 0), ann.get("y", 0)
            w, h = ann.get("width", 0), ann.get("height", 0)

            x1, y1 = int(x * img_width), int(y * img_height)
            x2, y2 = int((x + w) * img_width), int((y + h) * img_height)

            pixel_points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        if not pixel_points:
            return None

        return BoundingBox(
            type=bbox_type,
            points=pixel_points,
            annotation_index=ann_idx,
            conf=conf,
            text=text
        )

    @staticmethod
    def adjust_bboxes_for_crop(
        template_bbox: BoundingBox,
        other_bboxes: List[BoundingBox],
        crop_area: CropArea
    ) -> Tuple[BoundingBox, List[BoundingBox]]:
        """
        Adjust all bboxes for crop area offset.

        Args:
            template_bbox: Template bounding box
            other_bboxes: List of other bounding boxes
            crop_area: Crop area with offset

        Returns:
            Tuple of (adjusted_template_bbox, adjusted_other_bboxes)
        """
        offset_x = crop_area.x1
        offset_y = crop_area.y1

        adjusted_template = template_bbox.offset(offset_x, offset_y)
        adjusted_others = [bbox.offset(offset_x, offset_y) for bbox in other_bboxes]

        return adjusted_template, adjusted_others
