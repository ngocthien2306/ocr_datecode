"""
Matcher Configuration Dataclasses

Provides structured data types for matcher configuration.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class CropArea:
    """Represents a crop area in pixel coordinates"""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def to_dict(self) -> Dict[str, int]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height
        }

    def is_valid(self, min_size: int = 100) -> bool:
        """Check if crop area is valid (not too small)"""
        return self.width >= min_size and self.height >= min_size


@dataclass
class BoundingBox:
    """Represents a bounding box with 4 corner points"""
    type: str  # "template", "text", "barcode", "datecode", "product"
    points: List[List[int]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    annotation_index: int
    conf: float = 0.8
    text: str = ""  # For text type annotations

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "type": self.type,
            "points": self.points,
            "annotation_index": self.annotation_index,
            "conf": self.conf
        }
        if self.text:
            result["text"] = self.text
        return result

    def offset(self, offset_x: int, offset_y: int) -> 'BoundingBox':
        """Create new BoundingBox with offset applied"""
        new_points = [
            [pt[0] - offset_x, pt[1] - offset_y]
            for pt in self.points
        ]
        return BoundingBox(
            type=self.type,
            points=new_points,
            annotation_index=self.annotation_index,
            conf=self.conf,
            text=self.text
        )


@dataclass
class MatcherConfig:
    """Configuration for creating a matcher"""
    serial_number: str
    template_idx: int
    template_path: Path
    template_bbox: BoundingBox
    other_bboxes: List[BoundingBox] = field(default_factory=list)
    crop_area: Optional[CropArea] = None
    engine_path: str = ""
    verbose: bool = False

    def get_all_bboxes(self) -> List[Dict[str, Any]]:
        """Get all bboxes as list of dicts"""
        bboxes = [self.template_bbox.to_dict()]
        bboxes.extend([bbox.to_dict() for bbox in self.other_bboxes])
        return bboxes
