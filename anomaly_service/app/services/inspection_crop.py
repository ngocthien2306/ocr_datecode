"""
Resolve + crop frame images saved by backend's inference pipeline.
Standalone port of backend/app/api/endpoints/ml_training.py's
_resolve_inspection_image / _prefer_original / _crop_from_polygon — this
service reads the same files but doesn't import backend's package.
"""
import base64
from pathlib import Path
from typing import List, Optional

import cv2 as cv
import numpy as np

from app.core.config import BACKEND_UPLOADS_PATH


def _prefer_original(image_path: str) -> str:
    """Swap a `_viz.jpg` suffix to `_org.jpg` so we crop from the raw frame
    instead of the bbox-overlaid visualization."""
    if not image_path:
        return image_path
    for suf in ("_viz.jpg", "_viz.jpeg", "_viz.png"):
        if image_path.endswith(suf):
            return image_path[: -len(suf)] + "_org" + suf[4:]
    return image_path


def resolve_inspection_image(image_path: str) -> Optional[Path]:
    """Resolve a relative inspection image path to an absolute Path on disk.
    Prefers the original frame (`_org.<ext>`) over the visualization
    (`_viz.<ext>`); falls back to the literal path if `_org` doesn't exist."""
    if not image_path:
        return None

    def _resolve_one(candidate: str) -> Optional[Path]:
        p = Path(candidate)
        if p.is_absolute() and p.exists():
            return p
        joined = BACKEND_UPLOADS_PATH / candidate
        return joined if joined.exists() else None

    org_path = _prefer_original(image_path)
    if org_path != image_path:
        resolved = _resolve_one(org_path)
        if resolved is not None:
            return resolved
    return _resolve_one(image_path)


def crop_from_polygon(img: np.ndarray, points: List[List[float]], padding: int = 4) -> Optional[np.ndarray]:
    """Crop the axis-aligned bbox enclosing a polygon (with padding)."""
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    h, w = img.shape[:2]
    x0 = max(0, int(min(xs)) - padding)
    y0 = max(0, int(min(ys)) - padding)
    x1 = min(w, int(max(xs)) + padding)
    y1 = min(h, int(max(ys)) + padding)
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1].copy()


def img_to_b64(img: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")
