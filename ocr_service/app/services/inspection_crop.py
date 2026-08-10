"""
Resolve + crop OCR regions out of frame images saved by backend's inference
pipeline.

Cropping uses a perspective warp of the region's 4 points (crop_obb), NOT the
axis-aligned bounding box that anomaly_service uses. That is deliberate: the
production OCR path warps the oriented box before recognition, so an
axis-aligned crop would hand the model pixels it never sees at inference time —
extra background on rotated text, and a different aspect ratio. Ported from
backend/scripts/crop_ocr_training_data.py, which produced the data_ocr_merged
sample set.
"""
import base64
from pathlib import Path
from typing import List, Optional

import cv2 as cv
import numpy as np

from app.core.config import BACKEND_UPLOADS_PATH

# detected_regions[].type values that carry readable text.
OCR_REGION_TYPES = ("text", "datecode")


def _prefer_original(image_path: str) -> str:
    """Swap a `_viz` suffix for `_org` so crops come from the raw frame rather
    than the one with bbox overlays burned in."""
    if not image_path:
        return image_path
    for suf in ("_viz.jpg", "_viz.jpeg", "_viz.png"):
        if image_path.endswith(suf):
            return image_path[: -len(suf)] + "_org" + suf[4:]
    return image_path


def resolve_inspection_image(image_path: str) -> Optional[Path]:
    """Resolve a relative inspection image_path to an absolute Path on disk,
    preferring the original frame over the visualization. Returns None when
    neither exists — frames get pruned by storage_cleanup_scheduler while the
    Mongo record lives on, so a missing file is normal, not an error."""
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


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [TL, TR, BR, BL] for any rotation."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL
    rect[2] = pts[np.argmax(s)]   # BR
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # TR
    rect[3] = pts[np.argmax(d)]   # BL
    return rect


# A crop this small holds no readable text; below it the quad is a rounding
# artifact rather than a region.
MIN_CROP_SIDE = 6
# How far outside the frame a quad may legitimately sit. A partially visible
# label really does project past the edge; 20% of the dimension covers that
# while still rejecting the failed-alignment quads described in quad_is_sane.
_BOUNDS_TOLERANCE = 0.2


def quad_is_sane(points: List[List[float]], img_h: int, img_w: int) -> bool:
    """Reject annotation quads that template alignment projected nonsensically.

    When SuperPoint fails to locate the template, the recipe's annotation quad
    still gets projected — to coordinates like x=-38418 on a 2448px-wide frame.
    Measured on real inspections, ~19% of text/datecode regions on FAIL frames
    are this kind of garbage. warpPerspective happily produces a 43658x0 image
    from one, so the degenerate-size check alone catches it only by accident;
    checking the input is what makes the rejection deliberate.
    """
    if not points or len(points) < 4:
        return False
    pts = np.array(points, dtype=np.float32)[:4]
    if not np.isfinite(pts).all():
        return False
    mx, my = img_w * _BOUNDS_TOLERANCE, img_h * _BOUNDS_TOLERANCE
    if (pts[:, 0].min() < -mx or pts[:, 0].max() > img_w + mx
            or pts[:, 1].min() < -my or pts[:, 1].max() > img_h + my):
        return False
    tl, tr, br, bl = _order_points(pts)
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    return w >= MIN_CROP_SIDE and h >= MIN_CROP_SIDE


def crop_region(image: np.ndarray, points: List[List[float]]) -> Optional[np.ndarray]:
    """quad_is_sane + crop_obb + a JPEG-encodability check, as one call.

    Every caller wants all three: a crop that cannot be encoded is useless to
    both the review UI (it renders as a broken tile) and to training.
    """
    if image is None:
        return None
    h, w = image.shape[:2]
    if not quad_is_sane(points, h, w):
        return None
    crop = crop_obb(image, points)
    if crop is None or crop.size == 0:
        return None
    if not cv.imencode(".jpg", crop)[0]:
        return None
    return crop


def crop_obb(image: np.ndarray, points: List[List[float]]) -> Optional[np.ndarray]:
    """Perspective-correct crop of an oriented box — same transform the
    production OCR path applies. None if the box is degenerate."""
    if not points or len(points) < 4:
        return None
    pts = np.array(points, dtype=np.float32)[:4]
    tl, tr, br, bl = _order_points(pts)
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if w <= 0 or h <= 0:
        return None
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    return cv.warpPerspective(image, cv.getPerspectiveTransform(src, dst), (w, h))


def ann_key(value) -> str:
    """Normalize an annotation index for map lookups.

    detected_regions[].annotation_index and
    text_verification.results[].annotation_idx are both ints in current data
    (checked over 1800 regions), but they are written by different code paths.
    Keying on str() means a future int/str drift shows up as a wrong label
    rather than as a verification map that silently never matches — which would
    look like "this recipe has no OCR data" instead of a bug.
    """
    return str(value)


def build_verification_map(frame: dict) -> dict:
    """{ann_key: text_verification result} for one frame."""
    tv = frame.get("text_verification") or {}
    return {
        ann_key(vr.get("annotation_idx")): vr
        for vr in (tv.get("results") or [])
        if vr.get("annotation_idx") is not None
    }


def prefill_from_verification(vr: Optional[dict], region: dict) -> str:
    """Best guess at the ground truth for a crop, to seed the Label tab.

    A matching region read what the recipe expected, so `expected` IS the
    ground truth. A failing region read something else — and on a real misprint
    the printed text is what OCR saw, not what the recipe wanted, so
    `recognized` is the better starting point there. Neither is trustworthy
    enough to train on unreviewed, which is why imports land as need_review.
    """
    if vr is None:
        return region.get("text") or ""
    if vr.get("match"):
        return vr.get("expected") or ""
    return vr.get("recognized") or region.get("text") or ""


def img_to_b64(img: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")
