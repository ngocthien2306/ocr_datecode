"""
Anomaly candidates — crop the `label` region out of past inference_results,
filtered by recipe, for the Import-from-Recipe modal.

Standalone counterpart to backend's ml_training.py::get_inspection_candidates,
but crops the whole-label region (detected_regions[type='label']) instead of
per-character regions, and reads inference_results directly (this service has
its own MongoDB connection to the same database).
"""
import logging
from datetime import datetime as _dt
from typing import Any, Dict, List, Optional

import cv2 as cv
from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies.auth import CurrentUser, get_current_user
from app.db.mongodb import get_database
from app.repositories.anomaly_repository import AnomalyRepository
from app.services.inspection_crop import crop_from_polygon, img_to_b64, resolve_inspection_image

logger = logging.getLogger(__name__)

router = APIRouter()


def get_repo(db=Depends(get_database)) -> AnomalyRepository:
    return AnomalyRepository(db)


@router.get("/candidates", tags=["Anomaly Candidates"])
async def get_candidates(
    project_id: str,
    recipe_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 100,
    repo: AnomalyRepository = Depends(get_repo),
    db=Depends(get_database),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Return label-crop candidates from past inspections, filtered by recipe.

    Each candidate is one frame's `label` region, cropped from the saved
    frame image. Already-imported candidates are tagged with their split
    (train/test) so the FE can disable the checkbox.
    """
    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    imported_keys = await repo.get_imported_provenance_keys(project_id)

    query: Dict[str, Any] = {
        "camera_results.frames.detected_regions": {"$elemMatch": {"type": "label"}},
        # Most frames never get an image saved to disk (only FAIL/ERROR, or
        # PASS with save_pass_images on) and older saved images get pruned by
        # storage_cleanup_scheduler while the Mongo record stays forever. Filter
        # those out server-side, otherwise a flat `.limit(N)` on the raw cursor
        # (sorted newest-first) can burn through hundreds of dangling-image
        # docs before ever reaching one with a usable file, or a valid-image
        # window further back in time -- yielding zero candidates even when
        # plenty exist.
        "camera_results.frames.image_path": {"$ne": None},
    }
    if recipe_id:
        query["recipe_id"] = recipe_id
    if date_from or date_to:
        ts: Dict[str, Any] = {}
        if date_from:
            ts["$gte"] = _dt.fromisoformat(date_from.replace("Z", "+00:00"))
        if date_to:
            ts["$lte"] = _dt.fromisoformat(date_to.replace("Z", "+00:00"))
        query["timestamp"] = ts

    coll = db.get_collection("inference_results")
    # No hard doc-count limit: keep scanning (newest-first) until we've
    # collected `limit` candidates whose image file actually still exists.
    # max_docs_scanned is just a safety valve for the case where the whole
    # date range has zero resolvable images, so the request can't hang scanning
    # the entire collection.
    max_docs_scanned = max(limit * 50, 2000)
    cursor = coll.find(query).sort("timestamp", -1).limit(max_docs_scanned)

    candidates: List[Dict[str, Any]] = []

    docs_scanned = 0
    async for doc in cursor:
        if len(candidates) >= limit:
            break
        docs_scanned += 1
        recipe_name = doc.get("recipe_name", "")
        rid = str(doc.get("recipe_id", ""))
        ts = doc.get("timestamp")
        ts_iso = ts.isoformat() if ts else None
        inspection_id = str(doc.get("_id", ""))

        for cam in doc.get("camera_results", []):
            if len(candidates) >= limit:
                break
            cam_serial = cam.get("serial_number", "")
            for frame in cam.get("frames", []):
                if len(candidates) >= limit:
                    break
                regions = frame.get("detected_regions") or []
                label_region = next((r for r in regions if r.get("type") == "label"), None)
                if label_region is None:
                    continue
                points = label_region.get("points") or []
                if not points:
                    continue

                image_path_raw = frame.get("image_path")
                image_path = resolve_inspection_image(image_path_raw or "")
                if image_path is None:
                    continue

                img = cv.imread(str(image_path))
                if img is None:
                    continue
                crop = crop_from_polygon(img, points, padding=4)
                if crop is None or crop.size == 0:
                    continue

                frame_idx_i = int(frame.get("frame_idx", 0))
                key = f"{inspection_id}:{cam_serial}:{frame_idx_i}"
                imported_split = imported_keys.get(key)

                candidates.append({
                    "inspection_id":   inspection_id,
                    "recipe_id":       rid,
                    "recipe_name":     recipe_name,
                    "camera_serial":   cam_serial,
                    "frame_idx":       frame_idx_i,
                    "product_pass_fail": doc.get("product_pass_fail"),
                    "timestamp":       ts_iso,
                    "crop_b64":        img_to_b64(crop),
                    "imported_split":  imported_split,
                })

    return {"candidates": candidates, "count": len(candidates), "docs_scanned": docs_scanned}
