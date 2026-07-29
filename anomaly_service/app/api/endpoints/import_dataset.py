"""
Import selected label-crops into a project's dataset folder.

    normal   → dataset/train/good/{inspection_id}_{camera_serial}_{frame_idx}.jpg
    abnormal → dataset/test/{defect_type}/{...}.jpg

Train/test split for *normal* images happens at train time (see
AnomalyTrainRequest.test_split), not here — every imported normal crop lands
in train/good, and the training service carves out a held-out slice into
test/good when it builds the run.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List

import cv2 as cv
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies.auth import CurrentUser, get_current_user
from app.db.mongodb import get_database
from app.models.anomaly import AnomalyImportRequest
from app.repositories.anomaly_repository import AnomalyRepository
from app.services import dataset_fs
from app.services.inspection_crop import crop_from_polygon, resolve_inspection_image

logger = logging.getLogger(__name__)

router = APIRouter()


def get_repo(db=Depends(get_database)) -> AnomalyRepository:
    return AnomalyRepository(db)


@router.post("/projects/{project_id}/import", tags=["Anomaly Candidates"])
async def import_candidates(
    project_id: str,
    request: AnomalyImportRequest,
    repo: AnomalyRepository = Depends(get_repo),
    db=Depends(get_database),
    current_user: CurrentUser = Depends(get_current_user),
):
    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    if not request.selections:
        raise HTTPException(400, "No selections provided")

    for sel in request.selections:
        if sel.label not in ("normal", "abnormal"):
            raise HTTPException(400, f"Invalid label '{sel.label}' — must be 'normal' or 'abnormal'")
        if sel.label == "abnormal" and not sel.defect_type:
            raise HTTPException(400, "defect_type is required for abnormal selections")

    dataset_fs.ensure_project_dirs(project_id)
    already_imported = await repo.get_imported_provenance_keys(project_id)

    coll = db.get_collection("inference_results")
    imported = 0
    skipped: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for sel in request.selections:
        key = f"{sel.inspection_id}:{sel.camera_serial}:{sel.frame_idx}"
        if key in already_imported:
            skipped.append({"inspection_id": sel.inspection_id, "reason": "already imported"})
            continue

        if not ObjectId.is_valid(sel.inspection_id):
            errors.append({"inspection_id": sel.inspection_id, "reason": "invalid inspection_id"})
            continue

        # inference_results._id is stored as a plain string by backend
        # (inference_result_repository.py sets `_id = str(inserted_id)` and
        # queries it back as a string too) -- NOT a real BSON ObjectId, even
        # though it's 24 hex chars and looks like one. Querying with
        # ObjectId(...) here would never match.
        doc = await coll.find_one({"_id": sel.inspection_id})
        if not doc:
            errors.append({"inspection_id": sel.inspection_id, "reason": "inspection not found"})
            continue

        cam = next(
            (c for c in doc.get("camera_results", []) if c.get("serial_number") == sel.camera_serial),
            None,
        )
        if not cam:
            errors.append({"inspection_id": sel.inspection_id, "reason": "camera not found in inspection"})
            continue
        frame = next((f for f in cam.get("frames", []) if int(f.get("frame_idx", 0)) == sel.frame_idx), None)
        if not frame:
            errors.append({"inspection_id": sel.inspection_id, "reason": "frame not found"})
            continue

        regions = frame.get("detected_regions") or []
        label_region = next((r for r in regions if r.get("type") == "label"), None)
        points = (label_region or {}).get("points") or []
        image_path = resolve_inspection_image(frame.get("image_path") or "")
        if not points or image_path is None:
            errors.append({"inspection_id": sel.inspection_id, "reason": "no label region / image on frame"})
            continue

        img = cv.imread(str(image_path))
        if img is None:
            errors.append({"inspection_id": sel.inspection_id, "reason": "failed to read frame image"})
            continue
        crop = crop_from_polygon(img, points, padding=4)
        if crop is None or crop.size == 0:
            errors.append({"inspection_id": sel.inspection_id, "reason": "empty crop"})
            continue

        filename = f"{sel.inspection_id}_{sel.camera_serial}_{sel.frame_idx}.jpg"
        if sel.label == "normal":
            dest_dir = dataset_fs.train_good_dir(project_id)
            split = "train"
        else:
            dest_dir = dataset_fs.test_defect_dir(project_id, sel.defect_type)
            dest_dir.mkdir(parents=True, exist_ok=True)
            split = "test"

        dest_path = dest_dir / filename
        ok = cv.imwrite(str(dest_path), crop)
        if not ok:
            errors.append({"inspection_id": sel.inspection_id, "reason": "failed to write crop file"})
            continue

        await repo.insert_import_item({
            "project_id": project_id,
            "inspection_id": sel.inspection_id,
            "camera_serial": sel.camera_serial,
            "frame_idx": sel.frame_idx,
            "recipe_id": str(doc.get("recipe_id", "")),
            "recipe_name": doc.get("recipe_name", ""),
            "label": sel.label,
            "defect_type": sel.defect_type,
            "split": split,
            "image_path": str(dest_path.relative_to(dataset_fs.project_dir(project_id))),
            "created_at": datetime.utcnow(),
        })
        already_imported[key] = split
        imported += 1

    counts = dataset_fs.count_images(project_id)
    await repo.set_counts(project_id, counts["normal"], counts["abnormal"])

    return {
        "imported": imported,
        "skipped": len(skipped),
        "errors": errors,
        "normal_count": counts["normal"],
        "abnormal_count": counts["abnormal"],
    }
