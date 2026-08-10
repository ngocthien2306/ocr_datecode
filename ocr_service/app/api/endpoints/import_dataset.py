"""
Import crops into a project's dataset.

Two ways in:
  POST /projects/{id}/import         selected regions from past inspections
  POST /projects/{id}/import-folder  an existing OpenOCR-format dataset on disk

Both write the image under dataset/{split}/ and one ocr_dataset_items row per
crop. Inspection imports arrive as `need_review` — their label is a guess (see
prefill_from_verification) and training only reads `verified` items. Folder
imports arrive as `verified`, since a rec_gt file is already reviewed ground
truth.
"""
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2 as cv
from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies.auth import CurrentUser, get_current_user
from app.core.config import SERVICE_ROOT
from app.db.mongodb import get_database
from app.models.ocr import OCRImportFolderRequest, OCRImportRequest
from app.repositories.ocr_repository import OCRRepository
from app.services import dataset_fs
from app.services.inspection_crop import (
    ann_key,
    build_verification_map,
    crop_region,
    prefill_from_verification,
    resolve_inspection_image,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def get_repo(db=Depends(get_database)) -> OCRRepository:
    return OCRRepository(db)


@router.post("/projects/{project_id}/import")
async def import_candidates(
    project_id: str,
    request: OCRImportRequest,
    repo: OCRRepository = Depends(get_repo),
    db=Depends(get_database),
    current_user: CurrentUser = Depends(get_current_user),
):
    if not await repo.get_project(project_id):
        raise HTTPException(404, "Project not found")
    if not request.selections:
        raise HTTPException(400, "No selections provided")
    if request.split not in ("train", "test"):
        raise HTTPException(400, "split must be 'train' or 'test'")

    dataset_fs.ensure_project_dirs(project_id)
    dest_dir = dataset_fs.split_dir(project_id, request.split)
    coll = db.get_collection("inference_results")

    imported = 0
    skipped: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    # One inspection can supply several selections; cache the decoded frame
    # image so a 4-region frame is read from disk once, not four times.
    image_cache: Dict[str, Any] = {}

    for sel in request.selections:
        # _id is a plain string here, not an ObjectId — see candidates.py.
        doc = await coll.find_one({"_id": sel.inspection_id})
        if not doc:
            errors.append({"inspection_id": sel.inspection_id, "reason": "inspection not found"})
            continue

        cam = next(
            (c for c in doc.get("camera_results", []) if c.get("serial_number") == sel.camera_serial),
            None,
        )
        if not cam:
            errors.append({"inspection_id": sel.inspection_id, "reason": "camera not in inspection"})
            continue
        frame = next(
            (f for f in cam.get("frames", []) if int(f.get("frame_idx", 0)) == sel.frame_idx),
            None,
        )
        if not frame:
            errors.append({"inspection_id": sel.inspection_id, "reason": "frame not found"})
            continue

        region = next(
            (r for r in (frame.get("detected_regions") or [])
             if r.get("annotation_index") == sel.annotation_index),
            None,
        )
        if not region or not region.get("points"):
            errors.append({"inspection_id": sel.inspection_id,
                           "reason": f"region {sel.annotation_index} not found"})
            continue

        cache_key = f"{sel.inspection_id}:{sel.camera_serial}:{sel.frame_idx}"
        if cache_key not in image_cache:
            path = resolve_inspection_image(frame.get("image_path") or "")
            image_cache[cache_key] = cv.imread(str(path)) if path else None
        image = image_cache[cache_key]
        if image is None:
            errors.append({"inspection_id": sel.inspection_id, "reason": "frame image missing on disk"})
            continue

        crop = crop_region(image, region["points"])
        if crop is None:
            # Almost always a quad that template alignment projected outside the
            # frame — see quad_is_sane.
            errors.append({"inspection_id": sel.inspection_id,
                           "reason": f"region {sel.annotation_index}: degenerate crop"})
            continue

        vr = build_verification_map(frame).get(ann_key(sel.annotation_index))
        prefill = prefill_from_verification(vr, region)
        gt_text = sel.gt_text if sel.gt_text is not None else prefill

        filename = (f"{sel.inspection_id}_{sel.camera_serial}_f{sel.frame_idx}"
                    f"_ann{sel.annotation_index}.jpg")
        dest_path = dest_dir / filename
        if not cv.imwrite(str(dest_path), crop):
            errors.append({"inspection_id": sel.inspection_id, "reason": "failed to write crop"})
            continue

        item_id = await repo.insert_item({
            "project_id": project_id,
            "inspection_id": sel.inspection_id,
            "camera_serial": sel.camera_serial,
            "frame_idx": sel.frame_idx,
            "annotation_index": sel.annotation_index,
            "recipe_id": str(doc.get("recipe_id", "")),
            "recipe_name": doc.get("recipe_name", ""),
            "region_type": region.get("type"),
            "gt_text": gt_text,
            "prefill_text": prefill,
            "expected_text": (vr or {}).get("expected"),
            "recognized_text": (vr or {}).get("recognized") or region.get("text"),
            "ocr_confidence": (vr or {}).get("confidence"),
            "verify_match": bool(vr.get("match")) if vr else None,
            # Never 'verified' on import: the label is a guess, and training
            # reads only verified items. The Label tab is what promotes it.
            "status": "need_review",
            "split": request.split,
            "image_path": str(dest_path.relative_to(dataset_fs.project_dir(project_id))),
            "source": "import",
            "exclude_from_training": False,
            "created_at": datetime.utcnow(),
        })
        if item_id is None:
            # Lost the race against the unique import_dedup index, or the FE
            # sent a row the grid should have greyed out. The file we just wrote
            # is the same crop under the same deterministic name, so leaving it
            # is harmless.
            skipped.append({"inspection_id": sel.inspection_id, "reason": "already imported"})
            continue
        imported += 1

    counts = await repo.sync_project_counts(project_id)
    return {
        "imported": imported,
        "skipped": len(skipped),
        "errors": errors,
        **counts,
    }


@router.post("/projects/{project_id}/import-folder")
async def import_folder(
    project_id: str,
    request: OCRImportFolderRequest,
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Seed a project from an OpenOCR-format dataset directory.

    Expects rec_gt_train.txt / rec_gt_test.txt with "<relative/path>\\t<label>"
    lines. Images are copied (not linked) into the project so the project stays
    self-contained if the source folder moves or is deleted.
    """
    if not await repo.get_project(project_id):
        raise HTTPException(404, "Project not found")

    src = Path(request.folder)
    if not src.is_absolute():
        src = (SERVICE_ROOT / request.folder).resolve()
    if not src.is_dir():
        raise HTTPException(400, f"Folder not found: {src}")

    dataset_fs.ensure_project_dirs(project_id)
    now = datetime.utcnow()
    status = "verified" if request.mark_verified else "need_review"

    imported = 0
    per_split: Dict[str, int] = {}
    errors: List[Dict[str, Any]] = []

    for split in ("train", "test"):
        gt_file = src / f"rec_gt_{split}.txt"
        if not gt_file.is_file():
            continue
        dest_dir = dataset_fs.split_dir(project_id, split)
        n = 0
        for lineno, raw in enumerate(gt_file.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            rel, sep, label = line.partition("\t")
            if not sep:
                errors.append({"file": gt_file.name, "line": lineno, "reason": "no TAB separator"})
                continue
            src_img = src / rel
            if not src_img.is_file():
                errors.append({"file": gt_file.name, "line": lineno, "reason": f"missing image {rel}"})
                continue

            # Flatten the source's own subdirectories into one name so
            # train/x.jpg and test/x.jpg can't collide inside the project.
            dest_path = dest_dir / f"{split}_{Path(rel).name}"
            try:
                shutil.copyfile(src_img, dest_path)
            except OSError as e:
                errors.append({"file": gt_file.name, "line": lineno, "reason": f"copy failed: {e}"})
                continue

            item_id = await repo.insert_item({
                "project_id": project_id,
                "gt_text": label,
                "prefill_text": label,
                "status": status,
                "split": split,
                "image_path": str(dest_path.relative_to(dataset_fs.project_dir(project_id))),
                "source": "seed",
                "exclude_from_training": False,
                "created_at": now,
            })
            if item_id is None:
                continue
            imported += 1
            n += 1
        per_split[split] = n

    counts = await repo.sync_project_counts(project_id)
    return {
        "imported": imported,
        "per_split": per_split,
        # Truncated: a wrong --folder can produce one error per line, and the
        # count below is what tells you that happened.
        "errors": errors[:20],
        "error_count": len(errors),
        "source": str(src),
        **counts,
    }
