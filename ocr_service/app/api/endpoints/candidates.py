"""
OCR candidates — crop text/datecode regions out of past inference_results,
filtered by recipe, for the Import-from-Recipe modal.

Counterpart to anomaly_service's candidates endpoint, with two differences that
matter: it yields one candidate per REGION rather than per frame (a frame
usually carries several OCR annotations), and each candidate arrives with a
prefilled label guess so the operator reviews text instead of typing it.
"""
import logging
from datetime import datetime as _dt
from typing import Any, Dict, List, Optional

import cv2 as cv
from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies.auth import CurrentUser, get_current_user
from app.db.mongodb import get_database
from app.repositories.ocr_repository import OCRRepository
from app.services.inspection_crop import (
    OCR_REGION_TYPES,
    ann_key,
    build_verification_map,
    crop_region,
    img_to_b64,
    prefill_from_verification,
    resolve_inspection_image,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def get_repo(db=Depends(get_database)) -> OCRRepository:
    return OCRRepository(db)


@router.get("/candidates")
async def get_candidates(
    project_id: str,
    recipe_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    region_type: Optional[str] = Query(None, description="text | datecode"),
    match_filter: str = Query("all", description="all | pass | fail"),
    limit: int = 100,
    max_per_frame: int = Query(4, ge=1, description="cap regions taken per frame, for variety"),
    repo: OCRRepository = Depends(get_repo),
    db=Depends(get_database),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Candidate OCR crops from past inspections.

    `match_filter`:
      pass — only regions the recipe verified as correct. Their `expected` text
             IS the ground truth, so these are near-free training data.
      fail — only regions that failed. The most valuable data (hard crops, real
             misprints) and the ones that most need human review, since the
             prefill comes from what OCR guessed.
      all  — both.
    """
    if not await repo.get_project(project_id):
        raise HTTPException(404, "Project not found")
    if region_type and region_type not in OCR_REGION_TYPES:
        raise HTTPException(400, f"region_type must be one of {OCR_REGION_TYPES}")
    if match_filter not in ("all", "pass", "fail"):
        raise HTTPException(400, "match_filter must be all | pass | fail")

    imported = await repo.get_imported_provenance_keys(project_id)
    wanted_types = [region_type] if region_type else list(OCR_REGION_TYPES)

    query: Dict[str, Any] = {
        "camera_results.frames.detected_regions": {
            "$elemMatch": {"type": {"$in": wanted_types}}
        },
        # Most frames never get an image written (only FAIL/ERROR, or PASS with
        # save_pass_images on), and storage_cleanup_scheduler prunes old ones
        # while the Mongo record stays forever. Without this filter a flat
        # limit() can burn through hundreds of dangling records before reaching
        # one whose file still exists, and return nothing.
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
    # No hard doc-count limit: keep scanning newest-first until `limit`
    # candidates with a readable image are collected. max_docs_scanned is only a
    # safety valve so a date range with zero resolvable images can't scan the
    # whole 600k-document collection.
    max_docs_scanned = max(limit * 50, 2000)
    cursor = coll.find(query).sort("timestamp", -1).limit(max_docs_scanned)

    candidates: List[Dict[str, Any]] = []
    docs_scanned = 0
    # Regions rejected by quad_is_sane — reported rather than silently dropped,
    # because "this recipe yields few candidates" and "this recipe's template
    # alignment is failing" look identical from the UI otherwise.
    skipped_degenerate = 0

    async for doc in cursor:
        if len(candidates) >= limit:
            break
        docs_scanned += 1
        # inference_results._id is stored as a plain string by backend
        # (inference_result_repository sets _id = str(inserted_id)), NOT a BSON
        # ObjectId — even though it is 24 hex chars and looks like one.
        inspection_id = str(doc.get("_id", ""))
        ts = doc.get("timestamp")
        ts_iso = ts.isoformat() if ts else None

        for cam in doc.get("camera_results", []):
            if len(candidates) >= limit:
                break
            cam_serial = cam.get("serial_number", "")

            for frame in cam.get("frames", []):
                if len(candidates) >= limit:
                    break
                regions = [
                    r for r in (frame.get("detected_regions") or [])
                    if r.get("type") in wanted_types and r.get("points")
                ]
                if not regions:
                    continue
                verif = build_verification_map(frame)

                image = None  # loaded lazily: only if a region survives filtering
                taken = 0
                for region in regions:
                    if len(candidates) >= limit or taken >= max_per_frame:
                        break
                    ann_idx = region.get("annotation_index")
                    vr = verif.get(ann_key(ann_idx))
                    matched = bool(vr.get("match", False)) if vr else None
                    if match_filter == "pass" and matched is not True:
                        continue
                    if match_filter == "fail" and matched is not False:
                        continue

                    if image is None:
                        path = resolve_inspection_image(frame.get("image_path") or "")
                        if path is None:
                            break  # no file for this frame — skip the whole frame
                        image = cv.imread(str(path))
                        if image is None:
                            break
                    crop = crop_region(image, region["points"])
                    if crop is None:
                        skipped_degenerate += 1
                        continue

                    frame_idx = int(frame.get("frame_idx", 0))
                    key = f"{inspection_id}:{cam_serial}:{frame_idx}:{ann_idx}"
                    candidates.append({
                        "inspection_id": inspection_id,
                        "recipe_id": str(doc.get("recipe_id", "")),
                        "recipe_name": doc.get("recipe_name", ""),
                        "camera_serial": cam_serial,
                        "frame_idx": frame_idx,
                        "annotation_index": ann_idx,
                        "region_type": region.get("type"),
                        "timestamp": ts_iso,
                        "product_pass_fail": doc.get("product_pass_fail"),
                        "expected_text": (vr or {}).get("expected"),
                        "recognized_text": (vr or {}).get("recognized") or region.get("text"),
                        "ocr_confidence": (vr or {}).get("confidence"),
                        "verify_match": matched,
                        "prefill_text": prefill_from_verification(vr, region),
                        "crop_b64": img_to_b64(crop),
                        "imported_status": imported.get(key),
                    })
                    taken += 1

    return {
        "candidates": candidates,
        "count": len(candidates),
        "docs_scanned": docs_scanned,
        "skipped_degenerate": skipped_degenerate,
    }


@router.get("/candidates/recipes")
async def list_recipes_with_ocr_data(
    limit: int = 50,
    db=Depends(get_database),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Recipes that actually have OCR regions on record, for the modal's recipe
    dropdown. Reading the recipes collection would list every recipe ever
    created, most of which have no inspections to crop from."""
    coll = db.get_collection("inference_results")
    pipeline = [
        {"$match": {
            "camera_results.frames.detected_regions": {
                "$elemMatch": {"type": {"$in": list(OCR_REGION_TYPES)}}
            },
            "camera_results.frames.image_path": {"$ne": None},
        }},
        # Newest-first then cap before grouping: an unbounded $group over 600k
        # documents is a multi-second collection scan on every modal open, and
        # a recipe with no recent inspections has nothing croppable left anyway
        # (storage_cleanup_scheduler has long since pruned its images).
        {"$sort": {"timestamp": -1}},
        {"$limit": 20000},
        {"$group": {
            "_id": {"recipe_id": "$recipe_id", "recipe_name": "$recipe_name"},
            "n": {"$sum": 1},
            "latest": {"$max": "$timestamp"},
        }},
        {"$sort": {"latest": -1}},
        {"$limit": limit},
    ]
    out = []
    async for row in coll.aggregate(pipeline):
        out.append({
            "recipe_id": str(row["_id"].get("recipe_id", "")),
            "recipe_name": row["_id"].get("recipe_name", ""),
            "inspection_count": row["n"],
            "latest": row["latest"].isoformat() if row.get("latest") else None,
        })
    return {"recipes": out, "scanned_window": 20000}
