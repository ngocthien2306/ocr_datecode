"""
Dataset gallery + label editing.

The Label tab lives on these endpoints: it lists crops with their current
ground-truth text, lets an operator fix the text and promote the item from
need_review to verified, and offers the bulk actions that make reviewing a few
hundred crops tolerable.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2 as cv
from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies.auth import CurrentUser, get_current_user
from app.db.mongodb import get_database
from app.models.ocr import (
    LABEL_STATUSES,
    OCRBulkExcludeRequest,
    OCRBulkIdsRequest,
    OCRBulkSplitRequest,
    OCRBulkStatusRequest,
    OCRRelabelRequest,
)
from app.repositories.ocr_repository import OCRRepository
from app.services import dataset_fs
from app.services.dataset_builder import blocking_reason, build_dataset
from app.services.inspection_crop import img_to_b64

logger = logging.getLogger(__name__)

router = APIRouter()

# Thumbnails are height-normalised rather than box-fitted: these are single
# lines of text with wildly different aspect ratios, and squeezing them into a
# square makes the characters unreadable — which defeats the point of a review UI.
THUMB_HEIGHT = 48
THUMB_MAX_WIDTH = 640


def get_repo(db=Depends(get_database)) -> OCRRepository:
    return OCRRepository(db)


def _thumb_b64(project_id: str, image_path: str) -> str:
    path = dataset_fs.image_abs_path(project_id, image_path)
    img = cv.imread(str(path))
    if img is None:
        return ""
    h, w = img.shape[:2]
    scale = THUMB_HEIGHT / max(h, 1)
    new_w = min(max(int(w * scale), 1), THUMB_MAX_WIDTH)
    return img_to_b64(cv.resize(img, (new_w, THUMB_HEIGHT)), quality=80)


def _item_out(item, project_id: str, with_thumb: bool) -> Dict[str, Any]:
    out = item.model_dump(by_alias=False)
    if with_thumb:
        out["thumb_b64"] = _thumb_b64(project_id, item.image_path)
    return out


@router.get("/projects/{project_id}/dataset/items")
async def list_items(
    project_id: str,
    status: Optional[str] = Query(None, description="need_review | verified | rejected"),
    split: Optional[str] = Query(None, description="train | test"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    if not await repo.get_project(project_id):
        raise HTTPException(404, "Project not found")
    if status and status not in LABEL_STATUSES:
        raise HTTPException(400, f"status must be one of {LABEL_STATUSES}")
    if split and split not in ("train", "test"):
        raise HTTPException(400, "split must be 'train' or 'test'")

    items, total = await repo.list_items_page(
        project_id, status, split, skip=(page - 1) * page_size, limit=page_size,
    )
    return {
        "items": [_item_out(i, project_id, with_thumb=True) for i in items],
        "count": len(items),
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }


@router.get("/projects/{project_id}/dataset/item-ids")
async def list_item_ids(
    project_id: str,
    status: Optional[str] = None,
    split: Optional[str] = None,
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Ids only, no thumbnails — lets "select all" span pages without pulling
    every page's base64 images."""
    if not await repo.get_project(project_id):
        raise HTTPException(404, "Project not found")
    ids = await repo.list_item_ids(project_id, status, split)
    return {"ids": ids, "total": len(ids)}


@router.get("/projects/{project_id}/dataset/items/{item_id}/full")
async def get_item_full(
    project_id: str,
    item_id: str,
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Full-resolution crop, for zooming in on an ambiguous character."""
    item = await repo.get_item(item_id)
    if not item or item.project_id != project_id:
        raise HTTPException(404, "Item not found")
    img = cv.imread(str(dataset_fs.image_abs_path(project_id, item.image_path)))
    if img is None:
        raise HTTPException(404, "Image file missing on disk")
    return {"id": item.id, "full_b64": img_to_b64(img), "width": img.shape[1], "height": img.shape[0]}


@router.patch("/projects/{project_id}/dataset/items/{item_id}")
async def relabel_item(
    project_id: str,
    item_id: str,
    request: OCRRelabelRequest,
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Edit one item's ground truth, status or split."""
    item = await repo.get_item(item_id)
    if not item or item.project_id != project_id:
        raise HTTPException(404, "Item not found")
    if request.status and request.status not in LABEL_STATUSES:
        raise HTTPException(400, f"status must be one of {LABEL_STATUSES}")
    if request.split and request.split not in ("train", "test"):
        raise HTTPException(400, "split must be 'train' or 'test'")

    update: Dict[str, Any] = {}
    if request.gt_text is not None:
        update["gt_text"] = request.gt_text
    if request.split is not None:
        update["split"] = request.split
    if request.status is not None:
        update["status"] = request.status
        if request.status == "verified":
            update["verified_by"] = current_user.username
            update["verified_at"] = datetime.utcnow()
    if not update:
        raise HTTPException(400, "Nothing to update")

    # An empty label can never be verified: OpenOCR's encoder returns None for
    # an empty string and drops the sample, so it would be an invisible
    # shrinking of the training set rather than a visible error.
    final_text = update.get("gt_text", item.gt_text)
    final_status = update.get("status", item.status)
    if final_status == "verified" and not final_text.strip():
        raise HTTPException(400, "Cannot verify an item with an empty gt_text")

    await repo.update_item(item_id, update)
    counts = await repo.sync_project_counts(project_id)
    updated = await repo.get_item(item_id)
    return {"item": _item_out(updated, project_id, with_thumb=False), **counts}


@router.post("/projects/{project_id}/dataset/items/bulk-status")
async def bulk_set_status(
    project_id: str,
    request: OCRBulkStatusRequest,
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Promote/reject many items at once — e.g. verify every item whose recipe
    already confirmed the text (verify_match=true), which is the bulk of a
    typical import."""
    if request.status not in LABEL_STATUSES:
        raise HTTPException(400, f"status must be one of {LABEL_STATUSES}")
    if not await repo.get_project(project_id):
        raise HTTPException(404, "Project not found")

    update: Dict[str, Any] = {"status": request.status}
    if request.status == "verified":
        update["verified_by"] = current_user.username
        update["verified_at"] = datetime.utcnow()
        # Same rule as the single-item path, applied by filtering rather than
        # erroring: a bulk verify over a mixed selection should promote what it
        # can and report the rest, not fail wholesale.
        items = await repo.get_items(project_id, request.ids)
        ok_ids = [i.id for i in items if i.gt_text.strip()]
        skipped_empty = len(items) - len(ok_ids)
    else:
        ok_ids, skipped_empty = request.ids, 0

    modified = await repo.update_items(project_id, ok_ids, update)
    counts = await repo.sync_project_counts(project_id)
    return {"modified": modified, "skipped_empty_text": skipped_empty, **counts}


@router.post("/projects/{project_id}/dataset/items/bulk-split")
async def bulk_set_split(
    project_id: str,
    request: OCRBulkSplitRequest,
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Pin items to train or test.

    Note this moves the row, not the file: image_path keeps pointing at
    dataset/{original_split}/, and the label file written per run resolves paths
    against the dataset dir either way. Keeping the file put avoids a
    move-then-crash leaving a row that points nowhere.
    """
    if request.split not in ("train", "test"):
        raise HTTPException(400, "split must be 'train' or 'test'")
    if not await repo.get_project(project_id):
        raise HTTPException(404, "Project not found")
    modified = await repo.update_items(project_id, request.ids, {"split": request.split})
    return {"modified": modified}


@router.post("/projects/{project_id}/dataset/items/bulk-exclude")
async def bulk_exclude(
    project_id: str,
    request: OCRBulkExcludeRequest,
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Hold items out of the next run without deleting them."""
    if not await repo.get_project(project_id):
        raise HTTPException(404, "Project not found")
    modified = await repo.update_items(
        project_id, request.ids, {"exclude_from_training": request.excluded},
    )
    return {"modified": modified, "excluded": request.excluded}


@router.post("/projects/{project_id}/dataset/prepare")
async def prepare_dataset(
    project_id: str,
    test_split: float = Query(0.2, ge=0.0, lt=1.0),
    use_space_char: bool = Query(True),
    max_text_length: int = Query(25, ge=1),
    dry_run: bool = Query(True),
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Validate the verified items and write the two rec_gt label files.

    `dry_run=true` (the default) validates and reports without touching disk —
    that is what the Train tab calls to show what a run will see before the
    operator commits to it. Training calls it for real.

    The report exists because every failure mode here is silent otherwise:
    an over-long label makes OpenOCR substitute a different image, an unknown
    character is stripped without complaint, and a dataset that is 95% one
    recipe trains a model that only reads that recipe's font.
    """
    if not await repo.get_project(project_id):
        raise HTTPException(404, "Project not found")

    items = await repo.list_trainable_items(project_id)
    report = build_dataset(
        project_id, items,
        test_split=test_split,
        use_space_char=use_space_char,
        max_text_length=max_text_length,
        dry_run=dry_run,
    )
    report["blocking_reason"] = blocking_reason(report)
    return report


@router.post("/projects/{project_id}/dataset/items/bulk-delete")
async def bulk_delete(
    project_id: str,
    request: OCRBulkIdsRequest,
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Delete rows and their image files. The row is the only pointer to the
    file, so dropping one without the other leaks disk space forever."""
    if not await repo.get_project(project_id):
        raise HTTPException(404, "Project not found")
    deleted = await repo.delete_items(project_id, request.ids)
    for item in deleted:
        dataset_fs.image_abs_path(project_id, item.image_path).unlink(missing_ok=True)
    counts = await repo.sync_project_counts(project_id)
    return {"deleted": len(deleted), **counts}
