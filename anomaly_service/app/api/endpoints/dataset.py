"""
Browse / relabel / delete already-imported dataset images.

Counterpart to import_dataset.py (which only writes new items) and the
dataset-stats-only 'Dataset' tab — this is the actual gallery: list every
imported crop (paginated) with a thumbnail, move it between normal/abnormal
(+ defect type) on disk one-at-a-time or in bulk, or delete it (one or in
bulk). Roboflow-style dataset review flow.
"""
import base64
import logging
from typing import Any, Dict, List, Optional

import cv2 as cv
from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies.auth import CurrentUser, get_current_user
from app.db.mongodb import get_database
from app.models.anomaly import AnomalyBulkDeleteRequest, AnomalyBulkRelabelRequest, AnomalyRelabelRequest
from app.repositories.anomaly_repository import AnomalyRepository
from app.services import dataset_fs

logger = logging.getLogger(__name__)

router = APIRouter()

THUMB_MAX_SIDE = 220
# Cap for the click-to-enlarge lightbox -- full resolution for typical label
# crops (a few hundred px), only downscaled if something unusually large
# slipped in, to keep the response from ballooning.
FULL_MAX_SIDE = 1600
DEFAULT_PAGE_SIZE = 60


def get_repo(db=Depends(get_database)) -> AnomalyRepository:
    return AnomalyRepository(db)


def _encode_b64(img_path, max_side: int, quality: int) -> str:
    img = cv.imread(str(img_path))
    if img is None:
        return ""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img = cv.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
    ok, buf = cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("utf-8") if ok else ""


def _thumb_b64(img_path) -> str:
    return _encode_b64(img_path, THUMB_MAX_SIDE, quality=80)


def _full_b64(img_path) -> str:
    return _encode_b64(img_path, FULL_MAX_SIDE, quality=92)


async def _recount(repo: AnomalyRepository, project_id: str) -> Dict[str, int]:
    counts = dataset_fs.count_images(project_id)
    await repo.set_counts(project_id, counts["normal"], counts["abnormal"])
    return counts


def _cleanup_if_empty_defect_dir(d) -> None:
    """rmdir a test/<defect_type> dir left empty by a move/delete.

    Never touches "good" (train/good or test/good must always exist for
    anomalib), only ever a defect-type dir -- and only when it's genuinely
    empty. Without this, a stale empty test/<defect_type> dir makes
    dataset_fs.list_defect_types() (before its own has_images guard) or a
    human staring at the folder think there's abnormal data that isn't
    really there.
    """
    try:
        if d.is_dir() and d.name != "good" and not any(d.iterdir()):
            d.rmdir()
    except OSError:
        pass


def _relabel_one_on_disk(proj_dir, item, label: str, defect_type: Optional[str]) -> Dict[str, Any]:
    """Move one item's file to its new label's folder. Returns the Mongo
    $set update dict, or raises HTTPException if the source file is gone."""
    src_path = proj_dir / item.image_path
    if not src_path.exists():
        raise HTTPException(404, f"Image file missing on disk for {item.id}")

    if label == "normal":
        dest_dir = dataset_fs.train_good_dir(item.project_id)
        new_split = "train"
        new_defect_type = None
    else:
        dest_dir = dataset_fs.test_defect_dir(item.project_id, defect_type)
        dest_dir.mkdir(parents=True, exist_ok=True)
        new_split = "test"
        new_defect_type = defect_type

    old_parent = src_path.parent
    dest_path = dest_dir / src_path.name
    if src_path.resolve() != dest_path.resolve():
        src_path.rename(dest_path)
        _cleanup_if_empty_defect_dir(old_parent)

    return {
        "label": label,
        "defect_type": new_defect_type,
        "split": new_split,
        "image_path": str(dest_path.relative_to(proj_dir)),
    }


@router.get("/projects/{project_id}/dataset/images", tags=["Anomaly Dataset"])
async def list_dataset_images(
    project_id: str,
    label: Optional[str] = None,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    page = max(1, page)
    page_size = max(1, min(page_size, 200))

    items, total = await repo.list_import_items_page(
        project_id, label, skip=(page - 1) * page_size, limit=page_size,
    )

    proj_dir = dataset_fs.project_dir(project_id)
    out = []
    for it in items:
        abs_path = proj_dir / it.image_path
        out.append({
            "id": it.id,
            "inspection_id": it.inspection_id,
            "camera_serial": it.camera_serial,
            "frame_idx": it.frame_idx,
            "recipe_name": it.recipe_name,
            "label": it.label,
            "defect_type": it.defect_type,
            "split": it.split,
            "created_at": it.created_at.isoformat(),
            "thumb_b64": _thumb_b64(abs_path),
        })
    return {
        "images": out,
        "count": len(out),
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, -(-total // page_size)),  # ceil div
    }


@router.get("/projects/{project_id}/dataset/images/{item_id}/full", tags=["Anomaly Dataset"])
async def get_dataset_image_full(
    project_id: str,
    item_id: str,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    item = await repo.get_import_item(item_id)
    if not item or item.project_id != project_id:
        raise HTTPException(404, "Image not found")

    abs_path = dataset_fs.project_dir(project_id) / item.image_path
    if not abs_path.exists():
        raise HTTPException(404, "Image file missing on disk")

    return {"id": item_id, "full_b64": _full_b64(abs_path)}


@router.patch("/projects/{project_id}/dataset/images/{item_id}", tags=["Anomaly Dataset"])
async def relabel_dataset_image(
    project_id: str,
    item_id: str,
    request: AnomalyRelabelRequest,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    if request.label not in ("normal", "abnormal"):
        raise HTTPException(400, "label must be 'normal' or 'abnormal'")
    if request.label == "abnormal" and not request.defect_type:
        raise HTTPException(400, "defect_type is required for abnormal")

    item = await repo.get_import_item(item_id)
    if not item or item.project_id != project_id:
        raise HTTPException(404, "Image not found")

    proj_dir = dataset_fs.project_dir(project_id)
    update = _relabel_one_on_disk(proj_dir, item, request.label, request.defect_type)
    await repo.update_import_item(item_id, update)
    counts = await _recount(repo, project_id)

    return {"ok": True, "id": item_id, **update, **counts}


@router.delete("/projects/{project_id}/dataset/images/{item_id}", tags=["Anomaly Dataset"])
async def delete_dataset_image(
    project_id: str,
    item_id: str,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    item = await repo.get_import_item(item_id)
    if not item or item.project_id != project_id:
        raise HTTPException(404, "Image not found")

    proj_dir = dataset_fs.project_dir(project_id)
    abs_path = proj_dir / item.image_path
    if abs_path.exists():
        abs_path.unlink()
        _cleanup_if_empty_defect_dir(abs_path.parent)

    await repo.delete_import_item(item_id)
    counts = await _recount(repo, project_id)

    return {"ok": True, "id": item_id, **counts}


@router.post("/projects/{project_id}/dataset/images/bulk-relabel", tags=["Anomaly Dataset"])
async def bulk_relabel_dataset_images(
    project_id: str,
    request: AnomalyBulkRelabelRequest,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    if request.label not in ("normal", "abnormal"):
        raise HTTPException(400, "label must be 'normal' or 'abnormal'")
    if request.label == "abnormal" and not request.defect_type:
        raise HTTPException(400, "defect_type is required for abnormal")
    if not request.ids:
        raise HTTPException(400, "No ids provided")

    proj_dir = dataset_fs.project_dir(project_id)
    items = await repo.get_import_items(project_id, request.ids)
    found_ids = {it.id for it in items}

    updated = 0
    errors: List[Dict[str, str]] = []
    for it in items:
        try:
            update = _relabel_one_on_disk(proj_dir, it, request.label, request.defect_type)
            await repo.update_import_item(it.id, update)
            updated += 1
        except HTTPException as e:
            errors.append({"id": it.id, "reason": str(e.detail)})
    for missing_id in set(request.ids) - found_ids:
        errors.append({"id": missing_id, "reason": "not found"})

    counts = await _recount(repo, project_id)
    return {"updated": updated, "errors": errors, **counts}


@router.post("/projects/{project_id}/dataset/images/bulk-delete", tags=["Anomaly Dataset"])
async def bulk_delete_dataset_images(
    project_id: str,
    request: AnomalyBulkDeleteRequest,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    if not request.ids:
        raise HTTPException(400, "No ids provided")

    proj_dir = dataset_fs.project_dir(project_id)
    items = await repo.get_import_items(project_id, request.ids)
    found_ids = {it.id for it in items}

    deleted = 0
    for it in items:
        abs_path = proj_dir / it.image_path
        if abs_path.exists():
            abs_path.unlink()
            _cleanup_if_empty_defect_dir(abs_path.parent)
        await repo.delete_import_item(it.id)
        deleted += 1

    errors = [{"id": i, "reason": "not found"} for i in set(request.ids) - found_ids]
    counts = await _recount(repo, project_id)
    return {"deleted": deleted, "errors": errors, **counts}
