"""
OCR Projects — CRUD + dataset stats.
Mirrors anomaly_service/app/api/endpoints/projects.py.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies.auth import CurrentUser, get_current_user
from app.db.mongodb import get_database
from app.models.ocr import OCRProjectCreate, OCRProjectUpdate
from app.repositories.ocr_repository import OCRRepository
from app.services import dataset_fs

logger = logging.getLogger(__name__)

router = APIRouter()


def get_repo(db=Depends(get_database)) -> OCRRepository:
    return OCRRepository(db)


@router.post("/projects")
async def create_project(
    data: OCRProjectCreate,
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    project = await repo.create_project(data, current_user.username)
    dataset_fs.ensure_project_dirs(project.id)
    return project.model_dump(by_alias=False)


@router.get("/projects")
async def list_projects(
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    projects = await repo.list_projects()
    return [p.model_dump(by_alias=False) for p in projects]


@router.get("/projects/{project_id}")
async def get_project(
    project_id: str,
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    return project.model_dump(by_alias=False)


@router.patch("/projects/{project_id}")
async def update_project(
    project_id: str,
    data: OCRProjectUpdate,
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    project = await repo.update_project(project_id, data)
    if not project:
        raise HTTPException(404, "Project not found")
    return project.model_dump(by_alias=False)


@router.delete("/projects/{project_id}")
async def delete_project(
    project_id: str,
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    # Filesystem first: if the Mongo delete fails the dataset is gone but the
    # project row survives, which is recoverable by hand. The reverse leaves
    # orphaned gigabytes with nothing pointing at them.
    dataset_fs.delete_project_dir(project_id)
    deleted = await repo.delete_project(project_id)
    if not deleted:
        raise HTTPException(404, "Project not found")
    return {"ok": True}


@router.get("/projects/{project_id}/dataset-stats")
async def dataset_stats(
    project_id: str,
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Recount dataset items and sync onto the project doc.

    Counts come from Mongo, not from listing files: an image whose label is
    still `need_review` is on disk but not trainable, so a file count would
    overstate what a run will see. `trainable_count` is the number the Train
    tab gates on.
    """
    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    counts = await repo.sync_project_counts(project_id)
    return {
        "total_count": counts["total"],
        "verified_count": counts["verified"],
        "need_review_count": counts["need_review"],
        "rejected_count": counts["rejected"],
        "trainable_count": counts["verified"],
    }
