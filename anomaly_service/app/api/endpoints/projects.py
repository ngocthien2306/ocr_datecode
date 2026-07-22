"""
Anomaly Projects — CRUD.
Mirrors the shape of backend's /api/ml/projects endpoints.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies.auth import CurrentUser, get_current_user
from app.db.mongodb import get_database
from app.models.anomaly import AnomalyProjectCreate, AnomalyProjectUpdate
from app.repositories.anomaly_repository import AnomalyRepository
from app.services import dataset_fs

logger = logging.getLogger(__name__)

router = APIRouter()


def get_repo(db=Depends(get_database)) -> AnomalyRepository:
    return AnomalyRepository(db)


@router.post("/projects", tags=["Anomaly Projects"])
async def create_project(
    data: AnomalyProjectCreate,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    project = await repo.create_project(data, current_user.username)
    dataset_fs.ensure_project_dirs(project.id)
    return project.model_dump(by_alias=False)


@router.get("/projects", tags=["Anomaly Projects"])
async def list_projects(
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    projects = await repo.list_projects()
    return [p.model_dump(by_alias=False) for p in projects]


@router.get("/projects/{project_id}", tags=["Anomaly Projects"])
async def get_project(
    project_id: str,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    return project.model_dump(by_alias=False)


@router.patch("/projects/{project_id}", tags=["Anomaly Projects"])
async def update_project(
    project_id: str,
    data: AnomalyProjectUpdate,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    project = await repo.update_project(project_id, data)
    if not project:
        raise HTTPException(404, "Project not found")
    return project.model_dump(by_alias=False)


@router.delete("/projects/{project_id}", tags=["Anomaly Projects"])
async def delete_project(
    project_id: str,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    dataset_fs.delete_project_dir(project_id)
    deleted = await repo.delete_project(project_id)
    if not deleted:
        raise HTTPException(404, "Project not found")
    return {"ok": True}


@router.get("/projects/{project_id}/dataset-stats", tags=["Anomaly Projects"])
async def dataset_stats(
    project_id: str,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Recount images on disk and sync onto the project doc. Cheap enough to
    call on every page load — no need for a separate 'refresh' action."""
    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    counts = dataset_fs.count_images(project_id)
    await repo.set_counts(project_id, counts["normal"], counts["abnormal"])
    return {
        "normal_count": counts["normal"],
        "abnormal_count": counts["abnormal"],
        "defect_types": dataset_fs.list_defect_types(project_id),
    }
