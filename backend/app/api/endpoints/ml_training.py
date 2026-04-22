"""
ML Training API Endpoints
CRUD for projects, image management, segmentation, labeling, training, prediction.
"""
import asyncio
import base64
import logging
import shutil
import uuid
from pathlib import Path
from typing import List, Optional

import cv2 as cv
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.api.dependencies.auth import get_current_user
from app.db.mongodb import get_database
from app.models.ml_training import (
    MLAnnotationSave,
    MLProjectCreate,
    MLProjectUpdate,
    TrainRequest,
)
from app.models.user import UserInDB
from app.repositories.ml_training_repository import MLTrainingRepository
from app.services.ml_segment_service import segment_region
from app.services.ml_training_service import (
    get_labeled_crops,
    img_to_b64,
    predict_on_image,
    train_model,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Paths ──────────────────────────────────────────────────────────────────
# Navigate from this file up to project root: endpoints→api→app→backend→root
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
ML_BASE = _PROJECT_ROOT / "public" / "ml_projects"
PUBLIC_IMAGES_DIR = _PROJECT_ROOT / "public" / "images"

ML_BASE.mkdir(parents=True, exist_ok=True)


def _project_dir(project_id: str) -> Path:
    return ML_BASE / project_id


def _images_dir(project_id: str) -> Path:
    return ML_BASE / project_id / "images"


def _models_dir(project_id: str) -> Path:
    return ML_BASE / project_id / "models"


# ── Dependency ─────────────────────────────────────────────────────────────

def get_repo(db=Depends(get_database)) -> MLTrainingRepository:
    return MLTrainingRepository(db)


# ── Helpers ────────────────────────────────────────────────────────────────

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTS


def _sync_image_count(repo: MLTrainingRepository, project_id: str):
    """Helper: schedule image count refresh (to be called in background)."""
    images_dir = _images_dir(project_id)
    count = len([f for f in images_dir.glob("*") if f.suffix.lower() in ALLOWED_EXTS]) if images_dir.exists() else 0
    return count


def _img_thumbnail_b64(path: Path, size: int = 120) -> str:
    img = cv.imread(str(path))
    if img is None:
        return ""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    img = cv.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
    return img_to_b64(img, quality=70)


# ════════════════════════════════════════ PROJECTS ════════════════════════

@router.post("/ml/projects", tags=["ML Training"])
async def create_project(
    data: MLProjectCreate,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    project = await repo.create_project(data, str(current_user.id))
    _images_dir(project.id).mkdir(parents=True, exist_ok=True)
    _models_dir(project.id).mkdir(parents=True, exist_ok=True)
    return project.model_dump(by_alias=False)


@router.get("/ml/projects", tags=["ML Training"])
async def list_projects(
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    projects = await repo.list_projects()
    return [p.model_dump(by_alias=False) for p in projects]


@router.get("/ml/projects/{project_id}", tags=["ML Training"])
async def get_project(
    project_id: str,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    return project.model_dump(by_alias=False)


@router.patch("/ml/projects/{project_id}", tags=["ML Training"])
async def update_project(
    project_id: str,
    data: MLProjectUpdate,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    project = await repo.update_project(project_id, data)
    if not project:
        raise HTTPException(404, "Project not found")
    return project.model_dump(by_alias=False)


@router.delete("/ml/projects/{project_id}", tags=["ML Training"])
async def delete_project(
    project_id: str,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    # Remove files on disk
    project_dir = _project_dir(project_id)
    if project_dir.exists():
        shutil.rmtree(project_dir)
    deleted = await repo.delete_project(project_id)
    if not deleted:
        raise HTTPException(404, "Project not found")
    return {"ok": True}


# ════════════════════════════════════════ AVAILABLE IMAGES ════════════════

@router.get("/ml/available-images", tags=["ML Training"])
async def list_available_images(
    current_user: UserInDB = Depends(get_current_user),
):
    """List images in /public/images (the rolling 100-image buffer from cameras)."""
    if not PUBLIC_IMAGES_DIR.exists():
        return []
    files = sorted(
        [f for f in PUBLIC_IMAGES_DIR.glob("*") if f.suffix.lower() in ALLOWED_EXTS],
        key=lambda f: f.name,
        reverse=True,
    )
    result = []
    for f in files:
        result.append({
            "filename": f.name,
            "thumbnail_b64": _img_thumbnail_b64(f),
        })
    return result


# ════════════════════════════════════════ PROJECT IMAGES ══════════════════

@router.get("/ml/projects/{project_id}/images", tags=["ML Training"])
async def list_project_images(
    project_id: str,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    images_dir = _images_dir(project_id)
    if not images_dir.exists():
        return []

    files = sorted([f for f in images_dir.glob("*") if f.suffix.lower() in ALLOWED_EXTS])
    # Check which images have annotations
    annotations = await repo.list_annotations(project_id)
    annotated = {a.filename for a in annotations if any(
        seg.label in ("OK", "NG")
        for r in a.regions for seg in r.segments
    )}

    result = []
    for f in files:
        result.append({
            "filename": f.name,
            "thumbnail_b64": _img_thumbnail_b64(f),
            "has_annotation": f.name in annotated,
        })
    return result


@router.post("/ml/projects/{project_id}/images/copy", tags=["ML Training"])
async def copy_images_to_project(
    project_id: str,
    body: dict,
    background_tasks: BackgroundTasks,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    """Copy selected filenames from /public/images into the project folder."""
    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    filenames: List[str] = body.get("filenames", [])
    if not filenames:
        raise HTTPException(400, "No filenames provided")

    images_dir = _images_dir(project_id)
    images_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for fn in filenames:
        src = PUBLIC_IMAGES_DIR / fn
        if not src.exists() or not _allowed(fn):
            continue
        dst = images_dir / fn
        shutil.copy2(src, dst)
        copied.append(fn)

    # Refresh image count
    count = _sync_image_count(repo, project_id)
    await repo.set_image_count(project_id, count)

    return {"copied": len(copied), "filenames": copied}


@router.post("/ml/projects/{project_id}/images/upload", tags=["ML Training"])
async def upload_images_to_project(
    project_id: str,
    files: List[UploadFile] = File(...),
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    images_dir = _images_dir(project_id)
    images_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for file in files:
        if not _allowed(file.filename or ""):
            continue
        # Use original name, avoid overwrite collision
        stem = Path(file.filename).stem
        suffix = Path(file.filename).suffix.lower()
        dst = images_dir / f"{stem}{suffix}"
        if dst.exists():
            dst = images_dir / f"{stem}_{uuid.uuid4().hex[:6]}{suffix}"
        content = await file.read()
        dst.write_bytes(content)
        saved.append(dst.name)

    count = _sync_image_count(repo, project_id)
    await repo.set_image_count(project_id, count)

    return {"saved": len(saved), "filenames": saved}


@router.delete("/ml/projects/{project_id}/images/{filename}", tags=["ML Training"])
async def delete_project_image(
    project_id: str,
    filename: str,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    path = _images_dir(project_id) / filename
    if not path.exists():
        raise HTTPException(404, "Image not found")
    path.unlink()
    count = _sync_image_count(repo, project_id)
    await repo.set_image_count(project_id, count)
    return {"ok": True}


@router.get("/ml/projects/{project_id}/images/{filename}", tags=["ML Training"])
async def get_project_image_b64(
    project_id: str,
    filename: str,
    current_user: UserInDB = Depends(get_current_user),
):
    """Return a project image as base64 (for canvas display)."""
    path = _images_dir(project_id) / filename
    if not path.exists():
        raise HTTPException(404, "Image not found")
    img = cv.imread(str(path))
    if img is None:
        raise HTTPException(500, "Cannot read image")
    return {
        "filename": filename,
        "image_b64": img_to_b64(img, quality=90),
        "width": img.shape[1],
        "height": img.shape[0],
    }


# ════════════════════════════════════════ SEGMENTATION ════════════════════

@router.post("/ml/projects/{project_id}/segment", tags=["ML Training"])
async def segment_image_region(
    project_id: str,
    body: dict,
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Auto-segment characters inside a user-drawn region.
    Body: {filename, region: {x, y, w, h}}  (all coords normalized 0-1)
    Returns: [{id, x, y, w, h}, ...]
    """
    filename = body.get("filename")
    region = body.get("region")
    if not filename or not region:
        raise HTTPException(400, "filename and region are required")

    image_path = _images_dir(project_id) / filename
    if not image_path.exists():
        raise HTTPException(404, "Image not found in project")

    try:
        segments = await asyncio.get_event_loop().run_in_executor(
            None, segment_region, image_path, region
        )
    except Exception as e:
        logger.exception("Segmentation failed")
        raise HTTPException(500, f"Segmentation error: {e}")

    return {"segments": segments, "count": len(segments)}


# ════════════════════════════════════════ ANNOTATIONS ════════════════════

@router.get("/ml/projects/{project_id}/annotations/{filename}", tags=["ML Training"])
async def get_annotation(
    project_id: str,
    filename: str,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    ann = await repo.get_annotation(project_id, filename)
    if not ann:
        return {"project_id": project_id, "filename": filename, "regions": []}
    return ann.model_dump(by_alias=False)


@router.put("/ml/projects/{project_id}/annotations/{filename}", tags=["ML Training"])
async def save_annotation(
    project_id: str,
    filename: str,
    data: MLAnnotationSave,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    ann = await repo.save_annotation(project_id, filename, data)
    await repo.refresh_labeled_count(project_id)
    return ann.model_dump(by_alias=False)


# ════════════════════════════════════════ LABELED CROPS (Train tab) ═══════

@router.get("/ml/projects/{project_id}/labeled-crops", tags=["ML Training"])
async def get_labeled_crops_endpoint(
    project_id: str,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    """Return all labeled character crops for the Train tab preview."""
    annotations = await repo.list_annotations(project_id)
    images_dir = _images_dir(project_id)

    crops = await asyncio.get_event_loop().run_in_executor(
        None, get_labeled_crops, annotations, images_dir
    )
    return {"crops": crops, "count": len(crops)}


# ════════════════════════════════════════ TRAINING ════════════════════════

@router.post("/ml/projects/{project_id}/train", tags=["ML Training"])
async def start_training(
    project_id: str,
    request: TrainRequest,
    background_tasks: BackgroundTasks,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    annotations = await repo.list_annotations(project_id)
    if not annotations:
        raise HTTPException(400, "No annotations found. Please label images first.")

    # Create model record
    model_record = await repo.create_model_record(
        project_id=project_id,
        algorithm=request.algorithm,
        params=request.model_dump(),
        augment_factor=request.augment_factor,
    )
    model_id = model_record.id

    await repo.set_status(project_id, "training")
    await repo.update_model_record(model_id, {"status": "training"})

    # Launch training in background
    background_tasks.add_task(
        _run_training_bg, repo, project_id, model_id,
        annotations, request,
    )

    return {"model_id": model_id, "status": "training"}


async def _run_training_bg(
    repo: MLTrainingRepository,
    project_id: str,
    model_id: str,
    annotations,
    request: TrainRequest,
):
    images_dir = _images_dir(project_id)
    model_path = _models_dir(project_id) / f"{model_id}.joblib"

    try:
        loop = asyncio.get_event_loop()
        metrics = await loop.run_in_executor(
            None,
            train_model,
            annotations,
            images_dir,
            request,
            model_path,
        )
        await repo.update_model_record(model_id, {
            "status": "completed",
            "metrics": metrics,
            "model_path": str(model_path),
        })
        await repo.set_status(project_id, "trained")
        logger.info(f"[ML] Training completed for project {project_id}, model {model_id}")

    except Exception as e:
        logger.exception(f"[ML] Training failed for project {project_id}")
        await repo.update_model_record(model_id, {
            "status": "failed",
            "error": str(e),
        })
        await repo.set_status(project_id, "active")


@router.get("/ml/projects/{project_id}/models", tags=["ML Training"])
async def list_models(
    project_id: str,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    models = await repo.list_models(project_id)
    return [m.model_dump(by_alias=False) for m in models]


@router.get("/ml/projects/{project_id}/models/{model_id}/status", tags=["ML Training"])
async def get_model_status(
    project_id: str,
    model_id: str,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    models = await repo.list_models(project_id)
    for m in models:
        if m.id == model_id:
            return m.model_dump(by_alias=False)
    raise HTTPException(404, "Model not found")


# ════════════════════════════════════════ PREDICTION ═════════════════════

@router.post("/ml/projects/{project_id}/predict", tags=["ML Training"])
async def predict(
    project_id: str,
    file: UploadFile = File(...),
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    """Upload an image and predict OK/NG for each character."""
    model_record = await repo.get_latest_model(project_id)
    if not model_record:
        raise HTTPException(400, "No trained model found. Please train first.")

    model_path = Path(model_record.model_path)
    if not model_path.exists():
        raise HTTPException(500, "Model file not found on disk")

    # Save uploaded file temporarily
    tmp_path = _models_dir(project_id) / f"_predict_{uuid.uuid4().hex}.jpg"
    try:
        content = await file.read()
        tmp_path.write_bytes(content)

        results = await asyncio.get_event_loop().run_in_executor(
            None, predict_on_image, model_path, tmp_path, None, 0.5,
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return {
        "model_id": model_record.id,
        "algorithm": model_record.algorithm,
        "results": results,
    }
