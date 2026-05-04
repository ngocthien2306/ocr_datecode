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
from typing import Any, Dict, List, Optional

import cv2 as cv
import joblib
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.api.dependencies.auth import get_current_user
from app.db.mongodb import get_database
from app.models.ml_training import (
    AnnotationRegion,
    CharSegment,
    ImportFromRecipeRequest,
    ImportFromRecipeResponse,
    CharCoverageResponse,
    MLAnnotationSave,
    MLProjectCreate,
    MLProjectUpdate,
    SyntheticPreviewRequest,
    TrainRequest,
)
from app.models.user import UserInDB
from app.repositories.ml_training_repository import MLTrainingRepository
from app.repositories.recipe_repository import RecipeRepository
from app.services.ml_char_ocr_service import recognize_chars
from app.services.ml_segment_service import crop_segment, segment_region
from app.services.ml_training_service import (
    generate_synthetic_crops,
    get_labeled_crops,
    get_model_chars,
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
IMAGES_TEMP_DIR = _PROJECT_ROOT / "public" / "images_temp"

ML_BASE.mkdir(parents=True, exist_ok=True)
IMAGES_TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Static URL prefixes (served by main.py mounts)
_CAMERA_STATIC_PREFIX = "/api/camera-images"
_ML_FILES_STATIC_PREFIX = "/api/ml-files"


def _project_dir(project_id: str) -> Path:
    return ML_BASE / project_id


def _images_dir(project_id: str) -> Path:
    return ML_BASE / project_id / "images"


def _models_dir(project_id: str) -> Path:
    return ML_BASE / project_id / "models"


# ── Dependency ─────────────────────────────────────────────────────────────

def get_repo(db=Depends(get_database)) -> MLTrainingRepository:
    return MLTrainingRepository(db)


def get_recipe_repo(db=Depends(get_database)) -> RecipeRepository:
    return RecipeRepository(db)


# ── Helpers ────────────────────────────────────────────────────────────────

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTS


def _sync_image_count(repo: MLTrainingRepository, project_id: str):
    images_dir = _images_dir(project_id)
    count = len([f for f in images_dir.glob("*") if f.suffix.lower() in ALLOWED_EXTS]) if images_dir.exists() else 0
    return count


def _image_dimensions(path: Path) -> tuple[int, int]:
    """Return (width, height) of an image file without encoding it."""
    img = cv.imread(str(path))
    if img is None:
        return 0, 0
    return img.shape[1], img.shape[0]


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


# ════════════════════════════════════════ SNAPSHOT ════════════════════════

@router.post("/ml/snapshot-images", tags=["ML Training"])
async def snapshot_images(
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Copy /public/images → /public/images_temp so the ML session has a stable
    snapshot that won't be affected by the rolling 100-image buffer updates.
    """
    if IMAGES_TEMP_DIR.exists():
        shutil.rmtree(IMAGES_TEMP_DIR)
    IMAGES_TEMP_DIR.mkdir(parents=True, exist_ok=True)

    if not PUBLIC_IMAGES_DIR.exists():
        return {"copied": 0, "filenames": []}

    copied = []
    for f in PUBLIC_IMAGES_DIR.glob("*"):
        if f.suffix.lower() in ALLOWED_EXTS:
            shutil.copy2(f, IMAGES_TEMP_DIR / f.name)
            copied.append(f.name)

    logger.info(f"[ML] Snapshot: copied {len(copied)} images to images_temp")
    return {"copied": len(copied), "filenames": copied}


# ════════════════════════════════════════ AVAILABLE IMAGES ════════════════

@router.get("/ml/available-images", tags=["ML Training"])
async def list_available_images(
    current_user: UserInDB = Depends(get_current_user),
):
    """List images from the stable snapshot in /public/images_temp."""
    if not IMAGES_TEMP_DIR.exists():
        return []
    files = sorted(
        [f for f in IMAGES_TEMP_DIR.glob("*") if f.suffix.lower() in ALLOWED_EXTS],
        key=lambda f: f.name,
        reverse=True,
    )
    return [
        {"filename": f.name, "url": f"{_CAMERA_STATIC_PREFIX}/{f.name}"}
        for f in files
    ]


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

    return [
        {
            "filename": f.name,
            "url": f"{_ML_FILES_STATIC_PREFIX}/{project_id}/images/{f.name}",
            "has_annotation": f.name in annotated,
        }
        for f in files
    ]


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
        # Read from the stable snapshot, not the rolling buffer
        src = IMAGES_TEMP_DIR / fn
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


@router.get("/ml/projects/{project_id}/images/{filename}/meta", tags=["ML Training"])
async def get_project_image_meta(
    project_id: str,
    filename: str,
    current_user: UserInDB = Depends(get_current_user),
):
    """Return image URL + natural dimensions (for canvas scaling in Label tab)."""
    path = _images_dir(project_id) / filename
    if not path.exists():
        raise HTTPException(404, "Image not found")
    w, h = _image_dimensions(path)
    if w == 0:
        raise HTTPException(500, "Cannot read image")
    return {
        "filename": filename,
        "url": f"{_ML_FILES_STATIC_PREFIX}/{project_id}/images/{filename}",
        "width": w,
        "height": h,
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

    # Auto-OCR each segment to pre-populate char_id. Batch the ONNX call.
    # User can still edit char_id manually in Label tab afterwards.
    if segments:
        def _ocr_all():
            crops = [crop_segment(image_path, s) for s in segments]
            return recognize_chars([c for c in crops if c is not None]), crops

        try:
            char_ids, crops = await asyncio.get_event_loop().run_in_executor(None, _ocr_all)
            # char_ids is parallel to non-None crops; re-align back to segments
            it = iter(char_ids)
            for seg, crop in zip(segments, crops):
                seg["char_id"] = next(it) if crop is not None else ""
        except Exception as e:
            logger.warning(f"[segment] OCR pass failed, leaving char_id empty: {e}")
            for seg in segments:
                seg.setdefault("char_id", "")

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


@router.post("/ml/projects/{project_id}/preview-synthetic", tags=["ML Training"])
async def preview_synthetic_endpoint(
    project_id: str,
    request: SyntheticPreviewRequest,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Return synthetic augmented crops generated from OK samples (no training).

    Request `label` controls which variants to return:
      - 'NG'   → destructive augmentations (default)
      - 'OK'   → mild augmentations (for balanced training preview)
      - 'BOTH' → both OK and NG variants
    """
    annotations = await repo.list_annotations(project_id)
    images_dir = _images_dir(project_id)

    crops = await asyncio.get_event_loop().run_in_executor(
        None, generate_synthetic_crops,
        annotations, images_dir, request.augment_factor, request.label,
        request.severity_dist,
    )
    return {"crops": crops, "count": len(crops)}


# ════════════════════════════════════════ SYNTHETIC OK PREVIEW ═════════════

class _SynthOKRequest(BaseModel):
    target_n_per_char: int = 30
    only_below_threshold: bool = True
    char_filter: Optional[List[str]] = None
    rotation_max_deg: float = 5.0
    size_jitter: float = 0.30


@router.post("/ml/projects/{project_id}/preview-synthetic-ok", tags=["ML Training"])
async def preview_synthetic_ok_endpoint(
    project_id: str,
    request: _SynthOKRequest,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Generate synthetic OK crops via PIL render → composite on real BG → camera
    noise. Used to top-up under-represented chars in the training set.

    Each result item: {crop_b64, char_id, font_name, rotation_deg, source}.
    """
    from app.services.ml_ok_synthesize import synthesize_ok_from_annotations

    annotations = await repo.list_annotations(project_id)
    if not annotations:
        raise HTTPException(400, "No annotations — label some OK chars first")
    images_dir = _images_dir(project_id)

    def _build():
        synth = synthesize_ok_from_annotations(
            annotations, images_dir,
            target_n_per_char=request.target_n_per_char,
            only_below_threshold=request.only_below_threshold,
            char_filter=request.char_filter,
            rotation_max_deg=request.rotation_max_deg,
            size_jitter=request.size_jitter,
        )
        # Encode crops → b64 for transport (drop raw ndarray)
        return [{
            'crop_b64':     img_to_b64(item['crop']),
            'char_id':      item['char_id'],
            'font_name':    item['font_name'],
            'rotation_deg': item['rotation_deg'],
            'source':       item['source'],
        } for item in synth]

    try:
        crops = await asyncio.get_event_loop().run_in_executor(None, _build)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))
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
    model_id: Optional[str] = Form(None),
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    """Upload an image and predict OK/NG for each character."""
    if model_id:
        all_models = await repo.list_models(project_id)
        model_record = next((m for m in all_models if m.id == model_id), None)
        if not model_record:
            raise HTTPException(404, "Model not found")
    else:
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

        threshold = float(model_record.params.get("threshold", 0.5))
        results = await asyncio.get_event_loop().run_in_executor(
            None, predict_on_image, model_path, tmp_path, None, threshold,
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return {
        "model_id": model_record.id,
        "algorithm": model_record.algorithm,
        "results": results,
    }


@router.get("/ml/projects/{project_id}/models/{model_id}/test-set-crops", tags=["ML Training"])
async def get_test_set_crops(
    project_id: str,
    model_id: str,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Return the test-split crops saved during training, with true & predicted labels.
    Each item: { crop_b64, true_label, pred_label, prob_ok, correct }
    """
    import json as _json
    all_models = await repo.list_models(project_id)
    model_record = next((m for m in all_models if m.id == model_id), None)
    if not model_record:
        raise HTTPException(404, "Model not found")

    model_path = Path(model_record.model_path)
    test_set_path = model_path.parent / f"{model_path.stem}_test_set.json"
    if not test_set_path.exists():
        return {"crops": [], "count": 0}

    items = _json.loads(test_set_path.read_text())
    return {"crops": items, "count": len(items)}


@router.get(
    "/ml/projects/{project_id}/models/{model_id}/report",
    tags=["ML Training"],
)
async def download_model_report(
    project_id: str,
    model_id: str,
    include_testset: bool = True,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Full training report as a single JSON — metrics, per-char stats, and
    test-set crops with embedded base64 images. Intended for offline
    analysis / sharing.
    """
    import json as _json

    all_models = await repo.list_models(project_id)
    model_record = next((m for m in all_models if m.id == model_id), None)
    if not model_record:
        raise HTTPException(404, "Model not found")

    def _build() -> dict:
        report = {
            "project_id": project_id,
            "model_id": model_id,
            "algorithm": model_record.algorithm,
            "created_at": model_record.created_at.isoformat() if model_record.created_at else None,
            "status": model_record.status,
            "augment_factor": model_record.augment_factor,
            "params": model_record.params,
            "metrics": model_record.metrics.model_dump() if hasattr(model_record.metrics, "model_dump") else dict(model_record.metrics or {}),
        }

        if model_record.model_path:
            try:
                data = joblib.load(str(model_record.model_path))
                if isinstance(data, dict):
                    report["char_stats"] = data.get("char_stats") or {}
                else:
                    report["char_stats"] = {}
            except Exception as e:
                logger.warning(f"[report] bundle load failed: {e}")
                report["char_stats"] = {}

        if include_testset and model_record.model_path:
            ts_path = Path(model_record.model_path).parent / f"{Path(model_record.model_path).stem}_test_set.json"
            if ts_path.exists():
                try:
                    report["test_set"] = _json.loads(ts_path.read_text())
                except Exception as e:
                    logger.warning(f"[report] test_set read failed: {e}")
                    report["test_set"] = []
            else:
                report["test_set"] = []

        # Summarize per-char accuracy from test_set (handy for quick glance)
        if include_testset and report.get("test_set"):
            from collections import defaultdict
            buckets = defaultdict(lambda: {"n": 0, "correct": 0, "wrong": 0})
            for item in report["test_set"]:
                key = item.get("char_id") or "__null__"
                b = buckets[key]
                b["n"] += 1
                if item.get("correct"):
                    b["correct"] += 1
                else:
                    b["wrong"] += 1
            per_char = []
            for k, v in buckets.items():
                per_char.append({
                    "char_id": None if k == "__null__" else k,
                    "n": v["n"],
                    "correct": v["correct"],
                    "wrong": v["wrong"],
                    "accuracy": round(v["correct"] / v["n"], 4) if v["n"] else 0.0,
                })
            per_char.sort(key=lambda r: (-r["wrong"], r["char_id"] or "~"))
            report["per_char_accuracy"] = per_char

        return report

    report = await asyncio.get_event_loop().run_in_executor(None, _build)

    filename = f"model_{model_id}_report.json"
    return JSONResponse(
        content=report,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


# ════════════════════════════════════════ IMPORT FROM RECIPE ══════════════

def _bbox_to_normalized_xywh(ann, image_width: int, image_height: int):
    """
    Convert a recipe TemplateAnnotation (either rect or polygon) to
    normalized (x, y, w, h) in [0, 1] coords.
    Returns None if annotation has neither rect nor polygon points.
    """
    if ann.points and len(ann.points) >= 4:
        xs = [float(p[0]) for p in ann.points]
        ys = [float(p[1]) for p in ann.points]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
    elif (ann.x is not None and ann.y is not None
          and ann.width is not None and ann.height is not None):
        x0, y0 = float(ann.x), float(ann.y)
        x1, y1 = x0 + float(ann.width), y0 + float(ann.height)
    else:
        return None

    # Normalize to 0..1 (clamped)
    nx = max(0.0, min(1.0, x0 / image_width))
    ny = max(0.0, min(1.0, y0 / image_height))
    nw = max(0.0, min(1.0 - nx, (x1 - x0) / image_width))
    nh = max(0.0, min(1.0 - ny, (y1 - y0) / image_height))
    if nw <= 0 or nh <= 0:
        return None
    return nx, ny, nw, nh


@router.post(
    "/ml/projects/{project_id}/import-from-recipe",
    response_model=ImportFromRecipeResponse,
    tags=["ML Training"],
)
async def import_from_recipe(
    project_id: str,
    request: ImportFromRecipeRequest,
    repo: MLTrainingRepository = Depends(get_repo),
    recipe_repo: RecipeRepository = Depends(get_recipe_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Pre-populate ML project annotations from recipe template text/datecode
    bboxes.

    For each selected file: drop a REGION (no segments) for every text /
    datecode bbox found in the recipe's camera template. The user opens the
    file in the Label tab afterwards and clicks "Segment" — the segment
    endpoint runs OCR per char and fills char_id one character at a time.

    We don't pre-create segments here because a single annotation's `text`
    is the whole string ("LOT PL26050423") — using that as a per-segment
    `char_id` would store a multi-char string where a single character is
    expected.

    Partial-success semantics: files missing on disk or producing no valid
    region are skipped with a per-file error; other files proceed.
    """
    # Verify project exists
    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    # Load recipe
    recipe = await recipe_repo.get_by_id(request.recipe_id)
    if not recipe:
        raise HTTPException(404, f"Recipe {request.recipe_id} not found")

    # Find the CameraConfiguration matching the requested serial
    camera_cfg = next(
        (c for c in (recipe.cameras or []) if c.serial_number == request.camera_serial),
        None,
    )
    if not camera_cfg:
        raise HTTPException(
            400,
            f"Recipe does not reference camera {request.camera_serial}",
        )

    # Find the CameraTemplates for this camera_id
    cam_templates = next(
        (ct for ct in (recipe.camera_templates or []) if ct.camera_id == camera_cfg.camera_id),
        None,
    )
    if not cam_templates or not cam_templates.templates:
        raise HTTPException(
            400,
            f"Recipe has no templates for camera {request.camera_serial}",
        )

    imported = 0
    skipped = 0
    errors: List[dict] = []

    images_dir = _images_dir(project_id)

    for filename in request.filenames:
        try:
            image_path = images_dir / filename
            if not image_path.exists():
                raise FileNotFoundError(f"{filename} not present in project images")

            regions: List[AnnotationRegion] = []
            for tpl in cam_templates.templates:
                img_w = int(tpl.image_width) if tpl.image_width else 0
                img_h = int(tpl.image_height) if tpl.image_height else 0
                if img_w <= 0 or img_h <= 0:
                    continue

                for ann in tpl.annotations or []:
                    if ann.type not in ("text", "datecode"):
                        continue
                    box = _bbox_to_normalized_xywh(ann, img_w, img_h)
                    if box is None:
                        continue
                    nx, ny, nw, nh = box

                    # Empty `segments` — the user runs Segment in Label tab
                    # to detect chars one-by-one (with OCR-filled char_id).
                    regions.append(AnnotationRegion(
                        id=str(uuid.uuid4()),
                        x=nx, y=ny, w=nw, h=nh,
                        segments=[],
                    ))

            if not regions:
                raise ValueError("Recipe template has no text/datecode annotations")

            await repo.save_annotation(
                project_id, filename,
                MLAnnotationSave(regions=regions),
            )
            imported += 1

        except Exception as e:
            skipped += 1
            errors.append({"filename": filename, "reason": str(e)})
            logger.warning(f"[import-from-recipe] skip {filename}: {e}")

    # Refresh labeled count (still 0 since regions hold no labeled segments yet)
    await repo.refresh_labeled_count(project_id)

    logger.info(
        f"[import-from-recipe] project={project_id} recipe={request.recipe_id} "
        f"imported={imported} skipped={skipped} (regions only — segment in Label tab)"
    )

    return ImportFromRecipeResponse(
        imported=imported,
        skipped=skipped,
        errors=errors,
        char_ids=[],   # regions only; chars filled later by Label-tab segment
    )


# ════════════════════════════════════════ CHAR COVERAGE ═════════════════

@router.get(
    "/ml/projects/{project_id}/models/{model_id}/char-coverage",
    response_model=CharCoverageResponse,
    tags=["ML Training"],
)
async def char_coverage(
    project_id: str,
    model_id: str,
    chars: str,  # comma-separated: "A,B,C"
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Given a list of chars the recipe needs, return which are covered by
    the model's goldens and which are missing. Used by FE to warn when
    a recipe picks an ML model that doesn't have all its required chars.
    """
    all_models = await repo.list_models(project_id)
    model_record = next((m for m in all_models if m.id == model_id), None)
    if not model_record:
        raise HTTPException(404, "Model not found")
    if not model_record.model_path:
        raise HTTPException(400, "Model has no saved file")

    model_chars = set(get_model_chars(Path(model_record.model_path)))
    requested = [c.strip() for c in (chars or "").split(",") if c.strip()]

    covered = [c for c in requested if c in model_chars]
    missing = [c for c in requested if c not in model_chars]
    pct = round(len(covered) / len(requested) * 100, 1) if requested else 0.0

    return CharCoverageResponse(
        covered=covered,
        missing=missing,
        coverage_pct=pct,
        model_chars=sorted(model_chars),
    )


# ════════════════════════════════════════ IMPORT FROM INSPECTIONS ═════════
#
# Active learning loop: pull failed-prediction chars from production inference
# results back into the training set. Operator picks the wrong predictions in
# Label tab → labels them correctly → retrains.
#
# Source filter: char_verification.results from inference_results collection.
#   - hard_fail = True   → match=false (model said NG)
#   - borderline = True  → 0.03 ≤ ml_p_ok ≤ 0.3 (low-confidence OK)

_INSPECTION_UPLOADS = _PROJECT_ROOT / "backend" / "uploads"


def _resolve_inspection_image(image_path: str) -> Optional[Path]:
    """
    Inspection results store relative-ish paths. Resolve to an absolute path
    on disk under backend/uploads/. Returns None if file doesn't exist.
    """
    if not image_path:
        return None
    p = Path(image_path)
    if p.is_absolute() and p.exists():
        return p
    candidate = _INSPECTION_UPLOADS / image_path
    return candidate if candidate.exists() else None


def _crop_from_polygon(img, points: List[List[float]], padding: int = 4):
    """Crop the axis-aligned bbox enclosing a polygon (with padding)."""
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    h, w = img.shape[:2]
    x0 = max(0, int(min(xs)) - padding)
    y0 = max(0, int(min(ys)) - padding)
    x1 = min(w, int(max(xs)) + padding)
    y1 = min(h, int(max(ys)) + padding)
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1].copy()


@router.get(
    "/ml/projects/{project_id}/inspection-candidates",
    tags=["ML Training"],
)
async def get_inspection_candidates(
    project_id: str,
    recipe_id: Optional[str] = None,
    date_from: Optional[str] = None,        # ISO format
    date_to: Optional[str] = None,
    include_hard_fail: bool = True,
    include_borderline: bool = True,
    limit: int = 100,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
    db=Depends(get_database),
):
    """
    Return up to `limit` char-level training candidates pulled from past
    inspections that had ML predictions worth re-labeling.

    Each candidate = one char_verification result with bbox cropped from the
    saved frame image. Sorted newest first.
    """
    from datetime import datetime as _dt

    # Verify project exists
    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    if not (include_hard_fail or include_borderline):
        return {"candidates": [], "count": 0}

    # Build mongo query
    query: Dict[str, Any] = {}
    if recipe_id:
        query["recipe_id"] = recipe_id
    if date_from or date_to:
        ts: Dict[str, Any] = {}
        if date_from:
            ts["$gte"] = _dt.fromisoformat(date_from.replace("Z", "+00:00"))
        if date_to:
            ts["$lte"] = _dt.fromisoformat(date_to.replace("Z", "+00:00"))
        query["timestamp"] = ts
    # Only inspections that have at least one char_verification
    query["camera_results.frames.char_verification"] = {"$ne": None}

    coll = db.get_collection("inference_results")
    cursor = coll.find(query).sort("timestamp", -1).limit(500)  # over-fetch for filtering

    candidates: List[Dict[str, Any]] = []
    loop = asyncio.get_event_loop()

    async for doc in cursor:
        if len(candidates) >= limit:
            break
        recipe_name = doc.get("recipe_name", "")
        rid = str(doc.get("recipe_id", ""))
        ts = doc.get("timestamp")
        ts_iso = ts.isoformat() if ts else None
        inspection_id = str(doc.get("_id", ""))

        for cam in doc.get("camera_results", []):
            for frame in cam.get("frames", []):
                cv_data = frame.get("char_verification") or {}
                results = cv_data.get("results") or []
                if not results:
                    continue

                detected_regions = frame.get("detected_regions") or []
                # Index detected_regions by annotation_index for quick lookup
                region_by_idx = {
                    r.get("annotation_index"): r
                    for r in detected_regions
                    if r.get("type") == "char" and r.get("annotation_index") is not None
                }

                image_path_raw = frame.get("image_path")
                image_path = _resolve_inspection_image(image_path_raw or "")
                if image_path is None:
                    continue

                # Lazy-load image once per frame
                _img_cache = {"img": None}

                def _get_img():
                    if _img_cache["img"] is None:
                        _img_cache["img"] = cv.imread(str(image_path))
                    return _img_cache["img"]

                for r in results:
                    if len(candidates) >= limit:
                        break
                    is_hard_fail = (r.get("match") is False)
                    p_ok = float(r.get("ml_p_ok") or 0.0)
                    is_borderline = (0.03 <= p_ok <= 0.3) and not is_hard_fail
                    if is_hard_fail and not include_hard_fail:
                        continue
                    if is_borderline and not include_borderline:
                        continue
                    if not is_hard_fail and not is_borderline:
                        continue

                    ann_idx = r.get("annotation_idx")
                    region = region_by_idx.get(ann_idx)
                    if region is None:
                        continue
                    points = region.get("points") or []

                    img = _get_img()
                    if img is None:
                        break  # image unreadable → skip frame
                    char_crop = _crop_from_polygon(img, points, padding=4)
                    if char_crop is None or char_crop.size == 0:
                        continue

                    candidates.append({
                        "inspection_id":  inspection_id,
                        "recipe_id":      rid,
                        "recipe_name":    recipe_name,
                        "camera_serial":  cam.get("serial_number", ""),
                        "frame_idx":      frame.get("frame_idx", 0),
                        "annotation_idx": ann_idx,
                        "expected":       r.get("expected", ""),
                        "ml_label":       r.get("ml_label", "NG"),
                        "ml_p_ok":        round(p_ok, 4),
                        "kind":           "hard_fail" if is_hard_fail else "borderline",
                        "timestamp":      ts_iso,
                        "image_path":     image_path_raw,    # for import payload
                        "crop_b64":       img_to_b64(char_crop),
                    })
                if len(candidates) >= limit:
                    break
            if len(candidates) >= limit:
                break

    return {"candidates": candidates, "count": len(candidates)}


# ════════════════════════════════════════ IMPORT FROM INSPECTIONS (apply) ═

class _ImportSelection(BaseModel):
    inspection_id: str
    annotation_idx: int


class _ImportFromInspectionsRequest(BaseModel):
    selections: List[_ImportSelection]


@router.post(
    "/ml/projects/{project_id}/import-from-inspections",
    tags=["ML Training"],
)
async def import_from_inspections(
    project_id: str,
    body: _ImportFromInspectionsRequest,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
    db=Depends(get_database),
):
    """
    Import operator-picked char crops from inspection history into a project.

    Per unique frame:
      - copy frame image into project images/ (skip if already there)
      - find or create annotation for that filename
      - merge char segments — char_id from `expected`, label=None for human
        review. Segments deduplicated by annotation_idx provenance to avoid
        re-import on repeat clicks.
    """
    if not body.selections:
        return {"imported": 0, "skipped": 0, "errors": []}

    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    images_dir = _images_dir(project_id)
    images_dir.mkdir(parents=True, exist_ok=True)

    coll = db.get_collection("inference_results")

    # Group selections by inspection_id (so we open each frame doc once)
    by_inspection: Dict[str, List[int]] = {}
    for sel in body.selections:
        by_inspection.setdefault(sel.inspection_id, []).append(sel.annotation_idx)

    imported = 0
    skipped = 0
    errors: List[Dict[str, str]] = []

    from bson import ObjectId
    from app.models.ml_training import MLAnnotationSave, AnnotationRegion, CharSegment

    for inspection_id, ann_idxs in by_inspection.items():
        try:
            try:
                _id = ObjectId(inspection_id)
            except Exception:
                errors.append({"inspection_id": inspection_id, "reason": "invalid_id"})
                continue
            doc = await coll.find_one({"_id": _id})
            if not doc:
                errors.append({"inspection_id": inspection_id, "reason": "not_found"})
                continue

            # Walk frames, find char_verification + detected_regions
            for cam in doc.get("camera_results", []):
                for frame in cam.get("frames", []):
                    cv_data = frame.get("char_verification") or {}
                    results = cv_data.get("results") or []
                    if not results:
                        continue
                    region_by_idx = {
                        r.get("annotation_index"): r
                        for r in (frame.get("detected_regions") or [])
                        if r.get("type") == "char" and r.get("annotation_index") is not None
                    }
                    expected_by_idx = {r.get("annotation_idx"): r for r in results}

                    needed = [i for i in ann_idxs if i in region_by_idx and i in expected_by_idx]
                    if not needed:
                        continue

                    image_path_raw = frame.get("image_path") or ""
                    src_path = _resolve_inspection_image(image_path_raw)
                    if src_path is None:
                        errors.append({"inspection_id": inspection_id, "reason": "image_missing"})
                        continue

                    img = cv.imread(str(src_path))
                    if img is None:
                        errors.append({"inspection_id": inspection_id, "reason": "image_unreadable"})
                        continue
                    img_h, img_w = img.shape[:2]

                    # Stable filename derived from inspection so repeat imports merge
                    safe_name = f"insp_{inspection_id}_cam{cam.get('serial_number','')}_f{frame.get('frame_idx',0)}.jpg"
                    dst_path = images_dir / safe_name
                    if not dst_path.exists():
                        shutil.copy(str(src_path), str(dst_path))

                    # Convert each selected char bbox → normalized segment
                    new_segments: List[Dict[str, Any]] = []
                    for ann_idx in needed:
                        pts = region_by_idx[ann_idx].get("points") or []
                        if not pts:
                            continue
                        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                        x0, x1 = min(xs), max(xs)
                        y0, y1 = min(ys), max(ys)
                        nx = max(0.0, x0 / img_w)
                        ny = max(0.0, y0 / img_h)
                        nw = min(1.0 - nx, (x1 - x0) / img_w)
                        nh = min(1.0 - ny, (y1 - y0) / img_h)
                        if nw <= 0 or nh <= 0:
                            continue
                        new_segments.append({
                            "id":      f"seg-{inspection_id[-8:]}-{ann_idx}",
                            "x":       nx, "y": ny, "w": nw, "h": nh,
                            "label":   None,           # operator reviews in Label tab
                            "char_id": expected_by_idx[ann_idx].get("expected", "") or None,
                        })
                    if not new_segments:
                        skipped += len(needed)
                        continue

                    # Merge: load existing annotation, dedupe by segment.id, save
                    existing = await repo.get_annotation(project_id, safe_name)
                    if existing and existing.regions:
                        regions = [r.model_dump() for r in existing.regions]
                        # Dedup: collect existing segment ids
                        existing_ids = set()
                        for reg in regions:
                            for seg in reg.get("segments", []) or []:
                                if seg.get("id"):
                                    existing_ids.add(seg["id"])
                        fresh = [s for s in new_segments if s["id"] not in existing_ids]
                        if not fresh:
                            skipped += len(new_segments)
                            continue
                        # Append into the first region (or create one)
                        if regions:
                            regions[0].setdefault("segments", []).extend(fresh)
                        else:
                            regions = [_make_wrapper_region(fresh, img_w, img_h)]
                        new_imported = len(fresh)
                    else:
                        regions = [_make_wrapper_region(new_segments, img_w, img_h)]
                        new_imported = len(new_segments)

                    save_payload = MLAnnotationSave(
                        regions=[AnnotationRegion(**r) for r in regions]
                    )
                    await repo.save_annotation(project_id, safe_name, save_payload)
                    imported += new_imported

        except Exception as e:
            logger.exception(f"Import inspection {inspection_id} failed")
            errors.append({"inspection_id": inspection_id, "reason": str(e)})

    # Refresh project image_count
    new_count = _sync_image_count(repo, project_id)
    await repo.set_image_count(project_id, new_count)

    return {"imported": imported, "skipped": skipped, "errors": errors}


def _make_wrapper_region(segments: List[Dict[str, Any]], img_w: int, img_h: int) -> Dict[str, Any]:
    """Build a single region wrapping all segments — bounding box."""
    xs0 = [s["x"] for s in segments]
    ys0 = [s["y"] for s in segments]
    xs1 = [s["x"] + s["w"] for s in segments]
    ys1 = [s["y"] + s["h"] for s in segments]
    rx, ry = min(xs0), min(ys0)
    rw, rh = max(xs1) - rx, max(ys1) - ry
    return {
        "id":       f"region-import-{int(asyncio.get_event_loop().time() * 1000)}",
        "x":        rx, "y": ry, "w": rw, "h": rh,
        "segments": segments,
    }
