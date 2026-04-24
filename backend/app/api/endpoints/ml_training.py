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
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

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
from app.services.ml_segment_service import segment_region
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
    Pre-populate ML annotations from recipe template's text/datecode bboxes.

    For each filename in project images:
      - For each text/datecode annotation in the recipe's camera template:
        - Create a region + segment with char_id auto-filled from annotation.text
        - label left as None — user labels OK/NG in the UI afterwards

    Partial-success semantics: files that fail (missing on disk, no
    recipe annotations, invalid bbox) are skipped with error logged;
    other files proceed.
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
    char_ids_seen = set()

    images_dir = _images_dir(project_id)

    for filename in request.filenames:
        try:
            image_path = images_dir / filename
            if not image_path.exists():
                raise FileNotFoundError(f"{filename} not present in project images")

            regions: List[AnnotationRegion] = []
            # Loop over each template defined for the camera
            # Normally there's 1 template per camera, but support N.
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
                    char_id = (ann.text or "").strip() or None

                    seg = CharSegment(
                        id=str(uuid.uuid4()),
                        x=nx, y=ny, w=nw, h=nh,
                        label=None,
                        char_id=char_id,
                    )
                    regions.append(AnnotationRegion(
                        id=str(uuid.uuid4()),
                        x=nx, y=ny, w=nw, h=nh,
                        segments=[seg],
                    ))
                    if char_id:
                        char_ids_seen.add(char_id)

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

    # Refresh labeled count (may be 0 since we only created segments, not labels)
    await repo.refresh_labeled_count(project_id)

    logger.info(
        f"[import-from-recipe] project={project_id} recipe={request.recipe_id} "
        f"imported={imported} skipped={skipped} char_ids={sorted(char_ids_seen)}"
    )

    return ImportFromRecipeResponse(
        imported=imported,
        skipped=skipped,
        errors=errors,
        char_ids=sorted(char_ids_seen),
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
