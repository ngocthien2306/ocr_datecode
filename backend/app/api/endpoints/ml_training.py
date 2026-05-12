"""
ML Training API Endpoints
CRUD for projects, image management, segmentation, labeling, training, prediction.
"""
import asyncio
import base64
import logging
import shutil
import uuid
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    CharCoverageResponse,
    CharImportBatchResponse,
    CharImportBatchUpdate,
    CharImportBulkUpdate,
    CharImportCreateRequest,
    CharImportItemResponse,
    CharImportUpdate,
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
from app.services.ml_training_logs import (
    attach_handler as attach_log_handler,
    detach_handler as detach_log_handler,
    get_buffer as get_log_buffer,
)
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


class _CloneProjectRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


@router.post("/ml/projects/{project_id}/clone", tags=["ML Training"])
async def clone_project(
    project_id: str,
    body: _CloneProjectRequest,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Clone a project's training data (images + annotations) into a new project.
    Trained models are NOT copied — the new project starts in 'active' status
    and must be retrained.
    """
    src = await repo.get_project(project_id)
    if not src:
        raise HTTPException(404, "Project not found")

    new_name = (body.name or f"{src.name} (copy)").strip()
    if not new_name:
        raise HTTPException(400, "Name is required")
    new_desc = body.description if body.description is not None else src.description

    new_project = await repo.create_project(
        MLProjectCreate(name=new_name, description=new_desc),
        str(current_user.id),
    )
    new_id = new_project.id

    try:
        # Copy image files (annotations reference filenames so paths stay valid)
        src_images = _images_dir(project_id)
        dst_images = _images_dir(new_id)
        dst_images.mkdir(parents=True, exist_ok=True)
        _models_dir(new_id).mkdir(parents=True, exist_ok=True)
        if src_images.exists():
            for f in src_images.glob("*"):
                if f.is_file() and f.suffix.lower() in ALLOWED_EXTS:
                    shutil.copy2(f, dst_images / f.name)

        # Copy annotation documents under the new project_id
        annotations = await repo.list_annotations(project_id)
        for ann in annotations:
            await repo.save_annotation(
                new_id, ann.filename,
                MLAnnotationSave(regions=ann.regions),
            )

        # Refresh counters
        count = _sync_image_count(repo, new_id)
        await repo.set_image_count(new_id, count)
        await repo.refresh_labeled_count(new_id)

    except Exception as e:
        logger.exception(f"[clone] Failed to clone project {project_id} → {new_id}")
        # Best-effort rollback so we don't leave a half-cloned project around
        try:
            new_dir = _project_dir(new_id)
            if new_dir.exists():
                shutil.rmtree(new_dir)
            await repo.delete_project(new_id)
        except Exception:
            logger.exception("[clone] Rollback failed")
        raise HTTPException(500, f"Clone failed: {e}")

    cloned = await repo.get_project(new_id)
    return cloned.model_dump(by_alias=False) if cloned else new_project.model_dump(by_alias=False)


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


class _SegmentPatch(BaseModel):
    char_id: Optional[str] = None    # empty string clears char_id; None = no change
    label:   Optional[str] = None    # 'OK' | 'NG' | None = no change


@router.patch(
    "/ml/projects/{project_id}/annotations/{filename}/segments/{segment_id}",
    tags=["ML Training"],
)
async def patch_segment(
    project_id: str,
    filename: str,
    segment_id: str,
    patch: _SegmentPatch,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    """Inline-edit a single char segment from the Train-tab popover.

    Atomic at the doc level — multiple users can fix different segments in the
    same image without stepping on each other. Empty `char_id` clears it; null
    means "leave unchanged".
    """
    payload: Dict[str, Any] = {}
    if patch.char_id is not None:
        # Empty string → store None so downstream code treats it as "missing".
        payload["char_id"] = patch.char_id.strip() or None
    if patch.label is not None:
        if patch.label not in ("OK", "NG"):
            raise HTTPException(400, "label must be 'OK' or 'NG'")
        payload["label"] = patch.label
    if not payload:
        raise HTTPException(400, "Nothing to update")

    ann = await repo.patch_segment(project_id, filename, segment_id, payload)
    if ann is None:
        raise HTTPException(404, "Annotation or segment not found")
    await repo.refresh_labeled_count(project_id)
    return ann.model_dump(by_alias=False)


@router.delete(
    "/ml/projects/{project_id}/annotations/{filename}/segments/{segment_id}",
    tags=["ML Training"],
)
async def delete_segment(
    project_id: str,
    filename: str,
    segment_id: str,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    ann = await repo.delete_segment(project_id, filename, segment_id)
    if ann is None:
        raise HTTPException(404, "Annotation not found")
    await repo.refresh_labeled_count(project_id)
    return {"ok": True}


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
        request.severity_dist, request.force_defect_type, request.char_filter,
        request.enabled_defect_types,
    )
    return {"crops": crops, "count": len(crops)}


# ════════════════════════════════════════ SYNTHETIC OK PREVIEW ═════════════

class _SynthOKRequest(BaseModel):
    target_n_per_char: int = 30
    only_below_threshold: bool = True
    char_filter: Optional[List[str]] = None
    rotation_max_deg: float = 5.0
    size_jitter: float = 0.30
    font_paths: Optional[List[str]] = None
    style_sample_n: int = 64
    sample_strategy: str = "random"
    char_fill_min: float = 0.85
    char_fill_max: float = 0.95
    bg_per_char: int = 24
    fill_min: float = 0.10
    fill_max: float = 0.65
    min_contrast: float = 20.0
    max_retries: int = 4
    # When True, include project-derived glyph dict in the font pool. The
    # synth pipeline mixes it with any TTFs the user also selected.
    use_project_glyphs: bool = False


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
    images_dir = _images_dir(project_id)

    # Pull imported OK crops from the active-learning pool so preview matches
    # what training will actually use (E in the design).
    imported_chars = await repo.list_char_imports(project_id)
    imported_ok_paths = [
        (_PROJECT_ROOT / "public" / c.crop_path, c.char_id)
        for c in imported_chars
        if c.label == "OK" and c.char_id
    ]

    if not annotations and not imported_ok_paths:
        raise HTTPException(
            400,
            "No OK samples — label some in Label tab or import chars first",
        )

    def _build():
        from app.services.ml_project_glyphs import load_glyph_dict
        from app.services.ml_ok_synthesize import PROJECT_GLYPHS_TOKEN

        imported_ok: List[Tuple[Any, str]] = []
        for p, cid in imported_ok_paths:
            img = cv.imread(str(p))
            if img is not None:
                imported_ok.append((img, cid))

        # Decide whether to inject the project glyph token + load dict from
        # disk. font_paths may already contain the token; the flag is an
        # equivalent user-friendly switch.
        glyphs = None
        fp = list(request.font_paths or [])
        if request.use_project_glyphs and PROJECT_GLYPHS_TOKEN not in fp:
            fp.append(PROJECT_GLYPHS_TOKEN)
        if PROJECT_GLYPHS_TOKEN in fp:
            glyphs = load_glyph_dict(_project_dir(project_id))
            if glyphs is None:
                # Remove the token so synth doesn't fail — quietly fall back
                # to TTFs. UI will surface a warning via /glyphs/info.
                fp = [p for p in fp if p != PROJECT_GLYPHS_TOKEN]

        synth = synthesize_ok_from_annotations(
            annotations, images_dir,
            target_n_per_char=request.target_n_per_char,
            only_below_threshold=request.only_below_threshold,
            char_filter=request.char_filter,
            rotation_max_deg=request.rotation_max_deg,
            size_jitter=request.size_jitter,
            imported_ok_crops=imported_ok or None,
            font_paths=fp,
            style_sample_n=request.style_sample_n,
            sample_strategy=request.sample_strategy,
            bg_per_char=request.bg_per_char,
            char_fill_min=request.char_fill_min,
            char_fill_max=request.char_fill_max,
            fill_min=request.fill_min,
            fill_max=request.fill_max,
            min_contrast=request.min_contrast,
            max_retries=request.max_retries,
            project_glyphs=glyphs,
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


# ════════════════════════════════════════ FONT MANAGEMENT ══════════════════

_FONTS_ALLOWED_EXTS = {".ttf", ".otf", ".ttc"}
_FONTS_MAX_BYTES = 5 * 1024 * 1024


@router.get("/ml/fonts/discover", tags=["ML Training"])
async def fonts_discover(
    preview_chars: str = "",
    current_user: UserInDB = Depends(get_current_user),
):
    from app.services.ml_font_preview import list_fonts
    return await asyncio.get_event_loop().run_in_executor(
        None, list_fonts, preview_chars,
    )


@router.post("/ml/fonts/upload", tags=["ML Training"])
async def fonts_upload(
    file: UploadFile = File(...),
    current_user: UserInDB = Depends(get_current_user),
):
    from app.services.ml_font_preview import (
        _FONT_DIR, render_preview_b64, clear_preview_cache,
    )
    from app.services.ml_ok_synthesize import _measure_stroke_ratio

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _FONTS_ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported font format: {suffix}")
    content = await file.read()
    if len(content) > _FONTS_MAX_BYTES:
        raise HTTPException(400, "Font file exceeds 5 MB")

    _FONT_DIR.mkdir(parents=True, exist_ok=True)
    dst = _FONT_DIR / Path(file.filename).name
    if dst.exists():
        raise HTTPException(409, f"Font already exists: {dst.name}")
    dst.write_bytes(content)

    ratio = _measure_stroke_ratio(str(dst))
    if ratio is None:
        try:
            dst.unlink()
        except Exception:
            pass
        raise HTTPException(400, "Font file is invalid or unreadable")

    warning = None
    if not (0.08 <= ratio <= 0.22):
        warning = f"stroke {ratio:.2f} outside recommended [0.08, 0.22] — synthesis may look off"

    clear_preview_cache()
    return {
        "path": str(dst.resolve()),
        "filename": dst.name,
        "name": dst.stem,
        "stroke_ratio": round(ratio, 3),
        "source": "project",
        "preview_b64": render_preview_b64(str(dst), ""),
        "warning": warning,
    }


@router.delete("/ml/fonts/{filename}", tags=["ML Training"])
async def fonts_delete(
    filename: str,
    current_user: UserInDB = Depends(get_current_user),
):
    from app.services.ml_font_preview import _FONT_DIR, clear_preview_cache
    target = (_FONT_DIR / filename).resolve()
    try:
        target.relative_to(_FONT_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "Path traversal not allowed")
    if not target.exists():
        raise HTTPException(404, "Font not found")
    target.unlink()
    clear_preview_cache()
    return {"ok": True}


# ════════════════════════════════════════ SYNTH-OK INTROSPECTION ═══════════

class _SynthOKStyleRequest(BaseModel):
    style_sample_n: int = 64
    sample_strategy: str = "random"
    include_imported: bool = True
    n_thumbnails: int = 4


@router.post(
    "/ml/projects/{project_id}/synth-ok-style",
    tags=["ML Training"],
)
async def synth_ok_style(
    project_id: str,
    request: _SynthOKStyleRequest,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    from app.services.ml_ok_synthesize import _get_or_build_cache

    annotations = await repo.list_annotations(project_id)
    images_dir = _images_dir(project_id)

    imported_ok_paths: List[Tuple[Path, str]] = []
    if request.include_imported:
        imported_chars = await repo.list_char_imports(project_id)
        imported_ok_paths = [
            (_PROJECT_ROOT / "public" / c.crop_path, c.char_id)
            for c in imported_chars if c.label == "OK" and c.char_id
        ]

    if not annotations and not imported_ok_paths:
        raise HTTPException(400, "No OK samples to fingerprint")

    def _build():
        imported_ok: List[Tuple[Any, str]] = []
        for p, cid in imported_ok_paths:
            img = cv.imread(str(p))
            if img is not None:
                imported_ok.append((img, cid))
        bundle = _get_or_build_cache(
            annotations, images_dir, imported_ok or None,
            style_sample_n=request.style_sample_n,
            sample_strategy=request.sample_strategy,
        )
        st = bundle["style"]
        sample = bundle.get("sample_used") or []
        thumbs = sample[: max(0, request.n_thumbnails)]
        return {
            "ink_bgr":   list(st["ink_bgr"]),
            "bg_bgr":    list(st["bg_bgr"]),
            "mean_w":    st["mean_w"],
            "mean_h":    st["mean_h"],
            "blur_sigma": st["blur_sigma"],
            "noise_std": st["noise_std"],
            "n_analyzed": st["n_analyzed"],
            "ink_bg_contrast": st["ink_bg_contrast"],
            "sample_b64s": [img_to_b64(c) for c in thumbs],
        }

    try:
        return await asyncio.get_event_loop().run_in_executor(None, _build)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))


@router.get(
    "/ml/projects/{project_id}/synth-ok-bg-pool",
    tags=["ML Training"],
)
async def synth_ok_bg_pool(
    project_id: str,
    n_per_char: int = 4,
    chars: str = "",
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    from app.services.ml_ok_synthesize import _get_or_build_cache, _get_bgs_for_char
    import random as _random

    annotations = await repo.list_annotations(project_id)
    images_dir = _images_dir(project_id)
    imported_chars = await repo.list_char_imports(project_id)
    imported_ok_paths = [
        (_PROJECT_ROOT / "public" / c.crop_path, c.char_id)
        for c in imported_chars if c.label == "OK" and c.char_id
    ]
    if not annotations and not imported_ok_paths:
        raise HTTPException(400, "No OK samples for BG pool")

    requested = [c.strip() for c in (chars or "").split(",") if c.strip()]

    def _build():
        imported_ok: List[Tuple[Any, str]] = []
        for p, cid in imported_ok_paths:
            img = cv.imread(str(p))
            if img is not None:
                imported_ok.append((img, cid))
        bundle = _get_or_build_cache(annotations, images_dir, imported_ok or None)
        target_size = bundle["mean_size"]
        rng = _random.Random()
        ok_by_char = bundle["ok_crops_by_char"]
        target_chars = requested if requested else list(ok_by_char.keys())[:6]
        out: Dict[str, List[str]] = {}
        n = max(1, min(int(n_per_char), 12))
        for cid in target_chars:
            if cid not in ok_by_char:
                continue
            bgs = _get_bgs_for_char(bundle, cid, target_size, rng, n=n)
            out[cid] = [img_to_b64(b) for b in bgs[:n]]
        return out

    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _build)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))
    return result


@router.post(
    "/ml/projects/{project_id}/synth-ok-cache/clear",
    tags=["ML Training"],
)
async def synth_ok_cache_clear(
    project_id: str,
    current_user: UserInDB = Depends(get_current_user),
):
    from app.services.ml_ok_synthesize import clear_cache
    images_dir = _images_dir(project_id)
    clear_cache(images_dir)
    return {"ok": True}


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
    imported_chars = []
    if request.include_imported_chars:
        imported_chars = await repo.list_char_imports(project_id)
        # Restrict to chars with a definitive label (OK/NG).
        imported_chars = [c for c in imported_chars if c.label in ("OK", "NG")]

    if not annotations and not imported_chars:
        raise HTTPException(
            400,
            "No labeled data found. Label images in Label tab or import chars first.",
        )

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

    # Build (path, label, char_id) tuples for the service layer to read.
    public_root = _PROJECT_ROOT / "public"
    imported_files = [
        (public_root / c.crop_path, c.label, c.char_id)
        for c in imported_chars
    ]

    # Launch training in background
    background_tasks.add_task(
        _run_training_bg, repo, project_id, model_id,
        annotations, request, imported_files,
    )

    return {"model_id": model_id, "status": "training"}


_CANCEL_FLAGS: Dict[str, bool] = {}


async def _run_training_bg(
    repo: MLTrainingRepository,
    project_id: str,
    model_id: str,
    annotations,
    request: TrainRequest,
    imported_files: Optional[List[tuple]] = None,
):
    from app.services.ml_training_service import TrainingCancelled
    images_dir = _images_dir(project_id)
    model_path = _models_dir(project_id) / f"{model_id}.joblib"

    log_buffer = get_log_buffer()
    log_buffer.register(model_id)
    handler = attach_log_handler(model_id)
    log_buffer.push(model_id, f"[ML] Starting training for model {model_id} (algo={request.algorithm})")
    if imported_files:
        log_buffer.push(model_id, f"[ML] Including {len(imported_files)} imported chars in training set")

    loop = asyncio.get_event_loop()

    # Throttle DB writes / log lines: many batch-level updates within the same
    # phase only need to refresh the % field — log just on phase change or
    # after a meaningful jump (≥5 %).
    _last_phase = [None]
    _last_logged_pct = [-100.0]

    def _progress_cb(phase: str, progress: float) -> None:
        progress = float(progress)
        if _last_phase[0] != phase or progress - _last_logged_pct[0] >= 5.0 or progress >= 100.0:
            log_buffer.push(model_id, f"[phase] {phase} ({progress:.0f}%)")
            _last_phase[0] = phase
            _last_logged_pct[0] = progress
        try:
            asyncio.run_coroutine_threadsafe(
                repo.update_model_record(
                    model_id,
                    {"phase": phase, "progress": progress},
                ),
                loop,
            )
        except Exception:
            logger.exception("[ML] Failed to schedule progress update")

    _CANCEL_FLAGS[model_id] = False

    def _is_cancelled() -> bool:
        return _CANCEL_FLAGS.get(model_id, False)

    try:
        metrics = await loop.run_in_executor(
            None,
            partial(
                train_model,
                annotations,
                images_dir,
                request,
                model_path,
                progress_cb=_progress_cb,
                imported_char_files=imported_files or [],
                cancel_check=_is_cancelled,
            ),
        )
        await repo.update_model_record(model_id, {
            "status": "completed",
            "metrics": metrics,
            "model_path": str(model_path),
            "phase": "completed",
            "progress": 100.0,
        })
        await repo.set_status(project_id, "trained")
        log_buffer.push(model_id, f"[ML] Training completed (acc_test={metrics.get('accuracy_test', 0):.3f})")
        logger.info(f"[ML] Training completed for project {project_id}, model {model_id}")

    except TrainingCancelled:
        log_buffer.push(model_id, "[ML] Training cancelled by user", level="WARNING")
        await repo.update_model_record(model_id, {
            "status": "cancelled",
            "error": "Cancelled by user",
            "phase": "cancelled",
        })
        await repo.set_status(project_id, "active")
        try:
            if model_path.exists():
                model_path.unlink()
            sidecar = model_path.parent / f"{model_path.stem}_test_set.json"
            if sidecar.exists():
                sidecar.unlink()
        except Exception:
            logger.exception("[ML] Failed to clean cancelled model files")

    except Exception as e:
        logger.exception(f"[ML] Training failed for project {project_id}")
        log_buffer.push(model_id, f"[ML] Training failed: {e}", level="ERROR")
        await repo.update_model_record(model_id, {
            "status": "failed",
            "error": str(e),
            "phase": "failed",
        })
        await repo.set_status(project_id, "active")
    finally:
        detach_log_handler(handler)
        _CANCEL_FLAGS.pop(model_id, None)


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


@router.get("/ml/projects/{project_id}/models/{model_id}/logs", tags=["ML Training"])
async def get_training_logs(
    project_id: str,
    model_id: str,
    since: int = 0,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Real-time training log + status. Poll with `since` (use the previous
    response's `next_since`) to receive only new lines.
    """
    models = await repo.list_models(project_id)
    model_record = next((m for m in models if m.id == model_id), None)
    if not model_record:
        raise HTTPException(404, "Model not found")

    logs, next_since = get_log_buffer().get_since(model_id, since)
    return {
        "logs":       logs,
        "next_since": next_since,
        "phase":      model_record.phase,
        "progress":   float(model_record.progress or 0.0),
        "status":     model_record.status,
        "error":      model_record.error,
    }


@router.post(
    "/ml/projects/{project_id}/models/{model_id}/cancel",
    tags=["ML Training"],
)
async def cancel_training(
    project_id: str,
    model_id: str,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    models = await repo.list_models(project_id)
    model_record = next((m for m in models if m.id == model_id), None)
    if not model_record:
        raise HTTPException(404, "Model not found")
    if model_record.status not in ("training", "pending"):
        raise HTTPException(409, f"Model status is '{model_record.status}', not cancellable")

    if model_id in _CANCEL_FLAGS:
        _CANCEL_FLAGS[model_id] = True
        await repo.update_model_record(model_id, {"phase": "cancelling"})
        get_log_buffer().push(model_id, "[ML] Cancel requested", level="WARNING")
        return {"ok": True, "model_id": model_id, "mode": "cooperative"}

    await repo.update_model_record(model_id, {
        "status":   "failed",
        "error":    "Stuck record reset by user (no in-process task — likely from a previous service instance)",
        "phase":    "failed",
        "progress": 0.0,
    })
    await repo.set_status(project_id, "active")
    get_log_buffer().push(model_id, "[ML] Stuck record force-failed by user", level="WARNING")

    model_path = _models_dir(project_id) / f"{model_id}.joblib"
    try:
        if model_path.exists():
            model_path.unlink()
        sidecar = model_path.parent / f"{model_path.stem}_test_set.json"
        if sidecar.exists():
            sidecar.unlink()
    except Exception:
        logger.exception("[ML] Failed to clean stuck model files")

    return {"ok": True, "model_id": model_id, "mode": "force_failed"}


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




# ════════════════════════════════════════ CHAR IMPORTS ════════════════════
#
# Active-learning pool: char crops harvested from past production inferences,
# stored as JPEG files on disk + Mongo docs. The Imported Chars tab is the
# single UI for managing this pool; the Train pipeline merges it with the
# Label-tab annotations when `include_imported_chars=True`.
#
# Disk layout:
#   public/ml_projects/{project_id}/imported_chars/{batch_id}/{char_doc_id}.jpg
#
# Dedup key: (project_id, inspection_id, annotation_idx) — set in stone for
# each imported char, used by both list-candidates (badge) and create-batch
# (skip).

_INSPECTION_UPLOADS = _PROJECT_ROOT / "backend" / "uploads"


def _imported_chars_dir(project_id: str) -> Path:
    return ML_BASE / project_id / "imported_chars"


def _prefer_original(image_path: str) -> str:
    """Swap a `_viz.jpg` suffix to `_org.jpg` so we crop from the raw frame
    instead of the bbox-overlaid visualization. Other suffixes are returned
    unchanged."""
    if not image_path:
        return image_path
    if image_path.endswith("_viz.jpg"):
        return image_path[:-len("_viz.jpg")] + "_org.jpg"
    if image_path.endswith("_viz.jpeg"):
        return image_path[:-len("_viz.jpeg")] + "_org.jpeg"
    if image_path.endswith("_viz.png"):
        return image_path[:-len("_viz.png")] + "_org.png"
    return image_path


def _resolve_inspection_image(image_path: str) -> Optional[Path]:
    """Resolve a relative inspection image path to an absolute Path on disk.

    Prefers the original frame (`_org.<ext>`) over the visualization
    (`_viz.<ext>`); falls back to the literal path if `_org` doesn't exist.
    """
    if not image_path:
        return None

    def _resolve_one(candidate_path: str) -> Optional[Path]:
        p = Path(candidate_path)
        if p.is_absolute() and p.exists():
            return p
        candidate = _INSPECTION_UPLOADS / candidate_path
        return candidate if candidate.exists() else None

    org_path = _prefer_original(image_path)
    if org_path != image_path:
        resolved = _resolve_one(org_path)
        if resolved is not None:
            return resolved
    return _resolve_one(image_path)


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


def _crop_url(crop_path: str) -> str:
    """Build the static-mount URL for a crop_path stored relative to /public."""
    # crop_path looks like "ml_projects/{pid}/imported_chars/{bid}/{cid}.jpg"
    # Strip the leading "ml_projects/" since the /api/ml-files mount points
    # directly at the ml_projects directory.
    rel = crop_path
    if rel.startswith("ml_projects/"):
        rel = rel[len("ml_projects/"):]
    return f"{_ML_FILES_STATIC_PREFIX}/{rel}"


def _char_to_response(c) -> Dict[str, Any]:
    return {
        "id":               c.id,
        "batch_id":         c.batch_id,
        "char_id":          c.char_id,
        "label":            c.label,
        "crop_url":         _crop_url(c.crop_path),
        "ml_label":         c.ml_label,
        "ml_p_ok":          c.ml_p_ok,
        "inspection_id":    c.inspection_id,
        "annotation_idx":   c.annotation_idx,
        "recipe_name":      c.recipe_name,
        "camera_serial":    c.camera_serial,
        "frame_idx":        c.frame_idx,
        "source_timestamp": c.source_timestamp,
        "created_at":       c.created_at,
    }


# ── Inspection candidates (search) ─────────────────────────────────────────

@router.get(
    "/ml/projects/{project_id}/inspection-candidates",
    tags=["ML Training"],
)
async def get_inspection_candidates(
    project_id: str,
    recipe_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    include_pred_ok: bool = True,
    include_pred_ng: bool = True,
    limit: int = 100,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
    db=Depends(get_database),
):
    """
    Return char-level training candidates from past inspections, filtered by
    the model's predicted label.

    Each candidate is one `char_verification` result, cropped from the saved
    frame image. Already-imported candidates are tagged with the batch they
    live in so the FE can disable the checkbox.
    """
    from datetime import datetime as _dt

    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    if not (include_pred_ok or include_pred_ng):
        return {"candidates": [], "count": 0}

    # Pre-fetch dedup map: {f"{insp_id}:{ann_idx}": batch_id}
    imported_keys = await repo.get_imported_provenance_keys(project_id)
    batches = {b.id: b.name for b in await repo.list_char_import_batches(project_id)}

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
    query["camera_results.frames.char_verification"] = {"$ne": None}

    coll = db.get_collection("inference_results")
    cursor = coll.find(query).sort("timestamp", -1).limit(500)

    candidates: List[Dict[str, Any]] = []

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
                region_by_idx = {
                    r.get("annotation_index"): r
                    for r in detected_regions
                    if r.get("type") == "char" and r.get("annotation_index") is not None
                }

                image_path_raw = frame.get("image_path")
                image_path = _resolve_inspection_image(image_path_raw or "")
                if image_path is None:
                    continue

                _img_cache: Dict[str, Any] = {"img": None}

                def _get_img():
                    if _img_cache["img"] is None:
                        _img_cache["img"] = cv.imread(str(image_path))
                    return _img_cache["img"]

                for r in results:
                    if len(candidates) >= limit:
                        break
                    ml_label = (r.get("ml_label") or "NG").upper()
                    if ml_label == "OK" and not include_pred_ok:
                        continue
                    if ml_label == "NG" and not include_pred_ng:
                        continue

                    ann_idx = r.get("annotation_idx")
                    region = region_by_idx.get(ann_idx)
                    if region is None:
                        continue
                    points = region.get("points") or []

                    img = _get_img()
                    if img is None:
                        break
                    char_crop = _crop_from_polygon(img, points, padding=4)
                    if char_crop is None or char_crop.size == 0:
                        continue

                    cam_serial = cam.get("serial_number", "")
                    frame_idx_i = int(frame.get("frame_idx", 0))
                    key = f"{inspection_id}:{cam_serial}:{frame_idx_i}:{ann_idx}"
                    imported_batch_id = imported_keys.get(key)
                    p_ok = float(r.get("ml_p_ok") or 0.0)

                    candidates.append({
                        "inspection_id":      inspection_id,
                        "recipe_id":          rid,
                        "recipe_name":        recipe_name,
                        "camera_serial":      cam.get("serial_number", ""),
                        "frame_idx":          frame.get("frame_idx", 0),
                        "annotation_idx":     ann_idx,
                        "expected":           r.get("expected", ""),
                        "ml_label":           ml_label,
                        "ml_p_ok":            round(p_ok, 4),
                        "timestamp":          ts_iso,
                        "image_path":         image_path_raw,
                        "crop_b64":           img_to_b64(char_crop),
                        "imported_batch_id":  imported_batch_id,
                        "imported_batch_name": batches.get(imported_batch_id) if imported_batch_id else None,
                    })
                if len(candidates) >= limit:
                    break
            if len(candidates) >= limit:
                break

    return {"candidates": candidates, "count": len(candidates)}


# ── Char-import batches CRUD ───────────────────────────────────────────────

@router.get(
    "/ml/projects/{project_id}/char-imports/batches",
    tags=["ML Training"],
)
async def list_char_import_batches(
    project_id: str,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    batches = await repo.list_char_import_batches(project_id)
    counts = await repo.count_char_imports_by_batch(project_id)
    out: List[Dict[str, Any]] = []
    for b in batches:
        c = counts.get(b.id, {"OK": 0, "NG": 0})
        ok_n = c.get("OK", 0)
        ng_n = c.get("NG", 0)
        out.append({
            "id":         b.id,
            "name":       b.name,
            "created_at": b.created_at,
            "total":      ok_n + ng_n,
            "ok_count":   ok_n,
            "ng_count":   ng_n,
        })
    return out


@router.post(
    "/ml/projects/{project_id}/char-imports/batches",
    tags=["ML Training"],
)
async def create_char_import_batch(
    project_id: str,
    request: CharImportCreateRequest,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
    db=Depends(get_database),
):
    """
    Create a new batch and import the selected chars into it.

    For each `(inspection_id, annotation_idx)` selection: load the inspection
    doc, crop the char from the saved frame, write the JPEG into the batch
    folder, and insert one ml_char_imports doc with `label='NG'` (default).

    Already-imported `(inspection_id, annotation_idx)` pairs are skipped.
    """
    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    if not request.selections:
        raise HTTPException(400, "No selections provided")

    from bson import ObjectId
    from datetime import datetime as _dt

    now = _dt.utcnow()

    # Auto-generate name if missing
    name = (request.batch_name or "").strip()
    if not name:
        name = f"Import {now.strftime('%Y-%m-%d %H:%M')}"

    batch = await repo.create_char_import_batch(project_id, name)
    batch_id = batch.id

    batch_dir = _imported_chars_dir(project_id) / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    imported = 0
    skipped = 0
    errors: List[Dict[str, str]] = []

    # Pre-fetch dedup keys to skip already-imported items.
    imported_keys = await repo.get_imported_provenance_keys(project_id)

    # Group selections by inspection_id, then index per (camera_serial, frame_idx)
    # so we only crop chars from the exact frame the user picked — annotation_idx
    # alone is not unique within an inspection (each frame reuses the same
    # indices for its char positions).
    by_insp: Dict[str, Dict[Tuple[str, int], set]] = {}
    for sel in request.selections:
        comp_key = (
            f"{sel.inspection_id}:{sel.camera_serial}:"
            f"{sel.frame_idx}:{sel.annotation_idx}"
        )
        if comp_key in imported_keys:
            skipped += 1
            continue
        slot = by_insp.setdefault(sel.inspection_id, {})
        slot.setdefault((sel.camera_serial, sel.frame_idx), set()).add(sel.annotation_idx)

    coll = db.get_collection("inference_results")

    for insp_id, frame_map in by_insp.items():
        try:
            # inference_results._id is stored as a STRING (not ObjectId) — see
            # inference_result_repository.delete_by_id / get_by_id which query
            # with the raw string. Querying with ObjectId here would silently
            # miss every doc.
            if not insp_id:
                errors.append({"inspection_id": insp_id, "reason": "invalid_id"})
                continue
            doc = await coll.find_one({"_id": insp_id})
            if not doc:
                errors.append({"inspection_id": insp_id, "reason": "not_found"})
                continue

            recipe_id = str(doc.get("recipe_id", ""))
            recipe_name = doc.get("recipe_name", "")
            ts_doc = doc.get("timestamp")

            for cam in doc.get("camera_results", []):
                cam_serial = cam.get("serial_number", "")
                for frame in cam.get("frames", []):
                    frame_idx_i = int(frame.get("frame_idx", 0))
                    wanted_idxs = frame_map.get((cam_serial, frame_idx_i))
                    if not wanted_idxs:
                        continue

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

                    needed = [i for i in wanted_idxs if i in region_by_idx and i in expected_by_idx]
                    if not needed:
                        continue

                    image_path_raw = frame.get("image_path") or ""
                    src_path = _resolve_inspection_image(image_path_raw)
                    if src_path is None:
                        errors.append({"inspection_id": insp_id, "reason": "image_missing"})
                        continue

                    img = cv.imread(str(src_path))
                    if img is None:
                        errors.append({"inspection_id": insp_id, "reason": "image_unreadable"})
                        continue

                    for ann_idx in needed:
                        pts = region_by_idx[ann_idx].get("points") or []
                        char_crop = _crop_from_polygon(img, pts, padding=4)
                        if char_crop is None or char_crop.size == 0:
                            skipped += 1
                            continue

                        # Pre-allocate Mongo _id so it doubles as the file name.
                        oid = ObjectId()
                        rel_path = f"ml_projects/{project_id}/imported_chars/{batch_id}/{oid}.jpg"
                        abs_path = batch_dir / f"{oid}.jpg"
                        ok_w, jpg_buf = cv.imencode(".jpg", char_crop, [cv.IMWRITE_JPEG_QUALITY, 90])
                        if not ok_w:
                            skipped += 1
                            continue
                        abs_path.write_bytes(jpg_buf.tobytes())

                        r_data = expected_by_idx[ann_idx]
                        ml_label = (r_data.get("ml_label") or "NG").upper()
                        p_ok = float(r_data.get("ml_p_ok") or 0.0)
                        char_doc = {
                            "_id":              oid,
                            "project_id":       project_id,
                            "batch_id":         batch_id,
                            "inspection_id":    insp_id,
                            "annotation_idx":   ann_idx,
                            "frame_idx":        int(frame.get("frame_idx", 0)),
                            "camera_serial":    cam.get("serial_number", ""),
                            "recipe_id":        recipe_id,
                            "recipe_name":      recipe_name,
                            "char_id":          (r_data.get("expected") or None),
                            "label":            "NG",
                            "crop_path":        rel_path,
                            "ml_label":         ml_label,
                            "ml_p_ok":          round(p_ok, 4),
                            "source_timestamp": ts_doc,
                            "created_at":       now,
                            "updated_at":       now,
                        }
                        await repo.char_imports.insert_one(char_doc)
                        imported += 1

        except Exception as e:
            logger.exception(f"[char-imports] inspection {insp_id} failed")
            errors.append({"inspection_id": insp_id, "reason": str(e)})

    # If nothing was imported (all skipped/errored), drop the empty batch.
    if imported == 0:
        await repo.delete_char_import_batch(batch_id)
        try:
            shutil.rmtree(batch_dir)
        except Exception:
            pass
        return {
            "batch_id":  None,
            "imported":  0,
            "skipped":   skipped,
            "errors":    errors,
        }

    return {
        "batch_id":  batch_id,
        "batch_name": name,
        "imported":  imported,
        "skipped":   skipped,
        "errors":    errors,
    }


@router.patch(
    "/ml/projects/{project_id}/char-imports/batches/{batch_id}",
    tags=["ML Training"],
)
async def rename_char_import_batch(
    project_id: str,
    batch_id: str,
    update: CharImportBatchUpdate,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    name = (update.name or "").strip()
    if not name:
        raise HTTPException(400, "Name cannot be empty")
    batch = await repo.rename_char_import_batch(batch_id, name)
    if not batch:
        raise HTTPException(404, "Batch not found")
    return {"id": batch.id, "name": batch.name}


@router.delete(
    "/ml/projects/{project_id}/char-imports/batches/{batch_id}",
    tags=["ML Training"],
)
async def delete_char_import_batch(
    project_id: str,
    batch_id: str,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    batch = await repo.get_char_import_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")

    await repo.delete_char_import_batch(batch_id)
    # Best-effort folder cleanup. crop_paths are inside this folder.
    batch_dir = _imported_chars_dir(project_id) / batch_id
    try:
        if batch_dir.exists():
            shutil.rmtree(batch_dir)
    except Exception:
        logger.exception(f"[char-imports] failed to remove {batch_dir}")
    return {"ok": True}


# ── Char-import items CRUD ─────────────────────────────────────────────────

@router.get(
    "/ml/projects/{project_id}/char-imports/chars",
    tags=["ML Training"],
)
async def list_char_imports(
    project_id: str,
    batch_id: Optional[str] = None,
    label: Optional[str] = None,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    chars = await repo.list_char_imports(project_id, batch_id=batch_id, label=label)
    return [_char_to_response(c) for c in chars]


@router.patch(
    "/ml/projects/{project_id}/char-imports/chars/bulk",
    tags=["ML Training"],
)
async def bulk_update_char_imports(
    project_id: str,
    body: CharImportBulkUpdate,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    if not body.char_ids:
        return {"updated": 0}

    if body.delete:
        crop_paths = await repo.bulk_delete_char_imports(body.char_ids)
        for rel in crop_paths:
            abs_path = _PROJECT_ROOT / "public" / rel
            try:
                if abs_path.exists():
                    abs_path.unlink()
            except Exception:
                logger.exception(f"[char-imports] failed to unlink {abs_path}")
        return {"deleted": len(crop_paths)}

    if body.label is not None:
        if body.label not in ("OK", "NG"):
            raise HTTPException(400, "label must be 'OK' or 'NG'")
        n = await repo.bulk_update_char_imports(body.char_ids, {"label": body.label})
        return {"updated": n}

    raise HTTPException(400, "Nothing to do — set `label` or `delete=true`")


@router.patch(
    "/ml/projects/{project_id}/char-imports/chars/{char_id}",
    tags=["ML Training"],
)
async def update_char_import(
    project_id: str,
    char_id: str,
    update: CharImportUpdate,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    payload = {}
    if update.char_id is not None:
        payload["char_id"] = update.char_id.strip() or None
    if update.label is not None:
        if update.label not in ("OK", "NG"):
            raise HTTPException(400, "label must be 'OK' or 'NG'")
        payload["label"] = update.label
    if not payload:
        raise HTTPException(400, "Nothing to update")

    char = await repo.update_char_import(char_id, payload)
    if not char:
        raise HTTPException(404, "Char not found")
    return _char_to_response(char)


@router.delete(
    "/ml/projects/{project_id}/char-imports/chars/{char_id}",
    tags=["ML Training"],
)
async def delete_char_import(
    project_id: str,
    char_id: str,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    crop_path = await repo.delete_char_import(char_id)
    if crop_path is None:
        raise HTTPException(404, "Char not found")
    # Unlink crop file (best-effort)
    abs_path = _PROJECT_ROOT / "public" / crop_path
    try:
        if abs_path.exists():
            abs_path.unlink()
    except Exception:
        logger.exception(f"[char-imports] failed to unlink {abs_path}")
    return {"ok": True}


# ════════════════════════════════════════ PROJECT GLYPHS (custom font) ═════

@router.post(
    "/ml/projects/{project_id}/glyphs/rebuild",
    tags=["ML Training"],
)
async def rebuild_project_glyphs(
    project_id: str,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Build (or rebuild) the project-derived glyph dictionary from current OK
    pool (labeled annotations + imported OK). Each rebuild reflects all OK
    samples that exist at call time — so labeling more chars → richer glyphs.
    """
    from app.services.ml_project_glyphs import build_glyph_dict, save_glyph_dict

    annotations = await repo.list_annotations(project_id)
    imported_chars = await repo.list_char_imports(project_id)
    imported_ok_paths = [
        (_PROJECT_ROOT / "public" / c.crop_path, c.char_id)
        for c in imported_chars if c.label == "OK" and c.char_id
    ]
    if not annotations and not imported_ok_paths:
        raise HTTPException(400, "No OK samples to build glyphs from")

    images_dir = _images_dir(project_id)
    project_dir = _project_dir(project_id)

    def _build():
        # Collect OK crops by char_id — labeled first, then imported
        from app.services.ml_segment_service import crop_segment
        ok_by_char: Dict[str, List[Any]] = {}
        for ann in annotations:
            img_path = images_dir / ann.filename
            for region in ann.regions:
                for seg in region.segments:
                    if seg.label != "OK" or not seg.char_id:
                        continue
                    crop = crop_segment(img_path, {
                        "x": seg.x, "y": seg.y, "w": seg.w, "h": seg.h,
                    })
                    if crop is None:
                        continue
                    ok_by_char.setdefault(seg.char_id, []).append(crop)
        for p, cid in imported_ok_paths:
            img = cv.imread(str(p))
            if img is None:
                continue
            ok_by_char.setdefault(cid, []).append(img)

        if not any(ok_by_char.values()):
            raise ValueError("No usable OK crops")

        sample_counts = {c: len(v) for c, v in ok_by_char.items()}
        glyphs = build_glyph_dict(ok_by_char, min_samples=1)
        return save_glyph_dict(project_dir, glyphs, sample_counts=sample_counts)

    try:
        meta = await asyncio.get_event_loop().run_in_executor(None, _build)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))

    # Bust the synth cache so subsequent previews see the new glyphs.
    from app.services.ml_ok_synthesize import clear_cache
    clear_cache(images_dir)
    return meta


@router.get(
    "/ml/projects/{project_id}/glyphs/info",
    tags=["ML Training"],
)
async def project_glyphs_info(
    project_id: str,
    include_thumbnails: bool = True,
    current_user: UserInDB = Depends(get_current_user),
):
    """Return glyph metadata (chars covered, built_at, sample counts) and
    optionally per-char preview thumbnails (base64 PNG)."""
    from app.services.ml_project_glyphs import load_meta, load_thumbnails
    project_dir = _project_dir(project_id)
    meta = load_meta(project_dir)
    if meta is None:
        return {"built": False}
    thumbs = load_thumbnails(project_dir) if include_thumbnails else {}
    return {
        "built":          True,
        "built_at":       meta.get("built_at"),
        "chars_covered":  meta.get("chars_covered", []),
        "count":          meta.get("count", 0),
        "canvas":         meta.get("canvas"),
        "sample_counts":  meta.get("sample_counts", {}),
        "thumbnails":     thumbs,
    }


# ════════════════════════════════════════ TRAINING RESOURCE RELEASE ════════

class _ReleaseRequest(BaseModel):
    # When True, also drop the SupCon ONNX session. Penalty: ~3-8s reload on
    # the next embed/predict call. Default False so a follow-up predict from
    # Realtime tab stays fast.
    drop_onnx_session: bool = False


@router.post(
    "/ml/projects/{project_id}/release-training-resources",
    tags=["ML Training"],
)
async def release_training_resources_endpoint(
    project_id: str,
    request: Optional[_ReleaseRequest] = None,
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Drop in-memory training caches when the user finishes a session
    (e.g. closes the ML Training Studio). Hard-rejects with 409 if any
    training is still running for this project — closing the UI must NOT
    affect a background training job; the job keeps going and resources
    stay live.
    """
    # Block release while any training is mid-run for this project.
    # Status 'training' or 'pending' both count as "in progress".
    models = await repo.list_models(project_id)
    in_progress = [m for m in models if m.status in ("training", "pending")]
    if in_progress:
        raise HTTPException(
            409,
            f"Training in progress for {len(in_progress)} model(s); "
            "resources kept. Cancel training first if you really want to release.",
        )

    from app.services.ml_training_service import release_training_resources
    images_dir = _images_dir(project_id)
    drop_onnx = bool(request.drop_onnx_session) if request else False

    # Run cleanup off the event loop — gc.collect on big training arrays can
    # take a few hundred ms and we don't want to block other API requests.
    released = await asyncio.get_event_loop().run_in_executor(
        None, release_training_resources, images_dir, drop_onnx,
    )
    return {"released": True, **released}
