"""
Export a trained model to ONNX and to a standalone TensorRT .engine, and
download either.

Both run automatically at the end of a training run (train.py); these endpoints
are for re-exporting — after an engine is deleted, or when the TensorRT version
on the machine changes and old engines stop deserializing.
"""
import asyncio
import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from app.api.dependencies.auth import CurrentUser, get_current_user
from app.core.config import CHARACTER_DICT_PATH
from app.db.mongodb import get_database
from app.repositories.ocr_repository import OCRRepository
from app.services import dataset_fs, gpu_lock, onnx_export, trt_export

logger = logging.getLogger(__name__)

router = APIRouter()


def get_repo(db=Depends(get_database)) -> OCRRepository:
    return OCRRepository(db)


async def _require_completed(repo: OCRRepository, project_id: str, model_id: str):
    model = await repo.get_model(model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")
    if model.status != "completed" or not model.checkpoint_path:
        raise HTTPException(400, f"Model status is '{model.status}' — train to completion first")
    if not Path(model.checkpoint_path).is_file():
        raise HTTPException(500, f"Checkpoint missing on disk: {model.checkpoint_path}")
    return model


def export_onnx_sync(project_id: str, model_id: str, config_path: Path, ckpt_path: Path) -> dict:
    """ONNX export is CPU-only, so it does not take the GPU lock."""
    return onnx_export.export_onnx(
        config_path, ckpt_path, dataset_fs.export_dir(project_id, model_id), fp16=True,
    )


def build_engine_sync(project_id: str, model_id: str, onnx_path: Path, fp16: bool = True) -> dict:
    """Build the engine and copy the dict beside it.

    The dict is copied so the export directory is self-contained: at runtime the
    recipe hands ai_services an engine path and a dict path, and a decoder built
    from the wrong dict silently shifts every character index.
    """
    export_dir = dataset_fs.export_dir(project_id, model_id)
    engine_path = export_dir / "rec_smtr_fp16.engine"
    input_name, dims, outputs = trt_export.onnx_io_info(onnx_path)
    if len(outputs) != 2:
        raise RuntimeError(
            f"ONNX has {len(outputs)} outputs {outputs}; ai_services' "
            f"TextRecognizerSMTRTRT asserts exactly 2 (gtc_logits, ctc_logits)"
        )
    with gpu_lock.gpu_lock(f"ocr-trt:{model_id}"):
        trt_export.build_engine(onnx_path, engine_path, input_name=input_name, fp16=fp16)
    info = trt_export.inspect_engine(engine_path)

    dict_dest = export_dir / CHARACTER_DICT_PATH.name
    shutil.copyfile(CHARACTER_DICT_PATH, dict_dest)
    return {"engine_path": str(engine_path), "dict_path": str(dict_dest), **info}


@router.post("/projects/{project_id}/models/{model_id}/export-onnx")
async def export_model_onnx(
    project_id: str,
    model_id: str,
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    model = await _require_completed(repo, project_id, model_id)
    if not model.config_path or not Path(model.config_path).is_file():
        raise HTTPException(500, "Training config missing — cannot rebuild the graph")

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: export_onnx_sync(
                project_id, model_id, Path(model.config_path), Path(model.checkpoint_path),
            ),
        )
    except Exception as e:
        logger.exception(f"[ocr export] ONNX export failed for {model_id}")
        raise HTTPException(500, f"ONNX export failed: {e}")

    await repo.update_model_record(model_id, {
        "onnx_path": result["onnx_path"],
        "onnx_fp16_path": result["onnx_fp16_path"],
    })
    return result


@router.post("/projects/{project_id}/models/{model_id}/export-tensorrt")
async def export_model_tensorrt(
    project_id: str,
    model_id: str,
    fp16: bool = Query(True),
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    model = await _require_completed(repo, project_id, model_id)
    onnx_path = model.onnx_fp16_path if fp16 else model.onnx_path
    if not onnx_path or not Path(onnx_path).is_file():
        raise HTTPException(400, "No ONNX export for this model yet — export ONNX first")

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, lambda: build_engine_sync(project_id, model_id, Path(onnx_path), fp16),
        )
    except Exception as e:
        logger.exception(f"[ocr export] TensorRT build failed for {model_id}")
        raise HTTPException(500, f"TensorRT build failed: {e}")

    await repo.update_model_record(model_id, {
        "engine_path": result["engine_path"], "dict_path": result["dict_path"],
    })
    return result


@router.get("/projects/{project_id}/models/{model_id}/export/inspect")
async def inspect_export(
    project_id: str,
    model_id: str,
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Deserialize the engine and report its bindings + profile.

    `runtime_compatible` is the one to read: it is false when the engine has
    anything other than two outputs, which is the failure ai_services would hit
    at load time instead.
    """
    model = await repo.get_model(model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")
    if not model.engine_path or not Path(model.engine_path).is_file():
        raise HTTPException(404, "No TensorRT engine for this model yet")
    try:
        return trt_export.inspect_engine(Path(model.engine_path))
    except Exception as e:
        raise HTTPException(500, f"Engine inspect failed: {e}")


@router.get("/projects/{project_id}/models/{model_id}/export/{artifact}")
async def download_artifact(
    project_id: str,
    model_id: str,
    artifact: str,
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Download onnx | onnx_fp16 | engine | dict | checkpoint."""
    model = await repo.get_model(model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")

    field = {
        "onnx": ("onnx_path", f"rec_smtr_{model_id}.onnx"),
        "onnx_fp16": ("onnx_fp16_path", f"rec_smtr_fp16_{model_id}.onnx"),
        "engine": ("engine_path", f"rec_smtr_fp16_{model_id}.engine"),
        "dict": ("dict_path", CHARACTER_DICT_PATH.name),
        "checkpoint": ("checkpoint_path", f"{model_id}.pth"),
    }.get(artifact)
    if field is None:
        raise HTTPException(400, "artifact must be onnx | onnx_fp16 | engine | dict | checkpoint")

    path = getattr(model, field[0], None)
    if not path or not Path(path).is_file():
        raise HTTPException(404, f"No {artifact} for this model yet")
    return FileResponse(path, filename=field[1], media_type="application/octet-stream")
