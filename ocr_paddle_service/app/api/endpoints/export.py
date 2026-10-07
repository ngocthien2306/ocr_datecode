"""
Re-export, evaluate, predict and download for trained Paddle models.

Export + evaluation run automatically at the end of a training run (train.py);
these endpoints are for re-running a stage — after an engine is deleted, or
when the TensorRT version in the vision env changes and old engines stop
deserializing.
"""
import asyncio
import base64
import logging
from pathlib import Path

import cv2 as cv
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.api.dependencies.auth import CurrentUser, get_current_user
from app.db.mongodb import get_database
from app.repositories.ocr_repository import OCRRepository
from app.services import gpu_lock, paddle_export
from app.services.dataset_builder import build_dataset

logger = logging.getLogger(__name__)

router = APIRouter()


def get_repo(db=Depends(get_database)) -> OCRRepository:
    return OCRRepository(db)


async def _require_model(repo, project_id: str, model_id: str, completed: bool = True):
    model = await repo.get_model(model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")
    if completed and (model.status != "completed" or not model.checkpoint_path):
        raise HTTPException(400, f"Model status is '{model.status}' — train to completion first")
    return model


async def eval_items_for(repo, project_id: str, model) -> list:
    """The test items this model was measured on: same validation + split as
    its run, from the run's own recorded params."""
    params = model.params or {}
    items = await repo.list_trainable_items(project_id)
    report = build_dataset(
        project_id, items, arch=model.arch,
        test_split=float(params.get("test_split", 0.2)),
        max_text_length=int(params.get("max_text_length", 25)),
        exclude_item_ids=params.get("exclude_item_ids") or [],
        dry_run=True,
    )
    return report["test_items"]


@router.post("/projects/{project_id}/models/{model_id}/export-onnx")
async def export_model_onnx(
    project_id: str, model_id: str,
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    model = await _require_model(repo, project_id, model_id)
    if not model.config_path or not Path(model.config_path).is_file():
        raise HTTPException(500, "Training config missing — cannot rebuild the graph")
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, lambda: paddle_export.export_onnx(
            project_id, model_id, model.arch, Path(model.config_path), Path(model.checkpoint_path),
        ))
    except Exception as e:
        logger.exception(f"[paddle export] ONNX export failed for {model_id}")
        raise HTTPException(500, f"ONNX export failed: {e}")
    paddle_export.release_model(model_id)
    # A re-export wipes the export dir, engine included.
    await repo.update_model_record(model_id, {
        "inference_dir": result["inference_dir"], "onnx_path": result["onnx_path"],
        "dict_path": result["dict_path"], "num_classes": result["num_classes"],
        "engine_path": None, "metrics.acc_trt": None, "metrics.acc_exact_trt": None,
    })
    return result


@router.post("/projects/{project_id}/models/{model_id}/export-tensorrt")
async def export_model_tensorrt(
    project_id: str, model_id: str,
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    model = await _require_model(repo, project_id, model_id)
    if not model.onnx_path or not Path(model.onnx_path).is_file():
        raise HTTPException(400, "No ONNX export for this model yet — export ONNX first")
    loop = asyncio.get_event_loop()

    def _build():
        with gpu_lock.gpu_lock(f"paddle-ocr-trt:{model_id}"):
            return paddle_export.build_engine(project_id, model_id, Path(model.onnx_path))

    try:
        result = await loop.run_in_executor(None, _build)
    except Exception as e:
        logger.exception(f"[paddle export] TensorRT build failed for {model_id}")
        raise HTTPException(500, f"TensorRT build failed: {e}")
    await repo.update_model_record(model_id, {"engine_path": result["engine_path"]})
    return result


@router.post("/projects/{project_id}/models/{model_id}/evaluate")
async def evaluate_model(
    project_id: str, model_id: str,
    engine: str = Query("tensorrt", description="tensorrt | onnx"),
    with_thumbs: bool = Query(False),
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    if engine not in ("tensorrt", "onnx"):
        raise HTTPException(400, "engine must be 'tensorrt' or 'onnx'")
    model = await _require_model(repo, project_id, model_id)
    if not model.dict_path:
        raise HTTPException(400, "Model has no dict — export ONNX first")
    items = await eval_items_for(repo, project_id, model)
    if not items:
        raise HTTPException(400, "No verified items in the test split — nothing to evaluate against.")

    loop = asyncio.get_event_loop()

    def _run():
        args = (project_id,
                Path(model.onnx_path) if model.onnx_path else None,
                Path(model.engine_path) if model.engine_path else None,
                Path(model.dict_path), items, engine)
        if engine == "tensorrt":
            with gpu_lock.gpu_lock(f"paddle-ocr-eval:{model_id}"):
                return paddle_export.evaluate(*args)
        return paddle_export.evaluate(*args)

    try:
        result = await loop.run_in_executor(None, _run)
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception(f"[paddle eval] failed for {model_id}")
        raise HTTPException(500, f"Evaluation failed: {e}")

    s = result["scores"]
    suffix = "trt" if engine == "tensorrt" else "onnx"
    await repo.update_model_record(model_id, {
        f"metrics.acc_{suffix}": s["norm"], f"metrics.acc_exact_{suffix}": s["exact"],
    })
    if with_thumbs:
        from app.services import dataset_fs
        for entry in result["items"]:
            img = cv.imread(str(dataset_fs.image_abs_path(project_id, entry["image_path"])))
            ok, buf = cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, 80]) if img is not None else (False, None)
            entry["thumb_b64"] = base64.b64encode(buf.tobytes()).decode() if ok else ""
    result["train_metrics"] = {"acc": model.metrics.acc, "best_epoch": model.metrics.best_epoch}
    return result


@router.post("/projects/{project_id}/models/{model_id}/predict")
async def predict_image(
    project_id: str, model_id: str,
    file: UploadFile = File(...),
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Read one uploaded crop with the ONNX export and production
    preprocessing (CPU — no GPU context in this process)."""
    model = await _require_model(repo, project_id, model_id)
    if not model.onnx_path or not model.dict_path:
        raise HTTPException(400, "No ONNX export for this model yet")
    raw = await file.read()
    img = cv.imdecode(np.frombuffer(raw, np.uint8), cv.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode the uploaded image")
    loop = asyncio.get_event_loop()
    (res,), ms = await loop.run_in_executor(None, lambda: paddle_export.recognize_onnx(
        Path(model.onnx_path), Path(model.dict_path), [img],
    ))
    return {
        "engine": "onnx",
        "text": res[0],
        "confidence": res[1],
        "inference_ms": round(ms, 2),
        "image_b64": base64.b64encode(raw).decode(),
        "size": [img.shape[1], img.shape[0]],
    }


@router.get("/projects/{project_id}/models/{model_id}/export/{artifact}")
async def download_artifact(
    project_id: str, model_id: str, artifact: str,
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Download onnx | engine | dict | checkpoint."""
    model = await _require_model(repo, project_id, model_id, completed=False)
    field = {
        "onnx": ("onnx_path", f"{model.arch}_{model_id}.onnx"),
        "engine": ("engine_path", f"{model.arch}_{model_id}_fp16.engine"),
        "dict": ("dict_path", "dict.txt"),
        "checkpoint": ("checkpoint_path", f"{model_id}.pdparams"),
    }.get(artifact)
    if field is None:
        raise HTTPException(400, "artifact must be onnx | engine | dict | checkpoint")
    path = getattr(model, field[0], None)
    if not path or not Path(path).is_file():
        raise HTTPException(404, f"No {artifact} for this model yet")
    return FileResponse(path, filename=field[1], media_type="application/octet-stream")
