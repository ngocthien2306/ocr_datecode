"""
Export — ONNX (via anomalib Engine.export) + a TensorRT "verify" step that
builds/caches a real TensorRT engine from that ONNX file on THIS machine
(train + inference share the same GPU workstation — see
docs/anomaly_training_plan.md) and confirms it runs, using the same
onnxruntime TensorrtExecutionProvider lazy-build/cache pattern backend's
ml_training_service.py already uses for the SupCon embedder.
"""
import logging
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.api.dependencies.auth import CurrentUser, get_current_user
from app.db.mongodb import get_database
from app.repositories.anomaly_repository import AnomalyRepository
from app.services import dataset_fs
from app.services.anomaly_training import export_onnx

logger = logging.getLogger(__name__)

router = APIRouter()


def get_repo(db=Depends(get_database)) -> AnomalyRepository:
    return AnomalyRepository(db)


def _export_dir(project_id: str, model_id: str) -> Path:
    return dataset_fs.models_dir(project_id) / "export" / model_id


def _trt_cache_dir(project_id: str, model_id: str) -> Path:
    d = dataset_fs.models_dir(project_id) / "export" / model_id / "trt_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


@router.post("/projects/{project_id}/models/{model_id}/export-onnx", tags=["Anomaly Export"])
async def export_model_onnx(
    project_id: str,
    model_id: str,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    model = await repo.get_model(model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")
    if model.status != "completed" or not model.checkpoint_path:
        raise HTTPException(400, f"Model status is '{model.status}' — train to completion first")
    if not Path(model.checkpoint_path).exists():
        raise HTTPException(500, "Checkpoint file missing on disk")

    image_size = int(model.params.get("image_size", 256))
    export_dir = _export_dir(project_id, model_id)

    try:
        onnx_path = export_onnx(model.algorithm, Path(model.checkpoint_path), export_dir, image_size)
    except Exception as e:
        logger.exception(f"[export] ONNX export failed for model {model_id}")
        raise HTTPException(500, f"Export failed: {e}")

    await repo.update_model_record(model_id, {"onnx_path": str(onnx_path)})
    return {"onnx_path": str(onnx_path), "image_size": image_size}


@router.get("/projects/{project_id}/models/{model_id}/export/onnx", tags=["Anomaly Export"])
async def download_onnx(
    project_id: str,
    model_id: str,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    model = await repo.get_model(model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")
    if not model.onnx_path or not Path(model.onnx_path).exists():
        raise HTTPException(404, "No ONNX export for this model yet — export it first")
    return FileResponse(model.onnx_path, filename=f"anomaly_{model_id}.onnx", media_type="application/octet-stream")


@router.post("/projects/{project_id}/models/{model_id}/verify-tensorrt", tags=["Anomaly Export"])
async def verify_tensorrt(
    project_id: str,
    model_id: str,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Build (or load from cache) a TensorRT engine from the exported ONNX and
    run one dummy inference to confirm it works on this machine — same
    approach `ai_services` will use at live-inference time, so a pass here
    means the deployed model will load too.
    """
    import onnxruntime as ort

    model = await repo.get_model(model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")
    if not model.onnx_path or not Path(model.onnx_path).exists():
        raise HTTPException(400, "No ONNX export for this model yet — export it first")

    image_size = int(model.params.get("image_size", 256))
    cache_dir = _trt_cache_dir(project_id, model_id)
    cache_had_files_before = any(cache_dir.iterdir())

    available = set(ort.get_available_providers())
    if "TensorrtExecutionProvider" not in available:
        raise HTTPException(
            400,
            "TensorrtExecutionProvider not available in this onnxruntime build "
            f"(available: {sorted(available)})",
        )

    providers = [
        ("TensorrtExecutionProvider", {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(cache_dir),
            "trt_fp16_enable": True,
            "trt_max_workspace_size": 512 * 1024 * 1024,
        }),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    try:
        t0 = time.perf_counter()
        sess = ort.InferenceSession(model.onnx_path, providers=providers)
        build_ms = (time.perf_counter() - t0) * 1000

        x = np.random.rand(1, 3, image_size, image_size).astype(np.float32)
        input_name = sess.get_inputs()[0].name
        t1 = time.perf_counter()
        outputs = sess.run(None, {input_name: x})
        infer_ms = (time.perf_counter() - t1) * 1000
    except Exception as e:
        logger.exception(f"[export] TensorRT verify failed for model {model_id}")
        raise HTTPException(500, f"TensorRT verify failed: {e}")

    active_provider = sess.get_providers()[0]
    return {
        "active_provider": active_provider,
        "engine_cache_hit": cache_had_files_before,
        "build_or_load_ms": round(build_ms, 1),
        "inference_ms": round(infer_ms, 1),
        "output_shapes": [list(o.shape) for o in outputs],
        "cache_dir": str(cache_dir),
    }
