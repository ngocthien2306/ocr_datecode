"""
Export — ONNX (via anomalib Engine.export) + a standalone TensorRT .engine
built directly via the TensorRT Python Builder API (app/services/trt_export.py),
verified by deserializing it with tensorrt.Runtime on this machine.
"""
import logging
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.api.dependencies.auth import CurrentUser, get_current_user
from app.db.mongodb import get_database
from app.repositories.anomaly_repository import AnomalyRepository
from app.services import dataset_fs
from app.services.anomaly_training import export_onnx
from app.services.trt_export import build_engine_from_onnx, onnx_input_info

logger = logging.getLogger(__name__)

router = APIRouter()


def get_repo(db=Depends(get_database)) -> AnomalyRepository:
    return AnomalyRepository(db)


def _export_dir(project_id: str, model_id: str) -> Path:
    return dataset_fs.models_dir(project_id) / "export" / model_id


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


def _engine_path(project_id: str, model_id: str) -> Path:
    return _export_dir(project_id, model_id) / "model.engine"


@router.post("/projects/{project_id}/models/{model_id}/export-tensorrt", tags=["Anomaly Export"])
async def export_model_tensorrt(
    project_id: str,
    model_id: str,
    max_batch: int = 8,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Build a standalone, portable TensorRT .engine from the exported ONNX
    (dynamic batch 1..max_batch, fixed HxW = training image_size) --
    downloadable the same way OCR/pipeline engines ship in weights/, unlike
    verify-tensorrt's internal onnxruntime-managed engine cache below.
    """
    model = await repo.get_model(model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")
    if not model.onnx_path or not Path(model.onnx_path).exists():
        raise HTTPException(400, "No ONNX export for this model yet — export it first")
    if max_batch < 1:
        raise HTTPException(400, "max_batch must be >= 1")

    image_size = int(model.params.get("image_size", 256))
    onnx_path = Path(model.onnx_path)
    engine_path = _engine_path(project_id, model_id)

    try:
        input_name, dims = onnx_input_info(onnx_path)
        min_shape = (1, 3, image_size, image_size)
        opt_shape = (1, 3, image_size, image_size)
        max_shape = (max_batch, 3, image_size, image_size)

        logs: list = []
        build_engine_from_onnx(
            onnx_path, engine_path, input_name, min_shape, opt_shape, max_shape,
            fp16=True, log=logs.append,
        )
        for line in logs:
            logger.info(f"[export-tensorrt] {line}")
    except Exception as e:
        logger.exception(f"[export] TensorRT engine export failed for model {model_id}")
        raise HTTPException(500, f"TensorRT export failed: {e}")

    await repo.update_model_record(model_id, {"engine_path": str(engine_path)})
    return {
        "engine_path": str(engine_path),
        "input_name": input_name,
        "min_shape": list(min_shape), "opt_shape": list(opt_shape), "max_shape": list(max_shape),
    }


@router.get("/projects/{project_id}/models/{model_id}/export/tensorrt", tags=["Anomaly Export"])
async def download_tensorrt(
    project_id: str,
    model_id: str,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    model = await repo.get_model(model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")
    if not model.engine_path or not Path(model.engine_path).exists():
        raise HTTPException(404, "No TensorRT export for this model yet — export it first")
    return FileResponse(model.engine_path, filename=f"anomaly_{model_id}.engine", media_type="application/octet-stream")


@router.post("/projects/{project_id}/models/{model_id}/verify-tensorrt", tags=["Anomaly Export"])
async def verify_tensorrt(
    project_id: str,
    model_id: str,
    max_batch: int = 8,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Build (or reuse) the standalone TensorRT engine from the exported ONNX
    and confirm it deserializes with the expected I/O bindings on this
    machine's GPU -- same tensorrt.Runtime path as export-tensorrt below.

    Deliberately NOT onnxruntime's bundled TensorrtExecutionProvider: that
    EP is compiled against TensorRT 10.x (dlopens libnvinfer.so.10), while
    this service pins tensorrt==11.1.0.106 for the standalone .engine export
    (ships libnvinfer.so.11) -- the two major versions can't coexist here,
    so this endpoint doesn't touch onnxruntime's TRT EP at all anymore.
    """
    import tensorrt as trt

    model = await repo.get_model(model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")
    if not model.onnx_path or not Path(model.onnx_path).exists():
        raise HTTPException(400, "No ONNX export for this model yet — export it first")

    image_size = int(model.params.get("image_size", 256))
    engine_path = _engine_path(project_id, model_id)
    cache_hit = engine_path.exists()

    try:
        t0 = time.perf_counter()
        if not cache_hit:
            input_name, _ = onnx_input_info(Path(model.onnx_path))
            build_engine_from_onnx(
                Path(model.onnx_path), engine_path, input_name,
                (1, 3, image_size, image_size), (1, 3, image_size, image_size),
                (max_batch, 3, image_size, image_size), fp16=True,
            )
            await repo.update_model_record(model_id, {"engine_path": str(engine_path)})
        build_ms = (time.perf_counter() - t0) * 1000

        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError("Engine failed to deserialize")

        output_shapes = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                output_shapes.append(list(engine.get_tensor_shape(name)))
    except Exception as e:
        logger.exception(f"[export] TensorRT verify failed for model {model_id}")
        raise HTTPException(500, f"TensorRT verify failed: {e}")

    return {
        "active_provider": "TensorRT (standalone .engine)",
        "engine_cache_hit": cache_hit,
        "build_or_load_ms": round(build_ms, 1),
        "inference_ms": None,  # not run here -- no pycuda in this env; build+deserialize is the real check
        "output_shapes": output_shapes,
        "cache_dir": str(engine_path.parent),
    }
