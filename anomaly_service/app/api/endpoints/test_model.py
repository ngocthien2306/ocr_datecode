"""
Live "try it" inference — run a freshly uploaded image through a trained
model's exported ONNX or standalone TensorRT engine and return the anomaly
score + heatmap + inference time, for the Test tab (Roboflow-style "drop an
image, see the prediction" UI).

Distinct from eval.py (which only re-derives metrics from the *stored*
training-time test-set predictions) -- this actually runs the model again,
on demand, on whatever image the user hands it, through whichever backend
(onnx / tensorrt) the user picks -- so inference_ms is a real,
apples-to-apples comparison between the two.

Sessions/engines are cached per model_id (module-level dict) so repeated
Test-tab clicks measure steady-state inference time, not
session-creation/engine-load overhead. TensorRT execution uses torch's own
CUDA context for input/output buffers (context.set_tensor_address +
torch tensor .data_ptr()) instead of pycuda, which isn't installed in this
env (its build needs nvcc; torch is already here and already proven to
share a CUDA context fine with TensorRT and onnxruntime in this service).
"""
import logging
import time
from pathlib import Path
from typing import Dict

import cv2 as cv
import numpy as np
import tensorrt as trt
import torch
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.dependencies.auth import CurrentUser, get_current_user
from app.db.mongodb import get_database
from app.repositories.anomaly_repository import AnomalyRepository
from app.services.anomaly_training import encode_heatmap_overlay
from app.services.inspection_crop import img_to_b64

logger = logging.getLogger(__name__)

router = APIRouter()

_TRT_TORCH_DTYPE = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.BOOL: torch.bool,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
}

_onnx_sessions: Dict[str, "object"] = {}
_trt_state: Dict[str, dict] = {}
_warmed_up: set = set()  # {(model_id, engine)}


def get_repo(db=Depends(get_database)) -> AnomalyRepository:
    return AnomalyRepository(db)


def _get_onnx_session(model):
    import onnxruntime as ort

    sess = _onnx_sessions.get(model.id)
    if sess is None:
        sess = ort.InferenceSession(
            model.onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _onnx_sessions[model.id] = sess
    return sess


def _get_trt_state(model):
    state = _trt_state.get(model.id)
    if state is None:
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(model.engine_path, "rb") as f:
            engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        input_name = None
        output_names = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_name = name
            else:
                output_names.append(name)
        state = {
            "engine": engine, "context": context,
            "input_name": input_name, "output_names": output_names,
            "stream": torch.cuda.Stream(),
        }
        _trt_state[model.id] = state
    return state


def _run_onnx(model, x: np.ndarray) -> Dict[str, np.ndarray]:
    sess = _get_onnx_session(model)
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: x})
    return dict(zip((o.name for o in sess.get_outputs()), outputs)), sess.get_providers()[0]


def _run_tensorrt(model, x: np.ndarray) -> Dict[str, np.ndarray]:
    state = _get_trt_state(model)
    context = state["context"]
    stream = state["stream"]

    context.set_input_shape(state["input_name"], x.shape)
    x_t = torch.from_numpy(x).cuda().contiguous()
    context.set_tensor_address(state["input_name"], x_t.data_ptr())

    out_tensors = {}
    for name in state["output_names"]:
        shape = tuple(context.get_tensor_shape(name))
        tdtype = _TRT_TORCH_DTYPE[state["engine"].get_tensor_dtype(name)]
        t = torch.empty(shape, dtype=tdtype, device="cuda")
        out_tensors[name] = t
        context.set_tensor_address(name, t.data_ptr())

    with torch.cuda.stream(stream):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    return {k: v.cpu().numpy() for k, v in out_tensors.items()}, "TensorRT (standalone .engine)"


@router.post("/projects/{project_id}/models/{model_id}/predict", tags=["Anomaly Test"])
async def predict_image(
    project_id: str,
    model_id: str,
    engine: str = "onnx",
    file: UploadFile = File(...),
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    if engine not in ("onnx", "tensorrt"):
        raise HTTPException(400, "engine must be 'onnx' or 'tensorrt'")

    model = await repo.get_model(model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")
    if engine == "onnx" and (not model.onnx_path or not Path(model.onnx_path).exists()):
        raise HTTPException(400, "No ONNX export for this model yet — export it first")
    if engine == "tensorrt" and (not model.engine_path or not Path(model.engine_path).exists()):
        raise HTTPException(400, "No TensorRT export for this model yet — export it first (Export tab, step 3)")

    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv.imdecode(arr, cv.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode uploaded file as an image")

    image_size = int(model.params.get("image_size", 256))
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    resized = cv.resize(img_rgb, (image_size, image_size))
    # NOTE: no ImageNet mean/std normalization here -- anomalib's export_transform
    # (Resize + Normalize) is baked into the exported ONNX/TensorRT graph itself.
    # Verified against a real train/good image: applying normalization again here
    # flips a genuinely-normal image's pred_score from 0.0 to 1.0 (false abnormal).
    x = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

    run_fn = _run_onnx if engine == "onnx" else _run_tensorrt
    cache_key = (model_id, engine)
    try:
        if cache_key not in _warmed_up:
            run_fn(model, x)  # untimed warmup: first CUDA kernel launch / TRT context alloc is much slower
            _warmed_up.add(cache_key)

        t0 = time.perf_counter()
        out_map, active_provider = run_fn(model, x)
        inference_ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        logger.exception(f"[test] Predict failed for model {model_id} (engine={engine})")
        raise HTTPException(500, f"Inference failed: {e}")

    pred_score = float(np.asarray(out_map["pred_score"]).reshape(-1)[0])

    heatmap_b64 = ""
    amap = out_map.get("anomaly_map")
    if amap is not None:
        amap = np.asarray(amap)
        heatmap_b64 = encode_heatmap_overlay(img, amap.reshape(amap.shape[-2], amap.shape[-1]))

    return {
        "pred_score": round(pred_score, 4),
        "crop_b64": img_to_b64(img),
        "heatmap_b64": heatmap_b64,
        "image_size": image_size,
        "engine": engine,
        "active_provider": active_provider,
        "inference_ms": round(inference_ms, 2),
    }
