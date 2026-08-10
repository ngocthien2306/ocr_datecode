"""
Evaluate an exported model against the project's test split, and run it on a
single uploaded image.

Everything here scores through ai_services' own recognizer classes at batch=1 —
see ocr_inference for why both of those choices matter.
"""
import asyncio
import base64
import logging
import time
from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.api.dependencies.auth import CurrentUser, get_current_user
from app.db.mongodb import get_database
from app.repositories.ocr_repository import OCRRepository
from app.services import dataset_fs, gpu_lock, ocr_inference
from app.services.dataset_builder import build_dataset
from app.services.inspection_crop import img_to_b64

logger = logging.getLogger(__name__)

router = APIRouter()


def get_repo(db=Depends(get_database)) -> OCRRepository:
    return OCRRepository(db)


async def eval_items_for(repo: OCRRepository, project_id: str, model) -> list:
    """The test items this model was actually measured on.

    Runs the same validation + split as the training run did, using the run's own
    recorded params, rather than filtering by the stored `split` field. Filtering
    by split alone re-includes labels that validation dropped (over-long ones,
    which training never saw and no model can reproduce) and would report an
    accuracy over a set that differs from the one in the training log.
    """
    params = model.params or {}
    items = await repo.list_trainable_items(project_id)
    report = build_dataset(
        project_id, items,
        test_split=float(params.get("test_split", 0.2)),
        use_space_char=bool(params.get("use_space_char", True)),
        max_text_length=int(params.get("max_text_length", 25)),
        dry_run=True,
    )
    return report["test_items"]


def _artifact_for(model, engine: str) -> Path:
    path = model.engine_path if engine == "tensorrt" else (model.onnx_fp16_path or model.onnx_path)
    if not path or not Path(path).is_file():
        raise HTTPException(400, f"No {engine} export for this model yet")
    return Path(path)


def evaluate_sync(project_id: str, model, items, engine: str) -> dict:
    """Score every test item. Runs in a thread; takes the GPU lock because a
    TensorRT context and a training run do not share the card politely."""
    paths = [dataset_fs.image_abs_path(project_id, i.image_path) for i in items]
    labels = [i.gt_text for i in items]
    artifact = _artifact_for(model, engine)
    dict_path = Path(model.dict_path) if model.dict_path else None

    def _score():
        rec = ocr_inference.load_recognizer(artifact, engine, dict_path)
        t0 = time.perf_counter()
        preds = ocr_inference.recognize_paths(rec, paths, batch=1)
        return preds, (time.perf_counter() - t0) * 1000

    with gpu_lock.gpu_lock(f"ocr-eval:{model.id}"):
        # On the single CUDA-owning thread — see ocr_inference.run_on_gpu_thread.
        preds, elapsed_ms = ocr_inference.run_on_gpu_thread(_score)

    scores = ocr_inference.score_against_labels(preds, labels)
    per_image = []
    for item, (gtc, gtc_c, ctc, ctc_c) in zip(items, preds):
        norm_gt = ocr_inference.normalize(item.gt_text)
        per_image.append({
            "id": item.id,
            "gt_text": item.gt_text,
            "gtc_text": gtc,
            "gtc_conf": gtc_c,
            "ctc_text": ctc,
            "ctc_conf": ctc_c,
            "correct_norm": norm_gt in (ocr_inference.normalize(gtc), ocr_inference.normalize(ctc)),
            "correct_exact": item.gt_text in (gtc, ctc),
            "image_path": item.image_path,
        })
    return {
        "engine": engine,
        "scores": scores,
        "ms_per_image": round(elapsed_ms / max(len(paths), 1), 2),
        "items": per_image,
    }


@router.post("/projects/{project_id}/models/{model_id}/evaluate")
async def evaluate_model(
    project_id: str,
    model_id: str,
    engine: str = Query("tensorrt", description="tensorrt | onnx"),
    with_thumbs: bool = Query(False, description="include a thumbnail per item"),
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Run the exported model over the test split and store the accuracy.

    Two numbers per head, and they are not interchangeable. `norm_*` matches
    what train_rec.py prints and what production's compare_texts() effectively
    does (it strips spaces and punctuation). `exact_*` is raw string equality,
    which is much lower for a model trained without the space class — it cannot
    emit spaces at all.
    """
    if engine not in ("tensorrt", "onnx"):
        raise HTTPException(400, "engine must be 'tensorrt' or 'onnx'")
    model = await repo.get_model(model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")

    items = await eval_items_for(repo, project_id, model)
    if not items:
        raise HTTPException(
            400,
            "No verified items in the test split — nothing to evaluate against. "
            "Pin some items to test, or re-run prepare with test_split > 0.",
        )

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, lambda: evaluate_sync(project_id, model, items, engine),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[ocr eval] failed for {model_id}")
        raise HTTPException(500, f"Evaluation failed: {e}")

    s = result["scores"]
    key = "acc_trt" if engine == "tensorrt" else "acc_onnx"
    update = {f"metrics.{key}": s["norm_either"]}
    if engine == "tensorrt":
        update["metrics.acc_exact_trt"] = s["exact_either"]
    await repo.update_model_record(model_id, update)

    if with_thumbs:
        for entry in result["items"]:
            img = cv.imread(str(dataset_fs.image_abs_path(project_id, entry["image_path"])))
            entry["thumb_b64"] = img_to_b64(img, quality=80) if img is not None else ""

    # The trained checkpoint's own accuracy, for the comparison that matters:
    # a big gap between PyTorch and the engine means the export lost something,
    # not that the model is bad.
    result["train_metrics"] = {
        "min_acc": model.metrics.min_acc,
        "acc": model.metrics.acc,
        "gtc_acc": model.metrics.gtc_acc,
    }
    return result


@router.post("/projects/{project_id}/models/{model_id}/predict")
async def predict_image(
    project_id: str,
    model_id: str,
    file: UploadFile = File(...),
    engine: str = Query("tensorrt", description="tensorrt | onnx"),
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Read one uploaded crop. Recognizers are cached per artifact, so
    inference_ms reflects steady-state speed rather than model load time."""
    if engine not in ("tensorrt", "onnx"):
        raise HTTPException(400, "engine must be 'tensorrt' or 'onnx'")
    model = await repo.get_model(model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")

    raw = await file.read()
    img = cv.imdecode(np.frombuffer(raw, np.uint8), cv.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode the uploaded image")

    artifact = _artifact_for(model, engine)
    dict_path = Path(model.dict_path) if model.dict_path else None
    loop = asyncio.get_event_loop()

    def _infer():
        rec = ocr_inference.load_recognizer(artifact, engine, dict_path)
        t0 = time.perf_counter()
        return rec.recognize(img), (time.perf_counter() - t0) * 1000

    def _run():
        with gpu_lock.gpu_lock(f"ocr-predict:{model_id}"):
            return ocr_inference.run_on_gpu_thread(_infer)

    try:
        (gtc, ctc), ms = await loop.run_in_executor(None, _run)
    except Exception as e:
        logger.exception(f"[ocr predict] failed for {model_id}")
        raise HTTPException(500, f"Inference failed: {e}")

    return {
        "engine": engine,
        "gtc_text": gtc[0],
        "gtc_conf": float(gtc[1]),
        "ctc_text": ctc[0],
        "ctc_conf": float(ctc[1]),
        "inference_ms": round(ms, 2),
        "image_b64": base64.b64encode(raw).decode(),
        "size": [img.shape[1], img.shape[0]],
    }
