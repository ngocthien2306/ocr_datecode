"""
Anomaly model training — start/status/logs/cancel.
Mirrors backend's ml_training.py training endpoints' shape (background task
+ live-log polling), but talks to anomaly_training.py / train_logs.py.
"""
import asyncio
import logging
import shutil
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.api.dependencies.auth import CurrentUser, get_current_user
from app.db.mongodb import get_database
from app.models.anomaly import AnomalyTrainRequest
from app.repositories.anomaly_repository import AnomalyRepository
from app.services import dataset_fs
from app.services.anomaly_training import TrainingCancelled, export_onnx, train_model
from app.services.train_logs import attach_handler, detach_handler, get_buffer
from app.services.trt_export import build_engine_from_onnx, onnx_input_info

logger = logging.getLogger(__name__)

router = APIRouter()


def get_repo(db=Depends(get_database)) -> AnomalyRepository:
    return AnomalyRepository(db)


@router.post("/projects/{project_id}/train", tags=["Anomaly Training"])
async def start_training(
    project_id: str,
    request: AnomalyTrainRequest,
    background_tasks: BackgroundTasks,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    if project.normal_count == 0:
        raise HTTPException(
            400,
            "Need at least one 'normal' image imported before training.",
        )
    # abnormal_count == 0 is fine -- PatchCore/Padim fit on normal images only.
    # _build_datamodule/_compute_metrics already handle the no-abnormal case
    # (image_auroc/image_f1 just report as 0.0 / not meaningful until you
    # import some abnormal images to evaluate against).

    model_record = await repo.create_model_record(project_id, request.algorithm, request.model_dump())
    model_id = model_record.id

    await repo.set_status(project_id, "training")
    await repo.update_model_record(model_id, {"status": "training"})

    background_tasks.add_task(_run_training_bg, repo, project_id, model_id, request)

    return {"model_id": model_id, "status": "training"}


_CANCEL_FLAGS: Dict[str, bool] = {}


async def _run_training_bg(repo: AnomalyRepository, project_id: str, model_id: str, request: AnomalyTrainRequest):
    log_buffer = get_buffer()
    log_buffer.register(model_id)
    handler = attach_handler(model_id)
    log_buffer.push(model_id, f"[anomaly] Starting training for model {model_id} (algo={request.algorithm})")

    loop = asyncio.get_event_loop()
    _last_phase = [None]
    _last_logged_pct = [-100.0]

    def _progress_cb(phase: str, progress: float) -> None:
        progress = float(progress)
        if _last_phase[0] != phase or progress - _last_logged_pct[0] >= 10.0 or progress >= 100.0:
            log_buffer.push(model_id, f"[phase] {phase} ({progress:.0f}%)")
            _last_phase[0] = phase
            _last_logged_pct[0] = progress
        try:
            asyncio.run_coroutine_threadsafe(
                repo.update_model_record(model_id, {"phase": phase, "progress": progress}),
                loop,
            )
        except Exception:
            logger.exception("[anomaly train] Failed to schedule progress update")

    _CANCEL_FLAGS[model_id] = False

    def _is_cancelled() -> bool:
        return _CANCEL_FLAGS.get(model_id, False)

    checkpoint_dir = dataset_fs.models_dir(project_id)

    try:
        metrics = await loop.run_in_executor(
            None,
            lambda: train_model(
                project_id, model_id, request, checkpoint_dir,
                progress_cb=_progress_cb, cancel_check=_is_cancelled,
            ),
        )
        project = await repo.get_project(project_id)
        n_normal_train = max(0, (project.normal_count if project else 0) - metrics["n_normal_test"])
        await repo.update_model_record(model_id, {
            "status": "completed",
            "metrics": {
                "image_auroc": metrics["image_auroc"],
                "image_f1": metrics["image_f1"],
                "threshold": metrics["threshold"],
                "n_normal_train": n_normal_train,
                "n_normal_test": metrics["n_normal_test"],
                "n_abnormal_test": metrics["n_abnormal_test"],
            },
            "checkpoint_path": metrics["checkpoint_path"],
            "phase": "completed",
            "progress": 100.0,
        })
        await repo.set_status(project_id, "trained")
        log_buffer.push(model_id, f"[anomaly] Training completed (AUROC={metrics['image_auroc']:.3f})")

        # Auto-export ONNX then TensorRT so Eval/Export/Test don't need any
        # manual "Export" click first. Each step is best-effort and
        # independent: a failed export doesn't fail the training itself
        # (the checkpoint is already saved either way), and TensorRT only
        # runs if the ONNX export it depends on succeeded.
        export_dir = dataset_fs.models_dir(project_id) / "export" / model_id
        onnx_path = None
        try:
            log_buffer.push(model_id, "[anomaly] Auto-exporting ONNX...")
            onnx_path = await loop.run_in_executor(
                None,
                lambda: export_onnx(
                    request.algorithm, Path(metrics["checkpoint_path"]), export_dir, request.image_size,
                ),
            )
            await repo.update_model_record(model_id, {"onnx_path": str(onnx_path)})
            log_buffer.push(model_id, f"[anomaly] ONNX export complete: {onnx_path}")
        except Exception as e:
            logger.exception(f"[anomaly train] Auto ONNX export failed for model {model_id}")
            log_buffer.push(
                model_id, f"[anomaly] ONNX auto-export failed (training itself still succeeded): {e}",
                level="WARNING",
            )

        if onnx_path is not None:
            try:
                log_buffer.push(model_id, "[anomaly] Auto-exporting TensorRT engine...")
                engine_path = export_dir / "model.engine"
                input_name, _ = onnx_input_info(onnx_path)
                image_size = request.image_size
                await loop.run_in_executor(
                    None,
                    lambda: build_engine_from_onnx(
                        onnx_path, engine_path, input_name,
                        (1, 3, image_size, image_size), (1, 3, image_size, image_size),
                        (8, 3, image_size, image_size), fp16=True,
                    ),
                )
                await repo.update_model_record(model_id, {"engine_path": str(engine_path)})
                log_buffer.push(model_id, f"[anomaly] TensorRT export complete: {engine_path}")
            except Exception as e:
                logger.exception(f"[anomaly train] Auto TensorRT export failed for model {model_id}")
                log_buffer.push(
                    model_id, f"[anomaly] TensorRT auto-export failed (ONNX export still succeeded): {e}",
                    level="WARNING",
                )

    except TrainingCancelled:
        log_buffer.push(model_id, "[anomaly] Training cancelled by user", level="WARNING")
        await repo.update_model_record(model_id, {
            "status": "cancelled", "error": "Cancelled by user", "phase": "cancelled",
        })
        await repo.set_status(project_id, "active")

    except Exception as e:
        logger.exception(f"[anomaly train] Training failed for project {project_id}")
        log_buffer.push(model_id, f"[anomaly] Training failed: {e}", level="ERROR")
        await repo.update_model_record(model_id, {
            "status": "failed", "error": str(e), "phase": "failed",
        })
        await repo.set_status(project_id, "active")

    finally:
        detach_handler(handler)
        _CANCEL_FLAGS.pop(model_id, None)


@router.get("/projects/{project_id}/models", tags=["Anomaly Training"])
async def list_models(
    project_id: str,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    models = await repo.list_models(project_id)
    return [m.model_dump(by_alias=False) for m in models]


@router.get("/projects/{project_id}/models/{model_id}/status", tags=["Anomaly Training"])
async def get_model_status(
    project_id: str,
    model_id: str,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    model = await repo.get_model(model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")
    return model.model_dump(by_alias=False)


@router.get("/projects/{project_id}/models/{model_id}/logs", tags=["Anomaly Training"])
async def get_training_logs(
    project_id: str,
    model_id: str,
    since: int = 0,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    model = await repo.get_model(model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")
    logs, next_since = get_buffer().get_since(model_id, since)
    return {
        "logs": logs,
        "next_since": next_since,
        "phase": model.phase,
        "progress": float(model.progress or 0.0),
        "status": model.status,
        "error": model.error,
    }


@router.post("/projects/{project_id}/models/{model_id}/cancel", tags=["Anomaly Training"])
async def cancel_training(
    project_id: str,
    model_id: str,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    model = await repo.get_model(model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")
    if model.status not in ("training", "pending"):
        raise HTTPException(409, f"Model status is '{model.status}', not cancellable")

    if model_id in _CANCEL_FLAGS:
        _CANCEL_FLAGS[model_id] = True
        await repo.update_model_record(model_id, {"phase": "cancelling"})
        get_buffer().push(model_id, "[anomaly] Cancel requested", level="WARNING")
        return {"ok": True, "model_id": model_id, "mode": "cooperative"}

    # No in-process task (e.g. service restarted mid-train) — force-fail the
    # stuck record so the UI/project status aren't left hanging forever.
    await repo.update_model_record(model_id, {
        "status": "failed",
        "error": "Stuck record reset by user (no in-process task)",
        "phase": "failed",
        "progress": 0.0,
    })
    await repo.set_status(project_id, "active")
    return {"ok": True, "model_id": model_id, "mode": "force_failed"}


@router.delete("/projects/{project_id}/models/{model_id}", tags=["Anomaly Training"])
async def delete_model(
    project_id: str,
    model_id: str,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Delete a trained model's DB record and every file it wrote to disk
    (checkpoint, onnx/engine export, test-results sidecar) -- the whole
    point being to actually reclaim disk space, not just hide the row."""
    model = await repo.get_model(model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")
    if model.status in ("training", "pending"):
        raise HTTPException(409, "Cancel training before deleting this model")

    models_dir = dataset_fs.models_dir(project_id)
    for p in [
        models_dir / f"{model_id}.ckpt",
        models_dir / f"{model_id}_test_results.json",
    ]:
        p.unlink(missing_ok=True)
    shutil.rmtree(models_dir / "export" / model_id, ignore_errors=True)

    await repo.delete_model(model_id)
    return {"ok": True, "model_id": model_id}
