"""
OCR model training — start / list / status / logs / cancel / delete, plus the
base-checkpoint picker the Train tab's dropdown reads.

Same shape as anomaly_service's train endpoints: a FastAPI BackgroundTask runs
the job, progress and phase land on the model record, and the FE polls a log
endpoint with a `since` cursor.
"""
import asyncio
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.api.dependencies.auth import CurrentUser, get_current_user
from app.core.config import BASE_CKPT_DIR, BUILTIN_BASES, DEFAULT_BASE
from app.db.mongodb import get_database
from app.models.ocr import OCRTrainRequest
from app.repositories.ocr_repository import OCRRepository
from app.api.endpoints.eval import eval_items_for, evaluate_sync
from app.api.endpoints.export import build_engine_sync, export_onnx_sync
from app.services import dataset_fs, gpu_lock, ocr_inference, ocr_training
from app.services.dataset_builder import blocking_reason, build_dataset
from app.services.train_logs import attach_handler, detach_handler, get_buffer

logger = logging.getLogger(__name__)

router = APIRouter()

# model_id -> cancel requested. Lives in the process, like anomaly_service's:
# a cancel only means anything while the task that reads it is alive, and a
# restart is handled by reset_stuck_training instead.
_CANCEL_FLAGS: Dict[str, bool] = {}

# Where the bar sits when the checkpoint exists but export has not run.
TRAIN_DONE_PROGRESS = 85.0


def get_repo(db=Depends(get_database)) -> OCRRepository:
    return OCRRepository(db)


@router.get("/base-checkpoints")
async def list_base_checkpoints(
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Everything a run can fine-tune from: the built-in bases plus every
    completed model, grouped by project.

    Models from OTHER projects are included on purpose — a broad "all datecodes"
    project makes a good starting point for a narrow one, and the common request
    is "continue from last week's model" after adding images.
    """
    builtin = []
    for key, filename in BUILTIN_BASES.items():
        path = BASE_CKPT_DIR / filename
        builtin.append({
            "id": key,
            "filename": filename,
            "available": path.is_file(),
            "recommended": key == DEFAULT_BASE,
            # Both shipped bases predate use_space_char; a space run widens them
            # on the fly (see ckpt_vocab.needs_expansion).
            "use_space_char": False,
            "vocab_size": 99,
        })

    models = await repo.list_completed_models()
    projects = {p.id: p.name for p in await repo.list_projects()}
    grouped: Dict[str, Dict] = {}
    for m in models:
        if not m.checkpoint_path or not Path(m.checkpoint_path).is_file():
            continue  # row survived, file did not — not selectable
        g = grouped.setdefault(m.project_id, {
            "project_id": m.project_id,
            "project_name": projects.get(m.project_id, "(deleted project)"),
            "models": [],
        })
        g["models"].append({
            "model_id": m.id,
            "label": _model_label(m),
            "use_space_char": m.use_space_char,
            "vocab_size": m.vocab_size,
            "created_at": m.created_at.isoformat(),
            "min_acc": m.metrics.min_acc,
        })
    return {"builtin": builtin, "projects": list(grouped.values())}


def _model_label(m) -> str:
    acc = m.metrics.min_acc
    acc_s = f"{acc * 100:.1f}%" if acc is not None else "n/a"
    return f"min_acc {acc_s} · {m.created_at.strftime('%d/%m %H:%M')}"


@router.post("/projects/{project_id}/train")
async def start_training(
    project_id: str,
    request: OCRTrainRequest,
    background_tasks: BackgroundTasks,
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    # Resolve the base first: a missing checkpoint or a vocab conflict should be
    # a 4xx on the button press, not a failed run discovered minutes later.
    model_lookup = None
    if request.base.kind == "model":
        if not request.base.model_id:
            raise HTTPException(400, "base.kind='model' requires base.model_id")
        base_model = await repo.get_model(request.base.model_id)
        if not base_model:
            raise HTTPException(404, f"Base model {request.base.model_id} not found")
        if base_model.status != "completed":
            raise HTTPException(409, f"Base model status is '{base_model.status}', not completed")
        model_lookup = {
            "checkpoint_path": base_model.checkpoint_path,
            "id": base_model.id,
            "label": f"{base_model.project_id[:6]}·{_model_label(base_model)}",
        }
        # Narrowing the vocabulary would mean deleting the space class and every
        # weight that learned it. Almost certainly a mis-click.
        if base_model.use_space_char and not request.use_space_char:
            raise HTTPException(
                400,
                "This base was trained with use_space_char=True; training from it "
                "with spaces disabled would drop the space class it learned. "
                "Enable use_space_char, or pick a different base.",
            )
    try:
        base = ocr_training.resolve_base(request.base, model_lookup)
    except ocr_training.TrainingFailed as e:
        raise HTTPException(400, str(e))

    # Same validation the run will do, so an unusable dataset fails here with a
    # readable reason instead of after the GPU lock is taken.
    items = await repo.list_trainable_items(project_id)
    report = build_dataset(
        project_id, items,
        test_split=request.test_split,
        use_space_char=request.use_space_char,
        max_text_length=request.max_text_length,
        dry_run=True,
    )
    reason = blocking_reason(report)
    if reason:
        raise HTTPException(400, reason)

    params = request.model_dump()
    params["expanded_from"] = None
    model = await repo.create_model_record(
        project_id, params, base_label=base["label"], use_space_char=request.use_space_char,
    )
    await repo.update_model_record(model.id, {"status": "training", "phase": "queued"})
    await repo.set_status(project_id, "training")

    background_tasks.add_task(
        _run_training_bg, repo, project_id, model.id, request, base["path"], report,
    )
    return {
        "model_id": model.id,
        "status": "training",
        "n_train": report["n_train"],
        "n_test": report["n_test"],
        "dropped_count": report["dropped_count"],
        "gpu_holder": gpu_lock.current_holder(),
    }


async def _run_training_bg(
    repo: OCRRepository,
    project_id: str,
    model_id: str,
    request: OCRTrainRequest,
    base_path: Path,
    report: Dict,
):
    buf = get_buffer()
    buf.register(project_id, model_id)
    handler = attach_handler(model_id)
    buf.push(model_id, f"[ocr] starting run {model_id} (base={base_path.name})")
    buf.push(
        model_id,
        f"[ocr] dataset: {report['n_train']} train / {report['n_test']} test"
        + (f", {report['dropped_count']} dropped by validation" if report["dropped_count"] else "")
    )
    if report["unknown_chars"]:
        buf.push(
            model_id,
            f"[ocr] warning: {report['unknown_char_count']} label(s) contain characters "
            f"outside the dict {report['unknown_chars']} — they will be stripped silently",
            level="WARNING",
        )

    loop = asyncio.get_event_loop()
    _CANCEL_FLAGS[model_id] = False

    def _is_cancelled() -> bool:
        return _CANCEL_FLAGS.get(model_id, False)

    _last = {"phase": None, "pct": -100.0}

    def _progress(phase: str, pct: float) -> None:
        if _last["phase"] != phase or pct - _last["pct"] >= 10.0 or pct >= 100.0:
            buf.push(model_id, f"[phase] {phase} ({pct:.0f}%)")
            _last["phase"], _last["pct"] = phase, pct
        try:
            asyncio.run_coroutine_threadsafe(
                repo.update_model_record(model_id, {"phase": phase, "progress": pct}), loop,
            )
        except Exception:
            logger.exception("[ocr train] failed to schedule progress update")

    def _on_wait(waited: float) -> None:
        if waited < 1.0 or int(waited) % 30 == 0:
            buf.push(model_id, f"[ocr] waiting for the GPU ({gpu_lock.current_holder()})")
        try:
            asyncio.run_coroutine_threadsafe(
                repo.update_model_record(model_id, {"phase": "waiting_for_gpu"}), loop,
            )
        except Exception:
            pass

    def _blocking_run() -> Dict:
        # Written for real inside the lock, not from the dry run above: the
        # verified set can change between pressing Train and the GPU freeing up.
        with gpu_lock.gpu_lock(f"ocr-train:{model_id}", on_wait=_on_wait):
            return ocr_training.train_model(
                project_id, model_id, request, base_path,
                progress_cb=_progress, cancel_check=_is_cancelled,
            )

    try:
        items = await repo.list_trainable_items(project_id)
        build_dataset(
            project_id, items,
            test_split=request.test_split,
            use_space_char=request.use_space_char,
            max_text_length=request.max_text_length,
            dry_run=False,
        )
        result = await loop.run_in_executor(None, _blocking_run)

        metrics = result["metrics"]
        # NOT 'completed' yet: export still has to run, and the FE treats
        # status=='completed' as "the run is finished, artifacts are ready".
        # Flipping it here made a model look done while its engine was still
        # building, so the export endpoints would 400 on a model the UI showed
        # as ready.
        await repo.update_model_record(model_id, {
            "phase": "trained",
            "progress": TRAIN_DONE_PROGRESS,
            "checkpoint_path": result["checkpoint_path"],
            "config_path": result["config_path"],
            "vocab_size": result["vocab_size"],
            "params.expanded_from": result["expanded_from"],
            "metrics": {
                "acc": metrics.get("acc"),
                "gtc_acc": metrics.get("gtc_acc"),
                "min_acc": metrics.get("min_acc"),
                "norm_edit_dis": metrics.get("norm_edit_dis"),
                "best_epoch": metrics.get("best_epoch"),
                "n_train": report["n_train"],
                "n_test": report["n_test"],
            },
        })
        buf.push(
            model_id,
            f"[ocr] trained — min_acc={metrics.get('min_acc')} acc={metrics.get('acc')} "
            f"gtc_acc={metrics.get('gtc_acc')} best_epoch={metrics.get('best_epoch')}",
        )

        # Auto-export so Eval/Export/Test need no manual step. Each stage is
        # best-effort and independent: the checkpoint is already saved and the
        # run counts as completed either way, and TensorRT only runs if the ONNX
        # it depends on succeeded.
        await _auto_export(repo, project_id, model_id, result, buf)

        await repo.update_model_record(model_id, {
            "status": "completed", "phase": "completed", "progress": 100.0,
            "completed_at": datetime.utcnow(),
        })
        await repo.set_status(project_id, "trained")
        buf.push(model_id, "[ocr] run complete")

    except ocr_training.TrainingCancelled:
        buf.push(model_id, "[ocr] cancelled by user", level="WARNING")
        await repo.update_model_record(model_id, {
            "status": "cancelled", "phase": "cancelled", "error": "Cancelled by user",
        })
        await repo.set_status(project_id, "active")

    except Exception as e:
        logger.exception(f"[ocr train] run {model_id} failed")
        buf.push(model_id, f"[ocr] FAILED: {e}", level="ERROR")
        await repo.update_model_record(model_id, {
            "status": "failed", "phase": "failed", "error": str(e),
        })
        await repo.set_status(project_id, "active")

    finally:
        detach_handler(handler)
        _CANCEL_FLAGS.pop(model_id, None)


async def _auto_export(repo: OCRRepository, project_id: str, model_id: str, result: Dict, buf) -> None:
    """ONNX → TensorRT → measure acc_trt, after a successful run.

    Progress runs 85→100% here so the bar doesn't sit at 100% through a
    40-second engine build.
    """
    loop = asyncio.get_event_loop()
    onnx_fp16: Optional[str] = None

    try:
        await repo.update_model_record(model_id, {"phase": "exporting_onnx", "progress": 88.0})
        buf.push(model_id, "[ocr] exporting ONNX...")
        exported = await loop.run_in_executor(
            None,
            lambda: export_onnx_sync(
                project_id, model_id,
                Path(result["config_path"]), Path(result["checkpoint_path"]),
            ),
        )
        onnx_fp16 = exported["onnx_fp16_path"]
        await repo.update_model_record(model_id, {
            "onnx_path": exported["onnx_path"], "onnx_fp16_path": onnx_fp16,
        })
        buf.push(model_id, f"[ocr] ONNX ok — gtc {exported['gtc_shape']} ctc {exported['ctc_shape']}")
    except Exception as e:
        logger.exception(f"[ocr train] ONNX export failed for {model_id}")
        buf.push(model_id, f"[ocr] ONNX export failed (training still succeeded): {e}",
                 level="WARNING")
        return

    if not onnx_fp16:
        return
    try:
        await repo.update_model_record(model_id, {"phase": "exporting_tensorrt", "progress": 92.0})
        buf.push(model_id, "[ocr] building TensorRT engine...")
        built = await loop.run_in_executor(
            None, lambda: build_engine_sync(project_id, model_id, Path(onnx_fp16), True),
        )
        await repo.update_model_record(model_id, {
            "engine_path": built["engine_path"], "dict_path": built["dict_path"],
        })
        buf.push(model_id, f"[ocr] engine ok — {built['size_mb']} MB, "
                           f"outputs={[o['name'] for o in built['outputs']]}")
    except Exception as e:
        logger.exception(f"[ocr train] TensorRT build failed for {model_id}")
        buf.push(model_id, f"[ocr] TensorRT build failed (ONNX still exported): {e}",
                 level="WARNING")
        return

    # Score the engine, because the checkpoint's accuracy is not the engine's.
    # fp16 conversion and the export path can both lose something, and this is
    # the number the recipe picker gates on.
    try:
        await repo.update_model_record(model_id, {"phase": "measuring_engine", "progress": 96.0})
        model = await repo.get_model(model_id)
        items = await eval_items_for(repo, project_id, model)
        if not items:
            buf.push(model_id, "[ocr] no test split — skipping engine accuracy", level="WARNING")
            return
        evaluated = await loop.run_in_executor(
            None, lambda: evaluate_sync(project_id, model, items, "tensorrt"),
        )
        s = evaluated["scores"]
        await repo.update_model_record(model_id, {
            "metrics.acc_trt": s["norm_either"],
            "metrics.acc_exact_trt": s["exact_either"],
        })
        buf.push(
            model_id,
            f"[ocr] engine accuracy (batch=1): normalized {s['norm_either']:.4f} "
            f"exact {s['exact_either']:.4f} over {s['n']} test images, "
            f"{evaluated['ms_per_image']} ms/img",
        )
        train_acc = (model.metrics.min_acc or 0.0)
        if train_acc and s["norm_either"] < train_acc - 0.02:
            buf.push(
                model_id,
                f"[ocr] WARNING: engine is {(train_acc - s['norm_either']) * 100:.1f}pp below the "
                f"checkpoint ({train_acc:.4f}) — export may have lost accuracy",
                level="WARNING",
            )
    except Exception as e:
        logger.exception(f"[ocr train] engine evaluation failed for {model_id}")
        buf.push(model_id, f"[ocr] engine accuracy check failed: {e}", level="WARNING")


@router.get("/projects/{project_id}/models")
async def list_models(
    project_id: str,
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    models = await repo.list_models(project_id)
    return [m.model_dump(by_alias=False) for m in models]


@router.get("/projects/{project_id}/models/{model_id}/status")
async def get_model_status(
    project_id: str,
    model_id: str,
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    model = await repo.get_model(model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")
    return model.model_dump(by_alias=False)


@router.get("/projects/{project_id}/models/{model_id}/logs")
async def get_training_logs(
    project_id: str,
    model_id: str,
    since: int = 0,
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    model = await repo.get_model(model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")
    # No-op when this process already holds the log; otherwise lazy-loads it
    # from disk, which is what makes a run's log survive a service restart.
    get_buffer().register(project_id, model_id)
    logs, next_since = get_buffer().get_since(model_id, since)
    return {
        "logs": logs,
        "next_since": next_since,
        "phase": model.phase,
        "progress": float(model.progress or 0.0),
        "status": model.status,
        "error": model.error,
    }


@router.post("/projects/{project_id}/models/{model_id}/cancel")
async def cancel_training(
    project_id: str,
    model_id: str,
    repo: OCRRepository = Depends(get_repo),
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
        get_buffer().push(model_id, "[ocr] cancel requested", level="WARNING")
        return {"ok": True, "model_id": model_id, "mode": "cooperative"}

    # No task in this process — the service restarted mid-run. Force-fail the
    # record so the project isn't stuck showing a run nothing will finish.
    await repo.update_model_record(model_id, {
        "status": "failed", "phase": "failed", "progress": 0.0,
        "error": "Stuck record reset by user (no in-process task)",
    })
    await repo.set_status(project_id, "active")
    return {"ok": True, "model_id": model_id, "mode": "force_failed"}


@router.delete("/projects/{project_id}/models/{model_id}")
async def delete_model(
    project_id: str,
    model_id: str,
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Delete the record and every file the run wrote. Checkpoints are 84 MB
    and exports another ~150 MB, so hiding the row without reclaiming the disk
    is the wrong trade here."""
    model = await repo.get_model(model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")
    if model.status in ("training", "pending"):
        raise HTTPException(409, "Cancel this run before deleting it")

    models_dir = dataset_fs.models_dir(project_id)
    for p in (
        models_dir / f"{model_id}.pth",
        models_dir / f"{model_id}_config.yml",
        models_dir / f"{model_id}_base_space.pth",
    ):
        p.unlink(missing_ok=True)
    shutil.rmtree(models_dir / f"{model_id}_run", ignore_errors=True)
    shutil.rmtree(dataset_fs.export_dir(project_id, model_id), ignore_errors=True)
    get_buffer().drop(project_id, model_id)
    # Without this the deleted model's engine stays resident on the GPU until
    # the service restarts, and a re-export under the same id would keep
    # serving the stale one.
    ocr_inference.release_model(model_id)

    await repo.delete_model(model_id)
    return {"ok": True, "model_id": model_id}
