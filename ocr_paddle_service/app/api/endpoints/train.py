"""
PaddleOCR model training — start / list / status / logs / cancel / delete,
plus the architecture + base-weights picker the Train tab reads and the
trained-model list the recipe form reads.

Same shape as ocr_service's train endpoints (and the FE polls them the same
way): a FastAPI BackgroundTask runs the job, progress and phase land on the
model record, the FE polls a log endpoint with a `since` cursor.
"""
import asyncio
import logging
import shutil
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.api.dependencies.auth import CurrentUser, get_current_user
from app.core.config import DEFAULT_ARCH, PADDLE_ARCHS, arch_pretrained_path
from app.db.mongodb import get_database
from app.models.ocr import OCRTrainRequest
from app.repositories.ocr_repository import OCRRepository
from app.services import dataset_fs, gpu_lock, ocr_training, paddle_export, synthetic_defects
from app.services.dataset_builder import blocking_reason, build_dataset
from app.services.train_logs import attach_handler, detach_handler, get_buffer
from app.services.recipe_usage import in_use_message, recipes_using

logger = logging.getLogger(__name__)

router = APIRouter()

_CANCEL_FLAGS: Dict[str, bool] = {}

TRAIN_DONE_PROGRESS = 85.0


def get_repo(db=Depends(get_database)) -> OCRRepository:
    return OCRRepository(db)


# created_at is stored as naive UTC (datetime.utcnow()); labels are read by
# operators in Vietnam, so show factory local time.
_LOCAL_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def _local_time(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_LOCAL_TZ).strftime('%d/%m %H:%M')


def _model_label(m) -> str:
    acc = m.metrics.acc_trt if m.metrics.acc_trt is not None else m.metrics.acc
    acc_s = f"{acc * 100:.1f}%" if acc is not None else "n/a"
    return f"{m.arch.replace('_rec', '')} · acc {acc_s} · {_local_time(m.created_at)}"


@router.get("/archs")
async def list_archs(
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Architectures + what each can fine-tune from: its upstream pretrained
    weights, or any completed model of the same arch (any project)."""
    models = await repo.list_completed_models()
    projects = {p.id: p.name for p in await repo.list_projects()}
    out = []
    for key, spec in PADDLE_ARCHS.items():
        bases = []
        for m in models:
            if m.arch != key or not m.checkpoint_path or not Path(m.checkpoint_path).is_file():
                continue
            bases.append({
                "project_id": m.project_id,
                "project_name": projects.get(m.project_id, "(deleted project)"),
                "model_id": m.id,
                "label": _model_label(m),
            })
        out.append({
            "id": key,
            "label": spec["label"],
            "recommended": key == DEFAULT_ARCH,
            "pretrained_available": arch_pretrained_path(key).is_file(),
            "trained_bases": bases,
        })
    return out


@router.get("/trained-models")
async def list_trained_models(
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Completed models a recipe can run, grouped by project. Only models with
    a deployable artifact (engine or ONNX) plus dict are listed — the backend
    resolver rejects anything else anyway, and a selectable-but-dead entry is
    how a recipe silently falls back to the default OCR."""
    models = await repo.list_completed_models()
    projects = {p.id: p.name for p in await repo.list_projects()}
    grouped: Dict[str, Dict] = {}
    for m in models:
        has_engine = bool(m.engine_path and Path(m.engine_path).is_file())
        has_onnx = bool(m.onnx_path and Path(m.onnx_path).is_file())
        if not (has_engine or has_onnx) or not (m.dict_path and Path(m.dict_path).is_file()):
            continue
        g = grouped.setdefault(m.project_id, {
            "project_id": m.project_id,
            "project_name": projects.get(m.project_id, "(deleted project)"),
            "models": [],
        })
        g["models"].append({
            "model_id": m.id,
            "arch": m.arch,
            "label": _model_label(m),
            "runtime": "tensorrt" if has_engine else "onnx",
            "acc_trt": m.metrics.acc_trt,
            "acc_onnx": m.metrics.acc_onnx,
            "created_at": m.created_at.isoformat(),
        })
    return {"projects": list(grouped.values())}


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
    if request.arch not in PADDLE_ARCHS:
        raise HTTPException(400, f"Unknown arch '{request.arch}'")

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
            "arch": base_model.arch,
            "id": base_model.id,
            "label": f"{base_model.project_id[:6]}·{_model_label(base_model)}",
        }
    try:
        base = ocr_training.resolve_base(request.arch, request.base, model_lookup)
    except ocr_training.TrainingFailed as e:
        raise HTTPException(400, str(e))

    synth = []
    if request.use_synthetic:
        manifest = synthetic_defects.load_manifest(project_id)
        if not manifest or not manifest.get("n_train"):
            raise HTTPException(400, "use_synthetic is on but this project has no synthetic "
                                     "defect data — generate it first.")
        synth = manifest["entries"]

    items = await repo.list_trainable_items(project_id)
    report = build_dataset(
        project_id, items, arch=request.arch,
        test_split=request.test_split, max_text_length=request.max_text_length,
        exclude_item_ids=request.exclude_item_ids, dry_run=True, synth_entries=synth,
    )
    reason = blocking_reason(report, batch_size=request.batch_size)
    if reason:
        raise HTTPException(400, reason)
    if report["n_test"] == 0:
        # PaddleOCR only writes best_accuracy after an eval; with no test split
        # a run would finish with nothing saved.
        raise HTTPException(400, "No test images — set test_split > 0 or pin some items to test.")

    model = await repo.create_model_record(
        project_id, request.arch, request.model_dump(), base_label=base["label"],
    )
    await repo.update_model_record(model.id, {"status": "training", "phase": "queued"})

    background_tasks.add_task(_run_training_bg, repo, project_id, model.id, request, base["path"])
    return {
        "model_id": model.id,
        "status": "training",
        "n_train": report["n_train"],
        "n_test": report["n_test"],
        "n_synth_train": report["n_synth_train"],
        "dropped_count": report["dropped_count"],
        "gpu_holder": gpu_lock.current_holder(),
    }


async def _run_training_bg(
    repo: OCRRepository, project_id: str, model_id: str, request: OCRTrainRequest, base_path: Path,
):
    buf = get_buffer()
    buf.register(project_id, model_id)
    handler = attach_handler(model_id)
    buf.push(model_id, f"[paddle] starting run {model_id} ({request.arch}, base={base_path.name})")

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
            logger.exception("[paddle train] failed to schedule progress update")

    def _on_wait(waited: float) -> None:
        if waited < 1.0 or int(waited) % 30 == 0:
            buf.push(model_id, f"[paddle] waiting for the GPU ({gpu_lock.current_holder()})")
        try:
            asyncio.run_coroutine_threadsafe(
                repo.update_model_record(model_id, {"phase": "waiting_for_gpu"}), loop,
            )
        except Exception:
            pass

    try:
        # Written for real now, not from the dry run: the verified set can
        # change between pressing Train and the GPU freeing up.
        items = await repo.list_trainable_items(project_id)
        manifest = synthetic_defects.load_manifest(project_id)
        synth = manifest["entries"] if (request.use_synthetic and manifest) else []
        report = await loop.run_in_executor(None, lambda: build_dataset(
            project_id, items, arch=request.arch,
            test_split=request.test_split, max_text_length=request.max_text_length,
            model_id=model_id, exclude_item_ids=request.exclude_item_ids, dry_run=False,
            synth_entries=synth,
        ))
        buf.push(
            model_id,
            f"[paddle] dataset: {report['n_train']} train / {report['n_test']} test"
            + (f", {report['dropped_count']} dropped by validation" if report["dropped_count"] else "")
            + (f", {report['unreadable_count']} unreadable" if report["unreadable_count"] else "")
            + (f" + {report['n_synth_train']} synthetic defect crops" if report["n_synth_train"] else ""),
        )
        if report["unknown_chars"]:
            buf.push(
                model_id,
                f"[paddle] warning: {report['unknown_char_count']} label(s) contain characters "
                f"outside the dict {report['unknown_chars']} — they will be stripped silently",
                level="WARNING",
            )

        def _blocking_run() -> Dict:
            with gpu_lock.gpu_lock(f"paddle-ocr-train:{model_id}", on_wait=_on_wait):
                return ocr_training.train_model(
                    project_id, model_id, request, base_path, report["n_train"],
                    progress_cb=_progress, cancel_check=_is_cancelled,
                )

        result = await loop.run_in_executor(None, _blocking_run)
        metrics = result["metrics"]
        await repo.update_model_record(model_id, {
            "phase": "trained",
            "progress": TRAIN_DONE_PROGRESS,
            "checkpoint_path": result["checkpoint_path"],
            "config_path": result["config_path"],
            "metrics": {
                "acc": metrics.get("acc"),
                "norm_edit_dis": metrics.get("norm_edit_dis"),
                "best_epoch": metrics.get("best_epoch"),
                "n_train": report["n_train"],
                "n_test": report["n_test"],
                "n_synth_train": report["n_synth_train"],
            },
        })
        buf.push(model_id, f"[paddle] trained — acc={metrics.get('acc')} "
                           f"best_epoch={metrics.get('best_epoch')}")

        await _auto_export(repo, project_id, model_id, request.arch, result, report["test_items"], buf)

        await repo.update_model_record(model_id, {
            "status": "completed", "phase": "completed", "progress": 100.0,
            "completed_at": datetime.utcnow(),
        })
        buf.push(model_id, "[paddle] run complete")

    except ocr_training.TrainingCancelled:
        buf.push(model_id, "[paddle] cancelled by user", level="WARNING")
        await repo.update_model_record(model_id, {
            "status": "cancelled", "phase": "cancelled", "error": "Cancelled by user",
        })
    except Exception as e:
        logger.exception(f"[paddle train] run {model_id} failed")
        buf.push(model_id, f"[paddle] FAILED: {e}", level="ERROR")
        await repo.update_model_record(model_id, {
            "status": "failed", "phase": "failed", "error": str(e),
        })
    finally:
        detach_handler(handler)
        _CANCEL_FLAGS.pop(model_id, None)


async def _auto_export(repo, project_id: str, model_id: str, arch: str, result: Dict,
                       test_items, buf) -> None:
    """ONNX -> score ONNX -> TensorRT -> score engine. Best-effort per stage:
    the checkpoint is saved either way; the engine only builds from a verified
    ONNX."""
    loop = asyncio.get_event_loop()
    try:
        await repo.update_model_record(model_id, {"phase": "exporting_onnx", "progress": 87.0})
        buf.push(model_id, "[paddle] exporting ONNX (checked against Paddle)...")
        exported = await loop.run_in_executor(None, lambda: paddle_export.export_onnx(
            project_id, model_id, arch, Path(result["config_path"]), Path(result["checkpoint_path"]),
        ))
        await repo.update_model_record(model_id, {
            "inference_dir": exported["inference_dir"],
            "onnx_path": exported["onnx_path"],
            "dict_path": exported["dict_path"],
            "num_classes": exported["num_classes"],
        })
        buf.push(model_id, f"[paddle] ONNX ok — {exported['size_mb']} MB, "
                           f"{exported['num_classes']} classes")
    except Exception as e:
        logger.exception(f"[paddle train] ONNX export failed for {model_id}")
        buf.push(model_id, f"[paddle] ONNX export failed (training still succeeded): {e}",
                 level="WARNING")
        return

    if test_items:
        try:
            await repo.update_model_record(model_id, {"phase": "measuring_onnx", "progress": 90.0})
            ev = await loop.run_in_executor(None, lambda: paddle_export.evaluate(
                project_id, Path(exported["onnx_path"]), None, Path(exported["dict_path"]),
                test_items, "onnx",
            ))
            s = ev["scores"]
            await repo.update_model_record(model_id, {
                "metrics.acc_onnx": s["norm"], "metrics.acc_exact_onnx": s["exact"],
            })
            buf.push(model_id, f"[paddle] ONNX accuracy (production preprocessing): "
                               f"normalized {s['norm']:.4f} exact {s['exact']:.4f} over {s['n']}")
        except Exception as e:
            logger.exception(f"[paddle train] ONNX eval failed for {model_id}")
            buf.push(model_id, f"[paddle] ONNX accuracy check failed: {e}", level="WARNING")

    await _score_synthetic(repo, project_id, model_id, exported, None, "onnx", test_items, buf)

    try:
        await repo.update_model_record(model_id, {"phase": "exporting_tensorrt", "progress": 92.0})
        buf.push(model_id, "[paddle] building TensorRT engine (vision env)...")

        def _build():
            with gpu_lock.gpu_lock(f"paddle-ocr-trt:{model_id}"):
                return paddle_export.build_engine(project_id, model_id, Path(exported["onnx_path"]))

        built = await loop.run_in_executor(None, _build)
        await repo.update_model_record(model_id, {"engine_path": built["engine_path"]})
        buf.push(model_id, f"[paddle] engine ok — {built['size_mb']} MB in {built['build_s']} s")
    except Exception as e:
        logger.exception(f"[paddle train] TensorRT build failed for {model_id}")
        buf.push(model_id, f"[paddle] TensorRT build failed (ONNX still usable): {e}",
                 level="WARNING")
        return

    if not test_items:
        buf.push(model_id, "[paddle] no test split — skipping engine accuracy", level="WARNING")
        return
    try:
        await repo.update_model_record(model_id, {"phase": "measuring_engine", "progress": 96.0})

        def _eval():
            with gpu_lock.gpu_lock(f"paddle-ocr-eval:{model_id}"):
                return paddle_export.evaluate(
                    project_id, None, Path(built["engine_path"]), Path(exported["dict_path"]),
                    test_items, "tensorrt",
                )

        ev = await loop.run_in_executor(None, _eval)
        s = ev["scores"]
        await repo.update_model_record(model_id, {
            "metrics.acc_trt": s["norm"], "metrics.acc_exact_trt": s["exact"],
        })
        buf.push(model_id, f"[paddle] engine accuracy (ai_services TextRecognizerTRT): "
                           f"normalized {s['norm']:.4f} exact {s['exact']:.4f} over {s['n']}, "
                           f"{ev['ms_per_image']} ms/img")
    except Exception as e:
        logger.exception(f"[paddle train] engine eval failed for {model_id}")
        buf.push(model_id, f"[paddle] engine accuracy check failed: {e}", level="WARNING")

    await _score_synthetic(repo, project_id, model_id, exported, built["engine_path"], "tensorrt",
                           test_items, buf)


def synth_test_entries(project_id: str, test_items) -> list:
    """Synthetic defects made from THIS run's test items (never trained on)."""
    manifest = synthetic_defects.load_manifest(project_id)
    if not manifest:
        return []
    test_ids = {i.id for i in test_items}
    return [e for e in manifest["entries"] if e["source_id"] in test_ids]


async def _score_synthetic(repo, project_id: str, model_id: str, exported: Dict,
                           engine_path: Optional[str], engine: str, test_items, buf) -> None:
    """Reject rate on synthetic defects of test items — measured for every run
    when the project has synthetic data, so a model trained WITHOUT it has a
    baseline to compare against."""
    entries = synth_test_entries(project_id, test_items)
    if not entries:
        return
    loop = asyncio.get_event_loop()

    def _run():
        args = (project_id, Path(exported["onnx_path"]),
                Path(engine_path) if engine_path else None,
                Path(exported["dict_path"]), entries, engine)
        if engine == "tensorrt":
            with gpu_lock.gpu_lock(f"paddle-ocr-eval:{model_id}"):
                return paddle_export.evaluate_synthetic(*args)
        return paddle_export.evaluate_synthetic(*args)

    try:
        ev = await loop.run_in_executor(None, _run)
        key = "syn_reject_trt" if engine == "tensorrt" else "syn_reject_onnx"
        await repo.update_model_record(model_id, {f"metrics.{key}": ev["reject_rate"],
                                                  "metrics.syn_n": ev["n"]})
        per = ", ".join(f"{k} {v['reject_rate'] * 100:.0f}%" for k, v in sorted(ev["by_type"].items()))
        buf.push(model_id, f"[paddle] synthetic defects ({engine}): line would FAIL "
                           f"{ev['reject_rate'] * 100:.1f}% of {ev['n']} damaged test crops — {per}")
    except Exception as e:
        logger.exception(f"[paddle train] synthetic eval failed for {model_id}")
        buf.push(model_id, f"[paddle] synthetic defect check failed: {e}", level="WARNING")


@router.get("/projects/{project_id}/models")
async def list_models(
    project_id: str,
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    return [m.model_dump(by_alias=False) for m in await repo.list_models(project_id)]


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
        get_buffer().push(model_id, "[paddle] cancel requested", level="WARNING")
        return {"ok": True, "model_id": model_id, "mode": "cooperative"}

    await repo.update_model_record(model_id, {
        "status": "failed", "phase": "failed", "progress": 0.0,
        "error": "Stuck record reset by user (no in-process task)",
    })
    return {"ok": True, "model_id": model_id, "mode": "force_failed"}


@router.delete("/projects/{project_id}/models/{model_id}")
async def delete_model(
    project_id: str,
    model_id: str,
    repo: OCRRepository = Depends(get_repo),
    db=Depends(get_database),
    current_user: CurrentUser = Depends(get_current_user),
):
    model = await repo.get_model(model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")
    if model.status in ("training", "pending"):
        raise HTTPException(409, "Cancel this run before deleting it")
    used_by = await recipes_using(db, model_id=model_id)
    if used_by:
        raise HTTPException(409, in_use_message("OCR model", used_by))

    models_dir = dataset_fs.models_dir(project_id)
    for p in (models_dir / f"{model_id}.pdparams", models_dir / f"{model_id}_config.yml"):
        p.unlink(missing_ok=True)
    for split in ("train", "test"):
        dataset_fs.label_file(project_id, model_id, split).unlink(missing_ok=True)
    shutil.rmtree(dataset_fs.run_dir(project_id, model_id), ignore_errors=True)
    shutil.rmtree(dataset_fs.export_dir(project_id, model_id), ignore_errors=True)
    get_buffer().drop(project_id, model_id)
    paddle_export.release_model(model_id)
    await repo.delete_model(model_id)
    return {"ok": True, "model_id": model_id}
