"""
Test/Eval — read the per-image test_results.json sidecar written by
train_model(), and let the FE recompute metrics at an arbitrary threshold
without touching the model/GPU again (Eval UI's threshold slider).
"""
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies.auth import CurrentUser, get_current_user
from app.db.mongodb import get_database
from app.repositories.anomaly_repository import AnomalyRepository
from app.services import dataset_fs
from app.services.anomaly_training import recompute_metrics_at_threshold

router = APIRouter()


def get_repo(db=Depends(get_database)) -> AnomalyRepository:
    return AnomalyRepository(db)


def _test_results_path(project_id: str, model_id: str) -> Path:
    return dataset_fs.models_dir(project_id) / f"{model_id}_test_results.json"


@router.get("/projects/{project_id}/models/{model_id}/test-results", tags=["Anomaly Eval"])
async def get_test_results(
    project_id: str,
    model_id: str,
    threshold: float = 0.5,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Per-image test-set predictions + metrics recomputed at `threshold`
    (default 0.5, matching what training stored). Pure read — no retrain,
    no GPU — so the Eval UI can call this on every slider move.
    """
    model = await repo.get_model(model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")

    path = _test_results_path(project_id, model_id)
    if not path.exists():
        raise HTTPException(404, "No test results for this model (train it first)")

    if not (0.0 <= threshold <= 1.0):
        raise HTTPException(400, "threshold must be between 0 and 1")

    return recompute_metrics_at_threshold(path, threshold)


@router.get("/projects/{project_id}/models/{model_id}/report", tags=["Anomaly Eval"])
async def download_model_report(
    project_id: str,
    model_id: str,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Full report as JSON (params + metrics + test-set) for offline sharing —
    mirrors backend's ml_training model report download."""
    model = await repo.get_model(model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")

    path = _test_results_path(project_id, model_id)
    test_results = json.loads(path.read_text()) if path.exists() else []

    return {
        "project_id": project_id,
        "model_id": model_id,
        "algorithm": model.algorithm,
        "params": model.params,
        "status": model.status,
        "created_at": model.created_at.isoformat() if model.created_at else None,
        "metrics": model.metrics.model_dump(),
        "test_set": test_results,
    }
