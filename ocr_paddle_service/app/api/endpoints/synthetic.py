"""
Synthetic defect data ("enrich") for a project — generate / inspect / delete.

Generation is per project, not per run: the images are made once from the
verified crops and each training run opts in with use_synthetic. See
app/services/synthetic_defects.py for what is generated and how it is labelled.
"""
import asyncio
import base64
import logging
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2 as cv
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies.auth import CurrentUser, get_current_user
from app.core.config import DEFAULT_ARCH, REPO_ROOT
from app.db.mongodb import get_database
from app.models.ocr import OCRSyntheticRequest
from app.repositories.ocr_repository import OCRRepository
from app.services import dataset_fs, synthetic_defects as sd
from app.services.dataset_builder import build_dataset

logger = logging.getLogger(__name__)

router = APIRouter()

# Fallback localiser when the project has no trained Paddle model yet: the
# official PP-OCRv6 small that ai_services ships. It misreads more crops, and a
# misread crop is skipped (positions can't be trusted), so fewer get defects.
_OFFICIAL_ONNX = REPO_ROOT / "languages" / "english" / "ppocrv6_small_rec.onnx"
_OFFICIAL_DICT = REPO_ROOT / "languages" / "english" / "ppocrv6_dict.txt"


def get_repo(db=Depends(get_database)) -> OCRRepository:
    return OCRRepository(db)


async def _pick_localiser(repo: OCRRepository, project_id: str) -> Tuple[Path, Path, str]:
    """Best model to place characters with: this project's newest completed
    Paddle model (reads its own crops best), else the official one."""
    for m in await repo.list_models(project_id):
        if (m.status == "completed" and m.onnx_path and m.dict_path
                and Path(m.onnx_path).is_file() and Path(m.dict_path).is_file()):
            return Path(m.onnx_path), Path(m.dict_path), f"model:{m.id}"
    if _OFFICIAL_ONNX.is_file() and _OFFICIAL_DICT.is_file():
        return _OFFICIAL_ONNX, _OFFICIAL_DICT, "official:PP-OCRv6_small_rec"
    raise HTTPException(400, "No model available to locate characters — train a Paddle model first.")


def _thumb(img: np.ndarray, h: int = 40) -> np.ndarray:
    return cv.resize(img, (max(1, int(img.shape[1] * h / img.shape[0])), h))


def _samples(project_id: str, manifest: Dict, n: int, seed: int = 0) -> list:
    """Before/after pairs for the preview grid, spread across defect types."""
    entries = manifest.get("entries", [])
    rng = random.Random(seed)
    by_type: Dict[str, list] = {}
    for e in entries:
        by_type.setdefault(e["type"], []).append(e)
    picked = []
    while len(picked) < n and any(by_type.values()):
        for t in list(by_type):
            if by_type[t] and len(picked) < n:
                picked.append(by_type[t].pop(rng.randrange(len(by_type[t]))))
    out = []
    base = dataset_fs.dataset_dir(project_id)
    for e in picked:
        before = cv.imread(str(dataset_fs.image_abs_path(project_id, e["source_image"])), cv.IMREAD_GRAYSCALE)
        after = cv.imread(str(base / e["file"]), cv.IMREAD_GRAYSCALE)
        if before is None or after is None:
            continue
        pair = np.hstack([_thumb(before), np.full((40, 4), 128, np.uint8), _thumb(after)])
        ok, buf = cv.imencode(".png", pair)
        if ok:
            out.append({
                "type": e["type"], "char": e["char"], "severity": e["severity"],
                "split": e["split"], "source_label": e["source_label"], "label": e["label"],
                "image_b64": base64.b64encode(buf.tobytes()).decode(),
            })
    return out


@router.get("/projects/{project_id}/synthetic")
async def get_synthetic(
    project_id: str,
    samples: int = Query(12, ge=0, le=60),
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    manifest = sd.load_manifest(project_id)
    if not manifest:
        return {"exists": False, "reject_char": sd.REJECT_CHAR, "types": list(sd.DEFECT_TYPES)}
    loop = asyncio.get_event_loop()
    pics = await loop.run_in_executor(None, lambda: _samples(project_id, manifest, samples))
    return {"exists": True, **sd.summary(manifest), "samples": pics, "types_available": list(sd.DEFECT_TYPES)}


@router.post("/projects/{project_id}/synthetic")
async def generate_synthetic(
    project_id: str,
    request: OCRSyntheticRequest,
    repo: OCRRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    if not await repo.get_project(project_id):
        raise HTTPException(404, "Project not found")
    if not 1 <= request.per_image <= 5:
        raise HTTPException(400, "per_image must be 1..5")
    bad = [t for t in request.types if t not in sd.DEFECT_TYPES]
    if bad or not request.types:
        raise HTTPException(400, f"types must be a non-empty subset of {list(sd.DEFECT_TYPES)}")

    onnx_path, dict_path, loc_label = await _pick_localiser(repo, project_id)
    items = await repo.list_trainable_items(project_id)
    # Same validation + split a run with this test_split will use, so the
    # train/test tag on each synthetic crop matches what training sees.
    report = build_dataset(project_id, items, arch=DEFAULT_ARCH,
                           test_split=request.test_split, max_text_length=25, dry_run=True)
    split_of = {i.id: "train" for i in report["train_items"]}
    split_of.update({i.id: "test" for i in report["test_items"]})
    pool = report["train_items"] + report["test_items"]
    if not pool:
        raise HTTPException(400, "No verified items to build from.")

    loop = asyncio.get_event_loop()

    def _run():
        loc = sd.Localiser(onnx_path, dict_path)
        return sd.generate(project_id, pool, split_of, loc, per_image=request.per_image,
                           types=request.types, seed=request.seed, localiser_label=loc_label)

    try:
        manifest = await loop.run_in_executor(None, _run)
    except Exception as e:
        logger.exception(f"[synth] generation failed for {project_id}")
        raise HTTPException(500, f"Synthetic generation failed: {e}")
    pics = await loop.run_in_executor(None, lambda: _samples(project_id, manifest, 12))
    return {"exists": True, **sd.summary(manifest), "samples": pics}


@router.delete("/projects/{project_id}/synthetic")
async def delete_synthetic(
    project_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    sd.clear(project_id)
    return {"ok": True}
