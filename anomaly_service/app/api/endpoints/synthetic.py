"""
Synthetic NG generation — draw controlled defects onto clean label crops and
add them to the dataset as abnormal test samples.

Why this exists: PatchCore/Padim never train on abnormal images (anomalib forces
every abnormal sample to Split.TEST, and PatchCore has no optimizer). What
abnormal images do drive is the adaptive threshold, the score normalisation and
the reported metrics — all of which are calibrated on the validation loop, which
sees the test set. A project with a single real defect therefore has a threshold
fitted to one point, and a score scale with almost no spread. Generating a
population of defects across a Δ range is what gives the threshold something to
separate.

Generated images are written to their own `test/<name>` folder and tagged
`source="synthetic"` so a model can later be evaluated with and without them.
Calibrating on drawn marks and never checking against real defects is the way
this feature would quietly mislead, so the two stay distinguishable.

Placement is automatic (defect_sim.sample_flat_position) — marks land on blank
label surface rather than across printed text, otherwise Δ stops being a
meaningful knob.
"""
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2 as cv
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies.auth import CurrentUser, get_current_user
from app.db.mongodb import get_database
from app.repositories.anomaly_repository import AnomalyRepository
from app.services import dataset_fs, defect_sim
from app.services.inspection_crop import img_to_b64

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_GENERATE = 500          # guard against a grid that silently produces thousands
PREVIEW_LIMIT = 12          # thumbnails returned by /preview


def get_repo(db=Depends(get_database)) -> AnomalyRepository:
    return AnomalyRepository(db)


class SyntheticOptions(BaseModel):
    """Pools to draw each mark's parameters from — NOT a cartesian sweep.

    Every generated mark picks one value from each pool at random, so ticking
    everything gives a varied population rather than a combinatorial explosion.
    An exhaustive grid produces an evenly-spaced set a threshold can fit
    suspiciously well; random draws over the same pools look more like the
    spread of real defects.
    """
    deltas: List[float] = Field(default=[20, 30, 40, 55], description="Grey offset vs local background")
    widths: List[int] = Field(default=[4, 8], description="Stroke thickness (bubble: rim softness)")
    edges: List[str] = Field(default=["wrinkle"], description="hard | soft | wrinkle | bubble")
    polarities: List[str] = Field(default=["dark"], description="dark | bright (ignored by wrinkle/bubble)")
    marks_min: int = Field(default=1, ge=1, le=10, description="Fewest marks on one image")
    marks_max: int = Field(default=3, ge=1, le=10, description="Most marks on one image")


class SyntheticRequest(BaseModel):
    base_item_ids: List[str] = Field(default_factory=list, description="Normal dataset images to draw on")
    options: SyntheticOptions = Field(default_factory=SyntheticOptions)
    # Output size is expressed relative to the chosen base images so the
    # request stays meaningful as the dataset grows — x2 of 40 bases and x2 of
    # 400 bases both mean "twice what I selected".
    multiplier: int = Field(default=2, ge=1, le=8, description="How many images to make, as a multiple of the base images")
    defect_type: str = Field(default="synthetic_wrinkle", description="Folder name under dataset/test/")
    seed: int = Field(default=20260803, description="RNG seed — same seed reproduces the same batch")


def _validate(opts: SyntheticOptions) -> None:
    if not opts.edges or not opts.deltas or not opts.widths:
        raise HTTPException(400, "Pick at least one defect type, one Δ and one thickness")
    for edge in opts.edges:
        if edge not in defect_sim.EDGES:
            raise HTTPException(400, f"defect type must be one of {defect_sim.EDGES}, got {edge!r}")
    for pol in opts.polarities:
        if pol not in defect_sim.POLARITIES:
            raise HTTPException(400, f"direction must be dark|bright, got {pol!r}")
    if opts.marks_max < opts.marks_min:
        raise HTTPException(400, "marks_max must be >= marks_min")


def _safe_defect_type(name: str) -> str:
    """A new dataset/test/<name> folder becomes an evaluation defect class the
    moment a file lands in it (dataset_fs.list_defect_types), so a typo would
    silently create a class. Restrict to a predictable shape."""
    cleaned = "".join(c if (c.isalnum() or c in "_-") else "_" for c in name.strip())
    if not cleaned:
        raise HTTPException(400, "defect_type must contain at least one alphanumeric character")
    if cleaned == "good":
        raise HTTPException(400, "defect_type cannot be 'good' — that is the normal test folder")
    return cleaned


async def _load_bases(project_id: str, item_ids: List[str], repo: AnomalyRepository):
    """Resolve base images, refusing anything that isn't a normal sample —
    drawing a synthetic defect onto an already-defective label would make the
    resulting sample uninterpretable."""
    if not item_ids:
        raise HTTPException(400, "Pick at least one base image")
    items = await repo.get_import_items(project_id, item_ids)
    if not items:
        raise HTTPException(404, "No base images found in this project")
    proj_dir = dataset_fs.project_dir(project_id)
    bases = []
    for it in items:
        if it.label != "normal":
            raise HTTPException(400, f"Base image {it.id} is labelled '{it.label}' — pick normal images only")
        abs_path = proj_dir / it.image_path
        img = cv.imread(str(abs_path))
        if img is None:
            logger.warning(f"[synthetic] skipping unreadable base image {abs_path}")
            continue
        bases.append((it, img))
    if not bases:
        raise HTTPException(400, "None of the selected base images could be read from disk")
    return bases


def _render(bases, opts: SyntheticOptions, count: int, seed: int):
    """Generate `count` images, each a random base carrying a random number of
    randomly-parameterised marks. Yields (base_item, spec, image)."""
    rng = np.random.default_rng(seed)
    pols = opts.polarities or ["dark"]
    for _ in range(count):
        it, img = bases[int(rng.integers(len(bases)))]
        n_marks = int(rng.integers(opts.marks_min, opts.marks_max + 1))
        strokes = [
            defect_sim.random_stroke(img, rng, opts.deltas, opts.widths, opts.edges, pols)
            for _ in range(n_marks)
        ]
        out, _fp = defect_sim.apply_strokes(img, strokes)
        spec = {
            "n_marks": n_marks,
            "edges": sorted({s["edge"] for s in strokes}),
            "deltas": [s["delta"] for s in strokes],
            "widths": [s["width"] for s in strokes],
            "strokes": strokes,
        }
        yield it, spec, out


@router.post("/projects/{project_id}/synthetic/preview", tags=["Anomaly Synthetic"])
async def preview_synthetic(
    project_id: str,
    request: SyntheticRequest,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Render a sample of the grid without writing anything, so the marks can be
    eyeballed before committing a few hundred files to the dataset."""
    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    _validate(request.options)
    bases = await _load_bases(project_id, request.base_item_ids, repo)
    total = len(bases) * request.multiplier

    samples = []
    for _it, spec, img in _render(bases, request.options, min(total, PREVIEW_LIMIT), request.seed):
        samples.append({
            "n_marks": spec["n_marks"],
            "edges": spec["edges"],
            "deltas": spec["deltas"],
            "widths": spec["widths"],
            "image_b64": img_to_b64(img),
        })

    return {
        "total_to_generate": total,
        "base_images": len(bases),
        "multiplier": request.multiplier,
        "over_limit": total > MAX_GENERATE,
        "max_generate": MAX_GENERATE,
        "samples": samples,
    }


@router.post("/projects/{project_id}/synthetic/generate", tags=["Anomaly Synthetic"])
async def generate_synthetic(
    project_id: str,
    request: SyntheticRequest,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    defect_type = _safe_defect_type(request.defect_type)
    _validate(request.options)
    bases = await _load_bases(project_id, request.base_item_ids, repo)
    total = len(bases) * request.multiplier
    if total > MAX_GENERATE:
        raise HTTPException(
            400,
            f"That would generate {total} images (limit {MAX_GENERATE}). "
            f"Lower the multiplier or select fewer base images.",
        )

    # Synthetic defects are abnormal samples and must land under test/, never in
    # train/good — a defect in the training set would be absorbed into the
    # memory bank as normal appearance.
    dest_dir = dataset_fs.test_defect_dir(project_id, defect_type)
    dest_dir.mkdir(parents=True, exist_ok=True)
    proj_dir = dataset_fs.project_dir(project_id)
    batch_id = uuid.uuid4().hex[:8]

    written, errors = 0, []
    for base_item, spec, img in _render(bases, request.options, total, request.seed):
        filename = (
            f"syn_{batch_id}_{written:04d}_{'-'.join(spec['edges'])}"
            f"_n{spec['n_marks']}.jpg"
        )
        dest_path = dest_dir / filename
        if not cv.imwrite(str(dest_path), img):
            errors.append(f"Failed to write {filename}")
            continue
        await repo.insert_import_item({
            "project_id": project_id,
            "inspection_id": None,
            "camera_serial": None,
            "frame_idx": None,
            "recipe_id": None,
            "recipe_name": None,
            "label": "abnormal",
            "defect_type": defect_type,
            "split": "test",
            "image_path": str(dest_path.relative_to(proj_dir)),
            "created_at": datetime.utcnow(),
            "source": "synthetic",
            "synthetic_params": {
                "batch_id": batch_id,
                "seed": request.seed,
                "base_item_id": base_item.id,
                "base_image_path": base_item.image_path,
                "n_marks": spec["n_marks"],
                "edges": spec["edges"],
                "deltas": spec["deltas"],
                "widths": spec["widths"],
                "strokes": spec["strokes"],
            },
            "exclude_from_training": False,
        })
        written += 1

    counts = dataset_fs.count_images(project_id)
    await repo.set_counts(project_id, counts["normal"], counts["abnormal"])

    logger.info(
        f"[synthetic] project {project_id}: wrote {written}/{total} images "
        f"into test/{defect_type} (batch {batch_id})"
    )
    return {
        "generated": written,
        "requested": total,
        "batch_id": batch_id,
        "defect_type": defect_type,
        "errors": errors,
        **counts,
    }


class ExcludeRequest(BaseModel):
    ids: List[str]
    excluded: bool = True


@router.post("/projects/{project_id}/dataset/images/bulk-exclude", tags=["Anomaly Dataset"])
async def bulk_exclude_from_training(
    project_id: str,
    request: ExcludeRequest,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Flag images in/out of the next training run. Non-destructive: the files
    stay put and training stages a symlink tree of the included set instead."""
    project = await repo.get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    modified = await repo.set_exclude_from_training(project_id, request.ids, request.excluded)
    return {"ok": True, "modified": modified, "excluded": request.excluded}
