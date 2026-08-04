"""
Defect Studio — draw a synthetic wrinkle/scratch onto a clean label, run the
trained model on it, and see exactly what the model saw: score, heatmap, and
the predicted mask as polygons.

The point is to answer "how bad does a defect have to be before this model
catches it?" interactively, instead of waiting for a real defective bottle to
come down the line. Marks are drawn server-side (app/services/defect_sim.py)
rather than in the browser so the Δ-against-local-background and blur rules
match the offline POD study exactly — a canvas-drawn stroke would be a
different stimulus and the numbers wouldn't transfer.

Two independent verdicts are reported, because they can disagree and the
disagreement is informative:
  * caught_by_score — the image-level pred_score crossed the threshold
  * caught_by_mask  — the predicted mask actually landed ON the drawn mark
A high score with no mask overlap means the model reacted to something else in
the frame, which looks like a catch but would not localize the defect.
"""
import base64
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
from app.api.endpoints.test_model import run_model_inference, validate_export
from app.db.mongodb import get_database
from app.repositories.anomaly_repository import AnomalyRepository
from app.services import dataset_fs, defect_sim
from app.services.anomaly_training import encode_heatmap_overlay
from app.services.inspection_crop import img_to_b64

logger = logging.getLogger(__name__)

router = APIRouter()


def get_repo(db=Depends(get_database)) -> AnomalyRepository:
    return AnomalyRepository(db)


class Stroke(BaseModel):
    """One drawn mark, in SOURCE IMAGE pixel coordinates."""
    points: List[List[float]] = Field(..., min_length=2, description="[[x,y], ...] in source image pixels")
    width: int = Field(default=3, ge=1, le=200, description="Stroke thickness in source pixels")
    delta: float = Field(default=25.0, ge=0.0, le=255.0, description="Grey offset vs the LOCAL background under the stroke")
    polarity: str = Field(default="dark", description="'dark' | 'bright' — ignored for edge='wrinkle', which draws both")
    edge: str = Field(default="soft", description="'hard' (sharp) | 'soft' (blurred) | 'wrinkle' (highlight+shadow pair)")
    curvature: float = Field(default=0.0, ge=-1.0, le=1.0, description="Bow a straight 2-point stroke into an arc")


class SimulateRequest(BaseModel):
    strokes: List[Stroke] = Field(default_factory=list)
    engine: str = Field(default="onnx", description="'onnx' | 'tensorrt'")
    # Base image: exactly one of these.
    item_id: Optional[str] = Field(default=None, description="Dataset image id to use as the clean base")
    image_b64: Optional[str] = Field(default=None, description="Base image as base64, when not using a dataset image")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="pred_score >= threshold => abnormal")
    pixel_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Threshold anomaly_map at this value to build the mask instead of using the model's own pred_mask",
    )
    min_region_area: float = Field(default=0.0, ge=0.0, description="Drop mask regions smaller than this (mask pixels)")
    # Persist the drawn image into the dataset as an abnormal test sample. The
    # marks that sit right at the model's limit are the most useful ones to
    # keep, and they are exactly the ones you find by hand here rather than by
    # sweeping a grid.
    save_to_dataset: bool = Field(default=False)
    save_defect_type: str = Field(default="synthetic_wrinkle")


def _decode_b64_image(b64: str) -> Optional[np.ndarray]:
    if "," in b64[:64]:            # tolerate a data: URL prefix
        b64 = b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv.imdecode(arr, cv.IMREAD_COLOR)


async def _load_base_image(project_id: str, req: SimulateRequest, repo: AnomalyRepository) -> np.ndarray:
    if req.item_id:
        item = await repo.get_import_item(req.item_id)
        if not item or item.project_id != project_id:
            raise HTTPException(404, "Base image not found in this project")
        abs_path = dataset_fs.project_dir(project_id) / item.image_path
        if not abs_path.exists():
            raise HTTPException(404, "Base image file missing on disk")
        img = cv.imread(str(abs_path))
        if img is None:
            raise HTTPException(400, f"Could not read base image: {abs_path.name}")
        return img
    if req.image_b64:
        img = _decode_b64_image(req.image_b64)
        if img is None:
            raise HTTPException(400, "Could not decode image_b64 as an image")
        return img
    raise HTTPException(400, "Provide either item_id (a dataset image) or image_b64")


@router.post("/projects/{project_id}/models/{model_id}/simulate-defect", tags=["Anomaly Studio"])
async def simulate_defect(
    project_id: str,
    model_id: str,
    request: SimulateRequest,
    repo: AnomalyRepository = Depends(get_repo),
    current_user: CurrentUser = Depends(get_current_user),
):
    model = await repo.get_model(model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")
    validate_export(model, request.engine)

    base_img = await _load_base_image(project_id, request, repo)

    strokes = [s.model_dump() for s in request.strokes]
    defect_img, footprint = defect_sim.apply_strokes(base_img, strokes)

    try:
        out_map, active_provider, image_size, inference_ms = run_model_inference(
            model, defect_img, request.engine
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[studio] Inference failed for model {model_id} (engine={request.engine})")
        raise HTTPException(500, f"Inference failed: {e}")

    pred_score = float(np.asarray(out_map["pred_score"]).reshape(-1)[0])

    amap = out_map.get("anomaly_map")
    amap = np.asarray(amap).reshape(amap.shape[-2], amap.shape[-1]).astype(np.float32) if amap is not None else None

    # Which binary mask to report regions from. anomalib's baked-in pred_mask
    # comes from a pixel threshold calibrated during training; on a project
    # with no pixel-level ground truth that threshold has been observed never
    # to fire, so the UI can override it by thresholding anomaly_map directly.
    if request.pixel_threshold is not None and amap is not None:
        mask = amap > request.pixel_threshold
        mask_source = f"anomaly_map > {request.pixel_threshold}"
    else:
        pm = out_map.get("pred_mask")
        mask = (
            np.asarray(pm).reshape(pm.shape[-2], pm.shape[-1]).astype(bool)
            if pm is not None else None
        )
        mask_source = "model pred_mask"

    # Overlap is measured in the MASK's resolution: downscale the drawn
    # footprint rather than upscaling the mask, so a few stray mask pixels
    # can't be stretched into a false "hit".
    overlap = 0.0
    stroke_px = int(footprint.sum())
    fp_small = None
    if mask is not None and stroke_px:
        fp_small = cv.resize(
            footprint.astype(np.uint8), (mask.shape[1], mask.shape[0]), interpolation=cv.INTER_NEAREST
        ) > 0
        n_fp = int(fp_small.sum())
        if n_fp:
            overlap = float(np.logical_and(mask, fp_small).sum() / n_fp)

    polygons = (
        defect_sim.mask_polygons(
            mask, min_area=request.min_region_area, amap=amap, footprint=fp_small,
        )
        if mask is not None else []
    )
    total_area = sum(p["area"] for p in polygons)
    mask_px = int(mask.shape[0] * mask.shape[1]) if mask is not None else 0

    heatmap_b64 = encode_heatmap_overlay(defect_img, amap) if amap is not None else ""

    saved_id = None
    if request.save_to_dataset and strokes:
        from app.api.endpoints.synthetic import _safe_defect_type
        defect_type = _safe_defect_type(request.save_defect_type)
        dest_dir = dataset_fs.test_defect_dir(project_id, defect_type)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"studio_{uuid.uuid4().hex[:10]}.jpg"
        if not cv.imwrite(str(dest_path), defect_img):
            raise HTTPException(500, f"Could not write {dest_path.name} to the dataset")
        saved_id = await repo.insert_import_item({
            "project_id": project_id,
            "inspection_id": None, "camera_serial": None, "frame_idx": None,
            "recipe_id": None, "recipe_name": None,
            "label": "abnormal", "defect_type": defect_type, "split": "test",
            "image_path": str(dest_path.relative_to(dataset_fs.project_dir(project_id))),
            "created_at": datetime.utcnow(),
            "source": "synthetic",
            "synthetic_params": {"origin": "studio", "base_item_id": request.item_id, "strokes": strokes},
            "exclude_from_training": False,
        })
        counts = dataset_fs.count_images(project_id)
        await repo.set_counts(project_id, counts["normal"], counts["abnormal"])
        logger.info(f"[studio] saved hand-drawn NG sample {dest_path.name} to test/{defect_type}")

    return {
        "saved_item_id": saved_id,
        "pred_score": round(pred_score, 4),
        "threshold": request.threshold,
        "caught_by_score": bool(pred_score >= request.threshold),
        "caught_by_mask": bool(overlap >= 0.10),
        "stroke_overlap": round(overlap, 4),
        "stroke_pixels": stroke_px,

        "defect_b64": img_to_b64(defect_img),
        "heatmap_b64": heatmap_b64,
        "mask_polygons": polygons,
        "mask_source": mask_source,
        # Totals over the regions that survived min_region_area, so they match
        # what the list below actually shows.
        "region_count": len(polygons),
        "total_area": total_area,
        "total_area_pct": round(total_area / max(mask_px, 1) * 100, 3),
        # Polygons are in mask space (image_size x image_size); the UI scales
        # them onto whatever size it renders the image at.
        "mask_width": int(mask.shape[1]) if mask is not None else 0,
        "mask_height": int(mask.shape[0]) if mask is not None else 0,
        "source_width": int(base_img.shape[1]),
        "source_height": int(base_img.shape[0]),

        "anomaly_map_max": round(float(amap.max()), 4) if amap is not None else None,
        "image_size": image_size,
        "engine": request.engine,
        "active_provider": active_provider,
        "inference_ms": round(inference_ms, 2),
    }
