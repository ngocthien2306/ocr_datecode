"""
Live template realignment for the realtime inference screen.

POST /api/live-template/realign
    Take a frame the operator just captured and re-derive the template regions
    ON THAT FRAME, the way the AI service would (cap rotation → cap crop → text
    window → SuperPoint → RANSAC), then hand back the rotated frame plus
    regions in the AI result's `detected_regions` shape.

Replaces pairing a fresh frame with `detected_regions` from an older
inference result, which drifted whenever the bottle had moved — and, on cap
cameras, was in a different coordinate system altogether (the AI regions are
in the ROTATED frame, the shared-memory frame is not rotated).

All work runs on CPU in this process — see
app/services/live_template/onnx_models.py for why that is a hard rule here.
"""

import asyncio
import base64
import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.api.dependencies.auth import get_current_user
from app.db.mongodb import get_database
from app.models.user import UserInDB

logger = logging.getLogger(__name__)
router = APIRouter()

# One realign at a time: each is ~1 s of CPU on 4 threads. Several operators
# (or a double click) queue instead of piling that onto the box running the
# camera producer.
_realign_gate = asyncio.Semaphore(1)


class RealignRequest(BaseModel):
    recipe_id: str
    serial_number: str
    frame_base64: str = Field(..., description="Raw (unannotated) JPEG frame, base64, as returned by /cameras/{sn}/frames/latest")
    template_name: Optional[str] = Field(None, description="Template to align against; falls back to template_index")
    template_index: int = 0
    quality: int = Field(95, ge=50, le=100, description="JPEG quality of the returned frames")


def _decode_b64_image(b64: str) -> np.ndarray:
    if "," in b64[:64] and b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    try:
        buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    except Exception:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "frame_base64 is not valid base64")
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "frame_base64 is not a decodable image")
    return img


def _encode_jpeg_b64(img: np.ndarray, quality: int) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "failed to encode frame")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _find_template(recipe: Dict[str, Any], serial: str, name: Optional[str], index: int):
    cam = next((c for c in (recipe.get("cameras") or []) if c.get("serial_number") == serial), None)
    if cam is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"camera {serial} is not part of this recipe")
    ct = next((c for c in (recipe.get("camera_templates") or []) if c.get("camera_id") == cam.get("camera_id")), None)
    if ct is None or not ct.get("templates"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no templates for camera {serial}")
    templates = ct["templates"]
    tmpl = None
    if name:
        tmpl = next((t for t in templates if t.get("name") == name), None)
    if tmpl is None:
        if not (0 <= index < len(templates)):
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"template index {index} out of range")
        tmpl = templates[index]
    return cam, ct, tmpl


def _load_template_image(image_url: str) -> np.ndarray:
    # Same resolution rule the recipe preview endpoints use.
    from app.api.endpoints.recipes import _resolve_preview_image_path
    path = _resolve_preview_image_path(image_url or "")
    if path is None or not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"template image not found: {image_url}")
    img = cv2.imread(str(path))
    if img is None:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"cannot read template image: {path.name}")
    return img


@router.post("/live-template/realign")
async def realign_live_frame(
    body: RealignRequest,
    db=Depends(get_database),
    current_user: UserInDB = Depends(get_current_user),
):
    # Imported here so the backend doesn't load onnxruntime sessions until the
    # feature is actually used.
    from app.services.live_template.realign import RealignError, RealignOptions, realign

    try:
        rid = ObjectId(body.recipe_id)
    except Exception:
        rid = body.recipe_id
    recipe = await db.recipes.find_one({"_id": rid})
    if not recipe:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "recipe not found")

    cam, ct, tmpl = _find_template(recipe, body.serial_number, body.template_name, body.template_index)
    frame = _decode_b64_image(body.frame_base64)
    template_img = _load_template_image(tmpl.get("image_url", ""))

    opts = RealignOptions(
        function_type=ct.get("function_type", "") or "",
        cap_rotation_method=recipe.get("cap_rotation_method") or "yolo_obb",
        cap_crop_method=recipe.get("cap_crop_method") or "none",
        dual_rotation_check=bool(recipe.get("dual_rotation_check")),
        matching_conf=float(recipe.get("matching_conf") or 0.20),
        match_erosion_enabled=bool(recipe.get("match_erosion_enabled")),
        match_erosion_kernel_w=int(recipe.get("match_erosion_kernel_w") or 80),
        match_erosion_kernel_h=int(recipe.get("match_erosion_kernel_h") or 1),
        match_erosion_iterations=int(recipe.get("match_erosion_iterations") or 1),
        first_template_annotations=(ct["templates"][0].get("annotations") or []),
    )

    async with _realign_gate:
        try:
            out = await run_in_threadpool(realign, frame, template_img, tmpl.get("annotations") or [], opts)
        except RealignError as e:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e))
        except Exception as e:
            logger.exception(f"[live_template] realign failed for {body.serial_number}")
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"realign failed: {e}")

    candidates: List[Dict[str, Any]] = []
    for c in out.get("candidates", []):
        img = c.pop("frame")
        c["frame_base64"] = _encode_jpeg_b64(img, body.quality)
        c["width"], c["height"] = int(img.shape[1]), int(img.shape[0])
        candidates.append(c)
    out["candidates"] = candidates
    out["template_name"] = tmpl.get("name")
    out["serial_number"] = body.serial_number

    chosen = out.get("chosen")
    logger.info(
        f"[live_template] {body.serial_number} '{tmpl.get('name')}': mode={out.get('mode')} "
        f"rot={out.get('rotation_mode')} chosen={chosen} "
        + " ".join(f"{c['label']}(ok={c['success']},conf={c['confidence']:.2f},"
                   f"rot={c['rotation_deg']},mir={c['mirrored']})" for c in candidates)
        + f" in {out.get('elapsed_ms', 0):.0f}ms"
    )
    return out
