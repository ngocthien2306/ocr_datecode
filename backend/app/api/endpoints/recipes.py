import asyncio
import traceback
from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File, Request
from fastapi.responses import FileResponse, Response
from typing import Any, Dict, List, Optional
import os
import uuid
from pathlib import Path
import base64
from PIL import Image, ImageDraw, ImageFont
import json
import hashlib
import io
import logging
import time

logger = logging.getLogger(__name__)

from app.schemas.recipe import RecipeCreate, RecipeUpdate, RecipeResponse
from app.models.recipe import RecipeInDB
from app.models.user import UserInDB, UserRole
from app.repositories.recipe_repository import RecipeRepository
from app.repositories.action_log_repository import ActionLogRepository
from app.repositories.user_repository import UserRepository
from app.db.mongodb import get_database
from app.api.dependencies.auth import (
    get_current_user, 
    require_supervisor, 
    require_admin,
    require_operator,
    RoleChecker
)
from app.api.dependencies.action_log import get_action_log_repository
from app.schemas.receipt_load import ReceiptLoadCreate, ReceiptLoadResponse
from app.repositories.receipt_load_repository import ReceiptLoadRepository
from app.models.action_log import ActionLogCreate, ActionType
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from app.core.config import settings
from app.api.websocket.camera_ws import send_load_recipe, send_stop_recipe, send_set_inference_mode, camera_ws_manager

router = APIRouter()


# Template image upload directory
TEMPLATE_UPLOAD_DIR = Path("uploads/templates")
TEMPLATE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
# CV-method preview endpoint
# Compute char-quality score on N random pairs from latest test_result/emb_*/
# folder for the selected cv_method. Used by recipe form to preview before save.
# MUST stay BEFORE any `/{recipe_id}` route (FastAPI static-first rule).
# ────────────────────────────────────────────────────────────────────────────
@router.post("/cv-preview")
async def cv_method_preview(payload: Dict[str, Any]):
    """Run a given cv_method against N random char pairs from the latest
    inference debug folder and return base64 thumbnails + per-pair scores.

    Body: {"cv_method": "legacy"|"v4"|"shape_v7", "count": 5, "threshold": 0.80}
    """
    import sys as _sys
    import random as _random
    import re as _re
    import cv2 as _cv2

    cv_method = (payload.get("cv_method") or "legacy").lower()
    if cv_method not in ("legacy", "v3", "v4", "v5", "shape_v7"):
        cv_method = "legacy"
    count = max(1, min(20, int(payload.get("count") or 5)))
    threshold = float(payload.get("threshold") or 0.80)
    # Optional: caller pins specific pairs (folder + char_idx). When provided,
    # use these exact pairs instead of random — allows FE to keep the same
    # cards across method changes for fair comparison.
    pair_keys_in = payload.get("pair_keys") or []

    # Project root: backend/app/api/endpoints/recipes.py → up 4 dirs
    proj_root = Path(__file__).resolve().parents[4]
    if str(proj_root) not in _sys.path:
        _sys.path.insert(0, str(proj_root))

    test_result_dir = proj_root / "ai_services" / "test_result"
    if not test_result_dir.is_dir():
        return {"folder": None, "method": cv_method, "pairs": [],
                "error": f"test_result dir missing: {test_result_dir}"}

    emb_folders = [d for d in test_result_dir.iterdir()
                   if d.is_dir() and d.name.startswith("emb_")]
    if not emb_folders:
        return {"folder": None, "method": cv_method, "pairs": [],
                "error": "no emb_* folders found"}

    pat = _re.compile(r"^char(\d+)_(OK|NG)_p([\d.]+)_target\.png$")

    def _scan_folder(folder_path: Path) -> List[Dict[str, Any]]:
        items = []
        for f in folder_path.iterdir():
            m = pat.match(f.name)
            if not m:
                continue
            tmpl_path = folder_path / f.name.replace("_target.png", "_template.png")
            if not tmpl_path.exists():
                continue
            items.append({
                "char_idx": int(m.group(1)),
                "logged_label": m.group(2),
                "logged_p": float(m.group(3)),
                "folder": folder_path.name,
                "tmpl": tmpl_path,
                "tgt": f,
            })
        return items

    # Path A: caller specified exact pairs (folder + char_idx) → use those
    sample: List[Dict[str, Any]] = []
    folder_for_response: Optional[str] = None
    if pair_keys_in:
        for k in pair_keys_in:
            try:
                folder_name = str(k.get("folder"))
                char_idx = int(k.get("char_idx"))
            except Exception:
                continue
            f_dir = test_result_dir / folder_name
            if not f_dir.is_dir():
                continue
            for it in _scan_folder(f_dir):
                if it["char_idx"] == char_idx:
                    sample.append(it)
                    break
        if sample:
            folder_for_response = sample[0]["folder"]

    # Path B (default): latest folder, random shuffle
    if not sample:
        latest = max(emb_folders, key=lambda d: d.stat().st_mtime)
        pairs_on_disk = _scan_folder(latest)
        if not pairs_on_disk:
            return {"folder": latest.name, "method": cv_method, "pairs": [],
                    "error": "no template/target pairs in folder"}
        _random.shuffle(pairs_on_disk)
        sample = pairs_on_disk[:count]
        folder_for_response = latest.name

    # Lazy-import algorithms (only what we need)
    from ai_services.camera_management.verification.embedding_classifier import _compute_char_quality
    if cv_method == "v3":
        from ai_services.camera_management.verification.char_quality_v3 import compute_char_quality_v3
    elif cv_method == "v4":
        from ai_services.camera_management.verification.char_quality_v4 import compute_char_quality_v4
    elif cv_method == "v5":
        from ai_services.camera_management.verification.char_quality_v5 import compute_char_quality_v5
    elif cv_method == "shape_v7":
        from ai_services.camera_management.verification.char_quality_v7_shape import (
            compute_char_quality_v7,
            render_orientation_overlay,
        )

    import numpy as _np

    def _encode_png_b64(img) -> Optional[str]:
        if img is None or getattr(img, "size", 0) == 0:
            return None
        ok, buf = _cv2.imencode(".png", img)
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode()

    out_pairs: List[Dict[str, Any]] = []
    for p in sample:
        try:
            tmpl_bgr = _cv2.imread(str(p["tmpl"]))
            tgt_bgr  = _cv2.imread(str(p["tgt"]))
            if tmpl_bgr is None or tgt_bgr is None:
                continue
            tmpl_g = _cv2.cvtColor(tmpl_bgr, _cv2.COLOR_BGR2GRAY)
            tgt_g  = _cv2.cvtColor(tgt_bgr,  _cv2.COLOR_BGR2GRAY)

            result_img = None  # method-specific diff visualization

            if cv_method == "v3":
                m = compute_char_quality_v3(tmpl_g, tgt_g)
                conf = float(m["confidence"]); defect = m.get("defect_type")
                extra = {"ncc": round(float(m.get("ncc", 0)), 3),
                         "over": round(float(m.get("over_ink_score", 0)), 3),
                         "under": round(float(m.get("under_ink_score", 0)), 3)}
                # Colored diff: gray=match, red=over, blue=under (same as v4)
                t_bin = m.get("_t_bin")
                if t_bin is not None:
                    base = _cv2.cvtColor(t_bin, _cv2.COLOR_GRAY2BGR)
                    base[t_bin > 0] = (180, 180, 180)
                    if m.get("_extra_ink") is not None:
                        base[m["_extra_ink"] > 0] = (60, 60, 255)
                    if m.get("_missing_ink") is not None:
                        base[m["_missing_ink"] > 0] = (255, 100, 60)
                    result_img = base
            elif cv_method == "v4":
                m = compute_char_quality_v4(tmpl_g, tgt_g)
                conf = float(m["confidence"]); defect = m.get("defect_type")
                extra = {"ncc": round(float(m.get("ncc", 0)), 3),
                         "over": round(float(m.get("over_ink_score", 0)), 3),
                         "under": round(float(m.get("under_ink_score", 0)), 3)}
                t_bin = m.get("_t_bin")
                if t_bin is not None:
                    base = _cv2.cvtColor(t_bin, _cv2.COLOR_GRAY2BGR)
                    base[t_bin > 0] = (180, 180, 180)
                    if m.get("_extra_ink") is not None:
                        base[m["_extra_ink"] > 0] = (60, 60, 255)
                    if m.get("_missing_ink") is not None:
                        base[m["_missing_ink"] > 0] = (255, 100, 60)
                    result_img = base
            elif cv_method == "v5":
                m = compute_char_quality_v5(tmpl_g, tgt_g)
                conf = float(m["confidence"]); defect = m.get("defect_type")
                loc = m.get("defect_location")
                extra = {"ncc": round(float(m.get("ncc", 0)), 3),
                         "over_g": round(float(m.get("over_ink_score", 0)), 3),
                         "under_g": round(float(m.get("under_ink_score", 0)), 3),
                         "tile_over_max":  round(float(m.get("tile_over_max", 0)), 3),
                         "tile_under_max": round(float(m.get("tile_under_max", 0)), 3),
                         "n_bad_tiles":    int(m.get("n_bad_tiles", 0))}
                # Use tile heatmap as result visualization
                if m.get("_tile_heatmap") is not None:
                    result_img = m["_tile_heatmap"]
            elif cv_method == "shape_v7":
                m = compute_char_quality_v7(tmpl_g, tgt_g)
                conf = float(m["confidence"]); defect = m.get("defect_type")
                extra = {"match_pct": round(100 * float(m.get("orientation_match_pct", 0)), 1),
                         "partial_pct": round(100 * float(m.get("partial_match_pct", 0)), 1),
                         "coverage": round(float(m.get("coverage_ratio", 0)), 2)}
                # Orientation overlay (green=match, yellow=partial, red=mismatch, blue=missing)
                if m.get("_t_prep") is not None and m.get("_quant_t") is not None and m.get("_quant_g") is not None:
                    result_img = render_orientation_overlay(m["_t_prep"], m["_quant_t"], m["_quant_g"])
            else:
                m = _compute_char_quality(tmpl_g, tgt_g, denoise=False)
                conf = float(m["confidence"]); defect = None
                extra = {"blur_tm": round(float(m.get("blur_tm", 0)), 3),
                         "iou": round(float(m.get("iou", 0)), 3),
                         "pixel_conf": round(float(m.get("pixel_conf", 0)), 3)}
                # XOR of centroid-aligned masks
                if m.get("_mask_tmpl_aligned") is not None and m.get("_mask_tgt_aligned") is not None:
                    result_img = _cv2.bitwise_xor(m["_mask_tmpl_aligned"], m["_mask_tgt_aligned"])

            label = "OK" if (conf >= threshold and not defect) else "NG"

            with open(p["tmpl"], "rb") as fh:
                tmpl_b64 = base64.b64encode(fh.read()).decode()
            with open(p["tgt"], "rb") as fh:
                tgt_b64 = base64.b64encode(fh.read()).decode()
            result_b64 = _encode_png_b64(result_img)

            out_pairs.append({
                "char_idx": p["char_idx"],
                "folder": p.get("folder"),
                "logged_label": p["logged_label"],
                "logged_p": p["logged_p"],
                "tmpl_b64": tmpl_b64,
                "tgt_b64": tgt_b64,
                "result_b64": result_b64,
                "conf": round(conf, 4),
                "label": label,
                "defect_type": defect,
                "extra": extra,
            })
        except Exception as e:
            out_pairs.append({"char_idx": p["char_idx"], "folder": p.get("folder"), "error": str(e)})

    return {
        "folder": folder_for_response,
        "method": cv_method,
        "threshold": threshold,
        "pairs": out_pairs,
    }

TEMPLATE_VISUALIZE_DIR = Path("uploads/visualizations")
TEMPLATE_VISUALIZE_DIR.mkdir(parents=True, exist_ok=True)

# Base URL used when returning image/visualization links (frontend should use absolute URLs)
TEMPLATE_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000').rstrip('/')


def _coord(value, max_val: int) -> int:
    """Normalize a coordinate which may be absolute pixel or relative (0..1)."""
    try:
        v = float(value)
    except Exception:
        return 0
    if 0.0 <= v <= 1.0:
        return int(v * max_val)
    return int(round(v))


def _annotations_hash(filename: str, annotations: list, loaded_at: Optional[str] = None) -> str:
    payload = json.dumps({"file": filename, "annotations": annotations, "loaded_at": loaded_at}, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()


def _draw_template_visualization(image_path: Path, annotations: list, loaded_at: Optional[str] = None) -> str:
    """Draw rectangles, polygons and text onto image and save under TEMPLATE_VISUALIZE_DIR.

    Returns relative URL path to the generated visualization file.
    """
    if not image_path.exists():
        raise FileNotFoundError(str(image_path))

    # compute deterministic filename from annotations + source filename + optional load timestamp
    digest = _annotations_hash(image_path.name, annotations or [], loaded_at=loaded_at)
    out_name = f"viz_{digest}.png"
    out_path = TEMPLATE_VISUALIZE_DIR / out_name
    if out_path.exists():
        return f"/api/recipes/templates/visualizations/{out_name}"

    img = Image.open(str(image_path)).convert("RGBA")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # font size proportional to image width
    # Choose a readable font; fall back silently to default if unavailable
    font_size = max(12, int(width / 70))
    font = None
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except Exception:
        try:
            # Try a common system font as alternative
            font = ImageFont.truetype("Arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    def _get_text_size(draw_obj, text, font_obj):
        try:
            # PIL >=8.0
            bbox = draw_obj.textbbox((0, 0), text, font=font_obj)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            try:
                return font_obj.getsize(text)
            except Exception:
                # fallback estimate
                return (len(text) * (font_obj.size if hasattr(font_obj, 'size') else 10) * 0.6, (font_obj.size if hasattr(font_obj, 'size') else 10))

    for ann in annotations or []:
        # annotation examples in your data use keys: type, shape, text, x,y,width,height,points
        shape = (ann.get('shape') or ann.get('type') or '').lower()
        # If explicit text not provided, fall back to the annotation `type` so shapes still show a label
        label = ann.get('text') or (ann.get('type') or ann.get('annotationType') or None)
        color = ann.get('color') or ann.get('stroke') or '#ff0000'

        if shape == 'rectangle' or shape == 'rect' or shape == 'template':
            # rectangle stored as x,y,width,height
            x = ann.get('x')
            y = ann.get('y')
            w = ann.get('width') or ann.get('w')
            h = ann.get('height') or ann.get('h')
            if x is None or y is None or w is None or h is None:
                continue
            try:
                x = float(x); y = float(y); w = float(w); h = float(h)
            except Exception:
                continue
            x1 = _coord(x, width)
            y1 = _coord(y, height)
            x2 = _coord(x + w, width)
            y2 = _coord(y + h, height)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, int(width/800)))
            if label:
                # draw label background and text
                text_w, text_h = _get_text_size(draw, label, font)
                bx1, by1 = x1, max(0, y1 - text_h - 6)
                bx2, by2 = x1 + text_w + 6, by1 + text_h + 4
                draw.rectangle([bx1, by1, bx2, by2], fill=color)
                draw.text((bx1 + 3, by1 + 2), label, fill='#ffffff', font=font)

        elif shape == 'polygon' or shape == 'poly' or ann.get('points'):
            pts = ann.get('points') or ann.get('polygon') or []
            proc = []
            for p in pts:
                if not isinstance(p, (list, tuple)) or len(p) < 2:
                    continue
                try:
                    px = float(p[0]); py = float(p[1])
                except Exception:
                    continue
                proc.append((_coord(px, width), _coord(py, height)))
            if len(proc) >= 2:
                draw.line(proc + [proc[0]], fill=color, width=max(2, int(width/800)))
                if label:
                    lx, ly = proc[0]
                    text_w, text_h = _get_text_size(draw, label, font)
                    bx1, by1 = lx, max(0, ly - text_h - 6)
                    bx2, by2 = lx + text_w + 6, by1 + text_h + 4
                    draw.rectangle([bx1, by1, bx2, by2], fill=color)
                    draw.text((bx1 + 3, by1 + 2), label, fill='#ffffff', font=font)

    # save output as PNG
    img.convert('RGB').save(str(out_path), format='PNG')
    return f"/api/recipes/templates/visualizations/{out_name}"



async def get_recipe_repository(db=Depends(get_database)) -> RecipeRepository:
    """Dependency to get recipe repository"""
    return RecipeRepository(db)


async def get_receipt_load_repository(db=Depends(get_database)) -> ReceiptLoadRepository:
    """Dependency to get receipt load repository"""
    return ReceiptLoadRepository(db)


async def get_user_repository(db=Depends(get_database)) -> UserRepository:
    """Dependency to get user repository"""
    return UserRepository(db)


def recipe_to_response(recipe: RecipeInDB, user_names_map: dict = None) -> RecipeResponse:
    """Convert RecipeInDB to RecipeResponse"""
    # Convert cameras to dict if they are already CameraConfiguration objects
    cameras_data = []
    if hasattr(recipe, 'cameras') and recipe.cameras:
        for cam in recipe.cameras:
            if hasattr(cam, 'model_dump'):
                cameras_data.append(cam.model_dump())
            elif isinstance(cam, dict):
                cameras_data.append(cam)
            else:
                cameras_data.append(cam)
    
    # Convert camera_templates if present (always include, even if empty)
    camera_templates_data = []
    if hasattr(recipe, 'camera_templates') and recipe.camera_templates is not None:
        for cam_template in recipe.camera_templates:
            if hasattr(cam_template, 'model_dump'):
                camera_templates_data.append(cam_template.model_dump())
            elif isinstance(cam_template, dict):
                camera_templates_data.append(cam_template)
            else:
                camera_templates_data.append(cam_template)
    # Get user names from map if provided
    user_names = user_names_map or {}
    created_by_name = user_names.get(recipe.created_by)
    updated_by_name = user_names.get(recipe.updated_by)

    return RecipeResponse(
        id=recipe.id,
        name=recipe.name,
        product_code=recipe.product_code,
        description=recipe.description,
        delay_reject=recipe.delay_reject if hasattr(recipe, 'delay_reject') else 100.0,
        reject_pulse=recipe.reject_pulse if hasattr(recipe, 'reject_pulse') else 50.0,
        reject_method=recipe.reject_method if hasattr(recipe, 'reject_method') else 'DIO_OUT',
        do_reject_number=recipe.do_reject_number if hasattr(recipe, 'do_reject_number') else 0,
        do_alarm_number=recipe.do_alarm_number if hasattr(recipe, 'do_alarm_number') else 0,
        allow_late_reject=getattr(recipe, 'allow_late_reject', False),
        normal_pulse_ms=getattr(recipe, 'normal_pulse_ms', 0.0),
        cameras=cameras_data,
        camera_templates=camera_templates_data,
        camera_settings=recipe.camera_settings.model_dump() if hasattr(recipe.camera_settings, 'model_dump') and recipe.camera_settings else recipe.camera_settings,
        model_thresholds=recipe.model_thresholds.model_dump() if hasattr(recipe.model_thresholds, 'model_dump') else recipe.model_thresholds,
        template_config=recipe.template_config,
        roi_config=recipe.roi_config,
        is_active=recipe.is_active,
        ocr_model_type=getattr(recipe, 'ocr_model_type', None),
        ml_project_id=getattr(recipe, 'ml_project_id', None),
        ml_model_id=getattr(recipe, 'ml_model_id', None),
        defect_model=getattr(recipe, 'defect_model', None),
        classifier_backend=getattr(recipe, 'classifier_backend', None),
        cv_method=getattr(recipe, 'cv_method', None),
        template_bank_enabled=getattr(recipe, 'template_bank_enabled', False),
        template_bank_size=getattr(recipe, 'template_bank_size', 10),
        char_denoise_enabled=getattr(recipe, 'char_denoise_enabled', False),
        product_detection_method=getattr(recipe, 'product_detection_method', 'yolo_obb'),
        cap_rotation_method=getattr(recipe, 'cap_rotation_method', 'yolo_obb'),
        cap_crop_method=getattr(recipe, 'cap_crop_method', 'none'),
        crop_match_method=getattr(recipe, 'crop_match_method', 'superpoint'),
        dual_rotation_check=getattr(recipe, 'dual_rotation_check', False),
        product_box_wall_type=getattr(recipe, 'product_box_wall_type', 'outer'),
        wrinkle_conf=getattr(recipe, 'wrinkle_conf', None),
        wrinkle_show_when_pass=getattr(recipe, 'wrinkle_show_when_pass', None),
        matching_conf=getattr(recipe, 'matching_conf', None),
        mask_overlap_threshold=getattr(recipe, 'mask_overlap_threshold', None),
        match_erosion_enabled=getattr(recipe, 'match_erosion_enabled', False),
        match_erosion_kernel_w=getattr(recipe, 'match_erosion_kernel_w', 80),
        match_erosion_kernel_h=getattr(recipe, 'match_erosion_kernel_h', 1),
        match_erosion_iterations=getattr(recipe, 'match_erosion_iterations', 1),
        created_by=recipe.created_by,
        updated_by=recipe.updated_by,
        created_by_name=created_by_name,
        updated_by_name=updated_by_name,
        created_at=recipe.created_at,
        updated_at=recipe.updated_at
    )


def _to_primitive(obj):
    """Recursively convert Pydantic/model objects to Python primitives safe for BSON.

    - Handles None, primitives, dicts, lists/tuples/sets
    - Supports Pydantic v2 (`model_dump`) and v1 (`dict`) and falls back to `vars()`
    - Final fallback converts to str(obj)
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _to_primitive(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_primitive(v) for v in obj]
    # Pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            return _to_primitive(obj.model_dump())
        except Exception:
            pass
    # Pydantic v1
    if hasattr(obj, "dict"):
        try:
            return _to_primitive(obj.dict())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return _to_primitive(vars(obj))
        except Exception:
            pass
    return str(obj)


def _enrich_cameras_with_function_type(cameras_config, camera_templates_list):
    """
    Add function_type from camera_templates to each camera config.

    Args:
        cameras_config: List of camera configurations (primitives)
        camera_templates_list: List of camera templates with function_type

    Returns:
        List of enriched camera configs with function_type added
    """
    # Create mapping: camera_id -> function_type
    camera_function_map = {}
    for cam_template in camera_templates_list:
        if isinstance(cam_template, dict):
            camera_function_map[cam_template.get('camera_id')] = cam_template.get('function_type', 'OCR')
        else:
            camera_function_map[getattr(cam_template, 'camera_id', None)] = getattr(cam_template, 'function_type', 'OCR')

    # Enrich each camera config
    enriched_cameras = []
    for cam_config in cameras_config:
        cam_dict = cam_config.copy() if isinstance(cam_config, dict) else cam_config
        camera_id = cam_dict.get('camera_id')
        cam_dict['function_type'] = camera_function_map.get(camera_id, 'OCR')
        enriched_cameras.append(cam_dict)

    return enriched_cameras


async def _resolve_camera_serials(cameras: list, camera_repo) -> list:
    """
    Resolve stale serial_number/model_name in cameras list using current Camera table data.

    For each camera, lookup by camera_id → override serial_number and model_name
    with current values. This ensures recipes work correctly after camera hardware replacement.
    """
    resolved = []
    for cam in cameras:
        cam_copy = dict(cam)
        camera_id = cam_copy.get('camera_id')
        if camera_id:
            current = await camera_repo.get_by_id(camera_id)
            if current:
                old_serial = cam_copy.get('serial_number')
                new_serial = current.get('serial_number')
                if old_serial and new_serial and old_serial != new_serial:
                    logger.info(
                        f"[RESOLVE] Camera {camera_id}: serial {old_serial} → {new_serial}"
                    )
                if new_serial:
                    cam_copy['serial_number'] = new_serial
                if current.get('model_name'):
                    cam_copy['model_name'] = current['model_name']
        resolved.append(cam_copy)
    return resolved


@router.post("/", response_model=RecipeResponse, status_code=status.HTTP_201_CREATED)
async def create_recipe(
    recipe: RecipeCreate,
    request: Request,
    current_user: UserInDB = Depends(require_supervisor),
    recipe_repo: RecipeRepository = Depends(get_recipe_repository),
    action_log_repo: ActionLogRepository = Depends(get_action_log_repository)
):
    """
    Create a new recipe (Receipt).
    
    **Permission**: Supervisor and Admin only
    
    According to requirements:
    - Supervisor: Can create new receipts and edit old receipt content
    - Admin: Has all permissions
    """
    # Check if recipe with same name already exists
    existing_recipe = await recipe_repo.get_by_name(recipe.name)
    if existing_recipe:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Recipe with name '{recipe.name}' already exists"
        )
    
    # Check if recipe with same product code already exists
    existing_product = await recipe_repo.get_by_product_code(recipe.product_code)
    if existing_product:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Recipe with product code '{recipe.product_code}' already exists"
        )
    
    created_recipe = await recipe_repo.create(recipe, current_user.id)

    # Log the action
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    action_log = ActionLogCreate(
        user_id=current_user.id,
        username=current_user.username,
        action_type=ActionType.CREATE_RECIPE,
        resource_type="recipe",
        resource_id=created_recipe.id,
        description=f"Created recipe '{created_recipe.name}' with product code '{created_recipe.product_code}'",
        new_value={
            "name": created_recipe.name,
            "product_code": created_recipe.product_code,
            "description": created_recipe.description,
            "is_active": created_recipe.is_active
        },
        ip_address=client_ip,
        user_agent=user_agent
    )
    await action_log_repo.create_action_log(action_log)

    return recipe_to_response(created_recipe)


@router.get("/", response_model=List[RecipeResponse])
async def list_recipes(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    is_active: Optional[bool] = None,
    sort_by: str = Query(
        "created_desc",
        pattern="^(created|updated)_(desc|asc)$",
        description="Sort order: created_desc (default), created_asc, updated_desc, updated_asc",
    ),
    current_user: UserInDB = Depends(get_current_user),
    recipe_repo: RecipeRepository = Depends(get_recipe_repository),
    user_repo: UserRepository = Depends(get_user_repository)
):
    """
    List all recipes with pagination.

    **Permission**: All authenticated users (Operator, Supervisor, Admin)

    According to requirements:
    - Operator: Can only load receipt and input datecode field to verify with read content
    """
    recipes = await recipe_repo.get_all(skip=skip, limit=limit, is_active=is_active, sort_by=sort_by)

    # Collect unique user IDs for batch lookup
    user_ids = set()
    for recipe in recipes:
        if recipe.created_by:
            user_ids.add(recipe.created_by)
        if recipe.updated_by:
            user_ids.add(recipe.updated_by)

    # Batch lookup user names
    user_names_map = await user_repo.get_users_by_ids(list(user_ids)) if user_ids else {}
    return [recipe_to_response(recipe, user_names_map) for recipe in recipes]


@router.get("/search", response_model=List[RecipeResponse])
async def search_recipes(
    q: str = Query(..., min_length=1, description="Search query"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    sort_by: str = Query(
        "created_desc",
        pattern="^(created|updated)_(desc|asc)$",
        description="Sort order: created_desc (default), created_asc, updated_desc, updated_asc",
    ),
    current_user: UserInDB = Depends(get_current_user),
    recipe_repo: RecipeRepository = Depends(get_recipe_repository),
    user_repo: UserRepository = Depends(get_user_repository)
):
    """
    Search recipes by name, product code, or description.

    **Permission**: All authenticated users
    """
    recipes = await recipe_repo.search(q, skip=skip, limit=limit, sort_by=sort_by)

    # Batch lookup user names
    user_ids = set()
    for recipe in recipes:
        if recipe.created_by:
            user_ids.add(recipe.created_by)
        if recipe.updated_by:
            user_ids.add(recipe.updated_by)
    user_names_map = await user_repo.get_users_by_ids(list(user_ids)) if user_ids else {}

    return [recipe_to_response(recipe, user_names_map) for recipe in recipes]



# Move static routes before path-parameter routes so they are matched first.


@router.get("/loads/latest")
async def get_latest_load(
    load_repo: ReceiptLoadRepository = Depends(get_receipt_load_repository),
):
    """Public endpoint: return the most recent running recipe load (no auth required)."""
    item = await load_repo.get_latest_running()
    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No running recipe")

    # Resolve stale camera serials in the snapshot using current Camera table data
    try:
        from app.repositories.camera_repository import CameraRepository as _CameraRepo
        _camera_repo = _CameraRepo(get_database())
        metadata = item.get('metadata') or {}
        cameras = metadata.get('cameras') or []
        if cameras:
            metadata['cameras'] = await _resolve_camera_serials(cameras, _camera_repo)
            item['metadata'] = metadata
    except Exception as e:
        logger.warning(f"[RESOLVE] Failed to resolve camera serials in latest load: {e}")

    return item
@router.get("/loads")
async def list_all_loads(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    current_user: UserInDB = Depends(require_operator),
    load_repo: ReceiptLoadRepository = Depends(get_receipt_load_repository),
):
    """
    Return all load events (admin/supervisor only).
    """
    items = await load_repo.list_all(skip=skip, limit=limit)

    # Attach visualization_url for each template when possible (generate deterministic viz files)
    for item in items:
        try:
            metadata = item.get('metadata') or {}
            cams = metadata.get('camera_templates') or []
            for cam in cams:
                templates = cam.get('templates') or []
                for tpl in templates:
                    image_url = tpl.get('image_url')
                    tpl['visualization_url'] = None
                    if not image_url:
                        continue
                    filename = image_url.split('/')[-1]
                    src_path = TEMPLATE_UPLOAD_DIR / filename
                    if not src_path.exists():
                        continue
                    annotations = tpl.get('annotations') or []
                    if not annotations:
                        # no annotations: point to raw template image (relative URL)
                        tpl['visualization_url'] = f"/api/recipes/templates/images/{filename}"
                        continue
                    try:
                        # include the load timestamp to create per-load visualization filenames
                        viz_url = _draw_template_visualization(src_path, annotations, loaded_at=item.get('loaded_at'))
                        tpl['visualization_url'] = viz_url
                    except Exception:
                        traceback.print_exc()
                        tpl['visualization_url'] = None
        except Exception:
            # be resilient to malformed metadata
            continue

    return {
        'items': items,
        'count': await load_repo.count_all()
    }


@router.get("/loads/template-images")
async def get_template_images_by_timestamp(
    recipe_id: str = Query(..., description="Recipe ID"),
    timestamp: str = Query(..., description="Inference timestamp in ISO format (UTC)"),
    serial_number: Optional[str] = Query(None, description="Camera serial number to filter"),
    load_repo: ReceiptLoadRepository = Depends(get_receipt_load_repository),
):
    """
    Get template images for a recipe at a specific timestamp.

    This endpoint finds the recipe load that was active at the given inference timestamp
    and returns the template images and metadata.

    **Permission**: Public (no auth required for historical data access)

    Returns:
        {
            "recipe_load_id": "...",
            "loaded_at": "...",
            "templates": [
                {
                    "camera_id": "CAM004",
                    "serial_number": "40272812",
                    "function_type": "Check_Type_Product",
                    "templates": [
                        {
                            "name": "Template 1",
                            "image_url": "/api/recipes/templates/images/xxx.jpeg",
                            "visualization_url": "/api/recipes/templates/visualizations/xxx.png"
                        }
                    ]
                }
            ]
        }
    """
    # Parse timestamp
    try:
        from dateutil import parser
        ts = parser.isoparse(timestamp)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid timestamp format: {str(e)}"
        )

    # Get recipe load at the timestamp
    load_doc = await load_repo.get_template_images_at_timestamp(
        recipe_id=recipe_id,
        timestamp=ts,
        serial_number=serial_number
    )

    if not load_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No recipe load found for the given recipe_id and timestamp"
        )

    # Extract template data from metadata
    metadata = load_doc.get('metadata') or {}
    camera_templates = metadata.get('camera_templates') or []

    # Filter by serial_number if provided
    if serial_number:
        camera_templates = [
            ct for ct in camera_templates
            if metadata.get('cameras', [])
            and any(
                cam.get('serial_number') == serial_number
                for cam in metadata.get('cameras', [])
                if cam.get('camera_id') == ct.get('camera_id')
            )
        ]

    # Enrich each template with camera serial_number and generate visualization URLs
    templates_output = []
    cameras = metadata.get('cameras') or []

    for cam_template in camera_templates:
        camera_id = cam_template.get('camera_id')

        # Find matching camera config to get serial_number
        cam_config = next((c for c in cameras if c.get('camera_id') == camera_id), None)
        cam_serial = cam_config.get('serial_number') if cam_config else None

        templates_list = cam_template.get('templates') or []

        # Generate visualization URLs for each template
        enriched_templates = []
        for tpl in templates_list:
            image_url = tpl.get('image_url')
            viz_url = None

            if image_url:
                filename = image_url.split('/')[-1]
                src_path = TEMPLATE_UPLOAD_DIR / filename

                if src_path.exists():
                    annotations = tpl.get('annotations') or []
                    if annotations:
                        try:
                            viz_url = _draw_template_visualization(
                                src_path,
                                annotations,
                                loaded_at=load_doc.get('loaded_at')
                            )
                        except Exception:
                            traceback.print_exc()
                            viz_url = image_url  # Fallback to raw image
                    else:
                        viz_url = image_url  # No annotations, use raw image

            enriched_templates.append({
                'name': tpl.get('name'),
                'image_url': image_url,
                'visualization_url': viz_url
            })

        templates_output.append({
            'camera_id': camera_id,
            'serial_number': cam_serial,
            'function_type': cam_template.get('function_type'),
            'templates': enriched_templates
        })

    return {
        'recipe_load_id': load_doc.get('id'),
        'loaded_at': load_doc.get('loaded_at'),
        'templates': templates_output
    }


@router.get("/{recipe_id}/loads")
async def list_recipe_loads(
    recipe_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    current_user: UserInDB = Depends(get_current_user),
    load_repo: ReceiptLoadRepository = Depends(get_receipt_load_repository),
):
    """List load events for a specific recipe."""
    items = await load_repo.list_by_recipe(recipe_id=recipe_id, skip=skip, limit=limit)

    return {
        'items': items,
        'count': await load_repo.count_by_recipe(recipe_id)
    }



@router.post("/templates/detect-bottle-preview")
async def detect_bottle_preview(body: dict):
    """
    Run the image-proc bottle detector (`color_verifier._detect_bottle`) on a
    saved template image, using the same tuning params that ColorSetupModal will
    persist into `color_config`. Returns the detected bottle bbox so the FE can
    overlay it on the canvas for tuning feedback.

    Body:
        image_url:                str          (e.g. /api/recipes/templates/images/xxx.jpg)
        product_polygon:          [[x,y], ...] (TEMPLATE pixel coords)
        crop_area:                {x1, y1, x2, y2}  (optional — TEMPLATE pixel coords)
        sharp_threshold:          float        (default 0.30)
        min_height_ratio:         float        (default 0.20)
        min_aspect:               float        (default 1.2)

    Returns:
        {
          detected: bool,
          bbox: {x, y, w, h} | null   # in TEMPLATE pixel coords (offset applied)
          corners: [[x,y]×4] | null
          score: float | null
          reason: str | null
        }
    """
    import sys
    import numpy as np
    import cv2

    image_url = body.get("image_url", "")
    product_polygon = body.get("product_polygon")
    crop_area = body.get("crop_area")
    sharp_threshold = float(body.get("sharp_threshold", 0.30))
    min_height_ratio = float(body.get("min_height_ratio", 0.20))
    min_aspect = float(body.get("min_aspect", 1.2))

    if not image_url or not product_polygon:
        raise HTTPException(400, "image_url and product_polygon are required")
    if len(product_polygon) < 3:
        raise HTTPException(400, "product_polygon must have at least 3 points")

    filename = image_url.split("/")[-1]
    image_path = TEMPLATE_UPLOAD_DIR / filename
    if not image_path.exists():
        raise HTTPException(404, f"Template image not found: {filename}")

    # Add ai_services to sys.path so we can reuse ColorVerificationService._detect_bottle
    home = os.environ.get("HOME", "")
    ai_path = os.path.join(home, "Source", "ocr_datecode", "ai_services")
    if ai_path not in sys.path:
        sys.path.insert(0, ai_path)
    try:
        from camera_management.verification.color_verifier import ColorVerificationService
    except Exception as e:
        raise HTTPException(503, f"color_verifier import failed: {e}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise HTTPException(500, "Failed to read template image")

    H, W = img.shape[:2]

    # Apply crop_area if provided (already in pixel coords)
    crop_offset = (0, 0)
    crop_img = img
    if crop_area:
        x1 = max(0, int(crop_area.get("x1", 0)))
        y1 = max(0, int(crop_area.get("y1", 0)))
        x2 = min(W, int(crop_area.get("x2", W)))
        y2 = min(H, int(crop_area.get("y2", H)))
        if x2 > x1 and y2 > y1:
            crop_img = img[y1:y2, x1:x2]
            crop_offset = (x1, y1)

    # Build polygon in CROP coords (subtract crop offset)
    poly = np.array(product_polygon, dtype=np.float32)
    poly_in_crop = poly - np.array(crop_offset, dtype=np.float32)

    bottle = ColorVerificationService._detect_bottle(
        crop_img, poly_in_crop, crop_offset,
        sharp_threshold=sharp_threshold,
        min_height_ratio=min_height_ratio,
        min_aspect=min_aspect,
    )
    if bottle is None:
        return {
            "detected": False, "bbox": None, "corners": None,
            "score": None, "reason": "No bottle candidate matched the criteria",
        }

    box = bottle.get("box", [0, 0, 0, 0, 0])
    cx, cy, bw, bh = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    x = int(cx - bw / 2.0)
    y = int(cy - bh / 2.0)
    return {
        "detected": True,
        "bbox": {"x": x, "y": y, "w": int(bw), "h": int(bh)},
        "corners": bottle.get("corners"),
        "score": float(bottle.get("score", 0.0)),
        "reason": None,
    }


@router.post("/templates/upload")
async def upload_template_image(
    file: UploadFile = File(...),
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Upload a template image and return its URL and dimensions.
    
    **Permission**: All authenticated users
    
    Returns:
        {
            "url": "/api/recipes/templates/images/{filename}",
            "width": 1920,
            "height": 1080
        }
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )
    
    # Read image data
    image_data = await file.read()
    
    # Get image dimensions using PIL
    try:
        image = Image.open(io.BytesIO(image_data))
        width, height = image.size
        image_format = image.format.lower() if image.format else 'jpg'
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {str(e)}"
        )
    
    # Generate unique filename
    file_extension = file.filename.split('.')[-1] if '.' in file.filename else image_format
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = TEMPLATE_UPLOAD_DIR / unique_filename
    
    # Save file
    with open(file_path, 'wb') as f:
        f.write(image_data)
    
    # Return URL and dimensions
    return {
        "url": f"/api/recipes/templates/images/{unique_filename}",
        "width": width,
        "height": height
    }


@router.get("/templates/images/{filename}")
async def get_template_image(filename: str):
    """
    Serve a template image file.
    
    **Permission**: Public (no auth required for image serving)
    """
    file_path = TEMPLATE_UPLOAD_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
    
    return FileResponse(
        file_path,
        media_type="image/png",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )


@router.get("/templates/visualizations/{filename}")
async def get_visualization_image(filename: str):
    file_path = TEMPLATE_VISUALIZE_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Visualization not found")
    return FileResponse(
        file_path,
        media_type="image/png",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

@router.options("/templates/visualizations/{filename}")
async def options_visualization_image(filename: str):
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )


@router.options("/templates/images/{filename}")
async def options_template_image(filename: str):
    """Handle CORS preflight request"""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )



@router.post("/templates/segment")
async def segment_template_region(body: dict):
    """
    Auto-segment characters inside a drawn region on a template image.

    Body:
      image_url: str
      region:    {x, y, w, h}   (normalized 0-1)
      with_ocr:  bool           (optional, default False) — when true, runs
                                rec.onnx per crop and attaches `expected_text`
                                to each segment for FE auto-fill of char
                                annotations.
    Returns: {segments: [{id, x, y, w, h, expected_text?}, ...], count, full_text?}
    """
    image_url = body.get("image_url", "")
    region = body.get("region")
    with_ocr = bool(body.get("with_ocr", False))
    if not image_url or not region:
        raise HTTPException(400, "image_url and region are required")

    # Extract filename from URL (e.g. /api/recipes/templates/images/abc.jpg → abc.jpg)
    filename = image_url.split("/")[-1]
    image_path = TEMPLATE_UPLOAD_DIR / filename
    if not image_path.exists():
        raise HTTPException(404, f"Template image not found: {filename}")

    try:
        from app.services.ml_segment_service import segment_region, crop_segment
        loop = asyncio.get_event_loop()
        segments = await loop.run_in_executor(None, segment_region, image_path, region)

        full_text: Optional[str] = None
        if with_ocr and segments:
            # Reading-order sort: row-bucket by y, then x within each row.
            # Keeps single-line text in the right order even when box y's
            # jitter; multi-line text still groups rows correctly.
            heights = sorted((s.get("h", 0) for s in segments))
            median_h = heights[len(heights) // 2] if heights else 0.0
            bucket_sz = max(median_h * 0.4, 1e-6)
            segments.sort(key=lambda s: (int(s.get("y", 0) / bucket_sz), s.get("x", 0)))

            from app.services.ml_char_ocr_service import recognize_text, recognize_chars
            import cv2 as cv

            def _ocr_region_and_map() -> List[Optional[str]]:
                """
                Primary path — OCR the whole drawn region once (word-level),
                then position-map each character to a sorted segment box.

                Fallback — if character count from word-OCR doesn't match the
                segment count, OCR each crop individually (less accurate but
                at least gives 1:1 mapping the FE can edit).
                """
                # 1. Crop the full region from the original image
                img = cv.imread(str(image_path))
                if img is None:
                    return [None] * len(segments)
                ih, iw = img.shape[:2]
                rx = max(0, int(region["x"] * iw))
                ry = max(0, int(region["y"] * ih))
                rw = min(iw - rx, int(region["w"] * iw))
                rh = min(ih - ry, int(region["h"] * ih))
                if rw <= 0 or rh <= 0:
                    return [None] * len(segments)
                region_crop = img[ry:ry + rh, rx:rx + rw]

                # 2. Word-level OCR on the whole region
                word = recognize_text(region_crop)

                if word and len(word) == len(segments):
                    # 3a. Perfect match — map char-by-char to sorted boxes
                    return list(word)

                # 3b. Mismatch — fall back to per-char OCR of each sorted box
                crops = [crop_segment(image_path, s) for s in segments]
                valid_idx = [i for i, c in enumerate(crops) if c is not None]
                valid_crops = [crops[i] for i in valid_idx]
                if not valid_crops:
                    return [None] * len(segments)
                texts = recognize_chars(valid_crops)
                out: List[Optional[str]] = [None] * len(segments)
                for i, t in zip(valid_idx, texts):
                    out[i] = t
                logger.info(
                    f"[segment_with_ocr] word='{word}' ({len(word)} chars) "
                    f"vs segments={len(segments)} — fell back to per-char"
                )
                return out

            try:
                texts = await loop.run_in_executor(None, _ocr_region_and_map)
                for seg, text in zip(segments, texts):
                    seg["expected_text"] = ((text or "").strip()) or None
                full_text = "".join((t or "") for t in texts)
            except Exception:
                logger.exception("Region OCR failed; returning boxes only")
    except Exception as e:
        logger.exception("Template segmentation failed")
        raise HTTPException(500, f"Segmentation error: {e}")

    resp: Dict[str, Any] = {"segments": segments, "count": len(segments)}
    if full_text is not None:
        resp["full_text"] = full_text
    return resp


@router.get("/{recipe_id}", response_model=RecipeResponse)
async def get_recipe(
    recipe_id: str,
    current_user: UserInDB = Depends(get_current_user),
    recipe_repo: RecipeRepository = Depends(get_recipe_repository)
):
    """
    Load a specific recipe by ID.
    
    **Permission**: All authenticated users
    
    According to requirements:
    - Operator: Can load receipt
    """
    recipe = await recipe_repo.get_by_id(recipe_id)
    if not recipe:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recipe not found"
        )
    
    return recipe_to_response(recipe)


@router.post("/{recipe_id}/clone", response_model=RecipeResponse, status_code=status.HTTP_201_CREATED)
async def clone_recipe(
    recipe_id: str,
    request: Request,
    current_user: UserInDB = Depends(require_supervisor),
    recipe_repo: RecipeRepository = Depends(get_recipe_repository),
    action_log_repo: ActionLogRepository = Depends(get_action_log_repository)
):
    """
    Clone an existing recipe with a new ID.

    **Permission**: Supervisor and Admin only

    The cloned recipe will have:
    - A new unique ID (UUID)
    - Name with " (Copy)" suffix
    - Same is_active status as original
    - All other settings identical to the original
    """
    # Get original recipe
    original_recipe = await recipe_repo.get_by_id(recipe_id)
    if not original_recipe:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recipe not found"
        )

    # Create new recipe name with (Copy) suffix
    new_name = f"{original_recipe.name} (Copy)"

    # Check if recipe with this name already exists
    counter = 1
    final_name = new_name
    while await recipe_repo.get_by_name(final_name):
        counter += 1
        final_name = f"{original_recipe.name} (Copy {counter})"

    # Convert Pydantic models to dict for RecipeCreate
    # Helper function to convert model or list of models to dict
    def to_dict(obj):
        if obj is None:
            return None
        if isinstance(obj, list):
            return [item.model_dump() if hasattr(item, 'model_dump') else item for item in obj]
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        return obj

    # Create new recipe data from original
    recipe_data = RecipeCreate(
        name=final_name,
        product_code=original_recipe.product_code,
        description=original_recipe.description,
        cameras=to_dict(original_recipe.cameras) or [],
        camera_templates=to_dict(original_recipe.camera_templates) or [],
        delay_reject=getattr(original_recipe, 'delay_reject', 100.0),
        reject_pulse=getattr(original_recipe, 'reject_pulse', 50.0),
        reject_method=getattr(original_recipe, 'reject_method', 'DIO_OUT'),
        do_reject_number=getattr(original_recipe, 'do_reject_number', 0),
        do_alarm_number=getattr(original_recipe, 'do_alarm_number', 0),
        allow_late_reject=getattr(original_recipe, 'allow_late_reject', False),
        normal_pulse_ms=getattr(original_recipe, 'normal_pulse_ms', 0.0),
        camera_settings=to_dict(original_recipe.camera_settings),
        model_thresholds=to_dict(original_recipe.model_thresholds),
        template_config=original_recipe.template_config,
        roi_config=original_recipe.roi_config,
        is_active=original_recipe.is_active,
        ocr_model_type=getattr(original_recipe, 'ocr_model_type', None),
        ml_project_id=getattr(original_recipe, 'ml_project_id', None),
        ml_model_id=getattr(original_recipe, 'ml_model_id', None),
        defect_model=getattr(original_recipe, 'defect_model', None),
        classifier_backend=getattr(original_recipe, 'classifier_backend', None),
        cv_method=getattr(original_recipe, 'cv_method', None),
        template_bank_enabled=getattr(original_recipe, 'template_bank_enabled', False),
        template_bank_size=getattr(original_recipe, 'template_bank_size', 10),
        char_denoise_enabled=getattr(original_recipe, 'char_denoise_enabled', False),
        product_detection_method=getattr(original_recipe, 'product_detection_method', 'yolo_obb'),
        cap_rotation_method=getattr(original_recipe, 'cap_rotation_method', 'yolo_obb'),
        cap_crop_method=getattr(original_recipe, 'cap_crop_method', 'none'),
        crop_match_method=getattr(original_recipe, 'crop_match_method', 'superpoint'),
        dual_rotation_check=getattr(original_recipe, 'dual_rotation_check', False),
        product_box_wall_type=getattr(original_recipe, 'product_box_wall_type', 'outer'),
        wrinkle_conf=getattr(original_recipe, 'wrinkle_conf', None),
        wrinkle_show_when_pass=getattr(original_recipe, 'wrinkle_show_when_pass', None),
        matching_conf=getattr(original_recipe, 'matching_conf', None),
        mask_overlap_threshold=getattr(original_recipe, 'mask_overlap_threshold', None),
        match_erosion_enabled=getattr(original_recipe, 'match_erosion_enabled', False),
        match_erosion_kernel_w=getattr(original_recipe, 'match_erosion_kernel_w', 80),
        match_erosion_kernel_h=getattr(original_recipe, 'match_erosion_kernel_h', 1),
        match_erosion_iterations=getattr(original_recipe, 'match_erosion_iterations', 1),
    )

    # Create the cloned recipe
    cloned_recipe = await recipe_repo.create(recipe_data, current_user.id)

    # Log the action
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    action_log = ActionLogCreate(
        user_id=current_user.id,
        username=current_user.username,
        action_type=ActionType.CREATE_RECIPE,
        resource_type="recipe",
        resource_id=cloned_recipe.id,
        description=f"Cloned recipe '{original_recipe.name}' to '{cloned_recipe.name}' (from ID: {recipe_id})",
        new_value={
            "name": cloned_recipe.name,
            "product_code": cloned_recipe.product_code,
            "description": cloned_recipe.description,
            "is_active": cloned_recipe.is_active,
            "cloned_from": recipe_id
        },
        ip_address=client_ip,
        user_agent=user_agent
    )
    await action_log_repo.create_action_log(action_log)

    return recipe_to_response(cloned_recipe)


@router.post("/{recipe_id}/load", response_model=ReceiptLoadResponse)
async def load_recipe(
    recipe_id: str,
    request: Request,
    current_user: UserInDB = Depends(get_current_user),
    recipe_repo: RecipeRepository = Depends(get_recipe_repository),
    load_repo: ReceiptLoadRepository = Depends(get_receipt_load_repository),
    action_log_repo: ActionLogRepository = Depends(get_action_log_repository)
):
    """
    Record a recipe load event. Stores who loaded the recipe and when.
    Also applies camera trigger configuration from recipe to connected cameras.
    """
    from app.services.camera_settings_service import camera_settings_service

    # Ensure recipe exists
    recipe = await recipe_repo.get_by_id(recipe_id)
    if not recipe:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recipe not found"
        )
    # Enforce presence of camera_templates before allowing load
    camera_templates = getattr(recipe, 'camera_templates', None)
    if not camera_templates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Recipe has no camera templates. Please draw templates before loading."
        )

    # Check if CameraManagement service is connected
    if not camera_ws_manager.is_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Camera Management service is not connected. Please start the service."
        )

    # Check and auto-connect cameras if needed
    cameras_config = getattr(recipe, 'cameras', [])
    if cameras_config:
        from app.repositories.camera_repository import CameraRepository
        from app.db.mongodb import get_database
        from app.api.websocket.camera_ws import send_connect_camera

        db = get_database()
        camera_repo = CameraRepository(db)

        for cam_config in cameras_config:
            # Handle both dict and Pydantic model
            if isinstance(cam_config, dict):
                serial_number = cam_config.get('serial_number')
                pixel_format = cam_config.get('pixel_format', 'BGR8')
            else:
                # Pydantic model
                serial_number = getattr(cam_config, 'serial_number', None)
                pixel_format = getattr(cam_config, 'pixel_format', 'BGR8')


            if not serial_number:
                continue

            # Check if camera is connected in DB
            camera = await camera_repo.get_by_serial(serial_number)
            if camera and not camera.get('is_connected', False):
                logger.info(f"Auto-connecting camera {serial_number} for recipe {recipe.name}")

                # Send connect command via WebSocket
                success = await send_connect_camera(serial_number, pixel_format)

                if success:
                    logger.info(f"Successfully sent connect command for camera {serial_number}")
                else:
                    logger.warning(f"Failed to send connect command for camera {serial_number}")

    # Enrich cameras with function_type from camera_templates
    cameras_config = getattr(recipe, 'cameras', [])
    camera_templates_list = getattr(recipe, 'camera_templates', [])

    # Convert to primitive and enrich with function_type
    cameras_primitive = _to_primitive(cameras_config)
    enriched_cameras = _enrich_cameras_with_function_type(
        cameras_primitive,
        camera_templates_list
    )

    # Resolve stale serial_number/model_name using current Camera table data
    from app.repositories.camera_repository import CameraRepository as _CameraRepo
    _camera_repo = _CameraRepo(get_database())
    enriched_cameras = await _resolve_camera_serials(enriched_cameras, _camera_repo)

    # Build metadata snapshot (will be used for both storage and inference)
    metadata = {
        'recipe_id': recipe_id,
        'name': recipe.name,
        'delay_reject': recipe.delay_reject,
        'reject_pulse': recipe.reject_pulse if hasattr(recipe, 'reject_pulse') else 50.0,
        'reject_method': getattr(recipe, 'reject_method', 'DIO_OUT'),
        'do_reject_number': recipe.do_reject_number,
        'do_alarm_number': getattr(recipe, 'do_alarm_number', 0),
        'allow_late_reject': getattr(recipe, 'allow_late_reject', False),
        'normal_pulse_ms': getattr(recipe, 'normal_pulse_ms', 0.0),
        'product_code': recipe.product_code,
        'camera_templates': _to_primitive(camera_templates),
        'model_thresholds': _to_primitive(getattr(recipe, 'model_thresholds', None)),
        'cameras': enriched_cameras,  # Use enriched cameras with function_type
        'ocr_model_type': getattr(recipe, 'ocr_model_type', None),
        'ml_project_id': getattr(recipe, 'ml_project_id', None),
        'ml_model_id': getattr(recipe, 'ml_model_id', None),
        'defect_model': getattr(recipe, 'defect_model', None),
        'classifier_backend': getattr(recipe, 'classifier_backend', None),
        'cv_method': getattr(recipe, 'cv_method', None) or 'legacy',
        'template_bank_enabled': getattr(recipe, 'template_bank_enabled', False),
        'template_bank_size': getattr(recipe, 'template_bank_size', 10),
        'char_denoise_enabled': getattr(recipe, 'char_denoise_enabled', False),
        'product_detection_method': getattr(recipe, 'product_detection_method', None) or 'yolo_obb',
        'cap_rotation_method': getattr(recipe, 'cap_rotation_method', None) or 'yolo_obb',
        'cap_crop_method': getattr(recipe, 'cap_crop_method', None) or 'none',
        'crop_match_method': getattr(recipe, 'crop_match_method', None) or 'superpoint',
        'dual_rotation_check': bool(getattr(recipe, 'dual_rotation_check', False)),
        'product_box_wall_type': getattr(recipe, 'product_box_wall_type', None) or 'outer',
        'wrinkle_conf': getattr(recipe, 'wrinkle_conf', None),
        'wrinkle_show_when_pass': getattr(recipe, 'wrinkle_show_when_pass', None),
        'matching_conf': getattr(recipe, 'matching_conf', None),
        'mask_overlap_threshold': getattr(recipe, 'mask_overlap_threshold', None),
        'match_erosion_enabled': getattr(recipe, 'match_erosion_enabled', False),
        'match_erosion_kernel_w': getattr(recipe, 'match_erosion_kernel_w', 80),
        'match_erosion_kernel_h': getattr(recipe, 'match_erosion_kernel_h', 1),
        'match_erosion_iterations': getattr(recipe, 'match_erosion_iterations', 1),
    }

    # Send load recipe command via WebSocket to CameraManagement service
    # Convert recipe to dict for WebSocket transmission
    recipe_dict = {
        '_id': str(recipe.id) if hasattr(recipe, 'id') else recipe_id,
        'id': recipe_id,
        'name': recipe.name,
        'product_code': recipe.product_code,
        'delay_reject': recipe.delay_reject,
        'reject_pulse': recipe.reject_pulse if hasattr(recipe, 'reject_pulse') else 50.0,
        'reject_method': getattr(recipe, 'reject_method', 'DIO_OUT'),
        'do_reject_number': recipe.do_reject_number,
        'do_alarm_number': getattr(recipe, 'do_alarm_number', 0),
        'allow_late_reject': getattr(recipe, 'allow_late_reject', False),
        'normal_pulse_ms': getattr(recipe, 'normal_pulse_ms', 0.0),
        'cameras': enriched_cameras,  # Use enriched cameras with function_type
        'camera_templates': _to_primitive(camera_templates),
        'model_thresholds': _to_primitive(getattr(recipe, 'model_thresholds', None)),
        'camera_settings': _to_primitive(getattr(recipe, 'camera_settings', None)),
        'template_config': _to_primitive(getattr(recipe, 'template_config', None)),
        'roi_config': _to_primitive(getattr(recipe, 'roi_config', None)),
        'ocr_model_type': getattr(recipe, 'ocr_model_type', None),
        'ml_project_id': getattr(recipe, 'ml_project_id', None),
        'ml_model_id': getattr(recipe, 'ml_model_id', None),
        'defect_model': getattr(recipe, 'defect_model', None),
        'classifier_backend': getattr(recipe, 'classifier_backend', None),
        'cv_method': getattr(recipe, 'cv_method', None) or 'legacy',
        'template_bank_enabled': getattr(recipe, 'template_bank_enabled', False),
        'template_bank_size': getattr(recipe, 'template_bank_size', 10),
        'char_denoise_enabled': getattr(recipe, 'char_denoise_enabled', False),
        'product_detection_method': getattr(recipe, 'product_detection_method', None) or 'yolo_obb',
        'cap_rotation_method': getattr(recipe, 'cap_rotation_method', None) or 'yolo_obb',
        'cap_crop_method': getattr(recipe, 'cap_crop_method', None) or 'none',
        'crop_match_method': getattr(recipe, 'crop_match_method', None) or 'superpoint',
        'dual_rotation_check': bool(getattr(recipe, 'dual_rotation_check', False)),
        'product_box_wall_type': getattr(recipe, 'product_box_wall_type', None) or 'outer',
        'wrinkle_conf': getattr(recipe, 'wrinkle_conf', None),
        'wrinkle_show_when_pass': getattr(recipe, 'wrinkle_show_when_pass', None),
        'matching_conf': getattr(recipe, 'matching_conf', None),
        'mask_overlap_threshold': getattr(recipe, 'mask_overlap_threshold', None),
        'match_erosion_enabled': getattr(recipe, 'match_erosion_enabled', False),
        'match_erosion_kernel_w': getattr(recipe, 'match_erosion_kernel_w', 80),
        'match_erosion_kernel_h': getattr(recipe, 'match_erosion_kernel_h', 1),
        'match_erosion_iterations': getattr(recipe, 'match_erosion_iterations', 1),
    }

    success = await send_load_recipe(recipe_dict)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send load recipe command to Camera Management service"
        )

    logger.info(f"✅ [RECIPE LOAD] Sent load recipe command via WebSocket: {recipe.name}")

    # Persist loader's full name to avoid frontend-side lookups
    created = await load_repo.create(
        recipe_id=recipe_id,
        user_id=current_user.id,
        metadata=metadata,
        user_full_name=getattr(current_user, 'full_name', None),
    )

    # Log the action
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    action_log = ActionLogCreate(
        user_id=current_user.id,
        username=current_user.username,
        action_type=ActionType.LOAD_RECIPE,
        resource_type="recipe",
        resource_id=recipe_id,
        description=f"Loaded recipe '{recipe.name}' with product code '{recipe.product_code}'",
        ip_address=client_ip,
        user_agent=user_agent
    )
    await action_log_repo.create_action_log(action_log)

    # Emit SocketIO event for recipe status change
    try:
        from app.services.socketio_service import emit_recipe_status_change
        await emit_recipe_status_change({
            'event': 'recipe_loaded',
            'recipe_id': recipe_id,
            'recipe_name': recipe.name,
            'product_code': recipe.product_code,
            'loaded_by': current_user.username
        })
        logger.info(f"✅ Emitted recipe_loaded event via SocketIO")
    except Exception as e:
        logger.error(f"Error emitting SocketIO event: {e}")

    return ReceiptLoadResponse(
        id=created['id'],
        recipe_id=created['recipe_id'],
        loaded_by=created['loaded_by'],
        loaded_by_full_name=created.get('loaded_by_full_name'),
        loaded_at=created['loaded_at'],
        metadata=created.get('metadata')
    )


@router.post("/{recipe_id}/stop", response_model=dict)
async def stop_recipe(
    recipe_id: str,
    request: Request,
    current_user: UserInDB = Depends(get_current_user),
    recipe_repo: RecipeRepository = Depends(get_recipe_repository),
    action_log_repo: ActionLogRepository = Depends(get_action_log_repository),
    load_repo: ReceiptLoadRepository = Depends(get_receipt_load_repository)
):
    """
    Stop a running recipe and set cameras to idle mode.

    - **recipe_id**: Recipe ID to stop

    Returns: Status message
    """
    # Ensure recipe exists
    recipe = await recipe_repo.get_by_id(recipe_id)
    if not recipe:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recipe not found"
        )

    # Check if CameraManagement service is connected
    # if not camera_ws_manager.is_connected():
    #     raise HTTPException(
    #         status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    #         detail="Camera Management service is not connected"
    #     )

    # Send stop recipe command via WebSocket
    success = await send_stop_recipe(recipe_id)

    # if not success:
    #     raise HTTPException(
    #         status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    #         detail="Failed to send stop recipe command"
    #     )

    logger.info(f"✅ [RECIPE STOP] Sent stop recipe command via WebSocket: {recipe.name}")

    # Update recipe_load status to 'stopped'
    stopped_load = await load_repo.stop_recipe_load(
        recipe_id=recipe_id,
        user_id=current_user.id,
        user_full_name=current_user.full_name
    )

    if stopped_load:
        logger.info(f"✅ [RECIPE STOP] Updated recipe_load status to 'stopped': {stopped_load['id']}")
    else:
        logger.warning(f"⚠️ [RECIPE STOP] No running recipe_load found for recipe {recipe_id}")

    # Log the action
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    action_log = ActionLogCreate(
        user_id=current_user.id,
        username=current_user.username,
        action_type=ActionType.STOP_RECIPE,
        resource_type="recipe",
        resource_id=recipe_id,
        description=f"Stopped recipe '{recipe.name}'",
        ip_address=client_ip,
        user_agent=user_agent
    )
    await action_log_repo.create_action_log(action_log)

    # Emit SocketIO event for recipe status change
    try:
        from app.services.socketio_service import emit_recipe_status_change
        await emit_recipe_status_change({
            'event': 'recipe_stopped',
            'recipe_id': recipe_id,
            'recipe_name': recipe.name,
            'stopped_by': current_user.username
        })
        logger.info(f"✅ Emitted recipe_stopped event via SocketIO")
    except Exception as e:
        logger.error(f"Error emitting SocketIO event: {e}")

    return {
        "success": True,
        "message": f"Recipe '{recipe.name}' stopped successfully",
        "recipe_id": recipe_id
    }


@router.post("/{recipe_id}/set-inference-mode")
async def set_inference_mode(
    recipe_id: str,
    mode: dict,
    current_user: UserInDB = Depends(get_current_user),
    recipe_repo: RecipeRepository = Depends(get_recipe_repository)
):
    """
    Set inference mode for a running recipe (ONLINE/OFFLINE)

    - **ONLINE**: Inference runs when triggers occur
    - **OFFLINE**: Triggers detected but inference skipped

    Args:
        recipe_id: Recipe ID
        mode: {"enabled": true/false} or {"mode": "online"/"offline"}

    Returns:
        Status message
    """
    # Ensure recipe exists
    recipe = await recipe_repo.get_by_id(recipe_id)
    if not recipe:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recipe not found"
        )

    # Parse mode parameter
    enabled = mode.get("enabled")
    if enabled is None:
        # Support both formats: {"enabled": bool} or {"mode": "online"/"offline"}
        mode_str = mode.get("mode", "online")
        enabled = mode_str.lower() == "online"

    # Check if CameraManagement service is connected
    if not camera_ws_manager.is_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Camera Management service is not connected"
        )

    # Send command to Camera Service
    success = await send_set_inference_mode(recipe_id, enabled)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send set inference mode command"
        )

    mode_name = "ONLINE" if enabled else "OFFLINE"
    logger.info(f"✅ [INFERENCE MODE] Set to {mode_name} for recipe: {recipe.name}")

    return {
        "success": True,
        "message": f"Inference mode set to {mode_name}",
        "recipe_id": recipe_id,
        "enabled": enabled,
        "mode": mode_name
    }


@router.post("/{recipe_id}/update-realtime")
async def update_recipe_realtime(
    recipe_id: str,
    update_data: dict,
    request: Request,
    current_user: UserInDB = Depends(get_current_user),
    recipe_repo: RecipeRepository = Depends(get_recipe_repository),
    action_log_repo: ActionLogRepository = Depends(get_action_log_repository)
):
    """
    Update recipe parameters in realtime WITHOUT saving to database.
    Uses Stop + Load pattern to apply changes to Camera Management service.

    This endpoint:
    1. Loads the recipe from DB (to get full config)
    2. Merges update_data into recipe config (in-memory only)
    3. Stops the current running recipe
    4. Loads the updated recipe config
    5. Does NOT save changes to database

    Supports updating:
    - delay_reject, reject_pulse (Basic Info tab)
    - Per-camera: exposure_time, delay_trigger, delay_interval, gain (Camera tab)
    - Per-camera templates: annotations only (Template tab)

    Args:
        recipe_id: Recipe ID to update
        update_data: Dictionary containing fields to update

    Returns:
        Status message with restarted flag
    """
    # 1. Check if CameraManagement service is connected
    if not camera_ws_manager.is_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Camera Management service is not connected"
        )

    # 2. Load recipe from DB to get full config
    recipe = await recipe_repo.get_by_id(recipe_id)
    if not recipe:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recipe not found"
        )

    # 3. Build base recipe dict from DB recipe
    camera_templates = getattr(recipe, 'camera_templates', [])
    cameras_config = getattr(recipe, 'cameras', [])

    # Convert to primitive and enrich with function_type
    cameras_primitive = _to_primitive(cameras_config)
    enriched_cameras = _enrich_cameras_with_function_type(
        cameras_primitive,
        camera_templates
    )

    # Resolve stale serial_number/model_name using current Camera table data
    from app.repositories.camera_repository import CameraRepository as _CameraRepo
    _camera_repo = _CameraRepo(get_database())
    enriched_cameras = await _resolve_camera_serials(enriched_cameras, _camera_repo)

    recipe_dict = {
        '_id': str(recipe.id) if hasattr(recipe, 'id') else recipe_id,
        'id': recipe_id,
        'name': recipe.name,
        'product_code': recipe.product_code,
        'delay_reject': recipe.delay_reject,
        'reject_pulse': recipe.reject_pulse if hasattr(recipe, 'reject_pulse') else 50.0,
        'reject_method': getattr(recipe, 'reject_method', 'DIO_OUT'),
        'do_reject_number': recipe.do_reject_number,
        'do_alarm_number': getattr(recipe, 'do_alarm_number', 0),
        'allow_late_reject': getattr(recipe, 'allow_late_reject', False),
        'normal_pulse_ms': getattr(recipe, 'normal_pulse_ms', 0.0),
        'cameras': enriched_cameras,
        'camera_templates': _to_primitive(camera_templates),
        'model_thresholds': _to_primitive(getattr(recipe, 'model_thresholds', None)),
        'camera_settings': _to_primitive(getattr(recipe, 'camera_settings', None)),
        'template_config': _to_primitive(getattr(recipe, 'template_config', None)),
        'roi_config': _to_primitive(getattr(recipe, 'roi_config', None)),
        'ocr_model_type': getattr(recipe, 'ocr_model_type', None),
        'ml_project_id': getattr(recipe, 'ml_project_id', None),
        'ml_model_id': getattr(recipe, 'ml_model_id', None),
        'defect_model': getattr(recipe, 'defect_model', None),
        'classifier_backend': getattr(recipe, 'classifier_backend', None),
        'cv_method': getattr(recipe, 'cv_method', None) or 'legacy',
        'template_bank_enabled': getattr(recipe, 'template_bank_enabled', False),
        'template_bank_size': getattr(recipe, 'template_bank_size', 10),
        'char_denoise_enabled': getattr(recipe, 'char_denoise_enabled', False),
        'product_detection_method': getattr(recipe, 'product_detection_method', None) or 'yolo_obb',
        'cap_rotation_method': getattr(recipe, 'cap_rotation_method', None) or 'yolo_obb',
        'cap_crop_method': getattr(recipe, 'cap_crop_method', None) or 'none',
        'crop_match_method': getattr(recipe, 'crop_match_method', None) or 'superpoint',
        'dual_rotation_check': bool(getattr(recipe, 'dual_rotation_check', False)),
        'product_box_wall_type': getattr(recipe, 'product_box_wall_type', None) or 'outer',
        'wrinkle_conf': getattr(recipe, 'wrinkle_conf', None),
        'wrinkle_show_when_pass': getattr(recipe, 'wrinkle_show_when_pass', None),
        'matching_conf': getattr(recipe, 'matching_conf', None),
        'mask_overlap_threshold': getattr(recipe, 'mask_overlap_threshold', None),
        'match_erosion_enabled': getattr(recipe, 'match_erosion_enabled', False),
        'match_erosion_kernel_w': getattr(recipe, 'match_erosion_kernel_w', 80),
        'match_erosion_kernel_h': getattr(recipe, 'match_erosion_kernel_h', 1),
        'match_erosion_iterations': getattr(recipe, 'match_erosion_iterations', 1),
    }

    # 4. Merge update_data into recipe_dict (in-memory, NOT saved to DB)
    if 'delay_reject' in update_data:
        recipe_dict['delay_reject'] = update_data['delay_reject']
        logger.info(f"📝 [REALTIME UPDATE] delay_reject → {update_data['delay_reject']} ms")

    if 'allow_late_reject' in update_data:
        recipe_dict['allow_late_reject'] = update_data['allow_late_reject']
        logger.info(f"📝 [REALTIME UPDATE] allow_late_reject → {update_data['allow_late_reject']}")

    if 'reject_pulse' in update_data:
        recipe_dict['reject_pulse'] = update_data['reject_pulse']
        logger.info(f"📝 [REALTIME UPDATE] reject_pulse → {update_data['reject_pulse']} ms")

    if 'normal_pulse_ms' in update_data:
        recipe_dict['normal_pulse_ms'] = update_data['normal_pulse_ms']
        logger.info(f"📝 [REALTIME UPDATE] normal_pulse_ms → {update_data['normal_pulse_ms']} ms")

    if 'wrinkle_conf' in update_data:
        recipe_dict['wrinkle_conf'] = update_data['wrinkle_conf']
        logger.info(f"📝 [REALTIME UPDATE] wrinkle_conf → {update_data['wrinkle_conf']}")

    if 'wrinkle_show_when_pass' in update_data:
        recipe_dict['wrinkle_show_when_pass'] = update_data['wrinkle_show_when_pass']
        logger.info(f"📝 [REALTIME UPDATE] wrinkle_show_when_pass → {update_data['wrinkle_show_when_pass']}")

    if 'matching_conf' in update_data:
        recipe_dict['matching_conf'] = update_data['matching_conf']
        logger.info(f"📝 [REALTIME UPDATE] matching_conf → {update_data['matching_conf']}")

    if 'match_erosion_enabled' in update_data:
        recipe_dict['match_erosion_enabled'] = update_data['match_erosion_enabled']
        logger.info(f"📝 [REALTIME UPDATE] match_erosion_enabled → {update_data['match_erosion_enabled']}")
    if 'match_erosion_kernel_w' in update_data:
        recipe_dict['match_erosion_kernel_w'] = update_data['match_erosion_kernel_w']
        logger.info(f"📝 [REALTIME UPDATE] match_erosion_kernel_w → {update_data['match_erosion_kernel_w']}")
    if 'match_erosion_kernel_h' in update_data:
        recipe_dict['match_erosion_kernel_h'] = update_data['match_erosion_kernel_h']
        logger.info(f"📝 [REALTIME UPDATE] match_erosion_kernel_h → {update_data['match_erosion_kernel_h']}")
    if 'match_erosion_iterations' in update_data:
        recipe_dict['match_erosion_iterations'] = update_data['match_erosion_iterations']
        logger.info(f"📝 [REALTIME UPDATE] match_erosion_iterations → {update_data['match_erosion_iterations']}")

    # Update camera settings
    if 'cameras' in update_data:
        for updated_cam in update_data['cameras']:
            cam_id = updated_cam.get('camera_id')
            for idx, cam in enumerate(recipe_dict['cameras']):
                if cam.get('camera_id') == cam_id:
                    # Merge updated fields
                    for field in ['exposure_time', 'delay_trigger', 'delay_interval', 'gain']:
                        if field in updated_cam:
                            recipe_dict['cameras'][idx][field] = updated_cam[field]
                            logger.info(f"📝 [REALTIME UPDATE] Camera {cam_id}: {field} → {updated_cam[field]}")
                    break

    # Update templates (annotations + optional image replacement)
    if 'camera_templates' in update_data:
        for updated_ct in update_data['camera_templates']:
            cam_id = updated_ct.get('camera_id')
            for idx, ct in enumerate(recipe_dict['camera_templates']):
                if ct.get('camera_id') == cam_id:
                    for tmpl_idx, tmpl in enumerate(updated_ct.get('templates', [])):
                        if tmpl_idx < len(ct.get('templates', [])):
                            existing_tmpl = recipe_dict['camera_templates'][idx]['templates'][tmpl_idx]
                            existing_tmpl['annotations'] = tmpl.get('annotations', [])
                            if 'center_offset_threshold_left' in tmpl:
                                existing_tmpl['center_offset_threshold_left'] = tmpl['center_offset_threshold_left']
                                logger.info(f"📝 [REALTIME UPDATE] Camera {cam_id} Template {tmpl_idx}: center_offset_threshold_left → {tmpl['center_offset_threshold_left']}")
                            if 'center_offset_threshold_right' in tmpl:
                                existing_tmpl['center_offset_threshold_right'] = tmpl['center_offset_threshold_right']
                                logger.info(f"📝 [REALTIME UPDATE] Camera {cam_id} Template {tmpl_idx}: center_offset_threshold_right → {tmpl['center_offset_threshold_right']}")
                            if 'center_offset_unit' in tmpl:
                                existing_tmpl['center_offset_unit'] = tmpl['center_offset_unit']
                                logger.info(f"📝 [REALTIME UPDATE] Camera {cam_id} Template {tmpl_idx}: center_offset_unit → {tmpl['center_offset_unit']}")
                            if 'wrinkle_area' in tmpl:
                                existing_tmpl['wrinkle_area'] = tmpl['wrinkle_area']
                                logger.info(f"📝 [REALTIME UPDATE] Camera {cam_id} Template {tmpl_idx}: wrinkle_area → {tmpl['wrinkle_area']}")
                            if 'wrinkle_min_area' in tmpl:
                                existing_tmpl['wrinkle_min_area'] = tmpl['wrinkle_min_area']
                                logger.info(f"📝 [REALTIME UPDATE] Camera {cam_id} Template {tmpl_idx}: wrinkle_min_area → {tmpl['wrinkle_min_area']}")
                            if 'wrinkle_max_area' in tmpl:
                                existing_tmpl['wrinkle_max_area'] = tmpl['wrinkle_max_area']
                                logger.info(f"📝 [REALTIME UPDATE] Camera {cam_id} Template {tmpl_idx}: wrinkle_max_area → {tmpl['wrinkle_max_area']}")
                            if 'color_config' in tmpl:
                                existing_tmpl['color_config'] = tmpl['color_config']
                                logger.info(f"📝 [REALTIME UPDATE] Camera {cam_id} Template {tmpl_idx}: color_config updated")
                            # Support online template image replacement
                            if 'image_url' in tmpl:
                                existing_tmpl['image_url'] = tmpl['image_url']
                                logger.info(f"📝 [REALTIME UPDATE] Camera {cam_id} Template {tmpl_idx}: image_url → {tmpl['image_url']}")
                            if 'image_width' in tmpl:
                                existing_tmpl['image_width'] = tmpl['image_width']
                            if 'image_height' in tmpl:
                                existing_tmpl['image_height'] = tmpl['image_height']
                            logger.info(f"📝 [REALTIME UPDATE] Camera {cam_id} Template {tmpl_idx}: Updated annotations")
                    # Propagate function_type updates (per-camera, not per-template)
                    if 'function_type' in updated_ct:
                        recipe_dict['camera_templates'][idx]['function_type'] = updated_ct['function_type']
                        # Mirror onto enriched cameras so AI service picks it up
                        for cam_i, cam in enumerate(recipe_dict['cameras']):
                            if cam.get('camera_id') == cam_id:
                                recipe_dict['cameras'][cam_i]['function_type'] = updated_ct['function_type']
                                break
                        logger.info(f"📝 [REALTIME UPDATE] Camera {cam_id}: function_type → {updated_ct['function_type']}")
                    break

    # 5. Stop current recipe
    logger.info(f"🛑 [REALTIME UPDATE] Stopping recipe: {recipe.name}")
    stop_success = await send_stop_recipe(recipe_id)
    time.sleep(2)

    if not stop_success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stop current recipe"
        )

    # 6. Load updated recipe (with merged changes)
    logger.info(f"🔄 [REALTIME UPDATE] Loading updated recipe: {recipe.name}")
    load_success = await send_load_recipe(recipe_dict)

    if not load_success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load updated recipe"
        )

    logger.info(f"✅ [REALTIME UPDATE] Recipe updated successfully (NOT saved to DB): {recipe.name}")

    # 7. Log the action (optional - track realtime updates)
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    action_log = ActionLogCreate(
        user_id=current_user.id,
        username=current_user.username,
        action_type=ActionType.UPDATE_RECIPE,
        resource_type="recipe",
        resource_id=recipe_id,
        description=f"Updated recipe '{recipe.name}' in realtime (not saved to DB)",
        new_value=update_data,
        ip_address=client_ip,
        user_agent=user_agent
    )
    await action_log_repo.create_action_log(action_log)

    return {
        "success": True,
        "message": "Recipe updated in realtime (not saved to database)",
        "recipe_id": recipe_id,
        "restarted": True,
        "note": "Changes are temporary. Stop/Load recipe again to restore DB config."
    }


@router.put("/{recipe_id}", response_model=RecipeResponse)
async def update_recipe(
    recipe_id: str,
    recipe_update: RecipeUpdate,
    request: Request,
    current_user: UserInDB = Depends(require_operator),
    recipe_repo: RecipeRepository = Depends(get_recipe_repository),
    action_log_repo: ActionLogRepository = Depends(get_action_log_repository)
):
    """
    Update an existing recipe (Save Receipt).

    **Permission**: Supervisor and Admin only

    According to requirements:
    - Supervisor: Can create new receipt and edit old receipt content
    - Admin: Has all permissions
    """
    
    # Check if recipe exists
    existing_recipe = await recipe_repo.get_by_id(recipe_id)
    if not existing_recipe:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recipe not found"
        )
    
    # If name is being updated, check for duplicates
    if recipe_update.name and recipe_update.name != existing_recipe.name:
        duplicate = await recipe_repo.get_by_name(recipe_update.name)
        if duplicate:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Recipe with name '{recipe_update.name}' already exists"
            )
    
    # If product_code is being updated, check for duplicates
    if recipe_update.product_code and recipe_update.product_code != existing_recipe.product_code:
        duplicate = await recipe_repo.get_by_product_code(recipe_update.product_code)
        if duplicate:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Recipe with product code '{recipe_update.product_code}' already exists"
            )
    
    updated_recipe = await recipe_repo.update(recipe_id, recipe_update, current_user.id)
    if not updated_recipe:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update recipe"
        )

    # Log the action
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    # Prepare old and new values for logging
    # Include commonly-used recipe fields and convert complex objects to primitives
    old_value = {
        "name": getattr(existing_recipe, 'name', None),
        "product_code": getattr(existing_recipe, 'product_code', None),
        "description": getattr(existing_recipe, 'description', None),
        "is_active": getattr(existing_recipe, 'is_active', None),
        "delay_reject": getattr(existing_recipe, 'delay_reject', None),
        "cameras": _to_primitive(getattr(existing_recipe, 'cameras', None)),
        "camera_templates": _to_primitive(getattr(existing_recipe, 'camera_templates', None)),
        "camera_settings": _to_primitive(getattr(existing_recipe, 'camera_settings', None)),
        "model_thresholds": _to_primitive(getattr(existing_recipe, 'model_thresholds', None)),
        "template_config": _to_primitive(getattr(existing_recipe, 'template_config', None)),
        "roi_config": _to_primitive(getattr(existing_recipe, 'roi_config', None))
    }

    new_value = {}
    update_dict = recipe_update.model_dump(exclude_unset=True)
    for field in update_dict:
        if hasattr(updated_recipe, field):
            new_value[field] = getattr(updated_recipe, field)

    action_log = ActionLogCreate(
        user_id=current_user.id,
        username=current_user.username,
        action_type=ActionType.UPDATE_RECIPE,
        resource_type="recipe",
        resource_id=recipe_id,
        description=f"Updated recipe '{updated_recipe.name}'",
        old_value=old_value,
        new_value=new_value,
        ip_address=client_ip,
        user_agent=user_agent
    )
    await action_log_repo.create_action_log(action_log)

    return recipe_to_response(updated_recipe)


@router.delete("/{recipe_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_recipe(
    recipe_id: str,
    request: Request,
    current_user: UserInDB = Depends(require_admin),
    recipe_repo: RecipeRepository = Depends(get_recipe_repository),
    action_log_repo: ActionLogRepository = Depends(get_action_log_repository)
):
    """
    Delete a recipe permanently.
    
    **Permission**: Admin only
    
    According to requirements:
    - Admin: Can change/reset passwords for Operator and Supervisor, has all permissions
    """
    recipe = await recipe_repo.get_by_id(recipe_id)
    if not recipe:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recipe not found"
        )
    
    success = await recipe_repo.delete(recipe_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete recipe"
        )

    # Log the action
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    action_log = ActionLogCreate(
        user_id=current_user.id,
        username=current_user.username,
        action_type=ActionType.DELETE_RECIPE,
        resource_type="recipe",
        resource_id=recipe_id,
        description=f"Deleted recipe '{recipe.name}' with product code '{recipe.product_code}'",
        old_value={
            "name": recipe.name,
            "product_code": recipe.product_code,
            "description": recipe.description,
            "is_active": recipe.is_active
        },
        ip_address=client_ip,
        user_agent=user_agent
    )
    await action_log_repo.create_action_log(action_log)

    return None


@router.get("/stats/count")
async def get_recipe_count(
    is_active: Optional[bool] = None,
    current_user: UserInDB = Depends(get_current_user),
    recipe_repo: RecipeRepository = Depends(get_recipe_repository)
):
    """
    Get total count of recipes.
    
    **Permission**: All authenticated users
    """
    count = await recipe_repo.count(is_active=is_active)
    return {"count": count}

