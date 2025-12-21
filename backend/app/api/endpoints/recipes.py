import traceback
from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File
from fastapi.responses import FileResponse, Response
from typing import List, Optional
import os
import uuid
from pathlib import Path
import base64
from PIL import Image, ImageDraw, ImageFont
import json
import hashlib
import io

from app.schemas.recipe import RecipeCreate, RecipeUpdate, RecipeResponse
from app.models.recipe import RecipeInDB
from app.models.user import UserInDB, UserRole
from app.repositories.recipe_repository import RecipeRepository
from app.db.mongodb import get_database
from app.api.dependencies.auth import (
    get_current_user, 
    require_supervisor, 
    require_admin,
    RoleChecker
)
from app.schemas.receipt_load import ReceiptLoadCreate, ReceiptLoadResponse
from app.repositories.receipt_load_repository import ReceiptLoadRepository
from datetime import datetime

router = APIRouter()


# Template image upload directory
TEMPLATE_UPLOAD_DIR = Path("/Users/ngocthien.ai/Source/Projects/ocr_datecode/backend/uploads/templates")
TEMPLATE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE_VISUALIZE_DIR = Path("/Users/ngocthien.ai/Source/Projects/ocr_datecode/backend/uploads/visualizations")
TEMPLATE_VISUALIZE_DIR.mkdir(parents=True, exist_ok=True)


def _coord(value, max_val: int) -> int:
    """Normalize a coordinate which may be absolute pixel or relative (0..1)."""
    try:
        v = float(value)
    except Exception:
        return 0
    if 0.0 <= v <= 1.0:
        return int(v * max_val)
    return int(round(v))


def _annotations_hash(filename: str, annotations: list) -> str:
    payload = json.dumps({"file": filename, "annotations": annotations}, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()


def _draw_template_visualization(image_path: Path, annotations: list) -> str:
    """Draw rectangles, polygons and text onto image and save under TEMPLATE_VISUALIZE_DIR.

    Returns relative URL path to the generated visualization file.
    """
    if not image_path.exists():
        raise FileNotFoundError(str(image_path))

    # compute deterministic filename from annotations + source filename
    digest = _annotations_hash(image_path.name, annotations or [])
    out_name = f"viz_{digest}.png"
    out_path = TEMPLATE_VISUALIZE_DIR / out_name
    if out_path.exists():
        return f"/api/recipes/templates/visualizations/{out_name}"

    img = Image.open(str(image_path)).convert("RGBA")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # font size proportional to image width
    try:
        font_size = max(12, int(width / 100))
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
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
        label = ann.get('text')
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


def recipe_to_response(recipe: RecipeInDB) -> RecipeResponse:
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
    return RecipeResponse(
        id=recipe.id,
        name=recipe.name,
        product_code=recipe.product_code,
        description=recipe.description,
        delay_reject=recipe.delay_reject if hasattr(recipe, 'delay_reject') else 100.0,
        cameras=cameras_data,
        camera_templates=camera_templates_data,
        camera_settings=recipe.camera_settings.model_dump() if hasattr(recipe.camera_settings, 'model_dump') and recipe.camera_settings else recipe.camera_settings,
        model_thresholds=recipe.model_thresholds.model_dump() if hasattr(recipe.model_thresholds, 'model_dump') else recipe.model_thresholds,
        template_config=recipe.template_config,
        roi_config=recipe.roi_config,
        is_active=recipe.is_active,
        created_by=recipe.created_by,
        updated_by=recipe.updated_by,
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


@router.post("/", response_model=RecipeResponse, status_code=status.HTTP_201_CREATED)
async def create_recipe(
    recipe: RecipeCreate,
    current_user: UserInDB = Depends(require_supervisor),
    recipe_repo: RecipeRepository = Depends(get_recipe_repository)
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
    return recipe_to_response(created_recipe)


@router.get("/", response_model=List[RecipeResponse])
async def list_recipes(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    is_active: Optional[bool] = None,
    current_user: UserInDB = Depends(get_current_user),
    recipe_repo: RecipeRepository = Depends(get_recipe_repository)
):
    """
    List all recipes with pagination.
    
    **Permission**: All authenticated users (Operator, Supervisor, Admin)
    
    According to requirements:
    - Operator: Can only load receipt and input datecode field to verify with read content
    """
    recipes = await recipe_repo.get_all(skip=skip, limit=limit, is_active=is_active)
    return [recipe_to_response(recipe) for recipe in recipes]


@router.get("/search", response_model=List[RecipeResponse])
async def search_recipes(
    q: str = Query(..., min_length=1, description="Search query"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    current_user: UserInDB = Depends(get_current_user),
    recipe_repo: RecipeRepository = Depends(get_recipe_repository)
):
    """
    Search recipes by name, product code, or description.
    
    **Permission**: All authenticated users
    """
    recipes = await recipe_repo.search(q, skip=skip, limit=limit)
    return [recipe_to_response(recipe) for recipe in recipes]



# Move static routes before path-parameter routes so they are matched first.
@router.get("/loads")
async def list_all_loads(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    current_user: UserInDB = Depends(require_supervisor),
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
                        # no annotations: point to raw template image
                        tpl['visualization_url'] = f"/api/recipes/templates/images/{filename}"
                        continue
                    try:
                        viz_url = _draw_template_visualization(src_path, annotations)
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


@router.post("/{recipe_id}/load", response_model=ReceiptLoadResponse)
async def load_recipe(
    recipe_id: str,
    current_user: UserInDB = Depends(get_current_user),
    recipe_repo: RecipeRepository = Depends(get_recipe_repository),
    load_repo: ReceiptLoadRepository = Depends(get_receipt_load_repository)
):
    """
    Record a recipe load event. Stores who loaded the recipe and when.
    """
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

    # Build metadata snapshot (store useful fields only) and convert to primitives
    metadata = {
        'name': recipe.name,
        'product_code': recipe.product_code,
        'camera_templates': _to_primitive(camera_templates),
        'model_thresholds': _to_primitive(getattr(recipe, 'model_thresholds', None)),
        'cameras': _to_primitive(getattr(recipe, 'cameras', []))
    }

    # Persist loader's full name to avoid frontend-side lookups
    created = await load_repo.create(
        recipe_id=recipe_id,
        user_id=current_user.id,
        metadata=metadata,
        user_full_name=getattr(current_user, 'full_name', None),
    )

    return ReceiptLoadResponse(
        id=created['id'],
        recipe_id=created['recipe_id'],
        loaded_by=created['loaded_by'],
        loaded_by_full_name=created.get('loaded_by_full_name'),
        loaded_at=created['loaded_at'],
        metadata=created.get('metadata')
    )



@router.put("/{recipe_id}", response_model=RecipeResponse)
async def update_recipe(
    recipe_id: str,
    recipe_update: RecipeUpdate,
    current_user: UserInDB = Depends(require_supervisor),
    recipe_repo: RecipeRepository = Depends(get_recipe_repository)
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
    
    return recipe_to_response(updated_recipe)


@router.delete("/{recipe_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_recipe(
    recipe_id: str,
    current_user: UserInDB = Depends(require_admin),
    recipe_repo: RecipeRepository = Depends(get_recipe_repository)
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

