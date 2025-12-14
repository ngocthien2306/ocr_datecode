from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File
from fastapi.responses import FileResponse, Response
from typing import List, Optional
import os
import uuid
from pathlib import Path
import base64
from PIL import Image
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

router = APIRouter()


async def get_recipe_repository(db=Depends(get_database)) -> RecipeRepository:
    """Dependency to get recipe repository"""
    return RecipeRepository(db)


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


# Template image upload directory
TEMPLATE_UPLOAD_DIR = Path("/Users/ngocthien.ai/Source/Projects/ocr_datecode/backend/uploads/templates")
TEMPLATE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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
