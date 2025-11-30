from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from app.db.mongodb import get_database
from app.repositories.recipe_repository import RecipeRepository
from app.models.recipe import (
    RecipeCreate,
    RecipeUpdate,
    RecipeResponse,
    DatecodeValidation,
    DatecodeValidationResponse
)
from app.models.user import UserInDB
from app.api.dependencies.auth import get_current_user, require_operator, require_supervisor

router = APIRouter()


@router.post("/", response_model=RecipeResponse, status_code=status.HTTP_201_CREATED)
async def create_recipe(
    recipe_create: RecipeCreate,
    db=Depends(get_database),
    current_user: UserInDB = Depends(require_supervisor)
):
    """
    Create new recipe - SUPERVISOR and ADMIN only

    - **name**: Unique recipe name
    - **description**: Recipe description (optional)
    - **camera_settings**: Camera configuration (exposure_time, delay_trigger, etc.)
    - **model_thresholds**: List of AI model thresholds
    - **datecode_pattern**: Expected datecode pattern/regex (optional)
    """
    recipe_repo = RecipeRepository(db)

    # Check if recipe name already exists
    existing_recipe = await recipe_repo.get_recipe_by_name(recipe_create.name)
    if existing_recipe:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Recipe name already exists"
        )

    # Create recipe
    recipe = await recipe_repo.create_recipe(recipe_create, current_user.id)

    return RecipeResponse(**recipe.model_dump())


@router.get("/", response_model=List[RecipeResponse])
async def get_all_recipes(
    skip: int = 0,
    limit: int = 100,
    active_only: bool = True,
    db=Depends(get_database),
    current_user: UserInDB = Depends(require_operator)
):
    """
    Get all recipes - All authenticated users

    - **skip**: Number of records to skip (pagination)
    - **limit**: Maximum number of records to return
    - **active_only**: Filter only active recipes
    """
    recipe_repo = RecipeRepository(db)
    recipes = await recipe_repo.get_all_recipes(skip=skip, limit=limit, active_only=active_only)

    return [RecipeResponse(**recipe.model_dump()) for recipe in recipes]


@router.get("/{recipe_id}", response_model=RecipeResponse)
async def get_recipe(
    recipe_id: str,
    db=Depends(get_database),
    current_user: UserInDB = Depends(require_operator)
):
    """Get recipe by ID - All authenticated users"""
    recipe_repo = RecipeRepository(db)
    recipe = await recipe_repo.get_recipe_by_id(recipe_id)

    if not recipe:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recipe not found"
        )

    return RecipeResponse(**recipe.model_dump())


@router.get("/name/{recipe_name}", response_model=RecipeResponse)
async def get_recipe_by_name(
    recipe_name: str,
    db=Depends(get_database),
    current_user: UserInDB = Depends(require_operator)
):
    """Get recipe by name - All authenticated users"""
    recipe_repo = RecipeRepository(db)
    recipe = await recipe_repo.get_recipe_by_name(recipe_name)

    if not recipe:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recipe not found"
        )

    return RecipeResponse(**recipe.model_dump())


@router.put("/{recipe_id}", response_model=RecipeResponse)
async def update_recipe(
    recipe_id: str,
    recipe_update: RecipeUpdate,
    db=Depends(get_database),
    current_user: UserInDB = Depends(require_supervisor)
):
    """
    Update recipe - SUPERVISOR and ADMIN only

    - **name**: New recipe name (optional)
    - **description**: New description (optional)
    - **camera_settings**: Updated camera settings (optional)
    - **model_thresholds**: Updated model thresholds (optional)
    - **datecode_pattern**: Updated datecode pattern (optional)
    - **is_active**: Active status (optional)
    """
    recipe_repo = RecipeRepository(db)

    # Check if recipe exists
    existing_recipe = await recipe_repo.get_recipe_by_id(recipe_id)
    if not existing_recipe:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recipe not found"
        )

    # If updating name, check for duplicates
    if recipe_update.name and recipe_update.name != existing_recipe.name:
        duplicate = await recipe_repo.get_recipe_by_name(recipe_update.name)
        if duplicate:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Recipe name already exists"
            )

    # Update recipe
    recipe = await recipe_repo.update_recipe(recipe_id, recipe_update, current_user.id)

    if not recipe:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update recipe"
        )

    return RecipeResponse(**recipe.model_dump())


@router.delete("/{recipe_id}")
async def delete_recipe(
    recipe_id: str,
    hard_delete: bool = False,
    db=Depends(get_database),
    current_user: UserInDB = Depends(require_supervisor)
):
    """
    Delete recipe - SUPERVISOR and ADMIN only

    - **recipe_id**: ID of recipe to delete
    - **hard_delete**: If true, permanently delete. Otherwise soft delete (set is_active=False)
    """
    recipe_repo = RecipeRepository(db)

    # Check if recipe exists
    recipe = await recipe_repo.get_recipe_by_id(recipe_id)
    if not recipe:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recipe not found"
        )

    if hard_delete:
        success = await recipe_repo.hard_delete_recipe(recipe_id)
    else:
        success = await recipe_repo.delete_recipe(recipe_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete recipe"
        )

    delete_type = "permanently deleted" if hard_delete else "deactivated"
    return {"message": f"Recipe {delete_type} successfully"}


@router.post("/validate-datecode", response_model=DatecodeValidationResponse)
async def validate_datecode(
    validation: DatecodeValidation,
    db=Depends(get_database),
    current_user: UserInDB = Depends(require_operator)
):
    """
    Validate datecode against recipe - All authenticated users (OPERATOR can use)

    - **recipe_id**: ID of recipe to validate against
    - **datecode_input**: Datecode string to validate
    """
    recipe_repo = RecipeRepository(db)

    # Get recipe
    recipe = await recipe_repo.get_recipe_by_id(validation.recipe_id)
    if not recipe:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recipe not found"
        )

    # Simple validation logic (can be enhanced with regex pattern matching)
    is_valid = True
    validation_message = "Datecode validation successful"

    if recipe.datecode_pattern:
        import re
        try:
            pattern = re.compile(recipe.datecode_pattern)
            if not pattern.match(validation.datecode_input):
                is_valid = False
                validation_message = f"Datecode does not match pattern: {recipe.datecode_pattern}"
        except re.error:
            validation_message = "Invalid regex pattern in recipe"
            is_valid = False

    return DatecodeValidationResponse(
        is_valid=is_valid,
        recipe_name=recipe.name,
        expected_pattern=recipe.datecode_pattern,
        input_datecode=validation.datecode_input,
        validation_message=validation_message
    )
