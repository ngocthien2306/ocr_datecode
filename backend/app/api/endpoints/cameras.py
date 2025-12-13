"""
Camera API Endpoints
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status

from app.db.mongodb import get_database
from app.repositories.camera_repository import CameraRepository
from app.schemas.camera import (
    CameraCreate,
    CameraUpdate,
    CameraResponse,
    CameraConnectionStatus
)
from app.api.dependencies.auth import get_current_user

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.post(
    "/",
    response_model=CameraResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new camera"
)
async def create_camera(
    camera: CameraCreate,
    db=Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    """
    Create a new camera entry in the database.
    
    - **camera_id**: Unique identifier for the camera
    - **model_name**: Camera model name
    - **serial_number**: Camera serial number
    - **resolution_width**: Width in pixels
    - **resolution_height**: Height in pixels
    """
    repo = CameraRepository(db)
    
    # Check if camera_id already exists
    existing = await repo.get_by_id(camera.camera_id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Camera with ID '{camera.camera_id}' already exists"
        )
    
    # Check if serial number already exists
    existing_serial = await repo.get_by_serial(camera.serial_number)
    if existing_serial:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Camera with serial number '{camera.serial_number}' already exists"
        )
    
    created_by = current_user.get("username", "system")
    result = await repo.create(camera, created_by=created_by)
    
    return result


@router.get(
    "/",
    response_model=List[CameraResponse],
    summary="Get all cameras"
)
async def get_cameras(
    skip: int = 0,
    limit: int = 100,
    is_active: Optional[bool] = None,
    db=Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    """
    Get all cameras with pagination and filtering.
    
    - **skip**: Number of records to skip
    - **limit**: Maximum number of records to return
    - **is_active**: Filter by active status
    """
    repo = CameraRepository(db)
    cameras = await repo.get_all(skip=skip, limit=limit, is_active=is_active)
    return cameras


@router.get(
    "/connected",
    response_model=List[CameraResponse],
    summary="Get connected cameras"
)
async def get_connected_cameras(
    db=Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    """
    Get all currently connected cameras.
    """
    repo = CameraRepository(db)
    cameras = await repo.get_connected_cameras()
    return cameras


@router.get(
    "/{camera_id}",
    response_model=CameraResponse,
    summary="Get camera by ID"
)
async def get_camera(
    camera_id: str,
    db=Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    """
    Get a specific camera by its camera_id.
    """
    repo = CameraRepository(db)
    camera = await repo.get_by_id(camera_id)
    
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with ID '{camera_id}' not found"
        )
    
    return camera


@router.put(
    "/{camera_id}",
    response_model=CameraResponse,
    summary="Update camera"
)
async def update_camera(
    camera_id: str,
    camera_update: CameraUpdate,
    db=Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    """
    Update camera information.
    """
    repo = CameraRepository(db)
    
    # Check if camera exists
    existing = await repo.get_by_id(camera_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with ID '{camera_id}' not found"
        )
    
    # If updating serial number, check for duplicates
    if camera_update.serial_number:
        existing_serial = await repo.get_by_serial(camera_update.serial_number)
        if existing_serial and existing_serial["camera_id"] != camera_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Camera with serial number '{camera_update.serial_number}' already exists"
            )
    
    updated = await repo.update(camera_id, camera_update)
    
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update camera"
        )
    
    return updated


@router.patch(
    "/{camera_id}/connection",
    response_model=CameraResponse,
    summary="Update camera connection status"
)
async def update_connection_status(
    camera_id: str,
    is_connected: bool,
    db=Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    """
    Update the connection status of a camera.
    """
    repo = CameraRepository(db)
    
    updated = await repo.update_connection_status(camera_id, is_connected)
    
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with ID '{camera_id}' not found"
        )
    
    return updated


@router.delete(
    "/{camera_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete camera"
)
async def delete_camera(
    camera_id: str,
    db=Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    """
    Delete a camera from the database.
    """
    repo = CameraRepository(db)
    
    deleted = await repo.delete(camera_id)
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with ID '{camera_id}' not found"
        )
    
    return None


@router.get(
    "/count/total",
    summary="Get camera count"
)
async def get_camera_count(
    is_active: Optional[bool] = None,
    db=Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    """
    Get total count of cameras.
    """
    repo = CameraRepository(db)
    count = await repo.count(is_active=is_active)
    
    return {"count": count, "is_active": is_active}
