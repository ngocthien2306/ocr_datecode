"""
Camera API Endpoints
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
import io

from app.db.mongodb import get_database
from app.repositories.camera_repository import CameraRepository
from app.schemas.camera import (
    CameraCreate,
    CameraUpdate,
    CameraResponse,
    CameraConnectionStatus,
    CameraSettingsUpdate
)
from app.api.dependencies.auth import get_current_user, get_current_user_from_query
from app.services import camera_frame_service, camera_producer_service
from app.services.camera_settings_service import camera_settings_service

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


@router.get(
    "/{serial_number}/frame",
    summary="Get latest camera frame",
    responses={
        200: {
            "content": {"image/jpeg": {}},
            "description": "Returns the latest camera frame as JPEG image"
        },
        404: {
            "description": "Camera not found or not streaming"
        }
    }
)
async def get_camera_frame(
    serial_number: str,
    quality: int = 85,
    db=Depends(get_database),
    current_user: dict = Depends(get_current_user_from_query)
):
    """
    Lấy frame mới nhất từ camera.

    - **serial_number**: Serial number của camera
    - **quality**: JPEG quality (0-100), mặc định 85

    Returns: JPEG image
    """
    # Kiểm tra camera có tồn tại trong database không
    repo = CameraRepository(db)
    camera = await repo.get_by_serial(serial_number)

    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with serial number '{serial_number}' not found"
        )

    # Đọc frame từ shared memory
    frame_data = camera_frame_service.get_frame(serial_number)

    if frame_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera '{serial_number}' is not streaming. Please ensure camera producer is running."
        )

    # Encode frame thành JPEG
    jpeg_bytes = camera_frame_service.encode_frame_jpeg(frame_data['frame'], quality)

    if jpeg_bytes is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to encode frame"
        )

    # Return image as streaming response
    return StreamingResponse(
        io.BytesIO(jpeg_bytes),
        media_type="image/jpeg",
        headers={
            "X-Frame-Index": str(frame_data['metadata']['frame_idx']),
            "X-Timestamp": str(frame_data['metadata']['timestamp']),
            "X-Camera-Model": frame_data['metadata']['model_name']
        }
    )


@router.get(
    "/{serial_number}/frame/metadata",
    summary="Get latest camera frame metadata",
    response_model=dict
)
async def get_camera_frame_metadata(
    serial_number: str,
    db=Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    """
    Lấy metadata của frame mới nhất từ camera (không bao gồm ảnh).

    - **serial_number**: Serial number của camera

    Returns: Frame metadata (timestamp, frame_idx, shape, etc.)
    """
    # Kiểm tra camera có tồn tại trong database không
    repo = CameraRepository(db)
    camera = await repo.get_by_serial(serial_number)

    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with serial number '{serial_number}' not found"
        )

    # Đọc frame từ shared memory
    frame_data = camera_frame_service.get_frame(serial_number)

    if frame_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera '{serial_number}' is not streaming. Please ensure camera producer is running."
        )

    return frame_data['metadata']


@router.post(
    "/{serial_number}/connect",
    summary="Connect and start camera producer",
    response_model=dict
)
async def connect_camera(
    serial_number: str,
    device_index: int = 0,
    db=Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    """
    Connect camera and start producer process.

    - **serial_number**: Serial number của camera
    - **device_index**: Camera device index (default 0)

    Returns: Status message
    """
    # Kiểm tra camera có tồn tại trong database không
    repo = CameraRepository(db)
    camera = await repo.get_by_serial(serial_number)

    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with serial number '{serial_number}' not found"
        )

    # Start camera producer
    success = camera_producer_service.start_camera(serial_number, device_index)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start camera producer for '{serial_number}'"
        )

    # Update camera connection status in database
    await repo.update_connection_status(camera["camera_id"], True)

    return {
        "success": True,
        "message": f"Camera '{serial_number}' connected successfully",
        "serial_number": serial_number,
        "status": camera_producer_service.get_camera_status(serial_number)
    }


@router.post(
    "/{serial_number}/disconnect",
    summary="Disconnect and stop camera producer",
    response_model=dict
)
async def disconnect_camera(
    serial_number: str,
    db=Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    """
    Disconnect camera and stop producer process.

    - **serial_number**: Serial number của camera

    Returns: Status message
    """
    # Kiểm tra camera có tồn tại trong database không
    repo = CameraRepository(db)
    camera = await repo.get_by_serial(serial_number)

    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with serial number '{serial_number}' not found"
        )

    # Stop camera producer
    success = camera_producer_service.stop_camera(serial_number)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop camera producer for '{serial_number}'"
        )

    # Update camera connection status in database
    await repo.update_connection_status(camera["camera_id"], False)

    return {
        "success": True,
        "message": f"Camera '{serial_number}' disconnected successfully",
        "serial_number": serial_number
    }


@router.get(
    "/{serial_number}/status",
    summary="Get camera producer status",
    response_model=dict
)
async def get_camera_producer_status(
    serial_number: str,
    db=Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    """
    Get camera producer process status.

    - **serial_number**: Serial number của camera

    Returns: Producer status
    """
    # Kiểm tra camera có tồn tại trong database không
    repo = CameraRepository(db)
    camera = await repo.get_by_serial(serial_number)

    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with serial number '{serial_number}' not found"
        )

    status_info = camera_producer_service.get_camera_status(serial_number)

    return {
        "serial_number": serial_number,
        "camera_id": camera["camera_id"],
        "producer_status": status_info
    }


@router.post(
    "/{serial_number}/settings",
    summary="Update camera runtime settings",
    response_model=dict
)
async def update_camera_settings(
    serial_number: str,
    settings: CameraSettingsUpdate,
    db=Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    """
    Update camera runtime settings (exposure, gain).
    Settings are applied immediately if camera is running.

    - **serial_number**: Serial number của camera
    - **exposure_time**: Exposure time in microseconds (100-1000000)
    - **gain**: Camera gain value (0.0-20.0)

    Returns: Updated settings
    """
    # Kiểm tra camera có tồn tại trong database không
    repo = CameraRepository(db)
    camera = await repo.get_by_serial(serial_number)

    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with serial number '{serial_number}' not found"
        )

    try:
        updated_settings = camera_settings_service.update_settings(
            serial_number=serial_number,
            exposure_time=settings.exposure_time,
            gain=settings.gain
        )

        return {
            "success": True,
            "message": "Settings updated successfully",
            "serial_number": serial_number,
            "settings": updated_settings
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update settings: {str(e)}"
        )


@router.get(
    "/{serial_number}/settings",
    summary="Get camera runtime settings",
    response_model=dict
)
async def get_camera_settings(
    serial_number: str,
    db=Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    """
    Get current camera runtime settings.

    - **serial_number**: Serial number của camera

    Returns: Current settings
    """
    # Kiểm tra camera có tồn tại trong database không
    repo = CameraRepository(db)
    camera = await repo.get_by_serial(serial_number)

    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with serial number '{serial_number}' not found"
        )

    settings = camera_settings_service.get_settings(serial_number)

    if settings is None:
        # Return default settings if not found
        settings = {
            "exposure_time": 10000,
            "gain": 1.0
        }

    return {
        "serial_number": serial_number,
        "settings": settings
    }
