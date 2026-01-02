
"""
Camera API Endpoints
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import StreamingResponse, JSONResponse
import io

from app.db.mongodb import get_database
from app.repositories.camera_repository import CameraRepository
from app.repositories.action_log_repository import ActionLogRepository
from app.schemas.camera import (
    CameraCreate,
    CameraUpdate,
    CameraResponse,
    CameraConnectionStatus,
    CameraSettingsUpdate
)
from app.api.dependencies.auth import get_current_user, get_current_user_from_query
from app.api.dependencies.action_log import get_action_log_repository
from app.services import camera_frame_service, camera_producer_service, shared_memory_service
from app.services.camera_settings_service import camera_settings_service
from app.models.action_log import ActionLogCreate, ActionType
from app.models.user import UserInDB
from app.api.websocket.camera_ws import (
    send_connect_camera,
    send_disconnect_camera,
    send_set_camera_mode,
    send_update_camera_settings,
    camera_ws_manager
)

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.post(
    "/",
    response_model=CameraResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new camera"
)
async def create_camera(
    camera: CameraCreate,
    request: Request,
    db=Depends(get_database),
    current_user: UserInDB = Depends(get_current_user),
    action_log_repo: ActionLogRepository = Depends(get_action_log_repository)
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
    
    created_by = current_user.username
    result = await repo.create(camera, created_by=created_by)

    # Log the action
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    action_log = ActionLogCreate(
        user_id=current_user.id,
        username=current_user.username,
        action_type=ActionType.CREATE_CAMERA,
        resource_type="camera",
        resource_id=result.camera_id,
        description=f"Created camera '{result.camera_id}' with model '{result.model_name}'",
        new_value={
            "camera_id": result.camera_id,
            "model_name": result.model_name,
            "serial_number": result.serial_number,
            "resolution_width": result.resolution_width,
            "resolution_height": result.resolution_height,
            "is_active": result.is_active
        },
        ip_address=client_ip,
        user_agent=user_agent
    )
    await action_log_repo.create_action_log(action_log)
    
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
    request: Request,
    db=Depends(get_database),
    current_user: UserInDB = Depends(get_current_user),
    action_log_repo: ActionLogRepository = Depends(get_action_log_repository)
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

    # Log the action
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    # Prepare old and new values for logging
    print(existing)
    old_value = {
        "camera_id": existing['camera_id'],
        "model_name": existing['model_name'],
        "serial_number": existing['serial_number'],
        "resolution_width": existing['resolution_width'],
        "resolution_height": existing['resolution_height'],
        "is_active": existing['is_active']
    }

    new_value = {}
    update_dict = camera_update.model_dump(exclude_unset=True)
    for field in update_dict:
        if hasattr(updated, field):
            new_value[field] = getattr(updated, field)

    action_log = ActionLogCreate(
        user_id=current_user.id,
        username=current_user.username,
        action_type=ActionType.UPDATE_CAMERA,
        resource_type="camera",
        resource_id=camera_id,
        description=f"Updated camera '{camera_id}'",
        old_value=old_value,
        new_value=new_value,
        ip_address=client_ip,
        user_agent=user_agent
    )
    await action_log_repo.create_action_log(action_log)

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
    request: Request,
    db=Depends(get_database),
    current_user: UserInDB = Depends(get_current_user),
    action_log_repo: ActionLogRepository = Depends(get_action_log_repository)
):
    """
    Delete a camera from the database.
    """
    repo = CameraRepository(db)

    # Get camera info before deletion for logging
    camera = await repo.get_by_id(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with ID '{camera_id}' not found"
        )
    
    deleted = await repo.delete(camera_id)
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with ID '{camera_id}' not found"
        )

    # Log the action
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    action_log = ActionLogCreate(
        user_id=current_user.id,
        username=current_user.username,
        action_type=ActionType.DELETE_CAMERA,
        resource_type="camera",
        resource_id=camera_id,
        description=f"Deleted camera '{camera_id}' with model '{camera.get('model_name', '')}'",
        old_value={
            "camera_id": camera.get("camera_id"),
            "model_name": camera.get("model_name"),
            "serial_number": camera.get("serial_number"),
            "resolution_width": camera.get("resolution_width"),
            "resolution_height": camera.get("resolution_height"),
            "is_active": camera.get("is_active")
        },
        ip_address=client_ip,
        user_agent=user_agent
    )
    await action_log_repo.create_action_log(action_log)

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

    # Đọc frame từ shared memory (new architecture)
    jpeg_bytes = shared_memory_service.read_frame_as_jpeg(serial_number, quality)

    if jpeg_bytes is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera '{serial_number}' is not streaming. Please ensure camera is connected."
        )

    # Get metadata for headers
    result = shared_memory_service.read_frame(serial_number)
    metadata = result[1] if result else {}

    # Return image as streaming response
    return StreamingResponse(
        io.BytesIO(jpeg_bytes),
        media_type="image/jpeg",
        headers={
            "X-Frame-Index": str(metadata.get('frame_idx', 0)),
            "X-Timestamp": str(metadata.get('timestamp', 0)),
            "X-Camera-Model": metadata.get('model_name', 'Unknown')
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

    # Đọc frame từ shared memory (new architecture)
    result = shared_memory_service.read_frame(serial_number)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera '{serial_number}' is not streaming. Please ensure camera is connected."
        )

    _, metadata = result
    return metadata


@router.post(
    "/{serial_number}/connect",
    summary="Connect camera via WebSocket",
    response_model=dict
)
async def connect_camera(
    serial_number: str,
    pixel_format: str = "Mono8",
    db=Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    """
    Connect camera via WebSocket to CameraManagement service.

    - **serial_number**: Serial number of the camera
    - **pixel_format**: Pixel format (Mono8, RGB8, etc.) - default Mono8

    Returns: Status message
    """
    # Check if camera exists in database
    repo = CameraRepository(db)
    camera = await repo.get_by_serial(serial_number)

    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with serial number '{serial_number}' not found"
        )

    # Check if CameraManagement service is connected
    if not camera_ws_manager.is_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Camera Management service is not connected. Please start the service."
        )

    # Send connect command via WebSocket
    success = await send_connect_camera(serial_number, pixel_format)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send connect command for '{serial_number}'"
        )

    # Update camera connection status in database
    await repo.update_connection_status(camera["camera_id"], True)

    return {
        "success": True,
        "message": f"Camera '{serial_number}' connect command sent successfully",
        "serial_number": serial_number
    }


@router.post(
    "/{serial_number}/disconnect",
    summary="Disconnect camera via WebSocket",
    response_model=dict
)
async def disconnect_camera(
    serial_number: str,
    db=Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    """
    Disconnect camera via WebSocket from CameraManagement service.

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

    # Check if CameraManagement service is connected
    if not camera_ws_manager.is_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Camera Management service is not connected"
        )

    # Send disconnect command via WebSocket
    success = await send_disconnect_camera(serial_number)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send disconnect command for '{serial_number}'"
        )

    # Update camera connection status in database
    await repo.update_connection_status(camera["camera_id"], False)

    # Cleanup shared memory connection
    shared_memory_service.cleanup(serial_number)

    return {
        "success": True,
        "message": f"Camera '{serial_number}' disconnect command sent successfully",
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
        # Update settings in file
        updated_settings = camera_settings_service.update_settings(
            serial_number=serial_number,
            exposure_time=settings.exposure_time,
            gain=settings.gain,
            trigger_config=settings.trigger_config
        )

        # Send settings to camera via WebSocket (if camera is connected)
        if camera_ws_manager.is_connected():
            success = await send_update_camera_settings(serial_number, updated_settings)
            if success:
                print(f"✅ Sent settings update to camera {serial_number} via WebSocket")
            else:
                print(f"⚠️ Failed to send settings to camera {serial_number}")
        else:
            print(f"⚠️ Camera service not connected, settings saved to file only")

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


@router.post(
    "/{serial_number}/test-trigger",
    summary="Test hardware trigger - capture and save frame",
    response_model=dict
)
async def test_hardware_trigger(
    serial_number: str
):
    """
    Giả lập hardware trigger: Tạo trigger signal, chụp 1 frame và lưu vào test_trigger.png
    API này không cần authentication để dễ test.

    - **serial_number**: Serial number của camera

    Returns: Success message with saved file path
    """
    import cv2
    import os
    import asyncio
    from pathlib import Path

    # Tạo trigger file
    trigger_file = Path(f"/tmp/camera_trigger_{serial_number}")
    trigger_file.touch()

    # Đợi camera producer xử lý trigger và chụp frame (max 2s)
    max_wait = 2.0
    wait_interval = 0.05
    waited = 0.0
    initial_frame_idx = None

    while waited < max_wait:
        frame_data = camera_frame_service.get_frame(serial_number)
        if frame_data:
            current_idx = frame_data['metadata']['frame_idx']
            if initial_frame_idx is None:
                initial_frame_idx = current_idx
            elif current_idx > initial_frame_idx:
                # Frame mới đã được chụp
                break
        await asyncio.sleep(wait_interval)
        waited += wait_interval

    # Đọc frame cuối cùng
    frame_data = camera_frame_service.get_frame(serial_number)

    if frame_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera '{serial_number}' is not streaming. Please ensure camera producer is running."
        )

    # Lưu frame vào file
    output_path = os.path.join(os.getcwd(), "test_trigger.png")
    success = cv2.imwrite(output_path, frame_data['frame'])

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save frame to file"
        )

    return {
        "success": True,
        "message": "Hardware trigger simulated - frame captured and saved",
        "serial_number": serial_number,
        "file_path": output_path,
        "frame_index": frame_data['metadata']['frame_idx'],
        "timestamp": frame_data['metadata']['timestamp']
    }


