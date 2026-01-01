"""
Trigger Simulator API
Simulate hardware trigger for testing without physical DI pins
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional
import logging

from app.api.dependencies.auth import get_current_user
from app.api.websocket.camera_ws import camera_ws_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trigger-simulator", tags=["Trigger Simulator"])


class TriggerSimulateRequest(BaseModel):
    """Request to simulate trigger"""
    serial_number: Optional[str] = None  # If None, trigger all cameras
    trigger_type: str = "rising_edge"  # rising_edge, falling_edge, any_edge


@router.post(
    "/simulate",
    summary="Simulate hardware trigger",
    response_model=dict
)
async def simulate_trigger(
    request: TriggerSimulateRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Simulate hardware trigger event.

    This endpoint sends a trigger event to the camera management service
    without requiring physical DI pin signal.

    **Use cases:**
    - Testing recipe without hardware
    - Manual triggering for debug
    - Automated testing

    **Parameters:**
    - **serial_number**: Camera serial (if None, trigger all cameras in hardware_trigger mode)
    - **trigger_type**: Type of edge (rising_edge, falling_edge, any_edge)

    **Returns:**
    - Success status and message
    """

    # Check if camera service is connected
    if not camera_ws_manager.is_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Camera management service is not connected"
        )

    # Send trigger simulation command via WebSocket
    message = {
        "event": "simulate_trigger",
        "data": {
            "serial_number": request.serial_number,
            "trigger_type": request.trigger_type
        }
    }

    success = await camera_ws_manager.send_to_camera_service(message)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send trigger simulation command"
        )

    logger.info(
        f"Trigger simulated by {current_user['username']}: "
        f"camera={request.serial_number or 'all'}, type={request.trigger_type}"
    )

    return {
        "success": True,
        "message": f"Trigger simulation sent to {'camera ' + request.serial_number if request.serial_number else 'all cameras'}",
        "trigger_type": request.trigger_type
    }


@router.post(
    "/simulate-sequence",
    summary="Simulate multiple triggers in sequence",
    response_model=dict
)
async def simulate_trigger_sequence(
    serial_number: Optional[str] = None,
    count: int = 5,
    interval_ms: int = 1000,
    current_user: dict = Depends(get_current_user)
):
    """
    Simulate a sequence of trigger events for testing.

    **Parameters:**
    - **serial_number**: Camera serial (if None, trigger all cameras)
    - **count**: Number of triggers to simulate (1-100)
    - **interval_ms**: Interval between triggers in milliseconds (100-10000)

    **Returns:**
    - Success status and sequence info
    """

    if count < 1 or count > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Count must be between 1 and 100"
        )

    if interval_ms < 100 or interval_ms > 10000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Interval must be between 100ms and 10000ms"
        )

    # Check if camera service is connected
    if not camera_ws_manager.is_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Camera management service is not connected"
        )

    # Send trigger sequence command
    message = {
        "event": "simulate_trigger_sequence",
        "data": {
            "serial_number": serial_number,
            "count": count,
            "interval_ms": interval_ms
        }
    }

    success = await camera_ws_manager.send_to_camera_service(message)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send trigger sequence command"
        )

    logger.info(
        f"Trigger sequence started by {current_user['username']}: "
        f"camera={serial_number or 'all'}, count={count}, interval={interval_ms}ms"
    )

    return {
        "success": True,
        "message": f"Trigger sequence started: {count} triggers every {interval_ms}ms",
        "serial_number": serial_number,
        "count": count,
        "interval_ms": interval_ms
    }
