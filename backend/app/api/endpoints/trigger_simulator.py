"""
Trigger Simulator API
Simulate hardware trigger for testing without physical DI pins
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional
import logging
import asyncio

from app.api.dependencies.auth import get_current_user
from app.api.websocket.camera_ws import send_request_and_wait
from app.services.camera_service_supervisor import require_camera_service, notify_service_down

# Waiting for the ACK includes the actual capture time on the AI service side
TRIGGER_ACK_TIMEOUT = 15.0

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trigger-simulator", tags=["Trigger Simulator"])


class TriggerSimulateRequest(BaseModel):
    """Request to simulate trigger"""
    serial_number: Optional[str] = None  # If None, trigger all cameras
    trigger_type: str = "rising_edge"  # rising_edge, falling_edge, any_edge


class TriggerContinuousRequest(BaseModel):
    """Request to simulate continuous triggers"""
    serial_number: Optional[str] = None
    trigger_type: str = "rising_edge"
    count: int = Field(default=5, ge=1, le=100, description="Number of triggers (1-100)")
    interval_seconds: float = Field(default=1.0, ge=0.1, le=10.0, description="Interval between triggers in seconds (0.1-10.0)")


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

    # Ensure camera service is connected (notifies UI + triggers recovery if down)
    await require_camera_service("Camera management service is not connected")

    # Send trigger simulation command and wait for the AI service's ACK, so a
    # 200 here means the trigger was actually received AND executed.
    message = {
        "event": "simulate_trigger",
        "data": {
            "serial_number": request.serial_number,
            "trigger_type": request.trigger_type
        }
    }

    response = await send_request_and_wait(message, timeout=TRIGGER_ACK_TIMEOUT)

    if response is None:
        # Send failed or no ACK — connection is suspect: tell FE + kick recovery
        await notify_service_down("Camera service did not acknowledge trigger command")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Camera service did not acknowledge trigger command — please retry shortly"
        )

    if not response.get("success", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=response.get("error", "Trigger simulation failed")
        )

    logger.info(
        f"Trigger simulated by {current_user.username}: "
        f"camera={request.serial_number or 'all'}, type={request.trigger_type}, "
        f"triggered={response.get('cameras_triggered', [])}"
    )

    return {
        "success": True,
        "message": f"Trigger executed on {'camera ' + request.serial_number if request.serial_number else 'all cameras'}",
        "trigger_type": request.trigger_type,
        "cameras_triggered": response.get("cameras_triggered", [])
    }


@router.post(
    "/simulate-continuous",
    summary="Simulate continuous triggers with interval",
    response_model=dict
)
async def simulate_continuous_triggers(
    request: TriggerContinuousRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Simulate continuous trigger events with specified interval.

    **Backend loops and sends multiple trigger commands** to camera service.

    **Use cases:**
    - Stress testing with multiple triggers
    - Simulating production line with specific intervals
    - Performance testing

    **Parameters:**
    - **serial_number**: Camera serial (if None, trigger all cameras in hardware_trigger mode)
    - **trigger_type**: Type of edge (rising_edge, falling_edge, any_edge)
    - **count**: Number of triggers to send (1-100)
    - **interval_seconds**: Interval between each trigger in seconds (0.1-10.0)

    **Returns:**
    - Success status with statistics

    **Note:** This endpoint blocks until all triggers are sent.
    """

    # Ensure camera service is connected (notifies UI + triggers recovery if down)
    await require_camera_service("Camera management service is not connected")

    logger.info(
        f"Starting continuous trigger by {current_user.username}: "
        f"camera={request.serial_number or 'all'}, type={request.trigger_type}, "
        f"count={request.count}, interval={request.interval_seconds}s"
    )

    # Statistics
    sent_count = 0
    failed_count = 0

    # Loop and send triggers (each one waits for the AI service's ACK)
    for i in range(request.count):
        message = {
            "event": "simulate_trigger",
            "data": {
                "serial_number": request.serial_number,
                "trigger_type": request.trigger_type
            }
        }

        response = await send_request_and_wait(message, timeout=TRIGGER_ACK_TIMEOUT)

        if response is not None and response.get("success", False):
            sent_count += 1
            logger.debug(f"Trigger {i+1}/{request.count} acknowledged")
        else:
            failed_count += 1
            logger.warning(
                f"Trigger {i+1}/{request.count} failed: "
                f"{'no ACK' if response is None else response.get('error')}"
            )

        # Sleep before next trigger (except for last iteration)
        if i < request.count - 1:
            await asyncio.sleep(request.interval_seconds)

    # Log completion
    logger.info(
        f"Continuous trigger completed: {sent_count} sent, {failed_count} failed"
    )

    # Check if any failed
    if failed_count > 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Some triggers failed: {failed_count}/{request.count}"
        )

    return {
        "success": True,
        "message": f"Successfully sent {sent_count} triggers",
        "statistics": {
            "total_requested": request.count,
            "sent": sent_count,
            "failed": failed_count,
            "interval_seconds": request.interval_seconds,
            "serial_number": request.serial_number or "all",
            "trigger_type": request.trigger_type
        }
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

    # Ensure camera service is connected (notifies UI + triggers recovery if down)
    await require_camera_service("Camera management service is not connected")

    # Send trigger sequence command. The AI service runs the whole sequence
    # before responding, so scale the ACK timeout with count * interval.
    message = {
        "event": "simulate_trigger_sequence",
        "data": {
            "serial_number": serial_number,
            "count": count,
            "interval_ms": interval_ms
        }
    }

    sequence_timeout = count * (interval_ms / 1000.0 + 5.0) + 10.0
    response = await send_request_and_wait(message, timeout=sequence_timeout)

    if response is None:
        await notify_service_down("Camera service did not acknowledge trigger sequence")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Camera service did not acknowledge trigger sequence command"
        )

    if not response.get("success", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=response.get("error", "Trigger sequence failed")
        )

    logger.info(
        f"Trigger sequence started by {current_user.username}: "
        f"camera={serial_number or 'all'}, count={count}, interval={interval_ms}ms"
    )

    return {
        "success": True,
        "message": f"Trigger sequence started: {count} triggers every {interval_ms}ms",
        "serial_number": serial_number,
        "count": count,
        "interval_ms": interval_ms
    }
