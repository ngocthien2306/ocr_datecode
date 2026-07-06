"""
Camera Management WebSocket Endpoint
Handles bidirectional communication with AI Service (CameraManagement)
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import logging
from typing import Optional, Dict, Any
from collections import deque
import asyncio
import uuid

logger = logging.getLogger(__name__)

# Pending requests for request-response pattern
_pending_requests: Dict[str, asyncio.Future] = {}

# Recently processed inference_result msg_ids (dedup for at-least-once resend)
_recent_result_msg_ids: deque = deque(maxlen=500)

# App-level heartbeat: protocol ping/pong keeps working even when the app on
# either side has stopped reading messages, so a zombie connection can only be
# detected by a request/response at this layer.
HEARTBEAT_INTERVAL = 10.0
HEARTBEAT_TIMEOUT = 5.0
HEARTBEAT_MAX_MISSES = 2

router = APIRouter()


class CameraWebSocketManager:
    """
    Manages WebSocket connection with Camera Management Service

    Features:
    - Single connection to AI service
    - Message routing and forwarding
    - Connection state tracking
    """

    def __init__(self):
        self.camera_service_ws: Optional[WebSocket] = None
        self.connected = False
        self._lock = asyncio.Lock()

        logger.info("CameraWebSocketManager initialized")

    async def connect(self, websocket: WebSocket):
        """Connect camera service WebSocket"""
        async with self._lock:
            await websocket.accept()

            # Disconnect previous connection if exists
            if self.camera_service_ws and self.connected:
                logger.warning("Replacing existing camera service connection")
                try:
                    await self.camera_service_ws.close()
                except:
                    pass

            self.camera_service_ws = websocket
            self.connected = True

            logger.info("Camera service connected via WebSocket")

        # Notify frontend clients that camera service is back up so the
        # ServiceDownOverlay hides itself. Done outside the lock to avoid
        # holding it across the emit.
        try:
            from app.services.socketio_service import emit_camera_service_status
            await emit_camera_service_status({
                'connected': True,
                'message': 'Camera Management Service connected'
            })
        except Exception as e:
            logger.error(f"Failed to emit camera_service_status up: {e}")

    async def disconnect(self, websocket: Optional[WebSocket] = None):
        """
        Disconnect camera service WebSocket.

        Args:
            websocket: The socket owned by the calling handler. When the AI
                service reconnects, connect() replaces the old socket; the old
                handler's cleanup then runs AFTER the new connection is live and
                must not tear down the new connection's state.
        """
        async with self._lock:
            if websocket is not None and self.camera_service_ws is not websocket:
                logger.info("Stale camera service handler exited; keeping current connection")
                return

            self.connected = False
            self.camera_service_ws = None

            logger.info("Camera service disconnected")

        # Notify all frontend clients that camera service is down.
        # Done outside the lock to avoid holding it across the emit.
        try:
            from app.services.socketio_service import emit_camera_service_status
            await emit_camera_service_status({
                'connected': False,
                'message': 'Camera Management Service disconnected'
            })
        except Exception as e:
            logger.error(f"Failed to emit camera_service_status down: {e}")

    async def send_to_camera_service(self, message: dict) -> bool:
        """
        Send message to camera service

        Args:
            message: Message dict

        Returns:
            True if sent successfully
        """
        if not self.connected or not self.camera_service_ws:
            logger.warning("Cannot send to camera service: not connected")
            return False

        try:
            message_json = json.dumps(message)
            await self.camera_service_ws.send_text(message_json)
            logger.debug(f"Sent to camera service: {message.get('event', 'unknown')}")
            return True

        except Exception as e:
            logger.error(f"Error sending to camera service: {e}")
            self.connected = False
            return False

    def is_connected(self) -> bool:
        """Check if camera service is connected"""
        return self.connected


# Global singleton instance
camera_ws_manager = CameraWebSocketManager()


async def send_request_and_wait(message: dict, timeout: float = 10.0) -> Optional[dict]:
    """
    Send a command to the camera service and wait for the matching response
    (correlated via data.request_id, resolved in handle_camera_service_message).

    Returns:
        Response data dict, or None if the send failed or no response arrived
        within the timeout (i.e. the command may not have been executed).
    """
    request_id = str(uuid.uuid4())
    message.setdefault("data", {})["request_id"] = request_id

    future: asyncio.Future = asyncio.get_running_loop().create_future()
    _pending_requests[request_id] = future

    try:
        if not await camera_ws_manager.send_to_camera_service(message):
            return None
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(
            f"No response from camera service for {message.get('event')} "
            f"(request_id={request_id}, timeout={timeout}s)"
        )
        return None
    finally:
        _pending_requests.pop(request_id, None)


async def _heartbeat_loop(websocket: WebSocket):
    """
    Per-connection app-level heartbeat. After HEARTBEAT_MAX_MISSES consecutive
    unanswered pings, closes the socket so both sides tear down and the AI
    service reconnects cleanly.
    """
    misses = 0
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL)

        if camera_ws_manager.camera_service_ws is not websocket:
            return  # Connection was replaced; this loop is obsolete

        response = await send_request_and_wait(
            {"event": "ping", "data": {}},
            timeout=HEARTBEAT_TIMEOUT
        )

        if response is not None:
            misses = 0
            continue

        misses += 1
        logger.warning(f"Camera service heartbeat miss {misses}/{HEARTBEAT_MAX_MISSES}")

        if misses >= HEARTBEAT_MAX_MISSES:
            logger.error("Camera service heartbeat failed — closing zombie connection")
            try:
                await websocket.close()
            except Exception:
                pass
            return


@router.websocket("/ws/camera-management")
async def camera_management_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for Camera Management Service

    This endpoint:
    - Accepts connection from AI service
    - Forwards commands from backend to AI service
    - Receives events from AI service and processes them
    - Runs an app-level heartbeat to detect zombie connections
    """
    await camera_ws_manager.connect(websocket)

    heartbeat_task = asyncio.create_task(_heartbeat_loop(websocket))

    try:
        # Main receive loop — reads from THIS handler's socket, not the shared
        # manager state, so a replaced (stale) handler cannot steal messages.
        while True:
            message_raw = await websocket.receive_text()

            try:
                message = json.loads(message_raw)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from camera service: {e}")
                continue

            logger.debug(f"Received from camera service: {message.get('event', 'unknown')}")

            # Handle message from camera service
            await handle_camera_service_message(message)

    except WebSocketDisconnect:
        logger.info("Camera service WebSocket disconnected")

    except Exception as e:
        logger.error(f"Error in camera WebSocket: {e}")
        import traceback
        traceback.print_exc()

    finally:
        heartbeat_task.cancel()
        await camera_ws_manager.disconnect(websocket)


async def handle_camera_service_message(message: dict):
    """
    Handle message from camera service

    Events from AI service:
    - service_connected: Service startup notification
    - service_disconnected: Service shutdown notification
    - camera_connected: Camera connected successfully
    - camera_disconnected: Camera disconnected
    - camera_error: Camera error occurred
    - inference_result: Inference result ready
    - *_response: Response to commands
    """
    event = message.get("event")
    data = message.get("data", {})

    if event != "live_frame":
        logger.info(f"Camera service event: {event}")

    # Resolve request-response futures (ACK pattern). Any response carrying a
    # known request_id unblocks the send_request_and_wait() caller.
    if isinstance(data, dict):
        request_id = data.get("request_id")
        if request_id and request_id in _pending_requests:
            future = _pending_requests.pop(request_id)
            if not future.done():
                future.set_result(data)

    # At-least-once delivery: the AI service resends events that were never
    # acked (msg_id-tagged). Dedup already-processed ones — ack again and drop.
    msg_id = data.get("msg_id") if isinstance(data, dict) else None
    if msg_id and msg_id in _recent_result_msg_ids:
        logger.info(f"Duplicate {event} (msg_id={msg_id}) — ack only")
        await camera_ws_manager.send_to_camera_service({
            "event": "event_ack",
            "data": {"msg_id": msg_id}
        })
        return

    async def _ack_processed():
        """Record + ack a msg_id-tagged event once it is safely processed"""
        if msg_id:
            _recent_result_msg_ids.append(msg_id)
            await camera_ws_manager.send_to_camera_service({
                "event": "event_ack",
                "data": {"msg_id": msg_id}
            })

    try:
        if event == "service_connected":
            logger.info(f"Camera service connected: {data}")

        elif event == "service_disconnected":
            logger.info(f"Camera service disconnected: {data}")

        elif event == "service_shutdown":
            # Camera service is shutting down gracefully
            logger.info(f"Camera service shutting down: {data}")

            # Get list of cameras that were connected
            connected_cameras = data.get("connected_cameras", [])

            # Update database: mark all cameras as disconnected
            from app.repositories.camera_repository import CameraRepository
            from app.db.mongodb import get_database

            db = get_database()
            camera_repo = CameraRepository(db)

            for serial_number in connected_cameras:
                try:
                    # Get camera by serial number to get camera_id
                    camera = await camera_repo.get_by_serial(serial_number)
                    if camera:
                        camera_id = camera.get('camera_id')
                        await camera_repo.update_connection_status(
                            camera_id=camera_id,
                            is_connected=False
                        )
                        logger.info(f"Marked camera {serial_number} as disconnected in DB")
                    else:
                        logger.warning(f"Camera {serial_number} not found in DB")
                except Exception as e:
                    logger.error(f"Failed to disconnect {serial_number} in DB: {e}")

            # Notify frontend clients via SocketIO
            from app.services.socketio_service import emit_camera_service_status
            await emit_camera_service_status({
                'connected': False,
                'message': data.get('message', 'Camera Management Service disconnected'),
                'cameras_affected': connected_cameras
            })

        elif event == "camera_connected":
            serial_number = data.get('serial_number')
            logger.info(f"Camera connected: {serial_number}")

            # Update DB: mark camera as connected
            from app.repositories.camera_repository import CameraRepository
            from app.db.mongodb import get_database

            db = get_database()
            camera_repo = CameraRepository(db)

            try:
                camera = await camera_repo.get_by_serial(serial_number)
                if camera:
                    camera_id = camera.get('camera_id')
                    await camera_repo.update_connection_status(
                        camera_id=camera_id,
                        is_connected=True
                    )
                    logger.info(f"Updated camera {serial_number} status to connected in DB")

                    # Notify frontend via SocketIO
                    from app.services.socketio_service import emit_camera_status_update
                    await emit_camera_status_update({
                        'serial_number': serial_number,
                        'camera_id': camera_id,
                        'is_connected': True,
                        'event': 'camera_connected'
                    })
                else:
                    logger.warning(f"Camera {serial_number} not found in DB")
            except Exception as e:
                logger.error(f"Failed to update camera {serial_number} status: {e}")

        elif event == "camera_disconnected":
            serial_number = data.get('serial_number')
            logger.info(f"Camera disconnected: {serial_number}")

            # Update DB: mark camera as disconnected
            from app.repositories.camera_repository import CameraRepository
            from app.db.mongodb import get_database

            db = get_database()
            camera_repo = CameraRepository(db)

            try:
                camera = await camera_repo.get_by_serial(serial_number)
                if camera:
                    camera_id = camera.get('camera_id')
                    await camera_repo.update_connection_status(
                        camera_id=camera_id,
                        is_connected=False
                    )
                    logger.info(f"Updated camera {serial_number} status to disconnected in DB")

                    # Notify frontend via SocketIO
                    from app.services.socketio_service import emit_camera_status_update
                    await emit_camera_status_update({
                        'serial_number': serial_number,
                        'camera_id': camera_id,
                        'is_connected': False,
                        'event': 'camera_disconnected'
                    })
                else:
                    logger.warning(f"Camera {serial_number} not found in DB")
            except Exception as e:
                logger.error(f"Failed to update camera {serial_number} status: {e}")

        elif event == "camera_error":
            logger.error(f"Camera error: {data}")
            # Forward to frontend so the operator sees the failure
            from app.services.socketio_service import emit_camera_alert
            await emit_camera_alert({"type": "camera_error", **(data or {})})

        elif event == "trigger_error":
            # Camera capture failed after a trigger — the operator must know,
            # otherwise the trigger silently produces nothing.
            logger.error(f"Trigger error from camera service: {data}")
            from app.services.socketio_service import emit_camera_alert
            await emit_camera_alert({"type": "trigger_error", **(data or {})})
            await _ack_processed()

        elif event == "inference_skipped":
            logger.warning(f"Inference skipped: {data}")
            from app.services.socketio_service import emit_camera_alert
            await emit_camera_alert({"type": "inference_skipped", **(data or {})})
            await _ack_processed()

        elif event == "inference_result":
            # Ack only after the result is safely stored in DB, so a failed
            # save leads to a resend from the AI service instead of data loss.
            result = await process_inference_result(data)
            if result is not None:
                await _ack_processed()

        elif event == "live_frame":
            # Forward live frame to frontend via SocketIO
            from app.services.socketio_service import emit_live_frame
            await emit_live_frame(data)

        elif event == "discover_cameras_response":
            # Handle discover cameras response
            logger.info(f"Discover cameras response: {data.get('count', 0)} devices found")
            # Resolve pending request if exists
            if "discover_cameras" in _pending_requests:
                future = _pending_requests.pop("discover_cameras")
                if not future.done():
                    future.set_result(data)

        elif event == "ping_response":
            # Heartbeat reply — already resolved via request_id above
            pass

        elif event.endswith("_response"):
            # Response to command - will be handled by calling code
            logger.debug(f"Response received: {event}")

        else:
            logger.warning(f"Unknown event from camera service: {event}")

    except Exception as e:
        logger.error(f"Error handling camera service message: {e}")
        import traceback
        traceback.print_exc()


async def process_inference_result(data: dict):
    """
    Process inference result from camera service

    Args:
        data: Inference result data

    Returns:
        The saved result, or None if processing/saving failed (caller must
        NOT ack in that case so the AI service will resend).
    """
    try:
        # Import here to avoid circular dependency
        from app.db.mongodb import get_database
        from app.services.inference_result_service import create_inference_result_service

        db = get_database()
        inference_service = create_inference_result_service(db)

        # Process and save result
        result = await inference_service.process_inference_result(data)

        if result:
            logger.info(f"Inference result processed successfully: {result.id}")
        else:
            logger.error("Failed to process inference result")

        return result

    except Exception as e:
        logger.error(f"Error in process_inference_result: {e}")
        import traceback
        traceback.print_exc()
        return None


# Utility functions for other endpoints to use


async def send_connect_camera(serial_number: str, pixel_format: str = "BGR8") -> bool:
    """Send connect camera command to AI service"""
    return await camera_ws_manager.send_to_camera_service({
        "event": "connect_camera",
        "data": {
            "serial_number": serial_number,
            "pixel_format": pixel_format
        }
    })


async def send_disconnect_camera(serial_number: str) -> bool:
    """Send disconnect camera command to AI service"""
    return await camera_ws_manager.send_to_camera_service({
        "event": "disconnect_camera",
        "data": {
            "serial_number": serial_number
        }
    })


async def send_load_recipe(recipe_data: dict) -> bool:
    """Send load recipe command to AI service"""
    return await camera_ws_manager.send_to_camera_service({
        "event": "load_recipe",
        "data": {
            "recipe": recipe_data
        }
    })


async def send_stop_recipe(recipe_id: str) -> bool:
    """Send stop recipe command to AI service"""
    return await camera_ws_manager.send_to_camera_service({
        "event": "stop_recipe",
        "data": {
            "recipe_id": recipe_id
        }
    })


async def send_set_inference_mode(recipe_id: str, enabled: bool) -> bool:
    """Send set inference mode command to AI service"""
    return await camera_ws_manager.send_to_camera_service({
        "event": "set_inference_mode",
        "data": {
            "recipe_id": recipe_id,
            "enabled": enabled
        }
    })


async def send_set_camera_mode(serial_number: str, mode: str) -> bool:
    """Send set camera mode command to AI service"""
    return await camera_ws_manager.send_to_camera_service({
        "event": "set_camera_mode",
        "data": {
            "serial_number": serial_number,
            "mode": mode
        }
    })


async def send_get_camera_status(serial_number: Optional[str] = None) -> bool:
    """Send get camera status command to AI service"""
    data = {}
    if serial_number:
        data["serial_number"] = serial_number

    return await camera_ws_manager.send_to_camera_service({
        "event": "get_camera_status",
        "data": data
    })


async def send_start_live_view(serial_number: str, frame_rate: int = 10) -> bool:
    """Send start live view command to AI service"""
    return await camera_ws_manager.send_to_camera_service({
        "event": "start_live_view",
        "data": {
            "serial_number": serial_number,
            "frame_rate": frame_rate
        }
    })


async def send_stop_live_view(serial_number: str) -> bool:
    """Send stop live view command to AI service"""
    return await camera_ws_manager.send_to_camera_service({
        "event": "stop_live_view",
        "data": {
            "serial_number": serial_number
        }
    })


async def send_update_camera_settings(serial_number: str, settings: Dict[str, Any]) -> bool:
    """Send update camera settings command to AI service"""
    return await camera_ws_manager.send_to_camera_service({
        "event": "update_camera_settings",
        "data": {
            "serial_number": serial_number,
            "settings": settings
        }
    })


async def send_discover_cameras(timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    """
    Send discover cameras command to AI service and wait for response

    Args:
        timeout: Timeout in seconds to wait for response

    Returns:
        Response dict with discovered devices or None if failed/timeout
    """
    # Check if camera service is connected
    if not camera_ws_manager.is_connected():
        logger.warning("Cannot discover cameras: camera service not connected")
        return None

    # Create future for response
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    _pending_requests["discover_cameras"] = future

    try:
        # Send discover command
        success = await camera_ws_manager.send_to_camera_service({
            "event": "discover_cameras",
            "data": {}
        })

        if not success:
            logger.error("Failed to send discover_cameras command")
            return None

        # Wait for response with timeout
        result = await asyncio.wait_for(future, timeout=timeout)
        return result

    except asyncio.TimeoutError:
        logger.error(f"Discover cameras request timed out after {timeout}s")
        return None

    except Exception as e:
        logger.error(f"Error in send_discover_cameras: {e}")
        return None

    finally:
        # Cleanup pending request
        _pending_requests.pop("discover_cameras", None)
