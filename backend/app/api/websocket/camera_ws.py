"""
Camera Management WebSocket Endpoint
Handles bidirectional communication with AI Service (CameraManagement)
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import logging
from typing import Optional, Dict, Any
import asyncio

logger = logging.getLogger(__name__)

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

    async def disconnect(self):
        """Disconnect camera service WebSocket"""
        async with self._lock:
            self.connected = False
            self.camera_service_ws = None

            logger.info("Camera service disconnected")

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

    async def receive_from_camera_service(self) -> Optional[dict]:
        """
        Receive message from camera service

        Returns:
            Message dict or None
        """
        if not self.connected or not self.camera_service_ws:
            return None

        try:
            message_raw = await self.camera_service_ws.receive_text()
            message = json.loads(message_raw)
            logger.debug(f"Received from camera service: {message.get('event', 'unknown')}")
            return message

        except WebSocketDisconnect:
            logger.info("Camera service disconnected")
            self.connected = False
            return None

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from camera service: {e}")
            return None

        except Exception as e:
            logger.error(f"Error receiving from camera service: {e}")
            self.connected = False
            return None

    def is_connected(self) -> bool:
        """Check if camera service is connected"""
        return self.connected


# Global singleton instance
camera_ws_manager = CameraWebSocketManager()


@router.websocket("/ws/camera-management")
async def camera_management_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for Camera Management Service

    This endpoint:
    - Accepts connection from AI service
    - Forwards commands from backend to AI service
    - Receives events from AI service and processes them
    """
    await camera_ws_manager.connect(websocket)

    try:
        # Main receive loop
        while camera_ws_manager.is_connected():
            message = await camera_ws_manager.receive_from_camera_service()

            if message is None:
                break

            # Handle message from camera service
            await handle_camera_service_message(message)

    except WebSocketDisconnect:
        logger.info("Camera service WebSocket disconnected")

    except Exception as e:
        logger.error(f"Error in camera WebSocket: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await camera_ws_manager.disconnect()


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

    logger.info(f"Camera service event: {event}")

    try:
        if event == "service_connected":
            logger.info(f"Camera service connected: {data}")

        elif event == "service_disconnected":
            logger.info(f"Camera service disconnected: {data}")

        elif event == "camera_connected":
            logger.info(f"Camera connected: {data.get('serial_number')}")

        elif event == "camera_disconnected":
            logger.info(f"Camera disconnected: {data.get('serial_number')}")

        elif event == "camera_error":
            logger.error(f"Camera error: {data}")

        elif event == "inference_result":
            # Process inference result
            await process_inference_result(data)

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
            # TODO: Emit SocketIO event to frontend when SocketIO is implemented
        else:
            logger.error("Failed to process inference result")

    except Exception as e:
        logger.error(f"Error in process_inference_result: {e}")
        import traceback
        traceback.print_exc()


# Utility functions for other endpoints to use


async def send_connect_camera(serial_number: str, pixel_format: str = "Mono8") -> bool:
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


async def send_update_camera_settings(serial_number: str, settings: Dict[str, Any]) -> bool:
    """Send update camera settings command to AI service"""
    return await camera_ws_manager.send_to_camera_service({
        "event": "update_camera_settings",
        "data": {
            "serial_number": serial_number,
            "settings": settings
        }
    })
