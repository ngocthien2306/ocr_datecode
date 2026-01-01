#!/usr/bin/env python3
"""
Camera Management Service
Main entry point for the camera management system

This service:
- Manages multiple Basler cameras
- Communicates with Backend via WebSocket
- Handles trigger events and inference
- Writes frames to shared memory
"""

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from camera_management.camera_manager import CameraManager
from camera_management.websocket_client import CameraWebSocketClient

# Import for API calls
import requests

# Configure logging
LOGS_DIR = Path("/home/demo/Source/ocr_datecode/backend/logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / 'camera_management.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)


class CameraManagementService:
    """
    Main service class

    Coordinates:
    - CameraManager: Manages camera instances
    - WebSocketClient: Communication with backend
    - Startup initialization: Load running recipes from API
    """

    def __init__(self):
        """Initialize service"""
        # Get configuration from environment
        self.api_base = os.environ.get("API_BASE", "http://localhost:8000")
        ws_host = os.environ.get("WS_HOST", "localhost")
        ws_port = os.environ.get("WS_PORT", "8000")
        self.ws_url = f"ws://{ws_host}:{ws_port}/ws/camera-management"

        # Initialize camera manager
        self.camera_manager = CameraManager(
            event_callback=self._handle_camera_event
        )

        # Initialize WebSocket client
        self.ws_client = CameraWebSocketClient(
            server_url=self.ws_url,
            message_callback=self._handle_ws_message,
            heartbeat_interval=30
        )

        self.running = False

        logger.info(f"CameraManagementService initialized")
        logger.info(f"API Base: {self.api_base}")
        logger.info(f"WebSocket URL: {self.ws_url}")

    async def _handle_camera_event(self, event_type: str, data: Dict[str, Any]):
        """
        Handle events from CameraManager

        Events:
        - camera_connected
        - camera_disconnected
        - camera_error
        - inference_result
        """
        logger.info(f"Camera event: {event_type}, data: {data}")

        # Forward event to backend via WebSocket
        await self.ws_client.send_message({
            "event": event_type,
            "data": data
        })

    async def _handle_ws_message(self, message: Dict[str, Any]):
        """
        Handle WebSocket messages from backend

        Supported events:
        - connect_camera: Add and connect camera
        - disconnect_camera: Disconnect and remove camera
        - load_recipe: Load recipe to cameras
        - stop_recipe: Stop recipe
        - set_camera_mode: Set camera operation mode
        - get_camera_status: Get camera status
        """
        event = message.get("event")
        data = message.get("data", {})

        logger.debug(f"Handling WS message: {event}")

        try:
            if event == "connect_camera":
                result = self.camera_manager.add_camera(
                    serial_number=data["serial_number"],
                    device_index=data["device_index"]
                )
                await self.ws_client.send_message({
                    "event": "connect_camera_response",
                    "data": result
                })

            elif event == "disconnect_camera":
                result = self.camera_manager.remove_camera(
                    serial_number=data["serial_number"]
                )
                await self.ws_client.send_message({
                    "event": "disconnect_camera_response",
                    "data": result
                })

            elif event == "load_recipe":
                result = self.camera_manager.load_recipe(
                    recipe_data=data["recipe"]
                )
                await self.ws_client.send_message({
                    "event": "load_recipe_response",
                    "data": result
                })

            elif event == "stop_recipe":
                result = self.camera_manager.stop_recipe(
                    recipe_id=data["recipe_id"]
                )
                await self.ws_client.send_message({
                    "event": "stop_recipe_response",
                    "data": result
                })

            elif event == "set_camera_mode":
                result = self.camera_manager.set_camera_mode(
                    serial_number=data["serial_number"],
                    mode=data["mode"]
                )
                await self.ws_client.send_message({
                    "event": "set_camera_mode_response",
                    "data": result
                })

            elif event == "get_camera_status":
                serial_number = data.get("serial_number")
                if serial_number:
                    status = self.camera_manager.get_camera_status(serial_number)
                else:
                    status = self.camera_manager.get_all_cameras_status()

                await self.ws_client.send_message({
                    "event": "camera_status_response",
                    "data": status
                })

            else:
                logger.warning(f"Unknown event: {event}")

        except Exception as e:
            logger.error(f"Error handling message {event}: {e}")
            import traceback
            traceback.print_exc()

            # Send error response
            await self.ws_client.send_message({
                "event": f"{event}_error",
                "data": {
                    "error": str(e)
                }
            })

    async def _check_running_recipes(self):
        """
        Check for running recipes on startup and auto-initialize cameras

        This ensures service can recover from restart
        """
        try:
            logger.info("Checking for running recipes...")

            # Call API to get latest recipe load (only 1 recipe at a time)
            url = f"{self.api_base}/api/recipes/loads/latest"
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                logger.warning(f"Failed to get latest recipe: {response.status_code}")
                return

            load_data = response.json()

            if not load_data:
                logger.info("No running recipe found")
                return

            # Extract recipe info
            recipe_id = load_data.get("recipe_id")
            metadata = load_data.get("metadata", {})

            if not recipe_id:
                logger.warning("No recipe_id in latest load data")
                return

            logger.info(f"Found running recipe: {metadata.get('name', recipe_id)}")

            # Use metadata directly (it already contains full recipe data)
            recipe_data = metadata.copy()
            recipe_data["_id"] = recipe_id
            recipe_data["id"] = recipe_id

            # Connect cameras
            cameras = recipe_data.get("cameras", [])

            if not cameras:
                logger.warning("No cameras in recipe")
                return

            for idx, cam in enumerate(cameras):
                serial_number = cam.get("serial_number")

                # Try to connect
                result = self.camera_manager.add_camera(
                    serial_number=serial_number,
                    device_index=idx  # Assume device index matches order
                )

                if result["success"]:
                    logger.info(f"Connected camera {serial_number}")
                else:
                    logger.error(f"Failed to connect camera {serial_number}: {result.get('error')}")

            # Load recipe
            load_result = self.camera_manager.load_recipe(recipe_data)

            if load_result["success"]:
                logger.info(f"Recipe auto-loaded successfully: {metadata.get('name')}")
            else:
                logger.error(f"Failed to auto-load recipe: {load_result.get('error')}")

        except Exception as e:
            logger.error(f"Error checking running recipes: {e}")
            import traceback
            traceback.print_exc()

    async def start(self):
        """Start the service"""
        self.running = True

        logger.info("Starting CameraManagementService...")

        # Start WebSocket client
        await self.ws_client.start()

        # Wait for WebSocket connection
        if not await self.ws_client.wait_until_connected(timeout=30):
            logger.error("Failed to connect to WebSocket server within timeout")
            return

        logger.info("WebSocket connected, checking for running recipes...")

        # Check for running recipes and auto-initialize
        await self._check_running_recipes()

        logger.info("CameraManagementService started successfully")

        # Keep running
        while self.running:
            await asyncio.sleep(1)

    async def stop(self):
        """Stop the service"""
        logger.info("Stopping CameraManagementService...")

        self.running = False

        # Shutdown camera manager
        self.camera_manager.shutdown()

        # Stop WebSocket client
        await self.ws_client.stop()

        logger.info("CameraManagementService stopped")


# Global service instance
service: Optional[CameraManagementService] = None


def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {sig}, shutting down...")
    if service:
        asyncio.create_task(service.stop())


async def main():
    """Main entry point"""
    global service

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create and start service
    service = CameraManagementService()

    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Service error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await service.stop()


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("Camera Management Service v1.0")
    logger.info("=" * 80)

    asyncio.run(main())
