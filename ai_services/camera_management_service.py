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

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from camera_management.camera_manager import CameraManager
from camera_management.websocket_client import CameraWebSocketClient

# Import for API calls
import requests
home = os.environ.get('HOME')

# Configure logging
LOGS_DIR = Path(f"{home}/Source/ocr_datecode/backend/logs")
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
        - frames_captured (triggers inference in main thread)
        - inference_result
        """
        # Special handling for frames_captured event - submit async inference
        if event_type == "frames_captured":
            cameras = data.get("cameras", [])
            results_with_metadata = {
                'group_id': data.get('group_id'),
                'T_capture_complete': data.get('T_capture_complete'),  # For reject timing
                **data.get("results", {})
            }

            logger.info(
                f"Frames captured (Group #{data.get('group_id')}), "
                f"submitting async inference for {len(cameras)} camera(s)..."
            )

            # Submit async inference job (non-blocking)
            self.camera_manager.process_inference(cameras, results_with_metadata)
            return  # Don't forward frames_captured to backend

        # Hide base64 in logs (too long)
        log_data = data.copy() if isinstance(data, dict) else data
        if event_type == "inference_result" and isinstance(log_data, dict):
            # Remove base64 from camera_results.frames for logging
            if 'camera_results' in log_data:
                log_data = {**log_data, 'camera_results': [
                    {**cr, 'frames': [
                        {k: v for k, v in frame.items() if k != 'image_base64'}
                        for frame in cr.get('frames', [])
                    ]}
                    for cr in log_data['camera_results']
                ]}

        # logger.info(f"Camera event: {event_type}, data: {log_data}")

        # Forward event to backend via WebSocket (with full data including base64)
        await self.ws_client.send_message({
            "event": event_type,
            "data": data  # Send original data with base64
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
                    pixel_format=data.get("pixel_format", "BGR8")
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

            elif event == "set_inference_mode":
                # Set inference mode (ONLINE/OFFLINE)
                recipe_id = data.get("recipe_id")
                enabled = data.get("enabled", True)

                result = self.camera_manager.set_inference_mode(enabled)

                await self.ws_client.send_message({
                    "event": "set_inference_mode_response",
                    "data": result
                })

            elif event == "simulate_trigger":
                # Simulate hardware trigger
                serial_number = data.get("serial_number")
                trigger_type = data.get("trigger_type", "rising_edge")

                result = self.camera_manager.simulate_trigger(
                    serial_number=serial_number,
                    trigger_type=trigger_type
                )

                await self.ws_client.send_message({
                    "event": "simulate_trigger_response",
                    "data": result
                })

            elif event == "simulate_trigger_sequence":
                # Simulate trigger sequence
                serial_number = data.get("serial_number")
                count = data.get("count", 5)
                interval_ms = data.get("interval_ms", 1000)

                result = self.camera_manager.simulate_trigger_sequence(
                    serial_number=serial_number,
                    count=count,
                    interval_ms=interval_ms
                )

                await self.ws_client.send_message({
                    "event": "simulate_trigger_sequence_response",
                    "data": result
                })

            elif event == "update_camera_settings":
                # Update camera settings
                serial_number = data.get("serial_number")
                settings = data.get("settings", {})

                result = self.camera_manager.update_camera_settings(
                    serial_number=serial_number,
                    settings=settings
                )

                await self.ws_client.send_message({
                    "event": "update_camera_settings_response",
                    "data": result
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

            # Log cameras config from latest load
            logger.info(f"📋 Latest load - cameras config:")
            for idx, cam in enumerate(recipe_data.get("cameras", [])):
                logger.info(f"  Camera {idx + 1}:")
                logger.info(f"    - camera_id: {cam.get('camera_id')}")
                logger.info(f"    - serial_number: {cam.get('serial_number')}")
                logger.info(f"    - function_type: {cam.get('function_type', 'NOT SET')} ⭐")
                logger.info(f"    - trigger_mode: {cam.get('trigger_mode')}")

            # Connect cameras
            cameras = recipe_data.get("cameras", [])

            if not cameras:
                logger.warning("No cameras in recipe")
                return

            for cam in cameras:
                serial_number = cam.get("serial_number")
                pixel_format = cam.get("pixel_format", "Mono8")

                # Try to connect by serial number
                result = self.camera_manager.add_camera(
                    serial_number=serial_number,
                    pixel_format=pixel_format
                )

                if result["success"]:
                    logger.info(f"Connected camera {serial_number} with {pixel_format}")
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

        # Pass event loop to camera manager for async callbacks from threads
        import asyncio
        self.camera_manager.event_loop = asyncio.get_event_loop()

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

    async def _cleanup_cameras_on_shutdown(self):
        """
        Send shutdown notification via WebSocket to backend
        Backend will handle updating camera status and notifying frontend clients
        """
        try:
            # Get all connected cameras from camera manager
            connected_serials = list(self.camera_manager.cameras.keys())

            if not connected_serials:
                logger.info("No cameras to cleanup")
                return

            logger.info(f"Notifying backend about shutdown ({len(connected_serials)} cameras)...")

            # Send shutdown notification via WebSocket
            await self.ws_client.send_message({
                "event": "service_shutdown",
                "data": {
                    "connected_cameras": connected_serials,
                    "message": "Camera Management Service shutting down"
                }
            })

            logger.info("Shutdown notification sent to backend")

            # Give backend time to process
            await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Error during camera cleanup: {e}")

    async def stop(self):
        """Stop the service"""
        logger.info("Stopping CameraManagementService...")

        self.running = False

        # Update all cameras to disconnected in database
        await self._cleanup_cameras_on_shutdown()

        # Shutdown camera manager
        self.camera_manager.shutdown()

        # Stop WebSocket client
        await self.ws_client.stop()

        logger.info("CameraManagementService stopped")


# Global service instance
service: Optional[CameraManagementService] = None


def signal_handler(sig, _frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {sig}, shutting down...")
    if service:
        # Signal event loop to stop service
        # The actual cleanup happens in main's finally block
        service.running = False


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
