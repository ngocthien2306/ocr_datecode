"""
Camera Manager Module
Manages multiple camera instances and handles events
"""

import logging
import threading
from typing import Dict, Any, Optional, Callable
from .camera import Camera, CameraMode

logger = logging.getLogger(__name__)


class CameraManager:
    """
    Manages multiple Camera instances

    Features:
    - Add/remove cameras by serial number
    - Load/stop recipes across cameras
    - Event aggregation and forwarding
    - Thread-safe operations
    """

    def __init__(self, event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None):
        """
        Initialize CameraManager

        Args:
            event_callback: Callback for events from cameras (event_type, data)
        """
        self.cameras: Dict[str, Camera] = {}
        self.event_callback = event_callback
        self._lock = threading.RLock()

        logger.info("CameraManager initialized")

    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to callback"""
        if self.event_callback:
            try:
                # Check if callback is async
                import asyncio
                import inspect
                if inspect.iscoroutinefunction(self.event_callback):
                    # Schedule coroutine in event loop
                    try:
                        loop = asyncio.get_event_loop()
                        asyncio.ensure_future(self.event_callback(event_type, data), loop=loop)
                    except RuntimeError:
                        # No event loop running, create task without loop
                        asyncio.create_task(self.event_callback(event_type, data))
                else:
                    # Sync callback
                    self.event_callback(event_type, data)
            except Exception as e:
                logger.error(f"Error emitting event {event_type}: {e}")

    def _camera_event_handler(self, event_type: str, data: Dict[str, Any]):
        """Handle events from individual cameras"""
        logger.debug(f"Camera event: {event_type}, data: {data}")

        # Forward event to main callback
        self._emit_event(event_type, data)

    def add_camera(
        self,
        serial_number: str,
        pixel_format: str = "Mono8"
    ) -> Dict[str, Any]:
        """
        Add and connect a camera

        Args:
            serial_number: Camera serial number
            pixel_format: Pixel format (Mono8, RGB8, etc.)

        Returns:
            Response dict with status
        """
        with self._lock:
            # Check if already exists
            if serial_number in self.cameras:
                logger.warning(f"Camera {serial_number} already exists")
                return {
                    "success": False,
                    "error": "Camera already connected"
                }

            try:
                # Create camera instance
                camera = Camera(
                    serial_number=serial_number,
                    pixel_format=pixel_format,
                    event_callback=self._camera_event_handler
                )

                # Connect to hardware
                if not camera.connect():
                    return {
                        "success": False,
                        "error": "Failed to connect to camera hardware"
                    }

                # Set to continuous mode for live view and capture
                camera.set_mode(CameraMode.CONTINUOUS)

                # Store in dict
                self.cameras[serial_number] = camera

                logger.info(f"Camera {serial_number} added successfully (mode: {camera.mode.value})")

                return {
                    "success": True,
                    "serial_number": serial_number,
                    "mode": camera.mode.value
                }

            except Exception as e:
                logger.error(f"Error adding camera {serial_number}: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

    def remove_camera(self, serial_number: str) -> Dict[str, Any]:
        """
        Remove and disconnect a camera

        Args:
            serial_number: Camera serial number

        Returns:
            Response dict with status
        """
        with self._lock:
            if serial_number not in self.cameras:
                logger.warning(f"Camera {serial_number} not found")
                return {
                    "success": False,
                    "error": "Camera not found"
                }

            try:
                camera = self.cameras[serial_number]

                # Stop recipe if running
                if camera.recipe_id:
                    camera.stop_recipe()

                # Disconnect camera
                camera.disconnect()

                # Remove from dict
                del self.cameras[serial_number]

                logger.info(f"Camera {serial_number} removed successfully")

                return {
                    "success": True,
                    "serial_number": serial_number
                }

            except Exception as e:
                logger.error(f"Error removing camera {serial_number}: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

    def load_recipe(self, recipe_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load recipe to all cameras in the recipe

        Args:
            recipe_data: Full recipe JSON

        Returns:
            Response dict with status
        """
        with self._lock:
            try:
                recipe_id = recipe_data.get("_id") or recipe_data.get("id")
                recipe_name = recipe_data.get("name", "Unknown")

                # Find all cameras in recipe
                recipe_cameras = recipe_data.get("cameras", [])
                loaded_cameras = []
                errors = []

                for cam_config in recipe_cameras:
                    serial_number = cam_config.get("serial_number")

                    if serial_number not in self.cameras:
                        error_msg = f"Camera {serial_number} not connected"
                        logger.warning(error_msg)
                        errors.append(error_msg)
                        continue

                    camera = self.cameras[serial_number]

                    # Load recipe to camera
                    if camera.load_recipe(recipe_data):
                        loaded_cameras.append(serial_number)
                        logger.info(f"Recipe loaded to camera {serial_number}")
                    else:
                        error_msg = f"Failed to load recipe to camera {serial_number}"
                        logger.error(error_msg)
                        errors.append(error_msg)

                success = len(loaded_cameras) > 0

                logger.info(
                    f"Recipe '{recipe_name}' loaded to {len(loaded_cameras)}/{len(recipe_cameras)} cameras"
                )

                return {
                    "success": success,
                    "recipe_id": recipe_id,
                    "recipe_name": recipe_name,
                    "loaded_cameras": loaded_cameras,
                    "errors": errors if errors else None
                }

            except Exception as e:
                logger.error(f"Error loading recipe: {e}")
                import traceback
                traceback.print_exc()
                return {
                    "success": False,
                    "error": str(e)
                }

    def stop_recipe(self, recipe_id: str) -> Dict[str, Any]:
        """
        Stop recipe on all cameras

        Args:
            recipe_id: Recipe ID to stop

        Returns:
            Response dict with status
        """
        with self._lock:
            try:
                stopped_cameras = []

                for serial_number, camera in self.cameras.items():
                    if camera.recipe_id == recipe_id:
                        camera.stop_recipe()
                        stopped_cameras.append(serial_number)
                        logger.info(f"Recipe stopped on camera {serial_number}")

                logger.info(f"Recipe {recipe_id} stopped on {len(stopped_cameras)} cameras")

                return {
                    "success": True,
                    "recipe_id": recipe_id,
                    "stopped_cameras": stopped_cameras
                }

            except Exception as e:
                logger.error(f"Error stopping recipe: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

    def set_camera_mode(self, serial_number: str, mode: str) -> Dict[str, Any]:
        """
        Set camera operation mode

        Args:
            serial_number: Camera serial number
            mode: Mode string ("idle", "continuous", "hardware_trigger")

        Returns:
            Response dict with status
        """
        with self._lock:
            if serial_number not in self.cameras:
                return {
                    "success": False,
                    "error": "Camera not found"
                }

            try:
                camera = self.cameras[serial_number]
                camera_mode = CameraMode(mode)
                camera.set_mode(camera_mode)

                logger.info(f"Camera {serial_number} mode set to {mode}")

                return {
                    "success": True,
                    "serial_number": serial_number,
                    "mode": mode
                }

            except ValueError:
                return {
                    "success": False,
                    "error": f"Invalid mode: {mode}"
                }
            except Exception as e:
                logger.error(f"Error setting camera mode: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

    def get_camera_status(self, serial_number: str) -> Optional[Dict[str, Any]]:
        """
        Get camera status

        Args:
            serial_number: Camera serial number

        Returns:
            Status dict or None if not found
        """
        with self._lock:
            if serial_number not in self.cameras:
                return None

            camera = self.cameras[serial_number]

            return {
                "serial_number": serial_number,
                "mode": camera.mode.value,
                "recipe_id": camera.recipe_id,
                "recipe_name": camera.recipe_name,
                "frame_idx": camera.frame_idx
            }

    def get_all_cameras_status(self) -> Dict[str, Any]:
        """
        Get status of all cameras

        Returns:
            Dict mapping serial_number -> status
        """
        with self._lock:
            return {
                serial_number: self.get_camera_status(serial_number)
                for serial_number in self.cameras.keys()
            }

    def simulate_trigger(self, serial_number: str = None, trigger_type: str = "rising_edge") -> Dict[str, Any]:
        """
        Simulate hardware trigger without physical DI pin

        Args:
            serial_number: Camera serial (if None, trigger all cameras in hardware_trigger mode)
            trigger_type: Type of trigger edge

        Returns:
            Response dict
        """
        with self._lock:
            try:
                triggered_cameras = []

                if serial_number:
                    # Trigger specific camera
                    if serial_number not in self.cameras:
                        return {
                            "success": False,
                            "error": f"Camera {serial_number} not found"
                        }

                    camera = self.cameras[serial_number]
                    if camera.mode == CameraMode.HARDWARE_TRIGGER:
                        camera._handle_trigger_event()  # Simulate trigger
                        triggered_cameras.append(serial_number)
                        logger.info(f"Simulated trigger for camera {serial_number}")
                    else:
                        return {
                            "success": False,
                            "error": f"Camera {serial_number} not in hardware_trigger mode (current: {camera.mode.value})"
                        }
                else:
                    # Trigger all cameras in hardware_trigger mode
                    for sn, camera in self.cameras.items():
                        if camera.mode == CameraMode.HARDWARE_TRIGGER:
                            camera._handle_trigger_event()
                            triggered_cameras.append(sn)
                            logger.info(f"Simulated trigger for camera {sn}")

                if not triggered_cameras:
                    return {
                        "success": False,
                        "error": "No cameras in hardware_trigger mode"
                    }

                return {
                    "success": True,
                    "triggered_cameras": triggered_cameras,
                    "trigger_type": trigger_type
                }

            except Exception as e:
                logger.error(f"Error simulating trigger: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

    def simulate_trigger_sequence(self, serial_number: str = None, count: int = 5, interval_ms: int = 1000) -> Dict[str, Any]:
        """
        Simulate a sequence of triggers

        Args:
            serial_number: Camera serial (if None, trigger all cameras)
            count: Number of triggers
            interval_ms: Interval between triggers in ms

        Returns:
            Response dict
        """
        import threading
        import time

        def trigger_loop():
            for i in range(count):
                logger.info(f"Trigger sequence {i+1}/{count}")
                self.simulate_trigger(serial_number=serial_number)
                if i < count - 1:  # Don't sleep after last trigger
                    time.sleep(interval_ms / 1000.0)

        # Start trigger sequence in background thread
        thread = threading.Thread(target=trigger_loop, daemon=True)
        thread.start()

        return {
            "success": True,
            "message": f"Trigger sequence started: {count} triggers every {interval_ms}ms",
            "count": count,
            "interval_ms": interval_ms
        }

    def update_camera_settings(self, serial_number: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update camera settings (exposure, gain, etc.)

        Args:
            serial_number: Camera serial number
            settings: Settings dict with exposure_time, gain, etc.

        Returns:
            Response dict with status
        """
        with self._lock:
            if serial_number not in self.cameras:
                return {
                    "success": False,
                    "error": f"Camera {serial_number} not found"
                }

            try:
                camera = self.cameras[serial_number]

                # Update settings
                if "exposure_time" in settings:
                    camera.exposure_time = settings["exposure_time"]
                if "gain" in settings:
                    camera.gain = settings["gain"]
                if "trigger_activation" in settings:
                    camera.trigger_activation = settings["trigger_activation"]
                if "di_number" in settings:
                    camera.di_number = settings["di_number"]
                if "delay_trigger" in settings:
                    camera.delay_trigger = settings["delay_trigger"]

                # Apply settings to camera hardware
                camera._apply_settings()

                logger.info(f"Settings updated for camera {serial_number}: {settings}")

                return {
                    "success": True,
                    "serial_number": serial_number,
                    "settings": settings
                }

            except Exception as e:
                logger.error(f"Error updating camera settings: {e}")
                import traceback
                traceback.print_exc()
                return {
                    "success": False,
                    "error": str(e)
                }

    def shutdown(self):
        """Shutdown all cameras"""
        with self._lock:
            logger.info("Shutting down CameraManager...")

            serial_numbers = list(self.cameras.keys())
            for serial_number in serial_numbers:
                try:
                    self.remove_camera(serial_number)
                except Exception as e:
                    logger.error(f"Error shutting down camera {serial_number}: {e}")

            logger.info("CameraManager shutdown complete")
