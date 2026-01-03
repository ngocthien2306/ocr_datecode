"""
Camera Manager Module
Manages multiple camera instances and handles events

Refactored to use separate handlers for better organization:
- TriggerHandler: DI polling and software trigger logic
- InferenceHandler: Inference processing and result building
- utils: Common utility functions
"""

import logging
import threading
from typing import Dict, Any, Optional, Callable
from .camera import Camera, CameraMode
from .trigger_handler import TriggerHandler
from .inference_handler import InferenceHandler

logger = logging.getLogger(__name__)


class CameraManager:
    """
    Manages multiple Camera instances

    Features:
    - Add/remove cameras by serial number
    - Load/stop recipes across cameras
    - Event aggregation and forwarding
    - Thread-safe operations
    - Delegates trigger and inference logic to handlers
    """

    def __init__(
        self,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        event_loop: Optional[Any] = None
    ):
        """
        Initialize CameraManager

        Args:
            event_callback: Callback for events from cameras (event_type, data)
            event_loop: Event loop for async callbacks from threads
        """
        self.cameras: Dict[str, Camera] = {}
        self.event_callback = event_callback
        self.event_loop = event_loop
        self._lock = threading.RLock()

        # Initialize handlers
        self.trigger_handler = TriggerHandler(self)
        self.inference_handler = InferenceHandler()

        logger.info("CameraManager initialized with handlers")

    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to callback (handles both sync and async callbacks from thread)"""
        if self.event_callback:
            try:
                # If event_loop is set, schedule coroutine in event loop
                if self.event_loop:
                    self.event_loop.call_soon_threadsafe(
                        lambda: self.event_loop.create_task(self.event_callback(event_type, data))
                    )
                else:
                    # Synchronous callback
                    self.event_callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")

    def _camera_event_handler(self, event_type: str, data: Dict[str, Any]):
        """Forward camera events to external callback"""
        self._emit_event(event_type, data)

    def add_camera(
        self,
        serial_number: str,
        pixel_format: str = "BGR8",
        exposure_time: Optional[float] = None,
        gain: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Add and connect a camera by serial number

        Args:
            serial_number: Camera serial number
            pixel_format: Pixel format (BGR8, Mono8, etc)
            exposure_time: Exposure time in microseconds (optional)
            gain: Gain value (optional)

        Returns:
            Result dict with success status
        """
        with self._lock:
            if serial_number in self.cameras:
                return {
                    "success": False,
                    "error": f"Camera {serial_number} already added"
                }

            try:
                # Create camera instance
                camera = Camera(
                    serial_number=serial_number,
                    pixel_format=pixel_format,
                    event_callback=self._camera_event_handler
                )

                # Connect camera
                if not camera.connect():
                    return {
                        "success": False,
                        "error": f"Failed to connect camera {serial_number}"
                    }

                # Apply settings if provided
                if exposure_time is not None:
                    camera.set_exposure_time(exposure_time)

                if gain is not None:
                    camera.set_gain(gain)

                # Add to cameras dict
                self.cameras[serial_number] = camera

                # Note: Camera loop will start automatically when set_mode() is called
                # (e.g., when loading recipe with trigger_mode)

                logger.info(f"Camera {serial_number} added and connected")

                return {
                    "success": True,
                    "serial_number": serial_number,
                    "pixel_format": pixel_format
                }

            except Exception as e:
                logger.error(f"Error adding camera {serial_number}: {e}")
                import traceback
                traceback.print_exc()
                return {
                    "success": False,
                    "error": str(e)
                }

    def remove_camera(self, serial_number: str) -> Dict[str, Any]:
        """
        Disconnect and remove camera

        Args:
            serial_number: Camera serial number

        Returns:
            Result dict with success status
        """
        with self._lock:
            if serial_number not in self.cameras:
                return {
                    "success": False,
                    "error": f"Camera {serial_number} not found"
                }

            try:
                camera = self.cameras[serial_number]

                # Disconnect camera (will stop grabbing automatically)
                camera.disconnect()

                # Remove from dict
                del self.cameras[serial_number]

                logger.info(f"Camera {serial_number} removed")

                return {
                    "success": True,
                    "serial_number": serial_number
                }

            except Exception as e:
                logger.error(f"Error removing camera {serial_number}: {e}")
                import traceback
                traceback.print_exc()
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

                # Initialize inference matcher for first camera (in main thread - has CUDA context)
                if success and loaded_cameras:
                    first_serial = loaded_cameras[0]
                    first_camera = self.cameras[first_serial]

                    if first_camera.templates:
                        logger.info(f"Initializing inference matcher for camera {first_serial}")
                        self.inference_handler.init_matcher(first_camera)
                        if self.inference_handler.inference_matcher:
                            logger.info("✅ Inference matcher initialized successfully")
                        else:
                            logger.warning("⚠️ Failed to initialize inference matcher")
                    else:
                        logger.info("No templates found, skipping inference initialization")

                # Build DI camera map for Software Trigger mode
                self.trigger_handler.build_di_camera_map()

                # Start trigger polling if any cameras use Software Trigger
                self.trigger_handler.start_polling()

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

                # Clear inference matcher when recipe is stopped
                if stopped_cameras:
                    self.inference_handler.clear_matcher()

                # Rebuild DI map (remove stopped cameras)
                self.trigger_handler.build_di_camera_map()

                # Stop trigger polling if no more Software Trigger cameras
                has_software_trigger = any(
                    len(cams) > 0 for cams in self.trigger_handler.di_camera_map.values()
                )
                if not has_software_trigger:
                    self.trigger_handler.stop_polling()
                    logger.info("No more Software Trigger cameras, polling stopped")

                return {
                    "success": True,
                    "recipe_id": recipe_id,
                    "stopped_cameras": stopped_cameras
                }

            except Exception as e:
                logger.error(f"Error stopping recipe: {e}")
                import traceback
                traceback.print_exc()
                return {
                    "success": False,
                    "error": str(e)
                }

    def set_camera_mode(self, serial_number: str, mode: str) -> Dict[str, Any]:
        """
        Set camera operation mode

        Args:
            serial_number: Camera serial number
            mode: Mode string (continuous, software_trigger, hardware_trigger)

        Returns:
            Result dict
        """
        with self._lock:
            if serial_number not in self.cameras:
                return {
                    "success": False,
                    "error": f"Camera {serial_number} not found"
                }

            try:
                camera = self.cameras[serial_number]

                # Map string to CameraMode enum
                mode_map = {
                    "continuous": CameraMode.CONTINUOUS,
                    "software_trigger": CameraMode.SOFTWARE_TRIGGER,
                    "hardware_trigger": CameraMode.HARDWARE_TRIGGER
                }

                if mode not in mode_map:
                    return {
                        "success": False,
                        "error": f"Invalid mode: {mode}"
                    }

                camera.set_mode(mode_map[mode])

                logger.info(f"Camera {serial_number} mode set to {mode}")

                return {
                    "success": True,
                    "serial_number": serial_number,
                    "mode": mode
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
        camera = self.cameras.get(serial_number)
        if not camera:
            return None

        return {
            "serial_number": serial_number,
            "is_connected": camera.is_connected,
            "mode": camera.mode.value if camera.mode else None,
            "recipe_id": camera.recipe_id,
            "recipe_name": camera.recipe_name
        }

    def get_all_cameras_status(self) -> Dict[str, Any]:
        """Get status of all cameras"""
        return {
            "cameras": [
                self.get_camera_status(sn) for sn in self.cameras.keys()
            ],
            "total_cameras": len(self.cameras)
        }

    def update_camera_settings(
        self,
        serial_number: str,
        settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update camera settings (exposure, gain, etc.)

        Args:
            serial_number: Camera serial number
            settings: Settings dict
                {
                    "exposure_time": float (microseconds),
                    "gain": float,
                    "pixel_format": str
                }

        Returns:
            Result dict
        """
        with self._lock:
            if serial_number not in self.cameras:
                return {
                    "success": False,
                    "error": f"Camera {serial_number} not found"
                }

            try:
                camera = self.cameras[serial_number]
                updated = []

                # Update exposure time
                if "exposure_time" in settings:
                    exposure_time = settings["exposure_time"]
                    camera.set_exposure_time(exposure_time)
                    updated.append("exposure_time")

                # Update gain
                if "gain" in settings:
                    gain = settings["gain"]
                    camera.set_gain(gain)
                    updated.append("gain")

                # Update pixel format (requires reconnect)
                if "pixel_format" in settings:
                    # TODO: Implement pixel format change
                    logger.warning("Pixel format change not implemented yet")

                logger.info(f"Camera {serial_number} settings updated: {updated}")

                return {
                    "success": True,
                    "serial_number": serial_number,
                    "updated": updated
                }

            except Exception as e:
                logger.error(f"Error updating camera settings: {e}")
                import traceback
                traceback.print_exc()
                return {
                    "success": False,
                    "error": str(e)
                }

    # Delegate trigger methods to TriggerHandler

    def simulate_trigger(
        self,
        serial_number: str = None,
        trigger_type: str = "rising_edge"
    ) -> Dict[str, Any]:
        """Simulate hardware trigger (delegates to TriggerHandler)"""
        return self.trigger_handler.simulate_trigger(serial_number, trigger_type)

    def simulate_trigger_sequence(
        self,
        serial_number: str = None,
        count: int = 5,
        interval_ms: int = 1000
    ) -> Dict[str, Any]:
        """Simulate trigger sequence (delegates to TriggerHandler)"""
        return self.trigger_handler.simulate_trigger_sequence(serial_number, count, interval_ms)

    # Inference processing (called from event handler in main thread)

    def process_inference(self, cameras, results):
        """
        Process inference on captured frames (runs in main thread)

        Args:
            cameras: List of cameras that captured frames
            results: Capture results from trigger

        Delegates to InferenceHandler with emit callback
        """
        self.inference_handler.process_inference(
            cameras=cameras,
            results=results,
            emit_callback=self._emit_event
        )

    def shutdown(self):
        """Shutdown all cameras and stop polling"""
        with self._lock:
            logger.info("Shutting down CameraManager...")

            # Stop trigger polling first
            self.trigger_handler.stop_polling()

            # Stop all cameras
            serial_numbers = list(self.cameras.keys())
            for serial_number in serial_numbers:
                try:
                    self.remove_camera(serial_number)
                except Exception as e:
                    logger.error(f"Error shutting down camera {serial_number}: {e}")

            # Clear inference matcher
            self.inference_handler.clear_matcher()

            logger.info("CameraManager shutdown complete")
