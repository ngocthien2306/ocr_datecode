"""
Camera Manager Module
Manages multiple camera instances and handles events

Refactored to use separate handlers for better organization:
- TriggerHandler: DI polling and software trigger logic
- InferenceHandler: Inference processing and result building
- PLCController: PLC Modbus TCP control for reject
- utils: Common utility functions
"""

import logging
import os
import threading
from typing import Dict, Any, Optional, Callable
from datetime import datetime

home = os.environ.get('HOME')
from .camera import Camera, CameraMode
from .trigger_handler import TriggerHandler
from .inference_handler import InferenceHandler
from .reject_scheduler import RejectScheduler

logger = logging.getLogger(__name__)

# Import PLC Controller
try:
    from .plc_controller import PLCController
    PLC_AVAILABLE = True
except ImportError:
    logger.warning("PLCController not available, PLC mode disabled")
    PLC_AVAILABLE = False


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

        # Inference mode flag (ONLINE/OFFLINE)
        self.inference_enabled = True

        # Initialize PLC Controller
        self.plc_controller = None
        if PLC_AVAILABLE:
            try:
                self.plc_controller = PLCController()
                logger.info("PLCController initialized")
            except Exception as e:
                logger.error(f"Failed to initialize PLCController: {e}")

        # Initialize handlers
        self.trigger_handler = TriggerHandler(self)
        self.reject_scheduler = RejectScheduler(plc_controller=self.plc_controller)
        self.inference_handler = InferenceHandler(
            reject_scheduler=self.reject_scheduler,
            trigger_handler=self.trigger_handler
        )

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

                camera.set_mode(CameraMode.CONTINUOUS)

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

                # Log recipe data for debugging
                logger.info(f"📥 Loading recipe: {recipe_name} (ID: {recipe_id})")
                logger.info(f"📋 Recipe cameras config:")
                for idx, cam in enumerate(recipe_data.get("cameras", [])):
                    logger.info(f"  Camera {idx + 1}:")
                    logger.info(f"    - camera_id: {cam.get('camera_id')}")
                    logger.info(f"    - serial_number: {cam.get('serial_number')}")
                    logger.info(f"    - function_type: {cam.get('function_type', 'NOT SET')} ⭐")
                    logger.info(f"    - trigger_mode: {cam.get('trigger_mode')}")

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

                # Initialize inference matchers for all cameras (in main thread - has CUDA context)
                if success and loaded_cameras:
                    cameras_with_recipe = [self.cameras[sn] for sn in loaded_cameras]

                    logger.info(f"Initializing inference matchers for {len(cameras_with_recipe)} cameras")
                    num_initialized = self.inference_handler.init_matchers(cameras_with_recipe)

                    if num_initialized > 0:
                        logger.info(f"✅ {num_initialized} inference matchers initialized successfully")
                    else:
                        logger.warning("⚠️ Failed to initialize inference matchers")

                # Update normal_pulse_ms from recipe (for stuck bottle detection)
                normal_pulse_ms = recipe_data.get('normal_pulse_ms', 250.0) or 250.0
                self.trigger_handler.normal_pulse_ms = float(normal_pulse_ms)
                logger.info(f"normal_pulse_ms set to {self.trigger_handler.normal_pulse_ms} ms from recipe")

                # Build DI camera map for Software Trigger mode
                self.trigger_handler.build_di_camera_map()

                # Start trigger polling if any cameras use Software Trigger
                self.trigger_handler.start_polling()

                # Update reject configuration from recipe
                reject_pulse = recipe_data.get("reject_pulse", 50.0)
                reject_method = recipe_data.get("reject_method", "DIO").upper()

                logger.info(f"Reject config: method={reject_method}, pulse={reject_pulse}ms")

                # Connect PLC if using PLC mode
                if reject_method == "PLC" and self.plc_controller:
                    logger.info("Connecting to PLC...")
                    if self.plc_controller.connect():
                        logger.info("✅ PLC connected successfully")
                    else:
                        logger.warning("⚠️ PLC connection failed, will fallback to DIO")

                # Update reject scheduler config
                self.reject_scheduler.set_reject_config(reject_pulse, reject_method)

                # Start reject scheduler (if not already running)
                self.reject_scheduler.start()

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

                # Clear inference matchers when recipe is stopped
                if stopped_cameras:
                    self.inference_handler.clear_matchers()

                # Rebuild DI map (remove stopped cameras)
                self.trigger_handler.build_di_camera_map()

                # Stop trigger polling if no more Software Trigger cameras
                has_software_trigger = any(
                    len(cams) > 0 for cams in self.trigger_handler.di_camera_map.values()
                )
                if not has_software_trigger:
                    self.trigger_handler.stop_polling()
                    logger.info("No more Software Trigger cameras, polling stopped")

                # Disconnect PLC if connected
                if self.plc_controller and self.plc_controller.is_connected():
                    logger.info("Disconnecting from PLC...")
                    self.plc_controller.disconnect()
                    logger.info("PLC disconnected")

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

                # Update delay trigger
                if "delay_trigger" in settings:
                    delay_ms = settings["delay_trigger"]
                    camera.delay_trigger = delay_ms
                    updated.append("delay_trigger")
                    logger.info(f"Camera {serial_number} delay_trigger updated to {delay_ms}ms")

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
        # Check if inference is enabled (ONLINE mode)
        if not self.inference_enabled:
            logger.info("⏸️ [INFERENCE OFFLINE] Trigger detected but inference skipped")
            # Emit event to notify frontend
            self._emit_event("inference_skipped", {
                "reason": "Inference mode is OFFLINE",
                "cameras": [cam.serial_number for cam in cameras],
                "timestamp": str(datetime.now())
            })
            return

        # Submit async inference job (non-blocking)
        # Extract metadata from results if available
        group_id = results.get('group_id') if isinstance(results, dict) and 'group_id' in results else None
        T_capture_complete = results.get('T_capture_complete') if isinstance(results, dict) else None
        stuck_count = results.get('stuck_count', 1) if isinstance(results, dict) else 1
        pulse_width_ms = results.get('pulse_width_ms', 0.0) if isinstance(results, dict) else 0.0

        job_id = self.inference_handler.process_inference_async(
            cameras=cameras,
            results=results,
            emit_callback=self._emit_event,
            group_id=group_id,
            T_capture_complete=T_capture_complete,
            stuck_count=stuck_count,
            pulse_width_ms=pulse_width_ms
        )

        logger.debug(f"Submitted inference job #{job_id} for {len(cameras)} camera(s)")

    def handle_stuck_capture(
        self,
        cameras: list,
        results: dict,
        group_id: int,
        T_capture_complete: float,
        stuck_count: int,
        pulse_width_ms: float
    ):
        """
        Build and emit a properly-formatted auto-FAIL inference_result for stuck
        bottles (same structure as normal inference), then schedule N reject pulses.

        Args:
            cameras: List of Camera objects
            results: Capture results {serial: {'frames': [np.ndarray, ...]}}
            group_id: Trigger group ID
            T_capture_complete: Timestamp when primary capture completed
            stuck_count: Number of stuck bottles (N)
            pulse_width_ms: Total DI pulse width in ms
        """
        from .result_builder.builder import InferenceResultBuilder, CameraResultBuilder, FrameResultBuilder
        from .utils import save_and_encode_frame, encode_frame_for_display

        normal_pulse_ms = self.trigger_handler.normal_pulse_ms
        ratio = pulse_width_ms / normal_pulse_ms
        fail_reason = (
            f"STUCK_BOTTLES: {stuck_count} chai dính nhau "
            f"(pulse={pulse_width_ms:.0f}ms, ratio={ratio:.2f}x, normal={normal_pulse_ms:.0f}ms)"
        )

        if not cameras:
            logger.error(f"[Group #{group_id}] handle_stuck_capture: no cameras, abort")
            return

        first_camera = cameras[0]
        recipe_id = first_camera.recipe_id
        recipe_name = first_camera.recipe_name
        base_dir = f"{home}/Source/ocr_datecode/backend/uploads/inference_results"

        builder = InferenceResultBuilder(
            recipe_id=recipe_id,
            recipe_name=recipe_name
        ).with_overall_result("FAIL")

        for camera in cameras:
            serial = camera.serial_number
            cam_frames = results.get(serial, {}).get('frames', [])

            camera_builder = CameraResultBuilder(
                serial_number=serial,
                delay_trigger=camera.delay_trigger
            )

            for frame_idx, frame_img in enumerate(cam_frames):
                frame_builder = FrameResultBuilder(
                    frame_idx=frame_idx,
                    template_name=f"Stuck Bottle {frame_idx + 1}"
                ).with_inference_data("FAIL", 0.0)

                try:
                    image_path, image_base64 = save_and_encode_frame(
                        frame_img=frame_img,
                        serial_number=serial,
                        recipe_id=recipe_id,
                        pass_fail="FAIL",
                        frame_idx=frame_idx,
                        base_dir=base_dir,
                    )
                    frame_builder.set_encoded_image(image_path, image_base64)
                except Exception as e:
                    logger.error(
                        f"[Group #{group_id}] Failed to encode stuck frame "
                        f"camera={serial} idx={frame_idx}: {e}"
                    )

                camera_builder.add_frame(frame_builder.build())

            camera_result = camera_builder.build()
            builder.add_camera_result(camera_result)
            frames = camera_result.frames
            builder.add_camera_stats(
                serial_number=serial,
                avg_confidence=0.0,
                total_inliers=0,
                total_matches=0,
                timings={},
                pass_count=0,
                fail_count=len(frames),
                error_count=0
            )

        result = builder.build()
        # Attach stuck-specific info into metadata so it flows through to DB / frontend
        result['group_id'] = group_id
        result.setdefault('metadata', {})['stuck_info'] = {
            'fail_reason': fail_reason,
            'stuck_count': stuck_count,
            'pulse_width_ms': pulse_width_ms,
        }

        self._emit_event("inference_result", result)

        # Schedule N reject pulses (one per stuck bottle)
        self.schedule_stuck_reject(
            group_id=group_id,
            T_capture_complete=T_capture_complete,
            stuck_count=stuck_count,
            pulse_width_ms=pulse_width_ms,
            cameras=cameras,
        )

    def schedule_stuck_reject(
        self,
        group_id: int,
        T_capture_complete: float,
        stuck_count: int,
        pulse_width_ms: float,
        cameras: list
    ):
        """
        Schedule N reject pulses for N stuck bottles (bypasses inference).

        Bottle i arrives at reject station delta_t seconds after bottle 0,
        where delta_t = pulse_width_ms / stuck_count / 1000.

        Args:
            group_id: Original group ID (virtual IDs = group_id*1000 + i)
            T_capture_complete: Timestamp when primary capture completed
            stuck_count: Number of stuck bottles (N)
            pulse_width_ms: Total DI pulse width in ms
            cameras: Camera list (used to get delay_reject / do_number config)
        """
        if not cameras or not self.reject_scheduler or not T_capture_complete:
            logger.error(f"[Group #{group_id}] schedule_stuck_reject: missing params, abort")
            return

        camera = cameras[0]
        delay_reject = camera.delay_reject        # ms
        do_reject_number = camera.do_reject_number
        do_alarm_number = camera.do_alarm_number
        delta_t_sec = (pulse_width_ms / stuck_count) / 1000.0  # seconds per bottle

        logger.warning(
            f"[Group #{group_id}] Scheduling {stuck_count} stuck-bottle rejects "
            f"(delta_t={delta_t_sec*1000:.0f}ms, DO{do_reject_number})"
        )

        for i in range(stuck_count):
            fake_T_capture = T_capture_complete + i * delta_t_sec
            bottle_group_id = group_id * 1000 + i
            success = self.reject_scheduler.schedule_reject(
                group_id=bottle_group_id,
                T_capture_complete=fake_T_capture,
                inference_time=0.0,          # no inference for stuck
                delay_reject=delay_reject,
                do_number=do_reject_number,
                alarm_number=do_alarm_number if i == 0 else -1  # alarm once only
            )
            if success:
                logger.info(
                    f"[Group #{group_id}] Stuck reject {i+1}/{stuck_count} scheduled "
                    f"(+{i * delta_t_sec * 1000:.0f}ms offset, vgroup={bottle_group_id})"
                )
            else:
                logger.error(
                    f"[Group #{group_id}] Stuck reject {i+1}/{stuck_count} FAILED to schedule"
                )

    def set_inference_mode(self, enabled: bool) -> Dict[str, Any]:
        """
        Set inference mode (ONLINE/OFFLINE)
        Also sets DO2 output: ONLINE=1, OFFLINE=0

        Args:
            enabled: True for ONLINE, False for OFFLINE

        Returns:
            Result dict
        """
        with self._lock:
            self.inference_enabled = enabled
            mode_name = "ONLINE" if enabled else "OFFLINE"

            # Set DO2: ONLINE=1, OFFLINE=0
            from .utils import write_do_value
            do_value = 0 if enabled else 1
            do_success = write_do_value(2, do_value)  # DO2

            if do_success:
                logger.info(f"🔄 [INFERENCE MODE] Set to {mode_name}, DO2={do_value}")
            else:
                logger.warning(f"🔄 [INFERENCE MODE] Set to {mode_name}, but failed to set DO2={do_value}")

            return {
                "success": True,
                "enabled": enabled,
                "mode": mode_name,
                "do2_value": do_value,
                "do2_success": do_success,
                "message": f"Inference mode set to {mode_name}"
            }

    def get_reject_stats(self) -> Dict[str, Any]:
        """
        Get reject scheduler statistics including PLC stats

        Returns:
            Combined statistics from reject scheduler and PLC
        """
        stats = {
            "reject_scheduler": self.reject_scheduler.get_stats(),
            "plc": None
        }

        # Add PLC stats if available
        if self.plc_controller:
            try:
                stats["plc"] = self.plc_controller.get_stats()
            except Exception as e:
                logger.error(f"Error getting PLC stats: {e}")
                stats["plc"] = {"error": str(e)}

        return stats

    def shutdown(self):
        """Shutdown all cameras and stop polling"""
        with self._lock:
            logger.info("Shutting down CameraManager...")

            # Stop trigger polling first
            self.trigger_handler.stop_polling()

            # Stop reject scheduler
            self.reject_scheduler.stop()

            # Disconnect PLC if connected
            if self.plc_controller and self.plc_controller.is_connected():
                logger.info("Disconnecting from PLC during shutdown...")
                self.plc_controller.disconnect()

            # Stop all cameras
            serial_numbers = list(self.cameras.keys())
            for serial_number in serial_numbers:
                try:
                    self.remove_camera(serial_number)
                except Exception as e:
                    logger.error(f"Error shutting down camera {serial_number}: {e}")

            # Shutdown inference handler (waits for pending jobs)
            self.inference_handler.shutdown()

            logger.info("CameraManager shutdown complete")
