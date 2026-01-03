"""
Camera Manager Module
Manages multiple camera instances and handles events
"""

import logging
import threading
import subprocess
import time
import cv2
from typing import Dict, Any, Optional, Callable, List, Tuple
from pathlib import Path
from .camera import Camera, CameraMode
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import inference service
try:
    import sys
    ai_services_path = Path(__file__).parent.parent
    if str(ai_services_path) not in sys.path:
        sys.path.insert(0, str(ai_services_path))
    from inference_service import SuperPointMatcherTRT
    INFERENCE_AVAILABLE = True
except Exception as e:
    logging.warning(f"Inference service not available: {e}")
    INFERENCE_AVAILABLE = False
    SuperPointMatcherTRT = None

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

        # DI Polling for Software Trigger (centralized)
        self._trigger_polling = False
        self._trigger_thread: Optional[threading.Thread] = None
        self._trigger_lock = threading.Lock()

        # DI → Cameras mapping: {di_number: [(camera, trigger_activation), ...]}
        # Only cameras with trigger_mode="software_trigger" will be in this map
        self.di_camera_map: Dict[int, List[Tuple[Camera, str]]] = {0: [], 1: [], 2: [], 3: []}

        # Previous DI values for edge detection
        self.previous_di_values: Dict[int, Optional[int]] = {0: None, 1: None, 2: None, 3: None}

        # Inference matcher (Phase 1: single matcher for first camera)
        self.inference_matcher: Optional[Any] = None

        logger.info("CameraManager initialized")

    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to callback (handles both sync and async callbacks from thread)"""
        if self.event_callback:
            try:
                import asyncio
                import inspect

                # Check if callback is async
                if inspect.iscoroutinefunction(self.event_callback):
                    # Async callback - need to schedule in event loop
                    if self.event_loop:
                        # Use provided event loop (from main thread)
                        asyncio.run_coroutine_threadsafe(
                            self.event_callback(event_type, data),
                            self.event_loop
                        )
                    else:
                        logger.warning(f"Async callback but no event loop provided for {event_type}")
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
        pixel_format: str = "BGR8"
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
                # Get event loop for async callbacks from camera thread
                import asyncio
                try:
                    event_loop = asyncio.get_running_loop()
                except RuntimeError:
                    event_loop = None

                # Create camera instance
                camera = Camera(
                    serial_number=serial_number,
                    pixel_format=pixel_format,
                    event_callback=self._camera_event_handler,
                    event_loop=event_loop
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

                # Initialize inference matcher for first camera (in main thread - has CUDA context)
                if success and loaded_cameras:
                    first_serial = loaded_cameras[0]
                    first_camera = self.cameras[first_serial]

                    if INFERENCE_AVAILABLE and SuperPointMatcherTRT and first_camera.templates:
                        logger.info(f"Initializing inference matcher for camera {first_serial}")
                        self.inference_matcher = self._init_inference_matcher(first_camera)
                        if self.inference_matcher:
                            logger.info("✅ Inference matcher initialized successfully")
                        else:
                            logger.warning("⚠️ Failed to initialize inference matcher")
                    else:
                        logger.info("Inference not available or no templates found")

                # Build DI camera map for Software Trigger mode
                self._build_di_camera_map()

                # Start trigger polling if any cameras use Software Trigger
                self.start_trigger_polling()

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
                    self.inference_matcher = None
                    logger.info("Inference matcher cleared")

                # Rebuild DI map (remove stopped cameras)
                self._build_di_camera_map()

                # Stop trigger polling if no more Software Trigger cameras
                has_software_trigger = any(len(cams) > 0 for cams in self.di_camera_map.values())
                if not has_software_trigger:
                    self.stop_trigger_polling()
                    logger.info("No more Software Trigger cameras, polling stopped")

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
        Simulate software trigger without physical DI pin

        Args:
            serial_number: Camera serial (if None, trigger all cameras in software_trigger mode)
            trigger_type: Type of trigger edge (not used in simulation)

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
                    if camera.mode == CameraMode.SOFTWARE_TRIGGER:
                        triggered_cameras.append(camera)
                        logger.info(f"Simulating trigger for camera {serial_number}")
                    else:
                        return {
                            "success": False,
                            "error": f"Camera {serial_number} not in software_trigger mode (current: {camera.mode.value})"
                        }
                else:
                    # Trigger all cameras in software_trigger mode
                    for sn, camera in self.cameras.items():
                        if camera.mode == CameraMode.SOFTWARE_TRIGGER:
                            triggered_cameras.append(camera)
                            logger.info(f"Simulating trigger for camera {sn}")

                if not triggered_cameras:
                    return {
                        "success": False,
                        "error": "No cameras in software_trigger mode"
                    }

                # Trigger cameras group
                self.trigger_cameras_group(triggered_cameras)

                return {
                    "success": True,
                    "triggered_cameras": [cam.serial_number for cam in triggered_cameras],
                    "trigger_type": "simulated"
                }

            except Exception as e:
                logger.error(f"Error simulating trigger: {e}")
                import traceback
                traceback.print_exc()
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

    def _read_di_value(self, di_number: int) -> int:
        """
        Read Digital Input pin value (0 or 1)

        Args:
            di_number: DI pin number (0-3)

        Returns:
            Pin value (0 or 1), or 0 on error
        """
        try:
            result = subprocess.run(
                ["sudo", "dio_in", str(di_number)],
                capture_output=True,
                text=True,
                timeout=0.5
            )

            if result.returncode == 0:
                # Parse output format: "The id-X input gpio status = Y\nCompletion code = 0x00"
                output = result.stdout.strip()

                # Extract value from "status = Y" line
                for line in output.split('\n'):
                    if 'status' in line and '=' in line:
                        # Extract the value after '='
                        value_str = line.split('=')[-1].strip()
                        try:
                            value = int(value_str)
                            return value
                        except ValueError:
                            logger.warning(f"Failed to parse DI{di_number} value: {value_str}")
                            return 0

                # Fallback: couldn't find status line
                logger.warning(f"Unexpected DI{di_number} output format: {output}")
                return 0
            else:
                logger.warning(f"Failed to read DI{di_number}: {result.stderr.strip()}")
                return 0

        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout reading DI{di_number}")
            return 0
        except Exception as e:
            logger.error(f"Error reading DI{di_number}: {e}")
            return 0

    def _check_trigger_edge(self, current: int, previous: Optional[int], activation: str) -> bool:
        """
        Check if current value matches trigger activation edge

        Args:
            current: Current DI value (0 or 1)
            previous: Previous DI value (0, 1, or None)
            activation: Trigger activation type (RisingEdge, FallingEdge, AnyEdge)

        Returns:
            True if edge detected, False otherwise
        """
        if previous is None:
            return False

        if activation == "RisingEdge":
            return previous == 0 and current == 1
        elif activation == "FallingEdge":
            return previous == 1 and current == 0
        elif activation == "AnyEdge":
            return previous != current
        else:
            logger.warning(f"Unknown trigger activation: {activation}")
            return False

    def _build_di_camera_map(self):
        """
        Build DI → Cameras mapping from cameras with Software Trigger mode

        Called when recipe is loaded to update the mapping
        """
        with self._trigger_lock:
            # Clear existing map
            self.di_camera_map = {0: [], 1: [], 2: [], 3: []}

            # Build new map
            for serial_number, camera in self.cameras.items():
                if camera.trigger_mode == "software_trigger" and camera.mode == CameraMode.SOFTWARE_TRIGGER:
                    di_num = camera.di_number
                    activation = camera.trigger_activation

                    if 0 <= di_num <= 3:
                        self.di_camera_map[di_num].append((camera, activation))
                        logger.debug(f"Mapped DI{di_num} → Camera {serial_number} ({activation})")
                    else:
                        logger.warning(f"Invalid DI number {di_num} for camera {serial_number}")

            logger.info(f"DI camera map built: {[(di, len(cams)) for di, cams in self.di_camera_map.items() if cams]}")

    def start_trigger_polling(self):
        """Start DI polling thread for Software Trigger mode"""
        with self._trigger_lock:
            if self._trigger_polling:
                logger.warning("Trigger polling already running")
                return

            # Check if any cameras need polling
            has_software_trigger = any(len(cams) > 0 for cams in self.di_camera_map.values())

            if not has_software_trigger:
                logger.info("No cameras with Software Trigger, skipping polling")
                return

            # Reset previous values
            self.previous_di_values = {0: None, 1: None, 2: None, 3: None}

            # Start polling thread
            self._trigger_polling = True
            self._trigger_thread = threading.Thread(
                target=self._trigger_polling_loop,
                daemon=True,
                name="CameraManager-TriggerPolling"
            )
            self._trigger_thread.start()

            logger.info("Trigger polling started (100Hz)")

    def stop_trigger_polling(self):
        """Stop DI polling thread"""
        with self._trigger_lock:
            if not self._trigger_polling:
                return

            self._trigger_polling = False

            if self._trigger_thread:
                self._trigger_thread.join(timeout=2.0)
                self._trigger_thread = None

            logger.info("Trigger polling stopped")

    def _trigger_polling_loop(self):
        """
        Main DI polling loop - runs at 100Hz

        Polls only the DI pins that are used by cameras in Software Trigger mode
        When edge is detected, triggers all cameras mapped to that DI pin
        """
        logger.info("Trigger polling loop started")

        poll_interval = 0.01  # 100Hz = 10ms

        try:
            while self._trigger_polling:
                loop_start = time.time()

                # Poll each DI that has cameras mapped to it
                for di_number, camera_list in self.di_camera_map.items():
                    if not camera_list:
                        continue  # Skip DI with no cameras

                    # Read current value
                    current_value = self._read_di_value(di_number)
                    previous_value = self.previous_di_values[di_number]

                    # Check each camera's trigger activation
                    triggered_cameras = []
                    for camera, activation in camera_list:
                        if self._check_trigger_edge(current_value, previous_value, activation):
                            triggered_cameras.append(camera)

                    # Update previous value
                    self.previous_di_values[di_number] = current_value

                    # If any camera triggered, trigger the group
                    if triggered_cameras:
                        logger.info(f"DI{di_number} edge detected ({previous_value}→{current_value}), triggering {len(triggered_cameras)} camera(s)")
                        self.trigger_cameras_group(triggered_cameras)

                # Sleep to maintain 100Hz
                elapsed = time.time() - loop_start
                sleep_time = max(0, poll_interval - elapsed)
                time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Error in trigger polling loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            logger.info("Trigger polling loop stopped")

    def trigger_cameras_group(self, cameras: List[Camera]):
        """
        Trigger a group of cameras simultaneously (runs in polling thread)

        Args:
            cameras: List of Camera objects to trigger

        Flow:
            1. Trigger all cameras in parallel (ThreadPoolExecutor)
            2. Wait for all to complete capture
            3. If any fails → emit error event
            4. If all succeed → emit frames_captured event (inference runs in main thread)
        """
        if not cameras:
            logger.warning("trigger_cameras_group called with empty camera list")
            return

        logger.info(f"Triggering {len(cameras)} camera(s) simultaneously")

        # Step 1: Trigger all cameras in parallel
        results = {}

        with ThreadPoolExecutor(max_workers=len(cameras)) as executor:
            # Submit all camera triggers
            future_to_camera = {
                executor.submit(camera.execute_software_trigger): camera
                for camera in cameras
            }

            # Wait for all to complete
            for future in as_completed(future_to_camera):
                camera = future_to_camera[future]
                try:
                    result = future.result(timeout=10.0)  # 10s timeout per camera
                    results[camera.serial_number] = result
                except Exception as e:
                    logger.error(f"Exception triggering camera {camera.serial_number}: {e}")
                    results[camera.serial_number] = {
                        'success': False,
                        'error': f'Exception: {str(e)}'
                    }

        # Step 2: Check if all cameras succeeded
        failed_cameras = [sn for sn, res in results.items() if not res.get('success', False)]

        if failed_cameras:
            # Emit error, don't run inference
            error_msg = f"Camera capture failed: {', '.join(failed_cameras)}"
            logger.error(error_msg)

            self._emit_event("trigger_error", {
                "error": error_msg,
                "failed_cameras": failed_cameras,
                "results": results
            })
            return

        # Step 3: All cameras succeeded - emit event for inference in main thread
        logger.info(f"All {len(cameras)} cameras captured successfully")

        # Emit frames_captured event - inference will be handled in main thread
        self._emit_event("frames_captured", {
            "cameras": cameras,
            "results": results
        })

    def process_inference(self, cameras: List[Camera], results: Dict[str, Any]):
        """
        Process inference on captured frames (runs in main thread - has CUDA context)

        Args:
            cameras: List of cameras that captured frames
            results: Capture results from trigger_cameras_group

        This method is called from the event loop in main thread,
        where CUDA context is available for TensorRT inference.
        """
        try:
            # Use first camera's first frame only (Phase 1)
            first_camera = cameras[0]
            first_frame = results[first_camera.serial_number]['frames'][0]

            logger.info(f"Running inference on camera {first_camera.serial_number} frame 0")

            # Run inference using pre-initialized matcher
            inference_result_data = "PASS"  # Default
            confidence = 0.0
            inliers = 0
            total_matches = 0

            if self.inference_matcher:
                try:
                    # Run matcher inference using match_array (in main thread - CUDA context OK)
                    match_result = self.inference_matcher.match_array(
                        target_img_array=first_frame,
                        score_threshold=0.3,
                        ransac_threshold=5.0
                    )

                    # Check if matching succeeded
                    if match_result.get('success', False):
                        confidence = match_result.get('confidence', 0.0)
                        inliers = match_result.get('inliers', 0)
                        total_matches = match_result.get('total_matches', 0)

                        # Simple pass/fail: confidence > 0.5 and enough inliers
                        if confidence > 0.5 and inliers >= 10:
                            inference_result_data = "PASS"
                        else:
                            inference_result_data = "FAIL"
                    else:
                        inference_result_data = "FAIL"

                    logger.info(
                        f"Inference result: {inference_result_data}, "
                        f"confidence: {confidence:.2%}, "
                        f"inliers: {inliers}/{total_matches}"
                    )
                except Exception as e:
                    logger.error(f"Error running inference: {e}")
                    import traceback
                    traceback.print_exc()
                    inference_result_data = "ERROR"
            else:
                logger.warning("Inference matcher not available, skipping inference")

            # Build camera_results structure for backend
            camera_results = []
            for camera in cameras:
                camera_frames = results[camera.serial_number]['frames']

                # Build frame results (Phase 1: only process first frame of first camera)
                frame_results = []
                for idx, _frame in enumerate(camera_frames):
                    # For first camera first frame, use real inference result
                    if camera == first_camera and idx == 0:
                        frame_result = {
                            "template_name": camera.templates[idx].get('name', f'Template {idx+1}') if idx < len(camera.templates) else f'Template {idx+1}',
                            "frame_idx": idx,
                            "pass_fail": inference_result_data,
                            "confidence": confidence,
                            "detected_regions": None  # TODO: Add detected regions from match_result
                        }
                    else:
                        # For other frames/cameras, placeholder (will be implemented later)
                        frame_result = {
                            "template_name": camera.templates[idx].get('name', f'Template {idx+1}') if idx < len(camera.templates) else f'Template {idx+1}',
                            "frame_idx": idx,
                            "pass_fail": "PASS",  # Placeholder
                            "confidence": 0.0,
                            "detected_regions": None
                        }

                    frame_results.append(frame_result)

                # Build camera result
                camera_result = {
                    "camera_id": camera.serial_number,  # TODO: Use actual camera_id from DB
                    "serial_number": camera.serial_number,
                    "frames": frame_results
                }
                camera_results.append(camera_result)

            # Build inference result with correct structure for backend
            inference_result = {
                "recipe_id": first_camera.recipe_id,
                "recipe_name": first_camera.recipe_name,
                "product_pass_fail": inference_result_data,  # Overall result
                "camera_results": camera_results,
                "metadata": {
                    "total_cameras": len(cameras),
                    "total_frames": sum(len(r['frames']) for r in results.values()),
                    "inference_stats": {
                        "confidence": confidence,
                        "inliers": inliers,
                        "total_matches": total_matches
                    }
                }
            }

            # Emit inference result to backend
            self._emit_event("inference_result", inference_result)

            logger.info(f"Inference complete: {inference_result['product_pass_fail']}")

        except Exception as e:
            logger.error(f"Error in process_inference: {e}")
            import traceback
            traceback.print_exc()

    def _init_inference_matcher(self, camera: Camera):
        """
        Initialize inference matcher using first camera's templates

        Args:
            camera: Camera object with templates loaded

        Returns:
            SuperPointMatcherTRT instance or None
        """
        try:
            if not camera.templates:
                logger.error(f"No templates found for camera {camera.serial_number}")
                return None

            # Get first template
            template_data = camera.templates[0]
            image_url = template_data.get("image_url")

            if not image_url:
                logger.error("No template image URL")
                return None

            # Copy template to temp directory
            filename = image_url.split("/")[-1]
            backend_dir = Path(__file__).parent.parent.parent / "backend"
            source_path = backend_dir / "uploads" / "templates" / filename

            if not source_path.exists():
                logger.error(f"Template not found: {source_path}")
                return None

            temp_dir = Path("ocr_inference")
            temp_dir.mkdir(exist_ok=True)
            template_path = temp_dir / f"template_{camera.serial_number}.jpg"

            import shutil
            shutil.copy(source_path, template_path)

            # Parse annotations
            annotations = template_data.get("annotations", [])
            template_bbox = None
            other_bboxes = []

            template_img = cv2.imread(str(template_path))
            img_h, img_w = template_img.shape[:2]

            for ann in annotations:
                ann_type = ann.get("type", "")

                if ann_type == "template":
                    x, y = ann.get("x", 0), ann.get("y", 0)
                    w, h = ann.get("width", 0), ann.get("height", 0)

                    x1, y1 = int(x * img_w), int(y * img_h)
                    x2, y2 = int((x + w) * img_w), int((y + h) * img_h)

                    template_bbox = {
                        "type": "template",
                        "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    }

                elif ann_type == "text" and ann.get("points"):
                    pixel_points = [
                        [int(pt[0] * img_w), int(pt[1] * img_h)]
                        for pt in ann.get("points", [])
                    ]
                    other_bboxes.append({
                        "type": ann_type,
                        "text": ann.get("text", ""),
                        "points": pixel_points
                    })

            if not template_bbox:
                logger.error("No template bbox in annotations")
                return None

            # Create annotation file
            import json
            ann_json_path = temp_dir / f"annotations_manager.json"
            ann_data = {
                "_template_image": str(template_path),
                str(template_path): [template_bbox] + other_bboxes
            }

            with open(ann_json_path, "w") as f:
                json.dump(ann_data, f, indent=2)

            # Initialize matcher
            engine_path = "/home/demo/Source/ocr_datecode/weights/pipeline_fp16_small.engine"
            matcher = SuperPointMatcherTRT(
                json_path=str(ann_json_path),
                engine_path=engine_path,
                scale=1.0,
                verbose=True
            )

            logger.info(f"Inference matcher initialized in CameraManager for camera {camera.serial_number}")
            return matcher

        except Exception as e:
            logger.error(f"Error initializing inference matcher: {e}")
            import traceback
            traceback.print_exc()
            return None

    def shutdown(self):
        """Shutdown all cameras and stop trigger polling"""
        with self._lock:
            logger.info("Shutting down CameraManager...")

            # Stop trigger polling first
            self.stop_trigger_polling()

            serial_numbers = list(self.cameras.keys())
            for serial_number in serial_numbers:
                try:
                    self.remove_camera(serial_number)
                except Exception as e:
                    logger.error(f"Error shutting down camera {serial_number}: {e}")

            logger.info("CameraManager shutdown complete")
