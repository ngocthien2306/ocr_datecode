"""
Trigger Handler Module
Handles DI polling and software trigger logic
"""

import logging
import threading
import time
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed

from .utils import read_di_value, check_trigger_edge

if TYPE_CHECKING:
    from .camera import Camera

logger = logging.getLogger(__name__)


class TriggerHandler:
    """
    Handles Digital Input polling and software trigger execution

    Features:
    - 100Hz DI polling
    - Edge detection (Rising/Falling/Any)
    - Multi-camera simultaneous triggering
    - Trigger simulation for testing
    """

    def __init__(self, camera_manager):
        """
        Initialize TriggerHandler

        Args:
            camera_manager: Reference to CameraManager instance
        """
        self.camera_manager = camera_manager
        self._polling = False
        self._polling_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # DI → Cameras mapping: {di_number: [(camera, trigger_activation), ...]}
        self.di_camera_map: Dict[int, List[Tuple['Camera', str]]] = {
            0: [], 1: [], 2: [], 3: []
        }

        # Previous DI values for edge detection
        self.previous_di_values: Dict[int, Optional[int]] = {
            0: None, 1: None, 2: None, 3: None
        }

    def build_di_camera_map(self):
        """
        Build DI → Cameras mapping from cameras with Software Trigger mode

        Only cameras with trigger_mode="software_trigger" are included.
        Maps DI pin → list of (camera, trigger_activation) tuples
        """
        # Clear existing map
        for di_num in self.di_camera_map:
            self.di_camera_map[di_num] = []

        # Scan all cameras for software trigger mode
        for serial_number, camera in self.camera_manager.cameras.items():
            if camera.mode.value == "software_trigger":
                di_number = camera.di_number
                trigger_activation = camera.trigger_activation

                if 0 <= di_number <= 3:
                    self.di_camera_map[di_number].append((camera, trigger_activation))
                    logger.debug(
                        f"Mapped DI{di_number} → Camera {serial_number} "
                        f"(activation: {trigger_activation})"
                    )

        # Log summary
        for di_num, cameras in self.di_camera_map.items():
            if cameras:
                camera_serials = [cam.serial_number for cam, _ in cameras]
                logger.info(f"DI{di_num} → {len(cameras)} camera(s): {camera_serials}")

    def start_polling(self):
        """Start DI polling thread (100Hz)"""
        with self._lock:
            if self._polling:
                logger.warning("Trigger polling already running")
                return

            # Check if any cameras need polling
            has_cameras = any(len(cams) > 0 for cams in self.di_camera_map.values())

            if not has_cameras:
                logger.info("No cameras with Software Trigger mode, skipping polling")
                return

            self._polling = True
            self._polling_thread = threading.Thread(
                target=self._polling_loop,
                daemon=True,
                name="DIPollingThread"
            )
            self._polling_thread.start()

            logger.info("Trigger polling started (100Hz)")

    def stop_polling(self):
        """Stop DI polling thread"""
        with self._lock:
            if not self._polling:
                return

            self._polling = False

            if self._polling_thread:
                self._polling_thread.join(timeout=2.0)
                self._polling_thread = None

            logger.info("Trigger polling stopped")

    def _polling_loop(self):
        """Main polling loop - runs at 100Hz"""
        poll_interval = 0.01  # 10ms = 100Hz

        logger.info("DI polling loop started")

        try:
            while self._polling:
                loop_start = time.time()

                # Poll all DI pins that have cameras mapped
                for di_number, cameras in self.di_camera_map.items():
                    if not cameras:
                        continue

                    # Read current DI value
                    current_value = read_di_value(di_number)

                    # Get previous value
                    previous_value = self.previous_di_values[di_number]

                    # Check if any camera's trigger activation matches
                    # (all cameras on same DI should have same activation, but check first one)
                    if cameras:
                        _, trigger_activation = cameras[0]
                        if check_trigger_edge(current_value, previous_value, trigger_activation):
                            # Edge detected!
                            logger.info(
                                f"DI{di_number} edge detected ({previous_value}→{current_value}), "
                                f"triggering {len(cameras)} camera(s)"
                            )

                            # Trigger all cameras on this DI pin simultaneously
                            camera_list = [cam for cam, _ in cameras]
                            self.trigger_cameras_group(camera_list)

                    # Update previous value
                    self.previous_di_values[di_number] = current_value

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

    def trigger_cameras_group(self, cameras: List['Camera']):
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

            self.camera_manager._emit_event("trigger_error", {
                "error": error_msg,
                "failed_cameras": failed_cameras,
                "results": results
            })
            return

        # Step 3: All cameras succeeded - emit event for inference in main thread
        logger.info(f"All {len(cameras)} cameras captured successfully")

        # Emit frames_captured event - inference will be handled in main thread
        self.camera_manager._emit_event("frames_captured", {
            "cameras": cameras,
            "results": results
        })

    def simulate_trigger(
        self,
        serial_number: str = None,
        trigger_type: str = "rising_edge"
    ) -> Dict:
        """
        Simulate hardware trigger for testing (without real DI)

        Args:
            serial_number: Camera serial number (None = all software trigger cameras)
            trigger_type: "rising_edge", "falling_edge", or "any_edge"

        Returns:
            Result dict
        """
        try:
            # Find cameras to trigger
            cameras_to_trigger = []

            if serial_number:
                # Trigger specific camera
                camera = self.camera_manager.cameras.get(serial_number)
                if not camera:
                    return {"success": False, "error": f"Camera {serial_number} not found"}

                if camera.mode.value != "software_trigger":
                    return {
                        "success": False,
                        "error": f"Camera {serial_number} not in software_trigger mode"
                    }

                cameras_to_trigger = [camera]
            else:
                # Trigger all cameras with software_trigger mode
                for cam in self.camera_manager.cameras.values():
                    if cam.mode.value == "software_trigger":
                        cameras_to_trigger.append(cam)

            if not cameras_to_trigger:
                return {"success": False, "error": "No cameras in software_trigger mode"}

            # Trigger cameras
            logger.info(f"Simulating {trigger_type} for {len(cameras_to_trigger)} camera(s)")
            self.trigger_cameras_group(cameras_to_trigger)

            return {
                "success": True,
                "cameras_triggered": [cam.serial_number for cam in cameras_to_trigger],
                "trigger_type": trigger_type
            }

        except Exception as e:
            logger.error(f"Error in simulate_trigger: {e}")
            return {"success": False, "error": str(e)}

    def simulate_trigger_sequence(
        self,
        serial_number: str = None,
        count: int = 5,
        interval_ms: int = 1000
    ) -> Dict:
        """
        Simulate sequence of triggers for testing

        Args:
            serial_number: Camera serial number (None = all)
            count: Number of triggers
            interval_ms: Interval between triggers in milliseconds

        Returns:
            Result dict
        """
        try:
            for i in range(count):
                result = self.simulate_trigger(serial_number, trigger_type="rising_edge")

                if not result.get("success"):
                    return {
                        "success": False,
                        "error": f"Failed at trigger {i+1}/{count}: {result.get('error')}"
                    }

                # Wait between triggers (except last one)
                if i < count - 1:
                    time.sleep(interval_ms / 1000.0)

            return {
                "success": True,
                "triggers_sent": count,
                "interval_ms": interval_ms
            }

        except Exception as e:
            logger.error(f"Error in simulate_trigger_sequence: {e}")
            return {"success": False, "error": str(e)}
