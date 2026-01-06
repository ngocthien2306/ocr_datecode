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

        # Timer tracking for non-blocking trigger handling
        self._active_timers: List[threading.Timer] = []
        self._timer_counter = 0
        self._timer_lock = threading.Lock()

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
        """Stop DI polling thread and cancel all pending timers"""
        with self._lock:
            if not self._polling:
                return

            self._polling = False

            # Cancel all pending timers
            with self._timer_lock:
                cancelled_count = 0
                for timer in self._active_timers:
                    if timer.is_alive():
                        timer.cancel()
                        cancelled_count += 1
                        logger.debug(f"Cancelled timer: {timer.name}")
                self._active_timers.clear()

                if cancelled_count > 0:
                    logger.info(f"Cancelled {cancelled_count} pending timer(s)")

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
        Args:
            cameras: List of Camera objects to trigger

        Flow:
            1. Get delay_trigger from camera settings (dynamic from FE)
            2. Create Timer thread to fire after delay
            3. Return immediately (polling thread continues)
            4. Timer callback executes capture when fired
        """
        if not cameras:
            logger.warning("trigger_cameras_group called with empty camera list")
            return

        # Get delay from first camera (all cameras in group should have same delay)
        delay_ms = cameras[0].delay_trigger  # Dynamic value from FE (e.g. 2000, 3000)
        delay_sec = delay_ms / 1000.0

        # Generate unique timer ID
        with self._timer_lock:
            self._timer_counter += 1
            timer_id = self._timer_counter

        logger.info(
            f"[Trigger #{timer_id}] Scheduling capture for {len(cameras)} camera(s) "
            f"in {delay_sec:.2f}s (delay_trigger={delay_ms}ms)"
        )

        # Create Timer thread
        # Timer will automatically create its own thread and fire callback after delay_sec
        timer = threading.Timer(
            interval=delay_sec,
            function=self._execute_capture_now,
            args=(cameras, timer_id)
        )
        timer.daemon = True  # Daemon thread - auto cleanup when main exits
        timer.name = f"TriggerTimer-{timer_id}"

        # Track active timer
        with self._timer_lock:
            self._active_timers.append(timer)

        # Start timer
        timer.start()

        # ⭐ RETURN IMMEDIATELY - Polling thread continues!
        logger.debug(f"[Trigger #{timer_id}] Timer started, returning to polling loop")

    def _execute_capture_now(self, cameras: List['Camera'], timer_id: int):
        """
        Timer callback - executed when timer fires

        This runs in Timer's own thread, separate from polling thread

        Args:
            cameras: Cameras to capture
            timer_id: Unique timer identifier
        """
        thread_name = threading.current_thread().name
        logger.info(
            f"[Trigger #{timer_id}] Timer FIRED! Executing capture NOW "
            f"(Thread: {thread_name})"
        )

        try:
            # Step 1: Capture all cameras in parallel
            results = {}

            with ThreadPoolExecutor(max_workers=len(cameras)) as executor:
                # Submit capture jobs (IMMEDIATE capture, no delay inside!)
                future_to_camera = {
                    executor.submit(camera.execute_software_trigger_immediate): camera
                    for camera in cameras
                }

                # Wait for all cameras to complete
                for future in as_completed(future_to_camera):
                    camera = future_to_camera[future]
                    try:
                        result = future.result(timeout=10.0)
                        results[camera.serial_number] = result

                        if result.get('success'):
                            logger.info(
                                f"[Trigger #{timer_id}] Camera {camera.serial_number} "
                                f"captured {result.get('frame_count', 0)} frames"
                            )
                        else:
                            logger.error(
                                f"[Trigger #{timer_id}] Camera {camera.serial_number} "
                                f"failed: {result.get('error')}"
                            )

                    except Exception as e:
                        logger.error(
                            f"[Trigger #{timer_id}] Exception capturing "
                            f"camera {camera.serial_number}: {e}"
                        )
                        results[camera.serial_number] = {
                            'success': False,
                            'error': f'Exception: {str(e)}'
                        }

            # Step 2: Check results
            failed_cameras = [
                sn for sn, res in results.items()
                if not res.get('success', False)
            ]

            if failed_cameras:
                # Some cameras failed
                error_msg = f"Camera capture failed: {', '.join(failed_cameras)}"
                logger.error(f"[Trigger #{timer_id}] {error_msg}")

                self.camera_manager._emit_event("trigger_error", {
                    "trigger_id": timer_id,
                    "error": error_msg,
                    "failed_cameras": failed_cameras,
                    "results": results
                })
            else:
                # All cameras succeeded!
                logger.info(
                    f"[Trigger #{timer_id}] ✅ All {len(cameras)} cameras "
                    f"captured successfully"
                )

                self.camera_manager._emit_event("frames_captured", {
                    "trigger_id": timer_id,
                    "cameras": cameras,
                    "results": results
                })

        except Exception as e:
            logger.error(f"[Trigger #{timer_id}] Error in capture callback: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Cleanup: Remove timer from active list
            with self._timer_lock:
                # Filter out completed timers
                self._active_timers = [t for t in self._active_timers if t.is_alive()]

    def get_active_timer_count(self) -> int:
        """Get number of active timers (for monitoring)"""
        with self._timer_lock:
            # Clean up finished timers
            self._active_timers = [t for t in self._active_timers if t.is_alive()]
            return len(self._active_timers)

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
