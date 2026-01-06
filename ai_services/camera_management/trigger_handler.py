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

        # Capture group tracking for multi-camera sync with different delays
        self._capture_groups: Dict[int, Dict] = {}
        self._group_lock = threading.Lock()

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
        Multi-camera synchronization with individual delays

        Args:
            cameras: List of Camera objects to trigger

        Flow:
            1. Create capture group with unique ID
            2. Each camera gets individual timer based on its delay_trigger
            3. When timer fires, camera captures and registers result in group
            4. Last camera to complete triggers batch inference
            5. Return immediately (polling thread continues)
        """
        if not cameras:
            logger.warning("trigger_cameras_group called with empty camera list")
            return

        # Generate unique group ID
        with self._timer_lock:
            self._timer_counter += 1
            group_id = self._timer_counter

        # Log camera delays
        camera_delays = [(cam.serial_number, cam.delay_trigger) for cam in cameras]
        logger.info(
            f"[Group #{group_id}] Creating capture group for {len(cameras)} camera(s): "
            f"{camera_delays}"
        )

        # Create capture group
        with self._group_lock:
            self._capture_groups[group_id] = {
                'cameras': cameras,
                'results': {},  # {serial_number: result_dict}
                'expected_count': len(cameras),
                'completed_count': 0,
                'group_lock': threading.Lock(),
                'created_at': time.time()
            }

        # Schedule individual timer for each camera based on its delay_trigger
        for camera in cameras:
            delay_ms = camera.delay_trigger
            delay_sec = delay_ms / 1000.0

            logger.info(
                f"[Group #{group_id}] Scheduling Camera {camera.serial_number} "
                f"capture in {delay_sec:.2f}s"
            )

            # Create timer for this camera
            timer = threading.Timer(
                interval=delay_sec,
                function=self._capture_and_register,
                args=(camera, group_id)
            )
            timer.daemon = True
            timer.name = f"CaptureTimer-{group_id}-{camera.serial_number}"

            # Track active timer
            with self._timer_lock:
                self._active_timers.append(timer)

            # Start timer
            timer.start()

        # Schedule timeout protection (max delay + 10 seconds buffer)
        max_delay = max(cam.delay_trigger for cam in cameras) / 1000.0
        timeout_sec = max_delay + 10.0

        timeout_timer = threading.Timer(
            interval=timeout_sec,
            function=self._group_timeout,
            args=(group_id,)
        )
        timeout_timer.daemon = True
        timeout_timer.name = f"TimeoutTimer-{group_id}"

        with self._timer_lock:
            self._active_timers.append(timeout_timer)

        timeout_timer.start()

        logger.debug(
            f"[Group #{group_id}] All timers scheduled, timeout in {timeout_sec:.1f}s. "
            f"Returning to polling loop."
        )

    def _capture_and_register(self, camera: 'Camera', group_id: int):
        """
        Timer callback for single camera capture

        This runs in Timer's own thread when the camera's delay expires.

        Args:
            camera: Camera to capture
            group_id: Capture group ID
        """
        thread_name = threading.current_thread().name
        logger.info(
            f"[Group #{group_id}] Camera {camera.serial_number} timer FIRED! "
            f"Capturing now (Thread: {thread_name})"
        )

        # Check if group still exists
        with self._group_lock:
            if group_id not in self._capture_groups:
                logger.warning(
                    f"[Group #{group_id}] Group not found (timeout or cancelled), "
                    f"skipping Camera {camera.serial_number}"
                )
                return

            group = self._capture_groups[group_id]

        try:
            # Execute immediate capture (no delay inside!)
            result = camera.execute_software_trigger_immediate()

            # Register result in group (thread-safe)
            with group['group_lock']:
                group['results'][camera.serial_number] = result
                group['completed_count'] += 1
                completed = group['completed_count']
                expected = group['expected_count']

                if result.get('success'):
                    logger.info(
                        f"[Group #{group_id}] Camera {camera.serial_number} captured "
                        f"{result.get('frame_count', 0)} frames. "
                        f"Progress: {completed}/{expected}"
                    )
                else:
                    logger.error(
                        f"[Group #{group_id}] Camera {camera.serial_number} failed: "
                        f"{result.get('error')}. Progress: {completed}/{expected}"
                    )

                # Check if all cameras completed
                if completed >= expected:
                    logger.info(
                        f"[Group #{group_id}] ✅ All {expected} cameras completed! "
                        f"Triggering batch inference."
                    )

                    # Emit frames_captured event for batch inference
                    cameras_list = group['cameras']
                    results_dict = group['results']

                    self.camera_manager._emit_event("frames_captured", {
                        "group_id": group_id,
                        "cameras": cameras_list,
                        "results": results_dict
                    })

                    # Cleanup group
                    with self._group_lock:
                        if group_id in self._capture_groups:
                            del self._capture_groups[group_id]
                            logger.debug(f"[Group #{group_id}] Group cleaned up")

        except Exception as e:
            logger.error(
                f"[Group #{group_id}] Exception capturing Camera {camera.serial_number}: {e}"
            )
            import traceback
            traceback.print_exc()

            # Register failure
            with group['group_lock']:
                group['results'][camera.serial_number] = {
                    'success': False,
                    'error': f'Exception: {str(e)}'
                }
                group['completed_count'] += 1

        finally:
            # Cleanup timer from active list
            with self._timer_lock:
                self._active_timers = [t for t in self._active_timers if t.is_alive()]

    def _group_timeout(self, group_id: int):
        """
        Timeout handler for capture groups

        If a group doesn't complete within timeout, emit error and cleanup.

        Args:
            group_id: Capture group ID
        """
        logger.warning(f"[Group #{group_id}] Timeout reached!")

        with self._group_lock:
            if group_id not in self._capture_groups:
                logger.debug(f"[Group #{group_id}] Already completed, ignoring timeout")
                return

            group = self._capture_groups[group_id]

            with group['group_lock']:
                completed = group['completed_count']
                expected = group['expected_count']
                results = group['results']

                # Find missing cameras
                all_serials = {cam.serial_number for cam in group['cameras']}
                completed_serials = set(results.keys())
                missing_serials = all_serials - completed_serials

                logger.error(
                    f"[Group #{group_id}] Timeout! Only {completed}/{expected} completed. "
                    f"Missing: {list(missing_serials)}"
                )

                # Emit error event
                self.camera_manager._emit_event("trigger_error", {
                    "group_id": group_id,
                    "error": "Capture group timeout",
                    "completed_count": completed,
                    "expected_count": expected,
                    "missing_cameras": list(missing_serials),
                    "results": results
                })

            # Cleanup group
            del self._capture_groups[group_id]
            logger.debug(f"[Group #{group_id}] Timeout cleanup complete")

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
