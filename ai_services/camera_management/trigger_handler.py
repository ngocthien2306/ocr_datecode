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

        # Statistics tracking
        self._stats = {
            'total_triggers': 0,          # Total trigger events detected
            'total_groups_created': 0,    # Total capture groups created
            'total_groups_completed': 0,  # Groups that completed successfully
            'total_groups_timeout': 0,    # Groups that timed out
            'total_inferences': 0,        # Total inference runs
        }
        self._stats_lock = threading.Lock()

        # Statistics monitoring thread
        self._monitoring = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stats_interval = 10  # Log stats every 10 seconds

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

            # Start statistics monitoring
            self.start_monitoring()

    def stop_polling(self):
        """Stop DI polling thread and cancel all pending timers"""
        with self._lock:
            if not self._polling:
                return

            self._polling = False

            # Stop statistics monitoring
            self.stop_monitoring()

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

    def start_monitoring(self, interval: int = 10):
        """
        Start statistics monitoring thread

        Args:
            interval: Log interval in seconds (default: 10)
        """
        if self._monitoring:
            logger.warning("Statistics monitoring already running")
            return

        self._stats_interval = interval
        self._monitoring = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="StatsMonitoringThread"
        )
        self._monitoring_thread.start()

        logger.info(f"Statistics monitoring started (interval: {interval}s)")

    def stop_monitoring(self):
        """Stop statistics monitoring thread"""
        if not self._monitoring:
            return

        self._monitoring = False

        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=2.0)
            self._monitoring_thread = None

        logger.info("Statistics monitoring stopped")

    def _monitoring_loop(self):
        """Monitoring loop - logs statistics every N seconds"""
        import os

        # Setup file logger
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'trigger_stats.log')

        # Create file handler
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - STATS - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)

        # Create stats logger
        stats_logger = logging.getLogger('trigger_stats')
        stats_logger.setLevel(logging.INFO)
        stats_logger.addHandler(file_handler)
        stats_logger.propagate = False

        logger.info(f"Statistics logging to: {log_file}")

        try:
            while self._monitoring:
                time.sleep(self._stats_interval)

                if not self._monitoring:
                    break

                # Collect current stats
                with self._stats_lock:
                    stats = self._stats.copy()

                # Count active groups and collect details
                with self._group_lock:
                    active_groups = len(self._capture_groups)
                    # Collect group details for debugging
                    group_details = []
                    current_time = time.time()
                    for gid, group in self._capture_groups.items():
                        age = current_time - group.get('created_at', current_time)
                        completed_count = group.get('completed_count', 0)
                        expected_count = group.get('expected_count', 0)
                        group_details.append(
                            f"G{gid}({completed_count}/{expected_count}, {age:.1f}s)"
                        )

                # Count active timers
                with self._timer_lock:
                    self._active_timers = [t for t in self._active_timers if t.is_alive()]
                    active_timers = len(self._active_timers)

                # Calculate success rate
                total_groups = stats['total_groups_created']
                completed = stats['total_groups_completed']
                timeout = stats['total_groups_timeout']
                success_rate = (completed / total_groups * 100) if total_groups > 0 else 0.0

                # Log to file
                stats_logger.info(
                    f"Triggers: {stats['total_triggers']} | "
                    f"Groups: {total_groups} created, {completed} completed, {timeout} timeout | "
                    f"Success: {success_rate:.1f}% | "
                    f"Inferences: {stats['total_inferences']} | "
                    f"Active: {active_groups} groups, {active_timers} timers"
                )

                # If there are active groups, log details
                if active_groups > 0:
                    stats_logger.info(f"  Active groups detail: {', '.join(group_details)}")

                # Also log to main logger (console)
                logger.info(
                    f"📊 [STATS] Triggers={stats['total_triggers']}, "
                    f"Groups={completed}/{total_groups} ({success_rate:.1f}%), "
                    f"Inferences={stats['total_inferences']}, "
                    f"Active={active_groups}g/{active_timers}t"
                )

        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            stats_logger.removeHandler(file_handler)
            file_handler.close()
            logger.info("Monitoring loop stopped")

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

                            # Increment trigger counter
                            with self._stats_lock:
                                self._stats['total_triggers'] += 1

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

        # Increment groups created counter
        with self._stats_lock:
            self._stats['total_groups_created'] += 1

        # Create capture group
        with self._group_lock:
            self._capture_groups[group_id] = {
                'cameras': cameras,
                'results': {},  # {serial_number: result_dict}
                'expected_count': len(cameras),
                'completed_count': 0,
                'group_lock': threading.Lock(),
                'created_at': time.time(),
                'timeout_timer': None  # Will be set after creation
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

        # Store timeout timer in group for cancellation
        with self._group_lock:
            if group_id in self._capture_groups:
                self._capture_groups[group_id]['timeout_timer'] = timeout_timer

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

                    # Increment stats counters
                    with self._stats_lock:
                        self._stats['total_groups_completed'] += 1
                        self._stats['total_inferences'] += 1

                    # Emit frames_captured event for batch inference
                    cameras_list = group['cameras']
                    results_dict = group['results']

                    self.camera_manager._emit_event("frames_captured", {
                        "group_id": group_id,
                        "cameras": cameras_list,
                        "results": results_dict
                    })

                    # Cancel timeout timer before cleanup
                    timeout_timer = group.get('timeout_timer')
                    if timeout_timer and timeout_timer.is_alive():
                        timeout_timer.cancel()
                        logger.debug(f"[Group #{group_id}] Cancelled timeout timer")

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

                # Increment timeout counter
                with self._stats_lock:
                    self._stats['total_groups_timeout'] += 1

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
