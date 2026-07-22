"""
Trigger Handler Module
Handles DI polling and software trigger logic
"""
import os
import logging
import threading
import time
from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed

from .utils import read_di_value, check_trigger_edge
from .image_saver import ImageSaveWorker

if TYPE_CHECKING:
    from .camera import Camera

logger = logging.getLogger(__name__)

# Default nominal pulse width (ms) — overridden per-recipe via normal_pulse_ms field
_DEFAULT_NORMAL_PULSE_MS = 100000.0  # Internal default: treated as "disabled" until recipe sets a real value

home = os.environ.get('HOME')

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

        # Pulse width measurement: track rising edge timestamp per DI pin
        self._pulse_start_times: Dict[int, Optional[float]] = {
            0: None, 1: None, 2: None, 3: None
        }
        self._pulse_logger: Optional[logging.Logger] = None

        # Normal pulse width per recipe (ms) — used for stuck bottle detection
        self.normal_pulse_ms: float = _DEFAULT_NORMAL_PULSE_MS

        # Stuck bottle tracking: link DI pin → last triggered group_id
        self._last_group_id_by_di: Dict[int, Optional[int]] = {
            0: None, 1: None, 2: None, 3: None
        }
        # {group_id: {'stuck_count': N, 'pulse_width_ms': float}}
        self._stuck_info: Dict[int, Dict] = {}

        # Proactive stuck reject scheduling: how many rejects already scheduled
        # during the current HIGH pulse (before falling edge), per DI pin
        self._proactive_rejects_per_di: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}

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
            'total_capture_failures': 0,  # Camera capture failures (trigger received but no image)
        }
        self._stats_lock = threading.Lock()

        # Per-camera cumulative statistics (persistent across recipe reload)
        # Structure: {serial_number: {'total_pass': N, 'total_fail': M, 'total_error': K}}
        self._per_camera_cumulative_stats: Dict[str, Dict[str, int]] = {}
        self._camera_cumulative_lock = threading.Lock()

        # Auto-restart on consecutive capture failures per camera
        _CAPTURE_FAIL_RESTART_N = 5          # consecutive failures before restart
        _CAPTURE_FAIL_COOLDOWN_S = 20.0      # minimum seconds between restarts
        self._capture_fail_restart_n = _CAPTURE_FAIL_RESTART_N
        self._capture_fail_cooldown_s = _CAPTURE_FAIL_COOLDOWN_S
        self._consecutive_capture_failures: Dict[str, int] = {}  # {serial_number: count}
        self._last_restart_time: float = 0.0
        self._restart_lock = threading.Lock()

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
            self._setup_pulse_logger()
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
        from logging_config import make_handler

        # Setup dedicated stats logger → {repo_root}/logs/trigger_stats/{YYYY-MM-DD}.log
        stats_logger = logging.getLogger('trigger_stats')
        stats_logger.setLevel(logging.INFO)
        stats_logger.propagate = False
        if not any(getattr(h, "_marker", None) == "daily-trigger_stats" for h in stats_logger.handlers):
            file_handler = make_handler(
                "trigger_stats",
                fmt='%(asctime)s - STATS - %(message)s',
            )
            setattr(file_handler, "_marker", "daily-trigger_stats")
            stats_logger.addHandler(file_handler)
        logger.info("Statistics logging to: trigger_stats/{date}.log")

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
                capture_failures = stats['total_capture_failures']
                stats_logger.info(
                    f"Triggers: {stats['total_triggers']} | "
                    f"Groups: {total_groups} created, {completed} completed, {timeout} timeout | "
                    f"Success: {success_rate:.1f}% | "
                    f"Inferences: {stats['total_inferences']} | "
                    f"CaptureFailures: {capture_failures} | "
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
                    f"CaptureFail={capture_failures}, "
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

    def _setup_pulse_logger(self):
        """Setup dedicated file logger for pulse width measurements"""
        from logging_config import make_handler

        pulse_logger = logging.getLogger('pulse_width')
        pulse_logger.setLevel(logging.INFO)
        pulse_logger.propagate = False

        # Avoid duplicate handlers if called multiple times
        if not any(getattr(h, "_marker", None) == "daily-pulse_width" for h in pulse_logger.handlers):
            file_handler = make_handler(
                "pulse_width",
                fmt='%(asctime)s.%(msecs)03d | %(message)s',
            )
            # Override datefmt for millisecond format
            file_handler.formatter.datefmt = '%Y-%m-%d %H:%M:%S'
            setattr(file_handler, "_marker", "daily-pulse_width")
            pulse_logger.addHandler(file_handler)

        self._pulse_logger = pulse_logger
        logger.info("Pulse width logging to: pulse_width/{date}.log")

    def _log_pulse_width(self, di_number: int, rise_time: float, fall_time: float,
                         stuck_count: int = 1):
        """Log a single pulse width measurement"""
        if self._pulse_logger is None:
            return
        pulse_width_ms = (fall_time - rise_time) * 1000.0
        stuck_str = f" | STUCK N={stuck_count} (ratio={pulse_width_ms/self.normal_pulse_ms:.2f})" \
                    if stuck_count > 1 else ""
        self._pulse_logger.info(
            f"DI{di_number} | pulse_width={pulse_width_ms:.3f}ms | normal={self.normal_pulse_ms:.0f}ms{stuck_str}"
        )

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

                    # Pulse width measurement (independent of trigger activation)
                    if previous_value is not None and current_value != previous_value:
                        if previous_value == 0 and current_value == 1:
                            # Rising edge: record start time, reset proactive counter
                            self._pulse_start_times[di_number] = time.time()
                            self._proactive_rejects_per_di[di_number] = 0
                        elif previous_value == 1 and current_value == 0:
                            # Falling edge: compute pulse width and detect stuck bottles
                            start_t = self._pulse_start_times[di_number]
                            if start_t is not None:
                                fall_time = time.time()
                                pulse_width_ms = (fall_time - start_t) * 1000.0
                                ratio = pulse_width_ms / self.normal_pulse_ms
                                stuck_count = round(ratio) if ratio > 1.5 else 1

                                self._log_pulse_width(di_number, start_t, fall_time, stuck_count)

                                if stuck_count > 1:
                                    gid = self._last_group_id_by_di.get(di_number)
                                    if gid is not None:
                                        self._stuck_info[gid] = {
                                            'stuck_count': stuck_count,
                                            'pulse_width_ms': pulse_width_ms
                                        }
                                        logger.warning(
                                            f"DI{di_number} STUCK BOTTLES: "
                                            f"pulse={pulse_width_ms:.1f}ms, "
                                            f"ratio={ratio:.2f}, N={stuck_count} "
                                            f"(group #{gid})"
                                        )

                                        # Schedule secondary captures for bottles 2..N
                                        # cameras here is List[Tuple[Camera, str]] from di_camera_map
                                        cameras_list = [cam for cam, _ in cameras]
                                        if cameras_list:
                                            delay_trigger_ms = cameras_list[0].delay_trigger
                                            # HW TriggerDelay: exposure xảy ra +delay_trigger SAU
                                            # ExecuteSoftwareTrigger → timer phải bắn sớm hơn đúng
                                            # delay_trigger để thời điểm chụp thật không đổi.
                                            hw_offset_ms = (
                                                delay_trigger_ms
                                                if cameras_list[0].hw_delay_capture_ready() else 0.0
                                            )
                                            for i in range(1, stuck_count):
                                                # Delay from NOW (falling edge) for bottle i+1
                                                extra_ms = (delay_trigger_ms + i * self.normal_pulse_ms
                                                            - pulse_width_ms - hw_offset_ms)
                                                if extra_ms < 0:
                                                    logger.warning(
                                                        f"DI{di_number} stuck bottle {i+1}/{stuck_count}: "
                                                        f"capture window already missed ({extra_ms:.0f}ms)"
                                                    )
                                                    continue
                                                timer = threading.Timer(
                                                    interval=extra_ms / 1000.0,
                                                    function=self._capture_stuck_secondary,
                                                    args=(cameras_list, gid, i + 1, stuck_count)
                                                )
                                                timer.daemon = True
                                                timer.name = f"StuckTimer-{gid}-b{i+1}"
                                                with self._timer_lock:
                                                    self._active_timers.append(timer)
                                                timer.start()
                                                logger.info(
                                                    f"DI{di_number} stuck capture {i+1}/{stuck_count} "
                                                    f"scheduled in {extra_ms:.0f}ms"
                                                )

                                        # Schedule rejects for bottles NOT yet scheduled proactively
                                        already_proactive = self._proactive_rejects_per_di[di_number]
                                        if (cameras_list
                                                and self.camera_manager.reject_scheduler
                                                and self.camera_manager.inference_enabled):
                                            cam0 = cameras_list[0]
                                            delay_rej = int(cam0.delay_reject)
                                            do_num = cam0.do_reject_number
                                            alarm_num = cam0.do_alarm_number
                                            scheduled_now = 0
                                            for k in range(already_proactive, stuck_count):
                                                T_bottle = start_t + k * self.normal_pulse_ms / 1000.0
                                                bgid = gid * 1000 + k
                                                ok = self.camera_manager.reject_scheduler.schedule_reject(
                                                    group_id=bgid,
                                                    T_capture_complete=T_bottle,
                                                    inference_time=0.0,
                                                    delay_reject=delay_rej,
                                                    do_number=do_num,
                                                    alarm_number=alarm_num if (k == 0 and already_proactive == 0) else -1
                                                )
                                                if ok:
                                                    scheduled_now += 1
                                            logger.warning(
                                                f"[Group #{gid}] Stuck N={stuck_count}: "
                                                f"{already_proactive} proactive + {scheduled_now} at fall "
                                                f"= {already_proactive + scheduled_now} rejects total"
                                            )
                                        self._proactive_rejects_per_di[di_number] = 0

                                self._pulse_start_times[di_number] = None

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

                            # Trigger all cameras and record group_id for stuck linking
                            camera_list = [cam for cam, _ in cameras]
                            group_id = self.trigger_cameras_group(camera_list, di_number=di_number)
                            self._last_group_id_by_di[di_number] = group_id

                    # Update previous value
                    self.previous_di_values[di_number] = current_value

                    # Proactive stuck reject scheduling: fires while DI is still HIGH
                    # Each time a new full bottle-interval completes, schedule its reject immediately
                    # so early bottles aren't missed by the time the falling edge is detected.
                    if previous_value == 1 and current_value == 1:
                        start_t = self._pulse_start_times[di_number]
                        if (start_t is not None
                                and self.camera_manager.reject_scheduler
                                and self.camera_manager.inference_enabled):
                            duration_ms = (time.time() - start_t) * 1000.0
                            ratio = duration_ms / self.normal_pulse_ms
                            if ratio > 1.5:
                                confirmed = int(ratio)   # bottles fully past camera
                                already = self._proactive_rejects_per_di[di_number]
                                if confirmed > already:
                                    cameras_list = [cam for cam, _ in cameras]
                                    gid = self._last_group_id_by_di.get(di_number)
                                    if gid is not None and cameras_list:
                                        cam0 = cameras_list[0]
                                        delay_rej = int(cam0.delay_reject)
                                        do_num = cam0.do_reject_number
                                        alarm_num = cam0.do_alarm_number
                                        for k in range(already, confirmed):
                                            T_bottle = start_t + k * self.normal_pulse_ms / 1000.0
                                            bgid = gid * 1000 + k
                                            ok = self.camera_manager.reject_scheduler.schedule_reject(
                                                group_id=bgid,
                                                T_capture_complete=T_bottle,
                                                inference_time=0.0,
                                                delay_reject=delay_rej,
                                                do_number=do_num,
                                                alarm_number=alarm_num if (k == 0 and already == 0) else -1
                                            )
                                            t_in = T_bottle + delay_rej / 1000.0 - time.time()
                                            logger.info(
                                                f"[Proactive Stuck] DI{di_number} bottle {k+1} reject "
                                                f"{'scheduled' if ok else 'FAILED'} "
                                                f"(fires in {t_in*1000:.0f}ms)"
                                            )
                                        self._proactive_rejects_per_di[di_number] = confirmed

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

    def trigger_cameras_group(self, cameras: List['Camera'], di_number: Optional[int] = None):
        """
        Multi-camera synchronization with individual delays

        Args:
            cameras: List of Camera objects to trigger
            di_number: DI pin that triggered this group (None for simulated triggers).
                       Used in _capture_and_register to detect large-N stuck bottles
                       where pulse_width > delay_trigger (cameras complete before falling edge).

        Flow:
            1. Create capture group with unique ID
            2. Each camera gets individual timer based on its delay_trigger
            3. When timer fires, camera captures and registers result in group
            4. Last camera to complete triggers batch inference
            5. Return immediately (polling thread continues)
        """
        if not cameras:
            logger.warning("trigger_cameras_group called with empty camera list")
            return 0

        # ── HW TriggerDelay path: bắn trigger NGAY tại edge, TRƯỚC mọi log/lock ──
        # Camera tự đợi delay_trigger bằng phần cứng (µs-precision) rồi mới chụp
        # → threading.Timer (vốn trễ 30-350ms khi GIL bận) ra khỏi đường timing.
        # Camera nào chưa arm được thì rơi về timer legacy như cũ.
        hw_fired: List['Camera'] = []
        hw_missed: List['Camera'] = []
        legacy_cams: List['Camera'] = []
        for camera in cameras:
            if camera.hw_delay_capture_ready():
                if camera.fire_software_trigger():
                    hw_fired.append(camera)
                else:
                    # Capture trước chưa xong (2 xung DI sát hơn delay_trigger)
                    # → bỏ lượt, ghi FAIL cho group. KHÔNG rơi về legacy timer:
                    # bắn thêm trigger sẽ sinh frame mồ côi trong grab queue →
                    # mọi retrieve sau lấy nhầm frame chai trước (ảnh lệch).
                    hw_missed.append(camera)
            else:
                legacy_cams.append(camera)

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
                'timeout_timer': None,  # Will be set after creation
                'di_number': di_number  # DI pin that triggered this group (for stuck detection)
            }

        # HW-fired cameras: trigger đã bắn ở đầu hàm — chỉ cần thread lấy frame.
        # KHÔNG đưa vào _active_timers (stop_polling gọi .cancel() mà Thread
        # không có); thread tự kết thúc theo RetrieveResult timeout.
        for camera in hw_fired:
            logger.info(
                f"[Group #{group_id}] Camera {camera.serial_number} HW TriggerDelay "
                f"armed ({camera.delay_trigger:.0f}ms in-camera), retrieving..."
            )
            t = threading.Thread(
                target=self._capture_and_register,
                args=(camera, group_id, True),
                daemon=True,
                name=f"HwRetrieve-{group_id}-{camera.serial_number}",
            )
            t.start()

        # Camera bỏ lượt (xung DI quá sát): ghi FAIL vào group qua thread nhỏ
        # để group vẫn hoàn tất đủ expected_count và chai này bị reject an toàn.
        for camera in hw_missed:
            logger.warning(
                f"[Group #{group_id}] Camera {camera.serial_number} BỎ LƯỢT "
                f"trigger (capture trước chưa xong — xung DI sát hơn "
                f"delay_trigger). Ghi FAIL cho group."
            )
            t = threading.Thread(
                target=self._capture_and_register,
                args=(camera, group_id, True),
                kwargs={'precomputed_result': {
                    'success': False,
                    'error': 'HW trigger skipped: previous capture still in-flight '
                             '(DI pulses closer than delay_trigger)',
                }},
                daemon=True,
                name=f"HwMiss-{group_id}-{camera.serial_number}",
            )
            t.start()

        # Legacy path: schedule individual timer for each camera based on its delay_trigger
        for camera in legacy_cams:
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

        return group_id

    def _capture_and_register(self, camera: 'Camera', group_id: int,
                              hw_delayed: bool = False,
                              precomputed_result: Optional[Dict[str, Any]] = None):
        """
        Timer callback for single camera capture

        This runs in Timer's own thread when the camera's delay expires.
        hw_delayed=True: trigger đã bắn tại DI edge (camera tự đợi TriggerDelay)
        — thread này chỉ RetrieveResult (blocking C call, nhả GIL), không cần
        đúng giờ nữa.

        Args:
            camera: Camera to capture
            group_id: Capture group ID
            hw_delayed: True nếu dùng HW TriggerDelay path
        """
        thread_name = threading.current_thread().name
        if precomputed_result is not None:
            logger.info(
                f"[Group #{group_id}] Camera {camera.serial_number} registering "
                f"precomputed result (Thread: {thread_name})"
            )
        elif hw_delayed:
            logger.info(
                f"[Group #{group_id}] Camera {camera.serial_number} retrieving "
                f"HW-delayed frame (Thread: {thread_name})"
            )
        else:
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
            # HW path: frame đang trên đường về sau TriggerDelay in-camera.
            # Legacy path: execute immediate capture (no delay inside!)
            # Precomputed: camera bỏ lượt (xung DI quá sát) — chỉ ghi FAIL.
            if precomputed_result is not None:
                result = precomputed_result
            elif hw_delayed:
                result = camera.retrieve_hw_delayed_frame()
            else:
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
                    # Save frames to disk (non-blocking background thread)
                    ImageSaveWorker.instance().enqueue(
                        camera.serial_number,
                        result.get('frames', [])
                    )
                    # Reset consecutive failure counter on success
                    with self._restart_lock:
                        self._consecutive_capture_failures.pop(camera.serial_number, None)
                else:
                    logger.error(
                        f"[Group #{group_id}] Camera {camera.serial_number} failed: "
                        f"{result.get('error')}. Progress: {completed}/{expected}"
                    )
                    with self._stats_lock:
                        self._stats['total_capture_failures'] += 1
                    self._handle_capture_failure(camera.serial_number, group_id)

                # Check if all cameras completed
                if completed >= expected:
                    # Record capture completion time (for reject timing calculation)
                    T_capture_complete = time.time()

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

                    # Attach stuck bottle info if detected (falling edge set this before capture completes)
                    stuck_info = self._stuck_info.pop(group_id, None)
                    stuck_count = stuck_info['stuck_count'] if stuck_info else 1
                    stuck_pulse_ms = stuck_info['pulse_width_ms'] if stuck_info else 0.0

                    # Race condition fix: when pulse_width > delay_trigger, cameras complete
                    # BEFORE the falling edge, so _stuck_info has not been set yet.
                    # Detect this by checking if DI is still HIGH at capture-complete time.
                    if stuck_info is None:
                        di_num = group.get('di_number')
                        if di_num is not None:
                            pulse_start = self._pulse_start_times.get(di_num)
                            if pulse_start is not None:
                                # DI is still HIGH → pulse_width > delay_trigger → stuck case
                                current_duration_ms = (T_capture_complete - pulse_start) * 1000.0
                                estimated_n = int(current_duration_ms / self.normal_pulse_ms) + 1
                                if estimated_n >= 2:
                                    stuck_count = estimated_n
                                    stuck_pulse_ms = current_duration_ms
                                    logger.warning(
                                        f"[Group #{group_id}] DI{di_num} still HIGH at capture-complete "
                                        f"(duration={current_duration_ms:.0f}ms, estimated N≥{estimated_n}). "
                                        f"Marking as STUCK (final count will be confirmed at falling edge)."
                                    )

                    self.camera_manager._emit_event("frames_captured", {
                        "group_id": group_id,
                        "cameras": cameras_list,
                        "results": results_dict,
                        "T_capture_complete": T_capture_complete,
                        "stuck_count": stuck_count,
                        "pulse_width_ms": stuck_pulse_ms
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

    def _capture_stuck_secondary(
        self,
        cameras: List['Camera'],
        primary_group_id: int,
        bottle_index: int,
        total_bottles: int
    ):
        """
        Capture a secondary stuck bottle (bottle 2..N).
        No inference, no reject scheduling — just capture and log the image.
        Called by timer scheduled at falling edge detection.
        """
        logger.info(
            f"[Group #{primary_group_id}] STUCK secondary capture "
            f"{bottle_index}/{total_bottles} firing"
        )
        for camera in cameras:
            try:
                result = camera.execute_software_trigger_immediate()
                if result.get('success'):
                    logger.info(
                        f"[Group #{primary_group_id}] STUCK bottle {bottle_index}/{total_bottles} "
                        f"Camera {camera.serial_number}: "
                        f"{result.get('frame_count', 0)} frame(s) captured (log only, no inference)"
                    )
                else:
                    logger.error(
                        f"[Group #{primary_group_id}] STUCK bottle {bottle_index}/{total_bottles} "
                        f"Camera {camera.serial_number} failed: {result.get('error')}"
                    )

                # Emit event so frontend can log/display the image
                # NOTE: strip 'frames' (np.ndarray list) — not JSON-serializable
                self.camera_manager._emit_event("stuck_bottle_captured", {
                    "group_id": primary_group_id,
                    "bottle_index": bottle_index,
                    "total_bottles": total_bottles,
                    "camera_serial": camera.serial_number,
                    "success": result.get("success", False),
                    "frame_count": result.get("frame_count", 0),
                })

            except Exception as e:
                logger.error(
                    f"[Group #{primary_group_id}] Error in stuck secondary capture "
                    f"bottle {bottle_index}/{total_bottles}: {e}"
                )
            finally:
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

    def update_camera_cumulative_stats(self, serial_number: str, pass_count: int,
                                       fail_count: int, error_count: int):
        """
        Update cumulative per-camera statistics (persistent across recipe reload)

        Args:
            serial_number: Camera serial number
            pass_count: Number of pass frames in this inference
            fail_count: Number of fail frames in this inference
            error_count: Number of error frames in this inference
        """
        with self._camera_cumulative_lock:
            if serial_number not in self._per_camera_cumulative_stats:
                self._per_camera_cumulative_stats[serial_number] = {
                    'total_pass': 0,
                    'total_fail': 0,
                    'total_error': 0
                }

            self._per_camera_cumulative_stats[serial_number]['total_pass'] += pass_count
            self._per_camera_cumulative_stats[serial_number]['total_fail'] += fail_count
            self._per_camera_cumulative_stats[serial_number]['total_error'] += error_count

    def get_camera_cumulative_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get cumulative per-camera statistics (persistent across recipe reload)

        Returns:
            Dict mapping serial_number to cumulative stats
        """
        with self._camera_cumulative_lock:
            return {
                sn: stats.copy()
                for sn, stats in self._per_camera_cumulative_stats.items()
            }

    def _handle_capture_failure(self, serial_number: str, group_id: int):
        """
        Track consecutive capture failures per camera.
        If a single camera fails N times in a row, trigger service restart
        (subject to cooldown).
        """
        with self._restart_lock:
            count = self._consecutive_capture_failures.get(serial_number, 0) + 1
            self._consecutive_capture_failures[serial_number] = count

            logger.warning(
                f"[Group #{group_id}] Camera {serial_number} consecutive failures: "
                f"{count}/{self._capture_fail_restart_n}"
            )

            if count >= self._capture_fail_restart_n:
                now = time.time()
                elapsed = now - self._last_restart_time
                if elapsed < self._capture_fail_cooldown_s:
                    remaining = self._capture_fail_cooldown_s - elapsed
                    logger.warning(
                        f"[Group #{group_id}] Camera {serial_number} reached {count} consecutive "
                        f"failures but cooldown active ({remaining:.0f}s remaining). Skip restart."
                    )
                    return

                logger.error(
                    f"[Group #{group_id}] Camera {serial_number} failed {count} times consecutively. "
                    f"Triggering service restart!"
                )
                self._last_restart_time = now
                self._consecutive_capture_failures[serial_number] = 0
                # Run restart in background thread to not block capture loop
                t = threading.Thread(
                    target=self._restart_service,
                    args=(serial_number,),
                    daemon=True,
                    name="ServiceRestartThread"
                )
                t.start()

    def _restart_service(self, trigger_camera: str):
        """
        Execute 'sudo systemctl restart ocr-all' with auto password input.
        Called in a background thread.
        """
        import subprocess
        service = "ocr-all"
        logger.warning(
            f"[AUTO-RESTART] Camera {trigger_camera} triggered service restart → "
            f"sudo systemctl restart {service}"
        )
        try:
            result = subprocess.run(
                ['sudo', '-S', 'systemctl', 'restart', service],
                input='1\n',
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                logger.warning(
                    f"[AUTO-RESTART] Service '{service}' restarted successfully "
                    f"(triggered by camera {trigger_camera})"
                )
            else:
                logger.error(
                    f"[AUTO-RESTART] Restart failed (rc={result.returncode}): "
                    f"{result.stderr.strip()}"
                )
        except subprocess.TimeoutExpired:
            logger.error(f"[AUTO-RESTART] Restart command timed out after 30s")
        except Exception as e:
            logger.error(f"[AUTO-RESTART] Restart exception: {e}")
