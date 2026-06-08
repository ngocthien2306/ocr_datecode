import heapq
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Import PLC Controller
try:
    from .plc_controller import PLCController
    PLC_AVAILABLE = True
except ImportError:
    logger.warning("PLCController not available, PLC mode disabled")
    PLC_AVAILABLE = False


@dataclass(order=True)
class RejectEntry:
    """Single reject entry in priority queue"""

    # Primary sort key
    T_reject: float = field(compare=True)

    # Data fields (not used for comparison)
    group_id: int = field(compare=False)
    do_number: int = field(compare=False)
    alarm_number: int = field(compare=False)  # Alarm DO/D number
    inference_time: float = field(compare=False)
    scheduled_at: float = field(compare=False)

    # Cancellation flag
    cancelled: bool = field(default=False, compare=False)


class RejectScheduler:
    """
    Manages scheduled reject actions using priority queue

    Features:
    - Schedule reject at specific time based on capture completion time
    - Cancel scheduled reject (if result is PASS)
    - Thread-safe priority queue
    - Statistics tracking
    - Automatic cleanup
    """

    def __init__(self, do_control_callback=None, plc_controller: Optional['PLCController'] = None):
        """
        Initialize RejectScheduler

        Args:
            do_control_callback: Function(do_number, pulse_ms) to trigger DO pulse
                                If None, will use default implementation
            plc_controller: PLCController instance for PLC mode (optional)
        """
        # Priority queue: sorted by T_reject (earliest first)
        self._reject_queue = []  # Heap of RejectEntry
        self._queue_lock = threading.Lock()

        # Track scheduled rejects by group_id (for cancellation)
        self._scheduled_rejects: Dict[int, RejectEntry] = {}

        # DO control callback
        self._do_control_callback = do_control_callback

        # PLC Controller
        self._plc_controller = plc_controller

        # Reject configuration from recipe
        self._reject_pulse_ms = 50.0
        self._reject_method = "DIO"  # Default: DIO, can be "PLC"

        # Sensor-based mode (PLC owns timing via a sensor in front of the rejector).
        # When enabled, AI does NOT schedule pulses; it writes a verdict + handshake
        # to PLC on each inference and the PLC fires the pulse internally.
        self._sensor_based_mode = False
        self._plc_sensor_cfg: Optional[Dict[str, Any]] = None  # set via set_sensor_based_mode
        self._sensor_stats = {
            'verdicts_sent': 0,
            'verdicts_fail': 0,
            'verdicts_pass': 0,
            'ack_timeouts': 0,
            'plc_errors': 0,
        }
        self._sensor_stats_lock = threading.Lock()

        # Scheduler thread
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None

        # Statistics
        self._stats = {
            'total_scheduled': 0,
            'total_rejected': 0,     # Actually triggered
            'total_cancelled': 0,    # Cancelled (PASS)
            'total_missed': 0,       # Too late (negative delay) - dropped
            'total_late_rejects': 0, # Too late but fired anyway (allow_late_reject=True)
            'max_queue_depth': 0,
            'plc_rejects': 0,        # PLC rejects
            'dio_rejects': 0,        # DIO rejects
            'plc_fallbacks': 0,      # PLC → DIO fallbacks
            'plc_reconnects': 0      # PLC reconnect attempts (successful)
        }
        self._stats_lock = threading.Lock()

        # Sequential reject counter for logging
        self._reject_counter = 0
        self._reject_counter_lock = threading.Lock()

        # Statistics monitoring
        self._monitoring = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stats_interval = 10  # seconds

        logger.info("RejectScheduler initialized")

    def set_reject_pulse(self, pulse_ms: float):
        """
        Set reject pulse duration from recipe

        Args:
            pulse_ms: Pulse duration in milliseconds
        """
        self._reject_pulse_ms = pulse_ms
        logger.info(f"Reject pulse duration updated: {pulse_ms}ms")

    def set_reject_config(self, pulse_ms: float, reject_method: str = "DIO"):
        """
        Set reject configuration from recipe (time_based mode).

        Args:
            pulse_ms: Pulse duration in milliseconds
            reject_method: Reject method ("PLC" or "DIO")
        """
        self._sensor_based_mode = False
        self._plc_sensor_cfg = None
        self._reject_pulse_ms = pulse_ms
        self._reject_method = reject_method.upper()

        logger.info(
            f"Reject config updated (time_based): method={self._reject_method}, pulse={pulse_ms}ms"
        )

    def set_sensor_based_mode(
        self,
        plc_cfg: Dict[str, Any],
        pulse_width_ms: float,
    ) -> bool:
        """
        Switch the scheduler into sensor_based mode. The AI service no longer
        times the reject pulse; instead it writes a verdict + handshake to PLC
        on each inference, and the PLC fires its own pulse when its dedicated
        sensor in front of the rejector triggers.

        Pulse width is written to the configured register ONCE here (so the PLC
        ladder reads it at startup).

        Args:
            plc_cfg: dict with keys verdict_register / verdict_prefix /
                ready_coil / ready_prefix / ack_coil / ack_prefix /
                pulse_register / pulse_prefix / ack_timeout_ms
            pulse_width_ms: pulse width (ms) — written to pulse_register

        Returns:
            True if the pulse_width write succeeded, False otherwise (mode is
            still activated either way; failure is logged so operator can fix
            connectivity and retry).
        """
        self._sensor_based_mode = True
        self._plc_sensor_cfg = dict(plc_cfg)  # shallow copy
        self._reject_pulse_ms = pulse_width_ms

        v_label = f"{plc_cfg.get('verdict_prefix','D')}{plc_cfg['verdict_register']}"
        rdy_label = f"{plc_cfg.get('ready_prefix','M')}{plc_cfg['ready_coil']}"
        ack_label = f"{plc_cfg.get('ack_prefix','M')}{plc_cfg['ack_coil']}"
        pulse_label = f"{plc_cfg.get('pulse_prefix','D')}{plc_cfg['pulse_register']}"

        logger.info(
            f"Sensor-based mode ENABLED — verdict={v_label}, ready={rdy_label}, "
            f"ack={ack_label}, pulse_reg={pulse_label}, "
            f"ack_timeout={plc_cfg['ack_timeout_ms']}ms"
        )

        # Push pulse_width to the PLC pulse register so the ladder has it.
        if not PLC_AVAILABLE or self._plc_controller is None:
            logger.error(
                "Sensor-based mode: PLCController unavailable — pulse_width "
                "could not be pushed. Verdict writes will also fail until PLC "
                "is reachable."
            )
            return False
        if not self._plc_controller.is_connected():
            logger.info("Sensor-based mode: PLC not connected, attempting connect...")
            if not self._plc_controller.connect():
                logger.error(
                    "Sensor-based mode: PLC connect failed — pulse_width not pushed"
                )
                return False
        ok, _dt = self._plc_controller.write_register(
            int(plc_cfg['pulse_register']), int(round(pulse_width_ms))
        )
        if ok:
            logger.info(
                f"Sensor-based mode: pulse_width={int(round(pulse_width_ms))}ms "
                f"written to {pulse_label}"
            )
        else:
            logger.error(
                f"Sensor-based mode: failed to write pulse_width to {pulse_label}"
            )
        return ok

    def is_sensor_based(self) -> bool:
        """True when the scheduler is in sensor_based mode."""
        return self._sensor_based_mode

    def send_verdict(self, verdict: int, group_id: int = 0) -> bool:
        """
        Push a PASS/FAIL verdict to the PLC (sensor_based mode).

        Sequence:
            1. Pre-clear the ack coil (defensive — clears any stale ack).
            2. Write the verdict register (0=PASS, 1=FAIL).
            3. Set the ready coil HIGH.
            4. Poll the ack coil until either the PLC sets it HIGH or
               `ack_timeout_ms` elapses.
            5. Drop the ready coil LOW so the PLC sees a clean edge next time.

        Returns:
            True if PLC acknowledged within the timeout, False otherwise.
            Caller should still consider the verdict "delivered" on True;
            False means the PLC did not confirm and the product may not be
            handled correctly — operator should be alerted.
        """
        if not self._sensor_based_mode or self._plc_sensor_cfg is None:
            logger.error(
                f"[Group #{group_id}] send_verdict called outside sensor_based mode"
            )
            return False

        cfg = self._plc_sensor_cfg
        with self._sensor_stats_lock:
            self._sensor_stats['verdicts_sent'] += 1
            if verdict:
                self._sensor_stats['verdicts_fail'] += 1
            else:
                self._sensor_stats['verdicts_pass'] += 1

        if self._plc_controller is None or not self._plc_controller.is_connected():
            logger.warning(
                f"[Group #{group_id}] send_verdict: PLC not connected — attempting reconnect"
            )
            if self._plc_controller is None or not self._plc_controller.connect():
                logger.error(
                    f"[Group #{group_id}] send_verdict: PLC unreachable, verdict={verdict} dropped"
                )
                with self._sensor_stats_lock:
                    self._sensor_stats['plc_errors'] += 1
                return False

        t_start = time.perf_counter()

        # 1. Pre-clear ack
        self._plc_controller.write_coil(int(cfg['ack_coil']), False)

        # 2. Write verdict register
        ok_verdict, _ = self._plc_controller.write_register(
            int(cfg['verdict_register']), int(verdict)
        )
        if not ok_verdict:
            logger.error(
                f"[Group #{group_id}] send_verdict: failed to write verdict register"
            )
            with self._sensor_stats_lock:
                self._sensor_stats['plc_errors'] += 1
            return False

        # 3. Set ready coil HIGH
        if not self._plc_controller.write_coil(int(cfg['ready_coil']), True):
            logger.error(
                f"[Group #{group_id}] send_verdict: failed to set ready coil"
            )
            with self._sensor_stats_lock:
                self._sensor_stats['plc_errors'] += 1
            return False

        # 4. Poll ACK until timeout
        timeout_s = float(cfg['ack_timeout_ms']) / 1000.0
        poll_interval = 0.005  # 5 ms — Modbus read takes longer in practice
        ack_seen = False
        while (time.perf_counter() - t_start) < timeout_s:
            ack = self._plc_controller.read_coil(int(cfg['ack_coil']))
            if ack is True:
                ack_seen = True
                break
            time.sleep(poll_interval)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # 5. Drop ready coil regardless of ack outcome (clean edge for next cycle)
        self._plc_controller.write_coil(int(cfg['ready_coil']), False)

        if ack_seen:
            logger.info(
                f"[Group #{group_id}] verdict={verdict} delivered → ACK in {elapsed_ms:.1f}ms"
            )
            return True
        else:
            logger.error(
                f"[Group #{group_id}] verdict={verdict} sent but ACK timeout after "
                f"{elapsed_ms:.1f}ms (limit={cfg['ack_timeout_ms']}ms)"
            )
            with self._sensor_stats_lock:
                self._sensor_stats['ack_timeouts'] += 1
            return False

    def start(self):
        """Start scheduler thread"""
        if self._running:
            logger.warning("RejectScheduler already running")
            return

        self._running = True

        # Start scheduler thread
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="RejectScheduler"
        )
        self._scheduler_thread.start()

        # Start monitoring thread
        self._monitoring = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="RejectMonitor"
        )
        self._monitoring_thread.start()

        logger.info("RejectScheduler started")

    def stop(self):
        """Stop scheduler and clear queue"""
        if not self._running:
            return

        logger.info("Stopping RejectScheduler...")

        self._running = False
        self._monitoring = False

        # Wait for threads
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=2.0)

        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=2.0)

        # Clear queue
        with self._queue_lock:
            cancelled_count = len(self._reject_queue)
            self._reject_queue.clear()
            self._scheduled_rejects.clear()

        logger.info(f"RejectScheduler stopped (cancelled {cancelled_count} pending rejects)")

        # Log final stats
        with self._stats_lock:
            logger.info(
                f"📊 [REJECT FINAL STATS] "
                f"Scheduled: {self._stats['total_scheduled']}, "
                f"Rejected: {self._stats['total_rejected']} "
                f"(PLC: {self._stats['plc_rejects']}, DIO: {self._stats['dio_rejects']}, "
                f"Fallback: {self._stats['plc_fallbacks']}, Reconnect: {self._stats['plc_reconnects']}), "
                f"Cancelled: {self._stats['total_cancelled']}, "
                f"Missed: {self._stats['total_missed']}, "
                f"Late: {self._stats['total_late_rejects']}, "
                f"Max Queue: {self._stats['max_queue_depth']}"
            )

    def schedule_reject(
        self,
        group_id: int,
        T_capture_complete: float,
        inference_time: float,
        delay_reject: int,  # ms
        do_number: int,
        alarm_number: int = -1,  # Alarm DO/D number (-1 = no alarm)
        allow_late_reject: bool = False  # Fire immediately when inference_time > delay_reject
    ) -> bool:
        """
        Schedule a reject action (with optional alarm)

        Args:
            group_id: Group ID for tracking
            T_capture_complete: Time when cameras finished capturing (seconds)
            inference_time: Inference duration (seconds)
            delay_reject: Time from camera to reject station (milliseconds)
            do_number: DO/D pin number for reject
            alarm_number: DO/D pin number for alarm (-1 to disable)
            allow_late_reject: If True, fire reject immediately when inference_time > delay_reject
                              instead of dropping it. Use for carton/alarm-only systems where
                              late reject (firing after product has passed) is harmless.

        Returns:
            True if scheduled successfully
            False if delay is negative AND allow_late_reject=False (too late, dropped)

        Timing calculation:
            delay_reject_sec = delay_reject / 1000.0
            delay_needed = delay_reject_sec - inference_time
            T_reject = T_capture_complete + delay_reject_sec

        Example:
            T_capture_complete = 3.600s
            inference_time = 0.450s
            delay_reject = 4000ms = 4.0s

            → delay_needed = 4.0 - 0.450 = 3.550s
            → T_reject = 3.600 + 4.0 = 7.600s

            Timeline:
            t=3.600s: Capture complete
            t=4.050s: Inference done (3.600 + 0.450)
            t=7.600s: Reject fires! (product at reject station)
        """
        # Calculate delay
        delay_reject_sec = delay_reject / 1000.0
        delay_needed = delay_reject_sec - inference_time
        late_fire = False

        # Check if too late
        if delay_needed < 0:
            if allow_late_reject:
                logger.warning(
                    f"[Group #{group_id}] ⚠️ REJECT LATE — firing immediately "
                    f"(inference_time={inference_time:.3f}s > delay_reject={delay_reject_sec:.3f}s, "
                    f"deficit={-delay_needed:.3f}s)"
                )
                late_fire = True
                with self._stats_lock:
                    self._stats['total_late_rejects'] += 1
            else:
                logger.error(
                    f"[Group #{group_id}] ❌ REJECT TOO LATE! "
                    f"inference_time={inference_time:.3f}s > delay_reject={delay_reject_sec:.3f}s, "
                    f"deficit={-delay_needed:.3f}s"
                )
                with self._stats_lock:
                    self._stats['total_missed'] += 1
                return False

        # Calculate absolute reject time
        # Late fire: T_reject = now → scheduler picks it up on next tick (~1ms)
        T_reject = time.time() if late_fire else (T_capture_complete + delay_reject_sec)

        # Increment reject counter
        with self._reject_counter_lock:
            self._reject_counter += 1
            reject_count = self._reject_counter

        # Create entry
        entry = RejectEntry(
            T_reject=T_reject,
            group_id=group_id,
            do_number=do_number,
            alarm_number=alarm_number,
            inference_time=inference_time,
            scheduled_at=time.time()
        )

        # Log reject start
        from .utils import log_reject_start
        log_reject_start(
            reject_count=reject_count,
            group_id=group_id,
            do_number=do_number,
            alarm_number=alarm_number,
            T_capture_complete=T_capture_complete,
            T_reject=T_reject,
            delay_reject_ms=delay_reject,
            inference_time_ms=inference_time * 1000
        )

        # Add to queue (thread-safe)
        with self._queue_lock:
            heapq.heappush(self._reject_queue, entry)
            self._scheduled_rejects[group_id] = entry

            # Update stats
            with self._stats_lock:
                self._stats['total_scheduled'] += 1
                queue_depth = len(self._reject_queue)
                if queue_depth > self._stats['max_queue_depth']:
                    self._stats['max_queue_depth'] = queue_depth

        alarm_info = f" + Alarm(DO/D{alarm_number})" if alarm_number >= 0 else ""
        logger.info(
            f"[Group #{group_id}] 🕐 Reject scheduled @ "
            f"T={T_reject:.3f}s (in {delay_needed:.3f}s), "
            f"Reject(DO/D{do_number}){alarm_info}"
        )

        return True

    def cancel_reject(self, group_id: int) -> bool:
        """
        Cancel scheduled reject (when result is PASS)

        Args:
            group_id: Group ID to cancel

        Returns:
            True if cancelled successfully
            False if not found (already executed or never scheduled)
        """
        with self._queue_lock:
            if group_id not in self._scheduled_rejects:
                # Already executed or never scheduled
                return False

            entry = self._scheduled_rejects[group_id]

            # Mark as cancelled
            entry.cancelled = True

            # Remove from tracking dict
            del self._scheduled_rejects[group_id]

        # Update stats and get counter
        with self._stats_lock:
            self._stats['total_cancelled'] += 1
            # Counter for cancelled rejects is total_scheduled
            reject_count = self._stats['total_scheduled']

        # Log cancellation
        from .utils import log_reject_cancelled
        log_reject_cancelled(
            reject_count=reject_count,
            group_id=group_id,
            reason="PASS"
        )

        logger.info(
            f"[Group #{group_id}] ✅ Reject cancelled (result: PASS)"
        )

        return True

    def _scheduler_loop(self):
        """
        Main scheduler loop

        Strategy:
        - Check queue head every iteration
        - If ready: pop and execute
        - Else: sleep until next event (max 10ms)
        """
        logger.info("RejectScheduler thread started")

        try:
            while self._running:
                T_now = time.time()
                entry_to_execute = None

                with self._queue_lock:
                    if not self._reject_queue:
                        # Queue empty, sleep 10ms
                        pass
                    else:
                        # Peek at head (earliest reject)
                        next_entry = self._reject_queue[0]

                        if T_now >= next_entry.T_reject:
                            # TIME TO REJECT!
                            entry = heapq.heappop(self._reject_queue)

                            # Check if cancelled
                            if entry.cancelled:
                                logger.debug(
                                    f"[Group #{entry.group_id}] Skipping cancelled reject"
                                )
                                continue  # Skip to next iteration

                            # Remove from tracking dict
                            if entry.group_id in self._scheduled_rejects:
                                del self._scheduled_rejects[entry.group_id]

                            entry_to_execute = entry

                # Execute reject (outside lock to avoid blocking)
                if entry_to_execute:
                    self._execute_reject(entry_to_execute)
                else:
                    # Calculate sleep time
                    with self._queue_lock:
                        if self._reject_queue:
                            next_entry = self._reject_queue[0]
                            sleep_time = min(next_entry.T_reject - time.time(), 0.01)
                            sleep_time = max(sleep_time, 0.001)  # At least 1ms
                        else:
                            sleep_time = 0.01  # Queue empty

                    time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Error in scheduler loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            logger.info("RejectScheduler thread stopped")

    def _execute_reject(self, entry: RejectEntry):
        """Execute reject pulse (PLC or DIO with fallback)"""
        import time

        try:
            # Get reject counter for this entry
            with self._stats_lock:
                reject_count = self._stats['total_rejected'] + 1

            # Determine reject method
            method = self._reject_method
            reject_address = entry.do_number  # D0=0, D1=1, etc.
            alarm_address = entry.alarm_number  # D1=1, etc. (-1 = no alarm)

            alarm_info = f" + Alarm({alarm_address})" if alarm_address >= 0 else ""
            logger.info(
                f"[Group #{entry.group_id}] 🔴 REJECTING! "
                f"Method: {method}, Reject({reject_address}){alarm_info}, "
                f"Pulse: {self._reject_pulse_ms}ms"
            )

            # Measure actual pulse duration
            T_pulse_start = time.time()
            success = False
            used_method = method

            # Try PLC first if configured
            if method == "PLC":
                # Check prerequisites
                if not PLC_AVAILABLE:
                    logger.error(
                        f"[Group #{entry.group_id}] ❌ PLC mode requested but PLCController not available "
                        f"(import failed), skipping reject"
                    )
                    with self._stats_lock:
                        self._stats['plc_fallbacks'] += 1
                elif not self._plc_controller:
                    logger.error(
                        f"[Group #{entry.group_id}] ❌ PLC mode requested but PLCController not initialized, "
                        f"skipping reject"
                    )
                    with self._stats_lock:
                        self._stats['plc_fallbacks'] += 1
                else:
                    logger.info(f"[Group #{entry.group_id}] Attempting PLC reject (D{reject_address})")

                    # Check PLC connection — try reconnect once if disconnected
                    if not self._plc_controller.is_connected():
                        logger.warning(
                            f"[Group #{entry.group_id}] ⚠️ PLC not connected, attempting reconnect (up to 3 times)..."
                        )

                        reconnect_success = False
                        for attempt in range(1, 4):
                            reconnect_success = self._plc_controller.connect()
                            if reconnect_success:
                                logger.info(
                                    f"[Group #{entry.group_id}] ✅ PLC reconnected (attempt {attempt}/3)"
                                )
                                with self._stats_lock:
                                    self._stats['plc_reconnects'] += 1
                                break
                            logger.warning(
                                f"[Group #{entry.group_id}] ⚠️ PLC reconnect attempt {attempt}/3 failed"
                            )

                        if not reconnect_success:
                            logger.error(
                                f"[Group #{entry.group_id}] ❌ PLC reconnect failed after 3 attempts, skipping reject"
                            )
                            with self._stats_lock:
                                self._stats['plc_fallbacks'] += 1

                    # Execute PLC if connected
                    if self._plc_controller.is_connected():
                        # Check if reject and alarm use same address
                        if alarm_address >= 0 and alarm_address == reject_address:
                            logger.info(
                                f"[Group #{entry.group_id}] Reject and Alarm use same D{reject_address}, "
                                f"triggering once"
                            )
                            # Same address, pulse once
                            success, plc_duration = self._plc_controller.write_pulse(
                                reject_address, self._reject_pulse_ms
                            )
                        elif alarm_address >= 0:
                            # Different addresses, use optimized dual pulse
                            logger.info(
                                f"[Group #{entry.group_id}] Executing dual PLC pulse "
                                f"(D{reject_address} + D{alarm_address})"
                            )
                            success, plc_duration = self._plc_controller.write_dual_pulse(
                                reject_address, alarm_address, self._reject_pulse_ms
                            )
                        else:
                            # Only reject, no alarm
                            success, plc_duration = self._plc_controller.write_pulse(
                                reject_address, self._reject_pulse_ms
                            )

                        # Check results
                        if success:
                            logger.info(
                                f"[Group #{entry.group_id}] ✅ PLC pulse successful "
                                f"(Duration: {plc_duration:.2f}ms)"
                            )
                            with self._stats_lock:
                                self._stats['plc_rejects'] += 1
                        else:
                            logger.error(
                                f"[Group #{entry.group_id}] ❌ PLC pulse failed, skipping reject"
                            )
                            with self._stats_lock:
                                self._stats['plc_fallbacks'] += 1

            # Execute DIO only when method is DIO (no fallback from PLC failure)
            if method == "DIO_OUT":
                # Check if reject and alarm use same address
                if alarm_address >= 0 and alarm_address == reject_address:
                    logger.info(
                        f"[Group #{entry.group_id}] Reject and Alarm use same DO{reject_address}, "
                        f"triggering once"
                    )
                    # Same address, pulse once
                    if self._do_control_callback:
                        self._do_control_callback(reject_address, pulse_ms=self._reject_pulse_ms)
                    else:
                        from .utils import trigger_reject_pulse
                        trigger_reject_pulse(reject_address, pulse_ms=self._reject_pulse_ms)
                elif alarm_address >= 0:
                    # Different addresses, pulse in parallel (threading)
                    logger.info(
                        f"[Group #{entry.group_id}] Executing parallel DIO pulses "
                        f"(DO{reject_address} + DO{alarm_address})"
                    )

                    import threading

                    def pulse_reject_dio():
                        try:
                            if self._do_control_callback:
                                self._do_control_callback(reject_address, pulse_ms=self._reject_pulse_ms)
                            else:
                                from .utils import trigger_reject_pulse
                                trigger_reject_pulse(reject_address, pulse_ms=self._reject_pulse_ms)
                        except Exception as exc:
                            logger.error(f"[Group #{entry.group_id}] ❌ Reject pulse DO{reject_address} failed: {exc}")

                    def pulse_alarm_dio():
                        try:
                            if self._do_control_callback:
                                self._do_control_callback(alarm_address, pulse_ms=self._reject_pulse_ms)
                            else:
                                from .utils import trigger_reject_pulse
                                trigger_reject_pulse(alarm_address, pulse_ms=self._reject_pulse_ms)
                        except Exception as exc:
                            logger.error(f"[Group #{entry.group_id}] ❌ Alarm pulse DO{alarm_address} failed: {exc}")

                    # Start both threads
                    t_reject = threading.Thread(target=pulse_reject_dio, daemon=True)
                    t_alarm = threading.Thread(target=pulse_alarm_dio, daemon=True)

                    t_reject.start()
                    t_alarm.start()

                    # Wait for both to complete
                    t_reject.join()
                    t_alarm.join()
                else:
                    # Only reject, no alarm
                    logger.info(f"[Group #{entry.group_id}] Executing DIO reject (DO{reject_address})")
                    if self._do_control_callback:
                        self._do_control_callback(reject_address, pulse_ms=self._reject_pulse_ms)
                    else:
                        from .utils import trigger_reject_pulse
                        trigger_reject_pulse(reject_address, pulse_ms=self._reject_pulse_ms)

                success = True
                with self._stats_lock:
                    self._stats['dio_rejects'] += 1

                if alarm_address >= 0 and alarm_address != reject_address:
                    logger.info(
                        f"[Group #{entry.group_id}] ✅ DIO parallel pulses successful "
                        f"(Reject DO{reject_address} + Alarm DO{alarm_address})"
                    )
                else:
                    logger.info(
                        f"[Group #{entry.group_id}] ✅ DIO pulse successful (DO{reject_address})"
                    )

            # Calculate actual duration
            T_pulse_end = time.time()
            actual_duration_ms = (T_pulse_end - T_pulse_start) * 1000

            if success:
                # Update stats and log only when reject actually fired
                with self._stats_lock:
                    self._stats['total_rejected'] += 1

                from .utils import log_reject_end
                log_reject_end(
                    reject_count=reject_count,
                    group_id=entry.group_id,
                    do_number=reject_address,
                    alarm_number=alarm_address,
                    pulse_duration_ms=self._reject_pulse_ms,
                    actual_duration_ms=actual_duration_ms
                )
            else:
                logger.warning(
                    f"[Group #{entry.group_id}] ⚠️ Reject SKIPPED "
                    f"(method={method}, PLC unavailable/failed)"
                )

            logger.info(
                f"[Group #{entry.group_id}] Reject complete "
                f"(method: {used_method}, duration: {actual_duration_ms:.2f}ms)"
            )

        except Exception as e:
            logger.error(f"[Group #{entry.group_id}] Error executing reject: {e}")
            import traceback
            traceback.print_exc()

    def _monitoring_loop(self):
        """Monitoring loop - logs statistics every N seconds"""
        logger.info(f"Reject monitoring started (interval: {self._stats_interval}s)")

        try:
            while self._monitoring:
                time.sleep(self._stats_interval)

                if not self._monitoring:
                    break

                # Collect stats
                with self._stats_lock:
                    stats = self._stats.copy()

                with self._queue_lock:
                    active_queue = len(self._reject_queue)

                # Log stats
                logger.info(
                    f"📊 [REJECT STATS] "
                    f"Scheduled: {stats['total_scheduled']}, "
                    f"Rejected: {stats['total_rejected']} "
                    f"(PLC: {stats['plc_rejects']}, DIO: {stats['dio_rejects']}, "
                    f"Fallback: {stats['plc_fallbacks']}, Reconnect: {stats['plc_reconnects']}), "
                    f"Cancelled: {stats['total_cancelled']}, "
                    f"Missed: {stats['total_missed']}, "
                    f"Late: {stats['total_late_rejects']}, "
                    f"Max Queue: {stats['max_queue_depth']}, "
                    f"Active: {active_queue}"
                )

        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
        finally:
            logger.info("Reject monitoring stopped")

    def get_stats(self) -> dict:
        """Get scheduler statistics"""
        with self._stats_lock:
            stats = self._stats.copy()

        with self._queue_lock:
            stats['active_queue_depth'] = len(self._reject_queue)

        return stats
