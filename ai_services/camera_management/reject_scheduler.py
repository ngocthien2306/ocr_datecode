import heapq
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

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

        # Scheduler thread
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None

        # Statistics
        self._stats = {
            'total_scheduled': 0,
            'total_rejected': 0,     # Actually triggered
            'total_cancelled': 0,    # Cancelled (PASS)
            'total_missed': 0,       # Too late (negative delay)
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
        Set reject configuration from recipe

        Args:
            pulse_ms: Pulse duration in milliseconds
            reject_method: Reject method ("PLC" or "DIO")
        """
        self._reject_pulse_ms = pulse_ms
        self._reject_method = reject_method.upper()

        logger.info(
            f"Reject config updated: method={self._reject_method}, pulse={pulse_ms}ms"
        )

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
                f"Max Queue: {self._stats['max_queue_depth']}"
            )

    def schedule_reject(
        self,
        group_id: int,
        T_capture_complete: float,
        inference_time: float,
        delay_reject: int,  # ms
        do_number: int,
        alarm_number: int = -1  # Alarm DO/D number (-1 = no alarm)
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

        Returns:
            True if scheduled successfully
            False if delay is negative (too late)

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

        # Check if too late
        if delay_needed < 0:
            logger.error(
                f"[Group #{group_id}] ❌ REJECT TOO LATE! "
                f"inference_time={inference_time:.3f}s > delay_reject={delay_reject_sec:.3f}s, "
                f"deficit={-delay_needed:.3f}s"
            )
            with self._stats_lock:
                self._stats['total_missed'] += 1
            return False

        # Calculate absolute reject time
        T_reject = T_capture_complete + delay_reject_sec

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
                    logger.warning(
                        f"[Group #{entry.group_id}] ⚠️ PLC mode requested but PLCController not available "
                        f"(import failed), fallback to DIO"
                    )
                    used_method = "DIO"
                    with self._stats_lock:
                        self._stats['plc_fallbacks'] += 1
                elif not self._plc_controller:
                    logger.warning(
                        f"[Group #{entry.group_id}] ⚠️ PLC mode requested but PLCController not initialized, "
                        f"fallback to DIO"
                    )
                    used_method = "DIO"
                    with self._stats_lock:
                        self._stats['plc_fallbacks'] += 1
                else:
                    logger.info(f"[Group #{entry.group_id}] Attempting PLC reject (D{reject_address})")

                    # Check PLC connection
                    if not self._plc_controller.is_connected():
                        logger.warning(
                            f"[Group #{entry.group_id}] ⚠️ PLC not connected, attempting reconnect..."
                        )

                        # Retry connect once
                        reconnect_success = self._plc_controller.connect()

                        if not reconnect_success:
                            logger.warning(
                                f"[Group #{entry.group_id}] ❌ PLC reconnect failed, fallback to DIO"
                            )
                            used_method = "DIO"
                            with self._stats_lock:
                                self._stats['plc_fallbacks'] += 1
                        else:
                            logger.info(
                                f"[Group #{entry.group_id}] ✅ PLC reconnected successfully"
                            )
                            with self._stats_lock:
                                self._stats['plc_reconnects'] += 1

                    # Execute PLC if connected
                    if self._plc_controller.is_connected() and used_method == "PLC":
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
                                f"[Group #{entry.group_id}] ❌ PLC pulse failed, fallback to DIO"
                            )
                            used_method = "DIO"
                            with self._stats_lock:
                                self._stats['plc_fallbacks'] += 1

            # Fallback to DIO if PLC failed or method is DIO
            if used_method == "DIO" or not success:
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
                        if self._do_control_callback:
                            self._do_control_callback(reject_address, pulse_ms=self._reject_pulse_ms)
                        else:
                            from .utils import trigger_reject_pulse
                            trigger_reject_pulse(reject_address, pulse_ms=self._reject_pulse_ms)

                    def pulse_alarm_dio():
                        if self._do_control_callback:
                            self._do_control_callback(alarm_address, pulse_ms=self._reject_pulse_ms)
                        else:
                            from .utils import trigger_reject_pulse
                            trigger_reject_pulse(alarm_address, pulse_ms=self._reject_pulse_ms)

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

            # Update stats
            with self._stats_lock:
                self._stats['total_rejected'] += 1

            # Log reject end with timing
            from .utils import log_reject_end
            log_reject_end(
                reject_count=reject_count,
                group_id=entry.group_id,
                do_number=reject_address,
                alarm_number=alarm_address,
                pulse_duration_ms=self._reject_pulse_ms,
                actual_duration_ms=actual_duration_ms
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
