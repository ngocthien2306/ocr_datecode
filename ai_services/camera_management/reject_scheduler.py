import heapq
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(order=True)
class RejectEntry:
    """Single reject entry in priority queue"""

    # Primary sort key
    T_reject: float = field(compare=True)

    # Data fields (not used for comparison)
    group_id: int = field(compare=False)
    do_number: int = field(compare=False)
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

    def __init__(self, do_control_callback=None):
        """
        Initialize RejectScheduler

        Args:
            do_control_callback: Function(do_number, pulse_ms) to trigger DO pulse
                                If None, will use default implementation
        """
        # Priority queue: sorted by T_reject (earliest first)
        self._reject_queue = []  # Heap of RejectEntry
        self._queue_lock = threading.Lock()

        # Track scheduled rejects by group_id (for cancellation)
        self._scheduled_rejects: Dict[int, RejectEntry] = {}

        # DO control callback
        self._do_control_callback = do_control_callback

        # Reject pulse duration from recipe (default 50ms)
        self._reject_pulse_ms = 50.0

        # Scheduler thread
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None

        # Statistics
        self._stats = {
            'total_scheduled': 0,
            'total_rejected': 0,     # Actually triggered
            'total_cancelled': 0,    # Cancelled (PASS)
            'total_missed': 0,       # Too late (negative delay)
            'max_queue_depth': 0
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
                f"Rejected: {self._stats['total_rejected']}, "
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
        do_number: int
    ) -> bool:
        """
        Schedule a reject action

        Args:
            group_id: Group ID for tracking
            T_capture_complete: Time when cameras finished capturing (seconds)
            inference_time: Inference duration (seconds)
            delay_reject: Time from camera to reject station (milliseconds)
            do_number: DO pin number to trigger

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
            inference_time=inference_time,
            scheduled_at=time.time()
        )

        # Log reject start
        from .utils import log_reject_start
        log_reject_start(
            reject_count=reject_count,
            group_id=group_id,
            do_number=do_number,
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

        logger.info(
            f"[Group #{group_id}] 🕐 Reject scheduled @ "
            f"T={T_reject:.3f}s (in {delay_needed:.3f}s), DO{do_number}"
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
        """Execute reject pulse"""
        import time

        try:
            logger.info(
                f"[Group #{entry.group_id}] 🔴 REJECTING! "
                f"DO{entry.do_number} pulse ({self._reject_pulse_ms}ms)"
            )

            # Get reject counter for this entry
            with self._stats_lock:
                reject_count = self._stats['total_rejected'] + 1

            # Measure actual pulse duration
            T_pulse_start = time.time()

            # Trigger DO pulse
            if self._do_control_callback:
                self._do_control_callback(entry.do_number, pulse_ms=self._reject_pulse_ms)
            else:
                # Use default implementation
                from .utils import trigger_reject_pulse
                trigger_reject_pulse(entry.do_number, pulse_ms=self._reject_pulse_ms)

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
                do_number=entry.do_number,
                pulse_duration_ms=self._reject_pulse_ms,
                actual_duration_ms=actual_duration_ms
            )

            logger.info(f"[Group #{entry.group_id}] Reject pulse complete")

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
                    f"Rejected: {stats['total_rejected']}, "
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
