"""
Dedicated audit log for reject actions, kept separate from the service log so
the record of what was physically rejected stays readable and greppable.
"""

import logging
from pathlib import Path

# Service log. Distinct from `_reject_logger` below, which is the dedicated
# audit trail — this one reports on the audit logger's own setup, and its
# absence used to raise NameError at the end of _init_reject_logger().
logger = logging.getLogger(__name__)

# Dedicated logger for reject actions
_reject_logger = None
_reject_log_file = None

def _init_reject_logger():
    """Initialize dedicated reject action logger"""
    global _reject_logger, _reject_log_file

    if _reject_logger is not None:
        return _reject_logger

    from logging_config import make_handler, get_log_dir

    # Logger writes to {repo_root}/logs/reject_actions/{YYYY-MM-DD}.log (auto-rotated)
    _reject_logger = logging.getLogger("reject_actions")
    _reject_logger.setLevel(logging.INFO)
    _reject_logger.propagate = False
    _reject_logger.handlers.clear()

    file_handler = make_handler(
        "reject_actions",
        fmt='%(asctime)s.%(msecs)03d | %(message)s',
    )
    file_handler.formatter.datefmt = '%Y-%m-%d %H:%M:%S'
    setattr(file_handler, "_marker", "daily-reject_actions")
    _reject_logger.addHandler(file_handler)
    _reject_log_file = Path(file_handler.baseFilename)

    logger.info(f"Reject action logger initialized: {_reject_log_file}")

    return _reject_logger


def log_reject_start(reject_count: int, group_id: int, do_number: int,
                     T_capture_complete: float, T_reject: float,
                     delay_reject_ms: int, inference_time_ms: float,
                     alarm_number: int = -1):
    """
    Log reject action START with timing and counter

    Args:
        reject_count: Sequential reject counter (1, 2, 3, ...)
        group_id: Capture group ID
        do_number: Digital output pin number for reject
        T_capture_complete: Timestamp when capture completed (seconds)
        T_reject: Scheduled reject timestamp (seconds)
        delay_reject_ms: Configured delay from capture to reject (milliseconds)
        inference_time_ms: Inference duration (milliseconds)
        alarm_number: Digital output pin number for alarm (-1 = no alarm)
    """
    reject_log = _init_reject_logger()

    import time
    T_now = time.time()
    time_until_reject = (T_reject - T_now) * 1000  # ms

    alarm_info = f" + Alarm(DO{alarm_number})" if alarm_number >= 0 else ""
    reject_log.info(
        f"REJECT_START | #{reject_count:04d} | Group #{group_id} | Reject(DO{do_number}){alarm_info} | "
        f"Scheduled: T={T_reject:.3f}s (in {time_until_reject:.1f}ms) | "
        f"Delay: {delay_reject_ms}ms | Inference: {inference_time_ms:.1f}ms"
    )


def log_reject_end(reject_count: int, group_id: int, do_number: int,
                   pulse_duration_ms: float, actual_duration_ms: float,
                   alarm_number: int = -1):
    """
    Log reject action END with actual duration

    Args:
        reject_count: Sequential reject counter (same as start)
        group_id: Capture group ID
        do_number: Digital output pin number for reject
        pulse_duration_ms: Configured pulse duration (milliseconds)
        actual_duration_ms: Actual measured pulse duration (milliseconds)
        alarm_number: Digital output pin number for alarm (-1 = no alarm)
    """
    reject_log = _init_reject_logger()

    duration_diff = actual_duration_ms - pulse_duration_ms

    alarm_info = f" + Alarm(DO{alarm_number})" if alarm_number >= 0 else ""
    reject_log.info(
        f"REJECT_END   | #{reject_count:04d} | Group #{group_id} | Reject(DO{do_number}){alarm_info} | "
        f"Pulse: {pulse_duration_ms:.1f}ms | Actual: {actual_duration_ms:.1f}ms | "
        f"Diff: {duration_diff:+.1f}ms"
    )


def log_reject_cancelled(reject_count: int, group_id: int, reason: str = "PASS"):
    """
    Log reject action CANCELLED (when product passes)

    Args:
        reject_count: Sequential reject counter
        group_id: Capture group ID
        reason: Cancellation reason (default: "PASS")
    """
    reject_log = _init_reject_logger()

    reject_log.info(
        f"REJECT_CANCEL | #{reject_count:04d} | Group #{group_id} | "
        f"Reason: {reason}"
    )
