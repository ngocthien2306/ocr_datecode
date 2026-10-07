"""
Cross-process GPU lock.

There is one GPU on this workstation and four things want it: live inference in
ai_services (~3.5 GB, permanently resident), OCR fine-tuning (~2.6 GB at
batch 32), TensorRT engine builds (~0.6 GB), and anomaly training (which can
exceed 10 GB — see docs/memory/anomaly-training.md). OCR training coexists with
live inference comfortably; two training runs at once do not.

flock is the right primitive here: it is atomic, and the kernel releases it when
the holder dies for any reason including SIGKILL, so a crashed run cannot wedge
the queue. Same reasoning as camera_management_service.py's single-instance lock.

SCOPE: today only ocr_service takes this lock, so it serialises OCR runs against
each other and against OCR engine builds. It does NOT yet protect against
anomaly_service, which has no GPU arbitration of its own — that needs the same
context manager on its training path, pointed at the same file. Until then, two
operators training in the two studios simultaneously can still OOM both runs.
"""
import fcntl
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Deliberately outside either service's tree so both can reach it without
# knowing where the other is installed. Overridable for tests.
GPU_LOCK_PATH = Path(os.environ.get("OCR_GPU_LOCK_PATH", "/tmp/ocr_datecode_gpu.lock"))


class GPUBusy(RuntimeError):
    pass


@contextmanager
def gpu_lock(
    label: str,
    timeout: Optional[float] = None,
    poll_interval: float = 2.0,
    on_wait: Optional[Callable[[float], None]] = None,
):
    """Hold the GPU lock for the duration of the block.

    Blocks until the lock is free (or `timeout` seconds pass, then raises
    GPUBusy). Blocking rather than failing fast is deliberate: an operator
    starts a run and walks away, and a queued run that eventually starts is far
    better than one that failed thirty seconds after they stopped watching.

    on_wait(waited_seconds) fires on each poll while queued, so the caller can
    surface a waiting_for_gpu phase instead of the UI showing a run that appears
    stuck at 0%.
    """
    GPU_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(GPU_LOCK_PATH, "a+")
    started = time.monotonic()
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                waited = time.monotonic() - started
                if timeout is not None and waited >= timeout:
                    raise GPUBusy(
                        f"GPU still busy after {waited:.0f}s (holder: {_holder(fh)})"
                    )
                if on_wait is not None:
                    on_wait(waited)
                time.sleep(poll_interval)

        # Who holds it, for the error message the next waiter sees. Best effort:
        # a stale line is harmless, the flock itself is the source of truth.
        try:
            fh.seek(0)
            fh.truncate()
            fh.write(f"{label} pid={os.getpid()} since={time.time():.0f}\n")
            fh.flush()
        except OSError:
            pass

        logger.info(f"[gpu_lock] acquired by {label} after {time.monotonic() - started:.1f}s")
        yield
    finally:
        if acquired:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            logger.info(f"[gpu_lock] released by {label}")
        fh.close()


def _holder(fh) -> str:
    try:
        fh.seek(0)
        return fh.read().strip() or "unknown"
    except OSError:
        return "unknown"


def current_holder() -> Optional[str]:
    """Who holds the lock right now, or None if it is free. Racy by nature —
    only for showing "waiting for <x>" in the UI, never for deciding to skip
    the lock."""
    if not GPU_LOCK_PATH.exists():
        return None
    with open(GPU_LOCK_PATH, "a+") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            return None
        except BlockingIOError:
            return _holder(fh)
