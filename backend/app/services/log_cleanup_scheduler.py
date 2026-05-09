"""
Daily log cleanup scheduler.

- Runs once per day at the user-configured hour:minute
- Compresses .log files older than `compress_after_days` days into .log.gz
- Deletes any file (raw or .gz) older than `keep_days` days
- Re-reads its config every cycle, so changes via the API take effect on next run
- Reschedule signal: if the API updates the schedule time, call reschedule()
  to wake the loop and recompute the next firing
"""
from __future__ import annotations

import asyncio
import gzip
import logging
import re
import shutil
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from app.api.endpoints.system_logs import (
    DEFAULT_CLEANUP_CONFIG,
    _load_cleanup_config,
)
from app.utils.logging_config import CATEGORIES, LOGS_ROOT

logger = logging.getLogger(__name__)

DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.log(\.gz)?$")

_task: Optional[asyncio.Task] = None
_wake_event: Optional[asyncio.Event] = None


def _file_date(name: str) -> Optional[date]:
    m = DATE_RE.match(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d").date()
    except ValueError:
        return None


def _compress_file(path: Path) -> bool:
    """Gzip a .log file in place. Returns True on success."""
    gz_path = path.with_suffix(path.suffix + ".gz")
    try:
        with open(path, "rb") as src, gzip.open(gz_path, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)
        path.unlink()
        return True
    except Exception as e:
        logger.error(f"[log_cleanup] Failed to compress {path}: {e}")
        if gz_path.exists():
            try:
                gz_path.unlink()
            except OSError:
                pass
        return False


def run_cleanup(cfg: Dict[str, Any]) -> Dict[str, int]:
    """
    Walk every category folder and apply compression + retention.
    Returns counters for logging/observability.
    """
    today = date.today()
    keep_days = int(cfg.get("keep_days", 30))
    compress_after_days = int(cfg.get("compress_after_days", 7))

    stats = {"compressed": 0, "deleted": 0, "scanned": 0, "errors": 0}

    for category in CATEGORIES:
        folder = LOGS_ROOT / category
        if not folder.exists():
            continue
        for p in folder.iterdir():
            if not p.is_file():
                continue
            file_date = _file_date(p.name)
            if file_date is None:
                continue
            stats["scanned"] += 1
            age_days = (today - file_date).days
            try:
                if age_days >= keep_days:
                    p.unlink()
                    stats["deleted"] += 1
                elif age_days >= compress_after_days and p.suffix == ".log":
                    if _compress_file(p):
                        stats["compressed"] += 1
                    else:
                        stats["errors"] += 1
            except Exception as e:
                logger.error(f"[log_cleanup] Error on {p}: {e}")
                stats["errors"] += 1

    logger.info(
        f"[log_cleanup] done: scanned={stats['scanned']} "
        f"compressed={stats['compressed']} deleted={stats['deleted']} "
        f"errors={stats['errors']} (keep_days={keep_days}, compress_after={compress_after_days})"
    )
    return stats


def _next_run_at(cfg: Dict[str, Any]) -> datetime:
    """Compute next datetime the cleanup should fire (today or tomorrow)."""
    hour = int(cfg.get("schedule_hour", 0))
    minute = int(cfg.get("schedule_minute", 30))
    now = datetime.now()
    target = datetime.combine(now.date(), time(hour, minute))
    if target <= now:
        target = target + timedelta(days=1)
    return target


async def _scheduler_loop() -> None:
    assert _wake_event is not None
    logger.info("[log_cleanup] scheduler started")
    while True:
        cfg = _load_cleanup_config()
        if not cfg.get("enabled"):
            # Disabled — sleep up to 1 hour, then re-check (also wakes on reschedule)
            try:
                await asyncio.wait_for(_wake_event.wait(), timeout=3600)
            except asyncio.TimeoutError:
                pass
            _wake_event.clear()
            continue

        next_run = _next_run_at(cfg)
        wait_s = max(1.0, (next_run - datetime.now()).total_seconds())
        logger.info(f"[log_cleanup] next run at {next_run.isoformat(timespec='seconds')} (in {int(wait_s)}s)")

        try:
            await asyncio.wait_for(_wake_event.wait(), timeout=wait_s)
            # Woken early (config changed) — loop and recompute
            _wake_event.clear()
            continue
        except asyncio.TimeoutError:
            pass

        # Time to run
        try:
            await asyncio.to_thread(run_cleanup, _load_cleanup_config())
        except Exception as e:
            logger.error(f"[log_cleanup] run failed: {e}", exc_info=True)


def start() -> None:
    """Spawn the scheduler task on the current event loop."""
    global _task, _wake_event
    if _task and not _task.done():
        return
    _wake_event = asyncio.Event()
    _task = asyncio.create_task(_scheduler_loop(), name="log_cleanup_scheduler")


async def stop() -> None:
    global _task
    if _task and not _task.done():
        _task.cancel()
        try:
            await _task
        except (asyncio.CancelledError, Exception):
            pass
    _task = None


def reschedule(_cfg: Dict[str, Any] | None = None) -> None:
    """Signal the loop to re-read config and recompute next-run time."""
    if _wake_event is not None:
        _wake_event.set()
