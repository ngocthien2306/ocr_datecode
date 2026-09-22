"""Storage auto-cleanup scheduler.

- Periodically checks disk usage of the uploads partition
- If usage exceeds the configured threshold, runs `StorageService.run_auto_cleanup`
- Two schedule modes:
    * daily  → fires once per day at HH:MM
    * hourly → fires every N hours
- Re-reads its config every cycle so changes via the API take effect on the
  next loop iteration. Call `reschedule()` from the API to wake the loop
  immediately and recompute the next-firing time.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from app.models.storage import CleanupConfig, CleanupRunResult

logger = logging.getLogger(__name__)

# Defaults mirror CleanupConfig — kept here to avoid circular imports with the
# endpoint module. Source of truth for validation is still the pydantic model.
DEFAULT_CLEANUP_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "dry_run": True,
    "schedule_mode": "daily",
    "schedule_hour": 2,
    "schedule_minute": 0,
    "interval_hours": 6,
    "trigger_usage_percent": 70.0,
    "target_usage_percent": 60.0,
    "min_images_per_folder": 25,
    "keep_recent_dates_per_recipe": 3,
}

_task: Optional[asyncio.Task] = None
_wake_event: Optional[asyncio.Event] = None
_service = None  # injected by start()
_config_path: Optional[Path] = None
_last_run_path: Optional[Path] = None


def _load_config() -> Dict[str, Any]:
    if _config_path is None or not _config_path.exists():
        return DEFAULT_CLEANUP_CONFIG.copy()
    try:
        data = json.loads(_config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return DEFAULT_CLEANUP_CONFIG.copy()
    merged = DEFAULT_CLEANUP_CONFIG.copy()
    merged.update({k: data[k] for k in DEFAULT_CLEANUP_CONFIG if k in data})
    return merged


def save_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if _config_path is None:
        raise RuntimeError("storage_cleanup_scheduler not initialised")
    _config_path.parent.mkdir(parents=True, exist_ok=True)
    _config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


def load_config() -> Dict[str, Any]:
    return _load_config()


def save_last_run(result: CleanupRunResult) -> None:
    if _last_run_path is None:
        return
    try:
        _last_run_path.parent.mkdir(parents=True, exist_ok=True)
        _last_run_path.write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
    except OSError as e:
        logger.error(f"[storage_cleanup] could not save last_run: {e}")


def load_last_run() -> Optional[Dict[str, Any]]:
    if _last_run_path is None or not _last_run_path.exists():
        return None
    try:
        return json.loads(_last_run_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _next_run_at(cfg: Dict[str, Any]) -> datetime:
    """Compute next datetime the cleanup should fire."""
    mode = cfg.get("schedule_mode", "daily")
    now = datetime.now()
    if mode == "hourly":
        interval = max(1, int(cfg.get("interval_hours", 6)))
        # Anchor next run from last_run if available so the cadence is stable;
        # otherwise from now + interval.
        last = load_last_run()
        if last and last.get("ran_at"):
            try:
                base = datetime.fromisoformat(last["ran_at"])
            except ValueError:
                base = now
        else:
            base = now
        nxt = base + timedelta(hours=interval)
        if nxt <= now:
            nxt = now + timedelta(hours=interval)
        return nxt
    # daily
    hour = int(cfg.get("schedule_hour", 2))
    minute = int(cfg.get("schedule_minute", 0))
    target = datetime.combine(now.date(), time(hour, minute))
    if target <= now:
        target = target + timedelta(days=1)
    return target


def _run_cleanup_sync(cfg_dict: Dict[str, Any], trigger: str) -> CleanupRunResult:
    """Run cleanup synchronously (called via asyncio.to_thread)."""
    assert _service is not None
    cfg = CleanupConfig(**cfg_dict)
    result = _service.run_auto_cleanup(cfg, trigger=trigger)
    save_last_run(result)
    logger.info(
        f"[storage_cleanup] {result.message} "
        f"(scanned={result.leaf_folders_scanned}, duration={result.duration_seconds}s)"
    )
    return result


async def run_now(trigger: str = "manual") -> CleanupRunResult:
    """Run cleanup immediately (used by the manual-trigger endpoint)."""
    cfg = _load_config()
    return await asyncio.to_thread(_run_cleanup_sync, cfg, trigger)


async def _scheduler_loop() -> None:
    assert _wake_event is not None
    logger.info("[storage_cleanup] scheduler started")
    while True:
        cfg = _load_config()
        if not cfg.get("enabled"):
            try:
                await asyncio.wait_for(_wake_event.wait(), timeout=3600)
            except asyncio.TimeoutError:
                pass
            _wake_event.clear()
            continue

        next_run = _next_run_at(cfg)
        wait_s = max(1.0, (next_run - datetime.now()).total_seconds())
        logger.info(
            f"[storage_cleanup] next run at {next_run.isoformat(timespec='seconds')} "
            f"(in {int(wait_s)}s)"
        )

        try:
            await asyncio.wait_for(_wake_event.wait(), timeout=wait_s)
            # Woken early (config changed) — recompute next run
            _wake_event.clear()
            continue
        except asyncio.TimeoutError:
            pass

        try:
            await asyncio.to_thread(_run_cleanup_sync, _load_config(), "scheduled")
        except Exception as e:
            logger.error(f"[storage_cleanup] run failed: {e}", exc_info=True)


def start(service, config_path: Path, last_run_path: Path) -> None:
    """Start the scheduler task on the current event loop."""
    global _task, _wake_event, _service, _config_path, _last_run_path
    _service = service
    _config_path = config_path
    _last_run_path = last_run_path
    if _task and not _task.done():
        return
    _wake_event = asyncio.Event()
    _task = asyncio.create_task(_scheduler_loop(), name="storage_cleanup_scheduler")


async def stop() -> None:
    global _task
    if _task and not _task.done():
        _task.cancel()
        try:
            await _task
        except (asyncio.CancelledError, Exception):
            pass
    _task = None


def reschedule() -> None:
    """Signal the loop to re-read config and recompute next-run time."""
    if _wake_event is not None:
        _wake_event.set()
