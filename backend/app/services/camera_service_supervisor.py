"""
Camera Service Supervisor
==========================

Centralised detection + recovery for the Camera Management (AI) service.

Why this exists
---------------
~15 BE endpoints used to do, inline:

    if not camera_ws_manager.is_connected():
        raise HTTPException(503, "Camera Management service is not connected")

That only *reported* the failure. The UI showed a toast but took no action, and
the service was never recovered. This module centralises the behaviour so every
endpoint can call ONE helper that:

  1. Tells the frontend immediately (emit ``camera_service_status {connected:false}``)
     so the global ``ServiceDownOverlay`` shows.
  2. Kicks off a *debounced* background recovery instead of blindly killing the
     process. The AI service already auto-reconnects its WebSocket (exponential
     backoff 1s→60s), and ``ocr-ai.service`` runs with ``Restart=always``. So:
       - WS dropped but process alive  → it reconnects on its own within seconds.
       - process crashed               → systemd brings it back.
       - process hung (alive, WS stuck)→ THIS is the only case a forced restart
                                          actually helps.
     We therefore wait a short grace period for self-healing, and only escalate
     to a restart if the service is *still* down. A debounce window prevents a
     burst of failing API calls from triggering a "restart storm" that would
     kill a process mid-reconnect.
  3. Raises HTTP 503 so the caller still gets a clean error.

When the AI service (re)connects, ``camera_ws_manager.connect()`` emits
``camera_service_status {connected:true}`` which hides the overlay.
"""

import asyncio
import logging
import subprocess
import time
from typing import Optional

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

# ── Tunables ────────────────────────────────────────────────────────────────
# How long to let the AI service self-heal (auto-reconnect / systemd restart)
# before BE forcibly restarts it.
_GRACE_SECONDS = 15.0
# Minimum spacing between two BE-initiated restarts. Guards against restart
# storms when many API calls fail back-to-back while the service is down.
_MIN_RESTART_INTERVAL = 60.0
# systemd unit name for the AI camera service.
# NOTE: under this deployment the camera runs as a plain nohup process (started by
# start_services.sh), NOT this unit — so _do_restart_blocking() falls through to
# kill+respawn via service_tools. Leaving the name as-is keeps that fallback path.
_AI_UNIT = "ocr-ai.service"

# ── Internal state ───────────────────────────────────────────────────────────
_recovery_lock = asyncio.Lock()
_recovery_in_progress = False
_last_restart_ts: float = 0.0
# Stale-shm recovery (cache-clear + camera kill/respawn) gets its own debounce so
# a burst of failing get-frame calls (e.g. several cameras at once) only recovers
# once. Because the backend is NOT restarted, this state survives across the whole
# recovery — which is what prevents a restart loop.
_shm_recovery_in_progress = False
_last_shm_recovery_ts: float = 0.0


def _import_emit():
    """Lazy import to avoid circular import at module load time."""
    from app.services.socketio_service import emit_camera_service_status
    return emit_camera_service_status


def _is_connected() -> bool:
    from app.api.websocket.camera_ws import camera_ws_manager
    return camera_ws_manager.is_connected()


def _camera_process_alive() -> bool:
    """
    True if a `camera_management_service.py` process exists. Lets recovery tell
    apart 'process crashed/dead' (restart immediately, grace is pointless) from
    'process alive but WS blipped' (wait grace, it auto-reconnects). Conservative:
    on any error returns True so we fall back to the safe grace path.
    """
    try:
        import psutil
        for proc in psutil.process_iter(['cmdline']):
            try:
                cmd = proc.info.get('cmdline') or []
                if any('camera_management_service.py' in str(c) for c in cmd):
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False
    except Exception:
        return True


# ── Restart primitives (run in a worker thread — they block) ─────────────────

def _systemd_unit_installed() -> bool:
    """
    True if the unit file exists at all — regardless of enable state. ocr-ai is
    deliberately *static* (no [Install]); it's started by start_services.sh after
    the camera check, but is still systemctl-restartable with Restart=always.
    """
    try:
        out = subprocess.run(
            ["systemctl", "list-unit-files", _AI_UNIT, "--no-legend"],
            capture_output=True, text=True, timeout=5,
        )
        # Any state (enabled/disabled/static) means the unit is installed.
        # An absent unit yields empty stdout.
        return _AI_UNIT in out.stdout and "not-found" not in out.stdout
    except Exception:
        return False


def _restart_via_systemd() -> bool:
    """Try a clean `sudo -n systemctl restart`. Returns True on success."""
    try:
        res = subprocess.run(
            ["sudo", "-n", "systemctl", "restart", _AI_UNIT],
            capture_output=True, text=True, timeout=30,
        )
        if res.returncode == 0:
            logger.info("AI service restarted via systemctl")
            return True
        logger.warning(
            "systemctl restart failed (rc=%s): %s",
            res.returncode, (res.stderr or "").strip(),
        )
        return False
    except Exception as e:
        logger.warning("systemctl restart errored: %s", e)
        return False


def _restart_via_kill_respawn() -> bool:
    """
    Fallback when systemctl is unavailable / sudo needs a password.

    If the unit IS systemd-managed, simply killing the process is enough:
    systemd's Restart=always brings it back. If it's NOT managed (manual run),
    we stop then start it ourselves via the existing psutil-based tools.
    """
    try:
        from app.agent.tools.service_tools import stop_service, start_service
        stop_res = stop_service("camera_management")
        logger.info("kill_respawn: stop → %s", stop_res.get("message"))

        if _systemd_unit_installed():
            # systemd will respawn it (Restart=always). Don't double-start.
            logger.info("kill_respawn: relying on systemd Restart=always")
            return True

        # Manual mode — start it back ourselves.
        time.sleep(2)
        start_res = start_service("camera_management")
        logger.info("kill_respawn: start → %s", start_res.get("message"))
        return bool(start_res.get("success"))
    except Exception as e:
        logger.error("kill_respawn failed: %s", e)
        return False


def _do_restart_blocking() -> bool:
    if _systemd_unit_installed():
        if _restart_via_systemd():
            return True
        logger.info("Falling back to kill+respawn (systemd path failed)")
    return _restart_via_kill_respawn()


# ── Recovery orchestration ───────────────────────────────────────────────────

async def _recover():
    """
    Grace-then-restart recovery. Guarded so only one recovery runs at a time
    and restarts are spaced by _MIN_RESTART_INTERVAL.
    """
    global _recovery_in_progress, _last_restart_ts

    if _recovery_in_progress:
        return
    async with _recovery_lock:
        if _recovery_in_progress:
            return
        _recovery_in_progress = True

    try:
        # 1. Give the service a chance to self-heal (auto-reconnect WS).
        #    BUT if the process is actually DEAD, there's nothing to self-heal —
        #    skip the grace and restart immediately (this is what makes a real
        #    crash recover fast instead of waiting the full grace window).
        logger.info("Camera service down — waiting up to %.0fs for self-heal", _GRACE_SECONDS)
        waited = 0.0
        step = 1.0
        while waited < _GRACE_SECONDS:
            await asyncio.sleep(step)
            waited += step
            if _is_connected():
                logger.info("Camera service self-healed after %.0fs", waited)
                return  # connect() already emitted connected:true
            if not await asyncio.to_thread(_camera_process_alive):
                logger.warning("Camera service PROCESS is dead — skipping grace, restarting now")
                break

        # 2. Still down. Restart — but respect the debounce window.
        now = time.monotonic()
        since = now - _last_restart_ts
        if since < _MIN_RESTART_INTERVAL:
            logger.warning(
                "Skipping restart — last restart was %.0fs ago (< %.0fs debounce)",
                since, _MIN_RESTART_INTERVAL,
            )
            return

        logger.warning("Camera service still down — forcing restart of %s", _AI_UNIT)
        _last_restart_ts = now
        ok = await asyncio.to_thread(_do_restart_blocking)
        logger.info("AI service restart %s", "succeeded" if ok else "FAILED")
    finally:
        _recovery_in_progress = False
        # Reconcile the overlay with reality on EVERY exit path. `connected:false`
        # is edge-triggered from many callers (require_camera_service, watchdog,
        # send failures), but `connected:true` is otherwise only emitted when the
        # AI-service WS actually re-accepts. Re-announcing the true current status
        # here guarantees a {connected:true} eventually reaches the FE once the
        # service is back — so the ServiceDownOverlay can't stay stuck showing a
        # down state that no longer reflects reality (self-heal / debounce-skip
        # paths previously emitted nothing).
        try:
            emit = _import_emit()
            await emit({
                "connected": _is_connected(),
                "message": "Camera service status (post-recovery)",
            })
        except Exception as e:
            logger.error("Recovery reconcile emit failed: %s", e)


async def notify_service_down(reason: str = "Camera Management Service unreachable"):
    """
    Announce the outage to the frontend and start (debounced) recovery in the
    background. Does NOT block the caller.
    """
    try:
        emit = _import_emit()
        await emit({"connected": False, "message": reason})
    except Exception as e:
        logger.error("Failed to emit camera_service_status down: %s", e)

    # Fire-and-forget recovery.
    asyncio.create_task(_recover())


async def require_camera_service(reason: Optional[str] = None):
    """
    Endpoint guard. Use in place of the old inline check::

        await require_camera_service()

    If the AI service is connected, returns silently. Otherwise it notifies the
    UI, triggers background recovery, and raises HTTP 503.
    """
    if _is_connected():
        return
    await notify_service_down(reason or "Camera Management Service is not connected")
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Camera Management Service is not connected. Restarting — please retry shortly.",
    )


# ── Stale shared-memory recovery (cache-clear + camera respawn) ───────────────
# Recovery path for the "stale shared-memory" failure mode: the camera is still
# grabbing (DB says connected, AI service is connected) but the BE reads no frame
# from shared memory. Root cause: the AI service unlink()s + recreates its shm
# segment on every (re)connect, while SharedMemoryService caches its handle and
# keeps reading the OLD, orphaned segment (or fails to find the unlinked name).
#
# The cache lives in THIS (backend) process, so we don't need to restart the
# backend at all — we just drop the cached handle in-process. We then kill+respawn
# the camera (a plain nohup process here, restarted via service_tools — the same
# path the existing crash-recovery uses) so a fresh, correctly-named segment is
# created. The next get-frame re-attaches to it.

async def recover_stale_shm(serial_number: str, reason: str):
    """
    Recover from the stale-shm condition WITHOUT restarting the backend:
      1. Drop the (possibly stale) cached shm handle for this camera.
      2. Kill + respawn the camera service so it recreates a fresh segment.
    Debounced (single-flight + _MIN_RESTART_INTERVAL) so a burst of failing
    get-frame calls only recovers once. Fire-and-forget: returns immediately so
    the caller's 404 still reaches the FE.
    """
    global _shm_recovery_in_progress, _last_shm_recovery_ts

    if _shm_recovery_in_progress:
        return
    now = time.monotonic()
    since = now - _last_shm_recovery_ts
    if since < _MIN_RESTART_INTERVAL:
        logger.warning(
            "Skipping shm recovery — last recovery was %.0fs ago (< %.0fs debounce)",
            since, _MIN_RESTART_INTERVAL,
        )
        return
    _shm_recovery_in_progress = True
    _last_shm_recovery_ts = now

    logger.warning("STALE-SHM RECOVERY triggered: %s", reason)

    # Tell the UI (the camera kill below also drops its WS, which shows the overlay
    # via camera_service_status — emit here too for an immediate popup).
    try:
        emit = _import_emit()
        await emit({"connected": False, "message": reason})
    except Exception as e:
        logger.error("Shm recovery: emit failed: %s", e)

    asyncio.create_task(_do_shm_recovery(serial_number))


async def _do_shm_recovery(serial_number: str):
    global _shm_recovery_in_progress
    try:
        from app.services.shared_memory_service import shared_memory_service
        from app.agent.tools.service_tools import stop_service, start_service

        # 1. Drop the stale cached handle so the next read re-attaches fresh.
        shared_memory_service.cleanup(serial_number)

        # 2. Kill + respawn the camera (blocking psutil/Popen work → run in thread).
        stop_res = await asyncio.to_thread(stop_service, "camera_management")
        logger.info("Shm recovery: stop camera → %s", stop_res.get("message"))
        await asyncio.sleep(2)
        start_res = await asyncio.to_thread(start_service, "camera_management")
        logger.info("Shm recovery: start camera → %s", start_res.get("message"))

        # 3. Drop the handle again so the next read attaches to the freshly created
        #    segment rather than any handle re-cached during the window above.
        shared_memory_service.cleanup(serial_number)
        logger.info("Shm recovery: done for %s", serial_number)

        # The AI-service WS never dropped during shm recovery (stale-shm means the
        # service IS connected — only its shared-memory segment went stale), so no
        # camera_service_status {connected:true} was ever emitted to clear the
        # overlay we showed at the start (line ~308). Re-announce current status
        # now so the ServiceDownOverlay hides itself instead of staying stuck until
        # the user manually reloads.
        try:
            emit = _import_emit()
            await emit({
                "connected": _is_connected(),
                "message": "Camera service recovered (shared-memory)",
            })
        except Exception as e:
            logger.error("Shm recovery: recovered-emit failed: %s", e)
    except Exception as e:
        logger.error("Shm recovery failed for %s: %s", serial_number, e)
    finally:
        _shm_recovery_in_progress = False


async def handle_missing_frame(serial_number: str, is_connected_in_db: bool):
    """
    Call from a get-frame endpoint when a shared-memory read returned no frame.

    Distinguishes the two reasons a read can come back empty:
      (A) the camera was never connected / isn't streaming yet  → benign, no-op.
      (B) the camera SHOULD be live (DB connected + AI service connected) but the
          frame is missing → the stale-shm bug → clear cache + respawn camera.

    Only case (B) triggers recovery, so merely opening a recipe before a camera
    is connected never restarts anything.
    """
    if not is_connected_in_db:
        return  # (A) camera not marked connected — nothing should be streaming
    if not _is_connected():
        return  # camera service itself is down — handled by require_camera_service
    # (B) camera is supposed to be live but shm has no frame → recover.
    await recover_stale_shm(
        serial_number,
        f"Camera {serial_number} is connected but no frame in shared memory "
        f"(stale shm) — clearing cache + restarting camera",
    )


# ── Watchdog ─────────────────────────────────────────────────────────────────
# Proactively detect a dead/crashed AI camera service WITHOUT waiting for a user
# API call. Without this, if the service crashes overnight and nobody touches the
# UI, the line keeps running with no inspection until someone notices.

_WATCHDOG_INTERVAL = 8.0
_watchdog_task: Optional[asyncio.Task] = None


async def _watchdog_loop(interval: float):
    announced_down = False
    while True:
        try:
            await asyncio.sleep(interval)

            if _is_connected():
                announced_down = False
                continue

            # Camera service is down. Announce ONCE per outage (FE toast is
            # edge-triggered anyway), then keep nudging recovery each tick.
            if not announced_down:
                announced_down = True
                logger.warning("Watchdog: camera service DOWN — notifying UI + recovering")
                try:
                    emit = _import_emit()
                    await emit({
                        "connected": False,
                        "message": "Camera Management Service down (watchdog)",
                    })
                except Exception as e:
                    logger.error("Watchdog emit failed: %s", e)

            # _recover() is guarded (single-flight + debounce) so calling it every
            # tick is safe; it restarts as soon as the debounce window allows.
            asyncio.create_task(_recover())

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Watchdog loop error: %s", e)


async def start_watchdog(interval: float = _WATCHDOG_INTERVAL):
    """Start the periodic camera-service watchdog. Idempotent — call once at startup."""
    global _watchdog_task
    if _watchdog_task is not None and not _watchdog_task.done():
        return
    _watchdog_task = asyncio.create_task(_watchdog_loop(interval))
    logger.info("Camera service watchdog started (interval=%.0fs)", interval)
