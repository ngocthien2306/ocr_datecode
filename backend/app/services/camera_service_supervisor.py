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
import os
import signal
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
_AI_UNIT = "ocr-ai.service"
# Grace before the backend self-terminates during a FULL restart. Gives the
# in-flight HTTP response time to flush (so the FE still gets its 404) and the
# camera_service_status emit time to reach connected clients before the socket
# drops. systemd (Restart=always) respawns the backend afterwards.
_SELF_EXIT_DELAY = 2.0

# ── Internal state ───────────────────────────────────────────────────────────
_recovery_lock = asyncio.Lock()
_recovery_in_progress = False
_last_restart_ts: float = 0.0
# Full restart (backend + camera) is a heavier, separate operation from the
# WS-only recovery above; it gets its own debounce so a burst of failing
# get-frame calls (e.g. several cameras at once) only restarts the line once.
_full_restart_in_progress = False
_last_full_restart_ts: float = 0.0


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


# ── Full restart (backend + camera) ──────────────────────────────────────────
# Recovery path for the "stale shared-memory" failure mode: the camera is still
# grabbing (DB says connected, AI service is connected) but the BE reads no frame
# from shared memory. This happens because the AI service unlink()s + recreates
# its shm segment on every (re)connect, while SharedMemoryService caches its
# handle forever and keeps reading the OLD, orphaned segment. The cache lives in
# the BACKEND process, so restarting the camera alone never clears it — the
# backend must restart too. We therefore restart both.

def _self_terminate_backend():
    """SIGTERM ourselves so systemd (Restart=always) respawns the backend with a
    clean SharedMemoryService cache."""
    logger.warning("Self-terminating backend for full restart — systemd will respawn it")
    os.kill(os.getpid(), signal.SIGTERM)


async def restart_backend_and_camera(reason: str):
    """
    Kill + restart BOTH the backend and the camera (AI) service. Used when a
    frame is missing from shared memory despite the camera being live (stale-shm
    bug). Debounced so concurrent failing requests trigger only one restart.

    Flow:
      1. Announce the outage (popup) — though the backend death below also makes
         the FE socket drop, which shows the overlay regardless.
      2. Restart the camera service in a DETACHED process so it survives the
         backend's own death moments later.
      3. Self-terminate the backend after a short grace (so the current 404
         response still reaches the FE); systemd respawns it with a fresh cache.
    """
    global _full_restart_in_progress, _last_full_restart_ts

    if _full_restart_in_progress:
        return
    now = time.monotonic()
    since = now - _last_full_restart_ts
    if since < _MIN_RESTART_INTERVAL:
        logger.warning(
            "Skipping full restart — last full restart was %.0fs ago (< %.0fs debounce)",
            since, _MIN_RESTART_INTERVAL,
        )
        return
    _full_restart_in_progress = True
    _last_full_restart_ts = now

    logger.warning("FULL RESTART (backend + camera) triggered: %s", reason)

    # 1. Tell the UI (best-effort — backend death below shows the overlay anyway).
    try:
        emit = _import_emit()
        await emit({"connected": False, "message": reason})
    except Exception as e:
        logger.error("Full restart: emit failed: %s", e)

    # 2. Restart the camera service detached so it outlives the backend.
    #    `sudo -n systemctl restart ocr-ai.service` is already permitted (used by
    #    _restart_via_systemd). start_new_session=True detaches it from the
    #    backend's process group so our SIGTERM below doesn't take it down.
    try:
        subprocess.Popen(
            ["sh", "-c", "sleep 1; sudo -n systemctl restart ocr-ai.service"],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("Full restart: camera service restart spawned (detached)")
    except Exception as e:
        logger.error("Full restart: failed to spawn camera restart: %s", e)

    # 3. Schedule backend self-termination after the response flushes.
    try:
        loop = asyncio.get_running_loop()
        loop.call_later(_SELF_EXIT_DELAY, _self_terminate_backend)
    except Exception as e:
        logger.error("Full restart: failed to schedule self-exit: %s", e)


async def handle_missing_frame(serial_number: str, is_connected_in_db: bool):
    """
    Call from a get-frame endpoint when a shared-memory read returned no frame.

    Distinguishes the two reasons a read can come back empty:
      (A) the camera was never connected / isn't streaming yet  → benign, no-op.
      (B) the camera SHOULD be live (DB connected + AI service connected) but the
          frame is missing → the stale-shm bug → restart backend + camera.

    Only case (B) triggers a restart, so merely opening a recipe before a camera
    is connected never restarts the line.
    """
    if not is_connected_in_db:
        return  # (A) camera not marked connected — nothing should be streaming
    if not _is_connected():
        return  # camera service itself is down — handled by require_camera_service
    # (B) camera is supposed to be live but shm has no frame → recover hard.
    await restart_backend_and_camera(
        f"Camera {serial_number} is connected but no frame in shared memory "
        f"(stale shm) — restarting backend + camera"
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
