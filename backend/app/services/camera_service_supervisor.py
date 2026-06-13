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
_AI_UNIT = "ocr-ai.service"

# ── Internal state ───────────────────────────────────────────────────────────
_recovery_lock = asyncio.Lock()
_recovery_in_progress = False
_last_restart_ts: float = 0.0


def _import_emit():
    """Lazy import to avoid circular import at module load time."""
    from app.services.socketio_service import emit_camera_service_status
    return emit_camera_service_status


def _is_connected() -> bool:
    from app.api.websocket.camera_ws import camera_ws_manager
    return camera_ws_manager.is_connected()


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
        # 1. Give the service a chance to self-heal (auto-reconnect / systemd).
        logger.info("Camera service down — waiting %.0fs for self-heal", _GRACE_SECONDS)
        waited = 0.0
        step = 1.0
        while waited < _GRACE_SECONDS:
            await asyncio.sleep(step)
            waited += step
            if _is_connected():
                logger.info("Camera service self-healed after %.0fs", waited)
                return  # connect() already emitted connected:true

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
