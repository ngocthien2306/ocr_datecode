"""
System control endpoints — manage AI camera service lifecycle and trigger
system restart from the ML Training Studio.

The AI camera service is launched by `start_services.sh` as a plain detached
Python process (NOT a separate systemd unit despite the unit file existing),
so we manage it via pkill/spawn rather than systemctl. Only the full restart
(systemctl restart ocr-all) needs sudo — auto-input password "1" via stdin
the same way ai_services/camera_management/trigger_handler.py:_restart_service
does today.

Flag file `/tmp/ocr_ai_stopped_for_training.lock` tracks "we stopped AI for
training" intent. Created on stop, deleted on start, and cleaned up by
backend startup hook in case backend crashed without resetting state.
"""
import asyncio
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.dependencies.auth import get_current_user
from app.models.user import UserInDB
from fastapi import Depends

logger = logging.getLogger(__name__)
router = APIRouter()


# ─────────────────────────────────────────────── Constants

AI_SERVICE_PROCESS_NAME = "camera_management_service.py"

# Persisted across backend restarts so we know on next launch whether the user
# intentionally stopped AI (vs. ai-services dying for other reasons).
TRAINING_MODE_FLAG = Path("/tmp/ocr_ai_stopped_for_training.lock")

_USER_HOME = Path(os.environ.get("HOME", "/home/suntech"))
AI_SERVICE_DIR = _USER_HOME / "Source/ocr_datecode/ai_services"
AI_SERVICE_LOG = _USER_HOME / "Source/ocr_datecode/logs/start_services/ai_camera.log"


# ─────────────────────────────────────────────── Helpers

def _ai_service_pids() -> list[int]:
    """Find all running AI service process PIDs by name. Empty list if none."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", AI_SERVICE_PROCESS_NAME],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode != 0:
            return []
        return [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        logger.exception("[system] pgrep failed")
        return []


def _is_ai_service_active() -> bool:
    return len(_ai_service_pids()) > 0


def _kill_ai_service(graceful_timeout_s: float = 10.0) -> Dict[str, Any]:
    """SIGTERM → wait → SIGKILL fallback. Returns summary dict."""
    pids = _ai_service_pids()
    if not pids:
        return {"killed": 0, "method": "noop", "remaining": []}

    # Graceful SIGTERM first — lets the service cleanup ring buffer + camera.
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            logger.exception(f"[system] SIGTERM failed for pid={pid}")

    # Poll for exit
    deadline = time.time() + graceful_timeout_s
    while time.time() < deadline:
        if not _is_ai_service_active():
            return {"killed": len(pids), "method": "TERM", "remaining": []}
        time.sleep(0.5)

    # Force kill survivors
    survivors = _ai_service_pids()
    for pid in survivors:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            logger.exception(f"[system] SIGKILL failed for pid={pid}")
    time.sleep(1.0)
    return {
        "killed": len(pids),
        "method": "KILL" if survivors else "TERM",
        "remaining": _ai_service_pids(),
    }


def _spawn_ai_service() -> int:
    """Spawn detached AI service, returns new PID. Inherits current env."""
    AI_SERVICE_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(AI_SERVICE_LOG, "a")
    process = subprocess.Popen(
        ["python3", AI_SERVICE_PROCESS_NAME],
        cwd=str(AI_SERVICE_DIR),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,         # detach — survives backend exit
        env=os.environ.copy(),          # inherits CUDA / LD_LIBRARY_PATH from BE env
    )
    return process.pid


def _trigger_restart_all() -> None:
    """
    Fire `sudo systemctl restart ocr-all` with hardcoded password. Returns
    immediately — the restart kills this backend process, so we cannot await.
    Reuses the same pattern as trigger_handler._restart_service.
    """
    subprocess.Popen(
        ["bash", "-c",
         'sleep 1; echo "1" | sudo -S systemctl restart ocr-all >/dev/null 2>&1'],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def cleanup_stale_training_flag() -> bool:
    """Called from BE startup. Deletes flag if leftover from previous crash."""
    if TRAINING_MODE_FLAG.exists():
        try:
            TRAINING_MODE_FLAG.unlink()
            logger.info(f"[system] cleaned stale training-mode flag: {TRAINING_MODE_FLAG}")
            return True
        except Exception:
            logger.exception(f"[system] failed to unlink stale flag")
    return False


# ─────────────────────────────────────────────── Endpoints

@router.get("/health", tags=["System"])
async def api_health():
    """Lightweight health-check used by FE polling during restart."""
    return {"status": "ok"}


class _AIServiceStatus(BaseModel):
    active: bool
    pids: list[int]
    in_training_mode: bool


@router.get("/system/ai-service/status", tags=["System"])
async def ai_service_status(current_user: UserInDB = Depends(get_current_user)) -> _AIServiceStatus:
    pids = _ai_service_pids()
    return _AIServiceStatus(
        active=len(pids) > 0,
        pids=pids,
        in_training_mode=TRAINING_MODE_FLAG.exists(),
    )


@router.post("/system/ai-service/stop", tags=["System"])
async def ai_service_stop(current_user: UserInDB = Depends(get_current_user)):
    """
    Stop AI camera service so training can use its memory. Sets the
    training-mode flag so the matching /start call (or auto-cleanup) knows
    this was an intentional stop.
    """
    result = await asyncio.get_event_loop().run_in_executor(None, _kill_ai_service)

    # Mark intent so /start auto-fires on training-studio close
    try:
        TRAINING_MODE_FLAG.write_text(str(int(time.time())))
    except Exception:
        logger.exception("[system] failed to write training-mode flag")

    return {
        "stopped": True,
        **result,
        "flag_path": str(TRAINING_MODE_FLAG),
    }


@router.post("/system/ai-service/start", tags=["System"])
async def ai_service_start(current_user: UserInDB = Depends(get_current_user)):
    """
    Spawn AI camera service back up. Deletes the training-mode flag so we
    don't keep auto-starting it on every studio close.
    """
    if _is_ai_service_active():
        # Already running — nothing to do but clear stale flag
        if TRAINING_MODE_FLAG.exists():
            try:
                TRAINING_MODE_FLAG.unlink()
            except Exception:
                pass
        return {"started": False, "reason": "already running", "pids": _ai_service_pids()}

    pid = await asyncio.get_event_loop().run_in_executor(None, _spawn_ai_service)

    if TRAINING_MODE_FLAG.exists():
        try:
            TRAINING_MODE_FLAG.unlink()
        except Exception:
            pass

    return {"started": True, "pid": pid}


@router.post("/system/restart-all", tags=["System"])
async def restart_all(current_user: UserInDB = Depends(get_current_user)):
    """
    Trigger `sudo systemctl restart ocr-all`. This will kill the current
    backend process; the response goes out before the kill thanks to the
    1-second sleep inside the bash one-liner. FE polls /api/health afterwards
    to detect when the new backend is up.
    """
    _trigger_restart_all()
    return {"triggered": True, "note": "Backend will restart in ~1s"}
