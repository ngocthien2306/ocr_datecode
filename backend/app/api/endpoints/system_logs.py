"""
System logs API.

Read-only access for any authenticated user; delete/cleanup-config restricted
to supervisor+. Logs live under {repo_root}/logs/{category}/{YYYY-MM-DD}.log
(or {YYYY-MM-DD}.log.gz when compressed by the cleanup scheduler).
"""
from __future__ import annotations

import asyncio
import gzip
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, StreamingResponse

from app.api.dependencies.auth import get_current_user, require_supervisor
from app.models.user import UserInDB
from app.utils.logging_config import CATEGORIES, LOGS_ROOT

router = APIRouter()

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
META_DIR = LOGS_ROOT / "_meta"
CLEANUP_CONFIG_FILE = META_DIR / "cleanup_config.json"
DEFAULT_CLEANUP_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "keep_days": 30,
    "compress_after_days": 7,
    "schedule_hour": 0,
    "schedule_minute": 30,
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def _validate_category(category: str) -> Path:
    if category not in CATEGORIES:
        raise HTTPException(404, f"Unknown category '{category}'")
    folder = LOGS_ROOT / category
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _validate_date(date: str) -> str:
    if not DATE_RE.match(date):
        raise HTTPException(400, "Date must be YYYY-MM-DD")
    return date


def _resolve_log_file(folder: Path, date: str) -> Path:
    """Return the actual file path (raw or .gz) for a date, or 404."""
    raw = folder / f"{date}.log"
    gz = folder / f"{date}.log.gz"
    if raw.exists():
        return raw
    if gz.exists():
        return gz
    raise HTTPException(404, f"No log file for {date}")


def _open_log_text(path: Path) -> io.TextIOBase:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _list_dates(folder: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for p in folder.iterdir():
        if p.is_dir():
            continue
        name = p.name
        if name.endswith(".log"):
            date = name[:-4]
            compressed = False
        elif name.endswith(".log.gz"):
            date = name[:-7]
            compressed = True
        else:
            continue
        if not DATE_RE.match(date):
            continue
        try:
            stat = p.stat()
        except OSError:
            continue
        items.append({
            "date": date,
            "size": stat.st_size,
            "modified_at": stat.st_mtime,
            "compressed": compressed,
        })
    items.sort(key=lambda x: x["date"], reverse=True)
    return items


def _folder_size(folder: Path) -> int:
    total = 0
    for p in folder.glob("**/*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _load_cleanup_config() -> Dict[str, Any]:
    META_DIR.mkdir(parents=True, exist_ok=True)
    if not CLEANUP_CONFIG_FILE.exists():
        return DEFAULT_CLEANUP_CONFIG.copy()
    try:
        data = json.loads(CLEANUP_CONFIG_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return DEFAULT_CLEANUP_CONFIG.copy()
    merged = DEFAULT_CLEANUP_CONFIG.copy()
    merged.update({k: data[k] for k in DEFAULT_CLEANUP_CONFIG if k in data})
    return merged


def _save_cleanup_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    META_DIR.mkdir(parents=True, exist_ok=True)
    CLEANUP_CONFIG_FILE.write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return cfg


# ── Endpoints ────────────────────────────────────────────────────────────────
@router.get("/categories")
async def list_categories(
    current_user: UserInDB = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """Return each category with file count, total size, and latest date."""
    out: List[Dict[str, Any]] = []
    for cat in CATEGORIES:
        folder = LOGS_ROOT / cat
        if not folder.exists():
            out.append({"category": cat, "file_count": 0, "size": 0, "latest_date": None})
            continue
        dates = _list_dates(folder)
        out.append({
            "category": cat,
            "file_count": len(dates),
            "size": _folder_size(folder),
            "latest_date": dates[0]["date"] if dates else None,
        })
    return out


@router.get("/{category}/dates")
async def list_dates(
    category: str,
    current_user: UserInDB = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    folder = _validate_category(category)
    return _list_dates(folder)


@router.get("/{category}/{date}")
async def read_file(
    category: str,
    date: str,
    offset: int = Query(0, ge=0, description="Byte offset (raw files only)"),
    tail_lines: Optional[int] = Query(
        None, ge=1, le=10000,
        description="If set, returns the last N lines instead of paginating",
    ),
    level: Optional[str] = Query(
        None, regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        description="Filter lines by level",
    ),
    current_user: UserInDB = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Read a log file. For non-compressed files, supports tail_lines or paginated
    offset reads. For compressed files, decompresses and returns full content
    (or last N lines via tail_lines).
    """
    folder = _validate_category(category)
    date = _validate_date(date)
    path = _resolve_log_file(folder, date)

    lines: List[str]
    if tail_lines is not None or path.suffix == ".gz":
        with _open_log_text(path) as f:
            all_lines = f.readlines()
        lines = all_lines[-tail_lines:] if tail_lines else all_lines
        next_offset = None
    else:
        size = path.stat().st_size
        if offset >= size:
            return {"lines": [], "next_offset": size, "total_size": size, "compressed": False}
        with open(path, "rb") as f:
            f.seek(offset)
            chunk = f.read()
        text = chunk.decode("utf-8", errors="replace")
        lines = text.splitlines()
        next_offset = size

    if level:
        lines = [ln for ln in lines if f" {level} " in ln or f" - {level} -" in ln]

    return {
        "lines": lines,
        "next_offset": next_offset,
        "total_size": path.stat().st_size,
        "compressed": path.suffix == ".gz",
    }


@router.get("/{category}/{date}/download")
async def download_file(
    category: str,
    date: str,
    current_user: UserInDB = Depends(get_current_user),
):
    folder = _validate_category(category)
    date = _validate_date(date)
    path = _resolve_log_file(folder, date)
    return FileResponse(path, filename=path.name, media_type="application/octet-stream")


@router.get("/{category}/{date}/tail")
async def tail_stream(
    category: str,
    date: str,
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Live-tail a log file via SSE. Only meaningful for an active (non-compressed)
    file — typically today's log.
    """
    folder = _validate_category(category)
    date = _validate_date(date)
    path = folder / f"{date}.log"
    if path.suffix == ".gz" or (folder / f"{date}.log.gz").exists():
        raise HTTPException(400, "Cannot tail a compressed (rotated) file")

    async def event_gen():
        # Start from current end so the client only sees new lines
        last_size = path.stat().st_size if path.exists() else 0
        idle_count = 0
        # Heartbeat every ~15s so proxies don't drop the connection
        while True:
            try:
                if not path.exists():
                    await asyncio.sleep(1.0)
                    idle_count += 1
                    if idle_count >= 15:
                        idle_count = 0
                        yield ": ping\n\n"
                    continue
                size = path.stat().st_size
                if size < last_size:
                    # File rotated/truncated — restart from beginning
                    last_size = 0
                if size > last_size:
                    with open(path, "rb") as f:
                        f.seek(last_size)
                        chunk = f.read(size - last_size)
                    last_size = size
                    text = chunk.decode("utf-8", errors="replace")
                    for line in text.splitlines():
                        if line.strip():
                            payload = json.dumps({"line": line}, ensure_ascii=False)
                            yield f"data: {payload}\n\n"
                    idle_count = 0
                else:
                    await asyncio.sleep(0.5)
                    idle_count += 1
                    if idle_count >= 30:
                        idle_count = 0
                        yield ": ping\n\n"
            except asyncio.CancelledError:
                break
            except Exception as e:
                err = json.dumps({"error": str(e)}, ensure_ascii=False)
                yield f"event: error\ndata: {err}\n\n"
                await asyncio.sleep(1.0)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete("/{category}/{date}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(
    category: str,
    date: str,
    current_user: UserInDB = Depends(require_supervisor),
):
    folder = _validate_category(category)
    date = _validate_date(date)
    path = _resolve_log_file(folder, date)
    try:
        path.unlink()
    except OSError as e:
        raise HTTPException(500, f"Failed to delete: {e}")
    return None


@router.delete("/{category}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_category(
    category: str,
    current_user: UserInDB = Depends(require_supervisor),
):
    folder = _validate_category(category)
    deleted = 0
    for p in folder.iterdir():
        if p.is_file() and (p.suffix == ".log" or p.name.endswith(".log.gz")):
            try:
                p.unlink()
                deleted += 1
            except OSError:
                pass
    return None


@router.get("/cleanup-config")
async def get_cleanup_config(
    current_user: UserInDB = Depends(get_current_user),
) -> Dict[str, Any]:
    return _load_cleanup_config()


@router.put("/cleanup-config")
async def update_cleanup_config(
    cfg: Dict[str, Any],
    current_user: UserInDB = Depends(require_supervisor),
) -> Dict[str, Any]:
    current = _load_cleanup_config()
    # Coerce + validate user-supplied fields
    if "enabled" in cfg:
        current["enabled"] = bool(cfg["enabled"])
    if "keep_days" in cfg:
        try:
            v = int(cfg["keep_days"])
        except (ValueError, TypeError):
            raise HTTPException(400, "keep_days must be an integer")
        if v < 1 or v > 3650:
            raise HTTPException(400, "keep_days must be 1..3650")
        current["keep_days"] = v
    if "compress_after_days" in cfg:
        try:
            v = int(cfg["compress_after_days"])
        except (ValueError, TypeError):
            raise HTTPException(400, "compress_after_days must be an integer")
        if v < 1:
            raise HTTPException(400, "compress_after_days must be >= 1")
        current["compress_after_days"] = v
    if "schedule_hour" in cfg:
        try:
            v = int(cfg["schedule_hour"])
        except (ValueError, TypeError):
            raise HTTPException(400, "schedule_hour must be an integer")
        if v < 0 or v > 23:
            raise HTTPException(400, "schedule_hour must be 0..23")
        current["schedule_hour"] = v
    if "schedule_minute" in cfg:
        try:
            v = int(cfg["schedule_minute"])
        except (ValueError, TypeError):
            raise HTTPException(400, "schedule_minute must be an integer")
        if v < 0 or v > 59:
            raise HTTPException(400, "schedule_minute must be 0..59")
        current["schedule_minute"] = v
    if current["compress_after_days"] >= current["keep_days"]:
        raise HTTPException(400, "compress_after_days must be < keep_days")

    saved = _save_cleanup_config(current)

    # Notify the running scheduler so it picks up the new schedule immediately
    try:
        from app.services.log_cleanup_scheduler import reschedule
        reschedule(saved)
    except Exception:
        pass
    return saved
