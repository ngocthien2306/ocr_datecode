"""
Centralized logging config for OCR DateCode.

Logs are written to {repo_root}/logs/{category}/{YYYY-MM-DD}.log
with auto-rotation when the date changes.

Usage:
    from app.utils.logging_config import setup_category_logger

    # Configure root logger to also write to file
    setup_category_logger("backend")
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# repo_root = parents[3] (utils → app → backend → repo)
_REPO_ROOT = Path(__file__).resolve().parents[3]
LOGS_ROOT = _REPO_ROOT / "logs"

# All log categories used across the project. Adding a new one only needs
# adding the string here for documentation; folders are auto-created on first use.
CATEGORIES = [
    "backend",
    "camera_settings",
    "camera_management",
    "trigger_stats",
    "pulse_width",
    "reject_actions",
    "obb_rotation",
    "camera_check",
    "start_services",
]

DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_log_dir(category: str) -> Path:
    folder = LOGS_ROOT / category
    folder.mkdir(parents=True, exist_ok=True)
    return folder


class DailyRotatingFileHandler(logging.FileHandler):
    """
    File handler that writes to {log_dir}/{YYYY-MM-DD}.log and rotates
    automatically when the local date changes (checked on each emit).

    backupCount is intentionally absent — old files are managed by the
    cleanup scheduler (compress/delete based on user config).
    """

    def __init__(self, log_dir: Path, encoding: str = "utf-8"):
        self._log_dir = log_dir
        self._current_date = self._today()
        filename = log_dir / f"{self._current_date}.log"
        super().__init__(str(filename), mode="a", encoding=encoding, delay=False)

    @staticmethod
    def _today() -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def emit(self, record: logging.LogRecord) -> None:
        today = self._today()
        if today != self._current_date:
            try:
                self.close()
            except Exception:
                pass
            self._current_date = today
            self.baseFilename = str(self._log_dir / f"{today}.log")
            self.stream = self._open()
        super().emit(record)


def make_handler(category: str, level: int = logging.INFO,
                 fmt: str = DEFAULT_FORMAT) -> DailyRotatingFileHandler:
    handler = DailyRotatingFileHandler(get_log_dir(category))
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt))
    return handler


def setup_category_logger(
    category: str,
    *,
    level: int = logging.INFO,
    fmt: str = DEFAULT_FORMAT,
    add_console: bool = True,
    logger_name: Optional[str] = None,
) -> logging.Logger:
    """
    Attach DailyRotatingFileHandler (+ optional StreamHandler) to a logger.

    If logger_name is None, configures the root logger — use this from main
    entrypoints so any module's logger.info() ends up in the file.
    """
    target = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    target.setLevel(level)

    # Avoid attaching duplicate handlers if this is called more than once
    file_marker = f"daily-{category}"
    has_file = any(getattr(h, "_marker", None) == file_marker for h in target.handlers)
    if not has_file:
        fh = make_handler(category, level=level, fmt=fmt)
        setattr(fh, "_marker", file_marker)
        target.addHandler(fh)

    if add_console:
        has_console = any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
            for h in target.handlers
        )
        if not has_console:
            ch = logging.StreamHandler()
            ch.setLevel(level)
            ch.setFormatter(logging.Formatter(fmt))
            target.addHandler(ch)

    return target
