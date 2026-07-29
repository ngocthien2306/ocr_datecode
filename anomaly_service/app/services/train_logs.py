"""
Training log buffer + logging handler — standalone port of
backend/app/services/ml_training_logs.py for this service's training loop.
Buffer is keyed by model_id; entries have a monotonically increasing `idx`
so the FE can poll with a `since` cursor.

Backed by a JSONL file per model (data/projects/{pid}/models/{model_id}_train.log.jsonl),
written through on every push() -- the in-memory deque alone doesn't survive
a server restart (uvicorn --reload, crash, manual restart), which used to
make a model's training log disappear even though the model itself (status,
metrics, checkpoint/onnx/engine paths) is safely in MongoDB the whole time.
register() lazy-loads from that file on first access in a process lifetime,
so re-opening the Train tab after a restart shows the full history again.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_MAX_LINES_PER_MODEL = 500
_CAPTURED_LOGGERS = ("app.services.anomaly_training",)


def _log_file_path(project_id: str, model_id: str) -> Path:
    from app.services import dataset_fs

    return dataset_fs.models_dir(project_id) / f"{model_id}_train.log.jsonl"


class TrainingLogBuffer:
    def __init__(self, max_lines: int = _MAX_LINES_PER_MODEL) -> None:
        self._buffers: Dict[str, Deque[dict]] = {}
        self._counters: Dict[str, int] = {}
        self._log_paths: Dict[str, Path] = {}
        self._lock = threading.Lock()
        self._max_lines = max_lines

    def register(self, project_id: str, model_id: str) -> None:
        """Idempotent per process lifetime: first call for a model_id loads
        any existing on-disk log (older run, or this run before a restart);
        later calls are a no-op so an in-progress run's in-memory state
        isn't clobbered by a second register() (e.g. a log-polling GET
        racing the background training task's own register())."""
        with self._lock:
            if model_id in self._buffers:
                return
            log_path = _log_file_path(project_id, model_id)
            buf: Deque[dict] = deque(maxlen=self._max_lines)
            counter = 0
            if log_path.exists():
                try:
                    for line in log_path.read_text().splitlines():
                        entry = json.loads(line)
                        buf.append(entry)
                        counter = max(counter, entry["idx"] + 1)
                except Exception:
                    logger.exception(f"[train_logs] failed to load {log_path}")
            self._buffers[model_id] = buf
            self._counters[model_id] = counter
            self._log_paths[model_id] = log_path

    def push(self, model_id: str, msg: str, level: str = "INFO") -> None:
        with self._lock:
            buf = self._buffers.setdefault(model_id, deque(maxlen=self._max_lines))
            counter = self._counters.get(model_id, 0)
            entry = {"idx": counter, "ts": time.time(), "level": level, "msg": msg}
            buf.append(entry)
            self._counters[model_id] = counter + 1

            log_path = self._log_paths.get(model_id)
            if log_path is not None:
                try:
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(entry) + "\n")
                except Exception:
                    logger.exception(f"[train_logs] failed to append to {log_path}")

    def get_since(self, model_id: str, since: int) -> Tuple[List[dict], int]:
        with self._lock:
            buf = self._buffers.get(model_id)
            counter = self._counters.get(model_id, 0)
            if not buf:
                return [], counter
            return [e for e in buf if e["idx"] >= since], counter

    def drop(self, model_id: str) -> None:
        """Clear in-memory state and delete the on-disk log -- call this
        from model deletion so a removed model doesn't leave its log file
        behind (delete_model already removes the checkpoint/export files
        for the same reason)."""
        with self._lock:
            self._buffers.pop(model_id, None)
            self._counters.pop(model_id, None)
            log_path = self._log_paths.pop(model_id, None)
        if log_path is not None:
            log_path.unlink(missing_ok=True)


_buffer = TrainingLogBuffer()


def get_buffer() -> TrainingLogBuffer:
    return _buffer


class _ModelLogHandler(logging.Handler):
    def __init__(self, model_id: str, buf: TrainingLogBuffer) -> None:
        super().__init__(level=logging.INFO)
        self.model_id = model_id
        self._buffer = buf
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._buffer.push(self.model_id, self.format(record), record.levelname)
        except Exception:
            pass


def attach_handler(model_id: str) -> _ModelLogHandler:
    handler = _ModelLogHandler(model_id, _buffer)
    for name in _CAPTURED_LOGGERS:
        logging.getLogger(name).addHandler(handler)
    return handler


def detach_handler(handler: Optional[_ModelLogHandler]) -> None:
    if handler is None:
        return
    for name in _CAPTURED_LOGGERS:
        logging.getLogger(name).removeHandler(handler)
