"""
In-memory training log buffer + logging handler that captures real-time
output from the ML training service so the frontend can display logs
during background training.

Buffer is keyed by model_id. Each entry has a monotonically increasing
`idx` so the FE can poll with a `since` cursor and only fetch new lines.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple


_MAX_LINES_PER_MODEL = 500
_CAPTURED_LOGGERS = (
    "app.services.ml_training_service",
    "app.services.ml_ok_synthesize",
)


class TrainingLogBuffer:
    """Thread-safe ring buffer of training log entries grouped by model_id."""

    def __init__(self, max_lines: int = _MAX_LINES_PER_MODEL) -> None:
        self._buffers: Dict[str, Deque[dict]] = {}
        self._counters: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._max_lines = max_lines

    def register(self, model_id: str) -> None:
        with self._lock:
            self._buffers.setdefault(model_id, deque(maxlen=self._max_lines))
            self._counters.setdefault(model_id, 0)

    def push(self, model_id: str, msg: str, level: str = "INFO") -> None:
        with self._lock:
            buf = self._buffers.setdefault(model_id, deque(maxlen=self._max_lines))
            counter = self._counters.get(model_id, 0)
            buf.append({
                "idx":   counter,
                "ts":    time.time(),
                "level": level,
                "msg":   msg,
            })
            self._counters[model_id] = counter + 1

    def get_since(self, model_id: str, since: int) -> Tuple[List[dict], int]:
        """Return (new_entries, next_since) where new_entries have idx >= since."""
        with self._lock:
            buf = self._buffers.get(model_id)
            counter = self._counters.get(model_id, 0)
            if not buf:
                return [], counter
            new_entries = [e for e in buf if e["idx"] >= since]
            return new_entries, counter

    def drop(self, model_id: str) -> None:
        with self._lock:
            self._buffers.pop(model_id, None)
            self._counters.pop(model_id, None)


_buffer = TrainingLogBuffer()


def get_buffer() -> TrainingLogBuffer:
    return _buffer


class _ModelLogHandler(logging.Handler):
    """Captures logger output into the buffer for a specific model_id."""

    def __init__(self, model_id: str, buf: TrainingLogBuffer) -> None:
        super().__init__(level=logging.INFO)
        self.model_id = model_id
        self._buffer = buf
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            msg = self.format(record)
            self._buffer.push(self.model_id, msg, record.levelname)
        except Exception:
            # Never let logging crash training
            pass


def attach_handler(model_id: str) -> _ModelLogHandler:
    """
    Attach a model-scoped handler to the training-related loggers so all
    `logger.info(...)` calls inside `train_model` / `build_dataset` end up
    in the buffer for this model_id.
    """
    handler = _ModelLogHandler(model_id, _buffer)
    for name in _CAPTURED_LOGGERS:
        logging.getLogger(name).addHandler(handler)
    return handler


def detach_handler(handler: Optional[_ModelLogHandler]) -> None:
    if handler is None:
        return
    for name in _CAPTURED_LOGGERS:
        logging.getLogger(name).removeHandler(handler)
