"""
Pass-Image Pruner
=================
Keeps a ring buffer of the most-recent PASS-frame images per (recipe, camera).

PASS images are saved to disk so the user can re-test "missed defect" frames in
the Edge Setup modal. To bound disk usage we keep only the N most-recent PASS
images **per camera per recipe** (default 200) and delete older ones.

Mirror of `image_saver.ImageSaveWorker`: a daemon thread drains a queue so the
glob+unlink never blocks the inference pipeline. Pruning per (recipe, serial)
is throttled so a high frame rate doesn't trigger a directory scan every frame.
"""
import os
import queue
import threading
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_MAX = 200
_THROTTLE_SECONDS = 8.0   # min gap between scans of the same (recipe, serial)


class PassImagePruner:
    """Background pruner keeping only the newest `max_images` PASS image files
    per (recipe_id, serial_number). Both `*_viz.jpg` and `*_org.jpg` are pruned.
    """

    _instance: Optional['PassImagePruner'] = None
    _instance_lock = threading.Lock()

    @classmethod
    def instance(cls, base_dir: Optional[str] = None, max_images: int = _DEFAULT_MAX) -> 'PassImagePruner':
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(base_dir=base_dir, max_images=max_images)
        return cls._instance

    def __init__(self, base_dir: Optional[str] = None, max_images: int = _DEFAULT_MAX):
        home = os.environ.get("HOME", "")
        self._base_dir = Path(base_dir or f"{home}/Source/ocr_datecode/backend/uploads/inference_results")
        self._max_images = max_images
        self._queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self._last_pruned: dict = {}  # {(recipe_id, serial): monotonic_ts}
        self._thread = threading.Thread(
            target=self._worker_loop, name="PassImagePruner", daemon=True
        )
        self._thread.start()
        logger.info(
            f"[PassImagePruner] Started. base={self._base_dir}, max={self._max_images}/camera"
        )

    def enqueue(self, recipe_id: str, serial_number: str) -> None:
        """Request a prune for (recipe_id, serial). Non-blocking; throttled in worker."""
        if not recipe_id or not serial_number:
            return
        try:
            self._queue.put_nowait((str(recipe_id), str(serial_number)))
        except queue.Full:
            pass

    def _worker_loop(self) -> None:
        while True:
            try:
                recipe_id, serial = self._queue.get()
                key = (recipe_id, serial)
                now = time.monotonic()
                last = self._last_pruned.get(key, 0.0)
                if now - last < _THROTTLE_SECONDS:
                    continue
                self._last_pruned[key] = now
                self._prune(recipe_id, serial)
            except Exception:
                logger.exception("[PassImagePruner] worker error")

    def _prune(self, recipe_id: str, serial: str) -> None:
        """Delete oldest PASS images beyond max_images for this (recipe, serial).

        Layout: {base}/{recipe_id}/{YYYY-MM-DD}/{serial}/pass_f*_*_(viz|org).jpg
        """
        recipe_dir = self._base_dir / recipe_id
        if not recipe_dir.exists():
            return
        for suffix in ("_viz.jpg", "_org.jpg"):
            try:
                files = list(recipe_dir.glob(f"*/{serial}/pass_*{suffix}"))
                if len(files) <= self._max_images:
                    continue
                files.sort(key=lambda p: p.stat().st_mtime)  # oldest first
                excess = len(files) - self._max_images
                for f in files[:excess]:
                    try:
                        f.unlink(missing_ok=True)
                    except Exception:
                        pass
                logger.debug(
                    f"[PassImagePruner] recipe={recipe_id} cam={serial} {suffix}: "
                    f"deleted {excess} old (kept {self._max_images})"
                )
            except Exception:
                logger.exception(
                    f"[PassImagePruner] prune failed recipe={recipe_id} cam={serial}"
                )
