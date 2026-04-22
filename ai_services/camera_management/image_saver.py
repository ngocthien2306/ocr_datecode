"""
Image Saver Module
Background worker that saves captured frames to disk without blocking the main capture flow.
"""
import queue
import threading
import logging
import time
from pathlib import Path
from typing import List, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Save directory: <project_root>/public/images
_SAVE_DIR = Path(__file__).parent.parent.parent / "public" / "images"
_MAX_IMAGES = 100
_JPEG_QUALITY = 90


class ImageSaveWorker:
    """
    Background worker that saves camera frames to disk.

    Features:
    - Non-blocking: enqueue() returns immediately, saving happens in a daemon thread
    - All cameras share the same folder
    - Keeps only the 100 most recent images (oldest deleted automatically)
    - Safe singleton: call ImageSaveWorker.instance() to get the shared instance
    """

    _instance: Optional['ImageSaveWorker'] = None
    _instance_lock = threading.Lock()

    @classmethod
    def instance(cls) -> 'ImageSaveWorker':
        """Return shared singleton instance, creating it on first call."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self, save_dir: Path = _SAVE_DIR, max_images: int = _MAX_IMAGES):
        self._save_dir = Path(save_dir)
        self._max_images = max_images
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="ImageSaveWorker",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"[ImageSaver] Started. Save dir: {self._save_dir}, max images: {self._max_images}")

    def enqueue(self, serial_number: str, frames: List[np.ndarray]) -> None:
        """
        Enqueue a list of frames for saving. Non-blocking — returns immediately.

        Args:
            serial_number: Camera serial number (used in filename)
            frames: List of BGR numpy arrays (one per template)
        """
        if not frames:
            return
        # Use a single timestamp for the whole batch from this trigger
        timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
        try:
            self._queue.put_nowait((serial_number, frames, timestamp))
        except queue.Full:
            logger.warning("[ImageSaver] Queue full, dropping frame batch for camera %s", serial_number)

    def _worker_loop(self) -> None:
        """Daemon worker: dequeue tasks and save images one by one."""
        self._save_dir.mkdir(parents=True, exist_ok=True)

        while True:
            try:
                serial_number, frames, timestamp = self._queue.get()
                self._save_batch(serial_number, frames, timestamp)
                self._enforce_max_images()
            except Exception:
                logger.exception("[ImageSaver] Unexpected error in worker loop")

    def _save_batch(self, serial_number: str, frames: List[np.ndarray], timestamp: str) -> None:
        """Save all frames from one trigger event."""
        for idx, frame in enumerate(frames):
            filename = f"{timestamp}_{serial_number}_{idx}.jpg"
            filepath = self._save_dir / filename
            try:
                ok = cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
                if ok:
                    logger.debug("[ImageSaver] Saved %s", filename)
                else:
                    logger.warning("[ImageSaver] cv2.imwrite failed for %s", filename)
            except Exception:
                logger.exception("[ImageSaver] Error saving %s", filepath)

    def _enforce_max_images(self) -> None:
        """Delete the oldest images if the folder exceeds max_images."""
        try:
            files = sorted(self._save_dir.glob("*.jpg"))
            excess = len(files) - self._max_images
            if excess > 0:
                for f in files[:excess]:
                    f.unlink(missing_ok=True)
                logger.debug("[ImageSaver] Deleted %d old image(s)", excess)
        except Exception:
            logger.exception("[ImageSaver] Error enforcing max image limit")
