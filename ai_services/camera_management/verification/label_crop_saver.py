"""
LabelCropSaver — rolling store of the newest N label crops per camera.

Nhận crop label (đúng cái crop đưa vào wrinkle/anomaly model) từ luồng
inference qua submit(), ghi đĩa ở một worker thread riêng để không bao giờ
chặn inference. Mỗi camera (serial_number) giữ tối đa `max_per_camera` ảnh
MỚI NHẤT — ảnh cũ hơn bị xoá dần (rolling), khác với crop_samples hiện có
(dừng hẳn khi đủ 100 ảnh đầu tiên).

Thư mục: {base_dir}/{serial_number}/label_{epoch_ms}_{seq:06d}.jpg
Tên file có epoch-ms đứng trước nên sort theo tên = sort theo thời gian.
"""

import logging
import os
import queue
import threading
import time
from typing import Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class LabelCropSaver:
    def __init__(
        self,
        base_dir: str,
        max_per_camera: int = 100,
        queue_size: int = 64,
        jpeg_quality: int = 90,
    ):
        self.base_dir = base_dir
        self.max_per_camera = max_per_camera
        self.jpeg_quality = jpeg_quality

        # Queue đầy thì submit() drop ảnh thay vì chặn luồng inference.
        self._queue: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self._seq: Dict[str, int] = {}       # serial -> số thứ tự tăng dần
        self._dropped = 0
        self._saved = 0

        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="LabelCropSaver"
        )
        self._worker.start()
        logger.info(
            f"LabelCropSaver started: dir={base_dir}, "
            f"max {max_per_camera} newest crops per camera"
        )

    def submit(self, serial_number: str, crop: np.ndarray) -> None:
        """Non-blocking. Gọi từ luồng inference — copy crop rồi đẩy vào queue;
        queue đầy thì bỏ qua ảnh này (ưu tiên tuyệt đối cho inference)."""
        if crop is None or crop.size == 0:
            return
        try:
            self._queue.put_nowait((serial_number, crop.copy()))
        except queue.Full:
            self._dropped += 1
            if self._dropped == 1 or self._dropped % 100 == 0:
                logger.warning(
                    f"LabelCropSaver: queue full, dropped {self._dropped} crops so far"
                )

    def stop(self) -> None:
        self._stop.set()
        self._worker.join(timeout=2.0)

    # ────────────────────────────────────────────────────────────────────
    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                serial_number, crop = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._save_and_prune(serial_number, crop)
            except Exception as e:
                logger.warning(f"LabelCropSaver: save failed ({e})")

    def _save_and_prune(self, serial_number: str, crop: np.ndarray) -> None:
        cam_dir = os.path.join(self.base_dir, serial_number)
        os.makedirs(cam_dir, exist_ok=True)

        seq = self._seq.get(serial_number, 0)
        self._seq[serial_number] = seq + 1
        name = f"label_{int(time.time() * 1000)}_{seq:06d}.jpg"
        ok = cv2.imwrite(
            os.path.join(cam_dir, name), crop,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ok:
            logger.warning(f"LabelCropSaver: imwrite failed for {name}")
            return
        self._saved += 1

        # Prune: giữ đúng max_per_camera file mới nhất (tên file sort được
        # theo thời gian nhờ prefix epoch-ms).
        files = sorted(
            f for f in os.listdir(cam_dir)
            if f.startswith("label_") and f.endswith(".jpg")
        )
        for old in files[: max(0, len(files) - self.max_per_camera)]:
            try:
                os.remove(os.path.join(cam_dir, old))
            except OSError:
                pass

    def get_stats(self) -> dict:
        return {
            "saved": self._saved,
            "dropped": self._dropped,
            "queued": self._queue.qsize(),
        }
