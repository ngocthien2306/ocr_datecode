"""
OcrFailSaver — lưu crop của các vùng OCR bị FAIL lúc chạy thực tế để debug.

Crop lưu đúng là ảnh đưa vào OCR (trước augment), PNG để giữ nguyên chấm
dot-matrix — dùng lại được làm data fine-tune. Ghi đĩa ở worker thread riêng
(giống LabelCropSaver) nên không bao giờ chặn inference; queue đầy thì bỏ ảnh.

Thư mục:
    {base_dir}/{YYYY-MM-DD}/{serial}/{HHMMSS_mmm}_ann{idx}_{model}_exp-{..}_got-{..}.png
    {base_dir}/{YYYY-MM-DD}/fails.csv   — 1 dòng / ảnh: expected, recognized nguyên văn,
                                          conf, char_confs, model, file
Giữ `keep_days` ngày gần nhất, tối đa `max_per_day` ảnh / ngày để không đầy đĩa.
"""

import csv
import logging
import os
import queue
import re
import shutil
import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_CSV_FIELDS = ["time", "serial", "annotation_idx", "model", "expected", "recognized",
               "confidence", "threshold", "similarity", "char_confs", "file"]
_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _safe(s: str, n: int = 24) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", s or "").strip("_")[:n] or "EMPTY"


class OcrFailSaver:
    def __init__(
        self,
        base_dir: str,
        keep_days: int = 14,
        max_per_day: int = 3000,
        queue_size: int = 64,
    ):
        self.base_dir = base_dir
        self.keep_days = keep_days
        self.max_per_day = max_per_day

        self._queue: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self._day: Optional[str] = None
        self._day_count = 0
        self._saved = 0
        self._dropped = 0

        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="OcrFailSaver"
        )
        self._worker.start()
        logger.info(
            f"OcrFailSaver started: dir={base_dir}, keep {keep_days} days, "
            f"max {max_per_day} crops/day"
        )

    def submit(self, serial_number: str, crop: np.ndarray, region: Dict[str, Any],
               model: str) -> None:
        """Non-blocking. Gọi từ luồng inference với region result đã chốt FAIL."""
        if crop is None or crop.size == 0:
            return
        try:
            self._queue.put_nowait((time.time(), serial_number, crop.copy(), dict(region), model))
        except queue.Full:
            self._dropped += 1
            if self._dropped == 1 or self._dropped % 100 == 0:
                logger.warning(f"OcrFailSaver: queue full, dropped {self._dropped} crops so far")

    def stop(self) -> None:
        self._stop.set()
        self._worker.join(timeout=2.0)

    # ────────────────────────────────────────────────────────────────────
    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._save(*item)
            except Exception as e:
                logger.warning(f"OcrFailSaver: save failed ({e})")

    def _save(self, ts: float, serial: str, crop: np.ndarray, region: Dict[str, Any],
              model: str) -> None:
        day = time.strftime("%Y-%m-%d", time.localtime(ts))
        day_dir = os.path.join(self.base_dir, day)
        if day != self._day:
            self._day = day
            self._day_count = len([f for _, _, fs in os.walk(day_dir) for f in fs if f.endswith(".png")])
            self._prune_old_days(day)
        if self._day_count >= self.max_per_day:
            if self._day_count == self.max_per_day:
                logger.warning(f"OcrFailSaver: reached {self.max_per_day} crops for {day}, "
                               f"skipping the rest of the day")
                self._day_count += 1
            return

        expected = region.get("expected", "")
        recognized = region.get("recognized", "")
        ann_idx = region.get("annotation_idx", -1)
        cam_dir = os.path.join(day_dir, serial)
        os.makedirs(cam_dir, exist_ok=True)
        stamp = time.strftime("%H%M%S", time.localtime(ts)) + f"_{int(ts * 1000) % 1000:03d}"
        name = (f"{stamp}_ann{ann_idx}_{_safe(model, 12)}"
                f"_exp-{_safe(expected)}_got-{_safe(recognized)}.png")
        if not cv2.imwrite(os.path.join(cam_dir, name), crop):
            logger.warning(f"OcrFailSaver: imwrite failed for {name}")
            return
        self._day_count += 1
        self._saved += 1

        csv_path = os.path.join(day_dir, "fails.csv")
        new_file = not os.path.exists(csv_path)
        # utf-8-sig: mở thẳng bằng Excel không lỗi font
        with open(csv_path, "a", newline="", encoding="utf-8-sig" if new_file else "utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            if new_file:
                w.writeheader()
            w.writerow({
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
                "serial": serial,
                "annotation_idx": ann_idx,
                "model": model,
                "expected": expected,
                "recognized": recognized,
                "confidence": f"{float(region.get('confidence', 0.0)):.4f}",
                "threshold": region.get("threshold", ""),
                "similarity": region.get("similarity", ""),
                "char_confs": " ".join(f"{c['char']}:{c['conf']:.2f}"
                                       for c in (region.get("char_confs") or [])),
                "file": os.path.join(serial, name),
            })

    def _prune_old_days(self, today: str) -> None:
        if not os.path.isdir(self.base_dir):
            return
        # today counts toward keep_days even before its folder exists
        days = sorted({d for d in os.listdir(self.base_dir) if _DAY_RE.match(d)} | {today})
        for old in days[: max(0, len(days) - self.keep_days)]:
            shutil.rmtree(os.path.join(self.base_dir, old), ignore_errors=True)

    def get_stats(self) -> dict:
        return {"saved": self._saved, "dropped": self._dropped, "queued": self._queue.qsize()}


_instances: Dict[str, OcrFailSaver] = {}
_instances_lock = threading.Lock()


def get_ocr_fail_saver(base_dir: str) -> OcrFailSaver:
    """Một saver / thư mục cho cả process — TextVerificationService có thể được
    tạo lại (re-init verification services) mà không đẻ thêm worker thread."""
    with _instances_lock:
        if base_dir not in _instances:
            _instances[base_dir] = OcrFailSaver(base_dir)
        return _instances[base_dir]
