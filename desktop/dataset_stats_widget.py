"""
Dataset Stats — browse the per-character dataset built by CharLabelerWidget.
Reads `metadata.json` from the output folder and displays:
  - per-char counts (OK/NG)
  - thumbnails grid
  - toggle: group by char  vs  flat list by source image
"""
import json
import os
from collections import defaultdict

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QScrollArea, QFrame, QCheckBox, QGroupBox, QGridLayout,
)


_THUMB_LIMIT_PER_CHAR = 24    # cap thumbnails per row to keep stats tab snappy


def _bgr_to_pixmap(img, target_size=(48, 48)):
    if img is None or img.size == 0:
        return QPixmap()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    buf = np.ascontiguousarray(img)
    qimg = QImage(buf.data, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped().copy()
    return QPixmap.fromImage(qimg).scaled(
        target_size[0], target_size[1],
        Qt.KeepAspectRatio, Qt.SmoothTransformation,
    )


class DatasetStatsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.output_folder = None
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        # Toolbar
        bar = QHBoxLayout()
        load_btn = QPushButton("📁 Load output folder…")
        load_btn.clicked.connect(self._load_folder)
        bar.addWidget(load_btn)
        self.path_label = QLabel("(no folder loaded)")
        self.path_label.setStyleSheet("color: #888;")
        bar.addWidget(self.path_label, stretch=1)
        self.group_check = QCheckBox("Group by char")
        self.group_check.setChecked(True)
        self.group_check.toggled.connect(self._render)
        bar.addWidget(self.group_check)
        reload_btn = QPushButton("🔄 Reload")
        reload_btn.clicked.connect(self._render)
        bar.addWidget(reload_btn)
        outer.addLayout(bar)

        # Summary
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(
            "background-color: #2a2a2a; padding: 8px; border-radius: 4px;"
        )
        outer.addWidget(self.summary_label)

        # Scrollable content area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("background-color: #1e1e1e;")
        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(8, 8, 8, 8)
        self.container_layout.setSpacing(10)
        self.scroll.setWidget(self.container)
        outer.addWidget(self.scroll, stretch=1)

    # --------------------------------------------------------------- IO
    def _load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if not folder:
            return
        self.output_folder = folder
        self.path_label.setText(folder)
        self._render()

    def _read_metadata(self):
        if not self.output_folder:
            return None
        p = os.path.join(self.output_folder, 'metadata.json')
        if not os.path.exists(p):
            return None
        try:
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[stats] failed to read metadata: {e}")
            return None

    # --------------------------------------------------------------- render

    def _clear_container(self):
        while self.container_layout.count():
            item = self.container_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _render(self):
        self._clear_container()
        meta = self._read_metadata()
        if not meta:
            self.summary_label.setText(
                "<i>No metadata.json found. Load an output folder produced by the Char Labeler tab.</i>"
            )
            return

        items = meta.get('items') or []
        n_total = len(items)
        n_ok = sum(1 for it in items if it.get('label') == 'ok')
        n_ng = sum(1 for it in items if it.get('label') == 'ng')
        n_imgs = len(meta.get('processed_images') or [])

        per_char_count = defaultdict(lambda: {'ok': 0, 'ng': 0})
        for it in items:
            cid = it.get('char_id') or '?'
            lbl = it.get('label') or 'ok'
            per_char_count[cid][lbl] += 1

        self.summary_label.setText(
            f"<b>Total:</b> {n_total} chars across {n_imgs} images "
            f"(<span style='color:#28dc50'>{n_ok} OK</span>, "
            f"<span style='color:#f03c3c'>{n_ng} NG</span>) | "
            f"<b>Unique char ids:</b> {len(per_char_count)}"
        )

        if self.group_check.isChecked():
            self._render_grouped(items, per_char_count)
        else:
            self._render_flat(items)

    def _render_grouped(self, items, per_char_count):
        # Sort: alphanumeric char_ids first, then '?'/'unknown' last
        char_ids = sorted(per_char_count.keys(),
                           key=lambda c: (c in ('?', 'unknown'), c))
        # Bucket items by char_id
        by_cid = defaultdict(list)
        for it in items:
            by_cid[it.get('char_id') or '?'].append(it)

        for cid in char_ids:
            counts = per_char_count[cid]
            sec = QGroupBox(
                f"char_{cid}   —   "
                f"OK: {counts['ok']}    NG: {counts['ng']}    "
                f"Total: {counts['ok'] + counts['ng']}"
            )
            secl = QVBoxLayout(sec)

            # OK row
            if counts['ok'] > 0:
                ok_items = [it for it in by_cid[cid] if it.get('label') == 'ok']
                secl.addWidget(self._make_thumb_row("✓ OK", '#28dc50', ok_items))
            if counts['ng'] > 0:
                ng_items = [it for it in by_cid[cid] if it.get('label') == 'ng']
                secl.addWidget(self._make_thumb_row("✗ NG", '#f03c3c', ng_items))
            self.container_layout.addWidget(sec)

        self.container_layout.addStretch(1)

    def _render_flat(self, items):
        # Group by source image
        by_src = defaultdict(list)
        for it in items:
            by_src[it.get('source') or '?'].append(it)

        for src in sorted(by_src.keys()):
            its = by_src[src]
            sec = QGroupBox(f"{src}   —   {len(its)} chars")
            secl = QVBoxLayout(sec)
            secl.addWidget(self._make_thumb_row("", '#a0a0a0', its))
            self.container_layout.addWidget(sec)
        self.container_layout.addStretch(1)

    def _make_thumb_row(self, label, color, items):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        if label:
            lbl = QLabel(f"<b style='color:{color}'>{label}</b> ({len(items)})")
            v.addWidget(lbl)

        # Cap thumbs to avoid render blowup for huge datasets
        shown = items[:_THUMB_LIMIT_PER_CHAR]
        omitted = len(items) - len(shown)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedHeight(72)
        scroll.setStyleSheet("border: none;")
        inner = QWidget()
        h = QHBoxLayout(inner)
        h.setContentsMargins(2, 2, 2, 2)
        h.setSpacing(4)

        for it in shown:
            rel = it.get('saved_path') or ''
            full = os.path.join(self.output_folder or '', rel)
            img = cv2.imread(full) if os.path.exists(full) else None
            cell = QFrame()
            cell.setStyleSheet(
                f"QFrame {{ background-color: #2a2a2a; border: 1px solid {color}; "
                "padding: 2px; }}"
            )
            cell.setFixedSize(58, 58)
            cv = QVBoxLayout(cell)
            cv.setContentsMargins(2, 2, 2, 2)
            img_lbl = QLabel()
            img_lbl.setAlignment(Qt.AlignCenter)
            img_lbl.setStyleSheet("border: none;")
            if img is not None:
                img_lbl.setPixmap(_bgr_to_pixmap(img, (50, 50)))
            else:
                img_lbl.setText("(missing)")
                img_lbl.setStyleSheet("color: #888; font-size: 8px; border: none;")
            cv.addWidget(img_lbl)
            cell.setToolTip(f"{it.get('source')}  #{it.get('char_idx')}  "
                              f"char='{it.get('char_id')}'  {it.get('label')}\n{rel}")
            h.addWidget(cell)

        if omitted > 0:
            more = QLabel(f"+{omitted} more")
            more.setStyleSheet("color: #888;")
            h.addWidget(more)
        h.addStretch(1)
        scroll.setWidget(inner)
        v.addWidget(scroll)
        return w
