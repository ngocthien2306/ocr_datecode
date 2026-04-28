"""
Char Labeler — build a per-character training dataset from text-region crops.

Workflow:
  1. Load a folder of pre-cropped text images.
  2. For each image: auto-segment chars + run RapidOCR for char-id labels.
  3. User reviews / edits bboxes and char-ids, then marks each char OK or NG.
  4. Save → output/char_<id>/<ok|ng>/<file>_<idx>.png + metadata.json

Output structure is designed to be both human-browsable (by char) and
machine-readable (metadata.json is the source of truth, allowing arbitrary
re-grouping later).
"""
import json
import os
from datetime import datetime

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QListWidget, QListWidgetItem, QSplitter, QLineEdit, QGroupBox, QMessageBox,
    QScrollArea, QFrame, QSizePolicy, QShortcut, QComboBox, QCheckBox,
)
from PyQt5.QtGui import QKeySequence

from char_labeler_canvas import CharLabelerCanvas


# Files ending in these extensions are treated as input images
_IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def _alphanumeric_only(text):
    """Strip everything except [a-zA-Z0-9] and return the cleaned string."""
    return ''.join(ch for ch in (text or '') if ch.isalnum())


def _spatial_map_chars(boxes, ocr_chars, image_width):
    """
    Map OCR per-char predictions to segmentation bboxes.

    `boxes`: list of (x, y, w, h) sorted left→right
    `ocr_chars`: list of {'char', 'col', 'conf'} from RapidOCR.recognize_with_chars
                 (already filtered to alphanumeric, sorted by `col`)
    `image_width`: width of the source image (px) — used to map `col` to x

    Returns list aligned to `boxes`: [{'char_id', 'ocr_conf'}, ...]. When there
    is no good OCR match for a box, char_id='?' and ocr_conf=None.
    """
    n = len(boxes)
    out = [{'char_id': '?', 'ocr_conf': None} for _ in range(n)]
    if not ocr_chars or n == 0:
        return out

    # Fast path: 1:1 by index when counts match
    if len(ocr_chars) == n:
        for i, oc in enumerate(ocr_chars):
            out[i] = {'char_id': oc.get('char') or '?',
                       'ocr_conf': float(oc.get('conf') or 0.0)}
        return out

    # Spatial mapping: convert each OCR char's `col` (in feature-map units) to
    # an approximate pixel x by linear scaling against the max col seen.
    cols = [oc.get('col', 0) for oc in ocr_chars]
    max_col = max(cols) if cols else 1
    if max_col <= 0:
        max_col = 1
    box_centers = [bx + bw / 2 for (bx, _by, bw, _bh) in boxes]
    img_max_x = max(box_centers) if box_centers else image_width

    # For each box, find the OCR char whose mapped x is closest
    for i, cx in enumerate(box_centers):
        best_idx, best_dist = -1, float('inf')
        for j, oc in enumerate(ocr_chars):
            mapped_x = (oc.get('col', 0) / max_col) * img_max_x
            d = abs(mapped_x - cx)
            if d < best_dist:
                best_dist = d
                best_idx = j
        if best_idx >= 0 and best_dist < image_width * 0.5:
            oc = ocr_chars[best_idx]
            out[i] = {'char_id': oc.get('char') or '?',
                       'ocr_conf': float(oc.get('conf') or 0.0)}
    return out


class _CharThumbStrip(QScrollArea):
    """Horizontal strip of clickable thumbnails (one per char) with label borders."""
    thumbClicked = pyqtSignal(int)            # idx
    labelToggled = pyqtSignal(int, str)       # idx, label

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFixedHeight(140)
        self.setStyleSheet("background-color: #1e1e1e; border: 1px solid #3e3e3e;")
        self._container = QWidget()
        self._layout = QHBoxLayout(self._container)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(6)
        self.setWidget(self._container)
        self._selected = -1

    def populate(self, image, chars, selected=-1):
        # Clear
        while self._layout.count():
            item = self._layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._selected = selected
        if image is None:
            return

        for i, c in enumerate(chars):
            x, y, w, h = c['bbox']
            x = max(0, x); y = max(0, y)
            x2 = min(image.shape[1], x + w); y2 = min(image.shape[0], y + h)
            if x2 <= x or y2 <= y:
                continue
            crop = image[y:y2, x:x2]
            cell = self._make_cell(i, c, crop, selected=(i == selected))
            self._layout.addWidget(cell)
        self._layout.addStretch(1)

    def _make_cell(self, idx, char_data, bgr_crop, selected):
        label = char_data.get('label')
        if label == 'ok':
            border = '#28dc50'
        elif label == 'ng':
            border = '#f03c3c'
        else:
            border = '#a0a0a0'
        if selected:
            border = '#ffc800'

        wrap = QFrame()
        wrap.setStyleSheet(
            f"QFrame {{ background-color: #2a2a2a; border: 2px solid {border}; "
            "border-radius: 4px; padding: 2px; }}"
        )
        wrap.setFixedSize(80, 110)
        wrap.setCursor(Qt.PointingHandCursor)

        v = QVBoxLayout(wrap)
        v.setContentsMargins(2, 2, 2, 2)
        v.setSpacing(2)

        # Char id top
        cid = QLabel(str(char_data.get('char_id') or '?'))
        cid.setAlignment(Qt.AlignCenter)
        cid.setStyleSheet(f"color: {border}; font-weight: bold; font-size: 14px; border: none;")
        v.addWidget(cid)

        # Image
        img_lbl = QLabel()
        img_lbl.setAlignment(Qt.AlignCenter)
        img_lbl.setStyleSheet("border: none;")
        img_lbl.setFixedHeight(54)
        if bgr_crop is not None and bgr_crop.size > 0:
            h, w = bgr_crop.shape[:2]
            buf = np.ascontiguousarray(bgr_crop if bgr_crop.ndim == 3 else
                                        cv2.cvtColor(bgr_crop, cv2.COLOR_GRAY2BGR))
            qimg = QImage(buf.data, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped().copy()
            pix = QPixmap.fromImage(qimg).scaled(70, 50, Qt.KeepAspectRatio,
                                                  Qt.SmoothTransformation)
            img_lbl.setPixmap(pix)
        v.addWidget(img_lbl)

        # Index label bottom
        idx_lbl = QLabel(f"#{idx}  {label or '—'}")
        idx_lbl.setAlignment(Qt.AlignCenter)
        idx_lbl.setStyleSheet("color: #aaa; font-size: 9px; border: none;")
        v.addWidget(idx_lbl)

        wrap.mousePressEvent = lambda ev, i=idx: self.thumbClicked.emit(i)
        return wrap


class CharLabelerWidget(QWidget):
    """Main tab widget — load folder, label chars, save dataset."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.input_folder = None
        self.output_folder = None
        self.image_files = []
        self.current_index = -1
        self.current_image = None
        self.processed_set = set()
        # Cache OCR engines so we don't re-init across image switches
        self._recognizers = {}  # backend_key → recognizer instance

        self._build_ui()
        self._setup_shortcuts()

    # ----------------------------------------------------------------- UI
    def _build_ui(self):
        outer = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter)

        # --- Left panel ---
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(8, 8, 8, 8)

        load_btn = QPushButton("📂 Load source folder…")
        load_btn.clicked.connect(self._load_input_folder)
        lv.addWidget(load_btn)
        self.input_label = QLabel("(no folder loaded)")
        self.input_label.setWordWrap(True)
        self.input_label.setStyleSheet("color: #888;")
        lv.addWidget(self.input_label)

        out_btn = QPushButton("📁 Set output folder…")
        out_btn.clicked.connect(self._set_output_folder)
        lv.addWidget(out_btn)
        self.output_label = QLabel("(no output folder set)")
        self.output_label.setWordWrap(True)
        self.output_label.setStyleSheet("color: #888;")
        lv.addWidget(self.output_label)

        # Segmentation padding controls (extra px around each detected char bbox)
        seg_group = QGroupBox("Segmentation padding")
        sg = QVBoxLayout(seg_group)
        sg.setSpacing(4)
        pad_row = QHBoxLayout()
        pad_row.addWidget(QLabel("W:"))
        from PyQt5.QtWidgets import QSpinBox
        self.pad_w_input = QSpinBox()
        self.pad_w_input.setRange(0, 60)
        self.pad_w_input.setValue(4)
        self.pad_w_input.setToolTip(
            "Extra horizontal padding (px) applied to each segmented char bbox.\n"
            "Useful if segmentation produces tight crops that cut into the strokes."
        )
        pad_row.addWidget(self.pad_w_input)
        pad_row.addWidget(QLabel("H:"))
        self.pad_h_input = QSpinBox()
        self.pad_h_input.setRange(0, 60)
        self.pad_h_input.setValue(4)
        self.pad_h_input.setToolTip("Extra vertical padding (px).")
        pad_row.addWidget(self.pad_h_input)
        pad_row.addStretch(1)
        sg.addLayout(pad_row)
        # Apply to current image without re-running segmentation
        self.apply_pad_btn = QPushButton("Apply padding to current")
        self.apply_pad_btn.setToolTip(
            "Re-apply current padding values to the existing bboxes on this image\n"
            "(without re-running segmentation, so manual edits are kept)."
        )
        self.apply_pad_btn.clicked.connect(self._apply_padding_to_current)
        sg.addWidget(self.apply_pad_btn)
        lv.addWidget(seg_group)

        # OCR controls
        ocr_group = QGroupBox("Auto-label (OCR)")
        og = QVBoxLayout(ocr_group)
        og.setSpacing(4)
        og.addWidget(QLabel("Backend:"))
        self.ocr_backend_combo = QComboBox()
        # PP-OCR rec.onnx is the local model trained on this domain — has
        # per-char confidence built in. RapidOCR is OK for clean printed text
        # but mis-segments on noisy / dot-matrix crops. Tesseract per-char is
        # the fallback for single isolated digits/letters.
        self.ocr_backend_combo.addItem("PP-OCR rec.onnx (full line, per-char conf)", "ppocr")
        self.ocr_backend_combo.addItem("PP-OCR rec.onnx (per char)", "ppocr_per_char")
        self.ocr_backend_combo.addItem("RapidOCR (full line)", "rapidocr")
        self.ocr_backend_combo.addItem("RapidOCR (per char)", "rapidocr_per_char")
        self.ocr_backend_combo.addItem("Tesseract (per char, pytesseract)", "tesseract_per_char")
        self.ocr_backend_combo.addItem("Disabled", "off")
        og.addWidget(self.ocr_backend_combo)

        og.addWidget(QLabel("Char whitelist (Tesseract only):"))
        self.whitelist_input = QLineEdit(
            "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        )
        self.whitelist_input.setToolTip(
            "Tesseract restricts its output to these characters only.\n"
            "Useful when you know your domain (e.g. only digits + uppercase\n"
            "for LOT codes — clears up '0' vs 'O' or '1' vs 'l' confusion).\n\n"
            "Default = digits + A-Z + a-z. Leave empty to allow any char."
        )
        og.addWidget(self.whitelist_input)
        lv.addWidget(ocr_group)

        lv.addWidget(QLabel("Images:"))
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self._on_image_selected)
        lv.addWidget(self.image_list, stretch=1)

        self.progress_label = QLabel("0 / 0")
        self.progress_label.setStyleSheet("color: #aaa;")
        lv.addWidget(self.progress_label)

        splitter.addWidget(left)

        # --- Center panel ---
        center = QWidget()
        cv_layout = QVBoxLayout(center)
        cv_layout.setContentsMargins(8, 8, 8, 8)

        self.title_label = QLabel("No image loaded")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        cv_layout.addWidget(self.title_label)

        # Canvas (image + bboxes)
        canvas_scroll = QScrollArea()
        canvas_scroll.setWidgetResizable(False)
        canvas_scroll.setAlignment(Qt.AlignCenter)
        canvas_scroll.setStyleSheet("background-color: #2b2b2b;")
        canvas_scroll.setMinimumHeight(200)
        self.canvas = CharLabelerCanvas()
        self.canvas.selectionChanged.connect(self._on_canvas_selection)
        self.canvas.bboxesChanged.connect(self._on_canvas_changed)
        canvas_scroll.setWidget(self.canvas)
        cv_layout.addWidget(canvas_scroll, stretch=1)

        # Bbox edit mode toggle
        mode_row = QHBoxLayout()
        self.draw_mode_btn = QPushButton("✏️ Draw new char")
        self.draw_mode_btn.setCheckable(True)
        self.draw_mode_btn.toggled.connect(self._on_draw_mode_toggled)
        mode_row.addWidget(self.draw_mode_btn)
        re_segment_btn = QPushButton("🔁 Re-segment")
        re_segment_btn.clicked.connect(self._re_segment_current)
        mode_row.addWidget(re_segment_btn)
        re_ocr_btn = QPushButton("🔤 Re-OCR")
        re_ocr_btn.clicked.connect(self._re_ocr_current)
        mode_row.addWidget(re_ocr_btn)
        mode_row.addStretch(1)

        # Zoom controls (Ctrl + scroll on canvas also works)
        zoom_out_btn = QPushButton("−")
        zoom_out_btn.setFixedWidth(30)
        zoom_out_btn.setToolTip("Zoom out (Ctrl+−)")
        zoom_out_btn.clicked.connect(self._zoom_out)
        mode_row.addWidget(zoom_out_btn)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(48)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setStyleSheet("color: #aaa;")
        mode_row.addWidget(self.zoom_label)
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedWidth(30)
        zoom_in_btn.setToolTip("Zoom in (Ctrl++)")
        zoom_in_btn.clicked.connect(self._zoom_in)
        mode_row.addWidget(zoom_in_btn)
        zoom_fit_btn = QPushButton("Fit")
        zoom_fit_btn.setToolTip("Reset zoom to fit the viewport (Ctrl+0)")
        zoom_fit_btn.clicked.connect(self._zoom_reset)
        mode_row.addWidget(zoom_fit_btn)
        cv_layout.addLayout(mode_row)

        # Char thumbnail strip
        cv_layout.addWidget(QLabel("Chars:"))
        self.thumb_strip = _CharThumbStrip()
        self.thumb_strip.thumbClicked.connect(self._on_thumb_clicked)
        cv_layout.addWidget(self.thumb_strip)

        # Bulk actions row
        bulk_row = QHBoxLayout()
        b_ok = QPushButton("✓ All OK (A)")
        b_ok.clicked.connect(lambda: self.canvas.bulk_set_label('ok'))
        b_ok.setStyleSheet("color: #28dc50; font-weight: bold;")
        bulk_row.addWidget(b_ok)
        b_ng = QPushButton("✗ All NG (Shift+A)")
        b_ng.clicked.connect(lambda: self.canvas.bulk_set_label('ng'))
        b_ng.setStyleSheet("color: #f03c3c; font-weight: bold;")
        bulk_row.addWidget(b_ng)
        b_reset = QPushButton("Reset")
        b_reset.clicked.connect(lambda: self.canvas.bulk_set_label(None))
        bulk_row.addWidget(b_reset)
        bulk_row.addStretch(1)
        save_btn = QPushButton("💾 Save & Next (Ctrl+S)")
        save_btn.setStyleSheet("background-color: #2a72d4; font-weight: bold; padding: 6px 14px;")
        save_btn.clicked.connect(self._save_and_next)
        bulk_row.addWidget(save_btn)
        skip_btn = QPushButton("Skip (Space)")
        skip_btn.clicked.connect(self._skip_current)
        bulk_row.addWidget(skip_btn)
        cv_layout.addLayout(bulk_row)

        splitter.addWidget(center)

        # --- Right panel ---
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(8, 8, 8, 8)
        right.setMinimumWidth(220)
        right.setMaximumWidth(320)

        info_box = QGroupBox("Selected char")
        ib = QVBoxLayout(info_box)
        self.sel_index_label = QLabel("None")
        self.sel_index_label.setStyleSheet("color: #aaa;")
        ib.addWidget(self.sel_index_label)

        ib.addWidget(QLabel("Char id:"))
        self.char_id_input = QLineEdit()
        self.char_id_input.setMaxLength(2)
        self.char_id_input.editingFinished.connect(self._on_char_id_edited)
        ib.addWidget(self.char_id_input)

        ib.addWidget(QLabel("Label:"))
        lbl_row = QHBoxLayout()
        self.ok_btn = QPushButton("OK (O)")
        self.ok_btn.setStyleSheet("color: #28dc50; font-weight: bold;")
        self.ok_btn.clicked.connect(lambda: self._set_selected_label('ok'))
        lbl_row.addWidget(self.ok_btn)
        self.ng_btn = QPushButton("NG (N)")
        self.ng_btn.setStyleSheet("color: #f03c3c; font-weight: bold;")
        self.ng_btn.clicked.connect(lambda: self._set_selected_label('ng'))
        lbl_row.addWidget(self.ng_btn)
        ib.addLayout(lbl_row)

        del_btn = QPushButton("🗑 Delete char (Del)")
        del_btn.clicked.connect(self.canvas.delete_selected)
        ib.addWidget(del_btn)

        rv.addWidget(info_box)

        # Stats per current image
        self.stats_label = QLabel("")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("color: #aaa; padding: 4px;")
        rv.addWidget(self.stats_label)

        # Shortcut help
        help_box = QGroupBox("Shortcuts")
        hb = QVBoxLayout(help_box)
        hb.addWidget(QLabel(
            "<small>"
            "← → : prev/next char<br/>"
            "↑ ↓ : prev/next image<br/>"
            "<b>0-9 / A-Z</b> : set char_id + advance<br/>"
            "O : mark OK<br/>"
            "N : mark NG<br/>"
            "Shift+O / Shift+N : All OK / All NG<br/>"
            "Ctrl+S : save & next<br/>"
            "Space : skip<br/>"
            "Del : delete char<br/>"
            "Ctrl+ + / − / 0 : zoom in / out / fit<br/>"
            "Ctrl + scroll : zoom on canvas"
            "</small>"
        ))
        rv.addWidget(help_box)

        rv.addStretch(1)
        splitter.addWidget(right)
        splitter.setSizes([260, 800, 260])

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("O"), self, activated=lambda: self._set_selected_label('ok'))
        QShortcut(QKeySequence("Shift+O"), self, activated=lambda: self.canvas.bulk_set_label('ok'))
        QShortcut(QKeySequence("Shift+N"), self, activated=lambda: self.canvas.bulk_set_label('ng'))
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._save_and_next)
        QShortcut(QKeySequence("Space"), self, activated=self._skip_current)
        QShortcut(QKeySequence("Up"), self, activated=lambda: self._step_image(-1))
        QShortcut(QKeySequence("Down"), self, activated=lambda: self._step_image(1))
        # Zoom shortcuts
        QShortcut(QKeySequence(Qt.CTRL + Qt.Key_Plus), self, activated=self._zoom_in)
        QShortcut(QKeySequence(Qt.CTRL + Qt.Key_Equal), self, activated=self._zoom_in)
        QShortcut(QKeySequence(Qt.CTRL + Qt.Key_Minus), self, activated=self._zoom_out)
        QShortcut(QKeySequence(Qt.CTRL + Qt.Key_0), self, activated=self._zoom_reset)
        # NOTE: 0-9, A-Z, plus N (NG) are handled in keyPressEvent so we can
        # also auto-advance to the next char after the keystroke. QShortcut
        # would fire even while editing the QLineEdit, which we don't want.

    def keyPressEvent(self, ev):
        """Fast path for labelling the currently-selected char.
        Skipped if the QLineEdit (char-id text input) has focus so typing in
        the field works normally."""
        if self.char_id_input.hasFocus():
            return super().keyPressEvent(ev)

        idx = self.canvas.selected
        chars = self.canvas.get_chars()
        if 0 <= idx < len(chars):
            text = ev.text()
            key = ev.key()
            if key == Qt.Key_N:
                # Lowercase n → mark as NG (uppercase shift+n handled by shortcut as bulk)
                self._set_selected_label('ng')
                return
            # Single alphanumeric char → set char_id and advance
            if len(text) == 1 and text.isalnum():
                self.canvas.update_selected_char_id(text.upper())
                # Auto-advance to next unlabeled or just next index
                if idx + 1 < len(chars):
                    self.canvas.select_index(idx + 1)
                return
        super().keyPressEvent(ev)

    # ----------------------------------------------------------------- IO

    def _load_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select source folder")
        if not folder:
            return
        self.input_folder = folder
        self.input_label.setText(folder)
        self.image_files = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(_IMG_EXTS)
            and not f.startswith('.')
        ])
        self._reload_processed_set()
        self._refresh_image_list()
        if self.image_files:
            self._select_image(0)

    def _set_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if not folder:
            return
        self.output_folder = folder
        self.output_label.setText(folder)
        self._reload_processed_set()
        self._refresh_image_list()

    def _metadata_path(self):
        if not self.output_folder:
            return None
        return os.path.join(self.output_folder, 'metadata.json')

    def _load_metadata(self):
        p = self._metadata_path()
        if not p or not os.path.exists(p):
            return {'version': 1, 'items': [], 'processed_images': []}
        try:
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[char_labeler] failed to load metadata: {e}")
            return {'version': 1, 'items': [], 'processed_images': []}

    def _save_metadata(self, meta):
        p = self._metadata_path()
        if not p:
            return
        os.makedirs(self.output_folder, exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def _reload_processed_set(self):
        meta = self._load_metadata()
        self.processed_set = set(meta.get('processed_images', []))

    def _refresh_image_list(self):
        # Pre-compute per-source char counts from metadata so each list entry
        # can show "✓ name (12)" — number of chars saved for that image.
        meta = self._load_metadata()
        per_source = {}
        for it in meta.get('items') or []:
            src = it.get('source')
            if not src:
                continue
            entry = per_source.setdefault(src, {'ok': 0, 'ng': 0})
            lbl = it.get('label') or 'ok'
            entry[lbl] = entry.get(lbl, 0) + 1

        from PyQt5.QtGui import QColor, QBrush
        green = QColor(40, 220, 80)
        gray = QColor(170, 170, 170)

        self.image_list.clear()
        for fn in self.image_files:
            done = fn in self.processed_set
            counts = per_source.get(fn, {})
            n_ok = counts.get('ok', 0)
            n_ng = counts.get('ng', 0)
            n_total = n_ok + n_ng
            if done:
                txt = f"✓  {fn}    [{n_total} chars · {n_ok} OK · {n_ng} NG]"
            else:
                txt = f"○  {fn}"
            item = QListWidgetItem(txt)
            item.setForeground(QBrush(green if done else gray))
            if done:
                f = item.font()
                f.setBold(True)
                item.setFont(f)
            self.image_list.addItem(item)

        n_done = len([f for f in self.image_files if f in self.processed_set])
        n_total = len(self.image_files)
        pct = int(100 * n_done / n_total) if n_total else 0
        self.progress_label.setText(
            f"<b>Labeled: {n_done} / {n_total}</b>   ({pct}%)"
        )

    # -------------------------------------------------------- per-image flow

    def _on_image_selected(self, item):
        idx = self.image_list.row(item)
        self._select_image(idx)

    def _select_image(self, idx):
        if not (0 <= idx < len(self.image_files)):
            return
        self.current_index = idx
        self.image_list.setCurrentRow(idx)
        fn = self.image_files[idx]
        path = os.path.join(self.input_folder, fn)
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "Read error", f"Cannot read {fn}")
            return
        self.current_image = img
        self.title_label.setText(f"{idx + 1} / {len(self.image_files)} — {fn}")

        # Auto segment + OCR
        chars = self._auto_label(img)
        self.canvas.set_image(img)
        self.canvas.set_chars(chars)
        self.thumb_strip.populate(img, chars, selected=-1)
        self._refresh_stats()

    def _pad_box(self, x, y, w, h, img_shape):
        """Expand (x, y, w, h) by user-specified padding, clamped to image bounds."""
        pad_w = int(self.pad_w_input.value())
        pad_h = int(self.pad_h_input.value())
        ih, iw = img_shape[:2]
        nx = max(0, x - pad_w)
        ny = max(0, y - pad_h)
        nw = min(iw - nx, w + 2 * pad_w)
        nh = min(ih - ny, h + 2 * pad_h)
        return nx, ny, nw, nh

    def _auto_label(self, img):
        """Segment + run the user-selected OCR strategy. Returns list of char dicts."""
        try:
            from char_segmenter import segment_characters
        except ImportError as e:
            QMessageBox.critical(self, "Missing module", str(e))
            return []

        boxes, _, _, _, _ = segment_characters(img)
        # Apply user padding before any further processing
        boxes = [self._pad_box(x, y, w, h, img.shape) for (x, y, w, h) in boxes]
        boxes = sorted(boxes, key=lambda b: b[0])

        backend = self.ocr_backend_combo.currentData()
        if backend == 'off':
            return [{'bbox': (int(x), int(y), int(w), int(h)),
                     'char_id': '?', 'label': None, 'ocr_conf': None}
                    for (x, y, w, h) in boxes]

        if backend in ('tesseract_per_char', 'rapidocr_per_char', 'ppocr_per_char'):
            return self._label_per_char(img, boxes, backend)

        # Full-line backends: ppocr (default) or rapidocr — both expose
        # `recognize_with_chars(img) → (text, mean_conf, [{char, conf, col}, ...])`
        ocr_chars = []
        try:
            rec = self._get_recognizer(backend)
            _text, _mc, ocr_chars = rec.recognize_with_chars(img)
            ocr_chars = [c for c in ocr_chars if (c.get('char') or '').isalnum()]
            ocr_chars.sort(key=lambda c: c.get('col', 0))
        except Exception as e:
            print(f"[char_labeler] full-line OCR failed: {e}")

        mapped = _spatial_map_chars(boxes, ocr_chars, img.shape[1])
        out = []
        for (x, y, w, h), m in zip(boxes, mapped):
            out.append({
                'bbox': (int(x), int(y), int(w), int(h)),
                'char_id': m.get('char_id') or '?',
                'label': None,
                'ocr_conf': m.get('ocr_conf'),
            })
        return out

    def _label_per_char(self, img, boxes, backend):
        """OCR each segmented char crop independently — robust on noisy
        / dot-matrix text where full-line OCR mis-segments internally."""
        out = []
        try:
            rec = self._get_recognizer(backend)
        except Exception as e:
            print(f"[char_labeler] cannot init {backend}: {e}")
            for (x, y, w, h) in boxes:
                out.append({'bbox': (int(x), int(y), int(w), int(h)),
                            'char_id': '?', 'label': None, 'ocr_conf': None})
            return out

        for (x, y, w, h) in boxes:
            x1 = max(0, x); y1 = max(0, y)
            x2 = min(img.shape[1], x + w); y2 = min(img.shape[0], y + h)
            crop = img[y1:y2, x1:x2]
            try:
                text, conf = rec.recognize(crop)
            except Exception as e:
                print(f"[char_labeler] per-char OCR error: {e}")
                text, conf = '', 0.0
            ch = ''.join(c for c in (text or '') if c.isalnum())[:1] or '?'
            out.append({
                'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                'char_id': ch,
                'label': None,
                'ocr_conf': float(conf),
            })
        return out

    def _get_recognizer(self, backend):
        """Lazy-init + cache. Tesseract cache key includes whitelist so that
        editing the whitelist mid-session forces a fresh init."""
        if backend == 'tesseract_per_char':
            whitelist = self.whitelist_input.text().strip() or ''
            cache_key = f'tesseract_per_char|{whitelist}'
        else:
            cache_key = backend

        if cache_key in self._recognizers:
            return self._recognizers[cache_key]

        if backend in ('ppocr', 'ppocr_per_char'):
            # Local PP-OCRv5 ONNX model — has per-char confidence and is the
            # one originally trained for this domain (LOT codes, datecodes).
            from text_recognizer import TextRecognizer
            rec = TextRecognizer(
                model_path='../languages/english/rec.onnx',
                dict_path='../languages/english/dict.txt',
                use_gpu=False,
            )
        elif backend in ('rapidocr', 'rapidocr_per_char'):
            from text_recognizer_rapidocr import TextRecognizerRapidOCR
            rec = TextRecognizerRapidOCR()
        elif backend == 'tesseract_per_char':
            from text_recognizer_tesseract import TextRecognizerTesseract
            # Force pytesseract on macOS — `tesserocr` C-binding crashes with
            # `cysignals sigaltstack: Operation not permitted` due to signal
            # handler conflicts. pytesseract spawns a subprocess (slower, ~50ms)
            # but doesn't have that issue.
            # PSM 10 = treat the image as a single character.
            rec = TextRecognizerTesseract(
                lang='eng', psm=10, oem=1,
                char_whitelist=whitelist or None,
                library='pytesseract',
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self._recognizers[cache_key] = rec
        return rec

    def _re_segment_current(self):
        if self.current_image is None:
            return
        # Keep labels for chars whose bbox center still falls inside a new bbox
        old = self.canvas.get_chars()
        try:
            from char_segmenter import segment_characters
        except ImportError:
            return
        boxes, _, _, _, _ = segment_characters(self.current_image)
        boxes = [self._pad_box(x, y, w, h, self.current_image.shape)
                 for (x, y, w, h) in boxes]
        boxes = sorted(boxes, key=lambda b: b[0])
        new_chars = []
        for (x, y, w, h) in boxes:
            cx = x + w / 2; cy = y + h / 2
            inherited = None
            for o in old:
                ox, oy, ow, oh = o['bbox']
                if ox <= cx <= ox + ow and oy <= cy <= oy + oh:
                    inherited = o
                    break
            new_chars.append({
                'bbox': (int(x), int(y), int(w), int(h)),
                'char_id': (inherited or {}).get('char_id', '?'),
                'label': (inherited or {}).get('label'),
                'ocr_conf': (inherited or {}).get('ocr_conf'),
            })
        self.canvas.set_chars(new_chars)
        self.thumb_strip.populate(self.current_image, new_chars, selected=-1)

    def _re_ocr_current(self):
        """Re-run OCR using the currently selected backend, keeping the
        existing bbox positions and overwriting only the char_id/ocr_conf."""
        if self.current_image is None:
            return
        backend = self.ocr_backend_combo.currentData()
        if backend == 'off':
            return

        chars = self.canvas.get_chars()
        boxes = [c['bbox'] for c in chars]
        if not boxes:
            return

        try:
            if backend in ('tesseract_per_char', 'rapidocr_per_char', 'ppocr_per_char'):
                # Per-char: feed each crop into the selected engine
                relabel = self._label_per_char(self.current_image, boxes, backend)
                for c, m in zip(chars, relabel):
                    c['char_id'] = m.get('char_id') or '?'
                    c['ocr_conf'] = m.get('ocr_conf')
            else:
                # Full-line backend (ppocr or rapidocr) + spatial mapping
                rec = self._get_recognizer(backend)
                _t, _mc, ocr_chars = rec.recognize_with_chars(self.current_image)
                ocr_chars = [c for c in ocr_chars if (c.get('char') or '').isalnum()]
                ocr_chars.sort(key=lambda c: c.get('col', 0))
                mapped = _spatial_map_chars(boxes, ocr_chars,
                                             self.current_image.shape[1])
                for c, m in zip(chars, mapped):
                    c['char_id'] = m.get('char_id') or '?'
                    c['ocr_conf'] = m.get('ocr_conf')
        except Exception as e:
            QMessageBox.critical(self, "OCR error", str(e))
            return

        self.canvas.set_chars(chars)
        self.thumb_strip.populate(self.current_image, chars,
                                    selected=self.canvas.selected)

    # -------------------------------------------------------- selection / edit

    def _on_canvas_selection(self, idx):
        chars = self.canvas.get_chars()
        if 0 <= idx < len(chars):
            c = chars[idx]
            self.sel_index_label.setText(
                f"#{idx} — char_id={c.get('char_id') or '?'}, "
                f"label={c.get('label') or '—'}, "
                f"ocr_conf={c.get('ocr_conf')!r}"
            )
            self.char_id_input.blockSignals(True)
            self.char_id_input.setText(str(c.get('char_id') or ''))
            self.char_id_input.blockSignals(False)
        else:
            self.sel_index_label.setText("None")
            self.char_id_input.clear()
        # Sync thumbnail strip selection highlight
        self.thumb_strip.populate(self.current_image, chars, selected=idx)

    def _on_canvas_changed(self):
        chars = self.canvas.get_chars()
        self.thumb_strip.populate(self.current_image, chars,
                                    selected=self.canvas.selected)
        self._refresh_stats()

    def _on_thumb_clicked(self, idx):
        self.canvas.select_index(idx)

    def _on_char_id_edited(self):
        new_id = self.char_id_input.text().strip()
        self.canvas.update_selected_char_id(new_id or '?')

    def _on_draw_mode_toggled(self, on):
        self.canvas.set_mode(self.canvas.MODE_DRAW if on else self.canvas.MODE_SELECT)

    def _apply_padding_to_current(self):
        """Re-pad existing bboxes on the current image using current pad W/H values.
        Useful when the user tweaks padding after manual edits — segmentation is
        not re-run (so manual additions/edits are preserved). To make this idempotent,
        the new bbox is computed from each char's current bbox treated as the
        already-padded shape: we shrink by the previously-applied padding first.
        Since we don't track previous padding per char, we just re-expand from
        the current visible bbox — call multiple times → keeps growing. Reset
        with Re-segment if needed."""
        if self.current_image is None:
            return
        chars = self.canvas.get_chars()
        if not chars:
            return
        for c in chars:
            x, y, w, h = c['bbox']
            c['bbox'] = self._pad_box(x, y, w, h, self.current_image.shape)
        self.canvas.set_chars(chars)
        self.thumb_strip.populate(self.current_image, chars,
                                    selected=self.canvas.selected)

    def _zoom_in(self):
        self.canvas.zoom_in()
        self._refresh_zoom_label()

    def _zoom_out(self):
        self.canvas.zoom_out()
        self._refresh_zoom_label()

    def _zoom_reset(self):
        self.canvas.zoom_reset()
        self._refresh_zoom_label()

    def _refresh_zoom_label(self):
        pct = int(round(self.canvas.get_user_zoom() * 100))
        self.zoom_label.setText(f"{pct}%")

    def _set_selected_label(self, label):
        self.canvas.update_selected_label(label)

    def _refresh_stats(self):
        chars = self.canvas.get_chars()
        n = len(chars)
        n_ok = sum(1 for c in chars if c.get('label') == 'ok')
        n_ng = sum(1 for c in chars if c.get('label') == 'ng')
        n_un = n - n_ok - n_ng
        self.stats_label.setText(
            f"<b>This image:</b> {n} chars | "
            f"<span style='color:#28dc50'>{n_ok} OK</span> | "
            f"<span style='color:#f03c3c'>{n_ng} NG</span> | "
            f"<span style='color:#a0a0a0'>{n_un} unlabeled</span>"
        )

    # -------------------------------------------------------- save / nav

    def _step_image(self, delta):
        n = len(self.image_files)
        if n == 0:
            return
        idx = (self.current_index + delta) % n
        self._select_image(idx)

    def _skip_current(self):
        self._step_image(1)

    def _save_and_next(self):
        if self.current_image is None or self.current_index < 0:
            return
        if not self.output_folder:
            QMessageBox.warning(self, "No output folder",
                                 "Please set an output folder first.")
            return

        chars = self.canvas.get_chars()
        if not chars:
            QMessageBox.information(self, "Nothing to save",
                                     "No char bboxes on this image.")
            return

        # Warn if there are unlabeled chars
        unlabeled = sum(1 for c in chars if c.get('label') is None)
        if unlabeled > 0:
            reply = QMessageBox.question(
                self, "Unlabeled chars",
                f"{unlabeled} chars are unlabeled. Save anyway "
                f"(they will be tagged 'ok' by default)?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        fn = self.image_files[self.current_index]
        stem = os.path.splitext(fn)[0]

        meta = self._load_metadata()
        # Drop any prior items for this source — re-saving the same image
        # should not produce duplicates.
        meta['items'] = [it for it in meta.get('items', [])
                         if it.get('source') != fn]

        for idx, c in enumerate(chars):
            x, y, w, h = c['bbox']
            x = max(0, x); y = max(0, y)
            x2 = min(self.current_image.shape[1], x + w)
            y2 = min(self.current_image.shape[0], y + h)
            if x2 <= x or y2 <= y:
                continue
            crop = self.current_image[y:y2, x:x2]
            char_id = (c.get('char_id') or '?').strip() or '?'
            char_id_safe = ''.join(ch for ch in char_id if ch.isalnum()) or 'unknown'
            # Use `_low` suffix for lowercase letters so case-insensitive
            # filesystems (macOS, default Windows) don't collapse `char_A`
            # and `char_a` into the same folder. Digits + uppercase keep
            # their original folder name for backward compat with old datasets.
            if char_id_safe.islower() and char_id_safe.isalpha():
                folder = f'char_{char_id_safe}_low'
            else:
                folder = f'char_{char_id_safe}'
            label = c.get('label') or 'ok'
            sub = os.path.join(self.output_folder, folder, label)
            os.makedirs(sub, exist_ok=True)
            out_name = f"{stem}_{idx}.png"
            out_path = os.path.join(sub, out_name)
            cv2.imwrite(out_path, crop)
            meta['items'].append({
                'source': fn,
                'char_idx': idx,
                'char_id': char_id,
                'label': label,
                'bbox': {'x': int(x), 'y': int(y), 'w': int(x2 - x), 'h': int(y2 - y)},
                'ocr_conf': c.get('ocr_conf'),
                'saved_path': os.path.relpath(out_path, self.output_folder),
                'saved_at': datetime.now().isoformat(timespec='seconds'),
            })

        processed = set(meta.get('processed_images', []))
        processed.add(fn)
        meta['processed_images'] = sorted(processed)
        self._save_metadata(meta)
        self.processed_set = processed
        self._refresh_image_list()

        # Advance
        if self.current_index < len(self.image_files) - 1:
            self._select_image(self.current_index + 1)
