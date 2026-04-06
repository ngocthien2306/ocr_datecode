import sys
import os
import tempfile
import numpy as np
import cv2 as cv

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QFileDialog, QSlider, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QScrollArea, QSplitter, QListWidget,
    QListWidgetItem, QSizePolicy, QHeaderView, QGroupBox, QAbstractItemView,
    QMenu, QAction
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_segment import segment_characters, compute_char_quality, _save_comparison_strip

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
THUMB_SIZE = 72


def cv2_to_pixmap(img_bgr: np.ndarray) -> QPixmap:
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    return QPixmap.fromImage(
        QImage(img_rgb.data, w, h, w * ch, QImage.Format_RGB888)
    )


def load_thumb(path: str) -> QIcon:
    img = cv.imread(path)
    if img is None:
        return QIcon()
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return QIcon()
    scale = THUMB_SIZE / max(h, w)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv.resize(img, (nw, nh))
    return QIcon(cv2_to_pixmap(resized))


class WorkerThread(QThread):
    done  = pyqtSignal(np.ndarray, list)
    error = pyqtSignal(str)

    def __init__(self, tmpl_data, tgt_path, threshold, output_dir):
        super().__init__()
        self.tmpl_data  = tmpl_data
        self.tgt_path   = tgt_path
        self.threshold  = threshold
        self.output_dir = output_dir

    def run(self):
        try:
            tmpl_boxes, tmpl_chars, tmpl_thresh_chars, tmpl_img, tmpl_thresh = self.tmpl_data

            tgt_boxes, tgt_chars, tgt_thresh_chars, tgt_img, tgt_thresh = \
                segment_characters(self.tgt_path,
                                   os.path.join(self.output_dir, "target"),
                                   save=False)

            if not tmpl_chars or not tgt_chars:
                self.error.emit("Không tìm thấy ký tự trong ảnh.")
                return

            n = min(len(tmpl_chars), len(tgt_chars))
            results = []
            for i in range(n):
                metrics = compute_char_quality(tmpl_thresh_chars[i], tgt_thresh_chars[i])
                quality = "PASS" if metrics["confidence"] >= self.threshold else "FAIL"
                results.append((i, metrics, quality))

            _save_comparison_strip(
                tmpl_chars, tgt_chars, results, self.output_dir,
                tmpl_img, tmpl_thresh, tgt_img, tgt_thresh
            )

            comparison = cv.imread(os.path.join(self.output_dir, "comparison.png"))
            self.done.emit(comparison, results)

        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Character Quality Comparison")
        self.resize(1600, 960)

        self.tmpl_data    = None
        self.tmpl_path    = None
        self.image_files  = []       # all images in loaded folder
        self.current_idx  = -1       # current target index
        self.output_dir   = tempfile.mkdtemp()
        self.worker       = None

        self._build_ui()

    # ═══════════════════════════════════════════════════════════ Build UI ══
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 4)
        main_layout.setSpacing(6)

        # ── Top bar: load folder + threshold ──────────────────────────────
        top = QHBoxLayout()

        btn_load = QPushButton("📂  Load Folder")
        btn_load.setFixedHeight(32)
        btn_load.clicked.connect(self._load_folder)
        top.addWidget(btn_load)

        top.addWidget(QLabel("  Ngưỡng PASS/FAIL:"))
        self.thr_slider = QSlider(Qt.Horizontal)
        self.thr_slider.setRange(0, 100)
        self.thr_slider.setValue(75)
        self.thr_slider.setFixedWidth(200)
        self.thr_slider.valueChanged.connect(self._slider_changed)
        top.addWidget(self.thr_slider)

        self.thr_spin = QDoubleSpinBox()
        self.thr_spin.setRange(0.0, 1.0)
        self.thr_spin.setSingleStep(0.01)
        self.thr_spin.setDecimals(2)
        self.thr_spin.setValue(0.75)
        self.thr_spin.setFixedWidth(68)
        self.thr_spin.valueChanged.connect(self._spin_changed)
        top.addWidget(self.thr_spin)
        top.addStretch()

        self.tmpl_label = QLabel("Template: —")
        self.tmpl_label.setStyleSheet("color:#aaa;")
        top.addWidget(self.tmpl_label)

        main_layout.addLayout(top)

        # ── Main splitter ──────────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)

        # ── LEFT: image list ───────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(210)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self.list_widget.setSpacing(2)
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._list_context_menu)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        left_layout.addWidget(self.list_widget)

        btn_set_tmpl = QPushButton("⭐  Đặt làm Template")
        btn_set_tmpl.clicked.connect(self._set_selected_as_template)
        left_layout.addWidget(btn_set_tmpl)

        self.list_status = QLabel("0 ảnh")
        self.list_status.setStyleSheet("color:#888;font-size:11px;")
        left_layout.addWidget(self.list_status)

        splitter.addWidget(left)

        # ── MIDDLE: comparison result ──────────────────────────────────────
        mid = QWidget()
        mid_layout = QVBoxLayout(mid)
        mid_layout.setContentsMargins(4, 0, 4, 0)
        mid_layout.setSpacing(6)

        # Navigation
        nav = QHBoxLayout()
        self.btn_prev = QPushButton("◀  Prev")
        self.btn_prev.setFixedWidth(90)
        self.btn_prev.clicked.connect(self._prev)
        nav.addWidget(self.btn_prev)
        self.nav_label = QLabel("—")
        self.nav_label.setAlignment(Qt.AlignCenter)
        nav.addWidget(self.nav_label, 1)
        self.btn_next = QPushButton("Next  ▶")
        self.btn_next.setFixedWidth(90)
        self.btn_next.clicked.connect(self._next)
        nav.addWidget(self.btn_next)
        mid_layout.addLayout(nav)

        # Banner
        self.banner = QLabel("—")
        self.banner.setAlignment(Qt.AlignCenter)
        f = QFont()
        f.setPointSize(15)
        f.setBold(True)
        self.banner.setFont(f)
        self.banner.setFixedHeight(44)
        self.banner.setStyleSheet("background:#444;color:white;border-radius:4px;")
        mid_layout.addWidget(self.banner)

        # Comparison image (scrollable, fills remaining space)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignCenter)
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.img_label.setStyleSheet("background:#2a2a2a;")
        self.scroll.setWidget(self.img_label)
        mid_layout.addWidget(self.scroll, 1)

        splitter.addWidget(mid)

        # ── RIGHT: results table ───────────────────────────────────────────
        right = QWidget()
        right.setFixedWidth(360)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        tbl_label = QLabel("Chi tiết từng ký tự")
        tbl_label.setStyleSheet("font-weight:bold;padding:4px;")
        right_layout.addWidget(tbl_label)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["#", "Final", "TM", "Px Conf", "Px Tmpl", "Px Tgt"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        right_layout.addWidget(self.table, 1)

        splitter.addWidget(right)
        splitter.setSizes([210, 1030, 360])
        main_layout.addWidget(splitter, 1)

        # Status bar
        self.status = QLabel("Sẵn sàng. Load folder để bắt đầu.")
        self.statusBar().addWidget(self.status)

    # ══════════════════════════════════════════════════════════ Load folder ══
    def _load_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Chọn thư mục ảnh")
        if not path:
            return

        self.image_files = sorted([
            os.path.join(path, f) for f in os.listdir(path)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ])
        self.current_idx = -1
        self.tmpl_data   = None
        self.tmpl_path   = None
        self.tmpl_label.setText("Template: —")

        self.list_widget.clear()
        for fp in self.image_files:
            item = QListWidgetItem(load_thumb(fp), os.path.basename(fp))
            item.setData(Qt.UserRole, fp)
            item.setSizeHint(QSize(190, THUMB_SIZE + 24))
            self.list_widget.addItem(item)

        self.list_status.setText(f"{len(self.image_files)} ảnh")
        self.status.setText(f"Đã load {len(self.image_files)} ảnh từ: {path}")

    # ══════════════════════════════════════════════════════ Template / Target ══
    def _set_selected_as_template(self):
        item = self.list_widget.currentItem()
        if item is None:
            return
        self._set_template(item.data(Qt.UserRole))

    def _set_template(self, path: str):
        self.tmpl_path = path
        self.tmpl_data = None  # reset cache
        fname = os.path.basename(path)
        self.tmpl_label.setText(f"Template: {fname}")
        self.status.setText(f"Template đã đặt: {fname}")

        # Đánh dấu item template trong list
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.UserRole) == path:
                item.setBackground(QColor("#1a4a1a"))
                item.setText(f"⭐ {os.path.basename(path)}")
            else:
                item.setBackground(QColor(0, 0, 0, 0))
                item.setText(os.path.basename(item.data(Qt.UserRole)))

    def _on_item_clicked(self, item: QListWidgetItem):
        path = item.data(Qt.UserRole)
        idx  = self.image_files.index(path)
        self.current_idx = idx
        self._update_nav_label()
        self._run_current()

    def _list_context_menu(self, pos):
        item = self.list_widget.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        act_tmpl = QAction("⭐ Đặt làm Template", self)
        act_run  = QAction("▶  Chạy so sánh", self)
        act_tmpl.triggered.connect(lambda: self._set_template(item.data(Qt.UserRole)))
        act_run.triggered.connect(lambda: self._on_item_clicked(item))
        menu.addAction(act_tmpl)
        menu.addAction(act_run)
        menu.exec_(self.list_widget.mapToGlobal(pos))

    # ══════════════════════════════════════════════════════════ Navigation ══
    def _prev(self):
        if not self.image_files:
            return
        self.current_idx = (self.current_idx - 1) % len(self.image_files)
        self._sync_list_selection()
        self._update_nav_label()
        self._run_current()

    def _next(self):
        if not self.image_files:
            return
        self.current_idx = (self.current_idx + 1) % len(self.image_files)
        self._sync_list_selection()
        self._update_nav_label()
        self._run_current()

    def _sync_list_selection(self):
        self.list_widget.setCurrentRow(self.current_idx)

    def _update_nav_label(self):
        if 0 <= self.current_idx < len(self.image_files):
            fname = os.path.basename(self.image_files[self.current_idx])
            self.nav_label.setText(
                f"{self.current_idx + 1} / {len(self.image_files)}  —  {fname}"
            )
        else:
            self.nav_label.setText("—")

    # ══════════════════════════════════════════════════════════ Run ══════════
    def _run_current(self):
        if not self.tmpl_path:
            self.status.setText("Chưa đặt template — right-click ảnh → Đặt làm Template.")
            return
        if self.current_idx < 0 or self.current_idx >= len(self.image_files):
            return
        if self.worker and self.worker.isRunning():
            return

        tgt_path = self.image_files[self.current_idx]
        if tgt_path == self.tmpl_path:
            self.status.setText("Target trùng template, bỏ qua.")
            return

        # Cache template segmentation
        if self.tmpl_data is None:
            self.status.setText("Đang segment template...")
            QApplication.processEvents()
            tmpl_dir = os.path.join(self.output_dir, "template")
            self.tmpl_data = segment_characters(self.tmpl_path, tmpl_dir, save=True)
            if self.tmpl_data[0] is None:
                self.status.setText("Lỗi: không đọc được template.")
                return

        self._set_buttons_enabled(False)
        self.status.setText(f"Đang xử lý: {os.path.basename(tgt_path)} ...")

        self.worker = WorkerThread(
            self.tmpl_data, tgt_path,
            self.thr_spin.value(), self.output_dir
        )
        self.worker.done.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    # ══════════════════════════════════════════════════════ Result display ══
    def _on_done(self, comparison_img: np.ndarray, results: list):
        self._set_buttons_enabled(True)
        self.status.setText("Hoàn thành.")

        # Scale ảnh vừa với chiều rộng scroll area
        if comparison_img is not None:
            pixmap = cv2_to_pixmap(comparison_img)
            avail_w = self.scroll.viewport().width() - 4
            if pixmap.width() < avail_w:
                pixmap = pixmap.scaledToWidth(avail_w, Qt.SmoothTransformation)
            self.img_label.setPixmap(pixmap)
            self.img_label.resize(pixmap.size())

        # Banner
        overall_pass = all(r[2] == "PASS" for r in results)
        if overall_pass:
            self.banner.setText("✔   PASS")
            self.banner.setStyleSheet(
                "background:#1a7f1a;color:white;border-radius:4px;"
            )
        else:
            fail_n = sum(1 for r in results if r[2] == "FAIL")
            self.banner.setText(f"✘   FAIL  ({fail_n} ký tự lỗi)")
            self.banner.setStyleSheet(
                "background:#b01a1a;color:white;border-radius:4px;"
            )

        # Table
        self.table.setRowCount(len(results))
        for row, (i, metrics, quality) in enumerate(results):
            vals = [
                str(i),
                f"{metrics['confidence']:.3f}",
                f"{metrics['tm_conf']:.3f}",
                f"{metrics['pixel_conf']:.3f}",
                str(metrics['px_tmpl']),
                str(metrics['px_tgt']),
            ]
            color = QColor("#1aaa1a") if quality == "PASS" else QColor("#dd3333")
            for col, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                item.setForeground(color)
                self.table.setItem(row, col, item)

    def _on_error(self, msg: str):
        self._set_buttons_enabled(True)
        self.status.setText("Lỗi! Xem console.")
        print(msg)

    # ══════════════════════════════════════════════════════════ Threshold ══
    def _slider_changed(self, val):
        self.thr_spin.blockSignals(True)
        self.thr_spin.setValue(val / 100.0)
        self.thr_spin.blockSignals(False)

    def _spin_changed(self, val):
        self.thr_slider.blockSignals(True)
        self.thr_slider.setValue(int(val * 100))
        self.thr_slider.blockSignals(False)

    def _set_buttons_enabled(self, enabled: bool):
        self.btn_prev.setEnabled(enabled)
        self.btn_next.setEnabled(enabled)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
