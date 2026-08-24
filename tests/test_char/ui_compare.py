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
    QMenu, QAction, QCheckBox, QMessageBox, QDialog, QTabWidget, QGridLayout
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_segment import segment_characters, compute_char_quality, _save_comparison_strip
from ml_classifier import train_model, predict_image, load_model

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
THUMB_SIZE = 72
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rf_model.joblib")


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


# ═══════════════════════════════════════════════════ Worker Threads ══

class WorkerThread(QThread):
    """Template-matching comparison worker (original mode)."""
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


class TrainWorker(QThread):
    """Background worker for RF training."""
    done  = pyqtSignal(str, str)   # info message, save_dir
    error = pyqtSignal(str)

    def __init__(self, ok_paths, model_path, save_dir):
        super().__init__()
        self.ok_paths   = ok_paths
        self.model_path = model_path
        self.save_dir   = save_dir

    def run(self):
        try:
            model, info = train_model(self.ok_paths, n_augments=5,
                                      model_path=self.model_path,
                                      save_dir=self.save_dir)
            msg = (f"Train xong! OK={info['n_ok']}, NG(synth)={info['n_ng']}, "
                   f"Total={info['n_total']}, Accuracy={info['accuracy_train']:.3f}")
            self.done.emit(msg, self.save_dir)
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


class PredictWorker(QThread):
    """Background worker for RF prediction."""
    done  = pyqtSignal(list, np.ndarray, np.ndarray)  # results, img, thresh
    error = pyqtSignal(str)

    def __init__(self, model, tgt_path, threshold):
        super().__init__()
        self.model     = model
        self.tgt_path  = tgt_path
        self.threshold = threshold

    def run(self):
        try:
            results, img, thresh = predict_image(self.model, self.tgt_path,
                                                 self.threshold)
            if not results:
                self.error.emit("Không tìm thấy ký tự trong ảnh target.")
                return
            self.done.emit(results, img, thresh)
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


# ═══════════════════════════════════════════════ Crop Viewer Dialog ══

CROP_CELL = 80   # display size (px) for each crop thumbnail


def _make_crop_pixmap(img_path: str) -> QPixmap:
    img = cv.imread(img_path)
    if img is None:
        return QPixmap()
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return QPixmap()
    # Upscale small crops so they're visible
    scale = CROP_CELL / max(h, w)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv.resize(img, (nw, nh), interpolation=cv.INTER_NEAREST)
    return cv2_to_pixmap(resized)


class CropViewerDialog(QDialog):
    """Grid viewer for OK and NG (augmented) character crops."""

    def __init__(self, save_dir: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training Crops Viewer")
        self.resize(1000, 700)
        self.save_dir = save_dir
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        info = QLabel(f"Folder: {self.save_dir}")
        info.setStyleSheet("color:#aaa;font-size:11px;")
        layout.addWidget(info)

        tabs = QTabWidget()

        ok_dir = os.path.join(self.save_dir, "ok_chars")
        ng_dir = os.path.join(self.save_dir, "ng_chars")

        tabs.addTab(self._make_grid_tab(ok_dir, "#0a3a0a", "OK"), "OK chars")
        tabs.addTab(self._make_grid_tab(ng_dir, "#3a0a0a", "NG (augmented)"), "NG (augmented)")

        layout.addWidget(tabs, 1)

        btn_open = QPushButton("Open folder in Finder")
        btn_open.clicked.connect(lambda: os.system(f'open "{self.save_dir}"'))
        layout.addWidget(btn_open)

    def _make_grid_tab(self, folder: str, bg_color: str, label: str) -> QWidget:
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(4, 4, 4, 4)

        if not os.path.isdir(folder):
            vbox.addWidget(QLabel(f"Folder not found: {folder}"))
            return container

        files = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(".png")
        ])

        count_label = QLabel(f"{len(files)} images")
        count_label.setStyleSheet("font-weight:bold;padding:2px;")
        vbox.addWidget(count_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner.setStyleSheet(f"background:{bg_color};")
        grid = QGridLayout(inner)
        grid.setSpacing(4)
        grid.setContentsMargins(6, 6, 6, 6)

        COLS = 12
        for idx, fpath in enumerate(files):
            row, col = divmod(idx, COLS)
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(2, 2, 2, 2)
            cell_layout.setSpacing(1)

            pm = _make_crop_pixmap(fpath)
            img_lbl = QLabel()
            img_lbl.setFixedSize(CROP_CELL, CROP_CELL)
            img_lbl.setAlignment(Qt.AlignCenter)
            img_lbl.setStyleSheet("background:#1a1a1a;border:1px solid #444;")
            if not pm.isNull():
                img_lbl.setPixmap(pm)
            cell_layout.addWidget(img_lbl)

            name_lbl = QLabel(os.path.basename(fpath))
            name_lbl.setAlignment(Qt.AlignCenter)
            name_lbl.setStyleSheet("color:#bbb;font-size:8px;")
            name_lbl.setWordWrap(True)
            name_lbl.setFixedWidth(CROP_CELL)
            cell_layout.addWidget(name_lbl)

            grid.addWidget(cell, row, col)

        scroll.setWidget(inner)
        vbox.addWidget(scroll, 1)
        return container


# ═══════════════════════════════════════════════════ Main Window ══

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Character Quality Comparison")
        self.resize(1600, 960)

        self.tmpl_data    = None
        self.tmpl_path    = None
        self.image_files  = []
        self.current_idx  = -1
        self.output_dir   = tempfile.mkdtemp()
        self.worker       = None

        # ML state
        self.rf_model     = None
        self.ok_paths     = set()   # paths marked as OK for training
        self.ml_mode      = False
        self.crops_dir    = None    # dir where training crops were saved

        # Load saved model if exists
        if os.path.exists(MODEL_PATH):
            try:
                self.rf_model = load_model(MODEL_PATH)
            except Exception:
                pass

        self._build_ui()

    # ═══════════════════════════════════════════════════════════ Build UI ══
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 4)
        main_layout.setSpacing(6)

        # ── Top bar: load folder + threshold + mode toggle ────────────────
        top = QHBoxLayout()

        btn_load = QPushButton("Load Folder")
        btn_load.setFixedHeight(32)
        btn_load.clicked.connect(self._load_folder)
        top.addWidget(btn_load)

        top.addWidget(QLabel("  Threshold:"))
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

        # Mode toggle
        self.chk_ml = QCheckBox("ML Mode (Random Forest)")
        self.chk_ml.setChecked(False)
        self.chk_ml.toggled.connect(self._toggle_ml_mode)
        top.addWidget(self.chk_ml)

        top.addStretch()

        self.tmpl_label = QLabel("Template: --")
        self.tmpl_label.setStyleSheet("color:#aaa;")
        top.addWidget(self.tmpl_label)

        main_layout.addLayout(top)

        # ── ML control bar (hidden by default) ────────────────────────────
        self.ml_bar = QWidget()
        ml_layout = QHBoxLayout(self.ml_bar)
        ml_layout.setContentsMargins(0, 0, 0, 0)
        ml_layout.setSpacing(8)

        self.btn_mark_ok = QPushButton("Mark as OK")
        self.btn_mark_ok.setToolTip("Danh dau anh dang chon la OK de training")
        self.btn_mark_ok.clicked.connect(self._mark_selected_ok)
        ml_layout.addWidget(self.btn_mark_ok)

        self.ok_count_label = QLabel("OK: 0")
        self.ok_count_label.setStyleSheet("color:#0a0;font-weight:bold;")
        ml_layout.addWidget(self.ok_count_label)

        self.btn_train = QPushButton("Train RF")
        self.btn_train.setStyleSheet("background:#1a6a1a;color:white;padding:4px 12px;")
        self.btn_train.clicked.connect(self._train_rf)
        ml_layout.addWidget(self.btn_train)

        self.btn_load_model = QPushButton("Load Model")
        self.btn_load_model.clicked.connect(self._load_model_dialog)
        ml_layout.addWidget(self.btn_load_model)

        self.btn_save_model = QPushButton("Save Model")
        self.btn_save_model.clicked.connect(self._save_model_dialog)
        ml_layout.addWidget(self.btn_save_model)

        self.model_status = QLabel("Model: not loaded" if self.rf_model is None
                                   else "Model: loaded")
        self.model_status.setStyleSheet("color:#888;")
        ml_layout.addWidget(self.model_status)

        self.btn_view_crops = QPushButton("View Crops")
        self.btn_view_crops.setStyleSheet("padding:4px 12px;")
        self.btn_view_crops.setEnabled(False)
        self.btn_view_crops.clicked.connect(self._view_crops)
        ml_layout.addWidget(self.btn_view_crops)

        ml_layout.addStretch()
        self.ml_bar.setVisible(False)
        main_layout.addWidget(self.ml_bar)

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
        self.list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._list_context_menu)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        left_layout.addWidget(self.list_widget)

        btn_set_tmpl = QPushButton("Set as Template")
        btn_set_tmpl.clicked.connect(self._set_selected_as_template)
        left_layout.addWidget(btn_set_tmpl)

        self.list_status = QLabel("0 images")
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
        self.btn_prev = QPushButton("< Prev")
        self.btn_prev.setFixedWidth(90)
        self.btn_prev.clicked.connect(self._prev)
        nav.addWidget(self.btn_prev)
        self.nav_label = QLabel("--")
        self.nav_label.setAlignment(Qt.AlignCenter)
        nav.addWidget(self.nav_label, 1)
        self.btn_next = QPushButton("Next >")
        self.btn_next.setFixedWidth(90)
        self.btn_next.clicked.connect(self._next)
        nav.addWidget(self.btn_next)
        mid_layout.addLayout(nav)

        # Banner
        self.banner = QLabel("--")
        self.banner.setAlignment(Qt.AlignCenter)
        f = QFont()
        f.setPointSize(15)
        f.setBold(True)
        self.banner.setFont(f)
        self.banner.setFixedHeight(44)
        self.banner.setStyleSheet("background:#444;color:white;border-radius:4px;")
        mid_layout.addWidget(self.banner)

        # Comparison image (scrollable)
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

        tbl_label = QLabel("Character details")
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
        self.status = QLabel("Ready. Load folder to start.")
        self.statusBar().addWidget(self.status)

    # ══════════════════════════════════════════════════════ ML Mode Toggle ══
    def _toggle_ml_mode(self, checked):
        self.ml_mode = checked
        self.ml_bar.setVisible(checked)
        if checked:
            self.table.setColumnCount(4)
            self.table.setHorizontalHeaderLabels(["#", "P(OK)", "Label", "Box"])
        else:
            self.table.setColumnCount(6)
            self.table.setHorizontalHeaderLabels(["#", "Final", "TM", "Px Conf", "Px Tmpl", "Px Tgt"])
        self.table.setRowCount(0)

    # ══════════════════════════════════════════════════════════ Load folder ══
    def _load_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select image folder")
        if not path:
            return

        self.image_files = sorted([
            os.path.join(path, f) for f in os.listdir(path)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ])
        self.current_idx = -1
        self.tmpl_data   = None
        self.tmpl_path   = None
        self.ok_paths    = set()
        self.tmpl_label.setText("Template: --")
        self.ok_count_label.setText("OK: 0")

        self.list_widget.clear()
        for fp in self.image_files:
            item = QListWidgetItem(load_thumb(fp), os.path.basename(fp))
            item.setData(Qt.UserRole, fp)
            item.setSizeHint(QSize(190, THUMB_SIZE + 24))
            self.list_widget.addItem(item)

        self.list_status.setText(f"{len(self.image_files)} images")
        self.status.setText(f"Loaded {len(self.image_files)} images from: {path}")

    # ══════════════════════════════════════════════════ Template / Target ══
    def _set_selected_as_template(self):
        item = self.list_widget.currentItem()
        if item is None:
            return
        self._set_template(item.data(Qt.UserRole))

    def _set_template(self, path: str):
        self.tmpl_path = path
        self.tmpl_data = None
        fname = os.path.basename(path)
        self.tmpl_label.setText(f"Template: {fname}")
        self.status.setText(f"Template set: {fname}")
        self._refresh_list_decorations()

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

        act_tmpl = QAction("Set as Template", self)
        act_run  = QAction("Run comparison", self)
        act_tmpl.triggered.connect(lambda: self._set_template(item.data(Qt.UserRole)))
        act_run.triggered.connect(lambda: self._on_item_clicked(item))
        menu.addAction(act_tmpl)
        menu.addAction(act_run)

        if self.ml_mode:
            act_ok = QAction("Mark as OK (for training)", self)
            act_ok.triggered.connect(lambda: self._mark_path_ok(item.data(Qt.UserRole)))
            menu.addAction(act_ok)

        menu.exec_(self.list_widget.mapToGlobal(pos))

    # ══════════════════════════════════════════════════════ ML: Mark OK ══
    def _mark_selected_ok(self):
        """Mark all currently selected images as OK for training."""
        selected = self.list_widget.selectedItems()
        if not selected:
            self.status.setText("Select images first.")
            return
        for item in selected:
            self.ok_paths.add(item.data(Qt.UserRole))
        self.ok_count_label.setText(f"OK: {len(self.ok_paths)}")
        self._refresh_list_decorations()
        self.status.setText(f"Marked {len(selected)} images as OK. Total OK: {len(self.ok_paths)}")

    def _mark_path_ok(self, path):
        self.ok_paths.add(path)
        self.ok_count_label.setText(f"OK: {len(self.ok_paths)}")
        self._refresh_list_decorations()

    def _refresh_list_decorations(self):
        """Update list item colors: green=OK, dark green=template."""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            path = item.data(Qt.UserRole)
            fname = os.path.basename(path)

            if path == self.tmpl_path:
                item.setBackground(QColor("#1a4a1a"))
                prefix = "[T] "
            elif path in self.ok_paths:
                item.setBackground(QColor("#0a3a0a"))
                prefix = "[OK] "
            else:
                item.setBackground(QColor(0, 0, 0, 0))
                prefix = ""

            item.setText(f"{prefix}{fname}")

    # ══════════════════════════════════════════════════════ ML: Train ══
    def _train_rf(self):
        if not self.ok_paths:
            QMessageBox.warning(self, "No OK images",
                                "Mark at least a few images as OK before training.")
            return
        if self.worker and self.worker.isRunning():
            return

        self._set_buttons_enabled(False)
        self.btn_train.setEnabled(False)
        self.status.setText(f"Training RF with {len(self.ok_paths)} OK images...")

        crops_dir = os.path.join(self.output_dir, "training_crops")
        os.makedirs(crops_dir, exist_ok=True)

        self.worker = TrainWorker(list(self.ok_paths), MODEL_PATH, crops_dir)
        self.worker.done.connect(self._on_train_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_train_done(self, msg, save_dir):
        self._set_buttons_enabled(True)
        self.btn_train.setEnabled(True)
        self.crops_dir = save_dir
        self.btn_view_crops.setEnabled(True)
        try:
            self.rf_model = load_model(MODEL_PATH)
            self.model_status.setText("Model: loaded")
            self.model_status.setStyleSheet("color:#0a0;font-weight:bold;")
        except Exception:
            pass
        self.status.setText(msg + "  |  Click 'View Crops' to inspect training data.")

    def _view_crops(self):
        if not self.crops_dir or not os.path.isdir(self.crops_dir):
            QMessageBox.information(self, "No crops", "Train the model first to generate crops.")
            return
        dlg = CropViewerDialog(self.crops_dir, self)
        dlg.exec_()

    # ══════════════════════════════════════════════════ ML: Load/Save ══
    def _load_model_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load RF Model", "",
                                              "Joblib (*.joblib);;All (*)")
        if not path:
            return
        try:
            self.rf_model = load_model(path)
            self.model_status.setText(f"Model: {os.path.basename(path)}")
            self.model_status.setStyleSheet("color:#0a0;font-weight:bold;")
            self.status.setText(f"Model loaded from: {path}")
        except Exception as e:
            self.status.setText(f"Error loading model: {e}")

    def _save_model_dialog(self):
        if self.rf_model is None:
            QMessageBox.warning(self, "No model", "Train or load a model first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save RF Model", "rf_model.joblib",
                                              "Joblib (*.joblib);;All (*)")
        if not path:
            return
        try:
            import joblib
            joblib.dump(self.rf_model, path)
            self.status.setText(f"Model saved to: {path}")
        except Exception as e:
            self.status.setText(f"Error saving model: {e}")

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
                f"{self.current_idx + 1} / {len(self.image_files)}  --  {fname}"
            )
        else:
            self.nav_label.setText("--")

    # ══════════════════════════════════════════════════════════ Run ══════════
    def _run_current(self):
        if self.current_idx < 0 or self.current_idx >= len(self.image_files):
            return
        if self.worker and self.worker.isRunning():
            return

        if self.ml_mode:
            self._run_ml_predict()
        else:
            self._run_template_match()

    def _run_template_match(self):
        """Original template matching mode."""
        if not self.tmpl_path:
            self.status.setText("Set a template first (right-click -> Set as Template).")
            return

        tgt_path = self.image_files[self.current_idx]
        if tgt_path == self.tmpl_path:
            self.status.setText("Target = template, skipping.")
            return

        if self.tmpl_data is None:
            self.status.setText("Segmenting template...")
            QApplication.processEvents()
            tmpl_dir = os.path.join(self.output_dir, "template")
            self.tmpl_data = segment_characters(self.tmpl_path, tmpl_dir, save=True)
            if self.tmpl_data[0] is None:
                self.status.setText("Error: cannot read template.")
                return

        self._set_buttons_enabled(False)
        tgt_path = self.image_files[self.current_idx]
        self.status.setText(f"Processing: {os.path.basename(tgt_path)} ...")

        self.worker = WorkerThread(
            self.tmpl_data, tgt_path,
            self.thr_spin.value(), self.output_dir
        )
        self.worker.done.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _run_ml_predict(self):
        """ML Random Forest prediction mode."""
        if self.rf_model is None:
            self.status.setText("No model loaded. Train or load a model first.")
            return

        tgt_path = self.image_files[self.current_idx]
        self._set_buttons_enabled(False)
        self.status.setText(f"RF Predicting: {os.path.basename(tgt_path)} ...")

        self.worker = PredictWorker(
            self.rf_model, tgt_path, self.thr_spin.value()
        )
        self.worker.done.connect(self._on_ml_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    # ══════════════════════════════════════════════════════ Result display ══
    def _on_done(self, comparison_img: np.ndarray, results: list):
        """Handle template matching results."""
        self._set_buttons_enabled(True)
        self.status.setText("Done.")

        if comparison_img is not None:
            pixmap = cv2_to_pixmap(comparison_img)
            avail_w = self.scroll.viewport().width() - 4
            if pixmap.width() < avail_w:
                pixmap = pixmap.scaledToWidth(avail_w, Qt.SmoothTransformation)
            self.img_label.setPixmap(pixmap)
            self.img_label.resize(pixmap.size())

        overall_pass = all(r[2] == "PASS" for r in results)
        if overall_pass:
            self.banner.setText("PASS")
            self.banner.setStyleSheet(
                "background:#1a7f1a;color:white;border-radius:4px;"
            )
        else:
            fail_n = sum(1 for r in results if r[2] == "FAIL")
            self.banner.setText(f"FAIL  ({fail_n} chars)")
            self.banner.setStyleSheet(
                "background:#b01a1a;color:white;border-radius:4px;"
            )

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

    def _on_ml_done(self, results: list, img: np.ndarray, _thresh: np.ndarray):
        """Handle ML prediction results."""
        self._set_buttons_enabled(True)
        self.status.setText("RF Prediction done.")

        # Draw result visualization on image
        vis = img.copy()
        padding = 2
        h_img, w_img = img.shape[:2]
        for r in results:
            x, y, w, h = r["box"]
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w_img, x + w + padding)
            y2 = min(h_img, y + h + padding)
            color = (0, 220, 0) if r["label"] == "PASS" else (0, 0, 255)
            cv.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label_text = f"{r['prob_ok']:.2f}"
            cv.putText(vis, label_text, (x1, y1 - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Build a strip: original on top, annotated below
        # Scale images to reasonable display size
        display_scale = min(3.0, 800 / max(1, w_img))
        dw = int(w_img * display_scale)
        dh = int(h_img * display_scale)
        orig_resized = cv.resize(img, (dw, dh), interpolation=cv.INTER_NEAREST)
        vis_resized  = cv.resize(vis, (dw, dh), interpolation=cv.INTER_NEAREST)

        gap = 8
        canvas = np.ones((dh * 2 + gap * 3, dw, 3), dtype=np.uint8) * 50
        canvas[gap:gap + dh, :] = orig_resized
        canvas[gap * 2 + dh:gap * 2 + dh * 2, :] = vis_resized

        cv.putText(canvas, "ORIGINAL", (4, gap + 14),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv.putText(canvas, "RF PREDICTION", (4, gap * 2 + dh + 14),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        pixmap = cv2_to_pixmap(canvas)
        avail_w = self.scroll.viewport().width() - 4
        if pixmap.width() < avail_w:
            pixmap = pixmap.scaledToWidth(avail_w, Qt.SmoothTransformation)
        self.img_label.setPixmap(pixmap)
        self.img_label.resize(pixmap.size())

        # Banner
        overall_pass = all(r["label"] == "PASS" for r in results)
        if overall_pass:
            self.banner.setText("PASS")
            self.banner.setStyleSheet(
                "background:#1a7f1a;color:white;border-radius:4px;"
            )
        else:
            fail_n = sum(1 for r in results if r["label"] == "FAIL")
            self.banner.setText(f"FAIL  ({fail_n} chars)")
            self.banner.setStyleSheet(
                "background:#b01a1a;color:white;border-radius:4px;"
            )

        # Table
        self.table.setRowCount(len(results))
        for row, r in enumerate(results):
            x, y, w, h = r["box"]
            vals = [
                str(r["char_idx"]),
                f"{r['prob_ok']:.4f}",
                r["label"],
                f"({x},{y},{w},{h})",
            ]
            color = QColor("#1aaa1a") if r["label"] == "PASS" else QColor("#dd3333")
            for col, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                item.setForeground(color)
                self.table.setItem(row, col, item)

    def _on_error(self, msg: str):
        self._set_buttons_enabled(True)
        self.btn_train.setEnabled(True)
        self.status.setText("Error! See console.")
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
