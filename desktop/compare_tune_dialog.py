"""
Dialog for live-tuning char_segmenter parameters on a single (template, target)
pair, with the comparison strip rebuilt on every change.
"""
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QScrollArea,
    QSpinBox, QDoubleSpinBox, QPushButton, QWidget, QGroupBox, QSplitter,
    QProgressBar,
)

from char_segmenter import compare_arrays_full, DEFAULT_PARAMS
from auto_tune import auto_tune


class AutoTuneWorker(QThread):
    """Runs auto-tune in a background thread so the UI stays responsive."""
    progress = pyqtSignal(int, int, float)   # done, total, best_score_so_far
    done = pyqtSignal(dict, dict)            # best_params, best_metrics ({} if none)
    failed = pyqtSignal(str)

    def __init__(self, tmpl_img, tgt_img, base_params, n_trials, n_ng_samples):
        super().__init__()
        self.tmpl_img = tmpl_img
        self.tgt_img = tgt_img
        self.base_params = base_params
        self.n_trials = n_trials
        self.n_ng_samples = n_ng_samples
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            best_params, best_metrics, _hist = auto_tune(
                self.tmpl_img, self.tgt_img,
                base_params=self.base_params,
                n_trials=self.n_trials,
                n_ng_samples=self.n_ng_samples,
                progress_cb=lambda i, n, s: self.progress.emit(i, n, s),
                cancel_cb=lambda: self._cancelled,
            )
            self.done.emit(best_params, best_metrics or {})
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.failed.emit(str(e))


# (key, label, widget_kind, min, max, step) — order is the form layout order
_PARAM_SPECS = [
    ('min_proc_h',          'Upscale height (px)',           'int',    20,  400,  10),
    ('clahe_clip',          'CLAHE clipLimit',               'float',  0.5, 10.0, 0.5),
    ('clahe_grid',          'CLAHE grid size (NxN)',         'int',    2,   32,   1),
    ('blur_kernel',         'Pre-thresh blur kernel',        'int',    1,   15,   2),
    ('close_kernel_factor', 'Morph close (% h_proc)',        'float',  0.0, 0.20, 0.005),
    ('min_char_h_factor',   'Min char height (% h_proc)',    'float',  0.05, 0.9, 0.05),
    ('min_char_w',          'Min char width (px)',           'int',    1,   30,   1),
    ('padding',             'Char crop padding (px)',        'int',    0,   20,   1),
    ('compare_size',        'Comparison cell (px)',          'int',    16,  256,  8),
    ('tm_blur_sigma',       'Template-match blur σ',         'float',  0.1, 4.0,  0.1),
    ('iou_dilate',          'IoU dilation kernel',           'int',    1,   15,   2),
    ('pixel_dev_tol',       'Pixel ratio tolerance',         'float',  0.2, 5.0,  0.1),
    ('align_max_shift',     'Align max shift (px)',          'int',    0,   30,   1),
    ('align_scale_tol',     'Align scale tol (±)',           'float',  0.0, 0.5,  0.02),
    ('align_scale_steps',   'Align scale steps',             'int',    1,   11,   2),
    ('keep_largest_cc',     'Drop noise (keep largest CC)',  'int',    0,   1,    1),
    ('min_cc_area_ratio',   '└ min area ratio (when off)',   'float',  0.0, 0.5,  0.02),
    ('extent_target_fill',  'Foreground fill ratio',         'float',  0.3, 0.95, 0.05),
    ('use_ecc_align',       'ECC sub-pixel align',           'int',    0,   1,    1),
    ('char_local_search_radius', 'Char local search ±',      'float',  0.0, 0.5,  0.05),
    ('char_refine_min_score',    '└ refine min score',       'float',  0.0, 1.0,  0.05),
    ('pass_threshold',      'PASS threshold (per char)',     'float',  0.0, 1.0,  0.01),
]


class CompareTuneDialog(QDialog):
    def __init__(self, tmpl_img, tgt_img, region_type, region_idx, parent=None,
                 annotations_json_path=None):
        super().__init__(parent)
        self.setWindowTitle(f"Tune Comparison — [{region_type.upper()}] region #{region_idx}")
        self.resize(1200, 800)

        self.tmpl_img = tmpl_img
        self.tgt_img = tgt_img
        self.annotations_json_path = annotations_json_path
        self._inputs = {}
        self.params = dict(DEFAULT_PARAMS)

        # If a trained config exists for this template, use it as the
        # starting point so users can iterate from the trained baseline.
        self._trained_params = None
        if annotations_json_path:
            try:
                from params_store import load_trained_params
                rec = load_trained_params(annotations_json_path)
                if rec and 'params' in rec:
                    self._trained_params = dict(rec['params'])
            except Exception as e:
                print(f"[tune] could not load trained params: {e}")

        # Debounce so rapid spinbox changes don't trigger compare on each tick
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._debounce.timeout.connect(self._refresh)

        self._build_ui()
        self._refresh()  # initial render

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        outer = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter)

        # --- Left: parameter form ---
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)

        group = QGroupBox("Parameters")
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignRight)

        # Initial values: prefer trained-params if loaded, else factory defaults
        initial = dict(DEFAULT_PARAMS)
        if self._trained_params:
            initial.update(self._trained_params)

        for key, label, kind, lo, hi, step in _PARAM_SPECS:
            if kind == 'int':
                w = QSpinBox()
                w.setRange(int(lo), int(hi))
                w.setSingleStep(int(step))
                w.setValue(int(initial[key]))
                w.valueChanged.connect(self._on_changed)
            else:
                w = QDoubleSpinBox()
                w.setRange(float(lo), float(hi))
                w.setSingleStep(float(step))
                w.setDecimals(3 if step < 0.01 else 2)
                w.setValue(float(initial[key]))
                w.valueChanged.connect(self._on_changed)
            self._inputs[key] = w
            form.addRow(label, w)

        left_layout.addWidget(group)

        # Show whether we started from trained or factory defaults
        if self._trained_params:
            init_note = QLabel(
                "<i style='color:#0c0;'>● Initialised from trained params</i>"
            )
        else:
            init_note = QLabel(
                "<i style='color:#888;'>(no trained params for this template — using factory defaults)</i>"
            )
        init_note.setWordWrap(True)
        left_layout.addWidget(init_note)

        # Reset row: factory + trained options
        reset_row = QHBoxLayout()
        reset_factory = QPushButton("Reset to factory")
        reset_factory.clicked.connect(self._reset_factory)
        reset_row.addWidget(reset_factory)
        reset_trained = QPushButton("Reset to trained")
        reset_trained.setEnabled(self._trained_params is not None)
        reset_trained.clicked.connect(self._reset_trained)
        reset_row.addWidget(reset_trained)
        left_layout.addLayout(reset_row)

        # Save / close row
        btn_row = QHBoxLayout()
        save_btn = QPushButton("💾 Save as trained")
        save_btn.setEnabled(self.annotations_json_path is not None)
        save_btn.clicked.connect(self._save_as_trained)
        btn_row.addWidget(save_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        left_layout.addLayout(btn_row)

        # Auto-tune controls
        auto_group = QGroupBox("Auto-tune")
        auto_layout = QVBoxLayout(auto_group)
        auto_layout.setSpacing(4)

        trial_row = QHBoxLayout()
        trial_row.addWidget(QLabel("Trials:"))
        self.auto_trials_input = QSpinBox()
        self.auto_trials_input.setRange(5, 200)
        self.auto_trials_input.setValue(40)
        trial_row.addWidget(self.auto_trials_input)
        trial_row.addWidget(QLabel("NG samples:"))
        self.auto_ng_input = QSpinBox()
        self.auto_ng_input.setRange(4, 48)
        self.auto_ng_input.setValue(12)
        trial_row.addWidget(self.auto_ng_input)
        trial_row.addStretch(1)
        auto_layout.addLayout(trial_row)

        run_row = QHBoxLayout()
        self.auto_btn = QPushButton("🤖 Run auto-tune")
        self.auto_btn.clicked.connect(self._start_auto_tune)
        run_row.addWidget(self.auto_btn)
        self.auto_cancel_btn = QPushButton("Cancel")
        self.auto_cancel_btn.setEnabled(False)
        self.auto_cancel_btn.clicked.connect(self._cancel_auto_tune)
        run_row.addWidget(self.auto_cancel_btn)
        auto_layout.addLayout(run_row)

        self.auto_progress = QProgressBar()
        self.auto_progress.setVisible(False)
        auto_layout.addWidget(self.auto_progress)

        self.auto_status = QLabel("")
        self.auto_status.setWordWrap(True)
        self.auto_status.setStyleSheet("color: #aaa;")
        auto_layout.addWidget(self.auto_status)
        left_layout.addWidget(auto_group)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("color: #ccc; padding: 4px;")
        left_layout.addWidget(self.summary_label)

        left_layout.addStretch(1)
        splitter.addWidget(left)

        # --- Right: source previews + live comparison strip ---
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(6)

        src_row = QHBoxLayout()
        src_row.addWidget(self._make_source_label("TEMPLATE", self.tmpl_img))
        src_row.addWidget(self._make_source_label("TARGET", self.tgt_img))
        right_layout.addLayout(src_row)

        right_layout.addWidget(QLabel("Comparison (live):"))
        self.strip_scroll = QScrollArea()
        self.strip_scroll.setWidgetResizable(True)
        self.strip_scroll.setStyleSheet("background-color: #1e1e1e; border: 1px solid #3e3e3e;")
        self.strip_label = QLabel()
        self.strip_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.strip_scroll.setWidget(self.strip_label)
        right_layout.addWidget(self.strip_scroll, stretch=1)

        # Row of per-char "🧪 Test" buttons (populated after each refresh).
        right_layout.addWidget(QLabel("Per-char NG sample test:"))
        self.char_btn_scroll = QScrollArea()
        self.char_btn_scroll.setWidgetResizable(True)
        self.char_btn_scroll.setFixedHeight(64)
        self.char_btn_container = QWidget()
        self.char_btn_layout = QHBoxLayout(self.char_btn_container)
        self.char_btn_layout.setContentsMargins(4, 4, 4, 4)
        self.char_btn_layout.setSpacing(4)
        self.char_btn_scroll.setWidget(self.char_btn_container)
        right_layout.addWidget(self.char_btn_scroll)

        # State for NG-test dialogs spawned from this dialog
        self._open_ng_dialogs = []
        self._last_tmpl_chars = []
        self._last_tgt_chars = []

        splitter.addWidget(right)
        splitter.setSizes([350, 850])

    def _make_source_label(self, title, img):
        box = QGroupBox(title)
        v = QVBoxLayout(box)
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("background-color: #1e1e1e;")
        pix = self._np_to_pixmap(img)
        # Cap height so very tall crops don't dominate the dialog
        if pix.height() > 140:
            pix = pix.scaledToHeight(140, Qt.SmoothTransformation)
        lbl.setPixmap(pix)
        v.addWidget(lbl)
        return box

    @staticmethod
    def _np_to_pixmap(img_bgr):
        if img_bgr is None or img_bgr.size == 0:
            return QPixmap()
        if len(img_bgr.shape) == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        h, w, ch = img_bgr.shape
        # Force a contiguous copy so QImage doesn't read freed memory
        buf = np.ascontiguousarray(img_bgr)
        qimg = QImage(buf.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped().copy()
        return QPixmap.fromImage(qimg)

    # ------------------------------------------------------------ Behavior

    def _on_changed(self, *_):
        self._debounce.start()

    def _reset_factory(self):
        for key, widget in self._inputs.items():
            widget.blockSignals(True)
            widget.setValue(DEFAULT_PARAMS[key])
            widget.blockSignals(False)
        self._refresh()

    def _reset_trained(self):
        if not self._trained_params:
            return
        for key, widget in self._inputs.items():
            widget.blockSignals(True)
            widget.setValue(self._trained_params.get(key, DEFAULT_PARAMS[key]))
            widget.blockSignals(False)
        self._refresh()

    def _save_as_trained(self):
        if not self.annotations_json_path:
            return
        try:
            from params_store import save_trained_params
            from PyQt5.QtWidgets import QMessageBox
            params = self._collect_params()
            path = save_trained_params(
                self.annotations_json_path, params,
                metrics={'source': 'manual_tune'},
                trained_on=[],
            )
            self._trained_params = dict(params)
            QMessageBox.information(
                self, "Saved",
                f"Saved current params as trained config to:\n{path}\n\n"
                "Subsequent inferences will use these params."
            )
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Save failed", str(e))

    def _collect_params(self):
        out = dict(DEFAULT_PARAMS)
        for key, widget in self._inputs.items():
            out[key] = widget.value()
        return out

    def _refresh(self):
        params = self._collect_params()
        try:
            out = compare_arrays_full(self.tmpl_img, self.tgt_img, params=params)
        except Exception as e:
            self.summary_label.setText(f"<span style='color:#f55'>Error: {e}</span>")
            self.strip_label.clear()
            self._populate_char_buttons([], [])
            return

        strip = out['strip']
        results = out['results']
        overall_pass = out['overall_pass']

        if strip is None:
            self.summary_label.setText(
                "<span style='color:#fa0'>No characters segmented "
                "(try lower min char height / larger upscale).</span>"
            )
            self.strip_label.clear()
            self._populate_char_buttons([], [])
            return

        self.strip_label.setPixmap(self._np_to_pixmap(strip))

        # Cache per-char crops so the NG buttons can pass them straight to NgTestDialog
        self._last_tmpl_chars = out['tmpl_chars']
        self._last_tgt_chars = out['tgt_chars']
        self._populate_char_buttons(out['tmpl_chars'], out['tgt_chars'], results)

        n_pass = sum(1 for r in results if r[2] == 'PASS')
        n_total = len(results)
        avg_conf = float(np.mean([r[1]['confidence'] for r in results])) if results else 0.0
        color = "#0c0" if overall_pass else "#f55"
        self.summary_label.setText(
            f"<span style='color:{color}; font-weight:bold;'>"
            f"{'PASS' if overall_pass else 'FAIL'}</span>"
            f" — {n_pass}/{n_total} chars passed, avg confidence {avg_conf:.3f}"
        )

    def _populate_char_buttons(self, tmpl_chars, tgt_chars, results=None):
        # Clear existing buttons
        while self.char_btn_layout.count():
            item = self.char_btn_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        n = min(len(tmpl_chars), len(tgt_chars))
        if n == 0:
            self.char_btn_layout.addStretch(1)
            return

        for i in range(n):
            quality = results[i][2] if results and i < len(results) else None
            color = "#0c0" if quality == 'PASS' else ("#f55" if quality == 'FAIL' else "#888")
            btn = QPushButton(f"🧪 #{i}")
            btn.setStyleSheet(f"color: {color}; padding: 4px 8px;")
            btn.setToolTip(f"Open NG sample tester for char #{i}")
            btn.clicked.connect(
                lambda _checked, idx=i: self._open_ng_dialog(idx)
            )
            self.char_btn_layout.addWidget(btn)
        self.char_btn_layout.addStretch(1)

    # ------------------------------------------------------------ Auto-tune

    def _start_auto_tune(self):
        if getattr(self, '_tuner', None) is not None and self._tuner.isRunning():
            return
        n_trials = int(self.auto_trials_input.value())
        n_ng = int(self.auto_ng_input.value())
        base = self._collect_params()

        self._tuner = AutoTuneWorker(
            self.tmpl_img, self.tgt_img, base, n_trials, n_ng
        )
        self._tuner.progress.connect(self._on_tune_progress)
        self._tuner.done.connect(self._on_tune_done)
        self._tuner.failed.connect(self._on_tune_failed)
        self._tuner.start()

        self.auto_btn.setEnabled(False)
        self.auto_cancel_btn.setEnabled(True)
        self.auto_progress.setVisible(True)
        self.auto_progress.setRange(0, n_trials)
        self.auto_progress.setValue(0)
        self.auto_status.setText("Running…")

    def _cancel_auto_tune(self):
        if getattr(self, '_tuner', None) is not None:
            self._tuner.cancel()
            self.auto_status.setText("Cancelling…")

    def _on_tune_progress(self, done, total, best_score):
        self.auto_progress.setMaximum(total)
        self.auto_progress.setValue(done)
        self.auto_status.setText(f"Trial {done}/{total} — best score: {best_score:.3f}")

    def _on_tune_done(self, best_params, best_metrics):
        self.auto_btn.setEnabled(True)
        self.auto_cancel_btn.setEnabled(False)
        self.auto_progress.setVisible(False)

        if not best_metrics:
            self.auto_status.setText(
                "<span style='color:#fa0'>No working config found. Try more trials "
                "or a different image.</span>"
            )
            return

        # Apply best params to the spinboxes (will trigger _refresh via debounce)
        for key, widget in self._inputs.items():
            if key in best_params:
                widget.blockSignals(True)
                widget.setValue(best_params[key])
                widget.blockSignals(False)
        self._refresh()

        self.auto_status.setText(
            "<b style='color:#0c0'>Done</b> — applied best config. "
            f"score={best_metrics['score']:.3f}, "
            f"clean PASS rate={best_metrics['clean_pass_rate'] * 100:.0f}%, "
            f"NG catch rate={best_metrics['ng_catch_rate'] * 100:.0f}%, "
            f"clean conf={best_metrics['clean_avg_conf']:.3f}, "
            f"separation margin={best_metrics['margin']:+.3f}"
        )

    def _on_tune_failed(self, msg):
        self.auto_btn.setEnabled(True)
        self.auto_cancel_btn.setEnabled(False)
        self.auto_progress.setVisible(False)
        self.auto_status.setText(f"<span style='color:#f55'>Auto-tune failed: {msg}</span>")

    def _open_ng_dialog(self, char_idx):
        if char_idx >= len(self._last_tmpl_chars) or char_idx >= len(self._last_tgt_chars):
            return
        try:
            from ng_test_dialog import NgTestDialog
        except ImportError as e:
            print(f"Cannot open NG test dialog: {e}")
            return
        dlg = NgTestDialog(
            tmpl_char_bgr=self._last_tmpl_chars[char_idx],
            tgt_char_bgr=self._last_tgt_chars[char_idx],
            char_idx=char_idx,
            base_compare_params=self._collect_params(),
            parent=self,
        )
        dlg.setModal(False)
        dlg.show()
        self._open_ng_dialogs.append(dlg)
        dlg.finished.connect(
            lambda _: self._open_ng_dialogs.remove(dlg)
            if dlg in self._open_ng_dialogs else None
        )
