"""
NG sample tester — given one clean character (template + target), generate N
synthetic defective variants and verify whether the comparison pipeline flags
each one correctly. Lets the user tune both augmentation and comparison
parameters live to validate detection sensitivity.
"""
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QFormLayout, QLabel, QGroupBox,
    QSpinBox, QDoubleSpinBox, QPushButton, QScrollArea, QWidget, QGridLayout,
    QSplitter, QCheckBox,
)

from char_segmenter import (
    DEFAULT_PARAMS, _normalize_char_thresh, compute_char_quality,
    compute_diff_overlay,
)
from ng_augmenter import DEFAULT_AUG_PARAMS, NG_AUG_TYPES, generate_samples


# (key, label, kind, lo, hi, step)
_AUG_SPECS = [
    ('n_samples',           'Number of samples',         'int',   1,    96,    1),
    ('seed',                'Seed (0 = random)',         'int',   0,    9999,  1),
    ('noise_sigma',         'Noise σ',                   'int',   0,    120,   5),
    ('cut_count_min',       'Cut count min',             'int',   0,    8,     1),
    ('cut_count_max',       'Cut count max',             'int',   1,    10,    1),
    ('cut_size_frac_min',   'Cut size frac min',         'float', 0.05, 0.6,   0.02),
    ('cut_size_frac_max',   'Cut size frac max',         'float', 0.05, 0.8,   0.02),
    ('erode_k_min',         'Erode kernel min',          'int',   1,    20,    1),
    ('erode_k_max',         'Erode kernel max',          'int',   1,    25,    1),
    ('dilate_k_min',        'Dilate kernel min',         'int',   1,    20,    1),
    ('dilate_k_max',        'Dilate kernel max',         'int',   1,    25,    1),
    ('line_count_min',      'Line count min',            'int',   0,    6,     1),
    ('line_count_max',      'Line count max',            'int',   1,    8,     1),
    ('line_thick_min',      'Line thickness min',        'int',   1,    20,    1),
    ('line_thick_max_frac', 'Line thickness max frac',   'float', 0.02, 0.5,   0.02),
]


class NgTestDialog(QDialog):
    """
    Two clean inputs (template + target) → augment the TARGET to simulate defects,
    then compare each augmented variant against the original TEMPLATE using the
    same comparison pipeline as the main inference.
    """

    def __init__(self, tmpl_char_bgr, tgt_char_bgr, char_idx,
                 base_compare_params=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"NG sample tester — char #{char_idx}")
        self.resize(1300, 800)

        self.tmpl_char = tmpl_char_bgr
        self.tgt_char = tgt_char_bgr
        self.char_idx = char_idx
        self.compare_params = {**DEFAULT_PARAMS, **(base_compare_params or {})}
        self._aug_inputs = {}
        self._enable_inputs = {}

        # Pre-compute the threshold mask of the clean template once — it's the
        # reference everything is compared against.
        self.tmpl_thresh_clean = _normalize_char_thresh(
            cv2.cvtColor(self.tmpl_char, cv2.COLOR_BGR2GRAY)
            if len(self.tmpl_char.shape) == 3 else self.tmpl_char
        )

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(150)
        self._debounce.timeout.connect(self._refresh)

        self._build_ui()
        self._refresh()

    # --------------------------------------------------------------------- UI

    def _build_ui(self):
        outer = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter)

        # Left: aug params + enable checkboxes
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(8, 8, 8, 8)

        # Source preview (template + target)
        src_group = QGroupBox("Source")
        sg = QHBoxLayout(src_group)
        sg.addWidget(self._make_thumb("TEMPLATE", self.tmpl_char))
        sg.addWidget(self._make_thumb("TARGET (clean)", self.tgt_char))
        lv.addWidget(src_group)

        # Augmentation type toggles
        type_group = QGroupBox("Augmentation types")
        tg = QHBoxLayout(type_group)
        for t in NG_AUG_TYPES:
            cb = QCheckBox(t)
            cb.setChecked(t in DEFAULT_AUG_PARAMS['enabled'])
            cb.stateChanged.connect(self._on_changed)
            self._enable_inputs[t] = cb
            tg.addWidget(cb)
        tg.addStretch(1)
        lv.addWidget(type_group)

        # Param form
        param_group = QGroupBox("Augmentation parameters")
        form = QFormLayout(param_group)
        form.setLabelAlignment(Qt.AlignRight)
        for key, label, kind, lo, hi, step in _AUG_SPECS:
            if kind == 'int':
                w = QSpinBox()
                w.setRange(int(lo), int(hi))
                w.setSingleStep(int(step))
                w.setValue(int(DEFAULT_AUG_PARAMS[key]))
            else:
                w = QDoubleSpinBox()
                w.setRange(float(lo), float(hi))
                w.setSingleStep(float(step))
                w.setDecimals(3 if step < 0.01 else 2)
                w.setValue(float(DEFAULT_AUG_PARAMS[key]))
            w.valueChanged.connect(self._on_changed)
            self._aug_inputs[key] = w
            form.addRow(label, w)
        lv.addWidget(param_group)

        btn_row = QHBoxLayout()
        regen_btn = QPushButton("🎲 Regenerate (new random)")
        regen_btn.clicked.connect(self._regenerate)
        btn_row.addWidget(regen_btn)

        reset_btn = QPushButton("Reset defaults")
        reset_btn.clicked.connect(self._reset_defaults)
        btn_row.addWidget(reset_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        lv.addLayout(btn_row)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color: #ddd; padding: 4px;")
        lv.addWidget(self.summary)

        lv.addStretch(1)
        splitter.addWidget(left)

        # Right: scrollable grid of generated NG samples
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(8, 8, 8, 8)

        legend = QLabel(
            "Each cell shows: <b>augmented sample</b> (top) and "
            "<b>diff overlay vs template</b> (bottom). "
            "<span style='color:#ff5050'>Red=missing</span>, "
            "<span style='color:#50ff50'>Green=extra</span>, white=match. "
            "Border green=PASS (mistakenly), red=FAIL (correctly detected as NG)."
        )
        legend.setWordWrap(True)
        legend.setStyleSheet("color: #ccc;")
        rv.addWidget(legend)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("background-color: #1e1e1e; border: 1px solid #3e3e3e;")
        self.grid_container = QWidget()
        self.grid = QGridLayout(self.grid_container)
        self.grid.setContentsMargins(6, 6, 6, 6)
        self.grid.setSpacing(8)
        self.scroll.setWidget(self.grid_container)
        rv.addWidget(self.scroll, stretch=1)

        splitter.addWidget(right)
        splitter.setSizes([420, 880])

    def _make_thumb(self, title, img):
        box = QGroupBox(title)
        v = QVBoxLayout(box)
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("background-color: #1e1e1e;")
        pix = self._np_to_pixmap(img)
        if pix.height() > 120:
            pix = pix.scaledToHeight(120, Qt.SmoothTransformation)
        lbl.setPixmap(pix)
        v.addWidget(lbl)
        return box

    @staticmethod
    def _np_to_pixmap(img):
        if img is None or img.size == 0:
            return QPixmap()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w, ch = img.shape
        buf = np.ascontiguousarray(img)
        qimg = QImage(buf.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped().copy()
        return QPixmap.fromImage(qimg)

    # --------------------------------------------------------- behaviour

    def _on_changed(self, *_):
        self._debounce.start()

    def _regenerate(self):
        # Bumps a counter into the seed-effective space so we get fresh randoms
        seed_w = self._aug_inputs.get('seed')
        if seed_w is not None and seed_w.value() > 0:
            seed_w.setValue(seed_w.value() + 1)
        else:
            self._refresh()

    def _reset_defaults(self):
        for key, w in self._aug_inputs.items():
            w.blockSignals(True)
            w.setValue(DEFAULT_AUG_PARAMS[key])
            w.blockSignals(False)
        for t, cb in self._enable_inputs.items():
            cb.blockSignals(True)
            cb.setChecked(t in DEFAULT_AUG_PARAMS['enabled'])
            cb.blockSignals(False)
        self._refresh()

    def _collect_aug_params(self):
        out = dict(DEFAULT_AUG_PARAMS)
        for key, w in self._aug_inputs.items():
            out[key] = w.value()
        out['enabled'] = [t for t, cb in self._enable_inputs.items() if cb.isChecked()]
        return out

    # --------------------------------------------------------- compute

    def _refresh(self):
        # Clear grid
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        aug_params = self._collect_aug_params()
        if not aug_params['enabled']:
            self.summary.setText(
                "<span style='color:#fa0'>Enable at least one augmentation type.</span>"
            )
            return

        try:
            samples = generate_samples(self.tgt_char, params=aug_params)
        except Exception as e:
            self.summary.setText(f"<span style='color:#f55'>Augmenter error: {e}</span>")
            return

        cell_size = (96, 96)
        n_pass = 0
        n_total = len(samples)
        per_type = {}

        cols = 6
        for idx, sample in enumerate(samples):
            aug_bgr = sample['image']
            aug_type = sample['type']

            aug_gray = cv2.cvtColor(aug_bgr, cv2.COLOR_BGR2GRAY) if len(aug_bgr.shape) == 3 else aug_bgr
            aug_thresh = _normalize_char_thresh(aug_gray)

            try:
                metrics = compute_char_quality(
                    self.tmpl_thresh_clean, aug_thresh,
                    size=(int(self.compare_params['compare_size']),) * 2,
                    params=self.compare_params,
                )
                conf = float(metrics['confidence'])
                quality = 'PASS' if conf >= float(self.compare_params['pass_threshold']) else 'FAIL'
            except Exception as e:
                conf, quality = 0.0, 'FAIL'
                metrics = {'confidence': 0.0}
                print(f"NG compare error: {e}")

            try:
                overlay, diff_stats = compute_diff_overlay(
                    self.tmpl_thresh_clean, aug_thresh, size=cell_size,
                    max_shift=int(self.compare_params.get('align_max_shift', 8)),
                    scale_tol=float(self.compare_params.get('align_scale_tol', 0.15)),
                    scale_steps=int(self.compare_params.get('align_scale_steps', 5)),
                    keep_largest_cc=bool(self.compare_params.get('keep_largest_cc', 1)),
                    min_cc_area_ratio=float(self.compare_params.get('min_cc_area_ratio', 0.05)),
                    extent_target_fill=float(self.compare_params.get('extent_target_fill', 0.75)),
                    use_ecc_align=bool(self.compare_params.get('use_ecc_align', 1)),
                )
            except Exception:
                overlay = np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8)

            if quality == 'PASS':
                n_pass += 1
            per_type.setdefault(aug_type, [0, 0])
            per_type[aug_type][0] += 1 if quality == 'FAIL' else 0
            per_type[aug_type][1] += 1

            cell = self._make_sample_cell(aug_bgr, overlay, aug_type, conf, quality, cell_size)
            self.grid.addWidget(cell, idx // cols, idx % cols)

        # Summary: in the NG-tester, FAIL is the *desired* outcome (we injected a defect).
        n_caught = n_total - n_pass
        catch_rate = n_caught / n_total if n_total else 0.0
        per_type_str = " | ".join(
            f"{t}: {caught}/{tot}" for t, (caught, tot) in sorted(per_type.items())
        )
        color = "#0c0" if catch_rate >= 0.9 else ("#fa0" if catch_rate >= 0.5 else "#f55")
        self.summary.setText(
            f"<b>Detection rate:</b> "
            f"<span style='color:{color}; font-weight:bold;'>{n_caught}/{n_total} "
            f"({catch_rate * 100:.0f}%)</span> NG samples correctly flagged. "
            f"<br/><span style='color:#aaa;'>Per type (caught/total): {per_type_str}</span>"
        )

    def _make_sample_cell(self, aug_bgr, overlay_bgr, aug_type, conf, quality, cell_size):
        is_pass = quality == 'PASS'  # BAD outcome here — augmented sample slipped through
        # Border: red if PASS (defect not detected), green if FAIL (detected correctly)
        border = "#f55" if is_pass else "#0c0"
        wrap = QWidget()
        wrap.setStyleSheet(
            f"QWidget {{ background-color: #1a1a1a; border: 2px solid {border}; "
            "border-radius: 4px; padding: 4px; }}"
        )
        v = QVBoxLayout(wrap)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(2)

        title = QLabel(f"{aug_type}")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #fff; font-weight: bold; border: none;")
        v.addWidget(title)

        h = QHBoxLayout()
        h.setSpacing(2)
        h.addWidget(self._make_image_label(aug_bgr, cell_size))
        h.addWidget(self._make_image_label(overlay_bgr, cell_size))
        v.addLayout(h)

        verdict = "✓ caught" if not is_pass else "✗ MISSED"
        score = QLabel(f"conf {conf:.3f}  ({verdict})")
        score.setAlignment(Qt.AlignCenter)
        score.setStyleSheet(f"color: {border}; border: none;")
        v.addWidget(score)
        return wrap

    def _make_image_label(self, img, size):
        lbl = QLabel()
        lbl.setStyleSheet("border: none;")
        if img is not None and img.size > 0:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            disp = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
            lbl.setPixmap(self._np_to_pixmap(disp))
        return lbl
