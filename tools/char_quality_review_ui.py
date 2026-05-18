"""Char-quality review UI.

Workflow:
  1. Import a root folder containing emb_*/ subfolders (e.g. data/test_result).
  2. Click a subfolder → all template/target pairs load as cards.
  3. Toggle method (OLD vs v3), optional fragment removal.
  4. For each card, click "NG" or "False-OK" to label.
  5. Labels and image copies saved to chosen output dir.

Run:
  python tools/char_quality_review_ui.py
"""

import sys, os, json, glob, re, shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / 'tools'))

from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPalette
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QFileDialog, QScrollArea,
    QFrame, QComboBox, QCheckBox, QGroupBox, QSplitter, QMessageBox, QLineEdit,
    QSizePolicy, QDialog, QDoubleSpinBox, QSpinBox,
)

from ai_services.camera_management.verification.embedding_classifier import (
    _compute_char_quality,
)
from ai_services.camera_management.verification.char_preprocess import (
    remove_fragments_local_bg,
)
from char_quality_v3 import compute_char_quality_v3
from char_quality_v4 import compute_char_quality_v4
from char_quality_v5 import compute_char_quality_v5
from char_quality_v6_saml import compute_saml_frame, render_h_histogram, N_SIGMA_DEFAULT
from char_quality_v7_shape import compute_char_quality_v7, render_orientation_overlay


PAIR_RE = re.compile(r"^char(\d+)_(OK|NG)_p([\d.]+)_target\.png$")
CARD_W, CARD_H = 320, 280
THUMB_SIZE = 96
DEFAULT_THRESHOLD = 0.80

LABEL_NONE = None
LABEL_CORRECT = 'correct'   # algo prediction matches ground truth
LABEL_WRONG   = 'wrong'     # algo prediction is wrong → ground truth = opposite

COLOR_BORDER = {
    LABEL_NONE:    '#3a3a3a',
    LABEL_CORRECT: '#27ae60',
    LABEL_WRONG:   '#e74c3c',
}
COLOR_VERDICT = {'OK': '#27ae60', 'NG': '#e74c3c'}

# Confusion matrix cells
# verdict=predicted, actual=ground truth → outcome
def outcome_of(verdict: str, actual: str) -> str:
    if verdict == 'NG' and actual == 'NG': return 'TP'   # correctly caught
    if verdict == 'OK' and actual == 'OK': return 'TN'   # correctly passed
    if verdict == 'NG' and actual == 'OK': return 'FP'   # false alarm (over-reject)
    if verdict == 'OK' and actual == 'NG': return 'FN'   # escape (missed defect)
    return 'NA'


def cv_to_qpixmap(img: np.ndarray, target: int = THUMB_SIZE) -> QPixmap:
    """uint8 BGR/gray → QPixmap, resized to target×target square."""
    if img is None or img.size == 0:
        return QPixmap(target, target)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    scale = target / max(h, w)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((target, target, 3), 30, dtype=np.uint8)
    yo, xo = (target - nh) // 2, (target - nw) // 2
    canvas[yo:yo+nh, xo:xo+nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, target, target, 3 * target, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class PairCard(QFrame):
    """One pair card: thumbs + metrics + action buttons."""

    def __init__(self, pair_info: dict, parent_window):
        super().__init__()
        self.pair = pair_info
        self.parent_window = parent_window
        self.label = LABEL_NONE
        self.metrics = None
        self.verdict = '?'  # 'OK' | 'NG' — algo prediction at current threshold
        self.setFixedSize(CARD_W, CARD_H)
        self.setFrameShape(QFrame.StyledPanel)
        self._build()
        self._apply_border()

    def _build(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(4)

        # Row 1: thumbs
        row = QHBoxLayout()
        row.setSpacing(4)
        self.lbl_tmpl = QLabel(); self.lbl_tmpl.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        self.lbl_tgt  = QLabel(); self.lbl_tgt.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        self.lbl_diff = QLabel(); self.lbl_diff.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        for w in (self.lbl_tmpl, self.lbl_tgt, self.lbl_diff):
            w.setStyleSheet("background:#222;border:1px solid #555;")
        row.addWidget(self.lbl_tmpl)
        row.addWidget(self.lbl_tgt)
        row.addWidget(self.lbl_diff)
        v.addLayout(row)

        # Row 2: header line (char id + logged)
        self.lbl_head = QLabel()
        self.lbl_head.setFont(QFont('Menlo', 10, QFont.Bold))
        v.addWidget(self.lbl_head)

        # Row 3: predicted verdict — BIG colored badge
        self.lbl_verdict = QLabel()
        self.lbl_verdict.setFont(QFont('Menlo', 13, QFont.Bold))
        self.lbl_verdict.setAlignment(Qt.AlignCenter)
        self.lbl_verdict.setFixedHeight(28)
        v.addWidget(self.lbl_verdict)

        # Row 4: metric text
        self.lbl_metrics = QLabel()
        self.lbl_metrics.setFont(QFont('Menlo', 9))
        self.lbl_metrics.setStyleSheet("color:#ddd;")
        v.addWidget(self.lbl_metrics)

        # Row 5: action buttons — verify whether the verdict is correct
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self.btn_correct = QPushButton("✓ Đúng")
        self.btn_correct.setStyleSheet("background:#27ae60;color:white;padding:6px;font-weight:bold;")
        self.btn_correct.clicked.connect(lambda: self.mark(LABEL_CORRECT))

        self.btn_wrong = QPushButton("✗ Sai")
        self.btn_wrong.setStyleSheet("background:#c0392b;color:white;padding:6px;font-weight:bold;")
        self.btn_wrong.clicked.connect(lambda: self.mark(LABEL_WRONG))

        self.btn_clear = QPushButton("✕")
        self.btn_clear.setFixedWidth(28)
        self.btn_clear.setStyleSheet("background:#555;color:white;padding:6px;")
        self.btn_clear.clicked.connect(lambda: self.mark(LABEL_NONE))

        self.btn_detail = QPushButton("🔍")
        self.btn_detail.setFixedWidth(28)
        self.btn_detail.setStyleSheet("background:#2980b9;color:white;padding:6px;")
        self.btn_detail.clicked.connect(self.show_detail)

        btn_row.addWidget(self.btn_correct)
        btn_row.addWidget(self.btn_wrong)
        btn_row.addWidget(self.btn_clear)
        btn_row.addWidget(self.btn_detail)
        v.addLayout(btn_row)

    def _apply_border(self):
        color = COLOR_BORDER[self.label]
        self.setStyleSheet(f"PairCard {{ border:3px solid {color}; border-radius:4px; background:#1e1e1e; }}")

    def refresh(self):
        """Re-load images and recompute metrics with current method."""
        tmpl = cv2.imread(self.pair['tmpl_path'], cv2.IMREAD_GRAYSCALE)
        tgt  = cv2.imread(self.pair['tgt_path'],  cv2.IMREAD_GRAYSCALE)
        if tmpl is None or tgt is None:
            self.lbl_head.setText("(missing)")
            return

        # Optional fragment removal on target
        tgt_for_metric = tgt
        if self.parent_window.apply_frag_remove:
            try:
                tgt_proc, _ = remove_fragments_local_bg(tgt)
                if tgt_proc.ndim == 3:
                    tgt_proc = cv2.cvtColor(tgt_proc, cv2.COLOR_BGR2GRAY)
                tgt_for_metric = tgt_proc
            except Exception:
                pass

        method = self.parent_window.method
        threshold = self.parent_window.threshold
        try:
            if method == 'OLD':
                m = _compute_char_quality(tmpl, tgt_for_metric, denoise=False)
                conf = m['confidence']
                self.metrics = {
                    'method': 'OLD', 'conf': conf,
                    'blur_tm': m['blur_tm'], 'iou': m['iou'], 'pixel_conf': m['pixel_conf'],
                }
                metric_txt = (f"OLD conf={conf:.3f}\n"
                              f"tm={m['blur_tm']:.2f}  iou={m['iou']:.2f}  px={m['pixel_conf']:.2f}")
                diff_img = cv2.bitwise_xor(m['_mask_tmpl_aligned'], m['_mask_tgt_aligned'])
            elif method == 'v7':
                m = compute_char_quality_v7(tmpl, tgt_for_metric,
                                            pad_y=self.parent_window.pad_y,
                                            pad_x=self.parent_window.pad_x,
                                            clean_fragments=self.parent_window.clean_fragments)
                conf = m['confidence']
                self.metrics = {
                    'method': 'v7', 'conf': conf,
                    'orientation_match_pct': m['orientation_match_pct'],
                    'partial_match_pct':     m['partial_match_pct'],
                    'n_strong_pixels':       m['n_strong_pixels'],
                    'coverage_ratio':        m['coverage_ratio'],
                    'ecc_cc': m['ecc_cc'],
                    'motion': m['motion'],
                    'defect_type': m['defect_type'],
                }
                dt = m['defect_type'] or '-'
                metric_txt = (
                    f"v7/shape  conf={conf:.3f}  defect={dt}  {m['motion']}\n"
                    f"match={100*m['orientation_match_pct']:.0f}%  partial={100*m['partial_match_pct']:.0f}%\n"
                    f"strong px: t={m['n_strong_template']}  g={m['n_strong_target']}  "
                    f"ratio={m['coverage_ratio']:.2f}"
                )
                diff_img = render_orientation_overlay(m['_t_prep'], m['_quant_t'], m['_quant_g'])
            elif method == 'v6':
                # Lookup pre-computed SAML result for this (folder, char_idx)
                key = (self.pair['folder'], self.pair['char_idx'])
                per = self.parent_window._saml_per_char.get(key)
                frame_res = self.parent_window._saml_cache.get(self.pair['folder'])
                if per is None or frame_res is None:
                    self.lbl_head.setText("(SAML pending — click folder)")
                    self.verdict = '?'
                    return
                conf = per['confidence']
                self.metrics = {
                    'method': 'v6',
                    'conf': conf,
                    'h_target': per['h_target'],
                    'h_template': per['h_template'],
                    'dev_self': per['dev_self'],
                    'dev_baseline': per['dev_baseline'],
                    'outlier_self': per['outlier_self'],
                    'outlier_baseline': per['outlier_baseline'],
                    'is_outlier': per['is_outlier'],
                    'frame_bad': per.get('frame_bad', False),
                    'frame_mean': frame_res['frame_mean'],
                    'frame_std':  frame_res['frame_std'],
                    'baseline_mean': frame_res['baseline_mean'],
                    'baseline_std':  frame_res['baseline_std'],
                    'frame_dev_vs_baseline': frame_res['frame_dev_vs_baseline'],
                    'defect_type': per['defect_type'],
                }
                dt = per['defect_type'] or '-'
                fb = " FRAME-BAD" if per.get('frame_bad') else ""
                dev_b = f"{per['dev_baseline']:+.1f}σ" if per['dev_baseline'] is not None else "n/a"
                metric_txt = (
                    f"v6/SAML  conf={conf:.3f}  defect={dt}{fb}\n"
                    f"H={per['h_target']:.0f}  dev_self={per['dev_self']:+.1f}σ  "
                    f"dev_base={dev_b}\n"
                    f"frame: mu={frame_res['frame_mean']:.0f} sig={frame_res['frame_std']:.2f}  "
                    f"vs_base={frame_res['frame_dev_vs_baseline']:+.1f}σ"
                )
                # Diff thumbnail: histogram with marker for this char
                h_vals = np.array([p['h_target'] for p in frame_res['per_char']])
                diff_img = render_h_histogram(
                    h_vals, frame_res['frame_mean'], frame_res['frame_std'],
                    n_sigma=self.parent_window.n_sigma,
                    current_h=per['h_target'], size=THUMB_SIZE,
                )
            elif method == 'v5':
                m = compute_char_quality_v5(tmpl, tgt_for_metric,
                                            pad_y=self.parent_window.pad_y,
                                            pad_x=self.parent_window.pad_x,
                                            clean_fragments=self.parent_window.clean_fragments)
                conf = m['confidence']
                self.metrics = {
                    'method': 'v5', 'conf': conf, 'ncc': m['ncc'], 'motion': m['motion'],
                    'over_g': m['over_ink_score'], 'under_g': m['under_ink_score'],
                    'over_max_tile':  m['tile_over_max'],
                    'under_max_tile': m['tile_under_max'],
                    'n_bad_tiles': m['n_bad_tiles'],
                    'defect_loc': m['defect_location'],
                    'sw_ratio': m['sw_ratio'],
                    'defect_type': m['defect_type'],
                }
                dt = m['defect_type'] or '-'
                loc = m['defect_location']
                loc_str = f"({loc[0]},{loc[1]})" if loc else '-'
                metric_txt = (
                    f"v5  conf={conf:.3f}  defect={dt} @{loc_str}  {m['motion']}\n"
                    f"GLB  over={m['over_ink_score']:.2f}  under={m['under_ink_score']:.2f}\n"
                    f"TILE max_over={m['tile_over_max']:.2f}  max_under={m['tile_under_max']:.2f}  bad={m['n_bad_tiles']}"
                )
                # Use heatmap as diff thumbnail
                diff_img = m['_tile_heatmap']
            elif method == 'v4':
                m = compute_char_quality_v4(tmpl, tgt_for_metric,
                                            pad_y=self.parent_window.pad_y,
                                            pad_x=self.parent_window.pad_x,
                                            clean_fragments=self.parent_window.clean_fragments)
                conf = m['confidence']
                self.metrics = {
                    'method': 'v4', 'conf': conf, 'ncc': m['ncc'], 'ecc_cc': m['ecc_cc'],
                    'motion': m['motion'], 'scale_x': m['scale_x'], 'scale_y': m['scale_y'],
                    'sw_ratio': m['sw_ratio'],
                    'over_ink': m['over_ink_score'], 'under_ink': m['under_ink_score'],
                    'defect_type': m['defect_type'],
                }
                dt = m['defect_type'] or '-'
                metric_txt = (
                    f"v4  conf={conf:.3f}  defect={dt}  {m['motion']}\n"
                    f"ncc={m['ncc']:.2f}  over={m['over_ink_score']:.2f}  under={m['under_ink_score']:.2f}\n"
                    f"scale=({m['scale_x']:.2f},{m['scale_y']:.2f})  sw_ratio={m['sw_ratio']:.2f}"
                )
                t_bin, g_bin = m['_t_bin'], m['_g_bin']
                base = cv2.cvtColor(t_bin, cv2.COLOR_GRAY2BGR)
                base[t_bin > 0] = (180, 180, 180)
                base[m['_extra_ink'] > 0]   = (60, 60, 255)
                base[m['_missing_ink'] > 0] = (255, 100, 60)
                diff_img = base
            else:  # v3
                m = compute_char_quality_v3(tmpl, tgt_for_metric,
                                            pad_y=self.parent_window.pad_y,
                                            pad_x=self.parent_window.pad_x,
                                            clean_fragments=self.parent_window.clean_fragments)
                conf = m['confidence']
                self.metrics = {
                    'method': 'v3', 'conf': conf,
                    'ncc': m['ncc'], 'ecc_cc': m['ecc_cc'],
                    'over_ink': m['over_ink_score'], 'under_ink': m['under_ink_score'],
                    'defect_type': m['defect_type'],
                }
                dt = m['defect_type'] or '-'
                metric_txt = (f"v3  conf={conf:.3f}  defect={dt}\n"
                              f"ncc={m['ncc']:.2f}  over={m['over_ink_score']:.2f}  under={m['under_ink_score']:.2f}")
                t_bin, g_bin = m['_t_bin'], m['_g_bin']
                base = cv2.cvtColor(t_bin, cv2.COLOR_GRAY2BGR)
                base[t_bin > 0] = (180, 180, 180)
                base[m['_extra_ink'] > 0]   = (60, 60, 255)
                base[m['_missing_ink'] > 0] = (255, 100, 60)
                diff_img = base
        except Exception as e:
            self.lbl_head.setText(f"err: {e}")
            self.verdict = '?'
            return

        # Predict verdict: conf >= threshold = OK, else NG
        self.verdict = 'OK' if conf >= threshold else 'NG'
        # v5: tile analysis có thể surface defect cục bộ → ép NG dù conf cao
        if self.parent_window.method == 'v5' and self.metrics.get('defect_type'):
            self.verdict = 'NG'
        # v6: SAML outlier → ép NG (đã được phản ánh trong conf nhưng đảm bảo)
        if self.parent_window.method == 'v6' and self.metrics.get('is_outlier'):
            self.verdict = 'NG'
        # v7: shape mismatch defect → ép NG
        if self.parent_window.method == 'v7' and self.metrics.get('defect_type'):
            self.verdict = 'NG'
        self.metrics['verdict'] = self.verdict
        self.metrics['threshold'] = threshold

        self.lbl_head.setText(
            f"char{self.pair['char_idx']:02d}  |  logged {self.pair['logged_label']} {self.pair['logged_p']:.2f}"
        )
        bg = COLOR_VERDICT[self.verdict]
        self.lbl_verdict.setText(f"PREDICT: {self.verdict}   (conf {conf:.2f} vs thr {threshold:.2f})")
        self.lbl_verdict.setStyleSheet(f"background:{bg};color:white;border-radius:3px;")
        self.lbl_metrics.setText(metric_txt)
        self.lbl_tmpl.setPixmap(cv_to_qpixmap(tmpl))
        self.lbl_tgt.setPixmap(cv_to_qpixmap(tgt))
        self.lbl_diff.setPixmap(cv_to_qpixmap(diff_img))

    def mark(self, label):
        self.label = label
        self._apply_border()
        self.parent_window.on_label_changed(self)

    def show_detail(self):
        """Open a separate window with full visualization."""
        dlg = DetailDialog(self.pair, self.metrics, self.parent_window)
        dlg.exec_()


class DetailDialog(QDialog):
    """Larger visualization of a single pair."""

    def __init__(self, pair, metrics, parent_window):
        super().__init__()
        self.setWindowTitle(f"Detail: {pair['folder']} char{pair['char_idx']:02d}")
        self.resize(960, 540)
        self.pair = pair
        self.parent_window = parent_window
        v = QVBoxLayout(self)

        tmpl = cv2.imread(pair['tmpl_path'], cv2.IMREAD_GRAYSCALE)
        tgt  = cv2.imread(pair['tgt_path'],  cv2.IMREAD_GRAYSCALE)

        try:
            m_old = _compute_char_quality(tmpl, tgt, denoise=False)
            pad_y = parent_window.pad_y
            pad_x = parent_window.pad_x
            m_v3  = compute_char_quality_v3(tmpl, tgt, pad_y=pad_y, pad_x=pad_x)
            m_v4  = compute_char_quality_v4(tmpl, tgt, pad_y=pad_y, pad_x=pad_x)
            m_v5  = compute_char_quality_v5(tmpl, tgt, pad_y=pad_y, pad_x=pad_x)
        except Exception as e:
            v.addWidget(QLabel(f"Error: {e}"))
            return

        panel = self._build_panel(tmpl, tgt, m_old, m_v5)
        # Append v5 heatmap as an extra row underneath
        heat = m_v5['_tile_heatmap']
        heat = cv2.resize(heat, (panel.shape[1] // 4, panel.shape[1] // 4),
                          interpolation=cv2.INTER_NEAREST)
        cv2.putText(heat, "v5 tile heatmap (red=over,blue=under)",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        # center it under the panel
        canvas = np.full((heat.shape[0], panel.shape[1], 3), 30, dtype=np.uint8)
        xo = (panel.shape[1] - heat.shape[1]) // 2
        canvas[:, xo:xo + heat.shape[1]] = heat
        full = np.vstack([panel, canvas])

        lbl = QLabel()
        lbl.setPixmap(cv_to_qpixmap(full, target=900))
        lbl.setAlignment(Qt.AlignCenter)
        v.addWidget(lbl)

        loc = m_v5['defect_location']
        loc_str = f"@tile{loc}" if loc else ''
        info = QLabel(
            f"OLD  conf={m_old['confidence']:.3f}  blur_tm={m_old['blur_tm']:.3f}  iou={m_old['iou']:.3f}  px={m_old['pixel_conf']:.3f}\n"
            f"v3   conf={m_v3['confidence']:.3f}  ncc={m_v3['ncc']:.3f}  "
            f"over={m_v3['over_ink_score']:.3f}  under={m_v3['under_ink_score']:.3f}  defect={m_v3['defect_type']}\n"
            f"v4   conf={m_v4['confidence']:.3f}  ncc={m_v4['ncc']:.3f}  "
            f"over={m_v4['over_ink_score']:.3f}  under={m_v4['under_ink_score']:.3f}  defect={m_v4['defect_type']}  "
            f"motion={m_v4['motion']}  scale=({m_v4['scale_x']:.2f},{m_v4['scale_y']:.2f})  sw={m_v4['sw_ratio']:.2f}\n"
            f"v5   conf={m_v5['confidence']:.3f}  ncc={m_v5['ncc']:.3f}  defect={m_v5['defect_type']} {loc_str}  "
            f"GLB(over={m_v5['over_ink_score']:.2f},under={m_v5['under_ink_score']:.2f})  "
            f"TILE(max_over={m_v5['tile_over_max']:.2f},max_under={m_v5['tile_under_max']:.2f},bad={m_v5['n_bad_tiles']})"
        )
        info.setFont(QFont('Menlo', 10))
        v.addWidget(info)

    def _build_panel(self, tmpl, tgt, m_old, m_new, scale=4):
        t_prep, g_aligned = m_new['_t_prep'], m_new['_g_aligned']
        t_bin, g_bin = m_new['_t_bin'], m_new['_g_bin']
        extra, missing = m_new['_extra_ink'], m_new['_missing_ink']
        xor = cv2.bitwise_xor(t_bin, g_bin)
        base = cv2.cvtColor(t_bin, cv2.COLOR_GRAY2BGR)
        base[t_bin > 0] = (180, 180, 180)
        base[extra > 0]   = (60, 60, 255)
        base[missing > 0] = (255, 100, 60)

        tmpl_r = cv2.resize(tmpl, (64, 64), interpolation=cv2.INTER_AREA)
        tgt_r  = cv2.resize(tgt,  (64, 64), interpolation=cv2.INTER_AREA)
        rows = [
            [("tmpl raw", tmpl_r), ("tgt raw", tgt_r), ("tmpl prep", t_prep), ("tgt aligned", g_aligned)],
            [("tmpl bin", t_bin), ("tgt bin", g_bin), ("XOR", xor), ("LEM=red MAT=blue", base)],
        ]
        out_rows = []
        for row in rows:
            imgs = []
            for label, im in row:
                if im.ndim == 2:
                    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                im = cv2.resize(im, (64*scale, 64*scale), interpolation=cv2.INTER_NEAREST)
                cv2.putText(im, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                imgs.append(im)
            out_rows.append(np.hstack(imgs))
        return np.vstack(out_rows)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Char Quality Review Tool")
        self.resize(1500, 900)
        self.root_dir = None
        self.save_dir = str(Path.home() / 'Downloads' / 'char_review_labels')
        self.labels = {}            # key: tgt_path → dict with label, metrics, method
        self.cards: list = []
        self.method = 'v6'
        self.threshold = DEFAULT_THRESHOLD
        self.apply_frag_remove = False
        self.clean_fragments = True
        self.pad_y = 4
        self.pad_x = 1
        self.n_sigma = N_SIGMA_DEFAULT
        # SAML cache: per-folder result so all cards share frame statistics.
        self._saml_cache = {}  # key=folder_path → frame result dict
        self._saml_per_char = {}  # key=(folder_path, char_idx) → per-char dict
        self._build()
        self._load_existing_labels()

    def _build(self):
        # Central splitter: left list | center grid | right controls
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # --- LEFT panel: folder list ---
        left = QWidget()
        lv = QVBoxLayout(left)
        btn_import = QPushButton("📂 Import root folder")
        btn_import.clicked.connect(self.on_import_root)
        lv.addWidget(btn_import)

        self.lbl_root = QLabel("(no folder)")
        self.lbl_root.setWordWrap(True)
        self.lbl_root.setStyleSheet("color:#aaa;font-size:11px;")
        lv.addWidget(self.lbl_root)

        self.list_folders = QListWidget()
        self.list_folders.itemClicked.connect(self.on_folder_selected)
        lv.addWidget(self.list_folders, 1)

        splitter.addWidget(left)
        splitter.setStretchFactor(0, 0)

        # --- CENTER panel: scrollable grid of cards ---
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setSpacing(8)
        self.grid.setContentsMargins(8, 8, 8, 8)
        self.scroll.setWidget(self.grid_host)
        splitter.addWidget(self.scroll)
        splitter.setStretchFactor(1, 1)

        # --- RIGHT panel: controls ---
        right = QWidget()
        rv = QVBoxLayout(right)

        # Method selector
        gb_method = QGroupBox("Method")
        gm = QVBoxLayout(gb_method)
        self.cb_method = QComboBox()
        self.cb_method.addItems([
            'v7 (shape-based: gradient orientation)',
            'v6 (SAML stochastic: self+baseline)',
            'v5 (tile-wise + scale-invariant)',
            'v4 (scale-invariant + directional)',
            'v3 (directional diff)',
            'OLD (blur_tm + iou + px)',
        ])
        self.cb_method.currentTextChanged.connect(self.on_method_changed)
        gm.addWidget(self.cb_method)

        self.chk_frag = QCheckBox("Apply remove_fragments_local_bg trên target")
        self.chk_frag.toggled.connect(self.on_frag_toggled)
        gm.addWidget(self.chk_frag)

        self.chk_clean_cc = QCheckBox("Clean fragments (giữ largest CC)")
        self.chk_clean_cc.setChecked(True)
        self.chk_clean_cc.toggled.connect(self.on_clean_cc_toggled)
        gm.addWidget(self.chk_clean_cc)

        # Threshold spinner
        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel("Threshold OK ≥"))
        self.spin_thr = QDoubleSpinBox()
        self.spin_thr.setRange(0.0, 1.0)
        self.spin_thr.setSingleStep(0.01)
        self.spin_thr.setDecimals(2)
        self.spin_thr.setValue(DEFAULT_THRESHOLD)
        self.spin_thr.valueChanged.connect(self.on_threshold_changed)
        thr_row.addWidget(self.spin_thr)
        gm.addLayout(thr_row)

        # Padding spinners (around Otsu bbox before resize)
        pad_row = QHBoxLayout()
        pad_row.addWidget(QLabel("Pad Y (top+bot)"))
        self.spin_pad_y = QSpinBox()
        self.spin_pad_y.setRange(0, 20)
        self.spin_pad_y.setValue(self.pad_y)
        self.spin_pad_y.valueChanged.connect(self.on_pad_y_changed)
        pad_row.addWidget(self.spin_pad_y)
        pad_row.addWidget(QLabel("Pad X (l+r)"))
        self.spin_pad_x = QSpinBox()
        self.spin_pad_x.setRange(0, 20)
        self.spin_pad_x.setValue(self.pad_x)
        self.spin_pad_x.valueChanged.connect(self.on_pad_x_changed)
        pad_row.addWidget(self.spin_pad_x)
        gm.addLayout(pad_row)

        # SAML n_sigma spinner (only used for v6)
        sig_row = QHBoxLayout()
        sig_row.addWidget(QLabel("SAML n_sigma"))
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.5, 5.0)
        self.spin_sigma.setSingleStep(0.1)
        self.spin_sigma.setDecimals(1)
        self.spin_sigma.setValue(self.n_sigma)
        self.spin_sigma.valueChanged.connect(self.on_sigma_changed)
        sig_row.addWidget(self.spin_sigma)
        gm.addLayout(sig_row)
        rv.addWidget(gb_method)

        # Save dir
        gb_save = QGroupBox("Save labels to")
        gs = QVBoxLayout(gb_save)
        self.le_save = QLineEdit(self.save_dir)
        self.le_save.editingFinished.connect(self.on_save_dir_changed)
        gs.addWidget(self.le_save)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self.on_browse_save)
        gs.addWidget(btn_browse)
        btn_open = QPushButton("Open in Finder")
        btn_open.clicked.connect(self.on_open_save)
        gs.addWidget(btn_open)
        rv.addWidget(gb_save)

        # Stats
        gb_stats = QGroupBox("Stats")
        gst = QVBoxLayout(gb_stats)
        self.lbl_stats = QLabel("(empty)")
        self.lbl_stats.setFont(QFont('Menlo', 11))
        gst.addWidget(self.lbl_stats)
        rv.addWidget(gb_stats)

        # Hint
        hint = QLabel(
            "<b>Workflow</b><br>"
            "1. Import root folder (chứa <code>emb_*/</code>).<br>"
            "2. Click 1 folder bên trái.<br>"
            "3. Mỗi card hiện <b>PREDICT: OK</b> (xanh) hoặc <b>NG</b> (đỏ).<br>"
            "4. Bấm <span style='color:#27ae60'><b>✓ Đúng</b></span> / "
            "<span style='color:#c0392b'><b>✗ Sai</b></span> để verify.<br>"
            "5. 🔍 = visualize đầy đủ.<br>"
            "<br><b>Label theo method</b>: cùng 1 cặp có thể có label "
            "riêng cho v3/v4/v5. Đổi method → border reset theo label "
            "của method mới.<br>"
            "<br><b>Saved vào</b>: <code>&lt;save&gt;/&lt;method&gt;/&lt;outcome&gt;/</code><br>"
            "• <b>TP</b> = đúng NG (bắt defect đúng)<br>"
            "• <b>TN</b> = đúng OK (clean)<br>"
            "• <b>FP</b> = NG nhưng thực OK (false alarm)<br>"
            "• <b>FN</b> = OK nhưng thực NG (escape ⚠)<br>"
            "<br><b>Diff colors</b>: 🔴 LEM mực, 🔵 MẤT nét."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#bbb;font-size:11px;")
        rv.addWidget(hint)

        self.btn_refresh = QPushButton("🔄 Refresh metrics")
        self.btn_refresh.clicked.connect(self.refresh_all_cards)
        rv.addWidget(self.btn_refresh)

        rv.addStretch(1)
        right.setFixedWidth(320)
        splitter.addWidget(right)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([280, 900, 320])

        # Dark theme
        self.setStyleSheet("""
            QMainWindow { background:#252525; }
            QWidget { color: #e0e0e0; }
            QGroupBox { border:1px solid #444; margin-top:8px; padding-top:14px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; }
            QPushButton { background:#3a3a3a; padding:6px; border:1px solid #555; border-radius:3px; }
            QPushButton:hover { background:#4a4a4a; }
            QListWidget { background:#1e1e1e; border:1px solid #444; }
            QComboBox, QLineEdit { background:#1e1e1e; padding:4px; border:1px solid #444; }
            QScrollArea { background:#1a1a1a; border:1px solid #444; }
            QLabel { background: transparent; }
        """)

    # ------------------------------------------------------------------ events
    def on_import_root(self):
        d = QFileDialog.getExistingDirectory(self, "Choose root folder containing emb_*")
        if not d:
            return
        self.root_dir = d
        self.lbl_root.setText(d)
        self.list_folders.clear()
        subs = sorted([p for p in Path(d).glob("emb_*") if p.is_dir()])
        if not subs:
            # fallback: any subdir with target/template pairs
            subs = sorted([p for p in Path(d).iterdir()
                           if p.is_dir() and any(PAIR_RE.match(f.name) for f in p.iterdir())])
        for p in subs:
            item = QListWidgetItem(p.name)
            item.setData(Qt.UserRole, str(p))
            self.list_folders.addItem(item)
        self.statusBar().showMessage(f"Found {len(subs)} folders in {d}")

    def on_folder_selected(self, item: QListWidgetItem):
        folder = Path(item.data(Qt.UserRole))
        pairs = []
        for f in sorted(folder.iterdir()):
            m = PAIR_RE.match(f.name)
            if not m:
                continue
            tmpl = folder / f.name.replace("_target.png", "_template.png")
            if not tmpl.exists():
                continue
            pairs.append({
                'folder': folder.name,
                'char_idx': int(m.group(1)),
                'logged_label': m.group(2),
                'logged_p': float(m.group(3)),
                'tmpl_path': str(tmpl),
                'tgt_path': str(f),
            })
        self._populate_grid(pairs)
        # Pre-compute SAML for this folder once (cross-char statistics)
        self._recompute_saml_for_current_folder()
        self.refresh_all_cards()
        self.statusBar().showMessage(f"Loaded {len(pairs)} pairs from {folder.name}")

    def _populate_grid(self, pairs):
        # Clear old cards
        for c in self.cards:
            c.setParent(None)
            c.deleteLater()
        self.cards.clear()

        cols = max(1, (self.scroll.viewport().width() - 16) // (CARD_W + 10))
        for i, pair in enumerate(pairs):
            card = PairCard(pair, self)
            # Restore label if previously saved
            existing = self.labels.get(pair['tgt_path'])
            if existing:
                card.label = existing.get('label', LABEL_NONE)
                card._apply_border()
            card.refresh()
            r, c = divmod(i, cols)
            self.grid.addWidget(card, r, c)
            self.cards.append(card)
        self.update_stats()

    def on_method_changed(self, txt):
        if   txt.startswith('v7'): self.method = 'v7'
        elif txt.startswith('v6'): self.method = 'v6'
        elif txt.startswith('v5'): self.method = 'v5'
        elif txt.startswith('v4'): self.method = 'v4'
        elif txt.startswith('v3'): self.method = 'v3'
        else:                       self.method = 'OLD'
        self.refresh_all_cards()
        self._restore_card_labels_for_method()
        self.update_stats()

    def on_frag_toggled(self, on):
        self.apply_frag_remove = on
        self.refresh_all_cards()

    def on_clean_cc_toggled(self, on):
        self.clean_fragments = on
        # Invalidate SAML cache since H_MAP depends on this
        self._saml_cache.clear()
        self._saml_per_char.clear()
        if self.cards:
            self._recompute_saml_for_current_folder()
        self.refresh_all_cards()

    def on_threshold_changed(self, v):
        self.threshold = float(v)
        self.refresh_all_cards()

    def on_pad_y_changed(self, v):
        self.pad_y = int(v)
        self.refresh_all_cards()

    def on_pad_x_changed(self, v):
        self.pad_x = int(v)
        self.refresh_all_cards()

    def on_sigma_changed(self, v):
        self.n_sigma = float(v)
        # Invalidate SAML cache because thresholds depend on n_sigma
        self._saml_cache.clear()
        self._saml_per_char.clear()
        if self.cards:
            self._recompute_saml_for_current_folder()
        self.refresh_all_cards()

    def _recompute_saml_for_current_folder(self):
        """Compute SAML once for current folder; populate per-char cache."""
        if not self.cards:
            return
        folder = self.cards[0].pair['folder']
        targets, templates, char_idxs = [], [], []
        for c in self.cards:
            tmpl = cv2.imread(c.pair['tmpl_path'], cv2.IMREAD_GRAYSCALE)
            tgt  = cv2.imread(c.pair['tgt_path'],  cv2.IMREAD_GRAYSCALE)
            if tmpl is None or tgt is None:
                continue
            targets.append(tgt)
            templates.append(tmpl)
            char_idxs.append(c.pair['char_idx'])
        if not targets:
            return
        frame = compute_saml_frame(targets, templates, n_sigma=self.n_sigma,
                                    clean_fragments=self.clean_fragments)
        self._saml_cache[folder] = frame
        for i, ci in enumerate(char_idxs):
            self._saml_per_char[(folder, ci)] = frame['per_char'][i]

    def refresh_all_cards(self):
        for c in self.cards:
            c.refresh()

    def on_save_dir_changed(self):
        self.save_dir = self.le_save.text().strip()
        self._load_existing_labels()

    def on_browse_save(self):
        d = QFileDialog.getExistingDirectory(self, "Choose save directory", self.save_dir)
        if d:
            self.save_dir = d
            self.le_save.setText(d)
            self._load_existing_labels()

    def on_open_save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        os.system(f'open "{self.save_dir}"')

    # ------------------------------------------------------------------ label IO
    def _labels_json_path(self) -> Path:
        return Path(self.save_dir) / 'labels.json'

    def _load_existing_labels(self):
        """labels schema (nested by method): {tgt_path: {method: entry}}.
        Migrate old flat schema ({tgt_path: entry}) → wrap under 'unknown'."""
        self.labels = {}
        p = self._labels_json_path()
        if p.exists():
            try:
                raw = json.loads(p.read_text())
                for k, v in raw.items():
                    if isinstance(v, dict) and 'label' in v and 'verdict' in v:
                        # Old flat format → put under method from entry, or 'unknown'
                        meth = v.get('method', 'unknown')
                        self.labels[k] = {meth: v}
                    elif isinstance(v, dict):
                        self.labels[k] = v
            except Exception:
                pass
        self._restore_card_labels_for_method()
        self.update_stats()

    def _restore_card_labels_for_method(self):
        """Set each card's label to the entry stored for the current method."""
        for c in self.cards:
            entry = self.labels.get(c.pair['tgt_path'], {}).get(self.method)
            c.label = entry.get('label', LABEL_NONE) if entry else LABEL_NONE
            c._apply_border()

    def on_label_changed(self, card: PairCard):
        """Save label PER METHOD — same pair can have different labels for v3/v4/v5."""
        path = card.pair['tgt_path']
        method = self.method
        os.makedirs(self.save_dir, exist_ok=True)
        base = f"{card.pair['folder']}__char{card.pair['char_idx']:02d}"

        # Sweep old copies for THIS (path, method) from all outcome dirs
        for sub in ('TP', 'TN', 'FP', 'FN'):
            for variant in ('target.png', 'template.png'):
                p = Path(self.save_dir) / method / sub / f"{base}_{variant}"
                if p.exists():
                    try: p.unlink()
                    except Exception: pass

        if card.label == LABEL_NONE:
            self.labels.get(path, {}).pop(method, None)
            if path in self.labels and not self.labels[path]:
                del self.labels[path]
        else:
            verdict = card.verdict
            actual = verdict if card.label == LABEL_CORRECT else ('NG' if verdict == 'OK' else 'OK')
            outcome = outcome_of(verdict, actual)  # TP/TN/FP/FN

            sub_dir = Path(self.save_dir) / method / outcome
            sub_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(card.pair['tgt_path'],  sub_dir / f"{base}_target.png")
            shutil.copy2(card.pair['tmpl_path'], sub_dir / f"{base}_template.png")

            self.labels.setdefault(path, {})[method] = {
                'label': card.label,            # 'correct' | 'wrong'
                'verdict': verdict,             # what algo predicted
                'actual': actual,               # derived ground truth (under this method's verdict)
                'outcome': outcome,             # TP/TN/FP/FN
                'method': method,
                'threshold': self.threshold,
                'metrics': {k: v for k, v in (card.metrics or {}).items() if k not in ('method', 'verdict', 'threshold')},
                'folder': card.pair['folder'],
                'char_idx': card.pair['char_idx'],
                'logged_label': card.pair['logged_label'],
                'logged_p': card.pair['logged_p'],
                'tmpl_path': card.pair['tmpl_path'],
                'tgt_path': card.pair['tgt_path'],
                'timestamp': datetime.now().isoformat(timespec='seconds'),
            }

        self._labels_json_path().write_text(json.dumps(self.labels, indent=2, default=float))
        self.update_stats()

    def update_stats(self):
        """Confusion matrix cho method đang chọn + summary all methods."""
        # Per-method cells
        method_cells = {}
        for path, by_method in self.labels.items():
            for meth, entry in by_method.items():
                cells = method_cells.setdefault(meth, {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0})
                oc = entry.get('outcome')
                if oc in cells:
                    cells[oc] += 1

        # Current method detail
        cur = method_cells.get(self.method, {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0})
        tp, tn, fp, fn = cur['TP'], cur['TN'], cur['FP'], cur['FN']
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        lines = [
            f"-- {self.method} --",
            "                   actual",
            "                  NG    OK",
            f"predict NG  TP={tp:>3}  FP={fp:>3}",
            f"predict OK  FN={fn:>3}  TN={tn:>3}",
            "",
            f"total      = {total}",
            f"accuracy   = {accuracy:.2%}",
            f"precision  = {precision:.2%}  (TP / predict-NG)",
            f"recall     = {recall:.2%}  (TP / actual-NG)",
            f"F1         = {f1:.2%}",
            "",
            f"FN escape       = {fn}",
            f"FP false alarm  = {fp}",
        ]

        # Cross-method summary if multiple methods labeled
        if len(method_cells) > 1:
            lines.append("")
            lines.append("-- all methods --")
            lines.append(f"{'meth':<5}{'TP':>4}{'TN':>4}{'FP':>4}{'FN':>4}{'acc':>7}")
            for meth in sorted(method_cells.keys()):
                c = method_cells[meth]
                t = c['TP'] + c['TN'] + c['FP'] + c['FN']
                ac = (c['TP'] + c['TN']) / t if t else 0
                lines.append(f"{meth:<5}{c['TP']:>4}{c['TN']:>4}{c['FP']:>4}{c['FN']:>4}{ac:>6.0%} ")

        self.lbl_stats.setText("\n".join(lines))

    def resizeEvent(self, ev):
        # Re-layout grid when window resized — keeps card columns nice
        super().resizeEvent(ev)
        if self.cards:
            cols = max(1, (self.scroll.viewport().width() - 16) // (CARD_W + 10))
            for i, c in enumerate(self.cards):
                r, col = divmod(i, cols)
                self.grid.addWidget(c, r, col)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
