"""char_quality_v6_saml — SAML-QC stochastic intensity assessment.

Dựa trên paper "SAML-QC" (Hussain 2019, arxiv 1901.07370), Section II.C.1
(Stochastic Assessment of Printing Quality).

Khác v3/v4/v5: KHÔNG so sánh shape pixel-by-pixel. So sánh **mean ink intensity**
của char với 2 reference:

  1. Self-referenced (paper default):
     - H_MAP(u) = mean intensity of INK pixels (after Otsu) in char u
     - Compute mean & std across ALL chars in frame
     - Char outlier khi |H - E[H]| > n_sigma * σ_H
     - → Bắt defect CỤC BỘ (1 chữ lem giữa các chữ đều)

  2. Baseline-referenced (extension fix paper limitation):
     - Cùng tính H_MAP cho ALL TEMPLATES có sẵn
     - Compute mean & std across templates
     - Char outlier khi |H_target - E[H_template]| > n_sigma * σ_H_template
     - → Bắt defect TOÀN FRAME (cả frame lệch khỏi template baseline)

  3. Combined verdict (recommended):
     - NG nếu outlier ở 1 trong 2 → robust với cả 2 fail mode

Defect type:
  H thấp hơn mean → ink đậm hơn = LEM (over_ink)
  H cao hơn mean  → ink nhạt hơn = MẤT NÉT / fading (under_ink)
"""

from typing import Dict, List, Optional

import cv2
import numpy as np

import sys
sys.path.insert(0, '/Users/ngocthien.ai/Source/Projects/ocr_datecode/tools')
from char_quality_v3 import _keep_largest_cc


N_SIGMA_DEFAULT = 2.0    # n trong công thức (17) — quality index
MIN_INK_PIXELS = 10      # tile có < này pixel ink → fallback all pixels


def _h_map(crop: np.ndarray, clean_fragments: bool = True) -> float:
    """Compute H_MAP = mean intensity of INK pixels (Otsu mask).

    Theo paper: H_MAP ≈ (1/K) Σ y_i với K = số pixel trong vùng chữ.
    Ta dùng ink pixels (sau Otsu inverse) thay vì all pixels — nhạy hơn với ink quality.
    clean_fragments: nếu True, dilate-then-largest-CC loại fragment kế bên trước khi tính.
    """
    if crop is None or crop.size == 0:
        return 255.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if clean_fragments:
        mask = _keep_largest_cc(mask, dilate_iter=2)
    if int(np.count_nonzero(mask)) < MIN_INK_PIXELS:
        return float(gray.mean())  # fallback
    return float(gray[mask > 0].mean())


def _per_char_assessment(
    h_target: float,
    h_template: Optional[float],
    frame_mean: float, frame_std: float,
    baseline_mean: Optional[float], baseline_std: Optional[float],
    n_sigma: float,
) -> Dict:
    """Combine self-ref + baseline-ref outlier detection for 1 char."""
    # Self-referenced
    dev_self = (h_target - frame_mean) / (frame_std + 1e-6) if frame_std > 0 else 0.0
    outlier_self = abs(dev_self) > n_sigma

    # Baseline-referenced
    dev_base = None
    outlier_base = False
    if h_template is not None and baseline_std is not None and baseline_std > 0:
        dev_base = (h_target - baseline_mean) / (baseline_std + 1e-6)
        outlier_base = abs(dev_base) > n_sigma

    # Defect type — based on the worse deviation (whichever has larger |dev|)
    # H lower than mean = darker ink = OVER_INK (lem)
    # H higher than mean = lighter ink = UNDER_INK (fading)
    primary_dev = dev_self
    if outlier_base and (not outlier_self or (dev_base is not None and abs(dev_base) > abs(dev_self))):
        primary_dev = dev_base

    defect_type = None
    is_outlier = outlier_self or outlier_base
    if is_outlier:
        defect_type = 'over_ink' if primary_dev < 0 else 'under_ink'

    # Confidence: smooth function of deviation
    # conf = exp(-0.5 * (max_dev / n_sigma)^2)  — Gaussian-style decay
    # At dev = n_sigma → conf ≈ 0.61, at dev = 2*n_sigma → conf ≈ 0.14
    max_dev = abs(primary_dev) if primary_dev is not None else 0.0
    confidence = float(np.exp(-0.5 * (max_dev / max(n_sigma, 0.1)) ** 2))

    return {
        'h_target':      float(h_target),
        'h_template':    float(h_template) if h_template is not None else None,
        'dev_self':      float(dev_self),
        'dev_baseline':  float(dev_base) if dev_base is not None else None,
        'outlier_self':     bool(outlier_self),
        'outlier_baseline': bool(outlier_base),
        'is_outlier':       bool(is_outlier),
        'defect_type':   defect_type,
        'confidence':    confidence,
    }


def compute_saml_frame(
    targets: List[np.ndarray],
    templates: Optional[List[np.ndarray]] = None,
    n_sigma: float = N_SIGMA_DEFAULT,
    clean_fragments: bool = True,
) -> Dict:
    """Frame-wise SAML assessment.

    Args:
        targets: list of N char crops in 1 frame (BGR or grayscale)
        templates: optional list of N templates (same length, same indexing as targets)
                   khi provided → enable baseline-referenced check
        n_sigma:  quality index (1=strict, 2=balanced=paper default, 3=lenient)

    Returns dict với:
        'per_char': List[Dict]  — assessment cho từng char (index khớp với targets)
        'frame_mean', 'frame_std'           — self-ref statistics
        'baseline_mean', 'baseline_std'     — baseline-ref (nếu có templates)
        'frame_dev_vs_baseline'             — |frame_mean - baseline_mean| / σ_baseline
        'frame_bad'                         — True nếu frame_mean lệch ≥ n_sigma vs baseline
        'quality_score'                     — % char OK
        'n_chars', 'n_sigma'
    """
    n = len(targets)
    if n == 0:
        return {
            'per_char': [], 'frame_mean': 0.0, 'frame_std': 0.0,
            'baseline_mean': None, 'baseline_std': None,
            'frame_dev_vs_baseline': 0.0, 'frame_bad': False,
            'quality_score': 0.0, 'n_chars': 0, 'n_sigma': n_sigma,
        }

    # Compute H_MAP for all targets
    h_targets = np.array([_h_map(c, clean_fragments=clean_fragments) for c in targets], dtype=np.float64)
    frame_mean = float(h_targets.mean())
    frame_std  = float(h_targets.std()) if n > 1 else 0.0

    # Compute baseline if templates provided
    h_templates = None
    base_mean = None
    base_std = None
    if templates is not None and len(templates) == n:
        h_templates = np.array([_h_map(t, clean_fragments=clean_fragments) for t in templates], dtype=np.float64)
        base_mean = float(h_templates.mean())
        base_std  = float(h_templates.std()) if len(h_templates) > 1 else 0.0

    # Frame-level: cả frame có lệch khỏi baseline không?
    frame_dev_vs_baseline = 0.0
    frame_bad = False
    if base_mean is not None and base_std is not None and base_std > 0:
        frame_dev_vs_baseline = (frame_mean - base_mean) / (base_std + 1e-6)
        # Nếu frame_mean lệch hẳn baseline → CẢ FRAME LỖI
        # threshold: n_sigma vẫn (cùng strictness level)
        frame_bad = abs(frame_dev_vs_baseline) > n_sigma

    # Per-char assessment
    per_char = []
    for i in range(n):
        h_t = h_templates[i] if h_templates is not None else None
        info = _per_char_assessment(
            h_target=h_targets[i], h_template=h_t,
            frame_mean=frame_mean, frame_std=frame_std,
            baseline_mean=base_mean, baseline_std=base_std,
            n_sigma=n_sigma,
        )
        # Nếu cả frame bad (toàn lem hoặc toàn mất nét) → mark all chars NG
        if frame_bad:
            info['frame_bad'] = True
            info['is_outlier'] = True
            if info['defect_type'] is None:
                info['defect_type'] = 'over_ink' if frame_dev_vs_baseline < 0 else 'under_ink'
            # Reduce confidence to NG zone
            info['confidence'] = min(info['confidence'], 0.30)
        else:
            info['frame_bad'] = False
        per_char.append(info)

    n_good = sum(1 for c in per_char if not c['is_outlier'])
    qs = 100.0 * n_good / n

    return {
        'per_char': per_char,
        'frame_mean': frame_mean,
        'frame_std':  frame_std,
        'baseline_mean': base_mean,
        'baseline_std':  base_std,
        'frame_dev_vs_baseline': float(frame_dev_vs_baseline),
        'frame_bad': bool(frame_bad),
        'quality_score': qs,
        'n_chars': n,
        'n_sigma': n_sigma,
    }


def render_h_histogram(h_values: np.ndarray, mean: float, std: float,
                        n_sigma: float, current_h: Optional[float] = None,
                        size: int = 240) -> np.ndarray:
    """Render histogram đơn giản cho H values với marker char hiện tại.
    Returns BGR image (size, size, 3) uint8."""
    if len(h_values) == 0:
        return np.full((size, size, 3), 30, dtype=np.uint8)
    img = np.full((size, size, 3), 30, dtype=np.uint8)
    h_min, h_max = float(min(h_values.min(), 0)), float(max(h_values.max(), 255))
    # Pad range slightly
    lo = max(0, mean - 4 * std)
    hi = min(255, mean + 4 * std)
    if hi - lo < 10:
        lo, hi = 0, 255

    # Histogram bars
    bins = 24
    hist, edges = np.histogram(h_values, bins=bins, range=(lo, hi))
    max_h = max(1, int(hist.max()))
    bar_w = (size - 20) // bins
    for i, count in enumerate(hist):
        bh = int(count / max_h * (size - 40))
        x0 = 10 + i * bar_w
        y0 = size - 10 - bh
        cv2.rectangle(img, (x0, y0), (x0 + bar_w - 2, size - 10), (120, 120, 120), -1)

    # Mean line (green)
    mx = int(10 + (mean - lo) / (hi - lo + 1e-6) * (size - 20))
    cv2.line(img, (mx, 10), (mx, size - 10), (0, 200, 0), 1)
    # ±n_sigma lines (yellow)
    for s in (-n_sigma, n_sigma):
        x = int(10 + (mean + s * std - lo) / (hi - lo + 1e-6) * (size - 20))
        if 0 < x < size:
            cv2.line(img, (x, 10), (x, size - 10), (60, 200, 200), 1)
    # Current char (red)
    if current_h is not None:
        x = int(10 + (current_h - lo) / (hi - lo + 1e-6) * (size - 20))
        x = max(0, min(size - 1, x))
        cv2.line(img, (x, 5), (x, size - 5), (60, 60, 255), 2)

    cv2.putText(img, f"mu={mean:.0f}  sig={std:.2f}", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
    return img
