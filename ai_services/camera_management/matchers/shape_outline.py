"""
Shape outline matcher — phase-correlation-based alternative to SuperPoint.

Despite the historical filename, the current implementation uses
`cv2.phaseCorrelate` (NOT contour outlines). The function name
`match_shape_outline` is preserved as the public API so the rest of the
codebase + recipe config (`crop_match_method='shape_outline'`) don't have to
change.

Algorithm
─────────
1. Convert template + target to grayscale.
2. Resize template to target's pixel size so phaseCorrelate gets matching
   dimensions (phase correlation requires equal-sized inputs).
3. Apply a Hanning window to suppress edge-discontinuity artifacts.
4. `cv2.phaseCorrelate(template_resized, target)` → sub-pixel shift `(dx, dy)`
   and a response peak (confidence-like scalar).
5. Build a 2×3 affine combining the template→target scale + the translation
   shift. The shape_outline contract is a 2×3 affine homography projected to
   3×3 for `cv2.perspectiveTransform`.
6. Apply that warp to `template_bbox` + `other_bboxes` → annotation positions
   in target coords.

Why phase correlation (vs the old contour + RANSAC affine)
──────────────────────────────────────────────────────────
After the `cap_rotation` + `cap_crop` pipeline steps, template and target are
already roughly aligned (same orientation, similar size). The residual error
is mostly a few-pixel translation from imperfect cap-center detection. Phase
correlation handles that case in 5-15ms (vs 30-60ms for the contour path) and
sub-pixel accuracy. It does NOT model residual rotation or non-uniform scale
— if those are expected, switch the recipe to `crop_match_method='superpoint'`
instead.

Return shape matches SuperPointMatcher.match_batch() per-entry result so the
pipeline downstream code (`_transform_func`, `inverse_transform_bboxes`,
`dual_rotation` winner pick) doesn't need changes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


logger = logging.getLogger(__name__)


# Debug image dump — turn off via env var in production.
_DEBUG_DIR = Path("debug_shape")
_DEBUG_ENABLED = True
# Phase-correlation raw response is already roughly 0-1: identical inputs ≈
# 1.0; translation-only with matching content ≈ 0.7-0.95; rotation/scale
# mismatch ≈ 0.3-0.5 (orientation must be aligned by cap_rotation first); pure
# noise <0.1. Use raw response directly so the `matching_conf` gate (default
# 0.20) acts as a "cap_rotation actually aligned things" sanity check.
_CONF_GAIN = 1.0
# Downsample target's longer side to this before FFT. Phase correlation cost is
# O(N log N) on the FFT, so 1600×1200 takes ~150ms but 512×384 takes ~10-15ms.
# Sub-pixel accuracy is preserved well enough for cap-OCR bbox transfer.
_DOWNSAMPLE_TO = 512


def _save_debug(img: np.ndarray, name: str, serial: str = "x") -> None:
    if not _DEBUG_ENABLED or img is None or img.size == 0:
        return
    try:
        _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        if img.dtype != np.uint8:
            mn, mx = float(img.min()), float(img.max())
            vis = ((img - mn) / (mx - mn) * 255.0).astype(np.uint8) if mx > mn else img.astype(np.uint8)
        else:
            vis = img
        cv2.imwrite(str(_DEBUG_DIR / f"{serial}_{name}.jpg"), vis)
    except Exception as e:
        logger.warning(f"shape_outline debug save failed ({name}): {e}")


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _fail_result(target_img, err: str, t_start: float) -> Dict[str, Any]:
    import time as _time
    total = (_time.perf_counter() - t_start) * 1000
    return {
        'success': False,
        'error': err,
        'homography': None,
        'confidence': 0.0,
        'inliers': 0,
        'total_matches': 0,
        'transformed_bboxes': [],
        'target_img': target_img,
        'timings': {'method': 'shape_outline_phasecorr', 'total': total},
    }


def match_shape_outline(
    template_img: np.ndarray,
    target_img: np.ndarray,
    template_bbox: Optional[Dict[str, Any]] = None,
    other_bboxes: Optional[List[Dict[str, Any]]] = None,
    serial: str = "x",
) -> Dict[str, Any]:
    """
    Phase-correlation alignment of template ↔ target. Returns the same result
    dict shape as SuperPoint match (success / homography / confidence /
    transformed_bboxes / ...).
    """
    import time as _time
    t_total = _time.perf_counter()

    if template_img is None or target_img is None:
        return _fail_result(target_img, "null input image", t_total)

    _save_debug(template_img, "0_template_raw", serial)
    _save_debug(target_img, "0_target_raw", serial)

    t_gray = _to_gray(template_img)
    g_gray = _to_gray(target_img)
    if t_gray.size == 0 or g_gray.size == 0:
        return _fail_result(target_img, "empty image after gray", t_total)

    Ht, Wt = t_gray.shape[:2]
    Hg, Wg = g_gray.shape[:2]

    # Downsample both to a common small canvas so FFT stays fast. The shift
    # we recover is in down-sampled pixels; we scale it back to target coords
    # at the end. Aspect ratio preserved (we match target's aspect ratio so
    # template is resized to fit).
    ds_scale = float(_DOWNSAMPLE_TO) / float(max(Hg, Wg))
    if ds_scale < 1.0:
        ds_W = max(8, int(round(Wg * ds_scale)))
        ds_H = max(8, int(round(Hg * ds_scale)))
    else:
        ds_scale = 1.0
        ds_W, ds_H = Wg, Hg

    g_ds = cv2.resize(g_gray, (ds_W, ds_H), interpolation=cv2.INTER_AREA)
    t_ds = cv2.resize(t_gray, (ds_W, ds_H), interpolation=cv2.INTER_AREA)

    t_phase = _time.perf_counter()
    try:
        t_f = t_ds.astype(np.float32)
        g_f = g_ds.astype(np.float32)
        hann = cv2.createHanningWindow((ds_W, ds_H), cv2.CV_32F)
        (dx_ds, dy_ds), response = cv2.phaseCorrelate(t_f, g_f, hann)
    except cv2.error as e:
        return _fail_result(target_img, f"phaseCorrelate failed: {e}", t_total)
    t_phase_ms = (_time.perf_counter() - t_phase) * 1000

    # Phase shift was measured in down-sampled target pixels → scale back to
    # full target-pixel space (divide by ds_scale because ds_scale<1 means we
    # shrunk; a 1-px ds shift = 1/ds_scale px in full target).
    dx = float(dx_ds) / ds_scale
    dy = float(dy_ds) / ds_scale

    scale_x = float(Wg) / float(Wt)
    scale_y = float(Hg) / float(Ht)

    # 2×3 affine: scale template→target dims, then translate by phase shift.
    # `template_bbox` is in template coords → multiplying by (scale_x, scale_y)
    # lands in target-pixel space; adding (dx, dy) accounts for the residual
    # translation phase correlation detected.
    M = np.array([
        [scale_x, 0.0,     dx],
        [0.0,     scale_y, dy],
    ], dtype=np.float32)

    warp_3x3 = np.eye(3, dtype=np.float32)
    warp_3x3[:2] = M

    transformed: List[Dict[str, Any]] = []
    if template_bbox and template_bbox.get('points'):
        pts = np.array(template_bbox['points'], dtype=np.float32).reshape(-1, 1, 2)
        pts_out = cv2.perspectiveTransform(pts, warp_3x3)
        transformed.append({
            'type': 'template',
            'points': pts_out.reshape(-1, 2).tolist(),
            'conf': template_bbox.get('conf', 0.8),
        })
    for bbox in (other_bboxes or []):
        if not bbox.get('points'):
            continue
        pts = np.array(bbox['points'], dtype=np.float32).reshape(-1, 1, 2)
        pts_out = cv2.perspectiveTransform(pts, warp_3x3)
        entry: Dict[str, Any] = {
            'type': bbox.get('type', 'unknown'),
            'points': pts_out.reshape(-1, 2).tolist(),
            'conf': bbox.get('conf', 0.8),
        }
        if 'text' in bbox:
            entry['text'] = bbox['text']
        if 'annotation_index' in bbox:
            entry['annotation_index'] = bbox['annotation_index']
        transformed.append(entry)

    # Map phase-corr response to a 0-1 confidence comparable to inlier ratios.
    confidence = float(min(max(response * _CONF_GAIN, 0.0), 1.0))

    total_ms = (_time.perf_counter() - t_total) * 1000
    logger.info(
        f"[ShapeOutline][{serial}] phaseCorr OK in {total_ms:.1f}ms — "
        f"shift=({dx:+.2f},{dy:+.2f}) response={response:.4f} "
        f"conf={confidence:.3f} scale=({scale_x:.3f},{scale_y:.3f}) "
        f"(phasecorr={t_phase_ms:.1f}ms)"
    )

    try:
        vis_target = (
            cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR)
            if target_img.ndim == 2 else target_img.copy()
        )
        for bbox in transformed:
            pts = np.array(bbox['points'], dtype=np.int32)
            color = (0, 255, 255) if bbox.get('type') == 'template' else (0, 255, 0)
            cv2.polylines(vis_target, [pts], True, color, 2)
            if bbox.get('text'):
                p0 = tuple(pts[0])
                cv2.putText(vis_target, str(bbox['text']), p0,
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        _save_debug(vis_target, "5_target_with_bboxes", serial)
    except Exception as e:
        logger.warning(f"shape_outline bbox-overlay save failed: {e}")

    return {
        'success': True,
        'homography': warp_3x3,
        'confidence': confidence,
        # `inliers` / `total_matches` aren't meaningful for phase correlation;
        # keep the keys with 0 so downstream code that reads them still works.
        'inliers': 0,
        'total_matches': 0,
        'transformed_bboxes': transformed,
        'target_img': target_img,
        'timings': {
            'method': 'shape_outline_phasecorr',
            'total': total_ms,
            'phasecorr': t_phase_ms,
        },
    }
