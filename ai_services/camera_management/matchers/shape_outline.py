"""
Shape outline matcher — gradient-based alternative to SuperPoint for cases
where template and target are similar in size and orientation (e.g. cap-OCR
mode after cap rotation + cap-crop).

Core idea
─────────
1. Compute Sobel magnitude on both template and target. The magnitude image
   "lights up" edges/contours, robust to absolute pixel intensity changes
   (lighting, ink darkness variation).
2. Run `cv2.findTransformECC` with MOTION_AFFINE — gradient-based image
   alignment that returns a 2×3 affine matrix mapping template → target.
3. Apply the affine to template annotation bboxes → transformed_bboxes.

When to use
───────────
- Cap-OCR mode (Check_Color + no product annotation): cap already rotated
  upright via OBB/CV → orientation residual is tiny → ECC converges fast.
- Cap-crop active (cap_crop_method!='none'): both template and target are
  tight cap-squares of similar size → ideal ECC scenario.

Performance: ~20-40ms on ~500×500 grayscale, vs ~95ms for SuperPoint TRT.

Limitations
───────────
- Won't handle large viewpoint changes (perspective). Affine only.
- If template & target are very different sizes, accuracy drops.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


logger = logging.getLogger(__name__)


# Tunables — empirically fine for ~500×500 cap images
_ECC_NUM_ITER = 30
_ECC_EPS = 1e-4
_SOBEL_KSIZE = 3
_GAUSS_KSIZE = 5
_DOWNSAMPLE_TO = 240  # max dim before ECC (keeps cost <30ms)

# Debug image dump — overwrite each call so user can inspect latest match
_DEBUG_DIR = Path("debug_shape")
_DEBUG_ENABLED = True  # toggle to False to silence


def _save_debug(img: np.ndarray, name: str, serial: str = "x") -> None:
    if not _DEBUG_ENABLED or img is None or img.size == 0:
        return
    try:
        _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        if img.dtype != np.uint8:
            # normalize float images (gradient maps) → 0-255 for visualization
            mn, mx = float(img.min()), float(img.max())
            if mx > mn:
                vis = ((img - mn) / (mx - mn) * 255.0).astype(np.uint8)
            else:
                vis = img.astype(np.uint8)
        else:
            vis = img
        cv2.imwrite(str(_DEBUG_DIR / f"{serial}_{name}.jpg"), vis)
    except Exception as e:
        logger.warning(f"shape_outline debug save failed ({name}): {e}")


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        return img
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Sobel magnitude as float32 in [0, 1] (normalized for ECC)."""
    blurred = cv2.GaussianBlur(gray, (_GAUSS_KSIZE, _GAUSS_KSIZE), 0,
                                borderType=cv2.BORDER_REPLICATE)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=_SOBEL_KSIZE,
                    borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=_SOBEL_KSIZE,
                    borderType=cv2.BORDER_REPLICATE)
    mag = cv2.magnitude(gx, gy)
    # Normalize to [0,1] so ECC correlation is well-conditioned
    m_max = float(mag.max())
    if m_max > 1e-6:
        mag = mag / m_max
    return mag


def _downsample_for_ecc(gray: np.ndarray, max_dim: int = _DOWNSAMPLE_TO):
    """Downsample so the larger dimension == max_dim. Returns (img, scale)."""
    h, w = gray.shape[:2]
    long_side = max(h, w)
    if long_side <= max_dim:
        return gray, 1.0
    scale = max_dim / float(long_side)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def match_shape_outline(
    template_img: np.ndarray,
    target_img: np.ndarray,
    template_bbox: Optional[Dict[str, Any]] = None,
    other_bboxes: Optional[List[Dict[str, Any]]] = None,
    serial: str = "x",
) -> Dict[str, Any]:
    """
    Align template → target using ECC on Sobel magnitude, then transform bboxes.

    Args:
        template_img: Template image (BGR or grayscale).
        target_img:   Target image (BGR or grayscale). Should be similar size.
        template_bbox: Optional template bbox dict {'points': [[x,y]*4], ...}.
        other_bboxes:  Optional list of annotation bbox dicts.
        serial:       Camera serial for debug image naming.

    Returns:
        Result dict with same shape as SuperPoint match result:
            success, confidence, inliers, total_matches, transformed_bboxes,
            homography, target_img, timings.
    """
    import time as _time
    t_total = _time.perf_counter()

    if template_img is None or target_img is None:
        return _fail_result(target_img, "null input image")

    # Step 0: dump raw inputs
    _save_debug(template_img, "0_template_raw", serial)
    _save_debug(target_img, "0_target_raw", serial)

    # Step 1: gray + resize target to match template size (ECC requires same size)
    t0 = _time.perf_counter()
    t_gray = _to_gray(template_img)
    g_gray_full = _to_gray(target_img)

    th, tw = t_gray.shape[:2]
    g_h, g_w = g_gray_full.shape[:2]
    if abs(g_h - th) > 2 or abs(g_w - tw) > 2:
        g_gray = cv2.resize(g_gray_full, (tw, th), interpolation=cv2.INTER_AREA)
        resize_to_match = True
    else:
        g_gray = g_gray_full
        resize_to_match = False
    t_prep_a = (_time.perf_counter() - t0) * 1000

    _save_debug(t_gray, "1_template_gray", serial)
    _save_debug(g_gray, f"1_target_gray_resized_{int(resize_to_match)}", serial)

    # Step 2: downsample BOTH equally for faster ECC, then upscale matrix later
    t0 = _time.perf_counter()
    t_small, scale = _downsample_for_ecc(t_gray)
    if scale < 1.0:
        new_w = int(round(g_gray.shape[1] * scale))
        new_h = int(round(g_gray.shape[0] * scale))
        g_small = cv2.resize(g_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        g_small = g_gray
    t_prep_b = (_time.perf_counter() - t0) * 1000

    # Step 3: Sobel magnitude on downsampled images
    t0 = _time.perf_counter()
    t_grad = _gradient_magnitude(t_small)
    g_grad = _gradient_magnitude(g_small)
    t_grad_ms = (_time.perf_counter() - t0) * 1000

    _save_debug(t_small, "2_template_small", serial)
    _save_debug(g_small, "2_target_small", serial)
    _save_debug(t_grad, "3_template_grad", serial)
    _save_debug(g_grad, "3_target_grad", serial)
    # Side-by-side visualization for quick comparison
    if t_grad.shape == g_grad.shape:
        side_by_side = np.hstack([t_grad, g_grad])
        _save_debug(side_by_side, "3_grad_side_by_side", serial)

    # Step 4: ECC affine alignment on gradient magnitude
    t0 = _time.perf_counter()
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                _ECC_NUM_ITER, _ECC_EPS)
    try:
        cc, warp = cv2.findTransformECC(
            templateImage=t_grad,
            inputImage=g_grad,
            warpMatrix=warp,
            motionType=cv2.MOTION_AFFINE,
            criteria=criteria,
        )
    except cv2.error as e:
        t_ecc_ms = (_time.perf_counter() - t0) * 1000
        logger.warning(
            f"[ShapeOutline][{serial}] ECC failed: {e} "
            f"(template={t_grad.shape}, target={g_grad.shape}) "
            f"— debug images saved in {_DEBUG_DIR}/"
        )
        return _fail_result(target_img, f"ECC failed: {e}", t_total, t_grad_ms, t_ecc_ms)
    t_ecc_ms = (_time.perf_counter() - t0) * 1000

    # Step 4b: visualize ECC result by warping target_small back to template_small
    # and overlaying. Good warp → overlay should align with template.
    try:
        target_warped_small = cv2.warpAffine(
            g_small, warp, (t_small.shape[1], t_small.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )
        _save_debug(target_warped_small, "4_target_warped_small", serial)
        # RGB overlay: template=R, target_warped=G — good alignment = yellow
        if t_small.shape == target_warped_small.shape:
            overlay = np.zeros((*t_small.shape, 3), dtype=np.uint8)
            overlay[..., 2] = t_small        # R = template
            overlay[..., 1] = target_warped_small  # G = warped target
            _save_debug(overlay, "4_overlay_RG", serial)
    except Exception as e:
        logger.warning(f"shape_outline overlay save failed: {e}")

    # Step 5: unscale the affine matrix back to original template coords
    # (warp maps coordinates in *downsampled* template → downsampled target;
    # we want template-full → target-full coords)
    if scale < 1.0:
        S = np.array([[scale, 0, 0],
                       [0, scale, 0]], dtype=np.float32)
        S_inv = np.array([[1.0/scale, 0, 0],
                           [0, 1.0/scale, 0]], dtype=np.float32)
        # convert 2x3 → 3x3, then warp_full = S_inv·warp·S in homogeneous form
        warp_3x3 = np.eye(3, dtype=np.float32); warp_3x3[:2] = warp
        S_3x3 = np.eye(3, dtype=np.float32); S_3x3[:2] = S
        S_inv_3x3 = np.eye(3, dtype=np.float32); S_inv_3x3[:2] = S_inv
        warp_full_3x3 = S_inv_3x3 @ warp_3x3 @ S_3x3
        warp_full = warp_full_3x3[:2]
    else:
        warp_full = warp
        warp_full_3x3 = np.eye(3, dtype=np.float32); warp_full_3x3[:2] = warp

    # Step 6: if target was resized to match template, rescale x/y back to
    # original target-image coords
    if resize_to_match:
        sx = g_w / float(tw)
        sy = g_h / float(th)
        # warp maps template_coords → target_coords (in template-sized space);
        # to get to original target coords, multiply x,y by (sx, sy)
        S_back = np.eye(3, dtype=np.float32)
        S_back[0, 0] = sx
        S_back[1, 1] = sy
        warp_full_3x3 = S_back @ warp_full_3x3

    # Step 7: transform bboxes
    t0 = _time.perf_counter()
    transformed: List[Dict[str, Any]] = []
    if template_bbox and template_bbox.get('points'):
        pts = np.array(template_bbox['points'], dtype=np.float32).reshape(-1, 1, 2)
        pts_out = cv2.perspectiveTransform(pts, warp_full_3x3)
        transformed.append({
            'type': 'template',
            'points': pts_out.reshape(-1, 2).tolist(),
            'conf': template_bbox.get('conf', 0.8),
        })
    for bbox in (other_bboxes or []):
        if not bbox.get('points'):
            continue
        pts = np.array(bbox['points'], dtype=np.float32).reshape(-1, 1, 2)
        pts_out = cv2.perspectiveTransform(pts, warp_full_3x3)
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
    t_bbox_ms = (_time.perf_counter() - t0) * 1000

    total_ms = (_time.perf_counter() - t_total) * 1000
    logger.info(
        f"[ShapeOutline][{serial}] OK in {total_ms:.1f}ms — ecc_cc={cc:.3f} "
        f"(prep={t_prep_a+t_prep_b:.1f}ms, grad={t_grad_ms:.1f}ms, "
        f"ecc={t_ecc_ms:.1f}ms, bbox={t_bbox_ms:.1f}ms, scale={scale:.2f}) "
        f"warp={warp.flatten().tolist()}"
    )

    # Step 8: draw warped bboxes on a copy of the target for visual sanity check
    try:
        if target_img.ndim == 2:
            vis_target = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR)
        else:
            vis_target = target_img.copy()
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
        'homography': warp_full_3x3,
        'confidence': float(cc),  # ECC correlation coefficient in [-1, 1]
        'inliers': int(t_grad.size),  # synthetic; ECC has no inlier concept
        'total_matches': int(t_grad.size),
        'transformed_bboxes': transformed,
        'target_img': target_img,
        'timings': {
            'method': 'shape_outline',
            'total': total_ms,
            'gradient': t_grad_ms,
            'ecc': t_ecc_ms,
        },
    }


def _fail_result(target_img, err: str,
                  t_start: Optional[float] = None,
                  t_grad: float = 0.0,
                  t_ecc: float = 0.0) -> Dict[str, Any]:
    import time as _time
    total = (_time.perf_counter() - t_start) * 1000 if t_start else 0.0
    return {
        'success': False,
        'error': err,
        'homography': None,
        'confidence': 0.0,
        'inliers': 0,
        'total_matches': 0,
        'transformed_bboxes': [],
        'target_img': target_img,
        'timings': {
            'method': 'shape_outline',
            'total': total,
            'gradient': t_grad,
            'ecc': t_ecc,
        },
    }
