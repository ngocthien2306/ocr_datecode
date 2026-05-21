"""
Shape outline matcher — contour-based alternative to SuperPoint for cases
where template and target are similar in size and orientation (e.g. cap-OCR
mode after cap rotation + cap-crop).

Algorithm (contour-based, primary path)
───────────────────────────────────────
1. Canny edges on both template + target.
2. cv2.findContours → list of polygons.
3. Filter contours that look like glyph strokes:
     - bbox area in [_MIN_AREA, _MAX_AREA]
     - aspect ratio bounded
4. Get N largest contour CENTROIDS in each image (text characters).
5. Match centroids by nearest-neighbor with mutual-consistency check
   (template→target NN must agree with target→template NN).
6. cv2.estimateAffinePartial2D with RANSAC → robust 2×3 affine
   (translation + rotation + uniform scale).
7. Apply affine to template annotation bboxes → transformed_bboxes.

Why contour-based instead of ECC
────────────────────────────────
ECC is gradient-descent on intensity; if the background is shared and strong
(e.g. conveyor pattern) it can lock onto the background and ignore the cap.
Contour centroids are sparse anchors → RANSAC throws out background outliers
and finds the transform that aligns text characters specifically.

Cost
────
~30-60ms for 500×500 grayscale (Canny ~5ms, findContours ~3ms, RANSAC ~10ms).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


logger = logging.getLogger(__name__)


# Tunables — empirically fine for cap-OCR text glyphs
_CANNY_LOW = 50
_CANNY_HIGH = 150
_GAUSS_KSIZE = 3
_MIN_AREA = 10           # filter speckles (lowered — small glyphs after downsample)
_MAX_AREA = 15000        # filter giant blobs (cap silhouette is usually >50k)
_MAX_ASPECT = 8.0        # filter long horizontal/vertical edges
_TOP_N_CENTROIDS = 60    # match up to this many text-like contours
_RANSAC_THRESH = 3.0     # pixels in downsampled space
_DOWNSAMPLE_TO = 320     # downsample longer side before contour extraction

# Debug image dump
_DEBUG_DIR = Path("debug_shape")
_DEBUG_ENABLED = True


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


def _downsample(gray: np.ndarray, max_dim: int = _DOWNSAMPLE_TO) -> Tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    long_side = max(h, w)
    if long_side <= max_dim:
        return gray, 1.0
    scale = max_dim / float(long_side)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def _extract_text_centroids(
    gray: np.ndarray,
    serial: str = "x",
    tag: str = "tmpl",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract centroids of text-like contours.

    Returns
    -------
    centroids : (N, 2) float32 of (x, y)
    edge_map  : Canny image (for debug)
    """
    blurred = cv2.GaussianBlur(gray, (_GAUSS_KSIZE, _GAUSS_KSIZE), 0)
    edges = cv2.Canny(blurred, _CANNY_LOW, _CANNY_HIGH)
    _save_debug(edges, f"contour_1_{tag}_edges", serial)

    # RETR_LIST: get ALL contours regardless of hierarchy. RETR_EXTERNAL would
    # return ONLY the outermost contour — when the cap circle in the edge image
    # is a closed ring around all text, RETR_EXTERNAL excludes the text inside.
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    centroids: List[Tuple[float, float, float]] = []  # (cx, cy, area)
    # Track all bboxes (passed + filtered) for debug visualization
    all_bboxes: List[Tuple[int, int, int, int, bool]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        passed = True
        if area < _MIN_AREA or area > _MAX_AREA:
            passed = False
        else:
            ar = max(w, h) / float(min(w, h)) if min(w, h) > 0 else 99.0
            if ar > _MAX_ASPECT:
                passed = False
        all_bboxes.append((x, y, w, h, passed))
        if not passed:
            continue
        cx = x + w / 2.0
        cy = y + h / 2.0
        centroids.append((cx, cy, float(area)))

    # Pick largest N by area (text glyphs have moderate area; sort and clip)
    centroids.sort(key=lambda c: c[2], reverse=True)
    centroids = centroids[:_TOP_N_CENTROIDS]
    arr = np.array([[c[0], c[1]] for c in centroids], dtype=np.float32) if centroids else np.empty((0, 2), np.float32)

    logger.info(
        f"[ShapeOutline] {tag} contours raw={len(contours)} passed={len(centroids)} "
        f"(filter: area=[{_MIN_AREA}, {_MAX_AREA}], aspect<={_MAX_ASPECT})"
    )

    # Debug visualization — show ALL contour bboxes (red=filtered, green=passed)
    try:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for x, y, w, h, passed in all_bboxes:
            color = (0, 255, 0) if passed else (0, 0, 255)
            thick = 2 if passed else 1
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, thick)
        # Also draw centroids on top
        for cx, cy, _ in centroids:
            cv2.circle(vis, (int(cx), int(cy)), 3, (255, 255, 0), -1)
        _save_debug(vis, f"contour_2_{tag}_centroids", serial)
    except Exception:
        pass

    return arr, edges


def _match_centroids_nn(
    src: np.ndarray,
    dst: np.ndarray,
    max_dist: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mutual nearest-neighbor matching.

    For each src point, find nearest dst point. Keep only pairs where the
    reverse NN (dst→src) returns to the same src point. Filter pairs with
    distance > max_dist.
    """
    if len(src) == 0 or len(dst) == 0:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    # src→dst NN
    diffs = src[:, None, :] - dst[None, :, :]
    dists2 = (diffs * diffs).sum(axis=2)
    nn_dst_for_src = dists2.argmin(axis=1)

    # dst→src NN
    nn_src_for_dst = dists2.argmin(axis=0)

    matched_src: List[Tuple[float, float]] = []
    matched_dst: List[Tuple[float, float]] = []
    max_d2 = max_dist * max_dist
    for i in range(len(src)):
        j = int(nn_dst_for_src[i])
        if int(nn_src_for_dst[j]) != i:
            continue  # not mutual
        if dists2[i, j] > max_d2:
            continue
        matched_src.append((float(src[i, 0]), float(src[i, 1])))
        matched_dst.append((float(dst[j, 0]), float(dst[j, 1])))

    return (np.array(matched_src, dtype=np.float32),
            np.array(matched_dst, dtype=np.float32))


def match_shape_outline(
    template_img: np.ndarray,
    target_img: np.ndarray,
    template_bbox: Optional[Dict[str, Any]] = None,
    other_bboxes: Optional[List[Dict[str, Any]]] = None,
    serial: str = "x",
) -> Dict[str, Any]:
    """
    Match template ↔ target via text-contour centroid alignment, then transform
    bboxes. Returns result dict in same shape as SuperPoint match result.
    """
    import time as _time
    t_total = _time.perf_counter()

    if template_img is None or target_img is None:
        return _fail_result(target_img, "null input image", t_total)

    _save_debug(template_img, "0_template_raw", serial)
    _save_debug(target_img, "0_target_raw", serial)

    # Step 1: gray + downsample BOTH to similar resolution
    t_gray = _to_gray(template_img)
    g_gray = _to_gray(target_img)
    t_small, scale_t = _downsample(t_gray)
    g_small, scale_g = _downsample(g_gray)

    # Step 2: extract centroids from each
    t_ext = _time.perf_counter()
    tmpl_pts, _ = _extract_text_centroids(t_small, serial=serial, tag="tmpl")
    tgt_pts,  _ = _extract_text_centroids(g_small, serial=serial, tag="tgt")
    t_extract_ms = (_time.perf_counter() - t_ext) * 1000

    logger.info(
        f"[ShapeOutline][{serial}] centroids: tmpl={len(tmpl_pts)} "
        f"tgt={len(tgt_pts)} (scales: tmpl={scale_t:.2f}, tgt={scale_g:.2f})"
    )

    if len(tmpl_pts) < 4 or len(tgt_pts) < 4:
        logger.warning(
            f"[ShapeOutline][{serial}] not enough centroids "
            f"(tmpl={len(tmpl_pts)}, tgt={len(tgt_pts)}) — need ≥4 for RANSAC"
        )
        return _fail_result(target_img, "too few centroids", t_total)

    # Step 3: bring centroids to same coordinate space (max(W,H))
    # tmpl_pts are in t_small space; tgt_pts in g_small space.
    # We need to find affine that maps tmpl_full_coords → tgt_full_coords.
    # First convert both to full-image coords:
    tmpl_pts_full = tmpl_pts / scale_t
    tgt_pts_full  = tgt_pts  / scale_g

    # Step 4: nearest-neighbor matching with mutual-consistency
    # Use a generous max_dist (image diagonal / 3) since cap may have shifted
    H_full, W_full = t_gray.shape[:2]
    max_dist = float(max(H_full, W_full)) / 3.0
    src, dst = _match_centroids_nn(tmpl_pts_full, tgt_pts_full, max_dist)

    if len(src) < 4:
        # Fallback: relaxed (no mutual check), all pairs within max_dist
        logger.warning(
            f"[ShapeOutline][{serial}] mutual NN gave {len(src)} matches; "
            f"falling back to non-mutual NN"
        )
        # All src → nearest dst pairs:
        diffs = tmpl_pts_full[:, None, :] - tgt_pts_full[None, :, :]
        d2 = (diffs * diffs).sum(axis=2)
        nn = d2.argmin(axis=1)
        src = tmpl_pts_full
        dst = tgt_pts_full[nn]

    if len(src) < 4:
        logger.warning(f"[ShapeOutline][{serial}] still <4 matches; giving up")
        return _fail_result(target_img, "too few matched centroids", t_total)

    logger.info(f"[ShapeOutline][{serial}] matched pairs: {len(src)}")

    # Step 5: RANSAC affine (translation + rotation + uniform scale)
    t_ransac = _time.perf_counter()
    M, inliers_mask = cv2.estimateAffinePartial2D(
        src, dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=_RANSAC_THRESH,
        maxIters=500,
        confidence=0.99,
    )
    t_ransac_ms = (_time.perf_counter() - t_ransac) * 1000

    if M is None:
        logger.warning(f"[ShapeOutline][{serial}] estimateAffinePartial2D returned None")
        return _fail_result(target_img, "RANSAC failed", t_total)

    n_inliers = int(inliers_mask.sum()) if inliers_mask is not None else 0
    inlier_ratio = float(n_inliers) / float(len(src))
    logger.info(
        f"[ShapeOutline][{serial}] RANSAC inliers {n_inliers}/{len(src)} "
        f"({inlier_ratio*100:.1f}%) in {t_ransac_ms:.1f}ms"
    )

    # Step 6: transform bboxes via the affine (3x3 form for perspectiveTransform)
    warp_3x3 = np.eye(3, dtype=np.float32)
    warp_3x3[:2] = M.astype(np.float32)

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

    total_ms = (_time.perf_counter() - t_total) * 1000
    logger.info(
        f"[ShapeOutline][{serial}] OK in {total_ms:.1f}ms — inlier_ratio={inlier_ratio:.3f} "
        f"(extract={t_extract_ms:.1f}ms, ransac={t_ransac_ms:.1f}ms) "
        f"warp={M.flatten().tolist()}"
    )

    # Step 7: debug visualization — draw bboxes on target
    try:
        vis_target = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR) if target_img.ndim == 2 else target_img.copy()
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
        'confidence': inlier_ratio,
        'inliers': n_inliers,
        'total_matches': len(src),
        'transformed_bboxes': transformed,
        'target_img': target_img,
        'timings': {
            'method': 'shape_outline_contour',
            'total': total_ms,
            'extract': t_extract_ms,
            'ransac': t_ransac_ms,
        },
    }


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
        'timings': {'method': 'shape_outline_contour', 'total': total},
    }
