"""
Image Processing-based product (bottle) edge detection.

NOTE: Trong UI / recipe option này gọi là "yolo_segment" cho consistency với
naming convention, NHƯNG thực chất KHÔNG dùng model YOLO. Đây là phương pháp
xử lý ảnh thuần (Sobel + outer-anchored peak detection) để tìm cạnh chai.

Pipeline:
1. Lấy label polygon từ transformed_bboxes (SuperPoint matching) — đã có sẵn
2. Build rotated search strip 2 bên label (outward)
3. Fill label đen middle 60% theo Y (top/bot 20% giữ để bắt cạnh khi label che)
4. Specular suppression: zero-out Sobel ở blob sáng lớn (>230, blob >5x5px)
5. Sobel X → 1D profile + per-column height_ratio (2-pass: global + P95 robust)
6. OUTER wall = peak xa nhất với height_ratio >= 0.55
7. INNER wall = peak gần label nhất, height_ratio >= 0.20, trước outer
8. Predict inner từ outer - template_plastic_thickness
9. Output: product_box ở format YOLO OBB-compatible {box, score, class, corners}

Returns None nếu không detect được walls (frame nhiễu/extreme).

Refs:
- test_alignment.py: pipeline gốc với debug visualization
- tools/annotator/detect_v3.py: outer-anchored algorithm
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)


# ─── Tuned defaults (xem test_alignment.py để tweak) ─────────────────────────
OUTER_SEARCH_MAX  = 150
EDGE_MARGIN       = 2
Y_EXTENSION       = 0.2
FILL_KEEP_RATIO   = 0.6
PEAK_HEIGHT       = 0.05
PEAK_PROM         = 0.02
PEAK_DIST         = 4
STRONG_THR        = 0.15
OUTER_MIN_HRATIO  = 0.55
INNER_MIN_HRATIO  = 0.20
INNER_TOL_PX      = 12
SPECULAR_THR      = 230


# ─── Geometric helpers ───────────────────────────────────────────────────────
def fill_label_black(img_bgr: np.ndarray, label_pts, keep_ratio: float = FILL_KEEP_RATIO) -> np.ndarray:
    """Fill middle keep_ratio% theo Y label (top/bot (1-keep)/2 giữ pixel gốc)."""
    pts = np.asarray(label_pts, dtype=np.float32)
    TL, TR, BR, BL = pts[0], pts[1], pts[2], pts[3]
    y_skip = (1.0 - keep_ratio) / 2
    new_TL = TL + y_skip * (BL - TL)
    new_BL = BL - y_skip * (BL - TL)
    new_TR = TR + y_skip * (BR - TR)
    new_BR = BR - y_skip * (BR - TR)
    inner_quad = np.array([new_TL, new_TR, new_BR, new_BL], dtype=np.int32)
    out = img_bgr.copy()
    cv2.fillPoly(out, [inner_quad.reshape(-1, 1, 2)], (0, 0, 0))
    return out


def _compute_strip_profile(
    gray: np.ndarray, label_pts, side: str,
    search_w: float = OUTER_SEARCH_MAX
) -> Optional[Dict[str, Any]]:
    pts = np.asarray(label_pts, dtype=np.float32)
    if side == 'left':
        p_top, p_bot = pts[0], pts[3]
    else:
        p_top, p_bot = pts[1], pts[2]

    edge = p_bot - p_top
    elen = float(np.linalg.norm(edge)) or 1.0
    edge_dir = edge / elen
    if side == 'left':
        perp = np.array([-edge_dir[1], edge_dir[0]], dtype=np.float32)
    else:
        perp = np.array([edge_dir[1], -edge_dir[0]], dtype=np.float32)

    y_ext = Y_EXTENSION * elen
    p_top_ext = p_top - y_ext * edge_dir
    p_bot_ext = p_bot + y_ext * edge_dir
    h = int(round(elen * (1.0 + 2 * Y_EXTENSION)))
    w = int(search_w - EDGE_MARGIN)
    if h < 50 or w < 30:
        return None

    inner_top = p_top_ext + EDGE_MARGIN * perp
    outer_top = p_top_ext + search_w   * perp
    outer_bot = p_bot_ext + search_w   * perp
    inner_bot = p_bot_ext + EDGE_MARGIN * perp
    src = np.array([inner_top, outer_top, outer_bot, inner_bot], dtype=np.float32)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    strip = cv2.warpPerspective(
        gray, M, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )

    sobel = cv2.Scharr(strip.astype(np.float32), cv2.CV_32F, 1, 0)
    abs_sobel = np.abs(sobel)

    # Specular suppression: zero-out Sobel near LARGE bright blobs only
    bright_mask = (strip > SPECULAR_THR).astype(np.uint8)
    if bright_mask.sum() > 100:
        bright_blobs = cv2.erode(
            bright_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        )
        if bright_blobs.sum() > 50:
            bright_dilated = cv2.dilate(
                bright_blobs, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            )
            abs_sobel = abs_sobel * (1 - bright_dilated)

    gmax = float(abs_sobel.max()) + 1e-6
    strong = (abs_sobel > STRONG_THR * gmax).sum(axis=0)
    height_ratio = strong / float(h)
    robust_max = float(np.percentile(abs_sobel, 95)) + 1e-6
    strong_robust = (abs_sobel > STRONG_THR * robust_max).sum(axis=0)
    height_ratio_robust = np.minimum(strong_robust / float(h), 1.0)

    profile = abs_sobel.sum(axis=0)
    profile = gaussian_filter1d(profile, sigma=1.5)
    pmax = profile.max() + 1e-6
    profile_n = profile / pmax

    peaks, _ = find_peaks(
        profile_n, height=PEAK_HEIGHT, distance=PEAK_DIST, prominence=PEAK_PROM
    )

    return {
        'profile': profile_n,
        'peaks': peaks,
        'height_ratio': height_ratio,
        'height_ratio_robust': height_ratio_robust,
        'p_top': p_top, 'p_bot': p_bot,
        'perp': perp,
        'edge_dir': edge_dir,
        'edge_len': elen,
    }


def _find_outer_inner(pd: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """Return (inner_gap, outer_gap) — px outward từ mép label."""
    if pd is None:
        return None, None
    peaks = pd['peaks']
    hr_global = pd['height_ratio']
    hr_robust = pd['height_ratio_robust']

    # Pass 1: global threshold (chính xác cho frame clean)
    outer_q = [int(p) for p in peaks if hr_global[p] >= OUTER_MIN_HRATIO]
    if not outer_q:
        # Pass 2: robust (P95) — frame có specular/window light mạnh
        outer_q = [int(p) for p in peaks if hr_robust[p] >= OUTER_MIN_HRATIO]
    outer = (max(outer_q) + EDGE_MARGIN) if outer_q else None

    hr_max = np.maximum(hr_global, hr_robust)
    inner_q = [int(p) for p in peaks if hr_max[p] >= INNER_MIN_HRATIO]
    inner = None
    if outer is not None:
        cands = [p for p in inner_q if (p + EDGE_MARGIN) < (outer - 4)]
        inner = (min(cands) + EDGE_MARGIN) if cands else None
    elif inner_q:
        inner = min(inner_q) + EDGE_MARGIN
    return inner, outer


def _find_inner_near(pd: Dict[str, Any], predicted_gap: float, tol: int = INNER_TOL_PX) -> Optional[int]:
    if pd is None:
        return None
    peaks = pd['peaks']
    hr_global = pd['height_ratio']
    hr_robust = pd['height_ratio_robust']
    hr_max = np.maximum(hr_global, hr_robust)
    cands = []
    for p in peaks:
        gap = int(p) + EDGE_MARGIN
        if hr_max[p] >= INNER_MIN_HRATIO and abs(gap - predicted_gap) <= tol:
            cands.append((gap, abs(gap - predicted_gap)))
    if not cands:
        return None
    cands.sort(key=lambda x: x[1])
    return cands[0][0]


def _wall_point_in_frame(label_pts, side: str, gap: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (top, bot) of wall line in frame coords."""
    pts = np.asarray(label_pts, dtype=np.float32)
    if side == 'left':
        p_top, p_bot = pts[0], pts[3]
    else:
        p_top, p_bot = pts[1], pts[2]
    edge = p_bot - p_top
    elen = float(np.linalg.norm(edge)) or 1.0
    edge_dir = edge / elen
    if side == 'left':
        perp = np.array([-edge_dir[1], edge_dir[0]], dtype=np.float32)
    else:
        perp = np.array([edge_dir[1], -edge_dir[0]], dtype=np.float32)
    return p_top + gap * perp, p_bot + gap * perp


# ─── Public API ──────────────────────────────────────────────────────────────
def detect_template_walls(template_bgr: np.ndarray, label_pts) -> Optional[Dict[str, Any]]:
    """Detect template's inner/outer walls. Gọi 1 lần khi init matcher.

    Returns: {inner_L, inner_R, outer_L, outer_R, plastic_L, plastic_R}
    None nếu detect fail.
    """
    masked = fill_label_black(template_bgr, label_pts)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    pd_L = _compute_strip_profile(gray, label_pts, 'left')
    pd_R = _compute_strip_profile(gray, label_pts, 'right')
    inner_L, outer_L = _find_outer_inner(pd_L)
    inner_R, outer_R = _find_outer_inner(pd_R)
    if any(v is None for v in [inner_L, inner_R, outer_L, outer_R]):
        logger.warning(
            f"Template walls incomplete: inner_L={inner_L} inner_R={inner_R} "
            f"outer_L={outer_L} outer_R={outer_R}"
        )
        return None
    return {
        'inner_L': int(inner_L), 'inner_R': int(inner_R),
        'outer_L': int(outer_L), 'outer_R': int(outer_R),
        'plastic_L': int(outer_L - inner_L),
        'plastic_R': int(outer_R - inner_R),
    }


def detect_product_box(
    frame_img: np.ndarray,
    label_pts: List[List[float]],
    template_walls: Dict[str, Any],
    serial_number: str = "",
) -> Optional[Dict[str, Any]]:
    """Detect product (bottle) box từ frame bằng image processing.

    Args:
        frame_img: BGR image
        label_pts: [TL, TR, BR, BL] of label polygon trong frame coords (from SuperPoint)
        template_walls: dict từ detect_template_walls()
        serial_number: chỉ cho logging

    Returns: dict YOLO OBB-compatible:
        {
            'box': np.array([cx, cy, w, h, angle]),
            'score': float,
            'class': 'product',
            'corners': np.array([[x,y]×4]),  # ← _check_center_alignment dùng cái này
            'source': 'image_proc',
            'detection_info': {...}   # debug
        }
    None nếu detect fail (caller nên skip frame hoặc fallback).
    """
    if frame_img is None or label_pts is None or template_walls is None:
        return None

    # 1) Fill label đen + extract grayscale
    masked = fill_label_black(frame_img, label_pts)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    # 2) Compute profile + peaks 2 bên
    pd_L = _compute_strip_profile(gray, label_pts, 'left')
    pd_R = _compute_strip_profile(gray, label_pts, 'right')

    # 3) Detect outer wall mỗi bên (height_ratio strict)
    outer_L = _find_outer_inner(pd_L)[1]
    outer_R = _find_outer_inner(pd_R)[1]

    # 4) Predict inner từ outer - plastic; nếu outer fail thì fallback tìm
    #    inner near template với tol rộng hơn
    def _resolve(pd, outer, tmpl_inner, plastic):
        if outer is not None:
            pred = outer - plastic
            if pred < 0:
                return None, pred, True
            inner = _find_inner_near(pd, max(pred, EDGE_MARGIN))
            return inner, pred, False
        # Fallback
        pred = tmpl_inner
        inner = _find_inner_near(pd, max(pred, EDGE_MARGIN), tol=INNER_TOL_PX + 8)
        if inner is None:
            return None, pred, True
        return inner, pred, False

    inner_L, pred_L, hidden_L = _resolve(
        pd_L, outer_L, template_walls['inner_L'], template_walls['plastic_L']
    )
    inner_R, pred_R, hidden_R = _resolve(
        pd_R, outer_R, template_walls['inner_R'], template_walls['plastic_R']
    )

    # 5) Effective gaps cho QC
    eff_L = inner_L if inner_L is not None else (pred_L if not hidden_L else -1)
    eff_R = inner_R if inner_R is not None else (pred_R if not hidden_R else -1)

    # Nếu cả 2 hidden → không detect được → return None
    if hidden_L and hidden_R:
        logger.warning(f"[{serial_number}] image_proc: both walls hidden")
        return None

    # 6) Build product box corners từ OUTER wall positions
    #    Nếu outer_L/R None thì dùng inner + plastic ngược lại
    out_L_gap = outer_L if outer_L is not None else (eff_L + template_walls['plastic_L'])
    out_R_gap = outer_R if outer_R is not None else (eff_R + template_walls['plastic_R'])

    # Get 4 corners trong frame: TL/TR/BR/BL của bottle
    L_top, L_bot = _wall_point_in_frame(label_pts, 'left',  out_L_gap)
    R_top, R_bot = _wall_point_in_frame(label_pts, 'right', out_R_gap)
    corners = np.array([L_top, R_top, R_bot, L_bot], dtype=np.float32)  # TL, TR, BR, BL

    # 7) Box [cx, cy, w, h, angle] cho compatibility với YOLO OBB format
    cx = float(corners[:, 0].mean())
    cy = float(corners[:, 1].mean())
    w  = float(np.linalg.norm((R_top + R_bot)/2 - (L_top + L_bot)/2))   # perp width
    h  = float(np.linalg.norm(L_bot - L_top))                           # along-edge height
    # Angle: angle of label vertical edge (top→bot) so với trục Y
    edge = L_bot - L_top
    angle = float(np.arctan2(edge[0], edge[1]))  # 0 = vertical, +rad = nghiêng phải
    box = np.array([cx, cy, w, h, angle], dtype=np.float32)

    return {
        'box': box,
        'score': 1.0,
        'class': 'product',
        'corners': corners,
        'source': 'image_proc',
        'detection_info': {
            'inner_L': inner_L, 'inner_R': inner_R,
            'outer_L': outer_L, 'outer_R': outer_R,
            'eff_L': eff_L, 'eff_R': eff_R,
            'pred_inner_L': pred_L, 'pred_inner_R': pred_R,
            'hidden_L': hidden_L, 'hidden_R': hidden_R,
        }
    }


def label_box_from_pts(label_pts: List[List[float]]) -> Dict[str, Any]:
    """Build a label_box dict in YOLO OBB-compatible format từ SuperPoint label polygon.
    Dùng khi yolo_segment mode (không qua YOLO).
    """
    corners = np.asarray(label_pts, dtype=np.float32)
    cx = float(corners[:, 0].mean())
    cy = float(corners[:, 1].mean())
    # OBB w/h: dùng cạnh thực (perp + along)
    edge_left  = corners[3] - corners[0]   # TL→BL
    edge_top   = corners[1] - corners[0]   # TL→TR
    h = float(np.linalg.norm(edge_left))
    w = float(np.linalg.norm(edge_top))
    angle = float(np.arctan2(edge_left[0], edge_left[1]))
    return {
        'box': np.array([cx, cy, w, h, angle], dtype=np.float32),
        'score': 1.0,
        'class': 'label',
        'corners': corners,
        'source': 'superpoint',
    }
