"""
Alignment detection v3 — Outer-anchored inner wall detection.

Pipeline:
1. Pre-compute template:
   - Detect OUTER wall (height_ratio >= 0.55) and INNER wall (closest peak) per side
   - Compute plastic thickness = outer - inner

2. Mỗi target frame:
   - Fill label đen (middle 60%, top/bot 20% giữ)
   - Build rotated search strip (extended Y-band 20%)
   - Compute Sobel X + height_ratio per column
   - Find OUTER wall (peak xa nhất với height >= 0.55)
   - Predict INNER = outer_target - plastic_template
   - Find INNER peak gần predicted (tol ±12)
   - Nếu predicted < 0 hoặc không có peak → wall hidden → defect

3. Defect: min(inner_L, inner_R) < TOUCH_MARGIN

Visualize 5 ảnh/frame trong align_vis/{serial}/{stem}/
+ Save FAIL frames riêng align_fails/{serial}/

Run: python3 test_alignment.py
"""

import cv2
import json
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────
CROP_DIR  = Path("/home/demo/Source/ocr_datecode/crop_samples_1")
OUT_DIR   = CROP_DIR / "align_vis"
FAIL_DIR  = CROP_DIR / "align_fails"
ANN_DIR   = CROP_DIR / "annotations"     # ground truth (optional, for overlay)
OUT_DIR.mkdir(exist_ok=True)
FAIL_DIR.mkdir(exist_ok=True)

CAMERAS  = ['40767171', '40733814']
MAX_IMGS = 100

# Search ROI
OUTER_SEARCH_MAX = 150     # px outward — đủ xa để bắt outer wall
EDGE_MARGIN      = 2       # skip vùng sát fill (smaller, để bắt defect sát label)
Y_EXTENSION      = 0.2     # extend search Y-band 20% above/below label
FILL_KEEP_RATIO  = 0.6     # fill black middle 60%, top/bot 20% giữ
SHRINK_PX        = 0       # shrink label polygon inward (0 = no shrink)

# Peak detection
PEAK_HEIGHT      = 0.05    # normalized profile threshold
PEAK_PROM        = 0.02
PEAK_DIST        = 4
STRONG_THR       = 0.15    # pixel > 15% global max = "strong edge"

# Wall classification
OUTER_MIN_HRATIO = 0.55    # outer wall: cạnh full chiều cao chai (silhouette)
INNER_MIN_HRATIO = 0.20    # inner wall: có thể bị label che 60-70%
INNER_TOL_PX     = 12      # inner phải nằm gần predicted ± TOL

# Defect criteria
TOUCH_MARGIN     = 5       # min(inner_L, inner_R) < margin → defect
SAVE_DEBUG       = True


# ─── Geometric helpers ───────────────────────────────────────────────────────
def label_perp_width(label_pts):
    p = np.asarray(label_pts, dtype=np.float32)
    return float(np.linalg.norm(((p[1]+p[2])/2 - (p[0]+p[3])/2)))


def shrink_label_polygon(label_pts, shrink_px=SHRINK_PX):
    if shrink_px == 0:
        return list(label_pts)
    pts = np.asarray(label_pts, dtype=np.float32)
    TL, TR, BR, BL = pts[0], pts[1], pts[2], pts[3]
    L_edge = BL - TL; L_len = float(np.linalg.norm(L_edge)) or 1.0
    L_inward = np.array([L_edge[1], -L_edge[0]], dtype=np.float32) / L_len
    R_edge = BR - TR; R_len = float(np.linalg.norm(R_edge)) or 1.0
    R_inward = np.array([-R_edge[1], R_edge[0]], dtype=np.float32) / R_len
    return [(TL + shrink_px*L_inward).tolist(),
            (TR + shrink_px*R_inward).tolist(),
            (BR + shrink_px*R_inward).tolist(),
            (BL + shrink_px*L_inward).tolist()]


def fill_label_black(img_bgr, label_pts, keep_ratio=FILL_KEEP_RATIO):
    """Fill MIDDLE keep_ratio% của label với màu đen (top + bot 20% giữ pixel gốc)."""
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


# ─── Detection core ──────────────────────────────────────────────────────────
def compute_strip_profile(gray, label_pts, side, search_w=OUTER_SEARCH_MAX):
    """Build rotated search strip → Sobel X → 1D profile + per-column height_ratio.

    Sử dụng PER-ROW ADAPTIVE THRESHOLD để tính height_ratio:
    - Window light/specular dominate global max → global threshold ăn cạnh chai yếu
    - Per-row threshold (mỗi row có max riêng) giữ được cạnh chai dù lighting cục bộ mạnh
    - Apply vertical morphology để loại spurious noise + connect bottle wall column
    """
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
        perp = np.array([ edge_dir[1], -edge_dir[0]], dtype=np.float32)

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
    src_quad = np.array([inner_top, outer_top, outer_bot, inner_bot], dtype=np.float32)
    dst_quad = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_quad, dst_quad)
    strip = cv2.warpPerspective(gray, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)

    sobel = cv2.Scharr(strip.astype(np.float32), cv2.CV_32F, 1, 0)
    abs_sobel = np.abs(sobel)

    # ── Specular suppression ─────────────────────────────────────────────────
    # Pixel quá sáng (>230) = specular/window light. Chỉ mask BLOB LỚN
    # (window reflection rectangular), KHÔNG mask thin lines (cạnh plastic
    # phản chiếu sáng — đó là cạnh chai thật).
    SPECULAR_THR = 230
    bright_mask = (strip > SPECULAR_THR).astype(np.uint8)
    if bright_mask.sum() > 100:
        # Erode để loại thin lines (< 4px wide), giữ blobs lớn
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bright_blobs = cv2.erode(bright_mask, kernel_erode)
        if bright_blobs.sum() > 50:
            # Dilate ngược lại để mask cover edges của blob
            bright_dilated = cv2.dilate(
                bright_blobs,
                cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
            abs_sobel = abs_sobel * (1 - bright_dilated)

    # ── Height_ratio: dùng global max như bình thường ─────────────────────────
    gmax = float(abs_sobel.max()) + 1e-6
    strong = (abs_sobel > STRONG_THR * gmax).sum(axis=0)
    height_ratio = strong / float(h)
    # Backup: với specular vẫn rớt qua (vd reflection rộng), thêm P95 robust
    robust_max = float(np.percentile(abs_sobel, 95)) + 1e-6
    strong_robust = (abs_sobel > STRONG_THR * robust_max).sum(axis=0)
    height_ratio_robust = np.minimum(strong_robust / float(h), 1.0)

    # ── Profile for peak detection (unweighted Sobel sum) ────────────────────
    profile = abs_sobel.sum(axis=0)
    profile = gaussian_filter1d(profile, sigma=1.5)
    pmax = profile.max() + 1e-6
    profile_n = profile / pmax

    peaks, _ = find_peaks(profile_n, height=PEAK_HEIGHT,
                          distance=PEAK_DIST, prominence=PEAK_PROM)

    return {
        'side': side,
        'profile': profile_n,
        'peaks': peaks,
        'height_ratio': height_ratio,
        'height_ratio_robust': height_ratio_robust,
        'src_quad': src_quad.tolist(),
        'p_top': p_top, 'p_bot': p_bot,
        'perp': perp,
        'edge_dir': edge_dir,
        'edge_len': elen,
        'w': w, 'h': h,
    }


def find_outer_inner(pd):
    """Từ profile data → trả về (inner_gap, outer_gap) bằng px từ mép label.
    2-pass: thử global height_ratio trước; nếu outer fail thì fallback robust (P95)."""
    if pd is None:
        return None, None
    peaks = pd['peaks']
    hr_global = pd['height_ratio']
    hr_robust = pd.get('height_ratio_robust', hr_global)

    # Pass 1: dùng global threshold (chặt, chính xác cho frame clean)
    outer_q = [int(p) for p in peaks if hr_global[p] >= OUTER_MIN_HRATIO]
    if not outer_q:
        # Pass 2: fallback robust (P95) — frame có specular/window light mạnh
        outer_q = [int(p) for p in peaks if hr_robust[p] >= OUTER_MIN_HRATIO]
    outer = (max(outer_q) + EDGE_MARGIN) if outer_q else None

    # INNER = peak GẦN LABEL NHẤT, trước outer ít nhất 4px, height >= INNER_MIN_HRATIO
    # Dùng max của 2 hr để inner detection lenient hơn
    hr_max = np.maximum(hr_global, hr_robust)
    inner_q = [int(p) for p in peaks if hr_max[p] >= INNER_MIN_HRATIO]
    inner = None
    if outer is not None:
        cands = [p for p in inner_q if (p + EDGE_MARGIN) < (outer - 4)]
        inner = (min(cands) + EDGE_MARGIN) if cands else None
    elif inner_q:
        inner = min(inner_q) + EDGE_MARGIN
    return inner, outer


def find_inner_near(pd, predicted_gap, tol=INNER_TOL_PX):
    """Find inner peak gần predicted_gap nhất, trong ±tol."""
    if pd is None: return None
    peaks = pd['peaks']; hr = pd['height_ratio']
    candidates = []
    for p in peaks:
        gap = int(p) + EDGE_MARGIN
        if hr[p] >= INNER_MIN_HRATIO and abs(gap - predicted_gap) <= tol:
            candidates.append((gap, abs(gap - predicted_gap)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


def detect_walls_outer_anchored(gray, label_pts, tmpl):
    """Full v3 detection với fallback: nếu outer không detect được,
    vẫn tìm inner near template gap (BG nhiễu, scale thay đổi không hẳn là defect).
    """
    pd_L = compute_strip_profile(gray, label_pts, 'left')
    pd_R = compute_strip_profile(gray, label_pts, 'right')

    outer_L = find_outer_inner(pd_L)[1]
    outer_R = find_outer_inner(pd_R)[1]

    plastic_L = tmpl['outer_L'] - tmpl['inner_L']
    plastic_R = tmpl['outer_R'] - tmpl['inner_R']

    def resolve_side(pd, outer, tmpl_inner, plastic):
        """Trả (inner, pred_inner, hidden, fallback_used)."""
        if outer is not None:
            pred_inner = outer - plastic
            if pred_inner < 0:
                return None, pred_inner, True, False
            inner = find_inner_near(pd, max(pred_inner, EDGE_MARGIN))
            return inner, pred_inner, False, False
        # Outer not found — FALLBACK: search inner near template gap with wider tol
        pred_inner = tmpl_inner
        inner = find_inner_near(pd, max(pred_inner, EDGE_MARGIN), tol=INNER_TOL_PX + 8)
        if inner is None:
            # Truly hidden: no outer AND no inner near template
            return None, pred_inner, True, True
        return inner, pred_inner, False, True

    inner_L, pred_L, hidden_L, fb_L = resolve_side(pd_L, outer_L, tmpl['inner_L'], plastic_L)
    inner_R, pred_R, hidden_R, fb_R = resolve_side(pd_R, outer_R, tmpl['inner_R'], plastic_R)

    eff_L = inner_L if inner_L is not None else (pred_L if not hidden_L else -1)
    eff_R = inner_R if inner_R is not None else (pred_R if not hidden_R else -1)

    return {
        'inner_L': inner_L, 'inner_R': inner_R,
        'outer_L': outer_L, 'outer_R': outer_R,
        'pred_inner_L': pred_L, 'pred_inner_R': pred_R,
        'hidden_L': hidden_L, 'hidden_R': hidden_R,
        'fallback_L': fb_L, 'fallback_R': fb_R,
        'eff_L': eff_L, 'eff_R': eff_R,
        'profile_L': pd_L, 'profile_R': pd_R,
        'plastic_L': plastic_L, 'plastic_R': plastic_R,
    }


# ─── Geometric: gap → frame coords for visualization ─────────────────────────
def wall_line_in_frame(label_pts, side, gap_perp):
    """Return ((x_top, y_top), (x_bot, y_bot)) of wall line in frame."""
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
        perp = np.array([ edge_dir[1], -edge_dir[0]], dtype=np.float32)
    return p_top + gap_perp * perp, p_bot + gap_perp * perp


# ─── Visualization helpers ───────────────────────────────────────────────────
def draw_search_rois(img_bgr, pd_L, pd_R):
    out = img_bgr.copy()
    for pd, color, lbl in [(pd_L, (0, 255, 255), 'L search'),
                            (pd_R, (0, 255, 255), 'R search')]:
        if pd is None: continue
        quad = np.array(pd['src_quad'], dtype=np.int32)
        cv2.polylines(out, [quad], True, color, 2)
        cv2.putText(out, lbl, tuple(quad[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def plot_profile_with_walls(pd, picked_inner_gap, picked_outer_gap, pred_inner_gap,
                              canvas_h=240, scale=2):
    """1D profile + height_ratio overlay + peaks marked.
    Picked inner (lime), outer (orange), predicted inner (yellow dashed).
    """
    if pd is None or pd['profile'] is None or len(pd['profile']) == 0:
        c = np.full((canvas_h, 400, 3), 40, dtype=np.uint8)
        cv2.putText(c, "no profile", (10, canvas_h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        return c

    profile = pd['profile']
    hr = pd['height_ratio']
    peaks = pd['peaks']
    n = len(profile)
    canvas_w = max(300, n * scale)
    canvas = np.full((canvas_h, canvas_w, 3), 40, dtype=np.uint8)

    baseline_y = canvas_h - 25
    top_y      = 20
    plot_h     = baseline_y - top_y

    # Height ratio band (blue background where >= INNER_MIN_HRATIO, green where >= OUTER_MIN_HRATIO)
    for i in range(n):
        x = i * scale
        if hr[i] >= OUTER_MIN_HRATIO:
            col = (60, 90, 30)   # dark green
        elif hr[i] >= INNER_MIN_HRATIO:
            col = (60, 60, 30)   # dark teal
        else:
            continue
        cv2.line(canvas, (x, top_y), (x+scale, top_y),
                 col, 6)  # top stripe shows height_ratio category

    # Threshold lines
    th_y = baseline_y - int(PEAK_HEIGHT * plot_h)
    cv2.line(canvas, (0, th_y), (canvas_w, th_y), (80, 80, 180), 1)

    # Profile polyline
    for i in range(n - 1):
        x1, x2 = i * scale, (i+1) * scale
        y1 = baseline_y - int(profile[i] * plot_h)
        y2 = baseline_y - int(profile[i+1] * plot_h)
        cv2.line(canvas, (x1, y1), (x2, y2), (220, 220, 220), 1)

    # Mark all peaks (small orange dots)
    for p in peaks:
        x = int(p) * scale
        y = baseline_y - int(profile[p] * plot_h)
        cv2.circle(canvas, (x, y), 3, (100, 100, 200), -1)

    # Mark PICKED OUTER (orange large)
    if picked_outer_gap is not None:
        p = picked_outer_gap - EDGE_MARGIN
        if 0 <= p < n:
            x = int(p) * scale
            y = baseline_y - int(profile[p] * plot_h)
            cv2.circle(canvas, (x, y), 7, (0, 165, 255), -1)
            cv2.putText(canvas, f"OUT={picked_outer_gap}", (x+8, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    # Mark PICKED INNER (lime large)
    if picked_inner_gap is not None:
        p = picked_inner_gap - EDGE_MARGIN
        if 0 <= p < n:
            x = int(p) * scale
            y = baseline_y - int(profile[p] * plot_h)
            cv2.circle(canvas, (x, y), 7, (0, 255, 0), -1)
            cv2.putText(canvas, f"IN={picked_inner_gap}", (x+8, y+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Mark PREDICTED INNER (yellow vertical line)
    if pred_inner_gap is not None:
        p_pred = pred_inner_gap - EDGE_MARGIN
        if 0 <= p_pred < n:
            x = int(p_pred) * scale
            cv2.line(canvas, (x, top_y), (x, baseline_y), (0, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"pred={pred_inner_gap:.0f}", (x+2, baseline_y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 220), 1)

    side = pd['side'].upper()
    cv2.putText(canvas, f"{side} profile  (outer thresh={OUTER_MIN_HRATIO}, inner={INNER_MIN_HRATIO})",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)
    cv2.line(canvas, (0, baseline_y), (canvas_w, baseline_y), (100,100,100), 1)
    return canvas


def stack_profiles(det, pred_in_L, pred_in_R):
    p_L = plot_profile_with_walls(det['profile_L'],
                                    det['inner_L'], det['outer_L'], pred_in_L)
    p_R = plot_profile_with_walls(det['profile_R'],
                                    det['inner_R'], det['outer_R'], pred_in_R)
    h = max(p_L.shape[0], p_R.shape[0])
    sep = np.full((h, 4, 3), (60,60,60), dtype=np.uint8)
    return np.hstack([p_L, sep, p_R])


def draw_final(full, label_pts, det, tmpl, gt_bottle_pts=None):
    """Final visualization: label + outer (orange) + inner detected (green) + predicted (yellow dashed)."""
    vis = full.copy()

    # Label polygon (cyan)
    cv2.polylines(vis, [np.array(label_pts, dtype=np.int32)],
                  True, (220, 220, 0), 2)

    # GT bottle (red dashed, if provided)
    if gt_bottle_pts is not None:
        cv2.polylines(vis, [np.array(gt_bottle_pts, dtype=np.int32)],
                      True, (0, 0, 200), 1)

    # OUTER wall (orange)
    for side, gap in [('left', det['outer_L']), ('right', det['outer_R'])]:
        if gap is not None:
            t, b = wall_line_in_frame(label_pts, side, gap)
            cv2.line(vis, (int(t[0]), int(t[1])),
                          (int(b[0]), int(b[1])), (0, 165, 255), 2)

    # PREDICTED inner (yellow dashed-ish)
    for side, gap in [('left', det['pred_inner_L']), ('right', det['pred_inner_R'])]:
        if gap is not None and gap > 0:
            t, b = wall_line_in_frame(label_pts, side, gap)
            # Draw dashed line
            pts = np.linspace([t[0], t[1]], [b[0], b[1]], 30)
            for i in range(0, len(pts) - 1, 2):
                p1 = pts[i]; p2 = pts[i+1]
                cv2.line(vis, (int(p1[0]), int(p1[1])),
                              (int(p2[0]), int(p2[1])), (0, 220, 220), 1)

    # INNER wall detected (green thick)
    for side, gap in [('left', det['inner_L']), ('right', det['inner_R'])]:
        if gap is not None:
            t, b = wall_line_in_frame(label_pts, side, gap)
            cv2.line(vis, (int(t[0]), int(t[1])),
                          (int(b[0]), int(b[1])), (0, 255, 0), 3)

    # Eff (used for decision) - if no inner detected but predicted available
    for side, eff, inner in [('left', det['eff_L'], det['inner_L']),
                              ('right', det['eff_R'], det['inner_R'])]:
        if inner is None and eff is not None and eff != -1 and eff > 0:
            t, b = wall_line_in_frame(label_pts, side, eff)
            cv2.line(vis, (int(t[0]), int(t[1])),
                          (int(b[0]), int(b[1])), (0, 255, 0), 1, cv2.LINE_AA)

    # Status tag
    is_def = det['eff_L'] < TOUCH_MARGIN or det['eff_R'] < TOUCH_MARGIN
    col = (0, 0, 255) if is_def else (0, 220, 0)
    cv2.putText(vis,
                f"inner(L={det['inner_L']}, R={det['inner_R']})  "
                f"outer(L={det['outer_L']}, R={det['outer_R']})  "
                f"eff(L={det['eff_L']:.0f}, R={det['eff_R']:.0f})  "
                f"{'FAIL' if is_def else 'OK'}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    cv2.putText(vis,
                f"plastic_L={det['plastic_L']}  plastic_R={det['plastic_R']}  "
                f"hidden(L={det['hidden_L']}, R={det['hidden_R']})",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    cv2.putText(vis,
                "Cyan=label  Orange=outer  GreenSolid=inner detected  "
                "YellowDash=predicted inner  Red=GT bottle",
                (10, vis.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
    return vis


def save_debug(frame_dir, full, masked, pd_L, pd_R, det, vis):
    frame_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frame_dir / "00_original.jpg"),      full)
    cv2.imwrite(str(frame_dir / "01_a_label_filled.jpg"), masked)
    cv2.imwrite(str(frame_dir / "02_b_search_rois.jpg"),  draw_search_rois(masked, pd_L, pd_R))
    cv2.imwrite(str(frame_dir / "03_c_profiles.jpg"),
                stack_profiles(det, det['pred_inner_L'], det['pred_inner_R']))
    cv2.imwrite(str(frame_dir / "04_final.jpg"),         vis)


# ─── Template setup ──────────────────────────────────────────────────────────
def get_template(serial):
    cam_dir = CROP_DIR / serial
    tmpl_label = json.loads((cam_dir / 'template_label_bbox.json').read_text())
    tmpl_bgr = cv2.imread(str(cam_dir / 'template.jpg'))
    pts_t = shrink_label_polygon(tmpl_label['points'])
    masked = fill_label_black(tmpl_bgr, pts_t)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    pd_L = compute_strip_profile(gray, pts_t, 'left')
    pd_R = compute_strip_profile(gray, pts_t, 'right')
    inner_L, outer_L = find_outer_inner(pd_L)
    inner_R, outer_R = find_outer_inner(pd_R)
    return {
        'label_pts': pts_t,
        'label_pts_orig': tmpl_label['points'],
        'inner_L': inner_L, 'inner_R': inner_R,
        'outer_L': outer_L, 'outer_R': outer_R,
        'tmpl_bgr': tmpl_bgr,
        'masked': masked,
        'pd_L': pd_L, 'pd_R': pd_R,
    }


# ─── Main loop ───────────────────────────────────────────────────────────────
def run():
    for serial in CAMERAS:
        cam_dir  = CROP_DIR / serial
        out_cam  = OUT_DIR / serial
        fail_cam = FAIL_DIR / serial
        ann_cam  = ANN_DIR / serial
        if not cam_dir.exists():
            print(f"[{serial}] data not found → skip")
            continue
        out_cam.mkdir(exist_ok=True)
        fail_cam.mkdir(exist_ok=True)
        for f in fail_cam.glob('*.jpg'): f.unlink()

        # ── Template ──────────────────────────────────────────────────────────
        try:
            tmpl = get_template(serial)
        except Exception as e:
            print(f"[{serial}] template error: {e}")
            continue

        if any(v is None for v in [tmpl['inner_L'], tmpl['inner_R'],
                                    tmpl['outer_L'], tmpl['outer_R']]):
            print(f"[{serial}] template walls incomplete: "
                  f"inner=({tmpl['inner_L']},{tmpl['inner_R']}) "
                  f"outer=({tmpl['outer_L']},{tmpl['outer_R']}) → skip")
            continue

        print(f"\n=== {serial} ===")
        print(f"  template inner: L={tmpl['inner_L']} R={tmpl['inner_R']}")
        print(f"  template outer: L={tmpl['outer_L']} R={tmpl['outer_R']}")
        print(f"  plastic thickness: L={tmpl['outer_L']-tmpl['inner_L']} "
              f"R={tmpl['outer_R']-tmpl['inner_R']}")

        # Template debug visualization
        if SAVE_DEBUG:
            tdet = {
                'inner_L': tmpl['inner_L'], 'inner_R': tmpl['inner_R'],
                'outer_L': tmpl['outer_L'], 'outer_R': tmpl['outer_R'],
                'pred_inner_L': tmpl['inner_L'], 'pred_inner_R': tmpl['inner_R'],
                'eff_L': tmpl['inner_L'], 'eff_R': tmpl['inner_R'],
                'hidden_L': False, 'hidden_R': False,
                'profile_L': tmpl['pd_L'], 'profile_R': tmpl['pd_R'],
                'plastic_L': tmpl['outer_L']-tmpl['inner_L'],
                'plastic_R': tmpl['outer_R']-tmpl['inner_R'],
            }
            tvis = draw_final(tmpl['tmpl_bgr'], tmpl['label_pts'], tdet, tmpl)
            save_debug(out_cam / "_template_debug",
                        tmpl['tmpl_bgr'], tmpl['masked'],
                        tmpl['pd_L'], tmpl['pd_R'], tdet, tvis)

        # ── Targets ───────────────────────────────────────────────────────────
        n_pass = n_fail = n_skip = 0
        for full_path in sorted(cam_dir.glob('crop_*[0-9]_full.jpg'))[:MAX_IMGS]:
            stem = full_path.stem.replace('_full', '')
            jp = cam_dir / f"{stem}.json"
            if not jp.exists(): continue
            try:
                meta = json.loads(jp.read_text())
                label_pts = meta.get('label_points_frame')
                if label_pts is None: continue
                full = cv2.imread(str(full_path))
                if full is None: continue

                pts_s = shrink_label_polygon(label_pts)
                masked = fill_label_black(full, pts_s)
                gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
                det = detect_walls_outer_anchored(gray, pts_s, tmpl)

                # Defect check
                is_def = det['eff_L'] < TOUCH_MARGIN or det['eff_R'] < TOUCH_MARGIN

                # GT overlay if annotation exists
                gt_pts = None
                ann_p = ann_cam / f"{stem}.json"
                if ann_p.exists():
                    try:
                        gt_pts = json.loads(ann_p.read_text()).get('bottle_points')
                    except Exception: pass

                vis = draw_final(full, pts_s, det, tmpl, gt_bottle_pts=gt_pts)
                if SAVE_DEBUG:
                    save_debug(out_cam / stem, full, masked,
                                det['profile_L'], det['profile_R'], det, vis)
                else:
                    cv2.imwrite(str(out_cam / f"{stem}.jpg"), vis)

                if is_def:
                    n_fail += 1
                    cv2.imwrite(str(fail_cam / f"{stem}.jpg"), vis)
                else:
                    n_pass += 1
            except Exception as e:
                n_skip += 1
                print(f"  err {stem}: {e}")

        total = n_pass + n_fail + n_skip
        print(f"  results: pass={n_pass}  fail={n_fail}  skip={n_skip}  total={total}")
        print(f"  output: {out_cam}/")

    print("\nLegend:")
    print("  Cyan         = label polygon (SuperPoint)")
    print("  Orange       = OUTER wall detected (silhouette chai)")
    print("  Green solid  = INNER wall detected (cạnh trong gần label)")
    print("  Yellow dash  = PREDICTED inner (outer - plastic_template)")
    print("  Red          = GT bottle (if user annotated)")
    print(f"  PASS = min(inner_L, inner_R) >= {TOUCH_MARGIN}px")
    print(f"  FAIL = min(inner_L, inner_R) <  {TOUCH_MARGIN}px")


if __name__ == '__main__':
    run()
