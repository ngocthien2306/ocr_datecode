"""
Alignment detection v2 — fill-black + outward scan + gap metric.

Pipeline:
1. Pre-compute template:
   - Fill label region với pixel đen
   - Scan từ mép label ra ngoài → peak đầu tiên = cạnh trong chai
   - Lưu template_inner_width (sanity check sau)

2. Mỗi target frame:
   - Lấy label_points_frame từ JSON (reliable, SuperPoint bám nhãn chuẩn)
   - Fill label đen trên full frame
   - Scan outward → 2 cạnh trong chai
   - gap_L = label_left  - bottle_inner_left
   - gap_R = bottle_inner_right - label_right
   - Reject khi min(gap_L, gap_R) < TOUCH_MARGIN

3. Sanity check: |detected_width - template_inner_width| / template_inner_width > 15%
   → 1 trong 2 cạnh sai → extrapolate từ cạnh tin cậy hơn

Run: python3 test_alignment.py
"""

import cv2
import json
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────
CROP_DIR = Path("/home/demo/Source/ocr_datecode/crop_samples")
OUT_DIR  = CROP_DIR / "align_vis"
OUT_DIR.mkdir(exist_ok=True)

CAMERAS  = ['40767171', '40733814']
MAX_IMGS = 100

SEARCH_W      = 80     # px tìm ra ngoài 2 bên label
EDGE_MARGIN   = 4      # skip vùng sát mép label để tránh fake peak từ fill artifact
MIN_SEARCH_PX = 25     # search window < ngưỡng này → undetectable, fallback extrapolate
PEAK_HEIGHT   = 0.15   # normalized
PEAK_PROM     = 0.05
PEAK_DIST     = 5
TOUCH_MARGIN  = 5      # min(gap_L, gap_R) < margin → reject
WIDTH_TOL     = 0.15   # |measured_w - tmpl_w| / tmpl_w > tol → 1 cạnh sai
SAVE_DEBUG    = True   # save 5 ảnh debug/frame (original, a/b/c, final)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def fill_label_black(img_bgr, label_pts):
    """Fill label polygon với màu đen → loại text/hoa văn khi Sobel."""
    out = img_bgr.copy()
    pts = np.array(label_pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(out, [pts], (0, 0, 0))
    return out


def label_perp_width(label_pts):
    """Khoảng cách perpendicular giữa mép trái và mép phải của label (qua midpoint)."""
    p = np.asarray(label_pts, dtype=np.float32)
    left_mid  = (p[0] + p[3]) / 2.0   # midpoint mép trái (TL→BL)
    right_mid = (p[1] + p[2]) / 2.0   # midpoint mép phải (TR→BR)
    return float(np.linalg.norm(right_mid - left_mid))


def detect_inner_wall(gray, label_pts, side, search_w=SEARCH_W):
    """
    Search dọc rotated strip song song với cạnh label (handle label xoay).

    label_pts: 4 corner [TL, TR, BR, BL] trong frame
    side:      'left' (cạnh TL→BL) | 'right' (cạnh TR→BR)
    Trả về dict:
      x_offset:   perpendicular distance từ mép label ra ngoài đến cạnh chai (px)
      inner_line: [(x1,y1),(x2,y2)] toạ độ cạnh chai trong frame
      src_quad:   4 corner của search ROI trong frame (để visualize)
      profile, peaks, picked, peak_height: như cũ
      reason:     chuỗi giải thích nếu không detect được
    """
    pts = np.asarray(label_pts, dtype=np.float32)
    out = {'x_offset': None, 'peak_height': 0.0, 'side': side,
           'profile': None, 'peaks': np.array([]), 'picked': -1,
           'src_quad': None, 'inner_line': None,
           'p_top': None, 'p_bot': None,
           'reason': ''}

    if side == 'left':
        p_top, p_bot = pts[0], pts[3]   # TL, BL
    else:
        p_top, p_bot = pts[1], pts[2]   # TR, BR
    out['p_top'], out['p_bot'] = p_top.tolist(), p_bot.tolist()

    edge_vec = p_bot - p_top
    edge_len = float(np.linalg.norm(edge_vec))
    if edge_len < 10:
        out['reason'] = 'edge_too_short'
        return out
    edge_dir = edge_vec / edge_len

    # Perpendicular outward (away from label) — image-coord rotation
    # edge_dir đi từ TOP → BOTTOM. Trong toạ độ ảnh (Y xuống):
    #   outward LEFT  = (-edge_y,  edge_x)
    #   outward RIGHT = ( edge_y, -edge_x)
    if side == 'left':
        perp = np.array([-edge_dir[1], edge_dir[0]], dtype=np.float32)
    else:
        perp = np.array([ edge_dir[1], -edge_dir[0]], dtype=np.float32)

    # 4 corner search ROI (rotated quad trong frame):
    inner_top = p_top + EDGE_MARGIN * perp
    inner_bot = p_bot + EDGE_MARGIN * perp
    outer_top = p_top + float(search_w) * perp
    outer_bot = p_bot + float(search_w) * perp
    src_quad = np.array([inner_top, outer_top, outer_bot, inner_bot], dtype=np.float32)
    out['src_quad'] = src_quad.tolist()

    h = int(round(edge_len))
    w = int(search_w - EDGE_MARGIN)
    if w < MIN_SEARCH_PX or h < 10:
        out['reason'] = 'window_too_narrow'
        return out

    # Warp rotated strip thành ảnh axis-aligned (X = outward distance, Y = along edge)
    dst_quad = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_quad, dst_quad)
    strip = cv2.warpPerspective(gray, M, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)

    # Sobel X (gradient theo outward direction)
    sobel = cv2.Scharr(strip.astype(np.float32), cv2.CV_32F, 1, 0)
    profile = np.sum(np.abs(sobel), axis=0)
    profile = gaussian_filter1d(profile, sigma=1.5)

    pmax = profile.max()
    if pmax == 0:
        out['reason'] = 'flat_profile'
        return out
    profile_n = profile / pmax
    out['profile'] = profile_n

    peaks, props = find_peaks(profile_n, height=PEAK_HEIGHT,
                              distance=PEAK_DIST, prominence=PEAK_PROM)
    out['peaks'] = peaks
    if len(peaks) == 0:
        out['reason'] = 'no_peaks'
        return out

    # Với rotated strip: x=0 luôn sát label, x=w xa nhất → peak gần label = smallest x
    idx = 0
    inner_local = int(peaks[idx])
    out['picked'] = idx
    out['peak_height'] = float(props['peak_heights'][idx])

    # Perpendicular distance từ mép label
    out['x_offset'] = inner_local + EDGE_MARGIN

    # Cạnh chai trong frame: line từ top → bot, dịch outward (inner_local + EDGE_MARGIN)
    inner_pt_top = p_top + (inner_local + EDGE_MARGIN) * perp
    inner_pt_bot = p_bot + (inner_local + EDGE_MARGIN) * perp
    out['inner_line'] = [inner_pt_top.tolist(), inner_pt_bot.tolist()]
    return out


# ─── Debug visualization helpers ─────────────────────────────────────────────
def draw_search_rois(img_bgr, info_L, info_R):
    """Draw rotated search ROI quads cho cả 2 side."""
    out = img_bgr.copy()
    for info, color, lbl in [(info_L, (0, 255, 255), 'L search'),
                              (info_R, (0, 255, 255), 'R search')]:
        if info.get('src_quad') is None:
            continue
        quad = np.array(info['src_quad'], dtype=np.int32)
        cv2.polylines(out, [quad], True, color, 2)
        cv2.putText(out, lbl, tuple(quad[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def plot_profile(info, canvas_h=220, scale=2):
    """
    Vẽ profile 1D: trục X = outward distance từ label (px),
    red dot = peak chosen, cam = peak khác.
    """
    profile = info['profile']
    if profile is None or len(profile) == 0:
        canvas = np.full((canvas_h, 400, 3), 40, dtype=np.uint8)
        cv2.putText(canvas, f"no profile ({info.get('reason','')})",
                    (10, canvas_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        return canvas

    n = len(profile)
    canvas_w = max(200, n * scale)
    canvas = np.full((canvas_h, canvas_w, 3), 40, dtype=np.uint8)

    baseline_y = canvas_h - 15
    top_y      = 15

    th_y = baseline_y - int(PEAK_HEIGHT * (baseline_y - top_y))
    cv2.line(canvas, (0, th_y), (canvas_w, th_y), (80, 80, 180), 1)
    cv2.putText(canvas, f"thresh={PEAK_HEIGHT}", (5, th_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 220), 1)

    pts = []
    for i, v in enumerate(profile):
        x = i * scale
        y = baseline_y - int(v * (baseline_y - top_y))
        pts.append((x, y))
    for i in range(len(pts) - 1):
        cv2.line(canvas, pts[i], pts[i + 1], (220, 220, 220), 1)

    for j, p in enumerate(info['peaks']):
        x = int(p) * scale
        y = baseline_y - int(profile[p] * (baseline_y - top_y))
        is_picked = (j == info['picked'])
        col = (0, 0, 255) if is_picked else (0, 165, 255)
        cv2.circle(canvas, (x, y), 5, col, -1)
        cv2.line(canvas, (x, baseline_y), (x, y), col, 1)
        if is_picked:
            offset = int(p) + EDGE_MARGIN
            cv2.putText(canvas, f"gap={offset}", (x + 8, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

    side = info['side'].upper()
    cv2.putText(canvas, f"{side} profile (perp from label edge)",
                (5, top_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.line(canvas, (0, baseline_y), (canvas_w, baseline_y), (100, 100, 100), 1)
    return canvas


def stack_profiles(info_L, info_R):
    """Ghép profile L + R thành 1 ảnh."""
    p_L = plot_profile(info_L)
    p_R = plot_profile(info_R)
    h = max(p_L.shape[0], p_R.shape[0])
    if p_L.shape[0] != h:
        p_L = cv2.copyMakeBorder(p_L, 0, h - p_L.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=40)
    if p_R.shape[0] != h:
        p_R = cv2.copyMakeBorder(p_R, 0, h - p_R.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=40)
    sep = np.full((h, 4, 3), (60, 60, 60), dtype=np.uint8)
    return np.hstack([p_L, sep, p_R])


def save_debug(frame_dir, full, masked, info_L, info_R, final_vis):
    """5 ảnh: original, a=label_filled, b=search_rois, c=profiles, final."""
    frame_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frame_dir / "00_original.jpg"),     full)
    cv2.imwrite(str(frame_dir / "01_a_label_filled.jpg"), masked)
    cv2.imwrite(str(frame_dir / "02_b_search_rois.jpg"),  draw_search_rois(masked, info_L, info_R))
    cv2.imwrite(str(frame_dir / "03_c_profiles.jpg"),     stack_profiles(info_L, info_R))
    cv2.imwrite(str(frame_dir / "04_final.jpg"),          final_vis)


# ─── Main ────────────────────────────────────────────────────────────────────
for serial in CAMERAS:
    cam_dir = CROP_DIR / serial
    out_cam = OUT_DIR / serial
    out_cam.mkdir(exist_ok=True)

    tmpl_path = cam_dir / "template.jpg"
    tmpl_label_path = cam_dir / "template_label_bbox.json"
    if not tmpl_path.exists() or not tmpl_label_path.exists():
        print(f"[{serial}] missing template.jpg or template_label_bbox.json → skip")
        continue

    # ── Template: detect inner walls ──────────────────────────────────────────
    tmpl_bgr = cv2.imread(str(tmpl_path))
    with open(tmpl_label_path) as f:
        tmpl_label = json.load(f)

    tmpl_masked = fill_label_black(tmpl_bgr, tmpl_label['points'])
    tmpl_gray = cv2.cvtColor(tmpl_masked, cv2.COLOR_BGR2GRAY)

    t_info_L = detect_inner_wall(tmpl_gray, tmpl_label['points'], 'left')
    t_info_R = detect_inner_wall(tmpl_gray, tmpl_label['points'], 'right')
    if t_info_L['x_offset'] is None or t_info_R['x_offset'] is None:
        print(f"[{serial}] template inner walls NOT detected "
              f"(L={t_info_L['reason']} R={t_info_R['reason']}) → skip")
        continue

    tmpl_label_w = label_perp_width(tmpl_label['points'])
    tmpl_gap_L = t_info_L['x_offset']
    tmpl_gap_R = t_info_R['x_offset']
    tmpl_inner_w = tmpl_gap_L + tmpl_label_w + tmpl_gap_R   # perp inner-to-inner

    print(f"\n=== {serial} ===")
    print(f"  template: label_perp_w={tmpl_label_w:.1f}  "
          f"gap_L={tmpl_gap_L}  gap_R={tmpl_gap_R}  inner_w={tmpl_inner_w:.1f}")

    # Template visualization: rotated walls + label polygon
    tvis = tmpl_bgr.copy()
    cv2.polylines(tvis, [np.array(tmpl_label['points'], dtype=np.int32)],
                  True, (220, 220, 0), 2)
    for info, col in [(t_info_L, (0, 255, 0)), (t_info_R, (0, 255, 0))]:
        if info['inner_line']:
            p0, p1 = info['inner_line']
            cv2.line(tvis, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), col, 2)
    cv2.putText(tvis, f"gap_L={tmpl_gap_L}  gap_R={tmpl_gap_R}  "
                f"label_w={tmpl_label_w:.0f}  inner_w={tmpl_inner_w:.0f}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 220), 2)
    cv2.imwrite(str(out_cam / "_template_check.jpg"), tvis)
    if SAVE_DEBUG:
        save_debug(out_cam / "_template_debug", tmpl_bgr, tmpl_masked,
                   t_info_L, t_info_R, tvis)

    # ── Targets ───────────────────────────────────────────────────────────────
    imgs = sorted(cam_dir.glob("crop_*[0-9]_full.jpg"))[:MAX_IMGS]
    n_pass = n_fail = n_skip = 0
    gap_L_list, gap_R_list = [], []

    for full_path in imgs:
        stem_no_full = full_path.stem.replace('_full', '')
        json_path = cam_dir / (stem_no_full + '.json')
        if not json_path.exists():
            continue

        full = cv2.imread(str(full_path))
        if full is None:
            continue
        with open(json_path) as f:
            meta = json.load(f)
        label_pts = meta.get('label_points_frame')
        if label_pts is None:
            continue

        masked = fill_label_black(full, label_pts)
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

        info_L = detect_inner_wall(gray, label_pts, 'left')
        info_R = detect_inner_wall(gray, label_pts, 'right')
        gL_det, lh = info_L['x_offset'], info_L['peak_height']
        gR_det, rh = info_R['x_offset'], info_R['peak_height']
        target_label_w = label_perp_width(label_pts)
        expected_total = tmpl_inner_w   # tmpl_gap_L + tmpl_label_w + tmpl_gap_R

        # ── Sanity + extrapolate (theo perpendicular gap) ─────────────────────
        source = 'both'
        if gL_det is not None and gR_det is not None:
            total = gL_det + target_label_w + gR_det
            rel_err = abs(total - expected_total) / max(expected_total, 1)
            if rel_err > WIDTH_TOL:
                # 1 cạnh bắt sai → giữ peak cao hơn, extrapolate cạnh kia
                if lh >= rh:
                    gR_det = max(0, expected_total - target_label_w - gL_det)
                    source = 'extrap_R'
                else:
                    gL_det = max(0, expected_total - target_label_w - gR_det)
                    source = 'extrap_L'
        elif gL_det is not None:
            gR_det = max(0, expected_total - target_label_w - gL_det)
            source = 'extrap_R'
        elif gR_det is not None:
            gL_det = max(0, expected_total - target_label_w - gR_det)
            source = 'extrap_L'
        else:
            n_skip += 1
            if SAVE_DEBUG:
                fail_vis = full.copy()
                cv2.putText(fail_vis,
                            f"SKIP: L={info_L['reason']} R={info_R['reason']}",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                save_debug(out_cam / stem_no_full, full, masked,
                           info_L, info_R, fail_vis)
            continue

        gap_L = int(round(gL_det))
        gap_R = int(round(gR_det))
        gap_min = min(gap_L, gap_R)
        is_ok = gap_min >= TOUCH_MARGIN

        gap_L_list.append(gap_L)
        gap_R_list.append(gap_R)
        if is_ok:
            n_pass += 1
        else:
            n_fail += 1

        # ── Helper: build inner wall line cho cả case extrapolate ──────────────
        def build_wall_line(info, side, gap_perp):
            """Trả về (p_top, p_bot) trong frame của cạnh trong, dùng gap_perp px.
            Perp formula PHẢI match detect_inner_wall — outward direction.
            """
            pts_arr = np.asarray(label_pts, dtype=np.float32)
            if side == 'left':
                p_top, p_bot = pts_arr[0], pts_arr[3]
            else:
                p_top, p_bot = pts_arr[1], pts_arr[2]
            edge = p_bot - p_top
            edge_len = float(np.linalg.norm(edge)) or 1.0
            edge_dir = edge / edge_len
            if side == 'left':
                perp = np.array([-edge_dir[1], edge_dir[0]], dtype=np.float32)
            else:
                perp = np.array([ edge_dir[1], -edge_dir[0]], dtype=np.float32)
            return (p_top + gap_perp * perp, p_bot + gap_perp * perp)

        # ── Visualize ─────────────────────────────────────────────────────────
        vis = full.copy()

        # Label polygon (cyan)
        cv2.polylines(vis, [np.array(label_pts, dtype=np.int32)],
                      True, (220, 220, 0), 2)

        # Inner walls (green if detected, orange if extrapolated)
        c_l = (0, 165, 255) if source == 'extrap_L' else (0, 255, 0)
        c_r = (0, 165, 255) if source == 'extrap_R' else (0, 255, 0)
        wl_top, wl_bot = build_wall_line(info_L, 'left',  gap_L)
        wr_top, wr_bot = build_wall_line(info_R, 'right', gap_R)
        cv2.line(vis, (int(wl_top[0]), int(wl_top[1])),
                      (int(wl_bot[0]), int(wl_bot[1])), c_l, 2)
        cv2.line(vis, (int(wr_top[0]), int(wr_top[1])),
                      (int(wr_bot[0]), int(wr_bot[1])), c_r, 2)

        # Gap text gần midpoint của wall
        col_L = (0, 220, 0) if gap_L >= TOUCH_MARGIN else (0, 0, 255)
        col_R = (0, 220, 0) if gap_R >= TOUCH_MARGIN else (0, 0, 255)
        wl_mid = ((wl_top + wl_bot) / 2).astype(int)
        wr_mid = ((wr_top + wr_bot) / 2).astype(int)
        cv2.putText(vis, f"L={gap_L}", (int(wl_mid[0]) - 80, int(wl_mid[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col_L, 2)
        cv2.putText(vis, f"R={gap_R}", (int(wr_mid[0]) + 10, int(wr_mid[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col_R, 2)

        tag = (f"gapL={gap_L}  gapR={gap_R}  "
               f"min={gap_min}  [{source}]  {'OK' if is_ok else 'FAIL'}")
        col = (0, 220, 0) if is_ok else (0, 0, 255)
        cv2.putText(vis, tag, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)

        if SAVE_DEBUG:
            save_debug(out_cam / stem_no_full, full, masked, info_L, info_R, vis)
        else:
            cv2.imwrite(str(out_cam / f"{stem_no_full}.jpg"), vis)

    # ── Stats ─────────────────────────────────────────────────────────────────
    total = n_pass + n_fail + n_skip
    print(f"  total={total}  pass={n_pass}  fail={n_fail}  skip={n_skip}")
    if gap_L_list:
        gl = np.array(gap_L_list); gr = np.array(gap_R_list)
        print(f"  gap_L: mean={gl.mean():+.1f}  std={gl.std():.1f}  "
              f"min={gl.min():+d}  max={gl.max():+d}")
        print(f"  gap_R: mean={gr.mean():+.1f}  std={gr.std():.1f}  "
              f"min={gr.min():+d}  max={gr.max():+d}")
    print(f"  output: {out_cam}/")

print("\nLegend:")
print("  Green   = bottle inner wall (detected)")
print("  Orange  = bottle inner wall (extrapolated)")
print("  Cyan    = label bbox")
print(f"  PASS = min(gap_L, gap_R) >= {TOUCH_MARGIN}px (label chưa chạm chai)")
print(f"  FAIL = min(gap_L, gap_R) <  {TOUCH_MARGIN}px (label chạm/vượt cạnh chai)")
