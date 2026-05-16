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


def points_to_box(pts):
    arr = np.asarray(pts, dtype=np.float32)
    return {
        'x_min': int(arr[:, 0].min()), 'x_max': int(arr[:, 0].max()),
        'y_min': int(arr[:, 1].min()), 'y_max': int(arr[:, 1].max()),
    }


def detect_inner_wall(gray, label_box, side, search_w=SEARCH_W):
    """
    Scan từ mép label ra ngoài, lấy peak đầu tiên = cạnh TRONG chai.

    gray: full frame grayscale (đã fill label đen)
    side: 'left' hoặc 'right'
    Trả về dict với x_frame, peak_height, profile, peaks, picked, ROI.
    Nếu không detect được → x_frame=None.
    """
    H, W = gray.shape[:2]
    y1 = max(0, label_box['y_min'])
    y2 = min(H, label_box['y_max'])
    out = {'x_frame': None, 'peak_height': 0.0, 'side': side,
           'profile': None, 'peaks': np.array([]), 'picked': -1,
           'x_start': 0, 'x_end': 0, 'y1': y1, 'y2': y2,
           'reason': ''}
    if y2 <= y1:
        out['reason'] = 'empty_y_band'
        return out

    if side == 'left':
        x_start = max(0, label_box['x_min'] - search_w)
        x_end   = label_box['x_min'] - EDGE_MARGIN
    else:
        x_start = label_box['x_max'] + EDGE_MARGIN
        x_end   = min(W, label_box['x_max'] + search_w)
    out['x_start'], out['x_end'] = x_start, x_end
    if x_end - x_start < MIN_SEARCH_PX:
        out['reason'] = 'window_too_narrow'
        return out

    region = gray[y1:y2, x_start:x_end]
    sobel = cv2.Scharr(region.astype(np.float32), cv2.CV_32F, 1, 0)
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

    if side == 'left':
        idx = int(np.argmax(peaks))
    else:
        idx = 0
    inner_local = int(peaks[idx])
    out['picked'] = idx
    out['x_frame'] = x_start + inner_local
    out['peak_height'] = float(props['peak_heights'][idx])
    return out


# ─── Debug visualization helpers ─────────────────────────────────────────────
def draw_search_rois(img_bgr, info_L, info_R):
    """Draw search ROI rectangles for both sides on a copy of img_bgr."""
    out = img_bgr.copy()
    for info, color, lbl in [(info_L, (0, 255, 255), 'L search'),
                              (info_R, (0, 255, 255), 'R search')]:
        if info['x_end'] > info['x_start']:
            cv2.rectangle(out,
                          (info['x_start'], info['y1']),
                          (info['x_end'],   info['y2']),
                          color, 2)
            cv2.putText(out, lbl, (info['x_start'], max(20, info['y1'] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def plot_profile(info, canvas_h=220, scale=2):
    """
    Vẽ profile 1D thành ảnh: trục X = local x trong search ROI,
    marker peaks (đỏ = picked, cam = các peak khác).
    """
    profile = info['profile']
    if profile is None or len(profile) == 0:
        canvas = np.full((canvas_h, max(200, info['x_end'] - info['x_start']) * scale, 3),
                         40, dtype=np.uint8)
        cv2.putText(canvas, f"no profile ({info.get('reason','')})",
                    (10, canvas_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        return canvas

    n = len(profile)
    canvas_w = max(200, n * scale)
    canvas = np.full((canvas_h, canvas_w, 3), 40, dtype=np.uint8)

    baseline_y = canvas_h - 15
    top_y      = 15

    # Threshold line
    th_y = baseline_y - int(PEAK_HEIGHT * (baseline_y - top_y))
    cv2.line(canvas, (0, th_y), (canvas_w, th_y), (80, 80, 180), 1)
    cv2.putText(canvas, f"thresh={PEAK_HEIGHT}", (5, th_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 220), 1)

    # Profile polyline
    pts = []
    for i, v in enumerate(profile):
        x = i * scale
        y = baseline_y - int(v * (baseline_y - top_y))
        pts.append((x, y))
    for i in range(len(pts) - 1):
        cv2.line(canvas, pts[i], pts[i + 1], (220, 220, 220), 1)

    # Peaks
    for j, p in enumerate(info['peaks']):
        x = int(p) * scale
        y = baseline_y - int(profile[p] * (baseline_y - top_y))
        is_picked = (j == info['picked'])
        col = (0, 0, 255) if is_picked else (0, 165, 255)
        cv2.circle(canvas, (x, y), 5, col, -1)
        cv2.line(canvas, (x, baseline_y), (x, y), col, 1)
        if is_picked:
            x_frame = info['x_start'] + int(p)
            cv2.putText(canvas, f"x={x_frame}", (x + 8, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

    # Axis label
    side = info['side'].upper()
    cv2.putText(canvas, f"{side} profile (x_start={info['x_start']})",
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

    t_info_L = detect_inner_wall(tmpl_gray, tmpl_label, 'left')
    t_info_R = detect_inner_wall(tmpl_gray, tmpl_label, 'right')
    t_left, t_right = t_info_L['x_frame'], t_info_R['x_frame']

    if t_left is None or t_right is None:
        print(f"[{serial}] template inner walls NOT detected "
              f"(left={t_left}, right={t_right}) → skip")
        continue

    tmpl_inner_w = t_right - t_left
    tmpl_gap_L = tmpl_label['x_min'] - t_left
    tmpl_gap_R = t_right - tmpl_label['x_max']

    print(f"\n=== {serial} ===")
    print(f"  template: label_x=({tmpl_label['x_min']}..{tmpl_label['x_max']}) "
          f"inner_x=({t_left}..{t_right}) width={tmpl_inner_w}")
    print(f"            gap_L={tmpl_gap_L} gap_R={tmpl_gap_R}")

    # Template final vis + debug stack
    tvis = tmpl_bgr.copy()
    cv2.rectangle(tvis,
                  (tmpl_label['x_min'], max(0, tmpl_label['y_min'])),
                  (tmpl_label['x_max'], min(tvis.shape[0], tmpl_label['y_max'])),
                  (220, 220, 0), 2)
    H = tvis.shape[0]
    cv2.line(tvis, (t_left, 0),  (t_left, H),  (0, 255, 0), 2)
    cv2.line(tvis, (t_right, 0), (t_right, H), (0, 255, 0), 2)
    cv2.putText(tvis, f"gap_L={tmpl_gap_L}", (t_left + 5, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2)
    cv2.putText(tvis, f"gap_R={tmpl_gap_R}", (t_right - 200, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2)
    cv2.putText(tvis, f"inner_w={tmpl_inner_w}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
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

        label_box = points_to_box(label_pts)
        masked = fill_label_black(full, label_pts)
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

        info_L = detect_inner_wall(gray, label_box, 'left')
        info_R = detect_inner_wall(gray, label_box, 'right')
        left_x,  left_h  = info_L['x_frame'],  info_L['peak_height']
        right_x, right_h = info_R['x_frame'], info_R['peak_height']

        # ── Sanity + extrapolate ──────────────────────────────────────────────
        source = 'both'
        if left_x is not None and right_x is not None:
            measured_w = right_x - left_x
            rel_err = abs(measured_w - tmpl_inner_w) / max(tmpl_inner_w, 1)
            if rel_err > WIDTH_TOL:
                if left_h >= right_h:
                    right_x = left_x + tmpl_inner_w
                    source = 'extrap_R'
                else:
                    left_x = right_x - tmpl_inner_w
                    source = 'extrap_L'
        elif left_x is not None:
            right_x = left_x + tmpl_inner_w
            source = 'extrap_R'
        elif right_x is not None:
            left_x = right_x - tmpl_inner_w
            source = 'extrap_L'
        else:
            n_skip += 1
            if SAVE_DEBUG:
                # vẫn lưu debug để biết tại sao skip
                fail_vis = full.copy()
                cv2.putText(fail_vis,
                            f"SKIP: L={info_L['reason']} R={info_R['reason']}",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                save_debug(out_cam / stem_no_full, full, masked,
                           info_L, info_R, fail_vis)
            continue

        gap_L = int(label_box['x_min'] - left_x)
        gap_R = int(right_x - label_box['x_max'])
        gap_min = min(gap_L, gap_R)
        is_ok = gap_min >= TOUCH_MARGIN

        gap_L_list.append(gap_L)
        gap_R_list.append(gap_R)
        if is_ok:
            n_pass += 1
        else:
            n_fail += 1

        # ── Visualize ─────────────────────────────────────────────────────────
        vis = full.copy()
        Hv = vis.shape[0]

        # Label bbox (cyan)
        cv2.polylines(vis, [np.array(label_pts, dtype=np.int32)],
                      True, (220, 220, 0), 2)

        # Inner walls (green if detected, orange if extrapolated)
        c_l = (0, 165, 255) if source == 'extrap_L' else (0, 255, 0)
        c_r = (0, 165, 255) if source == 'extrap_R' else (0, 255, 0)
        cv2.line(vis, (int(left_x), 0),  (int(left_x), Hv),  c_l, 2)
        cv2.line(vis, (int(right_x), 0), (int(right_x), Hv), c_r, 2)

        # Gap text inline với label band
        y_text = (label_box['y_min'] + label_box['y_max']) // 2
        y_text = max(40, min(Hv - 20, y_text))
        col_L = (0, 220, 0) if gap_L >= TOUCH_MARGIN else (0, 0, 255)
        col_R = (0, 220, 0) if gap_R >= TOUCH_MARGIN else (0, 0, 255)
        cv2.putText(vis, f"L={gap_L}", (int(left_x) + 8, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col_L, 2)
        cv2.putText(vis, f"R={gap_R}", (int(right_x) - 110, y_text),
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
