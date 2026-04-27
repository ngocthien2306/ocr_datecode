import cv2 as cv
import numpy as np
import os


DEFAULT_PARAMS = {
    # --- Segmentation ---
    'min_proc_h': 80,           # upscale ảnh nhỏ tới ít nhất chiều cao này (px)
    'clahe_clip': 2.0,          # CLAHE clipLimit; cao = contrast mạnh hơn
    'clahe_grid': 8,            # CLAHE tile grid size (NxN)
    'blur_kernel': 3,           # Gaussian blur kernel trước threshold (lẻ)
    'close_kernel_factor': 0.025,  # MORPH_CLOSE kernel = % chiều cao ảnh proc
    'min_char_h_factor': 0.3,   # bỏ contour có h < factor * h_proc
    'min_char_w': 3,            # bỏ contour quá hẹp
    'padding': 2,               # padding khi crop char ra
    # --- Compare ---
    'pass_threshold': 0.75,     # ngưỡng PASS/FAIL mỗi char
    'compare_size': 64,         # ô vuông để so sánh (size, size)
    'tm_blur_sigma': 1.2,       # blur σ trước template matching
    'iou_dilate': 5,            # dilation kernel cho IoU (tha thứ stroke width)
    'pixel_dev_tol': 1.4,       # độ dung sai cho pixel ratio (cao = dễ pass)
    'align_max_shift': 8,       # brute-force shift radius khi align mask cho IoU/diff (px)
    'align_scale_tol': 0.15,    # ±tol cho scale search (0 = tắt). 0.15 = thử 0.85..1.15
    'align_scale_steps': 5,     # số scale candidates trong khoảng tol
    'keep_largest_cc': 1,       # 1=giữ chỉ connected-component lớn nhất khi normalize char (drop noise rời)
    'min_cc_area_ratio': 0.05,  # các CC < ratio*lớn-nhất sẽ bị drop (chỉ áp khi keep_largest_cc=0)
    # Smart alignment (extent-normalize + ECC fine-align)
    'extent_target_fill': 0.75, # fit foreground sao cho chiếm % này của canvas (decouple khỏi crop padding)
    'use_ecc_align': 0,         # 1=dùng cv2.findTransformECC fine-align (sub-pixel + rotation)
                                # OFF by default: empirically worse than brute-force scale+shift on
                                # binary masks. Bật khi ảnh có rotation rõ rệt mà brute-force không xử lý.
    # Per-char workflow (when annotations include `chars`)
    'char_local_search_radius': 0.15,  # ±frac của char dim — local matchTemplate refinement
                                        # bù cho SuperPoint homography hơi lệch. 0 = thuần homography
    'char_refine_min_score': 0.3,      # nếu match score < ngưỡng → fallback về coarse (tránh trượt char)
}


def char_polygon_to_points(char_dict):
    """Normalise a char dict (rectangle or polygon) into a numpy (4, 2) float32
    array of corner points. Used to bypass segmentation when the user has
    pre-marked individual char positions on the template."""
    if char_dict.get('shape') == 'polygon' and 'points' in char_dict:
        return np.array(char_dict['points'], dtype=np.float32)
    x = float(char_dict['x'])
    y = float(char_dict['y'])
    w = float(char_dict['width'])
    h = float(char_dict['height'])
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                    dtype=np.float32)


def warp_char_with_refinement(template_img, target_img, H, char_polygon,
                               search_radius=0.15, refine_min_score=0.3):
    """
    Get template+target crops for ONE character defined by `char_polygon`
    (4 corner points in template image coordinates).

    1. Coarse: warp char polygon to target via homography H
    2. Crop template at canonical (W×H) rect
    3. Crop target with extra search padding
    4. cv2.matchTemplate inside the padded target → exact char position
       (compensates for slight SuperPoint inaccuracy)
    5. Re-crop target at refined position. Falls back to coarse crop if the
       match score is below `refine_min_score` (avoids drifting onto neighbour
       chars when they look similar).

    `search_radius` = 0 disables refinement entirely.

    Returns (tmpl_crop_bgr, tgt_crop_bgr, dx_dy_offset, match_score).
    """
    pts_template = np.asarray(char_polygon, dtype=np.float32)

    # 1. Coarse target position via homography
    pts_target_coarse = cv.perspectiveTransform(
        pts_template.reshape(-1, 1, 2), H
    ).reshape(-1, 2)

    # 2. Canonical dims from coarse target geometry
    w = int(round(max(
        np.linalg.norm(pts_target_coarse[0] - pts_target_coarse[1]),
        np.linalg.norm(pts_target_coarse[2] - pts_target_coarse[3]),
    )))
    h = int(round(max(
        np.linalg.norm(pts_target_coarse[1] - pts_target_coarse[2]),
        np.linalg.norm(pts_target_coarse[3] - pts_target_coarse[0]),
    )))
    if w < 4 or h < 4:
        return None, None, (0, 0), 0.0

    dst_canon = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                         dtype=np.float32)

    # Template char (canonical)
    M_tmpl = cv.getPerspectiveTransform(pts_template, dst_canon)
    tmpl_crop = cv.warpPerspective(template_img, M_tmpl, (w, h))

    # Coarse target crop (used directly when refinement is off or fails)
    M_tgt_coarse = cv.getPerspectiveTransform(pts_target_coarse, dst_canon)
    tgt_crop_coarse = cv.warpPerspective(target_img, M_tgt_coarse, (w, h))

    if search_radius <= 0:
        return tmpl_crop, tgt_crop_coarse, (0, 0), 1.0

    # 3. Expanded target search window
    pad_x = max(2, int(round(w * search_radius)))
    pad_y = max(2, int(round(h * search_radius)))
    big_w = w + 2 * pad_x
    big_h = h + 2 * pad_y
    dst_expanded = np.array([
        [pad_x, pad_y],
        [pad_x + w - 1, pad_y],
        [pad_x + w - 1, pad_y + h - 1],
        [pad_x, pad_y + h - 1],
    ], dtype=np.float32)
    M_tgt_expanded = cv.getPerspectiveTransform(pts_target_coarse, dst_expanded)
    tgt_search = cv.warpPerspective(target_img, M_tgt_expanded, (big_w, big_h))

    # 4. Local matchTemplate to find exact char position
    tmpl_gray = cv.cvtColor(tmpl_crop, cv.COLOR_BGR2GRAY) \
        if tmpl_crop.ndim == 3 else tmpl_crop
    tgt_gray = cv.cvtColor(tgt_search, cv.COLOR_BGR2GRAY) \
        if tgt_search.ndim == 3 else tgt_search

    if (tgt_gray.shape[0] < tmpl_gray.shape[0] or
            tgt_gray.shape[1] < tmpl_gray.shape[1]):
        return tmpl_crop, tgt_crop_coarse, (0, 0), 0.0

    res = cv.matchTemplate(tgt_gray, tmpl_gray, cv.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv.minMaxLoc(res)

    # 5. Fallback: bad score → keep coarse crop (avoid jumping to wrong char)
    if max_val < refine_min_score:
        return tmpl_crop, tgt_crop_coarse, (0, 0), float(max_val)

    x, y = int(max_loc[0]), int(max_loc[1])
    tgt_crop_refined = tgt_search[y:y + h, x:x + w].copy()
    return tmpl_crop, tgt_crop_refined, (x - pad_x, y - pad_y), float(max_val)


def _drop_small_cc(mask, keep_largest_only=True, min_area_ratio=0.05):
    """Drop disconnected noise blobs from a binary mask.

    `keep_largest_only=True` → only the single biggest component survives
    (handles the typical "1 char + scattered noise" case).
    Otherwise, all components below `min_area_ratio * largest_area` are dropped.
    """
    if mask is None or mask.size == 0:
        return mask
    n_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    if n_labels <= 2:  # 0=bg, 1=single component → nothing to drop
        return mask

    areas = stats[1:, cv.CC_STAT_AREA]  # exclude background (label 0)
    if keep_largest_only:
        keep = {1 + int(np.argmax(areas))}
    else:
        threshold = float(np.max(areas)) * float(min_area_ratio)
        keep = {1 + i for i, a in enumerate(areas) if a >= threshold}

    out = np.zeros_like(mask)
    for lbl in keep:
        out[labels == lbl] = 255
    return out


def _ensure_fg_white(th, border_frac=0.10):
    """
    Make sure foreground = white. Polarity is decided by the BORDER pixels:
    the outer ring of a tight crop is almost certainly background, so if the
    border is mostly white we flip. Falls back to the global majority rule
    when the crop is too small to have a meaningful border.
    Works for any binary {0,255} image.
    """
    h, w = th.shape[:2]
    if h < 4 or w < 4:
        return cv.bitwise_not(th) if np.mean(th) > 127 else th

    bw = max(1, int(round(min(h, w) * border_frac)))
    border = np.concatenate([
        th[:bw, :].ravel(),       # top band
        th[-bw:, :].ravel(),      # bottom band
        th[bw:-bw, :bw].ravel(),  # left band (excluding corners already counted)
        th[bw:-bw, -bw:].ravel(), # right band
    ])
    if np.mean(border) > 127:
        return cv.bitwise_not(th)
    return th


def split_wide_box(thresh_roi, box, median_w):
    x, y, w, h = box
    if w < median_w * 1.5:
        return [box]

    roi = thresh_roi[y:y+h, x:x+w]
    v_proj = np.sum(roi, axis=0) / 255

    n_chars = max(2, round(w / median_w))
    split_points = []
    for i in range(1, n_chars):
        expected_x = int(i * w / n_chars)
        search_w = max(3, int(median_w * 0.3))
        left = max(1, expected_x - search_w)
        right = min(w - 1, expected_x + search_w)
        if left < right:
            window = v_proj[left:right]
            best = left + np.argmin(window)
            split_points.append(best)

    split_points = sorted(set(split_points))
    sub_boxes = []
    prev = 0
    for sp in split_points:
        if sp - prev > 3:
            sub_boxes.append((x + prev, y, sp - prev, h))
        prev = sp
    if w - prev > 3:
        sub_boxes.append((x + prev, y, w - prev, h))

    return sub_boxes if len(sub_boxes) > 1 else [box]


def segment_characters(image, output_dir=None, save=False, params=None):
    """Segment characters. `image` can be a file path (str) or a numpy BGR array.
    `params` overrides any keys in DEFAULT_PARAMS."""
    p = {**DEFAULT_PARAMS, **(params or {})}

    if save and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if isinstance(image, str):
        img = cv.imread(image)
        if img is None:
            return None, None, None, None, None
    else:
        img = image

    h_img, w_img = img.shape[:2]

    min_proc_h = max(1, int(p['min_proc_h']))
    scale = max(1.0, min_proc_h / h_img) if h_img < min_proc_h else 1.0
    proc_img = cv.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv.INTER_CUBIC) if scale > 1.0 else img

    gray = cv.cvtColor(proc_img, cv.COLOR_BGR2GRAY)
    grid = max(1, int(p['clahe_grid']))
    clahe = cv.createCLAHE(clipLimit=float(p['clahe_clip']), tileGridSize=(grid, grid))
    gray = clahe.apply(gray)
    bk = max(1, int(p['blur_kernel']))
    if bk % 2 == 0:
        bk += 1
    blurred = cv.GaussianBlur(gray, (bk, bk), 0)
    _, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    thresh = _ensure_fg_white(thresh)
    h_proc = proc_img.shape[0]
    close_k = max(1, int(h_proc * float(p['close_kernel_factor'])))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (close_k, close_k))
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    min_char_height = h_proc * float(p['min_char_h_factor'])
    min_char_width = max(1, int(p['min_char_w']))

    boxes = []
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        if h >= min_char_height and w >= min_char_width:
            boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[0])

    min_overlap_px = max(1, int(h_proc * 0.01))
    merged = []
    for box in boxes:
        if merged and box[0] < merged[-1][0] + merged[-1][2] - min_overlap_px:
            px, py, pw, ph = merged[-1]
            nx = min(px, box[0])
            ny = min(py, box[1])
            nx2 = max(px + pw, box[0] + box[2])
            ny2 = max(py + ph, box[1] + box[3])
            merged[-1] = (nx, ny, nx2 - nx, ny2 - ny)
        else:
            merged.append(box)

    if len(merged) > 0:
        widths = [b[2] for b in merged]
        median_w = float(np.median(widths))
        final_boxes_proc = []
        for box in merged:
            sub = split_wide_box(thresh, box, median_w)
            final_boxes_proc.extend(sub)
    else:
        final_boxes_proc = merged

    if scale > 1.0:
        final_boxes = [(int(x / scale), int(y / scale),
                        max(1, int(w / scale)), max(1, int(h / scale)))
                       for (x, y, w, h) in final_boxes_proc]
        thresh_display = cv.resize(thresh, (w_img, h_img), interpolation=cv.INTER_NEAREST)
    else:
        final_boxes = final_boxes_proc
        thresh_display = thresh

    padding = max(0, int(p['padding']))
    char_imgs = []
    thresh_char_imgs = []
    for i, (x, y, w, h) in enumerate(final_boxes):
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_img, x + w + padding)
        y2 = min(h_img, y + h + padding)
        char_imgs.append(img[y1:y2, x1:x2])
        thresh_char_imgs.append(thresh_display[y1:y2, x1:x2])
        if save and output_dir:
            cv.imwrite(os.path.join(output_dir, f"char_{i}.png"), char_imgs[-1])

    return final_boxes, char_imgs, thresh_char_imgs, img, thresh_display


def deskew_char(thresh_char, max_angle=15):
    coords = np.column_stack(np.where(thresh_char > 0))
    if len(coords) < 10:
        return thresh_char
    angle = cv.minAreaRect(coords[:, ::-1].astype(np.float32))[2]
    if angle < -45:
        angle += 90
    if abs(angle) > max_angle:
        return thresh_char
    h, w = thresh_char.shape
    M = cv.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv.warpAffine(thresh_char, M, (w, h),
                         flags=cv.INTER_NEAREST, borderValue=0)


def tight_crop(thresh):
    coords = np.where(thresh > 0)
    if len(coords[0]) == 0:
        return thresh
    y0, y1 = int(coords[0].min()), int(coords[0].max())
    x0, x1 = int(coords[1].min()), int(coords[1].max())
    return thresh[y0:y1 + 1, x0:x1 + 1]


def fit_to_square(img, size):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size[1], size[0]), dtype=np.uint8)
    scale = min(size[0] / w, size[1] / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv.resize(img, (nw, nh), interpolation=cv.INTER_NEAREST)
    canvas = np.zeros((size[1], size[0]), dtype=np.uint8)
    yo = (size[1] - nh) // 2
    xo = (size[0] - nw) // 2
    canvas[yo:yo + nh, xo:xo + nw] = resized
    return canvas


def _center_by_centroid(mask, size):
    H, W = size[1], size[0]
    canvas = np.zeros((H, W), dtype=np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return canvas
    cy, cx = float(ys.mean()), float(xs.mean())
    h, w = mask.shape
    yo = int(round(H / 2 - cy))
    xo = int(round(W / 2 - cx))
    y1s, y1e = max(0, yo), min(H, yo + h)
    x1s, x1e = max(0, xo), min(W, xo + w)
    y2s, y2e = max(0, -yo), max(0, -yo) + (y1e - y1s)
    x2s, x2e = max(0, -xo), max(0, -xo) + (x1e - x1s)
    if y1e > y1s and x1e > x1s:
        canvas[y1s:y1e, x1s:x1e] = mask[y2s:y2e, x2s:x2e]
    return canvas


def _iou(a, b):
    inter = int(np.count_nonzero((a > 0) & (b > 0)))
    union = int(np.count_nonzero((a > 0) | (b > 0)))
    return inter / union if union > 0 else 0.0


def _shift_mask(mask, dx, dy):
    """Shift a binary mask by (dx, dy), filling exposed area with 0."""
    H, W = mask.shape
    out = np.zeros_like(mask)
    ys, ye = max(0, dy), min(H, H + dy)
    xs, xe = max(0, dx), min(W, W + dx)
    sy, sx = max(0, -dy), max(0, -dx)
    out[ys:ye, xs:xe] = mask[sy:sy + (ye - ys), sx:sx + (xe - xs)]
    return out


def _scale_into_canvas(mask, scale, canvas_shape):
    """Resize a binary mask by `scale` (preserving aspect) and center-place
    into a canvas of `canvas_shape`. Used for the scale-search alignment."""
    H, W = canvas_shape
    h, w = mask.shape
    if h == 0 or w == 0:
        return np.zeros(canvas_shape, dtype=mask.dtype)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    resized = cv.resize(mask, (nw, nh), interpolation=cv.INTER_NEAREST)
    canvas = np.zeros(canvas_shape, dtype=mask.dtype)
    # Center the (possibly larger) resized mask, cropping if it overflows
    yo_dst = max(0, (H - nh) // 2)
    xo_dst = max(0, (W - nw) // 2)
    yo_src = max(0, (nh - H) // 2)
    xo_src = max(0, (nw - W) // 2)
    h_eff = min(nh, H)
    w_eff = min(nw, W)
    canvas[yo_dst:yo_dst + h_eff, xo_dst:xo_dst + w_eff] = \
        resized[yo_src:yo_src + h_eff, xo_src:xo_src + w_eff]
    return canvas


def _best_shift_align(a, b, max_shift=8):
    """
    Find translation (dx, dy) in [-max_shift, +max_shift]² that maximizes
    IoU(a, shift(b, dx, dy)). Returns (b_aligned, dx, dy, best_iou).
    Uses cv2.matchTemplate with binary masks for the heavy lifting:
    correlation peak ≈ best overlap for binary inputs, much faster than a
    Python loop for typical sizes.
    """
    if max_shift <= 0:
        return b, 0, 0, _iou(a, b)

    H, W = a.shape
    pad = int(max_shift)
    # Pad b so we can search a's position within b
    b_padded = cv.copyMakeBorder(b, pad, pad, pad, pad, cv.BORDER_CONSTANT, value=0)

    # Find where the foreground of `a` correlates best with `b_padded`.
    # TM_CCORR (no normalisation) on binary masks ≡ #overlap pixels at each
    # candidate offset, which monotonically tracks IoU when foreground areas
    # don't change with shift (they don't — we just translate).
    a_f = (a > 0).astype(np.float32)
    b_f = (b_padded > 0).astype(np.float32)
    if a_f.sum() == 0 or b_f.sum() == 0:
        return b, 0, 0, 0.0

    res = cv.matchTemplate(b_f, a_f, cv.TM_CCORR)
    _, _, _, max_loc = cv.minMaxLoc(res)
    # max_loc = (x, y) where a's TOP-LEFT best fits in b_padded.
    # Without padding, (0,0) means a starts at (0,0) of b → no shift.
    # With pad, (pad, pad) → no shift. So:
    dx = pad - int(max_loc[0])
    dy = pad - int(max_loc[1])
    b_aligned = _shift_mask(b, dx, dy)
    return b_aligned, dx, dy, _iou(a, b_aligned)


def fit_by_foreground_extent(mask, size, target_fill=0.75):
    """
    Resize so foreground EXTENT (after tight-cropping) fills `target_fill`
    of the canvas, preserving aspect ratio. Background padding around the
    char is IGNORED — makes the comparison robust to inconsistent crop
    tightness between template and target segmentation.

    `size` is (W, H). Returns uint8 canvas of shape (H, W).
    """
    W, H = int(size[0]), int(size[1])
    canvas = np.zeros((H, W), dtype=np.uint8)
    if mask is None or mask.size == 0:
        return canvas

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return canvas

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    fg_h = y1 - y0 + 1
    fg_w = x1 - x0 + 1

    # Scale so the LARGER foreground dim fills target_fill of the corresponding
    # canvas dim. Aspect ratio of foreground is preserved.
    scale_y = (target_fill * H) / fg_h
    scale_x = (target_fill * W) / fg_w
    scale = min(scale_x, scale_y)
    new_h = max(1, int(round(fg_h * scale)))
    new_w = max(1, int(round(fg_w * scale)))

    cropped = mask[y0:y1 + 1, x0:x1 + 1]
    resized = cv.resize(cropped, (new_w, new_h), interpolation=cv.INTER_NEAREST)

    yo = max(0, (H - new_h) // 2)
    xo = max(0, (W - new_w) // 2)
    h_eff = min(new_h, H - yo)
    w_eff = min(new_w, W - xo)
    canvas[yo:yo + h_eff, xo:xo + w_eff] = resized[:h_eff, :w_eff]
    return canvas


def _ecc_affine_align(a, b, motion_type='euclidean', max_iters=50, eps=1e-3):
    """
    Sub-pixel alignment of `b` to `a` using cv2.findTransformECC.
    `motion_type`: 'translation' (2 DOF) | 'euclidean' (4 DOF: tx,ty,rot,scale)
                   | 'affine' (6 DOF). Default euclidean — robust without
                   over-fitting shear that pixel-art chars don't have.

    Returns (b_aligned, success). On non-convergence returns (b, False) so the
    caller can fall back to brute-force align.
    """
    motion_map = {
        'translation': cv.MOTION_TRANSLATION,
        'euclidean': cv.MOTION_EUCLIDEAN,
        'affine': cv.MOTION_AFFINE,
    }
    motion = motion_map.get(motion_type, cv.MOTION_EUCLIDEAN)

    # ECC needs float inputs in [0,1]; mild blur stabilises gradient on binary
    a_f = cv.GaussianBlur((a > 0).astype(np.float32), (3, 3), 0)
    b_f = cv.GaussianBlur((b > 0).astype(np.float32), (3, 3), 0)

    # Need at least some signal in both, else ECC will fail
    if a_f.sum() < 5 or b_f.sum() < 5:
        return b, False

    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _cc, warp = cv.findTransformECC(
            a_f, b_f, warp, motion,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
                      int(max_iters), float(eps)),
            inputMask=None, gaussFiltSize=5,
        )
    except cv.error:
        return b, False

    H, W = a.shape[:2]
    b_aligned = cv.warpAffine(
        b, warp, (W, H),
        flags=cv.INTER_NEAREST + cv.WARP_INVERSE_MAP,
        borderValue=0,
    )
    return b_aligned, True


def _best_scale_shift_align(a, b, max_shift=8, scale_tol=0.15, n_scales=5):
    """
    Joint search over (scale, dx, dy) for shifting+resizing `b` to maximize
    IoU(a, b_transformed). `scale_tol` = 0 falls back to plain shift-only align.
    Returns (b_aligned, scale, dx, dy, iou).
    """
    if scale_tol <= 0 or n_scales <= 1:
        b_aligned, dx, dy, iou = _best_shift_align(a, b, max_shift)
        return b_aligned, 1.0, dx, dy, iou

    best = (b, 1.0, 0, 0, -1.0)
    scales = np.linspace(1.0 - scale_tol, 1.0 + scale_tol, n_scales)
    for s in scales:
        b_scaled = _scale_into_canvas(b, float(s), a.shape)
        b_aligned, dx, dy, iou = _best_shift_align(a, b_scaled, max_shift)
        if iou > best[4]:
            best = (b_aligned, float(s), dx, dy, iou)
    return best


def _normalize_char_thresh(raw, keep_largest_cc=True, min_cc_area_ratio=0.05):
    """Reusable: blur + Otsu (foreground = white via border-based polarity)
    + morph close + connected-component cleanup. Mirrors `_to_thresh_norm`
    inside compute_char_quality so overlay sees the same masks.

    `keep_largest_cc=True` drops every disconnected noise blob (e.g. the
    bottom-edge band shadow that shows up as a horizontal red strip in
    diff overlays)."""
    blurred = cv.GaussianBlur(raw, (5, 5), 0)
    _, th = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    th = _ensure_fg_white(th)
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    th = cv.morphologyEx(th, cv.MORPH_CLOSE, k, iterations=1)
    if keep_largest_cc:
        th = _drop_small_cc(th, keep_largest_only=True)
    elif min_cc_area_ratio > 0:
        th = _drop_small_cc(th, keep_largest_only=False, min_area_ratio=min_cc_area_ratio)
    return th


def compute_diff_overlay(tmpl_thresh_char, tgt_thresh_char, size=(96, 96),
                         max_shift=8, scale_tol=0.15, scale_steps=5,
                         keep_largest_cc=True, min_cc_area_ratio=0.05,
                         extent_target_fill=0.75, use_ecc_align=True):
    """
    Align two char masks (deskew + tight-crop + centroid-center + best-shift)
    and produce a BGR overlay highlighting the diff:
      - White  : pixels present in both (match)
      - Red    (BGR 0,0,255) : present in TEMPLATE only → MISSING IN TARGET (mất nét)
      - Green  (BGR 0,255,0) : present in TARGET only → EXTRA IN TARGET (nét thừa)

    `max_shift` runs a brute-force translation search after centroid centering
    so asymmetric chars (5, 3, J, L…) don't false-FAIL on horizontal/vertical drift.
    """
    a_raw = _normalize_char_thresh(
        tmpl_thresh_char, keep_largest_cc=keep_largest_cc,
        min_cc_area_ratio=min_cc_area_ratio,
    )
    b_raw = _normalize_char_thresh(
        tgt_thresh_char, keep_largest_cc=keep_largest_cc,
        min_cc_area_ratio=min_cc_area_ratio,
    )

    # Smart normalisation: scale by FOREGROUND extent (not bbox), so
    # inconsistent crop tightness between template/target is decoupled from
    # the comparison. Also deskew first so ECC has a near-aligned start.
    a_centered = fit_by_foreground_extent(deskew_char(a_raw), size, target_fill=extent_target_fill)
    b_centered = fit_by_foreground_extent(deskew_char(b_raw), size, target_fill=extent_target_fill)

    scale, dx, dy = 1.0, 0, 0
    aligned_by = 'extent'
    if use_ecc_align:
        # Sub-pixel ECC fine-align (rotation + uniform scale + translation).
        b_ecc, ok = _ecc_affine_align(a_centered, b_centered, motion_type='euclidean')
        if ok:
            b_centered = b_ecc
            aligned_by = 'ecc'
        else:
            # Fallback to brute-force scale+shift if ECC didn't converge
            b_centered, scale, dx, dy, _ = _best_scale_shift_align(
                a_centered, b_centered,
                max_shift=max_shift, scale_tol=scale_tol, n_scales=scale_steps,
            )
            aligned_by = 'brute'
    else:
        b_centered, scale, dx, dy, _ = _best_scale_shift_align(
            a_centered, b_centered,
            max_shift=max_shift, scale_tol=scale_tol, n_scales=scale_steps,
        )
        aligned_by = 'brute'

    overlay = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    both = (a_centered > 0) & (b_centered > 0)
    only_t = (a_centered > 0) & (b_centered == 0)  # template only → missing
    only_g = (a_centered == 0) & (b_centered > 0)  # target only → extra

    overlay[both] = (255, 255, 255)
    overlay[only_t] = (0, 0, 255)   # red = missing in target
    overlay[only_g] = (0, 255, 0)   # green = extra in target

    n_both = int(np.count_nonzero(both))
    n_missing = int(np.count_nonzero(only_t))
    n_extra = int(np.count_nonzero(only_g))
    n_union = n_both + n_missing + n_extra
    iou = n_both / n_union if n_union > 0 else 0.0
    miss_ratio = n_missing / max(1, np.count_nonzero(a_centered))
    extra_ratio = n_extra / max(1, np.count_nonzero(b_centered))

    return overlay, {
        'iou': float(iou),
        'missing_px': n_missing,
        'extra_px': n_extra,
        'miss_ratio': float(miss_ratio),
        'extra_ratio': float(extra_ratio),
        'shift_dx': int(dx),
        'shift_dy': int(dy),
        'scale': float(scale),
        'aligned_by': aligned_by,
    }


def compute_char_quality(tmpl_char, tgt_char, size=(64, 64), params=None):
    p = {**DEFAULT_PARAMS, **(params or {})}
    blur_sigma = float(p['tm_blur_sigma'])
    dilate_k = max(1, int(p['iou_dilate']))
    pixel_dev_tol = max(0.1, float(p['pixel_dev_tol']))
    align_max_shift = max(0, int(p['align_max_shift']))
    align_scale_tol = max(0.0, float(p['align_scale_tol']))
    align_scale_steps = max(1, int(p['align_scale_steps']))
    keep_largest_cc = bool(p['keep_largest_cc'])
    min_cc_area_ratio = max(0.0, float(p['min_cc_area_ratio']))
    extent_target_fill = float(p['extent_target_fill'])
    use_ecc_align = bool(p['use_ecc_align'])

    # Single source of truth for char-level normalisation (see _normalize_char_thresh).
    tmpl_char = _normalize_char_thresh(
        tmpl_char, keep_largest_cc=keep_largest_cc, min_cc_area_ratio=min_cc_area_ratio
    )
    tgt_char = _normalize_char_thresh(
        tgt_char, keep_largest_cc=keep_largest_cc, min_cc_area_ratio=min_cc_area_ratio
    )

    t1 = fit_to_square(deskew_char(tight_crop(tmpl_char)), size)
    g2_deskewed = deskew_char(tight_crop(tgt_char))
    t2_base = fit_to_square(g2_deskewed, size)

    px1 = int(np.count_nonzero(t1))
    px2 = int(np.count_nonzero(t2_base))
    ratio = px2 / (px1 + 1e-6)
    deviation = abs(ratio - 1.0)
    pixel_conf = float(np.clip(1.0 - deviation * (1.0 / pixel_dev_tol), 0.0, 1.0))

    t1_blur = cv.GaussianBlur(t1.astype(np.float32), (0, 0), sigmaX=blur_sigma)
    best_tm = 0.0
    for s_factor in [0.85, 0.92, 1.0, 1.08, 1.15]:
        s = (max(t1.shape[1], int(size[0] * s_factor)),
             max(t1.shape[0], int(size[1] * s_factor)))
        t2 = fit_to_square(g2_deskewed, s)
        t2_blur = cv.GaussianBlur(t2.astype(np.float32), (0, 0), sigmaX=blur_sigma)
        result = cv.matchTemplate(t2_blur, t1_blur, cv.TM_CCOEFF_NORMED)
        best_tm = max(best_tm, float(result.max()))
    blur_tm = float(np.clip(best_tm, 0.0, 1.0))

    # Smart normalisation (matches what compute_diff_overlay shows the user):
    # foreground-extent normalize → ECC fine-align (with brute-force fallback).
    a = fit_by_foreground_extent(t1, size, target_fill=extent_target_fill)
    b = fit_by_foreground_extent(t2_base, size, target_fill=extent_target_fill)
    if use_ecc_align:
        b_ecc, ok = _ecc_affine_align(a, b, motion_type='euclidean')
        if ok:
            b = b_ecc
        else:
            b, _, _, _, _ = _best_scale_shift_align(
                a, b,
                max_shift=align_max_shift,
                scale_tol=align_scale_tol,
                n_scales=align_scale_steps,
            )
    else:
        b, _, _, _, _ = _best_scale_shift_align(
            a, b,
            max_shift=align_max_shift,
            scale_tol=align_scale_tol,
            n_scales=align_scale_steps,
        )
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilate_k, dilate_k))
    a = cv.dilate(a, k, iterations=1)
    b = cv.dilate(b, k, iterations=1)
    iou = _iou(a, b)

    tm_conf = max(blur_tm, iou)

    return {
        "confidence": min(tm_conf, pixel_conf),
        "tm_conf": round(tm_conf, 4),
        "blur_tm": round(blur_tm, 4),
        "iou": round(iou, 4),
        "pixel_conf": round(pixel_conf, 3),
        "px_tmpl": px1,
        "px_tgt": px2,
    }


def build_comparison_strip(tmpl_chars, tgt_chars, results,
                           tmpl_img=None, tmpl_thresh=None,
                           tgt_img=None, tgt_thresh=None,
                           tmpl_thresh_chars=None, tgt_thresh_chars=None,
                           align_max_shift=None,
                           align_scale_tol=None, align_scale_steps=None,
                           keep_largest_cc=None, min_cc_area_ratio=None,
                           extent_target_fill=None, use_ecc_align=None):
    """Build the visual comparison strip and return as a BGR numpy array.
    If `tmpl_thresh_chars` and `tgt_thresh_chars` are provided, an additional
    row of mask-diff overlays (white=match, red=missing, green=extra) is shown."""
    n = len(results)
    if n == 0:
        return None

    cell_w, cell_h = 80, 80
    gap = 4
    score_h = 40
    strip_w = n * (cell_w + gap) + gap

    orig_h = 120
    has_orig = tmpl_img is not None
    orig_section_h = (orig_h + gap) * 2 + gap if has_orig else 0

    # Optional 3rd row: mask diff overlay
    has_diff = tmpl_thresh_chars is not None and tgt_thresh_chars is not None
    diff_rows = 1 if has_diff else 0
    char_strip_h = (2 + diff_rows) * (cell_h + gap) + score_h + gap * 2

    total_h = orig_section_h + char_strip_h
    total_w = strip_w

    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 50

    def place_image(src, y_start, x_start, w, h):
        if src is None:
            return
        img_bgr = cv.cvtColor(src, cv.COLOR_GRAY2BGR) if len(src.shape) == 2 else src
        ih, iw = img_bgr.shape[:2]
        scale = w / iw
        nw = w
        nh = min(h, max(1, int(ih * scale)))
        resized = cv.resize(img_bgr, (nw, nh))
        canvas[y_start:y_start + nh, x_start:x_start + nw] = resized

    if has_orig:
        half_w = strip_w // 2
        y0 = gap
        place_image(tmpl_img, y0, 0, half_w, orig_h)
        place_image(tgt_img, y0, half_w, half_w, orig_h)
        cv.line(canvas, (half_w, y0), (half_w, y0 + orig_h), (120, 120, 120), 1)
        cv.putText(canvas, "ORIG TEMPLATE", (4, y0 + 12),
                   cv.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv.putText(canvas, "ORIG TARGET", (half_w + 4, y0 + 12),
                   cv.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        y1 = y0 + orig_h + gap
        place_image(tmpl_thresh, y1, 0, half_w, orig_h)
        place_image(tgt_thresh, y1, half_w, half_w, orig_h)
        cv.line(canvas, (half_w, y1), (half_w, y1 + orig_h), (120, 120, 120), 1)
        cv.putText(canvas, "THRESH TEMPLATE", (4, y1 + 12),
                   cv.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv.putText(canvas, "THRESH TARGET", (half_w + 4, y1 + 12),
                   cv.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    def fit_char(src, w, h):
        ih, iw = src.shape[:2]
        scale = min(w / iw, h / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = cv.resize(src, (nw, nh))
        cell = np.ones((h, w, 3), dtype=np.uint8) * 50
        yo = (h - nh) // 2
        xo = (w - nw) // 2
        cell[yo:yo + nh, xo:xo + nw] = resized
        return cell

    y_base = orig_section_h
    diff_y0 = y_base + gap * 3 + cell_h * 2  # only used if has_diff
    for idx, (i, metrics, quality) in enumerate(results):
        x_off = gap + idx * (cell_w + gap)

        t = fit_char(tmpl_chars[i], cell_w, cell_h)
        canvas[y_base + gap:y_base + gap + cell_h, x_off:x_off + cell_w] = t

        tgt = fit_char(tgt_chars[i], cell_w, cell_h)
        canvas[y_base + gap * 2 + cell_h:y_base + gap * 2 + cell_h * 2, x_off:x_off + cell_w] = tgt

        is_pass = quality == "PASS"
        color = (0, 220, 0) if is_pass else (0, 0, 255)

        tgt_y0 = y_base + gap * 2 + cell_h
        tgt_y1 = tgt_y0 + cell_h
        cv.rectangle(canvas, (x_off, tgt_y0), (x_off + cell_w, tgt_y1), color, 2)

        # Mask diff row
        if has_diff:
            try:
                shift_arg = (
                    int(DEFAULT_PARAMS['align_max_shift'])
                    if align_max_shift is None
                    else int(align_max_shift)
                )
                stol = (DEFAULT_PARAMS['align_scale_tol']
                        if align_scale_tol is None else float(align_scale_tol))
                ssteps = (DEFAULT_PARAMS['align_scale_steps']
                          if align_scale_steps is None else int(align_scale_steps))
                klc = (bool(DEFAULT_PARAMS['keep_largest_cc'])
                       if keep_largest_cc is None else bool(keep_largest_cc))
                mar = (float(DEFAULT_PARAMS['min_cc_area_ratio'])
                       if min_cc_area_ratio is None else float(min_cc_area_ratio))
                etf = (float(DEFAULT_PARAMS['extent_target_fill'])
                       if extent_target_fill is None else float(extent_target_fill))
                uea = (bool(DEFAULT_PARAMS['use_ecc_align'])
                       if use_ecc_align is None else bool(use_ecc_align))
                overlay, _diff_stats = compute_diff_overlay(
                    tmpl_thresh_chars[i], tgt_thresh_chars[i],
                    size=(cell_w, cell_h), max_shift=shift_arg,
                    scale_tol=stol, scale_steps=ssteps,
                    keep_largest_cc=klc, min_cc_area_ratio=mar,
                    extent_target_fill=etf, use_ecc_align=uea,
                )
            except Exception:
                overlay = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            canvas[diff_y0:diff_y0 + cell_h, x_off:x_off + cell_w] = overlay

        y_text = (diff_y0 + cell_h + gap) if has_diff else (y_base + gap * 3 + cell_h * 2)

        cv.putText(canvas, f"{metrics['confidence']:.2f}", (x_off + 2, y_text + 14),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv.putText(canvas, quality, (x_off + 2, y_text + 28),
                   cv.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    cv.putText(canvas, "TEMPLATE", (2, y_base + gap + 12),
               cv.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv.putText(canvas, "TARGET", (2, y_base + gap * 2 + cell_h + 12),
               cv.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    if has_diff:
        cv.putText(canvas, "DIFF (R=miss, G=extra)", (2, diff_y0 + 12),
                   cv.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    overall_pass = all(r[2] == "PASS" for r in results)
    banner_color = (0, 180, 0) if overall_pass else (0, 0, 220)
    banner_label = "PASS" if overall_pass else "FAIL"
    banner_h = 28
    banner = np.ones((banner_h, total_w, 3), dtype=np.uint8)
    banner[:] = banner_color
    text_size = cv.getTextSize(banner_label, cv.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
    tx = (total_w - text_size[0]) // 2
    cv.putText(banner, banner_label, (tx, 21), cv.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    return np.vstack([canvas, banner])


def compare_arrays(tmpl_arr, tgt_arr, params=None):
    """
    Run segmentation + per-character comparison on two BGR arrays.
    Returns (comparison_image_bgr, results_list, overall_pass).
    `results_list` is [(idx, metrics_dict, "PASS"|"FAIL"), ...].
    Returns (None, [], False) if either side has no segmentable chars.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    pass_threshold = float(p['pass_threshold'])
    compare_size = max(8, int(p['compare_size']))

    _, tmpl_chars, tmpl_thresh_chars, tmpl_img, tmpl_thresh = segment_characters(tmpl_arr, params=p)
    _, tgt_chars, tgt_thresh_chars, tgt_img, tgt_thresh = segment_characters(tgt_arr, params=p)

    if not tmpl_chars or not tgt_chars:
        return None, [], False

    n_pairs = min(len(tmpl_chars), len(tgt_chars))
    results = []
    for i in range(n_pairs):
        metrics = compute_char_quality(
            tmpl_thresh_chars[i], tgt_thresh_chars[i],
            size=(compare_size, compare_size), params=p,
        )
        quality = "PASS" if metrics["confidence"] >= pass_threshold else "FAIL"
        results.append((i, metrics, quality))

    overall_pass = (
        len(tmpl_chars) == len(tgt_chars)
        and all(r[2] == "PASS" for r in results)
    )

    strip = build_comparison_strip(
        tmpl_chars, tgt_chars, results,
        tmpl_img, tmpl_thresh, tgt_img, tgt_thresh,
        tmpl_thresh_chars=tmpl_thresh_chars,
        tgt_thresh_chars=tgt_thresh_chars,
        align_max_shift=p['align_max_shift'],
        align_scale_tol=p['align_scale_tol'],
        align_scale_steps=p['align_scale_steps'],
        keep_largest_cc=p['keep_largest_cc'],
        min_cc_area_ratio=p['min_cc_area_ratio'],
        extent_target_fill=p['extent_target_fill'],
        use_ecc_align=p['use_ecc_align'],
    )
    return strip, results, overall_pass


def compare_char_pairs(char_pairs, params=None):
    """
    Compare a pre-paired list of (template_char_bgr, target_char_bgr) crops —
    used when annotation provides explicit char bboxes per text region, so
    `segment_characters` is bypassed entirely. Char count match is guaranteed
    by construction.

    `char_pairs` = [(tmpl_bgr, tgt_bgr), ...]

    Returns the same structure as compare_arrays_full so the rest of the
    pipeline (UI strip, scoring) doesn't care which path produced the pairs:
        {strip, results, overall_pass, tmpl_chars, tgt_chars,
         tmpl_thresh_chars, tgt_thresh_chars}
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    pass_threshold = float(p['pass_threshold'])
    compare_size = max(8, int(p['compare_size']))
    keep_largest_cc = bool(p['keep_largest_cc'])
    min_cc_area_ratio = max(0.0, float(p['min_cc_area_ratio']))

    if not char_pairs:
        return {
            'strip': None, 'results': [], 'overall_pass': False,
            'tmpl_chars': [], 'tgt_chars': [],
            'tmpl_thresh_chars': [], 'tgt_thresh_chars': [],
        }

    # Threshold each crop for the per-char comparison + diff overlay
    tmpl_chars = []
    tgt_chars = []
    tmpl_thresh_chars = []
    tgt_thresh_chars = []

    for tmpl_bgr, tgt_bgr in char_pairs:
        tmpl_gray = cv.cvtColor(tmpl_bgr, cv.COLOR_BGR2GRAY) \
            if tmpl_bgr.ndim == 3 else tmpl_bgr
        tgt_gray = cv.cvtColor(tgt_bgr, cv.COLOR_BGR2GRAY) \
            if tgt_bgr.ndim == 3 else tgt_bgr
        tmpl_th = _normalize_char_thresh(
            tmpl_gray, keep_largest_cc=keep_largest_cc,
            min_cc_area_ratio=min_cc_area_ratio,
        )
        tgt_th = _normalize_char_thresh(
            tgt_gray, keep_largest_cc=keep_largest_cc,
            min_cc_area_ratio=min_cc_area_ratio,
        )
        tmpl_chars.append(tmpl_bgr)
        tgt_chars.append(tgt_bgr)
        tmpl_thresh_chars.append(tmpl_th)
        tgt_thresh_chars.append(tgt_th)

    results = []
    for i in range(len(char_pairs)):
        metrics = compute_char_quality(
            tmpl_thresh_chars[i], tgt_thresh_chars[i],
            size=(compare_size, compare_size), params=p,
        )
        quality = "PASS" if metrics["confidence"] >= pass_threshold else "FAIL"
        results.append((i, metrics, quality))

    # Char-count is identical by construction — overall pass is just
    # "every char passed".
    overall_pass = all(r[2] == "PASS" for r in results)

    strip = build_comparison_strip(
        tmpl_chars, tgt_chars, results,
        tmpl_img=None, tmpl_thresh=None, tgt_img=None, tgt_thresh=None,
        tmpl_thresh_chars=tmpl_thresh_chars,
        tgt_thresh_chars=tgt_thresh_chars,
        align_max_shift=p['align_max_shift'],
        align_scale_tol=p['align_scale_tol'],
        align_scale_steps=p['align_scale_steps'],
        keep_largest_cc=p['keep_largest_cc'],
        min_cc_area_ratio=p['min_cc_area_ratio'],
        extent_target_fill=p['extent_target_fill'],
        use_ecc_align=p['use_ecc_align'],
    )
    return {
        'strip': strip,
        'results': results,
        'overall_pass': overall_pass,
        'tmpl_chars': tmpl_chars,
        'tgt_chars': tgt_chars,
        'tmpl_thresh_chars': tmpl_thresh_chars,
        'tgt_thresh_chars': tgt_thresh_chars,
    }


def compare_arrays_full(tmpl_arr, tgt_arr, params=None):
    """
    Same as `compare_arrays` but additionally returns the per-character crops
    so the caller (e.g. tune dialog) can drive a chooser without re-segmenting.
    Returns:
        {
          'strip': np.ndarray | None,
          'results': [...],
          'overall_pass': bool,
          'tmpl_chars':         [bgr...],
          'tgt_chars':          [bgr...],
          'tmpl_thresh_chars':  [gray-mask...],
          'tgt_thresh_chars':   [gray-mask...],
        }
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    pass_threshold = float(p['pass_threshold'])
    compare_size = max(8, int(p['compare_size']))

    _, tmpl_chars, tmpl_thresh_chars, tmpl_img, tmpl_thresh = segment_characters(tmpl_arr, params=p)
    _, tgt_chars, tgt_thresh_chars, tgt_img, tgt_thresh = segment_characters(tgt_arr, params=p)

    if not tmpl_chars or not tgt_chars:
        return {
            'strip': None, 'results': [], 'overall_pass': False,
            'tmpl_chars': tmpl_chars or [], 'tgt_chars': tgt_chars or [],
            'tmpl_thresh_chars': tmpl_thresh_chars or [],
            'tgt_thresh_chars': tgt_thresh_chars or [],
        }

    n_pairs = min(len(tmpl_chars), len(tgt_chars))
    results = []
    for i in range(n_pairs):
        metrics = compute_char_quality(
            tmpl_thresh_chars[i], tgt_thresh_chars[i],
            size=(compare_size, compare_size), params=p,
        )
        quality = "PASS" if metrics["confidence"] >= pass_threshold else "FAIL"
        results.append((i, metrics, quality))

    overall_pass = (
        len(tmpl_chars) == len(tgt_chars)
        and all(r[2] == "PASS" for r in results)
    )

    strip = build_comparison_strip(
        tmpl_chars, tgt_chars, results,
        tmpl_img, tmpl_thresh, tgt_img, tgt_thresh,
        tmpl_thresh_chars=tmpl_thresh_chars,
        tgt_thresh_chars=tgt_thresh_chars,
        align_max_shift=p['align_max_shift'],
        align_scale_tol=p['align_scale_tol'],
        align_scale_steps=p['align_scale_steps'],
        keep_largest_cc=p['keep_largest_cc'],
        min_cc_area_ratio=p['min_cc_area_ratio'],
        extent_target_fill=p['extent_target_fill'],
        use_ecc_align=p['use_ecc_align'],
    )
    return {
        'strip': strip,
        'results': results,
        'overall_pass': overall_pass,
        'tmpl_chars': tmpl_chars,
        'tgt_chars': tgt_chars,
        'tmpl_thresh_chars': tmpl_thresh_chars,
        'tgt_thresh_chars': tgt_thresh_chars,
    }
