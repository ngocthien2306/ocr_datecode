"""char_quality_v3 — fix v1 bugs found via visualization.

Changes from v1:
  - DROP CLAHE (it bombed bg paper texture into Otsu → bad bbox)
  - Use OLD-style _to_thresh_norm (proven) for binary mask + bbox extraction
  - Crop+resize GRAYSCALE using the SAME bbox as binary mask
  - ECC TRANSLATION only (rotation hiếm, gây fail), with safety bounds
  - Subtract noise floor before defect penalty
  - Cap penalty at 0.7 → NCC vẫn dominate
  - Bound over/under to [0, 1]
"""

from typing import Dict, Tuple

import cv2
import numpy as np

SIZE = 64
DILATE_PX = 3
NOISE_FLOOR_OVER  = 0.10   # natural binarize diff on OK pairs
NOISE_FLOOR_UNDER = 0.08
MAX_PENALTY = 0.70         # confidence cap loss from defects
THR_LEM  = 0.25            # over_ink threshold to flag NG-LEM
THR_DROP = 0.20            # under_ink threshold to flag NG-MẤT NÉT
PAD_Y_DEFAULT = 4          # extra padding top+bottom around Otsu bbox (recover ascender/descender)
PAD_X_DEFAULT = 1          # extra padding left+right (small to avoid fragment leak)


def _binarize_norm(g: np.ndarray) -> np.ndarray:
    """Same recipe as old _to_thresh_norm. Reliable on raw grayscale."""
    blurred = cv2.GaussianBlur(g, (5, 5), 0)
    _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.mean(th) > 127:
        th = cv2.bitwise_not(th)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)


def _keep_largest_cc(
    mask: np.ndarray,
    dilate_iter: int = 2,
    accent_max_ratio: float = 0.30,
    accent_col_overlap: float = 0.50,
    accent_max_dy: int = 15,
) -> np.ndarray:
    """Giữ main body (largest CC) + accent nhỏ có column-overlap với body.
    Drop true fragments (kế bên, không column-overlap).

    Workflow:
      1. Dilate `dilate_iter` lần để merge các piece sát nhau.
      2. Find largest CC = main body.
      3. Với mỗi CC khác: keep nếu (area ≤ ratio*body) AND (column overlap ≥ X)
         AND (vertical distance ≤ Y) — pattern của accent (dấu i/j, : ,!).
      4. Drop CC còn lại = fragment thật (chữ kế bên leak vào).
      5. AND ngược về mask gốc → stroke width nguyên.

    Args:
        accent_max_ratio: max area ratio so với largest (>0.30 → fragment, không phải accent)
        accent_col_overlap: min cột mà accent overlap với body để được giữ
        accent_max_dy: max khoảng cách dọc giữa accent và body
    """
    if mask.size == 0 or int(np.count_nonzero(mask)) == 0:
        return mask
    if dilate_iter > 0:
        kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        merged = cv2.dilate(mask, kk, iterations=dilate_iter)
    else:
        merged = mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    if n <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest_idx = 1 + int(np.argmax(areas))
    biggest_area = int(areas.max())

    body_region = (labels == biggest_idx)
    body_cols = body_region.any(axis=0)
    body_rows = np.where(body_region.any(axis=1))[0]
    if len(body_rows) > 0:
        body_y0, body_y1 = int(body_rows.min()), int(body_rows.max())
    else:
        body_y0 = body_y1 = 0

    keep = body_region.copy()
    for cc_idx in range(1, n):
        if cc_idx == biggest_idx:
            continue
        cc_area = int(stats[cc_idx, cv2.CC_STAT_AREA])
        if cc_area > accent_max_ratio * biggest_area:
            continue  # CC quá lớn → fragment chứ không phải accent
        cc_region = (labels == cc_idx)
        cc_cols = cc_region.any(axis=0)
        if cc_cols.sum() == 0:
            continue
        col_overlap = float((cc_cols & body_cols).sum()) / float(cc_cols.sum())
        if col_overlap < accent_col_overlap:
            continue  # nằm ở cột khác hẳn body → fragment
        cc_rows = np.where(cc_region.any(axis=1))[0]
        if len(cc_rows) == 0:
            continue
        cc_y0, cc_y1 = int(cc_rows.min()), int(cc_rows.max())
        dy = min(abs(cc_y0 - body_y1), abs(cc_y1 - body_y0))
        if dy > accent_max_dy:
            continue  # quá xa body dọc → fragment
        keep |= cc_region  # is accent → keep

    keep_mask = keep.astype(np.uint8) * 255
    return cv2.bitwise_and(mask, keep_mask)


def _crop_resize(g: np.ndarray, size: int,
                 pad_y: int = PAD_Y_DEFAULT,
                 pad_x: int = PAD_X_DEFAULT,
                 clean_fragments: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Tight-crop foreground (Otsu bbox) → resize keep-aspect → pad with bg median.
    Returns (grayscale_canvas, binary_canvas) both size×size.

    pad_y / pad_x: padding (px) quanh Otsu bbox trước khi resize.
    clean_fragments: nếu True, dilate-then-largest-CC để loại fragment kế bên,
                     noise, trước khi tính bbox. Stroke width của chữ chính giữ nguyên."""
    if g is None or g.size == 0:
        return np.full((size, size), 200, np.uint8), np.zeros((size, size), np.uint8)
    b = _binarize_norm(g)
    if clean_fragments:
        b = _keep_largest_cc(b, dilate_iter=2)
    coords = np.where(b > 0)
    if len(coords[0]) == 0:
        gr = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
        return gr, np.zeros((size, size), np.uint8)
    y0, y1 = int(coords[0].min()), int(coords[0].max())
    x0, x1 = int(coords[1].min()), int(coords[1].max())
    # Configurable padding around bbox (vertical can be larger to recover ascender/descender)
    y0 = max(0, y0 - pad_y); y1 = min(g.shape[0] - 1, y1 + pad_y)
    x0 = max(0, x0 - pad_x); x1 = min(g.shape[1] - 1, x1 + pad_x)
    gc = g[y0:y1 + 1, x0:x1 + 1]
    bc = b[y0:y1 + 1, x0:x1 + 1]
    h, w = gc.shape
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    rg = cv2.resize(gc, (nw, nh), interpolation=cv2.INTER_AREA)
    rb = cv2.resize(bc, (nw, nh), interpolation=cv2.INTER_NEAREST)
    bg_pixels = g[b == 0]
    bg = int(np.median(bg_pixels)) if bg_pixels.size > 50 else 200
    cg = np.full((size, size), bg, np.uint8)
    cb = np.zeros((size, size), np.uint8)
    yo, xo = (size - nh) // 2, (size - nw) // 2
    cg[yo:yo + nh, xo:xo + nw] = rg
    cb[yo:yo + nh, xo:xo + nw] = rb
    return cg, cb


def _align_translation(t: np.ndarray, g: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """ECC translation-only alignment with safety bounds."""
    try:
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-3)
        cc, warp = cv2.findTransformECC(t, g, warp, cv2.MOTION_TRANSLATION, criteria, None, 3)
        if not np.isfinite(warp).all() or abs(warp[0, 2]) > size / 4 or abs(warp[1, 2]) > size / 4:
            return g, None, 0.0, False
        return warp, cc
    except cv2.error:
        return None, 0.0


def compute_char_quality_v3(
    tmpl_gray: np.ndarray,
    tgt_gray: np.ndarray,
    size: int = SIZE,
    dilate_px: int = DILATE_PX,
    pad_y: int = PAD_Y_DEFAULT,
    pad_x: int = PAD_X_DEFAULT,
    clean_fragments: bool = True,
) -> Dict:
    t_g, t_b = _crop_resize(tmpl_gray, size, pad_y=pad_y, pad_x=pad_x, clean_fragments=clean_fragments)
    g_g, g_b = _crop_resize(tgt_gray,  size, pad_y=pad_y, pad_x=pad_x, clean_fragments=clean_fragments)

    # ECC translation-only on grayscale
    cc = 0.0
    motion = 'identity'
    try:
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-3)
        cc, warp = cv2.findTransformECC(t_g, g_g, warp, cv2.MOTION_TRANSLATION, criteria, None, 3)
        if np.isfinite(warp).all() and abs(warp[0, 2]) <= size / 4 and abs(warp[1, 2]) <= size / 4:
            g_g = cv2.warpAffine(
                g_g, warp, (size, size),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
            g_b = cv2.warpAffine(
                g_b, warp, (size, size),
                flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            motion = 'translation'
        else:
            cc = 0.0
    except cv2.error:
        pass

    # NCC on aligned grayscale — primary similarity
    res = cv2.matchTemplate(g_g, t_g, cv2.TM_CCOEFF_NORMED)
    ncc = float(np.clip(res.max(), 0.0, 1.0))

    # Directional diff on PRE-aligned binary masks (no fresh Otsu)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    t_dil = cv2.dilate(t_b, k)
    g_dil = cv2.dilate(g_b, k)

    extra   = cv2.bitwise_and(g_b, cv2.bitwise_not(t_dil))   # target có ink ngoài template = LEM
    missing = cv2.bitwise_and(t_b, cv2.bitwise_not(g_dil))   # template có ink ngoài target = MẤT NÉT

    t_px = max(1, int(np.count_nonzero(t_b)))
    over  = min(1.0, int(np.count_nonzero(extra))   / t_px)
    under = min(1.0, int(np.count_nonzero(missing)) / t_px)

    over_eff  = max(0.0, over  - NOISE_FLOOR_OVER)
    under_eff = max(0.0, under - NOISE_FLOOR_UNDER)
    defect_pen = min(MAX_PENALTY, over_eff + under_eff)
    confidence = ncc * (1.0 - defect_pen)

    if over > THR_LEM and over > under:
        defect_type = 'over_ink'
    elif under > THR_DROP and under > over:
        defect_type = 'under_ink'
    else:
        defect_type = None

    return {
        'confidence': float(confidence),
        'ncc': float(ncc),
        'ecc_cc': float(cc),
        'motion': motion,
        'over_ink_score':  float(over),
        'under_ink_score': float(under),
        'over_effective':  float(over_eff),
        'under_effective': float(under_eff),
        'defect_penalty':  float(defect_pen),
        'defect_type': defect_type,
        '_t_prep': t_g,
        '_g_aligned': g_g,
        '_t_bin': t_b,
        '_g_bin': g_b,
        '_extra_ink': extra,
        '_missing_ink': missing,
    }
