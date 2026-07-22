"""
Image Processing-based product (bottle) edge detection.

NOTE: Trong UI / recipe option này gọi là "yolo_segment" cho consistency với
naming convention, NHƯNG thực chất KHÔNG dùng model YOLO. Đây là phương pháp
xử lý ảnh thuần (Sobel + outer-anchored peak detection) để tìm cạnh chai.

Pipeline:
1. Lấy label polygon từ transformed_bboxes (SuperPoint matching) — đã có sẵn
2. Build rotated search strip 2 bên label. Product detection quét CẢ vào trong
   mép label (INNER_SEARCH_MAX px) → bắt được cạnh chai khi label lệch mạnh
   (gap có thể âm). Template detection vẫn outward-only (inner_search=0).
3. Fill label đen middle 60% theo Y (top/bot 20% giữ để bắt cạnh khi label che);
   product path co fill vào trong x_inset=INNER_SEARCH_MAX để mép đen không tạo
   peak giả trong cửa sổ inner search.
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
from dataclasses import dataclass, fields
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)


# ─── Tuned defaults (xem test_alignment.py để tweak) ─────────────────────────
OUTER_SEARCH_MAX  = 150
# Quét THÊM vào PHÍA TRONG mép label (px). Cần khi label lệch mạnh, cạnh chai
# thật rơi vào trong mép label → search outward-only sẽ miss và box bị ghim về
# sát label. Chỉ áp cho product detection; template detection vẫn dùng 0.
INNER_SEARCH_MAX  = 80
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


# ─── Tunable params (per-template edge_config) ───────────────────────────────
@dataclass
class EdgeParams:
    """Per-template edge-detection tuning. Defaults == module globals above so
    callers that don't pass params keep the original behaviour 1:1.

    Persisted into `TemplateImage.edge_config` from the FE EdgeSetupModal and
    rebuilt at inference via `EdgeParams.from_config(edge_config)`.
    """
    outer_search_max: float = OUTER_SEARCH_MAX
    inner_search_max: float = INNER_SEARCH_MAX
    edge_margin:      float = EDGE_MARGIN
    y_extension:      float = Y_EXTENSION
    fill_keep_ratio:  float = FILL_KEEP_RATIO
    peak_height:      float = PEAK_HEIGHT
    peak_prom:        float = PEAK_PROM
    peak_dist:        int   = PEAK_DIST
    strong_thr:       float = STRONG_THR
    outer_min_hratio: float = OUTER_MIN_HRATIO
    inner_min_hratio: float = INNER_MIN_HRATIO
    inner_tol_px:     int   = INNER_TOL_PX
    specular_thr:     int   = SPECULAR_THR
    # Detection signal:
    #   'gradient'   = |Scharr X| edge strength (default) + specular suppression
    #   'brightness' = SIGNED Scharr X edge of a chosen polarity: keeps only
    #                  edges whose dark/bright direction matches `edge_polarity`
    #                  while scanning outward. Specular suppression is applied
    #                  here too.
    detect_mode:      str   = "gradient"
    # Edge polarity for brightness mode (ignored by gradient):
    #   'light_to_dark' = bright→dark when scanning OUTWARD (bright bottle rim
    #                     against a darker background — the default product case)
    #   'dark_to_light' = dark→bright outward (inverted-contrast products)
    edge_polarity:    str   = "light_to_dark"
    # Which qualifying peak becomes the OUTER wall:
    #   'farthest'  = outermost qualifying peak (default, original behaviour)
    #   'nearest'   = innermost qualifying peak (closest to the label)
    #   'strongest' = highest-profile qualifying peak
    find_by:          str   = "farthest"
    # Expected edge transition width in px. Sets the 1D profile smoothing as
    # sigma = edge_width / 2 (→ 1.5 at the default of 3).
    edge_width:       float = 3.0

    @classmethod
    def from_config(cls, cfg: Optional[Dict[str, Any]]) -> "EdgeParams":
        """Build from an edge_config dict, ignoring unknown keys (e.g.
        'template_walls', 'wall_type'). Missing keys fall back to defaults."""
        if not cfg:
            return cls()
        known = {f.name for f in fields(cls)}
        kwargs = {k: cfg[k] for k in known if cfg.get(k) is not None}
        try:
            return cls(**kwargs)
        except (TypeError, ValueError) as e:
            logger.warning(f"EdgeParams.from_config invalid ({e}); using defaults")
            return cls()


DEFAULT_EDGE_PARAMS = EdgeParams()


# ─── Geometric helpers ───────────────────────────────────────────────────────
def fill_label_black(
    img_bgr: np.ndarray, label_pts,
    keep_ratio: float = FILL_KEEP_RATIO, x_inset: float = 0.0,
) -> np.ndarray:
    """Fill middle keep_ratio% theo Y label (top/bot (1-keep)/2 giữ pixel gốc).

    x_inset: co vùng fill VÀO TRONG theo trục X mỗi bên (px). Cần khi bật inner
    search — nếu không, mép đen↔nền tại biên label sẽ tạo peak giả full-height
    ngay tại gap≈0 và lấn át cạnh chai thật nằm trong mép label. Inset đẩy mép
    đen ra ngoài cửa sổ search nên artifact biến mất. Mặc định 0 = không co
    (giữ nguyên hành vi cho template detection / outward-only).
    """
    pts = np.asarray(label_pts, dtype=np.float32)
    TL, TR, BR, BL = pts[0], pts[1], pts[2], pts[3]
    y_skip = (1.0 - keep_ratio) / 2
    new_TL = TL + y_skip * (BL - TL)
    new_BL = BL - y_skip * (BL - TL)
    new_TR = TR + y_skip * (BR - TR)
    new_BR = BR - y_skip * (BR - TR)
    if x_inset > 0:
        top_w = float(np.linalg.norm(TR - TL)) or 1.0
        bot_w = float(np.linalg.norm(BR - BL)) or 1.0
        # Clamp để không lật ngược quad ở label hẹp (giữ ≥10% lõi giữa).
        ins_top = min(x_inset, 0.45 * top_w)
        ins_bot = min(x_inset, 0.45 * bot_w)
        dir_top = (TR - TL) / top_w
        dir_bot = (BR - BL) / bot_w
        new_TL = new_TL + ins_top * dir_top
        new_TR = new_TR - ins_top * dir_top
        new_BL = new_BL + ins_bot * dir_bot
        new_BR = new_BR - ins_bot * dir_bot
    inner_quad = np.array([new_TL, new_TR, new_BR, new_BL], dtype=np.int32)
    out = img_bgr.copy()
    cv2.fillPoly(out, [inner_quad.reshape(-1, 1, 2)], (0, 0, 0))
    return out


def _compute_strip_profile(
    gray: np.ndarray, label_pts, side: str,
    search_w: Optional[float] = None,
    inner_search: float = 0.0,
    params: EdgeParams = DEFAULT_EDGE_PARAMS,
) -> Optional[Dict[str, Any]]:
    """Build search strip 2 bên label + Sobel profile.

    inner_search: px quét vào PHÍA TRONG mép label (gap âm). 0 = chỉ outward
    (cột 0 bắt đầu ở +edge_margin, hành vi cũ). >0 → strip bắt đầu từ
    gap = edge_margin - inner_search (âm). Peak→gap qua 'gap_offset':
    gap = peak_col + gap_offset.
    """
    if search_w is None:
        search_w = params.outer_search_max
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

    y_ext = params.y_extension * elen
    p_top_ext = p_top - y_ext * edge_dir
    p_bot_ext = p_bot + y_ext * edge_dir
    h = int(round(elen * (1.0 + 2 * params.y_extension)))
    near_edge = params.edge_margin - inner_search   # gap tại cột 0 (âm nếu inner_search>0)
    far_edge  = search_w                     # gap tại cột w
    w = int(far_edge - near_edge)            # 1px/cột → gap = col + near_edge
    if h < 50 or w < 30:
        return None

    inner_top = p_top_ext + near_edge * perp
    outer_top = p_top_ext + far_edge  * perp
    outer_bot = p_bot_ext + far_edge  * perp
    inner_bot = p_bot_ext + near_edge * perp
    src = np.array([inner_top, outer_top, outer_bot, inner_bot], dtype=np.float32)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    strip = cv2.warpPerspective(
        gray, M, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )

    # ── Build the 1D detection signal (2D map, columns = gap from label) ──────
    # Columns increase OUTWARD (away from the label), so a SIGNED Scharr X gives
    # the dark/bright transition direction along the search direction.
    sobel = cv2.Scharr(strip.astype(np.float32), cv2.CV_32F, 1, 0)
    if (params.detect_mode or "gradient").lower() == "brightness":
        # Brightness mode (polarity-aware): keep only edges of the chosen
        # direction. 'light_to_dark' = bright→dark outward (intensity decreases →
        # negative Scharr), the bright-rim→background silhouette of a backlit
        # bottle. 'dark_to_light' = the opposite, for inverted-contrast products.
        if (params.edge_polarity or "light_to_dark").lower() == "dark_to_light":
            signal = np.maximum(sobel, 0.0)
        else:
            signal = np.maximum(-sobel, 0.0)
    else:
        # Gradient mode (default): unsigned horizontal edge strength.
        signal = np.abs(sobel)
    # Specular suppression (BOTH modes): zero-out large bright blobs that create
    # spurious edges (window light / glare on the glass). `spec_mask` (strip
    # space) is kept so callers can map it back to frame coords for the UI.
    spec_mask = None
    bright_mask = (strip > params.specular_thr).astype(np.uint8)
    if bright_mask.sum() > 100:
        bright_blobs = cv2.erode(
            bright_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        )
        if bright_blobs.sum() > 50:
            bright_dilated = cv2.dilate(
                bright_blobs, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            )
            signal = signal * (1 - bright_dilated)
            spec_mask = bright_dilated

    gmax = float(signal.max()) + 1e-6
    strong = (signal > params.strong_thr * gmax).sum(axis=0)
    height_ratio = strong / float(h)
    # np.percentile partition GIỮ GIL suốt (numpy sort không nhả GIL) — với strip
    # lớn nó chặn mọi thread khác (capture timer, OCR) vài ms. p95 chỉ là
    # normalizer robust nên strip to thì subsample stride-2 (lệch <0.5%).
    _sig_for_p95 = signal[::2, ::2] if signal.size > 65536 else signal
    robust_max = float(np.percentile(_sig_for_p95, 95)) + 1e-6
    strong_robust = (signal > params.strong_thr * robust_max).sum(axis=0)
    height_ratio_robust = np.minimum(strong_robust / float(h), 1.0)

    profile = signal.sum(axis=0)
    sigma = max(0.3, float(params.edge_width) / 2.0)   # edge transition width
    profile = gaussian_filter1d(profile, sigma=sigma)
    pmax = profile.max() + 1e-6
    profile_n = profile / pmax

    peaks, _ = find_peaks(
        profile_n, height=params.peak_height,
        distance=params.peak_dist, prominence=params.peak_prom,
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
        'gap_offset': float(near_edge),   # gap = peak_col + gap_offset (có thể âm)
        'M': M,                           # frame→strip perspective (for inverse map)
        'specular_mask': spec_mask,       # strip-space suppressed blobs (or None)
    }


def _find_outer_inner(
    pd: Dict[str, Any], params: EdgeParams = DEFAULT_EDGE_PARAMS,
) -> Tuple[Optional[int], Optional[int]]:
    """Return (inner_gap, outer_gap) — px outward từ mép label."""
    if pd is None:
        return None, None
    peaks = pd['peaks']
    hr_global = pd['height_ratio']
    hr_robust = pd['height_ratio_robust']
    go = pd['gap_offset']   # gap = peak_col + go (có thể âm khi bật inner search)

    # Pass 1: global threshold (chính xác cho frame clean)
    outer_q = [int(p) for p in peaks if hr_global[p] >= params.outer_min_hratio]
    if not outer_q:
        # Pass 2: robust (P95) — frame có specular/window light mạnh
        outer_q = [int(p) for p in peaks if hr_robust[p] >= params.outer_min_hratio]
    if outer_q:
        # Find-by strategy: pick which qualifying peak becomes the outer wall.
        fb = (params.find_by or "farthest").lower()
        if fb == "nearest":
            outer_col = min(outer_q)
        elif fb == "strongest":
            prof = pd['profile']
            outer_col = max(outer_q, key=lambda p: prof[p])
        else:  # 'farthest' (default, original behaviour)
            outer_col = max(outer_q)
        outer = outer_col + go
    else:
        outer = None

    hr_max = np.maximum(hr_global, hr_robust)
    inner_q = [int(p) for p in peaks if hr_max[p] >= params.inner_min_hratio]
    inner = None
    if outer is not None:
        cands = [p for p in inner_q if (p + go) < (outer - 4)]
        inner = (min(cands) + go) if cands else None
    elif inner_q:
        inner = min(inner_q) + go
    return inner, outer


def _find_inner_near(
    pd: Dict[str, Any], predicted_gap: float,
    tol: Optional[int] = None, params: EdgeParams = DEFAULT_EDGE_PARAMS,
) -> Optional[int]:
    if pd is None:
        return None
    if tol is None:
        tol = params.inner_tol_px
    peaks = pd['peaks']
    hr_global = pd['height_ratio']
    hr_robust = pd['height_ratio_robust']
    go = pd['gap_offset']
    hr_max = np.maximum(hr_global, hr_robust)
    cands = []
    for p in peaks:
        gap = int(p) + go
        if hr_max[p] >= params.inner_min_hratio and abs(gap - predicted_gap) <= tol:
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
def detect_template_walls(
    template_bgr: np.ndarray, label_pts,
    params: EdgeParams = DEFAULT_EDGE_PARAMS,
) -> Optional[Dict[str, Any]]:
    """Detect template's inner/outer walls. Gọi 1 lần khi init matcher.

    Returns: {inner_L, inner_R, outer_L, outer_R, plastic_L, plastic_R}
    None nếu detect fail.
    """
    masked = fill_label_black(template_bgr, label_pts, keep_ratio=params.fill_keep_ratio)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    pd_L = _compute_strip_profile(gray, label_pts, 'left', params=params)
    pd_R = _compute_strip_profile(gray, label_pts, 'right', params=params)
    inner_L, outer_L = _find_outer_inner(pd_L, params=params)
    inner_R, outer_R = _find_outer_inner(pd_R, params=params)
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
    wall_type: str = "outer",
    params: EdgeParams = DEFAULT_EDGE_PARAMS,
    return_profiles: bool = False,
) -> Optional[Dict[str, Any]]:
    """Detect product (bottle) box từ frame bằng image processing.

    return_profiles: khi True, đính kèm 'profiles' (curve + peaks + chosen
    outer/inner/pred per side) để FE vẽ chart chọn cạnh. Off mặc định (inference).

    Args:
        frame_img: BGR image
        label_pts: [TL, TR, BR, BL] of label polygon trong frame coords (from SuperPoint)
        template_walls: dict từ detect_template_walls()
        serial_number: chỉ cho logging
        wall_type: 'outer' (bottle silhouette, default) | 'inner' (sát label)
                   → quyết định corners được build từ wall nào

    Returns: dict YOLO OBB-compatible:
        {
            'box': np.array([cx, cy, w, h, angle]),
            'score': float,
            'class': 'product',
            'corners': np.array([[x,y]×4]),  # ← OUTER hoặc INNER tùy wall_type
            'inner_corners': ...   # always provided for debug
            'outer_corners': ...   # always provided for debug
            'source': 'image_proc',
            'detection_info': {...}
        }
    None nếu detect fail.
    """
    if frame_img is None or label_pts is None or template_walls is None:
        return None

    # 0) ROI = bbox của label + 2 dải search (chính xác theo quad từng dải).
    #    fill_label_black copy nguyên ảnh đầu vào và cvtColor chạy trên cả ảnh —
    #    trên frame 1920×1200 tốn ~7MB copy + 7MB đọc mỗi lần detect, tranh băng
    #    thông với TRT/capture. Mọi phép đo (gap, peak) là khoảng cách tương đối
    #    nên bất biến với phép tịnh tiến; chỉ M cần bù lại về frame coords.
    pts_f = np.asarray(label_pts, dtype=np.float32)
    strip_corners = [pts_f]
    for _side in ('left', 'right'):
        _p_top, _p_bot = (pts_f[0], pts_f[3]) if _side == 'left' else (pts_f[1], pts_f[2])
        _edge = _p_bot - _p_top
        _elen = float(np.linalg.norm(_edge)) or 1.0
        _edir = _edge / _elen
        if _side == 'left':
            _perp = np.array([-_edir[1], _edir[0]], dtype=np.float32)
        else:
            _perp = np.array([_edir[1], -_edir[0]], dtype=np.float32)
        _yext = params.y_extension * _elen
        _pt = _p_top - _yext * _edir
        _pb = _p_bot + _yext * _edir
        _near = params.edge_margin - params.inner_search_max
        _far = params.outer_search_max
        strip_corners.append(np.array([
            _pt + _near * _perp, _pt + _far * _perp,
            _pb + _near * _perp, _pb + _far * _perp,
        ], dtype=np.float32))
    _allc = np.vstack(strip_corners)
    _fh, _fw = frame_img.shape[:2]
    x0 = max(0, int(_allc[:, 0].min()) - 4)
    y0 = max(0, int(_allc[:, 1].min()) - 4)
    x1 = min(_fw, int(_allc[:, 0].max()) + 5)
    y1 = min(_fh, int(_allc[:, 1].max()) + 5)
    if (x1 - x0) < 30 or (y1 - y0) < 50:
        logger.warning(f"[{serial_number}] image_proc: degenerate ROI {x1-x0}x{y1-y0} — skip")
        return None
    roi_img = frame_img[y0:y1, x0:x1]
    roi_offset = np.array([x0, y0], dtype=np.float32)
    pts_roi = pts_f - roi_offset

    # 1) Fill label đen + extract grayscale (trên ROI)
    #    x_inset=INNER_SEARCH_MAX: co vùng đen vào trong để mép đen↔nền không tạo
    #    peak giả trong cửa sổ inner search.
    masked = fill_label_black(
        roi_img, pts_roi,
        keep_ratio=params.fill_keep_ratio, x_inset=params.inner_search_max,
    )
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    # 2) Compute profile + peaks 2 bên (quét cả vào trong mép label inner_search_max
    #    px — bắt được cạnh chai khi label lệch mạnh, gap có thể âm)
    pd_L = _compute_strip_profile(gray, pts_roi, 'left',  inner_search=params.inner_search_max, params=params)
    pd_R = _compute_strip_profile(gray, pts_roi, 'right', inner_search=params.inner_search_max, params=params)

    # Bù ROI: trả M/p_top/p_bot về frame coords để consumer (specular overlay,
    # debug) giữ nguyên hợp đồng cũ. gap/peak là scalar — không cần bù.
    _T_roi = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
    for _pd in (pd_L, pd_R):
        if _pd is not None:
            _pd['M'] = _pd['M'] @ _T_roi
            _pd['p_top'] = _pd['p_top'] + roi_offset
            _pd['p_bot'] = _pd['p_bot'] + roi_offset

    # 3) Detect outer wall mỗi bên (height_ratio strict)
    outer_L = _find_outer_inner(pd_L, params=params)[1]
    outer_R = _find_outer_inner(pd_R, params=params)[1]

    # 4) Predict inner từ outer - plastic; nếu outer fail thì fallback tìm
    #    inner near template với tol rộng hơn
    def _resolve(pd, outer, tmpl_inner, plastic):
        if outer is not None:
            # outer tìm được (kể cả gap âm = cạnh chai trong mép label) → side KHÔNG
            # hidden. inner chỉ phụ trợ (dùng khi wall_type='inner'); pred có thể âm.
            pred = outer - plastic
            inner = _find_inner_near(pd, pred, params=params)
            return inner, pred, False
        # Fallback: outer fail hẳn → bám template inner với tol rộng
        pred = tmpl_inner
        inner = _find_inner_near(pd, max(pred, params.edge_margin),
                                 tol=params.inner_tol_px + 8, params=params)
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

    # 6) Compute BOTH inner & outer corners
    in_L_gap = max(0.0, float(eff_L)) if eff_L != -1 else 0.0
    in_R_gap = max(0.0, float(eff_R)) if eff_R != -1 else 0.0
    out_L_gap = float(outer_L) if outer_L is not None else (in_L_gap + template_walls['plastic_L'])
    out_R_gap = float(outer_R) if outer_R is not None else (in_R_gap + template_walls['plastic_R'])

    # Inner corners
    L_top_in, L_bot_in = _wall_point_in_frame(label_pts, 'left',  in_L_gap)
    R_top_in, R_bot_in = _wall_point_in_frame(label_pts, 'right', in_R_gap)
    inner_corners = np.array([L_top_in, R_top_in, R_bot_in, L_bot_in], dtype=np.float32)

    # Outer corners
    L_top_out, L_bot_out = _wall_point_in_frame(label_pts, 'left',  out_L_gap)
    R_top_out, R_bot_out = _wall_point_in_frame(label_pts, 'right', out_R_gap)
    outer_corners = np.array([L_top_out, R_top_out, R_bot_out, L_bot_out], dtype=np.float32)

    # 7) Choose corners based on wall_type
    wt = (wall_type or "outer").lower()
    if wt == "inner":
        corners = inner_corners
        L_top, L_bot, R_top, R_bot = L_top_in, L_bot_in, R_top_in, R_bot_in
    else:
        corners = outer_corners
        L_top, L_bot, R_top, R_bot = L_top_out, L_bot_out, R_top_out, R_bot_out

    # 8) Box [cx, cy, w, h, angle] for YOLO OBB compatibility
    cx = float(corners[:, 0].mean())
    cy = float(corners[:, 1].mean())
    w  = float(np.linalg.norm((R_top + R_bot)/2 - (L_top + L_bot)/2))
    h  = float(np.linalg.norm(L_bot - L_top))
    edge = L_bot - L_top
    angle = float(np.arctan2(edge[0], edge[1]))
    box = np.array([cx, cy, w, h, angle], dtype=np.float32)

    result = {
        'box': box,
        'score': 1.0,
        'class': 'product',
        'corners': corners,                  # ← chosen wall (outer hoặc inner) — vẽ yellow OBB
        'inner_corners': inner_corners,      # debug / secondary draw
        'outer_corners': outer_corners,      # debug / secondary draw
        'wall_type_used': wt,
        'source': 'image_proc',
        'detection_info': {
            'inner_L': inner_L, 'inner_R': inner_R,
            'outer_L': outer_L, 'outer_R': outer_R,
            'eff_L': eff_L, 'eff_R': eff_R,
            'pred_inner_L': pred_L, 'pred_inner_R': pred_R,
            'hidden_L': hidden_L, 'hidden_R': hidden_R,
            'inner_L_gap': in_L_gap, 'inner_R_gap': in_R_gap,
            'outer_L_gap': out_L_gap, 'outer_R_gap': out_R_gap,
        }
    }

    if return_profiles:
        fh, fw = frame_img.shape[:2]

        def _specular_regions(pd):
            """Map the strip-space suppressed mask back to frame coords as
            simplified polygons, so the UI can overlay what specular_thr removes."""
            if pd is None or pd.get('specular_mask') is None:
                return []
            back = cv2.warpPerspective(
                pd['specular_mask'], pd['M'], (fw, fh),
                flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST,
            )
            cnts, _ = cv2.findContours(
                (back > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            out = []
            for c in cnts:
                if cv2.contourArea(c) < 20:
                    continue
                approx = cv2.approxPolyDP(c, 2.0, True).reshape(-1, 2)
                out.append([[int(x), int(y)] for x, y in approx])
            return out

        def _side_payload(pd, outer_gap, inner_gap, pred_gap):
            if pd is None:
                return None
            go = float(pd['gap_offset'])
            to_col = lambda g: (None if g is None else float(g) - go)
            # Per-peak height_ratio (the quantity outer/inner_min_hratio filters on):
            # max of the global + P95-robust passes, matching candidate selection.
            hr_max = np.maximum(pd['height_ratio'], pd['height_ratio_robust'])
            return {
                'profile': [round(float(x), 4) for x in pd['profile']],
                'peaks': [int(p) for p in pd['peaks']],
                'peak_hratio': [round(float(hr_max[int(p)]), 3) for p in pd['peaks']],
                'gap_offset': go,
                'outer_col': to_col(outer_gap),
                'inner_col': to_col(inner_gap),
                'pred_col': to_col(pred_gap),
                'outer_min_hratio': float(params.outer_min_hratio),
                'inner_min_hratio': float(params.inner_min_hratio),
                'detect_mode': (params.detect_mode or 'gradient'),
                'specular_regions': _specular_regions(pd),
            }
        result['profiles'] = {
            'left':  _side_payload(pd_L, outer_L, inner_L, pred_L),
            'right': _side_payload(pd_R, outer_R, inner_R, pred_R),
        }

    return result


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
