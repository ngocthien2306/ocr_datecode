"""
Template ↔ target OK/NG classifier (CV-based, no model).

Per-character similarity is measured via 3 metrics on binary character masks:
  - pixel_conf  : foreground pixel-count ratio
  - blur_tm     : multi-scale blurred template matching (TM_CCOEFF_NORMED)
  - iou         : IoU after centroid alignment + dilation

confidence = min(max(blur_tm, iou), pixel_conf)  ∈ [0, 1]

Class name `EmbeddingClassifierService` is preserved for caller compatibility,
but no embedding model is loaded — pure OpenCV pipeline.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Minimum grayscale std-dev for a crop to be considered a real character.
# A black/white uniform patch → std ≈ 0. Typical characters → std > 15.
MIN_CROP_STD: float = 8.0


def _crop_std(bgr: np.ndarray) -> float:
    """Return grayscale std-dev of an image crop."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    return float(np.std(gray))


"""
OLD: ImageNet preprocessing for ONNX inference (no longer used)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(bgr: np.ndarray, size: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"degenerate crop {w}x{h}")
    s = size / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    canvas[(size - nh) // 2:(size - nh) // 2 + nh, (size - nw) // 2:(size - nw) // 2 + nw] = img
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1).astype(np.float32)


def _extract_embeddings(sess, head, input_name, tensors):
    emb = sess.run(None, {input_name: tensors})[0]
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    return emb / norms
"""


# --------------------------------------------------------------------------- #
# Character quality comparison — copied from tests/test_segment.py logic
# --------------------------------------------------------------------------- #

def _to_thresh_norm(raw: np.ndarray) -> np.ndarray:
    """GaussianBlur(5,5) + Otsu inverse + morphClose(ellipse 3×3).
    Normalizes stroke width so chars like B/D/O are more comparable."""
    blurred = cv2.GaussianBlur(raw, (5, 5), 0)
    _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.mean(th) > 127:
        th = cv2.bitwise_not(th)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)


def _tight_crop(thresh: np.ndarray) -> np.ndarray:
    """Crop sát foreground, loại bỏ padding thừa."""
    coords = np.where(thresh > 0)
    if len(coords[0]) == 0:
        return thresh
    y0, y1 = int(coords[0].min()), int(coords[0].max())
    x0, x1 = int(coords[1].min()), int(coords[1].max())
    return thresh[y0:y1 + 1, x0:x1 + 1]


def _deskew_char(thresh_char: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Xoay ký tự về đứng nếu góc lệch < max_angle. Bỏ qua nếu lớn —
    tránh xoay sai với chữ bất đối xứng như L, J, F."""
    coords = np.column_stack(np.where(thresh_char > 0))
    if len(coords) < 10:
        return thresh_char
    angle = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))[2]
    if angle < -45:
        angle += 90
    if abs(angle) > max_angle:
        return thresh_char
    h, w = thresh_char.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(thresh_char, M, (w, h),
                          flags=cv2.INTER_NEAREST, borderValue=0)


def _fit_to_square(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize giữ aspect ratio, pad đen → size×size.
    Quan trọng cho ký tự hẹp (I, l, 1, j) — tránh stretch méo."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size[1], size[0]), dtype=np.uint8)
    scale = min(size[0] / w, size[1] / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size[1], size[0]), dtype=np.uint8)
    yo = (size[1] - nh) // 2
    xo = (size[0] - nw) // 2
    canvas[yo:yo + nh, xo:xo + nw] = resized
    return canvas


def _center_by_centroid(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Đặt mask vào khung size×size với khối tâm foreground ở giữa."""
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


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU của 2 mask nhị phân cùng kích thước."""
    inter = int(np.count_nonzero((a > 0) & (b > 0)))
    union = int(np.count_nonzero((a > 0) | (b > 0)))
    return inter / union if union > 0 else 0.0


def _compute_char_quality(
    tmpl_gray: np.ndarray,
    tgt_gray: np.ndarray,
    size: Tuple[int, int] = (64, 64),
) -> Dict[str, float]:
    """So sánh 2 ký tự bằng 3 metric, robust với binary shapes:
      1. pixel_conf  — ratio px_tgt/px_tmpl
      2. blur_tm     — multi-scale blurred TM_CCOEFF_NORMED
      3. iou         — IoU sau centroid alignment + dilation

    tm_conf    = max(blur_tm, iou)
    confidence = min(tm_conf, pixel_conf)

    Inputs MUST be single-channel (grayscale). Returns dict with confidence ∈ [0,1].
    """
    tmpl_b = _to_thresh_norm(tmpl_gray)
    tgt_b  = _to_thresh_norm(tgt_gray)

    t1 = _fit_to_square(_deskew_char(_tight_crop(tmpl_b)), size)
    g2_deskewed = _deskew_char(_tight_crop(tgt_b))
    t2_base = _fit_to_square(g2_deskewed, size)

    # (1) Pixel-count confidence
    px1 = int(np.count_nonzero(t1))
    px2 = int(np.count_nonzero(t2_base))
    ratio = px2 / (px1 + 1e-6)
    deviation = abs(ratio - 1.0)
    pixel_conf = float(np.clip(1.0 - deviation * (1.0 / 1.4), 0.0, 1.0))

    # (2) Blurred multi-scale template matching — blur biến binary thành soft mask,
    # giúp TM chịu được lệch 1–2 px và stroke khác nhau.
    t1_blur = cv2.GaussianBlur(t1.astype(np.float32), (0, 0), sigmaX=1.2)
    best_tm = 0.0
    for scale in (0.85, 0.92, 1.0, 1.08, 1.15):
        s = (max(t1.shape[1], int(size[0] * scale)),
             max(t1.shape[0], int(size[1] * scale)))
        t2 = _fit_to_square(g2_deskewed, s)
        t2_blur = cv2.GaussianBlur(t2.astype(np.float32), (0, 0), sigmaX=1.2)
        result = cv2.matchTemplate(t2_blur, t1_blur, cv2.TM_CCOEFF_NORMED)
        best_tm = max(best_tm, float(result.max()))
    blur_tm = float(np.clip(best_tm, 0.0, 1.0))

    # (3) IoU sau centroid alignment — dilate(ellipse 5×5) tha thứ stroke width
    a = _center_by_centroid(_tight_crop(t1),      size)
    b = _center_by_centroid(_tight_crop(t2_base), size)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    a = cv2.dilate(a, k, iterations=1)
    b = cv2.dilate(b, k, iterations=1)
    iou = _iou(a, b)

    tm_conf = max(blur_tm, iou)
    confidence = min(tm_conf, pixel_conf)

    return {
        "confidence": float(confidence),
        "tm_conf":    float(tm_conf),
        "blur_tm":    float(blur_tm),
        "iou":        float(iou),
        "pixel_conf": float(pixel_conf),
        "px_tmpl":    px1,
        "px_tgt":     px2,
    }


class EmbeddingClassifierService:
    """
    Per-character OK/NG classifier (CV-based, no model).

    classify_batch() compares each template/target pair using
    `_compute_char_quality` — multi-metric character similarity.

    `onnx_path` and `config_path` are kept in the signature for backward
    compatibility with existing callers but are NOT loaded or used.
    """

    def __init__(
        self,
        onnx_path: str,
        config_path: str,
        save_debug_images: bool = True,
        debug_path: Optional[str] = None,
    ):
        """
        OLD: ONNX MODEL LOADING (commented out — CV-based mode does not need a model)

        cfg = OmegaConf.load(config_path)
        self.head = cfg.model.head.type
        self.size = int(cfg.data.image_size)

        if self.head not in ("projection", "arcface"):
            raise ValueError(
                f"EmbeddingClassifierService requires projection/arcface head, got: {self.head}"
            )

        self.sess = ort.InferenceSession(
            str(onnx_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.input_name = self.sess.get_inputs()[0].name
        """

        # Params accepted but unused — kept for signature compatibility
        self._onnx_path = onnx_path
        self._config_path = config_path

        self.save_debug_images = save_debug_images
        self.debug_path = debug_path or f"{os.environ.get('HOME')}/Source/ocr_datecode/ai_services/test_result"
        if self.save_debug_images:
            os.makedirs(self.debug_path, exist_ok=True)

        logger.info(
            f"EmbeddingClassifierService (CV-based mode): no model loaded, "
            f"debug={self.debug_path}"
        )

    def classify_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Batch classify N character crops by template ↔ target CV similarity.

        Each item must supply both `region_img` (target) and `template_crop`.
        Per-item `conf_threshold` is compared against `p_ok` to decide OK/NG.

        Input item shape:
            {
                'region_img':     np.ndarray  — target crop (current frame)
                'template_crop':  np.ndarray  — template crop (required)
                'conf_threshold': float       — per-character OK threshold
                'serial_number':  str
                'annotation_idx': int
            }

        Returns list parallel to items:
            {'ml_pass', 'p_ok', 'label', 'threshold', 'time_ms', 'error'}
        """
        """
        OLD: ONNX EMBEDDING + COSINE-SIMILARITY IMPLEMENTATION (commented out)

        n = len(items)
        if n == 0:
            return []

        t0 = time.perf_counter()
        results: List[Optional[Dict[str, Any]]] = [None] * n

        tmpl_tensors: List[np.ndarray] = []
        tgt_tensors: List[np.ndarray] = []
        valid_idxs: List[int] = []

        for i, item in enumerate(items):
            region = item.get('region_img')
            template = item.get('template_crop')
            conf_thr = float(item.get('conf_threshold', 0.5))
            serial = item.get('serial_number', '')
            ann_idx = item.get('annotation_idx', -1)

            if region is None or region.size == 0:
                results[i] = {
                    'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                    'threshold': conf_thr, 'time_ms': 0.0, 'error': 'empty_region',
                }
                continue

            if template is None or template.size == 0:
                results[i] = {
                    'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                    'threshold': conf_thr, 'time_ms': 0.0, 'error': 'missing_template_crop',
                }
                continue

            tgt_std = _crop_std(region)
            if tgt_std < MIN_CROP_STD:
                results[i] = {
                    'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                    'threshold': conf_thr, 'time_ms': 0.0,
                    'error': f'low_variance_target:{tgt_std:.1f}',
                }
                continue

            tmpl_std = _crop_std(template)
            if tmpl_std < MIN_CROP_STD:
                results[i] = {
                    'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                    'threshold': conf_thr, 'time_ms': 0.0,
                    'error': f'low_variance_template:{tmpl_std:.1f}',
                }
                continue

            try:
                tgt_tensors.append(_preprocess(region, self.size))
                tmpl_tensors.append(_preprocess(template, self.size))
            except Exception as e:
                results[i] = {
                    'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                    'threshold': conf_thr, 'time_ms': 0.0,
                    'error': f'preprocess_failed:{e}',
                }
                continue

            valid_idxs.append(i)

        if not valid_idxs:
            ...

        m = len(valid_idxs)
        try:
            combined = np.stack(tmpl_tensors + tgt_tensors)
            all_embs = _extract_embeddings(self.sess, self.head, self.input_name, combined)
            tmpl_embs = all_embs[:m]
            tgt_embs  = all_embs[m:]
        except Exception as e:
            ...

        for row, i in enumerate(valid_idxs):
            sim = float(np.dot(tmpl_embs[row], tgt_embs[row]))
            p_ok = (sim + 1.0) / 2.0
            ...
        """

        # ====================================================================
        # NEW: CV-based per-pair character quality (no model inference)
        # ====================================================================
        n = len(items)
        if n == 0:
            return []

        t0 = time.perf_counter()
        results: List[Optional[Dict[str, Any]]] = [None] * n
        valid_idxs: List[int] = []

        # ---- Stage 1: validate inputs ----
        for i, item in enumerate(items):
            region = item.get('region_img')
            template = item.get('template_crop')
            conf_thr = float(item.get('conf_threshold', 0.5))
            serial = item.get('serial_number', '')
            ann_idx = item.get('annotation_idx', -1)

            if region is None or region.size == 0:
                results[i] = {
                    'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                    'threshold': conf_thr, 'time_ms': 0.0, 'error': 'empty_region',
                }
                continue

            if template is None or template.size == 0:
                results[i] = {
                    'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                    'threshold': conf_thr, 'time_ms': 0.0, 'error': 'missing_template_crop',
                }
                continue

            tgt_std = _crop_std(region)
            if tgt_std < MIN_CROP_STD:
                logger.debug(
                    f"[{serial}] ann {ann_idx}: target crop too uniform "
                    f"(std={tgt_std:.1f} < {MIN_CROP_STD}) → NG"
                )
                results[i] = {
                    'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                    'threshold': conf_thr, 'time_ms': 0.0,
                    'error': f'low_variance_target:{tgt_std:.1f}',
                }
                continue

            tmpl_std = _crop_std(template)
            if tmpl_std < MIN_CROP_STD:
                logger.warning(
                    f"[{serial}] ann {ann_idx}: template crop too uniform "
                    f"(std={tmpl_std:.1f} < {MIN_CROP_STD}), check recipe"
                )
                results[i] = {
                    'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                    'threshold': conf_thr, 'time_ms': 0.0,
                    'error': f'low_variance_template:{tmpl_std:.1f}',
                }
                continue

            valid_idxs.append(i)

        if not valid_idxs:
            for i, r in enumerate(results):
                if r is None:
                    results[i] = {
                        'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                        'threshold': float(items[i].get('conf_threshold', 0.5)),
                        'time_ms': 0.0, 'error': 'all_invalid',
                    }
            return results  # type: ignore

        # ---- Stage 2: per-pair comparison ----
        m = len(valid_idxs)

        debug_dir = None
        if self.save_debug_images:
            try:
                serial = items[valid_idxs[0]].get('serial_number', 'unknown')
                ts = time.strftime("%Y%m%d_%H%M%S")
                debug_dir = os.path.join(self.debug_path, f"emb_{serial}_{ts}")
                os.makedirs(debug_dir, exist_ok=True)
            except Exception:
                debug_dir = None

        for i in valid_idxs:
            item = items[i]
            region = item['region_img']
            template = item['template_crop']
            conf_thr = float(item.get('conf_threshold', 0.5))

            try:
                tmpl_gray = (
                    cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                    if template.ndim == 3 else template
                )
                tgt_gray = (
                    cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                    if region.ndim == 3 else region
                )

                metrics = _compute_char_quality(tmpl_gray, tgt_gray)
                p_ok = float(metrics['confidence'])
                label = "OK" if p_ok >= conf_thr else "NG"
                results[i] = {
                    'ml_pass': (label == "OK"),
                    'p_ok': round(p_ok, 4),
                    'label': label,
                    'threshold': conf_thr,
                    'time_ms': 0.0,  # filled after batch
                    'error': None,
                }

                if debug_dir is not None:
                    try:
                        ann = item.get('annotation_idx', i)
                        prefix = f"char{ann:02d}_{label}_p{p_ok:.2f}"
                        cv2.imwrite(os.path.join(debug_dir, f"{prefix}_template.png"),
                                    template)
                        cv2.imwrite(os.path.join(debug_dir, f"{prefix}_target.png"),
                                    region)
                    except Exception:
                        pass

                logger.debug(
                    f"[{item.get('serial_number', '')}] cv ann "
                    f"{item.get('annotation_idx', -1)}: "
                    f"conf={p_ok:.4f} tm={metrics['tm_conf']:.3f} "
                    f"blur_tm={metrics['blur_tm']:.3f} iou={metrics['iou']:.3f} "
                    f"px={metrics['pixel_conf']:.3f} {label} thr={conf_thr}"
                )
            except Exception as e:
                results[i] = {
                    'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                    'threshold': conf_thr, 'time_ms': 0.0,
                    'error': f'compare_failed:{e}',
                }

        elapsed = (time.perf_counter() - t0) * 1000
        time_per_item = round(elapsed / m, 2)
        for i in valid_idxs:
            if results[i] is not None:
                results[i]['time_ms'] = time_per_item

        logger.info(
            f"CV classify batch: N={m}, elapsed={elapsed:.1f}ms ({time_per_item:.2f}ms/item)"
        )

        for i, r in enumerate(results):
            if r is None:
                results[i] = {
                    'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                    'threshold': float(items[i].get('conf_threshold', 0.5)),
                    'time_ms': 0.0, 'error': 'unknown',
                }

        return results  # type: ignore
