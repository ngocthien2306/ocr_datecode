"""
Text Verification Service

Handles OCR-based text verification for inference results.
Supports both single-camera and multi-camera batch processing.
"""

import logging
import time
import os
import cv2
import numpy as np
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

from ..ocr_utils import crop_text_region, compare_texts, compare_texts_smart

if TYPE_CHECKING:
    from ..camera import Camera

logger = logging.getLogger(__name__)

home = os.environ.get('HOME')

AUGMENT_SIMILARITY_THRESHOLD = 0.70


def augment_laser_text(img_bgr: np.ndarray) -> dict:
    """
    Generate 5 enhanced versions optimized for difficult backgrounds (laser-engraved, low contrast).
    Copied from tests/test_trt_inference.py.

    Returns:
        dict with keys: 'original', 'clahe', 'bg_subtract', 'unsharp_clahe', 'tophat'
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    results = {'original': img_bgr.copy()}

    # 1. CLAHE – adaptive local contrast enhancement
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lab_eq = cv2.merge([clahe.apply(l), a, b])
    results['clahe'] = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # 2. Background subtraction – amplify residual text signal
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    diff_amp = cv2.convertScaleAbs(cv2.subtract(gray, bg), alpha=8)
    results['bg_subtract'] = cv2.cvtColor(diff_amp, cv2.COLOR_GRAY2BGR)

    # 3. Unsharp masking + CLAHE
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    unsharp = cv2.addWeighted(gray, 2.0, blurred, -1.0, 0)
    results['unsharp_clahe'] = cv2.cvtColor(clahe.apply(unsharp), cv2.COLOR_GRAY2BGR)

    # 4. Morphological TOPHAT – extracts bright regions smaller than kernel
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 20))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_morph)
    tophat_eq = cv2.convertScaleAbs(tophat, alpha=6)
    results['tophat'] = cv2.cvtColor(clahe.apply(tophat_eq), cv2.COLOR_GRAY2BGR)

    return results


def _apply_text_corrections(text: str) -> str:
    """Apply common OCR correction rules before comparison."""
    # if "BE" in text:
    #     text = text.replace("BE", "BB")
    # if "RL" in text:
    #     text = text.replace("RL", "PL")
    if "Pt" in text:
        text = text.replace("Pt", "PL")
    if "USsed" in text:
        text = text.replace("USsed", "Used")
    if "Iif" in text:
        text = text.replace("Iif", "If")


    return text


# ── Character-level quality analysis ──────────────────────────────────────

def _split_wide_box(thresh_roi, box, median_w):
    """Split a box that is too wide using vertical projection."""
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


def _segment_characters_from_image(img: np.ndarray) -> list:
    """
    Segment characters from a BGR image.
    Returns list of cropped char images (BGR), sorted left-to-right.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = img.shape[:2]
    min_char_height = h_img * 0.3
    min_char_width = 3

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= min_char_height and w >= min_char_width:
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[0])

    merged = []
    for box in boxes:
        if merged and box[0] < merged[-1][0] + merged[-1][2]:
            px, py, pw, ph = merged[-1]
            nx = min(px, box[0])
            ny = min(py, box[1])
            nx2 = max(px + pw, box[0] + box[2])
            ny2 = max(py + ph, box[1] + box[3])
            merged[-1] = (nx, ny, nx2 - nx, ny2 - ny)
        else:
            merged.append(box)

    if merged:
        widths = [b[2] for b in merged]
        median_w = float(np.median(widths))
        final_boxes = []
        for box in merged:
            sub = _split_wide_box(thresh, box, median_w)
            final_boxes.extend(sub)
    else:
        final_boxes = merged

    padding = 2
    char_imgs = []
    for (x, y, w, h) in final_boxes:
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(w_img, x + w + padding), min(h_img, y + h + padding)
        char_imgs.append(img[y1:y2, x1:x2])

    return char_imgs


def _tight_crop(thresh: np.ndarray) -> np.ndarray:
    """Crop to foreground bounding box, removing excess padding."""
    coords = np.where(thresh > 0)
    if len(coords[0]) == 0:
        return thresh
    y0, y1 = int(coords[0].min()), int(coords[0].max())
    x0, x1 = int(coords[1].min()), int(coords[1].max())
    return thresh[y0:y1 + 1, x0:x1 + 1]


def _deskew_char(thresh_char: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Rotate upright. Skip if angle > max_angle to avoid wrong rotation on L/J/F."""
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
    return cv2.warpAffine(thresh_char, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)


def _fit_to_square(img: np.ndarray, size: tuple) -> np.ndarray:
    """Resize keeping aspect ratio, pad with black. Prevents I/l/1 distortion."""
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


def _center_by_centroid(mask: np.ndarray, size: tuple) -> np.ndarray:
    """Place mask in canvas so its centroid is centered."""
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
    y2s = max(0, -yo)
    x2s = max(0, -xo)
    y2e = y2s + (y1e - y1s)
    x2e = x2s + (x1e - x1s)
    if y1e > y1s and x1e > x1s:
        canvas[y1s:y1e, x1s:x1e] = mask[y2s:y2e, x2s:x2e]
    return canvas


def _iou_binary(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two same-size binary masks."""
    inter = int(np.count_nonzero((a > 0) & (b > 0)))
    union = int(np.count_nonzero((a > 0) | (b > 0)))
    return inter / union if union > 0 else 0.0


def _char_quality(tmpl_char: np.ndarray, tgt_char: np.ndarray, size=(64, 64)) -> dict:
    """
    Per-character quality analysis (ported from test_segment.py).

    Pipeline: BGR → thresh → tight_crop → deskew → fit_to_square
    Metrics:
      1. Blurred Template Matching (TM_CCOEFF_NORMED, σ=1.2, multi-scale)
      2. IoU after centroid alignment
      3. Pixel ratio confidence
    tm_conf = max(blur_TM, IoU). final confidence = min(tm_conf, pixel_conf).
    Defect detection retained for production use.
    """
    def _to_thresh(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if np.mean(th) > 127:
            th = cv2.bitwise_not(th)
        return th

    tmpl_thresh = _to_thresh(tmpl_char)
    tgt_thresh = _to_thresh(tgt_char)

    t1 = _fit_to_square(_deskew_char(_tight_crop(tmpl_thresh)), size)
    g2_deskewed = _deskew_char(_tight_crop(tgt_thresh))
    t2_base = _fit_to_square(g2_deskewed, size)

    # (1) Pixel confidence
    px1 = int(np.count_nonzero(t1))
    px2 = int(np.count_nonzero(t2_base))
    pixel_ratio = px2 / (px1 + 1e-6)
    deviation = abs(pixel_ratio - 1.0)
    pixel_conf = float(np.clip(1.0 - deviation * (1.0 / 1.4), 0.0, 1.0))

    # (2) Blurred multi-scale Template Matching
    t1_blur = cv2.GaussianBlur(t1.astype(np.float32), (0, 0), sigmaX=1.2)
    best_tm = 0.0
    for scale in [0.85, 0.92, 1.0, 1.08, 1.15]:
        s = (max(t1.shape[1], int(size[0] * scale)),
             max(t1.shape[0], int(size[1] * scale)))
        t2 = _fit_to_square(g2_deskewed, s)
        t2_blur = cv2.GaussianBlur(t2.astype(np.float32), (0, 0), sigmaX=1.2)
        result = cv2.matchTemplate(t2_blur, t1_blur, cv2.TM_CCOEFF_NORMED)
        best_tm = max(best_tm, float(result.max()))
    blur_tm = float(np.clip(best_tm, 0.0, 1.0))

    # (3) IoU after centroid alignment
    a = _center_by_centroid(_tight_crop(t1), size)
    b = _center_by_centroid(_tight_crop(t2_base), size)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    a = cv2.dilate(a, k, iterations=1)
    b = cv2.dilate(b, k, iterations=1)
    iou = _iou_binary(a, b)

    tm_conf = max(blur_tm, iou)
    confidence = min(tm_conf, pixel_conf)

    # Sharpness from BGR for LEM detection
    g1 = cv2.cvtColor(tmpl_char, cv2.COLOR_BGR2GRAY) if len(tmpl_char.shape) == 3 else tmpl_char
    g2 = cv2.cvtColor(tgt_char, cv2.COLOR_BGR2GRAY) if len(tgt_char.shape) == 3 else tgt_char
    g1 = cv2.resize(g1, size)
    g2 = cv2.resize(g2, size)
    sharp_t = float(np.var(cv2.Laplacian(g1, cv2.CV_64F)))
    sharp_g = float(np.var(cv2.Laplacian(g2, cv2.CV_64F)))
    sharp_ratio = sharp_g / (sharp_t + 1e-8)

    # Connected components for GAY_NET
    cc_t = cv2.connectedComponents(t1)[0] - 1
    cc_g = cv2.connectedComponents(t2_base)[0] - 1

    # Defect detection
    defects = []
    if confidence < 0.5:
        defects.append("SAI_KY_TU")
    if sharp_ratio < 0.25:
        defects.append("LEM")
    if pixel_ratio > 1.3 and iou > 0.6:
        defects.append("IN_CHONG")
    if cc_g > cc_t + 2:
        defects.append("GAY_NET")
    if pixel_ratio < 0.5:
        defects.append("MAT_NET")

    return {
        "confidence": round(confidence, 4),
        "tm_conf": round(tm_conf, 4),
        "blur_tm": round(blur_tm, 4),
        "iou": round(iou, 4),
        "pixel_conf": round(pixel_conf, 3),
        "px_tmpl": px1,
        "px_tgt": px2,
        "sharp_ratio": round(sharp_ratio, 4),
        "cc_tmpl": cc_t,
        "cc_tgt": cc_g,
        "defects": defects,
    }


def _save_char_comparison(tmpl_img, tgt_img, tmpl_chars, tgt_chars, char_results, output_path, conf_threshold=0.75):
    """Save visual comparison strip — layout identical to test_segment.py:
      Row 1: ORIG TEMPLATE  |  ORIG TARGET        (side by side)
      Row 2: THRESH TEMPLATE | THRESH TARGET       (side by side)
      Row 3: template chars  (80×80 per char)
      Row 4: target chars    (80×80, green/red border)
      Row 5: confidence + PASS/FAIL per char
      Banner: overall PASS/FAIL
    """
    n = len(char_results)
    if n == 0:
        return

    cell_w, cell_h = 80, 80
    gap = 4
    score_h = 40
    strip_w = n * (cell_w + gap) + gap

    orig_h = 120
    orig_section_h = (orig_h + gap) * 2 + gap
    char_strip_h = 2 * (cell_h + gap) + score_h + gap * 2
    total_h = orig_section_h + char_strip_h
    total_w = max(strip_w, 200)

    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 50

    def place_image(src, y_start, x_start, w, h):
        img_bgr = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR) if len(src.shape) == 2 else src
        ih, iw = img_bgr.shape[:2]
        scale = w / iw
        nw = w
        nh = min(h, max(1, int(ih * scale)))
        resized = cv2.resize(img_bgr, (nw, nh))
        canvas[y_start:y_start + nh, x_start:x_start + nw] = resized

    def make_full_thresh(bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if len(bgr.shape) == 3 else bgr
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if np.mean(th) > 127:
            th = cv2.bitwise_not(th)
        return th

    def fit_char(src, w, h):
        img = src if len(src.shape) == 3 else cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        ih, iw = img.shape[:2]
        scale = min(w / iw, h / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = cv2.resize(img, (nw, nh))
        cell = np.ones((h, w, 3), dtype=np.uint8) * 50
        yo = (h - nh) // 2
        xo = (w - nw) // 2
        cell[yo:yo + nh, xo:xo + nw] = resized
        return cell

    tmpl_thresh = make_full_thresh(tmpl_img)
    tgt_thresh = make_full_thresh(tgt_img)
    half_w = total_w // 2

    # Row 1: original images side by side
    y0 = gap
    place_image(tmpl_img, y0, 0, half_w, orig_h)
    place_image(tgt_img, y0, half_w, half_w, orig_h)
    cv2.line(canvas, (half_w, y0), (half_w, y0 + orig_h), (120, 120, 120), 1)
    cv2.putText(canvas, "ORIG TEMPLATE", (4, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(canvas, "ORIG TARGET", (half_w + 4, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # Row 2: thresh images side by side
    y1 = y0 + orig_h + gap
    place_image(tmpl_thresh, y1, 0, half_w, orig_h)
    place_image(tgt_thresh, y1, half_w, half_w, orig_h)
    cv2.line(canvas, (half_w, y1), (half_w, y1 + orig_h), (120, 120, 120), 1)
    cv2.putText(canvas, "THRESH TEMPLATE", (4, y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(canvas, "THRESH TARGET", (half_w + 4, y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # Char rows
    y_base = orig_section_h
    cv2.putText(canvas, "TEMPLATE", (2, y_base + gap + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(canvas, "TARGET", (2, y_base + gap * 2 + cell_h + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    for idx, cr in enumerate(char_results):
        x_off = gap + idx * (cell_w + gap)
        if x_off + cell_w > total_w:
            break
        i = cr["idx"]
        is_pass = cr["confidence"] >= conf_threshold
        color = (0, 220, 0) if is_pass else (0, 0, 255)

        # Template char row
        canvas[y_base + gap:y_base + gap + cell_h, x_off:x_off + cell_w] = fit_char(tmpl_chars[i], cell_w, cell_h)

        # Target char row + border
        tgt_y0 = y_base + gap * 2 + cell_h
        tgt_y1 = tgt_y0 + cell_h
        canvas[tgt_y0:tgt_y1, x_off:x_off + cell_w] = fit_char(tgt_chars[i], cell_w, cell_h)
        cv2.rectangle(canvas, (x_off, tgt_y0), (x_off + cell_w, tgt_y1), color, 2)

        # Score row
        y_text = y_base + gap * 3 + cell_h * 2
        cv2.putText(canvas, f"{cr['confidence']:.2f}", (x_off + 2, y_text + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        label = "PASS" if is_pass else "FAIL"
        cv2.putText(canvas, label, (x_off + 2, y_text + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Overall PASS/FAIL banner
    overall_pass = all(cr["confidence"] >= conf_threshold for cr in char_results)
    banner_color = (0, 180, 0) if overall_pass else (0, 0, 220)
    banner_label = "PASS" if overall_pass else "FAIL"
    banner_h = 28
    banner = np.ones((banner_h, total_w, 3), dtype=np.uint8)
    banner[:] = banner_color
    text_size = cv2.getTextSize(banner_label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
    tx = (total_w - text_size[0]) // 2
    cv2.putText(banner, banner_label, (tx, 21), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    canvas = np.vstack([canvas, banner])

    cv2.imwrite(output_path, canvas)


@dataclass
class TextVerificationResult:
    """Result of text verification for a single annotation"""
    annotation_idx: int
    expected: str
    recognized: str
    match: bool
    confidence: float
    threshold: float
    error: Optional[str] = None


@dataclass
class TextVerificationSummary:
    """Summary of text verification for all regions"""
    all_match: bool
    results: List[TextVerificationResult] = field(default_factory=list)
    error: Optional[str] = None


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity ratio between two texts using SequenceMatcher.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity ratio (0.0 - 1.0)
    """
    # Normalize: strip whitespace, remove internal spaces, and convert to lowercase
    special_chars_to_space = ['_', '-', '－', '—', '–', ',', '.', ':', ';', '--']  # underscore, hyphen, dashes, punctuation
    for char in special_chars_to_space:
        text1 = text1.replace(char, " ")
        text2 = text2.replace(char, " ")

    text1_norm = text1.strip().replace(" ", "").lower()
    text2_norm = text2.strip().replace(" ", "").lower()

    # Calculate similarity ratio
    ratio = SequenceMatcher(None, text1_norm, text2_norm).ratio()
    return ratio


class TextVerificationService:
    """
    Service for verifying text regions using OCR.

    Responsibilities:
    - Crop text regions from frames
    - Run OCR on cropped regions
    - Compare recognized text with expected text
    - Support batch OCR for multiple cameras
    """

    # Default max dimension for resize optimization (reuse from TemplateVerificationService)
    SIM_MAX_DIMENSION = 200
    SIM_MAX_WORKERS = 4

    def __init__(
        self,
        text_recognizer: Any,
        ocr_backend: str,
        save_debug_images: bool = True,
        debug_path: Optional[str] = None,
        use_char_conf_check: bool = False,
        use_sim_check: bool = False,
    ):
        """
        Initialize TextVerificationService.

        Args:
            text_recognizer: OCR model instance (TensorRT or ONNX)
            ocr_backend: Backend name ("tensorrt" or "onnx")
            save_debug_images: Whether to save cropped regions for debugging
            debug_path: Path to save debug images
            use_sim_check: Whether to run similarity check on text/datecode regions
        """
        self.text_recognizer = text_recognizer
        self.ocr_backend = ocr_backend
        self.save_debug_images = save_debug_images
        self.debug_path = debug_path or f"{home}/Source/ocr_datecode/ai_services/test_result"
        self._debug_counter = 0
        self.use_char_conf_check = use_char_conf_check
        self.use_sim_check = use_sim_check
        self._sim_crop_cache = {}  # Cache for template crops (key: (serial, points_tuple))

    @property
    def is_available(self) -> bool:
        """Check if OCR is available"""
        return self.text_recognizer is not None

    def _resize_for_matching(self, img: np.ndarray, max_dim: int) -> tuple:
        """Resize image if larger than max_dim for faster matching."""
        if max_dim <= 0:
            return img, 1.0
        h, w = img.shape[:2]
        max_current = max(h, w)
        if max_current <= max_dim:
            return img, 1.0
        scale = max_dim / max_current
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale

    def _calculate_sim(
        self,
        template_crop: np.ndarray,
        target_crop: np.ndarray,
        method: int = cv2.TM_CCOEFF_NORMED
    ) -> float:
        """Calculate similarity score between two image crops using cv2.matchTemplate."""
        if len(template_crop.shape) == 3:
            template_gray = cv2.cvtColor(template_crop, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template_crop
        if len(target_crop.shape) == 3:
            target_gray = cv2.cvtColor(target_crop, cv2.COLOR_BGR2GRAY)
        else:
            target_gray = target_crop

        result = cv2.matchTemplate(target_gray, template_gray, method)
        if method == cv2.TM_SQDIFF_NORMED:
            return float(1.0 - result[0, 0])
        return float(result[0, 0])

    def _compute_single_sim(
        self,
        frame_img: np.ndarray,
        template_img: np.ndarray,
        transformed_points: List,
        original_points: List,
        serial_number: str,
        annotation_idx: int,
        conf_threshold: float,
    ) -> Dict[str, Any]:
        """
        Compute similarity for a single text/datecode region.
        Also runs per-character quality analysis and saves comparison image.
        Used as a unit of work for ThreadPoolExecutor.
        """
        t_start = time.perf_counter()
        try:
            # Crop from frame (transformed points)
            cropped_target = crop_text_region(frame_img, transformed_points)

            # Crop from template (original points) — use cache
            cache_key = (serial_number, tuple(map(tuple, original_points)))
            if cache_key in self._sim_crop_cache:
                cropped_template = self._sim_crop_cache[cache_key]
            else:
                cropped_template = crop_text_region(template_img, original_points)
                self._sim_crop_cache[cache_key] = cropped_template

            # ── Per-character quality analysis (threaded) ──
            t_char_start = time.perf_counter()
            tmpl_chars = _segment_characters_from_image(cropped_template)
            tgt_chars = _segment_characters_from_image(cropped_target)
            n_pairs = min(len(tmpl_chars), len(tgt_chars))

            char_results = []
            if n_pairs > 0:
                with ThreadPoolExecutor(max_workers=min(n_pairs, self.SIM_MAX_WORKERS)) as pool:
                    metrics_list = list(pool.map(
                        lambda i: _char_quality(tmpl_chars[i], tgt_chars[i]),
                        range(n_pairs)
                    ))
                for i, m in enumerate(metrics_list):
                    char_results.append({"idx": i, **m})

            t_char_ms = (time.perf_counter() - t_char_start) * 1000

            # Similarity = min confidence across all char pairs (weakest link)
            # If char counts mismatch, penalize to 0
            if n_pairs == 0:
                similarity = 0.0
            elif len(tmpl_chars) != len(tgt_chars):
                similarity = 0.0
            else:
                similarity = float(min(cr["confidence"] for cr in char_results))

            match_sim = bool(similarity >= conf_threshold)

            # Count defects
            total_defects = sum(1 for cr in char_results if cr["defects"])
            defect_summary = {}
            for cr in char_results:
                for d in cr["defects"]:
                    defect_summary[d] = defect_summary.get(d, 0) + 1

            # Save comparison image to debug folder
            if self.save_debug_images and n_pairs > 0:
                try:
                    ts = int(time.time())
                    out_path = os.path.join(
                        self.debug_path,
                        f"char_quality_{serial_number}_{annotation_idx}_{ts}.png"
                    )
                    _save_char_comparison(
                        cropped_template, cropped_target,
                        tmpl_chars, tgt_chars, char_results, out_path,
                        conf_threshold=conf_threshold
                    )
                except Exception as e_save:
                    logger.warning(f"[{serial_number}] Failed to save char comparison: {e_save}")

            elapsed = (time.perf_counter() - t_start) * 1000
            logger.info(
                f"[{serial_number}] Sim check annotation {annotation_idx}: "
                f"similarity={similarity:.4f}, threshold={conf_threshold}, "
                f"match_sim={match_sim}, "
                f"chars={n_pairs} (tmpl={len(tmpl_chars)}, tgt={len(tgt_chars)}), "
                f"defects={total_defects}/{n_pairs} {defect_summary}, "
                f"char_analysis={t_char_ms:.1f}ms, total={elapsed:.1f}ms"
            )

            return {
                'annotation_idx': annotation_idx,
                'match_sim': match_sim,
                'similarity': similarity,
                'threshold': conf_threshold,
                'time_ms': elapsed,
                'char_count': n_pairs,
                'char_defects': total_defects,
                'defect_summary': defect_summary,
                'char_results': char_results,
            }
        except Exception as e:
            elapsed = (time.perf_counter() - t_start) * 1000
            logger.error(
                f"[{serial_number}] Sim check error annotation {annotation_idx}: {e}, "
                f"time={elapsed:.1f}ms"
            )
            return {
                'annotation_idx': annotation_idx,
                'match_sim': False,
                'similarity': 0.0,
                'threshold': conf_threshold,
                'time_ms': elapsed,
                'error': str(e),
            }

    def verify_text_regions(
        self,
        frame_img: 'np.ndarray',
        transformed_bboxes: List[Dict[str, Any]],
        expected_texts: Dict[int, str],
        camera: 'Camera',
        recognition_threshold: float = 0.5,
        template_img: Optional[np.ndarray] = None,
        original_bboxes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Verify text in transformed regions match expected texts.

        Args:
            frame_img: Captured frame (numpy array)
            transformed_bboxes: List of transformed bbox dicts from matcher
            expected_texts: Dict mapping annotation_idx -> expected_text
            camera: Camera object for logging
            recognition_threshold: Minimum OCR confidence threshold
            template_img: Reference template image (for sim check)
            original_bboxes: Original bboxes from template (for sim check)

        Returns:
            {
                'all_match': bool,
                'results': [
                    {
                        'annotation_idx': 0,
                        'expected': '123',
                        'recognized': '123',
                        'match': True,
                        'match_sim': True,   # only when use_sim_check=True
                        'similarity': 0.92,  # only when use_sim_check=True
                        'confidence': 0.95,
                        'threshold': 0.8
                    },
                    ...
                ]
            }
        """
        if not self.is_available:
            logger.warning("OCR model not available, skipping text verification")
            return {'all_match': False, 'results': []}

        verification_results = []
        all_match = True
        serial_number = camera.serial_number

        # Filter only text type bboxes
        text_bboxes = [
            bbox for bbox in transformed_bboxes
            if bbox.get('type') in ['text', 'datecode']
        ]

        logger.info(f"[{serial_number}] Verifying {len(text_bboxes)} text regions")
        logger.info(f"[{serial_number}] Expected texts dict: {expected_texts}")
        logger.info(
            f"[{serial_number}] Text bbox annotation indices: "
            f"{[bbox.get('annotation_index') for bbox in text_bboxes]}"
        )

        # Build original bbox lookup for sim check
        original_bbox_map = {}
        if self.use_sim_check and template_img is not None and original_bboxes:
            for ob in original_bboxes:
                if ob.get('type') in ['text', 'datecode']:
                    ob_idx = ob.get('annotation_index')
                    if ob_idx is not None:
                        original_bbox_map[ob_idx] = ob

        # Collect sim tasks
        sim_tasks_local = []
        if self.use_sim_check and template_img is not None and original_bbox_map:
            for bbox in text_bboxes:
                annotation_idx = bbox.get('annotation_index')
                if annotation_idx is None:
                    continue
                original_bbox = original_bbox_map.get(annotation_idx)
                if not original_bbox:
                    continue
                points = bbox.get('points', [])
                original_points = original_bbox.get('points', [])
                if len(points) >= 4 and len(original_points) >= 4:
                    conf_threshold = bbox.get('conf', 0.8)
                    sim_tasks_local.append({
                        'frame_img': frame_img,
                        'template_img': template_img,
                        'transformed_points': points,
                        'original_points': original_points,
                        'serial_number': serial_number,
                        'annotation_idx': annotation_idx,
                        'conf_threshold': conf_threshold,
                    })

        # Run sim checks in parallel (non-blocking with OCR below)
        sim_results_map = {}
        if sim_tasks_local:
            logger.info(
                f"[{serial_number}] Sim check: {len(sim_tasks_local)} regions "
                f"(workers={self.SIM_MAX_WORKERS})"
            )
            with ThreadPoolExecutor(max_workers=self.SIM_MAX_WORKERS) as sim_executor:
                futures = {
                    sim_executor.submit(
                        self._compute_single_sim,
                        st['frame_img'],
                        st['template_img'],
                        st['transformed_points'],
                        st['original_points'],
                        st['serial_number'],
                        st['annotation_idx'],
                        st['conf_threshold'],
                    ): st['annotation_idx']
                    for st in sim_tasks_local
                }
                for future in as_completed(futures):
                    ann_idx = futures[future]
                    try:
                        sim_results_map[ann_idx] = future.result()
                    except Exception as e:
                        logger.error(f"[{serial_number}] Sim check thread error ann {ann_idx}: {e}")

        for bbox in text_bboxes:
            result = self._verify_single_text_region(
                frame_img=frame_img,
                bbox=bbox,
                expected_texts=expected_texts,
                serial_number=serial_number,
                recognition_threshold=recognition_threshold
            )

            # Merge sim result
            if self.use_sim_check:
                ann_idx = bbox.get('annotation_index')
                sim_result = sim_results_map.get(ann_idx)
                if sim_result:
                    result['match_sim'] = sim_result['match_sim']
                    result['similarity'] = sim_result.get('similarity', 0.0)
                    logger.info(
                        f"[{serial_number}] Annotation {ann_idx}: "
                        f"FINAL match={result['match']}, match_sim={sim_result['match_sim']}, "
                        f"similarity={sim_result.get('similarity', 0.0):.4f}"
                    )
                else:
                    result['match_sim'] = None
                    result['similarity'] = None

            verification_results.append(result)

            if not result.get('match', False):
                all_match = False

        return {
            'all_match': all_match,
            'results': verification_results
        }

    def _verify_single_text_region(
        self,
        frame_img: 'np.ndarray',
        bbox: Dict[str, Any],
        expected_texts: Dict[int, str],
        serial_number: str,
        recognition_threshold: float
    ) -> Dict[str, Any]:
        """
        Verify a single text region.

        Args:
            frame_img: Input frame
            bbox: Bounding box dict with 'points', 'annotation_index', etc.
            expected_texts: Dict mapping annotation_idx -> expected_text
            serial_number: Camera serial number for logging
            recognition_threshold: Minimum confidence threshold

        Returns:
            Result dict with verification details
        """
        annotation_idx = bbox.get('annotation_index')
        conf_threshold = bbox.get('conf', 0.8)
        expected_text = ''

        try:
            if annotation_idx is None:
                logger.warning(f"[{serial_number}] Bbox missing annotation_index, skipping")
                return {
                    'annotation_idx': None,
                    'expected': '',
                    'recognized': '',
                    'match': False,
                    'confidence': 0.0,
                    'threshold': conf_threshold,
                    'error': 'Missing annotation_index'
                }

            # Get expected text using annotation_index
            expected_text = expected_texts.get(annotation_idx, '')
            logger.info(
                f"[{serial_number}] Processing annotation {annotation_idx}: "
                f"expected_text='{expected_text}'"
            )

            # Validate bbox points
            points = bbox.get('points', [])
            if len(points) < 4:
                logger.warning(f"[{serial_number}] Invalid points for annotation {annotation_idx}")
                return {
                    'annotation_idx': annotation_idx,
                    'expected': expected_text,
                    'recognized': '',
                    'match': False,
                    'confidence': 0.0,
                    'threshold': conf_threshold,
                    'error': 'Invalid bbox points'
                }

            # Crop text region
            cropped_region = crop_text_region(frame_img, points)

            # Save debug image if enabled
            if self.save_debug_images:
                debug_file = f"{self.debug_path}/cropped_region_{serial_number}_{annotation_idx}_{self._debug_counter}_{int(time.time())}.png"
                cv2.imwrite(debug_file, cropped_region)
                self._debug_counter += 1

            # Run OCR (single inference - get char_confs nếu cần, tránh double infer)
            logger.debug(f"[{serial_number}] Running OCR with {self.ocr_backend} backend...")
            if self.use_char_conf_check and hasattr(self.text_recognizer, 'recognize_with_char_conf'):
                text, confidence, char_confs = self.text_recognizer.recognize_with_char_conf(cropped_region)
            else:
                text, confidence = self.text_recognizer.recognize(cropped_region, return_confidence=True)
                char_confs = None
            recognized_text = text.strip()
            logger.debug(f"[{serial_number}] OCR result: '{recognized_text}' (conf: {confidence:.2%})")

            # Check confidence threshold
            if confidence < conf_threshold:
                logger.warning(
                    f"[{serial_number}] Annotation {annotation_idx}: "
                    f"Low confidence {confidence:.2%} < threshold {conf_threshold:.2%}, "
                    f"treating as NO MATCH"
                )
                match = False
            elif self.use_char_conf_check and char_confs is not None:
                low_chars = [(c, cf) for c, cf in char_confs if c.isalnum() and cf < conf_threshold]
                if low_chars:
                    logger.warning(
                        f"[{serial_number}] Annotation {annotation_idx}: "
                        f"Low per-char conf {low_chars}, treating as NO MATCH"
                    )
                    match = False
                else:
                    match = None  # tiếp tục so sánh text bên dưới
            else:
                match = None  # tiếp tục so sánh text bên dưới

            if match is None:
                # Compare texts using similarity matching for specific patterns
                # if "BEST BEFORE" in expected_text.upper() or "PL" in expected_text.upper() or "MFG" in expected_text.upper() or "BB" in expected_text.upper():
                #     # Use similarity matching (default 80% threshold)
                #     similarity = calculate_text_similarity(recognized_text, expected_text)
                #     similarity_threshold = 0.90
                #     match = similarity >= similarity_threshold
                #     if match:
                #         recognized_text = expected_text[:]  # Override with expected text on match

                #     logger.info(
                #         f"[{serial_number}] Annotation {annotation_idx}: "
                #         f"Using similarity matching - similarity={similarity:.2%}, "
                #         f"threshold={similarity_threshold:.2%}, match={match}"
                #     )
                # else:
                # Use exact match
                if "USsed" in recognized_text:
                    recognized_text = recognized_text.replace("USsed", "Used")

                if "Iif" in recognized_text:
                    recognized_text = recognized_text.replace("Iif", "If")
                    
                if "Fo" in recognized_text:
                    recognized_text = recognized_text.replace("Fo", "FO")

                if "oR" in recognized_text:
                    recognized_text = recognized_text.replace("oR", "OR")
                
                # if "MRR" in recognized_text:
                #     recognized_text = recognized_text.replace("MRR", "MAR")
                
                # if "HAR" in recognized_text:    
                #     recognized_text = recognized_text.replace("HAR", "MAR")
                
                # if "HRR" in recognized_text:    
                #     recognized_text = recognized_text.replace("HRR", "MAR")
                
                # if "RL" in recognized_text:
                #     recognized_text = recognized_text.replace("RL", "PL")
                

                match = compare_texts(recognized_text, expected_text, case_sensitive=False, strip=True)
                if match: 
                    recognized_text = expected_text[:]

            logger.info(
                f"[{serial_number}] Annotation {annotation_idx}: "
                f"expected='{expected_text}', recognized='{recognized_text}', "
                f"match={match}, conf={confidence:.2%}"
            )

            return {
                'annotation_idx': annotation_idx,
                'expected': expected_text,
                'recognized': recognized_text,
                'match': match,
                'confidence': confidence,
                'threshold': conf_threshold,
                'char_confs': [
                    {'char': c, 'conf': round(cf, 4)} for c, cf in char_confs if c.isalnum()
                ] if char_confs else None,
            }

        except Exception as e:
            logger.error(f"[{serial_number}] Error verifying annotation {annotation_idx}: {e}")
            import traceback
            traceback.print_exc()

            return {
                'annotation_idx': annotation_idx,
                'expected': expected_text,
                'recognized': '',
                'match': False,
                'confidence': 0.0,
                'threshold': conf_threshold,
                'error': str(e)
            }

    def batch_verify_multi_camera(
        self,
        ocr_tasks: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Batch OCR verification for ALL cameras at once.

        Args:
            ocr_tasks: List of tasks, each containing:
                {
                    'serial_number': str,
                    'frame_img': np.ndarray,
                    'transformed_bboxes': list,
                    'expected_texts': dict,
                    'camera': Camera,
                    'recognition_threshold': float
                }

        Returns:
            Dict mapping serial_number -> verification_result
        """
        if not self.is_available:
            logger.warning("OCR model not available, skipping batch text verification")
            return {
                task['serial_number']: {'all_match': False, 'results': []}
                for task in ocr_tasks
            }

        # ========== PHASE 1: Collect ALL text regions from ALL cameras ==========
        all_cropped_regions = []
        all_metadata = []
        sim_tasks = []  # Similarity check tasks (only when use_sim_check=True)

        for task in ocr_tasks:
            serial_number = task['serial_number']
            frame_img = task['frame_img']
            transformed_bboxes = task['transformed_bboxes']
            expected_texts = task['expected_texts']
            template_img = task.get('template_img')        # for sim check
            original_bboxes = task.get('original_bboxes', [])  # for sim check

            # Build lookup: annotation_index -> original bbox (for sim check)
            original_bbox_map = {}
            if self.use_sim_check and template_img is not None:
                for ob in original_bboxes:
                    if ob.get('type') in ['text', 'datecode']:
                        ob_idx = ob.get('annotation_index')
                        if ob_idx is not None:
                            original_bbox_map[ob_idx] = ob

            # Filter text bboxes
            text_bboxes = [
                bbox for bbox in transformed_bboxes
                if bbox.get('type') in ['text', 'datecode']
            ]
            logger.info(f"[{serial_number}] Collecting {len(text_bboxes)} text regions for batch OCR")

            for bbox in text_bboxes:
                annotation_idx = bbox.get('annotation_index')
                if annotation_idx is None:
                    continue

                points = bbox.get('points', [])
                if len(points) < 4:
                    continue

                conf_threshold = bbox.get('conf', 0.8)
                expected_text = expected_texts.get(annotation_idx, '')

                try:
                    cropped_region = crop_text_region(frame_img, points)

                    # Save debug image if enabled
                    if self.save_debug_images:
                        debug_file = f"{self.debug_path}/cropped_region_{serial_number}_{annotation_idx}.png"
                        cv2.imwrite(debug_file, cropped_region)

                    all_cropped_regions.append(cropped_region)
                    all_metadata.append({
                        'serial_number': serial_number,
                        'annotation_idx': annotation_idx,
                        'conf_threshold': conf_threshold,
                        'expected_text': expected_text,
                        'camera': task['camera'],
                        'cropped_region': cropped_region  # kept for augment retry
                    })

                    # Collect sim task if enabled and original bbox exists
                    if self.use_sim_check and template_img is not None:
                        original_bbox = original_bbox_map.get(annotation_idx)
                        if original_bbox:
                            original_points = original_bbox.get('points', [])
                            if len(original_points) >= 4:
                                sim_tasks.append({
                                    'frame_img': frame_img,
                                    'template_img': template_img,
                                    'transformed_points': points,
                                    'original_points': original_points,
                                    'serial_number': serial_number,
                                    'annotation_idx': annotation_idx,
                                    'conf_threshold': conf_threshold,
                                })

                except Exception as e:
                    logger.error(f"[{serial_number}] Error cropping annotation {annotation_idx}: {e}")

        if not all_cropped_regions:
            logger.warning("No valid text regions to process")
            return {
                task['serial_number']: {'all_match': True, 'results': []}
                for task in ocr_tasks
            }

        # ========== PHASE 2: OCR batch + Sim checks IN PARALLEL ==========
        t_phase2_start = time.perf_counter()

        # --- Thread A: Sim checks (parallel per region) ---
        sim_results_map = {}  # (serial_number, annotation_idx) -> sim result
        sim_future = None

        if self.use_sim_check and sim_tasks:
            logger.info(
                f"Sim check ENABLED: {len(sim_tasks)} regions to check "
                f"(workers={self.SIM_MAX_WORKERS})"
            )

            def _run_all_sim_checks():
                t_sim_start = time.perf_counter()
                results = {}
                with ThreadPoolExecutor(max_workers=self.SIM_MAX_WORKERS) as sim_executor:
                    futures = {
                        sim_executor.submit(
                            self._compute_single_sim,
                            st['frame_img'],
                            st['template_img'],
                            st['transformed_points'],
                            st['original_points'],
                            st['serial_number'],
                            st['annotation_idx'],
                            st['conf_threshold'],
                        ): (st['serial_number'], st['annotation_idx'])
                        for st in sim_tasks
                    }
                    for future in as_completed(futures):
                        key = futures[future]
                        try:
                            results[key] = future.result()
                        except Exception as e:
                            logger.error(f"Sim check thread error {key}: {e}")
                            results[key] = {
                                'annotation_idx': key[1],
                                'match_sim': False,
                                'similarity': 0.0,
                                'error': str(e),
                            }
                t_sim_total = (time.perf_counter() - t_sim_start) * 1000
                logger.info(
                    f"Sim check ALL complete: {len(sim_tasks)} regions in {t_sim_total:.1f}ms"
                )
                return results

            # Submit sim checks to run concurrently with OCR
            sim_executor_outer = ThreadPoolExecutor(max_workers=1)
            sim_future = sim_executor_outer.submit(_run_all_sim_checks)
        elif self.use_sim_check:
            logger.info("Sim check ENABLED but no valid sim tasks (missing template_img or original_bboxes)")

        # --- Thread B (main thread): OCR batch ---
        logger.info(
            f"Running BATCH OCR on {len(all_cropped_regions)} regions "
            f"from {len(ocr_tasks)} cameras"
        )

        try:
            t0 = time.perf_counter()

            if hasattr(self.text_recognizer, 'recognize_batch'):
                ocr_results = self.text_recognizer.recognize_batch(all_cropped_regions)
            else:
                ocr_results = [
                    self.text_recognizer.recognize(img, return_confidence=True)
                    for img in all_cropped_regions
                ]

            ocr_time = (time.perf_counter() - t0) * 1000
            logger.info(f"Batch OCR complete: {len(all_cropped_regions)} regions in {ocr_time:.1f}ms")

        except Exception as e:
            logger.error(f"Batch OCR failed: {e}")
            import traceback
            traceback.print_exc()
            # Cancel sim future if running
            if sim_future:
                sim_future.cancel()
                sim_executor_outer.shutdown(wait=False)
            return {
                task['serial_number']: {'all_match': False, 'results': [], 'error': str(e)}
                for task in ocr_tasks
            }

        # --- Wait for sim checks to finish (if running) ---
        if sim_future:
            try:
                sim_results_map = sim_future.result(timeout=10)
            except Exception as e:
                logger.error(f"Sim check future error: {e}")
                sim_results_map = {}
            finally:
                sim_executor_outer.shutdown(wait=False)

        t_phase2_total = (time.perf_counter() - t_phase2_start) * 1000
        logger.info(
            f"Phase 2 (OCR + Sim parallel) complete: {t_phase2_total:.1f}ms"
        )

        # ========== PHASE 3: Distribute results back to cameras ==========
        camera_results = {
            task['serial_number']: {'all_match': True, 'results': []}
            for task in ocr_tasks
        }

        for i, result in enumerate(ocr_results):
            if isinstance(result, dict):
                text, confidence = result["text"], result["confidence"]
                batch_char_confs = result.get("char_confs")
            else:
                text, confidence = result
                batch_char_confs = None

            meta = all_metadata[i]
            serial_number = meta['serial_number']
            annotation_idx = meta['annotation_idx']
            conf_threshold = meta['conf_threshold']
            expected_text = meta['expected_text']

            recognized_text = text.strip()
            char_confs = batch_char_confs  # khởi tạo, có thể được cập nhật bên dưới

            # Check confidence threshold
            if confidence < conf_threshold:
                logger.warning(
                    f"[{serial_number}] Annotation {annotation_idx}: "
                    f"Low confidence {confidence:.2%} < threshold {conf_threshold:.2%}"
                )
                match = False
            elif self.use_char_conf_check:
                if char_confs is None and hasattr(self.text_recognizer, 'recognize_with_char_conf'):
                    _, _, char_confs = self.text_recognizer.recognize_with_char_conf(meta['cropped_region'])
                low_chars = [(c, cf) for c, cf in (char_confs or []) if c.isalnum() and cf < conf_threshold]
                if low_chars:
                    logger.warning(
                        f"[{serial_number}] Annotation {annotation_idx}: "
                        f"Low per-char conf {low_chars}, treating as NO MATCH"
                    )
                    match = False
                else:
                    recognized_text = _apply_text_corrections(recognized_text)
                    match = compare_texts(recognized_text, expected_text, case_sensitive=True, strip=True)
                    if match:
                        recognized_text = expected_text[:]
            else:
                recognized_text = _apply_text_corrections(recognized_text)
                match = compare_texts(recognized_text, expected_text, case_sensitive=True, strip=True)

            logger.info(
                f"[{serial_number}] Annotation {annotation_idx}: "
                f"expected='{expected_text}', recognized='{recognized_text}', "
                f"match={match}, conf={confidence:.2%}"
            )

            # ========== AUGMENT RETRY for failed regions ==========
            if not match:
                similarity = calculate_text_similarity(recognized_text, expected_text)
                if similarity >= AUGMENT_SIMILARITY_THRESHOLD:
                    logger.info(
                        f"[{serial_number}] Annotation {annotation_idx}: "
                        f"FAIL but similarity={similarity:.2%} >= {AUGMENT_SIMILARITY_THRESHOLD:.0%}, "
                        f"retrying with augmentation..."
                    )
                    match, recognized_text = self._augment_retry(
                        cropped_region=meta['cropped_region'],
                        expected_text=expected_text,
                        serial_number=serial_number,
                        annotation_idx=annotation_idx,
                        conf_threshold=conf_threshold,
                    )
                else:
                    logger.info(
                        f"[{serial_number}] Annotation {annotation_idx}: "
                        f"FAIL and similarity={similarity:.2%} < {AUGMENT_SIMILARITY_THRESHOLD:.0%}, "
                        f"skip augment retry (likely background/noise)"
                    )
            if match:
                recognized_text = expected_text[:]

            if not match:
                camera_results[serial_number]['all_match'] = False

            # Build result dict
            region_result = {
                'annotation_idx': annotation_idx,
                'expected': expected_text,
                'recognized': recognized_text,
                'match': match,
                'confidence': confidence,
                'threshold': conf_threshold,
                'char_confs': [
                    {'char': c, 'conf': round(cf, 4)} for c, cf in char_confs if c.isalnum()
                ] if char_confs else None,
            }

            # Merge sim result if available
            if self.use_sim_check:
                sim_key = (serial_number, annotation_idx)
                sim_result = sim_results_map.get(sim_key)
                if sim_result:
                    region_result['match_sim'] = sim_result['match_sim']
                    region_result['similarity'] = sim_result.get('similarity', 0.0)
                    logger.info(
                        f"[{serial_number}] Annotation {annotation_idx}: "
                        f"FINAL match={match}, match_sim={sim_result['match_sim']}, "
                        f"similarity={sim_result.get('similarity', 0.0):.4f}"
                    )
                else:
                    region_result['match_sim'] = None
                    region_result['similarity'] = None

            camera_results[serial_number]['results'].append(region_result)

        return camera_results

    def _augment_retry(
        self,
        cropped_region: np.ndarray,
        expected_text: str,
        serial_number: str,
        annotation_idx: int,
        conf_threshold: float = 0.8,
    ):
        """
        Run OCR on 5 augmented versions of cropped_region (batch per region).
        Returns (match, recognized_text) of the first matching version,
        or (False, best_recognized_text) if none match.
        """
        aug_versions = augment_laser_text(cropped_region)
        aug_names = list(aug_versions.keys())
        aug_images = list(aug_versions.values())

        try:
            t0 = time.time()
            if hasattr(self.text_recognizer, 'recognize_batch'):
                aug_results = self.text_recognizer.recognize_batch(aug_images)
            else:
                aug_results = [
                    self.text_recognizer.recognize(img, return_confidence=True)
                    for img in aug_images
                ]
            elapsed = (time.time() - t0) * 1000
            logger.info(
                f"[{serial_number}] Annotation {annotation_idx}: "
                f"augment batch ({len(aug_images)} versions) in {elapsed:.1f}ms"
            )
        except Exception as e:
            logger.error(f"[{serial_number}] Annotation {annotation_idx}: augment OCR failed: {e}")
            return False, ""

        best_text = ""
        best_conf = -1.0

        for ver_name, aug_result in zip(aug_names, aug_results):
            if isinstance(aug_result, dict):
                aug_text, aug_conf = aug_result["text"], aug_result["confidence"]
                aug_char_confs = aug_result.get("char_confs")
            else:
                aug_text, aug_conf = aug_result
                aug_char_confs = None
            aug_recognized = _apply_text_corrections(aug_text.strip())
            aug_match = compare_texts(aug_recognized, expected_text, case_sensitive=True, strip=True)

            logger.info(
                f"[{serial_number}] Annotation {annotation_idx} "
                f"augment[{ver_name}]: '{aug_recognized}' conf={aug_conf:.2%} match={aug_match}"
            )

            if aug_match:
                if self.use_char_conf_check and aug_char_confs is not None:
                    low_chars = [(c, cf) for c, cf in aug_char_confs if c.isalnum() and cf < conf_threshold]
                    if low_chars:
                        logger.info(
                            f"[{serial_number}] Annotation {annotation_idx}: "
                            f"augment[{ver_name}] text match but low char conf {low_chars}, skip"
                        )
                        if aug_conf > best_conf:
                            best_conf = aug_conf
                            best_text = aug_recognized
                        continue
                logger.info(
                    f"[{serial_number}] Annotation {annotation_idx}: "
                    f"PASS via augment[{ver_name}]"
                )
                return True, expected_text[:]

            if aug_conf > best_conf:
                best_conf = aug_conf
                best_text = aug_recognized

        logger.info(
            f"[{serial_number}] Annotation {annotation_idx}: "
            f"still FAIL after augment retry, best='{best_text}' conf={best_conf:.2%}"
        )
        return False, best_text

    @staticmethod
    def update_bboxes_with_recognized_text(
        transformed_bboxes: List[Dict[str, Any]],
        text_verification: Dict[str, Any]
    ) -> None:
        """
        Update transformed_bboxes with recognized text from text_verification.
        This modifies transformed_bboxes in-place to replace expected text with OCR result.

        Args:
            transformed_bboxes: List of bbox dicts (modified in-place)
            text_verification: Result from verify_text_regions containing recognized texts
        """
        if not text_verification or not text_verification.get('results'):
            return

        # Create lookup map: annotation_idx -> recognized_text
        recognized_map = {}
        for text_result in text_verification['results']:
            annotation_idx = text_result.get('annotation_idx')
            recognized_text = text_result.get('recognized', '')
            if annotation_idx is not None:
                recognized_map[annotation_idx] = recognized_text

        # Update text bboxes with recognized text
        for bbox in transformed_bboxes:
            if bbox.get('type') in ['text', 'datecode']:
                annotation_idx = bbox.get('annotation_index')
                if annotation_idx is not None and annotation_idx in recognized_map:
                    bbox['text'] = recognized_map[annotation_idx]
