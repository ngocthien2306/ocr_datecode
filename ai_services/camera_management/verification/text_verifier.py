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

def _segment_characters_from_image(img: np.ndarray) -> list:
    """
    Segment characters from a BGR image.
    Returns list of cropped char images (BGR), sorted left-to-right.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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

    # Merge overlapping boxes
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

    # Split wide boxes
    if merged:
        widths = [b[2] for b in merged]
        median_w = float(np.median(widths))
        final_boxes = []
        for box in merged:
            x, y, w, h = box
            if w < median_w * 1.5:
                final_boxes.append(box)
            else:
                roi = thresh[y:y+h, x:x+w]
                v_proj = np.sum(roi, axis=0) / 255
                n_chars = max(2, round(w / median_w))
                split_pts = []
                for ci in range(1, n_chars):
                    ex = int(ci * w / n_chars)
                    sw = max(3, int(median_w * 0.3))
                    left = max(1, ex - sw)
                    right = min(w - 1, ex + sw)
                    if left < right:
                        split_pts.append(left + int(np.argmin(v_proj[left:right])))
                split_pts = sorted(set(split_pts))
                prev = 0
                subs = []
                for sp in split_pts:
                    if sp - prev > 3:
                        subs.append((x + prev, y, sp - prev, h))
                    prev = sp
                if w - prev > 3:
                    subs.append((x + prev, y, w - prev, h))
                final_boxes.extend(subs if len(subs) > 1 else [box])
    else:
        final_boxes = merged

    padding = 2
    char_imgs = []
    for (x, y, w, h) in final_boxes:
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(w_img, x + w + padding), min(h_img, y + h + padding)
        char_imgs.append(img[y1:y2, x1:x2])

    return char_imgs


def _ssim_cv(gray1: np.ndarray, gray2: np.ndarray) -> float:
    """SSIM tính bằng OpenCV (không cần skimage)."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    f1 = gray1.astype(np.float64)
    f2 = gray2.astype(np.float64)
    mu1 = cv2.GaussianBlur(f1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(f2, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(f1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(f2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(f1 * f2, (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


def _char_quality(tmpl_char: np.ndarray, tgt_char: np.ndarray, size=(64, 64)) -> dict:
    """Compute per-char quality metrics: SSIM, sharpness, edge, stroke, CC."""
    t1 = cv2.resize(cv2.cvtColor(tmpl_char, cv2.COLOR_BGR2GRAY), size)
    t2 = cv2.resize(cv2.cvtColor(tgt_char, cv2.COLOR_BGR2GRAY), size)

    sim = _ssim_cv(t1, t2)

    # Sharpness (Laplacian variance)
    sharp_t = float(np.var(cv2.Laplacian(t1, cv2.CV_64F)))
    sharp_g = float(np.var(cv2.Laplacian(t2, cv2.CV_64F)))
    sharp_ratio = sharp_g / (sharp_t + 1e-8)

    # Edge density (Canny)
    edge_t = float(np.sum(cv2.Canny(t1, 50, 150) > 0))
    edge_g = float(np.sum(cv2.Canny(t2, 50, 150) > 0))
    n_pixels = t1.size
    edge_ratio = (edge_g / n_pixels) / (edge_t / n_pixels + 1e-8)

    # Stroke density
    _, bw1 = cv2.threshold(t1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, bw2 = cv2.threshold(t2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    stroke_t = float(np.sum(bw1 > 0)) / bw1.size
    stroke_g = float(np.sum(bw2 > 0)) / bw2.size
    stroke_ratio = stroke_g / (stroke_t + 1e-8)

    # Connected components
    cc_t = cv2.connectedComponents(bw1)[0] - 1
    cc_g = cv2.connectedComponents(bw2)[0] - 1

    # Defect detection
    defects = []
    if sim < 0.5:
        defects.append("SAI_KY_TU")
    if sharp_ratio < 0.25:
        defects.append("LEM")
    if edge_ratio > 1.5 and stroke_ratio > 1.3:
        defects.append("IN_CHONG")
    if cc_g > cc_t + 2:
        defects.append("GAY_NET")
    if stroke_ratio < 0.5:
        defects.append("MAT_NET")

    return {
        "ssim": sim, "sharp_ratio": sharp_ratio, "edge_ratio": edge_ratio,
        "stroke_ratio": stroke_ratio, "cc_tmpl": cc_t, "cc_tgt": cc_g,
        "defects": defects,
    }


def _save_char_comparison(tmpl_img, tgt_img, tmpl_chars, tgt_chars, char_results, output_path):
    """Save visual comparison image to disk."""
    n = len(char_results)
    if n == 0:
        return
    cell_w, cell_h = 64, 64
    gap = 4
    label_w = 70
    info_h = 40

    # Full images section
    full_w = max(350, label_w + n * (cell_w + gap) + gap)
    tmpl_r = cv2.resize(tmpl_img, (full_w, int(tmpl_img.shape[0] * full_w / tmpl_img.shape[1])))
    tgt_r = cv2.resize(tgt_img, (full_w, int(tgt_img.shape[0] * full_w / tgt_img.shape[1])))

    # 4 rows of chars + info row
    grid_h = 4 * (cell_h + gap) + gap
    total_h = tmpl_r.shape[0] + tgt_r.shape[0] + gap * 5 + 32 + grid_h + info_h
    canvas = np.ones((total_h, full_w, 3), dtype=np.uint8) * 40

    y = gap
    cv2.putText(canvas, "TEMPLATE", (4, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    y += 16
    canvas[y:y + tmpl_r.shape[0], :tmpl_r.shape[1]] = tmpl_r
    y += tmpl_r.shape[0] + gap
    cv2.putText(canvas, "TARGET", (4, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    y += 16
    canvas[y:y + tgt_r.shape[0], :tgt_r.shape[1]] = tgt_r
    y += tgt_r.shape[0] + gap * 2

    grid_y = y
    for ri, label in enumerate(["Tmpl", "TmplThr", "Target", "TgtThr"]):
        ry = grid_y + ri * (cell_h + gap)
        cv2.putText(canvas, label, (4, ry + cell_h // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (160, 160, 160), 1)

    for idx, cr in enumerate(char_results):
        x_off = label_w + idx * (cell_w + gap)
        if x_off + cell_w > full_w:
            break
        defects = cr["defects"]
        border = (0, 0, 255) if defects else (0, 200, 0)
        i = cr["idx"]

        # Row 0: template char
        ry0 = grid_y
        cv2.putText(canvas, str(i), (x_off + cell_w // 2 - 4, ry0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        t_c = cv2.resize(tmpl_chars[i], (cell_w, cell_h))
        canvas[ry0:ry0+cell_h, x_off:x_off+cell_w] = t_c

        # Row 1: template thresh
        ry1 = grid_y + (cell_h + gap)
        g1 = cv2.cvtColor(tmpl_chars[i], cv2.COLOR_BGR2GRAY)
        _, bw1 = cv2.threshold(cv2.resize(g1, (cell_w, cell_h)), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        canvas[ry1:ry1+cell_h, x_off:x_off+cell_w] = cv2.cvtColor(bw1, cv2.COLOR_GRAY2BGR)

        # Row 2: target char
        ry2 = grid_y + 2 * (cell_h + gap)
        t_g = cv2.resize(tgt_chars[i], (cell_w, cell_h))
        canvas[ry2:ry2+cell_h, x_off:x_off+cell_w] = t_g
        cv2.rectangle(canvas, (x_off-1, ry2-1), (x_off+cell_w, ry2+cell_h), border, 2)

        # Row 3: target thresh
        ry3 = grid_y + 3 * (cell_h + gap)
        g2 = cv2.cvtColor(tgt_chars[i], cv2.COLOR_BGR2GRAY)
        _, bw2 = cv2.threshold(cv2.resize(g2, (cell_w, cell_h)), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        canvas[ry3:ry3+cell_h, x_off:x_off+cell_w] = cv2.cvtColor(bw2, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(canvas, (x_off-1, ry3-1), (x_off+cell_w, ry3+cell_h), border, 2)

        # Info row
        y_info = grid_y + 4 * (cell_h + gap)
        color = border
        cv2.putText(canvas, f"S:{cr['ssim']:.2f}", (x_off, y_info + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
        if defects:
            cv2.putText(canvas, "+".join(defects[:2]), (x_off, y_info + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.22, (0, 0, 255), 1)
        else:
            cv2.putText(canvas, "OK", (x_off, y_info + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 0), 1)

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

            # Resize target to match template size if needed
            if cropped_template.shape != cropped_target.shape:
                cropped_target = cv2.resize(
                    cropped_target,
                    (cropped_template.shape[1], cropped_template.shape[0])
                )

            # Resize for faster matching
            cropped_template_r, _ = self._resize_for_matching(cropped_template, self.SIM_MAX_DIMENSION)
            cropped_target_r, _ = self._resize_for_matching(cropped_target, self.SIM_MAX_DIMENSION)

            # Calculate region similarity
            similarity = self._calculate_sim(cropped_template_r, cropped_target_r)
            match_sim = bool(similarity >= conf_threshold)

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
                        tmpl_chars, tgt_chars, char_results, out_path
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
                    {'char': c, 'conf': round(cf, 4)} for c, cf in char_confs
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
                    {'char': c, 'conf': round(cf, 4)} for c, cf in char_confs
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
