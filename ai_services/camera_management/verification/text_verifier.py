"""
Text Verification Service

Handles OCR-based text verification for inference results.
Supports both single-camera and multi-camera batch processing.

Helpers live in sibling modules:
  - char_quality:    character segmentation + per-char similarity/defects + debug strip
  - text_ocr_utils:  laser-text augmentation, OCR candidate selection, text similarity
  - ml_classifier:   sklearn OK/NG classifier loader + predictor
"""

import logging
import time
import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field

from ..ocr_utils import crop_text_region
from ..smtr_utils import _to_candidates

from .char_quality import (
    char_quality as compute_char_quality,
    img_to_b64,
    save_char_comparison,
    segment_characters_from_image,
)
from .text_ocr_utils import (
    AUGMENT_SIMILARITY_THRESHOLD,
    augment_dot_matrix_text,
    augment_laser_text,
    calculate_text_similarity,
    pick_winning_candidate,
)
from .char_preprocess import remove_fragments_local_bg

if TYPE_CHECKING:
    from ..camera import Camera

logger = logging.getLogger(__name__)

home = os.environ.get('HOME')

# Per-recipe classifier backend (read from recipe.classifier_backend → camera).
# Each char_item carries its own value; missing → fallback to embedding.
DEFAULT_CHAR_CLASSIFIER_BACKEND: str = "embedding"


_NUMERIC_CHAR_FIELDS = (
    'confidence', 'tm_conf', 'blur_tm', 'iou', 'pixel_conf',
    'px_tmpl', 'px_tgt', 'sharp_ratio', 'cc_tmpl', 'cc_tgt', 'idx',
)


def _sanitize_char_results(char_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert numpy scalars inside char_result dicts to native Python types
    so the websocket JSON serializer doesn't choke.
    """
    out: List[Dict[str, Any]] = []
    for cr in char_results or []:
        sanitized: Dict[str, Any] = dict(cr)
        for f in _NUMERIC_CHAR_FIELDS:
            if f in sanitized and sanitized[f] is not None:
                try:
                    sanitized[f] = float(sanitized[f])
                except (TypeError, ValueError):
                    pass
        out.append(sanitized)
    return out



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

    # Text/datecode bbox: a single character typically < 150px.
    # Anything larger is almost certainly a projection failure (template mismatch).
    # Use a generous ceiling (2×) so only truly bogus bboxes are rejected.
    MAX_TEXT_CROP_DIM = 2500

    # Char quad shape guard — SuperPoint homography failure makes the projected
    # quad severely trapezoidal / bow-tie / collapsed. ML classifier is then
    # forced to classify noise, slow + always wrong. Reject these early as NG.
    # All thresholds tuned loose; tighten after observing real recipe behavior.
    CHAR_QUAD_MIN_EDGE_RATIO    = 0.5    # min(top,bot)/max(top,bot)  AND  min(L,R)/max(L,R)
    CHAR_QUAD_MAX_TOP_BOT_ANGLE = 22.0   # degrees between top edge vec and bottom edge vec
    CHAR_QUAD_MIN_EDGE_PX       = 3.0    # any edge shorter than this → degenerate
    CHAR_QUAD_ASPECT_DEV_MIN    = 0.5    # transformed (w/h) / original (w/h) must be in
    CHAR_QUAD_ASPECT_DEV_MAX    = 2.0    # this band — else char shape got warped

    # ── V-suffix bypass: Check_Color / Check_Type_Product CIJ dot-matrix, ký tự V cuối ──
    # Khi expected kết thúc bằng 'V' và prefix expected[:-1] khớp ĐÚNG ký tự với
    # recognized, coi như PASS — bất kể ký tự cuối OCR đọc là gì (kể cả mất).
    # Lý do: chữ V dot-matrix dưới cùng có đáy chỉ 1-2 chấm, OCR thường xuyên
    # đọc thành U/chấm/missing dù mọi ký tự khác đúng 100%.
    # KHÔNG dùng counter — bypass này áp dụng vĩnh viễn cho mọi chai cùng pattern.
    # Rủi ro: nếu line có biến thể sản phẩm khác V ở ký tự cuối (vd ...-U thật)
    # → 100% pass nhầm. Cần đảm bảo line chỉ chạy 1 product type tại 1 thời điểm.
    V_SUFFIX_BYPASS_ENABLED                         = True
    V_SUFFIX_BYPASS_LAST_CHARS: Tuple[str, ...]     = ('V',)    # mở rộng nếu sau có chữ khác (vd 'W')
    V_SUFFIX_BYPASS_MIN_EXPECTED_LEN                = 3         # chống case pathological expected='V' hay 'XV'
    V_SUFFIX_BYPASS_FUNCTION_TYPES: Tuple[str, ...] = ("Check_Color", "Check_Type_Product")

    # Khi augment retry fail toàn bộ 5 versions, lưu composite (INPUT crop + 5
    # augments) ra ổ đĩa để review offline. Subfolder ngày trong debug_path:
    # → {debug_path}/augment_fails/{YYYY-MM-DD}/{HHMMSS_xxx}_{serial}_ann{idx}.png
    # Phụ thuộc save_debug_images flag (đã có sẵn). Đặt False nếu sợ đầy đĩa.
    SAVE_AUGMENT_FAIL_DEBUG = False

    # Camera Check_Color in dot-matrix (CIJ) lên nắp chai → cần augment khác hẳn:
    # gộp các chấm thành nét liền thay vì sharpen (vốn làm khoảng cách chấm rõ hơn).
    # Các function_type khác giữ nguyên augment cũ (chữ in liền, laser-engrave).
    DOT_MATRIX_AUGMENT_FUNCTION_TYPES: Tuple[str, ...] = ("Check_Color",)

    def __init__(
        self,
        text_recognizer: Any,
        ocr_backend: str,
        save_debug_images: bool = True,
        debug_path: Optional[str] = None,
        use_char_conf_check: bool = False,
        use_sim_check: bool = False,
        ml_classifier_service: Optional[Any] = None,
        embedding_classifier_service: Optional[Any] = None,
        embedding_classifier_services: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize TextVerificationService.

        Args:
            text_recognizer: OCR model instance (TensorRT or ONNX)
            ocr_backend: Backend name ("tensorrt" or "onnx")
            save_debug_images: Whether to save cropped regions for debugging
            debug_path: Path to save debug images
            use_sim_check: Whether to run similarity check on text/datecode regions
            ml_classifier_service: Optional MLClassifierService — ML OK/NG classification
                runs only when both the service and camera.ml_project_id/ml_model_id are set.
            embedding_classifier_service: Default embedding service (legacy, fallback).
            embedding_classifier_services: Registry {name: service} keyed by recipe.defect_model
                (e.g. {'arcface': ..., 'supcon': ...}). Picked at runtime per recipe.
        """
        self.text_recognizer = text_recognizer
        self.ocr_backend = ocr_backend
        self.save_debug_images = save_debug_images
        self.debug_path = debug_path or f"{home}/Source/ocr_datecode/ai_services/test_result"
        self._debug_counter = 0
        self.use_char_conf_check = use_char_conf_check
        self.use_sim_check = use_sim_check
        self.ml_classifier_service = ml_classifier_service
        self.embedding_classifier_service = embedding_classifier_service
        self.embedding_classifier_services = embedding_classifier_services or {}
        self._sim_crop_cache = {}  # Cache for template crops (key: (serial, points_tuple))

    @property
    def is_available(self) -> bool:
        """Check if OCR is available"""
        return self.text_recognizer is not None

    def _get_max_batch(self) -> int:
        """
        Resolve OCR chunk size.

        Prefer backend-configured `batch_size` (optimal, tuned for that backend)
        over engine `max_batch` (hard upper limit). Some TensorRT backends
        pre-allocate output buffers sized for a specific seq_len; pushing
        `max_batch` through them overflows the buffer. `batch_size` (default 4
        for OpenOCR) stays safely within the allocation.
        """
        cached = getattr(self, '_cached_max_batch', None)
        if cached is not None:
            return cached

        candidates = []
        for obj in (self.text_recognizer, getattr(self.text_recognizer, '_recognizer', None)):
            if obj is None:
                continue
            for attr in ('_batch_size', 'batch_size'):
                v = getattr(obj, attr, None)
                if v:
                    candidates.append(int(v))
                    break

        if candidates:
            mb = min(candidates)
        else:
            hard = getattr(self.text_recognizer, 'max_batch', None) or \
                   getattr(getattr(self.text_recognizer, '_recognizer', None), 'max_batch', None)
            mb = min(int(hard), 4) if hard else 4

        self._cached_max_batch = mb
        return mb

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
            tmpl_chars = segment_characters_from_image(cropped_template)
            tgt_chars = segment_characters_from_image(cropped_target)
            n_pairs = min(len(tmpl_chars), len(tgt_chars))

            char_results = []
            if n_pairs > 0:
                with ThreadPoolExecutor(max_workers=min(n_pairs, self.SIM_MAX_WORKERS)) as pool:
                    metrics_list = list(pool.map(
                        lambda i: compute_char_quality(tmpl_chars[i], tgt_chars[i]),
                        range(n_pairs)
                    ))
                for i, m in enumerate(metrics_list):
                    char_results.append({
                        "idx": i,
                        **m,
                        "tmpl_char_b64": img_to_b64(tmpl_chars[i]),
                        "tgt_char_b64":  img_to_b64(tgt_chars[i]),
                    })

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
                    save_char_comparison(
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

    def _compute_single_ml(
        self,
        frame_img: np.ndarray,
        transformed_points: List,
        serial_number: str,
        annotation_idx: int,
        conf_threshold: float,
        ml_project_id: str,
        ml_model_id: str,
    ) -> Dict[str, Any]:
        """
        Crop region from frame and run ML classifier to label OK/NG.
        Each bbox is assumed to be a single character (user draws per-char).
        Used as a unit of work for ThreadPoolExecutor.
        """
        try:
            cropped = crop_text_region(frame_img, transformed_points)
            res = self.ml_classifier_service.classify_region(
                region_img=cropped,
                project_id=ml_project_id,
                model_id=ml_model_id,
                conf_threshold=conf_threshold,
                serial_number=serial_number,
                annotation_idx=annotation_idx,
            )
            res['annotation_idx'] = annotation_idx
            return res
        except Exception as e:
            logger.error(
                f"[{serial_number}] ML classify crop error ann {annotation_idx}: {e}"
            )
            return {
                'annotation_idx': annotation_idx,
                'ml_pass': False,
                'p_ok': 0.0,
                'label': 'NG',
                'threshold': conf_threshold,
                'error': str(e),
            }

    # ── Shared core: batch OCR + parallel sim + parallel ML + augment retry ──

    def _run_ocr_batch_with_checks(
        self,
        ocr_items: List[Dict[str, Any]],
        sim_items: List[Dict[str, Any]],
        case_sensitive: bool = True,
    ) -> Dict[Tuple[str, int], Dict[str, Any]]:
        """
        Batch OCR across all text/datecode items, run similarity check in
        parallel, merge, and apply augment retry for failed regions.

        ML predictions are NOT mixed in here — `char` annotations get their
        own classify-only path (`_run_char_batch`).

        Item dict shapes:
          ocr_items: serial_number, annotation_idx, conf_threshold,
                     expected_text, cropped_region
          sim_items: serial_number, annotation_idx, conf_threshold,
                     frame_img, template_img, transformed_points, original_points

        Returns: {(serial_number, annotation_idx) -> region_result}
        """
        if not ocr_items:
            return {}

        # --- Launch sim future (parallel with OCR) ---
        sim_results_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
        sim_future = None
        sim_executor_outer = None
        if sim_items:
            logger.info(
                f"Sim check ENABLED: {len(sim_items)} regions "
                f"(workers={self.SIM_MAX_WORKERS})"
            )

            def _run_all_sim():
                t_sim_start = time.perf_counter()
                out: Dict[Tuple[str, int], Dict[str, Any]] = {}
                with ThreadPoolExecutor(max_workers=self.SIM_MAX_WORKERS) as pool:
                    futs = {
                        pool.submit(
                            self._compute_single_sim,
                            s['frame_img'], s['template_img'],
                            s['transformed_points'], s['original_points'],
                            s['serial_number'], s['annotation_idx'],
                            s['conf_threshold'],
                        ): (s['serial_number'], s['annotation_idx'])
                        for s in sim_items
                    }
                    for fut in as_completed(futs):
                        key = futs[fut]
                        try:
                            out[key] = fut.result()
                        except Exception as e:
                            logger.error(f"Sim thread error {key}: {e}")
                            out[key] = {
                                'annotation_idx': key[1],
                                'match_sim': False,
                                'similarity': 0.0,
                                'error': str(e),
                            }
                t_sim_total = (time.perf_counter() - t_sim_start) * 1000
                logger.info(
                    f"Sim check ALL complete: {len(sim_items)} regions in {t_sim_total:.1f}ms"
                )
                return out

            sim_executor_outer = ThreadPoolExecutor(max_workers=1)
            sim_future = sim_executor_outer.submit(_run_all_sim)

        # --- Batch OCR on main thread (auto-chunk by engine max_batch) ---
        crops = [it['cropped_region'] for it in ocr_items]
        max_batch = self._get_max_batch()
        has_batch_api = hasattr(self.text_recognizer, 'recognize_batch')
        logger.info(
            f"Running BATCH OCR on {len(crops)} regions "
            f"(chunked by max_batch={max_batch})"
        )

        try:
            t0 = time.perf_counter()
            ocr_results: List[Any] = []
            if has_batch_api:
                for i in range(0, len(crops), max_batch):
                    chunk = crops[i:i + max_batch]
                    chunk_results = self.text_recognizer.recognize_batch(chunk)
                    ocr_results.extend(chunk_results)
            else:
                ocr_results = [
                    self.text_recognizer.recognize(img, return_confidence=True)
                    for img in crops
                ]
            ocr_time = (time.perf_counter() - t0) * 1000
            n_chunks = (len(crops) + max_batch - 1) // max_batch if has_batch_api else len(crops)
            logger.info(
                f"Batch OCR complete: {len(crops)} regions in {n_chunks} chunk(s), "
                f"{ocr_time:.1f}ms"
            )
        except Exception as e:
            logger.error(f"Batch OCR failed: {e}")
            import traceback
            traceback.print_exc()
            if sim_future:
                sim_future.cancel()
                sim_executor_outer.shutdown(wait=False)
            return {
                (it['serial_number'], it['annotation_idx']): {
                    'annotation_idx': it['annotation_idx'],
                    'expected': it['expected_text'],
                    'recognized': '',
                    'match': False,
                    'confidence': 0.0,
                    'threshold': it['conf_threshold'],
                    'error': str(e),
                }
                for it in ocr_items
            }

        # --- Wait futures ---
        if sim_future:
            try:
                sim_results_map = sim_future.result(timeout=10)
            except Exception as e:
                logger.error(f"Sim future error: {e}")
            finally:
                sim_executor_outer.shutdown(wait=False)

        # --- Merge per-region: pick candidate → augment retry → sim ---
        region_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
        has_sim = bool(sim_items)

        for item, ocr_res in zip(ocr_items, ocr_results):
            serial = item['serial_number']
            ann_idx = item['annotation_idx']
            conf_thr = item['conf_threshold']
            expected = item['expected_text']
            cropped = item['cropped_region']

            candidates = _to_candidates(ocr_res)
            if self.use_char_conf_check and all(cc is None for _, _, cc in candidates):
                if hasattr(self.text_recognizer, 'recognize_with_char_conf'):
                    raw = self.text_recognizer.recognize_with_char_conf(cropped)
                    candidates = _to_candidates(raw)

            match, recognized, confidence, char_confs = pick_winning_candidate(
                candidates, conf_thr, self.use_char_conf_check, expected,
                case_sensitive=case_sensitive,
            )
            logger.info(
                f"[{serial}] Annotation {ann_idx}: expected='{expected}', "
                f"recognized='{recognized}', match={match}, conf={confidence:.2%}"
            )

            # Early V-suffix bypass: nếu kết quả OCR pass đầu đã đủ điều kiện
            # bypass (prefix khớp 100%, expected kết thúc V, function_type=Check_Color),
            # SKIP augment_retry — tiết kiệm ~44ms TRT call. Augment chỉ giúp đọc
            # đúng V, nhưng V-bypass cho phép trailing tự do nên không cần đọc.
            if not match:
                pre_bypassed, pre_reason = self._check_v_suffix_bypass(
                    expected=expected,
                    recognized=recognized,
                    function_type=item.get('function_type', 'OCR'),
                )
                if pre_bypassed:
                    logger.info(
                        f"[{serial}] Ann {ann_idx}: EARLY V-SUFFIX BYPASS PASS "
                        f"recognized='{recognized}' → expected='{expected}' "
                        f"({pre_reason}) — skipping augment_retry"
                    )
                    match = True

            # Augment retry for failed regions with near-similar text (chỉ chạy
            # nếu early bypass không kích hoạt). Sau augment vẫn thử bypass lần
            # nữa với recognized mới (augment có thể đổi recognized).
            augment_attempted = False
            if not match:
                sim_score = calculate_text_similarity(recognized, expected)
                if sim_score >= AUGMENT_SIMILARITY_THRESHOLD:
                    logger.info(
                        f"[{serial}] Annotation {ann_idx}: FAIL but similarity={sim_score:.2%} "
                        f">= {AUGMENT_SIMILARITY_THRESHOLD:.0%}, retrying with augmentation..."
                    )
                    match, recognized = self._augment_retry(
                        cropped_region=cropped,
                        expected_text=expected,
                        serial_number=serial,
                        annotation_idx=ann_idx,
                        conf_threshold=conf_thr,
                        function_type=item.get('function_type', 'OCR'),
                        frame_img=item.get('frame_img'),
                        points=item.get('points'),
                    )
                    augment_attempted = True
                else:
                    logger.info(
                        f"[{serial}] Annotation {ann_idx}: FAIL and similarity={sim_score:.2%} "
                        f"< {AUGMENT_SIMILARITY_THRESHOLD:.0%}, skip augment retry"
                    )

            # Post-augment V-suffix bypass — fallback nếu augment cũng fail.
            if not match and augment_attempted:
                bypassed, bypass_reason = self._check_v_suffix_bypass(
                    expected=expected,
                    recognized=recognized,
                    function_type=item.get('function_type', 'OCR'),
                )
                if bypassed:
                    logger.info(
                        f"[{serial}] Ann {ann_idx}: V-SUFFIX BYPASS PASS "
                        f"recognized='{recognized}' → expected='{expected}' ({bypass_reason})"
                    )
                    match = True
                else:
                    logger.info(
                        f"[{serial}] Ann {ann_idx}: V-suffix bypass not applied — {bypass_reason}"
                    )

            # Always overwrite recognized with expected for any match (real or bypass)
            # — keep FE display simple and consistent. Drift info stays in logs.
            if match:
                recognized = expected[:]

            # Cast numeric fields to native Python types — OCR backends can
            # return numpy scalars (e.g. np.float32) which are not
            # JSON-serializable and break the websocket send.
            region = {
                'annotation_idx': int(ann_idx),
                'expected': expected,
                'recognized': recognized,
                'match': bool(match),
                'confidence': float(confidence),
                'threshold': float(conf_thr),
                'char_confs': [
                    {'char': c, 'conf': round(float(cf), 4)}
                    for c, cf in char_confs if c.isalnum()
                ] if char_confs else None,
            }

            key = (serial, ann_idx)

            # Merge sim
            if has_sim:
                sim_res = sim_results_map.get(key)
                if sim_res:
                    region['match_sim'] = bool(sim_res['match_sim'])
                    region['similarity'] = float(sim_res.get('similarity', 0.0))
                    region['char_results'] = _sanitize_char_results(
                        sim_res.get('char_results', [])
                    )
                    region['match'] = bool(region['match'] and sim_res['match_sim'])
                    logger.info(
                        f"[{serial}] Annotation {ann_idx}: FINAL match={region['match']}, "
                        f"match_sim={sim_res['match_sim']}, "
                        f"similarity={sim_res.get('similarity', 0.0):.4f}"
                    )
                else:
                    region['match_sim'] = None
                    region['similarity'] = None
                    region['char_results'] = []

            region_map[key] = region

        return region_map

    # ── Char ML batch path ──

    def _run_char_batch(
        self,
        char_items: List[Dict[str, Any]],
    ) -> Dict[Tuple[str, int], Dict[str, Any]]:
        """
        Classify-only path for `char` annotations.

        Routes to EmbeddingClassifierService or MLClassifierService depending
        on each item's `classifier_backend` ('embedding' default | 'ml').

        Each char_item carries: serial_number, annotation_idx, conf_threshold,
        cropped_region, template_crop (embedding only), expected_text,
        ml_project_id, ml_model_id.

        Returns: {(serial, ann_idx) → region_result} with ML-style fields.
        """
        if not char_items:
            return {}

        t0 = time.perf_counter()

        # Per-recipe routing: items in one batch share the same recipe → same backend.
        backend = (char_items[0].get('classifier_backend') or DEFAULT_CHAR_CLASSIFIER_BACKEND).lower()
        if backend == "embedding":
            # Pick embedding service based on recipe.defect_model (carried per item).
            defect_model = (char_items[0].get('defect_model') or 'arcface').lower()
            embed_service = (
                self.embedding_classifier_services.get(defect_model)
                or self.embedding_classifier_service     # legacy fallback
            )
            if not embed_service:
                logger.warning(
                    f"No embedding service for defect_model='{defect_model}'. "
                    f"Available: {list(self.embedding_classifier_services.keys())}"
                )
                return {}
            batch_input = [
                {
                    'region_img':     m['cropped_region'],
                    'template_crop':  m.get('template_crop'),
                    'conf_threshold': m['conf_threshold'],
                    'serial_number':  m['serial_number'],
                    'annotation_idx': m['annotation_idx'],
                    'cv_method':                (m.get('cv_method') or 'v4'),
                    # Template bank HARDCODED disabled — single-template path always
                    'recipe_id':                m.get('recipe_id'),
                    'template_bank_enabled':    False,
                    'template_bank_size':       int(m.get('template_bank_size', 10)),
                    'template_version_key':     m.get('template_version_key'),
                    'char_denoise_enabled':     bool(m.get('char_denoise_enabled', False)),
                }
                for m in char_items
            ]
            results = embed_service.classify_batch(batch_input)
            # Commit bank adds for chars that PASSED — strips _bank_try_add from results.
            # Per-char gating only (frame-level gating could be added later via caller).
            try:
                embed_service.commit_bank_adds(results, should_commit=True)
            except Exception as e:
                logger.warning(f"commit_bank_adds (single-frame) failed: {e}")
            backend_label = f"Embedding[{defect_model}]"
        else:
            if not self.ml_classifier_service:
                return {}
            batch_input = [
                {
                    'region_img':     m['cropped_region'],
                    'project_id':     m['ml_project_id'],
                    'model_id':       m['ml_model_id'],
                    'conf_threshold': m['conf_threshold'],
                    'serial_number':  m['serial_number'],
                    'annotation_idx': m['annotation_idx'],
                    'expected_text':  m.get('expected_text', ''),
                }
                for m in char_items
            ]
            results = self.ml_classifier_service.classify_batch(batch_input)
            backend_label = "ML"

        out: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for m, r in zip(char_items, results):
            key = (m['serial_number'], m['annotation_idx'])
            ml_pass = bool(r.get('ml_pass', False))
            out[key] = {
                'annotation_idx': int(m['annotation_idx']),
                'expected':       m.get('expected_text', ''),
                'match':          ml_pass,
                'ml_label':       r.get('label', 'NG'),
                'ml_p_ok':        float(r.get('p_ok', 0.0)),
                'threshold':      float(m['conf_threshold']),
                'error':          r.get('error'),
                'mask_diff_b64':  r.get('mask_diff_b64'),
            }

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"Char {backend_label} batch: {len(char_items)} regions in {elapsed:.1f}ms")
        return out

    def _run_char_batch_for_frames(
        self,
        char_items_per_frame: List[List[Dict[str, Any]]],
    ) -> List[Dict[Tuple[str, int], Dict[str, Any]]]:
        """
        Char classify with a SINGLE batch call across crops from multiple frames.

        Amortizes the embedding model's fixed setup cost (≈50-310ms per call,
        whichever provider) by merging char crops from N frames into one
        classify_batch invocation, then splitting results back per frame.

        Args:
            char_items_per_frame: List of char_items lists (one per frame).
                Items follow the same shape as _run_char_batch input.

        Returns:
            List of {(serial, ann_idx) → result_dict} maps, one per input
            frame, in the same order as `char_items_per_frame`.
            Empty input frames return {} at that index.
        """
        n_frames = len(char_items_per_frame)
        if n_frames == 0:
            return []

        # Flatten with boundaries so we can split results back per frame.
        flat: List[Dict[str, Any]] = []
        boundaries: List[int] = []  # cumulative end index per frame
        for frame_items in char_items_per_frame:
            flat.extend(frame_items)
            boundaries.append(len(flat))

        if not flat:
            return [{} for _ in range(n_frames)]

        t0 = time.perf_counter()

        # Sanity: all items in a single pipeline tick come from the same camera
        # → same recipe → same backend. Warn if assumption is violated.
        backends = {(m.get('classifier_backend') or DEFAULT_CHAR_CLASSIFIER_BACKEND).lower() for m in flat}
        if len(backends) > 1:
            logger.warning(
                f"[_run_char_batch_for_frames] mixed backends {backends} across frames — "
                "falling back to per-frame char batches"
            )
            return [self._run_char_batch(items) for items in char_items_per_frame]

        backend = next(iter(backends))
        if backend == "embedding":
            defect_model = (flat[0].get('defect_model') or 'arcface').lower()
            embed_service = (
                self.embedding_classifier_services.get(defect_model)
                or self.embedding_classifier_service
            )
            if not embed_service:
                logger.warning(
                    f"No embedding service for defect_model='{defect_model}'. "
                    f"Available: {list(self.embedding_classifier_services.keys())}"
                )
                return [{} for _ in range(n_frames)]
            batch_input = [
                {
                    'region_img':     m['cropped_region'],
                    'template_crop':  m.get('template_crop'),
                    'conf_threshold': m['conf_threshold'],
                    'serial_number':  m['serial_number'],
                    'annotation_idx': m['annotation_idx'],
                    'cv_method':                (m.get('cv_method') or 'v4'),
                    'recipe_id':                m.get('recipe_id'),
                    'template_bank_enabled':    False,  # HARDCODED disabled
                    'template_bank_size':       int(m.get('template_bank_size', 10)),
                    'template_version_key':     m.get('template_version_key'),
                    'char_denoise_enabled':     bool(m.get('char_denoise_enabled', False)),
                }
                for m in flat
            ]
            results = embed_service.classify_batch(batch_input)
            try:
                embed_service.commit_bank_adds(results, should_commit=True)
            except Exception as e:
                logger.warning(f"commit_bank_adds (multi-frame) failed: {e}")
            backend_label = f"Embedding[{defect_model}]"
        else:
            if not self.ml_classifier_service:
                return [{} for _ in range(n_frames)]
            batch_input = [
                {
                    'region_img':     m['cropped_region'],
                    'project_id':     m['ml_project_id'],
                    'model_id':       m['ml_model_id'],
                    'conf_threshold': m['conf_threshold'],
                    'serial_number':  m['serial_number'],
                    'annotation_idx': m['annotation_idx'],
                    'expected_text':  m.get('expected_text', ''),
                }
                for m in flat
            ]
            results = self.ml_classifier_service.classify_batch(batch_input)
            backend_label = "ML"

        # Split results back per frame using boundaries.
        per_frame: List[Dict[Tuple[str, int], Dict[str, Any]]] = []
        prev = 0
        for end_idx, frame_items in zip(boundaries, char_items_per_frame):
            chunk_items = flat[prev:end_idx]
            chunk_results = results[prev:end_idx]
            out: Dict[Tuple[str, int], Dict[str, Any]] = {}
            for m, r in zip(chunk_items, chunk_results):
                key = (m['serial_number'], m['annotation_idx'])
                out[key] = {
                    'annotation_idx': int(m['annotation_idx']),
                    'expected':       m.get('expected_text', ''),
                    'match':          bool(r.get('ml_pass', False)),
                    'ml_label':       r.get('label', 'NG'),
                    'ml_p_ok':        float(r.get('p_ok', 0.0)),
                    'threshold':      float(m['conf_threshold']),
                    'error':          r.get('error'),
                    'mask_diff_b64':  r.get('mask_diff_b64'),
                }
            per_frame.append(out)
            prev = end_idx

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"Char {backend_label} batched-frames: {len(flat)} regions across "
            f"{n_frames} frame(s) in {elapsed:.1f}ms (1 batch call)"
        )
        return per_frame

    # ── Helper: build ocr/sim/ml items from a single (camera, template) context ──

    def _validate_text_bbox(
        self,
        points: List,
        frame_shape: tuple,
    ) -> Tuple[bool, Optional[str]]:
        """
        Cheap pre-crop sanity check: reject bboxes whose projected bounds
        are way outside the frame or produce absurd crop sizes.

        Returns (is_valid, reason_if_invalid).
        """
        try:
            xs = [float(p[0]) for p in points]
            ys = [float(p[1]) for p in points]
        except Exception:
            return False, 'malformed_points'

        frame_h, frame_w = frame_shape[:2]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        crop_w = int(max_x - min_x)
        crop_h = int(max_y - min_y)

        if crop_w <= 1 or crop_h <= 1:
            return False, f'degenerate_crop_{crop_w}x{crop_h}'

        max_dim = self.MAX_TEXT_CROP_DIM
        if crop_w > max_dim or crop_h > max_dim:
            return False, f'oversize_crop_{crop_w}x{crop_h}'

        # Bbox substantially outside frame (>50% of frame dim beyond edges) → bogus
        margin_x = frame_w * 0.5
        margin_y = frame_h * 0.5
        if (max_x < -margin_x or min_x > frame_w + margin_x or
                max_y < -margin_y or min_y > frame_h + margin_y):
            return False, f'out_of_frame_{int(min_x)},{int(min_y)}→{int(max_x)},{int(max_y)}'

        return True, None

    @classmethod
    def _validate_char_quad(
        cls,
        points: List,
        original_points: Optional[List] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Shape sanity for char quads coming from SuperPoint homography.

        When the template doesn't match, the projected quad becomes severely
        trapezoidal / bow-tie / collapsed. Cropping such a region warps noise
        into the ML classifier — slow AND always wrong. Cheap geometric checks
        catch these before crop+embed+predict.

        Checks (in order):
          1. min edge length ≥ MIN_EDGE_PX (degenerate guard)
          2. min(top,bot)/max(top,bot) ≥ MIN_EDGE_RATIO (horizontal trapezoid)
          3. min(left,right)/max(left,right) ≥ MIN_EDGE_RATIO (vertical trapezoid)
          4. angle between top edge and bottom edge ≤ MAX_TOP_BOT_ANGLE
             (top/bottom should be roughly parallel)
          5. convexity — cross products of consecutive edges same sign
             (bow-tie / self-intersecting quad = homography blew up)
          6. aspect ratio deviation vs original within [ASPECT_DEV_MIN, MAX]
             (skipped if original_points not provided)

        Returns (is_valid, reason_if_invalid).
        """
        try:
            pts = np.asarray(points, dtype=np.float32)
            if pts.shape != (4, 2):
                return False, 'quad_bad_shape'
        except Exception:
            return False, 'quad_malformed'

        # Reorder to TL, TR, BR, BL using same heuristic as crop_text_region.
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        try:
            tl = pts[int(np.argmin(s))]
            br = pts[int(np.argmax(s))]
            tr = pts[int(np.argmin(d))]
            bl = pts[int(np.argmax(d))]
        except Exception:
            return False, 'quad_order_failed'

        top    = tr - tl
        bottom = br - bl
        left   = bl - tl
        right  = br - tr

        len_top    = float(np.linalg.norm(top))
        len_bottom = float(np.linalg.norm(bottom))
        len_left   = float(np.linalg.norm(left))
        len_right  = float(np.linalg.norm(right))

        # (1) min edge length
        min_edge = min(len_top, len_bottom, len_left, len_right)
        if min_edge < cls.CHAR_QUAD_MIN_EDGE_PX:
            return False, f'edge_too_small_{min_edge:.1f}px'

        # (2) horizontal ratio (top vs bottom)
        h_ratio = min(len_top, len_bottom) / max(len_top, len_bottom)
        if h_ratio < cls.CHAR_QUAD_MIN_EDGE_RATIO:
            return False, f'h_ratio={h_ratio:.2f}<{cls.CHAR_QUAD_MIN_EDGE_RATIO}'

        # (3) vertical ratio (left vs right)
        v_ratio = min(len_left, len_right) / max(len_left, len_right)
        if v_ratio < cls.CHAR_QUAD_MIN_EDGE_RATIO:
            return False, f'v_ratio={v_ratio:.2f}<{cls.CHAR_QUAD_MIN_EDGE_RATIO}'

        # (4) top-bottom angle: both vectors should point in nearly the same direction.
        cos_a = float(np.dot(top, bottom) / max(len_top * len_bottom, 1e-6))
        cos_a = max(-1.0, min(1.0, cos_a))
        angle_deg = float(np.degrees(np.arccos(cos_a)))
        if angle_deg > cls.CHAR_QUAD_MAX_TOP_BOT_ANGLE:
            return False, f'top_bot_angle={angle_deg:.1f}°>{cls.CHAR_QUAD_MAX_TOP_BOT_ANGLE}'

        # (5) convexity — consecutive edge cross products must have consistent sign.
        ordered = [tl, tr, br, bl]
        crosses = []
        for i in range(4):
            a = ordered[i]
            b = ordered[(i + 1) % 4]
            c = ordered[(i + 2) % 4]
            crosses.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
        pos = sum(1 for x in crosses if x > 1e-3)
        neg = sum(1 for x in crosses if x < -1e-3)
        if pos > 0 and neg > 0:
            return False, 'non_convex_quad'

        # (6) aspect ratio vs original template quad — skip if no template ref.
        if original_points is not None:
            try:
                opts = np.asarray(original_points, dtype=np.float32)
                if opts.shape == (4, 2):
                    os_ = opts.sum(axis=1)
                    od_ = np.diff(opts, axis=1).ravel()
                    otl = opts[int(np.argmin(os_))]
                    obr = opts[int(np.argmax(os_))]
                    otr = opts[int(np.argmin(od_))]
                    obl = opts[int(np.argmax(od_))]
                    o_top = float(np.linalg.norm(otr - otl))
                    o_bot = float(np.linalg.norm(obr - obl))
                    o_left = float(np.linalg.norm(obl - otl))
                    o_right = float(np.linalg.norm(obr - otr))

                    tgt_w = (len_top + len_bottom) / 2.0
                    tgt_h = (len_left + len_right) / 2.0
                    orig_w = (o_top + o_bot) / 2.0
                    orig_h = (o_left + o_right) / 2.0

                    tgt_ar = tgt_w / max(tgt_h, 1e-6)
                    orig_ar = orig_w / max(orig_h, 1e-6)
                    dev = tgt_ar / max(orig_ar, 1e-6)
                    if not (cls.CHAR_QUAD_ASPECT_DEV_MIN <= dev <= cls.CHAR_QUAD_ASPECT_DEV_MAX):
                        return False, f'aspect_dev={dev:.2f}'
            except Exception:
                pass

        return True, None

    def _build_items_for_camera(
        self,
        serial_number: str,
        frame_img: np.ndarray,
        transformed_bboxes: List[Dict[str, Any]],
        expected_texts: Dict[int, str],
        camera: 'Camera',
        template_img: Optional[np.ndarray],
        original_bboxes: Optional[List[Dict[str, Any]]],
    ):
        """
        Build per-(camera, template) items for the two independent paths:

          OCR path  — text/datecode bboxes → ocr_items + sim_items
          ML path   — char bboxes (when camera has ml_project_id/model_id)
                       → char_items
                     Skipped entirely if camera has no ML model assigned.

        Returns
            (ocr_items, sim_items, char_items,
             text_bboxes, char_bboxes, invalid_map)
        """
        text_bboxes = [b for b in transformed_bboxes if b.get('type') in ['text', 'datecode']]
        char_bboxes = [b for b in transformed_bboxes if b.get('type') == 'char']

        use_sim_task = self.use_sim_check and template_img is not None
        original_bbox_map: Dict[int, Dict[str, Any]] = {}
        if use_sim_task:
            for ob in (original_bboxes or []):
                if ob.get('type') in ['text', 'datecode']:
                    idx = ob.get('annotation_index')
                    if idx is not None:
                        original_bbox_map[idx] = ob

        ml_project_id = getattr(camera, 'ml_project_id', None)
        ml_model_id = getattr(camera, 'ml_model_id', None)
        camera_backend = (getattr(camera, 'classifier_backend', None) or DEFAULT_CHAR_CLASSIFIER_BACKEND).lower()
        # Embedding path: only needs an embedding service (no ml_project/model required).
        # ML path: requires both service AND recipe.ml_project_id / ml_model_id.
        char_bboxes_count = sum(1 for b in transformed_bboxes if b.get('type') == 'char')
        if camera_backend == "ml":
            use_char_task = bool(self.ml_classifier_service and ml_project_id and ml_model_id)
            if char_bboxes_count > 0 and not (ml_project_id and ml_model_id):
                logger.warning(
                    f"[{serial_number}] Recipe has {char_bboxes_count} char bbox(es) "
                    f"with classifier_backend='ml' but ml_project_id/ml_model_id missing — "
                    "char verification skipped. Either set ml_project + model or switch backend to 'embedding'."
                )
        else:
            use_char_task = bool(self.embedding_classifier_service or self.embedding_classifier_services)

        # Original char points map — used for:
        #   (a) embedding backend: cropping template region per char
        #   (b) BOTH backends: aspect-ratio deviation check in _validate_char_quad
        # Build for every backend so the quad shape guard can compare against
        # the template's natural aspect ratio.
        original_char_bbox_map: Dict[int, Dict[str, Any]] = {}
        if use_char_task:
            for ob in (original_bboxes or []):
                if ob.get('type') == 'char':
                    idx = ob.get('annotation_index')
                    if idx is not None:
                        original_char_bbox_map[idx] = ob

        ocr_items: List[Dict[str, Any]] = []
        sim_items: List[Dict[str, Any]] = []
        char_items: List[Dict[str, Any]] = []
        invalid_map: Dict[Tuple[str, int], Dict[str, Any]] = {}

        # Aggregate crop timing — to spot slow crop_text_region calls
        crop_text_total_ms = 0.0
        crop_char_total_ms = 0.0
        crop_template_total_ms = 0.0
        debug_save_ms = 0.0

        # ── OCR path: text / datecode ──
        for bbox in text_bboxes:
            ann_idx = bbox.get('annotation_index')
            if ann_idx is None:
                continue
            points = bbox.get('points', [])
            if len(points) < 4:
                continue
            conf_threshold = bbox.get('conf', 0.8)
            expected_text = expected_texts.get(ann_idx, '')

            is_valid, reason = self._validate_text_bbox(points, frame_img.shape)
            if not is_valid:
                logger.warning(f"[{serial_number}] Ann {ann_idx}: skipping OCR ({reason})")
                invalid_map[(serial_number, ann_idx)] = {
                    'annotation_idx': int(ann_idx), 'expected': expected_text,
                    'recognized': '', 'match': False, 'confidence': 0.0,
                    'threshold': float(conf_threshold),
                    'error': f'invalid_bbox:{reason}',
                }
                continue

            try:
                _t = time.perf_counter()
                cropped = crop_text_region(frame_img, points)
                crop_text_total_ms += (time.perf_counter() - _t) * 1000
            except Exception as e:
                logger.error(f"[{serial_number}] Error cropping ann {ann_idx}: {e}")
                invalid_map[(serial_number, ann_idx)] = {
                    'annotation_idx': int(ann_idx), 'expected': expected_text,
                    'recognized': '', 'match': False, 'confidence': 0.0,
                    'threshold': float(conf_threshold),
                    'error': f'crop_failed:{e}',
                }
                continue

            if self.save_debug_images:
                _t = time.perf_counter()
                try:
                    cv2.imwrite(
                        f"{self.debug_path}/cropped_region_{serial_number}_{ann_idx}.png",
                        cropped,
                    )
                except Exception as e_save:
                    logger.debug(f"[{serial_number}] Save debug crop failed: {e_save}")
                debug_save_ms += (time.perf_counter() - _t) * 1000

            ocr_items.append({
                'serial_number': serial_number,
                'annotation_idx': ann_idx,
                'conf_threshold': conf_threshold,
                'expected_text': expected_text,
                'cropped_region': cropped,
                # Used by V-suffix bypass + dot-matrix augment gates — both chỉ áp cho Check_Color.
                'function_type': getattr(camera, 'function_type', 'OCR'),
                # Used by _augment_retry pad_x versions để re-crop với polygon
                # mở rộng (lấy pixel thật từ frame thay vì pad nhân tạo).
                'frame_img': frame_img,
                'points': points,
            })

            if use_sim_task and ann_idx in original_bbox_map:
                orig_points = original_bbox_map[ann_idx].get('points', [])
                if len(orig_points) >= 4:
                    sim_items.append({
                        'frame_img': frame_img, 'template_img': template_img,
                        'transformed_points': points, 'original_points': orig_points,
                        'serial_number': serial_number, 'annotation_idx': ann_idx,
                        'conf_threshold': conf_threshold,
                    })

        # ── Char path: char bboxes (ML or embedding, only when camera has model) ──
        if use_char_task:
            # One debug folder per camera per call (same pattern as embedding_classifier)
            char_debug_dir = None
            if self.save_debug_images and char_bboxes:
                try:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    char_debug_dir = os.path.join(
                        self.debug_path, f"char_{serial_number}_{ts}"
                    )
                    os.makedirs(char_debug_dir, exist_ok=True)
                except Exception:
                    char_debug_dir = None

            for bbox in char_bboxes:
                ann_idx = bbox.get('annotation_index')
                if ann_idx is None:
                    continue
                points = bbox.get('points', [])
                if len(points) < 4:
                    continue
                conf_threshold = bbox.get('conf', 0.8)
                expected_char = (bbox.get('text') or expected_texts.get(ann_idx, '') or '').strip()

                if not expected_char:
                    invalid_map[(serial_number, ann_idx)] = {
                        'annotation_idx': int(ann_idx), 'expected': '',
                        'match': False, 'ml_label': 'NG', 'ml_p_ok': 0.0,
                        'threshold': float(conf_threshold),
                        'error': 'no_expected_char',
                    }
                    continue

                is_valid, reason = self._validate_text_bbox(points, frame_img.shape)
                if not is_valid:
                    invalid_map[(serial_number, ann_idx)] = {
                        'annotation_idx': int(ann_idx), 'expected': expected_char,
                        'match': False, 'ml_label': 'NG', 'ml_p_ok': 0.0,
                        'threshold': float(conf_threshold),
                        'error': f'invalid_bbox:{reason}',
                    }
                    continue

                # Shape sanity on the transformed quad — SuperPoint sometimes
                # projects a wildly deformed bbox when the template doesn't
                # match. Reject as NG here so we don't waste a crop + embed +
                # classifier call on a region full of noise.
                orig_pts_for_check = None
                if ann_idx in original_char_bbox_map:
                    orig_pts_for_check = original_char_bbox_map[ann_idx].get('points')
                quad_ok, quad_reason = self._validate_char_quad(points, orig_pts_for_check)
                if not quad_ok:
                    logger.info(
                        f"[{serial_number}] Char ann {ann_idx} ('{expected_char}'): "
                        f"deformed quad → NG, reason={quad_reason}"
                    )
                    invalid_map[(serial_number, ann_idx)] = {
                        'annotation_idx': int(ann_idx), 'expected': expected_char,
                        'match': False, 'ml_label': 'NG', 'ml_p_ok': 0.0,
                        'threshold': float(conf_threshold),
                        'error': f'deformed_quad:{quad_reason}',
                    }
                    continue

                try:
                    _t = time.perf_counter()
                    cropped = crop_text_region(frame_img, points)
                    # Loại fragment kí tự lân cận lọt vào crop — bbox project
                    # qua SuperPoint đôi lúc lấn sang neighbor (vd crop '8' dính
                    # rìa '4'). Giữ chữ chính + defect bên trong, fill local-bg.
                    try:
                        cropped, _frag_mask = remove_fragments_local_bg(cropped)
                        logger.info(f"[{serial_number}] fragment-clean succeeded ann {ann_idx}")
                    except Exception as e_clean:
                        logger.info(
                            f"[{serial_number}] fragment-clean failed ann {ann_idx}: "
                            f"{e_clean} — using raw crop"
                        )
                    crop_char_total_ms += (time.perf_counter() - _t) * 1000
                except Exception as e:
                    invalid_map[(serial_number, ann_idx)] = {
                        'annotation_idx': int(ann_idx), 'expected': expected_char,
                        'match': False, 'ml_label': 'NG', 'ml_p_ok': 0.0,
                        'threshold': float(conf_threshold),
                        'error': f'crop_failed:{e}',
                    }
                    continue

                # Crop template region for embedding mode
                template_crop = None
                if camera_backend == "embedding" and ann_idx in original_char_bbox_map:
                    orig_points = original_char_bbox_map[ann_idx].get('points', [])
                    if len(orig_points) >= 4:
                        try:
                            _t = time.perf_counter()
                            template_crop = crop_text_region(template_img, orig_points)
                            try:
                                template_crop, _frag_mask_tpl = remove_fragments_local_bg(template_crop)
                            except Exception as e_clean:
                                logger.info(
                                    f"[{serial_number}] fragment-clean template failed ann {ann_idx}: "
                                    f"{e_clean} — using raw template crop"
                                )
                            crop_template_total_ms += (time.perf_counter() - _t) * 1000
                        except Exception as e:
                            logger.warning(
                                f"[{serial_number}] Failed to crop template for char "
                                f"ann {ann_idx}: {e}"
                            )

                if char_debug_dir is not None:
                    try:
                        prefix = f"char{ann_idx:02d}_{expected_char}"
                        cv2.imwrite(
                            os.path.join(char_debug_dir, f"{prefix}_target.png"), cropped
                        )
                        if template_crop is not None:
                            cv2.imwrite(
                                os.path.join(char_debug_dir, f"{prefix}_template.png"),
                                template_crop,
                            )
                    except Exception:
                        pass

                char_items.append({
                    'serial_number': serial_number,
                    'annotation_idx': ann_idx,
                    'conf_threshold': conf_threshold,
                    'expected_text': expected_char,
                    'cropped_region': cropped,
                    'template_crop': template_crop,
                    'ml_project_id': ml_project_id,
                    'ml_model_id': ml_model_id,
                    'defect_model': getattr(camera, 'defect_model', None) or 'arcface',
                    'classifier_backend': camera_backend,
                    # CV pipeline variant when classifier_backend='embedding': 'legacy' | 'v4' | 'shape_v7'
                    'cv_method':             (getattr(camera, 'cv_method', None) or 'v4'),
                    # Template bank HARDCODED disabled (per recipe-system refactor) — single-template path always
                    'recipe_id':             getattr(camera, 'recipe_id', None),
                    'template_bank_enabled': False,  # was: getattr(camera, 'template_bank_enabled', False)
                    'template_bank_size':    int(getattr(camera, 'template_bank_size', 10) or 10),
                    'template_version_key':  getattr(camera, 'template_version_key', None),
                    'char_denoise_enabled':  bool(getattr(camera, 'char_denoise_enabled', False)),
                })

        # Log crop-time aggregates so caller can see if cropping is the slow part
        total_crop_ms = crop_text_total_ms + crop_char_total_ms + crop_template_total_ms
        if total_crop_ms > 0 or debug_save_ms > 0:
            logger.info(
                f"[{serial_number}] _build_items crops: "
                f"text={crop_text_total_ms:.1f}ms (n={len(ocr_items)}), "
                f"char={crop_char_total_ms:.1f}ms (n={len(char_items)}), "
                f"template_crop={crop_template_total_ms:.1f}ms, "
                f"debug_save={debug_save_ms:.1f}ms"
            )

        return ocr_items, sim_items, char_items, text_bboxes, char_bboxes, invalid_map

    # ── Public API: single camera (one template) ──

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
        Verify text + char regions for a single (camera, template) invocation.

        Two independent paths:
          - text/datecode  → OCR (+ optional similarity)
          - char           → ML predict (per-char golden-aware)

        Returns:
            {
              'text': {'all_match': bool, 'results': [...]},
              'char': {'all_match': bool, 'results': [...]},
            }
        Frame-level pass = `text.all_match AND char.all_match`.
        Both sub-results have `all_match=True` when they have no items.
        """
        empty_pass = {'all_match': True, 'results': []}
        if not self.is_available:
            logger.warning("OCR model not available, skipping text verification")
            return {'text': {'all_match': False, 'results': []}, 'char': empty_pass}

        serial_number = camera.serial_number
        t_v_start = time.perf_counter()

        t_build_start = time.perf_counter()
        ocr_items, sim_items, char_items, text_bboxes, char_bboxes, invalid_map = (
            self._build_items_for_camera(
                serial_number=serial_number,
                frame_img=frame_img,
                transformed_bboxes=transformed_bboxes,
                expected_texts=expected_texts,
                camera=camera,
                template_img=template_img,
                original_bboxes=original_bboxes,
            )
        )
        t_build_ms = (time.perf_counter() - t_build_start) * 1000

        logger.info(
            f"[{serial_number}] Verifying {len(text_bboxes)} text regions, "
            f"{len(char_bboxes)} char regions  (build_items={t_build_ms:.1f}ms)"
        )

        # ── Text/datecode OCR path ──
        t_ocr_start = time.perf_counter()
        text_region_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
        if ocr_items:
            text_region_map = self._run_ocr_batch_with_checks(
                ocr_items, sim_items, case_sensitive=True,
            )
        text_region_map.update(
            (k, v) for k, v in invalid_map.items() if 'recognized' in v
        )
        t_ocr_ms = (time.perf_counter() - t_ocr_start) * 1000

        text_results: List[Dict[str, Any]] = []
        text_all_match = True
        for bbox in text_bboxes:
            ann_idx = bbox.get('annotation_index')
            if ann_idx is None:
                continue
            r = text_region_map.get((serial_number, ann_idx))
            if r is None:
                continue
            text_results.append(r)
            if not r.get('match', False):
                text_all_match = False
        if text_bboxes and not text_results:
            text_all_match = False

        # ── Char ML path ──
        t_char_start = time.perf_counter()
        char_region_map = self._run_char_batch(char_items)
        char_region_map.update(
            (k, v) for k, v in invalid_map.items() if 'ml_label' in v
        )
        t_char_ms = (time.perf_counter() - t_char_start) * 1000

        char_results: List[Dict[str, Any]] = []
        char_all_match = True
        for bbox in char_bboxes:
            ann_idx = bbox.get('annotation_index')
            if ann_idx is None:
                continue
            r = char_region_map.get((serial_number, ann_idx))
            if r is None:
                continue
            char_results.append(r)
            if not r.get('match', False):
                char_all_match = False
        # No char items at all → pass (empty); some bboxes but no result → fail.
        if char_bboxes and not char_results:
            char_all_match = False

        t_v_total = (time.perf_counter() - t_v_start) * 1000
        logger.info(
            f"[{serial_number}] verify_text_regions BREAKDOWN: "
            f"build={t_build_ms:.1f}ms, ocr={t_ocr_ms:.1f}ms (n={len(ocr_items)}), "
            f"char={t_char_ms:.1f}ms (n={len(char_items)}), total={t_v_total:.1f}ms"
        )

        return {
            'text': {'all_match': text_all_match, 'results': text_results},
            'char': {'all_match': char_all_match, 'results': char_results},
        }

    # ── Public API: single camera, multiple frames (one batched char ML call) ──

    def verify_text_regions_batched_frames(
        self,
        frames_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Verify text + char regions for N frames of the SAME camera using a
        SINGLE char ML batch call across all frames.

        Mirrors `verify_text_regions` semantics per frame but amortizes the
        embedder's fixed setup cost (≈50-310ms depending on provider) over
        all frames instead of paying it once per frame.

        Args:
            frames_data: List of dicts, each with keys matching the args of
                `verify_text_regions`:
                    frame_img: np.ndarray
                    transformed_bboxes: List[Dict]
                    expected_texts: Dict[int, str]
                    camera: Camera
                    template_img: Optional[np.ndarray]
                    original_bboxes: Optional[List[Dict]]

        Returns:
            List of {'text': {...}, 'char': {...}} dicts — one per input
            frame, in the same order. Identical shape to verify_text_regions.
        """
        n_frames = len(frames_data)
        empty_pass = {'all_match': True, 'results': []}
        empty_fail = {'all_match': False, 'results': []}

        if n_frames == 0:
            return []

        if not self.is_available:
            logger.warning("OCR model not available, skipping batched text verification")
            return [{'text': empty_fail, 'char': empty_pass} for _ in range(n_frames)]

        t_v_start = time.perf_counter()

        # ── Phase 1: build items for every frame ──
        t_build_start = time.perf_counter()
        per_frame_items: List[Dict[str, Any]] = []
        for fd in frames_data:
            camera = fd['camera']
            ocr_items, sim_items, char_items, text_bboxes, char_bboxes, invalid_map = (
                self._build_items_for_camera(
                    serial_number=camera.serial_number,
                    frame_img=fd['frame_img'],
                    transformed_bboxes=fd['transformed_bboxes'],
                    expected_texts=fd['expected_texts'],
                    camera=camera,
                    template_img=fd.get('template_img'),
                    original_bboxes=fd.get('original_bboxes'),
                )
            )
            per_frame_items.append({
                'ocr_items':   ocr_items,
                'sim_items':   sim_items,
                'char_items':  char_items,
                'text_bboxes': text_bboxes,
                'char_bboxes': char_bboxes,
                'invalid_map': invalid_map,
                'serial':      camera.serial_number,
            })
        t_build_ms = (time.perf_counter() - t_build_start) * 1000

        # ── Phase 2: per-frame OCR (already batched internally; cheap relative to char) ──
        t_ocr_start = time.perf_counter()
        per_frame_text_map: List[Dict[Tuple[str, int], Dict[str, Any]]] = []
        for items in per_frame_items:
            text_region_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
            if items['ocr_items']:
                text_region_map = self._run_ocr_batch_with_checks(
                    items['ocr_items'], items['sim_items'], case_sensitive=True,
                )
            text_region_map.update(
                (k, v) for k, v in items['invalid_map'].items() if 'recognized' in v
            )
            per_frame_text_map.append(text_region_map)
        t_ocr_ms = (time.perf_counter() - t_ocr_start) * 1000

        # ── Phase 3: MERGED char ML batch across all frames ──
        t_char_start = time.perf_counter()
        per_frame_char_map = self._run_char_batch_for_frames(
            [items['char_items'] for items in per_frame_items]
        )
        # Inject invalid-bbox entries per-frame
        for i, items in enumerate(per_frame_items):
            per_frame_char_map[i].update(
                (k, v) for k, v in items['invalid_map'].items() if 'ml_label' in v
            )
        t_char_ms = (time.perf_counter() - t_char_start) * 1000

        # ── Phase 4: assemble per-frame {text, char} results ──
        out: List[Dict[str, Any]] = []
        for items, text_map, char_map in zip(per_frame_items, per_frame_text_map, per_frame_char_map):
            serial = items['serial']

            text_results: List[Dict[str, Any]] = []
            text_all_match = True
            for bbox in items['text_bboxes']:
                ann_idx = bbox.get('annotation_index')
                if ann_idx is None:
                    continue
                r = text_map.get((serial, ann_idx))
                if r is None:
                    continue
                text_results.append(r)
                if not r.get('match', False):
                    text_all_match = False
            if items['text_bboxes'] and not text_results:
                text_all_match = False

            char_results: List[Dict[str, Any]] = []
            char_all_match = True
            for bbox in items['char_bboxes']:
                ann_idx = bbox.get('annotation_index')
                if ann_idx is None:
                    continue
                r = char_map.get((serial, ann_idx))
                if r is None:
                    continue
                char_results.append(r)
                if not r.get('match', False):
                    char_all_match = False
            if items['char_bboxes'] and not char_results:
                char_all_match = False

            out.append({
                'text': {'all_match': text_all_match, 'results': text_results},
                'char': {'all_match': char_all_match, 'results': char_results},
            })

        t_v_total = (time.perf_counter() - t_v_start) * 1000
        total_chars = sum(len(items['char_items']) for items in per_frame_items)
        total_ocrs = sum(len(items['ocr_items']) for items in per_frame_items)
        logger.info(
            f"verify_text_regions_batched_frames: frames={n_frames}, "
            f"build={t_build_ms:.1f}ms, ocr={t_ocr_ms:.1f}ms (n={total_ocrs}), "
            f"char={t_char_ms:.1f}ms (n={total_chars}, 1 batch call), "
            f"total={t_v_total:.1f}ms"
        )

        return out

    # ── Public API: multi camera batch ──

    def batch_verify_multi_camera(
        self,
        ocr_tasks: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Batch OCR verification across multiple cameras (1 template per camera).
        Wraps all cameras' regions into a single OCR batch call.
        """
        if not self.is_available:
            logger.warning("OCR model not available, skipping batch text verification")
            return {
                task['serial_number']: {'all_match': False, 'results': []}
                for task in ocr_tasks
            }

        ocr_items_all: List[Dict[str, Any]] = []
        sim_items_all: List[Dict[str, Any]] = []
        char_items_all: List[Dict[str, Any]] = []
        invalid_map_all: Dict[Tuple[str, int], Dict[str, Any]] = {}
        text_bboxes_by_serial: Dict[str, List[Dict[str, Any]]] = {}
        char_bboxes_by_serial: Dict[str, List[Dict[str, Any]]] = {}

        for task in ocr_tasks:
            serial_number = task['serial_number']
            logger.info(f"[{serial_number}] Collecting regions for batch OCR")
            ocr_items, sim_items, char_items, text_bboxes, char_bboxes, invalid_map = (
                self._build_items_for_camera(
                    serial_number=serial_number,
                    frame_img=task['frame_img'],
                    transformed_bboxes=task['transformed_bboxes'],
                    expected_texts=task['expected_texts'],
                    camera=task.get('camera'),
                    template_img=task.get('template_img'),
                    original_bboxes=task.get('original_bboxes', []),
                )
            )
            ocr_items_all.extend(ocr_items)
            sim_items_all.extend(sim_items)
            char_items_all.extend(char_items)
            invalid_map_all.update(invalid_map)
            text_bboxes_by_serial[serial_number] = text_bboxes
            char_bboxes_by_serial[serial_number] = char_bboxes

            n_inv = len(invalid_map)
            if n_inv:
                if n_inv == len(text_bboxes):
                    logger.warning(
                        f"[{serial_number}] ALL {n_inv} bboxes invalid — "
                        f"template likely mismatched."
                    )
                else:
                    logger.info(
                        f"[{serial_number}] {n_inv}/{len(text_bboxes)} bboxes invalid"
                    )

        empty_pass = {'all_match': True, 'results': []}
        if not ocr_items_all and not char_items_all and not invalid_map_all:
            logger.warning("No valid text/char regions to process")
            return {
                t['serial_number']: {'text': dict(empty_pass), 'char': dict(empty_pass)}
                for t in ocr_tasks
            }

        # ── Run OCR + ML batches independently ──
        text_region_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
        if ocr_items_all:
            text_region_map = self._run_ocr_batch_with_checks(
                ocr_items_all, sim_items_all, case_sensitive=True,
            )
        text_region_map.update(
            (k, v) for k, v in invalid_map_all.items() if 'recognized' in v
        )

        char_region_map = self._run_char_batch(char_items_all)
        char_region_map.update(
            (k, v) for k, v in invalid_map_all.items() if 'ml_label' in v
        )

        # ── Assemble per-camera results preserving original bbox order ──
        camera_results: Dict[str, Dict[str, Any]] = {
            t['serial_number']: {'text': dict(empty_pass), 'char': dict(empty_pass)}
            for t in ocr_tasks
        }

        for serial, text_bboxes in text_bboxes_by_serial.items():
            results: List[Dict[str, Any]] = []
            all_match = True
            for bbox in text_bboxes:
                ann_idx = bbox.get('annotation_index')
                if ann_idx is None:
                    continue
                r = text_region_map.get((serial, ann_idx))
                if r is None:
                    continue
                results.append(r)
                if not r.get('match', False):
                    all_match = False
            if text_bboxes and not results:
                all_match = False
            camera_results[serial]['text'] = {'all_match': all_match, 'results': results}

        for serial, char_bboxes in char_bboxes_by_serial.items():
            results = []
            all_match = True
            for bbox in char_bboxes:
                ann_idx = bbox.get('annotation_index')
                if ann_idx is None:
                    continue
                r = char_region_map.get((serial, ann_idx))
                if r is None:
                    continue
                results.append(r)
                if not r.get('match', False):
                    all_match = False
            if char_bboxes and not results:
                all_match = False
            camera_results[serial]['char'] = {'all_match': all_match, 'results': results}

        return camera_results

    @staticmethod
    def _sort_quad_tl_tr_br_bl(points) -> Optional[np.ndarray]:
        """
        Sort 4-point polygon thành thứ tự [TL, TR, BR, BL] dùng standard trick:
          - TL = point có x+y nhỏ nhất (góc trên-trái)
          - BR = point có x+y lớn nhất (góc dưới-phải)
          - TR = point có x-y lớn nhất (góc trên-phải)
          - BL = point có x-y nhỏ nhất (góc dưới-trái)
        Approach này robust với bbox xoay ±45° (vs y-then-x sort chỉ work
        với axis-aligned bbox).

        Returns None nếu shape != (4, 2).
        """
        try:
            pts = np.array(points, dtype=np.float32).reshape(-1, 2)
        except Exception:
            return None
        if pts.shape != (4, 2):
            return None
        s = pts[:, 0] + pts[:, 1]
        d = pts[:, 0] - pts[:, 1]
        tl = pts[int(np.argmin(s))]
        br = pts[int(np.argmax(s))]
        tr = pts[int(np.argmax(d))]
        bl = pts[int(np.argmin(d))]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    @classmethod
    def _compute_text_width(cls, points) -> float:
        """
        Tính độ rộng text bbox = trung bình của (top edge length + bot edge length).
        Returns 0.0 nếu points không hợp lệ.
        """
        ordered = cls._sort_quad_tl_tr_br_bl(points)
        if ordered is None:
            return 0.0
        tl, tr, br, bl = ordered
        top_len = float(np.linalg.norm(tr - tl))
        bot_len = float(np.linalg.norm(br - bl))
        return (top_len + bot_len) / 2.0

    @classmethod
    def _extend_quad_x(
        cls,
        points,
        pad_left_px: float,
        pad_right_px: float,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> Optional[List[List[float]]]:
        """
        Mở rộng quad polygon theo hướng text-x (TL→TR direction).
        Clip về biên frame nếu frame_shape (H, W) được cung cấp.

        Args:
            points: 4 corners [[x,y], ...] (any order)
            pad_left_px: extend mép trái thêm bao nhiêu pixel (≥0)
            pad_right_px: extend mép phải thêm bao nhiêu pixel (≥0)
            frame_shape: (h, w) để clip — pass frame.shape[:2]

        Returns:
            new_points [[x,y], ...] theo thứ tự [TL, TR, BR, BL], hoặc None nếu fail.
        """
        ordered = cls._sort_quad_tl_tr_br_bl(points)
        if ordered is None:
            return None
        tl, tr, br, bl = ordered

        # Unit vector hướng text (left → right)
        top_dir = tr - tl
        top_norm = float(np.linalg.norm(top_dir))
        bot_dir = br - bl
        bot_norm = float(np.linalg.norm(bot_dir))
        if top_norm < 1e-6 or bot_norm < 1e-6:
            return None
        top_dir /= top_norm
        bot_dir /= bot_norm

        # Extend
        new_tl = tl - top_dir * pad_left_px
        new_tr = tr + top_dir * pad_right_px
        new_bl = bl - bot_dir * pad_left_px
        new_br = br + bot_dir * pad_right_px

        # Clip về biên frame
        if frame_shape is not None and len(frame_shape) >= 2:
            h, w = frame_shape[0], frame_shape[1]
            for p in (new_tl, new_tr, new_br, new_bl):
                p[0] = max(0.0, min(float(w - 1), float(p[0])))
                p[1] = max(0.0, min(float(h - 1), float(p[1])))

        return [new_tl.tolist(), new_tr.tolist(), new_br.tolist(), new_bl.tolist()]

    def _augment_retry(
        self,
        cropped_region: np.ndarray,
        expected_text: str,
        serial_number: str,
        annotation_idx: int,
        conf_threshold: float = 0.8,
        function_type: str = "OCR",
        frame_img: Optional[np.ndarray] = None,
        points: Optional[List] = None,
    ):
        """
        Run OCR on 9 augmented versions of cropped_region (batch per region):
          - 5 từ augment_laser_text: original, clahe, bg_subtract, unsharp_clahe, tophat
          - 4 pad_x re-crop từ frame_img với polygon mở rộng (cần frame_img + points):
            pad_x_sym_5pct, pad_x_sym_10pct, pad_x_left_15pct, pad_x_right_15pct

        Returns (match, recognized_text) của version đầu tiên match, hoặc
        (False, best_recognized_text) nếu không có version nào match.

        Note: mọi function_type (kể cả Check_Color) đều dùng augment_laser_text.
        """
        # User chốt: mọi function_type (kể cả Check_Color) đều dùng
        # augment_laser_text. `augment_dot_matrix_text` vẫn import + giữ
        # trong text_ocr_utils.py phòng khi cần, nhưng không gọi ở đây nữa.
        aug_versions = augment_laser_text(cropped_region)
        profile = "laser_text"

        # ── Bổ sung 4 phiên bản pad_x (re-crop từ frame gốc với polygon mở rộng) ──
        # Apply cho mọi profile khi có sẵn frame_img + points. Lấy pixel THẬT
        # từ frame gốc → không pad nhân tạo.
        # Note: trước đây gate bằng `profile == "laser_text"` — đã bỏ vì
        # Check_Color đi vào nhánh profile="dot_matrix" (label) nên skip pad_x.
        # Nay apply cho tất cả khi có frame+points.
        #   - pad_x_sym_5pct:    đối xứng 5% width mỗi bên (insurance nhẹ)
        #   - pad_x_sym_10pct:   đối xứng 10% width mỗi bên (insurance vừa)
        #   - pad_x_left_15pct:  chỉ bên trái 15% (vd '0' đầu hay 'B' đầu bị cắt)
        #   - pad_x_right_15pct: chỉ bên phải 15% (vd 'V' cuối hay 'y' cuối bị cắt)
        if frame_img is not None and points is not None:
            try:
                text_w = self._compute_text_width(points)
                if text_w > 0:
                    pad_configs = [
                        ('pad_x_sym_5pct',    text_w * 0.05, text_w * 0.05),
                        ('pad_x_sym_10pct',   text_w * 0.10, text_w * 0.10),
                        ('pad_x_left_15pct',  text_w * 0.15, 0.0),
                        ('pad_x_right_15pct', 0.0,            text_w * 0.15),
                    ]
                    for name, pad_l, pad_r in pad_configs:
                        new_pts = self._extend_quad_x(
                            points,
                            pad_left_px=pad_l,
                            pad_right_px=pad_r,
                            frame_shape=frame_img.shape[:2],
                        )
                        if new_pts is None:
                            continue
                        try:
                            padded_crop = crop_text_region(frame_img, new_pts)
                            aug_versions[name] = padded_crop
                        except Exception as e_pad:
                            logger.warning(
                                f"[{serial_number}] Ann {annotation_idx}: "
                                f"pad version '{name}' crop failed: {e_pad}"
                            )
            except Exception as e_outer:
                logger.warning(
                    f"[{serial_number}] Ann {annotation_idx}: "
                    f"pad_x augment setup failed: {e_outer}"
                )

        aug_names = list(aug_versions.keys())
        aug_images = list(aug_versions.values())
        logger.info(
            f"[{serial_number}] Annotation {annotation_idx}: "
            f"augment profile='{profile}' (function_type={function_type}), "
            f"versions={aug_names}"
        )

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
        # Per-version results — used to build the augment-fail debug composite
        # when ALL 5 versions fail (lets operator inspect crops offline).
        per_version: List[Tuple[str, str, float, bool]] = []

        for ver_name, aug_result in zip(aug_names, aug_results):
            candidates = _to_candidates(aug_result)
            aug_match, aug_recognized, aug_conf, _ = pick_winning_candidate(
                candidates, conf_threshold, self.use_char_conf_check, expected_text, case_sensitive=True
            )
            logger.info(
                f"[{serial_number}] Annotation {annotation_idx} "
                f"augment[{ver_name}]: '{aug_recognized}' conf={aug_conf:.2%} match={aug_match}"
            )
            per_version.append((ver_name, aug_recognized, float(aug_conf), bool(aug_match)))
            if aug_match:
                logger.info(f"[{serial_number}] Annotation {annotation_idx}: PASS via augment[{ver_name}]")
                return True, expected_text[:]
            if aug_conf > best_conf:
                best_conf = aug_conf
                best_text = aug_recognized

        logger.info(
            f"[{serial_number}] Annotation {annotation_idx}: "
            f"still FAIL after augment retry, best='{best_text}' conf={best_conf:.2%}"
        )
        # Persist crops + augments to disk for offline review of why OCR misses
        # the target character (vd chữ V cuối). Fails silent — không ảnh hưởng
        # pipeline nếu disk write lỗi.
        self._save_augment_fail_debug(
            serial_number=serial_number,
            annotation_idx=annotation_idx,
            expected_text=expected_text,
            cropped_region=cropped_region,
            aug_versions=aug_versions,
            per_version=per_version,
        )
        return False, best_text

    def _save_augment_fail_debug(
        self,
        serial_number: str,
        annotation_idx: int,
        expected_text: str,
        cropped_region: np.ndarray,
        aug_versions: Dict[str, np.ndarray],
        per_version: List[Tuple[str, str, float, bool]],
    ) -> None:
        """
        Lưu composite 1 ảnh dọc gồm: INPUT crop + 5 augment versions, mỗi panel
        có 1 dải label trên cho biết tên version + text OCR đọc + conf.

        Chỉ kích hoạt khi cả save_debug_images và SAVE_AUGMENT_FAIL_DEBUG = True.
        Folder: {debug_path}/augment_fails/{YYYY-MM-DD}/
        Filename: {HHMMSS_msec}_{serial}_ann{idx}_exp-{expected}.png
        """
        if not (self.save_debug_images and self.SAVE_AUGMENT_FAIL_DEBUG):
            return
        try:
            date_str = time.strftime("%Y-%m-%d")
            out_dir = os.path.join(self.debug_path, "augment_fails", date_str)
            os.makedirs(out_dir, exist_ok=True)

            panels: List[Tuple[str, np.ndarray, str]] = [
                ("INPUT", cropped_region, f"expected='{expected_text}'"),
            ]
            for ver_name, recognized, conf, matched in per_version:
                tag = "PASS" if matched else "FAIL"
                label = f"{ver_name:14s} {tag}  '{recognized}'  conf={conf:.2%}"
                panels.append((ver_name, aug_versions[ver_name], label))

            target_w = max(p[1].shape[1] for p in panels)
            target_w = max(target_w, 480)  # đảm bảo text label đọc được

            rows: List[np.ndarray] = []
            for name, img, label in panels:
                # Resize ảnh về cùng width (giữ aspect)
                if img.shape[1] != target_w:
                    scale = target_w / img.shape[1]
                    new_h = max(1, int(img.shape[0] * scale))
                    img = cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)
                # Convert grayscale → BGR để vstack được
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # Dải label trên (dark grey, text trắng/xanh/cam)
                band = np.full((28, target_w, 3), 40, dtype=np.uint8)
                if name == "INPUT":
                    color = (220, 220, 220)
                elif "PASS" in label:
                    color = (60, 220, 60)
                else:
                    color = (60, 140, 255)
                cv2.putText(
                    band, label, (6, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
                )
                rows.append(band)
                rows.append(img)

            composite = np.vstack(rows)

            ts_ms = int(time.time() * 1000)
            ts_str = time.strftime("%H%M%S") + f"_{ts_ms % 1000:03d}"
            safe_exp = "".join(c if c.isalnum() else "_" for c in expected_text)[:24]
            fname = f"{ts_str}_{serial_number}_ann{annotation_idx}_exp-{safe_exp}.png"
            out_path = os.path.join(out_dir, fname)
            cv2.imwrite(out_path, composite)
            logger.info(
                f"[{serial_number}] Saved augment-fail debug: {out_path}"
            )
        except Exception as e:
            logger.warning(
                f"[{serial_number}] Failed to save augment-fail debug ann {annotation_idx}: {e}"
            )

    # ── V-suffix bypass (known weak char: V dot-matrix dưới cùng) ──

    @staticmethod
    def _alnum_only(s: str) -> str:
        """Gỡ tất cả ký tự non-alphanumeric. Dùng cho v-suffix bypass vì OCR
        đôi khi đọc text dot-matrix không kèm dấu '-' (vd '0614113311U' thay
        vì '06141-13311-U'). So sánh trên dạng alnum-only để robust."""
        return ''.join(c for c in s if c.isalnum())

    def _check_v_suffix_bypass(
        self,
        expected: str,
        recognized: str,
        function_type: str,
    ) -> Tuple[bool, str]:
        """
        Bypass cho Check_Color CIJ dot-matrix:
        - expected phải kết thúc bằng ký tự trong V_SUFFIX_BYPASS_LAST_CHARS ('V')
        - Sau khi gỡ ký tự non-alphanumeric khỏi cả 2 chuỗi, recognized phải
          bắt đầu ĐÚNG với expected_norm[:-1] (prefix match từng ký tự alnum)
        - recognized_norm chỉ được dài hơn prefix_norm tối đa 1 ký tự
          → cho phép mất ký tự cuối / sai ký tự cuối, chặn garbage dài

        OCR có thể đọc với hoặc không có dấu '-' tuỳ trường hợp (dot-matrix in
        dấu '-' bằng 3-4 chấm ngang, thi thoảng OCR bỏ qua). Normalize đảm bảo
        bypass hoạt động ổn định không phụ thuộc dấu phân cách.

        Returns (bypassed, reason).
        """
        if not self.V_SUFFIX_BYPASS_ENABLED:
            return False, "disabled"
        if function_type not in self.V_SUFFIX_BYPASS_FUNCTION_TYPES:
            return False, f"wrong_function_type={function_type!r}"
        if not expected or len(expected) < self.V_SUFFIX_BYPASS_MIN_EXPECTED_LEN:
            return False, f"expected_too_short_len={len(expected)}"
        if expected[-1] not in self.V_SUFFIX_BYPASS_LAST_CHARS:
            return False, f"expected_last_char={expected[-1]!r}_not_in_target"

        # Normalize: gỡ '-', ' ', etc. khỏi cả hai để so prefix
        expected_norm = self._alnum_only(expected)
        recognized_norm = self._alnum_only(recognized)
        if len(expected_norm) < 2:
            return False, f"expected_norm_too_short={expected_norm!r}"
        # Confirm normalized cũng kết thúc bằng V (an toàn — expected_norm bỏ '-'
        # cuối ra trước, V là ký tự cuối alnum).
        if expected_norm[-1] not in self.V_SUFFIX_BYPASS_LAST_CHARS:
            return False, f"expected_norm_last_char={expected_norm[-1]!r}_not_in_target"

        prefix_norm = expected_norm[:-1]
        if not recognized_norm.startswith(prefix_norm):
            return False, (
                f"prefix_mismatch (expected_prefix_norm={prefix_norm!r}, "
                f"recognized_norm[:{len(prefix_norm)}]={recognized_norm[:len(prefix_norm)]!r})"
            )

        trailing = recognized_norm[len(prefix_norm):]
        if len(trailing) > 1:
            return False, f"too_many_trailing_chars={trailing!r}"

        return True, f"bypassed (trailing={trailing!r})"

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
