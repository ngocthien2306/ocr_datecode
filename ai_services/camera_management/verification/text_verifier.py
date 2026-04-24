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
    augment_laser_text,
    calculate_text_similarity,
    pick_winning_candidate,
)

if TYPE_CHECKING:
    from ..camera import Camera

logger = logging.getLogger(__name__)

home = os.environ.get('HOME')


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

    def __init__(
        self,
        text_recognizer: Any,
        ocr_backend: str,
        save_debug_images: bool = True,
        debug_path: Optional[str] = None,
        use_char_conf_check: bool = False,
        use_sim_check: bool = False,
        ml_classifier_service: Optional[Any] = None,
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
        """
        self.text_recognizer = text_recognizer
        self.ocr_backend = ocr_backend
        self.save_debug_images = save_debug_images
        self.debug_path = debug_path or f"{home}/Source/ocr_datecode/ai_services/test_result"
        self._debug_counter = 0
        self.use_char_conf_check = use_char_conf_check
        self.use_sim_check = use_sim_check
        self.ml_classifier_service = ml_classifier_service
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
        ml_items: List[Dict[str, Any]],
        case_sensitive: bool = True,
    ) -> Dict[Tuple[str, int], Dict[str, Any]]:
        """
        Batch OCR across all items, run sim+ML checks in parallel, merge,
        and apply augment retry for failed regions.

        Item dict shapes:
          ocr_items: serial_number, annotation_idx, conf_threshold,
                     expected_text, cropped_region
          sim_items: serial_number, annotation_idx, conf_threshold,
                     frame_img, template_img, transformed_points, original_points
          ml_items:  serial_number, annotation_idx, conf_threshold,
                     frame_img, transformed_points, ml_project_id, ml_model_id

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

        # --- Launch ML future (parallel with OCR) ---
        ml_results_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
        ml_future = None
        ml_executor_outer = None
        if ml_items:
            logger.info(f"ML classify ENABLED: {len(ml_items)} regions (batched)")

            def _run_all_ml():
                t_ml_start = time.perf_counter()
                # Build classify_batch input: use the already-cropped region
                # from _build_items_for_camera so we don't re-warp.
                batch_input = [
                    {
                        'region_img': m['cropped_region'],
                        'project_id': m['ml_project_id'],
                        'model_id': m['ml_model_id'],
                        'conf_threshold': m['conf_threshold'],
                        'serial_number': m['serial_number'],
                        'annotation_idx': m['annotation_idx'],
                    }
                    for m in ml_items
                ]
                results_list = self.ml_classifier_service.classify_batch(batch_input)
                out: Dict[Tuple[str, int], Dict[str, Any]] = {}
                for m, r in zip(ml_items, results_list):
                    out[(m['serial_number'], m['annotation_idx'])] = r
                t_ml_total = (time.perf_counter() - t_ml_start) * 1000
                logger.info(
                    f"ML classify ALL complete: {len(ml_items)} regions in {t_ml_total:.1f}ms"
                )
                return out

            ml_executor_outer = ThreadPoolExecutor(max_workers=1)
            ml_future = ml_executor_outer.submit(_run_all_ml)

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
            if ml_future:
                ml_future.cancel()
                ml_executor_outer.shutdown(wait=False)
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

        if ml_future:
            try:
                ml_results_map = ml_future.result(timeout=10)
            except Exception as e:
                logger.error(f"ML future error: {e}")
            finally:
                ml_executor_outer.shutdown(wait=False)

        # --- Merge per-region: pick candidate → augment retry → sim → ml ---
        region_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
        has_sim = bool(sim_items)
        has_ml = bool(ml_items)

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

            # Augment retry for failed regions with near-similar text
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
                    )
                else:
                    logger.info(
                        f"[{serial}] Annotation {ann_idx}: FAIL and similarity={sim_score:.2%} "
                        f"< {AUGMENT_SIMILARITY_THRESHOLD:.0%}, skip augment retry"
                    )
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

            # Merge ML
            if has_ml:
                ml_res = ml_results_map.get(key)
                if ml_res:
                    region['ml_pass'] = bool(ml_res['ml_pass'])
                    region['ml_p_ok'] = float(ml_res.get('p_ok', 0.0))
                    region['ml_label'] = ml_res.get('label', 'NG')
                    region['match'] = bool(region['match'] and ml_res['ml_pass'])
                    logger.info(
                        f"[{serial}] Annotation {ann_idx}: FINAL match={region['match']}, "
                        f"ml_pass={ml_res['ml_pass']}, ml_label={ml_res.get('label')}, "
                        f"p_ok={ml_res.get('p_ok', 0.0):.4f}"
                    )
                else:
                    region['ml_pass'] = None
                    region['ml_p_ok'] = None
                    region['ml_label'] = None

            region_map[key] = region

        return region_map

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
        """Return (ocr_items, sim_items, ml_items, text_bboxes) for one camera/template."""
        text_bboxes = [
            b for b in transformed_bboxes
            if b.get('type') in ['text', 'datecode']
        ]

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
        use_ml_task = bool(self.ml_classifier_service and ml_project_id and ml_model_id)

        ocr_items: List[Dict[str, Any]] = []
        sim_items: List[Dict[str, Any]] = []
        ml_items: List[Dict[str, Any]] = []
        invalid_map: Dict[Tuple[str, int], Dict[str, Any]] = {}

        for bbox in text_bboxes:
            ann_idx = bbox.get('annotation_index')
            if ann_idx is None:
                continue
            points = bbox.get('points', [])
            if len(points) < 4:
                continue
            conf_threshold = bbox.get('conf', 0.8)
            expected_text = expected_texts.get(ann_idx, '')

            # Pre-crop sanity check: skip bogus bboxes from bad homography
            is_valid, reason = self._validate_text_bbox(points, frame_img.shape)
            if not is_valid:
                logger.warning(
                    f"[{serial_number}] Ann {ann_idx}: skipping OCR ({reason})"
                )
                invalid_map[(serial_number, ann_idx)] = {
                    'annotation_idx': int(ann_idx),
                    'expected': expected_text,
                    'recognized': '',
                    'match': False,
                    'confidence': 0.0,
                    'threshold': float(conf_threshold),
                    'error': f'invalid_bbox:{reason}',
                }
                continue

            try:
                cropped = crop_text_region(frame_img, points)
            except Exception as e:
                logger.error(f"[{serial_number}] Error cropping ann {ann_idx}: {e}")
                invalid_map[(serial_number, ann_idx)] = {
                    'annotation_idx': int(ann_idx),
                    'expected': expected_text,
                    'recognized': '',
                    'match': False,
                    'confidence': 0.0,
                    'threshold': float(conf_threshold),
                    'error': f'crop_failed:{e}',
                }
                continue

            if self.save_debug_images:
                try:
                    debug_file = f"{self.debug_path}/cropped_region_{serial_number}_{ann_idx}.png"
                    cv2.imwrite(debug_file, cropped)
                except Exception as e_save:
                    logger.debug(f"[{serial_number}] Save debug crop failed: {e_save}")

            ocr_items.append({
                'serial_number': serial_number,
                'annotation_idx': ann_idx,
                'conf_threshold': conf_threshold,
                'expected_text': expected_text,
                'cropped_region': cropped,
            })

            if use_sim_task and ann_idx in original_bbox_map:
                orig_points = original_bbox_map[ann_idx].get('points', [])
                if len(orig_points) >= 4:
                    sim_items.append({
                        'frame_img': frame_img,
                        'template_img': template_img,
                        'transformed_points': points,
                        'original_points': orig_points,
                        'serial_number': serial_number,
                        'annotation_idx': ann_idx,
                        'conf_threshold': conf_threshold,
                    })

            if use_ml_task:
                # Reuse the already-computed crop to avoid warping twice.
                ml_items.append({
                    'cropped_region': cropped,
                    'serial_number': serial_number,
                    'annotation_idx': ann_idx,
                    'conf_threshold': conf_threshold,
                    'ml_project_id': ml_project_id,
                    'ml_model_id': ml_model_id,
                })

        return ocr_items, sim_items, ml_items, text_bboxes, invalid_map

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
        Verify text regions for a single (camera, template) invocation.
        Uses batch OCR internally — no per-region looping.
        """
        if not self.is_available:
            logger.warning("OCR model not available, skipping text verification")
            return {'all_match': False, 'results': []}

        serial_number = camera.serial_number

        ocr_items, sim_items, ml_items, text_bboxes, invalid_map = self._build_items_for_camera(
            serial_number=serial_number,
            frame_img=frame_img,
            transformed_bboxes=transformed_bboxes,
            expected_texts=expected_texts,
            camera=camera,
            template_img=template_img,
            original_bboxes=original_bboxes,
        )

        logger.info(f"[{serial_number}] Verifying {len(text_bboxes)} text regions")
        logger.info(f"[{serial_number}] Expected texts dict: {expected_texts}")
        logger.info(
            f"[{serial_number}] Text bbox annotation indices: "
            f"{[b.get('annotation_index') for b in text_bboxes]}"
        )

        n_invalid = len(invalid_map)
        if n_invalid:
            if n_invalid == len(text_bboxes):
                logger.warning(
                    f"[{serial_number}] ALL {n_invalid} bboxes invalid — "
                    f"template likely mismatched. Skipping OCR batch."
                )
            else:
                logger.info(
                    f"[{serial_number}] {n_invalid}/{len(text_bboxes)} bboxes invalid, "
                    f"OCR will skip those"
                )

        region_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
        if ocr_items:
            region_map = self._run_ocr_batch_with_checks(
                ocr_items, sim_items, ml_items, case_sensitive=True,
            )
        region_map.update(invalid_map)

        # Preserve original bbox order in results
        results: List[Dict[str, Any]] = []
        all_match = True
        for bbox in text_bboxes:
            ann_idx = bbox.get('annotation_index')
            if ann_idx is None:
                continue
            r = region_map.get((serial_number, ann_idx))
            if r is None:
                continue
            results.append(r)
            if not r.get('match', False):
                all_match = False

        if not results:
            all_match = False

        return {'all_match': all_match, 'results': results}

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
        ml_items_all: List[Dict[str, Any]] = []
        invalid_map_all: Dict[Tuple[str, int], Dict[str, Any]] = {}
        text_bboxes_by_serial: Dict[str, List[Dict[str, Any]]] = {}

        for task in ocr_tasks:
            serial_number = task['serial_number']
            logger.info(f"[{serial_number}] Collecting regions for batch OCR")
            ocr_items, sim_items, ml_items, text_bboxes, invalid_map = self._build_items_for_camera(
                serial_number=serial_number,
                frame_img=task['frame_img'],
                transformed_bboxes=task['transformed_bboxes'],
                expected_texts=task['expected_texts'],
                camera=task.get('camera'),
                template_img=task.get('template_img'),
                original_bboxes=task.get('original_bboxes', []),
            )
            ocr_items_all.extend(ocr_items)
            sim_items_all.extend(sim_items)
            ml_items_all.extend(ml_items)
            invalid_map_all.update(invalid_map)
            text_bboxes_by_serial[serial_number] = text_bboxes

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

        if not ocr_items_all and not invalid_map_all:
            logger.warning("No valid text regions to process")
            return {
                task['serial_number']: {'all_match': True, 'results': []}
                for task in ocr_tasks
            }

        region_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
        if ocr_items_all:
            region_map = self._run_ocr_batch_with_checks(
                ocr_items_all, sim_items_all, ml_items_all, case_sensitive=True,
            )
        region_map.update(invalid_map_all)

        camera_results = {
            task['serial_number']: {'all_match': True, 'results': []}
            for task in ocr_tasks
        }
        # Preserve original bbox order per camera
        for serial, text_bboxes in text_bboxes_by_serial.items():
            any_result = False
            for bbox in text_bboxes:
                ann_idx = bbox.get('annotation_index')
                if ann_idx is None:
                    continue
                r = region_map.get((serial, ann_idx))
                if r is None:
                    continue
                any_result = True
                camera_results[serial]['results'].append(r)
                if not r.get('match', False):
                    camera_results[serial]['all_match'] = False
            if not any_result:
                camera_results[serial]['all_match'] = False

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
            candidates = _to_candidates(aug_result)
            aug_match, aug_recognized, aug_conf, _ = pick_winning_candidate(
                candidates, conf_threshold, self.use_char_conf_check, expected_text, case_sensitive=True
            )
            logger.info(
                f"[{serial_number}] Annotation {annotation_idx} "
                f"augment[{ver_name}]: '{aug_recognized}' conf={aug_conf:.2%} match={aug_match}"
            )
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
