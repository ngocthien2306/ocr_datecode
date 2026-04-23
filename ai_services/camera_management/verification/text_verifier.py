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

        # ── ML classification (only when camera has ml_project_id + ml_model_id) ──
        ml_project_id = getattr(camera, 'ml_project_id', None)
        ml_model_id = getattr(camera, 'ml_model_id', None)
        use_ml = bool(self.ml_classifier_service and ml_project_id and ml_model_id)
        ml_results_map: Dict[int, Dict[str, Any]] = {}
        ml_future = None
        ml_executor_outer = None

        if use_ml:
            ml_tasks_local = []
            for bbox in text_bboxes:
                ann_idx = bbox.get('annotation_index')
                if ann_idx is None:
                    continue
                points = bbox.get('points', [])
                if len(points) < 4:
                    continue
                ml_tasks_local.append({
                    'transformed_points': points,
                    'annotation_idx': ann_idx,
                    'conf_threshold': bbox.get('conf', 0.8),
                })

            if ml_tasks_local:
                logger.info(
                    f"[{serial_number}] ML classify: {len(ml_tasks_local)} regions "
                    f"(project={ml_project_id}, model={ml_model_id})"
                )

                def _run_all_ml():
                    out: Dict[int, Dict[str, Any]] = {}
                    with ThreadPoolExecutor(max_workers=self.SIM_MAX_WORKERS) as ml_pool:
                        futs = {
                            ml_pool.submit(
                                self._compute_single_ml,
                                frame_img,
                                mt['transformed_points'],
                                serial_number,
                                mt['annotation_idx'],
                                mt['conf_threshold'],
                                ml_project_id,
                                ml_model_id,
                            ): mt['annotation_idx']
                            for mt in ml_tasks_local
                        }
                        for fut in as_completed(futs):
                            aid = futs[fut]
                            try:
                                out[aid] = fut.result()
                            except Exception as e:
                                logger.error(
                                    f"[{serial_number}] ML thread error ann {aid}: {e}"
                                )
                    return out

                # Run ML in background, join after OCR loop (parallel with OCR)
                ml_executor_outer = ThreadPoolExecutor(max_workers=1)
                ml_future = ml_executor_outer.submit(_run_all_ml)

        # Wait for ML future to finish (started in parallel with OCR above)
        if ml_future is not None:
            try:
                ml_results_map = ml_future.result(timeout=10)
            except Exception as e:
                logger.error(f"[{serial_number}] ML future error: {e}")
                ml_results_map = {}
            finally:
                if ml_executor_outer is not None:
                    ml_executor_outer.shutdown(wait=False)

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
                    result['char_results'] = sim_result.get('char_results', [])
                    result['match'] = result['match'] and sim_result['match_sim']
                    logger.info(
                        f"[{serial_number}] Annotation {ann_idx}: "
                        f"FINAL match={result['match']}, match_sim={sim_result['match_sim']}, "
                        f"similarity={sim_result.get('similarity', 0.0):.4f}"
                    )
                else:
                    result['match_sim'] = None
                    result['similarity'] = None
                    result['char_results'] = []

            # Merge ML classification result
            if use_ml:
                ann_idx = bbox.get('annotation_index')
                ml_result = ml_results_map.get(ann_idx)
                if ml_result:
                    result['ml_pass'] = ml_result['ml_pass']
                    result['ml_p_ok'] = ml_result.get('p_ok', 0.0)
                    result['ml_label'] = ml_result.get('label', 'NG')
                    result['match'] = result['match'] and ml_result['ml_pass']
                    logger.info(
                        f"[{serial_number}] Annotation {ann_idx}: "
                        f"FINAL match={result['match']}, ml_pass={ml_result['ml_pass']}, "
                        f"ml_label={ml_result.get('label')}, p_ok={ml_result.get('p_ok', 0.0):.4f}"
                    )
                else:
                    result['ml_pass'] = None
                    result['ml_p_ok'] = None
                    result['ml_label'] = None

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
                raw = self.text_recognizer.recognize_with_char_conf(cropped_region)
            else:
                raw = self.text_recognizer.recognize(cropped_region, return_confidence=True)

            candidates = _to_candidates(raw)
            match, recognized_text, confidence, char_confs = pick_winning_candidate(
                candidates, conf_threshold, self.use_char_conf_check, expected_text, case_sensitive=False
            )
            logger.debug(f"[{serial_number}] OCR result: '{recognized_text}' (conf: {confidence:.2%})")

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
        ml_tasks = []   # ML classify tasks (only when camera has ml_project_id + ml_model_id)

        for task in ocr_tasks:
            serial_number = task['serial_number']
            frame_img = task['frame_img']
            transformed_bboxes = task['transformed_bboxes']
            expected_texts = task['expected_texts']
            template_img = task.get('template_img')        # for sim check
            original_bboxes = task.get('original_bboxes', [])  # for sim check
            camera_obj = task.get('camera')

            # ML classify eligibility (per-camera): needs service + ml_project_id + ml_model_id
            ml_project_id = getattr(camera_obj, 'ml_project_id', None)
            ml_model_id = getattr(camera_obj, 'ml_model_id', None)
            use_ml_for_task = bool(
                self.ml_classifier_service and ml_project_id and ml_model_id
            )

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

                    # Collect ML classify task (1 bbox = 1 char, no char segmentation)
                    if use_ml_for_task:
                        ml_tasks.append({
                            'frame_img': frame_img,
                            'transformed_points': points,
                            'serial_number': serial_number,
                            'annotation_idx': annotation_idx,
                            'conf_threshold': conf_threshold,
                            'ml_project_id': ml_project_id,
                            'ml_model_id': ml_model_id,
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

        # --- Thread A2: ML classify (parallel per region, per-camera enabled) ---
        ml_results_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
        ml_future = None
        ml_executor_outer = None

        if ml_tasks:
            logger.info(
                f"ML classify ENABLED: {len(ml_tasks)} regions to classify "
                f"(workers={self.SIM_MAX_WORKERS})"
            )

            def _run_all_ml_checks():
                t_ml_start = time.perf_counter()
                results: Dict[Tuple[str, int], Dict[str, Any]] = {}
                with ThreadPoolExecutor(max_workers=self.SIM_MAX_WORKERS) as ml_pool:
                    futs = {
                        ml_pool.submit(
                            self._compute_single_ml,
                            mt['frame_img'],
                            mt['transformed_points'],
                            mt['serial_number'],
                            mt['annotation_idx'],
                            mt['conf_threshold'],
                            mt['ml_project_id'],
                            mt['ml_model_id'],
                        ): (mt['serial_number'], mt['annotation_idx'])
                        for mt in ml_tasks
                    }
                    for fut in as_completed(futs):
                        key = futs[fut]
                        try:
                            results[key] = fut.result()
                        except Exception as e:
                            logger.error(f"ML classify thread error {key}: {e}")
                            results[key] = {
                                'annotation_idx': key[1],
                                'ml_pass': False,
                                'p_ok': 0.0,
                                'label': 'NG',
                                'error': str(e),
                            }
                t_ml_total = (time.perf_counter() - t_ml_start) * 1000
                logger.info(
                    f"ML classify ALL complete: {len(ml_tasks)} regions in {t_ml_total:.1f}ms"
                )
                return results

            ml_executor_outer = ThreadPoolExecutor(max_workers=1)
            ml_future = ml_executor_outer.submit(_run_all_ml_checks)

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
            # Cancel ml future if running
            if ml_future:
                ml_future.cancel()
                if ml_executor_outer is not None:
                    ml_executor_outer.shutdown(wait=False)
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

        # --- Wait for ML checks to finish (if running) ---
        if ml_future:
            try:
                ml_results_map = ml_future.result(timeout=10)
            except Exception as e:
                logger.error(f"ML classify future error: {e}")
                ml_results_map = {}
            finally:
                if ml_executor_outer is not None:
                    ml_executor_outer.shutdown(wait=False)

        t_phase2_total = (time.perf_counter() - t_phase2_start) * 1000
        logger.info(
            f"Phase 2 (OCR + Sim + ML parallel) complete: {t_phase2_total:.1f}ms"
        )

        # ========== PHASE 3: Distribute results back to cameras ==========
        camera_results = {
            task['serial_number']: {'all_match': True, 'results': []}
            for task in ocr_tasks
        }

        for i, result in enumerate(ocr_results):
            meta = all_metadata[i]
            serial_number = meta['serial_number']
            annotation_idx = meta['annotation_idx']
            conf_threshold = meta['conf_threshold']
            expected_text = meta['expected_text']

            candidates = _to_candidates(result)
            if self.use_char_conf_check and all(cc is None for _, _, cc in candidates):
                if hasattr(self.text_recognizer, 'recognize_with_char_conf'):
                    raw = self.text_recognizer.recognize_with_char_conf(meta['cropped_region'])
                    candidates = _to_candidates(raw)

            match, recognized_text, confidence, char_confs = pick_winning_candidate(
                candidates, conf_threshold, self.use_char_conf_check, expected_text, case_sensitive=True
            )

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
            sim_pass = True
            if self.use_sim_check:
                sim_key = (serial_number, annotation_idx)
                sim_result = sim_results_map.get(sim_key)
                if sim_result:
                    region_result['match_sim'] = sim_result['match_sim']
                    region_result['similarity'] = sim_result.get('similarity', 0.0)
                    region_result['char_results'] = sim_result.get('char_results', [])
                    sim_pass = sim_result['match_sim']
                    region_result['match'] = match and sim_pass
                    logger.info(
                        f"[{serial_number}] Annotation {annotation_idx}: "
                        f"FINAL match={region_result['match']}, match_sim={sim_result['match_sim']}, "
                        f"similarity={sim_result.get('similarity', 0.0):.4f}"
                    )
                else:
                    region_result['match_sim'] = None
                    region_result['similarity'] = None
                    region_result['char_results'] = []

            # Merge ML classify result if available (per-camera, keyed by serial+ann)
            ml_pass = True
            ml_key = (serial_number, annotation_idx)
            ml_result = ml_results_map.get(ml_key) if ml_results_map else None
            if ml_result:
                region_result['ml_pass'] = ml_result['ml_pass']
                region_result['ml_p_ok'] = ml_result.get('p_ok', 0.0)
                region_result['ml_label'] = ml_result.get('label', 'NG')
                ml_pass = ml_result['ml_pass']
                region_result['match'] = region_result['match'] and ml_pass
                logger.info(
                    f"[{serial_number}] Annotation {annotation_idx}: "
                    f"FINAL match={region_result['match']}, ml_pass={ml_pass}, "
                    f"ml_label={ml_result.get('label')}, p_ok={ml_result.get('p_ok', 0.0):.4f}"
                )

            if not (match and sim_pass and ml_pass):
                camera_results[serial_number]['all_match'] = False

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
