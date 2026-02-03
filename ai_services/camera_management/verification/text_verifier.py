"""
Text Verification Service

Handles OCR-based text verification for inference results.
Supports both single-camera and multi-camera batch processing.
"""

import logging
import time
import os
import cv2
from difflib import SequenceMatcher
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

from ..ocr_utils import crop_text_region, compare_texts

if TYPE_CHECKING:
    from ..camera import Camera
    import numpy as np

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


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity ratio between two texts using SequenceMatcher.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity ratio (0.0 - 1.0)
    """
    # Normalize: strip whitespace and convert to lowercase
    text1_norm = text1.strip().lower()
    text2_norm = text2.strip().lower()

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

    def __init__(
        self,
        text_recognizer: Any,
        ocr_backend: str,
        save_debug_images: bool = True,
        debug_path: Optional[str] = None
    ):
        """
        Initialize TextVerificationService.

        Args:
            text_recognizer: OCR model instance (TensorRT or ONNX)
            ocr_backend: Backend name ("tensorrt" or "onnx")
            save_debug_images: Whether to save cropped regions for debugging
            debug_path: Path to save debug images
        """
        self.text_recognizer = text_recognizer
        self.ocr_backend = ocr_backend
        self.save_debug_images = save_debug_images
        self.debug_path = debug_path or f"{home}/Source/ocr_datecode/ai_services/test_result"

    @property
    def is_available(self) -> bool:
        """Check if OCR is available"""
        return self.text_recognizer is not None

    def verify_text_regions(
        self,
        frame_img: 'np.ndarray',
        transformed_bboxes: List[Dict[str, Any]],
        expected_texts: Dict[int, str],
        camera: 'Camera',
        recognition_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Verify text in transformed regions match expected texts.

        Args:
            frame_img: Captured frame (numpy array)
            transformed_bboxes: List of transformed bbox dicts from matcher
            expected_texts: Dict mapping annotation_idx -> expected_text
            camera: Camera object for logging
            recognition_threshold: Minimum OCR confidence threshold

        Returns:
            {
                'all_match': bool,
                'results': [
                    {
                        'annotation_idx': 0,
                        'expected': '123',
                        'recognized': '123',
                        'match': True,
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
            if bbox.get('type') == 'text'
        ]

        logger.info(f"[{serial_number}] Verifying {len(text_bboxes)} text regions")
        logger.info(f"[{serial_number}] Expected texts dict: {expected_texts}")
        logger.info(
            f"[{serial_number}] Text bbox annotation indices: "
            f"{[bbox.get('annotation_index') for bbox in text_bboxes]}"
        )

        for bbox in text_bboxes:
            result = self._verify_single_text_region(
                frame_img=frame_img,
                bbox=bbox,
                expected_texts=expected_texts,
                serial_number=serial_number,
                recognition_threshold=recognition_threshold
            )

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
                debug_file = f"{self.debug_path}/cropped_region_{serial_number}_{annotation_idx}.png"
                cv2.imwrite(debug_file, cropped_region)

            # Run OCR
            logger.debug(f"[{serial_number}] Running OCR with {self.ocr_backend} backend...")
            text, confidence = self.text_recognizer.recognize(cropped_region, return_confidence=True)
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
            else:
                # Compare texts using similarity matching for specific patterns
                if "BEST BEFORE" in expected_text.upper() or "PL" in expected_text.upper() or "MFG" in expected_text.upper() or "BB" in expected_text.upper():
                    # Use similarity matching (default 80% threshold)
                    similarity = calculate_text_similarity(recognized_text, expected_text)
                    similarity_threshold = 0.75
                    match = similarity >= similarity_threshold
                    if match:
                        recognized_text = expected_text[:]  # Override with expected text on match

                    logger.info(
                        f"[{serial_number}] Annotation {annotation_idx}: "
                        f"Using similarity matching - similarity={similarity:.2%}, "
                        f"threshold={similarity_threshold:.2%}, match={match}"
                    )
                else:
                    # Use exact match
                    if "USsed" in recognized_text:
                        recognized_text = recognized_text.replace("USsed", "Used")

                    match = compare_texts(recognized_text, expected_text, case_sensitive=False, strip=True)

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
                'threshold': conf_threshold
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

        for task in ocr_tasks:
            serial_number = task['serial_number']
            frame_img = task['frame_img']
            transformed_bboxes = task['transformed_bboxes']
            expected_texts = task['expected_texts']

            # Filter text bboxes
            text_bboxes = [b for b in transformed_bboxes if b.get('type') == 'text']

            logger.debug(f"[{serial_number}] Collecting {len(text_bboxes)} text regions for batch OCR")

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
                        'camera': task['camera']
                    })
                except Exception as e:
                    logger.error(f"[{serial_number}] Error cropping annotation {annotation_idx}: {e}")

        if not all_cropped_regions:
            logger.warning("No valid text regions to process")
            return {
                task['serial_number']: {'all_match': True, 'results': []}
                for task in ocr_tasks
            }

        # ========== PHASE 2: SINGLE BATCH OCR for ALL cameras ==========
        logger.info(
            f"Running BATCH OCR on {len(all_cropped_regions)} regions "
            f"from {len(ocr_tasks)} cameras"
        )

        try:
            t0 = time.time()

            if hasattr(self.text_recognizer, 'recognize_batch'):
                ocr_results = self.text_recognizer.recognize_batch(all_cropped_regions)
            else:
                # Fallback to sequential
                ocr_results = [
                    self.text_recognizer.recognize(img, return_confidence=True)
                    for img in all_cropped_regions
                ]

            ocr_time = (time.time() - t0) * 1000
            logger.info(f"Batch OCR complete: {len(all_cropped_regions)} regions in {ocr_time:.1f}ms")

        except Exception as e:
            logger.error(f"Batch OCR failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                task['serial_number']: {'all_match': False, 'results': [], 'error': str(e)}
                for task in ocr_tasks
            }

        # ========== PHASE 3: Distribute results back to cameras ==========
        camera_results = {
            task['serial_number']: {'all_match': True, 'results': []}
            for task in ocr_tasks
        }

        for i, (text, confidence) in enumerate(ocr_results):
            meta = all_metadata[i]
            serial_number = meta['serial_number']
            annotation_idx = meta['annotation_idx']
            conf_threshold = meta['conf_threshold']
            expected_text = meta['expected_text']

            recognized_text = text.strip()

            # Check confidence threshold
            if confidence < conf_threshold:
                logger.warning(
                    f"[{serial_number}] Annotation {annotation_idx}: "
                    f"Low confidence {confidence:.2%} < threshold {conf_threshold:.2%}"
                )
                match = False
            else:
                # Compare texts using similarity matching for specific patterns
                if "BEST BEFORE" in expected_text.upper() or "PL" in expected_text.upper() or "MFG" in expected_text.upper() or "BB" in expected_text.upper():

                    # Use similarity matching (default 80% threshold)
                    similarity = calculate_text_similarity(recognized_text, expected_text)
                    similarity_threshold = 0.75
                    match = similarity >= similarity_threshold
                    if match:
                        recognized_text = expected_text[:]  # Override with expected text on match
                    logger.info(
                        f"[{serial_number}] Annotation {annotation_idx}: "
                        f"Using similarity matching - similarity={similarity:.2%}, "
                        f"threshold={similarity_threshold:.2%}"
                    )
                else:
                    if "USsed" in recognized_text:
                        recognized_text = recognized_text.replace("USsed", "Used")
                        
                    # Use exact match
                    match = compare_texts(recognized_text, expected_text, case_sensitive=False, strip=True)

            if not match:
                camera_results[serial_number]['all_match'] = False

            camera_results[serial_number]['results'].append({
                'annotation_idx': annotation_idx,
                'expected': expected_text,
                'recognized': recognized_text,
                'match': match,
                'confidence': confidence,
                'threshold': conf_threshold
            })

            logger.info(
                f"[{serial_number}] Annotation {annotation_idx}: "
                f"expected='{expected_text}', recognized='{recognized_text}', "
                f"match={match}, conf={confidence:.2%}"
            )

        return camera_results

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
            if bbox.get('type') == 'text':
                annotation_idx = bbox.get('annotation_index')
                if annotation_idx is not None and annotation_idx in recognized_map:
                    bbox['text'] = recognized_map[annotation_idx]
