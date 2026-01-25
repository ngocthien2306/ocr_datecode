"""
Template Verification Service

Handles template matching verification for inference results.
Uses OpenCV template matching to verify similarity between
detected regions and reference templates.
"""

import logging
import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass

from ..ocr_utils import crop_text_region

if TYPE_CHECKING:
    from ..camera import Camera

logger = logging.getLogger(__name__)

home = os.environ.get('HOME')


@dataclass
class TemplateVerificationResult:
    """Result of template verification"""
    match: bool
    similarity: float
    threshold: float
    method: str
    template_bbox: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TemplateVerificationService:
    """
    Service for verifying template regions using OpenCV template matching.

    Responsibilities:
    - Crop template regions from frames
    - Compare cropped regions with reference templates
    - Calculate similarity scores
    - Determine pass/fail based on threshold
    """

    # Template matching method constants
    METHOD_CCOEFF_NORMED = cv2.TM_CCOEFF_NORMED
    METHOD_CCORR_NORMED = cv2.TM_CCORR_NORMED
    METHOD_SQDIFF_NORMED = cv2.TM_SQDIFF_NORMED

    METHOD_NAMES = {
        cv2.TM_CCOEFF_NORMED: 'TM_CCOEFF_NORMED',
        cv2.TM_CCORR_NORMED: 'TM_CCORR_NORMED',
        cv2.TM_SQDIFF_NORMED: 'TM_SQDIFF_NORMED'
    }

    def __init__(
        self,
        save_debug_images: bool = True,
        debug_path: Optional[str] = None,
        default_method: int = cv2.TM_CCOEFF_NORMED,
        default_threshold: float = 0.85
    ):
        """
        Initialize TemplateVerificationService.

        Args:
            save_debug_images: Whether to save cropped regions for debugging
            debug_path: Path to save debug images
            default_method: Default OpenCV template matching method
            default_threshold: Default similarity threshold
        """
        self.save_debug_images = save_debug_images
        self.debug_path = debug_path or f"{home}/Source/ocr_datecode/ai_services/test_result"
        self.default_method = default_method
        self.default_threshold = default_threshold

    def verify_template_regions(
        self,
        frame_img: np.ndarray,
        template_img: np.ndarray,
        transformed_bboxes: List[Dict[str, Any]],
        original_template_bbox: Dict[str, Any],
        camera: 'Camera',
        similarity_threshold: Optional[float] = None,
        method: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Verify template region similarity between frame and reference template.

        Args:
            frame_img: Captured frame (numpy array)
            template_img: Reference template image (numpy array)
            transformed_bboxes: List of transformed bbox dicts from matcher
            original_template_bbox: Original template bbox from reference
            camera: Camera object for logging
            similarity_threshold: Minimum similarity score (0-1)
            method: OpenCV template matching method

        Returns:
            {
                'match': bool,
                'similarity': float,
                'threshold': float,
                'method': str,
                'template_bbox': dict,
                'error': str (optional)
            }
        """
        serial_number = camera.serial_number
        method = method or self.default_method
        similarity_threshold = similarity_threshold or self.default_threshold

        try:
            # Filter only template type bboxes
            template_bboxes = [
                bbox for bbox in transformed_bboxes
                if bbox.get('type') == 'template'
            ]

            # Try to get confidence threshold from bbox
            try:
                if template_bboxes:
                    similarity_threshold = template_bboxes[0].get('conf', similarity_threshold)
            except Exception:
                pass

            logger.info(f"[{serial_number}] Verifying {len(template_bboxes)} template regions")

            if not template_bboxes:
                logger.warning(f"[{serial_number}] No template bbox found in transformed_bboxes")
                return self._build_error_result(
                    similarity_threshold,
                    method,
                    error='No template bbox found'
                )

            # Get the transformed template bbox (for target crop)
            transformed_template_bbox = template_bboxes[0]
            transformed_points = transformed_template_bbox.get('points', [])

            if len(transformed_points) < 4:
                logger.warning(f"[{serial_number}] Invalid transformed template bbox points")
                return self._build_error_result(
                    similarity_threshold,
                    method,
                    template_bbox=transformed_template_bbox,
                    error='Invalid transformed bbox points'
                )

            # Get original template bbox points (for template crop)
            original_points = original_template_bbox.get('points', [])
            if len(original_points) < 4:
                logger.warning(f"[{serial_number}] Invalid original template bbox points")
                return self._build_error_result(
                    similarity_threshold,
                    method,
                    template_bbox=transformed_template_bbox,
                    error='Invalid original bbox points'
                )

            # Crop template region from both images using perspective transform
            cropped_template = crop_text_region(template_img, original_points)
            cropped_target = crop_text_region(frame_img, transformed_points)

            # Save debug images if enabled
            if self.save_debug_images:
                cv2.imwrite(
                    f"{self.debug_path}/template_crop_{serial_number}.png",
                    cropped_template
                )
                cv2.imwrite(
                    f"{self.debug_path}/target_crop_{serial_number}.png",
                    cropped_target
                )

            # Ensure both crops are the same size
            if cropped_template.shape != cropped_target.shape:
                logger.warning(
                    f"[{serial_number}] Template and target crop size mismatch: "
                    f"{cropped_template.shape} vs {cropped_target.shape}"
                )
                cropped_target = cv2.resize(
                    cropped_target,
                    (cropped_template.shape[1], cropped_template.shape[0])
                )

            # Calculate similarity score
            similarity_score = self._calculate_similarity(
                cropped_template,
                cropped_target,
                method
            )

            # Determine if it matches threshold
            match = bool(similarity_score >= similarity_threshold)

            method_name = self.METHOD_NAMES.get(method, 'UNKNOWN')

            logger.info(
                f"[{serial_number}] Template verification: "
                f"similarity={similarity_score:.4f}, threshold={similarity_threshold}, "
                f"match={match}, method={method_name}"
            )

            return {
                'match': match,
                'similarity': similarity_score,
                'threshold': similarity_threshold,
                'method': method_name,
                'template_bbox': transformed_template_bbox
            }

        except Exception as e:
            logger.error(f"[{serial_number}] Error verifying template region: {e}")
            import traceback
            traceback.print_exc()

            return self._build_error_result(
                similarity_threshold,
                method,
                error=str(e)
            )

    def _calculate_similarity(
        self,
        template: np.ndarray,
        target: np.ndarray,
        method: int
    ) -> float:
        """
        Calculate similarity score between template and target images.

        Args:
            template: Template image (cropped reference)
            target: Target image (cropped from frame)
            method: OpenCV template matching method

        Returns:
            Similarity score (0-1)
        """
        # Convert to grayscale for template matching (more robust)
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template

        if len(target.shape) == 3:
            target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        else:
            target_gray = target

        # Use cv2.matchTemplate for pixel-level comparison
        # Since both images are the same size, result will be a single value
        result = cv2.matchTemplate(target_gray, template_gray, method)

        # Get similarity score
        if method == cv2.TM_SQDIFF_NORMED:
            # For SQDIFF, lower is better, so invert: similarity = 1 - score
            similarity_score = float(1.0 - result[0, 0])
        else:
            # For CCOEFF_NORMED and CCORR_NORMED, higher is better
            similarity_score = float(result[0, 0])

        return similarity_score

    def _build_error_result(
        self,
        threshold: float,
        method: int,
        template_bbox: Optional[Dict[str, Any]] = None,
        error: str = 'Unknown error'
    ) -> Dict[str, Any]:
        """
        Build error result dict.

        Args:
            threshold: Similarity threshold used
            method: OpenCV method used
            template_bbox: Template bbox if available
            error: Error message

        Returns:
            Error result dict
        """
        method_name = self.METHOD_NAMES.get(method, 'UNKNOWN')

        return {
            'match': False,
            'similarity': 0.0,
            'threshold': threshold,
            'method': method_name,
            'template_bbox': template_bbox,
            'error': error
        }

    def batch_verify_templates(
        self,
        verification_tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Batch verify multiple template regions.

        Args:
            verification_tasks: List of tasks, each containing:
                {
                    'frame_img': np.ndarray,
                    'template_img': np.ndarray,
                    'transformed_bboxes': list,
                    'original_template_bbox': dict,
                    'camera': Camera,
                    'similarity_threshold': float (optional),
                    'method': int (optional)
                }

        Returns:
            List of verification results in same order as tasks
        """
        results = []

        for task in verification_tasks:
            result = self.verify_template_regions(
                frame_img=task['frame_img'],
                template_img=task['template_img'],
                transformed_bboxes=task['transformed_bboxes'],
                original_template_bbox=task['original_template_bbox'],
                camera=task['camera'],
                similarity_threshold=task.get('similarity_threshold'),
                method=task.get('method')
            )
            results.append(result)

        return results
