"""
Template Verification Service

Handles template matching verification for inference results.
Uses OpenCV template matching to verify similarity between
detected regions and reference templates.
"""

import logging
import os
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    timing: Optional[Dict[str, float]] = None


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

    # Default max dimension for resize optimization
    # Based on analysis: 59x547 crop takes 2.5ms matching
    # Larger crops (763x520) take 16.5ms - resizing helps significantly
    DEFAULT_MAX_DIMENSION = 200

    def __init__(
        self,
        save_debug_images: bool = True,
        debug_path: Optional[str] = None,
        default_method: int = cv2.TM_CCOEFF_NORMED,
        default_threshold: float = 0.85,
        max_dimension: int = DEFAULT_MAX_DIMENSION,
        max_workers: int = 4
    ):
        """
        Initialize TemplateVerificationService.

        Args:
            save_debug_images: Whether to save cropped regions for debugging
            debug_path: Path to save debug images
            default_method: Default OpenCV template matching method
            default_threshold: Default similarity threshold
            max_dimension: Max dimension for resize optimization (0 = disabled)
            max_workers: Max workers for parallel verification
        """
        self.save_debug_images = save_debug_images
        self.debug_path = debug_path or f"{home}/Source/ocr_datecode/ai_services/test_result"
        self.default_method = default_method
        self.default_threshold = default_threshold
        self.max_dimension = max_dimension
        self.max_workers = max_workers
        self._debug_counter = 0

        # Cache for cropped templates (key: (camera_serial, points_tuple))
        self._template_crop_cache = {}

    def _resize_for_matching(
        self,
        img: np.ndarray,
        max_dim: int
    ) -> tuple:
        """
        Resize image if larger than max_dimension for faster matching.

        Args:
            img: Input image
            max_dim: Maximum dimension (width or height)

        Returns:
            Tuple of (resized_img, scale_factor)
        """
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

        # Timing measurements
        t_start = time.perf_counter()
        t_crop = 0.0
        t_resize = 0.0
        t_matching = 0.0

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

            # Validate crop size before expensive perspective transform
            frame_h, frame_w = frame_img.shape[:2]
            pts = np.array(transformed_points, dtype=np.float32)

            # Calculate crop dimensions (same as cv2.boundingRect will compute)
            min_x = pts[:, 0].min()
            max_x = pts[:, 0].max()
            min_y = pts[:, 1].min()
            max_y = pts[:, 1].max()
            crop_width = int(max_x - min_x)
            crop_height = int(max_y - min_y)

            # Validate crop size (prevent huge allocations when template matching fails)
            MAX_CROP_DIM = 1500  # Reasonable limit for template regions
            if (crop_width > MAX_CROP_DIM or crop_height > MAX_CROP_DIM or
                crop_width <= 0 or crop_height <= 0):
                logger.error(
                    f"[{serial_number}] Invalid crop size: {crop_width}x{crop_height}. "
                    f"Bounds: ({min_x:.0f},{min_y:.0f})→({max_x:.0f},{max_y:.0f}), "
                    f"Frame: {frame_w}x{frame_h}. Skipping template verification."
                )
                return self._build_result(
                    serial_number,
                    False,
                    0.0,
                    method,
                    template_bbox=transformed_template_bbox,
                    error=f'Invalid bbox (crop={crop_width}x{crop_height})'
                )

            # Crop template region (cache template crop, only crop target each time)
            # Create cache key from camera serial and original points
            cache_key = (serial_number, tuple(map(tuple, original_points)))

            t_crop_start = time.perf_counter()

            # Check cache for template crop (template doesn't change)
            if cache_key in self._template_crop_cache:
                cropped_template = self._template_crop_cache[cache_key]
            else:
                # First time: crop and cache
                cropped_template = crop_text_region(template_img, original_points)
                self._template_crop_cache[cache_key] = cropped_template
                logger.debug(f"[{serial_number}] Cached template crop (size: {cropped_template.shape})")

            # Always crop target (transformed_points change each frame)
            cropped_target = crop_text_region(frame_img, transformed_points)

            t_crop = (time.perf_counter() - t_crop_start) * 1000  # Convert to ms

            # Save debug images if enabled
            if self.save_debug_images:
                cv2.imwrite(
                    f"{self.debug_path}/template_crop_{serial_number}_{self._debug_counter}.png",
                    cropped_template
                )
                cv2.imwrite(
                    f"{self.debug_path}/target_crop_{serial_number}_{self._debug_counter}.png",
                    cropped_target
                )

            # Ensure both crops are the same size
            if cropped_template.shape != cropped_target.shape:
                logger.warning(
                    f"[{serial_number}] Template and target crop size mismatch: "
                    f"{cropped_template.shape} vs {cropped_target.shape}"
                )
                t_resize_start = time.perf_counter()
                cropped_target = cv2.resize(
                    cropped_target,
                    (cropped_template.shape[1], cropped_template.shape[0])
                )
                t_resize = (time.perf_counter() - t_resize_start) * 1000  # Convert to ms

            # Resize for faster matching if images are large
            original_size = cropped_template.shape[:2]
            t_opt_resize_start = time.perf_counter()
            cropped_template, scale = self._resize_for_matching(cropped_template, self.max_dimension)
            cropped_target, _ = self._resize_for_matching(cropped_target, self.max_dimension)
            t_opt_resize = (time.perf_counter() - t_opt_resize_start) * 1000

            if scale < 1.0:
                logger.info(
                    f"[{serial_number}] Resized for matching: {original_size} -> "
                    f"{cropped_template.shape[:2]} (scale={scale:.2f})"
                )

            # Calculate similarity score
            t_matching_start = time.perf_counter()
            similarity_score = self._calculate_similarity(
                cropped_template,
                cropped_target,
                method
            )
            t_matching = (time.perf_counter() - t_matching_start) * 1000  # Convert to ms

            # Determine if it matches threshold
            match = bool(similarity_score >= similarity_threshold)

            method_name = self.METHOD_NAMES.get(method, 'UNKNOWN')

            # Calculate total time
            t_total = (time.perf_counter() - t_start) * 1000  # Convert to ms

            # Build timing dict
            timing = {
                'crop_ms': t_crop,
                'resize_ms': t_resize,
                'opt_resize_ms': t_opt_resize,
                'matching_ms': t_matching,
                'total_ms': t_total
            }

            logger.info(
                f"[{serial_number}] Template verification: "
                f"similarity={similarity_score:.4f}, threshold={similarity_threshold}, "
                f"match={match}, method={method_name}, "
                f"total={t_total:.1f}ms (crop={t_crop:.1f}ms, resize={t_resize:.1f}ms, "
                f"opt_resize={t_opt_resize:.1f}ms, matching={t_matching:.1f}ms)"
            )

            return {
                'match': match,
                'similarity': similarity_score,
                'threshold': similarity_threshold,
                'method': method_name,
                'template_bbox': transformed_template_bbox,
                'timing': timing
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
        verification_tasks: List[Dict[str, Any]],
        parallel: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Batch verify multiple template regions (optionally in parallel).

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
            parallel: Whether to run verification in parallel (default: True)

        Returns:
            List of verification results in same order as tasks
        """
        t_batch_start = time.perf_counter()
        n_cameras = len(verification_tasks)

        if n_cameras == 0:
            return []

        # Use parallel execution for multiple cameras
        if parallel and n_cameras > 1:
            results = self._batch_verify_parallel(verification_tasks)
        else:
            results = self._batch_verify_sequential(verification_tasks)

        t_batch_total = (time.perf_counter() - t_batch_start) * 1000  # Convert to ms

        # Calculate timing statistics
        avg_time = t_batch_total / n_cameras if n_cameras > 0 else 0.0

        # Sum up individual timings
        total_crop = sum(r.get('timing', {}).get('crop_ms', 0.0) for r in results)
        total_resize = sum(r.get('timing', {}).get('resize_ms', 0.0) for r in results)
        total_opt_resize = sum(r.get('timing', {}).get('opt_resize_ms', 0.0) for r in results)
        total_matching = sum(r.get('timing', {}).get('matching_ms', 0.0) for r in results)

        mode = "parallel" if parallel and n_cameras > 1 else "sequential"
        logger.info(
            f"Template verification batch ({mode}): {n_cameras} camera(s), "
            f"total={t_batch_total:.1f}ms (avg={avg_time:.1f}ms/camera), "
            f"crop={total_crop:.1f}ms, resize={total_resize:.1f}ms, "
            f"opt_resize={total_opt_resize:.1f}ms, matching={total_matching:.1f}ms"
        )

        return results

    def _batch_verify_sequential(
        self,
        verification_tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run verification sequentially"""
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

    def _batch_verify_parallel(
        self,
        verification_tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run verification in parallel using ThreadPoolExecutor"""
        results = [None] * len(verification_tasks)

        def verify_task(idx_task):
            idx, task = idx_task
            return idx, self.verify_template_regions(
                frame_img=task['frame_img'],
                template_img=task['template_img'],
                transformed_bboxes=task['transformed_bboxes'],
                original_template_bbox=task['original_template_bbox'],
                camera=task['camera'],
                similarity_threshold=task.get('similarity_threshold'),
                method=task.get('method')
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(verify_task, (idx, task))
                for idx, task in enumerate(verification_tasks)
            ]

            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    logger.error(f"Parallel verification error: {e}")

        return results
