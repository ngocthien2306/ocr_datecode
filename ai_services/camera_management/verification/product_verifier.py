"""
Product Verification Service

Handles product alignment verification for inference results.
Detects bottle edges and label edges to determine if label is tilted.
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
class ProductVerificationResult:
    """Result of product alignment verification"""
    match: bool
    angle_diff: float
    threshold: float
    bottle_angle: Optional[float] = None
    label_angle: Optional[float] = None
    skipped: bool = False
    error: Optional[str] = None


class ProductVerificationService:
    """
    Service for verifying product alignment using edge detection.

    Detects bottle edges and label edges to determine if label is tilted
    relative to the bottle.

    Responsibilities:
    - Crop product regions from frames
    - Detect edges using Scharr gradient
    - Classify bottle edges vs label edges
    - Calculate angle difference
    - Determine alignment based on threshold
    """

    def __init__(
        self,
        save_debug_images: bool = True,
        debug_path: Optional[str] = None,
        default_angle_threshold: float = 1.0
    ):
        """
        Initialize ProductVerificationService.

        Args:
            save_debug_images: Whether to save debug images with visualized edges
            debug_path: Path to save debug images
            default_angle_threshold: Default angle threshold in degrees (default: 1.0)
        """
        self.save_debug_images = save_debug_images
        self.debug_path = debug_path or f"{home}/Source/ocr_datecode/ai_services/test_result"
        self.default_angle_threshold = default_angle_threshold

    def verify_product_alignment(
        self,
        frame_img: np.ndarray,
        transformed_bboxes: List[Dict[str, Any]],
        camera: 'Camera'
    ) -> Dict[str, Any]:
        """
        Verify product alignment (check if label is tilted relative to bottle).

        Args:
            frame_img: Captured frame (numpy array)
            transformed_bboxes: List of transformed bbox dicts from matcher
            camera: Camera object for logging and config

        Returns:
            {
                'match': bool,
                'angle_diff': float,
                'threshold': float,
                'bottle_angle': float,
                'label_angle': float,
                'skipped': bool (if no product bbox),
                'error': str (optional)
            }
        """
        serial_number = camera.serial_number

        # Get angle threshold from camera settings or use default
        angle_threshold = getattr(camera, 'angle_threshold', self.default_angle_threshold)

        try:
            # Filter only product type bboxes
            product_bboxes = [
                bbox for bbox in transformed_bboxes
                if bbox.get('type') == 'product'
            ]

            if not product_bboxes:
                logger.debug(f"[{serial_number}] No product bbox found, skipping product verification")
                return {
                    'match': True,
                    'angle_diff': 0.0,
                    'threshold': angle_threshold,
                    'skipped': True
                }

            # Use first product bbox
            product_bbox = product_bboxes[0]
            points = product_bbox.get('points', [])

            if len(points) < 4:
                logger.warning(f"[{serial_number}] Invalid product bbox points")
                return self._build_error_result(
                    angle_threshold,
                    error='Invalid product bbox points'
                )

            logger.info(f"[{serial_number}] Verifying product alignment")

            # Crop product region
            cropped_product = crop_text_region(frame_img, points)

            # Detect edges and classify
            edge_result = self._detect_and_classify_edges(cropped_product, serial_number)

            if edge_result is None:
                return self._build_error_result(
                    angle_threshold,
                    error='Failed to detect edges'
                )

            bottle_edges, label_edges, edges_img = edge_result

            # Calculate angles
            bottle_left_angle = self._get_line_angle(bottle_edges[0])
            bottle_right_angle = self._get_line_angle(bottle_edges[1])
            bottle_avg_angle = (bottle_left_angle + bottle_right_angle) / 2

            label_left_angle = self._get_line_angle(label_edges[0])
            label_right_angle = self._get_line_angle(label_edges[1])
            label_avg_angle = (label_left_angle + label_right_angle) / 2

            # Calculate tilt difference
            angle_diff = float(label_avg_angle - bottle_avg_angle)
            is_aligned = abs(angle_diff) < angle_threshold

            logger.info(
                f"[{serial_number}] Product verification: "
                f"bottle_angle={bottle_avg_angle:.2f}°, label_angle={label_avg_angle:.2f}°, "
                f"angle_diff={angle_diff:+.2f}°, threshold={angle_threshold}°, match={is_aligned}"
            )

            # Visualize edges if debug enabled
            if self.save_debug_images:
                self._visualize_edges(
                    cropped_product,
                    bottle_edges,
                    label_edges,
                    angle_diff,
                    is_aligned,
                    serial_number
                )

            return {
                'match': is_aligned,
                'angle_diff': angle_diff,
                'threshold': angle_threshold,
                'bottle_angle': bottle_avg_angle,
                'label_angle': label_avg_angle,
                'skipped': False
            }

        except Exception as e:
            logger.error(f"[{serial_number}] Error verifying product alignment: {e}")
            import traceback
            traceback.print_exc()

            return self._build_error_result(
                angle_threshold,
                error=str(e)
            )

    def _detect_and_classify_edges(
        self,
        img: np.ndarray,
        serial_number: str
    ) -> Optional[tuple]:
        """
        Detect edges and classify into bottle edges and label edges.

        Args:
            img: Cropped product image
            serial_number: Camera serial number for logging

        Returns:
            (bottle_edges, label_edges, edges_img) or None if failed
            - bottle_edges: [left_contour, right_contour]
            - label_edges: [left_contour, right_contour]
            - edges_img: Edge detection result image
        """
        try:
            # Preprocessing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # Edge detection using Scharr gradient (vertical edges)
            grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
            grad_x = cv2.normalize(np.abs(grad_x), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, edges = cv2.threshold(grad_x, 40, 255, cv2.THRESH_BINARY)

            # Morphology to connect edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) < 4:
                logger.warning(
                    f"[{serial_number}] Not enough contours detected: {len(contours)} < 4 (FAIL)"
                )
                return None

            # Sort by height and take top 4
            contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[3], reverse=True)[:4]

            # Sort by x-coordinate (left to right)
            edges_x = sorted([(cv2.boundingRect(c)[0], c) for c in contours])

            # Classify: bottle edges (leftmost and rightmost), label edges (middle 2)
            bottle_edges = [edges_x[0][1], edges_x[3][1]]
            label_edges = [edges_x[1][1], edges_x[2][1]]

            return bottle_edges, label_edges, edges

        except Exception as e:
            logger.error(f"[{serial_number}] Error detecting edges: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_line_angle(self, contour: np.ndarray) -> float:
        """
        Calculate angle of a line fitted to contour.

        Args:
            contour: Contour points

        Returns:
            Angle in degrees (0-90)
        """
        [vx, vy, x0, y0] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        angle = np.arctan2(float(vy), float(vx)) * 180 / np.pi
        angle = 90 - abs(angle)
        return angle

    def _visualize_edges(
        self,
        img: np.ndarray,
        bottle_edges: List[np.ndarray],
        label_edges: List[np.ndarray],
        angle_diff: float,
        is_aligned: bool,
        serial_number: str
    ):
        """
        Visualize detected edges on image and save for debugging.

        Args:
            img: Original product image
            bottle_edges: Bottle edge contours
            label_edges: Label edge contours
            angle_diff: Angle difference
            is_aligned: Whether product is aligned
            serial_number: Camera serial number
        """
        try:
            result = img.copy()

            # Draw bottle edges (green)
            for c in bottle_edges:
                [vx, vy, x0, y0] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
                vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)
                lefty = int((-x0 * vy / vx) + y0)
                righty = int(((img.shape[1] - x0) * vy / vx) + y0)
                cv2.line(result, (img.shape[1]-1, righty), (0, lefty), (0, 255, 0), 2)

            # Draw label edges (blue)
            for c in label_edges:
                [vx, vy, x0, y0] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
                vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)
                lefty = int((-x0 * vy / vx) + y0)
                righty = int(((img.shape[1] - x0) * vy / vx) + y0)
                cv2.line(result, (img.shape[1]-1, righty), (0, lefty), (255, 0, 0), 2)

            # Add text status
            color = (0, 255, 0) if is_aligned else (0, 0, 255)
            status = "ALIGNED" if is_aligned else "TILTED"
            cv2.putText(
                result,
                f"{status}: {angle_diff:+.2f} deg",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

            # Save debug image
            debug_file = f"{self.debug_path}/product_alignment_{serial_number}.png"
            cv2.imwrite(debug_file, result)
            logger.debug(f"[{serial_number}] Saved product alignment debug image: {debug_file}")

        except Exception as e:
            logger.error(f"[{serial_number}] Error visualizing edges: {e}")

    def _build_error_result(
        self,
        threshold: float,
        error: str = 'Unknown error'
    ) -> Dict[str, Any]:
        """
        Build error result dict.

        Args:
            threshold: Angle threshold used
            error: Error message

        Returns:
            Error result dict
        """
        return {
            'match': False,
            'angle_diff': 0.0,
            'threshold': threshold,
            'bottle_angle': None,
            'label_angle': None,
            'skipped': False,
            'error': error
        }
