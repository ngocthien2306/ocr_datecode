"""
Product Verification Service

Handles product alignment verification using YOLO OBB detection.
Detects product and label boxes to check for rotation and misalignment.
"""

import logging
import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, TYPE_CHECKING, Tuple
from dataclasses import dataclass
from pathlib import Path
import sys

# Import YOLO OBB TensorRT
try:
    ai_services_path = Path(__file__).parent.parent.parent
    if str(ai_services_path) not in sys.path:
        sys.path.insert(0, str(ai_services_path))
    from yolo_obb_tensorrt import YOLOOBBTensorRT
    YOLO_OBB_AVAILABLE = True
except ImportError as e:
    logger.warning(f"YOLO OBB not available: {e}")
    YOLO_OBB_AVAILABLE = False
    YOLOOBBTensorRT = None

if TYPE_CHECKING:
    from ..camera import Camera

logger = logging.getLogger(__name__)

home = os.environ.get('HOME')


@dataclass
class ProductVerificationResult:
    """Result of product alignment verification"""
    match: bool
    skipped: bool = False
    error: Optional[str] = None
    reason: Optional[str] = None

    # Rotation check results
    rotation_check: Optional[Dict[str, Any]] = None

    # Misalignment check results
    misalignment_check: Optional[Dict[str, Any]] = None

    # Detected boxes info
    detected_boxes: Optional[List[Dict[str, Any]]] = None


class ProductVerificationService:
    """
    Service for verifying product alignment using YOLO OBB detection.

    Detects product and label boxes to check:
    1. Rotation: If label is rotated relative to product (> angle threshold)
    2. Misalignment: If label touches the label region boundaries (left/right sides)

    Responsibilities:
    - Load YOLO OBB model for product/label detection
    - Filter detected boxes inside product region
    - Check rotation angle between product and label
    - Check if label is misaligned (touching boundaries)
    - Support batch inference for multiple frames
    """

    def __init__(
        self,
        engine_path: Optional[str] = None,
        save_debug_images: str = "never",
        debug_path: Optional[str] = None,
        angle_threshold: float = 3.0,
        margin_pixels: int = 30,
        conf_threshold: float = 0.25
    ):
        """
        Initialize ProductVerificationService with YOLO OBB model.

        Args:
            engine_path: Path to YOLO OBB TensorRT engine
            save_debug_images: When to save debug images:
                - "never": Never save (fastest, default)
                - "on_fail": Only save when verification fails
                - "always": Always save (slowest, for debugging)
            debug_path: Path to save debug images
            angle_threshold: Rotation angle threshold in degrees (default: 3.0)
            margin_pixels: Margin for misalignment check in pixels (default: 30)
            conf_threshold: YOLO confidence threshold (default: 0.25)
        """
        self.save_debug_images = save_debug_images.lower()
        self.debug_path = debug_path or f"{home}/Source/ocr_datecode/ai_services/test_result"
        self.angle_threshold = angle_threshold
        self.margin_pixels = margin_pixels
        self.conf_threshold = conf_threshold

        # Validate save_debug_images option
        if self.save_debug_images not in ["never", "on_fail", "always"]:
            logger.warning(
                f"Invalid save_debug_images option '{save_debug_images}', "
                f"using 'never'. Valid options: never, on_fail, always"
            )
            self.save_debug_images = "never"

        # Initialize YOLO OBB model
        if engine_path is None:
            engine_path = f"{home}/Source/ocr_datecode/weights/yolo26n-ultralight-obb_fp16_dynamic.engine"

        self.obb_model = None
        if YOLO_OBB_AVAILABLE:
            try:
                self.obb_model = YOLOOBBTensorRT(
                    engine_path=engine_path,
                    class_names=['label', 'product']
                )
                logger.info(
                    f"YOLO OBB model loaded: {engine_path}, "
                    f"debug_images={self.save_debug_images}"
                )
            except Exception as e:
                logger.error(f"Failed to load YOLO OBB model: {e}")
                self.obb_model = None
        else:
            logger.warning("YOLO OBB not available, product verification will be skipped")

    def should_verify_frame(self, transformed_bboxes: List[Dict[str, Any]]) -> bool:
        """
        Check if a frame should be verified for product alignment.

        A frame needs verification if it has both product and label regions.

        Args:
            transformed_bboxes: List of transformed bbox dicts from matcher

        Returns:
            True if frame should be verified, False otherwise
        """
        has_product = any(bbox.get('type') == 'product' for bbox in transformed_bboxes)
        has_label = any(bbox.get('type') == 'label' for bbox in transformed_bboxes)
        return has_product and has_label

    def verify_product_alignment(
        self,
        frame_img: np.ndarray,
        transformed_bboxes: List[Dict[str, Any]],
        camera: 'Camera'
    ) -> Dict[str, Any]:
        """
        Verify product alignment (check rotation and misalignment).

        This is a wrapper for backward compatibility. Internally calls verify_batch.

        Args:
            frame_img: Captured frame (numpy array)
            transformed_bboxes: List of transformed bbox dicts from matcher
            camera: Camera object for logging and config

        Returns:
            {
                'match': bool,
                'skipped': bool,
                'error': str (optional),
                'reason': str (optional),
                'rotation_check': {...},
                'misalignment_check': {...},
                'detected_boxes': [...]
            }
        """
        serial_number = camera.serial_number

        # Check if model is available
        if self.obb_model is None:
            logger.warning(f"[{serial_number}] YOLO OBB model not available, skipping")
            return {
                'match': True,
                'skipped': True,
                'reason': 'YOLO OBB model not available'
            }

        # Check if frame should be verified
        if not self.should_verify_frame(transformed_bboxes):
            logger.debug(f"[{serial_number}] Frame doesn't have both product and label regions, skipping")
            return {
                'match': True,
                'skipped': True,
                'reason': 'No product or label region in template'
            }

        # Use batch method with single frame
        frames_data = [{
            'frame_img': frame_img,
            'transformed_bboxes': transformed_bboxes,
            'camera': camera
        }]

        results = self.verify_batch(frames_data)
        return results[0]

    def verify_batch(
        self,
        frames_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Batch verify multiple frames for product alignment.

        Args:
            frames_data: List of dicts containing:
                - 'frame_img': np.ndarray
                - 'transformed_bboxes': List[Dict]
                - 'camera': Camera object

        Returns:
            List of verification results (one per frame)
        """
        import time

        t_start = time.perf_counter()

        if self.obb_model is None:
            return [{
                'match': True,
                'skipped': True,
                'reason': 'YOLO OBB model not available',
                'timing': {'total': 0.0}
            } for _ in frames_data]

        # Filter frames that need verification
        t_filter_start = time.perf_counter()
        frames_to_check = []
        frame_indices = []

        for idx, data in enumerate(frames_data):
            if self.should_verify_frame(data['transformed_bboxes']):
                frames_to_check.append(data)
                frame_indices.append(idx)

        t_filter = (time.perf_counter() - t_filter_start) * 1000

        # If no frames need verification, return all skipped
        if not frames_to_check:
            logger.debug(f"Product verification: 0/{len(frames_data)} frames need verification (filter: {t_filter:.1f}ms)")
            return [{
                'match': True,
                'skipped': True,
                'reason': 'No product or label region in template',
                'timing': {'total': t_filter, 'filter': t_filter}
            } for _ in frames_data]

        # Batch predict YOLO OBB
        images = [data['frame_img'] for data in frames_to_check]

        try:
            t_yolo_start = time.perf_counter()
            batch_results = self.obb_model.predict(images, conf_threshold=self.conf_threshold, return_timing=True)

            # Extract results and timing
            if isinstance(batch_results, tuple):
                batch_results, yolo_timing = batch_results
            else:
                yolo_timing = {}

            t_yolo = (time.perf_counter() - t_yolo_start) * 1000
        except Exception as e:
            logger.error(f"YOLO OBB batch prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return [{
                'match': False,
                'skipped': False,
                'error': f'YOLO prediction failed: {str(e)}',
                'timing': {'total': (time.perf_counter() - t_start) * 1000}
            } for _ in frames_data]

        # Process each frame result
        t_process_start = time.perf_counter()
        results = [{
            'match': True,
            'skipped': True,
            'reason': 'No product or label region in template',
            'timing': {'total': 0.0}
        } for _ in frames_data]

        debug_images_saved = 0
        for i, orig_idx in enumerate(frame_indices):
            data = frames_to_check[i]
            boxes, scores, class_ids = batch_results[i]

            result = self._process_single_frame(
                boxes=boxes,
                scores=scores,
                class_ids=class_ids,
                transformed_bboxes=data['transformed_bboxes'],
                frame_img=data['frame_img'],
                camera=data['camera']
            )

            # Track if debug image was saved
            if self._should_save_debug_image(result.get('match', True)):
                debug_images_saved += 1

            results[orig_idx] = result

        t_process = (time.perf_counter() - t_process_start) * 1000
        t_total = (time.perf_counter() - t_start) * 1000

        # Add timing to all results
        timing_info = {
            'total': t_total,
            'filter': t_filter,
            'yolo_inference': t_yolo,
            'yolo_details': yolo_timing,
            'processing': t_process,
            'frames_checked': len(frames_to_check),
            'frames_total': len(frames_data)
        }

        for idx in frame_indices:
            if results[idx]:
                results[idx]['timing'] = timing_info.copy()

        logger.info(
            f"Product verification batch: {len(frames_to_check)}/{len(frames_data)} frames, "
            f"total={t_total:.1f}ms (filter={t_filter:.1f}ms, yolo={t_yolo:.1f}ms, process={t_process:.1f}ms), "
            f"debug_images={debug_images_saved}/{len(frames_to_check)} saved"
        )

        return results

    def _process_single_frame(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        transformed_bboxes: List[Dict[str, Any]],
        frame_img: np.ndarray,
        camera: 'Camera'
    ) -> Dict[str, Any]:
        """
        Process single frame verification.

        Args:
            boxes: Detected OBB boxes [cx, cy, w, h, angle]
            scores: Confidence scores
            class_ids: Class IDs (0=label, 1=product)
            transformed_bboxes: Template bounding boxes
            frame_img: Frame image for debugging
            camera: Camera object

        Returns:
            Verification result dict
        """
        serial_number = camera.serial_number

        # Filter boxes inside product region
        filtered_boxes = self._filter_boxes_inside_product_region(
            boxes, scores, class_ids, transformed_bboxes
        )

        if not filtered_boxes:
            logger.debug(f"[{serial_number}] No boxes detected inside product region")
            return {
                'match': True,
                'skipped': True,
                'reason': 'No boxes detected inside product region'
            }

        # Validate and select boxes
        validation = self._validate_filtered_boxes(filtered_boxes, serial_number)

        if not validation['valid']:
            logger.debug(f"[{serial_number}] Box validation failed: {validation['reason']}")
            return {
                'match': True,
                'skipped': True,
                'reason': validation['reason']
            }

        product_box = validation['product_box']
        label_box = validation['label_box']

        # Check rotation
        rotation_check = self._check_rotation(product_box, label_box, serial_number)

        # Check misalignment
        misalignment_check = self._check_misalignment(
            label_box, transformed_bboxes, serial_number
        )

        # Determine overall match
        overall_match = rotation_check['ok'] and misalignment_check['ok']

        logger.info(
            f"[{serial_number}] Product verification: "
            f"rotation={'OK' if rotation_check['ok'] else 'FAIL'}, "
            f"misalignment={'OK' if misalignment_check['ok'] else 'FAIL'}, "
            f"overall={'PASS' if overall_match else 'FAIL'}"
        )

        # Save debug image based on configuration
        if self._should_save_debug_image(overall_match):
            self._visualize_result(
                frame_img, product_box, label_box, rotation_check,
                misalignment_check, overall_match, serial_number
            )

        # Convert numpy arrays to Python native types for JSON serialization
        detected_boxes = {
            'product': {
                'box': product_box['box'].tolist() if isinstance(product_box['box'], np.ndarray) else product_box['box'],
                'score': float(product_box['score']),
                'class': str(product_box['class']),
                'corners': product_box['corners'].tolist() if isinstance(product_box['corners'], np.ndarray) else product_box['corners']
            },
            'label': {
                'box': label_box['box'].tolist() if isinstance(label_box['box'], np.ndarray) else label_box['box'],
                'score': float(label_box['score']),
                'class': str(label_box['class']),
                'corners': label_box['corners'].tolist() if isinstance(label_box['corners'], np.ndarray) else label_box['corners']
            }
        }

        return {
            'match': bool(overall_match),
            'skipped': False,
            'rotation_check': rotation_check,
            'misalignment_check': misalignment_check,
            'detected_boxes': detected_boxes
        }

    def _filter_boxes_inside_product_region(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        transformed_bboxes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter OBB boxes that are completely inside the product region.

        Args:
            boxes: Detected boxes [cx, cy, w, h, angle]
            scores: Confidence scores
            class_ids: Class IDs
            transformed_bboxes: Template bounding boxes

        Returns:
            List of filtered boxes with metadata
        """
        # Find product region
        product_region = next(
            (bbox for bbox in transformed_bboxes if bbox.get('type') == 'product'),
            None
        )

        if product_region is None:
            return []

        product_poly = np.array(product_region['points'], dtype=np.float32)

        filtered = []
        for box, score, cls_id in zip(boxes, scores, class_ids):
            # Convert OBB to corners
            corners = self._obb_to_corners(box)

            # Check if all corners are inside product region
            if self._is_box_inside_polygon(corners, product_poly):
                filtered.append({
                    'box': box,
                    'score': float(score),
                    'class': self.obb_model.class_names[cls_id],
                    'corners': corners
                })

        return filtered

    def _validate_filtered_boxes(
        self,
        filtered_boxes: List[Dict[str, Any]],
        serial_number: str
    ) -> Dict[str, Any]:
        """
        Validate filtered boxes and select one product + one label.

        Handles cases:
        - No boxes
        - Only one type
        - Multiple boxes of same type -> select highest confidence

        Args:
            filtered_boxes: List of filtered boxes
            serial_number: Camera serial for logging

        Returns:
            Dict with 'valid', 'product_box', 'label_box', 'reason'
        """
        products = [b for b in filtered_boxes if b['class'] == 'product']
        labels = [b for b in filtered_boxes if b['class'] == 'label']

        # Check if we have at least one of each
        if not products:
            return {'valid': False, 'reason': 'No product box detected'}
        if not labels:
            return {'valid': False, 'reason': 'No label box detected'}

        # Select highest confidence if multiple
        if len(products) > 1:
            logger.debug(f"[{serial_number}] Multiple product boxes, selecting highest confidence")
        if len(labels) > 1:
            logger.debug(f"[{serial_number}] Multiple label boxes, selecting highest confidence")

        product_box = max(products, key=lambda x: x['score'])
        label_box = max(labels, key=lambda x: x['score'])

        return {
            'valid': True,
            'product_box': product_box,
            'label_box': label_box
        }

    def _check_rotation(
        self,
        product_box: Dict[str, Any],
        label_box: Dict[str, Any],
        serial_number: str
    ) -> Dict[str, Any]:
        """
        Check rotation angle between product and label boxes.

        Args:
            product_box: Product box dict with 'box' field
            label_box: Label box dict with 'box' field
            serial_number: Camera serial for logging

        Returns:
            Dict with 'ok', 'angle_diff', 'threshold'
        """
        # Extract angles from OBB boxes (last element)
        product_angle = product_box['box'][4]  # radians
        label_angle = label_box['box'][4]      # radians

        # Calculate angle difference
        angle_diff = abs(product_angle - label_angle)

        # Normalize to 0-π
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff

        # Convert to degrees
        angle_diff_deg = np.degrees(angle_diff)

        ok = angle_diff_deg <= self.angle_threshold

        logger.debug(
            f"[{serial_number}] Rotation check: "
            f"product={np.degrees(product_angle):.2f}°, "
            f"label={np.degrees(label_angle):.2f}°, "
            f"diff={angle_diff_deg:.2f}°, "
            f"threshold={self.angle_threshold}°, "
            f"result={'OK' if ok else 'FAIL'}"
        )

        return {
            'ok': bool(ok),
            'angle_diff': float(angle_diff_deg),
            'threshold': float(self.angle_threshold),
            'product_angle': float(np.degrees(product_angle)),
            'label_angle': float(np.degrees(label_angle))
        }

    def _check_misalignment(
        self,
        label_box: Dict[str, Any],
        transformed_bboxes: List[Dict[str, Any]],
        serial_number: str
    ) -> Dict[str, Any]:
        """
        Check if label box touches the label region boundaries (left/right).

        Args:
            label_box: Label box dict with 'corners' field
            transformed_bboxes: Template bounding boxes
            serial_number: Camera serial for logging

        Returns:
            Dict with 'ok', 'touches', 'margin', 'details'
        """
        # Find label region
        label_region = next(
            (bbox for bbox in transformed_bboxes if bbox.get('type') == 'label'),
            None
        )

        if label_region is None:
            logger.warning(f"[{serial_number}] No label region found in template")
            return {
                'ok': True,
                'reason': 'No label region in template'
            }

        label_region_poly = np.array(label_region['points'], dtype=np.float32)
        label_corners = label_box['corners']

        # Check left and right edges
        touches_left, left_dist = self._check_edge_touch(
            label_corners, label_region_poly, 'left', self.margin_pixels
        )
        touches_right, right_dist = self._check_edge_touch(
            label_corners, label_region_poly, 'right', self.margin_pixels
        )

        touches = touches_left or touches_right
        ok = not touches

        logger.debug(
            f"[{serial_number}] Misalignment check: "
            f"left={'TOUCH' if touches_left else 'OK'} ({left_dist:.1f}px), "
            f"right={'TOUCH' if touches_right else 'OK'} ({right_dist:.1f}px), "
            f"margin={self.margin_pixels}px, "
            f"result={'OK' if ok else 'FAIL'}"
        )

        return {
            'ok': bool(ok),
            'touches': bool(touches),
            'touches_left': bool(touches_left),
            'touches_right': bool(touches_right),
            'margin': int(self.margin_pixels),
            'left_distance': float(left_dist),
            'right_distance': float(right_dist)
        }

    # ========== Helper Methods ==========

    def _should_save_debug_image(self, overall_match: bool) -> bool:
        """
        Determine if debug image should be saved based on configuration.

        Args:
            overall_match: Whether verification passed

        Returns:
            True if should save debug image
        """
        if self.save_debug_images == "never":
            return False
        elif self.save_debug_images == "on_fail":
            return not overall_match  # Only save when failed
        elif self.save_debug_images == "always":
            return True
        return False

    def _obb_to_corners(self, box: np.ndarray) -> np.ndarray:
        """
        Convert OBB box to 4 corners.

        Args:
            box: [cx, cy, w, h, angle]

        Returns:
            Array of 4 corners [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        cx, cy, w, h, angle = box
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        hw, hh = w / 2, h / 2

        corners = np.array([
            [cx + (-hw * cos_a + hh * sin_a), cy + (-hw * sin_a - hh * cos_a)],
            [cx + (hw * cos_a + hh * sin_a), cy + (hw * sin_a - hh * cos_a)],
            [cx + (hw * cos_a - hh * sin_a), cy + (hw * sin_a + hh * cos_a)],
            [cx + (-hw * cos_a - hh * sin_a), cy + (-hw * sin_a + hh * cos_a)]
        ], dtype=np.float32)

        return corners

    def _is_box_inside_polygon(
        self,
        box_corners: np.ndarray,
        polygon: np.ndarray
    ) -> bool:
        """
        Check if all corners of a box are inside a polygon.

        Args:
            box_corners: Box corners [[x1,y1], ...]
            polygon: Polygon points [[x1,y1], ...]

        Returns:
            True if all corners inside polygon
        """
        for corner in box_corners:
            if cv2.pointPolygonTest(polygon, tuple(corner), False) < 0:
                return False
        return True

    def _check_edge_touch(
        self,
        label_corners: np.ndarray,
        region_poly: np.ndarray,
        side: str,
        margin: int
    ) -> Tuple[bool, float]:
        """
        Check if label box edge touches region boundary.

        Args:
            label_corners: Label box corners
            region_poly: Region polygon
            side: 'left' or 'right'
            margin: Margin in pixels

        Returns:
            (touches: bool, distance: float)
        """
        # Determine left and right edges of label box
        x_coords = label_corners[:, 0]
        left_x = np.min(x_coords)
        right_x = np.max(x_coords)

        # Determine left and right boundaries of region
        region_x = region_poly[:, 0]
        region_left_x = np.min(region_x)
        region_right_x = np.max(region_x)

        if side == 'left':
            # Distance from label left edge to region left boundary
            dist = left_x - region_left_x
            touches = dist < -margin  # Label extends beyond left boundary
        else:  # right
            # Distance from label right edge to region right boundary
            dist = region_right_x - right_x
            touches = dist < -margin  # Label extends beyond right boundary

        return touches, abs(dist)

    def _visualize_result(
        self,
        frame_img: np.ndarray,
        product_box: Dict[str, Any],
        label_box: Dict[str, Any],
        rotation_check: Dict[str, Any],
        misalignment_check: Dict[str, Any],
        overall_match: bool,
        serial_number: str
    ):
        """
        Visualize verification result and save debug image.

        Args:
            frame_img: Frame image
            product_box: Product box dict
            label_box: Label box dict
            rotation_check: Rotation check result
            misalignment_check: Misalignment check result
            overall_match: Overall pass/fail
            serial_number: Camera serial
        """
        try:
            result_img = frame_img.copy()

            # Draw product box (green)
            product_corners = product_box['corners'].astype(np.int32)
            cv2.polylines(result_img, [product_corners], True, (0, 255, 0), 2)
            cv2.putText(
                result_img, f"Product: {product_box['score']:.2f}",
                tuple(product_corners[0]), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2
            )

            # Draw label box (blue if OK, red if fail)
            label_color = (255, 0, 0) if not overall_match else (0, 0, 255)
            label_corners = label_box['corners'].astype(np.int32)
            cv2.polylines(result_img, [label_corners], True, label_color, 2)
            cv2.putText(
                result_img, f"Label: {label_box['score']:.2f}",
                tuple(label_corners[0]), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, label_color, 2
            )

            # Add status text
            status_color = (0, 255, 0) if overall_match else (0, 0, 255)
            status = "PASS" if overall_match else "FAIL"

            y_offset = 30
            cv2.putText(
                result_img, f"Result: {status}",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                1, status_color, 2
            )

            y_offset += 35
            rotation_status = "OK" if rotation_check['ok'] else "FAIL"
            cv2.putText(
                result_img,
                f"Rotation: {rotation_status} ({rotation_check['angle_diff']:.2f} deg)",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, status_color, 2
            )

            y_offset += 30
            misalign_status = "OK" if misalignment_check['ok'] else "FAIL"
            cv2.putText(
                result_img,
                f"Alignment: {misalign_status}",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, status_color, 2
            )

            # Save debug image asynchronously (don't block pipeline)
            self._save_debug_image_async(result_img, serial_number)

        except Exception as e:
            logger.error(f"[{serial_number}] Error visualizing result: {e}")

    def _save_debug_image_async(self, result_img: np.ndarray, serial_number: str):
        """
        Save debug image asynchronously in background thread.

        Args:
            result_img: Result image to save
            serial_number: Camera serial number
        """
        import threading

        def _save():
            try:
                os.makedirs(self.debug_path, exist_ok=True)
                debug_file = f"{self.debug_path}/product_verification_{serial_number}.png"
                cv2.imwrite(debug_file, result_img)
                logger.debug(f"[{serial_number}] Saved debug image: {debug_file}")
            except Exception as e:
                logger.error(f"[{serial_number}] Failed to save debug image: {e}")

        # Start background thread to save image
        thread = threading.Thread(target=_save, daemon=True, name=f"DebugImageSaver-{serial_number}")
        thread.start()
