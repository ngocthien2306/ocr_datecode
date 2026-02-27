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

    def __init__(
        self,
        engine_path: Optional[str] = None,
        save_debug_images: str = "never",
        debug_path: Optional[str] = None,
        angle_threshold: float = 3.0,
        margin_pixels: int = 30,
        conf_threshold: float = 0.25,
        check_rotation: bool = True,
        check_label_boundary: bool = True,
        check_misalignment: bool = True,
        check_center_alignment: bool = True,
        check_wrinkled: bool = True,
        center_offset_threshold: float = 50.0
    ):
        self.save_debug_images = save_debug_images.lower()
        self.debug_path = debug_path or f"{home}/Source/ocr_datecode/ai_services/test_result"
        self.angle_threshold = angle_threshold
        self.margin_pixels = margin_pixels
        self.conf_threshold = conf_threshold
        self.check_rotation = False
        self.check_label_boundary = False
        self.check_misalignment = False
        self.check_center_alignment = True
        self.check_wrinkled = True
        self.wrinkle_min_area = 2000
        self.center_offset_threshold = center_offset_threshold

        # Validate save_debug_images option
        if self.save_debug_images not in ["never", "on_fail", "always"]:
            logger.warning(
                f"Invalid save_debug_images option '{save_debug_images}', "
                f"using 'never'. Valid options: never, on_fail, always"
            )
            self.save_debug_images = "never"

        # Initialize YOLO OBB model
        if engine_path is None:
            engine_path = f"{home}/Source/ocr_datecode/weights/best_bottle_obb_m_320.engine"

        self.obb_model = None
        if YOLO_OBB_AVAILABLE:
            try:
                self.obb_model = YOLOOBBTensorRT(
                    engine_path=engine_path,
                    class_names=['product', 'label', "wrinkled"],
                    img_size=320
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

        # Initialize Wrinkled Segmenter (TRT instance segmentation)
        self.wrinkle_seg = None
        if self.check_wrinkled:
            try:
                from .wrinkle_segmenter import WrinkledSegmenterTRT
                wrinkle_engine = (
                    f"{home}/Source/ocr_datecode/weights/"
                    "best_wrinkled_instance_segmentation_crop_bottle.engine"
                )
                self.wrinkle_seg = WrinkledSegmenterTRT(
                    engine_path=wrinkle_engine,
                    min_area=self.wrinkle_min_area,
                )
            except Exception as e:
                logger.error(f"Failed to load WrinkledSegmenterTRT: {e}")
                self.check_wrinkled = False

    def should_verify_frame(self, transformed_bboxes: List[Dict[str, Any]]) -> bool:
        has_product = any(bbox.get('type') == 'product' for bbox in transformed_bboxes)
        has_label = any(bbox.get('type') == 'label' for bbox in transformed_bboxes)
        return has_product and has_label

    def verify_product_alignment(
        self,
        frame_img: np.ndarray,
        transformed_bboxes: List[Dict[str, Any]],
        camera: 'Camera',
        center_offset_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
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
            'camera': camera,
            'center_offset_threshold': center_offset_threshold
        }]

        results = self.verify_batch(frames_data)
        return results[0]

    def verify_batch(
        self,
        frames_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
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

        # ── Batch wrinkle segmentation (1 lần cho tất cả frames) ──────────────
        t_wrinkle_start = time.perf_counter()
        wrinkled_checks = {}
        if self.check_wrinkled and self.wrinkle_seg is not None:
            wrinkled_checks = self._batch_wrinkle_check(
                frames_to_check, frame_indices, batch_results
            )
        t_wrinkle = (time.perf_counter() - t_wrinkle_start) * 1000

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
                camera=data['camera'],
                center_offset_threshold=data.get('center_offset_threshold'),
                center_offset_threshold_left=data.get('center_offset_threshold_left'),
                center_offset_threshold_right=data.get('center_offset_threshold_right'),
                pre_computed_wrinkled_check=wrinkled_checks.get(orig_idx),
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
            'wrinkle_seg': t_wrinkle,
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
        camera: 'Camera',
        center_offset_threshold: Optional[float] = None,
        center_offset_threshold_left: Optional[float] = None,
        center_offset_threshold_right: Optional[float] = None,
        pre_computed_wrinkled_check: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        serial_number = camera.serial_number

        # Wrinkled check: dùng kết quả batch đã tính sẵn (ưu tiên) hoặc tính lại
        if pre_computed_wrinkled_check is not None:
            wrinkled_check = pre_computed_wrinkled_check
        elif self.check_wrinkled:
            wrinkled_check = self._check_wrinkled(
                boxes, scores, class_ids, transformed_bboxes, serial_number
            )
        else:
            wrinkled_check = {'ok': True, 'skipped': True, 'reason': 'Check disabled', 'has_wrinkled': False, 'wrinkled_count': 0, 'wrinkled_boxes': []}

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
        has_product = validation['has_product']
        has_label = validation['has_label']

        # Check rotation - requires BOTH product_box AND label_box
        if self.check_rotation and has_product and has_label:
            rotation_check = self._check_rotation(product_box, label_box, serial_number)
        elif not self.check_rotation:
            rotation_check = {'ok': True, 'skipped': True, 'reason': 'Check disabled'}
        else:
            rotation_check = {'ok': True, 'skipped': True, 'reason': 'Missing product or label box'}

        # Check misalignment - requires label_box
        if self.check_misalignment and has_label:
            misalignment_check = self._check_misalignment(
                label_box, transformed_bboxes, serial_number
            )
        elif not self.check_misalignment:
            misalignment_check = {'ok': True, 'skipped': True, 'reason': 'Check disabled'}
        else:
            misalignment_check = {'ok': True, 'skipped': True, 'reason': 'No label box detected'}

        # Check label region boundary - requires product_box
        if self.check_label_boundary and has_product:
            label_region_check = self._check_label_region_boundary(
                product_box, transformed_bboxes, serial_number
            )
        elif not self.check_label_boundary:
            label_region_check = {'ok': True, 'skipped': True, 'reason': 'Check disabled'}
        else:
            label_region_check = {'ok': True, 'skipped': True, 'reason': 'No product box detected'}

        # Check center alignment - requires product_box
        if self.check_center_alignment and has_product:
            center_alignment_check = self._check_center_alignment(
                product_box, transformed_bboxes, serial_number, center_offset_threshold,
                center_offset_threshold_left, center_offset_threshold_right 
            )
        elif self.check_center_alignment and has_label:
            center_alignment_check = self._check_center_alignment(
                label_box, transformed_bboxes, serial_number, center_offset_threshold,
                center_offset_threshold_left, center_offset_threshold_right 
            )
        elif not self.check_center_alignment:
            center_alignment_check = {'ok': True, 'skipped': True, 'reason': 'Check disabled'}
        else:
            center_alignment_check = {'ok': True, 'skipped': True, 'reason': 'No product or label box detected'}

        # Determine overall match
        overall_match = (
            wrinkled_check['ok'] and
            rotation_check['ok'] and
            misalignment_check['ok'] and
            label_region_check['ok'] and
            center_alignment_check['ok']
        )

        def _check_status(check_result):
            if check_result.get('skipped', False):
                return 'SKIP'
            return 'OK' if check_result['ok'] else 'FAIL'

        logger.info(
            f"[{serial_number}] Product verification: "
            f"wrinkled={_check_status(wrinkled_check)}, "
            f"rotation={_check_status(rotation_check)}, "
            f"misalignment={_check_status(misalignment_check)}, "
            f"label_boundary={_check_status(label_region_check)}, "
            f"center_alignment={_check_status(center_alignment_check)}, "
            f"overall={'PASS' if overall_match else 'FAIL'}"
        )

        # Save debug image based on configuration
        if self._should_save_debug_image(overall_match):
            self._visualize_result(
                frame_img, product_box, label_box, wrinkled_check,
                rotation_check, misalignment_check, label_region_check,
                center_alignment_check, overall_match, serial_number, transformed_bboxes
            )

        # Convert numpy arrays to Python native types for JSON serialization
        detected_boxes = {}
        if has_product:
            detected_boxes['product'] = {
                'box': product_box['box'].tolist() if isinstance(product_box['box'], np.ndarray) else product_box['box'],
                'score': float(product_box['score']),
                'class': str(product_box['class']),
                'corners': product_box['corners'].tolist() if isinstance(product_box['corners'], np.ndarray) else product_box['corners']
            }
        if has_label:
            detected_boxes['label'] = {
                'box': label_box['box'].tolist() if isinstance(label_box['box'], np.ndarray) else label_box['box'],
                'score': float(label_box['score']),
                'class': str(label_box['class']),
                'corners': label_box['corners'].tolist() if isinstance(label_box['corners'], np.ndarray) else label_box['corners']
            }
        if wrinkled_check.get('has_wrinkled', False):
            detected_boxes['wrinkled'] = wrinkled_check['wrinkled_boxes']

        return {
            'match': bool(overall_match),
            'skipped': False,
            'wrinkled_check': wrinkled_check,
            'rotation_check': rotation_check,
            'misalignment_check': misalignment_check,
            'label_region_check': label_region_check,
            'center_alignment_check': center_alignment_check,
            'detected_boxes': detected_boxes
        }

    def _filter_boxes_inside_product_region(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        transformed_bboxes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
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

        products = [b for b in filtered_boxes if b['class'] == 'product']
        labels = [b for b in filtered_boxes if b['class'] == 'label']

        # Check if we have at least one of either type
        if not products and not labels:
            return {'valid': False, 'reason': 'No boxes detected'}

        # Select highest confidence if multiple
        product_box = None
        label_box = None

        if products:
            if len(products) > 1:
                logger.debug(f"[{serial_number}] Multiple product boxes, selecting highest confidence")
            product_box = max(products, key=lambda x: x['score'])

        if labels:
            if len(labels) > 1:
                logger.debug(f"[{serial_number}] Multiple label boxes, selecting highest confidence")
            label_box = max(labels, key=lambda x: x['score'])

        logger.debug(
            f"[{serial_number}] Validated boxes: "
            f"product={'YES' if product_box else 'NO'}, "
            f"label={'YES' if label_box else 'NO'}"
        )

        return {
            'valid': True,
            'product_box': product_box,
            'label_box': label_box,
            'has_product': product_box is not None,
            'has_label': label_box is not None
        }

    def _check_rotation(
        self,
        product_box: Dict[str, Any],
        label_box: Dict[str, Any],
        serial_number: str
    ) -> Dict[str, Any]:

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

    def _check_center_alignment(
        self,
        product_box: Dict[str, Any],
        transformed_bboxes: List[Dict[str, Any]],
        serial_number: str,
        center_offset_threshold: Optional[float] = None,
        center_offset_threshold_left: Optional[float] = None,
        center_offset_threshold_right: Optional[float] = None
    ) -> Dict[str, Any]:

        # Use provided thresholds or fall back to default
        default_threshold = center_offset_threshold if center_offset_threshold is not None else self.center_offset_threshold
        threshold_left = center_offset_threshold_left if center_offset_threshold_left is not None else default_threshold
        threshold_right = center_offset_threshold_right if center_offset_threshold_right is not None else default_threshold

        # Find template region from transformed_bboxes
        template_region = next(
            (bbox for bbox in transformed_bboxes if bbox.get('type') == 'template'),
            None
        )

        if template_region is None:
            logger.warning(f"[{serial_number}] No template region found in transformed_bboxes")
            return {
                'ok': True,
                'skipped': True,
                'reason': 'No template region in transformed_bboxes'
            }

        # Get template region polygon
        template_poly = np.array(template_region['points'], dtype=np.float32)
        template_center_x = np.mean(template_poly[:, 0])
        template_center_y = np.mean(template_poly[:, 1])

        # Get detected product box corners
        product_corners = product_box['corners']
        if isinstance(product_corners, list):
            product_corners = np.array(product_corners, dtype=np.float32)

        product_center_x = np.mean(product_corners[:, 0])
        product_center_y = np.mean(product_corners[:, 1])

        # Calculate offset on x-axis (positive = product is to the right of template)
        offset_x = product_center_x - template_center_x
        abs_offset_x = abs(offset_x)

        # Determine direction and check against appropriate threshold
        if offset_x > 0:
            # Product is to the LEFT of template center
            direction = 'left'
            threshold_used = threshold_left
            ok = abs_offset_x <= threshold_left
        else:
            # Product is to the RIGHT of template center (or centered)
            direction = 'right'
            threshold_used = threshold_right
            ok = abs_offset_x <= threshold_right

        logger.debug(
            f"[{serial_number}] Center alignment check: "
            f"template_center=({template_center_x:.1f}, {template_center_y:.1f}), "
            f"product_center=({product_center_x:.1f}, {product_center_y:.1f}), "
            f"offset_x={offset_x:.1f}px ({direction}), "
            f"threshold_left={threshold_left}px, threshold_right={threshold_right}px, "
            f"result={'OK' if ok else 'FAIL'}"
        )

        return {
            'ok': bool(ok),
            'offset_x': float(offset_x),
            'abs_offset_x': float(abs_offset_x),
            'direction': direction,
            'threshold_left': float(threshold_left),
            'threshold_right': float(threshold_right),
            'threshold_used': float(threshold_used),
            'template_center': [float(template_center_x), float(template_center_y)],
            'product_center': [float(product_center_x), float(product_center_y)]
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Batch wrinkle segmentation helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _batch_wrinkle_check(
        self,
        frames_to_check: List[Dict[str, Any]],
        frame_indices: List[int],
        batch_results: List,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Crop product region từ mỗi frame → batch predict_batch() 1 lần → build wrinkled_check.
        Returns: {orig_idx: wrinkled_check}
        """
        wrinkled_checks: Dict[int, Dict[str, Any]] = {}
        crops_info = []  # (orig_idx, crop, cx, cy, w, h, angle, crop_offset, frame_img)

        for i, orig_idx in enumerate(frame_indices):
            boxes, scores, class_ids = batch_results[i]
            data = frames_to_check[i]

            product_box = self._get_product_box(boxes, scores, class_ids, data['transformed_bboxes'])
            if product_box is None:
                continue

            cx, cy, w, h, angle = product_box['box']
            try:
                crop, crop_offset = self.wrinkle_seg.crop_from_obb(
                    data['frame_img'], cx, cy, w, h, angle
                )
            except Exception as e:
                logger.warning(f"crop_from_obb failed for frame {orig_idx}: {e}")
                continue

            if crop.size == 0:
                continue

            crops_info.append((orig_idx, crop, cx, cy, w, h, angle, crop_offset, data['frame_img']))

        if not crops_info:
            return wrinkled_checks

        # Batch predict tất cả crops trong 1 lần inference
        crops = [ci[1] for ci in crops_info]
        try:
            seg_results, seg_timing = self.wrinkle_seg.predict_batch(
                crops, conf_threshold=0.5, return_timing=True
            )
            logger.info(
                f"[WrinkleSeg] batch={len(crops)} | "
                f"pre={seg_timing['preprocess']:.1f}ms  "
                f"h2d={seg_timing['h2d']:.1f}ms  "
                f"infer={seg_timing['inference']:.1f}ms  "
                f"d2h={seg_timing['d2h']:.1f}ms  "
                f"post={seg_timing['postprocess']:.1f}ms  "
                f"total={seg_timing['total']:.1f}ms  "
                f"per_img={seg_timing['per_image']:.1f}ms"
            )
        except Exception as e:
            logger.error(f"WrinkledSegmenterTRT batch predict failed: {e}")
            return wrinkled_checks

        # Build wrinkled_check per frame
        for j, (orig_idx, crop, cx, cy, w, h, angle, crop_offset, frame_img) in enumerate(crops_info):
            seg_boxes, seg_masks = seg_results[j]
            wrinkled_checks[orig_idx] = self.wrinkle_seg.build_wrinkled_check(
                seg_boxes=seg_boxes,
                seg_masks=seg_masks,
                cx=cx, cy=cy, w=w, h=h, angle=angle,
                crop_offset=crop_offset,
                frame_shape=frame_img.shape,
            )

        return wrinkled_checks

    def _get_product_box(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        transformed_bboxes: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Tìm product box để crop cho wrinkle segmenter.
        Ưu tiên: YOLO product box nằm trong template polygon.
        Fallback: dùng trực tiếp template product polygon từ transformed_bboxes.
        """
        product_region = next(
            (b for b in transformed_bboxes if b.get('type') == 'product'), None
        )
        if product_region is None:
            return None

        # ── Ưu tiên: YOLO detected product box ──────────────────────────────
        if len(boxes) > 0:
            product_poly = np.array(product_region['points'], dtype=np.float32)
            for box, score, cls_id in zip(boxes, scores, class_ids):
                if self.obb_model.class_names[cls_id] != 'product':
                    continue
                corners = self._obb_to_corners(box)
                if self._is_box_inside_polygon(corners, product_poly):
                    return {'box': box, 'score': float(score), 'source': 'yolo'}

        # ── Fallback: dùng template product polygon ──────────────────────────
        pts = np.array(product_region['points'], dtype=np.float32)
        x, y, bw, bh = cv2.boundingRect(pts.astype(np.int32))
        cx = float(x + bw / 2)
        cy = float(y + bh / 2)
        # angle=0 vì template polygon là axis-aligned sau khi transform
        box_fallback = np.array([cx, cy, float(bw), float(bh), 0.0], dtype=np.float32)
        logger.debug(
            f"No YOLO product box — fallback to template polygon: "
            f"cx={cx:.0f} cy={cy:.0f} w={bw} h={bh}"
        )
        return {'box': box_fallback, 'score': 1.0, 'source': 'template'}

    # ──────────────────────────────────────────────────────────────────────────

    def _check_wrinkled(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        transformed_bboxes: List[Dict[str, Any]],
        serial_number: str
    ) -> Dict[str, Any]:
        # Find product region
        product_region = next(
            (bbox for bbox in transformed_bboxes if bbox.get('type') == 'product'),
            None
        )

        if product_region is None:
            logger.debug(f"[{serial_number}] No product region found for wrinkled check")
            return {
                'ok': True,
                'has_wrinkled': False,
                'wrinkled_count': 0,
                'wrinkled_boxes': [],
                'reason': 'No product region in template'
            }

        product_poly = np.array(product_region['points'], dtype=np.float32)

        # Filter wrinkled boxes inside product region
        wrinkled_boxes = []
        for box, score, cls_id in zip(boxes, scores, class_ids):
            # Check if this is a wrinkled class
            if self.obb_model.class_names[cls_id] != 'wrinkled':
                continue

            # Convert OBB to corners
            corners = self._obb_to_corners(box)

            # Check if all corners are inside product region
            if self._is_box_inside_polygon(corners, product_poly):
                wrinkled_boxes.append({
                    'box': box.tolist() if isinstance(box, np.ndarray) else box,
                    'score': float(score),
                    'class': 'wrinkled',
                    'corners': corners.tolist() if isinstance(corners, np.ndarray) else corners
                })

        has_wrinkled = len(wrinkled_boxes) > 0
        wrinkled_count = len(wrinkled_boxes)
        ok = not has_wrinkled

        logger.debug(
            f"[{serial_number}] Wrinkled check: "
            f"count={wrinkled_count}, "
            f"result={'FAIL' if has_wrinkled else 'OK'}"
        )

        return {
            'ok': bool(not has_wrinkled),
            'has_wrinkled': bool(has_wrinkled),
            'wrinkled_count': int(wrinkled_count),
            'wrinkled_boxes': wrinkled_boxes
        }

    # ========== Helper Methods ==========

    def _should_save_debug_image(self, overall_match: bool) -> bool:
        if self.save_debug_images == "never":
            return False
        elif self.save_debug_images == "on_fail":
            return not overall_match  # Only save when failed
        elif self.save_debug_images == "always":
            return True
        return False

    def _obb_to_corners(self, box: np.ndarray) -> np.ndarray:
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
        product_box: Optional[Dict[str, Any]],
        label_box: Optional[Dict[str, Any]],
        wrinkled_check: Dict[str, Any],
        rotation_check: Dict[str, Any],
        misalignment_check: Dict[str, Any],
        label_region_check: Dict[str, Any],
        center_alignment_check: Dict[str, Any],
        overall_match: bool,
        serial_number: str,
        transformed_bboxes: List[Dict[str, Any]]
    ):
        """
        Visualize verification result and save debug image.

        Args:
            frame_img: Frame image
            product_box: Product box dict
            label_box: Label box dict
            wrinkled_check: Wrinkled check result
            rotation_check: Rotation check result
            misalignment_check: Misalignment check result
            label_region_check: Label region boundary check result
            center_alignment_check: Center alignment check result
            overall_match: Overall pass/fail
            serial_number: Camera serial
            transformed_bboxes: Template bounding boxes for getting template center
        """
        try:
            result_img = frame_img.copy()

            # Draw product box (green) if available
            if product_box is not None:
                product_corners = product_box['corners'].astype(np.int32)
                cv2.polylines(result_img, [product_corners], True, (0, 255, 0), 2)
                cv2.putText(
                    result_img, f"Product: {product_box['score']:.2f}",
                    tuple(product_corners[0]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2
                )

            # Draw label box (blue if OK, red if fail) if available
            if label_box is not None:
                label_color = (255, 0, 0) if not overall_match else (0, 0, 255)
                label_corners = label_box['corners'].astype(np.int32)
                cv2.polylines(result_img, [label_corners], True, label_color, 2)
                cv2.putText(
                    result_img, f"Label: {label_box['score']:.2f}",
                    tuple(label_corners[0]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, label_color, 2
                )

            # Draw wrinkled regions (orange) — contour mask nếu có, fallback corners
            if wrinkled_check.get('has_wrinkled', False):
                wrinkled_boxes = wrinkled_check.get('wrinkled_boxes', [])
                wrinkled_color = (0, 165, 255)  # Orange
                overlay = result_img.copy()
                for idx, wrinkled_box in enumerate(wrinkled_boxes):
                    contour = wrinkled_box.get('contour')
                    if contour is not None:
                        contour_arr = np.array(contour, dtype=np.int32)
                        cv2.fillPoly(overlay, [contour_arr], wrinkled_color)
                        cv2.polylines(result_img, [contour_arr], True, wrinkled_color, 2)
                    else:
                        wrinkled_corners = np.array(wrinkled_box['corners'], dtype=np.int32)
                        cv2.polylines(result_img, [wrinkled_corners], True, wrinkled_color, 2)

                    area     = wrinkled_box.get('area', 0)
                    area_pct = wrinkled_box.get('area_pct', 0.0)
                    label_text = f"Wrinkled#{idx+1}: {wrinkled_box['score']:.2f} {area}px({area_pct:.1f}%)"
                    anchor = contour[0] if contour else wrinkled_box['corners'][0]
                    cv2.putText(
                        result_img, label_text,
                        tuple(map(int, anchor)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, wrinkled_color, 2
                    )
                # Semi-transparent overlay cho toàn bộ vùng wrinkled
                cv2.addWeighted(overlay, 0.35, result_img, 0.65, 0, result_img)

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
            if wrinkled_check.get('skipped', False):
                wrinkled_status = "SKIP"
                wrinkled_text = f"Wrinkled: {wrinkled_status}"
            else:
                wrinkled_status = "OK" if wrinkled_check['ok'] else "FAIL"
                wrinkled_count = wrinkled_check.get('wrinkled_count', 0)
                wrinkled_text = f"Wrinkled: {wrinkled_status} (count={wrinkled_count})"
            cv2.putText(
                result_img,
                wrinkled_text,
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, status_color, 2
            )

            y_offset += 30
            if rotation_check.get('skipped', False):
                rotation_status = "SKIP"
                rotation_text = f"Rotation: {rotation_status}"
            else:
                rotation_status = "OK" if rotation_check['ok'] else "FAIL"
                rotation_text = f"Rotation: {rotation_status} ({rotation_check['angle_diff']:.2f} deg)"
            cv2.putText(
                result_img,
                rotation_text,
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

            y_offset += 30
            label_region_status = "OK" if label_region_check['ok'] else "FAIL"
            cv2.putText(
                result_img,
                f"Label Boundary: {label_region_status}",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, status_color, 2
            )

            y_offset += 30
            center_status = "OK" if center_alignment_check['ok'] else "FAIL"
            offset_x_val = center_alignment_check.get('offset_x', 0)
            direction = center_alignment_check.get('direction', '')
            threshold_used = center_alignment_check.get('threshold_used', 0)
            direction_label = f" {direction.upper()}" if direction else ""
            cv2.putText(
                result_img,
                f"Center Alignment: {center_status} ({offset_x_val:.1f}px{direction_label}, thresh={threshold_used:.0f})",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, status_color, 2
            )

            # Draw centers if available
            if not center_alignment_check.get('skipped', False):
                template_center = center_alignment_check.get('template_center')
                product_center = center_alignment_check.get('product_center')

                if template_center:
                    # Draw template center (cyan color)
                    cv2.circle(
                        result_img,
                        (int(template_center[0]), int(template_center[1])),
                        8,  # radius
                        (255, 255, 0),  # cyan
                        -1  # filled
                    )
                    cv2.putText(
                        result_img, "Template",
                        (int(template_center[0]) + 12, int(template_center[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 2
                    )

                if product_center:
                    # Draw product center (magenta color)
                    cv2.circle(
                        result_img,
                        (int(product_center[0]), int(product_center[1])),
                        8,  # radius
                        (255, 0, 255),  # magenta
                        -1  # filled
                    )
                    cv2.putText(
                        result_img, "Product",
                        (int(product_center[0]) + 12, int(product_center[1]) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 255), 2
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
