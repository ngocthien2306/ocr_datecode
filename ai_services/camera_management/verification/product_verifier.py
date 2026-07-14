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

        # Crop sample collection (set True to collect samples)
        self.save_crop_samples = True
        self.crop_sample_dir = f"{home}/Source/ocr_datecode/crop_samples"
        self.crop_sample_margin = 0   # px mở rộng ra ngoài product bbox
        self.crop_sample_max = 100     # tối đa 100 ảnh / camera
        self._crop_sample_counts: dict = {}  # {serial_number: count}

        # Validate save_debug_images option
        if self.save_debug_images not in ["never", "on_fail", "always"]:
            logger.warning(
                f"Invalid save_debug_images option '{save_debug_images}', "
                f"using 'never'. Valid options: never, on_fail, always"
            )
            self.save_debug_images = "never"

        # Initialize YOLO OBB model
        if engine_path is None:
            engine_path = f"{home}/Source/ocr_datecode/weights/best_bottle_obb_l_320.engine"

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

        # Color verification service for Check_Color templates with a 'product'
        # annotation. Decoupled image-proc bottle detection + HSV pixel count.
        try:
            from .color_verifier import ColorVerificationService
            self.color_verifier = ColorVerificationService(
                save_debug_images=self.save_debug_images,
                debug_path=self.debug_path,
            )
            logger.info("ColorVerificationService initialized")
        except Exception as e:
            logger.error(f"Failed to load ColorVerificationService: {e}")
            self.color_verifier = None

        # Initialize Wrinkled Segmenter (TRT instance segmentation)
        self.wrinkle_seg = None
        if self.check_wrinkled:
            try:
                from .wrinkle_segmenter import WrinkledSegmenterTRT
                wrinkle_engine = (
                    f"{home}/Source/ocr_datecode/weights/"
                    "best_wrinkled_instance_segmentation_crop_bottle_n.engine"
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

        # ── Step 0: route Check_Color + product frames to color_verifier ─────
        # A frame is considered "color_check" when its camera has function_type
        # == 'Check_Color' AND the template has a 'product' annotation. Other
        # Check_Color cases (e.g. rotate-cap-and-OCR) keep flowing through the
        # YOLO/image-proc paths below — they simply won't pass should_verify_frame.
        def _is_color_check(d):
            cam = d.get('camera')
            if not cam or getattr(cam, 'function_type', '') != 'Check_Color':
                return False
            if not self.color_verifier:
                return False
            # Prefer the camera's raw template (decoupled from SuperPoint) so we
            # can route even when matching failed.
            template_idx = int(d.get('template_idx', 0) or 0)
            templates = getattr(cam, 'templates', None) or []
            if template_idx < len(templates):
                anns = templates[template_idx].get('annotations') or []
                if any(a.get('type') == 'product' for a in anns):
                    return True
            # Fallback: check transformed_bboxes (post-match path)
            return any(b.get('type') == 'product' for b in d.get('transformed_bboxes', []))

        color_flags = [_is_color_check(d) for d in frames_data]
        if color_flags and all(color_flags):
            return self.color_verifier.verify_batch(frames_data)
        if any(color_flags):
            color_indices = [i for i, c in enumerate(color_flags) if c]
            other_indices = [i for i, c in enumerate(color_flags) if not c]
            color_results = (self.color_verifier.verify_batch([frames_data[i] for i in color_indices])
                             if color_indices else [])
            other_results = (self.verify_batch([frames_data[i] for i in other_indices])
                             if other_indices else [])
            merged: List[Optional[Dict[str, Any]]] = [None] * len(frames_data)
            for k, i in enumerate(color_indices):
                merged[i] = color_results[k]
            for k, i in enumerate(other_indices):
                merged[i] = other_results[k]
            return merged  # type: ignore[return-value]

        # ── Branch by product_detection_method ──────────────────────────────
        # Per-camera setting. Frames from cameras using "yolo_segment" mode
        # skip YOLO OBB entirely (image processing).
        def _cam_method(d):
            cam = d.get('camera')
            return getattr(cam, 'product_detection_method', 'yolo_obb') if cam else 'yolo_obb'

        methods = [_cam_method(d) for d in frames_data]
        if methods and all(m == 'yolo_segment' for m in methods):
            return self._verify_batch_image_proc(frames_data)
        # Mixed batch (rare): split, process separately, merge.
        if 'yolo_segment' in methods:
            yolo_indices = [i for i, m in enumerate(methods) if m == 'yolo_obb']
            img_indices  = [i for i, m in enumerate(methods) if m == 'yolo_segment']
            yolo_results = (self._verify_batch_yolo([frames_data[i] for i in yolo_indices])
                            if yolo_indices else [])
            img_results  = (self._verify_batch_image_proc([frames_data[i] for i in img_indices])
                            if img_indices else [])
            merged = [None] * len(frames_data)
            for k, i in enumerate(yolo_indices): merged[i] = yolo_results[k]
            for k, i in enumerate(img_indices):  merged[i] = img_results[k]
            return merged
        # All yolo_obb (default path)
        return self._verify_batch_yolo(frames_data)

    def _verify_batch_yolo(
        self,
        frames_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Original YOLO OBB batch path (was verify_batch)."""
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
                center_offset_unit=data.get('center_offset_unit', 'px'),
                wrinkle_show_when_pass=data.get('wrinkle_show_when_pass', True),
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

    # ──────────────────────────────────────────────────────────────────────────
    # Image-processing path (recipe option product_detection_method='yolo_segment').
    # Tên 'yolo_segment' giữ cho UI consistency nhưng KHÔNG dùng YOLO model.
    # ──────────────────────────────────────────────────────────────────────────
    def _load_template_walls(self, serial_number: str) -> Optional[Dict[str, Any]]:
        """Load template_walls.json (saved by MatcherFactory) — cached per serial."""
        if not hasattr(self, '_template_walls_cache'):
            self._template_walls_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        if serial_number in self._template_walls_cache:
            return self._template_walls_cache[serial_number]
        try:
            import json
            home = os.environ.get('HOME', '')
            walls_path = Path(home) / 'Source' / 'ocr_datecode' / 'crop_samples' / serial_number / 'template_walls.json'
            if walls_path.exists():
                walls = json.loads(walls_path.read_text())
                self._template_walls_cache[serial_number] = walls
                return walls
        except Exception as e:
            logger.error(f"[{serial_number}] Failed to load template_walls: {e}")
        self._template_walls_cache[serial_number] = None
        return None

    def _verify_batch_image_proc(
        self,
        frames_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Image processing batch path. Skip YOLO OBB hoàn toàn.
        Wrinkle vẫn chạy qua WrinkledSegmenter (separate model, không phụ thuộc YOLO).
        """
        import time
        from .image_proc_detector import (
            detect_product_box, label_box_from_pts, EdgeParams, DEFAULT_EDGE_PARAMS,
        )
        t_start = time.perf_counter()

        # Filter frames that need verification (cùng logic như YOLO path)
        t_filter_start = time.perf_counter()
        frames_to_check = []
        frame_indices = []
        for idx, data in enumerate(frames_data):
            if self.should_verify_frame(data['transformed_bboxes']):
                frames_to_check.append(data)
                frame_indices.append(idx)
        t_filter = (time.perf_counter() - t_filter_start) * 1000

        if not frames_to_check:
            return [{
                'match': True, 'skipped': True,
                'reason': 'No product or label region in template',
                'timing': {'total': t_filter, 'filter': t_filter}
            } for _ in frames_data]

        # ── Run image_proc detection + wrinkle segmentation IN PARALLEL ─────
        # image_proc: CPU-bound (cv2/numpy release GIL)
        # wrinkle:    GPU-bound (TRT inference)
        # → Different resources → true parallelism. Save 20-30ms/batch.
        from concurrent.futures import ThreadPoolExecutor

        def _detect_one(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            cam = data['camera']
            serial = cam.serial_number
            label_region = next(
                (b for b in data['transformed_bboxes'] if b.get('type') == 'label'), None
            )
            if label_region is None:
                return None

            # Per-template edge_config (tuned in EdgeSetupModal). Falls back to
            # globals + factory-computed template_walls when not configured.
            template_idx = int(data.get('template_idx', 0) or 0)
            templates = getattr(cam, 'templates', None) or []
            edge_cfg = None
            if 0 <= template_idx < len(templates):
                edge_cfg = (templates[template_idx] or {}).get('edge_config')

            params = EdgeParams.from_config(edge_cfg) if edge_cfg else DEFAULT_EDGE_PARAMS

            # Walls: prefer the ones saved into edge_config at setup time; else
            # use the factory-computed template_walls.json for this camera.
            walls = (edge_cfg or {}).get('template_walls') if edge_cfg else None
            if not walls:
                walls = self._load_template_walls(serial)
            if walls is None:
                return None

            wall_type = getattr(cam, 'product_box_wall_type', 'outer') or 'outer'
            try:
                return detect_product_box(
                    data['frame_img'], label_region['points'], walls,
                    serial_number=serial, wall_type=wall_type, params=params,
                )
            except Exception as e:
                logger.warning(f"[{serial}] image_proc detect failed: {e}")
                return None

        def _run_detect_all():
            """Detect product box cho tất cả frames (CPU parallel)."""
            n = len(frames_to_check)
            if n > 1:
                with ThreadPoolExecutor(max_workers=min(n, 4)) as p:
                    return list(p.map(_detect_one, frames_to_check))
            return [_detect_one(frames_to_check[0])]

        def _run_wrinkle_all():
            """Wrinkle segmentation cho tất cả frames (GPU TRT)."""
            if self.check_wrinkled and self.wrinkle_seg is not None:
                empty_batch = [(np.empty((0, 5)), np.empty(0), np.empty(0, dtype=int))
                                for _ in frames_to_check]
                return self._batch_wrinkle_check(frames_to_check, frame_indices, empty_batch)
            return {}

        t_par_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as outer:
            detect_future  = outer.submit(_run_detect_all)
            wrinkle_future = outer.submit(_run_wrinkle_all)
            per_frame_product_boxes = detect_future.result()
            wrinkled_checks: Dict[int, Dict[str, Any]] = wrinkle_future.result()
        t_par = (time.perf_counter() - t_par_start) * 1000
        # Approximate individual phases for logging (actual was concurrent)
        t_detect = t_par
        t_wrinkle = t_par

        # Process each frame
        t_process_start = time.perf_counter()
        results = [{
            'match': True, 'skipped': True,
            'reason': 'No product or label region in template',
            'timing': {'total': 0.0}
        } for _ in frames_data]

        for i, orig_idx in enumerate(frame_indices):
            data = frames_to_check[i]
            product_box = per_frame_product_boxes[i]
            label_region = next(
                (b for b in data['transformed_bboxes'] if b.get('type') == 'label'), None
            )
            label_box = label_box_from_pts(label_region['points']) if label_region else None

            result = self._process_single_frame_image_proc(
                product_box=product_box,
                label_box=label_box,
                transformed_bboxes=data['transformed_bboxes'],
                frame_img=data['frame_img'],
                camera=data['camera'],
                center_offset_threshold=data.get('center_offset_threshold'),
                center_offset_threshold_left=data.get('center_offset_threshold_left'),
                center_offset_threshold_right=data.get('center_offset_threshold_right'),
                center_offset_unit=data.get('center_offset_unit', 'px'),
                wrinkle_show_when_pass=data.get('wrinkle_show_when_pass', True),
                pre_computed_wrinkled_check=wrinkled_checks.get(orig_idx),
            )
            results[orig_idx] = result

        t_process = (time.perf_counter() - t_process_start) * 1000
        t_total = (time.perf_counter() - t_start) * 1000

        timing_info = {
            'total': t_total, 'filter': t_filter,
            'detect_plus_wrinkle_parallel': t_par,
            'processing': t_process,
            'frames_checked': len(frames_to_check), 'frames_total': len(frames_data),
            'method': 'image_proc',
        }
        for idx in frame_indices:
            if results[idx]:
                results[idx]['timing'] = timing_info.copy()

        logger.info(
            f"Product verification (image_proc): {len(frames_to_check)}/{len(frames_data)} frames, "
            f"total={t_total:.1f}ms (filter={t_filter:.1f}ms, "
            f"detect+wrinkle_parallel={t_par:.1f}ms, process={t_process:.1f}ms)"
        )
        return results

    def _process_single_frame_image_proc(
        self,
        product_box: Optional[Dict[str, Any]],
        label_box: Optional[Dict[str, Any]],
        transformed_bboxes: List[Dict[str, Any]],
        frame_img: np.ndarray,
        camera: 'Camera',
        center_offset_threshold: Optional[float] = None,
        center_offset_threshold_left: Optional[float] = None,
        center_offset_threshold_right: Optional[float] = None,
        center_offset_unit: str = 'px',
        wrinkle_show_when_pass: bool = True,
        pre_computed_wrinkled_check: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Same as _process_single_frame but accept pre-built product/label boxes
        (skip YOLO filter+validate). Other checks (rotation/boundary/misalignment/wrinkle)
        share logic with YOLO path.
        """
        serial_number = camera.serial_number

        # Wrinkle check (use pre-computed batch result if available)
        if pre_computed_wrinkled_check is not None:
            wrinkled_check = pre_computed_wrinkled_check
        else:
            wrinkled_check = {
                'ok': True, 'skipped': True, 'reason': 'Check disabled',
                'has_wrinkled': False, 'wrinkled_count': 0, 'wrinkled_boxes': []
            }

        if product_box is None:
            logger.debug(f"[{serial_number}] image_proc: no product_box detected")
            return {
                'match': False,
                'skipped': False,
                'reason': 'Image processing failed to detect bottle walls',
                'wrinkled_check': wrinkled_check,
            }

        has_product = True
        has_label = label_box is not None

        # Rotation check — disabled by default
        if self.check_rotation and has_product and has_label:
            rotation_check = self._check_rotation(product_box, label_box, serial_number)
        elif not self.check_rotation:
            rotation_check = {'ok': True, 'skipped': True, 'reason': 'Check disabled'}
        else:
            rotation_check = {'ok': True, 'skipped': True, 'reason': 'Missing boxes'}

        # Label region/boundary check — disabled by default
        if self.check_label_boundary and has_product:
            label_region_check = self._check_edge_touch(
                product_box, transformed_bboxes, serial_number
            )
        else:
            label_region_check = {'ok': True, 'skipped': True, 'reason': 'Check disabled'}

        # Misalignment — disabled by default (not implemented for image_proc path)
        misalignment_check = {'ok': True, 'skipped': True, 'reason': 'Check disabled'}

        # Center alignment — chính cái user muốn dùng image_proc cho cái này
        if self.check_center_alignment and has_product:
            center_alignment_check = self._check_center_alignment(
                product_box, transformed_bboxes, serial_number,
                center_offset_threshold=center_offset_threshold,
                center_offset_threshold_left=center_offset_threshold_left,
                center_offset_threshold_right=center_offset_threshold_right,
                center_offset_unit=center_offset_unit,
                label_box=label_box,
            )
        else:
            center_alignment_check = {'ok': True, 'skipped': True, 'reason': 'Check disabled'}

        # Aggregate
        def _check_status(check_result):
            if not check_result or check_result.get('skipped'):
                return True
            return check_result.get('ok', True)

        overall_match = (
            _check_status(rotation_check)
            and _check_status(label_region_check)
            and _check_status(misalignment_check)
            and _check_status(center_alignment_check)
            and _check_status(wrinkled_check)
        )

        # Save debug image if needed (re-use YOLO path's visualization signature)
        if self._should_save_debug_image(overall_match):
            try:
                self._visualize_result(
                    frame_img, product_box, label_box, wrinkled_check,
                    rotation_check, misalignment_check, label_region_check,
                    center_alignment_check, overall_match, serial_number,
                    transformed_bboxes,
                )
            except Exception as e:
                logger.warning(f"[{serial_number}] debug image save failed: {e}")

        # Build detected_boxes dict (SAME FORMAT as YOLO path → visualizer works)
        # Yellow product OBB drawn from corners. Label hidden (template polygon shown
        # via transformed_bboxes path). Wrinkled added if any candidates to draw.
        detected_boxes: Dict[str, Any] = {}
        detected_boxes['product'] = {
            'box': product_box['box'].tolist() if isinstance(product_box['box'], np.ndarray) else product_box['box'],
            'score': float(product_box['score']),
            'class': str(product_box['class']),
            'corners': product_box['corners'].tolist() if isinstance(product_box['corners'], np.ndarray) else product_box['corners'],
        }
        if label_box is not None:
            detected_boxes['label'] = {
                'box': label_box['box'].tolist() if isinstance(label_box['box'], np.ndarray) else label_box['box'],
                'score': float(label_box['score']),
                'class': str(label_box['class']),
                'corners': label_box['corners'].tolist() if isinstance(label_box['corners'], np.ndarray) else label_box['corners'],
            }
        wrinkled_boxes_to_draw = wrinkled_check.get('wrinkled_boxes', [])
        if wrinkled_boxes_to_draw and (wrinkled_check.get('has_wrinkled', False) or wrinkle_show_when_pass):
            detected_boxes['wrinkled'] = wrinkled_boxes_to_draw

        # Return EXACTLY same shape as YOLO path's _process_single_frame
        return {
            'match': bool(overall_match),
            'skipped': False,
            'wrinkled_check': wrinkled_check,
            'rotation_check': rotation_check,
            'misalignment_check': misalignment_check,
            'label_region_check': label_region_check,
            'center_alignment_check': center_alignment_check,
            'detected_boxes': detected_boxes,
        }

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
        center_offset_unit: str = 'px',
        wrinkle_show_when_pass: bool = True,
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

        # Save crop sample for analysis (controlled by self.save_crop_samples)
        self._save_crop_sample(frame_img, transformed_bboxes, serial_number)

        # Check center alignment - requires product_box
        if self.check_center_alignment and has_product:
            center_alignment_check = self._check_center_alignment(
                product_box, transformed_bboxes, serial_number,
                center_offset_threshold, center_offset_threshold_left, center_offset_threshold_right,
                center_offset_unit=center_offset_unit,
                label_box=label_box if has_label else None,
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
        # Draw wrinkle candidates (already filtered by conf + min_region_area):
        # - When wrinkle FAILS (has_wrinkled=True): always draw
        # - When PASS: only draw if wrinkle_show_when_pass=True (recipe option)
        wrinkled_boxes_to_draw = wrinkled_check.get('wrinkled_boxes', [])
        if wrinkled_boxes_to_draw and (wrinkled_check.get('has_wrinkled', False) or wrinkle_show_when_pass):
            detected_boxes['wrinkled'] = wrinkled_boxes_to_draw

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
        center_offset_threshold_right: Optional[float] = None,
        center_offset_unit: str = 'px',
        label_box: Optional[Dict[str, Any]] = None,  # signature BC, no longer used
    ) -> Dict[str, Any]:
        # Use provided thresholds or fall back to default
        default_threshold = center_offset_threshold if center_offset_threshold is not None else self.center_offset_threshold
        threshold_left = center_offset_threshold_left if center_offset_threshold_left is not None else default_threshold
        threshold_right = center_offset_threshold_right if center_offset_threshold_right is not None else default_threshold

        # --- Reference: always use template polygon type='label' transformed by SuperPoint ---
        template_region = next(
            (bbox for bbox in transformed_bboxes if bbox.get('type') == 'label'),
            None
        )
        if template_region is None:
            logger.warning(f"[{serial_number}] No template label region found")
            return {
                'ok': True,
                'skipped': True,
                'reason': 'No template label region in transformed_bboxes'
            }
        template_poly = np.array(template_region['points'], dtype=np.float32)
        ref_center_x = float(np.mean(template_poly[:, 0]))
        ref_center_y = float(np.mean(template_poly[:, 1]))
        # Axis-aligned width of polygon — used to convert % → px
        ref_width = float(np.max(template_poly[:, 0]) - np.min(template_poly[:, 0]))

        # --- Convert thresholds based on unit ---
        unit = (center_offset_unit or 'px').lower()
        if unit == 'pct':
            threshold_left_px  = ref_width * (threshold_left  / 100.0)
            threshold_right_px = ref_width * (threshold_right / 100.0)
        else:
            threshold_left_px  = float(threshold_left)
            threshold_right_px = float(threshold_right)

        # --- Product box center ---
        product_corners = product_box['corners']
        if isinstance(product_corners, list):
            product_corners = np.array(product_corners, dtype=np.float32)
        product_center_x = float(np.mean(product_corners[:, 0]))
        product_center_y = float(np.mean(product_corners[:, 1]))

        # --- Compute X-axis offset ---
        # offset_x > 0: product is on the RIGHT of reference
        # offset_x < 0: product is on the LEFT of reference
        offset_x = product_center_x - ref_center_x
        abs_offset_x = abs(offset_x)

        if offset_x > 0:
            direction = 'left'
            threshold_used_px = threshold_left_px
            ok = abs_offset_x <= threshold_left_px
        else:
            direction = 'right'
            threshold_used_px = threshold_right_px
            ok = abs_offset_x <= threshold_right_px

        logger.debug(
            f"[{serial_number}] Center alignment (ref=template_region, unit={unit}, "
            f"ref_w={ref_width:.1f}px): "
            f"ref_center=({ref_center_x:.1f}, {ref_center_y:.1f}), "
            f"product_center=({product_center_x:.1f}, {product_center_y:.1f}), "
            f"offset_x={offset_x:.1f}px ({direction}), "
            f"thresh={threshold_left}{unit}/{threshold_right}{unit} "
            f"(={threshold_left_px:.1f}/{threshold_right_px:.1f}px), "
            f"result={'OK' if ok else 'FAIL'}"
        )

        return {
            'ok': bool(ok),
            'offset_x': float(offset_x),
            'abs_offset_x': float(abs_offset_x),
            'direction': direction,
            'unit': unit,
            'ref_width': ref_width,
            # Raw values as entered by user (in selected unit)
            'threshold_left': float(threshold_left),
            'threshold_right': float(threshold_right),
            # Values converted to px at runtime
            'threshold_left_px': float(threshold_left_px),
            'threshold_right_px': float(threshold_right_px),
            'threshold_used_px': float(threshold_used_px),
            'template_center': [ref_center_x, ref_center_y],
            'product_center': [product_center_x, product_center_y]
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
        Crop product region từ mỗi frame → build wrinkled_check.

        Per-frame routing: if the frame's template has anomaly_config
        enabled with an onnx_path, run anomaly_inference.py instead of
        WrinkledSegmenterTRT — same crop, swap-in model, same result shape
        (see anomaly_inference.build_anomaly_check). Frames without an
        active anomaly_config are completely unaffected: same batched
        wrinkle_seg.predict_batch() call as before.

        Returns: {orig_idx: wrinkled_check}
        """
        wrinkled_checks: Dict[int, Dict[str, Any]] = {}
        wrinkle_crops_info = []   # legacy path — batched through wrinkle_seg
        anomaly_crops_info = []   # new path — one onnxruntime call per frame

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

            anomaly_config = data.get('anomaly_config') or {}
            if anomaly_config.get('enabled') and anomaly_config.get('onnx_path'):
                anomaly_crops_info.append((orig_idx, crop, anomaly_config))
                continue

            wrinkle_area = data.get('wrinkle_area')              # per-template total threshold
            wrinkle_min_area = data.get('wrinkle_min_area', 0.0)  # per-template per-region min
            wrinkle_max_area = data.get('wrinkle_max_area', 0.0)  # per-template per-region critical
            wrinkle_conf = data.get('wrinkle_conf', 0.25)         # per-recipe conf threshold
            mask_overlap_threshold = data.get('mask_overlap_threshold', 0.6)
            # Extract 'mask' polygons from transformed_bboxes (frame-space coords)
            mask_polygons = [
                np.array(b['points'], dtype=np.float32)
                for b in data['transformed_bboxes']
                if b.get('type') == 'mask' and b.get('points')
            ]
            wrinkle_crops_info.append((
                orig_idx, crop, cx, cy, w, h, angle, crop_offset,
                data['frame_img'], wrinkle_area, wrinkle_min_area, wrinkle_max_area, wrinkle_conf,
                mask_polygons, mask_overlap_threshold
            ))

        # ── Anomaly path — one model call per frame (models can differ per
        # template, so this isn't batchable the way a single shared wrinkle
        # model is; frame counts per trigger are small enough this is fine) ──
        if anomaly_crops_info:
            from .anomaly_inference import build_anomaly_check, predict as anomaly_predict
            for orig_idx, crop, anomaly_config in anomaly_crops_info:
                pred = anomaly_predict(
                    anomaly_config['onnx_path'], crop,
                    image_size=int(anomaly_config.get('image_size', 256)),
                )
                wrinkled_checks[orig_idx] = build_anomaly_check(
                    pred, float(anomaly_config.get('threshold', 0.5)),
                )

        if not wrinkle_crops_info:
            return wrinkled_checks

        # Batch predict — predict_batch only accepts a single conf for the whole batch.
        # Use the min conf across frames as a lower bound; per-frame post-hoc filter is
        # applied inside build_wrinkled_check via conf_threshold.
        # NOTE: tuple index 12 is wrinkle_conf (kept in sync with crops_info layout above).
        batch_min_conf = min(ci[12] for ci in wrinkle_crops_info)
        crops = [ci[1] for ci in wrinkle_crops_info]
        try:
            seg_results, seg_timing = self.wrinkle_seg.predict_batch(
                crops, conf_threshold=batch_min_conf, return_timing=True
            )
            logger.info(
                f"[WrinkleSeg] batch={len(crops)} conf={batch_min_conf:.2f} | "
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

        # Build wrinkled_check per frame (each frame has its own thresholds)
        for j, ci in enumerate(wrinkle_crops_info):
            (orig_idx, _crop, cx, cy, w, h, angle, crop_offset, frame_img,
             wrinkle_area, wrinkle_min_area, wrinkle_max_area, wrinkle_conf,
             mask_polygons, mask_overlap_threshold) = ci
            seg_boxes, seg_masks = seg_results[j]
            wrinkled_checks[orig_idx] = self.wrinkle_seg.build_wrinkled_check(
                seg_boxes=seg_boxes,
                seg_masks=seg_masks,
                cx=cx, cy=cy, w=w, h=h, angle=angle,
                crop_offset=crop_offset,
                frame_shape=frame_img.shape,
                min_area=wrinkle_area,
                min_region_area=wrinkle_min_area,
                max_region_area=wrinkle_max_area,
                conf_threshold=wrinkle_conf,
                mask_polygons=mask_polygons,
                mask_overlap_threshold=mask_overlap_threshold,
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
            (b for b in transformed_bboxes if b.get('type') == 'label'), None
        )
        if product_region is None:
            return None

        # ── Ưu tiên: YOLO detected product box ──────────────────────────────
        # if len(boxes) > 0:
        #     product_poly = np.array(product_region['points'], dtype=np.float32)
        #     for box, score, cls_id in zip(boxes, scores, class_ids):
        #         if self.obb_model.class_names[cls_id] != 'product':
        #             continue
        #         corners = self._obb_to_corners(box)
        #         if self._is_box_inside_polygon(corners, product_poly):
        #             return {'box': box, 'score': float(score), 'source': 'yolo'}

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
            'min_area': self.wrinkle_min_area,
            'wrinkled_boxes': wrinkled_boxes
        }

    # ========== Helper Methods ==========

    def _save_crop_sample(
        self,
        frame_img: np.ndarray,
        transformed_bboxes: List[Dict[str, Any]],
        serial_number: str,
    ):
        if not self.save_crop_samples:
            return
        count = self._crop_sample_counts.get(serial_number, 0)
        if count >= self.crop_sample_max:
            return

        product_region = next(
            (b for b in transformed_bboxes if b.get('type') == 'product'), None
        )
        if product_region is None:
            return

        pts = np.array(product_region['points'], dtype=np.int32)
        h, w = frame_img.shape[:2]
        m = self.crop_sample_margin
        x1 = max(0, int(pts[:, 0].min()) - m)
        y1 = max(0, int(pts[:, 1].min()) - m)
        x2 = min(w, int(pts[:, 0].max()) + m)
        y2 = min(h, int(pts[:, 1].max()) + m)

        crop = frame_img[y1:y2, x1:x2]
        if crop.size == 0:
            return

        try:
            import time, json
            cam_dir = os.path.join(self.crop_sample_dir, serial_number)
            os.makedirs(cam_dir, exist_ok=True)
            ts = int(time.time() * 1000)
            stem = f"crop_{count:04d}_{ts}"
            cv2.imwrite(os.path.join(cam_dir, f"{stem}.jpg"), crop)
            cv2.imwrite(os.path.join(cam_dir, f"{stem}_full.jpg"), frame_img)

            # Lưu label bbox (toạ độ tương đối so với crop offset x1, y1)
            label_region = next(
                (b for b in transformed_bboxes if b.get('type') == 'label'), None
            )
            meta = {'crop_offset': [x1, y1], 'product_points': pts.tolist()}
            if label_region is not None:
                label_pts = np.array(label_region['points'], dtype=np.int32)
                # Toạ độ trong crop space (trừ offset)
                label_pts_crop = label_pts - np.array([x1, y1], dtype=np.int32)
                meta['label_points'] = label_pts_crop.tolist()
                meta['label_points_frame'] = label_region['points']
            json_path = os.path.join(cam_dir, f"{stem}.json")
            with open(json_path, 'w') as f:
                json.dump(meta, f)

            self._crop_sample_counts[serial_number] = count + 1
            logger.debug(f"[{serial_number}] Saved crop sample {count+1}/{self.crop_sample_max}")
        except Exception as e:
            logger.warning(f"[{serial_number}] Failed to save crop sample: {e}")

    def save_template_crop(
        self,
        template_img: np.ndarray,
        bboxes: List[Dict[str, Any]],
        serial_number: str,
    ):
        """Gọi 1 lần khi load template để lưu crop mẫu của template."""
        product_region = next(
            (b for b in bboxes if b.get('type') == 'product'), None
        )
        if product_region is None:
            return
        pts = np.array(product_region['points'], dtype=np.int32)
        h, w = template_img.shape[:2]
        m = self.crop_sample_margin
        x1 = max(0, int(pts[:, 0].min()) - m)
        y1 = max(0, int(pts[:, 1].min()) - m)
        x2 = min(w, int(pts[:, 0].max()) + m)
        y2 = min(h, int(pts[:, 1].max()) + m)
        crop = template_img[y1:y2, x1:x2]
        if crop.size == 0:
            return
        try:
            cam_dir = os.path.join(self.crop_sample_dir, serial_number)
            os.makedirs(cam_dir, exist_ok=True)
            cv2.imwrite(os.path.join(cam_dir, "template.jpg"), crop)
            logger.info(f"[{serial_number}] Saved template crop to {cam_dir}/template.jpg")
        except Exception as e:
            logger.warning(f"[{serial_number}] Failed to save template crop: {e}")

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
        # Only check X axis: box corners must be within polygon's X range.
        # Y axis (height) is ignored.
        poly_x = polygon[:, 0]
        x_min, x_max = float(poly_x.min()), float(poly_x.max())
        for corner in box_corners:
            if corner[0] < x_min or corner[0] > x_max:
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
