"""
Single Camera Pipeline

Handles inference for single camera scenarios:
- Single camera with single template
- Single camera with multiple templates (batch processing)
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np

from .base import InferencePipelineTemplate, PipelineContext
from .frame_data import (
    build_frame_verification_data,
    get_color_localization_method,
    is_color_check_frame,
)
from ..camera import Camera

logger = logging.getLogger(__name__)




class SingleCameraPipeline(InferencePipelineTemplate):
    """
    Pipeline for single camera inference.

    Supports both single template and multi-template scenarios.
    Uses batch inference when multiple templates are configured.
    """

    def __init__(
        self,
        crop_func=None,
        transform_func=None,
        result_builder=None,
        save_and_encode_func=None,
        encode_display_func=None
    ):
        """
        Initialize SingleCameraPipeline.

        Args:
            crop_func: Function to crop frame by crop_area
            transform_func: Function to transform results to full image coords
            result_builder: InferenceResultBuilder class for building results
            save_and_encode_func: Function to save and encode frames
            encode_display_func: Function to encode frames for display
        """
        self._crop_func = crop_func
        self._transform_func = transform_func
        self._result_builder = result_builder
        self._save_and_encode_func = save_and_encode_func
        self._encode_display_func = encode_display_func

    def prepare(self, context: PipelineContext) -> bool:
        """Validate single camera setup"""
        if not context.cameras:
            logger.warning(f"[Job #{context.job_id}] No cameras provided")
            return False

        if len(context.cameras) > 1:
            logger.warning(
                f"[Job #{context.job_id}] SingleCameraPipeline received "
                f"{len(context.cameras)} cameras, using first only"
            )

        camera = context.cameras[0]
        serial_number = camera.serial_number

        # Check if matcher exists
        if serial_number not in context.camera_matchers:
            logger.error(f"[Job #{context.job_id}] No matcher for camera {serial_number}")
            return False

        # Check if frames exist
        if serial_number not in context.results:
            logger.error(f"[Job #{context.job_id}] No results for camera {serial_number}")
            return False

        context.cameras_to_process = [camera]
        return True

    def preprocess(self, context: PipelineContext) -> Optional[Dict[str, Any]]:
        """Prepare frames for inference"""
        import time as _time
        _t_preproc_start = _time.perf_counter()
        _ms_cap_rotation = 0.0
        _ms_cap_crop = 0.0
        _ms_crop_apply = 0.0
        _n_cap_rotation = 0
        _n_cap_crop = 0
        camera = context.cameras_to_process[0]
        serial_number = camera.serial_number
        frames = context.results[serial_number].get('frames', [])

        matcher_or_list = context.camera_matchers.get(serial_number)
        is_multi_template = isinstance(matcher_or_list, list)

        if is_multi_template:
            matchers = matcher_or_list
            num_templates = len(matchers)
        else:
            matchers = [matcher_or_list]
            num_templates = 1

        # Rotation rotates the FULL frame so SuperPoint sees an upright bottle.
        # Applies to Check_Color cameras whose templates DON'T have a 'product'
        # annotation (i.e. OCR-on-cap mode). Check_Color + product → color check
        # mode → no rotation.
        #
        # Engine selection (recipe-level `cap_rotation_method`):
        #   - 'yolo_obb'      → context.obb_rotation_service (trained TRT engine)
        #   - 'yolo_segment'  → context.cv_rotation_service  (pure CV)
        _first_template_has_product = bool(
            camera.templates
            and any(a.get('type') == 'product' for a in (camera.templates[0].get('annotations') or []))
        )
        _method = getattr(camera, 'cap_rotation_method', 'yolo_obb')
        _rotation_service = (
            context.cv_rotation_service
            if _method == 'yolo_segment'
            else context.obb_rotation_service
        )
        use_obb_rotation = (
            camera.function_type == 'Check_Color'
            and not _first_template_has_product
            and _rotation_service is not None
            and getattr(_rotation_service, 'available', False)
        )
        # Dual rotation: emit BOTH no-flip + flip180 candidates per frame and let
        # SuperPoint match confidence pick the winner downstream. Bypasses the
        # shape-match `_need_flip` heuristic ("BEST" template). Mirrors
        # multi_camera.preprocess dual logic.
        _dual_rotation = (
            use_obb_rotation
            and bool(getattr(camera, 'dual_rotation_check', False))
            and hasattr(_rotation_service, 'rotate_frame_dual')
        )
        frame_rotation_matrices = [None] * len(frames)
        # Alt full-frame per template idx (parallel to `frames`); None when
        # rotation wasn't dual or dual emit failed for that frame.
        alt_full_frames: List[Optional[Any]] = [None] * len(frames)
        # cap_circle (cx, cy, r) derived từ OBB/CV rotation per frame. Dùng
        # downstream cho cap_crop step → bỏ HoughCircles lần 2.
        cap_circles_from_rot: List[Optional[Tuple[float, float, float]]] = [None] * len(frames)

        if use_obb_rotation:
            from ..preprocessing.obb_rotator import transform_crop_area
            rotated_frames = []
            for idx, frame in enumerate(frames):
                frame_tag = f"{serial_number}/frame{idx}"
                # Pass the user-drawn product region down to the rotation step
                # so OBB inference + CV fallback only search inside it (avoid
                # neighbouring bottles in the FOV). matchers parallels frames.
                _rot_crop = (
                    getattr(matchers[idx], 'crop_area', None)
                    if idx < len(matchers) else None
                )
                _t_rot = _time.perf_counter()
                if _dual_rotation:
                    cand_a, cand_b, cap_circle = _rotation_service.rotate_frame_dual(
                        frame, frame_tag=f"{frame_tag}-dual", crop_area=_rot_crop
                    )
                    _ms_cap_rotation += (_time.perf_counter() - _t_rot) * 1000
                    _n_cap_rotation += 1
                    if cand_a is not None and cand_b is not None:
                        rotated_frames.append(cand_a)
                        alt_full_frames[idx] = cand_b
                        frame_rotation_matrices[idx] = None  # cap-only rotation
                        cap_circles_from_rot[idx] = cap_circle
                        logger.info(
                            f"[{serial_number}] dual cap_rotation OK frame{idx} "
                            f"— both candidates emitted"
                        )
                    else:
                        # Dual failed → fall back to single rotate_frame for this frame
                        rotated, M, cap_circle = _rotation_service.rotate_frame(
                            frame, frame_tag=f"{frame_tag}-fallback", crop_area=_rot_crop
                        )
                        rotated_frames.append(rotated)
                        frame_rotation_matrices[idx] = M
                        cap_circles_from_rot[idx] = cap_circle
                        logger.info(
                            f"[{serial_number}] dual cap_rotation fallback to "
                            f"single frame{idx}"
                        )
                else:
                    rotated, M, cap_circle = _rotation_service.rotate_frame(
                        frame, frame_tag=frame_tag, crop_area=_rot_crop
                    )
                    _ms_cap_rotation += (_time.perf_counter() - _t_rot) * 1000
                    _n_cap_rotation += 1
                    rotated_frames.append(rotated)
                    frame_rotation_matrices[idx] = M  # None means rotation failed → original used
                    cap_circles_from_rot[idx] = cap_circle

            if getattr(_rotation_service, 'inverse_transform', False):
                # inverse_transform=True: output dùng ảnh gốc, bbox sẽ được inverse về tọa độ gốc
                # → KHÔNG replace frames trong context (giữ ảnh gốc cho display/OCR output)
                pass
            else:
                # inverse_transform=False: output dùng ảnh đã xoay
                # → replace frames để verify/display dùng ảnh xoay
                # In dual mode, store PRIMARY candidates here; run_inference may
                # swap individual entries to alt if alt wins.
                context.results[serial_number]['frames'] = rotated_frames

            frames = rotated_frames  # superpoint luôn dùng ảnh đã xoay (primary cand)

        # Prepare inputs: crop from (rotated or original) frame.
        #
        # In dual_rotation mode we emit TWO batch entries per template idx:
        # primary (no-flip) first, then alt (flip180). All parallel lists
        # (target_imgs / crop_areas / batch_matchers / is_alt_candidate /
        # template_idx_per_entry) keep the same length.
        target_imgs = []
        crop_areas = []
        batch_matchers: List[Any] = []
        is_alt_candidate: List[bool] = []
        template_idx_per_entry: List[int] = []
        # Map template idx → alt full-frame, for swapping into context.results
        # if the alt candidate wins.
        alt_full_frame_by_idx: Dict[int, Any] = {}

        def _apply_match_erosion(img, idx_tag: str):
            """Horizontal erosion to suppress variable text before SuperPoint."""
            if not getattr(camera, 'match_erosion_enabled', False):
                return img
            kw = getattr(camera, 'match_erosion_kernel_w', 80)
            kh = getattr(camera, 'match_erosion_kernel_h', 1)
            iters = getattr(camera, 'match_erosion_iterations', 1)
            kernel = np.ones((kh, kw), np.uint8)
            eroded = cv2.erode(img, kernel, iterations=iters)
            try:
                from pathlib import Path as _Path
                _dbg = _Path("ocr_inference") / (
                    f"debug_frame_{camera.serial_number}_{idx_tag}_eroded.jpg"
                )
                cv2.imwrite(str(_dbg), eroded)
            except Exception:
                pass
            return eroded

        for idx, (frame, matcher) in enumerate(zip(frames[:num_templates], matchers)):
            crop_area = getattr(matcher, 'crop_area', None)
            # Keep the user-drawn region pristine: cap_crop synthesis below
            # OVERWRITES `crop_area` with the auto-detected cap bbox. We need
            # the original to (a) bound HoughCircles to the product region and
            # (b) clip the resulting cap bbox so it stays inside.
            user_crop_area = crop_area

            # Transform crop_area coords into rotated frame space (if rotation succeeded)
            M = frame_rotation_matrices[idx] if idx < len(frame_rotation_matrices) else None
            if M is not None and crop_area:
                crop_area = transform_crop_area(crop_area, M)

            # ── cap_crop_method active: detect cap per-frame, override crop_area ──
            _cap_crop_method = getattr(matcher, 'cap_crop_method', 'none')
            # Cache cap circle from primary to reuse on alt (cap-only rotation
            # → cap position identical between candidates, skip HoughCircles 2nd time).
            _cached_cap_circle = None
            if _cap_crop_method and _cap_crop_method != 'none':
                _t_cap = _time.perf_counter()
                try:
                    from ..preprocessing.cv_rotator import (
                        detect_cap_circle, apply_cap_crop,
                    )
                    # Reuse cap_circle từ rotation step nếu có — bỏ HoughCircles
                    # lần 2 (tiết kiệm ~55ms). Fallback Hough khi rotation fail —
                    # restrict to user_crop_area so we don't pick up a neighbour.
                    _cached_cap_circle = (
                        cap_circles_from_rot[idx]
                        if idx < len(cap_circles_from_rot) and cap_circles_from_rot[idx] is not None
                        else detect_cap_circle(frame, crop_area=user_crop_area)
                    )
                    cap_result = (
                        apply_cap_crop(frame, _cached_cap_circle, margin_ratio=0.10)
                        if _cached_cap_circle is not None else None
                    )
                    _dt_cap = (_time.perf_counter() - _t_cap) * 1000
                    _ms_cap_crop += _dt_cap
                    _n_cap_crop += 1
                    if cap_result is not None:
                        # Clip cap bbox to user's product crop_area so the
                        # 10% margin doesn't reach outside the labelled region.
                        _fx1, _fy1, _fx2, _fy2 = cap_result[1]
                        if user_crop_area:
                            _fx1 = max(_fx1, int(user_crop_area['x1']))
                            _fy1 = max(_fy1, int(user_crop_area['y1']))
                            _fx2 = min(_fx2, int(user_crop_area['x2']))
                            _fy2 = min(_fy2, int(user_crop_area['y2']))
                            frame_for_inference = frame[_fy1:_fy2, _fx1:_fx2]
                        else:
                            frame_for_inference = cap_result[0]
                        # Synthesize crop_area from clipped cap bbox so
                        # `_transform_func` later maps bboxes back to full
                        # frame coords (otherwise they stay in cap-crop coords
                        # and verifiers crop the wrong region).
                        crop_area = {
                            'x1': int(_fx1), 'y1': int(_fy1),
                            'x2': int(_fx2), 'y2': int(_fy2),
                        }
                        logger.info(
                            f"[{serial_number}] cap_crop ({_cap_crop_method}) "
                            f"in {_dt_cap:.1f}ms: frame={frame.shape[:2]} → "
                            f"crop={frame_for_inference.shape[:2]} "
                            f"bbox=({_fx1},{_fy1},{_fx2},{_fy2}) "
                            f"clipped_to_user={user_crop_area is not None}"
                        )
                    else:
                        logger.warning(
                            f"[{serial_number}] cap_crop FAIL in {_dt_cap:.1f}ms — "
                            f"no cap detected in frame {idx}; falling back"
                        )
                        _t_cr = _time.perf_counter()
                        frame_for_inference = (
                            self._crop_func(frame, crop_area)
                            if self._crop_func and crop_area else frame
                        )
                        _ms_crop_apply += (_time.perf_counter() - _t_cr) * 1000
                except Exception as e:
                    _ms_cap_crop += (_time.perf_counter() - _t_cap) * 1000
                    logger.warning(f"[{serial_number}] cap_crop error: {e}")
                    _t_cr = _time.perf_counter()
                    frame_for_inference = (
                        self._crop_func(frame, crop_area)
                        if self._crop_func and crop_area else frame
                    )
                    _ms_crop_apply += (_time.perf_counter() - _t_cr) * 1000
            elif self._crop_func and crop_area:
                _t_cr = _time.perf_counter()
                frame_for_inference = self._crop_func(frame, crop_area)
                _ms_crop_apply += (_time.perf_counter() - _t_cr) * 1000
            else:
                frame_for_inference = frame

            frame_for_inference = _apply_match_erosion(frame_for_inference, f"t{idx}")

            target_imgs.append(frame_for_inference)
            crop_areas.append(crop_area)
            batch_matchers.append(matcher)
            is_alt_candidate.append(False)
            template_idx_per_entry.append(idx)

            # ── Dual rotation: emit ALT candidate as a separate batch entry ──
            alt_frame_full = alt_full_frames[idx] if idx < len(alt_full_frames) else None
            if alt_frame_full is not None:
                alt_crop_area = crop_area
                alt_frame_for_inference = None
                if _cap_crop_method and _cap_crop_method != 'none':
                    # Reuse cap circle from primary — cap_rotation is cap-only,
                    # so the cap position is identical in primary and alt.
                    _t_alt = _time.perf_counter()
                    try:
                        from ..preprocessing.cv_rotator import apply_cap_crop
                        alt_cap = (
                            apply_cap_crop(alt_frame_full, _cached_cap_circle, margin_ratio=0.10)
                            if _cached_cap_circle is not None else None
                        )
                        _dt_alt = (_time.perf_counter() - _t_alt) * 1000
                        if alt_cap is not None:
                            # Match primary's user_crop_area clipping.
                            _ax1, _ay1, _ax2, _ay2 = alt_cap[1]
                            if user_crop_area:
                                _ax1 = max(_ax1, int(user_crop_area['x1']))
                                _ay1 = max(_ay1, int(user_crop_area['y1']))
                                _ax2 = min(_ax2, int(user_crop_area['x2']))
                                _ay2 = min(_ay2, int(user_crop_area['y2']))
                                alt_frame_for_inference = alt_frame_full[_ay1:_ay2, _ax1:_ax2]
                            else:
                                alt_frame_for_inference = alt_cap[0]
                            alt_crop_area = {
                                'x1': int(_ax1), 'y1': int(_ay1),
                                'x2': int(_ax2), 'y2': int(_ay2),
                            }
                            logger.info(
                                f"[{serial_number}] alt cap_crop (cached cap) in "
                                f"{_dt_alt:.1f}ms — reused HoughCircles result"
                            )
                        else:
                            logger.warning(
                                f"[{serial_number}] dual alt cap_crop FAIL "
                                f"— primary cap not detected, skipping alt"
                            )
                    except Exception as e:
                        logger.warning(f"[{serial_number}] dual alt cap_crop error: {e}")
                elif self._crop_func and crop_area:
                    alt_frame_for_inference = self._crop_func(alt_frame_full, crop_area)
                else:
                    alt_frame_for_inference = alt_frame_full

                if alt_frame_for_inference is not None:
                    alt_frame_for_inference = _apply_match_erosion(
                        alt_frame_for_inference, f"t{idx}_alt"
                    )
                    target_imgs.append(alt_frame_for_inference)
                    crop_areas.append(alt_crop_area)
                    batch_matchers.append(matcher)
                    is_alt_candidate.append(True)
                    template_idx_per_entry.append(idx)
                    alt_full_frame_by_idx[idx] = alt_frame_full

        _dt_total = (_time.perf_counter() - _t_preproc_start) * 1000
        logger.info(
            f"[Job #{context.job_id}] [{serial_number}] preprocess breakdown: "
            f"total={_dt_total:.1f}ms | "
            f"cap_rotation={_ms_cap_rotation:.1f}ms (n={_n_cap_rotation}) | "
            f"cap_crop_detect={_ms_cap_crop:.1f}ms (n={_n_cap_crop}) | "
            f"crop_apply={_ms_crop_apply:.1f}ms"
        )

        return {
            'camera': camera,
            'serial_number': serial_number,
            'frames': frames,
            # `matchers` is parallel to `target_imgs` (length N normally, or 2N
            # in dual mode). run_inference uses it for `templates=` in match_batch.
            'matchers': batch_matchers,
            # `template_matchers` is the original 1-per-template list (length N),
            # kept for code paths that index by template idx (e.g. ColorCheck stub
            # path or shape_outline). Equals `batch_matchers` when not dual.
            'template_matchers': matchers,
            'target_imgs': target_imgs,
            'crop_areas': crop_areas,
            # `rotation_matrices` stays length N (one per template idx), NOT
            # parallel to target_imgs. Used for inverse_transform on the winning
            # candidate. Always None in dual mode (cap-only rotation).
            'rotation_matrices': frame_rotation_matrices,
            'is_alt_candidate': is_alt_candidate,
            'template_idx_per_entry': template_idx_per_entry,
            'alt_full_frame_by_idx': alt_full_frame_by_idx,
            'num_templates': num_templates,
            'is_multi_template': is_multi_template
        }

    def run_inference(
        self,
        context: PipelineContext,
        preprocessed: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Run batch inference"""
        # Sub-stage timings — mirrors MultiCameraPipeline.run_inference so the
        # RUN-INFERENCE line reads the same whichever pipeline ran. Splits the
        # SuperPoint match from the setup and the bbox transform / dual resolve.
        _t_stage = time.perf_counter()
        _sub: Dict[str, float] = {}

        def _lap(name: str) -> None:
            nonlocal _t_stage
            _now = time.perf_counter()
            _sub[name] = (_now - _t_stage) * 1000
            _t_stage = _now

        matchers = preprocessed['matchers']            # parallel to target_imgs
        template_matchers = preprocessed.get('template_matchers', matchers)  # length N
        target_imgs = preprocessed['target_imgs']
        crop_areas = preprocessed['crop_areas']
        rotation_matrices = preprocessed.get('rotation_matrices', [None] * len(target_imgs))
        is_alt_candidate = preprocessed.get(
            'is_alt_candidate', [False] * len(target_imgs)
        )
        template_idx_per_entry = preprocessed.get(
            'template_idx_per_entry', list(range(len(target_imgs)))
        )
        alt_full_frame_by_idx = preprocessed.get('alt_full_frame_by_idx', {})
        num_templates = preprocessed.get('num_templates', len(template_matchers))

        # Map template_idx → primary batch index. Used by stub fast-path to
        # pick the primary entry per template (skipping alt entries).
        primary_k_by_tidx: Dict[int, int] = {}
        for _k, _tidx in enumerate(template_idx_per_entry):
            if not is_alt_candidate[_k] and _tidx not in primary_k_by_tidx:
                primary_k_by_tidx[_tidx] = _k

        try:
            # Per-recipe matching-confidence gate (default 0.20 = 20%)
            camera = context.cameras_to_process[0]
            matching_conf = float(getattr(camera, 'matching_conf', 0.20) or 0.20)

            # If ALL matchers are ColorCheck stubs, skip SuperPoint entirely and
            # synthesize success results from the raw camera templates (denormalized
            # to frame coords). Mirrors multi_camera.run_inference's fast path.
            #
            # Note: stub mode is mutually exclusive with dual_rotation (stub is
            # for Check_Color+product; dual is for Check_Color WITHOUT product),
            # but we iterate by template_idx via primary_k_by_tidx defensively.
            all_color_stub = all(
                getattr(m, 'is_color_check_stub', False) for m in template_matchers
            )
            if all_color_stub:
                logger.info(
                    f"[{camera.serial_number}] Skipping SuperPoint match: all "
                    f"{len(template_matchers)} matchers are ColorCheck stubs"
                )
                transformed_results = []
                for t_idx in range(num_templates):
                    primary_k = primary_k_by_tidx.get(t_idx)
                    frame = (
                        target_imgs[primary_k]
                        if primary_k is not None and primary_k < len(target_imgs)
                        else None
                    )
                    if frame is not None and hasattr(frame, 'shape'):
                        fh, fw = frame.shape[:2]
                    else:
                        fh, fw = 1080, 1920
                    annotations = []
                    if camera.templates and t_idx < len(camera.templates):
                        annotations = camera.templates[t_idx].get('annotations') or []
                    tb: List[Dict[str, Any]] = []
                    for ann_idx, ann in enumerate(annotations):
                        new_ann: Dict[str, Any] = dict(ann)
                        new_ann['annotation_index'] = ann_idx
                        pts = ann.get('points')
                        new_pts: List[List[float]] = []
                        if pts and len(pts) >= 3:
                            for p in pts:
                                if isinstance(p, dict):
                                    new_pts.append([float(p.get('x', 0)) * fw, float(p.get('y', 0)) * fh])
                                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                                    new_pts.append([float(p[0]) * fw, float(p[1]) * fh])
                        elif ann.get('width') and ann.get('height'):
                            x1 = float(ann.get('x', 0)) * fw
                            y1 = float(ann.get('y', 0)) * fh
                            x2 = (float(ann.get('x', 0)) + float(ann['width'])) * fw
                            y2 = (float(ann.get('y', 0)) + float(ann['height'])) * fh
                            new_pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        new_ann['points'] = new_pts
                        tb.append(new_ann)
                    transformed_results.append({
                        'success': True,
                        'confidence': 1.0,
                        'inliers': 0,
                        'total_matches': 0,
                        'transformed_bboxes': tb,
                        'timings': {'method': 'color_check_no_match'},
                    })
                return {
                    'batch_result': None,
                    'transformed_results': transformed_results,
                    'timings': {'total': 0.0}
                }

            _lap('setup')

            # ── shape_outline path: skip SuperPoint, run ECC affine ─────────
            if getattr(camera, 'crop_match_method', 'superpoint') == 'shape_outline':
                from ..matchers.shape_outline import match_shape_outline
                def _to_dict(bb):
                    if bb is None:
                        return None
                    if isinstance(bb, dict):
                        return bb
                    if hasattr(bb, 'to_dict'):
                        return bb.to_dict()
                    return None
                synthetic_results = []
                for idx, matcher in enumerate(matchers):
                    target_img = target_imgs[idx]
                    template_img = getattr(matcher, 'template_img', None)
                    template_bbox = _to_dict(getattr(matcher, 'template_bbox', None))
                    other_bboxes = [
                        _to_dict(b) for b in (getattr(matcher, 'other_bboxes', None) or [])
                        if b is not None
                    ]
                    _tag = (
                        f"{camera.serial_number}_t{template_idx_per_entry[idx]}"
                        f"{'_alt' if is_alt_candidate[idx] else ''}"
                    )
                    if template_img is None or template_bbox is None:
                        synthetic_results.append({
                            'success': False,
                            'error': 'no template data',
                            'transformed_bboxes': [],
                            'confidence': 0.0, 'inliers': 0, 'total_matches': 0,
                            'target_img': target_img,
                        })
                    else:
                        synthetic_results.append(match_shape_outline(
                            template_img=template_img,
                            target_img=target_img,
                            template_bbox=template_bbox,
                            other_bboxes=other_bboxes,
                            serial=_tag,
                        ))
                batch_result = {'success': True, 'results': synthetic_results,
                                 'batch_timings': {'total': 0.0, 'trt_inference': 0.0}}
                _lap('shape_outline')
            else:
                # Batch SuperPoint match (matchers parallel to target_imgs)
                batch_result = matchers[0].match_batch(
                    target_imgs=target_imgs,
                    templates=matchers,
                    score_threshold=0.3,
                    ransac_threshold=5.0,
                    min_confidence=matching_conf,
                )
                # Log breakdown thực tế của match_batch để đo cost post-SuperPoint.
                # batch_timings có: preprocess, concat, trt_inference, postprocess, total.
                _lap('match_batch')
                _bt = batch_result.get('batch_timings', {}) or {}
                if _bt:
                    logger.info(
                        f"[{camera.serial_number}] SuperPoint match_batch breakdown: "
                        f"total={_bt.get('total', 0):.1f}ms "
                        f"| preprocess={_bt.get('preprocess', 0):.1f}ms "
                        f"| concat={_bt.get('concat', 0):.1f}ms "
                        f"| trt={_bt.get('trt_inference', 0):.1f}ms "
                        f"| postprocess={_bt.get('postprocess', 0):.1f}ms "
                        f"(pairs={len(target_imgs)})"
                    )

            if not batch_result.get('success', False):
                logger.error(f"Batch inference failed: {batch_result.get('error')}")
                return None

            # Stash transformed results by (template_idx, is_alt). For each
            # batch entry:
            #   1. crop offset (cropped-rotated → full-rotated coords)
            #   2. inverse rotation (full-rotated → full-original) if enabled
            #
            # Then resolve per template_idx — in dual mode pick higher-confidence
            # winner; in normal mode (no alt), use whichever entry is present.
            serial_number = camera.serial_number
            _stash: Dict[tuple, Any] = {}
            for k, result in enumerate(batch_result['results']):
                crop_area = crop_areas[k]
                if self._transform_func:
                    result = self._transform_func(result, crop_area)
                t_idx = template_idx_per_entry[k] if k < len(template_idx_per_entry) else k
                M = rotation_matrices[t_idx] if t_idx < len(rotation_matrices) else None
                if (M is not None
                        and context.obb_rotation_service
                        and context.obb_rotation_service.inverse_transform):
                    from ..preprocessing.obb_rotator import inverse_transform_bboxes
                    result = inverse_transform_bboxes(result, M)
                is_alt = is_alt_candidate[k] if k < len(is_alt_candidate) else False
                _stash[(t_idx, is_alt)] = result

            transformed_results: List[Dict[str, Any]] = []
            for t_idx in range(num_templates):
                r_primary = _stash.get((t_idx, False))
                r_alt     = _stash.get((t_idx, True))

                if r_primary is None and r_alt is None:
                    # Defensive: no match entries for this template idx
                    transformed_results.append({
                        'success': False,
                        'error': 'no match result',
                        'transformed_bboxes': [],
                        'confidence': 0.0, 'inliers': 0, 'total_matches': 0,
                    })
                    continue

                if r_alt is None:
                    # Non-dual path (or dual emit failed for this idx)
                    transformed_results.append(r_primary)
                    continue

                # Only alt available — promote it, swap frame.
                if r_primary is None:
                    transformed_results.append(r_alt)
                    if t_idx in alt_full_frame_by_idx:
                        _frames = context.results[serial_number].get('frames', [])
                        if t_idx < len(_frames):
                            _frames[t_idx] = alt_full_frame_by_idx[t_idx]
                    continue

                # Both available — keep both for OCR-based winner pick in
                # run_verification. SuperPoint conf is unreliable on caps
                # with low-contrast laser-etched text (cap-rim keypoints
                # dominate, no orientation signal). OCR result on the
                # expected date code is the ground truth.
                c_p = float(r_primary.get('confidence', 0.0) or 0.0)
                c_a = float(r_alt.get('confidence', 0.0) or 0.0)
                combined = dict(r_primary)
                # `success` is logically OR — if EITHER candidate matched,
                # verification should proceed. Without this, primary.success
                # =False would gate-out the dual frame and alt would never
                # get its chance to OCR (Bug 14 in the original draft).
                combined['success'] = bool(
                    r_primary.get('success') or r_alt.get('success')
                )
                combined['_dual_alt_result'] = r_alt
                combined['_dual_alt_frame']  = alt_full_frame_by_idx.get(t_idx)
                combined['_dual_t_idx']      = t_idx
                combined['_dual_sp_primary'] = c_p
                combined['_dual_sp_alt']     = c_a
                transformed_results.append(combined)
                logger.info(
                    f"[{serial_number}] dual_rotation t{t_idx}: deferring "
                    f"winner pick to OCR (sp_primary={c_p:.3f}, "
                    f"sp_alt={c_a:.3f}, primary_match={r_primary.get('success')}, "
                    f"alt_match={r_alt.get('success')})"
                )

            _lap('transform+dual')
            logger.info(
                f"[Job #{context.job_id}] RUN-INFERENCE "
                + " ".join(f"{k}={v:.1f}ms" for k, v in _sub.items() if v >= 0.05)
                + f" | entries={len(target_imgs)} templates={num_templates}"
            )

            return {
                'batch_result': batch_result,
                'transformed_results': transformed_results,
                'timings': batch_result.get('batch_timings', {})
            }

        except Exception as e:
            logger.error(f"Inference error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def verify_results(
        self,
        context: PipelineContext,
        inference_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify text/char/template/product for each frame.

        Char ML verification is BATCHED across all eligible frames in a single
        embedder call to amortize the model's fixed setup cost (≈50-310ms).
        Template + product verify stay per-frame (cheap, independent state).
        """
        camera = context.cameras_to_process[0]
        serial_number = camera.serial_number
        frames = context.results[serial_number].get('frames', [])
        matchers = context.camera_matchers.get(serial_number)
        if not isinstance(matchers, list):
            matchers = [matchers]

        transformed_results = inference_results['transformed_results']

        # Batch product verification (existing optimization)
        product_verification_results = self._batch_verify_products(
            context, camera, frames, transformed_results
        )

        # ── Phase 0: initialize per-frame result skeletons ──
        verified_frames: List[Dict[str, Any]] = []
        for idx, result in enumerate(transformed_results):
            verified_frames.append({
                'frame_idx':            idx,
                'result':               'PASS' if result.get('success') else 'FAIL',
                'confidence':           float(result.get('confidence', 0.0)),
                'inliers':              int(result.get('inliers', 0)),
                'total_matches':        int(result.get('total_matches', 0)),
                'timings':              dict(result.get('timings', {})),
                'transformed_bboxes':   result.get('transformed_bboxes', []),
                'text_verification':    None,
                'char_verification':    None,
                'template_verification': None,
                'product_verification': None,
            })

        # ── Phase 1: collect frames eligible for text/char verification ──
        # For dual_rotation entries we emit TWO verify-data items (primary +
        # alt) under the same logical frame idx; Phase 2a picks the winner
        # based on # matching text annotations, falling back to SuperPoint
        # conf only when match counts tie.
        text_verify_indices: List[int] = []
        text_verify_data: List[Dict[str, Any]] = []
        # t_idx → (primary_data_idx, alt_data_idx) for entries with both
        # candidates. Used by Phase 2a to pick winner via OCR.
        dual_text_pairs: Dict[int, Tuple[int, int]] = {}
        if context.text_verification_service and camera.expected_texts:
            for idx, result in enumerate(transformed_results):
                # For dual mode, `success` is OR'd in run_inference so we
                # don't gate-out alt-only successful cases here.
                if not result.get('success'):
                    continue
                frame_expected_texts = camera.expected_texts.get(idx, {})
                if not frame_expected_texts:
                    continue
                matcher = matchers[idx] if idx < len(matchers) else matchers[0]
                sim_template_img = None
                sim_original_bboxes = None
                if hasattr(matcher, 'template_img') and hasattr(matcher, 'other_bboxes'):
                    sim_template_img = matcher.template_img
                    sim_original_bboxes = matcher.other_bboxes

                primary_di = len(text_verify_data)
                text_verify_indices.append(idx)
                text_verify_data.append({
                    'frame_img':          frames[idx],
                    'transformed_bboxes': verified_frames[idx]['transformed_bboxes'],
                    'expected_texts':     frame_expected_texts,
                    'camera':             camera,
                    'template_img':       sim_template_img,
                    'original_bboxes':    sim_original_bboxes,
                })

                # Dual-rotation: also enqueue the alt candidate.
                alt_result = result.get('_dual_alt_result')
                alt_frame  = result.get('_dual_alt_frame')
                if alt_result is not None and alt_frame is not None:
                    alt_di = len(text_verify_data)
                    text_verify_indices.append(idx)
                    text_verify_data.append({
                        'frame_img':          alt_frame,
                        'transformed_bboxes': alt_result.get('transformed_bboxes', []),
                        'expected_texts':     frame_expected_texts,
                        'camera':             camera,
                        'template_img':       sim_template_img,
                        'original_bboxes':    sim_original_bboxes,
                    })
                    dual_text_pairs[idx] = (primary_di, alt_di)

        # ── Phase 2: SINGLE batched verify across eligible frames ──
        per_frame_ocr_ms = {idx: 0.0 for idx in range(len(transformed_results))}
        # data_idx → True means "loser of dual pair, skip when distributing".
        skip_data_idx: set = set()
        if text_verify_data:
            t_batched_start = time.perf_counter()
            batched_verifs = context.text_verification_service.verify_text_regions_batched_frames(
                text_verify_data
            )
            t_batched_total_ms = (time.perf_counter() - t_batched_start) * 1000
            # Distribute the batched cost evenly across participating frames so per-frame
            # timing logs stay meaningful.
            per_frame_share = t_batched_total_ms / max(1, len(text_verify_data))

            # ── Phase 2a: OCR-based dual_rotation winner pick ──
            # Count text-only matches (char ML can have false positives that
            # add noise). Strict majority wins; on tie, fall back to
            # SuperPoint conf so we don't always default to primary when
            # both candidates fail OCR equally.
            def _count_text(v: Dict[str, Any]) -> int:
                return sum(1 for r in (v.get('text') or {}).get('results', [])
                            if r.get('match', False))
            for t_idx, (p_di, a_di) in dual_text_pairs.items():
                v_p = batched_verifs[p_di] if p_di < len(batched_verifs) else {}
                v_a = batched_verifs[a_di] if a_di < len(batched_verifs) else {}
                n_p = _count_text(v_p)
                n_a = _count_text(v_a)
                tie_breaker = ""
                if n_a > n_p:
                    alt_wins = True
                elif n_p > n_a:
                    alt_wins = False
                else:
                    # Tie — fall back to SuperPoint conf (the signal we'd
                    # otherwise lose by always-defaulting-to-primary).
                    _r = transformed_results[t_idx]
                    sp_p = float(_r.get('_dual_sp_primary', 0.0) or 0.0)
                    sp_a = float(_r.get('_dual_sp_alt', 0.0) or 0.0)
                    alt_wins = sp_a > sp_p
                    tie_breaker = f" tiebreak by sp_conf (p={sp_p:.3f}, a={sp_a:.3f})"

                if alt_wins:
                    # Promote alt's frame + bboxes + match metrics into both
                    # verified_frames AND transformed_results so Phase 3
                    # (template/product verify) + downstream viz use alt.
                    skip_data_idx.add(p_di)
                    _ctx_frames = context.results[serial_number].get('frames', [])
                    if t_idx < len(_ctx_frames):
                        _ctx_frames[t_idx] = text_verify_data[a_di]['frame_img']
                    _r       = transformed_results[t_idx]
                    _alt_res = _r.get('_dual_alt_result') or {}
                    _alt_bboxes = list(_alt_res.get('transformed_bboxes', []))
                    # transformed_results swap
                    _r['transformed_bboxes'] = _alt_bboxes
                    if 'confidence' in _alt_res:
                        _r['confidence']    = _alt_res['confidence']
                    if 'inliers' in _alt_res:
                        _r['inliers']       = _alt_res['inliers']
                    if 'total_matches' in _alt_res:
                        _r['total_matches'] = _alt_res['total_matches']
                    if 'success' in _alt_res:
                        _r['success']       = _alt_res['success']
                    # verified_frames swap (fix Bug 15 — keep display consistent)
                    fr_t = verified_frames[t_idx]
                    fr_t['transformed_bboxes'] = _alt_bboxes
                    fr_t['confidence']    = float(_alt_res.get('confidence', fr_t['confidence']))
                    fr_t['inliers']       = int(_alt_res.get('inliers', fr_t['inliers']))
                    fr_t['total_matches'] = int(_alt_res.get('total_matches', fr_t['total_matches']))
                    fr_t['result']        = 'PASS' if _alt_res.get('success', _r.get('success')) else 'FAIL'
                    logger.info(
                        f"[{serial_number}] dual_rotation t{t_idx} OCR-pick: "
                        f"ALT wins ({n_a} text matches vs primary {n_p}{tie_breaker})"
                    )
                else:
                    skip_data_idx.add(a_di)
                    logger.info(
                        f"[{serial_number}] dual_rotation t{t_idx} OCR-pick: "
                        f"primary wins ({n_p} text matches vs alt {n_a}{tie_breaker})"
                    )

            # ── Phase 2b: distribute verifications back (skip losers) ──
            for data_idx, (idx, verification) in enumerate(zip(text_verify_indices, batched_verifs)):
                if data_idx in skip_data_idx:
                    continue
                fr = verified_frames[idx]
                text_verification = verification.get('text') or {}
                char_verification = verification.get('char') or {}
                fr['text_verification'] = text_verification
                fr['char_verification'] = char_verification
                per_frame_ocr_ms[idx] = per_frame_share

                # Update bboxes with recognized text
                context.text_verification_service.update_bboxes_with_recognized_text(
                    fr['transformed_bboxes'], text_verification
                )

                # Mark failed text bboxes
                for text_result in text_verification.get('results', []):
                    annotation_idx = text_result.get('annotation_idx')
                    if annotation_idx is not None and not text_result.get('match', False):
                        for bbox in fr['transformed_bboxes']:
                            if (bbox.get('type') in ['text', 'datecode'] and
                                    bbox.get('annotation_index') == annotation_idx):
                                bbox['verification_status'] = 'fail'

                # Mark failed char bboxes
                for char_result in char_verification.get('results', []):
                    annotation_idx = char_result.get('annotation_idx')
                    if annotation_idx is not None and not char_result.get('match', False):
                        for bbox in fr['transformed_bboxes']:
                            if (bbox.get('type') == 'char' and
                                    bbox.get('annotation_index') == annotation_idx):
                                bbox['verification_status'] = 'fail'

        # ── Phase 3: per-frame template + product verify + final decision ──
        for idx, result in enumerate(transformed_results):
            frame_result = verified_frames[idx]
            frame = frames[idx]
            matcher = matchers[idx] if idx < len(matchers) else matchers[0]

            # Skip template verification for Check_Color + product templates —
            # no SuperPoint match runs so there's no aligned crop to compare.
            template_annotations_now = (
                camera.templates[idx].get('annotations') or []
                if camera.templates and idx < len(camera.templates) else []
            )
            _is_color_frame_tv = (
                getattr(camera, 'function_type', '') == 'Check_Color'
                and any(a.get('type') == 'product' for a in template_annotations_now)
            )

            # Template verification
            if (not _is_color_frame_tv and
                result.get('success') and
                context.template_verification_service and
                hasattr(matcher, 'template_img') and
                hasattr(matcher, 'template_bbox')):

                template_verification = context.template_verification_service.verify_template_regions(
                    frame_img=frame,
                    template_img=matcher.template_img,
                    transformed_bboxes=frame_result['transformed_bboxes'],
                    original_template_bbox=matcher.template_bbox,
                    camera=camera
                )
                frame_result['template_verification'] = template_verification

                if template_verification and not template_verification.get('match', True):
                    for bbox in frame_result['transformed_bboxes']:
                        if bbox.get('type') == 'template':
                            bbox['verification_status'] = 'fail'

            # Product verification — also receives Check_Color color_check results
            # (routed inside ProductVerificationService.verify_batch).
            if (context.product_verification_service and
                idx < len(product_verification_results)):

                product_verification = product_verification_results[idx]
                frame_result['product_verification'] = product_verification

                if (product_verification and
                        not product_verification.get('skipped', True) and
                        not product_verification.get('match', True)):
                    for bbox in frame_result['transformed_bboxes']:
                        if bbox.get('type') == 'product':
                            bbox['verification_status'] = 'fail'

            # Merge verification timings
            timings = frame_result['timings']
            t_ocr_ms = per_frame_ocr_ms.get(idx, 0.0)
            if t_ocr_ms > 0:
                timings['ocr_ms'] = t_ocr_ms
            template_verif = frame_result.get('template_verification') or {}
            if template_verif.get('timing'):
                timings['template_verification_ms'] = template_verif['timing'].get('total_ms', 0.0)
            product_verif = frame_result.get('product_verification') or {}
            if product_verif.get('timing'):
                timings['product_verification_ms'] = product_verif['timing'].get('total', 0.0)
            timings['total'] = (
                timings.get('total', 0.0)
                + timings.get('ocr_ms', 0.0)
                + timings.get('template_verification_ms', 0.0)
                + timings.get('product_verification_ms', 0.0)
            )

            # Final pass/fail (text AND char AND template AND product)
            text_ok = (frame_result['text_verification'] is None or
                       frame_result['text_verification'].get('all_match', True))
            char_ok = (frame_result['char_verification'] is None or
                       frame_result['char_verification'].get('all_match', True))
            template_ok = (frame_result['template_verification'] is None or
                           frame_result['template_verification'].get('match', True))
            product_ok = (frame_result['product_verification'] is None or
                          frame_result['product_verification'].get('skipped', True) or
                          frame_result['product_verification'].get('match', True))

            # Color cameras: pass/fail decided ONLY by color check (product_ok),
            # not SuperPoint match success. Override the initial FAIL from match.
            template_annotations = (
                camera.templates[idx].get('annotations') or []
                if camera.templates and idx < len(camera.templates) else []
            )
            is_color_frame = (
                getattr(camera, 'function_type', '') == 'Check_Color'
                and any(a.get('type') == 'product' for a in template_annotations)
            )
            if is_color_frame:
                frame_result['result'] = 'PASS' if product_ok else 'FAIL'
            elif not (text_ok and char_ok and template_ok and product_ok):
                frame_result['result'] = 'FAIL'

        # Determine overall result
        overall = 'PASS'
        for fr in verified_frames:
            if fr['result'] in ['FAIL', 'ERROR']:
                overall = fr['result']
                break

        return {
            'serial_number': serial_number,
            'frames': verified_frames,
            'overall_result': overall
        }

    def _batch_verify_products(
        self,
        context: PipelineContext,
        camera: 'Camera',
        frames: List[np.ndarray],
        transformed_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Batch verify products for all frames.

        Args:
            context: Pipeline context
            camera: Camera object
            frames: List of frame images
            transformed_results: List of transformed inference results

        Returns:
            List of product verification results (one per frame)
        """
        if not context.product_verification_service:
            return [None] * len(frames)

        # Prepare frames data for batch verification.
        # Single-camera: template_idx == the frame index, one template per frame.
        frames_data = []
        for idx, result in enumerate(transformed_results):
            template = (
                camera.templates[idx]
                if camera.templates and idx < len(camera.templates)
                else None
            )
            has_frame = (
                result.get('success') or is_color_check_frame(camera, template)
            ) and idx < len(frames)
            frames_data.append(build_frame_verification_data(
                camera, template, idx,
                frame_img=frames[idx] if has_frame else None,
                transformed_bboxes=result.get('transformed_bboxes', []) if has_frame else [],
            ))

        # Filter valid frames for batch processing
        valid_frames = [
            data for data in frames_data
            if data['frame_img'] is not None
        ]

        if not valid_frames:
            return [None] * len(frames)

        # Check how many frames need verification (have both product and label regions)
        frames_needing_verification = [
            data for data in valid_frames
            if context.product_verification_service.should_verify_frame(data['transformed_bboxes'])
        ]

        logger.debug(
            f"[{camera.serial_number}] Product verification: "
            f"{len(frames_needing_verification)}/{len(valid_frames)} frames need verification"
        )

        # Batch verify
        try:
            import time
            t_start = time.perf_counter()
            verification_results = context.product_verification_service.verify_batch(frames_data)
            t_elapsed = (time.perf_counter() - t_start) * 1000

            # Log timing details if available (find first valid timing)
            timing = None
            for result in verification_results:
                if result and result.get('timing') and result['timing'].get('total', 0) > 0:
                    timing = result['timing']
                    break

            if timing:
                logger.info(
                    f"[{camera.serial_number}] Product verification complete: "
                    f"total={timing.get('total', 0):.1f}ms, "
                    f"yolo={timing.get('yolo_inference', 0):.1f}ms, "
                    f"frames={timing.get('frames_checked', 0)}/{timing.get('frames_total', 0)}"
                )

            return verification_results
        except Exception as e:
            logger.error(f"[{camera.serial_number}] Batch product verification failed: {e}")
            import traceback
            traceback.print_exc()
            # Return error results
            return [{
                'match': False,
                'skipped': False,
                'error': f'Batch verification failed: {str(e)}',
                'timing': {'total': 0.0}
            } for _ in frames]

    def postprocess(
        self,
        context: PipelineContext,
        verified_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build final result structure"""
        serial_number = verified_results['serial_number']
        overall = verified_results['overall_result']

        # Store in context
        context.camera_inference_results[serial_number] = verified_results
        context.overall_pass_fail = overall

        # Build result using builder if available
        if self._result_builder:
            return self._result_builder.from_cameras(
                cameras=context.cameras,
                results=context.results,
                camera_inference_results=context.camera_inference_results,
                overall_pass_fail=overall,
                camera_matchers=context.camera_matchers,
                save_and_encode_func=self._save_and_encode_func,
                encode_display_func=self._encode_display_func,
                statistics=context.statistics
            )

        # Fallback to simple result
        camera = context.cameras[0]
        return {
            "recipe_id": camera.recipe_id,
            "recipe_name": camera.recipe_name,
            "product_pass_fail": overall,
            "camera_results": [{
                "serial_number": serial_number,
                "frames": verified_results['frames']
            }],
            "metadata": {
                "total_cameras": 1,
                "total_frames": len(verified_results['frames'])
            }
        }
