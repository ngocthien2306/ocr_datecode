# ─────────────────────────────────────────────────────────────────────────────
# `TemplateConfig` and `CpuMatcher._postprocess_pair` are COPIED VERBATIM from
# ai_services/inference_engine_shared.py (source commit 70885df, sha1
# 07c8934d8b63): TemplateConfig = lines 42-104, _postprocess_pair = 641-850.
#
# That file cannot be imported here: its line 31-33 are
#     import tensorrt as trt
#     import pycuda.driver as cuda
#     import pycuda.autoinit
# which would create a CUDA context in the backend
# (docs/DEBUG_SERVICE_CRASH_RUNBOOK.md §6). Only the TRT `_infer` call is
# replaced — by the CPU ONNX session in onnx_models.py; post-processing
# (score filter, keypoint rescale, RANSAC similarity/homography, confidence
# gate, bbox transform) is the source's own code so the regions agree with
# what the AI service produced.
#
# Re-sync when the source changes, then re-run
#   backend/scripts/check_live_template_parity.py
# ─────────────────────────────────────────────────────────────────────────────
"""SuperPoint template → target matching on CPU (see header)."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .onnx_models import get_sp_lg

logger = logging.getLogger(__name__)


@dataclass
class TemplateConfig:
    """
    Lightweight container for template data.
    Does NOT load TensorRT engine - only stores template info.
    """
    template_path: str
    template_img: np.ndarray
    template_gray: np.ndarray
    template_bbox: Dict[str, Any]
    other_bboxes: List[Dict[str, Any]]
    scale: float = 1.0
    crop_area: Optional[Dict[str, int]] = None

    # Geometric model RANSAC fits between template and target.
    #   'homography' (8 DOF) — default. Correct for labels on a bottle's curved
    #       wall seen off-axis, where real perspective distortion exists.
    #   'similarity' (4 DOF: rotation + uniform scale + translation) — for a
    #       FLAT face shot square-on, such as a bottle cap under an overhead
    #       camera after the OBB step has already normalised its angle. There
    #       the extra 4 DOF of a homography model nothing physical and instead
    #       absorb noise from the cap rim's near-rotationally-symmetric
    #       keypoints, warping mapped bboxes off the text they should frame.
    transform_model: str = 'homography'

    # Original annotations for reference
    annotations: List[Dict[str, Any]] = field(default_factory=list)

    # Cache scale_matrix + inverse — chỉ phụ thuộc `scale`, không đổi giữa
    # các frame. Tính 1 lần khi __post_init__ thay vì mỗi call _postprocess_pair.
    _scale_matrix: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _scale_matrix_inv: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    # Cache pre-formatted bbox points arrays — không đổi giữa các frame.
    # _template_pts_arr: shape (-1, 1, 2) float32 sẵn sàng cho perspectiveTransform
    # _other_pts_arrs:   list các array tương ứng với template.other_bboxes
    _template_pts_arr: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _other_pts_arrs: Optional[List[np.ndarray]] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        inv_s = 1.0 / float(self.scale)
        self._scale_matrix = np.array([
            [inv_s, 0.0,   0.0],
            [0.0,   inv_s, 0.0],
            [0.0,   0.0,   1.0],
        ], dtype=np.float64)
        # inverse is just element-wise reciprocal cho ma trận diagonal
        self._scale_matrix_inv = np.array([
            [self.scale, 0.0,        0.0],
            [0.0,        self.scale, 0.0],
            [0.0,        0.0,        1.0],
        ], dtype=np.float64)
        # Pre-format bbox points cho cv2.perspectiveTransform (shape -1,1,2 float32).
        # Tránh tạo numpy array mỗi frame.
        if self.template_bbox and 'points' in self.template_bbox:
            self._template_pts_arr = np.array(
                self.template_bbox['points'], dtype=np.float32
            ).reshape(-1, 1, 2)
        self._other_pts_arrs = [
            np.array(bbox['points'], dtype=np.float32).reshape(-1, 1, 2)
            if bbox.get('points') else None
            for bbox in (self.other_bboxes or [])
        ]



class CpuMatcher:
    """Stand-in for SuperPointMatcherTRTOptimized, CPU-only, one pair at a time.

    Provides exactly the attributes the copied `_postprocess_pair` reads:
    `input_shape` (engine H, W) and `num_keypoints`.
    """

    def __init__(self):
        self._sp = get_sp_lg()
        self.input_shape = self._sp.input_shape
        self.num_keypoints = 1024  # sp_lg_pipeline: 1024 keypoints / image

    def match(
        self,
        target_img: np.ndarray,
        template: 'TemplateConfig',
        score_threshold: float = 0.3,
        ransac_threshold: float = 5.0,
        min_confidence: float = 0.20,
    ) -> Dict[str, Any]:
        """Same arguments/defaults the pipeline passes to match_batch
        (single_camera.py: score_threshold=0.3, ransac_threshold=5.0,
        min_confidence=camera.matching_conf)."""
        if template.scale != 1.0:
            target_scaled = cv2.resize(target_img, None, fx=template.scale, fy=template.scale)
        else:
            target_scaled = target_img
        target_gray = (cv2.cvtColor(target_scaled, cv2.COLOR_BGR2GRAY)
                       if target_scaled.ndim == 3 else target_scaled)
        kpts, matches, mscores, infer_ms = self._sp.run_pair(template.template_gray, target_gray)
        result = self._postprocess_pair(
            kpts_raw=kpts, matches_raw=matches, mscores_raw=mscores,
            template_idx=0, target_idx=1, template=template,
            target_img_full=target_img,
            score_threshold=score_threshold, ransac_threshold=ransac_threshold,
            min_confidence=min_confidence,
        )
        result['infer_ms'] = infer_ms
        result.pop('target_img', None)  # never ship the image back through JSON
        return result

    def _postprocess_pair(
        self,
        kpts_raw: np.ndarray,
        matches_raw: np.ndarray,
        mscores_raw: np.ndarray,
        template_idx: int,
        target_idx: int,
        template: TemplateConfig,
        target_img_full: np.ndarray,
        score_threshold: float,
        ransac_threshold: float,
        min_confidence: float = 0.20,
    ) -> Dict:
        """Post-process a single template-target pair from batch outputs"""

        # `keypoints` is the one output TensorRT can shape ahead of time, so
        # _infer already hands it back as (batch, K, 2) -- use it as is. K
        # differs per engine (512 for the small pipeline, 1024 for the 9-layer
        # sp_lg_pipeline one), hence self.num_keypoints from the engine rather
        # than a hardcoded 512; reshaping 1024-keypoint output as 512 splices
        # image N's second half onto image N+1 and blows up the match indices.
        try:
            if kpts_raw.ndim == 3:
                kpts = kpts_raw
            else:
                num_kpts = self.num_keypoints
                if not num_kpts:
                    raise ValueError("engine reports no static keypoint count")
                kpts_flat = kpts_raw.ravel()
                usable = (kpts_flat.size // (num_kpts * 2)) * num_kpts * 2
                kpts = kpts_flat[:usable].reshape(-1, num_kpts, 2)
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to reshape keypoints: {e}',
                'homography': None,
                'confidence': 0.0,
                'inliers': 0,
                'total_matches': 0,
                'transformed_bboxes': [],
                'target_img': target_img_full
            }

        kpts0 = kpts[template_idx].astype(np.float32)
        kpts1 = kpts[target_idx].astype(np.float32)

        # matches/mscores now come back exactly sized from the output allocator
        # (they used to be a 100MB buffer whose tail was uninitialized, hence
        # the old "count scores > 1e-6 to guess the length" heuristic).
        try:
            matches = matches_raw.reshape(-1, 3).astype(np.int32)
            mscores = mscores_raw.ravel()
            num_matches = len(matches)
            if len(mscores) != num_matches:
                raise ValueError(f"matches/mscores length mismatch: {num_matches} vs {len(mscores)}")
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to reshape matches: {e}',
                'homography': None,
                'confidence': 0.0,
                'inliers': 0,
                'total_matches': 0,
                'transformed_bboxes': [],
                'target_img': target_img_full
            }

        # Filter by batch index
        pair_idx = template_idx // 2
        batch_mask = matches[:, 0] == pair_idx
        batch_matches = matches[batch_mask]
        batch_mscores = mscores[batch_mask]

        # Filter by score
        valid_mask = batch_mscores > score_threshold
        valid_matches = batch_matches[valid_mask]

        if len(valid_matches) < 10:
            return {
                'success': False,
                'error': f'Too few matches: {len(valid_matches)}',
                'homography': None,
                'confidence': 0.0,
                'inliers': 0,
                'total_matches': len(valid_matches),
                'transformed_bboxes': [],
                'target_img': target_img_full
            }

        # Fancy indexing (advanced indexing với integer array) đã trả về copy
        # độc lập theo numpy spec — bỏ .copy() dư. In-place multiply bên dưới
        # an toàn, không ảnh hưởng kpts0/kpts1.
        m_kpts0 = kpts0[valid_matches[:, 1]]
        m_kpts1 = kpts1[valid_matches[:, 2]]

        # Scale keypoints back
        template_h, template_w = template.template_gray.shape[:2]
        engine_h, engine_w = self.input_shape[2:]
        template_scale = (template_w / engine_w, template_h / engine_h)

        # Target dimensions chỉ cần tính ratio — không cần resize + cvtColor
        # lại (đã thực hiện trong preprocess step trước đó). Trước đây gọi
        # cv2.resize + cv2.cvtColor trên ảnh 1200×1600 chỉ để lấy shape → phí
        # ~5-10ms / pair.
        target_h, target_w = target_img_full.shape[:2]
        if template.scale != 1.0:
            target_h = int(round(target_h * template.scale))
            target_w = int(round(target_w * template.scale))
        target_scale = (target_w / engine_w, target_h / engine_h)

        m_kpts0[:, 0] *= template_scale[0]
        m_kpts0[:, 1] *= template_scale[1]
        m_kpts1[:, 0] *= target_scale[0]
        m_kpts1[:, 1] *= target_scale[1]

        # RANSAC homography — cap maxIters=500 (default 2000) để bound
        # worst-case khi matches toàn nhiễu (FAIL frame). RANSAC trên noisy
        # matches iterate gần hết maxIters → tốn 20-30ms. Cap 500 giảm xuống
        # ~5-8ms mà vẫn đủ cho ca thật (good matches converge sau ~50-200 iter).
        if getattr(template, 'transform_model', 'homography') == 'similarity':
            # 4 DOF instead of 8. estimateAffinePartial2D returns a 2x3
            # [[a,-b,tx],[b,a,ty]]; promote it to 3x3 so every downstream
            # perspectiveTransform keeps working unchanged.
            M, mask = cv2.estimateAffinePartial2D(
                m_kpts0, m_kpts1, method=cv2.RANSAC,
                ransacReprojThreshold=ransac_threshold, maxIters=500,
            )
            H = None if M is None else np.vstack([M, [0.0, 0.0, 1.0]])
        else:
            H, mask = cv2.findHomography(
                m_kpts0, m_kpts1, cv2.RANSAC, ransac_threshold,
                maxIters=500,
            )

        if H is None:
            return {
                'success': False,
                'error': 'Homography estimation failed',
                'homography': None,
                'confidence': 0.0,
                'inliers': 0,
                'total_matches': len(valid_matches),
                'transformed_bboxes': [],
                'target_img': target_img_full
            }

        inliers = np.sum(mask)
        confidence = inliers / len(m_kpts0)

        # ── Matching-confidence gate ────────────────────────────────────────
        # If RANSAC inlier ratio is too low, the homography is unreliable and
        # transformed bboxes can degenerate (huge / negative / outside frame),
        # making downstream OCR + char-ML waste 1-3s per frame on garbage.
        # Bail out early so the pipeline marks FAIL without running verify.
        if confidence < min_confidence:
            logger.warning(
                f"[MATCH-GATE] Low match confidence: {confidence:.3f} < {min_confidence:.2f} "
                f"(inliers={int(inliers)}, total={len(m_kpts0)}) — skip verify"
            )
            return {
                'success': False,
                'error': f'low_match_conf:{confidence:.3f}<{min_confidence:.2f}',
                'homography': None,
                'confidence': float(confidence),
                'inliers': int(inliers),
                'total_matches': len(valid_matches),
                'transformed_bboxes': [],
                'target_img': target_img_full,
            }

        # Transform bboxes — reuse cached scale_matrix + inverse từ TemplateConfig
        # (không phụ thuộc frame, chỉ phụ thuộc template.scale).
        H_full = template._scale_matrix @ H @ template._scale_matrix_inv

        transformed_bboxes = []

        # Transform template bbox — reuse cached np.array (pre-built in
        # TemplateConfig.__post_init__, không thay đổi giữa frames).
        if template._template_pts_arr is not None:
            template_transformed = cv2.perspectiveTransform(template._template_pts_arr, H_full)
            transformed_bboxes.append({
                'type': 'template',
                'points': template_transformed.reshape(-1, 2).tolist(),
                'conf': template.template_bbox.get('conf', 0.8)
            })

        # Transform other bboxes — dùng cached arrays parallel với other_bboxes
        cached_pts_list = template._other_pts_arrs or []
        for bbox, pts_arr in zip(template.other_bboxes, cached_pts_list):
            if pts_arr is None:
                continue
            pts_transformed = cv2.perspectiveTransform(pts_arr, H_full)
            transformed_bboxes.append({
                'type': bbox['type'],
                'points': pts_transformed.reshape(-1, 2).tolist(),
                'text': bbox.get('text', ''),
                'annotation_index': bbox.get('annotation_index'),
                'conf': bbox.get('conf', 0.8)
            })

        return {
            'success': True,
            'homography': H_full,
            'confidence': float(confidence),
            'inliers': int(inliers),
            'total_matches': len(valid_matches),
            'transformed_bboxes': transformed_bboxes,
            'target_img': target_img_full
        }

