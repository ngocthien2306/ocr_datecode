"""
Realign a fresh camera frame against an existing template — on CPU, in the
backend — so "Get Live Template" gets an image and regions that belong to
the SAME frame.

Why this exists
---------------
The realtime screen used to pair a frame grabbed at click time
(/cameras/{sn}/frames/latest) with `detected_regions` from the last inference
result. Two things made that wrong:

  1. Different moments: the regions were computed on an earlier frame; the
     bottle had moved/rotated since.
  2. Different coordinate systems (cap cameras): the AI pipeline rotates the
     cap disc before matching and, with inverse_transform=False, its regions
     live in the ROTATED frame — while shared memory hands out the raw,
     unrotated frame. Even the same moment would not line up.

So this module re-runs the AI service's own template-alignment steps on the
fresh frame and returns the rotated frame together with regions computed on
it, in the same shape the AI result uses (`detected_regions`).

Fidelity
--------
The steps and their order mirror, one for one:
  * template side — ai_services/camera_management/matchers/factory.py
      create_matcher(): parse annotations → cap-crop via the rotation
      service (fallback HoughCircles) with the annotations-inside guard →
      crop_area crop → erosion → JPEG round-trip (the factory writes each
      stage to disk and TemplateConfig.from_json re-reads it) → OBB text_box
      on the FULL template → narrow to the date-code window.
  * target side — ai_services/camera_management/pipeline/single_camera.py
      preprocess(): rotation gate (Check_Color and no 'product' on the first
      template) → dual / single cap rotation → cap-crop with cached circle,
      clipped to the user crop_area → narrow to text window
      (pipeline/base.py _narrow_target_to_text) → erosion;
      run_inference(): match (score 0.3, RANSAC 5.0, min_conf =
      matching_conf) → add crop offset (inference_handler
      _transform_results_to_full_image) → dual: mirror the matched side's
      regions through the cap centre when only one candidate matched.

One deliberate difference: the pipeline defers the dual winner to OCR
(SuperPoint confidence carries no orientation signal on laser-etched caps).
OCR on GPU is not available here, so the winner is the candidate whose
recovered rotation is closest to 0° — the template is upright, so the
correctly-oriented candidate matches with a near-identity rotation while the
flipped one, if it matches at all, comes back near 180°. BOTH candidates are
returned so the operator can flip if the pick is wrong.

CPU only; see onnx_models.py for why that is non-negotiable in this process.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .annotation_parser import AnnotationParser
from .cv_rotator import CVRotationService, apply_cap_crop, detect_cap_circle
from .matcher import CpuMatcher, TemplateConfig
from .obb_rotator import (
    OBBRotationService,
    mirror_bboxes_through_cap,
    text_window,
    transform_crop_area,
)
from .onnx_models import OBB_MODEL_PATH

logger = logging.getLogger(__name__)


class RealignError(Exception):
    """Realignment cannot run for this template / frame (bad input)."""


@dataclass
class RealignOptions:
    """Recipe/camera knobs the AI pipeline reads, with the same defaults as
    ai_services/camera_management/camera.py."""
    function_type: str = ""
    cap_rotation_method: str = "yolo_obb"      # camera.py:319
    cap_crop_method: str = "none"              # camera.py:322
    dual_rotation_check: bool = False          # camera.py:331
    matching_conf: float = 0.20                # camera.py:296
    match_erosion_enabled: bool = False        # camera.py:298
    match_erosion_kernel_w: int = 80
    match_erosion_kernel_h: int = 1
    match_erosion_iterations: int = 1
    # Annotations of the camera's FIRST template: the pipeline gates rotation
    # on whether *templates[0]* has a 'product' annotation (single_camera.py
    # 119-122), not on the template being matched.
    first_template_annotations: Optional[List[Dict[str, Any]]] = None


# ─── Rotation services (singletons) ──────────────────────────────────────────

_svc_lock = threading.Lock()
_obb_svc: Optional[OBBRotationService] = None
_cv_svc: Optional[CVRotationService] = None


def _services() -> Tuple[OBBRotationService, CVRotationService]:
    """Same construction as inference_handler.py:517-528 — conf 0.25,
    inverse_transform=False — with the ONNX CPU model instead of the engine."""
    global _obb_svc, _cv_svc
    if _obb_svc is None or _cv_svc is None:
        with _svc_lock:
            if _obb_svc is None:
                _obb_svc = OBBRotationService(
                    engine_path=str(OBB_MODEL_PATH),
                    conf_threshold=0.25,
                    inverse_transform=False,
                )
            if _cv_svc is None:
                _cv_svc = CVRotationService(inverse_transform=False)
    return _obb_svc, _cv_svc


# ─── Small helpers ───────────────────────────────────────────────────────────

def _jpeg_roundtrip(img: np.ndarray) -> np.ndarray:
    """The factory cv2.imwrite()s each template stage and TemplateConfig
    re-reads it with cv2.imread — default JPEG quality 95. Mimic it so the
    template SuperPoint sees carries the same compression artefacts."""
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        return img
    out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out if out is not None else img


def _erode(img: np.ndarray, opts: RealignOptions) -> np.ndarray:
    kernel = np.ones((int(opts.match_erosion_kernel_h), int(opts.match_erosion_kernel_w)), np.uint8)
    return cv2.erode(img, kernel, iterations=int(opts.match_erosion_iterations))


def _recovered_rotation_deg(H: Optional[np.ndarray]) -> Optional[float]:
    """Rotation angle of the template→target transform, in (-180, 180]."""
    if H is None:
        return None
    return float(math.degrees(math.atan2(float(H[1, 0]), float(H[0, 0]))))


def _wrap180(a: float) -> float:
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def _offset_bboxes(bboxes: List[Dict[str, Any]], crop_area: Optional[Dict[str, int]]):
    """inference_handler._transform_results_to_full_image: add the crop's
    (x1, y1) back so regions land in full-frame coordinates."""
    if not crop_area:
        return bboxes
    ox, oy = crop_area["x1"], crop_area["y1"]
    for bb in bboxes:
        if bb.get("points"):
            bb["points"] = [[p[0] + ox, p[1] + oy] for p in bb["points"]]
    return bboxes


# ─── Template side (factory.create_matcher) ──────────────────────────────────

@dataclass
class _PreparedTemplate:
    tc: TemplateConfig
    user_crop_area: Optional[Dict[str, int]]
    cap_crop_method: str            # 'none' unless the template cap-crop succeeded
    text_window: Optional[Tuple[int, int, int, int]]
    notes: List[str]


def _detect_cap_crop_for_template(template_img, cap_crop_method, user_crop_area,
                                  obb_svc, cv_svc, notes):
    """factory._detect_cap_crop_for_template (lines 133-208)."""
    crop_dict = None
    if user_crop_area is not None and user_crop_area.is_valid():
        crop_dict = {'x1': user_crop_area.x1, 'y1': user_crop_area.y1,
                     'x2': user_crop_area.x2, 'y2': user_crop_area.y2}
    svc = cv_svc if cap_crop_method == 'yolo_segment' else obb_svc
    cap_circle = None
    if svc is not None and getattr(svc, 'available', False):
        try:
            _, _, cap_circle = svc.rotate_frame(template_img, frame_tag="live/template",
                                                crop_area=crop_dict)
        except Exception as e:
            notes.append(f"template: rotation-service cap detect failed ({e})")
    if cap_circle is None:
        cap_circle = detect_cap_circle(template_img, crop_area=crop_dict)
        if cap_circle is not None:
            notes.append("template: cap via HoughCircles fallback")
    if cap_circle is None:
        return None
    return apply_cap_crop(template_img, cap_circle, margin_ratio=0.10)


def _annotations_outside_crop(template_bbox, other_bboxes, crop_bbox):
    """factory._annotations_outside_crop (lines 273-294)."""
    cx1, cy1, cx2, cy2 = crop_bbox
    outside = []
    for bb in ([template_bbox] + list(other_bboxes or [])):
        if bb is None:
            continue
        xs = [p[0] for p in bb.points]
        ys = [p[1] for p in bb.points]
        if min(xs) < cx1 or min(ys) < cy1 or max(xs) > cx2 or max(ys) > cy2:
            outside.append(bb.type + (f":{bb.text!r}" if getattr(bb, 'text', '') else ""))
    return outside


def _narrow_template_to_text(tc: TemplateConfig, text_box):
    """factory._narrow_template_to_text (lines 210-270)."""
    img = tc.template_img
    h, w = img.shape[:2]
    cap = (w / 2.0, h / 2.0, (min(w, h) / 2.0) / 1.10)
    win = text_window(cap, text_box, img.shape)
    if win is None:
        return None
    x1, y1, x2, y2 = win
    narrowed = img[y1:y2, x1:x2]
    if narrowed.size == 0:
        return None

    def _shift(bb):
        if not isinstance(bb, dict) or not bb.get('points'):
            return bb
        nb = dict(bb)
        nb['points'] = [[p[0] - x1, p[1] - y1] for p in bb['points']]
        return nb

    tc.template_img = narrowed
    tc.template_gray = cv2.cvtColor(narrowed, cv2.COLOR_BGR2GRAY) if narrowed.ndim == 3 else narrowed
    tc.template_bbox = _shift(tc.template_bbox)
    tc.other_bboxes = [_shift(b) for b in (tc.other_bboxes or [])]
    tc.__post_init__()
    return win


def prepare_template(template_img: np.ndarray, annotations: List[Dict[str, Any]],
                     opts: RealignOptions, use_rotation: bool) -> _PreparedTemplate:
    obb_svc, cv_svc = _services()
    notes: List[str] = []
    img_h, img_w = template_img.shape[:2]
    full_template_img = template_img

    template_bbox, other_bboxes, crop_area = AnnotationParser.parse_annotations(
        annotations=annotations, img_width=img_w, img_height=img_h,
        serial_number="live", template_idx=0,
    )
    if not template_bbox:
        raise RealignError("template has no 'template' annotation — nothing to align on")

    user_crop_area_obj = crop_area
    cap_crop_method = (opts.cap_crop_method or 'none')
    cap_crop_bbox = None
    if cap_crop_method != 'none':
        res = _detect_cap_crop_for_template(template_img, cap_crop_method, user_crop_area_obj,
                                            obb_svc, cv_svc, notes)
        if res is None:
            notes.append("template: cap_crop requested but no cap found → plain crop_area")
        else:
            cap_crop_img, (cx1, cy1, cx2, cy2) = res
            outside = _annotations_outside_crop(template_bbox, other_bboxes, (cx1, cy1, cx2, cy2))
            if outside:
                notes.append(f"template: cap_crop dropped, annotations outside it: {outside}")
            else:
                cap_crop_bbox = (cx1, cy1, cx2, cy2)
                template_bbox = template_bbox.offset(cx1, cy1)
                other_bboxes = [ob.offset(cx1, cy1) for ob in (other_bboxes or [])]
                template_img = _jpeg_roundtrip(cap_crop_img)
                crop_area = None

    if crop_area and crop_area.is_valid():
        template_img = _jpeg_roundtrip(template_img[crop_area.y1:crop_area.y2,
                                                    crop_area.x1:crop_area.x2])
        template_bbox = template_bbox.offset(crop_area.x1, crop_area.y1)
        other_bboxes = [b.offset(crop_area.x1, crop_area.y1) for b in other_bboxes]

    if opts.match_erosion_enabled:
        template_img = _jpeg_roundtrip(_erode(template_img, opts))

    # TemplateConfig.from_json: template = the 'template' bbox, others = the
    # rest (crop_area was never written to the JSON).
    tc = TemplateConfig(
        template_path="",
        template_img=template_img,
        template_gray=cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY),
        template_bbox=template_bbox.to_dict(),
        other_bboxes=[b.to_dict() for b in other_bboxes],
        scale=1.0,
    )
    tc.transform_model = 'similarity' if use_rotation else 'homography'

    tw = None
    if obb_svc is not None and obb_svc.available:
        tb = obb_svc.detect_text_box(full_template_img)
        if tb is not None and cap_crop_bbox:
            tb = (tb[0] - cap_crop_bbox[0], tb[1] - cap_crop_bbox[1], tb[2], tb[3], tb[4])
        has_product = any(b.get('type') == 'product' for b in tc.other_bboxes)
        if tb is not None and cap_crop_bbox and not has_product:
            tw = _narrow_template_to_text(tc, tb)
            if tw is None:
                notes.append("template: text window could not be built")

    return _PreparedTemplate(
        tc=tc,
        user_crop_area=(user_crop_area_obj.to_dict() if user_crop_area_obj else None),
        cap_crop_method=(cap_crop_method if cap_crop_bbox else 'none'),
        text_window=tw,
        notes=notes,
    )


# ─── Target side (single_camera.preprocess + run_inference) ──────────────────

def _narrow_target_to_text(obb_svc, frame, cap_circle, cap_bbox):
    """pipeline/base.py _narrow_target_to_text (lines 257-300)."""
    if obb_svc is None or not obb_svc.available or cap_circle is None:
        return None
    tb = obb_svc.detect_text_box(frame)
    if tb is None:
        return None
    win = text_window(cap_circle, tb, frame.shape)
    if win is None:
        return None
    x1, y1, x2, y2 = win
    cx1, cy1, cx2, cy2 = cap_bbox
    x1, y1 = max(x1, int(cx1)), max(y1, int(cy1))
    x2, y2 = min(x2, int(cx2)), min(y2, int(cy2))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    sub = frame[y1:y2, x1:x2]
    if sub.size == 0:
        return None
    return sub, {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)}


def _prepare_target(frame, tpl: _PreparedTemplate, cap_circle, M, opts, obb_svc,
                    notes, tag, cached_cap=None):
    """One candidate's (frame_for_inference, crop_area), single_camera 262-366.
    Returns (img, crop_area, cap_circle_used)."""
    crop_area = tpl.user_crop_area
    user_crop_area = crop_area
    if M is not None and crop_area:
        crop_area = transform_crop_area(crop_area, M)

    def _plain_crop(img, ca):
        return img[ca["y1"]:ca["y2"], ca["x1"]:ca["x2"]] if ca else img

    if tpl.cap_crop_method and tpl.cap_crop_method != 'none':
        circle = cached_cap if cached_cap is not None else (
            cap_circle if cap_circle is not None
            else detect_cap_circle(frame, crop_area=user_crop_area))
        cap_result = apply_cap_crop(frame, circle, margin_ratio=0.10) if circle is not None else None
        if cap_result is not None:
            fx1, fy1, fx2, fy2 = cap_result[1]
            if user_crop_area:
                fx1 = max(fx1, int(user_crop_area['x1'])); fy1 = max(fy1, int(user_crop_area['y1']))
                fx2 = min(fx2, int(user_crop_area['x2'])); fy2 = min(fy2, int(user_crop_area['y2']))
                img = frame[fy1:fy2, fx1:fx2]
            else:
                img = cap_result[0]
            crop_area = {'x1': int(fx1), 'y1': int(fy1), 'x2': int(fx2), 'y2': int(fy2)}
            if tpl.text_window is not None:
                tw = _narrow_target_to_text(obb_svc, frame, circle, (fx1, fy1, fx2, fy2))
                if tw is not None:
                    img, crop_area = tw
                else:
                    notes.append(f"{tag}: target text window not found → whole cap crop")
        else:
            notes.append(f"{tag}: no cap found on frame → plain crop_area")
            img = _plain_crop(frame, crop_area)
        used = circle
    else:
        img = _plain_crop(frame, crop_area)
        used = cap_circle

    if opts.match_erosion_enabled:
        img = _erode(img, opts)
    return img, crop_area, used


def _to_region_list(bboxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Plain-JSON regions in the AI result's `detected_regions` shape."""
    out = []
    for bb in bboxes or []:
        if not bb.get("points"):
            continue
        r = {
            "type": bb.get("type"),
            "points": [[float(p[0]), float(p[1])] for p in bb["points"]],
            "conf": float(bb.get("conf", 0.8)),
        }
        if bb.get("annotation_index") is not None:
            r["annotation_index"] = int(bb["annotation_index"])
        if bb.get("text"):
            r["text"] = bb["text"]
        out.append(r)
    return out


def realign(frame_bgr: np.ndarray, template_img: np.ndarray,
            annotations: List[Dict[str, Any]], opts: RealignOptions) -> Dict[str, Any]:
    """Run the alignment. Returns a JSON-serialisable dict (images are numpy
    arrays under `candidates[i]['frame']`; the endpoint encodes them)."""
    t_start = time.perf_counter()
    if frame_bgr is None or frame_bgr.size == 0:
        raise RealignError("empty frame")
    if template_img is None or template_img.size == 0:
        raise RealignError("empty template image")

    obb_svc, cv_svc = _services()
    notes: List[str] = []

    first_anns = opts.first_template_annotations
    if first_anns is None:
        first_anns = annotations
    first_has_product = any(a.get('type') == 'product' for a in (first_anns or []))

    if opts.function_type == 'Check_Color' and first_has_product:
        # factory fast path: Check_Color + product → colour-check stub, no
        # SuperPoint alignment at all in the AI service. Nothing to realign.
        return {"mode": "not_applicable",
                "reason": "Check_Color camera with a 'product' annotation uses the "
                          "colour-check path, which has no template alignment",
                "candidates": [], "chosen": None, "notes": notes,
                "elapsed_ms": (time.perf_counter() - t_start) * 1000}

    rot_svc = cv_svc if opts.cap_rotation_method == 'yolo_segment' else obb_svc
    use_rotation = (opts.function_type == 'Check_Color' and not first_has_product
                    and rot_svc is not None and getattr(rot_svc, 'available', False))

    tpl = prepare_template(template_img, annotations, opts, use_rotation)
    notes.extend(tpl.notes)

    # ── rotation (single_camera 112-222) ──
    primary, alt, cap_circle, M = frame_bgr, None, None, None
    rotation_mode = "none"
    if use_rotation:
        if opts.dual_rotation_check and hasattr(rot_svc, 'rotate_frame_dual'):
            a, b, cap_circle = rot_svc.rotate_frame_dual(frame_bgr, frame_tag="live-dual",
                                                         crop_area=tpl.user_crop_area)
            if a is not None and b is not None:
                primary, alt, rotation_mode = a, b, "dual"
            else:
                primary, M, cap_circle = rot_svc.rotate_frame(frame_bgr, frame_tag="live-fallback",
                                                              crop_area=tpl.user_crop_area)
                rotation_mode = "single(dual-fallback)"
        else:
            primary, M, cap_circle = rot_svc.rotate_frame(frame_bgr, frame_tag="live",
                                                          crop_area=tpl.user_crop_area)
            rotation_mode = "single"
        if cap_circle is None:
            notes.append("rotation: cap/text not detected — frame used unrotated")

    matcher = CpuMatcher()
    cands: List[Dict[str, Any]] = []
    cached_cap = None
    for is_alt, full in ((False, primary), (True, alt)):
        if full is None:
            continue
        tag = "alt(flip180)" if is_alt else "primary"
        img, crop_area, used_cap = _prepare_target(
            full, tpl, cap_circle, M, opts, obb_svc, notes, tag,
            cached_cap=(cached_cap if is_alt else None))
        if not is_alt:
            cached_cap = used_cap  # cap-only rotation: same cap position for alt
        res = matcher.match(img, tpl.tc, score_threshold=0.3, ransac_threshold=5.0,
                            min_confidence=float(opts.matching_conf or 0.20))
        H = res.pop("homography", None)
        bboxes = _offset_bboxes(res.get("transformed_bboxes") or [], crop_area)
        cands.append({
            "is_alt": is_alt,
            "label": "flip180" if is_alt else "no_flip",
            "frame": full,
            "success": bool(res.get("success")),
            "error": res.get("error"),
            "confidence": float(res.get("confidence") or 0.0),
            "inliers": int(res.get("inliers") or 0),
            "total_matches": int(res.get("total_matches") or 0),
            "rotation_deg": _recovered_rotation_deg(H) if res.get("success") else None,
            "mirrored": False,
            "bboxes": bboxes,
            "match_input_hw": list(img.shape[:2]),
            "infer_ms": float(res.get("infer_ms") or 0.0),
        })

    # ── dual: mirror the matched side into the unmatched one (single_camera 725-760) ──
    if len(cands) == 2 and cap_circle is not None:
        p, a = cands
        if bool(p["bboxes"]) != bool(a["bboxes"]):
            src, dst = (p, a) if p["bboxes"] else (a, p)
            dst["bboxes"] = mirror_bboxes_through_cap(src["bboxes"], cap_circle)
            for k in ("success", "confidence", "inliers", "total_matches"):
                dst[k] = src[k]
            # Its own match failed; the regions it now carries are the
            # source's, so its old "Too few matches" no longer describes it.
            dst["error"] = None
            dst["rotation_deg"] = (_wrap180(src["rotation_deg"] + 180.0)
                                   if src["rotation_deg"] is not None else None)
            dst["mirrored"] = True

    # ── winner (see module docstring: rotation-closest-to-0, not OCR) ──
    ok = [c for c in cands if c["success"] and c["bboxes"]]
    chosen = None
    if ok:
        chosen = min(ok, key=lambda c: (abs(c["rotation_deg"] or 0.0), -c["confidence"]))
    for c in cands:
        c["detected_regions"] = _to_region_list(c.pop("bboxes"))
        c["chosen"] = c is chosen

    return {
        "mode": "cap_rotation" if use_rotation else "plain",
        "rotation_mode": rotation_mode,
        "transform_model": tpl.tc.transform_model,
        "cap_crop": tpl.cap_crop_method != 'none',
        "text_window": tpl.text_window is not None,
        "cap_circle": list(cap_circle) if cap_circle is not None else None,
        "candidates": cands,
        "chosen": (cands.index(chosen) if chosen is not None else None),
        "notes": notes,
        "elapsed_ms": (time.perf_counter() - t_start) * 1000,
    }
