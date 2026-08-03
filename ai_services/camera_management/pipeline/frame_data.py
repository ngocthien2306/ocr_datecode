"""
Shared per-frame plumbing for the single- and multi-camera pipelines.

Both pipelines feed ProductVerificationService the same dict of per-frame
settings, assembled from the template config plus a few camera-level
attributes. That assembly used to be copy-pasted three times (twice in
single_camera for the has-frame / no-frame branches, once in multi_camera),
which is how the two branches in single_camera drifted apart — the no-frame
one silently stopped carrying center_offset_threshold_left/right. Building it
in one place keeps every consumer seeing the same key set, and means adding a
field (anomaly_config was the last one) is a one-line change instead of three.
"""

from typing import Any, Dict, List, Optional


def get_color_localization_method(template: Optional[Dict[str, Any]]) -> str:
    """Which localization the Check_Color path should use for this template.

    Reads the template-level override first, then the nested color_config,
    falling back to the legacy image-proc detector.
    """
    color_cfg = (template or {}).get('color_config') or {}
    method = (
        (template or {}).get('color_localization_method')
        or color_cfg.get('localization_method')
        or 'image_proc'
    )
    return str(method).strip().lower()


def bbox_to_dict(bbox: Any) -> Optional[Dict[str, Any]]:
    """Normalize a matcher bbox (dict, object with .to_dict(), or None) to a
    plain dict — matchers return different shapes depending on backend."""
    if bbox is None:
        return None
    if isinstance(bbox, dict):
        return bbox
    if hasattr(bbox, 'to_dict'):
        return bbox.to_dict()
    return None


def build_frame_verification_data(
    camera: Any,
    template: Optional[Dict[str, Any]],
    template_idx: int,
    frame_img: Any = None,
    transformed_bboxes: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Assemble one frame's entry for ProductVerificationService.

    `frame_img=None` marks a frame that won't be verified (no match, or no
    image); it still gets a full entry so the result list stays index-aligned
    with the input frames.
    """
    # "no template" and "empty template dict" are NOT the same thing here:
    # center_offset_threshold stays None only when the frame has no template at
    # all, and falls back to 50.0 for a template that simply omits the key. Test
    # by identity, not truthiness — an empty dict is falsy and would otherwise
    # collapse the two cases.
    has_template = template is not None
    template = template or {}
    return {
        'frame_img': frame_img,
        'transformed_bboxes': transformed_bboxes or [],
        'camera': camera,
        'template_idx': template_idx,

        'color_config': template.get('color_config'),
        'color_localization_method': get_color_localization_method(template),

        'center_offset_threshold': template.get('center_offset_threshold', 50.0) if has_template else None,
        'center_offset_threshold_left': template.get('center_offset_threshold_left', 50.0),
        'center_offset_threshold_right': template.get('center_offset_threshold_right', 50.0),
        'center_offset_unit': template.get('center_offset_unit', 'px') or 'px',

        'wrinkle_area': template.get('wrinkle_area', None),
        'wrinkle_min_area': template.get('wrinkle_min_area', 0.0) or 0.0,
        'wrinkle_max_area': template.get('wrinkle_max_area', 0.0) or 0.0,
        # Camera-level, not per-template.
        'wrinkle_conf': getattr(camera, 'wrinkle_conf', 0.25),
        'wrinkle_show_when_pass': getattr(camera, 'wrinkle_show_when_pass', True),
        'mask_overlap_threshold': getattr(camera, 'mask_overlap_threshold', 0.6),

        # When set + enabled, product_verifier's _batch_wrinkle_check routes
        # this template's label-defect check to anomaly_inference.py instead
        # of WrinkledSegmenterTRT.
        'anomaly_config': template.get('anomaly_config', None),
    }


def is_color_check_frame(camera: Any, template: Optional[Dict[str, Any]]) -> bool:
    """Colour-check frames must still run when SuperPoint found no match —
    but only on the legacy image-proc localization path, which doesn't need
    the match to locate the bottle."""
    return (
        getattr(camera, 'function_type', '') == 'Check_Color'
        and (template or {}).get('color_config') is not None
        and get_color_localization_method(template) != 'superpoint'
    )
