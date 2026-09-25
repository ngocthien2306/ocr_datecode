"""
Wall-anchored region placement for Check_Color cameras (edge_regions mode).

Why this exists
---------------
On the label cameras the bottle stands upright in a fixed camera, so its two
walls are straight vertical lines. SuperPoint, however, keys mostly on the
cap — a nearly rotationally-symmetric object — and the 8-DOF homography it
returns is free to shear. Every region derived through that homography
(template outline, product, label, ...) inherits the shear, and the error
grows with distance from the cap. Measured on a full day of colour FAILs:
73% carried a product polygon tilted >3° (up to 24.5°) while the bottle
itself stood straight; the sheared box sat on the glass neck instead of the
label and failed good bottles.

Role split applied here
-----------------------
    SuperPoint  → only places the two edge SEARCH regions near the cap
                  (that is what it is good at: coarse, shift-tolerant)
    edge detect → xL, xR — the wall positions, measured directly on the image
    wall frame  → vertical by construction; width = xR - xL
    every other annotation → rebuilt from its template-side fractions inside
                  that wall frame: x as a fraction of wall-to-wall width, y
                  as an offset from the cap-axis height in the same unit

So position, scale and uprightness come from physical structure; the
homography contributes nothing to region SHAPE any more. This runs once in
run_inference, so visualisation, the colour verifier and the product
verifier all see the same upright regions.

Every source is checked against the template: the product region drawn on the
template is a physical reference (fixed camera-to-bottle distance), so a
rebuilt product that comes out >15% bigger or smaller than drawn means the
walls it was built from were noise, however confident the detector was.

Fallback order when the walls can't be read (or fail that size check):
  1. the sharpness-based bottle detector (image_proc), run on the FULL frame —
     independent of SuperPoint, so it still works when the match is garbage
     (e.g. bottle turned to show the back label); same size check applies;
  2. the template's drawn coordinates copied as-is — triggering is regular
     and the bottle stops centred, so they are usually within a few px, and
     unlike anything derived from a bad match they cannot be skewed.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .image_proc_detector import EdgeParams, detect_cap_axis_from_regions

logger = logging.getLogger(__name__)

# Annotation kinds that define the wall frame (or are pure crop hints) and
# therefore must NOT be rebuilt from it.
_SKIP_TYPES = {"edge_left", "edge_right", "crop_area"}

_REF_CACHE: Dict[Tuple[str, int], Optional[Dict[str, float]]] = {}

# The product region drawn on the template is a PHYSICAL reference: the
# camera-to-bottle distance is fixed, so the same product must come out the
# same size in every frame. A rebuilt region noticeably bigger or smaller
# than the drawn one means the walls (or the bottle box) it was built from
# were noise — even if the detector reported them confidently.
DEFAULT_SIZE_TOL = 0.15


def _size_ok(scale: float, tol: float) -> bool:
    return (1.0 - tol) <= scale <= (1.0 + tol)


def template_identity_bboxes(
    bboxes: List[Dict[str, Any]], anns: List[Dict[str, Any]], fw: int, fh: int,
) -> List[Dict[str, Any]]:
    """Last resort: copy the template's own annotation coordinates onto the
    target frame. Triggering is regular and the bottle stops centred, so the
    drawn positions are usually within a few px of where the bottle actually
    is — and unlike anything derived from a bad match, they can't be skewed."""
    tmpl_by_type: Dict[str, List[np.ndarray]] = {}
    for a in anns:
        t = a.get("type")
        if t in _SKIP_TYPES:
            continue
        poly = annotation_polygon_px(a, fw, fh)
        if poly is not None:
            tmpl_by_type.setdefault(t, []).append(poly)
    out: List[Dict[str, Any]] = []
    counters: Dict[str, int] = {}
    for b in bboxes:
        t = b.get("type")
        if t in _SKIP_TYPES or not b.get("points"):
            out.append(b)
            continue
        k = counters.get(t, 0)
        counters[t] = k + 1
        polys = tmpl_by_type.get(t) or []
        if k >= len(polys):
            out.append(b)
            continue
        nb = dict(b)
        nb["points"] = polys[k].tolist()
        nb["anchor"] = "template"
        out.append(nb)
    return out


def load_cap_edge_ref(serial: str, template_idx: int = 0) -> Optional[Dict[str, float]]:
    """Template cap-edge reference written by MatcherFactory at build time
    (where the wall sits inside each hand-drawn edge region). Cached, misses
    included, so a template without the file costs no disk hit per frame."""
    key = (serial, template_idx)
    if key in _REF_CACHE:
        return _REF_CACHE[key]
    ref: Optional[Dict[str, float]] = None
    try:
        home = os.environ.get("HOME", "")
        suffix = f"_t{template_idx}" if template_idx > 0 else ""
        p = (Path(home) / "Source" / "ocr_datecode" / "crop_samples"
             / serial / f"template{suffix}_cap_edges.json")
        if p.exists():
            ref = json.loads(p.read_text())
    except Exception as e:
        logger.error(f"[{serial}] failed to load cap-edge reference: {e}")
    _REF_CACHE[key] = ref
    return ref


def annotation_polygon_px(ann: Dict[str, Any], fw: int, fh: int) -> Optional[np.ndarray]:
    """Recipe annotation → Nx2 float32 polygon in frame pixels.

    Recipes store normalized [0, 1] coords, either as a `points` list (dicts
    or pairs) or as a rect `x/y/width/height` — the edge regions use the
    rect form, product/template use points."""
    pts = ann.get("points")
    if pts and len(pts) >= 3:
        out = []
        for p in pts:
            if isinstance(p, dict):
                out.append([float(p.get("x", 0)) * fw, float(p.get("y", 0)) * fh])
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                out.append([float(p[0]) * fw, float(p[1]) * fh])
        if len(out) >= 3:
            return np.asarray(out, np.float32)
    x, y, w, h = ann.get("x"), ann.get("y"), ann.get("width"), ann.get("height")
    if x is not None and y is not None and w and h:
        x1, y1 = float(x) * fw, float(y) * fh
        x2, y2 = (float(x) + float(w)) * fw, (float(y) + float(h)) * fh
        return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    return None


def _edge_quads(bboxes: List[Dict[str, Any]]):
    qL = qR = None
    for b in bboxes:
        if not b.get("points") or len(b["points"]) < 4:
            continue
        if b.get("type") == "edge_left":
            qL = b["points"]
        elif b.get("type") == "edge_right":
            qR = b["points"]
    return qL, qR


def template_wall_frame(
    anns: List[Dict[str, Any]], ref: Dict[str, float], fw: int, fh: int,
) -> Optional[Dict[str, float]]:
    """Where the walls and the cap axis sit on the TEMPLATE, in frame pixels.

    frac_L/frac_R (from the cap-edge reference) say where the wall lies inside
    each hand-drawn edge region. Returns None if the template has no edge
    regions or they are degenerate."""
    tqL = tqR = None
    for a in anns:
        if a.get("type") == "edge_left":
            tqL = annotation_polygon_px(a, fw, fh)
        elif a.get("type") == "edge_right":
            tqR = annotation_polygon_px(a, fw, fh)
    if tqL is None or tqR is None:
        return None
    txL = float(tqL[:, 0].min()) + float(ref.get("frac_L", 0.5)) * float(tqL[:, 0].max() - tqL[:, 0].min())
    txR = float(tqR[:, 0].min()) + float(ref.get("frac_R", 0.5)) * float(tqR[:, 0].max() - tqR[:, 0].min())
    if txR - txL < 10.0:
        return None
    both = np.vstack([tqL, tqR])
    return {
        "xL": txL, "xR": txR, "w": txR - txL,
        "yC": float(both[:, 1].mean()),     # cap-axis height
        "yTop": float(both[:, 1].min()),    # ≈ cap top
    }


def _rebuild_in_wall_frame(
    bboxes: List[Dict[str, Any]],
    anns: List[Dict[str, Any]],
    tf: Dict[str, float],
    dxL: float, dxR: float, d_yc: float,
    fw: int, fh: int,
    anchor_tag: str,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Map every non-edge template annotation through the wall frame.

    Template polygons are grouped by type in order, so the k-th matched region
    of a type maps to the k-th template annotation of that type. Regions with
    no template counterpart, or landing (almost) entirely off-frame, are left
    as they came in — this path must never make a region worse."""
    d_w = dxR - dxL
    tmpl_by_type: Dict[str, List[np.ndarray]] = {}
    for a in anns:
        t = a.get("type")
        if t in _SKIP_TYPES:
            continue
        poly = annotation_polygon_px(a, fw, fh)
        if poly is not None:
            tmpl_by_type.setdefault(t, []).append(poly)

    out: List[Dict[str, Any]] = []
    counters: Dict[str, int] = {}
    n_rebuilt = n_kept = 0
    for b in bboxes:
        t = b.get("type")
        if t in _SKIP_TYPES or not b.get("points"):
            out.append(b)
            continue
        k = counters.get(t, 0)
        counters[t] = k + 1
        polys = tmpl_by_type.get(t) or []
        if k >= len(polys):
            out.append(b)
            n_kept += 1
            continue
        prod = polys[k]
        fx = (prod[:, 0] - tf["xL"]) / tf["w"]
        fy = (prod[:, 1] - tf["yC"]) / tf["w"]
        pts = np.stack([dxL + fx * d_w, d_yc + fy * d_w], axis=1)
        full_area = float(max(1.0, (pts[:, 0].max() - pts[:, 0].min())
                                    * (pts[:, 1].max() - pts[:, 1].min())))
        pts[:, 0] = np.clip(pts[:, 0], 0.0, fw - 1.0)
        pts[:, 1] = np.clip(pts[:, 1], 0.0, fh - 1.0)
        vis_area = float((pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min()))
        if vis_area < 0.30 * full_area:
            out.append(b)
            n_kept += 1
            continue
        nb = dict(b)
        nb["points"] = pts.tolist()
        nb["anchor"] = anchor_tag
        out.append(nb)
        n_rebuilt += 1
    return out, n_rebuilt, n_kept


def deskew_bboxes(bboxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Last-resort fallback: keep each region's centre and size but drop the
    shear/rotation the homography put on it. Position may still be off;
    shape no longer is."""
    out = []
    for b in bboxes:
        pts = b.get("points")
        if b.get("type") in _SKIP_TYPES or not pts or len(pts) != 4:
            out.append(b)
            continue
        p = np.asarray(pts, np.float64)
        cx, cy = float(p[:, 0].mean()), float(p[:, 1].mean())
        # Sides 0-1 / 2-3 are the drawn width, 1-2 / 3-0 the drawn height —
        # keep that orientation as-is. (Swapping to make w >= h turned the tall
        # product box into a wide one; measured 570x401 for a 410x600 region.)
        w = 0.5 * (np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[3] - p[2]))
        h = 0.5 * (np.linalg.norm(p[2] - p[1]) + np.linalg.norm(p[0] - p[3]))
        nb = dict(b)
        nb["points"] = [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                        [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]]
        nb["anchor"] = "deskew"
        out.append(nb)
    return out


def anchor_regions_on_walls(
    frame: np.ndarray,
    transformed_bboxes: List[Dict[str, Any]],
    template: Dict[str, Any],
    serial: str,
    template_idx: int = 0,
    size_tol: float = DEFAULT_SIZE_TOL,
) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Rebuild every non-edge region from the walls detected inside the
    SuperPoint-placed edge regions. Returns (new_bboxes, info); new_bboxes is
    None when the walls could not be used, with info['reason'] saying why.
    """
    anns = (template or {}).get("annotations") or []
    if not anns:
        return None, {"reason": "template has no annotations"}
    qL, qR = _edge_quads(transformed_bboxes)
    if qL is None or qR is None:
        return None, {"reason": "no edge_left/edge_right region in match"}
    ref = load_cap_edge_ref(serial, template_idx)
    if not ref:
        return None, {"reason": "no cap-edge reference"}

    color_cfg = (template or {}).get("color_config") or {}
    params = EdgeParams.from_config(color_cfg.get("edge_config") or {})
    axis = detect_cap_axis_from_regions(
        frame, qL, qR, ref_frac=ref, params=params, serial_number=serial,
    )
    if not axis or "left_line" not in axis:
        return None, {"reason": "wall edge not found"}

    fh, fw = frame.shape[:2]
    tf = template_wall_frame(anns, ref, fw, fh)
    if tf is None:
        return None, {"reason": "template lacks usable edge regions"}

    # Frame-side wall frame — forced to the physical model: walls vertical,
    # each contributes one x; the axis height is the mean y of the detected
    # line endpoints.
    lp = np.vstack([np.asarray(p, np.float32).reshape(-1)[:2] for p in axis["left_line"]])
    rp = np.vstack([np.asarray(p, np.float32).reshape(-1)[:2] for p in axis["right_line"]])
    dxL, dxR = float(lp[:, 0].mean()), float(rp[:, 0].mean())
    d_yc = float(np.vstack([lp, rp])[:, 1].mean())
    scale = (dxR - dxL) / tf["w"]
    if not _size_ok(scale, size_tol):
        # Walls found, but the product they imply is the wrong size → noise.
        return None, {"reason": f"product size {scale:.2f}x template (tol ±{size_tol:.0%})",
                      "scale": scale}

    out, n_rebuilt, n_kept = _rebuild_in_wall_frame(
        transformed_bboxes, anns, tf, dxL, dxR, d_yc, fw, fh, "walls",
    )
    y_top = float(min(np.asarray(qL, np.float32)[:, 1].min(), np.asarray(qR, np.float32)[:, 1].min()))
    return out, {
        "source": "walls", "xL": dxL, "xR": dxR, "yC": d_yc, "scale": scale,
        "n_rebuilt": n_rebuilt, "n_kept": n_kept,
        "lines": {
            "left_line": [[dxL, y_top], [dxL, float(fh - 1)]],
            "right_line": [[dxR, y_top], [dxR, float(fh - 1)]],
        },
    }


def anchor_regions_on_bottle_box(
    frame: np.ndarray,
    transformed_bboxes: List[Dict[str, Any]],
    template: Dict[str, Any],
    serial: str,
    bottle_box: Dict[str, Any],
    template_idx: int = 0,
    size_tol: float = DEFAULT_SIZE_TOL,
) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Same wall-frame rebuild, but with the walls taken from the sharpness-based
    bottle detector's bbox (frame coords, `box` = cx, cy, w, h, angle). That
    detector doesn't depend on SuperPoint, so this still places regions when
    the match is garbage. Cap-axis height is derived from the bbox top: on the
    template the axis sits (yC - yTop) below the cap top, scaled by the
    detected/template width ratio.
    """
    anns = (template or {}).get("annotations") or []
    if not anns or not bottle_box or not bottle_box.get("box"):
        return None, {"reason": "no bottle box"}
    ref = load_cap_edge_ref(serial, template_idx)
    if not ref:
        return None, {"reason": "no cap-edge reference"}
    fh, fw = frame.shape[:2]
    tf = template_wall_frame(anns, ref, fw, fh)
    if tf is None:
        return None, {"reason": "template lacks usable edge regions"}

    cx, cy, bw, bh = (float(v) for v in bottle_box["box"][:4])
    if bw < 10.0 or bh < 10.0:
        return None, {"reason": "bottle box degenerate"}
    dxL, dxR = cx - bw / 2.0, cx + bw / 2.0
    y_top = cy - bh / 2.0
    scale = bw / tf["w"]
    if not _size_ok(scale, size_tol):
        return None, {"reason": f"product size {scale:.2f}x template (tol ±{size_tol:.0%})",
                      "scale": scale}
    d_yc = y_top + (tf["yC"] - tf["yTop"]) * scale

    out, n_rebuilt, n_kept = _rebuild_in_wall_frame(
        transformed_bboxes, anns, tf, dxL, dxR, d_yc, fw, fh, "bottle_box",
    )
    return out, {
        "source": "bottle_box", "xL": dxL, "xR": dxR, "yC": d_yc, "scale": scale,
        "n_rebuilt": n_rebuilt, "n_kept": n_kept,
        "lines": {
            "left_line": [[dxL, y_top], [dxL, float(fh - 1)]],
            "right_line": [[dxR, y_top], [dxR, float(fh - 1)]],
        },
    }
