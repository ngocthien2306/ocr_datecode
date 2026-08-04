"""
Synthetic label-defect drawing for the Studio tab.

Ported from the standalone wrinkle-limit study script so the interactive tool
and any offline POD sweep produce the same marks. The two properties that make
these marks behave like real defects rather than pasted graphics:

  * `delta` is measured against the LOCAL background under the stroke, not
    against absolute grey. A +25 mark on a dark area and on a bright area are
    then equally visible, which is what makes a Δ value comparable across
    positions and across label designs.
  * `soft` / `wrinkle` blur the stroke's alpha. A hard-edged line has a step
    gradient no real crease produces, and gradient-sensitive models react to
    that edge rather than to the mark itself.

`wrinkle` draws a parallel highlight + shadow pair (a crease reflects light on
one side and shades the other) instead of a single line — closest to a real
wrinkle, and the reason a wrinkle mark can be caught at a lower Δ than a plain
line of the same width.

Coordinates arrive in the source image's own pixel space; the caller is
responsible for scaling from whatever the UI displayed.
"""
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np

EDGES = ("hard", "soft", "wrinkle", "bubble")
POLARITIES = {"dark": -1, "bright": +1}

# A patch this uniform counts as blank label surface. Also from the reference
# study: printed text and graphics push the local std well above this, so it
# doubles as a "don't draw on the artwork" test.
FLAT_PATCH_STD = 14.0
FLAT_PATCH_HALF = 16          # half-size of the patch sampled when testing flatness


def _local_background(gray: np.ndarray, x: int, y: int, r: int = 10) -> float:
    h, w = gray.shape[:2]
    y0, y1 = max(0, y - r), min(h, y + r)
    x0, x1 = max(0, x - r), min(w, x + r)
    patch = gray[y0:y1, x0:x1]
    return float(patch.mean()) if patch.size else float(gray.mean())


def _blend(img_rgb: np.ndarray, alpha: np.ndarray, target: float) -> np.ndarray:
    a3 = np.repeat(alpha[..., None], 3, axis=2)
    out = img_rgb.astype(np.float32) * (1 - a3) + target * a3
    return out.clip(0, 255).astype(np.uint8)


def _stroke_alpha(shape: Tuple[int, int], pts: np.ndarray, width: int, edge: str) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterize a polyline. Returns (alpha, hard_footprint).

    The footprint is taken BEFORE blurring: it marks where the defect actually
    is, so overlap with the model's predicted mask is measured against the mark
    itself rather than against its blur halo.
    """
    alpha = np.zeros(shape, np.float32)
    closed = False
    cv.polylines(alpha, [pts.astype(np.int32)], closed, 1.0, max(1, int(width)), lineType=cv.LINE_AA)
    footprint = alpha > 0.3
    if edge in ("soft", "wrinkle"):
        k = max(3, int(width) * 2 + 1)
        if k % 2 == 0:
            k += 1
        alpha = cv.GaussianBlur(alpha, (k, k), sigmaX=max(0.1, width * 0.8))
    return alpha, footprint


def _curve_points(pts: np.ndarray, curvature: float, n: int = 24) -> np.ndarray:
    """Bow a 2-point stroke into an arc. Multi-point strokes (freehand) are
    already curved by the user and pass through untouched."""
    if len(pts) != 2 or abs(curvature) < 1e-3:
        return pts
    (x0, y0), (x1, y1) = pts[0], pts[1]
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < 1:
        return pts
    ux, uy = dx / length, dy / length
    px, py = -uy, ux  # perpendicular
    out = []
    for t in np.linspace(0.0, 1.0, n):
        off = curvature * length * math.sin(t * math.pi)
        out.append([x0 + dx * t + off * px, y0 + dy * t + off * py])
    return np.array(out, dtype=np.float32)


def _offset(pts: np.ndarray, dist: float) -> np.ndarray:
    """Shift a polyline sideways by `dist` along its local normal."""
    out = np.empty_like(pts, dtype=np.float32)
    n = len(pts)
    for i in range(n):
        j0, j1 = max(0, i - 1), min(n - 1, i + 1)
        dx, dy = pts[j1][0] - pts[j0][0], pts[j1][1] - pts[j0][1]
        norm = math.hypot(dx, dy) or 1.0
        out[i] = [pts[i][0] - dy / norm * dist, pts[i][1] + dx / norm * dist]
    return out


def sample_flat_position(
    img_bgr: np.ndarray,
    rng: np.random.Generator,
    tries: int = 40,
) -> Tuple[int, int]:
    """Pick a spot on blank label surface, avoiding printed text and graphics.

    Ported from the reference detection-limit study. A mark drawn across
    printed text is not a controlled stimulus — the model may react to the
    disturbed text rather than to the mark, so Δ stops meaning anything. This
    rejects positions whose surroundings vary too much and falls back to the
    last candidate if the label has no flat area at all.
    """
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    cx = cy = 0
    for _ in range(tries):
        cx = int(rng.integers(int(w * 0.18), max(int(w * 0.82), int(w * 0.18) + 1)))
        cy = int(rng.integers(int(h * 0.15), max(int(h * 0.85), int(h * 0.15) + 1)))
        patch = gray[
            max(0, cy - FLAT_PATCH_HALF):cy + FLAT_PATCH_HALF,
            max(0, cx - FLAT_PATCH_HALF):cx + FLAT_PATCH_HALF,
        ]
        if patch.size and patch.std() < FLAT_PATCH_STD:
            return cx, cy
    return cx, cy


def auto_stroke(
    img_bgr: np.ndarray,
    rng: np.random.Generator,
    width: int,
    delta: float,
    edge: str,
    polarity: str = "dark",
    length_frac: float = 0.18,
) -> Dict[str, Any]:
    """Build one randomly placed/oriented stroke spec for batch generation.

    Length is a fraction of the image's short side rather than a fixed pixel
    count, so the same grid produces comparable marks on differently sized
    label crops.
    """
    h, w = img_bgr.shape[:2]
    cx, cy = sample_flat_position(img_bgr, rng)
    angle = float(rng.uniform(0, math.pi))
    length = max(12.0, min(h, w) * length_frac)
    dx, dy = math.cos(angle) * length / 2, math.sin(angle) * length / 2
    return {
        "points": [[cx - dx, cy - dy], [cx + dx, cy + dy]],
        "width": int(width),
        "delta": float(delta),
        "polarity": polarity,
        "edge": edge,
        "curvature": float(rng.uniform(-0.3, 0.3)),
    }


def random_stroke(
    img_bgr: np.ndarray,
    rng: np.random.Generator,
    deltas: List[float],
    widths: List[int],
    edges: List[str],
    polarities: List[str],
) -> Dict[str, Any]:
    """One mark with every parameter drawn at random from the given pools.

    Used for dataset generation, where the goal is coverage rather than a
    controlled sweep: an exhaustive grid produces a rigid, evenly-spaced set
    that a threshold can fit suspiciously well, while random draws over the
    same pools give the varied population real defects come in.

    Size varies per mark too. A batch where every scratch is exactly the same
    length is another regularity the model could latch onto instead of the
    defect itself.
    """
    edge = str(edges[int(rng.integers(len(edges)))])
    delta = float(deltas[int(rng.integers(len(deltas)))])
    width = int(widths[int(rng.integers(len(widths)))])
    polarity = str(polarities[int(rng.integers(len(polarities)))]) if polarities else "dark"
    # Bubbles are blobs, not lines: they need a wider span to read as a dome,
    # and they look wrong at scratch proportions.
    lo, hi = (0.10, 0.22) if edge == "bubble" else (0.08, 0.24)
    stroke = auto_stroke(
        img_bgr, rng, width=width, delta=delta, edge=edge, polarity=polarity,
        length_frac=float(rng.uniform(lo, hi)),
    )
    if edge == "bubble":
        # `curvature` carries the light direction for bubbles — spread it over
        # the full range so a batch is not all lit from the same side.
        stroke["curvature"] = float(rng.uniform(-1.0, 1.0))
    return stroke


def _draw_bubble(
    img_rgb: np.ndarray,
    pts: np.ndarray,
    width: int,
    delta: float,
    local: float,
    light_angle: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Air bubble / blister trapped under the label.

    Unlike the line-shaped defects, a bubble is a shallow dome: it has no ink
    of its own, it only redirects light. So it is drawn as a directional
    shading field over an elliptical footprint — bright on the slope facing the
    light, dark on the slope away from it, and nearly unchanged at the centre
    where the surface is still parallel to the label. Filling the ellipse with
    a flat offset instead would read as a printed blob, which is a different
    defect and would teach the wrong thing.

    Geometry comes from the two stroke points: they span the major axis, so
    the same auto-placement code that positions a scratch also positions a
    bubble. `width` sets how soft the rim is.
    """
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx = max(4.0, math.hypot(x1 - x0, y1 - y0) / 2.0)
    ry = rx * 0.72                      # slightly oval — a perfect circle looks synthetic
    angle = math.atan2(y1 - y0, x1 - x0)

    h, w = img_rgb.shape[:2]
    # Work in a local window; a full-frame meshgrid on a 1600px image is wasteful.
    pad = int(rx + width * 3 + 6)
    x_lo, x_hi = max(0, int(cx) - pad), min(w, int(cx) + pad)
    y_lo, y_hi = max(0, int(cy) - pad), min(h, int(cy) + pad)
    if x_hi <= x_lo or y_hi <= y_lo:
        return img_rgb, np.zeros((h, w), bool)

    ys, xs = np.mgrid[y_lo:y_hi, x_lo:x_hi].astype(np.float32)
    dx, dy = xs - cx, ys - cy
    ca, sa = math.cos(-angle), math.sin(-angle)
    xr = (dx * ca - dy * sa) / rx        # normalised ellipse coords
    yr = (dx * sa + dy * ca) / ry
    r = np.sqrt(xr * xr + yr * yr)

    inside = r <= 1.0
    # Direction of the incoming light, in the ellipse's own frame.
    lx, ly = math.cos(light_angle - angle), math.sin(light_angle - angle)
    shade = np.clip(xr * lx + yr * ly, -1.0, 1.0)
    # Contrast peaks on the slopes and fades at the centre, where the dome is
    # flat and reflects like the surrounding label.
    slope = np.clip(r, 0.0, 1.0) ** 0.7
    field = np.where(inside, shade * slope, 0.0).astype(np.float32)

    # Soften the rim so the bubble fades into the label instead of ending on a
    # step edge — nothing under a label has a hard boundary.
    k = max(3, int(width) * 2 + 1)
    if k % 2 == 0:
        k += 1
    field = cv.GaussianBlur(field, (k, k), sigmaX=max(0.6, width * 0.9))

    win = img_rgb[y_lo:y_hi, x_lo:x_hi].astype(np.float32)
    target = np.clip(local + delta * field, 0, 255)[..., None]
    # |field| doubles as the blend weight: no shift where the dome is flat.
    a = np.abs(field)[..., None]
    out = img_rgb.copy()
    out[y_lo:y_hi, x_lo:x_hi] = (win * (1 - a) + target * a).clip(0, 255).astype(np.uint8)

    footprint = np.zeros((h, w), bool)
    footprint[y_lo:y_hi, x_lo:x_hi] = inside
    return out, footprint


def apply_stroke(img_bgr: np.ndarray, stroke: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Draw one stroke. Returns (image, hard footprint mask)."""
    pts = np.asarray(stroke.get("points") or [], dtype=np.float32).reshape(-1, 2)
    if len(pts) < 2:
        return img_bgr, np.zeros(img_bgr.shape[:2], bool)

    width = max(1, int(stroke.get("width", 3)))
    delta = float(stroke.get("delta", 25))
    edge = stroke.get("edge", "soft")
    if edge not in EDGES:
        edge = "soft"
    sign = POLARITIES.get(stroke.get("polarity", "dark"), -1)
    pts = _curve_points(pts, float(stroke.get("curvature", 0.0)))

    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY).astype(np.float32)
    mid = pts[len(pts) // 2]
    local = _local_background(gray, int(mid[0]), int(mid[1]))
    shape = img_bgr.shape[:2]

    if edge == "bubble":
        # `curvature` is reused as the light direction — auto-generation already
        # randomises it, so a batch gets bubbles lit from varying angles.
        light = float(stroke.get("curvature", 0.0)) * math.pi + math.pi / 4
        img_rgb, footprint = _draw_bubble(img_rgb, pts, width, delta, local, light)
    elif edge == "wrinkle":
        # Crease = highlight on one side, shadow on the other. Offset by the
        # stroke width so the two bands sit next to each other, not on top.
        off = max(2.0, width * 1.0)
        hi_pts, lo_pts = _offset(pts, +off), _offset(pts, -off)
        a_hi, fp_hi = _stroke_alpha(shape, hi_pts, width, "soft")
        img_rgb = _blend(img_rgb, a_hi, float(np.clip(local + delta, 0, 255)))
        a_lo, fp_lo = _stroke_alpha(shape, lo_pts, width, "soft")
        img_rgb = _blend(img_rgb, a_lo, float(np.clip(local - delta, 0, 255)))
        footprint = fp_hi | fp_lo
    else:
        alpha, footprint = _stroke_alpha(shape, pts, width, edge)
        img_rgb = _blend(img_rgb, alpha, float(np.clip(local + sign * delta, 0, 255)))

    return cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR), footprint


def apply_strokes(img_bgr: np.ndarray, strokes: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Draw every stroke in order. Returns (image, union of footprints)."""
    out = img_bgr.copy()
    union = np.zeros(img_bgr.shape[:2], bool)
    for stroke in strokes or []:
        out, fp = apply_stroke(out, stroke)
        union |= fp
    return out, union


def mask_polygons(
    mask: np.ndarray,
    min_area: float = 0.0,
    max_points: int = 120,
    amap: Optional[np.ndarray] = None,
    footprint: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Split a binary mask into regions, each with its outline and statistics.

    Regions come from connected components rather than findContours alone, so
    `area` is an exact pixel count and the heatmap statistics can be read from
    the region's real membership. cv.contourArea would give the area of the
    simplified polygon instead, which drifts from the pixel count it is
    reported next to.

    `amap` (the anomaly heatmap) yields a per-region confidence — a large weak
    region and a small intense one are very different findings and a bare area
    cannot tell them apart. `footprint` (the drawn mark, when there is one)
    yields per-region overlap, which is what separates a region that landed on
    the defect from one the model raised somewhere else.

    Outlines are simplified with approxPolyDP so the payload stays drawable in
    the browser — a raw contour on a 512px map runs to thousands of points.
    """
    if mask is None or not mask.any():
        return []
    n_labels, labels = cv.connectedComponents(mask.astype(np.uint8), connectivity=8)
    total_px = int(mask.shape[0] * mask.shape[1])
    out = []
    for label_id in range(1, n_labels):
        blob = labels == label_id
        area = int(blob.sum())
        if area < min_area:
            continue
        contours, _ = cv.findContours(blob.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv.contourArea)
        eps = 0.001 * cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) > max_points:
            approx = approx[:: int(np.ceil(len(approx) / max_points))]
        if len(approx) < 3:
            continue
        x, y, w, h = cv.boundingRect(c)

        region: Dict[str, Any] = {
            "points": approx.astype(int).tolist(),
            "area": area,
            "area_pct": round(area / max(total_px, 1) * 100, 3),
            "bbox": [int(x), int(y), int(w), int(h)],
        }
        if amap is not None:
            vals = amap[blob]
            region["score_mean"] = round(float(vals.mean()), 4)
            region["score_max"] = round(float(vals.max()), 4)
        if footprint is not None:
            # What fraction of THIS region sits on the drawn mark. Low means the
            # model reacted to something else, however large the region is.
            region["stroke_overlap"] = round(float(np.logical_and(blob, footprint).sum() / area), 4)
        out.append(region)
    out.sort(key=lambda r: -r["area"])
    return out
