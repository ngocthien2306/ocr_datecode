#!/usr/bin/env python3
"""
Parity check: does the backend's CPU copy of the AI alignment
(app/services/live_template/) put regions where the AI service did?

For recent inspections of one camera it takes the frame the AI saved
(`*_org.jpg`), runs `live_template.realign.realign` on it with the recipe's
own settings, and compares every region against the `detected_regions` the
AI stored in Mongo for that same frame (matched by annotation_index, or by
type for the 'template' region). Reports the mean corner distance in pixels.

The saved `_org.jpg` of a cap camera is already the ROTATED frame (the
pipeline runs with inverse_transform=False and saves the frame it matched
on). realign's own rotation step then rotates the cap a little further — the
OBB still reads a residual 2-4° off it — so realign's regions live in a
slightly different frame than the AI's. Comparing them raw measures that
residual rotation, not the alignment. So the residual cap rotation is
reproduced here (same model, same input, same formula as rotate_frame_dual)
and undone about the cap centre before measuring.

Expect small differences, not zero: the AI runs SuperPoint as a TensorRT
fp16 engine, this runs the fp32 ONNX model; RANSAC is also randomised.
A few pixels is agreement; tens of pixels means the copy has drifted from
the source and needs re-syncing.

Only inspections run AFTER the template was last changed are usable —
older ones were aligned against a different template — so only inspections
newer than the recipe's `updated_at` + a grace period are compared. The
grace matters: the AI service rebuilds its matchers a few seconds after the
recipe is saved (measured 7 s on 2026-09-28: recipe saved 22:34:56, matcher
rebuilt 22:35:03 local), and frames in that gap were still aligned against the
OLD template — they showed up here as ~31 px "disagreement". (Region `text` can't be
used for that: the AI overwrites it with what OCR read.)

Usage (backend dir, conda env `vision`):
    python scripts/check_live_template_parity.py --recipe "Minced Onion 1 (Copy)" \
        --serial 24026290 --n 20
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

from pymongo import MongoClient  # noqa: E402

from app.services.cv_detectors import cuda_modules_present  # noqa: E402
from app.services.live_template.realign import RealignOptions, _services, realign  # noqa: E402

UPLOADS = BACKEND / "uploads"


def _opts(recipe, ct):
    return RealignOptions(
        function_type=ct.get("function_type", "") or "",
        cap_rotation_method=recipe.get("cap_rotation_method") or "yolo_obb",
        cap_crop_method=recipe.get("cap_crop_method") or "none",
        dual_rotation_check=bool(recipe.get("dual_rotation_check")),
        matching_conf=float(recipe.get("matching_conf") or 0.20),
        match_erosion_enabled=bool(recipe.get("match_erosion_enabled")),
        match_erosion_kernel_w=int(recipe.get("match_erosion_kernel_w") or 80),
        match_erosion_kernel_h=int(recipe.get("match_erosion_kernel_h") or 1),
        match_erosion_iterations=int(recipe.get("match_erosion_iterations") or 1),
        first_template_annotations=(ct.get("templates") or [{}])[0].get("annotations") or [],
    )


def _residual_rotation_deg(frame, crop_area):
    """The angle rotate_frame_dual rotated this frame's cap by: OBB text_box
    angle, +90 when the box is taller than wide (obb_rotator.py, same lines).
    None when the OBB path didn't run (CV fallback) — those frames are skipped."""
    obb, _ = _services()
    tb = obb.detect_text_box(frame, crop_area=crop_area)
    if tb is None:
        return None
    _, _, tw, th, ang = tb
    a = float(np.degrees(ang))
    return a + 90.0 if th > tw else a


def _unrotate(points, center, angle_deg):
    M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle_deg, 1.0)
    Mi = cv2.invertAffineTransform(M)
    pts = np.asarray(points, np.float32).reshape(-1, 2)
    return (np.hstack([pts, np.ones((len(pts), 1), np.float32)]) @ Mi.T).tolist()


def _key(r):
    return ("template", None) if r.get("type") == "template" else ("idx", r.get("annotation_index"))


def _corner_dist(a, b):
    a = np.asarray(a, np.float32).reshape(-1, 2)
    b = np.asarray(b, np.float32).reshape(-1, 2)
    if a.shape != b.shape:
        return None
    return float(np.linalg.norm(a - b, axis=1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--serial", required=True)
    ap.add_argument("--template-index", type=int, default=0)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--grace-seconds", type=float, default=15.0,
                    help="skip inspections this soon after the recipe update (AI reload lag)")
    ap.add_argument("--mongo", default=os.environ.get("MONGODB_URL", "mongodb://admin:password@localhost:27017/"))
    ap.add_argument("--db", default="ocr_datecode_db")
    args = ap.parse_args()

    db = MongoClient(args.mongo, serverSelectionTimeoutMS=4000).get_database(args.db)
    recipe = db.recipes.find_one({"name": args.recipe})
    if not recipe:
        sys.exit(f"recipe not found: {args.recipe}")
    cam = next((c for c in recipe.get("cameras", []) if c.get("serial_number") == args.serial), None)
    if not cam:
        sys.exit(f"camera {args.serial} not in recipe")
    ct = next(c for c in recipe["camera_templates"] if c["camera_id"] == cam["camera_id"])
    tmpl = ct["templates"][args.template_index]
    tpl_img = cv2.imread(str(UPLOADS / "templates" / tmpl["image_url"].split("/")[-1]))
    anns = tmpl["annotations"]
    since = recipe.get("updated_at")
    if since is not None:
        from datetime import timedelta
        since = since + timedelta(seconds=args.grace_seconds)
    opts = _opts(recipe, ct)

    rows, skipped = [], 0
    q = {"recipe_id": str(recipe["_id"])}
    if since is not None:
        q["timestamp"] = {"$gt": since}
    cur = db.inference_results.find(q, sort=[("timestamp", -1)]).limit(5000)
    for d in cur:
        if len(rows) >= args.n:
            break
        for c in d.get("camera_results", []):
            if c.get("serial_number") != args.serial:
                continue
            frames = c.get("frames", [])
            if args.template_index >= len(frames):
                continue
            fr = frames[args.template_index]
            p = fr.get("image_path") or ""
            org = UPLOADS / (p[:-8] + "_org.jpg" if p.endswith("_viz.jpg") else p)
            regs = fr.get("detected_regions") or []
            if not p or not org.exists() or not regs:
                skipped += 1
                continue
            frame = cv2.imread(str(org))
            if frame is None:  # still being written by the live line
                skipped += 1
                continue
            out = realign(frame, tpl_img, anns, opts)
            if out.get("chosen") is None:
                rows.append((d["timestamp"], None, None, out["candidates"][0]["error"] if out["candidates"] else out.get("reason")))
                continue
            chosen = out["candidates"][out["chosen"]]
            mine_regs = chosen["detected_regions"]
            if out.get("mode") == "cap_rotation" and out.get("cap_circle"):
                crop = next(({"x1": int(a["x"] * frame.shape[1]), "y1": int(a["y"] * frame.shape[0]),
                              "x2": int((a["x"] + a["width"]) * frame.shape[1]),
                              "y2": int((a["y"] + a["height"]) * frame.shape[0])}
                             for a in anns if a.get("type") == "crop_area"), None)
                phi = _residual_rotation_deg(frame, crop)
                if phi is None:
                    skipped += 1
                    continue
                if chosen["is_alt"]:
                    phi += 180.0
                c0 = out["cap_circle"][:2]
                mine_regs = [dict(r, points=_unrotate(r["points"], c0, phi)) for r in mine_regs]
            mine = {_key(r): r for r in mine_regs}
            dists = []
            for r in regs:
                m = mine.get(_key(r))
                if m is not None and r.get("points"):
                    dd = _corner_dist(r["points"], m["points"])
                    if dd is not None:
                        dists.append(dd)
            rows.append((d["timestamp"], float(np.mean(dists)) if dists else None,
                         float(np.max(dists)) if dists else None, None))

    print(f"recipe={args.recipe!r} serial={args.serial} template={tmpl.get('name')!r} "
          f"since {since} | compared {len(rows)} frames, skipped {skipped} (no saved image / no regions)")
    good = [r for r in rows if r[1] is not None]
    for ts, mean_d, max_d, err in rows:
        print(f"  {ts}  " + (f"mean {mean_d:6.2f} px  max {max_d:6.2f} px" if mean_d is not None else f"NO MATCH ({err})"))
    if good:
        m = np.array([r[1] for r in good]); x = np.array([r[2] for r in good])
        print(f"\nmean corner distance: median {np.median(m):.2f} px, p90 {np.percentile(m, 90):.2f} px, "
              f"worst region {x.max():.2f} px  ({len(good)}/{len(rows)} frames aligned)")
    print("CUDA modules loaded in this process:", cuda_modules_present() or "NONE")


if __name__ == "__main__":
    main()
