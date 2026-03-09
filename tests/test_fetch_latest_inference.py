#!/usr/bin/env python3
"""
Fetch latest inference_results records and crop text/datecode regions from FAIL cameras.
Filename: exp[expected]_got[recognized]_PASS/FAIL_c{conf}
Usage:
  python tests/test_fetch_latest_inference.py --recipe_id 69a1b1bc8ef588eb280877b0 --pass_fail FAIL --limit 10
"""

import os
import re
import argparse
import cv2
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://admin:password@localhost:27018/")
DATABASE_NAME = os.getenv("DATABASE_NAME", "ocr_datecode_db")
IMAGE_BASE = "/home/demo/Source/ocr_datecode/backend/uploads"
CROP_OUT_DIR = "tests/crop_output"
HANDLE_TYPES = {"text", "datecode"}


def safe(text: str) -> str:
    return re.sub(r'[^\w\-.]', '_', str(text).strip())[:40] or "empty"


def crop_frame(img, regions: list, text_verif: dict, out_prefix: str):
    # Build lookup: annotation_index -> verification result
    verif_map = {
        r["annotation_idx"]: r
        for r in (text_verif or {}).get("results", [])
        if "annotation_idx" in r
    }

    for i, region in enumerate(regions):
        rtype = region.get("type", "")
        if rtype not in HANDLE_TYPES:
            continue

        pts = region.get("points", [])
        if not pts:
            continue

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1 = max(0, int(min(xs)))
        y1 = max(0, int(min(ys)))
        x2 = min(img.shape[1], int(max(xs)))
        y2 = min(img.shape[0], int(max(ys)))

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        ann_idx = region.get("annotation_index")
        verif = verif_map.get(ann_idx)

        if verif:
            expected   = safe(verif.get("expected", ""))
            recognized = safe(verif.get("recognized", ""))
            match      = "PASS" if verif.get("match") else "FAIL"
            conf       = verif.get("confidence", 0.0)
            suffix = f"_exp[{expected}]_got[{recognized}]_{match}_c{conf:.2f}"
        else:
            # region chưa có trong text_verification (e.g. datecode chưa verify)
            expected = safe(region.get("text", ""))
            suffix = f"_exp[{expected}]_no_verif"

        out_path = f"{out_prefix}_{rtype}_{i}{suffix}.jpg"
        cv2.imwrite(out_path, crop)
        print(f"    {os.path.basename(out_path)}")


def run(recipe_id=None, pass_fail=None, limit=1):
    client = MongoClient(MONGODB_URL)
    db = client[DATABASE_NAME]

    query = {}
    if recipe_id:
        query["recipe_id"] = recipe_id
    if pass_fail:
        query["product_pass_fail"] = pass_fail.upper()

    total = db.inference_results.count_documents(query)
    print(f"Filter: {query if query else '(none)'} | Total: {total} | Processing: {limit}")

    records = list(db.inference_results.find(query, sort=[("_id", -1)]).limit(limit))
    client.close()

    os.makedirs(CROP_OUT_DIR, exist_ok=True)

    for rec_idx, record in enumerate(records):
        rec_id = str(record["_id"])
        print(f"\n[{rec_idx+1}] {rec_id} | {record.get('recipe_name')} | {record.get('product_pass_fail')}")

        for cam in record.get("camera_results", []):
            if cam.get("pass_fail") != "FAIL":
                continue
            cam_id = cam.get("camera_id", "cam")
            print(f"  Camera {cam_id} FAIL")

            for frame in cam.get("frames", []):
                if not frame.get("image_path"):
                    continue

                img_path = os.path.join(IMAGE_BASE, frame["image_path"].replace("_viz", "_org"))
                img = cv2.imread(img_path)
                if img is None:
                    print(f"  [WARN] Cannot read: {img_path}")
                    continue

                regions    = frame.get("detected_regions", [])
                text_verif = frame.get("text_verification")
                frame_name = frame.get("template_name", "frame").replace(" ", "_")
                out_prefix = os.path.join(CROP_OUT_DIR, f"{rec_idx+1}_{cam_id}_{frame_name}")

                text_regions = [r for r in regions if r.get("type") in HANDLE_TYPES]
                print(f"  Frame: {frame['image_path']} | text/datecode regions: {len(text_regions)}")
                crop_frame(img, regions, text_verif, out_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe_id", help="Filter by recipe_id")
    parser.add_argument("--pass_fail", help="Filter by product_pass_fail (PASS/FAIL)")
    parser.add_argument("--limit", type=int, default=1, help="Number of records (default: 1)")
    args = parser.parse_args()

    run(recipe_id=args.recipe_id, pass_fail=args.pass_fail, limit=args.limit)
