"""
Crop datecode regions from FAIL inference results.

Usage:
    python crop_fail_datecodes.py --recipe_id <id> --limit <n> --output_dir <path>

Examples:
    python crop_fail_datecodes.py --recipe_id 69c0efb30700293edfd62c7a --limit 100 --output_dir /tmp/datecodes
    python crop_fail_datecodes.py --recipe_id 69c0efb30700293edfd62c7a  # no limit, default output dir
"""

import argparse
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from pymongo import MongoClient
from bson import ObjectId

# ─── Config ──────────────────────────────────────────────────────────────────
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://admin:password@localhost:27017/")
DATABASE_NAME = os.getenv("DATABASE_NAME", "ocr_datecode_db")
UPLOADS_BASE = Path(os.getenv("UPLOADS_BASE", "/home/demo/Source/ocr_datecode/backend/uploads"))


# ─── Helpers ─────────────────────────────────────────────────────────────────

def safe_filename(text: str) -> str:
    """Replace characters that are invalid in filenames."""
    return re.sub(r'[\\/*?:"<>|/]', "_", text).strip()


def crop_obb(image: np.ndarray, points: list) -> np.ndarray:
    """
    Perspective-correct crop of an oriented bounding box.
    points: list of 4 [x, y] corners (any order).
    Returns the straightened rectangular crop.
    """
    pts = np.array(points, dtype=np.float32)

    # Order: top-left, top-right, bottom-right, bottom-left
    # Sort by y to get top / bottom pairs, then by x within each pair
    sorted_y = pts[np.argsort(pts[:, 1])]
    top_pts = sorted_y[:2][np.argsort(sorted_y[:2, 0])]   # tl, tr
    bot_pts = sorted_y[2:][np.argsort(sorted_y[2:, 0])]   # bl, br

    tl, tr = top_pts[0], top_pts[1]
    bl, br = bot_pts[0], bot_pts[1]

    src = np.array([tl, tr, br, bl], dtype=np.float32)

    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

    if w <= 0 or h <= 0:
        return None

    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h))


def process(recipe_id: str, limit: int | None, output_dir: Path):
    client = MongoClient(MONGODB_URL)
    db = client[DATABASE_NAME]
    collection = db["inference_results"]

    query = {
        "recipe_id": recipe_id,
        "product_pass_fail": "FAIL",
    }

    cursor = collection.find(query).sort("created_at", -1)
    if limit:
        cursor = cursor.limit(limit)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    skipped = 0

    for doc in cursor:
        result_id = str(doc["_id"])

        for cam in doc.get("camera_results", []):
            if cam.get("pass_fail") != "FAIL":
                continue
            camera_id = cam.get("camera_id", "cam")

            for frame in cam.get("frames", []):
                frame_idx = frame.get("frame_idx", 0)
                image_path = frame.get("image_path")

                if not image_path:
                    continue

                image_path = image_path.replace("_viz", "_org")

                full_image_path = UPLOADS_BASE / image_path
                if not full_image_path.exists():
                    print(f"  [WARN] Image not found: {full_image_path}")
                    skipped += 1
                    continue

                image = cv2.imread(str(full_image_path))
                if image is None:
                    print(f"  [WARN] Could not read image: {full_image_path}")
                    skipped += 1
                    continue

                # Build lookup: annotation_idx -> verification result
                verif_map = {}
                tv = frame.get("text_verification") or {}
                for vr in tv.get("results", []):
                    verif_map[vr.get("annotation_idx")] = vr

                for region in frame.get("detected_regions", []):
                    if region.get("type") != "datecode":
                        continue

                    points = region.get("points")
                    text = region.get("text", "")
                    annotation_index = region.get("annotation_index", 0)

                    if not points or len(points) < 4:
                        continue

                    if not text or not text.strip():
                        continue

                    crop = crop_obb(image, points)
                    if crop is None:
                        continue

                    # Pull verification metadata
                    vr = verif_map.get(annotation_index, {})
                    expected  = vr.get("expected", text)
                    vc_conf   = vr.get("confidence", -1.0)
                    threshold = vr.get("threshold", -1.0)
                    predicted = vr.get("recognized", "")

                    if expected == predicted:
                        continue  # Skip correct predictions, we only want fails


                    safe_text     = safe_filename(text)
                    safe_expected = safe_filename(expected)

                    # Encode meta into filename so the report can parse it back
                    # Separator __exp__ / __vc__ / __thr__ are distinct enough
                    filename = (
                        f"{result_id}_{camera_id}_f{frame_idx}_ann{annotation_index}"
                        f"__{safe_text}"
                        f"__exp__{safe_expected}"
                        f"__vc__{vc_conf:.4f}"
                        f"__thr__{threshold:.2f}"
                        f".jpg"
                    )
                    out_path = output_dir / filename

                    cv2.imwrite(str(out_path), crop)
                    print(f"  [SAVED] {filename}")
                    saved += 1

    client.close()
    print(f"\nDone. Saved: {saved}, Skipped (no image): {skipped}")
    print(f"Output dir: {output_dir.resolve()}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Crop datecode regions from FAIL inference results.")
    parser.add_argument("--recipe_id", required=True, help="Recipe ID to filter (hex string)")
    parser.add_argument("--limit", type=int, default=None, help="Max number of records to process (default: all)")
    parser.add_argument(
        "--output_dir",
        default="./cropped_datecodes",
        help="Directory to save cropped images (default: ./cropped_datecodes)",
    )
    parser.add_argument("--mongodb_url", default=None, help="Override MONGODB_URL")
    parser.add_argument("--uploads_base", default=None, help="Override UPLOADS_BASE path")

    args = parser.parse_args()

    global MONGODB_URL, UPLOADS_BASE
    if args.mongodb_url:
        MONGODB_URL = args.mongodb_url
    if args.uploads_base:
        UPLOADS_BASE = Path(args.uploads_base)

    process(
        recipe_id=args.recipe_id,
        limit=args.limit,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
