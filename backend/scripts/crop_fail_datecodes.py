"""
Crop datecode regions from FAIL inference results for ALL recipes.

Usage:
    python crop_all_recipes_datecodes.py [options]

Examples:
    # Mỗi recipe 150 ảnh, mỗi recipe một subfolder
    python crop_all_recipes_datecodes.py --limit_per_recipe 150

    # Chỉ recipe có ít nhất 1 FAIL, output tùy chỉnh
    python crop_all_recipes_datecodes.py --limit_per_recipe 150 --output_dir /tmp/all_datecodes

    # Giới hạn 50 ảnh mỗi camera, mỗi recipe
    python crop_all_recipes_datecodes.py --limit_per_recipe 150 --limit_per_camera 50
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pymongo import MongoClient

# ─── Config ──────────────────────────────────────────────────────────────────
MONGODB_URL  = os.getenv("MONGODB_URL",  "mongodb://admin:password@localhost:27017/")
DATABASE_NAME = os.getenv("DATABASE_NAME", "ocr_datecode_db")
UPLOADS_BASE  = Path(os.getenv("UPLOADS_BASE", "/home/demo/Source/ocr_datecode/backend/uploads"))


# ─── Helpers ─────────────────────────────────────────────────────────────────

def safe_filename(text: str) -> str:
    """Replace characters that are invalid in filenames."""
    return re.sub(r'[\\/*?:"<>|/]', "_", text).strip()


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as [TL, TR, BR, BL] — giống hệt ocr_utils._order_points.

    Logic:
      - TL = tổng (x+y) nhỏ nhất
      - BR = tổng (x+y) lớn nhất
      - TR = hiệu (x-y) nhỏ nhất  (x lớn, y nhỏ)
      - BL = hiệu (x-y) lớn nhất  (x nhỏ, y lớn)

    Cách này đúng với mọi góc xoay, không bị sai khi box nghiêng nhiều
    (khác với cách sort theo Y rồi sort theo X).
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL
    rect[2] = pts[np.argmax(s)]   # BR
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # TR
    rect[3] = pts[np.argmax(d)]   # BL
    return rect


def crop_obb(image: np.ndarray, points: list) -> np.ndarray | None:
    """
    Perspective-correct crop của oriented bounding box.
    Dùng cùng logic với crop_text_region trong TextVerificationService.

    points: list of 4 [x, y] corners (bất kỳ thứ tự nào).
    Trả về ảnh crop đã warp thẳng, hoặc None nếu thất bại.
    """
    pts = np.array(points, dtype=np.float32)
    tl, tr, br, bl = _order_points(pts)

    # Tính kích thước đích từ độ dài cạnh thực tế (không dùng AABB)
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

    if w <= 0 or h <= 0:
        return None

    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h))


def get_all_recipe_ids(db) -> list[str]:
    """Lấy tất cả recipe_id unique, sort theo created_at mới nhất đến cũ nhất."""
    pipeline = [
        {"$match": {"recipe_id": {"$exists": True, "$ne": None}}},
        {"$group": {"_id": "$recipe_id", "latest": {"$max": "$created_at"}}},
        {"$sort": {"latest": -1}},
    ]
    results = db["inference_results"].aggregate(pipeline)
    return [str(r["_id"]) for r in results if r["_id"]]


def process_recipe(
    db,
    recipe_id: str,
    output_dir: Path,
    limit_per_recipe: int | None,
    limit_per_camera: int | None,
) -> tuple[int, int]:
    """
    Crop ảnh FAIL cho một recipe.
    Trả về (saved, skipped).
    """
    collection = db["inference_results"]

    query = {
        "recipe_id": recipe_id,
        "product_pass_fail": "FAIL",
    }

    cursor = collection.find(query).sort("created_at", -1)

    output_dir.mkdir(parents=True, exist_ok=True)

    saved   = 0
    skipped = 0

    # Đếm số crop đã lưu theo camera (để giới hạn per-camera nếu cần)
    camera_counts: defaultdict[str, int] = defaultdict(int)

    def reached_recipe_limit() -> bool:
        return bool(limit_per_recipe and saved >= limit_per_recipe)

    def reached_camera_limit(cam_id: str) -> bool:
        return bool(limit_per_camera and camera_counts[cam_id] >= limit_per_camera)

    for doc in cursor:
        if reached_recipe_limit():
            break

        result_id = str(doc["_id"])

        for cam in (doc.get("camera_results") or []):
            if reached_recipe_limit():
                break
            if cam.get("pass_fail") != "FAIL":
                continue

            camera_id = cam.get("camera_id", "cam")

            for frame in (cam.get("frames") or []):
                if reached_recipe_limit():
                    break
                if reached_camera_limit(camera_id):
                    break

                frame_idx  = frame.get("frame_idx", 0)
                image_path = frame.get("image_path")

                if not image_path:
                    continue

                image_path = image_path.replace("_viz", "_org")
                full_image_path = UPLOADS_BASE / image_path

                if not full_image_path.exists():
                    print(f"    [WARN] Image not found: {full_image_path}")
                    skipped += 1
                    continue

                image = cv2.imread(str(full_image_path))
                if image is None:
                    print(f"    [WARN] Could not read: {full_image_path}")
                    skipped += 1
                    continue

                # Build lookup: annotation_idx -> verification result
                verif_map: dict = {}
                tv = frame.get("text_verification") or {}
                for vr in (tv.get("results") or []):
                    verif_map[vr.get("annotation_idx")] = vr

                for region in (frame.get("detected_regions") or []):
                    if reached_recipe_limit():
                        break
                    if reached_camera_limit(camera_id):
                        break
                    if region.get("type") != "datecode":
                        continue

                    points           = region.get("points")
                    text             = region.get("text", "")
                    annotation_index = region.get("annotation_index", 0)

                    if not points or len(points) < 4:
                        continue
                    if not text or not text.strip():
                        continue

                    crop = crop_obb(image, points)
                    if crop is None:
                        continue

                    vr        = verif_map.get(annotation_index, {})
                    expected  = vr.get("expected", text)
                    vc_conf   = vr.get("confidence", -1.0)
                    threshold = vr.get("threshold", -1.0)
                    predicted = vr.get("recognized", "")

                    # Bỏ qua các dự đoán đúng — chỉ lấy ảnh sai
                    if expected == predicted:
                        continue

                    safe_text     = safe_filename(text)
                    safe_expected = safe_filename(expected)

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

                    # Tăng counter SAU KHI lưu thành công
                    saved += 1
                    camera_counts[camera_id] += 1
                    print(f"    [SAVED] {filename}")

    return saved, skipped


def process_all(
    output_root: Path,
    limit_per_recipe: int | None,
    limit_per_camera: int | None,
):
    client = MongoClient(MONGODB_URL)
    db     = client[DATABASE_NAME]

    recipe_ids = get_all_recipe_ids(db)
    if not recipe_ids:
        print("[ERROR] Không tìm thấy recipe nào trong DB.")
        client.close()
        sys.exit(1)

    print(f"Tìm thấy {len(recipe_ids)} recipe(s).\n")

    total_saved   = 0
    total_skipped = 0
    summary: list[dict] = []

    for idx, recipe_id in enumerate(recipe_ids, 1):
        recipe_output_dir = output_root / safe_filename(recipe_id)
        print(f"[{idx}/{len(recipe_ids)}] Recipe: {recipe_id}")
        print(f"  → Output: {recipe_output_dir}")

        saved, skipped = process_recipe(
            db=db,
            recipe_id=recipe_id,
            output_dir=recipe_output_dir,
            limit_per_recipe=limit_per_recipe,
            limit_per_camera=limit_per_camera,
        )

        total_saved   += saved
        total_skipped += skipped
        summary.append({"recipe_id": recipe_id, "saved": saved, "skipped": skipped})
        print(f"  → Saved: {saved}, Skipped: {skipped}\n")

    client.close()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Recipe ID':<30} {'Saved':>8} {'Skipped':>9}")
    print("-" * 60)
    for row in summary:
        print(f"{row['recipe_id']:<30} {row['saved']:>8} {row['skipped']:>9}")
    print("-" * 60)
    print(f"{'TOTAL':<30} {total_saved:>8} {total_skipped:>9}")
    print("=" * 60)
    print(f"\nOutput root: {output_root.resolve()}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Crop datecode FAIL regions for ALL recipes."
    )
    parser.add_argument(
        "--output_dir",
        default="./cropped_datecodes_all",
        help="Root output directory. Each recipe gets its own subfolder. (default: ./cropped_datecodes_all)",
    )
    parser.add_argument(
        "--limit_per_recipe",
        type=int,
        default=150,
        help="Max wrong-prediction crops to save per recipe. 0 = unlimited. (default: 150)",
    )
    parser.add_argument(
        "--limit_per_camera",
        type=int,
        default=None,
        help="Optional: max crops per camera per recipe. (default: no limit)",
    )
    parser.add_argument("--mongodb_url",  default=None, help="Override MONGODB_URL env var")
    parser.add_argument("--uploads_base", default=None, help="Override UPLOADS_BASE env var")

    args = parser.parse_args()

    global MONGODB_URL, UPLOADS_BASE
    if args.mongodb_url:
        MONGODB_URL = args.mongodb_url
    if args.uploads_base:
        UPLOADS_BASE = Path(args.uploads_base)

    limit_per_recipe = args.limit_per_recipe if args.limit_per_recipe > 0 else None

    process_all(
        output_root=Path(args.output_dir),
        limit_per_recipe=limit_per_recipe,
        limit_per_camera=args.limit_per_camera,
    )


if __name__ == "__main__":
    main()