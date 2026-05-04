"""
Crop datecode regions from FAIL inference results for ALL recipes.

Usage:
    python crop_fail_datecodes.py [options]

Examples:
    # Mỗi recipe 150 ảnh, mỗi recipe một subfolder
    python crop_fail_datecodes.py --limit_per_recipe 150

    # Chỉ recipe cụ thể
    python crop_fail_datecodes.py --recipe_ids abc123 def456

    # Bật crop text khi product_verification fail
    python crop_fail_datecodes.py --crop_on_product_fail

    # Kết hợp tất cả
    python crop_fail_datecodes.py --recipe_ids 69dc5ae76f4ad1bfd551ee6a 69da40d924420f8c793c3a6b 69d9bb6c3e6c89aea642915a 69d8c53f709ff5dc8fd91338 --limit_per_recipe 200 --limit_per_camera 50 --crop_on_product_fail
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

# ─── Config ───────────────────────────────────────────────────────────────────
MONGODB_URL   = os.getenv("MONGODB_URL",   "mongodb://admin:password@localhost:27017/")
DATABASE_NAME = os.getenv("DATABASE_NAME", "ocr_datecode_db")
UPLOADS_BASE  = Path(os.getenv("UPLOADS_BASE", "/home/demo/Source/ocr_datecode/backend/uploads"))


# ─── Helpers ──────────────────────────────────────────────────────────────────

def safe_filename(text: str) -> str:
    """Replace characters that are invalid in filenames."""
    return re.sub(r'[\\/*?:"<>|/]', "_", text).strip()


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as [TL, TR, BR, BL].

    - TL = tổng (x+y) nhỏ nhất
    - BR = tổng (x+y) lớn nhất
    - TR = hiệu (x-y) nhỏ nhất  (x lớn, y nhỏ)
    - BL = hiệu (x-y) lớn nhất  (x nhỏ, y lớn)

    Đúng với mọi góc xoay, không bị sai khi box nghiêng nhiều.
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

    points: list of 4 [x, y] corners (bất kỳ thứ tự nào).
    Trả về ảnh crop đã warp thẳng, hoặc None nếu thất bại.
    """
    pts = np.array(points, dtype=np.float32)
    tl, tr, br, bl = _order_points(pts)

    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

    if w <= 0 or h <= 0:
        return None

    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h))


def _get_product_fail_reason(pv: dict) -> str:
    """Trả về tên check đầu tiên bị fail, để encode vào filename."""
    checks = [
        "center_alignment_check",
        "wrinkled_check",
        "rotation_check",
        "misalignment_check",
        "label_region_check",
    ]
    for check in checks:
        result = pv.get(check, {})
        if result and not result.get("ok", True) and not result.get("skipped", False):
            return check.replace("_check", "")
    return "unknown"


def load_image(image_path: str) -> np.ndarray | None:
    """Load ảnh gốc từ disk, tự động strip _viz → _org."""
    path = image_path.replace("_viz", "_org")
    full_path = UPLOADS_BASE / path
    if not full_path.exists():
        print(f"    [WARN] Image not found: {full_path}")
        return None
    img = cv2.imread(str(full_path))
    if img is None:
        print(f"    [WARN] Could not read: {full_path}")
    return img


# ─── DB Helpers ───────────────────────────────────────────────────────────────

def get_all_recipe_ids(db) -> list[str]:
    """Lấy tất cả recipe_id unique, sort theo created_at mới nhất đến cũ nhất."""
    pipeline = [
        {"$match": {"recipe_id": {"$exists": True, "$ne": None}}},
        {"$group": {"_id": "$recipe_id", "latest": {"$max": "$created_at"}}},
        {"$sort": {"latest": -1}},
    ]
    results = db["inference_results"].aggregate(pipeline)
    return [str(r["_id"]) for r in results if r["_id"]]


# ─── Core Processing ──────────────────────────────────────────────────────────

def process_frame(
    *,
    frame: dict,
    image: np.ndarray,
    result_id: str,
    camera_id: str,
    output_dir: Path,
    crop_on_product_fail: bool,
    camera_counts: defaultdict,
    limit_per_recipe: int | None,
    limit_per_camera: int | None,
    saved_ref: list,   # [int] — mutable counter
    skipped_ref: list, # [int] — mutable counter
):
    """
    Xử lý một frame: crop datecode sai + (tùy chọn) text khi product fail.
    Dùng list wrapper để thay đổi counter từ caller.
    """
    frame_idx  = frame.get("frame_idx", 0)
    image_path = frame.get("image_path")

    # Build lookup: annotation_idx -> verification result
    verif_map: dict = {}
    tv = frame.get("text_verification") or {}
    for vr in (tv.get("results") or []):
        verif_map[vr.get("annotation_idx")] = vr

    # Kiểm tra product_verification
    pv = frame.get("product_verification") or {}
    product_fail = (
        crop_on_product_fail
        and not pv.get("match", True)
        and not pv.get("skipped", False)
    )
    fail_reason = _get_product_fail_reason(pv) if product_fail else ""

    def reached_recipe_limit() -> bool:
        return bool(limit_per_recipe and saved_ref[0] >= limit_per_recipe)

    def reached_camera_limit() -> bool:
        return bool(limit_per_camera and camera_counts[camera_id] >= limit_per_camera)

    def save_crop(crop: np.ndarray, filename: str) -> bool:
        out_path = output_dir / filename
        cv2.imwrite(str(out_path), crop)
        saved_ref[0] += 1
        camera_counts[camera_id] += 1
        print(f"    [SAVED] {filename}")
        return True

    for region in (frame.get("detected_regions") or []):
        if reached_recipe_limit() or reached_camera_limit():
            break

        region_type      = region.get("type")
        points           = region.get("points")
        text             = region.get("text", "").strip()
        annotation_index = region.get("annotation_index", 0)

        if not points or len(points) < 4:
            continue

        # ── Datecode sai: crop bình thường ────────────────────────────────────
        if region_type == "datecode":
            if not text:
                continue

            crop = crop_obb(image, points)
            if crop is None:
                continue

            vr        = verif_map.get(annotation_index, {})
            expected  = vr.get("expected", text)
            predicted = vr.get("recognized", "")
            vc_conf   = vr.get("confidence", -1.0)
            threshold = vr.get("threshold", -1.0)

            # Chỉ lưu nếu dự đoán SAI
            if expected == predicted:
                continue

            filename = (
                f"{result_id}_{camera_id}_f{frame_idx}_ann{annotation_index}"
                f"__{safe_filename(text)}"
                f"__exp__{safe_filename(expected)}"
                f"__vc__{vc_conf:.4f}"
                f"__thr__{threshold:.2f}"
                f".jpg"
            )
            save_crop(crop, filename)

        # ── Text region khi product fail: crop kể cả đúng ────────────────────
        elif region_type == "text" and product_fail:
            if not text:
                continue

            crop = crop_obb(image, points)
            if crop is None:
                continue

            vr        = verif_map.get(annotation_index, {})
            expected  = vr.get("expected", text)
            predicted = vr.get("recognized", "")
            vc_conf   = vr.get("confidence", -1.0)
            threshold = vr.get("threshold", -1.0)

            filename = (
                f"PRODFAIL_{result_id}_{camera_id}_f{frame_idx}_ann{annotation_index}"
                f"__{safe_filename(text)}"
                f"__exp__{safe_filename(expected)}"
                f"__pred__{safe_filename(predicted)}"
                f"__vc__{vc_conf:.4f}"
                f"__reason__{fail_reason}"
                f".jpg"
            )
            save_crop(crop, filename)


def process_recipe(
    db,
    recipe_id: str,
    output_dir: Path,
    limit_per_recipe: int | None,
    limit_per_camera: int | None,
    crop_on_product_fail: bool = False,
) -> tuple[int, int]:
    """
    Crop ảnh FAIL cho một recipe.
    Trả về (saved, skipped).
    """
    collection = db["inference_results"]
    cursor     = collection.find(
        {"recipe_id": recipe_id, "product_pass_fail": "FAIL"}
    ).sort("created_at", -1)

    output_dir.mkdir(parents=True, exist_ok=True)

    saved_ref   = [0]   # mutable wrapper
    skipped_ref = [0]
    camera_counts: defaultdict[str, int] = defaultdict(int)

    def reached_recipe_limit() -> bool:
        return bool(limit_per_recipe and saved_ref[0] >= limit_per_recipe)

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

                image_path = frame.get("image_path")
                if not image_path:
                    skipped_ref[0] += 1
                    continue

                image = load_image(image_path)
                if image is None:
                    skipped_ref[0] += 1
                    continue

                process_frame(
                    frame=frame,
                    image=image,
                    result_id=result_id,
                    camera_id=camera_id,
                    output_dir=output_dir,
                    crop_on_product_fail=crop_on_product_fail,
                    camera_counts=camera_counts,
                    limit_per_recipe=limit_per_recipe,
                    limit_per_camera=limit_per_camera,
                    saved_ref=saved_ref,
                    skipped_ref=skipped_ref,
                )

    return saved_ref[0], skipped_ref[0]


def process_all(
    output_root: Path,
    limit_per_recipe: int | None,
    limit_per_camera: int | None,
    recipe_ids: list[str] | None = None,
    crop_on_product_fail: bool = False,
):
    client = MongoClient(MONGODB_URL)
    db     = client[DATABASE_NAME]

    if recipe_ids:
        print(f"Dùng {len(recipe_ids)} recipe ID từ tham số --recipe_ids.")
    else:
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
            crop_on_product_fail=crop_on_product_fail,
        )

        total_saved   += saved
        total_skipped += skipped
        summary.append({"recipe_id": recipe_id, "saved": saved, "skipped": skipped})
        print(f"  → Saved: {saved}, Skipped: {skipped}\n")

    client.close()

    # ── Summary ───────────────────────────────────────────────────────────────
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


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Crop datecode FAIL regions for ALL (or selected) recipes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--recipe_ids",
        nargs="+",
        default=None,
        metavar="ID",
        help="Danh sách recipe ID cụ thể. Nếu không truyền, lấy tất cả từ DB.",
    )
    parser.add_argument(
        "--output_dir",
        default="./cropped_datecodes_all1",
        help="Root output directory. Mỗi recipe một subfolder. (default: ./cropped_datecodes_all)",
    )
    parser.add_argument(
        "--limit_per_recipe",
        type=int,
        default=150,
        help="Max crops sai per recipe. 0 = unlimited. (default: 150)",
    )
    parser.add_argument(
        "--limit_per_camera",
        type=int,
        default=None,
        metavar="N",
        help="Max crops per camera per recipe. (default: no limit)",
    )
    parser.add_argument(
        "--crop_on_product_fail",
        action="store_true",
        default=False,
        help=(
            "Nếu product_verification.match=False, crop TẤT CẢ text regions "
            "của frame đó, kể cả OCR đúng. Filename có prefix PRODFAIL_ và lý do fail."
        ),
    )
    parser.add_argument(
        "--mongodb_url",
        default=None,
        help="Override MONGODB_URL env var.",
    )
    parser.add_argument(
        "--uploads_base",
        default=None,
        help="Override UPLOADS_BASE env var.",
    )

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
        recipe_ids=args.recipe_ids,
        crop_on_product_fail=args.crop_on_product_fail,
    )


if __name__ == "__main__":
    main()