"""
Crop các vùng OCR (text/datecode) từ inference_results để LABEL LẠI + chuẩn bị training.

Khác với crop_fail_datecodes.py (chỉ lấy datecode đoán SAI), script này lấy CẢ:
  - vùng PASS  (text_verification.match == True)  — khoảng N_PASS ảnh
  - vùng FAIL  (text_verification.match == False) — khoảng N_FAIL ảnh
cho mỗi/nhiều recipe, mỗi loại một thư mục riêng để label cho tiện.

Crop dùng perspective-warp (crop_obb) GIỐNG pipeline production — đúng pixel mà OCR thấy.

Layout output:
    <output_dir>/
        pass/<recipe_id>/<...>.jpg
        fail/<recipe_id>/<...>.jpg

Filename mã hoá sẵn expected/recognized/conf để tham chiếu khi label:
    {tag}__{recipe8}__{result_id}__{cam}__f{fi}__ann{idx}__exp-{EXP}__got-{GOT}__c{CONF}.jpg

Usage:
    # mặc định: 2 recipe, 150 pass + 400 fail (chia đều mỗi recipe)
    python backend/scripts/crop_ocr_training_data.py

    # tuỳ chỉnh
    python backend/scripts/crop_ocr_training_data.py \
        --recipe_ids 6a31ec96aa6d85bb656662a6 69ae5d9a48cb43b33a655486 \
        --limit_pass 150 --limit_fail 400 \
        --output_dir ./ocr_training_data --max_per_doc 2
"""

import argparse
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────
MONGODB_URL   = os.getenv("MONGODB_URL",   "mongodb://admin:password@localhost:27017/")
DATABASE_NAME = os.getenv("DATABASE_NAME", "ocr_datecode_db")
UPLOADS_BASE  = Path(os.getenv("UPLOADS_BASE", "/home/demo/Source/ocr_datecode/backend/uploads"))

OCR_TYPES = {"text", "datecode"}
DEFAULT_RECIPES = [
    "6a31ec96aa6d85bb656662a6",   # garlic powder
    "69ae5d9a48cb43b33a655486",   # Colect data
    "6a0fae86df97746461eb33b5",   # HEB (nhiều ngày)
]


# ─── Helpers (giống crop_fail_datecodes.py) ─────────────────────────────────────

def safe_filename(text: str) -> str:
    """Thay ký tự không hợp lệ trong tên file. Rỗng → 'empty'."""
    cleaned = re.sub(r'[\\/*?:"<>|/ ]', "_", str(text)).strip("_")
    return cleaned[:48] or "empty"


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Sắp 4 điểm thành [TL, TR, BR, BL] — đúng với mọi góc xoay."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL
    rect[2] = pts[np.argmax(s)]   # BR
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # TR
    rect[3] = pts[np.argmax(d)]   # BL
    return rect


def crop_obb(image: np.ndarray, points: list):
    """Perspective-correct crop của oriented bbox (giống pipeline). None nếu fail."""
    pts = np.array(points, dtype=np.float32)
    tl, tr, br, bl = _order_points(pts)
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if w <= 0 or h <= 0:
        return None
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h))


def load_image(image_path: str):
    """Load ảnh gốc (_org) từ disk."""
    full_path = UPLOADS_BASE / image_path.replace("_viz", "_org")
    if not full_path.exists():
        return None
    return cv2.imread(str(full_path))


# ─── Core ───────────────────────────────────────────────────────────────────────

def _distinct_days(coll, recipe_id: str, product_filter: str) -> list:
    """Danh sách ngày (YYYY-MM-DD, UTC) có data cho recipe+bucket, cũ→mới."""
    pipe = [
        {"$match": {"recipe_id": recipe_id, "product_pass_fail": product_filter}},
        {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}}}},
        {"$sort": {"_id": 1}},
    ]
    return [r["_id"] for r in coll.aggregate(pipe) if r["_id"]]


def collect_bucket(
    coll,
    recipe_id: str,
    *,
    product_filter: str,   # "PASS" | "FAIL" — doc nào quét
    want_match: bool,      # True = lấy region OCR đúng, False = lấy region OCR sai
    limit: int,
    output_dir: Path,
    max_per_doc: int,
    per_day: bool = True,
) -> int:
    """
    Crop các region text/datecode có text_verification.match == want_match.

    per_day=True: chia đều quota cho từng NGÀY có data (tránh dồn hết vào 1 ngày
    với recipe chạy nhiều ngày). per_day=False: lấy tuần tự mới→cũ.

    Trả về số ảnh đã lưu.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    seen: set = set()   # filenames đã lưu — chống đếm/ghi trùng giữa các ngày + lượt bù

    if not per_day:
        return _collect_range(coll, recipe_id, product_filter, want_match,
                              limit, output_dir, max_per_doc, seen, day=None)

    days = _distinct_days(coll, recipe_id, product_filter)
    if not days:
        return 0
    share = math.ceil(limit / len(days))
    for d in days:
        if len(seen) >= limit:
            break
        take = min(share, limit - len(seen))
        _collect_range(coll, recipe_id, product_filter, want_match,
                       take, output_dir, max_per_doc, seen, day=d)

    # Bù: nếu vài ngày thiếu ảnh (vd PASS hiếm được lưu) → quét toàn bộ lấp đủ.
    if len(seen) < limit:
        _collect_range(coll, recipe_id, product_filter, want_match,
                       limit, output_dir, max_per_doc, seen, day=None)
    return len(seen)


def _collect_range(
    coll, recipe_id, product_filter, want_match,
    limit, output_dir, max_per_doc, seen: set, day=None,
) -> int:
    """
    Quét doc (1 ngày nếu `day`, hoặc toàn bộ) sort mới→cũ, lưu region cho tới khi
    `len(seen) >= limit`. Chỉ lấy doc CÓ ảnh lưu trên đĩa. Bỏ qua file đã có trong
    `seen` (chống trùng). Trả về số file MỚI thêm trong lượt này.
    """
    tag = "pass" if want_match else "fail"
    query = {
        "recipe_id": recipe_id,
        "product_pass_fail": product_filter,
        # chỉ doc có ít nhất 1 frame ảnh lưu trên đĩa (PASS thường ko lưu ảnh)
        "camera_results.frames.image_path": {"$type": "string"},
    }
    if day:
        start = datetime.strptime(day, "%Y-%m-%d")
        query["created_at"] = {"$gte": start, "$lt": start + timedelta(days=1)}

    start_count = len(seen)
    cursor = coll.find(query).sort("created_at", -1)

    for doc in cursor:
        if len(seen) >= limit:
            break
        result_id = str(doc["_id"])

        for cam in (doc.get("camera_results") or []):
            if len(seen) >= limit:
                break
            camera_id = cam.get("camera_id", "cam")

            for frame in (cam.get("frames") or []):
                if len(seen) >= limit:
                    break
                image_path = frame.get("image_path")
                if not image_path:
                    continue

                # verif lookup: annotation_idx -> result
                tv = frame.get("text_verification") or {}
                verif_map = {
                    vr.get("annotation_idx"): vr
                    for vr in (tv.get("results") or [])
                    if vr.get("annotation_idx") is not None
                }
                if not verif_map:
                    continue

                image = None  # lazy-load chỉ khi có region cần crop
                per_doc = 0

                for region in (frame.get("detected_regions") or []):
                    if len(seen) >= limit or per_doc >= max_per_doc:
                        break
                    if region.get("type") not in OCR_TYPES:
                        continue
                    points = region.get("points")
                    if not points or len(points) < 4:
                        continue

                    ann_idx = region.get("annotation_index")
                    vr = verif_map.get(ann_idx)
                    if vr is None:
                        continue
                    if bool(vr.get("match", False)) != want_match:
                        continue

                    if image is None:
                        image = load_image(image_path)
                        if image is None:
                            break  # ảnh hỏng → bỏ cả frame

                    crop = crop_obb(image, points)
                    if crop is None or crop.size == 0:
                        continue

                    expected   = vr.get("expected", region.get("text", "")) or ""
                    recognized = vr.get("recognized", "") or ""
                    conf       = vr.get("confidence", -1.0) or -1.0

                    filename = (
                        f"{tag}__{recipe_id[:8]}__{result_id}__{camera_id}"
                        f"__f{frame.get('frame_idx', 0)}__ann{ann_idx}"
                        f"__exp-{safe_filename(expected)}"
                        f"__got-{safe_filename(recognized)}"
                        f"__c{float(conf):.3f}.jpg"
                    )
                    if filename in seen:
                        continue
                    cv2.imwrite(str(output_dir / filename), crop)
                    seen.add(filename)
                    per_doc += 1

    return len(seen) - start_count


def main():
    parser = argparse.ArgumentParser(
        description="Crop vùng OCR (pass + fail) cho training/relabel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--recipe_ids", nargs="+", default=DEFAULT_RECIPES,
                        metavar="ID", help="Danh sách recipe_id.")
    parser.add_argument("--limit_pass", type=int, default=150,
                        help="TỔNG số ảnh PASS (chia đều các recipe). (default: 150)")
    parser.add_argument("--limit_fail", type=int, default=400,
                        help="TỔNG số ảnh FAIL (chia đều các recipe). (default: 400)")
    parser.add_argument("--output_dir", default="./ocr_training_data",
                        help="Thư mục output (default: ./ocr_training_data)")
    parser.add_argument("--max_per_doc", type=int, default=2,
                        help="Max region lấy mỗi inference doc — tăng đa dạng. (default: 2)")
    parser.add_argument("--no_per_day", action="store_true",
                        help="Tắt sampling cân bằng theo ngày (mặc định BẬT).")
    parser.add_argument("--mongodb_url", default=None, help="Override MONGODB_URL.")
    parser.add_argument("--uploads_base", default=None, help="Override UPLOADS_BASE.")
    args = parser.parse_args()

    global MONGODB_URL, UPLOADS_BASE
    if args.mongodb_url:
        MONGODB_URL = args.mongodb_url
    if args.uploads_base:
        UPLOADS_BASE = Path(args.uploads_base)

    n = len(args.recipe_ids)
    # chia đều quota cho từng recipe (recipe cuối nhận phần dư)
    base_pass, rem_pass = divmod(args.limit_pass, n)
    base_fail, rem_fail = divmod(args.limit_fail, n)

    client = MongoClient(MONGODB_URL)
    coll = client[DATABASE_NAME]["inference_results"]
    out_root = Path(args.output_dir)

    print(f"DB: {DATABASE_NAME} @ {MONGODB_URL}")
    print(f"Recipes: {n} | quota TỔNG: {args.limit_pass} pass / {args.limit_fail} fail | "
          f"max_per_doc={args.max_per_doc}\n")

    summary = []
    for i, rid in enumerate(args.recipe_ids):
        lim_pass = base_pass + (rem_pass if i == n - 1 else 0)
        lim_fail = base_fail + (rem_fail if i == n - 1 else 0)
        print(f"[{i+1}/{n}] Recipe {rid}: target {lim_pass} pass / {lim_fail} fail")

        per_day = not args.no_per_day
        p = collect_bucket(coll, rid, product_filter="PASS", want_match=True,
                           limit=lim_pass, output_dir=out_root / "pass" / rid,
                           max_per_doc=args.max_per_doc, per_day=per_day)
        f = collect_bucket(coll, rid, product_filter="FAIL", want_match=False,
                           limit=lim_fail, output_dir=out_root / "fail" / rid,
                           max_per_doc=args.max_per_doc, per_day=per_day)
        print(f"      → saved {p} pass / {f} fail\n")
        summary.append((rid, p, f))

    client.close()

    print("=" * 64)
    print(f"{'Recipe':<28}{'PASS':>8}{'FAIL':>8}")
    print("-" * 64)
    tp = tf = 0
    for rid, p, f in summary:
        print(f"{rid:<28}{p:>8}{f:>8}"); tp += p; tf += f
    print("-" * 64)
    print(f"{'TOTAL':<28}{tp:>8}{tf:>8}")
    print("=" * 64)
    print(f"\nOutput: {out_root.resolve()}  (pass/ và fail/ tách riêng theo recipe)")


if __name__ == "__main__":
    main()
