#!/usr/bin/env python3
"""
Lấy 1 inference_result từ MongoDB theo _id và in ra (hoặc lưu file JSON),
và/hoặc crop các vùng `text`/`datecode` ra 1 folder để xem.

Lưu ý: collection `inference_results` lưu _id dưới dạng string (không phải ObjectId),
nên ta query theo string trước, rồi fallback sang ObjectId nếu không thấy.

Usage:
  python tests/get_inference_by_id.py 6a2cec50b3a81f6f2705ae1d
  python tests/get_inference_by_id.py 6a2cec50b3a81f6f2705ae1d --summary
  python tests/get_inference_by_id.py 6a2cec50b3a81f6f2705ae1d --out result.json
  python tests/get_inference_by_id.py 6a2cec50b3a81f6f2705ae1d --crop
  python tests/get_inference_by_id.py 6a2cec50b3a81f6f2705ae1d --crop --crop_dir tests/crop_output/<id>
"""

import os
import re
import json
import argparse

import cv2
from bson import ObjectId
from bson.json_util import dumps as bson_dumps
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://admin:password@localhost:27018/")
DATABASE_NAME = os.getenv("DATABASE_NAME", "ocr_datecode_db")
IMAGE_BASE = "/home/demo/Source/ocr_datecode/backend/uploads"
HANDLE_TYPES = {"text", "datecode"}


def fetch(result_id: str):
    client = MongoClient(MONGODB_URL)
    try:
        coll = client[DATABASE_NAME]["inference_results"]

        # _id lưu dạng string -> thử string trước
        doc = coll.find_one({"_id": result_id})

        # fallback: nếu lưu dạng ObjectId
        if doc is None and ObjectId.is_valid(result_id):
            doc = coll.find_one({"_id": ObjectId(result_id)})

        return doc
    finally:
        client.close()


def safe(text: str) -> str:
    return re.sub(r'[^\w\-.]', '_', str(text).strip())[:40] or "empty"


def _warp_crop_text_region(img, pts):
    """Crop GIỐNG HỆT pipeline thật: perspective-warp polygon 4 điểm về chữ nhật.

    Pipeline production dùng ocr_utils.crop_text_region (warpPerspective theo
    độ dài cạnh thật), KHÔNG phải axis-aligned bbox. Đây là điểm khác biệt khiến
    'crop nhìn đúng' (axis-aligned) nhưng OCR thật lại sai.
    """
    import sys
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ai_services_path = os.path.join(repo_root, "ai_services")
    if ai_services_path not in sys.path:
        sys.path.insert(0, ai_services_path)
    from camera_management.ocr_utils import crop_text_region
    return crop_text_region(img, pts)


def collect_crops(doc: dict, warp: bool = False) -> list:
    """Đọc ảnh _org của từng frame, crop các vùng text/datecode.

    Args:
        warp: nếu True crop bằng perspective-warp y hệt pipeline (crop_text_region);
              nếu False crop axis-aligned bbox (chỉ để xem nhanh, KHÔNG giống OCR thật).

    Trả về list dict:
      {crop, cam_id, frame_name, rtype, idx,
       expected, recognized_db, match_db, conf_db}
    """
    items = []

    for cam in doc.get("camera_results", []):
        cam_id = cam.get("camera_id", "cam")

        for frame in cam.get("frames", []):
            if not frame.get("image_path"):
                continue

            # dùng ảnh gốc (_org) thay vì ảnh đã vẽ overlay (_viz)
            img_path = os.path.join(IMAGE_BASE, frame["image_path"].replace("_viz", "_org"))
            img = cv2.imread(img_path)
            if img is None:
                print(f"  [WARN] Không đọc được ảnh: {img_path}")
                continue

            regions = frame.get("detected_regions", [])
            text_verif = frame.get("text_verification") or {}
            frame_name = str(frame.get("template_name", "frame")).replace(" ", "_")

            # map annotation_index -> verification result
            verif_map = {
                r["annotation_idx"]: r
                for r in text_verif.get("results", [])
                if "annotation_idx" in r
            }

            for i, region in enumerate(regions):
                rtype = region.get("type", "")
                if rtype not in HANDLE_TYPES:
                    continue

                pts = region.get("points", [])
                if not pts:
                    continue

                if warp and len(pts) >= 4:
                    # GIỐNG pipeline thật: perspective-warp polygon
                    crop = _warp_crop_text_region(img, pts)
                else:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    x1 = max(0, int(min(xs)))
                    y1 = max(0, int(min(ys)))
                    x2 = min(img.shape[1], int(max(xs)))
                    y2 = min(img.shape[0], int(max(ys)))
                    crop = img[y1:y2, x1:x2]
                if crop is None or crop.size == 0:
                    continue

                verif = verif_map.get(region.get("annotation_index"))
                items.append({
                    "crop": crop,
                    "cam_id": cam_id,
                    "frame_name": frame_name,
                    "rtype": rtype,
                    "idx": i,
                    "expected": (verif.get("expected") if verif else region.get("text", "")) or "",
                    "recognized_db": verif.get("recognized", "") if verif else None,
                    "match_db": verif.get("match") if verif else None,
                    "conf_db": verif.get("confidence", 0.0) if verif else None,
                })

    return items


def crop_record(doc: dict, out_dir: str, warp: bool = False):
    """Crop tất cả vùng text/datecode của record ra out_dir.

    Tên file: {cam}_{frame}_{type}_{i}_exp[expected]_got[recognized]_{PASS|FAIL}_c{conf}.jpg
    """
    os.makedirs(out_dir, exist_ok=True)
    items = collect_crops(doc, warp=warp)

    for it in items:
        expected = safe(it["expected"])
        if it["recognized_db"] is not None:
            recognized = safe(it["recognized_db"])
            match = "PASS" if it["match_db"] else "FAIL"
            suffix = f"_exp[{expected}]_got[{recognized}]_{match}_c{it['conf_db']:.2f}"
        else:
            suffix = f"_exp[{expected}]_no_verif"

        out_path = os.path.join(
            out_dir, f"{it['cam_id']}_{it['frame_name']}_{it['rtype']}_{it['idx']}{suffix}.jpg"
        )
        cv2.imwrite(out_path, it["crop"])
        print(f"  {os.path.basename(out_path)}")

    print(f"\n✅ Đã crop {len(items)} vùng -> {out_dir}")


def reocr_smtr(doc: dict, save_dir: str = None, warp: bool = False):
    """Gọi lại OCR bằng model SMTR (predict batch) trên các crop và so sánh
    với recognized có sẵn trong DB."""
    import sys
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ai_services_path = os.path.join(repo_root, "ai_services")
    if ai_services_path not in sys.path:
        sys.path.insert(0, ai_services_path)

    from camera_management.ocr import OCRBackendFactory, OCRBackendType
    from camera_management.ocr.factory import OCRModelType

    items = collect_crops(doc, warp=warp)
    if not items:
        print("Không có vùng text/datecode nào để OCR.")
        return
    print(f"Crop mode: {'WARP (giống pipeline thật)' if warp else 'AXIS-ALIGNED bbox'}")

    print(f"Khởi tạo OCR backend SMTR (AUTO → ưu tiên TensorRT)...")
    backend = OCRBackendFactory.create(
        backend_type=OCRBackendType.AUTO,
        model_type=OCRModelType.SMTR,
    )
    if backend is None or not backend.is_available:
        print("❌ Không khởi tạo được SMTR backend.")
        raise SystemExit(1)
    print(f"✅ Backend: {backend.backend_name}\n")

    crops = [it["crop"] for it in items]
    # SMTR dual-head: mỗi phần tử = [(gtc_text, gtc_conf, chars), (ctc_text, ctc_conf, chars)]
    results = backend.recognize_batch(crops)

    print(f"{'region':<22} {'expected':<18} {'DB recog':<14} "
          f"{'SMTR-GTC':<14} {'cGTC':>5}  {'SMTR-CTC':<14} {'cCTC':>5}  match")
    print("-" * 110)

    for it, res in zip(items, results):
        gtc_text, gtc_conf = res[0][0], res[0][1]
        ctc_text, ctc_conf = res[1][0], res[1][1]
        region = f"{it['cam_id']}/{it['frame_name']}/{it['rtype']}{it['idx']}"
        db_recog = it["recognized_db"] if it["recognized_db"] is not None else "-"

        exp = str(it["expected"])
        # so khớp SMTR (ưu tiên GTC head) với expected
        ok = "✓" if gtc_text == exp else ("~" if ctc_text == exp else "✗")

        print(f"{region:<22.22} {exp:<18.18} {str(db_recog):<14.14} "
              f"{gtc_text:<14.14} {gtc_conf:>5.2f}  {ctc_text:<14.14} {ctc_conf:>5.2f}  {ok}")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fn = (f"{it['cam_id']}_{it['frame_name']}_{it['rtype']}_{it['idx']}"
                  f"_exp[{safe(exp)}]_db[{safe(str(db_recog))}]"
                  f"_smtr[{safe(gtc_text)}].jpg")
            cv2.imwrite(os.path.join(save_dir, fn), it["crop"])

    if save_dir:
        print(f"\n✅ Đã lưu crop kèm kết quả SMTR -> {save_dir}")


def print_summary(doc: dict):
    print(f"_id              : {doc.get('_id')}")
    print(f"recipe_id        : {doc.get('recipe_id')}")
    print(f"recipe_name      : {doc.get('recipe_name')}")
    print(f"product_pass_fail: {doc.get('product_pass_fail')}")
    print(f"timestamp        : {doc.get('timestamp')}")
    print(f"created_at       : {doc.get('created_at')}")

    cams = doc.get("camera_results", [])
    print(f"cameras          : {len(cams)}")
    for cam in cams:
        frames = cam.get("frames", [])
        print(f"  - camera {cam.get('camera_id')} | {cam.get('pass_fail')} | frames: {len(frames)}")
        for fr in frames:
            tv = fr.get("text_verification") or {}
            print(
                f"      frame: {fr.get('template_name')} "
                f"| img: {fr.get('image_path')} "
                f"| text_match: {tv.get('all_match')}"
            )


def main():
    parser = argparse.ArgumentParser(description="Fetch an inference_result by _id")
    parser.add_argument("id", help="Inference result _id")
    parser.add_argument("--out", help="Ghi document ra file JSON")
    parser.add_argument("--summary", action="store_true", help="Chỉ in tóm tắt thay vì full JSON")
    parser.add_argument("--crop", action="store_true", help="Crop các vùng text/datecode ra folder")
    parser.add_argument("--crop_dir", help="Folder lưu crop (mặc định: tests/crop_output/<id>)")
    parser.add_argument("--reocr", action="store_true",
                        help="OCR lại các vùng bằng model SMTR (predict batch) và so sánh với DB")
    parser.add_argument("--reocr_dir", help="Lưu crop kèm kết quả SMTR vào folder này")
    parser.add_argument("--warp", action="store_true",
                        help="Crop bằng perspective-warp GIỐNG pipeline thật (crop_text_region) "
                             "thay vì axis-aligned bbox")
    args = parser.parse_args()

    doc = fetch(args.id)
    if doc is None:
        print(f"❌ Không tìm thấy inference_result với _id = {args.id}")
        print(f"   (DB: {DATABASE_NAME} @ {MONGODB_URL})")
        raise SystemExit(1)

    if args.reocr:
        print(f"Re-OCR (SMTR) các vùng text/datecode của {args.id}:")
        reocr_smtr(doc, save_dir=args.reocr_dir, warp=args.warp)
        return

    if args.crop:
        crop_dir = args.crop_dir or os.path.join("tests/crop_output", str(args.id))
        print(f"Crop vùng text/datecode của {args.id} ({'WARP' if args.warp else 'bbox'}):")
        crop_record(doc, crop_dir, warp=args.warp)
        return

    if args.summary:
        print_summary(doc)
        return

    # bson_dumps xử lý được ObjectId/datetime
    pretty = json.dumps(json.loads(bson_dumps(doc)), indent=2, ensure_ascii=False)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(pretty)
        print(f"✅ Đã ghi {args.out} ({len(pretty)} bytes)")
    else:
        print(pretty)


if __name__ == "__main__":
    main()
