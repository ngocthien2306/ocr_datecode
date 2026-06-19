"""
Repro: chiều ĐÚNG nhưng SuperPoint trả vùng text SAI → crop sai → OCR rác.

Chạy đúng path production:
  - MatcherFactory.create_matcher() build matcher từ recipe (gồm cap-crop template
    bằng detect_cap_and_crop, parse bbox text, load engine SuperPoint).
  - Cap-crop target cùng cách, rồi matcher.match_array() để lấy homography +
    transformed_bboxes (vùng text suy ra trên target).

In ra: confidence, inliers/total, so với matching_conf gate (0.09).
Vẽ: target cap-crop + transformed text bbox (đỏ) → để thấy bbox LỆCH khỏi text.

Tuỳ chọn: chạy lại với mask rim (inner_mask_ratio<1) + CLAHE để xem conf cải thiện.
"""
import sys, json
from pathlib import Path
from types import SimpleNamespace

ROOT = "/home/demo/Source/ocr_datecode"
sys.path.insert(0, f"{ROOT}/ai_services")
sys.path.insert(0, f"{ROOT}/ai_services/camera_management")

import cv2
import numpy as np
import pymongo
from bson import ObjectId

from camera_management.matchers.factory import MatcherFactory
from camera_management.preprocessing.cv_rotator import detect_cap_and_crop

RECIPE_ID = "6a32b270aa6d85bb6566c95b"
CAM_ID = "Camera004_Ktra code tren nap"
SP_ENGINE = f"{ROOT}/weights/pipeline_fp16_dynamic_512_512.engine"
OUT = Path(f"{ROOT}/tests/crop_output/repro_superpoint"); OUT.mkdir(parents=True, exist_ok=True)

TARGETS = [
    ("FAIL_garbage", f"{ROOT}/backend/uploads/inference_results/{RECIPE_ID}/2026-06-18/24026290/fail_f0_20260618_130714555803_org.jpg"),
    ("PASS_good",    f"{ROOT}/backend/uploads/inference_results/{RECIPE_ID}/2026-06-18/24026290/pass_f0_20260618_144829982743_org.jpg"),
]


def load_recipe():
    c = pymongo.MongoClient("mongodb://admin:password@localhost:27017/", serverSelectionTimeoutMS=5000)
    r = c["ocr_datecode_db"].recipes.find_one({"_id": ObjectId(RECIPE_ID)})
    ct = next(x for x in r["camera_templates"] if x.get("camera_id") == CAM_ID)
    return r, ct


def build_camera(recipe, ct):
    return SimpleNamespace(
        serial_number="24026290",
        function_type=ct.get("function_type", "Check_Color"),
        templates=ct.get("templates", []),
        cap_crop_method=recipe.get("cap_crop_method", "yolo_obb"),
        cap_rotation_method=recipe.get("cap_rotation_method", "yolo_obb"),
        match_erosion_enabled=recipe.get("match_erosion_enabled", False),
        match_erosion_kernel_w=recipe.get("match_erosion_kernel_w", 80),
        match_erosion_kernel_h=recipe.get("match_erosion_kernel_h", 1),
        match_erosion_iterations=recipe.get("match_erosion_iterations", 1),
    )


def draw_bboxes(img, bboxes):
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    colors = {"template": (0, 255, 0), "text": (0, 0, 255)}
    for b in bboxes:
        pts = np.array(b["points"], dtype=np.int32)
        cv2.polylines(out, [pts], True, colors.get(b["type"], (255, 0, 0)), 2)
        if b.get("text"):
            p = pts[0]
            cv2.putText(out, b["text"], (int(p[0]), int(p[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out


def main():
    recipe, ct = load_recipe()
    print(f"matching_conf gate = {recipe.get('matching_conf')}  cap_crop={recipe.get('cap_crop_method')}\n")
    cam = build_camera(recipe, ct)

    factory = MatcherFactory(engine_path=SP_ENGINE,
                             temp_dir=Path("/tmp/repro_matcher"),
                             backend_dir=Path(f"{ROOT}/backend"))
    matcher = factory.create_matcher(cam, cam.templates[0], 0, verbose=True)
    print("matcher:", matcher is not None,
          "| template_gray:", None if matcher is None else matcher.template_gray.shape,
          "| n other_bboxes:", None if matcher is None else len(matcher.other_bboxes))
    if matcher is None:
        return

    # crop_area (template fell back to crop_area flow → match target cùng framing).
    ca = next(a for a in cam.templates[0]["annotations"] if a["type"] == "crop_area")
    W, H = 1600, 1200
    cx1, cy1 = int(ca["x"] * W), int(ca["y"] * H)
    cx2, cy2 = int((ca["x"] + ca["width"]) * W), int((ca["y"] + ca["height"]) * H)
    print(f"crop_area px = [{cy1}:{cy2}, {cx1}:{cx2}] (template_gray={matcher.template_gray.shape})\n")

    for tag, path in TARGETS:
        img = cv2.imread(path)
        if img is None:
            print(f"[{tag}] read fail {path}"); continue
        target_crop = img[cy1:cy2, cx1:cx2]
        # min_confidence=0.0 để LẤY bbox kể cả khi conf thấp (xem nó lệch ra sao)
        res = matcher.engine.match_batch(
            target_imgs=[target_crop], templates=[matcher.template_config],
            score_threshold=0.3, ransac_threshold=5.0, min_confidence=0.0,
        )["results"][0]
        conf = res.get("confidence", 0.0); inl = res.get("inliers", 0); tot = res.get("total_matches", 0)
        gate = float(recipe.get("matching_conf") or 0.2)
        passed = "PASS gate" if conf >= gate else "REJECTED by gate"
        print(f"[{tag}] conf={conf:.3f} inliers={inl}/{tot} vs gate={gate} -> {passed} "
              f"| n_bbox={len(res.get('transformed_bboxes', []))}")
        viz = draw_bboxes(target_crop, res.get("transformed_bboxes", []))
        cv2.putText(viz, f"{tag} conf={conf:.3f} inl={inl}/{tot} gate={gate} {passed}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(str(OUT / f"sp_{tag}.jpg"), viz)
    print(f"\nviz -> {OUT}")


if __name__ == "__main__":
    main()
