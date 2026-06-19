"""
Repro TRUNG THỰC framing production: cap-crop ~square quanh nắp (yolo_obb) cho CẢ
template lẫn target, rồi SuperPoint match. Tái hiện việc conf tụt + bbox text lệch.

Khác với repro_superpoint_region.py (framing crop_area rộng → conf ~0.37, không lỗi),
bản này crop chặt quanh nắp đen (giống log production crop=(776,776)) để lộ nguyên nhân:
nắp tối, ít texture, ĐỐI XỨNG XOAY → SuperPoint ít keypoint ổn định → homography bất ổn.
"""
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = "/home/demo/Source/ocr_datecode"
sys.path.insert(0, f"{ROOT}/ai_services")
sys.path.insert(0, f"{ROOT}/ai_services/camera_management")

from inference_engine_shared import TemplateConfig, SuperPointEngineTRT
from camera_management.preprocessing.obb_rotator import OBBRotationService
from camera_management.preprocessing.cv_rotator import apply_cap_crop, detect_cap_circle

SP_ENGINE = f"{ROOT}/weights/pipeline_fp16_dynamic_512_512.engine"
OBB_ENGINE = f"{ROOT}/weights/best_bottle_m.engine"
TMPL = f"{ROOT}/backend/uploads/templates/589adc6a-45db-4bc5-91e5-8f82ff100640.jpg"
OUT = Path(f"{ROOT}/tests/crop_output/repro_sp_capcrop"); OUT.mkdir(parents=True, exist_ok=True)
W, H = 1600, 1200

# recipe annotations (normalized, top-left x/y + w/h) — giống _parse_rectangle_bbox
ANN = {
    "template": (0.5856209113561157, 0.5064173841279782, 0.26029559135437014, 0.11009730021158855),
    "JUN 17 2029": (0.5840375383012094, 0.5421565810980444, 0.16179579734802246, 0.036563297912680294),
    "06169-13019-V": (0.5893510380007605, 0.5673387596510668, 0.17590209007263186, 0.03768839518229167),
}
TARGETS = [
    ("FAIL_garbage", f"{ROOT}/backend/uploads/inference_results/6a32b270aa6d85bb6566c95b/2026-06-18/24026290/fail_f0_20260618_130714555803_org.jpg"),
    ("PASS_good",    f"{ROOT}/backend/uploads/inference_results/6a32b270aa6d85bb6566c95b/2026-06-18/24026290/pass_f0_20260618_144829982743_org.jpg"),
]


def px_pts(norm):
    x, y, w, h = norm
    x1, y1 = int(x * W), int(y * H)
    x2, y2 = int((x + w) * W), int((y + h) * H)
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def shift(pts, ox, oy):
    return [[p[0] - ox, p[1] - oy] for p in pts]


def main():
    obb = OBBRotationService(engine_path=OBB_ENGINE, conf_threshold=0.25, inverse_transform=False)
    engine = SuperPointEngineTRT.get_instance(SP_ENGINE, verbose=False)

    # ── Cap-crop TEMPLATE (cap circle via OBB, vì HoughCircles fail trên template) ──
    tmpl = cv2.imread(TMPL)
    _, _, cap_t = obb.rotate_frame(tmpl, frame_tag="template")
    if cap_t is None:
        cap_t = detect_cap_circle(tmpl)
    print("template cap circle:", cap_t)
    tcrop, (tx1, ty1, tx2, ty2) = apply_cap_crop(tmpl, cap_t, margin_ratio=0.10)
    print(f"template cap-crop = {tcrop.shape[:2]} origin=({tx1},{ty1})")

    tmpl_bbox = {"type": "template", "points": shift(px_pts(ANN["template"]), tx1, ty1), "conf": 0.8}
    others = [{"type": "text", "text": t, "points": shift(px_pts(ANN[t]), tx1, ty1), "conf": 0.5}
              for t in ("JUN 17 2029", "06169-13019-V")]
    tcfg = TemplateConfig(template_path=TMPL, template_img=tcrop,
                          template_gray=cv2.cvtColor(tcrop, cv2.COLOR_BGR2GRAY),
                          template_bbox=tmpl_bbox, other_bboxes=others, scale=1.0,
                          annotations=[tmpl_bbox] + others)

    for tag, path in TARGETS:
        img = cv2.imread(path)
        if img is None:
            print(f"[{tag}] read fail"); continue
        cap_hough = detect_cap_circle(img)
        _, _, cap_obb = obb.rotate_frame(img, frame_tag=tag)  # cap circle như production
        print(f"[{tag}] cap_hough={cap_hough}  cap_obb={cap_obb}")
        # So sánh 2 nguồn cap circle: HoughCircles vs OBB (production dùng OBB)
        for src, cap in [("hough", cap_hough), ("obb", cap_obb)]:
            if cap is None:
                print(f"   [{src}] cap None"); continue
            gcrop = apply_cap_crop(img, cap, margin_ratio=0.10)
            if gcrop is None:
                print(f"   [{src}] crop None"); continue
            target_crop = gcrop[0]
            res = engine.match_batch(target_imgs=[target_crop], templates=[tcfg],
                                     score_threshold=0.3, ransac_threshold=5.0,
                                     min_confidence=0.0)["results"][0]
            conf, inl, tot = res.get("confidence", 0), res.get("inliers", 0), res.get("total_matches", 0)
            gate = 0.09
            verdict = "PASS gate" if conf >= gate else "REJECT(FAIL)"
            print(f"   [{src}] crop={target_crop.shape[:2]} conf={conf:.3f} inl={inl}/{tot} "
                  f"r={cap[2]:.0f} -> {verdict}")
            viz = target_crop.copy()
            for b in res.get("transformed_bboxes", []):
                pts = np.array(b["points"], np.int32)
                col = (0, 255, 0) if b["type"] == "template" else (0, 0, 255)
                cv2.polylines(viz, [pts], True, col, 2)
            cv2.putText(viz, f"{tag}/{src} conf={conf:.3f} inl={inl}/{tot} {verdict}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.imwrite(str(OUT / f"sp_{tag}_{src}.jpg"), viz)
    print(f"\nviz -> {OUT}")


if __name__ == "__main__":
    main()
