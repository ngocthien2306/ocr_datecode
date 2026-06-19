"""
Compare classical CV vs YOLO OBB for bottle detection on 4 sample images.
Outputs a side-by-side visualization per image to tests/crop_output/bottle_compare/.
"""

import os
import sys
import cv2
import numpy as np
import onnxruntime as ort

REPO = "/home/demo/Source/ocr_datecode"
sys.path.insert(0, REPO)

IMAGES = [
    f"{REPO}/backend/uploads/inference_results/6a0d53cae10e970b7a50c76d/2026-05-20/40733814/fail_f0_20260520_095647734672_org.jpg",
    f"{REPO}/backend/uploads/inference_results/6a0d53cae10e970b7a50c76d/2026-05-20/40733814/fail_f0_20260520_095652648110_org.jpg",
    f"{REPO}/backend/uploads/inference_results/6a0d53cae10e970b7a50c76d/2026-05-20/40767171/fail_f0_20260520_075757115153_org.jpg",
    f"{REPO}/backend/uploads/inference_results/6a0d53cae10e970b7a50c76d/2026-05-20/40767171/fail_f0_20260520_142251354757_org.jpg",
]

OUT_DIR = f"{REPO}/tests/crop_output/bottle_compare"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Classical CV — vertical-edge + projection (robust to bright OR dark labels
# because both produce strong vertical edges against a dark blurred backdrop)
# ---------------------------------------------------------------------------
def detect_bottle_cv(bgr: np.ndarray):
    """
    Idea: the bottle is the only IN-FOCUS object — background is heavily
    blurred. Sharpness (local variance of Laplacian) lights up the bottle
    body even when its label is dark on dark background.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 1) local sharpness map via |Laplacian| smoothed in a 31x31 window
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    sharp = cv2.boxFilter(lap * lap, ddepth=cv2.CV_32F, ksize=(31, 31))
    sharp = np.sqrt(np.maximum(sharp, 0))

    # 2) normalize and threshold — pixels >35% of max sharpness are likely
    #    the in-focus object.
    s_norm = sharp / (sharp.max() + 1e-6)
    mask = (s_norm > 0.30).astype(np.uint8) * 255

    # 3) clean up
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    # 4) connected components, keep the largest one touching image center band
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if nlabels <= 1:
        return None, mask

    best = None
    best_score = 0.0
    for i in range(1, nlabels):
        x, y, bw, bh, area = stats[i]
        # bottle aspect ratio: taller than wide (h/w > 1.2) and reasonably tall
        if bh < 0.25 * h or bh < 1.2 * bw:
            continue
        # score = area * (closer to center is better, but not strict)
        center_dist = abs((x + bw / 2) - w / 2) / w
        score = area * (1.0 - 0.5 * center_dist)
        if score > best_score:
            best_score = score
            best = (x, y, x + bw, y + bh)

    return best, mask


# ---------------------------------------------------------------------------
# YOLO OBB ONNX  (model already has built-in NMS, output [cx,cy,w,h,conf,cls,a])
# ---------------------------------------------------------------------------
class YoloObb:
    def __init__(self, model_path: str, input_size=(320, 320)):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size  # (w, h)

    def _letterbox(self, image):
        iw, ih = self.input_size
        h, w = image.shape[:2]
        scale = min(iw / w, ih / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (nw, nh))
        padded = np.full((ih, iw, 3), 114, dtype=np.uint8)
        pt = (ih - nh) // 2
        pl = (iw - nw) // 2
        padded[pt:pt + nh, pl:pl + nw] = resized
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        return tensor[None], scale, (pt, pl)

    def predict(self, image, conf=0.25):
        tensor, scale, (pt, pl) = self._letterbox(image)
        out = self.session.run(None, {self.input_name: tensor})[0]  # 1x300x7
        out = out[0]
        # [cx, cy, w, h, conf, cls, angle]
        mask = (out[:, 4] >= conf) & (out[:, 4] > 0)
        det = out[mask]
        if det.shape[0] == 0:
            return []
        # rescale to original
        det[:, 0] = (det[:, 0] - pl) / scale
        det[:, 1] = (det[:, 1] - pt) / scale
        det[:, 2] /= scale
        det[:, 3] /= scale
        return det

    @staticmethod
    def obb_to_corners(cx, cy, w, h, angle):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        hw, hh = w / 2, h / 2
        return np.array([
            [cx + (-hw * cos_a + hh * sin_a), cy + (-hw * sin_a - hh * cos_a)],
            [cx + (hw * cos_a + hh * sin_a), cy + (hw * sin_a - hh * cos_a)],
            [cx + (hw * cos_a - hh * sin_a), cy + (hw * sin_a + hh * cos_a)],
            [cx + (-hw * cos_a - hh * sin_a), cy + (-hw * sin_a + hh * cos_a)],
        ], dtype=np.int32)


def label_panel(img, title):
    pad = 40
    out = np.zeros((img.shape[0] + pad, img.shape[1], 3), dtype=np.uint8)
    out[pad:] = img
    cv2.putText(out, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2)
    return out


def main():
    yolo_main = YoloObb(f"{REPO}/weights/yolo26_bottle_obb.onnx", input_size=(640, 640))
    yolo_alt = YoloObb(f"{REPO}/weights/yolo26n-ultralight-obb.onnx", input_size=(320, 320))

    summary = []
    for idx, path in enumerate(IMAGES, 1):
        img = cv2.imread(path)
        if img is None:
            print(f"[skip] cannot read {path}")
            continue

        # --- CV ---
        cv_img = img.copy()
        bbox_cv, cv_mask = detect_bottle_cv(img)
        if bbox_cv is not None:
            x1, y1, x2, y2 = bbox_cv
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 255), 4)
            cv2.putText(cv_img, "CV bottle", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv_status = "ok"
        else:
            cv_status = "FAIL"

        # --- YOLO --- run main model; if no det fall back to ultralight @ 320
        yolo_img = img.copy()
        det = yolo_main.predict(img, conf=0.25)
        used = "obb_m@640"
        if len(det) == 0:
            det = yolo_alt.predict(img, conf=0.05)
            used = "ultralight@320 (low-conf fallback)"
        n_det = len(det)
        if n_det:
            for d in det:
                cx, cy, w, h, conf, cls, a = d
                corners = YoloObb.obb_to_corners(cx, cy, w, h, a)
                cv2.polylines(yolo_img, [corners], True, (0, 255, 0), 4)
                cv2.putText(yolo_img, f"yolo {conf:.2f}",
                            tuple(corners[0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(yolo_img, used, (10, yolo_img.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)

        # --- side by side ---
        cvp = label_panel(cv_img, f"Classical CV [{cv_status}]")
        yp = label_panel(yolo_img, f"YOLO OBB  [{n_det} det]")
        side = np.hstack([cvp, yp])

        out_path = os.path.join(OUT_DIR, f"compare_{idx}.jpg")
        cv2.imwrite(out_path, side)
        summary.append((idx, os.path.basename(path), cv_status, n_det, out_path))
        print(f"[{idx}] cv={cv_status:>4}  yolo={n_det} det  ->  {out_path}")

    print("\n=== summary ===")
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
