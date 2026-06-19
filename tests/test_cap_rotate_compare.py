"""
Compare YOLO OBB vs pure-CV cap rotation on the 10 most recent org images.

Output: side-by-side viz per image at tests/crop_output/cap_rotate_compare/
  ┌─────────┬─────────┬─────────┐
  │ ORIG    │ YOLO    │ CV      │
  └─────────┴─────────┴─────────┘
"""

import sys
from pathlib import Path

sys.path.insert(0, "/home/demo/Source/ocr_datecode/backend")

import cv2
import numpy as np

from app.services.rotate_obb_service import get_rotate_obb_model, rotate_frame
from app.services.rotate_cv_service import rotate_frame_cv

IMAGES = [
    "/home/demo/Source/ocr_datecode/backend/uploads/inference_results/6a0d53cae10e970b7a50c76d/2026-05-20/24026290/fail_f0_20260520_223159334935_org.jpg",
    "/home/demo/Source/ocr_datecode/backend/uploads/inference_results/6a0d53cae10e970b7a50c76d/2026-05-20/24026290/fail_f0_20260520_223155626640_org.jpg",
    "/home/demo/Source/ocr_datecode/backend/uploads/inference_results/6a0d53cae10e970b7a50c76d/2026-05-20/24026290/fail_f0_20260520_223152866493_org.jpg",
    "/home/demo/Source/ocr_datecode/backend/uploads/inference_results/6a0d53cae10e970b7a50c76d/2026-05-20/24026290/fail_f0_20260520_223151797748_org.jpg",
    "/home/demo/Source/ocr_datecode/backend/uploads/inference_results/6a0d53cae10e970b7a50c76d/2026-05-20/24026290/fail_f0_20260520_223150860900_org.jpg",
    "/home/demo/Source/ocr_datecode/backend/uploads/inference_results/6a0d53cae10e970b7a50c76d/2026-05-20/24026290/fail_f0_20260520_223150450946_org.jpg",
    "/home/demo/Source/ocr_datecode/backend/uploads/inference_results/6a0d53cae10e970b7a50c76d/2026-05-20/24026290/fail_f0_20260520_223150034921_org.jpg",
    "/home/demo/Source/ocr_datecode/backend/uploads/inference_results/6a0d53cae10e970b7a50c76d/2026-05-20/24026290/fail_f0_20260520_223149614159_org.jpg",
    "/home/demo/Source/ocr_datecode/backend/uploads/inference_results/6a0d53cae10e970b7a50c76d/2026-05-20/24026290/fail_f0_20260520_223148629967_org.jpg",
    "/home/demo/Source/ocr_datecode/backend/uploads/inference_results/6a0d53cae10e970b7a50c76d/2026-05-20/24026290/fail_f0_20260520_223148121458_org.jpg",
]

OUT_DIR = Path("/home/demo/Source/ocr_datecode/tests/crop_output/cap_rotate_compare")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def banner(img: np.ndarray, txt: str) -> np.ndarray:
    h_b = 40
    out = np.zeros((img.shape[0] + h_b, img.shape[1], 3), dtype=np.uint8)
    out[h_b:] = img
    cv2.putText(out, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out


def main():
    obb_model = get_rotate_obb_model()
    print(f"OBB model loaded: {obb_model is not None}")

    for idx, path in enumerate(IMAGES, 1):
        img = cv2.imread(path)
        if img is None:
            print(f"[{idx}] read fail: {path}")
            continue

        out_yolo = rotate_frame(img, obb_model) if obb_model else img.copy()
        out_cv = rotate_frame_cv(img)

        # Resize to common height for side-by-side
        H = 600
        def fit(im):
            scale = H / im.shape[0]
            return cv2.resize(im, (int(im.shape[1] * scale), H))

        side = np.hstack([
            banner(fit(img), f"#{idx} ORIGINAL"),
            banner(fit(out_yolo), "YOLO OBB"),
            banner(fit(out_cv), "CV (HoughCircles + projection + shape-match)"),
        ])
        out_path = OUT_DIR / f"compare_{idx:02d}.jpg"
        cv2.imwrite(str(out_path), side)
        print(f"[{idx}] saved {out_path}")


if __name__ == "__main__":
    main()
