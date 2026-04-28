#!/usr/bin/env python3
"""
Quick test: segment characters from a text strip, classify each as OK/NG.

Usage
    python test_strip.py path/to/strip.png [--out result.png]

Reuses desktop/char_segmenter.py + training/ok_ng_model.onnx + model_meta.json.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "desktop"))
from char_segmenter import segment_characters


def preprocess(bgr, size, mean, std):
    h, w = bgr.shape[:2]
    s = size / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = img
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - mean) / std).transpose(2, 0, 1)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", default=str(HERE / "ok_ng_model.onnx"))
    ap.add_argument("--meta",  default=str(HERE / "model_meta.json"))
    ap.add_argument("--out",   default="strip_result.png")
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--pad-w", type=int, default=4, help="Pixels to pad left+right of each char crop")
    ap.add_argument("--pad-h", type=int, default=2, help="Pixels to pad top+bottom of each char crop")
    args = ap.parse_args()

    meta = json.load(open(args.meta))
    SIZE = meta["image_size"]
    TH = args.threshold if args.threshold is not None else meta["threshold"]
    mean = np.array(meta["normalization"]["mean"], dtype=np.float32)
    std  = np.array(meta["normalization"]["std"],  dtype=np.float32)

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    boxes, char_imgs, _, proc_img, _ = segment_characters(args.image)
    if not boxes:
        print(f"No characters segmented from {args.image}")
        return 1
    print(f"Segmented {len(boxes)} characters. Predicting...")

    # Optionally pad each crop before feeding to model
    def pad_crop(img, pw, ph):
        if pw == 0 and ph == 0:
            return img
        return cv2.copyMakeBorder(img, ph, ph, pw, pw, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    padded_imgs = [pad_crop(c, args.pad_w, args.pad_h) for c in char_imgs]

    # Batch predict
    tensors = np.stack([preprocess(c, SIZE, mean, std) for c in padded_imgs]).astype(np.float32)
    logits = sess.run(None, {"input": tensors})[0]
    p_ng = softmax(logits, axis=-1)[:, 1]

    # Draw on a copy of the processed image (same coord system as boxes)
    vis = proc_img.copy()
    n_ng = 0
    print(f"\n{'idx':>3}  {'P_NG':>6}  verdict  bbox")
    for i, (box, p) in enumerate(zip(boxes, p_ng)):
        x, y, w, h = box
        ng = p >= TH
        n_ng += int(ng)
        color = (0, 0, 255) if ng else (0, 200, 0)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis, f"{p:.2f}", (x, max(15, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        print(f"{i:>3}  {p:.4f}   {'NG' if ng else 'OK'}      {box}")

    cv2.imwrite(args.out, vis)
    print(f"\n{n_ng}/{len(boxes)} predicted NG  (threshold {TH:.3f})")
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
