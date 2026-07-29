#!/usr/bin/env python3
"""
Benchmark ONNX Runtime (CUDA EP) vs the rebuilt TensorRT engines for the OCR
recognition models in languages/english/, using the sample crops in
test_image/ocr/.

Usage:
    conda activate vision
    python tests/benchmark_ocr_onnx_vs_trt.py
    python tests/benchmark_ocr_onnx_vs_trt.py --iters 50 --warmup 10
"""
import argparse
import glob
import os
import sys
import time

import cv2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "ai_services"))

from camera_management.text_recognizer_openocr_trt import TextRecognizerOpenOCRTRT
from camera_management.text_recognizer_openocr_onnx import TextRecognizerOpenOCRONNX
from camera_management.text_recognizer_smtr_trt import TextRecognizerSMTRTRT
from camera_management.text_recognizer_smtr_onnx import TextRecognizerSMTRONNX

LANG_DIR = os.path.join(REPO_ROOT, "languages", "english")


def bench(name, model, imgs, warmup, iters):
    for img in imgs[:warmup]:
        model.recognize(img)

    t0 = time.perf_counter()
    n = 0
    for _ in range(iters):
        for img in imgs:
            model.recognize(img)
            n += 1
    dt = time.perf_counter() - t0

    ms_per_img = dt / n * 1000
    print(f"  {name:12s} {n:5d} calls in {dt:6.2f}s -> {ms_per_img:6.2f} ms/img  ({n/dt:7.1f} img/s)")
    return ms_per_img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-dir", default=os.path.join(REPO_ROOT, "test_image", "ocr"))
    ap.add_argument("--iters", type=int, default=20, help="passes over all images (after warmup)")
    ap.add_argument("--warmup", type=int, default=5, help="warmup calls before timing")
    args = ap.parse_args()

    img_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.png")))
    if not img_paths:
        print(f"No .png images found in {args.image_dir}", file=sys.stderr)
        sys.exit(1)
    imgs = [cv2.imread(p) for p in img_paths]
    print(f"{len(imgs)} images, {args.iters} passes ({len(imgs) * args.iters} calls/model), "
          f"{args.warmup} warmup calls\n")

    print("=== openocr ===")
    onnx_ms = bench(
        "onnx (cuda)",
        TextRecognizerOpenOCRONNX(os.path.join(LANG_DIR, "openocr_rec_model.onnx"),
                                   os.path.join(LANG_DIR, "ppocr_keys_v1.txt")),
        imgs, args.warmup, args.iters,
    )
    trt_ms = bench(
        "trt",
        TextRecognizerOpenOCRTRT(os.path.join(LANG_DIR, "openocr_rec_model_batch.engine"),
                                  os.path.join(LANG_DIR, "ppocr_keys_v1.txt")),
        imgs, args.warmup, args.iters,
    )
    print(f"  -> TRT is {onnx_ms / trt_ms:.2f}x faster than ONNX\n")

    print("=== smtr ===")
    onnx_ms = bench(
        "onnx (cuda)",
        TextRecognizerSMTRONNX(os.path.join(LANG_DIR, "rec_smtr_fp16.onnx"),
                                os.path.join(LANG_DIR, "EN_symbol_dict.txt"), device="cuda"),
        imgs, args.warmup, args.iters,
    )
    trt_ms = bench(
        "trt",
        TextRecognizerSMTRTRT(os.path.join(LANG_DIR, "rec_smtr_fp16.engine"),
                               os.path.join(LANG_DIR, "EN_symbol_dict.txt")),
        imgs, args.warmup, args.iters,
    )
    print(f"  -> TRT is {onnx_ms / trt_ms:.2f}x faster than ONNX\n")


if __name__ == "__main__":
    main()
