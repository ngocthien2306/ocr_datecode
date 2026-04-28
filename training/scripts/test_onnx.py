#!/usr/bin/env python3
"""
Test ONNX model on a strip image: segment chars → predict each OK/NG.
Works with linear (logits) and projection (embedding/centroid) heads.

Usage:
    # Linear head (CE / focal / poly)
    python scripts/test_onnx.py --run runs/ce_baseline_... --image strip.png

    # Projection head (SupCon) — needs dataset to build centroids
    python scripts/test_onnx.py --run runs/supcon_128_... --dataset /data/output2 --image strip.png

    # Multiple strips
    python scripts/test_onnx.py --run runs/supcon_128_... --dataset /data/output2 --dir /path/to/strips/
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent.parent          # training/
sys.path.insert(0, str(HERE.parent / "desktop"))       # for char_segmenter
from char_segmenter import segment_characters

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ── preprocessing (same as test_strip.py) ──────────────────────────────────────

def preprocess(bgr, size, mean, std):
    h, w = bgr.shape[:2]
    s = size / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    canvas[(size-nh)//2:(size-nh)//2+nh, (size-nw)//2:(size-nw)//2+nw] = img
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - mean) / std).transpose(2, 0, 1).astype(np.float32)


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


# ── centroid builder (projection head) ─────────────────────────────────────────

def build_centroids(sess, input_name, feat_dim, dataset_root, multi_to_idx, size, mean, std,
                    cache_path: Path | None = None):
    if cache_path and cache_path.exists():
        centroids = np.load(str(cache_path))
        print(f"[centroids] loaded from cache: {cache_path}")
        return centroids

    print(f"[centroids] scanning {dataset_root} ...")
    sums   = np.zeros((len(multi_to_idx), feat_dim), dtype=np.float64)
    counts = np.zeros(len(multi_to_idx), dtype=np.int64)

    root = Path(dataset_root)
    for char_dir in sorted(root.iterdir()):
        if not char_dir.is_dir() or not char_dir.name.startswith("char_"):
            continue
        for sub in ("ok", "ng"):
            cls_key = f"{char_dir.name}__{sub}"
            if cls_key not in multi_to_idx:
                continue
            idx  = multi_to_idx[cls_key]
            imgs = [p for p in sorted((char_dir / sub).iterdir()) if p.suffix.lower() in VALID_EXT]
            if not imgs:
                continue
            batch = np.stack([preprocess(cv2.imread(str(p)), size, mean, std) for p in imgs])
            embs  = sess.run(None, {input_name: batch})[0]
            embs  = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
            sums[idx]  += embs.sum(axis=0)
            counts[idx] += len(imgs)

    centroids = (sums / np.maximum(counts[:, None], 1)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8

    if cache_path:
        np.save(str(cache_path), centroids)
        print(f"[centroids] {len(multi_to_idx)} classes built → saved to {cache_path}")
    else:
        print(f"[centroids] {len(multi_to_idx)} classes built")
    return centroids


# ── inference ───────────────────────────────────────────────────────────────────

def run_linear(sess, input_name, tensors):
    logits = sess.run(None, {input_name: tensors})[0]
    return softmax(logits)[:, 1]          # P_NG per crop


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def run_projection(sess, input_name, tensors, centroids, idx_to_ok_ng, temperature=5.0):
    embs = sess.run(None, {input_name: tensors})[0]
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    sims    = embs @ centroids.T
    ng_mask = np.array([idx_to_ok_ng[i] for i in range(len(idx_to_ok_ng))], dtype=bool)
    raw     = np.where(ng_mask, sims, -2.0).max(axis=1) - np.where(~ng_mask, sims, -2.0).max(axis=1)
    return sigmoid(raw * temperature)   # → [0, 1], same scale as linear head


# ── per-strip test ───────────────────────────────────────────────────────────────

def test_strip(image_path, sess, input_name, head, cfg_size, mean, std,
               centroids, idx_to_ok_ng, threshold, pad_w, pad_h, out_path):
    boxes, char_imgs, _, proc_img, _ = segment_characters(str(image_path))
    if not boxes:
        print(f"  [warn] no characters segmented")
        return

    # pad each crop before model
    def pad(img):
        return cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w,
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))

    tensors = np.stack([preprocess(pad(c), cfg_size, mean, std) for c in char_imgs])

    if head == "linear":
        p_ng = run_linear(sess, input_name, tensors)
    else:
        p_ng = run_projection(sess, input_name, tensors, centroids, idx_to_ok_ng, temperature=5.0)

    vis    = proc_img.copy()
    n_ng   = 0
    print(f"  {'idx':>3}  {'P_NG':>6}  {'conf%':>6}  verdict  bbox")
    for i, (box, p) in enumerate(zip(boxes, p_ng)):
        x, y, w, h = box
        ng    = p >= threshold
        n_ng += int(ng)
        color = (0, 0, 255) if ng else (0, 200, 0)
        cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
        cv2.putText(vis, f"{p*100:.0f}%", (x, max(15, y-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        print(f"  {i:>3}  {p:.4f}  {p*100:>5.1f}%   {'NG' if ng else 'OK':2}       {box}")

    cv2.imwrite(str(out_path), vis)
    print(f"  → {n_ng}/{len(boxes)} NG  (th={threshold:.3f})  saved: {out_path}")
    return n_ng, len(boxes)


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run",       required=True, help="Run dir with model.onnx + config.yaml + best.pt")
    ap.add_argument("--image",     default=None,  help="Single strip image")
    ap.add_argument("--dir",       default=None,  help="Folder of strip images")
    ap.add_argument("--dataset",   default=None,  help="Training dataset root (required for projection head)")
    ap.add_argument("--out-dir",   default="onnx_results", help="Output dir for annotated images")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--pad-w",     type=int,   default=6)
    ap.add_argument("--pad-h",     type=int,   default=6)
    ap.add_argument("--checkpoint", default="best.pt")
    args = ap.parse_args()

    if not args.image and not args.dir:
        ap.error("Need --image or --dir")

    run_dir = Path(args.run).resolve()
    cfg     = OmegaConf.load(run_dir / "config.yaml")
    SIZE    = cfg.data.image_size
    MEAN    = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD     = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    head    = cfg.model.head.type

    import torch
    ckpt         = torch.load(run_dir / args.checkpoint, map_location="cpu")
    multi_to_idx = ckpt.get("multi_to_idx", {})
    idx_to_ok_ng = {v: (0 if k.endswith("__ok") else 1) for k, v in multi_to_idx.items()}
    best_metrics = ckpt.get("metrics", {})

    if args.threshold is not None:
        TH = args.threshold
    elif head == "projection":
        # stored threshold is raw (sim_ng - sim_ok) scale → convert to sigmoid scale
        raw_th = best_metrics.get("threshold", 0.0)
        TH = float(sigmoid(np.array([raw_th]) * 5.0)[0])
    else:
        TH = best_metrics.get("threshold", 0.5)

    sess = ort.InferenceSession(
        str(run_dir / "model.onnx"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name
    feat_dim   = sess.get_outputs()[0].shape[-1]

    print(f"[model]  {run_dir.name}")
    print(f"         head={head}  size={SIZE}  threshold={TH:.4f}  pad=({args.pad_w},{args.pad_h})")

    # build centroids for projection head
    centroids = None
    if head == "projection":
        cache_path = run_dir / "centroids.npy"
        if not args.dataset and not cache_path.exists():
            print("[error] --dataset required to build centroids (first run)", file=sys.stderr)
            return 1
        centroids = build_centroids(sess, input_name, feat_dim,
                                    args.dataset, multi_to_idx, SIZE, MEAN, STD,
                                    cache_path=cache_path)

    # collect strip images
    if args.image:
        paths = [Path(args.image)]
    else:
        paths = sorted(p for p in Path(args.dir).iterdir() if p.suffix.lower() in VALID_EXT)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[strips] {len(paths)} image(s)\n" + "─" * 55)
    for p in paths:
        print(f"\n{p.name}")
        out_path = out_dir / f"result_{p.name}"
        test_strip(p, sess, input_name, head, SIZE, MEAN, STD,
                   centroids, idx_to_ok_ng, TH,
                   args.pad_w, args.pad_h, out_path)


if __name__ == "__main__":
    sys.exit(main())

# python scripts/test_onnx.py --run runs/ce_center_b0_convnextv2_pico_20260428-133136  --image /Users/ngocthien.ai/Source/Projects/ocr_datecode/ai_generate/test_result_1/cropped_region_40767169_2_115_1775438651.png
# python scripts/test_onnx.py --run runs/supcon_128_convnext_tiny_20260428-131108 --image /Users/ngocthien.ai/Source/Projects/ocr_datecode/ai_generate/test_result_1/cropped_region_40767169_2_115_1775438651.png
