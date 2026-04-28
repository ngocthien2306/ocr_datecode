#!/usr/bin/env python3
"""
Template vs Target embedding similarity test.

Usage:
    python test_embedding.py --run runs/supcon_128_efficientnet_b3_20260428-180622
    python test_embedding.py --run runs/arcface_128_b3_20260428-190614 --n-template 3
    python test_embedding.py --run runs/supcon_128_efficientnet_b3_20260428-180622 \\
        --dataset /Users/ngocthien.ai/Downloads/output2 --chars A B 0 1

Head types:
    projection  → embedding from output[0] (L2-normalized)
    arcface     → cos_sim from output[1]   (used as feature vector)
    linear      → softmax(logits)          (limited, not designed for similarity)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from omegaconf import OmegaConf

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(bgr: np.ndarray, size: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    s = size / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    canvas[(size-nh)//2:(size-nh)//2+nh, (size-nw)//2:(size-nw)//2+nw] = img
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1).astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract(sess: ort.InferenceSession, head: str, paths: List[str],
            size: int, batch_size: int = 64) -> np.ndarray:
    """Return (N, D) embedding matrix for the given paths."""
    input_name = sess.get_inputs()[0].name
    all_embs: List[np.ndarray] = []

    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        tensors = []
        for p in chunk:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[warn] cannot read {p}", file=sys.stderr)
                tensors.append(np.zeros((3, size, size), dtype=np.float32))
            else:
                tensors.append(preprocess(img, size))
        x = np.stack(tensors).astype(np.float32)

        if head == "projection":
            emb = sess.run(None, {input_name: x})[0]          # (B, D) L2-normalized
        elif head == "arcface":
            emb = sess.run(None, {input_name: x})[1]          # (B, C) cos_sim
        else:  # linear
            logits = sess.run(None, {input_name: x})[0]
            emb = softmax(logits)                              # (B, num_classes)

        # L2-normalize so cosine sim = dot product
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        emb = emb / norms
        all_embs.append(emb)

    return np.concatenate(all_embs, axis=0)


# ---------------------------------------------------------------------------
# Dataset scan
# ---------------------------------------------------------------------------

def scan_char(char_dir: Path) -> Tuple[List[str], List[str]]:
    """Return (ok_paths, ng_paths) for a char_* directory."""
    ok_paths, ng_paths = [], []
    for p in sorted((char_dir / "ok").iterdir()) if (char_dir / "ok").is_dir() else []:
        if p.suffix.lower() in VALID_EXT:
            ok_paths.append(str(p))
    for p in sorted((char_dir / "ng").iterdir()) if (char_dir / "ng").is_dir() else []:
        if p.suffix.lower() in VALID_EXT:
            ng_paths.append(str(p))
    return ok_paths, ng_paths


# ---------------------------------------------------------------------------
# Per-char similarity test
# ---------------------------------------------------------------------------

def test_char(char: str, ok_paths: List[str], ng_paths: List[str],
              sess: ort.InferenceSession, head: str, size: int,
              n_template: int) -> Dict:
    if len(ok_paths) < n_template + 1:
        return None

    template_paths = ok_paths[:n_template]
    test_ok_paths  = ok_paths[n_template:]

    # Extract
    t_embs  = extract(sess, head, template_paths, size)     # (n_template, D)
    ok_embs = extract(sess, head, test_ok_paths,  size)     # (N_ok, D)
    ng_embs = extract(sess, head, ng_paths,       size) if ng_paths else None

    # Template = mean of template embeddings (re-normalize)
    template = t_embs.mean(axis=0)
    template /= (np.linalg.norm(template) + 1e-8)

    # Cosine similarity = dot product (both L2-normalized)
    sim_ok = ok_embs @ template                    # (N_ok,)
    sim_ng = ng_embs @ template if ng_embs is not None else np.array([])

    # Best threshold on this char
    if len(sim_ng) > 0:
        all_sims = np.concatenate([sim_ok, sim_ng])
        all_labels = np.concatenate([np.zeros(len(sim_ok)), np.ones(len(sim_ng))])
        best_bal, best_th = -1.0, 0.0
        for th in np.linspace(all_sims.min(), all_sims.max(), 61):
            pred = (all_sims < th).astype(int)   # low sim → NG
            tp = ((pred == 1) & (all_labels == 1)).sum()
            tn = ((pred == 0) & (all_labels == 0)).sum()
            fp = ((pred == 1) & (all_labels == 0)).sum()
            fn = ((pred == 0) & (all_labels == 1)).sum()
            bal = 0.5 * tn / max(1, tn+fp) + 0.5 * tp / max(1, tp+fn)
            if bal > best_bal:
                best_bal, best_th = bal, th
        separation = float(sim_ok.mean() - sim_ng.mean())
    else:
        best_bal, best_th, separation = float("nan"), float("nan"), float("nan")

    return {
        "char":        char,
        "n_template":  len(template_paths),
        "n_ok":        len(sim_ok),
        "n_ng":        len(sim_ng),
        "sim_ok_mean": float(sim_ok.mean()),
        "sim_ok_min":  float(sim_ok.min()),
        "sim_ng_mean": float(sim_ng.mean()) if len(sim_ng) else float("nan"),
        "sim_ng_max":  float(sim_ng.max())  if len(sim_ng) else float("nan"),
        "separation":  separation,
        "best_bal":    best_bal,
        "best_th":     best_th,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    here = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True,
                        help="Run directory (contains config.yaml + model.onnx)")
    parser.add_argument("--dataset", default="/Users/ngocthien.ai/Downloads/output2",
                        help="Dataset root with char_*/ok|ng structure")
    parser.add_argument("--chars", nargs="*", default=None,
                        help="Filter specific chars (e.g. --chars A B 0 1). Default: all")
    parser.add_argument("--n-template", type=int, default=1,
                        help="Number of OK images to use as template per char")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = here / run_dir

    cfg = OmegaConf.load(run_dir / "config.yaml")
    head = cfg.model.head.type
    size = int(cfg.data.image_size)

    sess = ort.InferenceSession(
        str(run_dir / "model.onnx"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    print(f"[model]   {run_dir.name}")
    print(f"[head]    {head}  |  image_size={size}")
    print(f"[dataset] {args.dataset}")
    print(f"[template] n_template={args.n_template} per char\n")

    dataset_root = Path(args.dataset)
    char_dirs = sorted([d for d in dataset_root.iterdir()
                        if d.is_dir() and d.name.startswith("char_")])

    if args.chars:
        char_set = set(args.chars)
        char_dirs = [d for d in char_dirs
                     if d.name.replace("char_", "") in char_set]

    results = []
    for char_dir in char_dirs:
        char = char_dir.name.replace("char_", "")
        ok_paths, ng_paths = scan_char(char_dir)
        if not ok_paths:
            continue
        r = test_char(char, ok_paths, ng_paths, sess, head, size, args.n_template)
        if r is None:
            print(f"  [{char}] skip — not enough OK images (need >{args.n_template})")
            continue
        results.append(r)

        # Per-char inline print
        sep_str = f"{r['separation']:+.4f}" if r['separation'] == r['separation'] else "  n/a  "
        bal_str = f"{r['best_bal']:.4f}"    if r['best_bal']   == r['best_bal']   else "  n/a  "
        print(f"  [{char:>10}]  "
              f"sim_ok={r['sim_ok_mean']:.4f}(min {r['sim_ok_min']:.4f})  "
              f"sim_ng={r['sim_ng_mean']:.4f}(max {r['sim_ng_max']:.4f})  "
              f"sep={sep_str}  bal@best={bal_str}  "
              f"n_ok={r['n_ok']}  n_ng={r['n_ng']}")

    if not results:
        print("No results."); return

    # Summary
    seps  = [r["separation"] for r in results if r["separation"] == r["separation"]]
    bals  = [r["best_bal"]   for r in results if r["best_bal"]   == r["best_bal"]]
    print(f"\n=== Summary ({len(results)} chars) ===")
    print(f"  Mean separation (sim_ok - sim_ng):  {np.mean(seps):.4f}")
    print(f"  Min  separation:                    {np.min(seps):.4f}")
    print(f"  Mean best balanced accuracy:        {np.mean(bals):.4f}")

    # Worst 5 by separation
    worst = sorted(results, key=lambda r: r["separation"] if r["separation"]==r["separation"] else 1.0)[:5]
    print(f"\n  Worst 5 chars by separation:")
    for r in worst:
        print(f"    {r['char']:>10}  sep={r['separation']:+.4f}  bal={r['best_bal']:.4f}")


if __name__ == "__main__":
    main()
