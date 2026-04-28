#!/usr/bin/env python3
"""
Quick test script for the binary OK/NG ONNX model.

Three modes:

    # 1. Single image — print verdict
    python test_model.py --image /path/to/crop.png

    # 2. Folder of images (no labels) — write CSV of predictions
    python test_model.py --dir /path/to/folder --csv predictions.csv

    # 3. Labeled dataset (char_*/ok|ng) — full eval: AUC, OK_pass, NG_catch,
    #    per-char breakdown, optional FN/FP visualization
    python test_model.py --dataset /Users/ngocthien.ai/Downloads/output2 \\
        --visualize-fn fn_gallery.png

Defaults assume ./ok_ng_model.onnx + ./model_meta.json next to this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Loading + preprocessing
# ---------------------------------------------------------------------------

def load_meta(meta_path: Path) -> dict:
    with open(meta_path) as f:
        return json.load(f)


def make_session(onnx_path: Path) -> ort.InferenceSession:
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    return ort.InferenceSession(str(onnx_path), providers=providers)


def preprocess(bgr: np.ndarray, size: int, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Letterbox to (size, size) → RGB float32 normalized → CHW."""
    h, w = bgr.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    y0 = (size - nh) // 2; x0 = (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = img
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - mean) / std
    return rgb.transpose(2, 0, 1)        # (3, H, W)


def softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def predict_batch(sess, paths: List[str], size: int, mean, std,
                  batch_size: int = 64) -> Tuple[np.ndarray, List[str]]:
    """Predict P(NG) for a list of image paths. Returns (probs_ng, valid_paths)."""
    probs: List[float] = []
    valid_paths: List[str] = []
    failed: List[str] = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        tensors = []
        kept = []
        for p in chunk:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                failed.append(p); continue
            tensors.append(preprocess(img, size, mean, std))
            kept.append(p)
        if not tensors:
            continue
        x = np.stack(tensors, axis=0).astype(np.float32)
        logits = sess.run(None, {"input": x})[0]
        p_ng = softmax(logits, axis=-1)[:, 1]
        probs.extend(p_ng.tolist())
        valid_paths.extend(kept)
    if failed:
        print(f"[warn] failed to load {len(failed)} image(s)", file=sys.stderr)
    return np.array(probs), valid_paths


# ---------------------------------------------------------------------------
# Mode 1: single image
# ---------------------------------------------------------------------------

def run_single(args, sess, meta):
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Cannot read {args.image}"); return 2
    mean = np.array(meta["normalization"]["mean"], dtype=np.float32)
    std = np.array(meta["normalization"]["std"], dtype=np.float32)
    th = args.threshold if args.threshold is not None else meta["threshold"]

    t0 = time.time()
    x = preprocess(img, meta["image_size"], mean, std)[None]
    logits = sess.run(None, {"input": x.astype(np.float32)})[0][0]
    p = softmax(logits)
    p_ng = float(p[1])
    dt_ms = (time.time() - t0) * 1000
    verdict = "NG" if p_ng >= th else "OK"
    print(f"{args.image}")
    print(f"  P(OK)={p[0]:.4f}  P(NG)={p_ng:.4f}  threshold={th:.3f}")
    print(f"  → {verdict}  ({dt_ms:.1f} ms)")
    return 0


# ---------------------------------------------------------------------------
# Mode 2: folder (no labels)
# ---------------------------------------------------------------------------

def list_images(folder: Path, recursive: bool) -> List[str]:
    it = folder.rglob("*") if recursive else folder.iterdir()
    return [str(p) for p in sorted(it) if p.is_file() and p.suffix.lower() in VALID_EXT]


def run_dir(args, sess, meta):
    folder = Path(args.dir)
    paths = list_images(folder, args.recursive)
    if not paths:
        print(f"No images in {folder}"); return 2
    print(f"Predicting on {len(paths)} images...")
    mean = np.array(meta["normalization"]["mean"], dtype=np.float32)
    std = np.array(meta["normalization"]["std"], dtype=np.float32)
    th = args.threshold if args.threshold is not None else meta["threshold"]

    t0 = time.time()
    probs, kept = predict_batch(sess, paths, meta["image_size"], mean, std, args.batch_size)
    dt = time.time() - t0
    verdicts = ["NG" if p >= th else "OK" for p in probs]
    n_ng = sum(1 for v in verdicts if v == "NG")
    print(f"Done in {dt:.1f}s  ({len(probs)/dt:.1f} img/s)")
    print(f"  Predicted NG: {n_ng}/{len(probs)} ({n_ng/len(probs)*100:.1f}%)")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "p_ng", "verdict"])
            for path, p, v in zip(kept, probs, verdicts):
                w.writerow([path, f"{p:.6f}", v])
        print(f"  → wrote {args.csv}")
    return 0


# ---------------------------------------------------------------------------
# Mode 3: labeled dataset (char_*/ok|ng)
# ---------------------------------------------------------------------------

def scan_labeled(root: Path) -> List[Tuple[str, str, int]]:
    """Return list of (path, char, true_label) where label 0=OK, 1=NG."""
    out = []
    for char_dir in sorted(root.iterdir()):
        if not char_dir.is_dir() or not char_dir.name.startswith("char_"):
            continue
        for label, sub in [(0, "ok"), (1, "ng")]:
            folder = char_dir / sub
            if not folder.is_dir():
                continue
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in VALID_EXT:
                    out.append((str(p), char_dir.name, label))
    return out


def run_dataset(args, sess, meta):
    root = Path(args.dataset)
    rows = scan_labeled(root)
    if not rows:
        print(f"No char_*/ok|ng samples under {root}"); return 2
    print(f"Found {len(rows)} labeled samples under {root}")

    paths = [r[0] for r in rows]
    chars = [r[1] for r in rows]
    labels = np.array([r[2] for r in rows])

    mean = np.array(meta["normalization"]["mean"], dtype=np.float32)
    std = np.array(meta["normalization"]["std"], dtype=np.float32)
    th = args.threshold if args.threshold is not None else meta["threshold"]

    t0 = time.time()
    probs, kept = predict_batch(sess, paths, meta["image_size"], mean, std, args.batch_size)
    dt = time.time() - t0
    print(f"Inference: {dt:.1f}s  ({len(probs)/dt:.1f} img/s)")

    # Align labels with kept (in case some images failed to load)
    if len(kept) < len(paths):
        keep_idx = [paths.index(p) for p in kept]
        labels = labels[keep_idx]
        chars = [chars[i] for i in keep_idx]

    preds = (probs >= th).astype(int)

    # Confusion + summary
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    ok_pass = tn / max(1, tn + fp)
    ng_catch = tp / max(1, tp + fn)
    bal = 0.5 * ok_pass + 0.5 * ng_catch

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(labels, probs))
    except Exception:
        auc = float("nan")

    print()
    print(f"=== Threshold {th:.3f} ===")
    print(f"  AUC:        {auc:.4f}")
    print(f"  Accuracy:   {acc:.4f}")
    print(f"  OK pass:    {ok_pass:.4f}  ({tn}/{tn+fp})")
    print(f"  NG catch:   {ng_catch:.4f}  ({tp}/{tp+fn})")
    print(f"  Balanced:   {bal:.4f}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    # Per-char breakdown
    print(f"\n=== Worst 10 chars by NG catch ===")
    rows_pc = []
    for ch in sorted(set(chars)):
        mask = np.array([c == ch for c in chars])
        ok_m = mask & (labels == 0); ng_m = mask & (labels == 1)
        rows_pc.append({
            "char": ch,
            "n_ok": int(ok_m.sum()), "n_ng": int(ng_m.sum()),
            "ok_pass":  float((preds[ok_m] == 0).mean()) if ok_m.any() else float("nan"),
            "ng_catch": float((preds[ng_m] == 1).mean()) if ng_m.any() else float("nan"),
            "fn": int((preds[ng_m] == 0).sum()),
            "fp": int((preds[ok_m] == 1).sum()),
        })
    by_ng = sorted(rows_pc, key=lambda r: r["ng_catch"] if r["ng_catch"] == r["ng_catch"] else 1.0)
    for r in by_ng[:10]:
        print(f"  {r['char']:18s} n_ok={r['n_ok']:3d} n_ng={r['n_ng']:3d}"
              f"  ok_pass={r['ok_pass']:.3f}  ng_catch={r['ng_catch']:.3f}"
              f"  FN={r['fn']}  FP={r['fp']}")

    # Save CSVs
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "char", "true_label", "p_ng", "verdict", "kind"])
            kind_map = {(0,0):"TN", (0,1):"FP", (1,0):"FN", (1,1):"TP"}
            for path, ch, lbl, p, pred in zip(kept, chars, labels, probs, preds):
                w.writerow([path, ch, int(lbl), f"{p:.6f}",
                            "NG" if pred==1 else "OK",
                            kind_map[(int(lbl), int(pred))]])
        print(f"\n[csv] {args.csv}")

        per_char_csv = Path(args.csv).with_name(Path(args.csv).stem + "_per_char.csv")
        with open(per_char_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_pc[0].keys()))
            w.writeheader(); w.writerows(rows_pc)
        print(f"[csv] {per_char_csv}")

    # Threshold sweep
    if args.sweep:
        print(f"\n=== Threshold sweep ===")
        ths = np.linspace(0.05, 0.95, 19)
        for t in ths:
            pr = (probs >= t).astype(int)
            okp = ((pr == 0) & (labels == 0)).sum() / max(1, (labels == 0).sum())
            ngc = ((pr == 1) & (labels == 1)).sum() / max(1, (labels == 1).sum())
            b = 0.5 * okp + 0.5 * ngc
            star = " ★" if abs(t - th) < 0.03 else ""
            print(f"  th={t:.2f}  OK_pass={okp:.3f}  NG_catch={ngc:.3f}  bal={b:.3f}{star}")

    # FN visualization
    if args.visualize_fn:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[warn] matplotlib not installed; skipping visualization", file=sys.stderr)
        else:
            fn_idx = np.where((preds == 0) & (labels == 1))[0]
            print(f"\n[viz] {len(fn_idx)} false negatives → {args.visualize_fn}")
            if len(fn_idx) > 0:
                # Sort by P_NG descending (closest to threshold = most concerning)
                fn_idx = fn_idx[np.argsort(-probs[fn_idx])]
                show = fn_idx[:args.viz_max]
                cols = 8; n = len(show); rows_n = (n + cols - 1) // cols
                fig, axes = plt.subplots(rows_n, cols, figsize=(cols*1.7, rows_n*1.9))
                axes = np.atleast_2d(axes).ravel()
                for i, ax in enumerate(axes):
                    ax.axis("off")
                    if i < n:
                        idx = show[i]
                        img = cv2.cvtColor(cv2.imread(kept[idx]), cv2.COLOR_BGR2RGB)
                        ax.imshow(img)
                        ax.set_title(f"{chars[idx]}\nP_NG={probs[idx]:.2f}",
                                      fontsize=7, color="red")
                fig.suptitle(f"False Negatives ({len(fn_idx)}) — NG escaped as OK",
                              fontsize=11)
                plt.tight_layout()
                plt.savefig(args.visualize_fn, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"[viz] saved {args.visualize_fn}")

    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(here / "ok_ng_model.onnx"),
                        help="Path to ONNX model")
    parser.add_argument("--meta", default=str(here / "model_meta.json"),
                        help="Path to model_meta.json")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override threshold (default = meta.threshold)")
    parser.add_argument("--batch-size", type=int, default=64)

    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="Single image path")
    g.add_argument("--dir", help="Folder of images (no labels)")
    g.add_argument("--dataset", help="Root with char_*/ok|ng (full eval)")

    parser.add_argument("--recursive", action="store_true",
                        help="Recurse into --dir")
    parser.add_argument("--csv", default=None,
                        help="Write per-image CSV (modes 2 & 3)")
    parser.add_argument("--sweep", action="store_true",
                        help="(mode 3) Print threshold sweep table")
    parser.add_argument("--visualize-fn", default=None,
                        help="(mode 3) Path to save FN gallery PNG")
    parser.add_argument("--viz-max", type=int, default=32,
                        help="Max images in FN gallery")
    args = parser.parse_args()

    onnx_path = Path(args.model); meta_path = Path(args.meta)
    if not onnx_path.is_file():
        print(f"ONNX not found: {onnx_path}", file=sys.stderr); return 2
    if not meta_path.is_file():
        print(f"Meta not found: {meta_path}", file=sys.stderr); return 2

    sess = make_session(onnx_path)
    meta = load_meta(meta_path)
    print(f"[model]  {onnx_path}  ({onnx_path.stat().st_size/1e6:.1f} MB)")
    print(f"[meta]   image_size={meta['image_size']}  threshold={meta['threshold']:.3f}  "
          f"AUC={meta['metrics']['auc']:.4f}")
    print(f"[ort]    providers={sess.get_providers()}\n")

    if args.image:
        return run_single(args, sess, meta)
    if args.dir:
        return run_dir(args, sess, meta)
    if args.dataset:
        return run_dataset(args, sess, meta)
    return 1


if __name__ == "__main__":
    sys.exit(main())


