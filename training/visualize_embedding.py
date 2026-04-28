#!/usr/bin/env python3
"""
Visualize template vs target embedding similarity.

Usage:
    python training/visualize_embedding.py \\
        --run training/runs/supcon_128_efficientnet_b3_20260428-180622 \\
        --ok-dir training/test_data/ok \\
        --ng-dir training/test_data/ng \\
        --out training/test_data/result.png

The first image in --ok-dir is used as the template.
All others are compared against it via cosine similarity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import onnxruntime as ort
from omegaconf import OmegaConf

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


def embed(sess: ort.InferenceSession, head: str, paths: list[str], size: int) -> np.ndarray:
    input_name = sess.get_inputs()[0].name
    tensors = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            tensors.append(np.zeros((3, size, size), dtype=np.float32))
        else:
            tensors.append(preprocess(img, size))
    x = np.stack(tensors).astype(np.float32)

    if head == "projection":
        emb = sess.run(None, {input_name: x})[0]
    elif head == "arcface":
        emb = sess.run(None, {input_name: x})[1]
    else:  # linear
        emb = softmax(sess.run(None, {input_name: x})[0])

    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    return emb / norms


def load_images_rgb(paths: list[str]) -> list[np.ndarray]:
    imgs = []
    for p in paths:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None
                    else np.full((64, 64, 3), 200, dtype=np.uint8))
    return imgs


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="training/runs/supcon_128_efficientnet_b3_20260428-180622")
    parser.add_argument("--ok-dir", default="training/test_data/ok")
    parser.add_argument("--ng-dir", default="training/test_data/ng")
    parser.add_argument("--out",    default="training/test_data/result.png")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = Path.cwd() / run_dir

    cfg  = OmegaConf.load(run_dir / "config.yaml")
    head = cfg.model.head.type
    size = int(cfg.data.image_size)

    sess = ort.InferenceSession(
        str(run_dir / "model.onnx"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    ok_paths = sorted(p for p in Path(args.ok_dir).iterdir() if p.suffix.lower() in VALID_EXT)
    ng_paths = sorted(p for p in Path(args.ng_dir).iterdir() if p.suffix.lower() in VALID_EXT)

    assert ok_paths, "No OK images found"

    # Template = first OK image
    template_path = ok_paths[0]
    test_ok_paths  = ok_paths[1:]
    test_ng_paths  = ng_paths

    print(f"[model]    {run_dir.name}  head={head}")
    print(f"[template] {template_path.name}")
    print(f"[test]     {len(test_ok_paths)} OK  +  {len(test_ng_paths)} NG")

    # Extract embeddings
    template_emb = embed(sess, head, [str(template_path)], size)[0]
    all_paths    = [str(p) for p in test_ok_paths + test_ng_paths]
    all_labels   = [0] * len(test_ok_paths) + [1] * len(test_ng_paths)
    all_embs     = embed(sess, head, all_paths, size)
    sims         = all_embs @ template_emb   # cosine similarity

    # Find best threshold
    best_bal, best_th = -1.0, 0.5
    for th in np.linspace(sims.min(), sims.max(), 101):
        pred = (sims < th).astype(int)
        y    = np.array(all_labels)
        tp   = ((pred == 1) & (y == 1)).sum()
        tn   = ((pred == 0) & (y == 0)).sum()
        fp   = ((pred == 1) & (y == 0)).sum()
        fn   = ((pred == 0) & (y == 1)).sum()
        bal  = 0.5 * tn / max(1, tn+fp) + 0.5 * tp / max(1, tp+fn)
        if bal > best_bal:
            best_bal, best_th = bal, float(th)

    print(f"[result]   best_th={best_th:.4f}  balanced_acc={best_bal:.4f}")

    # ----------------------------------------------------------------
    # Figure layout:
    #   Row 0          : template image (centered)
    #   Row 1..end     : test images sorted by similarity desc
    #   Right panel    : similarity bar chart
    # ----------------------------------------------------------------
    all_imgs = load_images_rgb(all_paths)
    order    = np.argsort(-sims)   # high → low similarity

    n_test = len(all_paths)
    cols   = min(10, n_test)
    rows   = (n_test + cols - 1) // cols

    fig = plt.figure(figsize=(max(14, cols * 1.5), rows * 1.8 + 5))
    fig.patch.set_facecolor("#1a1a2e")

    # -- Template strip --
    ax_tmpl = fig.add_axes([0.01, 0.88, 0.18, 0.10])
    ax_tmpl.imshow(load_images_rgb([str(template_path)])[0])
    ax_tmpl.set_title("TEMPLATE", color="white", fontsize=9, fontweight="bold", pad=3)
    ax_tmpl.axis("off")
    for spine in ax_tmpl.spines.values():
        spine.set_edgecolor("#00d4ff"); spine.set_linewidth(2)

    # -- Title --
    fig.text(0.5, 0.95,
             f"Embedding Similarity  |  model: {run_dir.name}  |  head: {head}",
             ha="center", color="white", fontsize=11, fontweight="bold")
    fig.text(0.5, 0.92,
             f"threshold = {best_th:.4f}   (sim < th → NG)   balanced_acc = {best_bal:.4f}",
             ha="center", color="#aaaaaa", fontsize=9)

    # -- Image grid --
    grid_top    = 0.82
    grid_height = 0.78
    cell_w = 0.98 / cols
    cell_h = grid_height / rows

    for rank, idx in enumerate(order):
        row_i = rank // cols
        col_i = rank  % cols
        sim   = float(sims[idx])
        label = all_labels[idx]
        pred  = 1 if sim < best_th else 0
        correct = pred == label

        left   = 0.01 + col_i * cell_w
        bottom = grid_top - (row_i + 1) * cell_h + 0.01

        ax = fig.add_axes([left, bottom, cell_w * 0.92, cell_h * 0.78])
        ax.imshow(all_imgs[idx])
        ax.axis("off")

        # Border color: green=correct, red=wrong
        edge_color = "#00ff88" if correct else "#ff4444"
        for spine in ax.spines.values():
            spine.set_edgecolor(edge_color)
            spine.set_linewidth(2.5)

        true_str  = "OK" if label == 0 else "NG"
        pred_str  = "OK" if pred  == 0 else "NG"
        ok_ng_color = "#44dd88" if label == 0 else "#ff6666"

        ax.set_title(
            f"sim={sim:.3f}\n{true_str}→{pred_str}",
            color=ok_ng_color if correct else "#ff4444",
            fontsize=6.5, pad=2,
        )

        # Synth badge
        fname = Path(all_paths[idx]).name
        if "synth" in fname:
            ax.text(0.02, 0.98, "S", transform=ax.transAxes,
                    color="#ffcc00", fontsize=7, va="top",
                    bbox=dict(boxstyle="round,pad=0.1", fc="#333300", ec="none"))

    # -- Similarity bar chart (right side) --
    ax_bar = fig.add_axes([0.0, 0.02, 0.99, 0.09])
    bar_sims   = sims[order]
    bar_labels = [all_labels[i] for i in order]
    bar_colors = ["#44dd88" if l == 0 else "#ff6666" for l in bar_labels]
    ax_bar.bar(range(n_test), bar_sims, color=bar_colors, width=0.8, alpha=0.85)
    ax_bar.axhline(best_th, color="#ffcc00", linewidth=1.5, linestyle="--", label=f"th={best_th:.3f}")
    ax_bar.set_xlim(-0.5, n_test - 0.5)
    ax_bar.set_ylabel("cosine sim", color="white", fontsize=8)
    ax_bar.set_facecolor("#0d0d1a")
    ax_bar.tick_params(colors="white", labelsize=7)
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#444444")
    ax_bar.legend(fontsize=8, labelcolor="white", facecolor="#222222", edgecolor="#444444")

    # Legend
    fig.text(0.22, 0.93, "█ OK  █ NG  S=synthetic  border: green=correct / red=wrong",
             color="#aaaaaa", fontsize=8)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[saved]    {out_path}")


if __name__ == "__main__":
    main()
