"""
Training loop.

train(cfg, run_dir) is the single entry point. It:
    1. Builds data, model, loss, optimizer, scheduler.
    2. Auto-derives `data.task` / head / inference if loss requires it.
    3. Runs `cfg.train.epochs` epochs of train + val.
    4. Saves best.pt by `cfg.eval.primary_metric`.
    5. Final eval at best ckpt + threshold sweep + per-char metrics.
    6. Optional ONNX export.
    7. Writes config.yaml, env.json, metrics.json, per_char_metrics.csv.

The module never reaches into the model's internals — it only consumes the
dict returned by Model.forward and passes both halves to loss_fn.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from .data import OCRDataset, build_augment, collate_fn, make_loaders, scan_dataset, stratified_split
from .evaluator import (
    best_threshold,
    collect_predictions,
    compute_centroids,
    compute_metrics,
    per_char_breakdown,
)
from .losses import build_loss, supcon_from_embeddings
from .models import build_model
from .utils import save_config


# ---------------------------------------------------------------------------
# Optimizer & scheduler
# ---------------------------------------------------------------------------

def build_optimizer(cfg, params):
    if cfg.train.optimizer == "adamw":
        return AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    if cfg.train.optimizer == "sgd":
        return SGD(params, lr=cfg.train.lr, momentum=0.9,
                   weight_decay=cfg.train.weight_decay, nesterov=True)
    raise ValueError(f"Unknown optimizer: {cfg.train.optimizer}")


def build_scheduler(cfg, optimizer, steps_per_epoch: int):
    epochs = cfg.train.epochs
    warmup = max(0, int(cfg.train.warmup_epochs))
    if cfg.train.scheduler == "cosine":
        if warmup == 0:
            return CosineAnnealingLR(optimizer, T_max=epochs)
        # linear warmup to lr, then cosine to 0
        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            t = (epoch - warmup) / max(1, epochs - warmup)
            return 0.5 * (1 + math.cos(math.pi * t))
        return LambdaLR(optimizer, lr_lambda)
    if cfg.train.scheduler == "none":
        return None
    raise ValueError(f"Unknown scheduler: {cfg.train.scheduler}")


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def _is_two_view(imgs) -> bool:
    """DataLoader sometimes converts our tuple to a list — accept both."""
    return isinstance(imgs, (tuple, list)) and len(imgs) == 2 and torch.is_tensor(imgs[0])


def train_one_epoch(model, loader, loss_fn, optimizer, device, cfg, scaler=None) -> float:
    model.train()
    total = 0.0; n = 0
    use_amp = cfg.train.amp and device == "cuda"

    for imgs, target in loader:
        if _is_two_view(imgs):
            v1, v2 = imgs
            imgs_f = torch.cat([v1.to(device, non_blocking=True),
                                v2.to(device, non_blocking=True)], dim=0)
            # Duplicate targets across the two views
            target = {
                "ok_ng":     torch.cat([target["ok_ng"],     target["ok_ng"]],     dim=0),
                "multi_idx": torch.cat([target["multi_idx"], target["multi_idx"]], dim=0),
                "char":      list(target["char"])  + list(target["char"]),
                "path":      list(target["path"])  + list(target["path"]),
            }
        else:
            imgs_f = imgs.to(device, non_blocking=True)

        labels_for_head = target["multi_idx"].to(device) if cfg.model.head.type == "arcface" else None

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                out = model(imgs_f, labels=labels_for_head) if labels_for_head is not None else model(imgs_f)
                loss = loss_fn(out, target)
            scaler.scale(loss).backward()
            if cfg.train.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            out = model(imgs_f, labels=labels_for_head) if labels_for_head is not None else model(imgs_f)
            loss = loss_fn(out, target)
            loss.backward()
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

        bs = imgs_f.size(0)
        total += loss.item() * bs; n += bs
    return total / max(1, n)


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def _resolve_inference_mode(cfg) -> str:
    """Pick eval inference mode based on head."""
    if cfg.model.head.type == "projection":
        return "centroid"
    if cfg.model.head.type == "arcface":
        return "argmax"     # uses cos_sim internally
    return "argmax"


def _build_centroid_loader(cfg, train_samples):
    """Single-view loader on train, with eval transform."""
    _, val_tf = build_augment("minimal", cfg.data.image_size)   # always minimal for centroid
    ds = OCRDataset(train_samples, transform=val_tf)
    return torch.utils.data.DataLoader(
        ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=True, collate_fn=collate_fn,
    )


def evaluate(model, val_loader, cfg, device, multi_to_idx, train_samples,
             multi_idx_to_ok_ng) -> Tuple[Dict, Dict]:
    """Returns (metrics_dict, predictions_dict)."""
    centroids = None
    if cfg.model.head.type == "projection":
        centroid_loader = _build_centroid_loader(cfg, train_samples)
        centroids = compute_centroids(model, centroid_loader,
                                      num_classes=len(multi_to_idx), device=device,
                                      feat_key="embedding")

    preds = collect_predictions(
        model, val_loader, cfg, device,
        multi_idx_to_ok_ng=multi_idx_to_ok_ng,
        centroids=centroids,
    )
    metrics = compute_metrics(preds)
    th = best_threshold(preds)
    metrics["threshold"]    = th["threshold"]
    metrics["th_balanced"]  = th["balanced"]
    metrics["th_ok_pass"]   = th["ok_pass"]
    metrics["th_ng_catch"]  = th["ng_catch"]
    return metrics, preds


def _select_metric(metrics: Dict, primary: str) -> float:
    return {
        "balanced_score": metrics.get("th_balanced", metrics["argmax_balanced"]),
        "bin_acc":        metrics["argmax_acc"],
        "auc":            metrics.get("auc", 0.0),
        "argmax_balanced": metrics["argmax_balanced"],
    }.get(primary, metrics["argmax_balanced"])


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(model, cfg, out_path: Path):
    import onnx
    model.eval()
    h = w = int(cfg.data.image_size)
    dummy = torch.zeros(1, 3, h, w).to(next(model.parameters()).device)

    output_names = []
    head = cfg.model.head.type
    if head == "linear":
        output_names = ["logits"]
    elif head == "projection":
        output_names = ["embedding"]
    elif head == "arcface":
        output_names = ["logits", "cos_sim"]

    # Wrap to flatten dict output to tuple for ONNX
    class WrappedONNX(nn.Module):
        def __init__(self, m, head_type):
            super().__init__()
            self.m = m; self.head_type = head_type
        def forward(self, x):
            o = self.m(x)
            if self.head_type == "linear":
                return o["logits"]
            if self.head_type == "projection":
                return o["embedding"]
            if self.head_type == "arcface":
                return o["logits"], o["cos_sim"]

    wrapper = WrappedONNX(model, head)
    torch.onnx.export(
        wrapper, dummy, str(out_path),
        input_names=["input"], output_names=output_names,
        dynamic_axes={"input": {0: "batch"},
                      **{n: {0: "batch"} for n in output_names}},
        opset_version=int(cfg.output.onnx_opset),
        do_constant_folding=True,
    )
    # Force single-file
    m = onnx.load(str(out_path), load_external_data=True)
    onnx.save_model(m, str(out_path), save_as_external_data=False)
    for sidecar in out_path.parent.glob(f"{out_path.stem}*.data"):
        sidecar.unlink()


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def train(cfg, run_dir: Path) -> Dict:
    """Run training, write artifacts to run_dir, return final metrics dict."""
    from .utils import pick_device, set_seed

    device = pick_device(cfg.device)
    set_seed(cfg.seed)
    print(f"[run_dir] {run_dir}")
    print(f"[device]  {device}")

    # ---- data ----
    train_loader, val_loader, multi_to_idx, train_samples, val_samples = make_loaders(cfg)
    multi_idx_to_ok_ng = {idx: (0 if name.endswith("__ok") else 1)
                          for name, idx in multi_to_idx.items()}
    print(f"[data]    samples: {len(train_samples):,} train, {len(val_samples):,} val")
    print(f"          multi classes: {len(multi_to_idx)}")

    # ---- model ----
    model = build_model(cfg, num_multi_classes=len(multi_to_idx)).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model]   {cfg.model.backbone} + {cfg.model.head.type}  "
          f"feat_dim={model.feat_dim}  params={n_params:.2f}M")

    # ---- loss ----
    loss_fn = build_loss(cfg, num_multi_classes=len(multi_to_idx),
                         feat_dim=model.feat_dim, train_samples=train_samples)
    loss_fn = loss_fn.to(device)
    print(f"[loss]    {cfg.loss.type}  ({type(loss_fn).__name__})")

    # ---- optim ----
    params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = build_optimizer(cfg, params)
    scheduler = build_scheduler(cfg, optimizer, len(train_loader))
    scaler = torch.amp.GradScaler("cuda") if (cfg.train.amp and device == "cuda") else None
    print(f"[optim]   {cfg.train.optimizer}  lr={cfg.train.lr}  amp={scaler is not None}")

    # ---- train ----
    save_config(cfg, run_dir / "config.yaml")
    best_score = -1.0; best_metrics = {}; best_preds = None
    history = {"train_loss": [], "val": []}

    for epoch in range(cfg.train.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, cfg, scaler)
        if scheduler is not None:
            scheduler.step()

        metrics, preds = evaluate(model, val_loader, cfg, device,
                                   multi_to_idx, train_samples, multi_idx_to_ok_ng)
        score = _select_metric(metrics, cfg.eval.primary_metric)
        history["train_loss"].append(train_loss)
        history["val"].append(metrics)

        flag = ""
        if score > best_score:
            best_score = score
            best_metrics = metrics
            best_preds = preds
            torch.save({
                "model": model.state_dict(),
                "loss_state": loss_fn.state_dict(),
                "epoch": epoch,
                "score": score,
                "metrics": metrics,
                "multi_to_idx": multi_to_idx,
            }, run_dir / "best.pt")
            flag = "  ← saved best"

        dt = time.time() - t0
        print(f"Ep {epoch+1:02d}/{cfg.train.epochs}  "
              f"loss={train_loss:.4f}  "
              f"argmax_bal={metrics['argmax_balanced']:.3f}  "
              f"th_bal={metrics['th_balanced']:.3f}  "
              f"auc={metrics['auc']:.4f}  "
              f"({dt:.1f}s){flag}")

    # ---- final eval ----
    print(f"\n[done]    best {cfg.eval.primary_metric}: {best_score:.4f}")
    print(f"          best metrics: {json.dumps(best_metrics, indent=2)}")

    per_char = per_char_breakdown(best_preds, threshold=best_metrics.get("threshold"))
    with open(run_dir / "per_char_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_char[0].keys()))
        w.writeheader(); w.writerows(per_char)

    # Save metrics history
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({
            "best_score": best_score,
            "best_metrics": best_metrics,
            "history": {"train_loss": history["train_loss"],
                         "val_argmax_bal":   [m["argmax_balanced"] for m in history["val"]],
                         "val_th_bal":       [m["th_balanced"]     for m in history["val"]],
                         "val_auc":          [m["auc"]              for m in history["val"]]},
        }, f, indent=2)

    # Save best predictions for downstream FN/FP analysis
    np.savez(run_dir / "val_predictions.npz",
             score_ng=best_preds["score_ng"], pred_argmax=best_preds["pred_argmax"],
             pred_multi=best_preds["pred_multi"], true_ok_ng=best_preds["true_ok_ng"],
             true_multi=best_preds["true_multi"], chars=best_preds["chars"],
             paths=best_preds["paths"])

    # ---- onnx ----
    if cfg.output.export_onnx:
        ckpt = torch.load(run_dir / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        export_onnx(model, cfg, run_dir / "model.onnx")
        print(f"[onnx]    exported model.onnx")

    return best_metrics
