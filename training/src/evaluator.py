"""
Validation logic.

Per-sample score_ng (continuous, higher = more NG) is computed differently
depending on (task, head_type):

    binary + linear            → softmax(logits)[:, 1]
    multi_128 + linear (CE)    → P_ng_sum  =  Σ softmax over NG classes
    multi_128 + arcface        → max(cos_sim_ng) - max(cos_sim_ok)
    multi_128 + projection     → max(emb·c_ng) - max(emb·c_ok)   (centroids from train)

Binary OK/NG prediction:
    threshold mode      pred = (score_ng >= th).int()
    argmax  mode        pred = argmax(logits) → if multi, lookup ok/ng of that class

We always report metrics under both modes (when applicable) plus AUC.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Centroid computation (for projection / SupCon)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_centroids(
    model: torch.nn.Module,
    loader,
    num_classes: int,
    device: str,
    feat_key: str = "embedding",
) -> torch.Tensor:
    """Mean L2-normalized embedding per class on `loader`. Returns (C, D)."""
    model.eval()
    sums: Optional[torch.Tensor] = None
    counts = torch.zeros(num_classes, device=device)
    for imgs, target in loader:
        imgs = imgs.to(device, non_blocking=True) if not isinstance(imgs, tuple) else imgs[0].to(device, non_blocking=True)
        out = model(imgs)
        emb = out[feat_key] if feat_key in out else F.normalize(out["feat"], dim=1)
        if sums is None:
            sums = torch.zeros(num_classes, emb.shape[1], device=device)
        labels = target["multi_idx"].to(device)
        for c in labels.unique():
            mask = labels == c
            sums[c] += emb[mask].sum(dim=0)
            counts[c] += mask.sum()
    centroids = sums / counts.clamp(min=1).unsqueeze(1)
    return F.normalize(centroids, dim=1)


# ---------------------------------------------------------------------------
# Per-sample scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    val_loader,
    cfg,
    device: str,
    multi_idx_to_ok_ng: Optional[Dict[int, int]] = None,
    centroids: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """
    Run model on val and return arrays:
        score_ng        (N,) continuous, higher = more NG
        pred_argmax     (N,) hard prediction from argmax (binary 0/1)
        true_ok_ng      (N,) ground truth 0/1
        true_multi      (N,) ground truth multi_idx
        pred_multi      (N,) argmax in multi space (None → -1 if not applicable)
        chars           list[str]
        paths           list[str]
    """
    model.eval()
    head_type = cfg.model.head.type
    task = cfg.data.task

    # Build NG/OK class masks once
    ng_mask = ok_mask = None
    if task == "multi_128" and multi_idx_to_ok_ng is not None:
        n = len(multi_idx_to_ok_ng)
        ng_mask = torch.tensor(
            [multi_idx_to_ok_ng[i] == 1 for i in range(n)],
            dtype=torch.bool, device=device,
        )
        ok_mask = ~ng_mask

    score_ng_all: List[np.ndarray] = []
    pred_arg_all: List[np.ndarray] = []
    pred_multi_all: List[np.ndarray] = []
    true_ok_ng_all: List[np.ndarray] = []
    true_multi_all: List[np.ndarray] = []
    chars_all: List[str] = []
    paths_all: List[str] = []

    for imgs, target in val_loader:
        if isinstance(imgs, tuple):           # paranoia: shouldn't be in val
            imgs = imgs[0]
        imgs = imgs.to(device, non_blocking=True)
        out = model(imgs)

        if task == "binary":
            probs = F.softmax(out["logits"], dim=1)
            score_ng = probs[:, 1]
            pred_argmax = out["logits"].argmax(dim=1)
            pred_multi = torch.full_like(pred_argmax, -1)

        else:                                  # multi_128
            if head_type == "projection":
                # centroid-based
                assert centroids is not None, "centroids required for projection head"
                emb = out["embedding"]
                sims = emb @ centroids.t()
            elif head_type == "arcface":
                sims = out["cos_sim"]
            else:                              # linear
                sims = F.softmax(out["logits"], dim=1)

            pred_multi = sims.argmax(dim=1)
            sim_ng = sims[:, ng_mask].max(dim=1).values
            sim_ok = sims[:, ok_mask].max(dim=1).values
            score_ng = sim_ng - sim_ok
            # Binary prediction from argmax: lookup ok/ng of predicted class
            pred_argmax = torch.tensor(
                [multi_idx_to_ok_ng[int(p)] for p in pred_multi.cpu().tolist()],
                device=device, dtype=torch.long,
            )

        score_ng_all.append(score_ng.detach().cpu().numpy())
        pred_arg_all.append(pred_argmax.detach().cpu().numpy())
        pred_multi_all.append(pred_multi.detach().cpu().numpy())
        true_ok_ng_all.append(target["ok_ng"].numpy())
        true_multi_all.append(target["multi_idx"].numpy())
        chars_all.extend(target["char"])
        paths_all.extend(target["path"])

    return {
        "score_ng":   np.concatenate(score_ng_all),
        "pred_argmax": np.concatenate(pred_arg_all),
        "pred_multi": np.concatenate(pred_multi_all),
        "true_ok_ng": np.concatenate(true_ok_ng_all),
        "true_multi": np.concatenate(true_multi_all),
        "chars":      np.array(chars_all),
        "paths":      np.array(paths_all),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute AUC, OK_pass, NG_catch, balanced (under argmax) and best-threshold."""
    s = preds["score_ng"]
    y = preds["true_ok_ng"]
    pa = preds["pred_argmax"]

    out: Dict[str, float] = {}

    # Argmax-based (no threshold tuning)
    tp = int(((pa == 1) & (y == 1)).sum())
    fp = int(((pa == 1) & (y == 0)).sum())
    tn = int(((pa == 0) & (y == 0)).sum())
    fn = int(((pa == 0) & (y == 1)).sum())
    out["argmax_ok_pass"] = tn / max(1, tn + fp)
    out["argmax_ng_catch"] = tp / max(1, tp + fn)
    out["argmax_balanced"] = 0.5 * out["argmax_ok_pass"] + 0.5 * out["argmax_ng_catch"]
    out["argmax_acc"] = (tp + tn) / max(1, tp + tn + fp + fn)

    # AUC
    if len(np.unique(y)) == 2:
        from sklearn.metrics import roc_auc_score
        out["auc"] = float(roc_auc_score(y, s))
    else:
        out["auc"] = float("nan")

    # Multi-class accuracy (only meaningful for multi_128)
    if "true_multi" in preds and (preds["pred_multi"] >= 0).all():
        out["multi_acc"] = float((preds["pred_multi"] == preds["true_multi"]).mean())
    else:
        out["multi_acc"] = float("nan")

    return out


def best_threshold(preds: Dict[str, np.ndarray], n_steps: int = 91) -> Dict[str, float]:
    """Sweep thresholds on score_ng, return best by balanced (OK_pass + NG_catch) / 2."""
    s = preds["score_ng"]; y = preds["true_ok_ng"]
    if len(s) == 0:
        return {"threshold": 0.0, "balanced": 0.0, "ok_pass": 0.0, "ng_catch": 0.0}

    lo, hi = float(s.min()), float(s.max())
    if math.isclose(lo, hi):
        lo -= 0.01; hi += 0.01
    ths = np.linspace(lo, hi, n_steps)
    best = {"threshold": float(lo), "balanced": -1.0, "ok_pass": 0.0, "ng_catch": 0.0}
    for th in ths:
        pred = (s >= th).astype(int)
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tp = int(((pred == 1) & (y == 1)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        ok_pass = tn / max(1, tn + fp); ng_catch = tp / max(1, tp + fn)
        bal = 0.5 * ok_pass + 0.5 * ng_catch
        if bal > best["balanced"]:
            best.update(threshold=float(th), balanced=float(bal),
                        ok_pass=float(ok_pass), ng_catch=float(ng_catch))
    return best


# ---------------------------------------------------------------------------
# Per-char breakdown
# ---------------------------------------------------------------------------

def per_char_breakdown(
    preds: Dict[str, np.ndarray],
    threshold: Optional[float] = None,
) -> List[Dict[str, float]]:
    """Per-char OK_pass / NG_catch under either argmax or threshold-based prediction."""
    s = preds["score_ng"]; y = preds["true_ok_ng"]; chars = preds["chars"]
    pred = (s >= threshold).astype(int) if threshold is not None else preds["pred_argmax"]
    rows: List[Dict[str, float]] = []
    for ch in sorted(set(chars.tolist())):
        m = chars == ch
        yi, pi = y[m], pred[m]
        ok = (yi == 0); ng = (yi == 1)
        rows.append({
            "char": ch,
            "n_ok": int(ok.sum()),
            "n_ng": int(ng.sum()),
            "ok_pass":  float((pi[ok] == 0).mean()) if ok.any() else float("nan"),
            "ng_catch": float((pi[ng] == 1).mean()) if ng.any() else float("nan"),
            "fn_count": int(((pi[ng] == 0)).sum()) if ng.any() else 0,
            "fp_count": int(((pi[ok] == 1)).sum()) if ok.any() else 0,
        })
    return rows
