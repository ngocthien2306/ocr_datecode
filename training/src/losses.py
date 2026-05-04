"""
Loss factory.

All losses share a uniform call signature:
    loss = loss_fn(model_out, target_dict)

where:
    model_out:    dict from Model.forward (keys: feat, logits, embedding, cos_sim)
    target_dict:  dict from data.collate_fn (keys: ok_ng, multi_idx, char, path)

Each loss decides internally which output / target field it needs and how to
combine them. The trainer passes both verbatim — no special-case branches.

Implemented:
    CELoss          standard CE + class weights + label smoothing
    FocalLoss       γ-focused CE for hard examples
    PolyLoss        Poly-1: CE + ε * (1 - p_t)
    CECenterLoss    CE + λ * Center Loss (hybrid binary metric)
    SupConLoss      Supervised Contrastive (pulls same class together in cosine geom)
    ArcFaceCELoss   CE on margin-modified logits (head already applies margin)

build_loss(cfg, num_multi_classes, train_samples) → nn.Module
    Computes class weights ('auto' → inverse frequency), passes everything
    the loss needs.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_target(cfg, target: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Pick the right target field based on task."""
    if cfg.data.task == "binary":
        return target["ok_ng"]
    elif cfg.data.task == "multi_128":
        return target["multi_idx"]
    raise ValueError(f"Unknown task: {cfg.data.task}")


def _auto_class_weights(samples, task: str, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights normalized so mean weight = 1.0."""
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for s in samples:
        c = s.ok_ng if task == "binary" else s.multi_idx
        counts[c] += 1
    counts = counts.clamp(min=1.0)
    inv = counts.sum() / (num_classes * counts)
    return inv  # mean ≈ 1.0


# ---------------------------------------------------------------------------
# CE / Focal / Poly — all binary or multi_128
# ---------------------------------------------------------------------------

class CELoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0, task: str = "binary"):
        super().__init__()
        self.task = task
        self.fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    def forward(self, out: Dict, target: Dict) -> torch.Tensor:
        logits = out["logits"]
        y = target["ok_ng"] if self.task == "binary" else target["multi_idx"]
        return self.fn(logits, y.to(logits.device))


class FocalLoss(nn.Module):
    """
    Multi-class focal loss with optional class weights.
        L = -α_y * (1 - p_y)^γ * log p_y
    """

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None,
                 task: str = "binary"):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("weight", weight if weight is not None else torch.tensor([]))
        self.task = task

    def forward(self, out: Dict, target: Dict) -> torch.Tensor:
        logits = out["logits"]
        y = (target["ok_ng"] if self.task == "binary" else target["multi_idx"]).to(logits.device)
        log_p = F.log_softmax(logits, dim=-1)
        p = log_p.exp()
        log_p_t = log_p.gather(1, y.unsqueeze(1)).squeeze(1)
        p_t = p.gather(1, y.unsqueeze(1)).squeeze(1)
        loss = -((1.0 - p_t) ** self.gamma) * log_p_t
        if self.weight.numel() > 0:
            w = self.weight.to(logits.device)[y]
            loss = loss * w
        return loss.mean()


class PolyLoss(nn.Module):
    """
    PolyLoss (Poly-1, Leng et al. 2022, https://arxiv.org/abs/2204.12511):
        L = CE + ε * (1 - p_t)
    """

    def __init__(self, eps: float = 2.0, weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0, task: str = "binary"):
        super().__init__()
        self.eps = float(eps)
        self.label_smoothing = float(label_smoothing)
        self.register_buffer("weight", weight if weight is not None else torch.tensor([]))
        self.task = task

    def forward(self, out: Dict, target: Dict) -> torch.Tensor:
        logits = out["logits"]
        y = (target["ok_ng"] if self.task == "binary" else target["multi_idx"]).to(logits.device)

        ce = F.cross_entropy(
            logits, y,
            weight=self.weight.to(logits.device) if self.weight.numel() > 0 else None,
            label_smoothing=self.label_smoothing, reduction="none",
        )
        p = F.softmax(logits, dim=-1)
        p_t = p.gather(1, y.unsqueeze(1)).squeeze(1)
        poly = ce + self.eps * (1.0 - p_t)
        return poly.mean()


# ---------------------------------------------------------------------------
# CE + Center Loss
# ---------------------------------------------------------------------------

class CenterLoss(nn.Module):
    """
    Wen et al. 2016. Pulls each sample's feature toward its class center.
    Centers are nn.Parameters updated via optimizer.
    """

    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        c = self.centers[labels]                    # (B, feat_dim)
        d = (feat - c).pow(2).sum(dim=1).mean()
        return 0.5 * d


class CECenterLoss(nn.Module):
    """CE + λ * CenterLoss. Operates on backbone features (out['feat'])."""

    def __init__(self, num_classes: int, feat_dim: int,
                 weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0,
                 center_lambda: float = 0.005, task: str = "binary"):
        super().__init__()
        self.task = task
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
        self.center = CenterLoss(num_classes, feat_dim)
        self.lam = float(center_lambda)

    def forward(self, out: Dict, target: Dict) -> torch.Tensor:
        logits = out["logits"]
        feat = out["feat"]
        y = (target["ok_ng"] if self.task == "binary" else target["multi_idx"]).to(logits.device)
        ce_loss = self.ce(logits, y)
        center_loss = self.center(feat, y)
        return ce_loss + self.lam * center_loss


# ---------------------------------------------------------------------------
# SupCon
# ---------------------------------------------------------------------------

class SupConLoss(nn.Module):
    """
    Khosla et al. 2020. Multi-positive contrastive loss using class labels.

        L_i = -1/|P(i)| * Σ_{p∈P(i)} log( exp(z_i·z_p/τ) / Σ_{a≠i} exp(z_i·z_a/τ) )

    Works with two-view (each sample appears twice) AND with single-view
    (positives = other batch members of same class).

    task='binary'      → labels = ok_ng (2 super-clusters: OK vs NG)
    task='multi_128'   → labels = multi_idx (~132 (char×OK/NG) clusters)
    """

    def __init__(self, temperature: float = 0.07, task: str = "multi_128"):
        super().__init__()
        self.t = float(temperature)
        self.task = task

    def forward(self, out: Dict, target: Dict) -> torch.Tensor:
        z = out["embedding"]                                        # (B, D), L2-normed
        key = "ok_ng" if self.task == "binary" else "multi_idx"
        y = target[key].to(z.device)
        return supcon_from_embeddings(z, y, self.t)


def supcon_from_embeddings(z: torch.Tensor, y: torch.Tensor, t: float) -> torch.Tensor:
    """SupCon loss on already-stacked (2B, D) embeddings + (2B,) labels."""
    n = z.size(0)
    sim = (z @ z.t()) / t                                           # (N, N)
    # numerical stability
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    mask_self = torch.eye(n, dtype=torch.bool, device=z.device)
    mask_pos = (y.unsqueeze(0) == y.unsqueeze(1)) & ~mask_self      # (N, N)

    exp_sim = sim.exp() * (~mask_self).float()
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    pos_count = mask_pos.float().sum(dim=1).clamp(min=1.0)
    mean_log_prob_pos = (mask_pos.float() * log_prob).sum(dim=1) / pos_count

    # Skip anchors with zero positives (avoid NaN inflation)
    has_pos = (mask_pos.sum(dim=1) > 0).float()
    loss = -(mean_log_prob_pos * has_pos).sum() / has_pos.sum().clamp(min=1.0)
    return loss


# ---------------------------------------------------------------------------
# ArcFace CE — head already applied margin, here we just CE on the result
# ---------------------------------------------------------------------------

class ArcFaceCELoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, out: Dict, target: Dict) -> torch.Tensor:
        logits = out["logits"]
        y = target["multi_idx"].to(logits.device)
        return self.fn(logits, y)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_loss(cfg, num_multi_classes: int, feat_dim: int,
               train_samples: List) -> nn.Module:
    """
    cfg.loss.type: 'ce' | 'focal' | 'poly' | 'ce_center' | 'supcon' | 'arcface'
    """
    loss_type = cfg.loss.type
    task = cfg.data.task

    # Determine number of CE classes (for class_weights)
    if task == "binary":
        n_cls = 2
    elif task == "multi_128":
        n_cls = num_multi_classes
    else:
        raise ValueError(f"Unknown task: {task}")

    # Class weights resolution
    weight: Optional[torch.Tensor] = None
    cw = cfg.loss.class_weights
    if cw == "auto":
        weight = _auto_class_weights(train_samples, task, n_cls)
    elif cw is None or cw is False:
        weight = None
    else:
        weight = torch.tensor(list(cw), dtype=torch.float32)

    if loss_type == "ce":
        return CELoss(weight=weight, label_smoothing=cfg.loss.label_smoothing, task=task)
    if loss_type == "focal":
        return FocalLoss(gamma=cfg.loss.gamma, weight=weight, task=task)
    if loss_type == "poly":
        return PolyLoss(eps=cfg.loss.poly_eps, weight=weight,
                        label_smoothing=cfg.loss.label_smoothing, task=task)
    if loss_type == "ce_center":
        return CECenterLoss(num_classes=n_cls, feat_dim=feat_dim,
                            weight=weight, label_smoothing=cfg.loss.label_smoothing,
                            center_lambda=cfg.loss.center_lambda, task=task)
    if loss_type == "supcon":
        if cfg.data.task not in ("multi_128", "binary"):
            raise ValueError("supcon requires data.task=multi_128 or binary")
        return SupConLoss(temperature=cfg.loss.temperature, task=task)
    if loss_type == "arcface":
        if cfg.data.task != "multi_128":
            raise ValueError("arcface requires data.task=multi_128")
        if cfg.model.head.type != "arcface":
            raise ValueError("arcface loss requires model.head.type=arcface")
        return ArcFaceCELoss(label_smoothing=cfg.loss.label_smoothing)

    raise ValueError(f"Unknown loss type: {loss_type}")
