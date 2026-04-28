"""
Models = timm backbone + pluggable head.

Heads
    linear     feat → Linear(num_classes)              for CE / Focal / Poly / CE+Center
    projection feat → MLP(hidden) → Linear(embed_dim) → L2-normalize    for SupCon
    arcface    feat → cosine sim with learnable class anchors (margin applied during train)

build_model(cfg, num_multi_classes) → nn.Module
    forward(x, labels=None) returns dict:
        feat        always present, raw backbone features
        logits      linear or arcface (with margin)
        cos_sim     arcface only (no margin, for inference)
        embedding   projection only (L2-normalized)

Calling convention
    forward returns a dict so the trainer can ask for whatever it needs without
    forcing every head into the same tuple shape.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(feat))


class ProjectionHead(nn.Module):
    """SimCLR / SupCon style 2-layer MLP with L2-normalized output."""

    def __init__(self, in_dim: int, hidden: int = 512, out_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        z = self.fc1(feat)
        z = self.bn(z)
        z = F.relu(z, inplace=True)
        z = self.fc2(z)
        return F.normalize(z, dim=-1)


class ArcMarginHead(nn.Module):
    """
    ArcFace-style head.

    During training (labels provided): adds an additive angular margin `m`
    to the target class before scaling by `s`, then returns logits.
    During eval (no labels): returns scaled cosine similarities directly.

    Always also returns the un-modified cosine sim (useful for inference,
    nearest-centroid eval, and threshold tuning).
    """

    def __init__(self, in_dim: int, num_classes: int,
                 margin: float = 0.30, scale: float = 30.0):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.xavier_normal_(self.W)
        self.margin = float(margin)
        self.scale = float(scale)

    def forward(self, feat: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        feat_n = F.normalize(feat, dim=1)
        W_n = F.normalize(self.W, dim=1)
        cos = feat_n @ W_n.t()                                       # (B, C)
        out = {"cos_sim": cos}

        if self.training and labels is not None:
            # Apply additive margin on the target class only
            cos_t = cos.clamp(-1 + 1e-7, 1 - 1e-7)
            theta = torch.acos(cos_t)
            target_cos_with_m = torch.cos(theta + self.margin)
            one_hot = F.one_hot(labels, num_classes=cos.shape[1]).float()
            logits = self.scale * (one_hot * target_cos_with_m + (1.0 - one_hot) * cos)
        else:
            logits = self.scale * cos
        out["logits"] = logits
        return out


# ---------------------------------------------------------------------------
# Combined model
# ---------------------------------------------------------------------------

class Model(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        head_type: str,
        head_kwargs: Dict,
        pretrained: bool = True,
        probe_size: int = 64,
    ):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained,
            num_classes=0, global_pool="avg",
        )
        # Determine true feat dim by forward probe — handles models like
        # MobileNetV3 where num_features (576) != actual head input (1024).
        self.backbone.eval()
        with torch.no_grad():
            probe = torch.zeros(1, 3, probe_size, probe_size)
            self.feat_dim = int(self.backbone(probe).shape[-1])
        self.backbone.train()
        self.head_type = head_type

        if head_type == "linear":
            self.head = LinearHead(self.feat_dim, head_kwargs["num_classes"])
        elif head_type == "projection":
            self.head = ProjectionHead(
                self.feat_dim,
                hidden=head_kwargs.get("hidden", 512),
                out_dim=head_kwargs["embed_dim"],
            )
        elif head_type == "arcface":
            self.head = ArcMarginHead(
                self.feat_dim,
                num_classes=head_kwargs["num_classes"],
                margin=head_kwargs.get("margin", 0.30),
                scale=head_kwargs.get("scale", 30.0),
            )
        else:
            raise ValueError(f"Unknown head type: {head_type}")

    def forward(self, x: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x)
        out: Dict[str, torch.Tensor] = {"feat": feat}
        if self.head_type == "linear":
            out["logits"] = self.head(feat)
        elif self.head_type == "projection":
            out["embedding"] = self.head(feat)
        elif self.head_type == "arcface":
            arc_out = self.head(feat, labels)
            out["logits"] = arc_out["logits"]
            out["cos_sim"] = arc_out["cos_sim"]
        return out

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw backbone feat (always available regardless of head)."""
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg, num_multi_classes: int) -> Model:
    """
    Build a Model from cfg.

    The number of classes is task-dependent:
        binary    + linear  → 2
        multi_128 + linear  → num_multi_classes
        any       + projection → out_dim from cfg.model.head.out_dim (embed)
        any       + arcface → num_multi_classes
    """
    head_type = cfg.model.head.type
    head_kwargs: Dict = {}

    if head_type == "linear":
        if cfg.data.task == "binary":
            head_kwargs["num_classes"] = 2
        elif cfg.data.task == "multi_128":
            head_kwargs["num_classes"] = num_multi_classes
        else:
            raise ValueError(f"Unknown task: {cfg.data.task}")
    elif head_type == "projection":
        head_kwargs["embed_dim"] = cfg.model.head.out_dim
        head_kwargs["hidden"] = cfg.model.head.hidden
    elif head_type == "arcface":
        head_kwargs["num_classes"] = num_multi_classes
        head_kwargs["margin"] = cfg.model.head.margin
        head_kwargs["scale"] = cfg.model.head.scale
    else:
        raise ValueError(f"Unknown head type: {head_type}")

    model = Model(
        backbone_name=cfg.model.backbone,
        head_type=head_type,
        head_kwargs=head_kwargs,
        pretrained=cfg.model.pretrained,
        probe_size=int(cfg.data.image_size),
    )

    # Sanity check: forward pass at the configured image size
    h = w = int(cfg.data.image_size)
    with torch.no_grad():
        try:
            dummy = torch.zeros(2, 3, h, w)
            model.eval()
            _ = model(dummy)
        except Exception as e:
            raise RuntimeError(
                f"Backbone '{cfg.model.backbone}' failed at {h}x{w} input: {e}"
            ) from e
        model.train()
    return model
