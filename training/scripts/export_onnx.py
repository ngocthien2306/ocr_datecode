#!/usr/bin/env python3
"""
Export ONNX from an existing training run directory.

Usage
    python scripts/export_onnx.py --run runs/ce_baseline_20240101-120000
    python scripts/export_onnx.py --run runs/ce_baseline_20240101-120000 --out my_model.onnx
    python scripts/export_onnx.py --run runs/ce_baseline_20240101-120000 --opset 17

Loads config.yaml + best.pt from the run dir, exports a single-file ONNX.
Default output: <run_dir>/model.onnx  (overwrites if exists).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, metavar="RUN_DIR",
                        help="Path to run directory containing config.yaml and best.pt")
    parser.add_argument("--out", default=None, metavar="PATH",
                        help="Output .onnx path (default: <run_dir>/model.onnx)")
    parser.add_argument("--opset", type=int, default=None,
                        help="ONNX opset version (default: from config)")
    parser.add_argument("--checkpoint", default="best.pt", metavar="FILENAME",
                        help="Checkpoint filename inside run dir (default: best.pt)")
    args = parser.parse_args()

    run_dir = Path(args.run).resolve()
    if not run_dir.is_dir():
        print(f"[error] run dir not found: {run_dir}", file=sys.stderr)
        return 1

    config_path = run_dir / "config.yaml"
    ckpt_path = run_dir / args.checkpoint
    for p, label in [(config_path, "config.yaml"), (ckpt_path, args.checkpoint)]:
        if not p.exists():
            print(f"[error] {label} not found in {run_dir}", file=sys.stderr)
            return 1

    out_path = Path(args.out).resolve() if args.out else run_dir / "model.onnx"

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.utils import load_config, pick_device
    from src.models import build_model
    from src.trainer import export_onnx

    cfg = load_config(str(config_path), overrides=[])
    if args.opset is not None:
        cfg.output.onnx_opset = args.opset

    device = pick_device(cfg.device)
    print(f"[device]     {device}")
    print(f"[run_dir]    {run_dir}")
    print(f"[checkpoint] {ckpt_path.name}")
    print(f"[out]        {out_path}")

    import torch
    ckpt = torch.load(str(ckpt_path), map_location=device)
    multi_to_idx = ckpt.get("multi_to_idx", {})
    num_multi_classes = len(multi_to_idx) if multi_to_idx else 2

    model = build_model(cfg, num_multi_classes=num_multi_classes).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model]      {cfg.model.backbone} + {cfg.model.head.type}  "
          f"feat_dim={model.feat_dim}  params={n_params:.2f}M  "
          f"multi_classes={num_multi_classes}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(model, cfg, out_path)
    print(f"[done]       {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
