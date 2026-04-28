#!/usr/bin/env python3
"""
Train a single experiment.

Usage
    python scripts/train.py --config configs/ce_baseline.yaml
    python scripts/train.py --config configs/ce_baseline.yaml \
        --override model.backbone=resnet18 train.lr=1e-4 experiment_name=ce_resnet18
    python scripts/train.py --resume runs/ce_baseline_20240101-120000

The override syntax is OmegaConf dotlist: <key.path>=<value>. Strings, numbers
and booleans are auto-typed.

--resume: path to an existing run_dir. Loads config.yaml from that dir, resumes
training from best.pt checkpoint (model + optimizer + scheduler state), continues
for the remaining epochs. --config/--override/--name are ignored when resuming.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to .yaml config")
    parser.add_argument("--override", nargs="*", default=[],
                        help='Dotlist overrides, e.g. train.lr=1e-4 model.backbone=resnet18')
    parser.add_argument("--name", default=None,
                        help="Override experiment_name (shortcut for --override experiment_name=<name>)")
    parser.add_argument("--resume", default=None, metavar="RUN_DIR",
                        help="Resume from an existing run directory (loads config.yaml + best.pt)")
    args = parser.parse_args()

    if args.resume is None and args.config is None:
        parser.error("Either --config or --resume is required")

    # Make `src` importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.utils import load_config, make_run_dir, save_env, setup_log_mirror, teardown_log_mirror
    from src.trainer import train

    if args.resume:
        resume_dir = Path(args.resume).resolve()
        if not resume_dir.is_dir():
            print(f"[error]   resume dir not found: {resume_dir}", file=sys.stderr)
            return 1
        config_path = resume_dir / "config.yaml"
        if not config_path.exists():
            print(f"[error]   config.yaml not found in {resume_dir}", file=sys.stderr)
            return 1
        checkpoint_path = resume_dir / "best.pt"
        if not checkpoint_path.exists():
            print(f"[error]   best.pt not found in {resume_dir}", file=sys.stderr)
            return 1
        cfg = load_config(str(config_path), overrides=[])
        run_dir = resume_dir
        print(f"[resume]  {resume_dir}")
        print(f"[ckpt]    {checkpoint_path}")
    else:
        overrides = list(args.override)
        if args.name:
            overrides.append(f"experiment_name={args.name}")
        cfg = load_config(args.config, overrides=overrides)
        if "experiment_name" not in cfg:
            cfg.experiment_name = Path(args.config).stem
        run_dir = make_run_dir(cfg.output.base_dir, cfg.experiment_name)
        checkpoint_path = None
        print(f"[config]  {args.config}")
        if overrides:
            print(f"[override] {overrides}")

    save_env(run_dir)
    log_fh = setup_log_mirror(run_dir)
    try:
        train(cfg, run_dir, resume_checkpoint=checkpoint_path)
    finally:
        teardown_log_mirror(log_fh)

    return 0


if __name__ == "__main__":
    sys.exit(main())
