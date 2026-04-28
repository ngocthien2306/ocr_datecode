#!/usr/bin/env python3
"""
Train a single experiment.

Usage
    python scripts/train.py --config configs/ce_baseline.yaml
    python scripts/train.py --config configs/ce_baseline.yaml \
        --override model.backbone=resnet18 train.lr=1e-4 experiment_name=ce_resnet18

The override syntax is OmegaConf dotlist: <key.path>=<value>. Strings, numbers
and booleans are auto-typed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to .yaml config")
    parser.add_argument("--override", nargs="*", default=[],
                        help='Dotlist overrides, e.g. train.lr=1e-4 model.backbone=resnet18')
    parser.add_argument("--name", default=None,
                        help="Override experiment_name (shortcut for --override experiment_name=<name>)")
    args = parser.parse_args()

    # Make `src` importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.utils import load_config, make_run_dir, save_env, setup_log_mirror, teardown_log_mirror
    from src.trainer import train

    overrides = list(args.override)
    if args.name:
        overrides.append(f"experiment_name={args.name}")
    cfg = load_config(args.config, overrides=overrides)

    if "experiment_name" not in cfg:
        cfg.experiment_name = Path(args.config).stem

    run_dir = make_run_dir(cfg.output.base_dir, cfg.experiment_name)
    save_env(run_dir)
    log_fh = setup_log_mirror(run_dir)
    try:
        print(f"[config]  {args.config}")
        if overrides:
            print(f"[override] {overrides}")
        train(cfg, run_dir)
    finally:
        teardown_log_mirror(log_fh)

    return 0


if __name__ == "__main__":
    sys.exit(main())
