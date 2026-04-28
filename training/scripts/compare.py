#!/usr/bin/env python3
"""
Sweep multiple configs / overrides and aggregate metrics into a single
comparison table.

Two modes:

1. Multi-config (one config file per experiment):
    python scripts/compare.py \
        --configs configs/ce_baseline.yaml configs/ce_focal.yaml configs/supcon_128.yaml \
        --output runs/sweep_$(date +%Y%m%d)

2. Sweep one key (Cartesian product of values, all using the same base config):
    python scripts/compare.py \
        --base-config configs/ce_baseline.yaml \
        --sweep model.backbone=mobilenetv3_small_100,efficientnet_b0,resnet18 \
        --output runs/backbone_sweep

Sweeps multiple keys: pass multiple --sweep flags (full Cartesian product).

After all runs complete, writes:
    <output>/comparison.csv     summary metrics table
    <output>/comparison.md      markdown table for sharing
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


def _discover_configs(args) -> List[Tuple[str, List[str]]]:
    """Return list of (config_path, overrides) tuples, one per planned run."""
    plan: List[Tuple[str, List[str]]] = []

    if args.configs:
        for c in args.configs:
            plan.append((c, []))
        return plan

    if args.base_config and args.sweep:
        # Parse each --sweep "key=v1,v2,v3" → [(key, v1), (key, v2), ...]
        axes: List[List[Tuple[str, str]]] = []
        for s in args.sweep:
            key, vals = s.split("=", 1)
            axes.append([(key, v.strip()) for v in vals.split(",")])
        for combo in itertools.product(*axes):
            overrides = [f"{k}={v}" for k, v in combo]
            # auto experiment_name suffix
            tag = "_".join(f"{k.split('.')[-1]}-{v}" for k, v in combo)
            overrides.append(f"experiment_name=sweep_{tag}")
            plan.append((args.base_config, overrides))
        return plan

    raise SystemExit("Either --configs or (--base-config + --sweep) is required.")


def _short(value, n=120):
    s = str(value)
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*",
                        help="One config file per experiment.")
    parser.add_argument("--base-config",
                        help="Base config to use with --sweep.")
    parser.add_argument("--sweep", action="append",
                        help='Sweep axis "key=v1,v2,...". Pass multiple for Cartesian product.')
    parser.add_argument("--output", required=True,
                        help="Sweep output directory; each run lives in a subfolder.")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Common overrides applied to every run.")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="If one run fails, keep going with the others.")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.utils import load_config, make_run_dir, save_env, setup_log_mirror, teardown_log_mirror
    from src.trainer import train

    plan = _discover_configs(args)
    common = list(args.override)

    sweep_dir = Path(args.output)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] planned {len(plan)} runs → {sweep_dir}\n")

    summary: List[Dict] = []
    for i, (cfg_path, run_overrides) in enumerate(plan, start=1):
        ovr = run_overrides + common + [f"output.base_dir={sweep_dir}"]
        print(f"\n========== Run {i}/{len(plan)}: {cfg_path}  ovr={ovr}")
        cfg = load_config(cfg_path, overrides=ovr)
        if "experiment_name" not in cfg:
            cfg.experiment_name = Path(cfg_path).stem

        run_dir = make_run_dir(cfg.output.base_dir, cfg.experiment_name)
        save_env(run_dir)
        log_fh = setup_log_mirror(run_dir)

        t0 = time.time()
        status = "ok"; metrics: Dict = {}
        try:
            metrics = train(cfg, run_dir)
        except Exception as e:
            status = f"FAIL: {type(e).__name__}: {e}"
            print(f"[error] {status}")
            if not args.continue_on_error:
                teardown_log_mirror(log_fh)
                raise
        finally:
            teardown_log_mirror(log_fh)

        duration = time.time() - t0
        summary.append({
            "experiment_name": cfg.experiment_name,
            "config":          cfg_path,
            "backbone":        str(cfg.model.backbone),
            "loss":            str(cfg.loss.type),
            "task":            str(cfg.data.task),
            "head":            str(cfg.model.head.type),
            "image_size":      int(cfg.data.image_size),
            "epochs":          int(cfg.train.epochs),
            "duration_s":      round(duration, 1),
            "auc":             metrics.get("auc"),
            "argmax_balanced": metrics.get("argmax_balanced"),
            "argmax_acc":      metrics.get("argmax_acc"),
            "th_balanced":     metrics.get("th_balanced"),
            "th_ok_pass":      metrics.get("th_ok_pass"),
            "th_ng_catch":     metrics.get("th_ng_catch"),
            "threshold":       metrics.get("threshold"),
            "status":          status,
            "run_dir":         str(run_dir),
            "overrides":       _short(run_overrides),
        })

    # Write CSV
    csv_path = sweep_dir / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        if summary:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)

    # Markdown table
    md_path = sweep_dir / "comparison.md"
    if summary:
        keys = ["experiment_name", "backbone", "loss", "auc",
                "th_balanced", "th_ok_pass", "th_ng_catch", "duration_s", "status"]
        with open(md_path, "w") as f:
            f.write("# Comparison\n\n")
            f.write("| " + " | ".join(keys) + " |\n")
            f.write("| " + " | ".join("---" for _ in keys) + " |\n")
            # Sort by th_balanced desc (None last)
            rows = sorted(summary, key=lambda r: (r.get("th_balanced") or -1), reverse=True)
            for r in rows:
                cells = []
                for k in keys:
                    v = r.get(k)
                    if isinstance(v, float):
                        cells.append(f"{v:.4f}")
                    else:
                        cells.append(str(v))
                f.write("| " + " | ".join(cells) + " |\n")

    print(f"\n[sweep] wrote {csv_path}\n        wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
