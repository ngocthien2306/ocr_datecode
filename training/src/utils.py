"""
Foundation utilities: config loading, seeding, logging, env capture.

Config flow:
    1. Load YAML at <config_path>
    2. If `extends` field present → recursively load + merge parent first
    3. Apply CLI overrides (dotlist style: ["train.lr=1e-4", "model.backbone=resnet18"])
    4. Resolve, return frozen OmegaConf DictConfig

Run output structure:
    runs/<experiment_name>_<timestamp>/
        config.yaml       resolved frozen config
        env.json          torch/timm/git versions
        train_log.txt     stdout mirror
        ...               (other artifacts written by trainer)
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_yaml_with_extends(path: Path, _seen: Optional[set] = None) -> DictConfig:
    """Load a YAML file. If it has `extends: <relative_path>`, merge parent first."""
    path = path.resolve()
    if _seen is None:
        _seen = set()
    if path in _seen:
        raise ValueError(f"Circular extends detected at {path}")
    _seen.add(path)

    cfg = OmegaConf.load(path)
    extends = cfg.pop("extends", None) if isinstance(cfg, DictConfig) else None
    if extends:
        parent_path = (path.parent / extends).resolve()
        parent_cfg = _load_yaml_with_extends(parent_path, _seen)
        cfg = OmegaConf.merge(parent_cfg, cfg)
    return cfg


def load_config(
    config_path: str | Path,
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """
    Load a config file with `extends` support and apply CLI dotlist overrides.

    Args:
        config_path: path to a .yaml file
        overrides: list of strings like ["train.lr=1e-4", "model.backbone=resnet18"]
    """
    cfg = _load_yaml_with_extends(Path(config_path))
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    return cfg


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Save resolved config to disk for audit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path, resolve=True)


# ---------------------------------------------------------------------------
# Seeding & device
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(prefer: str = "auto") -> str:
    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Run directory & env capture
# ---------------------------------------------------------------------------

def make_run_dir(base_dir: str | Path, experiment_name: str) -> Path:
    """Create runs/<exp_name>_<YYYYMMDD-HHMMSS>/ and return it."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / f"{experiment_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def capture_env() -> dict:
    """Capture python/torch/timm versions + git commit + dataset hash placeholder."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "git_commit": _git_commit(),
        "captured_at": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        import timm
        info["timm"] = timm.__version__
    except Exception:
        pass
    try:
        import albumentations
        info["albumentations"] = albumentations.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return info


def save_env(run_dir: Path) -> None:
    env_path = run_dir / "env.json"
    env_path.write_text(json.dumps(capture_env(), indent=2))


# ---------------------------------------------------------------------------
# Logging — stdout mirror to file
# ---------------------------------------------------------------------------

class StreamTee:
    """Mirror writes to multiple streams (stdout + file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def setup_log_mirror(run_dir: Path) -> "object":
    """
    Redirect stdout to also write to runs/<exp>/train_log.txt.
    Returns the file handle so the caller can close it at teardown.
    """
    log_path = run_dir / "train_log.txt"
    fh = open(log_path, "a", buffering=1)
    sys.stdout = StreamTee(sys.__stdout__, fh)
    sys.stderr = StreamTee(sys.__stderr__, fh)
    return fh


def teardown_log_mirror(fh) -> None:
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    try:
        fh.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def fmt_n(n: int) -> str:
    """Human-readable number (1.2k, 3.4M)."""
    for unit in ["", "k", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}".rstrip("0").rstrip(".")
        n /= 1000.0
    return f"{n:.1f}T"


def now_ms() -> int:
    return int(time.time() * 1000)
