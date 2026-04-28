"""
Data layer.

Two task modes:
    - binary       label ∈ {0:OK, 1:NG}, char info kept for stratification & per-char eval.
    - multi_128    label = encoded (char_folder, ok|ng) → int. Number of effective
                   classes = num_chars × 2 (minus any class filtered by min_per_class).

Augment presets:
    minimal  — translate ±3%, rotate ±2°, brightness/contrast ±0.10  (synth-rich data)
    medium   — adds blur/noise variants (use for SupCon two-view)
    strong   — adds CoarseDropout, more aggressive (for ablation)

API
    samples, char_to_idx, multi_to_idx = scan_dataset(root, min_per_class, task)
    train_samples, val_samples = stratified_split(samples, val_ratio, seed, task)
    train_tf, val_tf = build_augment(preset, image_size)
    train_loader, val_loader, n_multi_classes = make_loaders(cfg)
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Sample record
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    path: str
    char: str          # e.g. "char_A", "char_a_low", "char_hash"
    ok_ng: int         # 0 = ok, 1 = ng
    multi_idx: int     # 0..(N_multi-1), encoded (char, ok/ng); -1 if not built


# ---------------------------------------------------------------------------
# Scan + split
# ---------------------------------------------------------------------------

def scan_dataset(
    root: str | Path,
    min_per_class: int = 4,
) -> Tuple[List[Sample], Dict[str, int]]:
    """
    Walk root/char_*/ok|ng/*.png. Build samples + multi_to_idx mapping.

    Returns:
        samples: list of Sample (multi_idx already populated)
        multi_to_idx: {f"{char_folder}__{ok|ng}": int}

    Classes (multi keys) with fewer than `min_per_class` samples are dropped
    along with all their samples.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    raw: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)
    for char_dir in sorted(root.iterdir()):
        if not char_dir.is_dir() or not char_dir.name.startswith("char_"):
            continue
        for ok_ng_str, sub in [("ok", "ok"), ("ng", "ng")]:
            folder = char_dir / sub
            if not folder.is_dir():
                continue
            cls_key = f"{char_dir.name}__{sub}"
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in VALID_EXT:
                    raw[cls_key].append((str(p), char_dir.name, 0 if sub == "ok" else 1))

    raw = {k: v for k, v in raw.items() if len(v) >= min_per_class}
    if not raw:
        raise RuntimeError(f"No usable samples under {root} with min_per_class={min_per_class}")

    multi_to_idx = {k: i for i, k in enumerate(sorted(raw.keys()))}
    samples: List[Sample] = []
    for cls_key, items in raw.items():
        midx = multi_to_idx[cls_key]
        for path, char, ok_ng in items:
            samples.append(Sample(path=path, char=char, ok_ng=ok_ng, multi_idx=midx))
    return samples, multi_to_idx


def stratified_split(
    samples: List[Sample],
    val_ratio: float = 0.15,
    seed: int = 42,
    task: str = "binary",
) -> Tuple[List[Sample], List[Sample]]:
    """
    Stratify by either binary (char, ok/ng) for binary task to keep both
    OK and NG of each char in val, or by multi_idx for multi_128.
    Both end up identical here because we always group by (char, ok/ng).
    """
    rng = random.Random(seed)
    by_key = defaultdict(list)
    for s in samples:
        by_key[(s.char, s.ok_ng)].append(s)
    train, val = [], []
    for items in by_key.values():
        items = list(items); rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_ratio)))
        if n_val >= len(items):
            n_val = max(1, len(items) - 1)
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


# ---------------------------------------------------------------------------
# Dataset wrappers
# ---------------------------------------------------------------------------

class OCRDataset(Dataset):
    """
    Returns (img_tensor, target_dict).

    target_dict keys:
        ok_ng      int 0/1
        multi_idx  int 0..N_multi-1
        char       str (folder name, kept for per-char eval)
        path       str
    """

    def __init__(self, samples: List[Sample], transform: Optional[Callable] = None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            from PIL import Image
            img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
        return img

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = self._load(s.path)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        target = {
            "ok_ng": s.ok_ng,
            "multi_idx": s.multi_idx,
            "char": s.char,
            "path": s.path,
        }
        return img, target


class TwoViewDataset(Dataset):
    """Each __getitem__ returns ((view1, view2), target). For SupCon."""

    def __init__(self, base: OCRDataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        v1, t1 = self.base[idx]
        v2, _ = self.base[idx]
        return (v1, v2), t1


def collate_fn(batch):
    """Custom collate because target is a dict with mixed types (str fields)."""
    if isinstance(batch[0][0], tuple):  # TwoView
        v1 = torch.stack([b[0][0] for b in batch], dim=0)
        v2 = torch.stack([b[0][1] for b in batch], dim=0)
        imgs = (v1, v2)
    else:
        imgs = torch.stack([b[0] for b in batch], dim=0)
    tgt = {
        "ok_ng":     torch.tensor([b[1]["ok_ng"]     for b in batch], dtype=torch.long),
        "multi_idx": torch.tensor([b[1]["multi_idx"] for b in batch], dtype=torch.long),
        "char":      [b[1]["char"] for b in batch],
        "path":      [b[1]["path"] for b in batch],
    }
    return imgs, tgt


# ---------------------------------------------------------------------------
# PerClassBatchSampler — ensure each batch has multiple samples per class
# ---------------------------------------------------------------------------

class PerClassBatchSampler(Sampler[List[int]]):
    """
    Batch = n_classes_per_batch × n_samples_per_class.

    Useful for metric-learning losses (SupCon, ArcFace) when the dataset has
    many classes — random sampling rarely gives 2+ samples of same class
    in one batch.
    """

    def __init__(
        self,
        dataset: OCRDataset,
        key: str = "multi_idx",      # which target field to group by
        n_classes_per_batch: int = 16,
        n_samples_per_class: int = 4,
        num_batches: Optional[int] = None,
        seed: int = 0,
    ):
        self.key = key
        self.n_classes = n_classes_per_batch
        self.n_samples = n_samples_per_class
        self.batch_size = self.n_classes * self.n_samples

        self.cls_to_indices: Dict[int, List[int]] = defaultdict(list)
        for i, s in enumerate(dataset.samples):
            cls = s.multi_idx if key == "multi_idx" else s.ok_ng
            self.cls_to_indices[cls].append(i)
        self.classes = list(self.cls_to_indices.keys())

        if num_batches is None:
            num_batches = max(1, len(dataset) // self.batch_size)
        self.num_batches = num_batches
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            chosen = self.rng.sample(self.classes,
                                     k=min(self.n_classes, len(self.classes)))
            batch: List[int] = []
            for c in chosen:
                pool = self.cls_to_indices[c]
                if len(pool) >= self.n_samples:
                    batch.extend(self.rng.sample(pool, k=self.n_samples))
                else:
                    batch.extend(self.rng.choices(pool, k=self.n_samples))
            self.rng.shuffle(batch)
            yield batch


# ---------------------------------------------------------------------------
# Augment presets
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_augment(preset: str, image_size: int):
    """
    Returns (train_tf, val_tf). Album pipelines emitting torch tensors normalized.

    preset:
        minimal  — synth-rich data, keep aug very light
        medium   — for SupCon two-view contrastive (needs view diversity)
        strong   — adds CoarseDropout / heavier
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    common_tail = [
        A.LongestMaxSize(max_size=image_size, interpolation=1),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                      border_mode=0, value=(255, 255, 255)),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ]

    if preset == "minimal":
        train_ops = [
            A.Affine(translate_percent=(-0.03, 0.03), rotate=(-2, 2),
                     interpolation=1, cval=255, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.4),
        ]
    elif preset == "medium":
        train_ops = [
            A.Affine(translate_percent=(-0.05, 0.05), rotate=(-3, 3),
                     scale=(0.92, 1.08), interpolation=1, cval=255, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.3, 0.9), p=1.0),
                A.GaussNoise(var_limit=(5.0, 20.0), p=1.0),
            ], p=0.25),
            A.CoarseDropout(max_holes=2, max_height=5, max_width=5, fill_value=255, p=0.10),
        ]
    elif preset == "strong":
        train_ops = [
            A.Affine(translate_percent=(-0.07, 0.07), rotate=(-5, 5),
                     scale=(0.85, 1.15), interpolation=1, cval=255, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.6),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.3, 1.5), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
                A.GaussNoise(var_limit=(8.0, 40.0), p=1.0),
            ], p=0.4),
            A.CoarseDropout(max_holes=3, max_height=8, max_width=8, fill_value=255, p=0.15),
        ]
    else:
        raise ValueError(f"Unknown augment preset: {preset}")

    return A.Compose(train_ops + common_tail), A.Compose(common_tail)


# ---------------------------------------------------------------------------
# High-level factory: build everything from a config
# ---------------------------------------------------------------------------

def make_loaders(cfg) -> Tuple[DataLoader, DataLoader, Dict[str, int], List[Sample], List[Sample]]:
    """
    Args:
        cfg: OmegaConf DictConfig (with `data`, `augment`, `train` sections)
    Returns:
        train_loader, val_loader, multi_to_idx, train_samples, val_samples
    """
    samples, multi_to_idx = scan_dataset(cfg.data.root, cfg.data.min_per_class)
    train_samples, val_samples = stratified_split(
        samples, cfg.data.val_ratio, cfg.seed, cfg.data.task
    )

    train_tf, val_tf = build_augment(cfg.augment.preset, cfg.data.image_size)
    train_base = OCRDataset(train_samples, transform=train_tf)
    val_ds = OCRDataset(val_samples, transform=val_tf)

    if cfg.train.use_two_view:
        train_ds: Dataset = TwoViewDataset(train_base)
    else:
        train_ds = train_base

    common_kwargs = dict(
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    if cfg.train.sampler == "per_class":
        # PerClassBatchSampler operates on the BASE dataset (indexes into samples list).
        # When wrapped in TwoView the indexes still align (TwoView is a 1-to-1 wrapper).
        sampler = PerClassBatchSampler(
            train_base,
            key="multi_idx",
            n_classes_per_batch=cfg.train.per_class_n_classes,
            n_samples_per_class=cfg.train.per_class_n_samples,
            seed=cfg.seed,
        )
        train_loader = DataLoader(train_ds, batch_sampler=sampler, **common_kwargs)
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            drop_last=True,
            **common_kwargs,
        )

    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False, **common_kwargs
    )

    return train_loader, val_loader, multi_to_idx, train_samples, val_samples
