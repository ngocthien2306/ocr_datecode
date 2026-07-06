#!/usr/bin/env python3
"""
Merge current OCR crops into an existing OCR recognition dataset.

Expected target layout:
  target/
    train/
    test/
    rec_gt_train.txt   lines: train/file.jpg<TAB>label
    rec_gt_test.txt    lines: test/file.jpg<TAB>label

The source is the augmented OCR inference JSON created for the bottle report.
Only the original crop for each region is merged by default, not the visual
augmentation variants.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_JSON = PROJECT_ROOT / "outputs" / "augmented_ocr_inference_results.json"
DEFAULT_TARGET = Path("/Users/ngocthien.ai/Downloads/data_ocr_merged")
DEFAULT_SUMMARY = PROJECT_ROOT / "outputs" / "ocr_training_merge_summary.json"


@dataclass(frozen=True)
class CropItem:
    src_path: Path
    label: str
    filename: str


def safe_part(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    return text.strip("_") or "text"


def image_stem(image_name: str) -> str:
    return Path(image_name).stem.replace("Image_", "")


def read_gt(path: Path) -> Tuple[List[str], Dict[str, str]]:
    if not path.exists():
        return [], {}
    lines = path.read_text(encoding="utf-8").splitlines()
    entries: Dict[str, str] = {}
    for line in lines:
        if "\t" not in line:
            continue
        rel_path, label = line.split("\t", 1)
        entries[rel_path] = label
    return lines, entries


def collect_items(source_json: Path) -> List[CropItem]:
    data = json.loads(source_json.read_text(encoding="utf-8"))
    items: List[CropItem] = []

    for dataset in data.get("datasets", []):
        dataset_slug = safe_part(dataset.get("name", "dataset")).lower()
        for image_name, result in dataset.get("results", {}).items():
            stem = image_stem(image_name)
            for region in result.get("ocr_results", []):
                region_index = int(region.get("region_index", 0)) + 1
                label = str(region.get("expected", "")).strip()
                src = Path(region.get("shown_paths", {}).get("original", ""))
                if not label or not src.exists():
                    continue
                filename = (
                    f"cur_{dataset_slug}_{stem}_r{region_index:02d}"
                    f"__exp__{safe_part(label)}.jpg"
                )
                items.append(CropItem(src_path=src, label=label, filename=filename))

    return items


def write_jpeg(src_path: Path, dst_path: Path, quality: int = 95) -> None:
    image = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read crop image: {src_path}")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"Could not write image: {dst_path}")


def backup_file(path: Path, timestamp: str) -> str | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.name}.bak_{timestamp}")
    shutil.copy2(path, backup)
    return str(backup)


def merge_dataset(
    source_json: Path,
    target: Path,
    train_ratio: float,
    seed: int,
    dry_run: bool,
) -> dict:
    items = collect_items(source_json)
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)

    train_gt = target / "rec_gt_train.txt"
    test_gt = target / "rec_gt_test.txt"
    train_lines, train_entries = read_gt(train_gt)
    test_lines, test_entries = read_gt(test_gt)
    existing_rel_paths = set(train_entries) | set(test_entries)

    new_items = [
        item for item in shuffled
        if f"train/{item.filename}" not in existing_rel_paths
        and f"test/{item.filename}" not in existing_rel_paths
    ]
    split = int(round(len(new_items) * train_ratio))
    train_items = new_items[:split]
    test_items = new_items[split:]

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backups = {}

    if not dry_run:
        target.mkdir(parents=True, exist_ok=True)
        (target / "train").mkdir(parents=True, exist_ok=True)
        (target / "test").mkdir(parents=True, exist_ok=True)
        backups["rec_gt_train"] = backup_file(train_gt, timestamp)
        backups["rec_gt_test"] = backup_file(test_gt, timestamp)

        for split_name, split_items, gt_path, lines in (
            ("train", train_items, train_gt, train_lines),
            ("test", test_items, test_gt, test_lines),
        ):
            for item in split_items:
                write_jpeg(item.src_path, target / split_name / item.filename)
                lines.append(f"{split_name}/{item.filename}\t{item.label}")
            gt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return {
        "source_json": str(source_json),
        "target": str(target),
        "dry_run": dry_run,
        "source_items": len(items),
        "new_items": len(new_items),
        "skipped_existing": len(items) - len(new_items),
        "added_train": len(train_items),
        "added_test": len(test_items),
        "train_ratio": train_ratio,
        "seed": seed,
        "expected_after_train": len(train_entries) + len(train_items),
        "expected_after_test": len(test_entries) + len(test_items),
        "backups": backups,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge current bottle OCR crops into old OCR dataset.")
    parser.add_argument("--source-json", type=Path, default=DEFAULT_SOURCE_JSON)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = merge_dataset(
        source_json=args.source_json,
        target=args.target,
        train_ratio=args.train_ratio,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
