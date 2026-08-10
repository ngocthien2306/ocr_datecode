"""
Turn ocr_dataset_items rows into the rec_gt_train.txt / rec_gt_test.txt pair
that OpenOCR's SimpleDataSet reads, and validate the labels before a run can
waste GPU time on them.

Mongo is the source of truth for labels and for the train/test assignment; the
label files are regenerated from it on every run. See dataset_fs's module
docstring.
"""
import hashlib
import logging
from typing import Dict, List, Optional, Sequence, Set, Tuple

from app.core.config import CHARACTER_DICT_PATH
from app.models.ocr import OCRDatasetItemInDB
from app.services import dataset_fs

logger = logging.getLogger(__name__)

_charset_cache: Dict[bool, Set[str]] = {}


def load_charset(use_space_char: bool) -> Set[str]:
    """Characters the model can represent, from EN_symbol_dict.txt (94 entries:
    digits, both cases, ASCII punctuation — and notably NO space, which is why
    use_space_char adds one)."""
    if use_space_char in _charset_cache:
        return _charset_cache[use_space_char]
    chars = {
        line.rstrip("\n").rstrip("\r")
        for line in CHARACTER_DICT_PATH.read_text(encoding="utf-8").splitlines()
    }
    chars.discard("")
    if use_space_char:
        chars.add(" ")
    _charset_cache[use_space_char] = chars
    return chars


def label_issues(text: str, charset: Set[str], max_text_length: int) -> List[str]:
    """What OpenOCR will silently do to this label.

    'too_long' is the dangerous one. BaseRecLabelEncode.encode returns None when
    len(text) > max_text_len (measured on the RAW text, before unknown
    characters are stripped), and SimpleDataSet.__getitem__ responds to None by
    fetching a DIFFERENT sample — a random one while training, idx+1 while
    evaluating (simple_dataset.py:163). So an over-long label does not shrink
    the dataset, it duplicates some other image into its slot. Nothing about the
    run looks wrong, and during eval the reported accuracy is computed over a
    set that is no longer the test set. materialize() drops these rather than
    letting that happen.

    'unknown_chars' is milder but still wrong: encode() skips characters it
    doesn't know, so the model is taught to read the image as the stripped text.
    """
    issues = []
    if not text.strip():
        issues.append("empty")
    if len(text) > max_text_length:
        issues.append("too_long")
    if any(c not in charset for c in text):
        issues.append("unknown_chars")
    return issues


def unknown_chars_in(text: str, charset: Set[str]) -> Set[str]:
    return {c for c in text if c not in charset}


def _test_bucket(item_id: str, test_split: float) -> bool:
    """Deterministic, size-independent test-set membership.

    Hashing the item id rather than slicing an ordered list means an item's
    assignment never changes as the dataset grows. That keeps two runs
    comparable across an import: with an index-based slice, adding 100 images
    reshuffles which images are held out, and the accuracy difference between
    runs would partly reflect a different eval set rather than a better model.
    """
    h = int(hashlib.md5(item_id.encode()).hexdigest()[:8], 16)
    return (h % 10_000) < int(test_split * 10_000)


def plan_split(
    items: Sequence[OCRDatasetItemInDB], test_split: float,
) -> Tuple[List[OCRDatasetItemInDB], List[OCRDatasetItemInDB]]:
    """Split into (train, test).

    An item's stored `split` wins — folder seeds arrive with a meaningful split
    of their own, and bulk-split exists so an operator can pin hard crops into
    eval. test_split only carves a held-out slice when nothing has been assigned
    to test at all, which is the state a fresh inspection-import leaves.
    """
    pinned_test = [i for i in items if i.split == "test"]
    if pinned_test:
        return [i for i in items if i.split != "test"], pinned_test
    if test_split <= 0:
        return list(items), []
    train, test = [], []
    for item in items:
        (test if _test_bucket(item.id, test_split) else train).append(item)
    return train, test


def build_dataset(
    project_id: str,
    items: Sequence[OCRDatasetItemInDB],
    test_split: float,
    use_space_char: bool,
    max_text_length: int,
    dry_run: bool = False,
) -> Dict:
    """Validate, split, and (unless dry_run) write the two label files.

    Returns a report the Train tab can render before committing to a run:
    per-split counts, what was dropped and why, and the per-recipe breakdown —
    a dataset that is 95% one recipe trains a model that only reads that
    recipe's font, and the count is the only place that shows up.
    """
    charset = load_charset(use_space_char)

    kept: List[OCRDatasetItemInDB] = []
    dropped: List[Dict] = []
    warnings: List[Dict] = []
    unknown_chars: Set[str] = set()

    for item in items:
        issues = label_issues(item.gt_text, charset, max_text_length)
        if "too_long" in issues or "empty" in issues:
            dropped.append({
                "id": item.id,
                "gt_text": item.gt_text[:60],
                "length": len(item.gt_text),
                "reason": "too_long" if "too_long" in issues else "empty",
            })
            continue
        if "unknown_chars" in issues:
            bad = unknown_chars_in(item.gt_text, charset)
            unknown_chars |= bad
            warnings.append({
                "id": item.id,
                "gt_text": item.gt_text[:60],
                "chars": sorted(bad),
                "reason": "unknown_chars",
            })
        kept.append(item)

    train, test = plan_split(kept, test_split)

    if not dry_run:
        dataset_fs.ensure_project_dirs(project_id)
        dataset_fs.write_label_file(project_id, "train", [(i.image_path, i.gt_text) for i in train])
        dataset_fs.write_label_file(project_id, "test", [(i.image_path, i.gt_text) for i in test])

    def _by_recipe(pool: Sequence[OCRDatasetItemInDB]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for i in pool:
            key = i.recipe_name or i.source or "unknown"
            out[key] = out.get(key, 0) + 1
        return out

    return {
        # The actual item objects for each pool. Evaluation must score the SAME
        # test set training was measured against: filtering ocr_dataset_items by
        # split alone would re-include the over-long labels dropped above, which
        # training never saw and which no model can match.
        "train_items": train,
        "test_items": test,
        "dry_run": dry_run,
        "n_candidates": len(items),
        "n_train": len(train),
        "n_test": len(test),
        "split_source": "pinned" if any(i.split == "test" for i in kept) else "test_split",
        "dropped_count": len(dropped),
        # Truncated: a wrong max_text_length can drop hundreds, and the count
        # above is what says so.
        "dropped": dropped[:20],
        "unknown_char_count": len(warnings),
        "unknown_chars": sorted(unknown_chars),
        "unknown_char_samples": warnings[:20],
        "by_recipe_train": _by_recipe(train),
        "by_recipe_test": _by_recipe(test),
        "label_files": {
            "train": str(dataset_fs.label_file(project_id, "train")),
            "test": str(dataset_fs.label_file(project_id, "test")),
        } if not dry_run else None,
    }


def blocking_reason(report: Dict, min_train: int = 8) -> Optional[str]:
    """Why this dataset cannot be trained on, or None.

    min_train is a floor, not a recommendation — under a handful of images the
    run completes and reports a meaningless accuracy, which is worse than a
    refusal.
    """
    if report["n_train"] < min_train:
        return (f"Only {report['n_train']} trainable image(s) after validation "
                f"(need at least {min_train}). Verify more labels in the Label tab.")
    return None
