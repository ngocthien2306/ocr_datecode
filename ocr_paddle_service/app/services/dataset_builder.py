"""
Turn ocr_dataset_items rows (shared with ocr_service, read-only) into the
per-run label files PaddleOCR reads, and validate the labels before a run can
waste GPU time on them.

The split is computed exactly as ocr_service computes it (same hash, same
pinned-test rule, same item order), so an SMTR run and a Paddle run on the same
project are measured on the same test images and their accuracies compare.
"""
import hashlib
import logging
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2 as cv

from app.core.config import arch_dict_path
from app.models.ocr import OCRDatasetItemInDB
from app.services import dataset_fs

logger = logging.getLogger(__name__)

_charset_cache: Dict[str, Set[str]] = {}


def load_charset(arch: str) -> Set[str]:
    """Characters the arch's CTC head can emit: its dict plus space (every
    PP-OCRv5/v6 rec config sets use_space_char: true)."""
    if arch in _charset_cache:
        return _charset_cache[arch]
    chars = {
        line.rstrip("\n").rstrip("\r")
        for line in arch_dict_path(arch).read_text(encoding="utf-8").splitlines()
    }
    chars.discard("")
    chars.add(" ")
    _charset_cache[arch] = chars
    return chars


def label_issues(text: str, charset: Set[str], max_text_length: int) -> List[str]:
    """What PaddleOCR will silently do to this label.

    'too_long' is the dangerous one. BaseRecLabelEncode.encode returns None when
    len(text) > max_text_len, and SimpleDataSet.__getitem__ responds to None by
    fetching a DIFFERENT sample (idx+1). So an over-long label does not shrink
    the dataset, it duplicates some other image into its slot — during eval the
    reported accuracy is then over a set that is no longer the test set.
    build_dataset drops these rather than letting that happen.

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


def _materialize_gray(project_id: str, item: OCRDatasetItemInDB) -> Optional[Tuple[str, int, int]]:
    """Write (or reuse) the grayscale copy of one item. Returns
    (path relative to dataset_dir, width, height), or None if the source is
    unreadable. Reused when newer than the source, so re-runs cost nothing."""
    src = dataset_fs.image_abs_path(project_id, item.image_path)
    dst = dataset_fs.gray_dir(project_id) / f"{item.id}.png"
    if dst.is_file() and src.is_file() and dst.stat().st_mtime >= src.stat().st_mtime:
        img = cv.imread(str(dst), cv.IMREAD_GRAYSCALE)
    else:
        img = cv.imread(str(src), cv.IMREAD_GRAYSCALE)
        if img is None:
            return None
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(dst), img)
    if img is None:
        return None
    h, w = img.shape[:2]
    return f"gray/{dst.name}", w, h


def build_dataset(
    project_id: str,
    items: Sequence[OCRDatasetItemInDB],
    arch: str,
    test_split: float,
    max_text_length: int,
    model_id: Optional[str] = None,
    exclude_item_ids: Sequence[str] = (),
    dry_run: bool = False,
    synth_entries: Sequence[Dict] = (),
) -> Dict:
    """Validate, split, and (unless dry_run) write the run's label files.

    Train lines carry the image size ("path\\tlabel\\tw\\th") because
    MultiScaleDataSet with ds_width batches crops of similar aspect together
    and resizes each batch to H=48 x (48*aspect) — so long crops keep their
    shape instead of being squeezed into 320 px, as they are in production.
    """
    charset = load_charset(arch)
    excluded = set(exclude_item_ids or ())

    kept: List[OCRDatasetItemInDB] = []
    dropped: List[Dict] = []
    warnings: List[Dict] = []
    unknown_chars: Set[str] = set()

    for item in items:
        if item.id in excluded:
            continue
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

    # Synthetic defects only ever come from TRAIN items of this split: a damaged
    # copy of a test crop in training would leak the test image.
    train_ids = {i.id for i in train}
    synth_train = [e for e in synth_entries if e.get("source_id") in train_ids]

    unreadable: List[str] = []
    if not dry_run:
        if not model_id:
            raise ValueError("model_id is required to write label files")
        dataset_fs.dataset_dir(project_id).mkdir(parents=True, exist_ok=True)
        for split, pool in (("train", train), ("test", test)):
            path = dataset_fs.label_file(project_id, model_id, split)
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for item in pool:
                    got = _materialize_gray(project_id, item)
                    if got is None:
                        unreadable.append(item.id)
                        continue
                    rel, w, h = got
                    f.write(f"{rel}\t{item.gt_text}\t{w}\t{h}\n" if split == "train"
                            else f"{rel}\t{item.gt_text}\n")
                if split == "train":
                    for e in synth_train:
                        f.write(f"{e['file']}\t{e['label']}\t{e['w']}\t{e['h']}\n")
            tmp.replace(path)
        if unreadable:
            logger.warning(f"[paddle-ocr] {len(unreadable)} source image(s) unreadable, skipped")

    def _by_recipe(pool: Sequence[OCRDatasetItemInDB]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for i in pool:
            key = i.recipe_name or i.source or "unknown"
            out[key] = out.get(key, 0) + 1
        return out

    return {
        # Evaluation must score the SAME test set training was measured on —
        # filtering by split alone would re-include labels dropped above.
        "train_items": train,
        "test_items": test,
        "dry_run": dry_run,
        "n_candidates": len(items),
        "n_train": len(train),
        "n_synth_train": len(synth_train),
        "n_test": len(test),
        "split_source": "pinned" if any(i.split == "test" for i in kept) else "test_split",
        "dropped_count": len(dropped),
        "dropped": dropped[:20],
        "unknown_char_count": len(warnings),
        "unknown_chars": sorted(unknown_chars),
        "unknown_char_samples": warnings[:20],
        "unreadable_count": len(unreadable),
        "by_recipe_train": _by_recipe(train),
        "by_recipe_test": _by_recipe(test),
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
