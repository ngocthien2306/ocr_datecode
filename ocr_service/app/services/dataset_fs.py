"""
Dataset filesystem layout — OpenOCR SimpleDataSet convention.

    data/projects/{project_id}/
        dataset/
            train/*.jpg
            test/*.jpg
            rec_gt_train.txt      # "train/<file>.jpg\t<label>" per line
            rec_gt_test.txt
        models/
            {model_id}.pth, {model_id}_config.yml, {model_id}_train.log.jsonl
            export/{model_id}/{rec_smtr.onnx, rec_smtr_fp16.onnx, model.engine}

Note the label files are DERIVED, not authoritative: MongoDB
(ocr_dataset_items) owns the labels and the train/test split, and every run
regenerates rec_gt_*.txt from it. That way an operator's relabel takes effect
on the next run without anyone hand-editing a text file, and a half-written
label file can never silently train the wrong thing.
"""
import shutil
from pathlib import Path
from typing import Iterable, Tuple

from app.core.config import PROJECTS_DIR

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def project_dir(project_id: str) -> Path:
    return PROJECTS_DIR / project_id


def dataset_dir(project_id: str) -> Path:
    return project_dir(project_id) / "dataset"


def split_dir(project_id: str, split: str) -> Path:
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    return dataset_dir(project_id) / split


def label_file(project_id: str, split: str) -> Path:
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    return dataset_dir(project_id) / f"rec_gt_{split}.txt"


def models_dir(project_id: str) -> Path:
    return project_dir(project_id) / "models"


def export_dir(project_id: str, model_id: str) -> Path:
    return models_dir(project_id) / "export" / model_id


def ensure_project_dirs(project_id: str) -> None:
    split_dir(project_id, "train").mkdir(parents=True, exist_ok=True)
    split_dir(project_id, "test").mkdir(parents=True, exist_ok=True)
    models_dir(project_id).mkdir(parents=True, exist_ok=True)


def delete_project_dir(project_id: str) -> None:
    d = project_dir(project_id)
    if d.exists():
        shutil.rmtree(d)


def write_label_file(project_id: str, split: str, rows: Iterable[Tuple[str, str]]) -> int:
    """Write rec_gt_{split}.txt from (image_path, label) pairs.

    image_path is project-relative and includes the leading "dataset/", which
    is how it is stored on the item; the label file needs it relative to
    data_dir (= the dataset dir), so the prefix is stripped here. Written to a
    temp file and moved into place so a crash mid-write cannot leave training
    pointed at a truncated file.
    """
    path = label_file(project_id, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for image_path, label in rows:
            rel = image_path[len("dataset/"):] if image_path.startswith("dataset/") else image_path
            f.write(f"{rel}\t{label}\n")
            n += 1
    tmp.replace(path)
    return n


def image_abs_path(project_id: str, image_path: str) -> Path:
    """Resolve a stored (project-relative) image_path to an absolute Path."""
    return project_dir(project_id) / image_path
