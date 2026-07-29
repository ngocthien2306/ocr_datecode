"""
Dataset filesystem layout — anomalib Folder datamodule convention.

    data/projects/{project_id}/dataset/
        train/good/*.jpg               ← normal only, used to fit the model
        test/good/*.jpg                ← held-out normal, for eval
        test/<defect_type>/*.jpg       ← abnormal, for eval (image-level only;
                                          no ground_truth/ masks — see plan doc)
"""
import shutil
from pathlib import Path
from typing import Dict

from app.core.config import PROJECTS_DIR

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def project_dir(project_id: str) -> Path:
    return PROJECTS_DIR / project_id


def dataset_dir(project_id: str) -> Path:
    return project_dir(project_id) / "dataset"


def train_good_dir(project_id: str) -> Path:
    return dataset_dir(project_id) / "train" / "good"


def test_good_dir(project_id: str) -> Path:
    return dataset_dir(project_id) / "test" / "good"


def test_defect_dir(project_id: str, defect_type: str) -> Path:
    return dataset_dir(project_id) / "test" / defect_type


def models_dir(project_id: str) -> Path:
    return project_dir(project_id) / "models"


def ensure_project_dirs(project_id: str) -> None:
    train_good_dir(project_id).mkdir(parents=True, exist_ok=True)
    test_good_dir(project_id).mkdir(parents=True, exist_ok=True)
    models_dir(project_id).mkdir(parents=True, exist_ok=True)


def delete_project_dir(project_id: str) -> None:
    d = project_dir(project_id)
    if d.exists():
        shutil.rmtree(d)


def count_images(project_id: str) -> Dict[str, int]:
    """Walk the dataset dir and count normal (train/good + test/good) vs
    abnormal (test/<defect_type> for every defect_type) images."""
    ds = dataset_dir(project_id)
    if not ds.exists():
        return {"normal": 0, "abnormal": 0}

    def _count(d: Path) -> int:
        if not d.exists():
            return 0
        return len([f for f in d.glob("*") if f.suffix.lower() in ALLOWED_EXTS])

    normal = _count(ds / "train" / "good") + _count(ds / "test" / "good")
    abnormal = 0
    test_dir = ds / "test"
    if test_dir.exists():
        for child in test_dir.iterdir():
            if child.is_dir() and child.name != "good":
                abnormal += _count(child)
    return {"normal": normal, "abnormal": abnormal}


def list_defect_types(project_id: str) -> list:
    """Defect-type subfolders under test/ that actually contain images.

    Relabeling/deleting can leave an empty test/<defect_type> dir behind
    (last image moved/removed out of it). anomalib's Folder datamodule
    hard-errors ("Found 0 abnormal images in ...") if handed a defect dir
    with no images, so an empty folder must not be reported here -- it would
    otherwise break training even though the dataset genuinely has zero
    abnormal images.
    """
    test_dir = dataset_dir(project_id) / "test"
    if not test_dir.exists():
        return []
    return sorted(
        c.name for c in test_dir.iterdir()
        if c.is_dir() and c.name != "good" and has_images(c)
    )


def has_images(d: Path) -> bool:
    return d.exists() and any(f.suffix.lower() in ALLOWED_EXTS for f in d.glob("*"))
