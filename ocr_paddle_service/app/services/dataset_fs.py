"""
Filesystem layout.

Source images are ocr_service's (shared, read-only):
    <SHARED_PROJECTS_DIR>/{project_id}/<item.image_path>

This service's own tree:
    data/projects/{project_id}/
        dataset/
            gray/{item_id}.png     # grayscale training copies (see below)
            {model_id}_train.txt   # "gray/<id>.png\\t<label>\\t<w>\\t<h>"
            {model_id}_test.txt    # "gray/<id>.png\\t<label>"
        models/
            {model_id}_config.yml, {model_id}_train.log.jsonl
            {model_id}_run/        # PaddleOCR save_model_dir (best_accuracy.*)
            export/{model_id}/{inference/, rec.onnx, rec_fp16.engine, dict.txt}

Why grayscale copies: ai_services feeds the recogniser a grayscale crop
replicated to 3 channels (ocr_utils.preprocess_text_image). PaddleOCR trains on
colour BGR by default, so a model fine-tuned on colour crops is evaluated on a
different input distribution on the line. A gray PNG decodes in PaddleOCR's
BGR mode as exactly that replicated-gray image.

Label files are per run, not per project: two runs with different settings can
be queued at once, and a shared file would be rewritten under the first.
"""
from pathlib import Path

from app.core.config import PROJECTS_DIR, SHARED_PROJECTS_DIR


def project_dir(project_id: str) -> Path:
    return PROJECTS_DIR / project_id


def dataset_dir(project_id: str) -> Path:
    return project_dir(project_id) / "dataset"


def gray_dir(project_id: str) -> Path:
    return dataset_dir(project_id) / "gray"


def label_file(project_id: str, model_id: str, split: str) -> Path:
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    return dataset_dir(project_id) / f"{model_id}_{split}.txt"


def models_dir(project_id: str) -> Path:
    return project_dir(project_id) / "models"


def run_dir(project_id: str, model_id: str) -> Path:
    return models_dir(project_id) / f"{model_id}_run"


def export_dir(project_id: str, model_id: str) -> Path:
    return models_dir(project_id) / "export" / model_id


def image_abs_path(project_id: str, image_path: str) -> Path:
    """Resolve a stored (project-relative) image_path in ocr_service's tree."""
    return SHARED_PROJECTS_DIR / project_id / image_path
