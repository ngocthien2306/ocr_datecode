"""
OCR Training models.

Project and dataset-item schemas mirror ocr_service's (those collections are
shared and read-only here). The model record is this service's own
(paddle_ocr_models) and describes a PaddleOCR CTC recogniser.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ─────────────────────────────────── Project ──────────────────────────────


class OCRProjectInDB(BaseModel):
    id: str = Field(alias="_id")
    name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = None
    # Counts come from ocr_dataset_items, not from walking the filesystem:
    # an image on disk with no verified label is not trainable, so file count
    # would overstate what a run will actually see.
    total_count: int = 0
    verified_count: int = 0
    need_review_count: int = 0
    status: str = "active"  # active | training | trained

    model_config = {"populate_by_name": True}


# ─────────────────────────────────── Dataset item ─────────────────────────
#
# Dedup key = (inspection_id, camera_serial, frame_idx, annotation_index).
# Unlike anomaly's label crops, one frame can carry several OCR regions
# (a datecode and a lot code, say), so annotation_index is part of the key.

LABEL_STATUSES = ("need_review", "verified", "rejected")
REGION_TYPES = ("text", "datecode")


class OCRDatasetItemInDB(BaseModel):
    id: str = Field(alias="_id")
    project_id: str

    # Provenance of the inspection frame this crop came from. All optional:
    # items seeded from a folder or uploaded by hand have no inspection.
    inspection_id: Optional[str] = None
    camera_serial: Optional[str] = None
    frame_idx: Optional[int] = None
    annotation_index: Optional[int] = None
    recipe_id: Optional[str] = None
    recipe_name: Optional[str] = None
    region_type: Optional[str] = None  # 'text' | 'datecode'

    # gt_text is what training uses. prefill_text is what import guessed, kept
    # separately so "did the operator actually change anything?" stays
    # answerable after the fact.
    gt_text: str = ""
    prefill_text: str = ""
    expected_text: Optional[str] = None
    recognized_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    verify_match: Optional[bool] = None

    status: str = "need_review"  # need_review | verified | rejected
    split: str = "train"         # train | test
    # Relative to the PROJECT dir (includes the leading "dataset/"), so every
    # reader joins it against dataset_fs.project_dir().
    image_path: str
    source: str = "import"       # import | upload | seed
    exclude_from_training: bool = False

    created_at: datetime
    updated_at: Optional[datetime] = None
    verified_by: Optional[str] = None
    verified_at: Optional[datetime] = None

    model_config = {"populate_by_name": True}


# ─────────────────────────────────── Trained model ────────────────────────


class OCRBaseRef(BaseModel):
    """Which weights a run fine-tunes from. No from-scratch option: a few
    thousand factory crops cannot train a recogniser from random init."""
    kind: str = "pretrained"              # 'pretrained' | 'model'
    # kind='model': any completed Paddle model with the SAME arch, from any
    # project — a broad project's model can seed a narrower one.
    project_id: Optional[str] = None
    model_id: Optional[str] = None


class OCRTrainRequest(BaseModel):
    arch: str = "PP-OCRv6_small_rec"      # key in config.PADDLE_ARCHS
    base: OCRBaseRef = OCRBaseRef()
    # 15 from pretrained measured 93-95% engine accuracy on MINCED ONION.
    epoch_num: int = 15
    batch_size: int = 32
    lr: float = 0.0005
    test_split: float = 0.2
    # Widest crop (px at H=48) a training batch is resized to; wider crops are
    # squeezed. Production keeps the aspect up to 2000 px, so this only needs
    # to cover the widest real crop (factory datecodes: p90 ~420, max ~690).
    max_train_width: int = 768
    max_text_length: int = 25
    exclude_item_ids: List[str] = []
    # Add the project's synthetic defect crops (see synthetic_defects.py) of
    # TRAIN items to the train set, so the model learns to emit '□' for a
    # badly printed char instead of reading through it.
    use_synthetic: bool = False


class OCRSyntheticRequest(BaseModel):
    per_image: int = 1                  # defect variants per localisable crop
    types: List[str] = ["stroke_break", "dot_dropout", "overprint", "blot", "char_missing"]
    test_split: float = 0.2             # must match the run's, so test crops stay out of training
    seed: int = 0
    max_chars: int = 1                  # each variant damages a random 1..max_chars characters


class OCRModelMetrics(BaseModel):
    # acc / norm_edit_dis: PaddleOCR's own eval of the CTC head during training
    # (fixed-width eval resize — for picking the best epoch only).
    acc: Optional[float] = None
    norm_edit_dis: Optional[float] = None
    best_epoch: Optional[int] = None
    n_train: int = 0
    n_test: int = 0
    # Measured on the exported artifacts with ai_services' own preprocessing
    # (grayscale, keep aspect, black pad), batch=1. These are the numbers that
    # predict the line.
    acc_onnx: Optional[float] = None
    acc_exact_onnx: Optional[float] = None
    acc_trt: Optional[float] = None
    acc_exact_trt: Optional[float] = None
    # Share of synthetic defect crops (made from TEST items, never trained on)
    # that the line would FAIL. High = catches damaged prints; compare against
    # acc_trt, which must stay high on the clean crops.
    syn_reject_onnx: Optional[float] = None
    syn_reject_trt: Optional[float] = None
    syn_n: int = 0
    n_synth_train: int = 0


class OCRModelInDB(BaseModel):
    id: str = Field(alias="_id")
    project_id: str
    # Model family, for the recipe picker and the backend resolver: every
    # record in this collection is a PaddleOCR CTC model (recipe type
    # CUSTOM_PADDLE), never an SMTR one.
    family: str = "paddle_ctc"
    arch: str = "PP-OCRv6_small_rec"
    params: Dict[str, Any] = {}
    base_label: str = ""
    # Output classes = blank + dict + space. ai_services pads a missing space
    # class, so a mismatched dict would decode silently wrong — stored so the
    # export step can assert it.
    num_classes: Optional[int] = None
    metrics: OCRModelMetrics = OCRModelMetrics()
    checkpoint_path: str = ""          # .pdparams (best_accuracy)
    config_path: Optional[str] = None
    inference_dir: Optional[str] = None
    onnx_path: Optional[str] = None
    engine_path: Optional[str] = None
    dict_path: Optional[str] = None
    status: str = "pending"  # pending | training | completed | failed | cancelled
    error: Optional[str] = None
    phase: Optional[str] = None
    progress: float = 0.0
    created_at: datetime
    completed_at: Optional[datetime] = None

    model_config = {"populate_by_name": True}
