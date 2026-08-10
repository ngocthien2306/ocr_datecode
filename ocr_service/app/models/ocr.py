"""
OCR Training models.

Pydantic schemas for ocr_projects, ocr_dataset_items and ocr_models. Shaped
after anomaly_service/app/models/anomaly.py where it makes sense, but the
dataset item is genuinely different: OCR ground truth is a string an operator
has to review, not a normal/abnormal class, so every item carries a label
lifecycle (need_review → verified) that anomaly has no equivalent of.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ─────────────────────────────────── Project ──────────────────────────────


class OCRProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None


class OCRProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


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
    """Which checkpoint a run fine-tunes from. There is no from-scratch
    option: a few thousand factory crops cannot train SVTRv2 from random
    init."""
    kind: str = "builtin"                 # 'builtin' | 'model'
    builtin: Optional[str] = None         # key in config.BUILTIN_BASES
    # For kind='model'. May point at a project OTHER than the one training —
    # deliberately, so a broad project's model can seed a narrower one.
    project_id: Optional[str] = None
    model_id: Optional[str] = None


class OCRTrainRequest(BaseModel):
    base: OCRBaseRef = OCRBaseRef()
    # Defaults established by measurement in Phase 0, see
    # docs/ocr_training_plan.md §2 — read it before changing them.
    use_space_char: bool = True
    epoch_num: int = 50
    batch_size: int = 32
    lr: float = 0.0001
    test_split: float = 0.2
    image_h: int = 32
    image_w: int = 128
    max_text_length: int = 25
    exclude_item_ids: List[str] = []


class OCRModelMetrics(BaseModel):
    # acc is the CTC head, gtc_acc the SMTR head, min_acc the min of the two —
    # and min_acc is what best-checkpoint selection tracks. See §3.9: picking on
    # acc alone can save a run whose SMTR head is still unusable.
    acc: Optional[float] = None
    gtc_acc: Optional[float] = None
    min_acc: Optional[float] = None
    norm_edit_dis: Optional[float] = None
    best_epoch: Optional[int] = None
    n_train: int = 0
    n_test: int = 0
    # Accuracy of the exported artifacts, measured at batch=1 (batched
    # inference pads crops with -1 and costs real accuracy — a batched number
    # is not comparable to the PyTorch one).
    acc_onnx: Optional[float] = None
    acc_trt: Optional[float] = None
    acc_exact_trt: Optional[float] = None


class OCRModelInDB(BaseModel):
    id: str = Field(alias="_id")
    project_id: str
    params: Dict[str, Any] = {}
    # Human-readable description of the base, for the lineage column in the UI.
    base_label: str = ""
    # Duplicated out of params for cheap querying/filtering.
    use_space_char: bool = True
    # 99 without the space class, 100 with it. The runtime decoder and the
    # base-checkpoint picker both need this: two models with different vocab
    # sizes cannot share an engine or a post-processor.
    vocab_size: int = 100
    metrics: OCRModelMetrics = OCRModelMetrics()
    checkpoint_path: str = ""
    config_path: Optional[str] = None
    onnx_path: Optional[str] = None
    onnx_fp16_path: Optional[str] = None
    engine_path: Optional[str] = None
    dict_path: Optional[str] = None
    status: str = "pending"  # pending | training | completed | failed | cancelled
    error: Optional[str] = None
    phase: Optional[str] = None
    progress: float = 0.0
    created_at: datetime
    completed_at: Optional[datetime] = None

    model_config = {"populate_by_name": True}
