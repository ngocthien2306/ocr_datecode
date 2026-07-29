"""
Anomaly Training Models
Pydantic models for anomaly projects, imported dataset items, and trained models.
Mirrors backend/app/models/ml_training.py's shape where it makes sense, but is
a separate schema — this service owns its own collections.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ─────────────────────────────────── Project ──────────────────────────────


class AnomalyProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None


class AnomalyProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class AnomalyProjectInDB(BaseModel):
    id: str = Field(alias="_id")
    name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = None
    normal_count: int = 0     # images in train/good + test/good
    abnormal_count: int = 0   # images in test/<defect_type>/*
    status: str = "active"    # active | training | trained

    model_config = {"populate_by_name": True}


# ─────────────────────────────────── Import provenance ────────────────────
#
# One doc per imported label-crop, dedup key = (inspection_id, camera_serial,
# frame_idx) — a recipe run produces at most one label region per frame, so
# unlike char imports we don't need an annotation_idx in the key.


class AnomalyImportSelection(BaseModel):
    inspection_id: str
    camera_serial: str
    frame_idx: int
    label: str  # "normal" | "abnormal"
    defect_type: Optional[str] = None  # required when label == "abnormal"


class AnomalyImportRequest(BaseModel):
    selections: List[AnomalyImportSelection]


class AnomalyRelabelRequest(BaseModel):
    label: str  # "normal" | "abnormal"
    defect_type: Optional[str] = None  # required when label == "abnormal"


class AnomalyBulkRelabelRequest(BaseModel):
    ids: List[str]
    label: str  # "normal" | "abnormal"
    defect_type: Optional[str] = None  # required when label == "abnormal"


class AnomalyBulkDeleteRequest(BaseModel):
    ids: List[str]


class AnomalyImportItemInDB(BaseModel):
    id: str = Field(alias="_id")
    project_id: str
    inspection_id: str
    camera_serial: str
    frame_idx: int
    recipe_id: Optional[str] = None
    recipe_name: Optional[str] = None
    label: str  # "normal" | "abnormal"
    defect_type: Optional[str] = None
    split: str  # "train" | "test"
    image_path: str  # relative to this project's dataset dir
    created_at: datetime

    model_config = {"populate_by_name": True}


# ─────────────────────────────────── Trained Model ─────────────────────────


class AnomalyTrainRequest(BaseModel):
    algorithm: str = "patchcore"  # "patchcore" | "padim"
    backbone: str = "wide_resnet50_2"
    layers: List[str] = ["layer2", "layer3"]
    coreset_sampling_ratio: float = 0.1  # patchcore only
    image_size: int = 256
    test_split: float = 0.2  # held-out slice of normal images added to test/good


class AnomalyModelMetrics(BaseModel):
    image_auroc: float = 0.0
    image_f1: float = 0.0
    threshold: float = 0.5
    n_normal_train: int = 0
    n_normal_test: int = 0
    n_abnormal_test: int = 0


class AnomalyModelInDB(BaseModel):
    id: str = Field(alias="_id")
    project_id: str
    algorithm: str
    params: Dict[str, Any] = {}
    metrics: AnomalyModelMetrics = AnomalyModelMetrics()
    checkpoint_path: str = ""
    onnx_path: Optional[str] = None
    engine_path: Optional[str] = None
    status: str = "pending"  # pending | training | completed | failed
    error: Optional[str] = None
    created_at: datetime
    phase: Optional[str] = None
    progress: float = 0.0

    model_config = {"populate_by_name": True}
