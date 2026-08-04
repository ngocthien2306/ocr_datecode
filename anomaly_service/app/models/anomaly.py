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
    # Provenance of the inspection frame this crop came from. Optional because
    # synthetically generated images have no inspection behind them.
    inspection_id: Optional[str] = None
    camera_serial: Optional[str] = None
    frame_idx: Optional[int] = None
    recipe_id: Optional[str] = None
    recipe_name: Optional[str] = None
    label: str  # "normal" | "abnormal"
    defect_type: Optional[str] = None
    split: str  # "train" | "test"
    # Relative to the PROJECT dir (i.e. it includes the leading "dataset/"),
    # not the dataset dir — every reader joins it against dataset_fs.project_dir().
    image_path: str
    created_at: datetime

    # "import" (cropped from a real inspection) | "synthetic" (drawn by
    # defect_sim). Synthetic images are kept distinguishable so a model can be
    # evaluated with and without them — calibrating a threshold purely on drawn
    # marks and never checking against real defects is the failure mode here.
    source: str = "import"
    # Everything needed to reproduce a synthetic image: base item, stroke
    # geometry and parameters, seed.
    synthetic_params: Optional[Dict[str, Any]] = None
    # Soft exclusion — training builds a staging tree from the included set
    # rather than deleting anything (see anomaly_training._build_datamodule).
    exclude_from_training: bool = False

    model_config = {"populate_by_name": True}


# ─────────────────────────────────── Trained Model ─────────────────────────


class AnomalyTrainRequest(BaseModel):
    algorithm: str = "patchcore"  # "patchcore" | "padim"
    backbone: str = "wide_resnet50_2"
    layers: List[str] = ["layer2", "layer3"]
    coreset_sampling_ratio: float = 0.1  # patchcore only
    image_size: int = 256
    test_split: float = 0.2  # held-out slice of normal images added to test/good
    # Dataset images to leave out of this run. Training reads the filesystem,
    # so a non-empty list makes _build_datamodule stage a symlink tree of the
    # remaining images instead of pointing at the dataset directly.
    exclude_item_ids: List[str] = []


class AnomalyModelMetrics(BaseModel):
    # None (not 0.0) when the test set is single-class -- i.e. no abnormal
    # images imported yet, so AUROC/F1 are undefined rather than bad. The FE
    # renders that as "N/A". See _compute_metrics.
    image_auroc: Optional[float] = None
    image_f1: Optional[float] = None
    metrics_available: bool = False
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
