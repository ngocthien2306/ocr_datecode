"""
ML Training Models
Pydantic models for ML training projects, annotations, and trained models.
"""
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from bson import ObjectId


class PyObjectId(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return str(v)


# ─────────────────────────────────── Segment / Annotation structures ──

class CharSegment(BaseModel):
    """One auto-segmented character box with an optional OK/NG label."""
    id: str
    x: float   # normalized 0-1, relative to full image
    y: float
    w: float
    h: float
    label: Optional[str] = None  # "OK" | "NG" | None
    char_id: Optional[str] = None  # "A", "B", "0", "/"... — enables per-char golden template


class AnnotationRegion(BaseModel):
    """User-drawn region containing auto-segmented character boxes."""
    id: str
    x: float   # normalized 0-1, relative to full image
    y: float
    w: float
    h: float
    segments: List[CharSegment] = []


# ─────────────────────────────────── ML Project ──────────────────────

class MLProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None


class MLProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class MLProjectInDB(BaseModel):
    id: str = Field(alias="_id")
    name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = None
    image_count: int = 0
    labeled_count: int = 0   # images that have at least one labeled segment
    status: str = "active"   # active | training | trained

    model_config = {"populate_by_name": True}


# ─────────────────────────────────── Annotation ──────────────────────

class MLAnnotationSave(BaseModel):
    regions: List[AnnotationRegion] = []


class MLAnnotationInDB(BaseModel):
    id: str = Field(alias="_id")
    project_id: str
    filename: str
    regions: List[AnnotationRegion] = []
    updated_at: datetime

    model_config = {"populate_by_name": True}


# ─────────────────────────────────── Trained Model ───────────────────

class MLModelMetrics(BaseModel):
    accuracy_train: float = 0.0
    accuracy_test: float = 0.0
    n_ok: int = 0
    n_ng: int = 0
    n_total: int = 0
    confusion_matrix: List[List[int]] = []
    report: str = ""


class MLModelInDB(BaseModel):
    id: str = Field(alias="_id")
    project_id: str
    algorithm: str          # "rf" | "svm" | "mlp"
    params: Dict[str, Any] = {}
    augment_factor: int = 0  # 0=off, 2=x2 ...
    metrics: MLModelMetrics = MLModelMetrics()
    model_path: str = ""
    status: str = "pending"  # pending | training | completed | failed
    error: Optional[str] = None
    created_at: datetime

    model_config = {"populate_by_name": True}


# ─────────────────────────────────── Request / Response bodies ───────

class TrainRequest(BaseModel):
    algorithm: str = "rf"           # "rf" | "svm" | "mlp"
    augment_factor: int = 0         # 0=off, 2=x2 ...
    test_split: float = 0.2         # fraction for test set
    threshold: float = 0.5          # prob_ok >= threshold → label OK
    n_estimators: int = 100         # RF only
    max_iter: int = 500             # MLP/SVM only
    C: float = 1.0                  # SVM only
    hidden_layer_sizes: List[int] = [128, 64]  # MLP only


class SyntheticPreviewRequest(BaseModel):
    augment_factor: int             # must be >= 2
    label: str = "NG"               # 'NG', 'OK', or 'BOTH'


class TestSetRequest(BaseModel):
    model_id: str


class SegmentRequest(BaseModel):
    filename: str
    region: Dict[str, float]        # {x, y, w, h} normalized


class LabeledCrop(BaseModel):
    segment_id: str
    region_id: str
    filename: str
    label: Optional[str]
    crop_b64: str                   # base64 JPEG
    char_id: Optional[str] = None   # for FE badge + golden template


class PredictResult(BaseModel):
    segment_id: str
    x: float
    y: float
    w: float
    h: float
    prob_ok: float
    label: str                      # "OK" | "NG"
    crop_b64: str
    char_id: Optional[str] = None
    # Diff heatmap preview (only when char_id has a golden in model)
    aligned_b64: Optional[str] = None
    golden_b64: Optional[str] = None
    diff_b64: Optional[str] = None


class ImportFromRecipeRequest(BaseModel):
    recipe_id: str
    camera_serial: str
    filenames: List[str]


class ImportFromRecipeResponse(BaseModel):
    imported: int
    skipped: int
    errors: List[Dict[str, str]] = []  # [{filename, reason}, ...]
    char_ids: List[str] = []            # unique char_ids auto-populated


class CharCoverageResponse(BaseModel):
    covered: List[str]
    missing: List[str]
    coverage_pct: float
    model_chars: List[str]              # full list of chars the model has goldens for


class PredictResponse(BaseModel):
    model_id: str
    algorithm: str
    results: List[PredictResult]
