from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings

SERVICE_ROOT = Path(__file__).parent.parent.parent  # ocr_service/



class Settings(BaseSettings):
    # MongoDB — same cluster/DB as backend. Shared collections (recipes,
    # inference_results) are read-only here; this service owns ocr_projects,
    # ocr_dataset_items and ocr_models.
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "ocr_datecode_db"

    # JWT — MUST match backend's SECRET_KEY/ALGORITHM so tokens issued by
    # backend's /api/auth/login are accepted here without a second login.
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"

    CORS_ORIGINS: List[str] = ["*"]

    APP_NAME: str = "OCR Training API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    # 8000 backend, 8001 anomaly_service, 8002 here.
    PORT: int = 8002

    # Root of backend/uploads — where inference_results frame images
    # (image_path field) are resolved from when cropping candidates.
    BACKEND_UPLOADS_DIR: str = "../backend/uploads"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# Absolute, resolved once at import time.
BACKEND_UPLOADS_PATH: Path = (SERVICE_ROOT / settings.BACKEND_UPLOADS_DIR).resolve()

# Per-project datasets + model artifacts. Gitignored — 15 GB of run output is
# normal here, see .gitignore.
DATA_DIR: Path = (SERVICE_ROOT / "data").resolve()
PROJECTS_DIR: Path = DATA_DIR / "projects"
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

# Vendored OpenOCR checkout. train_rec.py must be launched with this as cwd:
# training configs carry a relative character_dict_path
# ('./tools/utils/EN_symbol_dict.txt') that only resolves from here.
OPENOCR_DIR: Path = (SERVICE_ROOT / "OpenOCR").resolve()
CHARACTER_DICT_PATH: Path = OPENOCR_DIR / "tools" / "utils" / "EN_symbol_dict.txt"

# Built-in base checkpoints to fine-tune from. Shipped by the installer, not
# git (81 MB each). Keys are the `builtin` ids in OCRBaseRef.
BASE_CKPT_DIR: Path = (SERVICE_ROOT / "weights" / "base").resolve()
BUILTIN_BASES = {
    # Already fine-tuned on factory datecodes — converges in ~4 epochs.
    "datecode_2406": "svtrv2_datecode_2406.pth",
    # Upstream general pretrained (114.5k steps), has not seen factory data.
    "general": "svtrv2_smtr_gtc_rctc.pth",
}
DEFAULT_BASE = "datecode_2406"
