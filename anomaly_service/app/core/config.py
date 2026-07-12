from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings

_SERVICE_ROOT = Path(__file__).parent.parent.parent  # anomaly_service/


class Settings(BaseSettings):
    # MongoDB — same cluster/DB as backend (shared collections: recipes,
    # inference_results). This service never writes to those collections,
    # only reads them.
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "ocr_datecode_db"

    # JWT — MUST match backend's SECRET_KEY/ALGORITHM so tokens issued by
    # backend's /api/auth/login are accepted here without a second login.
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"

    CORS_ORIGINS: List[str] = ["*"]

    APP_NAME: str = "Anomaly Training API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    PORT: int = 8001

    # Root of backend/uploads — where inference_results frame images
    # (image_path field) are resolved from. Relative paths are resolved
    # against this service's own root at startup.
    BACKEND_UPLOADS_DIR: str = "../backend/uploads"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# Absolute, resolved once at import time.
BACKEND_UPLOADS_PATH: Path = (_SERVICE_ROOT / settings.BACKEND_UPLOADS_DIR).resolve()

# Where per-project datasets + model artifacts live.
DATA_DIR: Path = (_SERVICE_ROOT / "data").resolve()
PROJECTS_DIR: Path = DATA_DIR / "projects"
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
