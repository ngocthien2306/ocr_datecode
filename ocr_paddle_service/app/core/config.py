from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings

SERVICE_ROOT = Path(__file__).parent.parent.parent  # ocr_paddle_service/
REPO_ROOT = SERVICE_ROOT.parent


class Settings(BaseSettings):
    # MongoDB — same cluster/DB as backend.
    #
    # Projects and labelled dataset items are SHARED with ocr_service (:8002)
    # and read-only here: the operator labels once in the OCR Studio and picks
    # the architecture on the Train tab. This service owns only its model
    # records, in a separate collection so neither service ever sees the
    # other's runs as its own.
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "ocr_datecode_db"
    PROJECTS_COLLECTION: str = "ocr_projects"
    ITEMS_COLLECTION: str = "ocr_dataset_items"
    MODELS_COLLECTION: str = "paddle_ocr_models"

    # JWT — MUST match backend's SECRET_KEY/ALGORITHM so tokens issued by
    # backend's /api/auth/login are accepted here without a second login.
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"

    CORS_ORIGINS: List[str] = ["*"]

    APP_NAME: str = "PaddleOCR Training API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    # 8000 backend, 8001 anomaly_service, 8002 ocr_service, 8003 here.
    PORT: int = 8003

    # ocr_service's data dir — where the shared dataset images live. Item
    # image_path values are relative to <this>/projects/{project_id}/.
    SHARED_OCR_DATA_DIR: str = "../ocr_service/data"

    # Conda env whose TensorRT matches ai_services' runtime (engines are not
    # portable across TensorRT versions). Engine builds and the engine
    # accuracy check run as subprocesses in it — never in this process.
    TRT_CONDA_ENV: str = "vision"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

SHARED_PROJECTS_DIR: Path = (SERVICE_ROOT / settings.SHARED_OCR_DATA_DIR / "projects").resolve()

# This service's own per-project dirs: grayscale training copies, label files,
# checkpoints, exports. Gitignored.
DATA_DIR: Path = (SERVICE_ROOT / "data").resolve()
PROJECTS_DIR: Path = DATA_DIR / "projects"
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

# PaddleOCR checkout (git clone --depth 1, gitignored). tools/train.py and
# tools/export_model.py run from here — configs carry relative dict paths.
PADDLEOCR_DIR: Path = (SERVICE_ROOT / "PaddleOCR").resolve()
# Official *_pretrained.pdparams used as Global.pretrained_model.
PADDLE_PRETRAINED_DIR: Path = (SERVICE_ROOT / "weights" / "paddle_pretrained").resolve()
PRETRAINED_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/"
    "{name}_pretrained.pdparams"
)

# Architectures the Train tab offers. Every one is a CTC recogniser at H=48 —
# the same contract as ai_services' PADDLEV5/PADDLEV6 runtime
# (TextRecognizerTRT: grayscale, keep aspect, (x/255-0.5)/0.5, greedy CTC) —
# so an exported model runs there with only its own dict swapped in.
# The FULL upstream dict is kept on purpose: the pretrained CTC head is sized to
# it, and shrinking it would throw that head away.
PADDLE_ARCHS = {
    "PP-OCRv6_small_rec": {
        "label": "PP-OCRv6 small (5.2M)",
        "config": "configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml",
        "dict": "ppocr/utils/dict/ppocrv6_dict.txt",
        "recommended": True,
    },
    "PP-OCRv6_tiny_rec": {
        "label": "PP-OCRv6 tiny",
        "config": "configs/rec/PP-OCRv6/PP-OCRv6_tiny_rec.yml",
        "dict": "ppocr/utils/dict/ppocrv6_tiny_dict.txt",
    },
    "PP-OCRv6_medium_rec": {
        "label": "PP-OCRv6 medium",
        "config": "configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml",
        "dict": "ppocr/utils/dict/ppocrv6_dict.txt",
    },
    "PP-OCRv5_mobile_rec": {
        "label": "PP-OCRv5 mobile",
        "config": "configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml",
        "dict": "ppocr/utils/dict/ppocrv5_dict.txt",
    },
    "PP-OCRv5_server_rec": {
        "label": "PP-OCRv5 server",
        "config": "configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml",
        "dict": "ppocr/utils/dict/ppocrv5_dict.txt",
    },
}
DEFAULT_ARCH = "PP-OCRv6_small_rec"

# Input contract shared with ai_services' TextRecognizerTRT.
REC_IMAGE_H = 48
# Engine optimization profile, same as scripts/build_ocr_engines.py 'paddlev6'.
TRT_MIN_SHAPE = (1, 3, 48, 50)
TRT_OPT_SHAPE = (1, 3, 48, 320)
TRT_MAX_SHAPE = (1, 3, 48, 2000)


def arch_pretrained_path(arch: str) -> Path:
    return PADDLE_PRETRAINED_DIR / f"{arch}_pretrained.pdparams"


def arch_dict_path(arch: str) -> Path:
    return PADDLEOCR_DIR / PADDLE_ARCHS[arch]["dict"]
