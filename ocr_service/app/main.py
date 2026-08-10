import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import candidates, dataset, import_dataset, projects, train
from app.core.config import (
    BASE_CKPT_DIR,
    BUILTIN_BASES,
    CHARACTER_DICT_PATH,
    OPENOCR_DIR,
    settings,
)
from app.db.mongodb import close_mongo_connection, connect_to_mongo, get_database
from app.repositories.ocr_repository import OCRRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _check_training_prerequisites() -> None:
    """Warn at startup about anything training will need but cannot find.

    Deliberately warnings, not failures: project CRUD, the dataset tabs and
    the label tab all work without OpenOCR or the base checkpoints present,
    and the alternative is a service that refuses to boot on a machine where
    someone only wanted to review labels. The training endpoint checks again
    and fails loudly there.
    """
    if not OPENOCR_DIR.is_dir():
        logger.warning(
            f"⚠️  OpenOCR not found at {OPENOCR_DIR} — training will fail. "
            f"Clone + patch it, see ocr_service/README.md"
        )
    elif not CHARACTER_DICT_PATH.is_file():
        logger.warning(f"⚠️  Character dict missing: {CHARACTER_DICT_PATH}")

    missing = [f for f in BUILTIN_BASES.values() if not (BASE_CKPT_DIR / f).is_file()]
    if missing:
        logger.warning(
            f"⚠️  Base checkpoint(s) missing from {BASE_CKPT_DIR}: {missing} — "
            f"fine-tuning from them will fail (they are gitignored, 81 MB each)"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_mongo()
    repo = OCRRepository(get_database())
    await repo.create_indexes()
    # Training runs in a background task, so a restart mid-run orphans the
    # record — nothing left alive will ever move it off 'training'.
    stuck = await repo.reset_stuck_training()
    if stuck:
        logger.warning(f"⚠️  Reset {stuck} training run(s) orphaned by a restart")
    _check_training_prerequisites()
    yield
    await close_mongo_connection()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects.router, prefix="/api/ocr", tags=["OCR Projects"])
app.include_router(candidates.router, prefix="/api/ocr", tags=["OCR Candidates"])
app.include_router(import_dataset.router, prefix="/api/ocr", tags=["OCR Candidates"])
app.include_router(dataset.router, prefix="/api/ocr", tags=["OCR Dataset"])
app.include_router(train.router, prefix="/api/ocr", tags=["OCR Training"])


@app.get("/")
async def root():
    return {"app": settings.APP_NAME, "version": settings.APP_VERSION, "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=settings.DEBUG)
