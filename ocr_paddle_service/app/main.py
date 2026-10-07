import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import export, synthetic, train
from app.core.config import PADDLE_ARCHS, PADDLEOCR_DIR, arch_pretrained_path, settings
from app.db.mongodb import close_mongo_connection, connect_to_mongo, get_database
from app.repositories.ocr_repository import OCRRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _check_training_prerequisites() -> None:
    """Warnings, not failures: listing models and serving exports work without
    the PaddleOCR checkout; the train endpoint checks again and fails loudly."""
    if not (PADDLEOCR_DIR / "tools" / "train.py").is_file():
        logger.warning(f"⚠️  PaddleOCR not found at {PADDLEOCR_DIR} — training will fail")
    missing = [a for a in PADDLE_ARCHS if not arch_pretrained_path(a).is_file()]
    if missing:
        logger.warning(f"⚠️  No pretrained weights for {missing} — those archs can only "
                       f"fine-tune from an already-trained model")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_mongo()
    repo = OCRRepository(get_database())
    await repo.create_indexes()
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

# Same /api/ocr prefix as ocr_service: the FE studio talks to either service
# with the same paths, switching only the base URL by model family.
app.include_router(train.router, prefix="/api/ocr", tags=["PaddleOCR Training"])
app.include_router(export.router, prefix="/api/ocr", tags=["PaddleOCR Export"])
app.include_router(synthetic.router, prefix="/api/ocr", tags=["PaddleOCR Synthetic defects"])


@app.get("/")
async def root():
    return {"app": settings.APP_NAME, "version": settings.APP_VERSION, "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=settings.DEBUG)
