import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import (
    candidates, dataset, eval as eval_endpoints, export, import_dataset, projects, test_model, train,
)
from app.core.config import settings
from app.db.mongodb import close_mongo_connection, connect_to_mongo, get_database
from app.repositories.anomaly_repository import AnomalyRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_mongo()
    await AnomalyRepository(get_database()).create_indexes()
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

app.include_router(projects.router, prefix="/api/anomaly", tags=["Anomaly Projects"])
app.include_router(candidates.router, prefix="/api/anomaly", tags=["Anomaly Candidates"])
app.include_router(import_dataset.router, prefix="/api/anomaly", tags=["Anomaly Candidates"])
app.include_router(dataset.router, prefix="/api/anomaly", tags=["Anomaly Dataset"])
app.include_router(train.router, prefix="/api/anomaly", tags=["Anomaly Training"])
app.include_router(eval_endpoints.router, prefix="/api/anomaly", tags=["Anomaly Eval"])
app.include_router(export.router, prefix="/api/anomaly", tags=["Anomaly Export"])
app.include_router(test_model.router, prefix="/api/anomaly", tags=["Anomaly Test"])


@app.get("/")
async def root():
    return {"app": settings.APP_NAME, "version": settings.APP_VERSION, "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=settings.DEBUG)
