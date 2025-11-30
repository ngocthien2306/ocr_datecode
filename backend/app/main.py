from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.core.config import settings
from app.db.mongodb import connect_to_mongo, close_mongo_connection, get_database
from app.repositories.user_repository import UserRepository

from app.api.endpoints import auth, users


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_mongo()

    db = get_database()
    user_repo = UserRepository(db)

    await user_repo.create_indexes()

    print("✅ Database indexes created")

    yield

    await close_mongo_connection()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/users", tags=["Users"])


@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
