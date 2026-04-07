from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from app.core.config import settings
from app.db.mongodb import connect_to_mongo, close_mongo_connection, get_database
from app.repositories.user_repository import UserRepository
from app.repositories.recipe_repository import RecipeRepository
from app.repositories.action_log_repository import ActionLogRepository
import logging
from pathlib import Path

from app.api.endpoints import auth, users, recipes, cameras, upload, action_logs, inference_results, trigger_simulator, agent, jetson_monitoring, storage
from app.api.websocket import camera_ws
from app.services.socketio_service import socket_app

# Setup logging to file
LOGS_DIR = Path(__file__).parent.parent.parent / "backend" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler(LOGS_DIR / 'backend.log', mode='a')  # File output
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect to MongoDB
    await connect_to_mongo()

    db = get_database()
    user_repo = UserRepository(db)
    recipe_repo = RecipeRepository(db)
    action_log_repo = ActionLogRepository(db)

    # Import and create indexes for inference results
    from app.repositories.inference_result_repository import InferenceResultRepository
    inference_result_repo = InferenceResultRepository(db)

    await user_repo.create_indexes()
    await recipe_repo.create_indexes()
    await inference_result_repo.create_indexes()
    await action_log_repo.collection.create_index([("timestamp", -1)])
    await action_log_repo.collection.create_index([("user_id", 1)])
    await action_log_repo.collection.create_index([("action_type", 1)])
    await action_log_repo.collection.create_index([("resource_type", 1)])

    print("✅ Database indexes created")

    # Initialize AI Agent system
    try:
        from app.agent.agents.orchestrator_agent import OrchestratorAgent
        from app.agent.agents.service_agent import ServiceManagementAgent
        from app.agent.agents.historical_agent import HistoricalAnalyticsAgent
        print("✅ AI Agent system initialized (3 agents registered: orchestrator, service_management, historical_analytics)")
    except Exception as e:
        print(f"⚠️ Warning: AI Agent system failed to initialize: {e}")

    # Start Jetson monitoring service
    try:
        from app.services.jetson_monitoring_service import jetson_monitoring_service
        await jetson_monitoring_service.start_monitoring()
        print("✅ Jetson monitoring service started")
    except Exception as e:
        print(f"⚠️ Warning: Jetson monitoring service failed to start: {e}")

    yield

    # Stop Jetson monitoring service
    try:
        from app.services.jetson_monitoring_service import jetson_monitoring_service
        await jetson_monitoring_service.stop_monitoring()
        print("🛑 Jetson monitoring service stopped")
    except Exception:
        pass

    await close_mongo_connection()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    first_error = exc.errors()[0]
    field = first_error["loc"][-1] if first_error.get("loc") else "field"
    msg = first_error.get("msg", "Invalid value")
    return JSONResponse(
        status_code=422,
        content={"detail": f"{field}: {msg}"}
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/users", tags=["Users"])
app.include_router(recipes.router, prefix="/api/recipes", tags=["Recipes"])
app.include_router(cameras.router, prefix="/api", tags=["Cameras"])
app.include_router(upload.router, prefix="/api/upload", tags=["Upload"])
app.include_router(action_logs.router, prefix="/api/action-logs", tags=["Action Logs"])
app.include_router(inference_results.router, prefix="/api", tags=["Inference Results"])
app.include_router(trigger_simulator.router, prefix="/api", tags=["Trigger Simulator"])
app.include_router(agent.router, prefix="/api", tags=["AI Agent"])
app.include_router(jetson_monitoring.router, prefix="/api", tags=["Jetson Monitoring"])
app.include_router(storage.router, prefix="/api", tags=["Storage Management"])

# WebSocket endpoints
app.include_router(camera_ws.router, tags=["WebSocket"])

# Mount static files for serving inference result images
UPLOADS_DIR = Path(__file__).parent.parent / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/api/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")


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


# Mount SocketIO app
app.mount("/socket.io", socket_app)
