"""
OCR Datecode — AI Agent Service

Process riêng, tách khỏi backend chính (:8000) để việc phát triển / restart
agent không đụng tới dây chuyền đang chạy.

Ranh giới với backend:
- Dùng chung MongoDB (đọc inference_results / recipe_loads / users;
  sở hữu riêng collection agent_conversations)
- Dùng chung SECRET_KEY để verify JWT do backend phát hành
- Gọi backend qua HTTP đúng 1 việc: đọc trạng thái WebSocket của camera
  service (state in-memory của backend) — xem core/backend_client.py
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agent_app.api import auth, chat, fleet
from agent_app.core.config import settings
from agent_app.db.mongodb import (
    close_mongo_connection,
    connect_to_mongo,
    ensure_indexes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_mongo()
    await ensure_indexes()

    # Import các agent để decorator @AgentRegistry.register chạy.
    # KHÔNG bọc try/except: agent service mà không có agent nào thì vô nghĩa —
    # thà fail ngay lúc start còn hơn chạy im lặng rồi trả 404 khó hiểu.
    from agent_app.agents.orchestrator_agent import OrchestratorAgent  # noqa: F401
    from agent_app.agents.service_agent import ServiceManagementAgent  # noqa: F401
    from agent_app.agents.historical_agent import HistoricalAnalyticsAgent  # noqa: F401
    from agent_app.agents.log_agent import LogAnalysisAgent  # noqa: F401
    from agent_app.agents.equipment_agent import EquipmentHealthAgent  # noqa: F401

    from agent_app.core.registry import AgentRegistry

    registered = [a["agent_id"] for a in AgentRegistry.list_agents()]
    logger.info("AI Agent system ready — %d agents: %s", len(registered), ", ".join(registered))

    if not settings.OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY chưa được cấu hình — /agent/chat sẽ lỗi khi gọi LLM")

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
    expose_headers=["*"],
)

# Cùng prefix với backend cũ (/api/agent/...) để FE chỉ cần đổi base URL.
app.include_router(auth.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
# Đường XÁC ĐỊNH cho tầng fleet: gọi thẳng tool, không qua LLM. Fleet poll 5 máy
# mỗi phút, đi qua /agent/chat thì mỗi vòng là 5 lượt LLM.
app.include_router(fleet.router, prefix="/api")


# Ảnh kết quả inference. Backend cũng serve thư mục này ở :8000, nhưng tunnel
# chỉ mở :8100 — không mount ở đây thì khách hàng xem qua tunnel sẽ thấy ảnh vỡ.
# Chỉ đọc, cùng máy nên không phải copy hay proxy.
_UPLOADS = settings.project_root / "backend" / "uploads"
if _UPLOADS.is_dir():
    app.mount("/api/uploads", StaticFiles(directory=str(_UPLOADS)), name="uploads")
else:
    logger.warning("Không thấy thư mục uploads: %s — ảnh sẽ không hiển thị", _UPLOADS)


# File báo cáo do `tools/report_tools.generate_report` sinh ra. Mount ở đây để
# link tải trong câu trả lời chat dùng được ngay, không phải nhờ backend serve.
from agent_app.tools.report_tools import REPORTS_DIR, REPORTS_URL_PREFIX  # noqa: E402

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount(REPORTS_URL_PREFIX, StaticFiles(directory=str(REPORTS_DIR)), name="reports")


@app.get("/test", include_in_schema=False)
async def test_ui():
    """
    UI test nhanh (http://localhost:8100/test).

    Serve từ chính service này để trình duyệt cùng origin — mở file:// trực tiếp
    sẽ vướng CORS khi gọi API.
    """
    return FileResponse(Path(__file__).resolve().parent.parent / "static" / "test.html")


@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "backend_url": settings.BACKEND_URL,
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
