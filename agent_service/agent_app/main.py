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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agent_app.api import auth, chat, fleet, station
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
# Line Station: màn hình đặt cạnh dây chuyền, chạy trên CHÍNH máy này. Chỉ đọc —
# không endpoint nào start/stop recipe hay sửa cấu hình.
app.include_router(station.router, prefix="/api")


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


# Trang Line Station. Mount trước route /station để file tĩnh đi trước.
_STATION = Path(__file__).resolve().parent.parent / "static" / "station"

# Code web DÙNG CHUNG với fleet_service: sơ đồ nhà máy 3D (1.282 dòng từng tồn
# tại hai bản và đã bắt đầu trôi khác nhau) cùng bản vendor three.js r184.
#
# Phục vụ từ ĐĨA CỦA MÁY NÀY. Line Station không được phụ thuộc fleet service
# lúc chạy — đó là toàn bộ lý do bề mặt này tồn tại — nên dùng chung ở đây là
# dùng chung MÃ NGUỒN trong repo, không phải gọi qua HTTP sang :8200.
_SHARED_WEB = Path(__file__).resolve().parent.parent.parent / "shared" / "web"


def _static_file(root: Path, asset: str, default: str = "index.html") -> FileResponse:
    """Trả một file tĩnh nằm trong `root`, chặn đường dẫn thoát ra ngoài.

    `asset` do client đưa, nên phải resolve rồi kiểm tra vẫn nằm trong `root`.
    """
    # Đường dẫn rỗng hoặc trỏ vào thư mục thì trả trang — `/station/` là cách
    # trình duyệt tự thêm dấu gạch, không phải một yêu cầu file.
    if not asset or asset.endswith("/"):
        asset = default
    target = (root / asset).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        raise HTTPException(404)
    if not target.is_file():
        raise HTTPException(404)
    mime = {".js": "text/javascript", ".css": "text/css",
            ".html": "text/html", ".svg": "image/svg+xml"}.get(target.suffix, None)
    # no-cache: màn hình treo tường chạy nhiều ngày liền, sửa xong mà tablet giữ
    # bản cũ thì không ai biết là đã sửa.
    return FileResponse(target, media_type=mime,
                        headers={"Cache-Control": "no-cache, must-revalidate"})


@app.get("/shared/{asset:path}", include_in_schema=False)
async def shared_asset(asset: str):
    """File web dùng chung với fleet_service (xem _SHARED_WEB)."""
    if not _SHARED_WEB.is_dir():
        # Nói to trong log, không chỉ trả 404: thiếu thư mục này thì sơ đồ nhà
        # máy không dựng được, mà lỗi import ES module chỉ hiện trong console
        # của trình duyệt — không ai đứng cạnh dây chuyền mở console.
        logger.error("Không thấy %s — sơ đồ nhà máy 3D sẽ không tải được. "
                     "Deploy phải copy cả thư mục shared/, không chỉ agent_service/.",
                     _SHARED_WEB)
        raise HTTPException(404)
    return _static_file(_SHARED_WEB, asset)

@app.get("/station", include_in_schema=False)
async def station_ui():
    """Màn hình cạnh dây chuyền (http://<máy>:8100/station)."""
    return FileResponse(_STATION / "index.html", media_type="text/html",
                        headers={"Cache-Control": "no-cache, must-revalidate"})


@app.get("/station/{asset:path}", include_in_schema=False)
async def station_asset(asset: str):
    """
    File tĩnh của Line Station.

    Một route chung thay vì một route mỗi file: bản trước khai riêng station.css
    và station.js, nên thêm floor.js là 404 ngay — và một màn hình xưởng thiếu
    một module thì trắng trơn, không báo gì.
    """
    return _static_file(_STATION, asset)


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
