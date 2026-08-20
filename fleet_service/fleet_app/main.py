"""
Fleet Service — quản lý nhiều máy trạm chạy OCR Datecode.

Chạy trên máy dev hoặc VPS, KHÔNG chạy trên Jetson. Nguyên tắc xuyên suốt:
fleet chỉ ĐỌC và không được là thành phần thiết yếu — 5 máy phải chạy bình
thường kể cả khi service này tắt cả tuần. Vì vậy fleet KÉO dữ liệu về (pull)
chứ không bắt edge đẩy lên (push): chiều phụ thuộc đi từ trung tâm xuống thiết
bị, không bao giờ ngược lại.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from fleet_app.api import chat, fleet
from fleet_app.core.config import SERVICE_ROOT, settings
from fleet_app.core.edge_client import client
from fleet_app.core.registry import registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


async def _poll_loop():
    """
    Quét nền: tailnet có máy nào, mỗi máy leo được tới bậc nào.

    Nuốt mọi exception rồi chạy tiếp. Vòng lặp này chết âm thầm thì dashboard
    đứng yên với dữ liệu cũ mà trông vẫn bình thường — đúng lỗi vừa gặp ở
    `jetson_monitoring_service`, nơi một lần ném exception làm tắt vòng giám sát
    vĩnh viễn trên ba máy suốt nhiều tuần.
    """
    while True:
        try:
            ms = await registry.refresh()
            ok = sum(1 for m in ms if m.state() == "ok")
            logger.info("Quét xong: %d/%d máy đang tốt", ok, len(ms))
        except Exception:
            logger.exception("Vòng quét lỗi — vẫn chạy tiếp")
        await asyncio.sleep(settings.POLL_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await client.start()
    registry.load_labels()
    try:
        await registry.refresh()
    except Exception:
        logger.exception("Lần quét đầu lỗi — service vẫn lên")
    task = asyncio.create_task(_poll_loop())
    logger.info("Fleet service sẵn sàng — %d máy", len(registry.all()))
    yield
    task.cancel()
    await client.close()


app = FastAPI(title="OCR Datecode — Fleet Service", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])
app.include_router(fleet.router)
app.include_router(chat.router)
# File tĩnh: ES module thuần, không bundler. Fleet service là một tiến trình
# FastAPI phục vụ vài file; thêm toolchain build chỉ để tách module là cái giá
# không đáng ở quy mô này.
class _NoCacheStatic(StaticFiles):
    """Buộc trình duyệt revalidate mỗi lần tải file tĩnh.

    Chuỗi `?v=` chỉ đánh dấu được file mà HTML trỏ tới trực tiếp; các module do
    ES import lồng nhau (`app.js` import `factory-map.js`) là mục cache riêng và
    vẫn được phục vụ bản cũ. Đây là dashboard nội bộ, vài chục KB trên LAN —
    revalidate mỗi lần rẻ hơn nhiều so với việc nhìn nhầm một bản đã sửa.
    """

    def file_response(self, *args, **kwargs):  # type: ignore[override]
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        return resp


app.mount("/static", _NoCacheStatic(directory=str(SERVICE_ROOT / "static")), name="static")


@app.get("/health", include_in_schema=False)
async def health():
    ms = registry.all()
    return {"status": "healthy",
            "machines": len(ms),
            "ok": sum(1 for m in ms if m.state() == "ok")}


@app.get("/", include_in_schema=False)
async def dashboard():
    # Cùng chính sách no-cache như /static. Trang này KHÔNG đi qua mount đó, nên
    # nó từng là mảnh duy nhất bị giữ cache: JS mới, HTML cũ — và importmap khai
    # báo trong HTML thì không tới được trình duyệt, three.js không resolve nổi.
    return FileResponse(SERVICE_ROOT / "static" / "index.html",
                        headers={"Cache-Control": "no-cache, must-revalidate"})
