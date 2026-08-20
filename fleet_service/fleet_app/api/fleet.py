"""
API tầng fleet — đường XÁC ĐỊNH, không LLM.

Rẻ, nhanh, chạy lại ra y hệt, và quan trọng nhất là **vẫn sống khi OpenAI hết
credit** — đã xảy ra thật, và lúc đó cả 5 agent im tiếng cùng lúc trong khi các
endpoint này vẫn trả lời bình thường.

Phép tính nằm ở `core/queries.py` để tool của agent dùng chung ĐÚNG MỘT bản. Giữ
hai bản thì dashboard và agent sẽ nói hai con số khác nhau trên cùng dữ liệu, và
không ai phát hiện cho tới khi có người mở hai màn hình cạnh nhau.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query

from fleet_app.core import queries
from fleet_app.core.config import settings
from fleet_app.core.edge_client import client
from fleet_app.core.registry import registry

router = APIRouter(prefix="/api/fleet", tags=["Fleet"])


def _one_or_409(key: str):
    """Tên khớp nhiều máy thì trả 409 kèm ứng viên, không tự chọn."""
    r = queries.resolve(key)
    if r["ok"]:
        return r["machine"]
    if r.get("ambiguous"):
        raise HTTPException(409, {"message": r["error"], "candidates": r["ambiguous"]})
    raise HTTPException(404, r["error"])


@router.get("/machines", summary="Danh sách máy và bậc năng lực")
async def list_machines() -> Dict[str, Any]:
    ms = registry.all()
    return {"count": len(ms), "machines": [m.to_dict() for m in ms],
            "poll_interval": settings.POLL_INTERVAL}


@router.get("/status", summary="Trạng thái cả đội hình, gộp song song")
async def fleet_status() -> Dict[str, Any]:
    return await queries.fleet_status()


@router.get("/production", summary="Sản lượng và vân tay kiểu lỗi của cả đội hình")
async def fleet_production(days: int = Query(7, ge=1, le=90),
                           causes: bool = Query(True)) -> Dict[str, Any]:
    return await queries.fleet_production(days=days, causes=causes)


@router.get("/machine/{key}", summary="Thẻ chi tiết một máy")
async def machine_detail(key: str) -> Dict[str, Any]:
    _one_or_409(key)
    return await queries.machine_detail(key)


@router.post("/refresh", summary="Quét lại tailnet ngay")
async def refresh_now() -> Dict[str, Any]:
    ms = await registry.refresh()
    return {"count": len(ms), "machines": [m.to_dict() for m in ms]}


@router.get("/ask/{key}", summary="Ủy quyền một câu hỏi cho agent của máy đó")
async def ask_machine(key: str, q: str = Query(..., description="Câu hỏi")) -> Dict[str, Any]:
    """
    Đường ỦY QUYỀN — đắt (một lượt LLM ở máy đích), 4–20s.

    Trả NGUYÊN VĂN câu trả lời của edge, không tóm tắt lại: agent của máy đó
    chính là chuyên gia về máy đó. Đứng giữa viết lại chỉ mất chi tiết, tốn thêm
    một lượt LLM, và thêm một chỗ để bịa.
    """
    m = _one_or_409(key)
    res = await client.chat(m.node_id, m.ip, q)
    if not res.ok:
        raise HTTPException(502, f"{m.name}: {res.error}")
    return {"machine": m.name, "node_id": m.node_id, "answer": res.data}


@router.get("/report/{name}", summary="Tải file báo cáo đã sinh")
async def download_report(name: str):
    """
    Trả file báo cáo. Chỉ nhận tên file trần trong thư mục sinh ra — chặn
    `../` để một tên file bịa ra không đọc được file khác trên máy.
    """
    from fastapi.responses import FileResponse

    from fleet_app.reports import builder

    safe = Path(name).name
    path = builder.OUT_DIR / safe
    if not path.is_file():
        raise HTTPException(404, f"Không có báo cáo {safe!r}")
    return FileResponse(str(path), filename=safe)
