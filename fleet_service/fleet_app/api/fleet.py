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
from typing import Any, Dict, Optional

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
                           causes: bool = Query(True),
                           granularity: str = Query("day", pattern="^(hour|day|week|shift)$")) -> Dict[str, Any]:
    return await queries.fleet_production(days=days, causes=causes,
                                          granularity=granularity)


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


@router.get("/staff", summary="Nhân sự toàn nhà máy, khoá theo (máy, username)")
async def fleet_staff() -> Dict[str, Any]:
    return await queries.fleet_staff()


@router.get("/failure-images", summary="Ảnh sản phẩm lỗi từ mọi máy")
async def fleet_failure_images(
    days: int = Query(7, ge=1, le=90),
    per_machine: int = Query(6, ge=1, le=24),
    cause: Optional[str] = Query(None),
) -> Dict[str, Any]:
    return await queries.fleet_failure_images(days=days, per_machine=per_machine,
                                              cause=cause)


@router.get("/failure-image/{machine}/{img_id}", summary="Proxy ảnh thu nhỏ từ edge")
async def fleet_failure_image(machine: str, img_id: str,
                              w: int = Query(480, ge=64, le=1600)):
    """
    Trình duyệt không gắn được Bearer token vào thẻ <img>, nên ảnh đi qua fleet:
    fleet giữ token với edge, edge vẫn giữ xác thực, trình duyệt chỉ thấy fleet.
    """
    from fastapi.responses import Response

    m = _one_or_409(machine)
    r = await client.failure_image_bytes(m.node_id, m.ip, img_id, w=w)
    if not r.ok:
        raise HTTPException(r.status or 502, f"{m.name}: {r.error}")
    return Response(content=r.data, media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=86400"})


@router.get("/audit", summary="Nhật ký thao tác gộp từ mọi máy")
async def fleet_audit(
    days: int = Query(7, ge=1, le=90),
    username: Optional[str] = Query(None),
    action_type: Optional[str] = Query(None),
    include_simulated: bool = Query(False),
    per_machine: int = Query(40, ge=1, le=200),
) -> Dict[str, Any]:
    return await queries.fleet_audit(days=days, username=username,
                                     action_type=action_type,
                                     include_simulated=include_simulated,
                                     per_machine=per_machine)


@router.get("/log-errors", summary="Lỗi hệ thống của mọi máy")
async def fleet_log_errors(
    date: Optional[str] = Query(None, description="YYYY-MM-DD, mặc định hôm nay"),
    top: int = Query(8, ge=1, le=30),
) -> Dict[str, Any]:
    return await queries.fleet_log_errors(date=date, top=top)


@router.get("/avatar/{machine}", summary="Proxy ảnh đại diện từ backend của máy")
async def fleet_avatar(machine: str, p: str = Query(..., description="avatar_url của user")):
    """
    Ảnh đại diện nằm trên backend từng máy. Proxy qua fleet cùng lý do với ảnh
    lỗi: trình duyệt không gắn được token vào thẻ <img>.

    Chỉ nhận đường dẫn bắt đầu bằng `/api/upload/` — một `p` bịa ra không được
    biến endpoint này thành cổng đọc mọi URL trên máy đích.
    """
    from fastapi.responses import Response

    if not p.startswith("/api/upload/"):
        raise HTTPException(400, "Đường dẫn ảnh không hợp lệ")
    m = _one_or_409(machine)
    token = await client.token_for(m.node_id, m.ip)
    if not token:
        raise HTTPException(502, f"{m.name}: không đăng nhập được")
    url = f"http://{m.ip}:{settings.EDGE_BACKEND_PORT}{p}"
    try:
        r = await client._client.get(url, timeout=settings.EDGE_TIMEOUT,
                                     headers={"Authorization": f"Bearer {token}"})
        if r.status_code >= 400:
            raise HTTPException(r.status_code, "không lấy được ảnh")
        return Response(content=r.content,
                        media_type=r.headers.get("content-type", "image/png"),
                        headers={"Cache-Control": "public, max-age=86400"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"{m.name}: {type(e).__name__}") from e
