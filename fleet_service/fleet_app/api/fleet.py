"""
API tầng fleet — giai đoạn 1, không có LLM.

Mọi endpoint ở đây đi đường XÁC ĐỊNH: gọi thẳng endpoint không-LLM của edge.
Chúng rẻ, nhanh, chạy lại ra y hệt, và quan trọng nhất là **vẫn sống khi
OpenAI hết credit** — đã xảy ra thật, và lúc đó cả 5 agent im tiếng cùng lúc
trong khi các endpoint này vẫn trả lời bình thường.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from fleet_app.core.config import settings
from fleet_app.core.edge_client import client
from fleet_app.core.fanout import coverage, fan_out
from fleet_app.core.registry import Machine, registry

router = APIRouter(prefix="/api/fleet", tags=["Fleet"])


def _metrics_view(raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Rút gọn số liệu phần cứng cho dashboard.

    Nhiệt độ `None` được GIỮ NGUYÊN là None. Không đổi thành 0: PC-Auto-1 là máy
    x86 không có cảm biến kiểu Jetson, và "0°C" đọc như máy đang rất mát — sai
    đúng theo hướng nguy hiểm nhất. Giao diện hiện "—" cho None.
    """
    if not raw:
        return None
    cpu, gpu = raw.get("cpu") or {}, raw.get("gpu") or {}
    ram, disk = raw.get("ram") or {}, raw.get("disk") or {}
    return {
        "cpu_percent": cpu.get("usage_percent"),
        "cpu_temp": cpu.get("temperature_celsius"),
        "gpu_percent": gpu.get("usage_percent"),
        "gpu_temp": gpu.get("temperature_celsius"),
        "ram_percent": ram.get("usage_percent"),
        "ram_used_gb": round(ram["used_mb"] / 1024, 2) if ram.get("used_mb") else None,
        "ram_total_gb": round(ram["total_mb"] / 1024, 2) if ram.get("total_mb") else None,
        "disk_percent": disk.get("usage_percent"),
        "disk_free_gb": round(disk["available_gb"], 1) if disk.get("available_gb") else None,
        "uptime_seconds": raw.get("uptime_seconds"),
        # nvp_model là CHẾ ĐỘ ĐIỆN của Jetson (MAXN_SUPER, 15W, 25W…), không phải
        # model thiết bị. Model lấy từ registry, xem Machine.model.
        "power_mode": raw.get("nvp_model"),
    }


@router.get("/machines", summary="Danh sách máy và bậc năng lực")
async def list_machines() -> Dict[str, Any]:
    ms = registry.all()
    return {
        "count": len(ms),
        "machines": [m.to_dict() for m in ms],
        "poll_interval": settings.POLL_INTERVAL,
    }


@router.get("/status", summary="Trạng thái cả đội hình, gộp song song")
async def fleet_status() -> Dict[str, Any]:
    """Một màn hình cho cả đội hình. Không LLM, không tốn tiền."""
    ms = [m for m in registry.all() if m.online]

    async def one(m: Machine) -> Dict[str, Any]:
        met = await client.system_metrics(m.ip)
        svc = await client.service_status(m.node_id, m.ip)
        return {
            "metrics": _metrics_view((met.data or {}).get("data") if met.ok else None),
            "metrics_error": None if met.ok else met.error,
            "service": svc.data if svc.ok else None,
            "service_error": None if svc.ok else svc.error,
        }

    results = await fan_out(ms, one, timeout=settings.EDGE_TIMEOUT + 2)
    by_id = {r["node_id"]: r for r in results}

    rows, degraded = [], []
    for m in registry.all():
        r = by_id.get(m.node_id) or {}
        d = r.get("data") or {}
        errors = [e for e in (d.get("metrics_error"), d.get("service_error"),
                              r.get("error")) if e]
        rows.append({
            **m.to_dict(),
            "metrics": d.get("metrics"),
            "service": d.get("service"),
            "errors": errors,
        })
        # Máy trả về được một phần vẫn phải đếm là thiếu. Lời gọi fan-out không
        # ném exception không có nghĩa là dữ liệu đầy đủ — tắt agent thì
        # `system_metrics` vẫn về (backend còn sống) còn `service_status` thì
        # lỗi, và nếu chỉ nhìn `ok` thì báo "đủ cả 5 máy" trong khi một máy hỏng.
        if errors and r.get("ok"):
            degraded.append({"machine": m.name, "reason": "; ".join(errors)})

    return {
        "generated_at": time.time(),
        "coverage": coverage(results, degraded),
        "machines": rows,
    }


@router.get("/machine/{key}", summary="Thẻ chi tiết một máy")
async def machine_detail(key: str) -> Dict[str, Any]:
    """
    Gộp mọi nguồn rẻ của MỘT máy. Đo được ~0,5s, 0 lượt LLM.

    Tên mơ hồ thì trả 409 kèm danh sách ứng viên chứ không tự chọn: bốn máy ở đây
    trùng hostname `suntech-desktop`, đoán bừa nghĩa là đưa số liệu của máy khác
    mà người dùng không hề biết.
    """
    cands = registry.resolve(key)
    if not cands:
        raise HTTPException(404, f"Không có máy nào khớp {key!r}")
    if len(cands) > 1:
        raise HTTPException(409, {
            "message": f"{key!r} khớp nhiều máy — nêu rõ hơn",
            "candidates": [{"node_id": m.node_id, "name": m.name, "ip": m.ip}
                           for m in cands],
        })

    m = cands[0]
    await registry.probe(m)
    met = await client.system_metrics(m.ip)
    svc = await client.service_status(m.node_id, m.ip)

    return {
        **m.to_dict(),
        "metrics": _metrics_view((met.data or {}).get("data") if met.ok else None),
        "metrics_error": None if met.ok else met.error,
        "service": svc.data if svc.ok else None,
        "service_error": None if svc.ok else svc.error,
    }


@router.post("/refresh", summary="Quét lại tailnet ngay")
async def refresh_now() -> Dict[str, Any]:
    ms = await registry.refresh()
    return {"count": len(ms), "machines": [m.to_dict() for m in ms]}


@router.get("/ask/{key}", summary="Ủy quyền một câu hỏi cho agent của máy đó")
async def ask_machine(key: str, q: str = Query(..., description="Câu hỏi")) -> Dict[str, Any]:
    """
    Đường ỦY QUYỀN — đắt (một lượt LLM ở máy đích), 4–20s.

    Fleet cố ý KHÔNG tóm tắt lại câu trả lời: agent của máy đó chính là chuyên
    gia về máy đó. Đứng giữa viết lại chỉ mất chi tiết, tốn thêm một lượt LLM, và
    thêm một chỗ để bịa. Trả nguyên văn, chỉ gắn thêm nhãn máy.
    """
    cands = registry.resolve(key)
    if not cands:
        raise HTTPException(404, f"Không có máy nào khớp {key!r}")
    if len(cands) > 1:
        raise HTTPException(409, {
            "message": f"{key!r} khớp nhiều máy — nêu rõ hơn",
            "candidates": [{"node_id": m.node_id, "name": m.name} for m in cands],
        })

    m = cands[0]
    res = await client.chat(m.node_id, m.ip, q)
    if not res.ok:
        raise HTTPException(502, f"{m.name}: {res.error}")
    return {"machine": m.name, "node_id": m.node_id, "answer": res.data}
