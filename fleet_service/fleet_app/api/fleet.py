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
        "disk_used_gb": round(disk["used_gb"], 1) if disk.get("used_gb") else None,
        "disk_free_gb": round(disk["available_gb"], 1) if disk.get("available_gb") else None,
        "disk_total_gb": round(disk["total_gb"], 1) if disk.get("total_gb") else None,
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


@router.get("/production", summary="Sản lượng và vân tay kiểu lỗi của cả đội hình")
async def fleet_production(
    days: int = Query(7, ge=1, le=90),
    causes: bool = Query(True, description="Kèm vân tay kiểu lỗi"),
) -> Dict[str, Any]:
    """
    Số liệu sản xuất gộp từ 5 máy, KHÔNG qua LLM.

    **Không xếp hạng theo pass rate.** Năm máy chạy năm sản phẩm khác nhau —
    recipe gần như không trùng nhau (đúng một recipe chung giữa hai máy trên cả
    đội hình). "M2 pass 69% còn PC-Auto-1 pass 98%" không nói lên máy nào tệ
    hơn, nó chỉ nói hai máy đang chạy hai mặt hàng khác độ khó. Xếp hạng thẳng ở
    đây là dựng sẵn một kết luận sai mà trông rất thuyết phục.

    Thứ so sánh được là **vân tay kiểu lỗi**: tỉ lệ giữa các nguyên nhân thuộc về
    pipeline OCR chứ không thuộc về mặt hàng. Một máy có `no_detection` lệch hẳn
    khỏi mặt bằng là vấn đề camera/trigger/ánh sáng, và kết luận đó đúng dù bốn
    máy kia đang chạy thứ khác.
    """
    ms = [m for m in registry.all() if m.online]

    async def one(m: Machine) -> Dict[str, Any]:
        r = await client.rollup(m.node_id, m.ip, days=days, causes=causes)
        if not r.ok:
            raise RuntimeError(r.error or "rollup lỗi")
        return r.data

    results = await fan_out(ms, one, timeout=settings.EDGE_ROLLUP_TIMEOUT + 3)
    by_id = {r["node_id"]: r for r in results}

    rows, totals = [], {"products": 0, "pass": 0, "fail": 0}
    for m in registry.all():
        r = by_id.get(m.node_id) or {}
        d = (r.get("data") or {}) if r.get("ok") else {}
        prod = d.get("production") or None
        fm = d.get("failure_modes") or None
        if prod:
            totals["products"] += prod.get("total_products") or 0
            totals["pass"] += prod.get("pass") or 0
            totals["fail"] += prod.get("fail") or 0
        rows.append({
            "node_id": m.node_id, "machine": m.name, "line": m.line,
            "state": m.state(),
            "production": prod,
            "failure_modes": fm,
            "recipes": d.get("recipes"),
            "error": r.get("error") or d.get("production_error"),
        })

    totals["pass_rate"] = (round(totals["pass"] * 100.0 / totals["products"], 2)
                           if totals["products"] else None)

    # Ma trận vân tay: hàng là máy, cột là nguyên nhân, ô là % của mẫu máy đó.
    # Đây là bảng để MẮT so sánh — chỗ nào một máy lệch hẳn khỏi cột của nó thì
    # nhìn ra ngay, mà không cần đem pass rate của hai mặt hàng khác nhau ra so.
    matrix: Dict[str, Dict[str, Any]] = {}
    if causes:
        keys: List[str] = []
        for row in rows:
            for c in ((row.get("failure_modes") or {}).get("by_cause") or []):
                if c["cause"] not in keys:
                    keys.append(c["cause"])
        for row in rows:
            fm = row.get("failure_modes") or {}
            share = {c["cause"]: c["percent_of_sample"]
                     for c in (fm.get("by_cause") or [])}
            matrix[row["machine"]] = {
                "by_cause": {k: share.get(k) for k in keys},
                "sample_products": fm.get("sample_products"),
                # Mẫu chưa phủ hết kỳ thì đây là TỈ LỆ của mẫu, không phải của cả
                # kỳ. Nói ra để không ai đọc thành số tuyệt đối.
                "sample_covers_all": fm.get("sample_covers_all"),
            }
        matrix = {"causes": keys, "by_machine": matrix} if keys else {}

    return {
        "generated_at": time.time(),
        "period_days": days,
        "coverage": coverage(results),
        "fleet_total": totals,
        "note": ("Không xếp hạng theo pass rate: các máy chạy recipe khác nhau nên "
                 "tỉ lệ pass không so trực tiếp được. Dùng vân tay kiểu lỗi để so."),
        "failure_fingerprint": matrix,
        "machines": rows,
    }
