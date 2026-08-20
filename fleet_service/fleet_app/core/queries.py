"""
Phép tính dùng chung cho cả API REST lẫn tool của agent.

Vì sao tách ra: nếu để logic nằm trong hàm endpoint rồi tool tự gọi lại edge và
tự gộp, ta có HAI bản của cùng một phép tính — và chúng sẽ trôi khỏi nhau. Lúc đó
dashboard nói một đằng, agent nói một nẻo, trên cùng một dữ liệu. Sửa một chỗ,
quên chỗ kia, là kiểu lỗi không ai phát hiện cho tới khi có người đối chiếu hai
màn hình.

Mọi hàm ở đây trả về dict thuần, không ném exception ra ngoài, và máy hỏng luôn
nằm trong kết quả kèm lý do.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fleet_app.core.config import settings
from fleet_app.core.edge_client import client
from fleet_app.core.fanout import coverage, fan_out
from fleet_app.core.registry import Machine, registry


def metrics_view(raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Rút gọn số liệu phần cứng.

    Nhiệt độ `None` được GIỮ NGUYÊN. Không đổi thành 0: máy x86 không có cảm biến
    kiểu Jetson, và "0°C" đọc như máy đang rất mát — sai đúng theo hướng nguy
    hiểm nhất.
    """
    if not raw:
        return None
    cpu, gpu = raw.get("cpu") or {}, raw.get("gpu") or {}
    ram, disk = raw.get("ram") or {}, raw.get("disk") or {}
    r1 = lambda v: round(v, 1) if v is not None else None      # noqa: E731
    return {
        "cpu_percent": cpu.get("usage_percent"),
        "cpu_temp": cpu.get("temperature_celsius"),
        "gpu_percent": gpu.get("usage_percent"),
        "gpu_temp": gpu.get("temperature_celsius"),
        "ram_percent": ram.get("usage_percent"),
        "ram_used_gb": round(ram["used_mb"] / 1024, 2) if ram.get("used_mb") else None,
        "ram_total_gb": round(ram["total_mb"] / 1024, 2) if ram.get("total_mb") else None,
        "disk_percent": disk.get("usage_percent"),
        "disk_used_gb": r1(disk.get("used_gb")),
        "disk_free_gb": r1(disk.get("available_gb")),
        "disk_total_gb": r1(disk.get("total_gb")),
        "uptime_seconds": raw.get("uptime_seconds"),
        # nvp_model là CHẾ ĐỘ ĐIỆN của Jetson (MAXN_SUPER, 15W…), không phải model
        # thiết bị. Model là thuộc tính tĩnh, lấy từ registry.
        "power_mode": raw.get("nvp_model"),
    }


def resolve(key: str) -> Dict[str, Any]:
    """
    Tìm máy theo tên người gõ.

    Trả `{"ok": False, "ambiguous": [...]}` khi khớp nhiều máy thay vì chọn cái
    đầu tiên. Bốn máy trong đội hình này cùng hostname `suntech-desktop`; đoán bừa
    nghĩa là trả số liệu của máy khác mà người hỏi không hề biết — sai còn tệ hơn
    là không trả lời.
    """
    cands = registry.resolve(key)
    if not cands:
        known = [m.name for m in registry.all()]
        return {"ok": False, "error": f"Không có máy nào tên {key!r}",
                "known_machines": known}
    if len(cands) > 1:
        return {"ok": False, "ambiguous": [m.name for m in cands],
                "error": f"{key!r} khớp nhiều máy — cần nêu rõ tên nào"}
    return {"ok": True, "machine": cands[0]}


async def fleet_status() -> Dict[str, Any]:
    """Trạng thái + phần cứng cả đội hình, gộp song song. Không LLM."""
    ms = [m for m in registry.all() if m.online]

    async def one(m: Machine) -> Dict[str, Any]:
        met = await client.system_metrics(m.ip)
        svc = await client.service_status(m.node_id, m.ip)
        return {
            "metrics": metrics_view((met.data or {}).get("data") if met.ok else None),
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
        rows.append({**m.to_dict(), "metrics": d.get("metrics"),
                     "service": d.get("service"), "errors": errors})
        # Lời gọi không ném exception KHÔNG có nghĩa là dữ liệu đủ: tắt agent thì
        # `system_metrics` vẫn về (backend còn sống) còn `service_status` lỗi.
        if errors and r.get("ok"):
            degraded.append({"machine": m.name, "reason": "; ".join(errors)})

    return {"generated_at": time.time(),
            "coverage": coverage(results, degraded), "machines": rows}


async def machine_detail(key: str) -> Dict[str, Any]:
    """Thẻ chi tiết MỘT máy. Gộp các nguồn rẻ, ~0,5s, 0 lượt LLM."""
    r = resolve(key)
    if not r["ok"]:
        return r
    m: Machine = r["machine"]
    await registry.probe(m)
    met = await client.system_metrics(m.ip)
    svc = await client.service_status(m.node_id, m.ip)
    return {
        "ok": True,
        **m.to_dict(),
        "metrics": metrics_view((met.data or {}).get("data") if met.ok else None),
        "metrics_error": None if met.ok else met.error,
        "service": svc.data if svc.ok else None,
        "service_error": None if svc.ok else svc.error,
    }


async def fleet_production(days: int = 7, causes: bool = True) -> Dict[str, Any]:
    """
    Sản lượng + vân tay kiểu lỗi cả đội hình.

    **Không xếp hạng theo tỉ lệ pass.** Các máy chạy recipe khác nhau — trên đội
    hình này đúng MỘT recipe được chia sẻ giữa hai máy — nên "máy A pass 98%, máy
    B pass 69%" là so hành tây với quế, không phải so máy với máy. Thứ so được là
    phân bố nguyên nhân lỗi, vì các nguyên nhân đó thuộc pipeline OCR chứ không
    thuộc mặt hàng.
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
        if prod:
            totals["products"] += prod.get("total_products") or 0
            totals["pass"] += prod.get("pass") or 0
            totals["fail"] += prod.get("fail") or 0
        rows.append({
            "node_id": m.node_id, "machine": m.name, "line": m.line,
            "state": m.state(), "production": prod,
            "failure_modes": d.get("failure_modes"),
            "recipes": d.get("recipes"),
            "error": r.get("error") or d.get("production_error"),
        })

    totals["pass_rate"] = (round(totals["pass"] * 100.0 / totals["products"], 2)
                           if totals["products"] else None)

    matrix: Dict[str, Any] = {}
    if causes:
        keys: List[str] = []
        for row in rows:
            for c in ((row.get("failure_modes") or {}).get("by_cause") or []):
                if c["cause"] not in keys:
                    keys.append(c["cause"])
        if keys:
            # Nhãn tiếng người đi kèm mã bước kiểm tra. Chỉ đưa mã trần
            # (`char_verification`) thì mô hình đọc nó như một thực thể lọc được
            # và truyền xuống agent máy như thể đó là tên recipe — đã xảy ra
            # thật: câu hỏi ủy quyền thành "recipe char_verification", edge trả
            # "không có dữ liệu". Cùng cách chữa như `only_in` → `absent_from`
            # ở tầng edge: đổi cách gọi tên, không phải cấm mô hình.
            labels: Dict[str, str] = {}
            for row in rows:
                for c in ((row.get("failure_modes") or {}).get("by_cause") or []):
                    if c.get("label"):
                        labels[c["cause"]] = c["label"]

            by_machine = {}
            for row in rows:
                fm = row.get("failure_modes") or {}
                share = {c["cause"]: c.get("share_of_causes_pct")
                         for c in (fm.get("by_cause") or [])}
                by_machine[row["machine"]] = {
                    "by_cause": {k: share.get(k) for k in keys},
                    "sample_products": fm.get("sample_products"),
                    # Mẫu chưa phủ hết kỳ ⇒ đây là tỉ lệ của MẪU, không phải của
                    # cả kỳ. Nói ra để không ai đọc thành số tuyệt đối.
                    "sample_covers_all": fm.get("sample_covers_all"),
                }
            matrix = {"causes": keys, "cause_labels": labels,
                      "by_machine": by_machine}

    return {
        "generated_at": time.time(), "period_days": days,
        "coverage": coverage(results),
        "fleet_total": totals,
        "note": ("Không xếp hạng theo pass rate: các máy chạy recipe khác nhau nên "
                 "tỉ lệ pass không so trực tiếp được. Dùng vân tay kiểu lỗi để so."),
        "failure_fingerprint": matrix,
        "machines": rows,
    }
