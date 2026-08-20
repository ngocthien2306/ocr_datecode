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


# --- báo cáo so sánh ---------------------------------------------------------

PERIOD_CHOICES = [
    {"key": "today",      "label": "Hôm nay",          "days": 1,  "granularity": "day"},
    {"key": "yesterday",  "label": "Hôm qua",          "days": 2,  "granularity": "day"},
    {"key": "this_week",  "label": "7 ngày qua",       "days": 7,  "granularity": "day"},
    {"key": "last_30d",   "label": "30 ngày qua",      "days": 30, "granularity": "week"},
    {"key": "last_90d",   "label": "90 ngày qua",      "days": 90, "granularity": "week"},
]
PERIOD_BY_KEY = {p["key"]: p for p in PERIOD_CHOICES}


def _norm(t: str) -> str:
    """Bỏ dấu, gộp khoảng trắng — để 'Tuần này' và 'tuan nay' về cùng một chuỗi."""
    import re
    import unicodedata
    t = unicodedata.normalize("NFD", str(t or "").strip().lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"[\s_-]+", " ", t)


def resolve_period(value: str) -> Optional[Dict[str, Any]]:
    """
    Nhận kỳ theo khoá, theo nhãn, hoặc theo cách người ta nói tự nhiên.

    Bản đầu chỉ tra đúng khoá nội bộ (`this_week`), nên mô hình gửi "7 ngày" —
    tức là nhắc lại chính cái nhãn mà tool vừa in ra cho người dùng chọn — thì bị
    từ chối. Bắt người gọi học khoá nội bộ là đẩy công việc sai chỗ: tool in ra
    nhãn nào thì phải nhận lại được nhãn đó.
    """
    if not value:
        return None
    v = _norm(value)
    for p in PERIOD_CHOICES:
        if v in (_norm(p["key"]), _norm(p["label"])):
            return p
    # "7 ngày", "30 ngay qua", "90d" — bắt lấy con số rồi khớp với kỳ gần nhất.
    import re
    m = re.search(r"(\d+)\s*(?:ngay|day|d\b)", v)
    if m:
        want = int(m.group(1))
        return min(PERIOD_CHOICES, key=lambda p: abs(p["days"] - want))
    if "hom nay" in v or "today" in v:
        return PERIOD_BY_KEY["today"]
    if "hom qua" in v or "yesterday" in v:
        return PERIOD_BY_KEY["yesterday"]
    if "tuan" in v or "week" in v:
        return PERIOD_BY_KEY["this_week"]
    if "thang" in v or "month" in v:
        return PERIOD_BY_KEY["last_30d"]
    return None


async def report_data(machine_names: List[str], days: int,
                      period_label: str) -> Dict[str, Any]:
    """
    Dữ liệu cho báo cáo so sánh, CHỈ gồm các máy được chọn.

    `coverage` tính lại trên đúng tập máy được chọn, không phải trên cả đội hình:
    người dùng chọn 2 máy thì "thiếu 3/5" là câu vô nghĩa, còn "đủ 2/2" mới đúng.
    """
    full = await fleet_production(days=days, causes=True)
    wanted = {n.lower() for n in machine_names}
    rows = [r for r in full["machines"] if r["machine"].lower() in wanted]

    totals = {"products": 0, "pass": 0, "fail": 0}
    for r in rows:
        p = r.get("production") or {}
        totals["products"] += p.get("total_products") or 0
        totals["pass"] += p.get("pass") or 0
        totals["fail"] += p.get("fail") or 0
    totals["pass_rate"] = (round(totals["pass"] * 100.0 / totals["products"], 2)
                           if totals["products"] else None)

    fp = full.get("failure_fingerprint") or {}
    if fp:
        kept = {k: v for k, v in (fp.get("by_machine") or {}).items()
                if k.lower() in wanted}
        # Bỏ luôn cột nguyên nhân mà không máy nào được chọn có số — giữ lại thì
        # báo cáo có một cột toàn dấu gạch ngang, đọc như "đo được nhưng bằng 0".
        keys = [c for c in fp.get("causes", [])
                if any((v["by_cause"].get(c) or 0) > 0 for v in kept.values())]
        fp = {"causes": keys, "cause_labels": fp.get("cause_labels", {}),
              "by_machine": {k: {**v, "by_cause": {c: v["by_cause"].get(c) for c in keys}}
                             for k, v in kept.items()}} if kept and keys else {}

    missing = [{"machine": r["machine"], "reason": r["error"]}
               for r in rows if r.get("error")]
    return {
        "period_days": days, "period_label": period_label,
        "fleet_total": totals,
        "failure_fingerprint": fp,
        "machines": rows,
        "coverage": {"machines_total": len(rows),
                     "machines_ok": len(rows) - len(missing),
                     "machines_missing": missing,
                     "machines_degraded": [],
                     "complete": not missing},
    }
