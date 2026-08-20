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

import asyncio
import re
import time
import unicodedata
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


async def fleet_production(days: int = 7, causes: bool = True,
                           granularity: str = "day") -> Dict[str, Any]:
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
        r = await client.rollup(m.node_id, m.ip, days=days, causes=causes,
                                granularity=granularity)
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
            "by_shift": d.get("by_shift"),
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
    """Bỏ dấu, gộp khoảng trắng — để 'Tuần này' và 'tuan nay' về cùng một chuỗi.

    `đ` phải tự tay đổi thành `d`: NFD không tách nó ra thành d + dấu, nên tên
    "Đặng Văn Sáu" gõ không dấu vẫn trượt nếu chỉ dựa vào NFD.
    """
    t = unicodedata.normalize("NFD", str(t or "").strip().lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"[\s_-]+", " ", t.replace("đ", "d"))


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


async def fleet_staff() -> Dict[str, Any]:
    """
    Nhân sự toàn nhà máy, gộp từ mọi máy.

    Mỗi bản ghi mang `machine` và khoá là **(máy, username)** — username KHÔNG
    duy nhất giữa các máy (kiểm chứng: `admin`/`operator`/`supervisor` tồn tại
    trên cả 5 máy, là các tài khoản khác nhau trùng tên). Gộp theo username trần
    là trộn 5 người thành một. Danh tính xuyên máy duy nhất là `employee_code`.
    """
    ms = [m for m in registry.all() if m.online]

    async def one(m: Machine) -> Dict[str, Any]:
        r = await client.staff(m.node_id, m.ip)
        if not r.ok:
            raise RuntimeError(r.error or "staff lỗi")
        return r.data

    results = await fan_out(ms, one, timeout=settings.EDGE_TIMEOUT + 2)

    users, by_machine = [], {}
    for r in results:
        if not r.get("ok"):
            continue
        rows = (r["data"] or {}).get("users") or []
        by_machine[r["machine"]] = len(rows)
        for u in rows:
            users.append({**u, "machine": r["machine"],
                          "key": f"{r['machine']}/{u.get('username')}"})

    return {
        "generated_at": time.time(),
        "coverage": coverage(results),
        "count": len(users),
        "by_machine": by_machine,
        "users": users,
    }


# ── Tra người ────────────────────────────────────────────────────────────────

async def resolve_person(text: str) -> Dict[str, Any]:
    """
    Đổi thứ người ta gọi một con người thành thứ bảng audit tra được.

    Bảng audit chỉ biết `username`. Còn trên màn hình — và trong đầu người hỏi —
    một người là HỌ TÊN. Đo được: mô hình truyền thẳng "Lâm Thị Tuyết Mai" vào ô
    username, cả 5 máy trả "không có tài khoản này", và câu trả lời thành
    "người này chưa thao tác gì" trong khi thực tế có bản ghi đăng nhập. Sai đó
    không sửa được bằng cách dặn mô hình kỹ hơn: chỗ hổng nằm ở việc không ai
    dịch tên người sang tên tài khoản.

    Trả về `status`:
      found      — ra đúng một username (có thể tồn tại trên nhiều máy)
      ambiguous  — nhiều username khác nhau cùng khớp, phải hỏi lại
      not_found  — không khớp ai; KHÁC HẲN với "có người này mà không làm gì"
    """
    staff = await fleet_staff()
    users = staff.get("users") or []
    q = _norm(text)
    if not q:
        return {"status": "not_found", "query": text, "candidates": []}

    def pick(hits: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        names = {u.get("username") for u in hits}
        if not hits:
            return None
        brief = [{"username": u.get("username"), "machine": u.get("machine"),
                  "full_name": u.get("full_name"),
                  "employee_code": u.get("employee_code"),
                  "job_title": u.get("job_title")} for u in hits]
        if len(names) == 1:
            return {"status": "found", "query": text,
                    "username": hits[0].get("username"), "matches": brief}
        return {"status": "ambiguous", "query": text, "candidates": brief}

    # Thứ tự có chủ đích: khớp chính xác trước, đoán mò sau cùng.
    for keys in (("username",), ("employee_code",), ("full_name",)):
        hit = [u for u in users if any(_norm(u.get(k)) == q for k in keys)]
        if hit:
            return pick(hit)
    hit = [u for u in users if q in _norm(u.get("full_name"))]
    if hit:
        return pick(hit)
    return {"status": "not_found", "query": text, "candidates": []}


# ── Ảnh sản phẩm: cache TTL nằm Ở ĐÂY ────────────────────────────────────────
#
# Nhịp "60 giây một ảnh" phải do MÁY CHỦ giữ, không phải do trình duyệt. Nếu để
# setInterval phía FE quyết định thì ba người mở ba tab là Jetson nhận ba lần
# tải — mà chính con Jetson đó đang chạy inference cho dây chuyền. Có TTL ở đây
# thì bao nhiêu người xem cũng chỉ còn 1 lần hỏi mỗi 60 giây cho mỗi máy.

FRAME_TTL = 60.0
_frame_cache: Dict[str, tuple] = {}


async def _frame_cached(key: str, make, ttl: float = FRAME_TTL) -> Dict[str, Any]:
    hit = _frame_cache.get(key)
    now = time.time()
    if hit and now - hit[0] < ttl:
        return {**hit[1], "cached": True, "cache_age": round(now - hit[0], 1)}
    val = await make()
    _frame_cache[key] = (now, val)
    return {**val, "cached": False, "cache_age": 0.0}


async def machine_frame(machine: str,
                        template: Optional[str] = None) -> Dict[str, Any]:
    """
    Ảnh lỗi gần nhất của một máy + template chuẩn của chính nó.

    Trả về cả khi máy không với tới được: lúc đó là bản đã cache lần cuối kèm
    tuổi của nó. Máy mất mạng mà màn hình quay vòng chờ mãi thì người xem không
    biết là mất mạng hay là chưa có sản phẩm nào.
    """
    ms = [m for m in registry.all() if m.name == machine or m.node_id == machine]
    if not ms:
        return {"success": False, "error": f"không có máy tên {machine}"}
    m = ms[0]

    async def fetch() -> Dict[str, Any]:
        r = await client.frame_pair(m.node_id, m.ip, template=template)
        if not r.ok:
            stale = _frame_cache.get(f"pair:{m.node_id}:{template or ''}")
            if stale:
                return {**stale[1], "stale": True, "error": r.error}
            return {"success": False, "machine": m.name, "error": r.error}
        d = r.data or {}
        return {"success": True, "machine": m.name, "node_id": m.node_id,
                "state": m.state(), "found": d.get("found"),
                "frame": d.get("frame"), "template": d.get("template"),
                "template_matched": d.get("template_matched"),
                "templates": d.get("templates"),
                "selected_template": d.get("selected_template"),
                "cameras": d.get("cameras"),
                "note": d.get("note")}

    # Khoá cache PHẢI kèm template: thiếu nó thì bấm sang Frame 3 lại nhận về
    # bản đã cache của Frame 4, và người xem tưởng hai vị trí lỗi giống hệt nhau.
    return await _frame_cached(f"pair:{m.node_id}:{template or ''}", fetch)


async def fleet_frames() -> Dict[str, Any]:
    """
    Ảnh gần nhất của MỌI máy — tường ảnh.

    Đi qua cùng một cache với khung ảnh trong drawer: mở tường ảnh rồi mở tiếp
    drawer của một máy thì máy đó không bị hỏi lại.
    """
    ms = [m for m in registry.all() if m.online]
    out = await asyncio.gather(*(machine_frame(m.name) for m in ms),
                               return_exceptions=True)
    rows = [o for o in out if isinstance(o, dict)]
    return {"generated_at": time.time(),
            "count": len(rows),
            "machines": rows,
            "ttl_seconds": FRAME_TTL}


async def frames_around(machine: str, ts: str) -> Dict[str, Any]:
    """Ảnh ngay trước và ngay sau một mốc — để ghép với nhật ký thao tác."""
    ms = [m for m in registry.all() if m.name == machine or m.node_id == machine]
    if not ms:
        return {"success": False, "error": f"không có máy tên {machine}"}
    m = ms[0]

    async def fetch() -> Dict[str, Any]:
        r = await client.frames_around(m.node_id, m.ip, ts)
        if not r.ok:
            return {"success": False, "machine": m.name, "error": r.error}
        d = r.data or {}
        return {"success": True, "machine": m.name, "at": ts,
                "before": d.get("before"), "after": d.get("after")}

    # Ảnh quanh một mốc trong QUÁ KHỨ không đổi nữa, nên cache lâu hơn nhiều.
    return await _frame_cached(f"around:{m.node_id}:{ts}", fetch, ttl=900.0)


async def fleet_failure_images(days: int = 7, per_machine: int = 6,
                               cause: Optional[str] = None) -> Dict[str, Any]:
    """
    Ảnh sản phẩm lỗi từ mọi máy, mỗi máy tối đa `per_machine` ảnh.

    Giới hạn THEO MÁY chứ không lấy top-N toàn cục: top-N toàn cục sẽ bị máy
    nhiều fail nhất chiếm sạch, và lưới "toàn nhà máy" hoá ra chỉ có ảnh của một
    line. Ảnh của mỗi máy vốn đã được edge rải đều theo nguyên nhân.
    """
    ms = [m for m in registry.all() if m.online]

    async def one(m: Machine) -> Dict[str, Any]:
        r = await client.failure_images(m.node_id, m.ip, days=days,
                                        limit=per_machine, cause=cause)
        if not r.ok:
            raise RuntimeError(r.error or "failure-images lỗi")
        return r.data

    results = await fan_out(ms, one, timeout=settings.EDGE_ROLLUP_TIMEOUT + 3)

    images = []
    for r in results:
        if not r.get("ok"):
            continue
        d = r["data"] or {}
        for img in d.get("images") or []:
            images.append({**img, "machine": r["machine"],
                           # URL đi QUA fleet — trình duyệt không gắn được token
                           # vào <img>, nên fleet proxy hộ.
                           "url": f"/api/fleet/failure-image/{r['machine']}/{img['id']}"})

    # Trộn xen kẽ theo máy thay vì nối đuôi: nối đuôi thì 6 ảnh đầu toàn của một
    # máy, người xem tưởng lưới chỉ có một line.
    by_m: Dict[str, List[Dict[str, Any]]] = {}
    for img in images:
        by_m.setdefault(img["machine"], []).append(img)
    mixed, idx = [], 0
    while any(by_m.values()):
        for k in sorted(by_m):
            if by_m[k]:
                mixed.append(by_m[k].pop(0))
        idx += 1

    return {
        "generated_at": time.time(),
        "coverage": coverage(results),
        "count": len(mixed),
        "images": mixed,
    }


async def fleet_audit(days: int = 7, username: Optional[str] = None,
                      action_type: Optional[str] = None,
                      include_simulated: bool = False,
                      per_machine: int = 40) -> Dict[str, Any]:
    """
    Nhật ký thao tác gộp từ mọi máy, sắp theo thời gian.

    Phân biệt ba tình huống mà gộp lại là nói sai:
      - máy trả lời, có bản ghi          → nằm trong `entries`
      - máy trả lời, KHÔNG có người này  → `machines_without_user`
      - máy không trả lời                → `coverage.machines_missing`

    Tình huống giữa rất dễ bị đọc thành lỗi: `truongca_m2` chỉ tồn tại trên M2,
    nên bốn máy kia "không có" là chuyện bình thường, không phải bốn máy hỏng.
    """
    ms = [m for m in registry.all() if m.online]

    async def one(m: Machine) -> Dict[str, Any]:
        r = await client.audit(m.node_id, m.ip, days=days, username=username,
                               action_type=action_type,
                               include_simulated=include_simulated,
                               limit=per_machine)
        if not r.ok:
            raise RuntimeError(r.error or "audit lỗi")
        return r.data

    results = await fan_out(ms, one, timeout=settings.EDGE_TIMEOUT + 2)

    entries, by_action, no_user, totals = [], {}, [], 0
    for r in results:
        if not r.get("ok"):
            continue
        d = r["data"] or {}
        if username and not d.get("matched"):
            no_user.append(r["machine"])
            continue
        totals += d.get("total_in_period") or 0
        for k, v in (d.get("by_action") or {}).items():
            by_action[k] = by_action.get(k, 0) + v
        for e in d.get("entries") or []:
            entries.append({**e, "machine": r["machine"],
                            "key": f"{r['machine']}/{e.get('username')}"})

    entries.sort(key=lambda e: e.get("time") or "", reverse=True)

    return {
        "generated_at": time.time(),
        "coverage": coverage(results),
        "period_days": days,
        "total_in_period": totals,
        "by_action": dict(sorted(by_action.items(), key=lambda x: -x[1])),
        "machines_without_user": no_user,
        "count": len(entries),
        "entries": entries,
        "note": ("username chỉ duy nhất TRONG một máy; mỗi bản ghi có `key` dạng "
                 "máy/username. Bản ghi demo (simulated) mặc định đã bị loại."),
    }


async def fleet_log_errors(date: Optional[str] = None,
                           top: int = 8) -> Dict[str, Any]:
    """Tóm tắt lỗi hệ thống của mọi máy. Chỉ nhận bản tóm tắt, không nhận file log."""
    ms = [m for m in registry.all() if m.online]

    async def one(m: Machine) -> Dict[str, Any]:
        r = await client.log_errors(m.node_id, m.ip, date=date, top=top)
        if not r.ok:
            raise RuntimeError(r.error or "log-errors lỗi")
        return r.data

    results = await fan_out(ms, one, timeout=settings.EDGE_ROLLUP_TIMEOUT + 3)

    rows, total = [], 0
    for r in results:
        if not r.get("ok"):
            rows.append({"machine": r["machine"], "error": r.get("error")})
            continue
        d = r["data"] or {}
        if not d.get("success"):
            rows.append({"machine": r["machine"], "error": d.get("error")})
            continue
        total += d.get("total_problem_lines") or 0
        rows.append({
            "machine": r["machine"],
            "total_problem_lines": d.get("total_problem_lines"),
            "distinct_problems": d.get("distinct_problems"),
            "by_level": d.get("by_level"),
            "problems": d.get("problems"),
        })

    return {
        "generated_at": time.time(),
        "coverage": coverage(results),
        "date": date or "hôm nay",
        "fleet_total_problem_lines": total,
        "machines": rows,
    }
