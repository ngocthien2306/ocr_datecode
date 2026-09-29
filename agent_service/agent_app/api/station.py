"""
API cho Line Station — màn hình đặt cạnh MỘT dây chuyền.

Khác Fleet Console ở một điểm quyết định mọi thứ khác: người dùng đang đứng cạnh
máy đang chạy, đeo găng, nhìn từ khoảng 2 m. Nên:

  * MỘT lời gọi trả cả màn hình. Sáu lời gọi từ một tablet qua Wi-Fi xưởng là sáu
    cơ hội để màn hình hiện dở dang; ở đây mọi số liệu đọc từ MongoDB của chính
    máy này nên gộp lại gần như không tốn thêm gì.
  * Không có gì so sánh với line khác. Line này chạy quế, line kia chạy muối —
    đặt cạnh nhau là mời người vận hành so sai.
  * Không endpoint nào gây tác dụng phụ. Cả file này chỉ đọc.
  * Không bao giờ trả `0` thay cho `None`. "0 sản phẩm, đạt 0%" đọc như máy đang
    hỏng, còn "0°C" đọc như máy rất mát — hai câu nói sai về hai chuyện khác
    nhau. Chưa đo được thì là `None`, và lớp vẽ hiện dấu gạch.

Sản lượng "ca này" tính theo CỬA SỔ CA, không theo ngày lịch. Ca C chạy
22:00–06:00 vắt qua nửa đêm; lấy theo ngày thì 00:05 màn hình nhảy về 0 trong khi
ca vẫn đang chạy dở.
"""

from __future__ import annotations

import logging
import socket
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query

from agent_app.api.deps import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/station", tags=["Line Station"])

# Ca làm việc. Giữ ở đây vì Line Station là nơi duy nhất cần BIÊN của ca (để nói
# "ca chưa bắt đầu", "còn 3 giờ nữa"), chứ không chỉ cần gán một bản ghi vào ca.
SHIFTS = [
    {"name": "A", "from": 6, "to": 14},
    {"name": "B", "from": 14, "to": 22},
    {"name": "C", "from": 22, "to": 6},     # vắt qua nửa đêm
]


def _station_identity() -> Dict[str, Any]:
    """
    Máy này là máy nào.

    Phải do env nói, không suy từ hostname: bốn trong năm máy đều tên
    `suntech-desktop`, nên hostname không định danh được gì. Thiếu env thì nói
    thẳng là chưa cấu hình, đừng đoán rồi hiện sai tên line lên màn hình xưởng.
    """
    from agent_app.core.config import settings

    # Đọc qua `settings`, không qua os.getenv: pydantic nạp .env vào settings
    # chứ KHÔNG đẩy vào os.environ, nên os.getenv ở đây luôn rỗng.
    name = (settings.STATION_NAME or "").strip()
    return {
        "name": name or socket.gethostname(),
        "configured": bool(name),
        "line": (settings.STATION_LINE or "").strip() or None,
        "model": (settings.STATION_MODEL or "").strip() or None,
    }


def _shift_at(now: datetime) -> Dict[str, Any]:
    """Ca đang chạy tại `now`, kèm mốc bắt đầu/kết thúc dạng datetime thật."""
    h = now.hour
    for s in SHIFTS:
        inside = (s["from"] <= h < s["to"]) if s["from"] < s["to"] \
            else (h >= s["from"] or h < s["to"])
        if inside:
            start = now.replace(hour=s["from"], minute=0, second=0, microsecond=0)
            # Ca C bắt đầu HÔM QUA nếu bây giờ đã qua nửa đêm.
            if s["from"] > s["to"] and h < s["to"]:
                start -= timedelta(days=1)
            span = (s["to"] - s["from"]) % 24 or 24
            return {"name": s["name"], "from_hour": s["from"], "to_hour": s["to"],
                    "start": start, "end": start + timedelta(hours=span),
                    "hours": span}
    return {"name": "?", "from_hour": None, "to_hour": None,
            "start": now, "end": now, "hours": 0}


def _pct(part: int, whole: int) -> Optional[float]:
    return round(part * 100 / whole, 1) if whole else None


def _to_utc(dt_local: datetime) -> datetime:
    """Giờ địa phương → naive-UTC, đúng cách MongoDB lưu.

    Không có bước này thì mọi cửa sổ ca lệch 7 tiếng: biên ca tính bằng giờ VN
    (06:00) đem so với `created_at` lưu UTC. Đo được trên Auto2: màn hình báo
    "ca A chưa bắt đầu" sau 3,4 giờ ca, trong khi ca trước hiện 2.128 sản phẩm —
    vì cửa sổ ca A quy ra UTC nằm ở tương lai. Cảnh báo này đã ghi sẵn trong
    analytics_tools và tôi vẫn rơi vào đúng nó.
    """
    from agent_app.core.config import settings

    tz = ZoneInfo(settings.TIMEZONE)
    if dt_local.tzinfo is None:
        dt_local = dt_local.replace(tzinfo=tz)
    return dt_local.astimezone(timezone.utc).replace(tzinfo=None)


@router.get("/overview", summary="Toàn bộ màn hình Line Station trong một lời gọi")
async def overview(
    hours_back_for_prev: int = Query(24, ge=8, le=72),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    from agent_app.core.config import settings
    from agent_app.db.mongodb import get_sync_database

    db = get_sync_database()
    col = db["inference_results"]
    now = datetime.now()
    sh = _shift_at(now)

    def window(start: datetime, end: datetime) -> Dict[str, Any]:
        """Đếm đạt/không đạt trong một cửa sổ thời gian."""
        rows = list(col.aggregate([
            # Biên là giờ địa phương; DB lưu naive-UTC ⇒ quy đổi tại đây.
            {"$match": {"created_at": {"$gte": _to_utc(start), "$lt": _to_utc(end)}}},
            {"$group": {"_id": "$product_pass_fail", "n": {"$sum": 1}}},
        ]))
        got = {r["_id"]: r["n"] for r in rows}
        ok, bad = got.get("PASS", 0), got.get("FAIL", 0)
        tot = ok + bad
        return {"total": tot or None, "pass": ok, "fail": bad,
                "pass_rate": _pct(ok, tot)}

    cur = window(sh["start"], now)
    prev = window(sh["start"] - timedelta(hours=sh["hours"]), sh["start"])

    # Delta ghi bằng ĐIỂM phần trăm, không phải "%": 69,4% so 77,6% lệch 8,2
    # điểm, không phải "lệch 8,2%".
    delta = None
    if cur["pass_rate"] is not None and prev["pass_rate"] is not None:
        delta = round(cur["pass_rate"] - prev["pass_rate"], 1)

    # Theo giờ TRONG CA. Giờ chưa tới trả None để lớp vẽ để trống — vẽ cột 0 cho
    # giờ chưa xảy ra là nói rằng giờ đó không sản xuất được gì.
    hourly: List[Dict[str, Any]] = []
    for i in range(sh["hours"]):
        h0 = sh["start"] + timedelta(hours=i)
        label = f"{h0.hour:02d}h"
        if h0 > now:
            hourly.append({"hour": label, "total": None, "pass_rate": None})
            continue
        w = window(h0, min(h0 + timedelta(hours=1), now))
        hourly.append({"hour": label, "total": w["total"],
                       "pass_rate": w["pass_rate"]})

    # Recipe đang chạy: lấy từ bản ghi kiểm gần nhất, không đọc cấu hình — cấu
    # hình nói cái ĐƯỢC chọn, bản ghi nói cái đang thật sự chạy qua camera.
    last = col.find_one(sort=[("created_at", -1)])
    recipe = (last or {}).get("recipe_name")
    last_seen = (last or {}).get("created_at")

    return {
        "success": True,
        "generated_at": now.isoformat(timespec="seconds"),
        "timezone": settings.TIMEZONE,
        "machine": _station_identity(),
        "recipe_name": recipe,
        "last_inspection": last_seen.isoformat(timespec="seconds") if last_seen else None,
        "shift": {
            "name": sh["name"],
            "from": f"{sh['from_hour']:02d}:00" if sh["from_hour"] is not None else None,
            "to": f"{sh['to_hour']:02d}:00" if sh["to_hour"] is not None else None,
            "started_at": sh["start"].isoformat(timespec="seconds"),
            "hours_elapsed": round((now - sh["start"]).total_seconds() / 3600, 1),
            "hours_total": sh["hours"],
            # Ca vừa sang mà chưa có bản ghi nào là "ca chưa bắt đầu", không phải
            # "sản lượng 0" — hai câu khác nhau hoàn toàn với người đứng máy.
            "not_started": cur["total"] is None,
        },
        "output": cur,
        "previous_shift": {**prev, "delta_points": delta},
        "hourly": hourly,
    }


# Bố trí mặt bằng. Line Station chỉ cần biết "mình ở đâu so với các máy khác",
# nên giữ một bản mặc định ngay đây và cho phép ghi đè bằng file trên máy —
# không đi hỏi fleet service, vì màn hình này phải chạy khi fleet tắt.
_DEFAULT_FLOOR = [
    {"name": "Auto2", "line": "Line 1", "floor": {"x": 0, "y": 0}},
    {"name": "M1", "line": "Line 2", "floor": {"x": 1, "y": 0}},
    {"name": "M2", "line": "Line 3", "floor": {"x": 2, "y": 0}},
    {"name": "LineTine", "line": "Tine Line", "floor": {"x": 0.5, "y": 1.35}},
    {"name": "PC-Auto-1", "line": "Auto Line", "floor": {"x": 1.5, "y": 1.35}},
]


@router.get("/floor", summary="Bố trí mặt bằng, và máy nào là máy này")
async def floor(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Toạ độ các máy trên sàn. Đọc `config/station_floor.json` nếu có, không thì
    dùng bản mặc định.

    Đổi bố trí xưởng thì sửa một file trên máy, không phải sửa code — nhưng cũng
    KHÔNG đi hỏi fleet service: sơ đồ định vị phải hiện được cả khi máy tổng tắt.
    """
    import json

    from agent_app.core.config import settings

    rows = _DEFAULT_FLOOR
    override = settings.project_root / "agent_service" / "config" / "station_floor.json"
    if override.is_file():
        try:
            rows = json.loads(override.read_text())
        except Exception as e:                              # pragma: no cover
            logger.warning("station_floor.json không đọc được: %s", e)

    me = _station_identity()
    return {"success": True, "self": me["name"], "machines": rows,
            "source": "file" if override.is_file() else "default"}


@router.get("/failures", summary="Ảnh sản phẩm lỗi gần nhất của chính máy này")
async def failures(
    limit: int = Query(4, ge=1, le=12),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Dùng lại đúng bộ chọn mẫu của `/fleet/failure-images` — không viết lại logic
    chọn ảnh. Ở đây chỉ ít ảnh hơn và không lọc theo nguyên nhân: màn hình xưởng
    cần "vừa rồi lỗi gì", không cần một cuộc điều tra.
    """
    from agent_app.api.fleet import failure_images

    return await failure_images(days=1, limit=limit, cause=None,
                                sample_limit=200, current_user=current_user)


@router.get("/image/{img_id}", summary="Ảnh sản phẩm cho thẻ <img> của Line Station")
async def station_image(
    img_id: str,
    w: int = Query(480, ge=64, le=1600),
    token: str = Query("", description="JWT — thẻ <img> không gắn được header"),
):
    """Cùng ảnh với `/fleet/frame/{id}`, chỉ khác chỗ đọc token."""
    from agent_app.api.deps import get_current_user_from_query
    from agent_app.api.fleet import failure_image

    user = await get_current_user_from_query(token)
    return await failure_image(img_id=img_id, w=w, current_user=user)


@router.get("/health-metrics", summary="Nhiệt độ, RAM, đĩa của chính máy này")
async def health_metrics(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Đọc qua `system_tools`, tức qua backend cùng máy — không đọc thẳng /sys.

    Máy x86 (PC-Auto-1) không có cảm biến kiểu Jetson nên nhiệt về `None`; lớp
    vẽ hiện dấu gạch. Trả 0 ở đây là nói máy đang rất mát.
    """
    from agent_app.tools.system_tools import get_system_metrics

    try:
        return get_system_metrics()
    except Exception as e:                                  # pragma: no cover
        logger.warning("Không đọc được phần cứng: %s", e)
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


@router.get("/crew", summary="Người trong ca hiện tại")
async def crew(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Ai đang trong ca, suy từ khung giờ ca của từng người.

    Không có trường nào trong DB nói "đang trong ca" — phải tách giờ từ chuỗi
    `shift` dạng "Shift B (14:00–22:00)". Dấu ở đây là gạch ngang dài, đúng như
    dữ liệu thật đang lưu.
    """
    import re

    from agent_app.db.mongodb import get_sync_database

    now = datetime.now()
    cur = now.hour * 60 + now.minute
    out = []
    for u in get_sync_database()["users"].find({}, {"hashed_password": 0}):
        m = re.search(r"(\d{1,2}):(\d{2})\s*[–-]\s*(\d{1,2}):(\d{2})",
                      str(u.get("shift") or ""))
        if not m:
            continue
        a = int(m[1]) * 60 + int(m[2])
        b = int(m[3]) * 60 + int(m[4])
        inside = (a <= cur < b) if a < b else (cur >= a or cur < b)
        if not inside:
            continue
        login = str(u.get("last_login") or "")
        out.append({
            "full_name": u.get("full_name") or u.get("username"),
            "username": u.get("username"),
            "job_title": u.get("job_title"),
            "shift": u.get("shift"),
            # "vào ca 14:02" chỉ đúng nếu lần đăng nhập là HÔM NAY; hôm khác thì
            # để trống chứ không gán bừa vào ca hiện tại.
            "since": login[11:16] if login[:10] == now.strftime("%Y-%m-%d") else None,
        })
    out.sort(key=lambda x: (x.get("job_title") or "", x["full_name"]))
    return {"success": True, "count": len(out), "crew": out}


@router.get("/handover", summary="Bản bàn giao ca (không qua LLM)")
async def handover(
    shift: Optional[str] = Query(None, description="A/B/C, để trống là ca hiện tại"),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Một lời gọi `get_shift_handover`, đúng tool đã có ở edge.

    Đi thẳng vào tool, KHÔNG qua mô hình: hôm hết credit OpenAI thì cả năm trợ
    lý im tiếng cùng lúc, mà bàn giao ca là việc bắt buộc phải làm được lúc
    22:00 dù có trợ lý hay không.
    """
    from agent_app.tools.analytics_tools import get_shift_handover

    name = shift or _shift_at(datetime.now())["name"]
    try:
        return get_shift_handover(shift=name)
    except Exception as e:                                  # pragma: no cover
        logger.error("Bàn giao ca lỗi: %s", e)
        return {"success": False, "error": f"{type(e).__name__}: {e}"}
