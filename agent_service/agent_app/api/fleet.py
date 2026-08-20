"""
Endpoint cho tầng fleet đọc: số liệu sản xuất gọn, KHÔNG đi qua LLM.

Vì sao cần, khi đã có `/api/agent/chat`: fleet phải poll 5 máy mỗi phút. Đi qua
agent thì mỗi vòng là 5 lượt LLM — đo thật trên đội hình này là 4–20s mỗi lượt,
và với `gpt-4o-mini` thì một cái dashboard poll 60s tốn cỡ 200 USD/tháng để hiện
lại đúng con số mà `get_pass_fail_stats` tính ra trong 0,2s miễn phí.

Endpoint ở đây gọi THẲNG các tool đã có, không dựng lại phép tính. Đó là chủ ý:
logic trong `analytics_tools` không phải "đọc dữ liệu" mà là tri thức đúng/sai đã
tích luỹ — `created_at` thay vì `timestamp` (198s → 0,2s), đếm theo SẢN PHẨM chứ
không theo FRAME, mốc ngày là giờ địa phương còn DB lưu naive-UTC. Viết lại ở
tầng fleet là rước lại đủ các lỗi đó, mà không test nào bắt được vì chúng đều
"hiện ra vẫn đẹp".

Khuôn theo `/api/agent/service/status` đã có: gọi thẳng tool, không tốn token.
"""

from __future__ import annotations

import base64
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from agent_app.api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fleet", tags=["Fleet"])

# Bộ nhớ đệm trong tiến trình. Fleet poll mỗi 60s và câu hỏi luôn cùng tham số,
# nên không cache thì mỗi vòng lại mổ lại đúng ngần ấy document fail. TTL ngắn
# hơn nhịp poll để số không bao giờ cũ quá một vòng.
_CACHE: Dict[str, tuple] = {}
_CACHE_TTL = 45.0


def _cached(key: str, build):
    now = time.monotonic()
    hit = _CACHE.get(key)
    if hit and hit[0] > now:
        return hit[1]
    val = build()
    # Chỉ cache kết quả thành công: lỗi thường là tạm thời (Mongo nghẽn), giữ
    # lại 45s nghĩa là người dùng thử lại ngay vẫn nhận đúng cái lỗi cũ.
    if isinstance(val, dict) and val.get("success"):
        _CACHE[key] = (now + _CACHE_TTL, val)
    return val


def _dates(days: int) -> tuple:
    end = datetime.now()
    start = end - timedelta(days=max(1, days) - 1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _fingerprint(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Vân tay kiểu lỗi: phân bố nguyên nhân, dạng TỈ LỆ chứ không phải số đếm.

    Đây là thứ duy nhất so sánh được giữa các máy khi chúng chạy sản phẩm khác
    nhau. Tỉ lệ pass tuyệt đối thì không: M2 pass 69% trên hạt tiêu và
    PC-Auto-1 pass 98% trên muối không nói lên máy nào tệ hơn. Nhưng
    "M2 có no_detection 43% còn mặt bằng ~10%" thì đúng dù khác sản phẩm, vì
    no_detection thuộc về camera/trigger/ánh sáng chứ không thuộc về hàng.

    Chỉ trả tỉ lệ, KHÔNG trả số đếm tuyệt đối. `explain_failures`
    lấy mẫu tối đa `sample_limit`, nên số đếm trong `causes` là của MẪU chứ không
    phải của cả kỳ — đưa nó ra cạnh `total_fail` là dựng sẵn cái bẫy đã ghi trong
    PIPELINE.md: "tổng fail 1.036" rồi liệt kê "144/106/102", ba số sau là của
    mẫu 294.
    """
    if not raw.get("success"):
        return None
    causes = raw.get("causes") or []
    total = sum(c.get("products", 0) for c in causes) or 1
    return {
        "sample_products": raw.get("failed_camera_frames_examined"),
        "sample_covers_all": raw.get("sample_covers_all"),
        "sampling": raw.get("sampling"),
        "by_cause": [
            {
                "cause": c["cause"],
                "label": c.get("label"),
                # Chuẩn hoá để tổng bằng 100: đây là TỈ TRỌNG giữa các nguyên
                # nhân, không phải "bao nhiêu % sản phẩm trượt bước này". Một sản
                # phẩm trượt được nhiều bước, nên con số chưa chuẩn hoá của
                # `explain_failures` cộng lại vượt 100 — trộn hai thứ đó vào một
                # câu trả lời là ra hai số khác nhau cho cùng một nguyên nhân.
                "share_of_causes_pct": round(c.get("products", 0) * 100.0 / total, 1),
            }
            for c in causes
        ],
        "template_similarity_avg": raw.get("template_similarity_avg"),
    }


def _shift_stats(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Pass/fail gộp theo CA — dùng lại `_shift_expr` của analytics_tools chứ không
    viết lại: ca C (22:00–06:00) vắt qua nửa đêm, điều kiện là `giờ >= 22 HOẶC
    giờ < 6` chứ không phải một khoảng liên tục, và biểu thức đó đã trả giá xong
    ở tầng tool. Viết lại bằng khoảng liên tục là mất sạch ca đêm.
    """
    from agent_app.db.mongodb import get_sync_database
    from agent_app.tools.analytics_tools import _TIME_FIELD, _local_bound, _shift_expr

    db = get_sync_database()
    pipeline = [
        {"$match": {_TIME_FIELD: {"$gte": _local_bound(start_date, end=False),
                                  "$lte": _local_bound(end_date, end=True)}}},
        {"$group": {
            "_id": {"shift": _shift_expr(), "result": "$product_pass_fail"},
            "count": {"$sum": 1},
        }},
    ]
    out: Dict[str, Dict[str, int]] = {}
    for r in db["inference_results"].aggregate(pipeline):
        sh = r["_id"]["shift"] or "?"
        d = out.setdefault(sh, {"pass": 0, "fail": 0})
        key = "pass" if r["_id"]["result"] == "PASS" else "fail"
        d[key] += r["count"]
    for d in out.values():
        t = d["pass"] + d["fail"]
        d["total"] = t
        d["pass_rate"] = round(d["pass"] * 100.0 / t, 2) if t else None
    return {"success": True, "by_shift": out}


@router.get("/rollup", summary="Số liệu sản xuất gọn cho tầng fleet (không LLM)")
async def rollup(
    days: int = Query(7, ge=1, le=90, description="Số ngày tính ngược từ hôm nay"),
    granularity: str = Query("day", pattern="^(hour|day|week|shift)$",
                             description="Chia trend theo giờ/ngày/tuần, hoặc gộp theo ca"),
    causes: bool = Query(True, description="Kèm vân tay kiểu lỗi (chậm hơn)"),
    sample_limit: int = Query(200, ge=50, le=1000),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Gói dữ liệu vài KB cho fleet: sản lượng, pass/fail, xu hướng, vân tay lỗi.

    Cố ý KHÔNG trả ảnh mẫu. `explain_failures` có trường `samples` chứa ảnh sản
    phẩm lỗi; đẩy chúng qua link tới Jetson (đo được vài chục KB/s) là biến một
    gói 3 KB thành vài MB. Fleet cần tỉ lệ, không cần ảnh — ai muốn xem ảnh thì
    đi đường ủy quyền hỏi thẳng agent của máy đó.
    """
    from agent_app.tools.analytics_tools import (
        explain_failures,
        get_pass_fail_stats,
        list_recipes,
    )

    start_date, end_date = _dates(days)
    out: Dict[str, Any] = {
        "success": True,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "period": {"start": start_date, "end": end_date, "days": days},
    }

    # "shift" không phải một cách chia trend theo thời gian mà là một cách GỘP
    # khác hẳn, nên trend vẫn chia theo ngày còn kết quả theo ca nằm ở khối riêng.
    trend_gran = granularity if granularity in ("hour", "day", "week") else "day"
    stats = _cached(f"stats:{start_date}:{end_date}:{trend_gran}",
                    lambda: get_pass_fail_stats(start_date=start_date,
                                                end_date=end_date, group_by=trend_gran))
    if stats.get("success"):
        s = stats.get("summary") or {}
        out["production"] = {
            "total_products": s.get("total_products"),
            "pass": s.get("pass_count"),
            "fail": s.get("fail_count"),
            "pass_rate": s.get("pass_rate"),
            # Chuẩn hoá theo ngày: hai máy chạy số ngày khác nhau trong kỳ thì
            # sản lượng tuyệt đối không so được, per-day thì so được.
            "per_day": round((s.get("total_products") or 0) / max(days, 1), 1),
            "trend_granularity": trend_gran,
            "trend": stats.get("trend") or {},
        }
    else:
        out["production"] = None
        out["production_error"] = stats.get("error")

    if causes:
        raw = _cached(f"causes:{start_date}:{end_date}:{sample_limit}",
                      lambda: explain_failures(start_date=start_date,
                                               end_date=end_date,
                                               sample_limit=sample_limit))
        out["failure_modes"] = _fingerprint(raw)
        if not raw.get("success"):
            out["failure_modes_error"] = raw.get("error")

    if granularity == "shift":
        out["by_shift"] = _cached(f"shift:{start_date}:{end_date}",
                                  lambda: _shift_stats(start_date, end_date)).get("by_shift")

    recs = _cached(f"recipes:{days}", lambda: list_recipes(days=days))
    if recs.get("success"):
        items: List[Dict[str, Any]] = recs.get("recipes") or recs.get("items") or []
        out["recipes"] = [
            {"name": r.get("name") or r.get("recipe_name"),
             "products": r.get("total") or r.get("total_products")}
            for r in items[:10]
        ]

    return out


# ═══════════════════════════════════════════════════════════════════════════
# GĐ 1 — ba khoảng trống dữ liệu (docs/ui/06-data-contracts.md)
# ═══════════════════════════════════════════════════════════════════════════

# Gốc ảnh sản phẩm: image_path trong DB là đường TƯƠNG ĐỐI so với backend/uploads
# (đo thật trên M2: "inference_results/<recipe>/<ngày>/<id>/..._viz.jpg").
_UPLOADS_ROOT = Path(__file__).resolve().parents[3] / "backend" / "uploads"
_IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}


@router.get("/staff", summary="Hồ sơ nhân sự đầy đủ (không LLM)")
async def staff(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Trả về user kèm CẢ các field hồ sơ mà `/api/users/` của backend làm rơi.

    Kiểm chứng trên M2: MongoDB có `employee_code`, `department`, `job_title`,
    `shift`, `production_line`, `hire_date` — response model của backend map
    field tường minh nên cả sáu không ra khỏi API (cùng lớp lỗi đã ghi cho
    `recipe_to_response`). Không có chúng thì tab Nhân sự nhóm theo line/bộ
    phận/ca không dựng được.

    Endpoint này đọc thẳng collection thay vì sửa backend: sửa backend là restart
    5 tiến trình đang phục vụ dây chuyền, còn thêm endpoint ở đây chỉ restart
    agent — thứ vốn không nằm trên đường sản xuất.
    """
    from agent_app.db.mongodb import get_sync_database

    db = get_sync_database()
    out = []
    # Loại hashed_password ngay ở projection, không phải lọc sau — trường mật
    # khẩu không được rời khỏi DB rồi mới bị bỏ.
    for u in db["users"].find({}, {"hashed_password": 0}):
        u["_id"] = str(u["_id"])
        for k in ("created_at", "updated_at", "last_login", "hire_date"):
            if u.get(k) is not None and not isinstance(u[k], str):
                u[k] = str(u[k])
        out.append(u)

    return {
        "success": True,
        "count": len(out),
        "users": out,
        # Nhắc tầng trên: username KHÔNG duy nhất giữa các máy (admin tồn tại
        # trên mọi máy) — fleet phải khoá theo (máy, username).
        "note": "username chỉ duy nhất TRONG một máy; định danh xuyên máy là employee_code.",
    }


def _img_id(rel_path: str) -> str:
    return base64.urlsafe_b64encode(rel_path.encode()).decode().rstrip("=")


def _img_path(img_id: str) -> Optional[Path]:
    """
    Giải id ảnh về đường dẫn, CHỈ chấp nhận file nằm trong uploads/.

    Id là base64 của đường dẫn tương đối — không phải để giấu, mà để endpoint
    không bao giờ nhận đường dẫn tuỳ ý: giải xong phải resolve nằm trong
    _UPLOADS_ROOT và có đuôi ảnh, không thì coi như không tồn tại. Một id bịa
    ra (`../../etc/...`) chết ở đây chứ không tới được đĩa.
    """
    try:
        pad = "=" * (-len(img_id) % 4)
        rel = base64.urlsafe_b64decode(img_id + pad).decode()
    except (ValueError, UnicodeDecodeError):
        return None
    p = (_UPLOADS_ROOT / rel).resolve()
    try:
        p.relative_to(_UPLOADS_ROOT.resolve())
    except ValueError:
        return None
    if p.suffix.lower() not in _IMG_SUFFIXES or not p.is_file():
        return None
    return p


@router.get("/failure-images", summary="Danh sách ảnh sản phẩm lỗi (không LLM)")
async def failure_images(
    days: int = Query(7, ge=1, le=90),
    limit: int = Query(12, ge=1, le=48),
    cause: Optional[str] = Query(None, description="Lọc theo nguyên nhân, vd no_detection"),
    sample_limit: int = Query(200, ge=50, le=1000),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Metadata ảnh sản phẩm lỗi — ảnh thật lấy qua `/fleet/failure-image/{id}`.

    Tái dùng mẫu của `explain_failures` (cùng cache với rollup, nên gọi thêm
    endpoint này không mổ lại document): mẫu đã được RẢI ĐỀU qua các nguyên nhân
    và ưu tiên ảnh có `expected` — ảnh kèm dòng "mong X → đọc Y" tự giải thích,
    ảnh không có thì người xem phải tự đoán. Không viết lại logic chọn ảnh.
    """
    from agent_app.tools.analytics_tools import _CAUSE_LABELS, explain_failures

    start_date, end_date = _dates(days)
    raw = _cached(f"causes:{start_date}:{end_date}:{sample_limit}",
                  lambda: explain_failures(start_date=start_date,
                                           end_date=end_date,
                                           sample_limit=sample_limit))
    if not raw.get("success"):
        return {"success": False, "error": raw.get("error")}

    items = []
    for s in raw.get("samples") or []:
        rel = s.get("image_path")
        if not rel:
            continue
        # `cause` do tool gán lúc tạo mẫu; `_causes` không tới được đây vì
        # attach_templates dọn khoá nội bộ trước khi trả.
        primary = s.get("cause") or "unknown"
        if cause and cause != primary:
            continue
        items.append({
            "id": _img_id(rel),
            "cause": primary,
            "cause_label": _CAUSE_LABELS.get(primary, primary),
            "camera": s.get("camera"),
            "timestamp": s.get("timestamp"),
            "recipe_name": s.get("recipe_name"),
            "expected": s.get("expected"),
            "recognized": s.get("recognized"),
        })
        if len(items) >= limit:
            break

    return {
        "success": True,
        "count": len(items),
        "images": items,
        "sampling": raw.get("sampling"),
        "sample_covers_all": raw.get("sample_covers_all"),
    }


@router.get("/failure-image/{img_id}", summary="Ảnh sản phẩm lỗi, thu nhỏ")
async def failure_image(
    img_id: str,
    w: int = Query(480, ge=64, le=1600),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Phục vụ ảnh, mặc định thu về 480px.

    Thu nhỏ không phải tuỳ chọn: ảnh gốc 1920×1200 ~500 KB, đường tới Jetson đo
    được vài chục KB/s — lưới 12 ảnh gốc là ~10 phút tải. 480px JPEG ~40–60 KB.
    """
    from io import BytesIO

    from fastapi.responses import Response
    from PIL import Image

    p = _img_path(img_id)
    if p is None:
        raise HTTPException(404, "Không có ảnh này")

    def _render() -> bytes:
        im = Image.open(p)
        im = im.convert("RGB")
        if im.width > w:
            im.thumbnail((w, w * im.height // max(im.width, 1)))
        buf = BytesIO()
        im.save(buf, "JPEG", quality=78)
        return buf.getvalue()

    # PIL chạy trong threadpool để không chặn event loop của agent — resize một
    # ảnh mất ~100ms trên Jetson, và agent này còn đang phục vụ chat.
    import anyio

    data = await anyio.to_thread.run_sync(_render)
    return Response(content=data, media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=86400"})
