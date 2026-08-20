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


# ═══════════════════════════════════════════════════════════════════════════
# Ảnh sản phẩm vừa kiểm — nguồn cho khung ảnh trên Fleet Console
# ═══════════════════════════════════════════════════════════════════════════

_TIME_FIELD = "created_at"   # cùng lý do như analytics_tools: đây là field CÓ INDEX


def _frame_payload(doc: Dict[str, Any], frame: Dict[str, Any],
                   serial: str) -> Dict[str, Any]:
    """Một frame + đủ ngữ cảnh để câu chú thích dưới ảnh tự đứng được."""
    tv = frame.get("text_verification") or {}
    bad = next((r for r in (tv.get("results") or []) if not r.get("match")), None)
    ts = doc.get("timestamp") or doc.get(_TIME_FIELD)
    return {
        "id": _img_id(frame["image_path"]),
        "product_id": str(doc.get("_id")),
        "verdict": doc.get("product_pass_fail"),
        "frame_verdict": frame.get("pass_fail"),
        "timestamp": str(ts) if ts is not None else None,
        "age_seconds": (int((datetime.now() - ts).total_seconds())
                        if isinstance(ts, datetime) else None),
        "camera": serial,
        "recipe_name": doc.get("recipe_name"),
        "recipe_id": str(doc.get("recipe_id")) if doc.get("recipe_id") else None,
        "confidence": frame.get("confidence"),
        # `expected` / `recognized` chỉ có khi chuỗi đọc sai. Có nó thì tấm ảnh
        # tự giải thích được; không có thì người xem phải tự đoán vì sao fail.
        "expected": (bad or {}).get("expected"),
        "recognized": (bad or {}).get("recognized"),
        "template_name": frame.get("template_name"),
    }


def _pick_frame(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Frame đại diện của một sản phẩm: ưu tiên frame FAIL, vì đó là frame giải
    thích được vì sao sản phẩm trượt. Sản phẩm đạt thì lấy frame đầu có ảnh."""
    best = None
    for cam in doc.get("camera_results") or []:
        serial = str(cam.get("serial_number"))
        for frame in cam.get("frames") or []:
            if not frame.get("image_path"):
                continue
            if frame.get("pass_fail") == "FAIL":
                return _frame_payload(doc, frame, serial)
            if best is None:
                best = _frame_payload(doc, frame, serial)
    return best


def _scan(query: Dict[str, Any], sort_dir: int, scan: int) -> Optional[Dict[str, Any]]:
    """Quét ngược từ mốc thời gian, trả frame CÓ ẢNH đầu tiên gặp được.

    Phải quét nhiều document chứ không lấy đúng một: không phải sản phẩm nào
    cũng lưu ảnh (chỉ frame fail và một phần frame pass mới có `image_path`),
    nên `find_one` rất hay trả về một bản ghi không có gì để xem.
    """
    from agent_app.db.mongodb import get_sync_database

    cur = (get_sync_database()["inference_results"]
           .find(query)
           .sort(_TIME_FIELD, sort_dir)
           .limit(scan))
    for doc in cur:
        got = _pick_frame(doc)
        if got:
            return got
    return None


@router.get("/latest-frame", summary="Ảnh sản phẩm kiểm gần nhất (không LLM)")
async def latest_frame(
    verdict: str = Query("any", pattern="^(any|PASS|FAIL)$"),
    within_hours: int = Query(24, ge=1, le=168),
    scan: int = Query(60, ge=5, le=400),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Metadata của sản phẩm vừa được kiểm — ảnh lấy qua `/fleet/frame/{id}`.

    Đây KHÔNG phải luồng camera trực tiếp, và tầng trên phải gọi đúng tên như
    vậy. Ảnh camera thô 60 giây lấy một tấm thì phần lớn rơi vào lúc băng tải
    trống; còn "sản phẩm vừa kiểm" thì tấm nào cũng có nội dung và có phán
    quyết đi kèm — đó mới là thứ người đứng máy cần nhìn.

    `within_hours` giữ truy vấn có biên: không chặn thì một máy đã dừng ba
    tháng sẽ kéo cả collection ra để tìm tấm ảnh cuối cùng.
    """
    since = datetime.now() - timedelta(hours=within_hours)
    query: Dict[str, Any] = {_TIME_FIELD: {"$gte": since}}
    if verdict != "any":
        query["product_pass_fail"] = verdict

    got = _scan(query, -1, scan)
    return {"success": True, "found": got is not None, "frame": got,
            "verdict_filter": verdict, "within_hours": within_hours}


@router.get("/frame-pair", summary="Ảnh đạt và ảnh lỗi mới nhất, cùng lúc")
async def frame_pair(
    within_hours: int = Query(24, ge=1, le=168),
    scan: int = Query(60, ge=5, le=400),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Cặp ảnh đạt / lỗi gần nhất của cùng một máy.

    Một tấm ảnh lỗi đứng riêng không nói được lỗi nằm ở đâu — phải có tấm đạt
    bên cạnh thì mắt mới so ra được là in mờ, lệch khung hay sai chuỗi. Trả cả
    hai trong MỘT lần gọi vì chúng luôn được xem cùng nhau, và mỗi lần gọi thêm
    là thêm một vòng qua đường tới Jetson.
    """
    since = datetime.now() - timedelta(hours=within_hours)
    base = {_TIME_FIELD: {"$gte": since}}
    ok = _scan({**base, "product_pass_fail": "PASS"}, -1, scan)
    bad = _scan({**base, "product_pass_fail": "FAIL"}, -1, scan)
    return {"success": True, "pass_frame": ok, "fail_frame": bad,
            "within_hours": within_hours}


@router.get("/frames-around", summary="Ảnh ngay trước và ngay sau một mốc thời gian")
async def frames_around(
    ts: str = Query(..., description="Mốc thời gian ISO, vd 2026-08-20T12:15:00"),
    scan: int = Query(60, ge=5, le=400),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Ảnh cuối TRƯỚC một mốc và ảnh đầu SAU nó.

    Dùng để ghép nhật ký thao tác với ảnh: nhật ký biết chính xác lúc nào ai
    chạy `update_recipe`, còn hai tấm này cho thấy việc đó đổi gì trên sản phẩm
    thật. Không có cặp ảnh này thì "đã sửa recipe" mãi chỉ là một dòng chữ.
    """
    try:
        at = datetime.fromisoformat(ts.replace("Z", ""))
    except ValueError:
        raise HTTPException(400, "ts phải là ISO datetime")
    before = _scan({_TIME_FIELD: {"$lt": at}}, -1, scan)
    after = _scan({_TIME_FIELD: {"$gte": at}}, 1, scan)
    return {"success": True, "at": ts, "before": before, "after": after}


@router.get("/frame/{img_id}", summary="Ảnh sản phẩm, thu nhỏ")
@router.get("/failure-image/{img_id}", summary="Ảnh sản phẩm lỗi (tên cũ)")
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


@router.get("/audit", summary="Nhật ký thao tác, lọc bản ghi giả lập (không LLM)")
async def audit(
    days: int = Query(7, ge=1, le=90),
    username: Optional[str] = Query(None),
    action_type: Optional[str] = Query(None),
    include_simulated: bool = Query(False, description="Kèm bản ghi demo (simulated)"),
    limit: int = Query(100, ge=1, le=500),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Nhật ký thao tác của máy này.

    Hai điểm khác `get_audit_logs` của tool, và cả hai đều cần cho tầng fleet:

    **Lọc `simulated` mặc định BẬT.** Dữ liệu demo đã seed mang cờ
    `simulated: true`; trộn chúng vào nhật ký thật thì không ai phân biệt được,
    mà bảng vẫn trông hoàn toàn bình thường.

    **Username lạ KHÔNG phải lỗi.** Tool trả lỗi kèm danh sách hợp lệ — đúng cho
    một máy, nhưng ở fleet thì `truongca_m2` chỉ tồn tại trên M2, và bốn máy kia
    trả lỗi sẽ bị đọc thành "bốn máy hỏng". Ở đây trả `matched: false` để tầng
    trên phân biệt "không có người này" với "máy không trả lời".
    """
    from agent_app.db.mongodb import get_sync_database
    from agent_app.tools.analytics_tools import _local_bound

    db = get_sync_database()
    q: Dict[str, Any] = {
        "timestamp": {"$gte": _local_bound(_dates(days)[0], end=False),
                      "$lte": _local_bound(_dates(days)[1], end=True)},
    }
    if not include_simulated:
        q["simulated"] = {"$ne": True}
    if action_type:
        q["action_type"] = action_type
    if username:
        known = db["action_logs"].distinct("username")
        if username not in known:
            return {"success": True, "matched": False, "count": 0, "entries": [],
                    "by_action": {},
                    "message": f"Máy này không có bản ghi nào của '{username}'."}
        q["username"] = username

    cur = (db["action_logs"]
           .find(q, {"old_value": 0, "new_value": 0, "user_agent": 0})
           .sort("timestamp", -1).limit(limit))
    entries = []
    for d in cur:
        entries.append({
            "time": d["timestamp"].isoformat() if d.get("timestamp") else None,
            "username": d.get("username"),
            "action_type": d.get("action_type"),
            "resource_type": d.get("resource_type"),
            "resource_id": d.get("resource_id"),
            "description": d.get("description"),
            "ip_address": d.get("ip_address"),
        })

    # Thống kê theo action tính trên TOÀN BỘ match, không phải trên `entries` đã
    # bị `limit` cắt — đúng lỗi đã sửa ở tầng tool: thẻ ghi "14:12 → 14:23" trong
    # khi bản ghi đầu tiên là 11:05, vì thống kê tính trên phần đã cắt.
    by_action = {r["_id"]: r["n"] for r in db["action_logs"].aggregate([
        {"$match": q}, {"$group": {"_id": "$action_type", "n": {"$sum": 1}}},
        {"$sort": {"n": -1}},
    ]) if r["_id"]}

    return {"success": True, "matched": True, "count": len(entries),
            "total_in_period": sum(by_action.values()),
            "by_action": by_action, "entries": entries}


@router.get("/log-errors", summary="Tóm tắt lỗi hệ thống của máy này (không LLM)")
async def log_errors(
    date: Optional[str] = Query(None, description="YYYY-MM-DD, mặc định hôm nay"),
    top: int = Query(8, ge=1, le=30),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Nhóm ERROR/WARNING/CRITICAL từ file log, đã gom nhóm giống nhau.

    Fleet chỉ nhận BẢN TÓM TẮT, không bao giờ nhận file log. File log trên máy
    này từng phình tới 1,4 GB; đọc trọn một file như vậy là treo agent và ăn hết
    RAM của Jetson (lý do đã ghi trong `log_tools.py`). Muốn sâu hơn thì ủy quyền
    câu hỏi cho agent của chính máy đó.
    """
    from agent_app.tools.log_tools import summarize_log_errors

    r = _cached(f"logerr:{date or 'today'}:{top}",
                lambda: summarize_log_errors(date=date, top=top))
    if not r.get("success"):
        return {"success": False, "error": r.get("error")}
    return {
        "success": True,
        "date": r.get("date"),
        "total_problem_lines": r.get("total_problem_lines"),
        "distinct_problems": r.get("distinct_problems"),
        "by_level": r.get("by_level"),
        "scanned_files": len(r.get("scanned_files") or []),
        "skipped_categories": r.get("skipped_categories"),
        # Giữ chữ ký nhóm + số lần, và MỘT dòng ví dụ đã được tool cắt ngắn.
        # Không đưa dòng log thô đầy đủ: một traceback dài vài KB, nhân 5 máy là
        # vài trăm KB qua link chậm chỉ để dựng một cái bảng.
        "problems": [{"signature": g.get("signature"),
                      "level": g.get("level"),
                      "category": g.get("category"),
                      "count": g.get("count"),
                      "first_seen": g.get("first_seen"),
                      "last_seen": g.get("last_seen"),
                      "example": g.get("example")}
                     for g in (r.get("problems") or [])[:top]],
    }
