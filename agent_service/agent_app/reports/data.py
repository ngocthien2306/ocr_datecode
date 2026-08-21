"""
Số liệu cho báo cáo sản xuất, gom thẳng từ MongoDB.

Tương đương hai endpoint `/inference-results/statistics/summary` và
`/statistics/timeseries` của backend, nhưng viết lại ở đây thay vì gọi sang:
agent service chạy process riêng và phải làm việc được kể cả khi backend đang
restart. Cấu trúc trả về giữ đúng như backend để phần render port từ
`reportGenerator.ts` không phải sửa gì.

Lọc theo `created_at` chứ không phải `timestamp` — xem ghi chú dài trong
`analytics_tools.py`: chỉ `created_at` mới nằm trong compound index, và chênh
lệch giữa hai field tối đa 1 ms.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from agent_app.core.config import settings
from agent_app.db.mongodb import get_sync_database
from agent_app.tools.analytics_tools import _TIME_FIELD, _local_bound, _TZ

Granularity = str  # 'hour' | 'day' | 'week'


def _rate(pass_: int, total: int) -> float:
    return round(pass_ / total * 100, 2) if total > 0 else 0.0


def build_summary(
    start_dt: datetime,
    end_dt: datetime,
    recipe_ids: Optional[List[str]] = None,
    include_camera: bool = False,
) -> Dict[str, Any]:
    """
    Tổng quan kỳ báo cáo, tách theo recipe và (tuỳ chọn) theo camera.

    `include_camera` mặc định TẮT vì nó đắt hơn hẳn phần còn lại. Nhóm theo
    recipe chỉ đụng tới các field nằm trong compound index nên Mongo trả lời
    ngay; còn tách theo camera phải `$unwind` mảng `camera_results`, tức nạp
    trọn document — mà mỗi document nặng trung bình 62,8 KB vì nhúng ảnh base64
    của từng ký tự. Đo thực tế:

        theo recipe, 7 ngày (73k bản ghi) :  ~0,4 s
        thêm theo camera, cùng kỳ         : ~51,8 s

    Bản báo cáo HTML không dùng `by_camera` (kiểm chứng: `reportGenerator.ts`
    không nhắc tới nó lần nào), nên chỉ các định dạng bảng mới cần bật, và chỉ
    khi kỳ báo cáo đủ ngắn.
    """
    db = get_sync_database()
    col = db["inference_results"]
    match: Dict[str, Any] = {_TIME_FIELD: {"$gte": start_dt, "$lte": end_dt}}
    if recipe_ids:
        match["recipe_id"] = {"$in": recipe_ids}

    by_recipe_rows = list(col.aggregate([
        {"$match": match},
        {"$group": {
            "_id": {"id": "$recipe_id", "name": "$recipe_name"},
            "total": {"$sum": 1},
            "pass": {"$sum": {"$cond": [{"$eq": ["$product_pass_fail", "PASS"]}, 1, 0]}},
        }},
        {"$sort": {"total": -1}},
    ]))

    by_recipe = []
    total = passed = 0
    for r in by_recipe_rows:
        t, p = r["total"], r["pass"]
        total += t
        passed += p
        by_recipe.append({
            "recipe_id": r["_id"]["id"],
            "recipe_name": r["_id"]["name"] or "(không tên)",
            "total": t, "pass": p, "fail": t - p, "pass_rate": _rate(p, t),
        })

    by_camera: List[Dict[str, Any]] = []
    if not include_camera:
        return {
            "total": total, "pass": passed, "fail": total - passed,
            "pass_rate": _rate(passed, total),
            "period": {"start_date": start_dt.isoformat(), "end_date": end_dt.isoformat()},
            "by_camera": by_camera,
            "by_recipe": by_recipe,
        }

    # Verdict của camera nằm ở `camera_results.pass_fail`, KHÔNG phải
    # `product_pass_fail`: dùng verdict sản phẩm thì mọi camera trên một sản
    # phẩm fail đều bị tính là fail, kể cả camera chụp đạt.
    by_camera_rows = list(col.aggregate([
        {"$match": match},
        {"$unwind": "$camera_results"},
        {"$group": {
            "_id": {"id": "$camera_results.camera_id", "sn": "$camera_results.serial_number"},
            "total": {"$sum": 1},
            "pass": {"$sum": {"$cond": [{"$eq": ["$camera_results.pass_fail", "PASS"]}, 1, 0]}},
        }},
        {"$sort": {"total": -1}},
    ]))
    by_camera = [{
        "camera_id": c["_id"]["id"],
        "serial_number": c["_id"]["sn"],
        "total": c["total"], "pass": c["pass"], "fail": c["total"] - c["pass"],
        "pass_rate": _rate(c["pass"], c["total"]),
    } for c in by_camera_rows]

    return {
        "total": total, "pass": passed, "fail": total - passed,
        "pass_rate": _rate(passed, total),
        "period": {"start_date": start_dt.isoformat(), "end_date": end_dt.isoformat()},
        "by_camera": by_camera,
        "by_recipe": by_recipe,
    }


def _bucket_expr(granularity: Granularity) -> Dict[str, Any]:
    """
    Biểu thức gom mốc thời gian, tính theo giờ địa phương.

    `$dateToString` với `timezone` để mốc "ngày" là ngày theo giờ VN chứ không
    phải theo UTC — lệch 7 tiếng thì ca đêm bị đẩy sang ngày hôm sau.

    Tuần dùng `%G-%V` (ISO week-year + ISO week) chứ không phải `%Y-%U`:
    `%U` đếm tuần bắt đầu từ Chủ nhật và tính tuần đầu năm khác chuẩn ISO, nên
    cuối tháng 12 sẽ gom nhầm sang tuần 00 của năm sau.
    """
    fmt = {"hour": "%Y-%m-%d %H:00", "day": "%Y-%m-%d", "week": "%G-W%V"}[granularity]
    return {"$dateToString": {"format": fmt, "date": f"${_TIME_FIELD}",
                              "timezone": settings.TIMEZONE}}


def build_timeseries(
    start_dt: datetime,
    end_dt: datetime,
    granularity: Granularity = "day",
    recipe_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Số liệu theo mốc thời gian, mỗi mốc tách tiếp theo recipe."""
    db = get_sync_database()
    match: Dict[str, Any] = {_TIME_FIELD: {"$gte": start_dt, "$lte": end_dt}}
    if recipe_ids:
        match["recipe_id"] = {"$in": recipe_ids}

    rows = list(db["inference_results"].aggregate([
        {"$match": match},
        {"$group": {
            "_id": {
                "bucket": _bucket_expr(granularity),
                "recipe_id": "$recipe_id",
                "recipe_name": "$recipe_name",
            },
            "total": {"$sum": 1},
            "pass": {"$sum": {"$cond": [{"$eq": ["$product_pass_fail", "PASS"]}, 1, 0]}},
        }},
        {"$sort": {"_id.bucket": 1}},
    ]))

    buckets: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        key = r["_id"]["bucket"]
        pt = buckets.setdefault(key, {
            "timestamp": key, "total": 0, "pass": 0, "fail": 0,
            "pass_rate": 0.0, "by_recipe": [],
        })
        t, p = r["total"], r["pass"]
        pt["total"] += t
        pt["pass"] += p
        pt["fail"] += t - p
        pt["by_recipe"].append({
            "recipe_id": r["_id"]["recipe_id"],
            "recipe_name": r["_id"]["recipe_name"] or "(không tên)",
            "total": t, "pass": p, "fail": t - p, "pass_rate": _rate(p, t),
        })

    data = [buckets[k] for k in sorted(buckets)]
    for pt in data:
        pt["pass_rate"] = _rate(pt["pass"], pt["total"])

    return {
        "granularity": granularity,
        "period": {"start_date": start_dt.isoformat(), "end_date": end_dt.isoformat()},
        "data": data,
    }


# ── Khoảng thời gian theo tên ────────────────────────────────────────────────

PERIOD_LABELS = {
    "today": "Today", "yesterday": "Yesterday",
    "thisweek": "This Week", "lastweek": "Last Week",
    "7days": "Last 7 Days", "thismonth": "This Month",
    "lastmonth": "Last Month", "30days": "Last 30 Days",
}


def resolve_period(
    period: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
) -> tuple[datetime, datetime, str]:
    """
    (start, end, nhãn) từ tên kỳ có sẵn hoặc từ ngày cụ thể.

    Ngày do người dùng đưa luôn được hiểu theo giờ địa phương rồi quy về
    naive-UTC bằng `_local_bound`, đúng cách MongoDB đang lưu.
    """
    if start_date or end_date:
        s = _local_bound(start_date, end=False)
        e = _local_bound(end_date, end=True)
        return s, e, f"{(start_date or '')[:10]} → {(end_date or '')[:10]}".strip(" →")

    today = datetime.now(_TZ)
    d = lambda n: (today - timedelta(days=n)).strftime("%Y-%m-%d")  # noqa: E731
    p = (period or "today").lower()

    if p == "yesterday":
        return _local_bound(d(1), end=False), _local_bound(d(1), end=True), PERIOD_LABELS[p]
    if p == "7days":
        return _local_bound(d(6), end=False), _local_bound(d(0), end=True), PERIOD_LABELS[p]
    if p == "30days":
        return _local_bound(d(29), end=False), _local_bound(d(0), end=True), PERIOD_LABELS[p]
    if p == "thisweek":
        back = today.weekday()                      # 0 = thứ Hai
        return _local_bound(d(back), end=False), _local_bound(d(0), end=True), PERIOD_LABELS[p]
    if p == "lastweek":
        back = today.weekday() + 7
        return _local_bound(d(back), end=False), _local_bound(d(back - 6), end=True), PERIOD_LABELS[p]
    if p == "thismonth":
        first = today.replace(day=1).strftime("%Y-%m-%d")
        return _local_bound(first, end=False), _local_bound(d(0), end=True), PERIOD_LABELS[p]
    if p == "lastmonth":
        first_this = today.replace(day=1)
        last_prev = first_this - timedelta(days=1)
        first_prev = last_prev.replace(day=1)
        return (_local_bound(first_prev.strftime("%Y-%m-%d"), end=False),
                _local_bound(last_prev.strftime("%Y-%m-%d"), end=True),
                PERIOD_LABELS[p])

    return _local_bound(d(0), end=False), _local_bound(d(0), end=True), PERIOD_LABELS["today"]
