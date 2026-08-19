"""
Analytics Tools for Historical Data Analysis
Provides tools for querying and analyzing production data
"""

import re
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pydantic import BaseModel, Field
from agent_app.core.config import settings
from agent_app.db.mongodb import get_sync_database
from agent_app.tools.base_tool import BaseTool, ToolMetadata
import logging

logger = logging.getLogger(__name__)

_TZ = ZoneInfo(settings.TIMEZONE)


# ============================================================================
# Vì sao lọc theo `created_at` chứ không phải `timestamp`
#
# Hai field này là một: trên 72.550 bản ghi 7 ngày, chênh lệch tối đa 1 ms
# (231 bản ghi lệch đúng 1 ms, còn lại bằng 0). Nhưng chỉ `created_at` nằm
# trong compound index mà scripts/optimize_mongodb.sh dựng sẵn:
#
#     {created_at: 1, product_pass_fail: 1, recipe_id: 1, recipe_name: 1}
#
# Lọc bằng `timestamp` thì index `{timestamp: -1}` chỉ lọc được khoảng ngày,
# sau đó Mongo phải nạp TOÀN BỘ document để đọc `product_pass_fail`. Mỗi
# document nặng trung bình 62,8 KB vì `char_verification.results[].mask_diff_b64`
# nhúng một ảnh PNG base64 cho từng ký tự. Đo trên 7 ngày (72k bản ghi):
#
#     $group lọc theo timestamp   : 18,82 s
#     $group lọc theo created_at  :  0,20 s   ← index bao trọn, không nạp document
#
# Đổi field là đủ, không cần thêm index mới.
# ============================================================================

_TIME_FIELD = "created_at"

# `find()` trong explain_failures phải nạp document thật (cần frames chi tiết),
# nên bỏ hẳn ảnh base64 của từng ký tự — thứ chiếm gần hết 62,8 KB kia mà phần
# phân tích không dùng tới.
_FAIL_PROJECTION = {"camera_results.frames.char_verification.results.mask_diff_b64": 0}


# ============================================================================
# Quy đổi múi giờ
#
# MongoDB lưu timestamp là naive-UTC (kiểm chứng: bản ghi mới nhất lệch 0 phút
# so với utcnow(), lệch 420 phút so với now()). User thì nói "hôm nay" theo giờ
# VN. Query thẳng bằng datetime.now() ⇒ "hôm nay" hoá ra là 07:00 hôm nay →
# 07:00 ngày mai, tức mọi con số lệch 7 tiếng.
# ============================================================================

def _to_utc(dt_local: datetime) -> datetime:
    """Naive/aware local datetime → naive UTC (đúng cách DB lưu)."""
    if dt_local.tzinfo is None:
        dt_local = dt_local.replace(tzinfo=_TZ)
    return dt_local.astimezone(timezone.utc).replace(tzinfo=None)


def _local_bound(value: Optional[str], *, end: bool) -> datetime:
    """
    Tham số ngày do user/LLM đưa (giờ địa phương) → mốc naive-UTC để query.

    - None            → hôm nay theo giờ địa phương
    - 'YYYY-MM-DD'    → nới ra trọn ngày (00:00:00 hoặc 23:59:59.999999)
    - có kèm giờ      → tôn trọng đúng giá trị đó
    """
    if value:
        dt = datetime.fromisoformat(value)
        date_only = len(value.strip()) == 10
    else:
        dt = datetime.now(_TZ).replace(tzinfo=None)
        date_only = True

    if date_only:
        dt = (
            dt.replace(hour=23, minute=59, second=59, microsecond=999999)
            if end
            else dt.replace(hour=0, minute=0, second=0, microsecond=0)
        )

    return _to_utc(dt)


def _to_local_str(dt_utc: datetime) -> str:
    """Naive-UTC → chuỗi ISO giờ địa phương, để hiển thị lại cho user."""
    return dt_utc.replace(tzinfo=timezone.utc).astimezone(_TZ).isoformat()


# ============================================================================
# Lọc theo tên hoặc theo ID
#
# recipe_id/loaded_by trong DB là ObjectId dạng chuỗi ('6a32b270aa6d85bb...').
# Nhưng user luôn gọi recipe/người dùng bằng TÊN, và LLM không có cách nào tra
# ra ObjectId. Trước đây truyền tên vào là query trả 0 bản ghi, agent báo
# "không có dữ liệu" — sai mà không hề có lỗi.
# ============================================================================

_OBJECT_ID = re.compile(r"^[0-9a-fA-F]{24}$")


def _id_or_name(value: str, id_field: str, name_field: str) -> Dict[str, Any]:
    """24 ký tự hex → lọc theo ID; còn lại → khớp tên, không phân biệt hoa thường."""
    if _OBJECT_ID.match(value.strip()):
        return {id_field: value.strip()}
    return {name_field: {"$regex": re.escape(value.strip()), "$options": "i"}}


def _matching_recipes(value: str, start_dt: datetime, end_dt: datetime) -> List[Dict[str, Any]]:
    """
    Các recipe khớp `value` VÀ có sản lượng trong đúng khoảng đang hỏi.

    Khớp theo tên là khớp một phần, nên một chuỗi ngắn có thể trúng nhiều
    recipe: 'minced onion' hiện trúng 5 recipe riêng biệt ('minced onion',
    'minced onion (Copy)', 'minced onion (Copy 2)', 'Minced Onion',
    'Minced Onion 1'). Trước đây tool cộng gộp cả 5 rồi trả về một con số duy
    nhất, user không hề biết mình đang xem số của 5 recipe trộn lại.

    Giới hạn theo đúng khoảng thời gian đang hỏi để không gợi ý những recipe
    vốn không có dữ liệu trong kỳ đó.
    """
    db = get_sync_database()
    match: Dict[str, Any] = {_TIME_FIELD: {"$gte": start_dt, "$lte": end_dt}}
    match.update(_id_or_name(value, "recipe_id", "recipe_name"))

    rows = db["inference_results"].aggregate([
        {"$match": match},
        {"$group": {
            "_id": {"id": "$recipe_id", "name": "$recipe_name"},
            "total": {"$sum": 1},
            "last": {"$max": "$created_at"},
        }},
        {"$sort": {"total": -1}},
    ])

    return [
        {
            "recipe_id": r["_id"]["id"],
            "recipe_name": r["_id"]["name"],
            "total": r["total"],
            "last_seen": _to_local_str(r["last"]),
        }
        for r in rows
    ]


def _disambiguation(value: str, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Payload báo cho LLM biết phải hỏi lại user thay vì đoán bừa."""
    return {
        "success": False,
        "needs_disambiguation": True,
        "message": (
            f"'{value}' khớp {len(matches)} recipe khác nhau. "
            f"KHÔNG được tự cộng gộp — hãy hỏi user chọn recipe nào."
        ),
        "matches": matches,
    }


# Tool argument schemas
class PassFailStatsArgs(BaseModel):
    """Arguments for get_pass_fail_stats"""
    recipe_id: Optional[str] = Field(None, description="Lọc theo recipe: truyền TÊN recipe (vd 'minced onion') hoặc ObjectId 24 ký tự. Tên khớp một phần, không phân biệt hoa thường.")
    start_date: Optional[str] = Field(None, description="Start date in ISO format YYYY-MM-DD (default: today)")
    end_date: Optional[str] = Field(None, description="End date in ISO format YYYY-MM-DD (default: today)")
    group_by: str = Field("day", description="Group results by 'hour', 'day', 'week', or 'month'")


class ProductionSummaryArgs(BaseModel):
    """Arguments for get_production_summary"""
    date: Optional[str] = Field(None, description="Một ngày cụ thể 'YYYY-MM-DD' (mặc định: hôm nay)")
    group_by: str = Field("recipe", description="Gom theo 'recipe', 'camera', hoặc 'hour'")
    start_date: Optional[str] = Field(None, description="Mốc đầu, KÈM GIỜ được: '2026-07-22T16:00:00'. Dùng khi user hỏi theo khung giờ.")
    end_date: Optional[str] = Field(None, description="Mốc cuối, kèm giờ được: '2026-07-22T18:00:00'")
    recipe_id: Optional[str] = Field(None, description="Chỉ tính một recipe: tên hoặc ObjectId 24 ký tự")


class ExplainFailuresArgs(BaseModel):
    """Arguments for explain_failures"""
    recipe_id: Optional[str] = Field(None, description="Lọc theo recipe: tên hoặc ObjectId 24 ký tự")
    camera: Optional[str] = Field(None, description="Lọc theo serial number camera, vd '24026290'")
    start_date: Optional[str] = Field(None, description="Mốc đầu, kèm giờ được: '2026-07-22T16:00:00'")
    end_date: Optional[str] = Field(None, description="Mốc cuối, kèm giờ được")
    sample_limit: int = Field(300, description="Số sản phẩm fail gần nhất đem mổ (mặc định 300)")


class ListRecipesArgs(BaseModel):
    """Arguments for list_recipes"""
    days: int = Field(7, description="Nhìn lại bao nhiêu ngày (mặc định 7)")
    name_filter: Optional[str] = Field(None, description="Chỉ lấy recipe có tên chứa chuỗi này")


class RecipeLoadHistoryArgs(BaseModel):
    """Arguments for get_recipe_load_history"""
    recipe_id: Optional[str] = Field(None, description="Lọc theo recipe: truyền TÊN recipe (vd 'minced onion') hoặc ObjectId 24 ký tự. Tên khớp một phần, không phân biệt hoa thường.")
    user_id: Optional[str] = Field(None, description="Lọc theo người dùng: truyền HỌ TÊN (vd 'System Administrator') hoặc ObjectId 24 ký tự.")
    limit: int = Field(10, description="Number of records to return (default: 10)")


def get_pass_fail_stats(
    recipe_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    group_by: str = "day"
) -> Dict[str, Any]:
    """
    Get pass/fail statistics for production data

    Args:
        recipe_id: Filter by specific recipe ID (optional)
        start_date: Start date in ISO format YYYY-MM-DD (default: today)
        end_date: End date in ISO format YYYY-MM-DD (default: today)
        group_by: Group results by 'hour', 'day', 'week', 'month' (default: day)

    Returns:
        Dictionary with pass/fail statistics and trends
    """
    try:
        db = get_sync_database()
        collection = db["inference_results"]

        # Mốc ngày user đưa là giờ địa phương; DB lưu naive-UTC.
        start_dt = _local_bound(start_date, end=False)
        end_dt = _local_bound(end_date, end=True)

        # Build query
        query = {
            _TIME_FIELD: {
                "$gte": start_dt,
                "$lte": end_dt
            }
        }

        if recipe_id:
            matches = _matching_recipes(recipe_id, start_dt, end_dt)
            if len(matches) > 1:
                return _disambiguation(recipe_id, matches)
            if matches:
                # Chốt bằng ID để không dính recipe khác trùng tiền tố.
                query["recipe_id"] = matches[0]["recipe_id"]
            else:
                # Không có recipe nào khớp trong kỳ này — cứ lọc như cũ để trả
                # về kết quả rỗng trung thực thay vì báo lỗi.
                query.update(_id_or_name(recipe_id, "recipe_id", "recipe_name"))

        # Aggregate statistics
        pipeline = [
            {"$match": query},
            {
                "$group": {
                    "_id": "$product_pass_fail",
                    "count": {"$sum": 1}
                }
            }
        ]

        results = list(collection.aggregate(pipeline))

        # Calculate totals
        pass_count = 0
        fail_count = 0

        for result in results:
            if result["_id"] == "PASS":
                pass_count = result["count"]
            elif result["_id"] == "FAIL":
                fail_count = result["count"]

        total_count = pass_count + fail_count
        pass_rate = (pass_count / total_count * 100) if total_count > 0 else 0
        fail_rate = (fail_count / total_count * 100) if total_count > 0 else 0

        # Get trend data (group by time)
        if group_by == "hour":
            group_format = "%Y-%m-%d %H:00"
        elif group_by == "day":
            group_format = "%Y-%m-%d"
        elif group_by == "week":
            group_format = "%Y-W%U"
        else:  # month
            group_format = "%Y-%m"

        trend_pipeline = [
            {"$match": query},
            {
                "$group": {
                    "_id": {
                        "time": {"$dateToString": {"format": group_format, "date": "$created_at",
                                                   "timezone": settings.TIMEZONE}},
                        "result": "$product_pass_fail"
                    },
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"_id.time": 1}}
        ]

        trend_results = list(collection.aggregate(trend_pipeline))

        # Format trend data
        trend_data = {}
        for item in trend_results:
            time_key = item["_id"]["time"]
            result_type = item["_id"]["result"]

            if time_key not in trend_data:
                trend_data[time_key] = {"pass": 0, "fail": 0}

            if result_type == "PASS":
                trend_data[time_key]["pass"] = item["count"]
            elif result_type == "FAIL":
                trend_data[time_key]["fail"] = item["count"]

        return {
            "success": True,
            "period": {
                "start": _to_local_str(start_dt),
                "end": _to_local_str(end_dt),
                "recipe_id": recipe_id or "all"
            },
            "summary": {
                "total_products": total_count,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "pass_rate": round(pass_rate, 2),
                "fail_rate": round(fail_rate, 2)
            },
            "trend": trend_data
        }

    except Exception as e:
        logger.error(f"Error getting pass/fail stats: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "message": "Không thể lấy thống kê pass/fail"
        }


def get_production_summary(
    date: Optional[str] = None,
    group_by: str = "recipe",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    recipe_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get production summary with breakdown by recipe / camera / hour.

    Args:
        date: Một ngày cụ thể 'YYYY-MM-DD' (mặc định: hôm nay)
        group_by: Gom theo 'recipe', 'camera', hoặc 'hour'
        start_date: Mốc đầu, có thể kèm giờ ('2026-07-22T16:00:00')
        end_date: Mốc cuối, có thể kèm giờ
        recipe_id: Chỉ tính một recipe (tên hoặc ObjectId)

    Returns:
        Production summary with breakdown
    """
    try:
        db = get_sync_database()
        collection = db["inference_results"]

        # start/end thắng `date` khi có — cần để trả lời "trong khung 16h-18h".
        # Thiếu hai tham số này thì mọi câu hỏi theo khung giờ đều âm thầm trả
        # số của TRỌN NGÀY.
        if start_date or end_date:
            start_dt = _local_bound(start_date or end_date, end=False)
            end_dt = _local_bound(end_date or start_date, end=True)
            target_date = datetime.fromisoformat((start_date or end_date))
        else:
            target_date = datetime.fromisoformat(date) if date else datetime.now(_TZ).replace(tzinfo=None)
            start_dt = _local_bound(date, end=False)
            end_dt = _local_bound(date, end=True)

        query = {
            _TIME_FIELD: {
                "$gte": start_dt,
                "$lte": end_dt
            }
        }

        if recipe_id:
            matches = _matching_recipes(recipe_id, start_dt, end_dt)
            if len(matches) > 1:
                return _disambiguation(recipe_id, matches)
            if matches:
                query["recipe_id"] = matches[0]["recipe_id"]
            else:
                query.update(_id_or_name(recipe_id, "recipe_id", "recipe_name"))

        # Build aggregation based on group_by
        if group_by == "recipe":
            group_field = "$recipe_name"
            group_key = "recipe"
        elif group_by == "hour":
            group_field = {"$hour": {"date": "$created_at", "timezone": settings.TIMEZONE}}
            group_key = "hour"
        elif group_by == "camera":
            # Verdict phải lấy TỪNG CAMERA (`camera_results.pass_fail`), không
            # phải verdict của cả sản phẩm (`product_pass_fail`). Dùng nhầm
            # product-level thì mọi camera trên một sản phẩm fail đều bị tính
            # là fail, kể cả camera chụp đạt.
            #
            # Đo trên dữ liệu thật 1 ngày: cách cũ báo 3 camera fail xấp xỉ
            # nhau (10.942 / 10.936 / 10.936) trong khi thực tế là
            # 10.902 / 64 / 64 — tức là che mất đúng camera đang hỏng.
            pipeline = [
                {"$match": query},
                {"$unwind": "$camera_results"},
                {
                    "$group": {
                        "_id": "$camera_results.serial_number",
                        "total": {"$sum": 1},
                        "pass": {
                            "$sum": {
                                "$cond": [{"$eq": ["$camera_results.pass_fail", "PASS"]}, 1, 0]
                            }
                        },
                        "fail": {
                            "$sum": {
                                "$cond": [{"$eq": ["$camera_results.pass_fail", "FAIL"]}, 1, 0]
                            }
                        }
                    }
                },
                {"$sort": {"_id": 1}}
            ]

            results = list(collection.aggregate(pipeline))

            breakdown = []
            for item in results:
                total = item["total"]
                pass_count = item["pass"]
                fail_count = item["fail"]

                breakdown.append({
                    "camera": item["_id"],
                    "total": total,
                    "pass": pass_count,
                    "fail": fail_count,
                    "pass_rate": round(pass_count / total * 100, 2) if total > 0 else 0
                })

            # Get overall total
            total_count = collection.count_documents(query)
            pass_pipeline = [
                {"$match": {**query, "product_pass_fail": "PASS"}},
                {"$count": "pass_count"}
            ]
            pass_result = list(collection.aggregate(pass_pipeline))[:1]
            pass_count = pass_result[0]["pass_count"] if pass_result else 0
            fail_count = total_count - pass_count

            return {
                "success": True,
                "date": target_date.strftime("%Y-%m-%d"),
                "group_by": group_by,
                "summary": {
                    "total_products": total_count,
                    "pass_count": pass_count,
                    "fail_count": fail_count,
                    "pass_rate": round(pass_count / total_count * 100, 2) if total_count > 0 else 0
                },
                "breakdown": breakdown
            }
        else:
            group_field = "$recipe_name"
            group_key = "recipe"

        # Standard aggregation for recipe/hour
        pipeline = [
            {"$match": query},
            {
                "$group": {
                    "_id": group_field,
                    "total": {"$sum": 1},
                    "pass": {
                        "$sum": {
                            "$cond": [{"$eq": ["$product_pass_fail", "PASS"]}, 1, 0]
                        }
                    },
                    "fail": {
                        "$sum": {
                            "$cond": [{"$eq": ["$product_pass_fail", "FAIL"]}, 1, 0]
                        }
                    }
                }
            },
            {"$sort": {"_id": 1}}
        ]

        results = list(collection.aggregate(pipeline))

        breakdown = []
        total_all = 0
        pass_all = 0
        fail_all = 0

        for item in results:
            total = item["total"]
            pass_count = item["pass"]
            fail_count = item["fail"]

            total_all += total
            pass_all += pass_count
            fail_all += fail_count

            breakdown.append({
                group_key: item["_id"],
                "total": total,
                "pass": pass_count,
                "fail": fail_count,
                "pass_rate": round(pass_count / total * 100, 2) if total > 0 else 0
            })

        return {
            "success": True,
            "date": target_date.strftime("%Y-%m-%d"),
            "group_by": group_by,
            "summary": {
                "total_products": total_all,
                "pass_count": pass_all,
                "fail_count": fail_all,
                "pass_rate": round(pass_all / total_all * 100, 2) if total_all > 0 else 0
            },
            "breakdown": breakdown
        }

    except Exception as e:
        logger.error(f"Error getting production summary: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Không thể lấy thống kê sản xuất"
        }


def get_recipe_load_history(
    recipe_id: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Get recipe load/stop history

    Args:
        recipe_id: Filter by specific recipe ID (optional)
        user_id: Filter by specific user ID (optional)
        limit: Maximum number of records to return (default: 10)

    Returns:
        Recipe load history with user information
    """
    try:
        db = get_sync_database()
        # Collection thật tên là 'receipt_loads' (xem
        # backend/app/repositories/receipt_load_repository.py) — không phải
        # 'recipe_loads'. Query sai tên chỉ trả collection rỗng chứ không lỗi,
        # nên tool im lặng báo "không có dữ liệu" dù DB có 2.2k bản ghi.
        collection = db["receipt_loads"]

        # Build query. Lưu ý receipt_loads KHÔNG có field 'recipe_name' —
        # tên recipe nằm trong metadata.name.
        query = {}
        if recipe_id:
            query.update(_id_or_name(recipe_id, "recipe_id", "metadata.name"))
        if user_id:
            query.update(_id_or_name(user_id, "loaded_by", "loaded_by_full_name"))

        # Get records
        cursor = collection.find(query).sort("loaded_at", -1).limit(limit)
        records = list(cursor)

        # Format results
        history = []
        for record in records:
            # Calculate runtime if still running
            runtime_seconds = None
            if record["status"] == "running":
                # loaded_at lưu naive-UTC ⇒ phải so với utcnow(), dùng now() sẽ
                # phồng runtime đúng bằng offset múi giờ (+7h ở VN).
                runtime_seconds = (datetime.utcnow() - record["loaded_at"]).total_seconds()
            elif record.get("stopped_at"):
                runtime_seconds = (record["stopped_at"] - record["loaded_at"]).total_seconds()

            runtime_str = None
            if runtime_seconds:
                hours = int(runtime_seconds // 3600)
                minutes = int((runtime_seconds % 3600) // 60)
                runtime_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"

            # Mọi mốc thời gian trả cho LLM đều là giờ địa phương. Trước đây
            # trả kèm cả bản UTC thô, LLM chọn nhầm rồi báo "09:30 (UTC)" cho
            # user Việt Nam.
            history.append({
                "recipe_id": record["recipe_id"],
                "recipe_name": record["metadata"].get("name", "Unknown"),
                "loaded_by": record["loaded_by_full_name"],
                "loaded_at": _to_local_str(record["loaded_at"]),
                "status": record["status"],
                "stopped_by": record.get("stopped_by_full_name"),
                "stopped_at": _to_local_str(record["stopped_at"]) if record.get("stopped_at") else None,
                "runtime": runtime_str,
                "timezone": settings.TIMEZONE,
            })

        # Get statistics if filtering by recipe
        stats = None
        if recipe_id and history:
            # Count loads by user
            user_counts = {}
            for h in history:
                user = h["loaded_by"]
                user_counts[user] = user_counts.get(user, 0) + 1

            stats = {
                "total_loads": len(history),
                "by_user": user_counts,
                "current_status": history[0]["status"] if history else None
            }

        return {
            "success": True,
            "filters": {
                "recipe_id": recipe_id or "all",
                "user_id": user_id or "all",
                "limit": limit
            },
            "history": history,
            "statistics": stats
        }

    except Exception as e:
        logger.error(f"Error getting recipe load history: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Không thể lấy lịch sử load recipe"
        }


# ============================================================================
# Phân loại một cặp expected → recognized
#
# Đo trên 101 cặp sai của ngày 19/08: 85% là chuỗi đọc được nằm gọn bên trong
# chuỗi mong đợi (cắt đầu 60, cắt đuôi 16, rỗng hẳn 9) — tức camera chỉ nhìn
# thấy một phần nhãn, không phải OCR đọc nhầm ký tự. Nhóm này đi kèm template
# similarity trung bình 0,08–0,44 (ngưỡng đạt là 0,50), xác nhận thùng bị lệch
# khỏi khung. Nhóm sai ký tự thật chỉ có 5 cặp và similarity trung bình 0,53,
# tức thùng nằm đúng chỗ.
#
# Tách được hai nhóm này thì câu trả lời mới đúng việc cần làm: chỉnh cơ khí /
# trigger, hay chỉnh model OCR.
# ============================================================================

def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _mismatch_kind(expected: str, recognized: str) -> str:
    """Nhãn ngắn gọn cho một cặp sai, dùng để gộp nhóm trong báo cáo."""
    if not recognized:
        return "khong_doc_duoc"
    if recognized == expected:
        return "khop"
    # Bỏ khoảng trắng rồi mới so: OCR hay chèn/bỏ space giữa các từ
    # ('BESTifUsedbyAUG182028' → 'BEST if Used by AUG'), đó là chuyện tách từ
    # chứ không phải nhìn thiếu nhãn.
    e_c, r_c = expected.replace(" ", ""), recognized.replace(" ", "")
    if r_c and r_c in e_c and r_c != e_c:
        return "doc_thieu_dau_cuoi"
    if e_c == r_c:
        return "chi_khac_khoang_trang"
    if e_c in r_c:
        return "doc_thua_ky_tu"
    d = _levenshtein(expected, recognized)
    if d <= 2:
        return "sai_it_ky_tu"
    return "khac_han"


_CAUSE_LABELS = {
    "text_verification": "OCR đọc sai chuỗi",
    "char_verification": "Ký tự dưới ngưỡng tin cậy",
    "template_verification": "Ảnh không khớp template",
    "product_verification": "Không nhận ra sản phẩm",
    "no_detection": "Detector không thấy vùng nào trong khung",
    "unknown": "Chưa xác định",
}


_MISMATCH_LABELS = {
    "khong_doc_duoc": "không đọc được ký tự nào (nhãn ngoài khung hoặc bị che)",
    "doc_thieu_dau_cuoi": "chỉ đọc được một phần nhãn (thùng lệch / vào khung chưa đủ)",
    "chi_khac_khoang_trang": "đúng chữ, chỉ khác khoảng trắng (lỗi tách từ của OCR)",
    "doc_thua_ky_tu": "đọc dư ký tự so với nhãn",
    "sai_it_ky_tu": "sai 1–2 ký tự — lỗi nhận dạng thật",
    "khac_han": "khác hẳn nhãn mong đợi (kiểm tra xem có đang chạy lô khác không)",
}


def explain_failures(
    recipe_id: Optional[str] = None,
    camera: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sample_limit: int = 300,
) -> Dict[str, Any]:
    """
    Vì sao sản phẩm bị fail — thống kê nguyên nhân, không chỉ đếm số lượng.

    Mỗi frame đi qua 4 bước kiểm tra; bước nào không đạt thì đó là nguyên nhân:
      - text_verification     : OCR đọc ra chuỗi khác chuỗi mong đợi
      - char_verification     : ký tự riêng lẻ dưới ngưỡng tin cậy
      - template_verification : ảnh không khớp template (similarity < threshold)
      - product_verification  : không nhận ra sản phẩm/nhãn

    Args:
        recipe_id: Lọc theo recipe (tên hoặc ObjectId)
        camera: Lọc theo serial number của camera, vd '24026290'
        start_date / end_date: ISO, có thể kèm giờ ('2026-07-22T16:00:00')
        sample_limit: Số bản ghi fail lấy về để mổ (mặc định 300)

    Returns:
        Số fail theo từng nguyên nhân + các cặp expected/recognized sai nhiều nhất
    """
    try:
        db = get_sync_database()
        start_dt = _local_bound(start_date, end=False)
        end_dt = _local_bound(end_date, end=True)

        query: Dict[str, Any] = {
            _TIME_FIELD: {"$gte": start_dt, "$lte": end_dt},
            "product_pass_fail": "FAIL",
        }

        if recipe_id:
            matches = _matching_recipes(recipe_id, start_dt, end_dt)
            if len(matches) > 1:
                return _disambiguation(recipe_id, matches)
            if matches:
                query["recipe_id"] = matches[0]["recipe_id"]
            else:
                query.update(_id_or_name(recipe_id, "recipe_id", "recipe_name"))

        total_fail = db["inference_results"].count_documents(query)
        docs = db["inference_results"].find(query, _FAIL_PROJECTION).sort(_TIME_FIELD, -1).limit(sample_limit)

        _CAUSE_KEYS = (
            "text_verification",
            "char_verification",
            "template_verification",
            "product_verification",
            "no_detection",
            "unknown",
        )
        causes = {k: 0 for k in _CAUSE_KEYS}          # đếm theo FRAME
        causes_products = {k: set() for k in _CAUSE_KEYS}   # đếm theo SẢN PHẨM
        per_camera: Dict[str, int] = {}
        mismatches: Dict[tuple, int] = {}
        mismatch_kinds: Dict[str, int] = {}
        sims: List[float] = []
        samples: List[Dict[str, Any]] = []
        examined = 0

        for doc in docs:
            for cam in doc.get("camera_results") or []:
                serial = str(cam.get("serial_number"))
                if camera and serial != str(camera):
                    continue
                if cam.get("pass_fail") != "FAIL":
                    continue

                per_camera[serial] = per_camera.get(serial, 0) + 1
                examined += 1

                for frame in cam.get("frames") or []:
                    if frame.get("pass_fail") != "FAIL":
                        continue
                    def _mark(key: str) -> None:
                        causes[key] += 1
                        causes_products[key].add(doc["_id"])

                    hit = False

                    tv = frame.get("text_verification") or {}
                    if tv.get("all_match") is False:
                        _mark("text_verification")
                        hit = True
                        for r in tv.get("results") or []:
                            if not r.get("match"):
                                expected = str(r.get("expected"))
                                recognized = str(r.get("recognized"))
                                key = (expected, recognized)
                                mismatches[key] = mismatches.get(key, 0) + 1
                                kind = _mismatch_kind(expected, recognized)
                                mismatch_kinds[kind] = mismatch_kinds.get(kind, 0) + 1

                    cv = frame.get("char_verification") or {}
                    if cv.get("all_match") is False:
                        _mark("char_verification")
                        hit = True

                    tpl = frame.get("template_verification") or {}
                    if tpl.get("match") is False:
                        _mark("template_verification")
                        hit = True
                        if isinstance(tpl.get("similarity"), (int, float)):
                            sims.append(tpl["similarity"])

                    pv = frame.get("product_verification") or {}
                    if pv.get("match") is False and not pv.get("skipped"):
                        _mark("product_verification")
                        hit = True

                    if not hit:
                        # Frame FAIL mà không bước kiểm tra nào báo sai: detector
                        # không tìm thấy vùng nào nên pipeline dừng ngay sau khi
                        # detect — cả 4 block verification đều null và `timings`
                        # không có `ocr_ms`. Đây KHÔNG phải "chưa rõ nguyên nhân",
                        # mà là "camera không nhìn thấy nhãn": thùng lệch, che
                        # khuất, hoặc trigger sai thời điểm. Gộp chung vào
                        # `unknown` như trước khiến nguyên nhân phổ biến thứ nhì
                        # bị giấu sau một cái nhãn vô nghĩa.
                        if not (frame.get("detected_regions") or []):
                            _mark("no_detection")
                        else:
                            _mark("unknown")

                    # Ảnh visualize của frame fail — để FE hiện kèm câu trả lời.
                    # Giữ tối đa 8 cái, đủ để nhìn ra quy luật mà không nặng UI.
                    if frame.get("image_path") and len(samples) < 8:
                        bad = next(
                            (r for r in (tv.get("results") or []) if not r.get("match")),
                            None,
                        )
                        samples.append({
                            "image_path": frame["image_path"],
                            "camera": serial,
                            "timestamp": _to_local_str(doc["timestamp"]),
                            "recipe_name": doc.get("recipe_name"),
                            "confidence": frame.get("confidence"),
                            "expected": (bad or {}).get("expected"),
                            "recognized": (bad or {}).get("recognized"),
                        })

        top_mismatch = [
            {"expected": e, "recognized": r, "count": n}
            for (e, r), n in sorted(mismatches.items(), key=lambda x: -x[1])[:10]
        ]

        mismatch_summary = [
            {"kind": k, "label": _MISMATCH_LABELS.get(k, k), "count": n}
            for k, n in sorted(mismatch_kinds.items(), key=lambda x: -x[1])
        ]

        return {
            "success": True,
            "period": {"start": _to_local_str(start_dt), "end": _to_local_str(end_dt)},
            "filters": {"recipe_id": recipe_id or "all", "camera": camera or "all"},
            "total_failed_products": total_fail,
            "examined_products": min(total_fail, sample_limit),
            "failed_camera_frames_examined": examined,
            # Một sản phẩm có nhiều frame và một frame trượt được nhiều bước, nên
            # hai cách đếm ra hai con số khác nhau (113 frame nhưng 112 sản phẩm).
            # Để hai con số cạnh nhau trong CÙNG một hàng, kèm tên trường nói rõ
            # đơn vị: tách thành hai dict riêng thì LLM vẫn bốc nhầm số frame rồi
            # gọi là "sản phẩm", vì ở chỗ đó không còn gì nhắc nó đơn vị nào.
            "causes": [
                {
                    "cause": k,
                    "label": _CAUSE_LABELS.get(k, k),
                    "products": len(causes_products[k]),
                    "frames": causes[k],
                }
                for k in sorted(causes, key=lambda x: -len(causes_products[x]))
                if causes[k]
            ],
            "causes_by_product": {k: len(v) for k, v in causes_products.items() if v},
            "failed_frames_by_camera": dict(sorted(per_camera.items(), key=lambda x: -x[1])),
            "top_text_mismatches": top_mismatch,
            "text_mismatch_kinds": mismatch_summary,
            "template_similarity_avg": round(sum(sims) / len(sims), 4) if sims else None,
            "samples": samples,
            "note": (
                f"Mổ {min(total_fail, sample_limit)} sản phẩm fail gần nhất trên tổng "
                f"{total_fail}. Mỗi hàng trong `causes` có hai con số: `products` "
                f"là số SẢN PHẨM, `frames` là số FRAME — khi trả lời user hãy dùng "
                f"`products`. Tổng các hàng lớn hơn tổng sản phẩm fail là bình "
                f"thường: một sản phẩm trượt được nhiều bước cùng lúc. "
                f"`no_detection` = detector không tìm thấy vùng nào trong frame "
                f"(thùng lệch/che khuất/trigger sai), khác với `unknown`. "
                f"Xem `text_mismatch_kinds` để biết lỗi OCR là do nhìn thiếu nhãn "
                f"hay do nhận dạng sai thật."
            ),
        }

    except Exception as e:
        logger.error(f"Error explaining failures: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e), "message": "Không thể phân tích nguyên nhân fail"}


def list_recipes(days: int = 7, name_filter: Optional[str] = None) -> Dict[str, Any]:
    """
    Liệt kê các recipe CÓ SẢN LƯỢNG trong N ngày gần nhất, kèm số sản phẩm.

    Dùng khi user hỏi thống kê theo recipe nhưng chưa nói rõ recipe nào — thay
    vì đoán, agent gọi tool này rồi đưa danh sách cho user chọn.

    Args:
        days: Nhìn lại bao nhiêu ngày (mặc định 7)
        name_filter: Chỉ lấy recipe có tên chứa chuỗi này (không bắt buộc)

    Returns:
        Danh sách recipe, sắp xếp theo sản lượng giảm dần
    """
    try:
        end_dt = _local_bound(None, end=True)
        start_dt = end_dt - timedelta(days=days)
        recipes = _matching_recipes(name_filter or "", start_dt, end_dt)

        return {
            "success": True,
            "period_days": days,
            "count": len(recipes),
            "recipes": recipes,
            "message": (
                f"Có {len(recipes)} recipe chạy trong {days} ngày qua."
                if recipes else
                f"Không có recipe nào có sản lượng trong {days} ngày qua."
            ),
        }
    except Exception as e:
        logger.error(f"Error listing recipes: {e}")
        return {"success": False, "error": str(e), "message": "Không thể liệt kê recipe"}


# Register tools
list_recipes_tool = BaseTool.create_tool(
    func=list_recipes,
    metadata=ToolMetadata(
        name="list_recipes",
        description=(
            "Liệt kê các recipe đang có sản lượng, kèm số sản phẩm. "
            "DÙNG KHI user hỏi thống kê theo recipe nhưng chưa nói rõ recipe nào — "
            "gọi tool này rồi đưa danh sách cho user chọn, TUYỆT ĐỐI không tự đoán."
        ),
        category="analytics"
    ),
    args_schema=ListRecipesArgs
)

explain_failures_tool = BaseTool.create_tool(
    func=explain_failures,
    metadata=ToolMetadata(
        name="explain_failures",
        description=(
            "Phân tích NGUYÊN NHÂN sản phẩm bị fail — dùng khi user hỏi 'tại sao', "
            "'vì sao fail', 'lỗi gì', 'nguyên nhân'. Trả về số fail theo từng bước "
            "kiểm tra (OCR đọc sai chuỗi / ký tự dưới ngưỡng / ảnh không khớp "
            "template / không nhận ra sản phẩm / không thấy nhãn trong khung), "
            "kèm các cặp expected-recognized sai nhiều nhất. "
            "QUAN TRỌNG khi diễn giải: mỗi hàng trong `causes` có `products` "
            "(số SẢN PHẨM) và `frames` (số FRAME) — trả lời user bằng `products`. "
            "Tổng các hàng lớn hơn tổng sản phẩm fail là bình thường vì một sản "
            "phẩm trượt được nhiều bước cùng lúc. Trường "
            "`text_mismatch_kinds` cho biết lỗi OCR là do camera chỉ nhìn thấy "
            "một phần nhãn (thùng lệch) hay do nhận dạng sai thật — hãy nêu rõ "
            "điều này thay vì mặc định quy cho chất lượng ảnh. "
            "Lọc được theo recipe, camera và khung giờ."
        ),
        category="analytics"
    ),
    args_schema=ExplainFailuresArgs
)

get_pass_fail_stats_tool = BaseTool.create_tool(
    func=get_pass_fail_stats,
    metadata=ToolMetadata(
        name="get_pass_fail_stats",
        description="Lấy thống kê pass/fail theo recipe và khoảng thời gian. Hỗ trợ phân tích xu hướng theo giờ/ngày/tuần/tháng.",
        category="analytics"
    ),
    args_schema=PassFailStatsArgs
)

get_production_summary_tool = BaseTool.create_tool(
    func=get_production_summary,
    metadata=ToolMetadata(
        name="get_production_summary",
        description=(
            "Tổng quan sản xuất, gom theo recipe/camera/giờ. Lọc được theo KHUNG GIỜ "
            "(start_date/end_date kèm giờ) và theo recipe. Dùng cho câu hỏi kiểu "
            "'camera nào fail nhiều nhất trong khung 16h-18h'."
        ),
        category="analytics"
    ),
    args_schema=ProductionSummaryArgs
)

get_recipe_load_history_tool = BaseTool.create_tool(
    func=get_recipe_load_history,
    metadata=ToolMetadata(
        name="get_recipe_load_history",
        description="Xem lịch sử load/stop recipe, bao gồm thông tin user, thời gian chạy. Dùng để theo dõi ai làm gì với recipe.",
        category="analytics"
    ),
    args_schema=RecipeLoadHistoryArgs
)


# ── So sánh hai kỳ ───────────────────────────────────────────────────────────
#
# Vì sao cần một tool riêng thay vì để LLM gọi get_pass_fail_stats hai lần rồi
# tự trừ: chênh lệch pass rate là phép tính trên phần trăm, và LLM làm số học
# kiểu đó rất hay sai — sai theo hướng nghe hợp lý. Tệ hơn, "tăng 1,2 điểm phần
# trăm" và "tăng 1,2%" là hai con số khác nhau mà LLM thường dùng lẫn. Tính tất
# định ở đây thì câu trả lời không thể lệch khỏi dữ liệu.

#: Nhãn kỳ bằng tiếng Việt. `PERIOD_LABELS` của report_tools là tiếng Anh và đi
#: thẳng vào tiêu đề bảng, phụ đề ô KPI — tức người vận hành đọc "so với
#: Yesterday" giữa một câu tiếng Việt. Khoá vẫn giữ tiếng Anh cho phần bên trong.
PERIOD_VI = {
    "today": "hôm nay", "yesterday": "hôm qua",
    "thisweek": "tuần này", "lastweek": "tuần trước",
    "7days": "7 ngày qua", "thismonth": "tháng này",
    "lastmonth": "tháng trước", "30days": "30 ngày qua",
}

#: Số bản ghi tối thiểu để một kỳ được coi là có nền so sánh.
#:
#: Dưới ngưỡng này thì mọi tỷ lệ đều là nhiễu. Đo trên dữ liệu thật: một recipe
#: có 2 bản ghi ở kỳ trước (cả 2 đều fail) và 66.199 bản ghi ở kỳ này cho ra
#: "+3.309.850%" và "+97,79 điểm pass rate" — hai con số vô nghĩa, và tệ hơn là
#: chúng trông như lỗi hiển thị nên người xem mất tin vào cả bảng.
MIN_BASELINE = 30


class ComparePeriodsArgs(BaseModel):
    """Arguments for compare_periods"""
    period_a: Optional[str] = Field(
        default=None,
        description="Kỳ CẦN ĐÁNH GIÁ: today, yesterday, thisweek, lastweek, 7days, "
                    "thismonth, lastmonth, 30days. Bỏ trống = today.",
    )
    period_b: Optional[str] = Field(
        default=None,
        description="Kỳ ĐỐI CHIẾU. Bỏ trống = kỳ liền trước period_a có cùng độ dài "
                    "(khuyến nghị bỏ trống, vì hai kỳ cùng độ dài mới so được).",
    )
    recipe_id: Optional[str] = Field(default=None, description="Chỉ so sánh một recipe: tên hoặc ObjectId")


def _period_bounds(period: Optional[str]) -> tuple:
    """
    (start, end, nhãn tiếng Việt) — dùng lại bảng kỳ của report_tools.

    Tên kỳ lạ phải BÁO LỖI, không được im lặng rơi về "today": `resolve_period`
    mặc định là hôm nay, nên `compare_periods(period_a='2019')` từng trả về
    success kèm nhãn "Today" — user hỏi năm 2019 mà nhận số của hôm nay, không có
    dấu hiệu gì cho biết câu hỏi đã bị hiểu khác.
    """
    from agent_app.reports.data import PERIOD_LABELS, resolve_period

    p = (period or "today").lower()
    if p not in PERIOD_LABELS:
        raise ValueError(
            f"Kỳ '{period}' không hợp lệ. Hợp lệ: {', '.join(PERIOD_LABELS)}. "
            f"Cần khoảng ngày cụ thể thì dùng get_pass_fail_stats với start_date/end_date."
        )
    start, end, _ = resolve_period(p, None, None)
    return start, end, PERIOD_VI[p]


def _shift_back(start: datetime, end: datetime) -> tuple:
    """
    Kỳ liền trước, cùng độ dài, KẾ TIẾP NGAY TRƯỚC `start`.

    Lùi đúng bằng độ dài kỳ chứ không lùi cứng một ngày: "7 ngày qua" phải đối
    chiếu với 7 ngày trước đó, không phải với hôm qua.

    Mốc cuối là `start` trừ 1 micro-giây. Trừ thẳng span cho cả hai đầu thì cửa
    sổ trước lại BAO GỒM đúng mốc `start` — tức micro-giây đầu tiên của kỳ hiện
    tại bị tính sang kỳ trước, đồng thời bỏ mất micro-giây đầu của kỳ trước.
    """
    span = end - start
    prev_end = start - timedelta(microseconds=1)
    return prev_end - span, prev_end


def _span_days(start: datetime, end: datetime) -> int:
    return max(1, round((end - start).total_seconds() / 86400))


def _delta(a: Optional[float], b: Optional[float], *, comparable: bool = True) -> Dict[str, Any]:
    """
    Chênh lệch a so với b, kèm cả hai cách đọc.

    `diff` là hiệu tuyệt đối — với pass rate thì đây chính là "điểm phần trăm".
    `change_pct` là thay đổi tương đối. Hai con số khác nhau, nên trả cả hai và
    gọi tên rõ; `kind` cho lớp hiển thị biết đơn vị nào đang dùng.

    `comparable=False` khi kỳ đối chiếu không có nền (rỗng, hoặc quá ít bản ghi):
    khi đó giữ nguyên giá trị thực nhưng BỎ HẲN mọi con số chênh lệch. Để nguyên
    phép trừ sẽ ra "+98 điểm" từ một nền bằng 0 — đúng số học mà đọc thành "cải
    thiện vượt bậc", trong khi kỳ kia đơn giản là không chạy.
    """
    out: Dict[str, Any] = {"current": a, "previous": b,
                           "diff": None, "change_pct": None, "direction": "n/a"}
    if not comparable or a is None or b is None:
        return out
    diff = round(a - b, 2)
    out["diff"] = diff
    out["change_pct"] = round((a - b) / b * 100, 2) if b else None
    out["direction"] = "up" if diff > 0 else "down" if diff < 0 else "flat"
    return out


def compare_periods(
    period_a: Optional[str] = None,
    period_b: Optional[str] = None,
    recipe_id: Optional[str] = None,
    **_ignored: Any,
) -> Dict[str, Any]:
    """So sánh sản lượng và chất lượng giữa hai kỳ."""
    try:
        from agent_app.reports.data import build_summary

        start_a, end_a, label_a = _period_bounds(period_a)
        if period_b:
            start_b, end_b, label_b = _period_bounds(period_b)
        else:
            start_b, end_b = _shift_back(start_a, end_a)
            # Nhãn phải mang khoảng ngày: "kỳ liền trước" một mình không cho biết
            # là 7 ngày nào, mà nhãn này đi thẳng vào tiêu đề bảng.
            lo, hi = _to_local_str(start_b)[:10], _to_local_str(end_b)[:10]
            label_b = lo if lo == hi else f"{lo} → {hi}"

        # So một kỳ với chính nó thì mọi chênh lệch bằng 0, và câu trả lời thành
        # "không có gì thay đổi" — một kết luận về sản xuất rút ra từ một phép so
        # sánh chưa từng diễn ra. Đã gặp thật: user hỏi "tháng trước so với tháng
        # trước nữa", LLM không có từ vựng cho "tháng trước nữa" nên điền
        # lastmonth cho cả hai.
        if (start_a, end_a) == (start_b, end_b):
            return {
                "success": False,
                "error": (f"Hai kỳ trùng khít nhau ({_to_local_str(start_a)[:10]} → "
                          f"{_to_local_str(end_a)[:10]}), không so sánh được. "
                          f"Chỉ có 8 kỳ dựng sẵn ({', '.join(PERIOD_VI.values())}); "
                          f"cần khoảng khác thì bỏ trống period_b để lấy kỳ liền trước, "
                          f"hoặc dùng get_pass_fail_stats với start_date/end_date."),
            }

        days_a, days_b = _span_days(start_a, end_a), _span_days(start_b, end_b)

        recipe_ids = None
        if recipe_id:
            matches = _matching_recipes(recipe_id, min(start_a, start_b), max(end_a, end_b))
            if len(matches) > 1:
                return _disambiguation(recipe_id, matches)
            if not matches:
                return {"success": False,
                        "error": f"Không có recipe nào khớp '{recipe_id}' trong hai kỳ này"}
            recipe_ids = [matches[0]["recipe_id"]]

        a = build_summary(start_a, end_a, recipe_ids)
        b = build_summary(start_b, end_b, recipe_ids)

        if a["total"] == 0 and b["total"] == 0:
            return {"success": False,
                    "error": f"Cả hai kỳ đều không có dữ liệu ({label_a} và {label_b})"}

        # Nền đủ lớn thì mới so; không thì chỉ trình bày giá trị thực.
        baseline_ok = b["total"] >= MIN_BASELINE
        baseline_note = None
        if b["total"] == 0:
            baseline_note = f"Kỳ đối chiếu ({label_b}) KHÔNG CÓ dữ liệu — không so sánh được."
        elif not baseline_ok:
            baseline_note = (f"Kỳ đối chiếu ({label_b}) chỉ có {b['total']} sản phẩm, "
                             f"dưới {MIN_BASELINE} — quá ít để làm nền so sánh.")

        # Recipe gom theo recipe_id, không theo tên: `build_summary` nhóm theo
        # (recipe_id, recipe_name) nên hai recipe trùng tên sẽ đè nhau nếu khoá
        # bằng tên, và bảng thôi không cộng đủ về tổng.
        by_id_a = {r["recipe_id"]: r for r in a["by_recipe"]}
        by_id_b = {r["recipe_id"]: r for r in b["by_recipe"]}
        recipes = []
        for rid in set(by_id_a) | set(by_id_b):
            ra, rb = by_id_a.get(rid), by_id_b.get(rid)
            only_in = None if (ra and rb) else (label_a if ra else label_b)
            # Recipe chỉ chạy một kỳ: cả sản lượng lẫn pass rate đều không có nền.
            # Trước đây chỉ pass_rate bị bỏ, còn sản lượng vẫn ra "-100.0%" —
            # một recipe không được lên lịch đọc thành sụp sản lượng hoàn toàn.
            # Và giá trị vắng mặt trả None chứ không phải 0, để số 0 in ra luôn
            # có nghĩa là "đo được và bằng 0" — nếu không, một recipe 18 sản phẩm
            # fail 100% sẽ trông y như một ô không có dữ liệu.
            comparable = baseline_ok and not only_in and (rb or {}).get("total", 0) >= MIN_BASELINE
            recipes.append({
                "recipe_id": rid,
                "recipe_name": (ra or rb)["recipe_name"],
                "total": _delta(ra["total"] if ra else None,
                                rb["total"] if rb else None, comparable=comparable),
                "pass_rate": _delta(ra["pass_rate"] if ra else None,
                                    rb["pass_rate"] if rb else None, comparable=comparable),
                "only_in": only_in,
                # Nền quá ít nhưng vẫn có: nói rõ vì sao không có chênh lệch.
                "baseline_too_small": (not only_in and 0 < (rb or {}).get("total", 0) < MIN_BASELINE),
                # 100% fail trên một lô nhỏ là sự cố thật, đừng để nó lẫn vào các
                # hàng "không có dữ liệu".
                "all_failed": bool(ra and ra["total"] > 0 and ra["pass_rate"] == 0.0),
            })

        # Sắp theo sản lượng LỚN NHẤT của hai kỳ, tên làm khoá phụ.
        #
        # Khoá cũ chỉ dùng sản lượng kỳ A nên recipe vắng ở A đều bằng 0, xếp
        # xuống cuối và thứ tự giữa chúng phụ thuộc thứ tự băm của set — cùng câu
        # hỏi, mỗi lần chạy ra một bảng khác. Mà chính những hàng đó lại là thay
        # đổi lớn nhất: một recipe 34.736 sản phẩm biến mất bị đẩy xuống dòng cuối.
        recipes.sort(key=lambda r: (-max(r["total"]["current"] or 0, r["total"]["previous"] or 0),
                                    r["recipe_name"]))

        return {
            "success": True,
            "period_a": {"label": label_a, "days": days_a,
                         "start": _to_local_str(start_a), "end": _to_local_str(end_a)},
            "period_b": {"label": label_b, "days": days_b,
                         "start": _to_local_str(start_b), "end": _to_local_str(end_b)},
            # Hai kỳ dài khác nhau thì so tổng là vô nghĩa: tháng này (19 ngày đã
            # qua) so tháng trước (31 ngày) cho ra "+59,75%" trong khi bình quân
            # mỗi ngày thực tế +160%. Cờ này để lớp trả lời buộc phải nói ra.
            "same_length": days_a == days_b,
            "per_day": {
                "current": round(a["total"] / days_a, 1),
                "previous": round(b["total"] / days_b, 1),
            },
            "baseline_usable": baseline_ok,
            "baseline_note": baseline_note,
            "recipe_scope": recipe_id or "tất cả recipe",
            "total": _delta(a["total"], b["total"], comparable=baseline_ok),
            "pass": _delta(a["pass"], b["pass"], comparable=baseline_ok),
            "fail": _delta(a["fail"], b["fail"], comparable=baseline_ok),
            "pass_rate": _delta(a["pass_rate"], b["pass_rate"], comparable=baseline_ok),
            "by_recipe": recipes,
            "note": (
                "Với `pass_rate`, `diff` là chênh lệch ĐIỂM PHẦN TRĂM còn `change_pct` "
                "là thay đổi tương đối — đừng dùng lẫn. 98% so với 96% là +2 điểm phần "
                "trăm, tức +2,08% tương đối. "
                "`diff`/`change_pct` bằng null nghĩa là KHÔNG có nền để so (kỳ đối chiếu "
                "rỗng, quá ít bản ghi, hoặc recipe chỉ chạy một kỳ) — đừng tự tính hiệu. "
                "Với recipe có `only_in`: cách diễn đạt DUY NHẤT được phép là 'không có "
                "bản ghi trong kỳ <tên kỳ>'. Cấm các cụm 'recipe mới', 'mới xuất hiện', "
                "'đã ngừng chạy', 'bị dừng', 'ngừng sản xuất' — kể cả trong tiêu đề mục. "
                "Trong một cửa sổ vài ngày đó chỉ là luân phiên sản phẩm bình thường, và "
                "ONION POWDER vắng ở kỳ trước KHÔNG có nghĩa nó là recipe mới — nó vẫn "
                "chạy hôm nay và cả tháng trước. Khẳng định như vậy là bịa ra một sự kiện "
                "kinh doanh mà dữ liệu không hề nói. "
                "`same_length` bằng false thì PHẢI nêu rõ hai kỳ dài khác nhau và dùng "
                "`per_day` để so, đừng so tổng. "
                "`all_failed` là sự cố thật: recipe có sản lượng mà pass rate bằng 0. "
                "Ô KPI và bảng so sánh đã được hệ thống gắn sẵn — nêu KẾT LUẬN và nguyên "
                "nhân, đừng đọc lại các con số đó."
            ),
        }
    except ValueError as e:
        # Chỉ lỗi tên kỳ mới tới đây; ValueError từ tầng dưới đã được các nhánh
        # trên chặn, nhưng vẫn ghi log để không nuốt mất nguyên nhân thật.
        logger.warning("compare_periods: %s", e)
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"compare_periods lỗi: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e), "message": "Không so sánh được hai kỳ"}


compare_periods_tool = BaseTool.create_tool(
    func=compare_periods,
    metadata=ToolMetadata(
        name="compare_periods",
        description=(
            "So sánh sản lượng và pass rate giữa HAI KỲ. Dùng khi user nói 'so sánh với "
            "hôm qua', 'tuần này so tuần trước', 'tháng này thế nào so tháng rồi', "
            "'có tốt hơn không'. Bỏ trống `period_b` thì tự lấy kỳ liền trước cùng độ dài. "
            "ĐỪNG gọi get_pass_fail_stats hai lần rồi tự trừ — chênh lệch phần trăm tính "
            "tay rất dễ sai, tool này tính sẵn cả hiệu tuyệt đối và thay đổi tương đối."
        ),
        category="analytics",
    ),
    args_schema=ComparePeriodsArgs,
)

logger.info("✅ Compare tool registered")
