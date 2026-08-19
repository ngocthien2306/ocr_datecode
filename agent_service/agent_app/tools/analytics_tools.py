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
                    hit = False

                    tv = frame.get("text_verification") or {}
                    if tv.get("all_match") is False:
                        causes["text_verification"] += 1
                        hit = True
                        for r in tv.get("results") or []:
                            if not r.get("match"):
                                key = (str(r.get("expected")), str(r.get("recognized")))
                                mismatches[key] = mismatches.get(key, 0) + 1

                    cv = frame.get("char_verification") or {}
                    if cv.get("all_match") is False:
                        causes["char_verification"] += 1
                        hit = True

                    tpl = frame.get("template_verification") or {}
                    if tpl.get("match") is False:
                        causes["template_verification"] += 1
                        hit = True
                        if isinstance(tpl.get("similarity"), (int, float)):
                            sims.append(tpl["similarity"])

                    pv = frame.get("product_verification") or {}
                    if pv.get("match") is False and not pv.get("skipped"):
                        causes["product_verification"] += 1
                        hit = True

                    if not hit:
                        causes["unknown"] += 1

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

        return {
            "success": True,
            "period": {"start": _to_local_str(start_dt), "end": _to_local_str(end_dt)},
            "filters": {"recipe_id": recipe_id or "all", "camera": camera or "all"},
            "total_failed_products": total_fail,
            "examined_products": min(total_fail, sample_limit),
            "failed_camera_frames_examined": examined,
            "causes": {k: v for k, v in causes.items() if v},
            "failed_frames_by_camera": dict(sorted(per_camera.items(), key=lambda x: -x[1])),
            "top_text_mismatches": top_mismatch,
            "template_similarity_avg": round(sum(sims) / len(sims), 4) if sims else None,
            "samples": samples,
            "note": (
                f"Mổ {min(total_fail, sample_limit)} sản phẩm fail gần nhất "
                f"trên tổng {total_fail}. Một frame có thể trượt nhiều bước "
                f"nên tổng các nguyên nhân có thể lớn hơn số frame."
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
            "template / không nhận ra sản phẩm), kèm các cặp expected-recognized "
            "sai nhiều nhất. Lọc được theo recipe, camera và khung giờ."
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
