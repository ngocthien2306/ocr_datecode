"""
Analytics Tools for Historical Data Analysis
Provides tools for querying and analyzing production data
"""

import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from app.db.mongodb import get_sync_database
from app.agent.tools.base_tool import BaseTool, ToolMetadata
import logging

logger = logging.getLogger(__name__)


# Tool argument schemas
class PassFailStatsArgs(BaseModel):
    """Arguments for get_pass_fail_stats"""
    recipe_id: Optional[str] = Field(None, description="Filter by specific recipe ID")
    start_date: Optional[str] = Field(None, description="Start date in ISO format YYYY-MM-DD (default: today)")
    end_date: Optional[str] = Field(None, description="End date in ISO format YYYY-MM-DD (default: today)")
    group_by: str = Field("day", description="Group results by 'hour', 'day', 'week', or 'month'")


class ProductionSummaryArgs(BaseModel):
    """Arguments for get_production_summary"""
    date: Optional[str] = Field(None, description="Date in ISO format YYYY-MM-DD (default: today)")
    group_by: str = Field("recipe", description="Group by 'recipe', 'camera', or 'hour'")


class RecipeLoadHistoryArgs(BaseModel):
    """Arguments for get_recipe_load_history"""
    recipe_id: Optional[str] = Field(None, description="Filter by specific recipe ID")
    user_id: Optional[str] = Field(None, description="Filter by specific user ID")
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

        # Parse dates
        if not start_date:
            start_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            start_dt = datetime.fromisoformat(start_date)

        if not end_date:
            end_dt = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999)
        else:
            end_dt = datetime.fromisoformat(end_date)

        # Build query
        query = {
            "timestamp": {
                "$gte": start_dt,
                "$lte": end_dt
            }
        }

        if recipe_id:
            query["recipe_id"] = recipe_id

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
                        "time": {"$dateToString": {"format": group_format, "date": "$timestamp"}},
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
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
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
    group_by: str = "recipe"
) -> Dict[str, Any]:
    """
    Get production summary with breakdown by recipe or time

    Args:
        date: Date in ISO format YYYY-MM-DD (default: today)
        group_by: Group by 'recipe', 'hour', or 'camera' (default: recipe)

    Returns:
        Production summary with breakdown
    """
    try:
        db = get_sync_database()
        collection = db["inference_results"]

        # Parse date
        if not date:
            target_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            target_date = datetime.fromisoformat(date)

        start_dt = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)

        query = {
            "timestamp": {
                "$gte": start_dt,
                "$lte": end_dt
            }
        }

        # Build aggregation based on group_by
        if group_by == "recipe":
            group_field = "$recipe_name"
            group_key = "recipe"
        elif group_by == "hour":
            group_field = {"$hour": "$timestamp"}
            group_key = "hour"
        elif group_by == "camera":
            # Unwind camera results first
            pipeline = [
                {"$match": query},
                {"$unwind": "$camera_results"},
                {
                    "$group": {
                        "_id": "$camera_results.serial_number",
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
        collection = db["recipe_loads"]

        # Build query
        query = {}
        if recipe_id:
            query["recipe_id"] = recipe_id
        if user_id:
            query["loaded_by"] = user_id

        # Get records
        cursor = collection.find(query).sort("loaded_at", -1).limit(limit)
        records = list(cursor)

        # Format results
        history = []
        for record in records:
            # Calculate runtime if still running
            runtime_seconds = None
            if record["status"] == "running":
                runtime_seconds = (datetime.now() - record["loaded_at"]).total_seconds()
            elif record.get("stopped_at"):
                runtime_seconds = (record["stopped_at"] - record["loaded_at"]).total_seconds()

            runtime_str = None
            if runtime_seconds:
                hours = int(runtime_seconds // 3600)
                minutes = int((runtime_seconds % 3600) // 60)
                runtime_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"

            history.append({
                "recipe_id": record["recipe_id"],
                "recipe_name": record["metadata"].get("name", "Unknown"),
                "loaded_by": record["loaded_by_full_name"],
                "loaded_at": record["loaded_at"].isoformat(),
                "loaded_at_local": record.get("loaded_at_local", ""),
                "status": record["status"],
                "stopped_by": record.get("stopped_by_full_name"),
                "stopped_at": record.get("stopped_at").isoformat() if record.get("stopped_at") else None,
                "runtime": runtime_str
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


# Register tools
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
        description="Lấy tổng quan sản xuất theo ngày, phân loại theo recipe/camera/giờ. Hữu ích để xem tổng quan nhanh.",
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
