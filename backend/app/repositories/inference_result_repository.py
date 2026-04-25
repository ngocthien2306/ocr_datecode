"""
Inference Result Repository
Database operations for inference results
"""

from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime, timedelta, timezone
from bson import ObjectId
import logging

from app.models.inference_result import (
    InferenceResultCreate,
    InferenceResultInDB,
    InferenceResultResponse
)
from app.models.statistics import (
    SummaryStatisticsResponse,
    TimeseriesStatisticsResponse,
    RecipeStats,
    TimeseriesDataPoint,
    TimeseriesRecipeData,
    PeriodInfo
)

logger = logging.getLogger(__name__)

# Vietnam timezone offset (UTC+7)
VIETNAM_TIMEZONE_OFFSET_HOURS = 7


class InferenceResultRepository:
    """Repository for inference results"""

    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection: AsyncIOMotorCollection = db["inference_results"]

    async def create_indexes(self):
        """Create database indexes"""
        await self.collection.create_index([("timestamp", -1)])
        await self.collection.create_index([("recipe_id", 1)])
        await self.collection.create_index([("product_pass_fail", 1)])
        await self.collection.create_index([("created_at", -1)])
        logger.info("✅ Inference result indexes created")

    async def create(
        self,
        result_data: InferenceResultCreate
    ) -> InferenceResultResponse:
        """
        Create new inference result

        Args:
            result_data: Inference result data

        Returns:
            Created result
        """
        # Create result document
        result_doc = InferenceResultInDB(
            **result_data.dict(),
            timestamp=datetime.utcnow(),
            created_at=datetime.utcnow()
        )

        # Convert to dict for MongoDB
        result_dict = result_doc.dict(by_alias=True)

        # Insert into database
        insert_result = await self.collection.insert_one(result_dict)

        # Return response
        result_dict["_id"] = str(insert_result.inserted_id)

        return InferenceResultResponse(**result_dict)

    async def get_by_id(self, result_id: str) -> Optional[InferenceResultResponse]:
        """
        Get inference result by ID

        Args:
            result_id: Result ID

        Returns:
            Result or None
        """
        try:
            logger.debug(f"Looking up result by ID: {result_id}")

            # Validate ObjectId format
            if not ObjectId.is_valid(result_id):
                logger.warning(f"Invalid ObjectId format: {result_id}")
                return None

            
            result = await self.collection.find_one({"_id": result_id})

            if result:
                result["_id"] = str(result["_id"])
                logger.debug(f"Found result: {result['_id']}")
                return InferenceResultResponse(**result)

            logger.debug(f"No result found for ID: {result_id}")
            return None

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error getting result by ID {result_id}: {e}")
            return None

    def _build_query(
        self,
        recipe_id: Optional[str] = None,
        pass_fail: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        fail_reasons: Optional[List[str]] = None,
        wrinkled_min_area: Optional[float] = None,
        center_direction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build MongoDB filter query from params."""
        query: Dict[str, Any] = {}

        if recipe_id:
            query["recipe_id"] = recipe_id

        if pass_fail:
            query["product_pass_fail"] = pass_fail

        if start_date or end_date:
            date_filter: Dict[str, Any] = {}
            if start_date:
                date_filter["$gte"] = start_date
            if end_date:
                date_filter["$lte"] = end_date
            query["timestamp"] = date_filter

        if fail_reasons:
            conditions: List[Dict[str, Any]] = []
            for reason in fail_reasons:
                if reason == "text":
                    conditions.append(
                        {"camera_results.frames.text_verification.all_match": False}
                    )
                elif reason == "char":
                    # ML per-character inspection failed at least once
                    conditions.append(
                        {"camera_results.frames.char_verification.all_match": False}
                    )
                elif reason == "template":
                    conditions.append(
                        {"camera_results.frames.template_verification.match": False}
                    )
                elif reason == "wrinkled":
                    wrinkled_cond: Dict[str, Any] = {
                        "camera_results.frames.product_verification.wrinkled_check.ok": False
                    }
                    if wrinkled_min_area is not None:
                        wrinkled_cond[
                            "camera_results.frames.product_verification.wrinkled_check.wrinkled_boxes.area"
                        ] = {"$gte": wrinkled_min_area}
                    conditions.append(wrinkled_cond)
                elif reason == "center":
                    center_cond: Dict[str, Any] = {
                        "camera_results.frames.product_verification.center_alignment_check.ok": False
                    }
                    if center_direction and center_direction != "any":
                        center_cond[
                            "camera_results.frames.product_verification.center_alignment_check.direction"
                        ] = center_direction
                    conditions.append(center_cond)

            if conditions:
                query["$and"] = conditions

        return query

    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        recipe_id: Optional[str] = None,
        pass_fail: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        fail_reasons: Optional[List[str]] = None,
        wrinkled_min_area: Optional[float] = None,
        center_direction: Optional[str] = None,
    ) -> List[InferenceResultResponse]:
        query = self._build_query(
            recipe_id=recipe_id,
            pass_fail=pass_fail,
            start_date=start_date,
            end_date=end_date,
            fail_reasons=fail_reasons,
            wrinkled_min_area=wrinkled_min_area,
            center_direction=center_direction,
        )

        cursor = self.collection.find(query).sort("timestamp", -1).skip(skip).limit(limit)

        results = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(InferenceResultResponse(**doc))

        return results

    async def count(
        self,
        recipe_id: Optional[str] = None,
        pass_fail: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        fail_reasons: Optional[List[str]] = None,
        wrinkled_min_area: Optional[float] = None,
        center_direction: Optional[str] = None,
    ) -> int:
        query = self._build_query(
            recipe_id=recipe_id,
            pass_fail=pass_fail,
            start_date=start_date,
            end_date=end_date,
            fail_reasons=fail_reasons,
            wrinkled_min_area=wrinkled_min_area,
            center_direction=center_direction,
        )
        return await self.collection.count_documents(query)

    async def delete_by_id(self, result_id: str) -> bool:
        """
        Delete inference result by ID

        Args:
            result_id: Result ID (stored as string in MongoDB)

        Returns:
            True if deleted
        """
        try:
            # Query by string _id (not ObjectId) since documents are stored with string _id
            delete_result = await self.collection.delete_one(
                {"_id": result_id}
            )
            return delete_result.deleted_count > 0

        except Exception as e:
            logger.error(f"Error deleting result: {e}")
            return False

    def _convert_vn_to_utc(self, vn_time: datetime) -> datetime:
        """
        Convert Vietnam time to UTC

        Args:
            vn_time: Datetime in VN timezone

        Returns:
            Datetime in UTC
        """
        return vn_time - timedelta(hours=VIETNAM_TIMEZONE_OFFSET_HOURS)

    def _convert_utc_to_vn(self, utc_time: datetime) -> datetime:
        """
        Convert UTC time to Vietnam time

        Args:
            utc_time: Datetime in UTC

        Returns:
            Datetime in VN timezone
        """
        return utc_time + timedelta(hours=VIETNAM_TIMEZONE_OFFSET_HOURS)

    async def get_summary_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        recipe_id: Optional[str] = None
    ) -> SummaryStatisticsResponse:
        """
        Get summary statistics with breakdown by camera and recipe

        Args:
            start_date: Start date (already in UTC from ISO string parse)
            end_date: End date (already in UTC from ISO string parse)
            recipe_id: Filter by recipe ID

        Returns:
            Summary statistics
        """
        # start_date and end_date are already UTC from FastAPI's datetime parsing
        # No conversion needed for DB query
        start_utc = start_date
        end_utc = end_date

        # Build match filter
        match_filter: Dict[str, Any] = {}
        if start_utc or end_utc:
            date_filter: Dict[str, Any] = {}
            if start_utc:
                date_filter["$gte"] = start_utc
            if end_utc:
                date_filter["$lte"] = end_utc
            match_filter["created_at"] = date_filter

        if recipe_id:
            match_filter["recipe_id"] = recipe_id

        # Aggregation pipeline
        pipeline = [
            {"$match": match_filter} if match_filter else {"$match": {}},
            {
                "$facet": {
                    # Overall stats
                    "overall": [
                        {
                            "$group": {
                                "_id": None,
                                "total": {"$sum": 1},
                                "pass": {
                                    "$sum": {
                                        "$cond": [
                                            {"$eq": ["$product_pass_fail", "PASS"]},
                                            1,
                                            0
                                        ]
                                    }
                                },
                                "fail": {
                                    "$sum": {
                                        "$cond": [
                                            {"$eq": ["$product_pass_fail", "FAIL"]},
                                            1,
                                            0
                                        ]
                                    }
                                }
                            }
                        }
                    ],
                    # Stats by recipe
                    "by_recipe": [
                        {
                            "$group": {
                                "_id": "$recipe_id",
                                "recipe_name": {"$first": "$recipe_name"},
                                "total": {"$sum": 1},
                                "pass": {
                                    "$sum": {
                                        "$cond": [
                                            {"$eq": ["$product_pass_fail", "PASS"]},
                                            1,
                                            0
                                        ]
                                    }
                                },
                                "fail": {
                                    "$sum": {
                                        "$cond": [
                                            {"$eq": ["$product_pass_fail", "FAIL"]},
                                            1,
                                            0
                                        ]
                                    }
                                }
                            }
                        },
                        {"$sort": {"recipe_name": 1}}
                    ]
                }
            }
        ]

        result = await self.collection.aggregate(pipeline).to_list(length=1)

        if not result:
            # No data
            return SummaryStatisticsResponse(
                total=0,
                **{"pass": 0},  # Use dict unpacking with "pass" key
                fail=0,
                pass_rate=0.0,
                period=PeriodInfo(
                    start_date=start_date or datetime.utcnow(),
                    end_date=end_date or datetime.utcnow()
                ),
                by_recipe=[]
            )

        data = result[0]

        # Parse overall stats
        overall = data["overall"][0] if data["overall"] else {
            "total": 0, "pass": 0, "fail": 0
        }
        total = overall["total"]
        pass_count = overall["pass"]
        fail_count = overall["fail"]
        pass_rate = (pass_count / total * 100) if total > 0 else 0.0

        # Parse recipe stats
        recipe_stats = []
        for recipe in data["by_recipe"]:
            recipe_total = recipe["total"]
            recipe_pass = recipe.get("pass", 0)
            recipe_fail = recipe.get("fail", 0)
            recipe_pass_rate = (recipe_pass / recipe_total * 100) if recipe_total > 0 else 0.0

            recipe_stats.append(RecipeStats(
                recipe_id=recipe["_id"],
                recipe_name=recipe["recipe_name"],
                total=recipe_total,
                **{"pass": recipe_pass},  # Use dict unpacking with "pass" key
                fail=recipe_fail,
                pass_rate=round(recipe_pass_rate, 1)
            ))

        return SummaryStatisticsResponse(
            total=total,
            **{"pass": pass_count},  # Use dict unpacking with "pass" key
            fail=fail_count,
            pass_rate=round(pass_rate, 1),
            period=PeriodInfo(
                start_date=start_date or datetime.utcnow(),
                end_date=end_date or datetime.utcnow()
            ),
            by_recipe=recipe_stats
        )

    async def get_timeseries_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        granularity: Literal["minute", "hour", "day"] = "day",
        recipe_ids: Optional[List[str]] = None
    ) -> TimeseriesStatisticsResponse:
        """
        Get timeseries statistics grouped by recipe

        Args:
            start_date: Start date (already in UTC from ISO string parse)
            end_date: End date (already in UTC from ISO string parse)
            granularity: Time granularity (minute, hour or day)
            recipe_ids: Filter by recipes (list)

        Returns:
            Timeseries statistics
        """
        # start_date and end_date are already UTC from FastAPI's datetime parsing
        # No conversion needed for DB query
        start_utc = start_date
        end_utc = end_date

        # Build match filter
        match_filter: Dict[str, Any] = {}
        if start_utc or end_utc:
            date_filter: Dict[str, Any] = {}
            if start_utc:
                date_filter["$gte"] = start_utc
            if end_utc:
                date_filter["$lte"] = end_utc
            match_filter["created_at"] = date_filter

        if recipe_ids and len(recipe_ids) > 0:
            match_filter["recipe_id"] = {"$in": recipe_ids}

        # Aggregation pipeline
        pipeline = [
            {"$match": match_filter} if match_filter else {"$match": {}},
            # Convert created_at to VN time
            {
                "$addFields": {
                    "created_at_vn": {
                        "$dateAdd": {
                            "startDate": "$created_at",
                            "unit": "hour",
                            "amount": VIETNAM_TIMEZONE_OFFSET_HOURS
                        }
                    }
                }
            },
            # Group by time + recipe
            {
                "$group": {
                    "_id": {
                        "timestamp": {
                            "$dateTrunc": {
                                "date": "$created_at_vn",
                                "unit": granularity
                            }
                        },
                        "recipe_id": "$recipe_id"
                    },
                    "recipe_name": {"$first": "$recipe_name"},
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
            # Group by timestamp to aggregate recipes
            {
                "$group": {
                    "_id": "$_id.timestamp",
                    "total": {"$sum": "$total"},
                    "pass": {"$sum": "$pass"},
                    "fail": {"$sum": "$fail"},
                    "recipes": {
                        "$push": {
                            "recipe_id": "$_id.recipe_id",
                            "recipe_name": "$recipe_name",
                            "total": "$total",
                            "pass": "$pass",
                            "fail": "$fail"
                        }
                    }
                }
            },
            {"$sort": {"_id": 1}}
        ]

        results = await self.collection.aggregate(pipeline).to_list(length=None)

        # Build timeseries data points
        data_points = []
        for item in results:
            timestamp_vn = item["_id"]
            total = item["total"]
            pass_count = item["pass"]
            fail_count = item["fail"]
            pass_rate = (pass_count / total * 100) if total > 0 else 0.0

            # Build recipe data
            recipe_data = []
            for recipe in item["recipes"]:
                recipe_total = recipe["total"]
                recipe_pass = recipe["pass"]
                recipe_fail = recipe["fail"]
                recipe_pass_rate = (recipe_pass / recipe_total * 100) if recipe_total > 0 else 0.0

                recipe_data.append(TimeseriesRecipeData(
                    recipe_id=recipe["recipe_id"],
                    recipe_name=recipe["recipe_name"],
                    total=recipe_total,
                    **{"pass": recipe_pass},
                    fail=recipe_fail,
                    pass_rate=round(recipe_pass_rate, 1)
                ))

            data_points.append(TimeseriesDataPoint(
                timestamp=timestamp_vn,
                total=total,
                **{"pass": pass_count},
                fail=fail_count,
                pass_rate=round(pass_rate, 1),
                by_recipe=recipe_data
            ))

        return TimeseriesStatisticsResponse(
            granularity=granularity,
            period=PeriodInfo(
                start_date=start_date or datetime.utcnow(),
                end_date=end_date or datetime.utcnow()
            ),
            data=data_points
        )
