"""
Inference Result Repository
Database operations for inference results
"""

from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection
from typing import List, Optional, Dict, Any
from datetime import datetime
from bson import ObjectId
import logging

from app.models.inference_result import (
    InferenceResultCreate,
    InferenceResultInDB,
    InferenceResultResponse
)

logger = logging.getLogger(__name__)


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
            result = await self.collection.find_one({"_id": ObjectId(result_id)})

            if result:
                result["_id"] = str(result["_id"])
                return InferenceResultResponse(**result)

            return None

        except Exception as e:
            logger.error(f"Error getting result by ID: {e}")
            return None

    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        recipe_id: Optional[str] = None,
        pass_fail: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[InferenceResultResponse]:
        """
        Get all inference results with filters

        Args:
            skip: Number to skip
            limit: Max number to return
            recipe_id: Filter by recipe ID
            pass_fail: Filter by pass/fail status
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            List of results
        """
        # Build filter query
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

        # Query database
        cursor = self.collection.find(query).sort(
            "timestamp", -1
        ).skip(skip).limit(limit)

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
        end_date: Optional[datetime] = None
    ) -> int:
        """
        Count inference results with filters

        Args:
            recipe_id: Filter by recipe ID
            pass_fail: Filter by pass/fail status
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            Count of results
        """
        # Build filter query
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

        return await self.collection.count_documents(query)

    async def delete_by_id(self, result_id: str) -> bool:
        """
        Delete inference result by ID

        Args:
            result_id: Result ID

        Returns:
            True if deleted
        """
        try:
            delete_result = await self.collection.delete_one(
                {"_id": ObjectId(result_id)}
            )
            return delete_result.deleted_count > 0

        except Exception as e:
            logger.error(f"Error deleting result: {e}")
            return False
