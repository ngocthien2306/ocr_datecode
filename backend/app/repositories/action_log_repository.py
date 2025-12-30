from motor.motor_asyncio import AsyncIOMotorCollection
from typing import List, Optional
from datetime import datetime
from app.models.action_log import ActionLogCreate, ActionLogResponse, ActionLogInDB


class ActionLogRepository:
    def __init__(self, database):
        self.collection: AsyncIOMotorCollection = database.action_logs

    async def create_action_log(self, action_log: ActionLogCreate) -> ActionLogResponse:
        """Create a new action log entry"""
        action_log_dict = action_log.model_dump()
        action_log_dict["timestamp"] = datetime.utcnow()

        result = await self.collection.insert_one(action_log_dict)
        action_log_dict["_id"] = str(result.inserted_id)

        return ActionLogResponse(**action_log_dict)

    async def get_action_logs(
        self,
        skip: int = 0,
        limit: int = 100,
        user_id: Optional[str] = None,
        action_type: Optional[str] = None,
        resource_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[ActionLogResponse]:
        """Get action logs with optional filtering"""
        query = {}

        if user_id:
            query["user_id"] = user_id
        if action_type:
            query["action_type"] = action_type
        if resource_type:
            query["resource_type"] = resource_type

        if start_date or end_date:
            query["timestamp"] = {}
            if start_date:
                query["timestamp"]["$gte"] = start_date
            if end_date:
                query["timestamp"]["$lte"] = end_date

        cursor = self.collection.find(query).sort("timestamp", -1).skip(skip).limit(limit)
        action_logs = await cursor.to_list(length=None)

        return [ActionLogResponse(**log) for log in action_logs]

    async def get_action_log_by_id(self, action_log_id: str) -> Optional[ActionLogResponse]:
        """Get a specific action log by ID"""
        action_log = await self.collection.find_one({"_id": action_log_id})
        if action_log:
            return ActionLogResponse(**action_log)
        return None

    async def get_user_action_logs(
        self,
        user_id: str,
        skip: int = 0,
        limit: int = 50
    ) -> List[ActionLogResponse]:
        """Get action logs for a specific user"""
        cursor = self.collection.find({"user_id": user_id}).sort("timestamp", -1).skip(skip).limit(limit)
        action_logs = await cursor.to_list(length=None)

        return [ActionLogResponse(**log) for log in action_logs]

    async def get_recent_action_logs(self, limit: int = 20) -> List[ActionLogResponse]:
        """Get the most recent action logs"""
        cursor = self.collection.find().sort("timestamp", -1).limit(limit)
        action_logs = await cursor.to_list(length=None)

        return [ActionLogResponse(**log) for log in action_logs]