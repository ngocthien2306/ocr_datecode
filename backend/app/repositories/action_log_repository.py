from motor.motor_asyncio import AsyncIOMotorCollection
from typing import List, Optional
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from app.core.config import settings
from bson import ObjectId
from app.models.action_log import ActionLogCreate, ActionLogResponse, ActionLogInDB


class ActionLogRepository:
    def __init__(self, database):
        self.collection: AsyncIOMotorCollection = database.action_logs

    async def create_action_log(self, action_log: ActionLogCreate) -> ActionLogResponse:
        """Create a new action log entry"""
        action_log_dict = action_log.model_dump()
        # Store timestamps in UTC (best practice). We'll convert to configured
        # local timezone when returning results to API callers.
        action_log_dict["timestamp"] = datetime.utcnow()

        result = await self.collection.insert_one(action_log_dict)
        action_log_dict["_id"] = str(result.inserted_id)

        return ActionLogResponse(**action_log_dict)

    async def get_action_logs(
        self,
        skip: int = 0,
        limit: int = 100,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        action_type: Optional[str] = None,
        resource_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[ActionLogResponse]:
        """Get action logs with optional filtering"""
        query = {}

        if user_id:
            query["user_id"] = user_id
        if username:
            query["username"] = username
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

        # Convert ObjectId to string for each log and convert timestamps to
        # the configured local timezone for presentation.
        tz = ZoneInfo(settings.TIMEZONE)
        for log in action_logs:
            log["_id"] = str(log["_id"])
            ts = log.get("timestamp")
            if isinstance(ts, datetime):
                # If naive, assume it's UTC; make it aware then convert.
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                log["timestamp"] = ts.astimezone(tz)

        return [ActionLogResponse(**log) for log in action_logs]

    async def get_action_log_by_id(self, action_log_id: str) -> Optional[ActionLogResponse]:
        """Get a specific action log by ID"""
        try:
            action_log = await self.collection.find_one({"_id": ObjectId(action_log_id)})
        except:
            return None

        if action_log:
            action_log["_id"] = str(action_log["_id"])
            ts = action_log.get("timestamp")
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                action_log["timestamp"] = ts.astimezone(ZoneInfo(settings.TIMEZONE))
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

        # Convert ObjectId to string for each log and convert timestamps
        tz = ZoneInfo(settings.TIMEZONE)
        for log in action_logs:
            log["_id"] = str(log["_id"])
            ts = log.get("timestamp")
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                log["timestamp"] = ts.astimezone(tz)

        return [ActionLogResponse(**log) for log in action_logs]

    async def get_recent_action_logs(self, limit: int = 20) -> List[ActionLogResponse]:
        """Get the most recent action logs"""
        cursor = self.collection.find().sort("timestamp", -1).limit(limit)
        action_logs = await cursor.to_list(length=None)

        # Convert ObjectId to string for each log and convert timestamps
        tz = ZoneInfo(settings.TIMEZONE)
        for log in action_logs:
            log["_id"] = str(log["_id"])
            ts = log.get("timestamp")
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                log["timestamp"] = ts.astimezone(tz)

        return [ActionLogResponse(**log) for log in action_logs]