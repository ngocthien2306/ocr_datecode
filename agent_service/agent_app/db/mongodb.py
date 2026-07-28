"""
MongoDB connections for the agent service.

Hai client, giống backend:
- async (motor)  → dùng trong FastAPI endpoint / conversation memory
- sync (pymongo) → dùng trong LangGraph tools (tool function là sync, chạy
  trong threadpool nên không được đụng vào event loop)

Kết nối tới CÙNG database với backend — chỉ đọc, trừ collection
`agent_conversations` do service này sở hữu.
"""

import logging

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient

from agent_app.core.config import settings

logger = logging.getLogger(__name__)


class _Async:
    client: AsyncIOMotorClient = None
    db = None


class _Sync:
    client: MongoClient = None
    db = None


_async_mongo = _Async()
_sync_mongo = _Sync()


async def connect_to_mongo():
    _async_mongo.client = AsyncIOMotorClient(settings.MONGODB_URL)
    _async_mongo.db = _async_mongo.client[settings.DATABASE_NAME]

    _sync_mongo.client = MongoClient(settings.MONGODB_URL)
    _sync_mongo.db = _sync_mongo.client[settings.DATABASE_NAME]

    logger.info("Connected to MongoDB (async + sync): %s", settings.DATABASE_NAME)


async def close_mongo_connection():
    if _async_mongo.client:
        _async_mongo.client.close()
    if _sync_mongo.client:
        _sync_mongo.client.close()
    logger.info("Closed MongoDB connections")


async def ensure_indexes():
    """Index cho collection hội thoại — agent service tự quản lý."""
    coll = _async_mongo.db["agent_conversations"]
    await coll.create_index([("session_id", 1)], unique=True)
    await coll.create_index([("user_id", 1), ("updated_at", -1)])
    logger.info("agent_conversations indexes ensured")


def get_database():
    """Async database (motor)."""
    return _async_mongo.db


def get_sync_database():
    """Sync database (pymongo) — dùng trong LangGraph tools."""
    return _sync_mongo.db
