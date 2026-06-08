from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class MongoDB:
    client: AsyncIOMotorClient = None
    db = None

class SyncMongoDB:
    """Sync MongoDB client for use in sync contexts (e.g., LangGraph tools)"""
    client: MongoClient = None
    db = None

mongodb = MongoDB()
sync_mongodb = SyncMongoDB()


async def connect_to_mongo():
    """Connect to MongoDB (async)"""
    mongodb.client = AsyncIOMotorClient(settings.MONGODB_URL)
    mongodb.db = mongodb.client[settings.DATABASE_NAME]
    print(f"✅ Connected to MongoDB (async): {settings.DATABASE_NAME}")

    # Also initialize sync client
    sync_mongodb.client = MongoClient(settings.MONGODB_URL)
    sync_mongodb.db = sync_mongodb.client[settings.DATABASE_NAME]
    print(f"✅ Connected to MongoDB (sync): {settings.DATABASE_NAME}")


async def close_mongo_connection():
    """Close MongoDB connection"""
    mongodb.client.close()
    print("❌ Closed MongoDB connection (async)")

    if sync_mongodb.client:
        sync_mongodb.client.close()
        print("❌ Closed MongoDB connection (sync)")


async def ensure_ttl_index(collection, field: str, expire_seconds: int, direction: int = -1):
    """Ensure a TTL index exists on `field` so MongoDB auto-deletes old documents.

    Idempotent and safe to call on every startup:
    - Fresh DB: creates the index directly with expireAfterSeconds.
    - Existing DB where a non-TTL index already exists on the same key: drops
      that index once and recreates it with expireAfterSeconds.
    - expire_seconds <= 0: no-op (TTL disabled for this collection).

    Never raises — TTL setup must not block app startup; failures are logged.
    """
    if expire_seconds <= 0:
        return

    keys = [(field, direction)]
    try:
        await collection.create_index(keys, expireAfterSeconds=expire_seconds)
        return
    except OperationFailure as e:
        # 85 IndexOptionsConflict / 86 IndexKeySpecsConflict: an index on this
        # key already exists with different options (e.g. no TTL). Recreate it.
        if e.code not in (85, 86):
            logger.warning(f"TTL index on {collection.name}.{field} failed: {e}")
            return

    try:
        existing_name = None
        async for idx in collection.list_indexes():
            if list(idx.get("key", {}).keys()) == [field]:
                existing_name = idx["name"]
                break
        if existing_name:
            await collection.drop_index(existing_name)
        await collection.create_index(keys, expireAfterSeconds=expire_seconds)
        logger.info(
            f"♻️  Recreated {collection.name}.{field} as TTL index "
            f"({expire_seconds // 86400} days)"
        )
    except Exception as e:
        logger.warning(f"TTL index recreate on {collection.name}.{field} failed: {e}")


def get_database():
    """Get database instance (async)"""
    return mongodb.db


def get_sync_database():
    """Get sync database instance for use in sync contexts"""
    return sync_mongodb.db
