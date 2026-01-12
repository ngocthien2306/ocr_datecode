from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from app.core.config import settings

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


def get_database():
    """Get database instance (async)"""
    return mongodb.db


def get_sync_database():
    """Get sync database instance for use in sync contexts"""
    return sync_mongodb.db
