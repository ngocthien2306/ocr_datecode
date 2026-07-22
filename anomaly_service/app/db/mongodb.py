from motor.motor_asyncio import AsyncIOMotorClient

from app.core.config import settings


class MongoDB:
    client: AsyncIOMotorClient = None
    db = None


mongodb = MongoDB()


async def connect_to_mongo():
    mongodb.client = AsyncIOMotorClient(settings.MONGODB_URL)
    mongodb.db = mongodb.client[settings.DATABASE_NAME]
    print(f"✅ [anomaly_service] Connected to MongoDB: {settings.DATABASE_NAME}")


async def close_mongo_connection():
    if mongodb.client:
        mongodb.client.close()
        print("❌ [anomaly_service] Closed MongoDB connection")


def get_database():
    return mongodb.db
