"""
Initialize MongoDB indexes for optimal performance
Run this once when setting up the database
"""

from app.db.mongodb import get_database
import asyncio


async def create_conversation_indexes():
    """Create indexes for agent_conversations collection"""
    db = get_database()
    collection = db["agent_conversations"]

    # Index on session_id (unique, for fast lookups)
    await collection.create_index("session_id", unique=True)
    print("✅ Created index: session_id (unique)")

    # Index on user_id (for listing user's conversations)
    await collection.create_index("user_id")
    print("✅ Created index: user_id")

    # Compound index on user_id + updated_at (for sorted user conversations)
    await collection.create_index([("user_id", 1), ("updated_at", -1)])
    print("✅ Created compound index: user_id + updated_at")

    # TTL index on updated_at (optional: auto-delete old conversations after 90 days)
    # Uncomment if you want automatic cleanup:
    # await collection.create_index("updated_at", expireAfterSeconds=90*24*60*60)
    # print("✅ Created TTL index: updated_at (90 days)")


async def init_all_indexes():
    """Initialize all database indexes"""
    print("🔧 Initializing MongoDB indexes...")

    await create_conversation_indexes()

    print("✅ All indexes created successfully!")


if __name__ == "__main__":
    # Run this script to create indexes
    from app.db.mongodb import connect_to_mongo

    async def main():
        await connect_to_mongo()
        await init_all_indexes()

    asyncio.run(main())
