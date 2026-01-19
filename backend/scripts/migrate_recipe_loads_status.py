#!/usr/bin/env python3
"""
Migration Script: Add status field to recipe_loads collection

This script adds the following fields to existing recipe_load documents:
- status: "stopped" (default for existing records)
- stopped_at: None
- stopped_by: None
- stopped_by_full_name: None

Usage:
    python3 scripts/migrate_recipe_loads_status.py
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings


async def migrate():
    """Run migration to add status field to recipe_loads"""

    print("=" * 80)
    print("Recipe Loads Status Migration")
    print("=" * 80)
    print()

    # Connect to MongoDB
    print(f"Connecting to MongoDB: {settings.MONGO_URL}")
    client = AsyncIOMotorClient(settings.MONGO_URL)
    db = client[settings.MONGO_DB_NAME]
    collection = db['receipt_loads']

    # Count documents without status field
    count_no_status = await collection.count_documents({'status': {'$exists': False}})

    print(f"Found {count_no_status} documents without status field")
    print()

    if count_no_status == 0:
        print("✅ All documents already have status field. Nothing to migrate.")
        client.close()
        return

    # Ask for confirmation
    print("This will update all documents to add:")
    print("  - status: 'stopped'")
    print("  - stopped_at: None")
    print("  - stopped_by: None")
    print("  - stopped_by_full_name: None")
    print()

    response = input(f"Proceed with migration of {count_no_status} documents? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Migration cancelled.")
        client.close()
        return

    print()
    print("Running migration...")

    # Update all documents without status field
    result = await collection.update_many(
        {'status': {'$exists': False}},
        {
            '$set': {
                'status': 'stopped',
                'stopped_at': None,
                'stopped_by': None,
                'stopped_by_full_name': None
            }
        }
    )

    print(f"✅ Migration complete!")
    print(f"   Modified {result.modified_count} documents")
    print()

    # Verify migration
    count_running = await collection.count_documents({'status': 'running'})
    count_stopped = await collection.count_documents({'status': 'stopped'})
    count_total = await collection.count_documents({})

    print("Status after migration:")
    print(f"  - Running: {count_running}")
    print(f"  - Stopped: {count_stopped}")
    print(f"  - Total: {count_total}")
    print()

    # Close connection
    client.close()
    print("Migration completed successfully!")


if __name__ == "__main__":
    asyncio.run(migrate())
