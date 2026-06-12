#!/usr/bin/env python3
"""
Migration Script: Set fixed model config on all recipes

Sets the following fields on every recipe document:
  - classifier_backend: "embedding"
  - cv_method:          "v4"   (Ink Defect Detector Scale-Tolerant)
  - defect_model:       "supcon"

Usage:
    python3 scripts/migrate_set_model_config.py
"""

import asyncio
import sys
from pathlib import Path

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

TARGET = {
    "classifier_backend": "embedding",
    "cv_method": "v4",
    "defect_model": "supcon",
}


async def migrate():
    print("=" * 70)
    print("Recipe Model Config Migration")
    print("=" * 70)
    print()
    for k, v in TARGET.items():
        print(f"  {k:25s} → {v}")
    print()

    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DATABASE_NAME]
    collection = db["recipes"]

    total = await collection.count_documents({})
    print(f"Total recipes in DB: {total}")

    # Show how many already match / need updating
    already_ok = await collection.count_documents(TARGET)
    need_update = total - already_ok
    print(f"Already correct:     {already_ok}")
    print(f"Need update:         {need_update}")
    print()

    if need_update == 0:
        print("✅ All recipes already have the correct config. Nothing to do.")
        client.close()
        return

    response = input(f"Proceed with updating {need_update} recipe(s)? (yes/no): ")
    if response.strip().lower() not in ("yes", "y"):
        print("Migration cancelled.")
        client.close()
        return

    print()
    print("Running update...")

    result = await collection.update_many(
        {},
        {"$set": TARGET}
    )

    print(f"✅ Done! Modified: {result.modified_count} / {total} documents")
    print()

    # Verify
    for k, v in TARGET.items():
        count = await collection.count_documents({k: v})
        status = "✅" if count == total else "⚠️ "
        print(f"  {status} {k}={v!r}: {count}/{total}")

    client.close()
    print()
    print("Migration completed.")


if __name__ == "__main__":
    asyncio.run(migrate())
