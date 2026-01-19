"""
Migration script to update trigger schema in recipes collection

Old schema: trigger_config.mode = "continuous" | "software" | "hardware"
            trigger_config.trigger_source = "Software" | "Line0" | ...

New schema: trigger_mode = "continuous" | "software_trigger" | "hardware_trigger" (top level)
            trigger_config.trigger_selector = "FrameStart" | ...
            trigger_config.trigger_activation = "RisingEdge" | ...
            trigger_config.di_number = 0-3
            trigger_config.trigger_source = "Line0" | "Line1" | ... (NO "Software")
"""

import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import UpdateOne
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "ocr_datecode")


async def migrate_recipes():
    """Migrate all recipes to new trigger schema"""

    # Connect to MongoDB
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DATABASE_NAME]
    collection = db["recipes"]

    print(f"Connected to MongoDB: {MONGODB_URL}/{DATABASE_NAME}")

    # Find all recipes with cameras
    recipes = await collection.find({"cameras": {"$exists": True, "$ne": []}}).to_list(None)

    print(f"Found {len(recipes)} recipes with cameras")

    updates = []
    migrated_count = 0

    for recipe in recipes:
        recipe_id = recipe["_id"]
        recipe_name = recipe.get("name", "Unknown")
        cameras = recipe.get("cameras", [])

        needs_migration = False

        for i, cam in enumerate(cameras):
            trigger_config = cam.get("trigger_config", {})

            # Check if already migrated (has trigger_mode at top level)
            if "trigger_mode" in cam:
                print(f"  ✓ Recipe '{recipe_name}' - Camera {i} already migrated")
                continue

            # Check if old schema exists (has mode in trigger_config)
            if "mode" in trigger_config:
                old_mode = trigger_config["mode"]

                # Map old mode to new trigger_mode
                if old_mode == "software":
                    cam["trigger_mode"] = "software_trigger"
                elif old_mode == "hardware":
                    cam["trigger_mode"] = "hardware_trigger"
                else:
                    cam["trigger_mode"] = "continuous"

                # Remove old "mode" field from trigger_config
                del trigger_config["mode"]

                # Fix trigger_source: "Software" → "Line0"
                if trigger_config.get("trigger_source") == "Software":
                    trigger_config["trigger_source"] = "Line0"

                needs_migration = True
                print(f"  → Migrating Recipe '{recipe_name}' - Camera {i}: {old_mode} → {cam['trigger_mode']}")
            else:
                # No old schema, add default trigger_mode
                cam["trigger_mode"] = "continuous"
                needs_migration = True
                print(f"  → Adding default trigger_mode to Recipe '{recipe_name}' - Camera {i}")

        if needs_migration:
            # Add update operation
            updates.append(
                UpdateOne(
                    {"_id": recipe_id},
                    {"$set": {"cameras": cameras}}
                )
            )
            migrated_count += 1

    # Execute bulk update
    if updates:
        result = await collection.bulk_write(updates)
        print(f"\n✅ Migration completed:")
        print(f"   - Modified {result.modified_count} recipes")
        print(f"   - Total recipes migrated: {migrated_count}")
    else:
        print("\n✅ No recipes need migration (all already migrated)")

    # Close connection
    client.close()


async def verify_migration():
    """Verify migration results"""

    # Connect to MongoDB
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DATABASE_NAME]
    collection = db["recipes"]

    print("\n🔍 Verifying migration...")

    # Check for recipes with old schema
    old_schema_recipes = await collection.find({
        "cameras": {
            "$elemMatch": {
                "trigger_config.mode": {"$exists": True}
            }
        }
    }).to_list(None)

    if old_schema_recipes:
        print(f"⚠️  WARNING: Found {len(old_schema_recipes)} recipes still using old schema:")
        for recipe in old_schema_recipes:
            print(f"   - {recipe.get('name', 'Unknown')} (ID: {recipe['_id']})")
    else:
        print("✅ All recipes migrated successfully (no old schema found)")

    # Check for recipes with new schema
    new_schema_recipes = await collection.find({
        "cameras": {
            "$elemMatch": {
                "trigger_mode": {"$exists": True}
            }
        }
    }).to_list(None)

    print(f"\n📊 Summary:")
    print(f"   - Recipes with new schema: {len(new_schema_recipes)}")
    print(f"   - Recipes with old schema: {len(old_schema_recipes)}")

    # Show sample of migrated data
    if new_schema_recipes:
        sample = new_schema_recipes[0]
        print(f"\n📝 Sample migrated recipe: {sample.get('name', 'Unknown')}")
        for i, cam in enumerate(sample.get("cameras", [])[:1]):  # Show first camera only
            print(f"   Camera {i}:")
            print(f"      trigger_mode: {cam.get('trigger_mode', 'N/A')}")
            print(f"      trigger_config: {cam.get('trigger_config', {})}")

    # Close connection
    client.close()


async def main():
    """Main migration function"""
    print("=" * 60)
    print("Recipe Trigger Schema Migration")
    print("=" * 60)

    # Run migration
    await migrate_recipes()

    # Verify migration
    await verify_migration()

    print("\n" + "=" * 60)
    print("Migration completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
