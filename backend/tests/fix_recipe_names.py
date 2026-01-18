"""
Fix recipe names in inference_results collection
Match recipe_id with actual recipe name from recipes collection
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings


async def fix_recipe_names():
    # Connect to MongoDB
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.MONGODB_DB_NAME]

    recipes_collection = db["recipes"]
    inference_results_collection = db["inference_results"]

    # Get all recipes
    recipes = {}
    async for recipe in recipes_collection.find({}):
        recipes[str(recipe["_id"])] = recipe["name"]

    print(f"✅ Found {len(recipes)} recipes")
    for recipe_id, recipe_name in recipes.items():
        print(f"  - {recipe_id}: {recipe_name}")

    # Update inference results
    updated_count = 0
    async for result in inference_results_collection.find({}):
        recipe_id = result.get("recipe_id")
        current_name = result.get("recipe_name")
        correct_name = recipes.get(recipe_id)

        if correct_name and current_name != correct_name:
            await inference_results_collection.update_one(
                {"_id": result["_id"]},
                {"$set": {"recipe_name": correct_name}}
            )
            updated_count += 1
            print(f"✅ Updated {result['_id']}: '{current_name}' -> '{correct_name}'")

    print(f"\n🎉 Updated {updated_count} inference results")

    client.close()


if __name__ == '__main__':
    asyncio.run(fix_recipe_names())
