from typing import Optional, List
from datetime import datetime
from bson import ObjectId
from app.models.recipe import RecipeCreate, RecipeUpdate, RecipeInDB


class RecipeRepository:
    def __init__(self, database):
        self.collection = database.recipes

    async def create_recipe(self, recipe: RecipeCreate, user_id: str) -> RecipeInDB:
        """Create a new recipe"""
        recipe_dict = recipe.model_dump()
        recipe_dict["created_by"] = user_id
        recipe_dict["updated_by"] = user_id
        recipe_dict["created_at"] = datetime.utcnow()
        recipe_dict["updated_at"] = datetime.utcnow()
        recipe_dict["version"] = 1

        result = await self.collection.insert_one(recipe_dict)
        recipe_dict["_id"] = str(result.inserted_id)

        return RecipeInDB(**recipe_dict)

    async def get_recipe_by_id(self, recipe_id: str) -> Optional[RecipeInDB]:
        """Get recipe by ID"""
        if not ObjectId.is_valid(recipe_id):
            return None

        recipe = await self.collection.find_one({"_id": ObjectId(recipe_id)})
        if recipe:
            recipe["_id"] = str(recipe["_id"])
            return RecipeInDB(**recipe)
        return None

    async def get_recipe_by_name(self, name: str) -> Optional[RecipeInDB]:
        """Get recipe by name"""
        recipe = await self.collection.find_one({"name": name})
        if recipe:
            recipe["_id"] = str(recipe["_id"])
            return RecipeInDB(**recipe)
        return None

    async def get_all_recipes(self, skip: int = 0, limit: int = 100, active_only: bool = False) -> List[RecipeInDB]:
        """Get all recipes with pagination"""
        query = {"is_active": True} if active_only else {}
        cursor = self.collection.find(query).skip(skip).limit(limit).sort("created_at", -1)

        recipes = []
        async for recipe in cursor:
            recipe["_id"] = str(recipe["_id"])
            recipes.append(RecipeInDB(**recipe))
        return recipes

    async def update_recipe(self, recipe_id: str, recipe_update: RecipeUpdate, user_id: str) -> Optional[RecipeInDB]:
        """Update recipe"""
        if not ObjectId.is_valid(recipe_id):
            return None

        update_data = {k: v for k, v in recipe_update.model_dump(exclude_unset=True).items() if v is not None}
        if not update_data:
            return await self.get_recipe_by_id(recipe_id)

        update_data["updated_by"] = user_id
        update_data["updated_at"] = datetime.utcnow()

        # Increment version number
        result = await self.collection.update_one(
            {"_id": ObjectId(recipe_id)},
            {
                "$set": update_data,
                "$inc": {"version": 1}
            }
        )

        if result.modified_count:
            return await self.get_recipe_by_id(recipe_id)
        return None

    async def delete_recipe(self, recipe_id: str) -> bool:
        """Delete recipe (soft delete by setting is_active=False)"""
        if not ObjectId.is_valid(recipe_id):
            return False

        result = await self.collection.update_one(
            {"_id": ObjectId(recipe_id)},
            {"$set": {"is_active": False, "updated_at": datetime.utcnow()}}
        )

        return result.modified_count > 0

    async def hard_delete_recipe(self, recipe_id: str) -> bool:
        """Permanently delete recipe"""
        if not ObjectId.is_valid(recipe_id):
            return False

        result = await self.collection.delete_one({"_id": ObjectId(recipe_id)})
        return result.deleted_count > 0

    async def create_indexes(self):
        """Create database indexes"""
        await self.collection.create_index("name", unique=True)
        await self.collection.create_index("is_active")
        await self.collection.create_index("created_at")
