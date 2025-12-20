from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import Optional, List, Any
from datetime import datetime
from bson import ObjectId

from app.models.recipe import RecipeInDB, RecipeCreate, RecipeUpdate


class RecipeRepository:
    def __init__(self, database: Any):
        self.collection = database.get_collection("recipes")
    
    async def create_indexes(self):
        """Create indexes for recipe collection"""
        # Get existing indexes
        existing_indexes = await self.collection.index_information()
        
        # Create indexes only if they don't exist
        indexes_to_create = []
        
        # Individual indexes
        if "name_1" not in existing_indexes:
            indexes_to_create.append(("name", 1))
        if "product_code_1" not in existing_indexes:
            indexes_to_create.append(("product_code", 1))
        if "created_by_1" not in existing_indexes:
            indexes_to_create.append(("created_by", 1))
        
        # Create individual indexes
        for field, direction in indexes_to_create:
            await self.collection.create_index([(field, direction)])
        
        # Create unique compound index if not exists
        if "name_1_product_code_1" not in existing_indexes:
            await self.collection.create_index([("name", 1), ("product_code", 1)], unique=True)
    
    async def create(self, recipe: RecipeCreate, user_id: str) -> RecipeInDB:
        """Create a new recipe"""
        recipe_dict = recipe.model_dump()
        recipe_dict["created_by"] = user_id
        recipe_dict["updated_by"] = user_id
        recipe_dict["created_at"] = datetime.utcnow()
        recipe_dict["updated_at"] = datetime.utcnow()
        
        result = await self.collection.insert_one(recipe_dict)
        recipe_dict["_id"] = str(result.inserted_id)
        
        return RecipeInDB(**recipe_dict)
    
    async def get_by_id(self, recipe_id: str) -> Optional[RecipeInDB]:
        """Get recipe by ID"""
        try:
            recipe = await self.collection.find_one({"_id": ObjectId(recipe_id)})
            if recipe:
                recipe["_id"] = str(recipe["_id"])
                # Ensure camera_templates exists for backward compatibility
                if "camera_templates" not in recipe:
                    recipe["camera_templates"] = []
                return RecipeInDB(**recipe)
        except Exception:
            return None
        return None
    
    async def get_by_name(self, name: str) -> Optional[RecipeInDB]:
        """Get recipe by name"""
        recipe = await self.collection.find_one({"name": name})
        if recipe:
            recipe["_id"] = str(recipe["_id"])
            # Ensure camera_templates exists for backward compatibility
            if "camera_templates" not in recipe:
                recipe["camera_templates"] = []
            return RecipeInDB(**recipe)
        return None
    
    async def get_by_product_code(self, product_code: str) -> Optional[RecipeInDB]:
        """Get recipe by product code"""
        recipe = await self.collection.find_one({"product_code": product_code})
        if recipe:
            recipe["_id"] = str(recipe["_id"])
            # Ensure camera_templates exists for backward compatibility
            if "camera_templates" not in recipe:
                recipe["camera_templates"] = []
            return RecipeInDB(**recipe)
        return None
    
    async def get_all(
        self, 
        skip: int = 0, 
        limit: int = 100,
        is_active: Optional[bool] = None
    ) -> List[RecipeInDB]:
        """Get all recipes with pagination"""
        query = {}
        if is_active is not None:
            query["is_active"] = is_active
        
        cursor = self.collection.find(query).skip(skip).limit(limit).sort("created_at", -1)
        recipes = []
        async for recipe in cursor:
            recipe["_id"] = str(recipe["_id"])
            # Ensure camera_templates exists for backward compatibility
            if "camera_templates" not in recipe:
                recipe["camera_templates"] = []
            recipes.append(RecipeInDB(**recipe))
        
        return recipes
    
    async def update(self, recipe_id: str, recipe_update: RecipeUpdate, user_id: str) -> Optional[RecipeInDB]:
        """Update a recipe"""
        # Use exclude_none instead of exclude_unset to allow empty arrays and default values
        update_data = recipe_update.model_dump(exclude_none=True, exclude_unset=False)
        
        # Remove None values manually but keep empty arrays
        update_data = {k: v for k, v in update_data.items() if v is not None}
        
        if not update_data:
            return await self.get_by_id(recipe_id)
        
        update_data["updated_by"] = user_id
        update_data["updated_at"] = datetime.utcnow()
        
        try:
            result = await self.collection.find_one_and_update(
                {"_id": ObjectId(recipe_id)},
                {"$set": update_data},
                return_document=True
            )
            
            if result:
                result["_id"] = str(result["_id"])
                # Ensure camera_templates exists for backward compatibility
                if "camera_templates" not in result:
                    result["camera_templates"] = []
                return RecipeInDB(**result)
        except Exception as e:
            print(f"Error updating recipe: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        return None
    
    async def delete(self, recipe_id: str) -> bool:
        """Delete a recipe"""
        try:
            result = await self.collection.delete_one({"_id": ObjectId(recipe_id)})
            return result.deleted_count > 0
        except Exception:
            return False
    
    async def count(self, is_active: Optional[bool] = None) -> int:
        """Count recipes"""
        query = {}
        if is_active is not None:
            query["is_active"] = is_active
        
        return await self.collection.count_documents(query)
    
    async def search(self, query: str, skip: int = 0, limit: int = 100) -> List[RecipeInDB]:
        """Search recipes by name or product code"""
        search_query = {
            "$or": [
                {"name": {"$regex": query, "$options": "i"}},
                {"product_code": {"$regex": query, "$options": "i"}},
                {"description": {"$regex": query, "$options": "i"}}
            ]
        }
        
        cursor = self.collection.find(search_query).skip(skip).limit(limit).sort("created_at", -1)
        recipes = []
        async for recipe in cursor:
            recipe["_id"] = str(recipe["_id"])
            # Ensure camera_templates exists for backward compatibility
            if "camera_templates" not in recipe:
                recipe["camera_templates"] = []
            recipes.append(RecipeInDB(**recipe))
        
        return recipes
