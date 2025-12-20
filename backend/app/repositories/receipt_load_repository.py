from typing import Optional, Dict, Any
from datetime import datetime


class ReceiptLoadRepository:
    def __init__(self, db):
        self.db = db
        self.collection = db['receipt_loads']

    async def create(
        self,
        recipe_id: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_full_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a receipt load event and persist the loader's full name.

        Storing the full name at creation time avoids additional lookups
        from the frontend and ensures historical correctness even if
        the user's profile changes later.
        """
        doc = {
            'recipe_id': recipe_id,
            'loaded_by': user_id,
            'loaded_by_full_name': user_full_name,
            'loaded_at': datetime.utcnow(),
            'metadata': metadata or {}
        }
        result = await self.collection.insert_one(doc)
        # Normalize id/_id to strings for JSON safety
        doc['_id'] = str(result.inserted_id)
        doc['id'] = doc['_id']
        return doc

    async def get_by_id(self, load_id: str) -> Optional[Dict[str, Any]]:
        from bson.objectid import ObjectId
        doc = await self.collection.find_one({'_id': ObjectId(load_id)})
        if not doc:
            return None
        doc['_id'] = str(doc['_id'])
        doc['id'] = doc['_id']
        return doc

    async def list_by_recipe(self, recipe_id: str, skip: int = 0, limit: int = 100) -> list:
        """List load events for a given recipe id with pagination."""
        cursor = self.collection.find({'recipe_id': recipe_id}).sort('loaded_at', -1).skip(skip).limit(limit)
        docs = []
        async for doc in cursor:
            doc['_id'] = str(doc['_id'])
            doc['id'] = doc['_id']
            docs.append(doc)
        return docs

    async def list_all(self, skip: int = 0, limit: int = 100) -> list:
        """List all load events with pagination."""
        cursor = self.collection.find({}).sort('loaded_at', -1).skip(skip).limit(limit)
        docs = []
        async for doc in cursor:
            doc['_id'] = str(doc['_id'])
            doc['id'] = doc['_id']
            docs.append(doc)
        return docs

    async def count_by_recipe(self, recipe_id: str) -> int:
        return await self.collection.count_documents({'recipe_id': recipe_id})

    async def count_all(self) -> int:
        return await self.collection.count_documents({})
