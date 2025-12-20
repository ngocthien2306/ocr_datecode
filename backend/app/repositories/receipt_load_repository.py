from typing import Optional, Dict, Any
from datetime import datetime


class ReceiptLoadRepository:
    def __init__(self, db):
        self.db = db
        self.collection = db['receipt_loads']

    async def create(self, recipe_id: str, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        doc = {
            'recipe_id': recipe_id,
            'loaded_by': user_id,
            'loaded_at': datetime.utcnow(),
            'metadata': metadata or {}
        }
        result = await self.collection.insert_one(doc)
        doc['id'] = str(result.inserted_id)
        return doc

    async def get_by_id(self, load_id: str) -> Optional[Dict[str, Any]]:
        from bson.objectid import ObjectId
        doc = await self.collection.find_one({'_id': ObjectId(load_id)})
        if not doc:
            return None
        doc['id'] = str(doc['_id'])
        return doc
