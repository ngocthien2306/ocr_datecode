import traceback
from typing import Optional, Dict, Any
from datetime import datetime
from zoneinfo import ZoneInfo
from app.core.config import settings


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
        # Use configured timezone when recording timestamps (defaults to Asia/Ho_Chi_Minh)
        try:
            tz = ZoneInfo(settings.TIMEZONE)
        except Exception:
            tz = None
        loaded_at = datetime.now(tz) if tz is not None else datetime.utcnow()
        # Also store a local ISO string for easy human-readable display in DB
        loaded_at_local = loaded_at.isoformat()

        doc = {
            'recipe_id': recipe_id,
            'loaded_by': user_id,
            'loaded_by_full_name': user_full_name,
            'loaded_at': loaded_at,
            'loaded_at_local': loaded_at_local,
            'metadata': metadata or {},
            'status': 'running',  # Default status when loading recipe
            'stopped_at': None,
            'stopped_by': None,
            'stopped_by_full_name': None
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

    async def stop_recipe_load(
        self,
        recipe_id: str,
        user_id: str,
        user_full_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Stop a running recipe load by setting status to 'stopped'.
        Finds the latest running load for the given recipe_id.
        """
        from bson.objectid import ObjectId

        # Use configured timezone
        try:
            tz = ZoneInfo(settings.TIMEZONE)
        except Exception:
            tz = None
        stopped_at = datetime.now(tz) if tz is not None else datetime.utcnow()

        # Find the latest running load for this recipe
        doc = await self.collection.find_one(
            {'recipe_id': recipe_id, 'status': 'running'},
            sort=[('loaded_at', -1)]
        )

        if not doc:
            return None

        # Update status to stopped
        await self.collection.update_one(
            {'_id': doc['_id']},
            {
                '$set': {
                    'status': 'stopped',
                    'stopped_at': stopped_at,
                    'stopped_by': user_id,
                    'stopped_by_full_name': user_full_name
                }
            }
        )

        # Return updated document
        updated_doc = await self.collection.find_one({'_id': doc['_id']})
        if updated_doc:
            updated_doc['_id'] = str(updated_doc['_id'])
            updated_doc['id'] = updated_doc['_id']

        return updated_doc

    async def get_latest_running(self) -> Optional[Dict[str, Any]]:
        """Get the latest running recipe load (any recipe)."""
        doc = await self.collection.find_one(
            {'status': 'running'},
            sort=[('loaded_at', -1)]
        )

        if not doc:
            return None

        doc['_id'] = str(doc['_id'])
        doc['id'] = doc['_id']
        return doc

    async def get_template_images_at_timestamp(
        self,
        recipe_id: str,
        timestamp: datetime,
        serial_number: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get template images for a recipe at a specific timestamp.

        Finds the recipe load that was active at the given timestamp:
        - loaded_at <= timestamp
        - (stopped_at is None) OR (stopped_at > timestamp)

        Returns the recipe load metadata with camera templates.
        """
        # Build query: recipe_id matches and loaded_at <= timestamp
        query = {
            'recipe_id': recipe_id,
            'loaded_at': {'$lte': timestamp}
        }

        # Find the latest load before or at the timestamp
        # Sort by loaded_at descending to get the most recent
        doc = await self.collection.find_one(
            query,
            sort=[('loaded_at', -1)]
        )

        if not doc:
            return None

        # Verify this load was still active at the given timestamp
        stopped_at = doc.get('stopped_at')
        if stopped_at is not None and stopped_at <= timestamp:
            # This load was already stopped before the timestamp
            return None

        # Normalize IDs
        doc['_id'] = str(doc['_id'])
        doc['id'] = doc['_id']

        return doc
