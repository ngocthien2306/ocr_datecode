"""
OCR Repository for the Paddle training service.

Projects and dataset items belong to ocr_service (:8002) — labelling happens
there — and are only READ here. Every write in this file goes to this
service's own model collection (settings.MODELS_COLLECTION). Keep it that way:
a write to the shared collections from here would race ocr_service's own
bookkeeping (project status, item counts) with nothing to arbitrate.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId

from app.core.config import settings
from app.models.ocr import OCRDatasetItemInDB, OCRModelInDB, OCRProjectInDB


def _to_str_id(doc: dict) -> dict:
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


class OCRRepository:
    def __init__(self, database):
        self.projects = database.get_collection(settings.PROJECTS_COLLECTION)
        self.items = database.get_collection(settings.ITEMS_COLLECTION)
        self.models = database.get_collection(settings.MODELS_COLLECTION)

    # ─────────────────────────── Shared, read-only ───────────────────────

    async def list_projects(self) -> List[OCRProjectInDB]:
        cursor = self.projects.find().sort("created_at", -1)
        return [OCRProjectInDB(**_to_str_id(doc)) async for doc in cursor]

    async def get_project(self, project_id: str) -> Optional[OCRProjectInDB]:
        if not ObjectId.is_valid(project_id):
            return None
        doc = await self.projects.find_one({"_id": ObjectId(project_id)})
        return OCRProjectInDB(**_to_str_id(doc)) if doc else None

    async def list_trainable_items(self, project_id: str) -> List[OCRDatasetItemInDB]:
        """Verified, not excluded, ordered by _id — the same query ocr_service
        trains on, so both services split the same project identically."""
        cursor = self.items.find({
            "project_id": project_id,
            "status": "verified",
            "exclude_from_training": {"$ne": True},
        }).sort("_id", 1)
        return [OCRDatasetItemInDB(**_to_str_id(doc)) async for doc in cursor]

    # ─────────────────────────────── Models ──────────────────────────────

    async def create_model_record(self, project_id: str, arch: str, params: Dict[str, Any],
                                  base_label: str) -> OCRModelInDB:
        doc = {
            "project_id": project_id,
            "family": "paddle_ctc",
            "arch": arch,
            "params": params,
            "base_label": base_label,
            "metrics": {},
            "checkpoint_path": "",
            "status": "pending",
            "error": None,
            "phase": None,
            "progress": 0.0,
            "created_at": datetime.utcnow(),
        }
        result = await self.models.insert_one(doc)
        doc["_id"] = str(result.inserted_id)
        return OCRModelInDB(**doc)

    async def update_model_record(self, model_id: str, update: Dict[str, Any]) -> None:
        await self.models.update_one({"_id": ObjectId(model_id)}, {"$set": update})

    async def list_models(self, project_id: str) -> List[OCRModelInDB]:
        cursor = self.models.find({"project_id": project_id}).sort("created_at", -1)
        return [OCRModelInDB(**_to_str_id(doc)) async for doc in cursor]

    async def get_model(self, model_id: str) -> Optional[OCRModelInDB]:
        if not ObjectId.is_valid(model_id):
            return None
        doc = await self.models.find_one({"_id": ObjectId(model_id)})
        return OCRModelInDB(**_to_str_id(doc)) if doc else None

    async def delete_model(self, model_id: str) -> bool:
        if not ObjectId.is_valid(model_id):
            return False
        result = await self.models.delete_one({"_id": ObjectId(model_id)})
        return result.deleted_count > 0

    async def list_completed_models(self) -> List[OCRModelInDB]:
        cursor = self.models.find({"status": "completed"}).sort("created_at", -1)
        return [OCRModelInDB(**_to_str_id(doc)) async for doc in cursor]

    async def reset_stuck_training(self) -> int:
        """Fail runs orphaned by a restart (the background task died with the
        process, so nothing would ever move them off 'training')."""
        result = await self.models.update_many(
            {"status": {"$in": ("training", "pending")}},
            {"$set": {
                "status": "failed",
                "error": "Interrupted by an ocr_paddle_service restart",
                "phase": "failed",
            }},
        )
        return result.modified_count

    async def create_indexes(self) -> None:
        # Own collection only — ocr_service owns the indexes on the shared ones.
        await self.models.create_index([("project_id", 1), ("created_at", -1)])
        await self.models.create_index([("status", 1), ("created_at", -1)])
