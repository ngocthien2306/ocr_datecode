"""
ML Training Repository
Async MongoDB CRUD for ml_projects, ml_annotations, ml_models collections.
"""
from typing import Any, Dict, List, Optional
from datetime import datetime
from bson import ObjectId

from app.models.ml_training import (
    MLProjectCreate, MLProjectUpdate, MLProjectInDB,
    MLAnnotationSave, MLAnnotationInDB,
    MLModelInDB,
)


def _to_str_id(doc: dict) -> dict:
    """Convert ObjectId _id to str."""
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


class MLTrainingRepository:
    def __init__(self, database):
        self.projects = database.get_collection("ml_projects")
        self.annotations = database.get_collection("ml_annotations")
        self.models = database.get_collection("ml_models")

    # ─────────────────────────────── Projects ────────────────────────

    async def create_project(self, data: MLProjectCreate, user_id: str) -> MLProjectInDB:
        now = datetime.utcnow()
        doc = {
            "name": data.name,
            "description": data.description,
            "created_at": now,
            "updated_at": now,
            "created_by": user_id,
            "image_count": 0,
            "labeled_count": 0,
            "status": "active",
        }
        result = await self.projects.insert_one(doc)
        doc["_id"] = str(result.inserted_id)
        return MLProjectInDB(**doc)

    async def list_projects(self) -> List[MLProjectInDB]:
        cursor = self.projects.find().sort("created_at", -1)
        out = []
        async for doc in cursor:
            out.append(MLProjectInDB(**_to_str_id(doc)))
        return out

    async def get_project(self, project_id: str) -> Optional[MLProjectInDB]:
        doc = await self.projects.find_one({"_id": ObjectId(project_id)})
        return MLProjectInDB(**_to_str_id(doc)) if doc else None

    async def update_project(self, project_id: str, data: MLProjectUpdate) -> Optional[MLProjectInDB]:
        update = {k: v for k, v in data.model_dump().items() if v is not None}
        update["updated_at"] = datetime.utcnow()
        doc = await self.projects.find_one_and_update(
            {"_id": ObjectId(project_id)},
            {"$set": update},
            return_document=True,
        )
        return MLProjectInDB(**_to_str_id(doc)) if doc else None

    async def delete_project(self, project_id: str) -> bool:
        result = await self.projects.delete_one({"_id": ObjectId(project_id)})
        # Clean up related data
        await self.annotations.delete_many({"project_id": project_id})
        await self.models.delete_many({"project_id": project_id})
        return result.deleted_count > 0

    async def set_image_count(self, project_id: str, count: int):
        await self.projects.update_one(
            {"_id": ObjectId(project_id)},
            {"$set": {"image_count": count, "updated_at": datetime.utcnow()}},
        )

    async def refresh_labeled_count(self, project_id: str):
        """Count images that have at least one labeled segment."""
        count = await self.annotations.count_documents({
            "project_id": project_id,
            "regions.segments.label": {"$in": ["OK", "NG"]},
        })
        await self.projects.update_one(
            {"_id": ObjectId(project_id)},
            {"$set": {"labeled_count": count, "updated_at": datetime.utcnow()}},
        )

    async def set_status(self, project_id: str, status: str):
        await self.projects.update_one(
            {"_id": ObjectId(project_id)},
            {"$set": {"status": status, "updated_at": datetime.utcnow()}},
        )

    # ─────────────────────────────── Annotations ─────────────────────

    async def get_annotation(self, project_id: str, filename: str) -> Optional[MLAnnotationInDB]:
        doc = await self.annotations.find_one({"project_id": project_id, "filename": filename})
        return MLAnnotationInDB(**_to_str_id(doc)) if doc else None

    async def save_annotation(self, project_id: str, filename: str, data: MLAnnotationSave) -> MLAnnotationInDB:
        now = datetime.utcnow()
        regions_raw = [r.model_dump() for r in data.regions]
        doc = await self.annotations.find_one_and_update(
            {"project_id": project_id, "filename": filename},
            {"$set": {"regions": regions_raw, "updated_at": now}},
            upsert=True,
            return_document=True,
        )
        return MLAnnotationInDB(**_to_str_id(doc))

    async def list_annotations(self, project_id: str) -> List[MLAnnotationInDB]:
        cursor = self.annotations.find({"project_id": project_id})
        out = []
        async for doc in cursor:
            out.append(MLAnnotationInDB(**_to_str_id(doc)))
        return out

    # ─────────────────────────────── Models ──────────────────────────

    async def create_model_record(self, project_id: str, algorithm: str,
                                  params: Dict, augment_factor: int) -> MLModelInDB:
        now = datetime.utcnow()
        doc = {
            "project_id": project_id,
            "algorithm": algorithm,
            "params": params,
            "augment_factor": augment_factor,
            "metrics": {},
            "model_path": "",
            "status": "pending",
            "error": None,
            "created_at": now,
        }
        result = await self.models.insert_one(doc)
        doc["_id"] = str(result.inserted_id)
        return MLModelInDB(**doc)

    async def update_model_record(self, model_id: str, update: Dict[str, Any]):
        await self.models.update_one(
            {"_id": ObjectId(model_id)},
            {"$set": update},
        )

    async def list_models(self, project_id: str) -> List[MLModelInDB]:
        cursor = self.models.find({"project_id": project_id}).sort("created_at", -1)
        out = []
        async for doc in cursor:
            out.append(MLModelInDB(**_to_str_id(doc)))
        return out

    async def get_latest_model(self, project_id: str) -> Optional[MLModelInDB]:
        doc = await self.models.find_one(
            {"project_id": project_id, "status": "completed"},
            sort=[("created_at", -1)],
        )
        return MLModelInDB(**_to_str_id(doc)) if doc else None

    async def create_indexes(self):
        await self.projects.create_index([("created_at", -1)])
        await self.annotations.create_index([("project_id", 1), ("filename", 1)], unique=True)
        await self.models.create_index([("project_id", 1), ("created_at", -1)])
