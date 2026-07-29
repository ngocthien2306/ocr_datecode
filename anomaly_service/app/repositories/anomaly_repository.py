"""
Anomaly Repository
Async MongoDB CRUD for anomaly_projects, anomaly_import_items, anomaly_models
collections. Separate database from backend's ml_* collections — this
service owns these three.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId

from app.models.anomaly import (
    AnomalyImportItemInDB,
    AnomalyModelInDB,
    AnomalyProjectCreate,
    AnomalyProjectInDB,
    AnomalyProjectUpdate,
)


def _to_str_id(doc: dict) -> dict:
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


class AnomalyRepository:
    def __init__(self, database):
        self.projects = database.get_collection("anomaly_projects")
        self.import_items = database.get_collection("anomaly_import_items")
        self.models = database.get_collection("anomaly_models")

    # ─────────────────────────────── Projects ────────────────────────

    async def create_project(self, data: AnomalyProjectCreate, user: str) -> AnomalyProjectInDB:
        now = datetime.utcnow()
        doc = {
            "name": data.name,
            "description": data.description,
            "created_at": now,
            "updated_at": now,
            "created_by": user,
            "normal_count": 0,
            "abnormal_count": 0,
            "status": "active",
        }
        result = await self.projects.insert_one(doc)
        doc["_id"] = str(result.inserted_id)
        return AnomalyProjectInDB(**doc)

    async def list_projects(self) -> List[AnomalyProjectInDB]:
        cursor = self.projects.find().sort("created_at", -1)
        return [AnomalyProjectInDB(**_to_str_id(doc)) async for doc in cursor]

    async def get_project(self, project_id: str) -> Optional[AnomalyProjectInDB]:
        doc = await self.projects.find_one({"_id": ObjectId(project_id)})
        return AnomalyProjectInDB(**_to_str_id(doc)) if doc else None

    async def update_project(self, project_id: str, data: AnomalyProjectUpdate) -> Optional[AnomalyProjectInDB]:
        update = {k: v for k, v in data.model_dump().items() if v is not None}
        update["updated_at"] = datetime.utcnow()
        doc = await self.projects.find_one_and_update(
            {"_id": ObjectId(project_id)},
            {"$set": update},
            return_document=True,
        )
        return AnomalyProjectInDB(**_to_str_id(doc)) if doc else None

    async def delete_project(self, project_id: str) -> bool:
        result = await self.projects.delete_one({"_id": ObjectId(project_id)})
        await self.import_items.delete_many({"project_id": project_id})
        await self.models.delete_many({"project_id": project_id})
        return result.deleted_count > 0

    async def set_counts(self, project_id: str, normal_count: int, abnormal_count: int):
        await self.projects.update_one(
            {"_id": ObjectId(project_id)},
            {"$set": {
                "normal_count": normal_count,
                "abnormal_count": abnormal_count,
                "updated_at": datetime.utcnow(),
            }},
        )

    async def set_status(self, project_id: str, status: str):
        await self.projects.update_one(
            {"_id": ObjectId(project_id)},
            {"$set": {"status": status, "updated_at": datetime.utcnow()}},
        )

    # ─────────────────────────────── Import provenance ───────────────

    async def get_imported_provenance_keys(self, project_id: str) -> Dict[str, str]:
        """Return {composite_key: split} for all imports in project.

        composite_key = f"{inspection_id}:{camera_serial}:{frame_idx}"
        A recipe run produces at most one label crop per frame, so this key
        (unlike char imports) needs no annotation_idx component.
        """
        out: Dict[str, str] = {}
        cursor = self.import_items.find(
            {"project_id": project_id},
            {"inspection_id": 1, "camera_serial": 1, "frame_idx": 1, "split": 1},
        )
        async for doc in cursor:
            key = f"{doc.get('inspection_id')}:{doc.get('camera_serial', '')}:{doc.get('frame_idx', 0)}"
            out[key] = doc.get("split", "")
        return out

    async def insert_import_item(self, doc: Dict[str, Any]) -> str:
        result = await self.import_items.insert_one(doc)
        return str(result.inserted_id)

    async def list_import_items(self, project_id: str) -> List[AnomalyImportItemInDB]:
        cursor = self.import_items.find({"project_id": project_id}).sort("created_at", -1)
        return [AnomalyImportItemInDB(**_to_str_id(doc)) async for doc in cursor]

    async def list_import_items_page(
        self, project_id: str, label: Optional[str], skip: int, limit: int,
    ) -> tuple[List[AnomalyImportItemInDB], int]:
        """One page of import items (newest first) + total matching count,
        for the dataset gallery. Filters server-side so thumbnail generation
        (disk reads) only happens for the page actually being shown."""
        query: Dict[str, Any] = {"project_id": project_id}
        if label:
            query["label"] = label
        total = await self.import_items.count_documents(query)
        cursor = self.import_items.find(query).sort("created_at", -1).skip(skip).limit(limit)
        items = [AnomalyImportItemInDB(**_to_str_id(doc)) async for doc in cursor]
        return items, total

    async def get_import_item(self, item_id: str) -> Optional[AnomalyImportItemInDB]:
        if not ObjectId.is_valid(item_id):
            return None
        doc = await self.import_items.find_one({"_id": ObjectId(item_id)})
        return AnomalyImportItemInDB(**_to_str_id(doc)) if doc else None

    async def get_import_items(self, project_id: str, item_ids: List[str]) -> List[AnomalyImportItemInDB]:
        valid_ids = [ObjectId(i) for i in item_ids if ObjectId.is_valid(i)]
        if not valid_ids:
            return []
        cursor = self.import_items.find({"_id": {"$in": valid_ids}, "project_id": project_id})
        return [AnomalyImportItemInDB(**_to_str_id(doc)) async for doc in cursor]

    async def update_import_item(self, item_id: str, update: Dict[str, Any]) -> None:
        await self.import_items.update_one({"_id": ObjectId(item_id)}, {"$set": update})

    async def delete_import_item(self, item_id: str) -> bool:
        result = await self.import_items.delete_one({"_id": ObjectId(item_id)})
        return result.deleted_count > 0

    async def count_import_items(self, project_id: str) -> Dict[str, int]:
        pipeline = [
            {"$match": {"project_id": project_id}},
            {"$group": {"_id": "$label", "n": {"$sum": 1}}},
        ]
        out = {"normal": 0, "abnormal": 0}
        async for row in self.import_items.aggregate(pipeline):
            if row["_id"] in out:
                out[row["_id"]] = row["n"]
        return out

    # ─────────────────────────────── Models ───────────────────────────

    async def create_model_record(self, project_id: str, algorithm: str, params: Dict) -> AnomalyModelInDB:
        now = datetime.utcnow()
        doc = {
            "project_id": project_id,
            "algorithm": algorithm,
            "params": params,
            "metrics": {},
            "checkpoint_path": "",
            "onnx_path": None,
            "status": "pending",
            "error": None,
            "created_at": now,
        }
        result = await self.models.insert_one(doc)
        doc["_id"] = str(result.inserted_id)
        return AnomalyModelInDB(**doc)

    async def update_model_record(self, model_id: str, update: Dict[str, Any]):
        await self.models.update_one({"_id": ObjectId(model_id)}, {"$set": update})

    async def list_models(self, project_id: str) -> List[AnomalyModelInDB]:
        cursor = self.models.find({"project_id": project_id}).sort("created_at", -1)
        return [AnomalyModelInDB(**_to_str_id(doc)) async for doc in cursor]

    async def get_model(self, model_id: str) -> Optional[AnomalyModelInDB]:
        doc = await self.models.find_one({"_id": ObjectId(model_id)})
        return AnomalyModelInDB(**_to_str_id(doc)) if doc else None

    async def create_indexes(self):
        await self.projects.create_index([("created_at", -1)])
        await self.import_items.create_index([("project_id", 1), ("created_at", -1)])
        await self.import_items.create_index(
            [("project_id", 1), ("inspection_id", 1), ("camera_serial", 1), ("frame_idx", 1)],
        )
        await self.models.create_index([("project_id", 1), ("created_at", -1)])
