"""
OCR Repository — async MongoDB CRUD for the three collections this service
owns: ocr_projects, ocr_dataset_items, ocr_models.

Shaped after anomaly_service/app/repositories/anomaly_repository.py. Only the
project + counting methods are filled in at this step; dataset-item and model
methods land with the endpoints that use them (see docs/ocr_training_plan.md §7).
"""
from datetime import datetime
from typing import Dict, List, Optional

from bson import ObjectId

from app.models.ocr import OCRProjectCreate, OCRProjectInDB, OCRProjectUpdate


def _to_str_id(doc: dict) -> dict:
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


class OCRRepository:
    def __init__(self, database):
        self.projects = database.get_collection("ocr_projects")
        self.items = database.get_collection("ocr_dataset_items")
        self.models = database.get_collection("ocr_models")

    # ─────────────────────────────── Projects ────────────────────────────

    async def create_project(self, data: OCRProjectCreate, user: str) -> OCRProjectInDB:
        now = datetime.utcnow()
        doc = {
            "name": data.name,
            "description": data.description,
            "created_at": now,
            "updated_at": now,
            "created_by": user,
            "total_count": 0,
            "verified_count": 0,
            "need_review_count": 0,
            "status": "active",
        }
        result = await self.projects.insert_one(doc)
        doc["_id"] = str(result.inserted_id)
        return OCRProjectInDB(**doc)

    async def list_projects(self) -> List[OCRProjectInDB]:
        cursor = self.projects.find().sort("created_at", -1)
        return [OCRProjectInDB(**_to_str_id(doc)) async for doc in cursor]

    async def get_project(self, project_id: str) -> Optional[OCRProjectInDB]:
        if not ObjectId.is_valid(project_id):
            return None
        doc = await self.projects.find_one({"_id": ObjectId(project_id)})
        return OCRProjectInDB(**_to_str_id(doc)) if doc else None

    async def update_project(self, project_id: str, data: OCRProjectUpdate) -> Optional[OCRProjectInDB]:
        if not ObjectId.is_valid(project_id):
            return None
        update = {k: v for k, v in data.model_dump().items() if v is not None}
        update["updated_at"] = datetime.utcnow()
        doc = await self.projects.find_one_and_update(
            {"_id": ObjectId(project_id)},
            {"$set": update},
            return_document=True,
        )
        return OCRProjectInDB(**_to_str_id(doc)) if doc else None

    async def delete_project(self, project_id: str) -> bool:
        if not ObjectId.is_valid(project_id):
            return False
        result = await self.projects.delete_one({"_id": ObjectId(project_id)})
        await self.items.delete_many({"project_id": project_id})
        await self.models.delete_many({"project_id": project_id})
        return result.deleted_count > 0

    async def set_status(self, project_id: str, status: str) -> None:
        await self.projects.update_one(
            {"_id": ObjectId(project_id)},
            {"$set": {"status": status, "updated_at": datetime.utcnow()}},
        )

    # ─────────────────────────────── Counts ──────────────────────────────

    async def count_items_by_status(self, project_id: str) -> Dict[str, int]:
        """Item counts per label status. Only 'verified' items are trainable,
        which is why the project doc tracks that separately from the total."""
        out = {"total": 0, "need_review": 0, "verified": 0, "rejected": 0}
        pipeline = [
            {"$match": {"project_id": project_id}},
            {"$group": {"_id": "$status", "n": {"$sum": 1}}},
        ]
        async for row in self.items.aggregate(pipeline):
            if row["_id"] in out:
                out[row["_id"]] = row["n"]
            out["total"] += row["n"]
        return out

    async def sync_project_counts(self, project_id: str) -> Dict[str, int]:
        """Recount items and write the result onto the project doc. Cheap
        enough to call on every page load — it is one aggregate over an
        indexed field."""
        counts = await self.count_items_by_status(project_id)
        await self.projects.update_one(
            {"_id": ObjectId(project_id)},
            {"$set": {
                "total_count": counts["total"],
                "verified_count": counts["verified"],
                "need_review_count": counts["need_review"],
                "updated_at": datetime.utcnow(),
            }},
        )
        return counts

    # ─────────────────────────────── Indexes ─────────────────────────────

    async def create_indexes(self) -> None:
        await self.projects.create_index([("created_at", -1)])
        await self.items.create_index([("project_id", 1), ("created_at", -1)])
        await self.items.create_index([("project_id", 1), ("status", 1)])
        await self.items.create_index([("project_id", 1), ("split", 1)])
        # Import dedup: one frame can hold several OCR regions, so
        # annotation_index is part of the key (anomaly's label crops are one
        # per frame and omit it). Partial so folder-seeded and uploaded items,
        # which have no provenance at all, don't collide on a row of nulls.
        await self.items.create_index(
            [("project_id", 1), ("inspection_id", 1), ("camera_serial", 1),
             ("frame_idx", 1), ("annotation_index", 1)],
            name="import_dedup",
            unique=True,
            partialFilterExpression={"inspection_id": {"$type": "string"}},
        )
        await self.models.create_index([("project_id", 1), ("created_at", -1)])
        # Feeds GET /api/ocr/base-checkpoints and the recipe model picker, both
        # of which query completed models across every project.
        await self.models.create_index([("status", 1), ("created_at", -1)])
