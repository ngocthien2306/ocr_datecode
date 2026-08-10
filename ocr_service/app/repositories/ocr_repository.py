"""
OCR Repository — async MongoDB CRUD for the three collections this service
owns: ocr_projects, ocr_dataset_items, ocr_models.

Shaped after anomaly_service/app/repositories/anomaly_repository.py. Only the
project + counting methods are filled in at this step; dataset-item and model
methods land with the endpoints that use them (see docs/ocr_training_plan.md §7).
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from app.models.ocr import (
    OCRDatasetItemInDB,
    OCRModelInDB,
    OCRProjectCreate,
    OCRProjectInDB,
    OCRProjectUpdate,
)


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

    # ─────────────────────────────── Dataset items ───────────────────────

    async def get_imported_provenance_keys(self, project_id: str) -> Dict[str, str]:
        """{composite_key: status} for every item imported from an inspection.

        composite_key = "{inspection_id}:{camera_serial}:{frame_idx}:{annotation_index}"
        — annotation_index included because one frame can hold several OCR
        regions. Lets the candidates grid grey out what is already imported
        without a query per row.
        """
        out: Dict[str, str] = {}
        cursor = self.items.find(
            {"project_id": project_id, "inspection_id": {"$type": "string"}},
            {"inspection_id": 1, "camera_serial": 1, "frame_idx": 1,
             "annotation_index": 1, "status": 1},
        )
        async for doc in cursor:
            key = (f"{doc.get('inspection_id')}:{doc.get('camera_serial', '')}"
                   f":{doc.get('frame_idx', 0)}:{doc.get('annotation_index', 0)}")
            out[key] = doc.get("status", "")
        return out

    async def insert_item(self, doc: Dict[str, Any]) -> Optional[str]:
        """Insert one dataset item. Returns None when the unique import_dedup
        index rejects it as a duplicate, so callers can count it as skipped
        rather than failing the whole batch."""
        try:
            result = await self.items.insert_one(doc)
        except DuplicateKeyError:
            return None
        return str(result.inserted_id)

    async def list_items_page(
        self,
        project_id: str,
        status: Optional[str],
        split: Optional[str],
        skip: int,
        limit: int,
    ) -> Tuple[List[OCRDatasetItemInDB], int]:
        """One page of items (newest first) + the total matching count.
        Filtered server-side so thumbnail generation only reads the images on
        the page actually being shown."""
        query = self._item_query(project_id, status, split)
        total = await self.items.count_documents(query)
        cursor = self.items.find(query).sort("created_at", -1).skip(skip).limit(limit)
        items = [OCRDatasetItemInDB(**_to_str_id(doc)) async for doc in cursor]
        return items, total

    async def list_item_ids(
        self, project_id: str, status: Optional[str], split: Optional[str],
    ) -> List[str]:
        """Ids only — lets "select all" in the Label tab span pages without
        pulling every page's base64 thumbnails."""
        cursor = self.items.find(
            self._item_query(project_id, status, split), {"_id": 1}
        ).sort("created_at", -1)
        return [str(doc["_id"]) async for doc in cursor]

    @staticmethod
    def _item_query(project_id: str, status: Optional[str], split: Optional[str]) -> Dict[str, Any]:
        query: Dict[str, Any] = {"project_id": project_id}
        if status:
            query["status"] = status
        if split:
            query["split"] = split
        return query

    async def get_item(self, item_id: str) -> Optional[OCRDatasetItemInDB]:
        if not ObjectId.is_valid(item_id):
            return None
        doc = await self.items.find_one({"_id": ObjectId(item_id)})
        return OCRDatasetItemInDB(**_to_str_id(doc)) if doc else None

    async def update_item(self, item_id: str, update: Dict[str, Any]) -> bool:
        if not ObjectId.is_valid(item_id):
            return False
        update["updated_at"] = datetime.utcnow()
        result = await self.items.update_one({"_id": ObjectId(item_id)}, {"$set": update})
        return result.matched_count > 0

    async def update_items(
        self, project_id: str, item_ids: List[str], update: Dict[str, Any],
    ) -> int:
        valid = [ObjectId(i) for i in item_ids if ObjectId.is_valid(i)]
        if not valid:
            return 0
        update["updated_at"] = datetime.utcnow()
        result = await self.items.update_many(
            {"_id": {"$in": valid}, "project_id": project_id}, {"$set": update},
        )
        return result.modified_count

    async def get_items(self, project_id: str, item_ids: List[str]) -> List[OCRDatasetItemInDB]:
        valid = [ObjectId(i) for i in item_ids if ObjectId.is_valid(i)]
        if not valid:
            return []
        cursor = self.items.find({"_id": {"$in": valid}, "project_id": project_id})
        return [OCRDatasetItemInDB(**_to_str_id(doc)) async for doc in cursor]

    async def delete_items(self, project_id: str, item_ids: List[str]) -> List[OCRDatasetItemInDB]:
        """Delete rows and return what was deleted, so the caller can remove
        the image files too (the row is the only pointer to them)."""
        items = await self.get_items(project_id, item_ids)
        if items:
            await self.items.delete_many(
                {"_id": {"$in": [ObjectId(i.id) for i in items]}, "project_id": project_id}
            )
        return items

    async def list_trainable_items(self, project_id: str) -> List[OCRDatasetItemInDB]:
        """Items a run will actually see: verified and not excluded.

        Ordered by _id so the train/test split derived from this list is stable
        across runs — a shuffled order would silently reshuffle the eval set
        between two runs on the same data and make their metrics incomparable.
        """
        cursor = self.items.find({
            "project_id": project_id,
            "status": "verified",
            "exclude_from_training": {"$ne": True},
        }).sort("_id", 1)
        return [OCRDatasetItemInDB(**_to_str_id(doc)) async for doc in cursor]

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

    # ─────────────────────────────── Models ──────────────────────────────

    async def create_model_record(self, project_id: str, params: Dict[str, Any], base_label: str,
                                  use_space_char: bool) -> OCRModelInDB:
        now = datetime.utcnow()
        doc = {
            "project_id": project_id,
            "params": params,
            "base_label": base_label,
            "use_space_char": use_space_char,
            # Provisional: the real value is read back off the trained
            # checkpoint, since a vocab expansion can change it mid-run.
            "vocab_size": 100 if use_space_char else 99,
            "metrics": {},
            "checkpoint_path": "",
            "status": "pending",
            "error": None,
            "phase": None,
            "progress": 0.0,
            "created_at": now,
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
        """Every completed model across ALL projects — feeds the base-checkpoint
        picker (a broad project's model can seed a narrow one) and, later, the
        recipe model dropdown."""
        cursor = self.models.find({"status": "completed"}).sort("created_at", -1)
        return [OCRModelInDB(**_to_str_id(doc)) async for doc in cursor]

    async def reset_stuck_training(self) -> int:
        """Fail any model left in training/pending by a service restart.

        Training runs in a background task, so a restart mid-run orphans the
        record: nothing will ever move it off 'training' and the project stays
        stuck showing a run that no longer exists.
        """
        result = await self.models.update_many(
            {"status": {"$in": ("training", "pending")}},
            {"$set": {
                "status": "failed",
                "error": "Interrupted by an ocr_service restart",
                "phase": "failed",
            }},
        )
        if result.modified_count:
            await self.projects.update_many(
                {"status": "training"}, {"$set": {"status": "active"}},
            )
        return result.modified_count

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
