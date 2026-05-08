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
    MLCharImportBatchInDB, MLCharImportInDB,
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
        self.char_import_batches = database.get_collection("ml_char_import_batches")
        self.char_imports = database.get_collection("ml_char_imports")

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
        await self.char_imports.delete_many({"project_id": project_id})
        await self.char_import_batches.delete_many({"project_id": project_id})
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

    # ─────────────────────────────── Char Imports ────────────────────

    async def create_char_import_batch(self, project_id: str, name: str) -> MLCharImportBatchInDB:
        now = datetime.utcnow()
        doc = {
            "project_id": project_id,
            "name": name,
            "created_at": now,
        }
        result = await self.char_import_batches.insert_one(doc)
        doc["_id"] = str(result.inserted_id)
        return MLCharImportBatchInDB(**doc)

    async def list_char_import_batches(self, project_id: str) -> List[MLCharImportBatchInDB]:
        cursor = self.char_import_batches.find({"project_id": project_id}).sort("created_at", -1)
        out = []
        async for doc in cursor:
            out.append(MLCharImportBatchInDB(**_to_str_id(doc)))
        return out

    async def get_char_import_batch(self, batch_id: str) -> Optional[MLCharImportBatchInDB]:
        doc = await self.char_import_batches.find_one({"_id": ObjectId(batch_id)})
        return MLCharImportBatchInDB(**_to_str_id(doc)) if doc else None

    async def rename_char_import_batch(self, batch_id: str, name: str) -> Optional[MLCharImportBatchInDB]:
        doc = await self.char_import_batches.find_one_and_update(
            {"_id": ObjectId(batch_id)},
            {"$set": {"name": name}},
            return_document=True,
        )
        return MLCharImportBatchInDB(**_to_str_id(doc)) if doc else None

    async def delete_char_import_batch(self, batch_id: str) -> List[str]:
        """Delete a batch and all its chars. Returns list of crop_path strings to unlink."""
        crop_paths = []
        async for doc in self.char_imports.find({"batch_id": batch_id}, {"crop_path": 1}):
            cp = doc.get("crop_path")
            if cp:
                crop_paths.append(cp)
        await self.char_imports.delete_many({"batch_id": batch_id})
        await self.char_import_batches.delete_one({"_id": ObjectId(batch_id)})
        return crop_paths

    async def insert_char_import(self, doc: Dict[str, Any]) -> str:
        result = await self.char_imports.insert_one(doc)
        return str(result.inserted_id)

    async def list_char_imports(self, project_id: str,
                                batch_id: Optional[str] = None,
                                label: Optional[str] = None) -> List[MLCharImportInDB]:
        query: Dict[str, Any] = {"project_id": project_id}
        if batch_id:
            query["batch_id"] = batch_id
        if label:
            query["label"] = label
        cursor = self.char_imports.find(query).sort("created_at", -1)
        out = []
        async for doc in cursor:
            out.append(MLCharImportInDB(**_to_str_id(doc)))
        return out

    async def get_imported_provenance_keys(self, project_id: str) -> Dict[str, str]:
        """Return {f"{inspection_id}:{annotation_idx}": batch_id} for all imports in project.
        Used to dedup the inspection-candidates list."""
        out: Dict[str, str] = {}
        cursor = self.char_imports.find(
            {"project_id": project_id},
            {"inspection_id": 1, "annotation_idx": 1, "batch_id": 1},
        )
        async for doc in cursor:
            key = f"{doc.get('inspection_id')}:{doc.get('annotation_idx')}"
            out[key] = str(doc.get("batch_id", ""))
        return out

    async def update_char_import(self, char_id: str, update: Dict[str, Any]) -> Optional[MLCharImportInDB]:
        update["updated_at"] = datetime.utcnow()
        doc = await self.char_imports.find_one_and_update(
            {"_id": ObjectId(char_id)},
            {"$set": update},
            return_document=True,
        )
        return MLCharImportInDB(**_to_str_id(doc)) if doc else None

    async def get_char_import(self, char_id: str) -> Optional[MLCharImportInDB]:
        doc = await self.char_imports.find_one({"_id": ObjectId(char_id)})
        return MLCharImportInDB(**_to_str_id(doc)) if doc else None

    async def delete_char_import(self, char_id: str) -> Optional[str]:
        """Delete one char doc. Returns its crop_path so the caller can unlink."""
        doc = await self.char_imports.find_one_and_delete({"_id": ObjectId(char_id)})
        return doc.get("crop_path") if doc else None

    async def bulk_update_char_imports(self, char_ids: List[str], update: Dict[str, Any]) -> int:
        if not char_ids:
            return 0
        update["updated_at"] = datetime.utcnow()
        oids = [ObjectId(c) for c in char_ids]
        result = await self.char_imports.update_many(
            {"_id": {"$in": oids}},
            {"$set": update},
        )
        return result.modified_count

    async def bulk_delete_char_imports(self, char_ids: List[str]) -> List[str]:
        """Delete many char docs. Returns crop_paths to unlink."""
        if not char_ids:
            return []
        oids = [ObjectId(c) for c in char_ids]
        crop_paths = []
        async for doc in self.char_imports.find({"_id": {"$in": oids}}, {"crop_path": 1}):
            cp = doc.get("crop_path")
            if cp:
                crop_paths.append(cp)
        await self.char_imports.delete_many({"_id": {"$in": oids}})
        return crop_paths

    async def count_char_imports_by_batch(self, project_id: str) -> Dict[str, Dict[str, int]]:
        """Aggregate count of OK/NG per batch_id for a project."""
        pipeline = [
            {"$match": {"project_id": project_id}},
            {"$group": {
                "_id": {"batch_id": "$batch_id", "label": "$label"},
                "n": {"$sum": 1},
            }},
        ]
        out: Dict[str, Dict[str, int]] = {}
        async for row in self.char_imports.aggregate(pipeline):
            bid = row["_id"]["batch_id"]
            label = row["_id"]["label"]
            slot = out.setdefault(bid, {"OK": 0, "NG": 0})
            slot[label] = row["n"]
        return out

    async def create_indexes(self):
        await self.projects.create_index([("created_at", -1)])
        await self.annotations.create_index([("project_id", 1), ("filename", 1)], unique=True)
        await self.models.create_index([("project_id", 1), ("created_at", -1)])
        await self.char_import_batches.create_index([("project_id", 1), ("created_at", -1)])
        await self.char_imports.create_index([("project_id", 1), ("batch_id", 1)])
        await self.char_imports.create_index([
            ("project_id", 1), ("inspection_id", 1), ("annotation_idx", 1),
        ])
