"""Which inspection recipes point at an OCR project or trained model.

Recipes live in the backend's `recipes` collection, in the same Mongo database
this service uses. A recipe depends on a trained model only when its
ocr_model_type is CUSTOM (SMTR) or CUSTOM_PADDLE; ids left on a recipe with a
built-in type are stale and ignored, the same way the backend ignores them.
"""
from typing import List, Optional

CUSTOM_OCR_TYPES = ("CUSTOM", "CUSTOM_PADDLE")


async def recipes_using(db, *, project_id: Optional[str] = None,
                        model_id: Optional[str] = None) -> List[str]:
    query = {"ocr_model_type": {"$in": list(CUSTOM_OCR_TYPES)}}
    if project_id:
        query["ocr_project_id"] = project_id
    if model_id:
        query["ocr_model_id"] = model_id
    docs = await db["recipes"].find(query, {"name": 1}).to_list(None)
    return sorted(d.get("name") or str(d["_id"]) for d in docs)


def in_use_message(kind: str, recipe_names: List[str]) -> str:
    return (
        f"Cannot delete this {kind}: it is used by {len(recipe_names)} recipe(s): "
        f"{', '.join(recipe_names)}. Open each recipe (Model tab → OCR Model Type), "
        f"choose a different OCR model and save, then delete again."
    )
