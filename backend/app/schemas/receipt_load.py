from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime


class ReceiptLoadCreate(BaseModel):
    recipe_id: str
    metadata: Optional[Dict[str, Any]] = None


class ReceiptLoadResponse(BaseModel):
    id: str
    recipe_id: str
    loaded_by: str
    loaded_by_full_name: Optional[str] = None
    loaded_at: datetime
    metadata: Optional[Dict[str, Any]] = None
