"""
Recipe Load Models
Tracks recipe loading and execution history
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any
from datetime import datetime


class RecipeLoadBase(BaseModel):
    """Base schema for recipe load"""
    recipe_id: str = Field(..., description="ID of the recipe being loaded")
    loaded_by: str = Field(..., description="User ID who loaded the recipe")
    loaded_by_full_name: str = Field(..., description="Full name of user who loaded")
    metadata: Dict[str, Any] = Field(..., description="Full recipe data snapshot at load time")
    status: str = Field(default="running", description="Load status: running, stopped")


class RecipeLoadCreate(RecipeLoadBase):
    """Schema for creating a new recipe load record"""
    pass


class RecipeLoadInDB(RecipeLoadBase):
    """Schema for recipe load stored in database"""
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(alias="_id")
    loaded_at: datetime = Field(..., description="When recipe was loaded")
    loaded_at_local: str = Field(..., description="Local time string when loaded")
    stopped_at: Optional[datetime] = Field(None, description="When recipe was stopped")
    stopped_by: Optional[str] = Field(None, description="User ID who stopped the recipe")
    stopped_by_full_name: Optional[str] = Field(None, description="Full name of user who stopped")


class RecipeLoadResponse(RecipeLoadBase):
    """Schema for recipe load response"""
    model_config = ConfigDict(from_attributes=True)

    id: str
    loaded_at: datetime
    loaded_at_local: str
    stopped_at: Optional[datetime] = None
    stopped_by: Optional[str] = None
    stopped_by_full_name: Optional[str] = None


class RecipeLoadStop(BaseModel):
    """Schema for stopping a recipe load"""
    stopped_by: str = Field(..., description="User ID who stopped the recipe")
    stopped_by_full_name: str = Field(..., description="Full name of user who stopped")
