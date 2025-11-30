from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class CameraSettings(BaseModel):
    """Camera configuration settings"""
    exposure_time: float = Field(..., description="Exposure time in milliseconds")
    delay_trigger: float = Field(..., description="Delay trigger in milliseconds")
    # Additional camera settings can be added here
    gain: Optional[float] = None
    brightness: Optional[float] = None
    contrast: Optional[float] = None


class ModelThresholds(BaseModel):
    """Model thresholds for OCR processing"""
    detection_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    recognition_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    # Additional thresholds
    min_text_size: Optional[int] = None
    max_text_size: Optional[int] = None


class RecipeBase(BaseModel):
    """Base recipe schema"""
    name: str = Field(..., min_length=1, max_length=100, description="Recipe name")
    product_code: str = Field(..., min_length=1, max_length=50, description="Product code")
    description: Optional[str] = Field(None, max_length=500)
    camera_settings: CameraSettings
    model_thresholds: ModelThresholds
    template_config: Optional[Dict[str, Any]] = Field(None, description="Template matching configuration")
    roi_config: Optional[Dict[str, Any]] = Field(None, description="Region of interest configuration")
    is_active: bool = True


class RecipeCreate(RecipeBase):
    """Schema for creating a new recipe"""
    pass


class RecipeUpdate(BaseModel):
    """Schema for updating a recipe"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    product_code: Optional[str] = Field(None, min_length=1, max_length=50)
    description: Optional[str] = Field(None, max_length=500)
    camera_settings: Optional[CameraSettings] = None
    model_thresholds: Optional[ModelThresholds] = None
    template_config: Optional[Dict[str, Any]] = None
    roi_config: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class RecipeInDB(RecipeBase):
    """Schema for recipe stored in database"""
    id: str = Field(alias="_id")
    created_by: str = Field(..., description="User ID who created this recipe")
    updated_by: str = Field(..., description="User ID who last updated this recipe")
    created_at: datetime
    updated_at: datetime

    class Config:
        populate_by_name = True


class RecipeResponse(RecipeBase):
    """Schema for recipe response"""
    id: str
    created_by: str
    updated_by: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
