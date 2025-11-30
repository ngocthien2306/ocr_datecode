from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class CameraSettings(BaseModel):
    """Camera configuration settings"""
    exposure_time: float = Field(..., gt=0, description="Exposure time in milliseconds")
    delay_trigger: float = Field(..., ge=0, description="Delay trigger in milliseconds")
    gain: Optional[float] = Field(None, ge=0, description="Camera gain")
    resolution: Optional[str] = Field(None, description="Image resolution (e.g., 1920x1080)")


class ModelThreshold(BaseModel):
    """AI Model threshold settings"""
    model_name: str = Field(..., description="Name of the trained model")
    confidence_threshold: float = Field(..., ge=0, le=1, description="Confidence threshold (0-1)")
    iou_threshold: Optional[float] = Field(None, ge=0, le=1, description="IoU threshold for detection")


class RecipeBase(BaseModel):
    """Base Recipe schema"""
    name: str = Field(..., min_length=1, max_length=100, description="Recipe name")
    description: Optional[str] = Field(None, max_length=500, description="Recipe description")
    camera_settings: CameraSettings
    model_thresholds: list[ModelThreshold] = Field(..., description="List of model thresholds")
    datecode_pattern: Optional[str] = Field(None, description="Expected datecode pattern/regex")
    is_active: bool = Field(True, description="Whether recipe is active")
    additional_params: Optional[Dict[str, Any]] = Field(None, description="Additional parameters")


class RecipeCreate(RecipeBase):
    """Schema for creating a new recipe"""
    pass


class RecipeUpdate(BaseModel):
    """Schema for updating a recipe"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    camera_settings: Optional[CameraSettings] = None
    model_thresholds: Optional[list[ModelThreshold]] = None
    datecode_pattern: Optional[str] = None
    is_active: Optional[bool] = None
    additional_params: Optional[Dict[str, Any]] = None


class RecipeInDB(RecipeBase):
    """Recipe model stored in database"""
    id: str = Field(alias="_id")
    created_by: str = Field(..., description="User ID who created the recipe")
    updated_by: str = Field(..., description="User ID who last updated the recipe")
    created_at: datetime
    updated_at: datetime
    version: int = Field(1, description="Recipe version number")

    class Config:
        populate_by_name = True


class RecipeResponse(RecipeBase):
    """Recipe response schema"""
    id: str = Field(alias="_id")
    created_by: str
    updated_by: str
    created_at: datetime
    updated_at: datetime
    version: int

    class Config:
        populate_by_name = True


class DatecodeValidation(BaseModel):
    """Schema for validating datecode against recipe"""
    recipe_id: str = Field(..., description="Recipe ID to validate against")
    datecode_input: str = Field(..., description="Datecode string to validate")


class DatecodeValidationResponse(BaseModel):
    """Response for datecode validation"""
    is_valid: bool
    recipe_name: str
    expected_pattern: Optional[str]
    input_datecode: str
    validation_message: str
    confidence: Optional[float] = None
