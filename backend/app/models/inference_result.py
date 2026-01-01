"""
Inference Result Model
Stores per-product inspection results (aggregate of all cameras)
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from bson import ObjectId


class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


class FrameResult(BaseModel):
    """Result for a single frame/template"""
    template_id: Optional[str] = Field(None, description="Template identifier")
    template_name: str = Field(..., description="Template name")
    frame_idx: int = Field(..., description="Frame index")
    pass_fail: str = Field(..., description="Pass or Fail")
    confidence: Optional[float] = Field(None, description="Confidence score")
    image_path: Optional[str] = Field(None, description="Path to saved image")
    detected_regions: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Detected regions with bounding boxes"
    )


class CameraResult(BaseModel):
    """Result for a single camera"""
    camera_id: str = Field(..., description="Camera identifier")
    serial_number: str = Field(..., description="Camera serial number")
    frames: List[FrameResult] = Field(
        default_factory=list,
        description="List of frame results"
    )


class InferenceResultCreate(BaseModel):
    """Schema for creating inference result"""
    recipe_id: str = Field(..., description="Recipe ID used for inspection")
    recipe_name: str = Field(..., description="Recipe name")
    product_pass_fail: str = Field(..., description="Overall product pass/fail")
    camera_results: List[CameraResult] = Field(
        default_factory=list,
        description="Results from all cameras"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata"
    )


class InferenceResultInDB(BaseModel):
    """Schema for inference result stored in database"""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    recipe_id: str = Field(..., description="Recipe ID")
    recipe_name: str = Field(..., description="Recipe name")
    product_pass_fail: str = Field(..., description="Overall product result")
    camera_results: List[CameraResult] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {ObjectId: str}
        populate_by_name = True


class InferenceResultResponse(BaseModel):
    """Schema for inference result API response"""
    id: str = Field(..., alias="_id", description="Result ID")
    recipe_id: str
    recipe_name: str
    product_pass_fail: str
    camera_results: List[CameraResult]
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime
    created_at: datetime

    class Config:
        json_encoders = {ObjectId: str}
        populate_by_name = True
