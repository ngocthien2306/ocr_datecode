"""
Inference Result Model
Stores per-product inspection results (aggregate of all cameras)
"""

from pydantic import BaseModel, Field, ConfigDict, field_serializer
from pydantic_core import core_schema
from typing import List, Optional, Dict, Any, Annotated
from datetime import datetime
from bson import ObjectId


class PyObjectId(ObjectId):
    """Pydantic v2 compatible ObjectId"""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return core_schema.union_schema([
            core_schema.is_instance_schema(ObjectId),
            core_schema.chain_schema([
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(cls.validate),
            ])
        ],
        serialization=core_schema.plain_serializer_function_ser_schema(
            lambda x: str(x)
        ))

    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return v
        if isinstance(v, str):
            if not ObjectId.is_valid(v):
                raise ValueError("Invalid ObjectId")
            return ObjectId(v)
        raise ValueError("Invalid ObjectId")


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
    timings: Optional[Dict[str, Any]] = Field(
        None,
        description="Inference timing breakdown (ms)"
    )
    text_verification: Optional[Dict[str, Any]] = Field(
        None,
        description="Text verification results for Check_Type_Product function"
    )
    # NOTE: image_base64 is NOT stored in DB - only used for realtime SocketIO emission


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
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True
    )

    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    recipe_id: str = Field(..., description="Recipe ID")
    recipe_name: str = Field(..., description="Recipe name")
    product_pass_fail: str = Field(..., description="Overall product result")
    camera_results: List[CameraResult] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class InferenceResultResponse(BaseModel):
    """Schema for inference result API response"""
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., alias="_id", description="Result ID")
    recipe_id: str
    recipe_name: str
    product_pass_fail: str
    camera_results: List[CameraResult]
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime
    created_at: datetime
