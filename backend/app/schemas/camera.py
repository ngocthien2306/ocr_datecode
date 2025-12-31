"""
Camera Schemas
Pydantic schemas for camera API requests and responses
"""
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class CameraBase(BaseModel):
    """Base camera schema"""
    camera_id: str = Field(..., description="Unique camera identifier", min_length=1, max_length=50)
    model_name: str = Field(..., description="Camera model name", min_length=1)
    serial_number: str = Field(..., description="Camera serial number", min_length=1)
    resolution_width: int = Field(..., ge=1, description="Camera resolution width in pixels")
    resolution_height: int = Field(..., ge=1, description="Camera resolution height in pixels")
    ip_address: Optional[str] = Field(None, description="Camera IP address")
    location: Optional[str] = Field(None, description="Physical location")
    description: Optional[str] = Field(None, description="Additional description")
    max_frame_rate: Optional[float] = Field(None, ge=0, description="Maximum frame rate")
    pixel_format: Optional[str] = Field("Mono8", description="Pixel format")
    is_active: bool = Field(True, description="Whether camera is active")


class CameraCreate(CameraBase):
    """Schema for creating a camera"""
    pass


class CameraUpdate(BaseModel):
    """Schema for updating a camera"""
    model_name: Optional[str] = Field(None, description="Camera model name")
    serial_number: Optional[str] = Field(None, description="Camera serial number")
    resolution_width: Optional[int] = Field(None, ge=1, description="Resolution width")
    resolution_height: Optional[int] = Field(None, ge=1, description="Resolution height")
    ip_address: Optional[str] = Field(None, description="Camera IP address")
    location: Optional[str] = Field(None, description="Physical location")
    description: Optional[str] = Field(None, description="Additional description")
    max_frame_rate: Optional[float] = Field(None, ge=0, description="Maximum frame rate")
    pixel_format: Optional[str] = Field(None, description="Pixel format")
    is_connected: Optional[bool] = Field(None, description="Connection status")
    is_active: Optional[bool] = Field(None, description="Active status")


class CameraResponse(CameraBase):
    """Schema for camera response"""
    id: Optional[str] = Field(None, alias="_id", description="MongoDB document ID")
    is_connected: bool = Field(..., description="Connection status")
    created_at: datetime
    updated_at: datetime
    created_by: str
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
    
    class Config:
        from_attributes = True


class CameraConnectionStatus(BaseModel):
    """Schema for camera connection status update"""
    camera_id: str
    is_connected: bool
    last_checked: datetime = Field(default_factory=datetime.utcnow)


class CameraSettingsUpdate(BaseModel):
    """Schema for updating camera runtime settings"""
    exposure_time: Optional[int] = Field(None, ge=100, le=1000000, description="Exposure time in microseconds (μs)")
    gain: Optional[float] = Field(None, ge=0.0, le=20.0, description="Camera gain value")
    trigger_config: Optional[Dict[str, Any]] = Field(None, description="Trigger configuration (mode, source, activation, di_number)")
