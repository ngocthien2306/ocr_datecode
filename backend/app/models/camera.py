"""
Camera Model
Defines the camera data structure
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class CameraModel(BaseModel):
    """Camera model for database"""
    model_config = ConfigDict(protected_namespaces=())

    camera_id: str = Field(..., description="Unique camera identifier")
    model_name: str = Field(..., description="Camera model name (e.g., Basler acA1920-40gm)")
    serial_number: str = Field(..., description="Camera serial number")
    resolution_width: int = Field(..., description="Camera resolution width in pixels")
    resolution_height: int = Field(..., description="Camera resolution height in pixels")
    is_connected: bool = Field(default=False, description="Camera connection status")
    is_active: bool = Field(default=True, description="Whether camera is active/enabled")
    ip_address: Optional[str] = Field(None, description="Camera IP address (for GigE cameras)")
    location: Optional[str] = Field(None, description="Physical location of camera")
    description: Optional[str] = Field(None, description="Additional description")
    
    # Camera capabilities
    max_frame_rate: Optional[float] = Field(None, description="Maximum frame rate (fps)")
    pixel_format: Optional[str] = Field(default="Mono8", description="Default pixel format")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = Field(default="system", description="User who created the camera entry")
    
    class Config:
        json_schema_extra = {
            "example": {
                "camera_id": "CAM001",
                "model_name": "Basler acA1920-40gm",
                "serial_number": "40123456",
                "resolution_width": 1920,
                "resolution_height": 1200,
                "is_connected": True,
                "is_active": True,
                "ip_address": "192.168.1.100",
                "location": "Production Line 1",
                "max_frame_rate": 40.0,
                "pixel_format": "Mono8"
            }
        }
