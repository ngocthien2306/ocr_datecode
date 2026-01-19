from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

from ..models.jetson_monitoring import (
    SystemMetrics,
    Alert,
    MonitoringConfig,
    AlertThreshold,
    CPUMetrics,
    GPUMetrics,
    RAMMetrics,
    DiskMetrics
)


class SystemMetricsResponse(BaseModel):
    success: bool = Field(default=True)
    data: SystemMetrics
    message: Optional[str] = Field(default=None)


class AlertsResponse(BaseModel):
    success: bool = Field(default=True)
    data: List[Alert]
    count: int
    message: Optional[str] = Field(default=None)


class MonitoringConfigResponse(BaseModel):
    success: bool = Field(default=True)
    data: MonitoringConfig
    message: Optional[str] = Field(default=None)


class UpdateMonitoringConfigRequest(BaseModel):
    enabled: Optional[bool] = Field(None, description="Enable/disable monitoring")
    update_interval_seconds: Optional[float] = Field(None, description="Update interval in seconds", ge=0.5, le=60.0)
    alert_thresholds: Optional[AlertThreshold] = Field(None, description="Alert threshold values")
    alert_cooldown_seconds: Optional[float] = Field(None, description="Cooldown between alerts", ge=10.0, le=600.0)


class UpdateMonitoringConfigResponse(BaseModel):
    success: bool = Field(default=True)
    data: MonitoringConfig
    message: str = Field(default="Monitoring configuration updated successfully")


class MonitoringStatusResponse(BaseModel):
    success: bool = Field(default=True)
    data: dict = Field(
        ...,
        description="Monitoring service status",
        example={
            "monitoring_enabled": True,
            "update_interval_seconds": 2.0,
            "last_update": "2026-01-18T10:30:00",
            "active_alerts_count": 0
        }
    )
    message: Optional[str] = Field(default=None)
