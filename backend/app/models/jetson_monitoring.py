from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class CPUMetrics(BaseModel):
    usage_percent: float = Field(..., description="CPU usage percentage (0-100)")
    frequency_mhz: float = Field(..., description="Current CPU frequency in MHz")
    temperature_celsius: Optional[float] = Field(None, description="CPU temperature in Celsius")
    per_core_usage: Dict[str, float] = Field(default_factory=dict, description="Usage per CPU core")


class GPUMetrics(BaseModel):
    usage_percent: float = Field(..., description="GPU usage percentage (0-100)")
    frequency_mhz: float = Field(..., description="Current GPU frequency in MHz")
    temperature_celsius: Optional[float] = Field(None, description="GPU temperature in Celsius")


class RAMMetrics(BaseModel):
    total_mb: float = Field(..., description="Total RAM in MB")
    used_mb: float = Field(..., description="Used RAM in MB")
    available_mb: float = Field(..., description="Available RAM in MB")
    usage_percent: float = Field(..., description="RAM usage percentage (0-100)")
    swap_total_mb: float = Field(..., description="Total swap space in MB")
    swap_used_mb: float = Field(..., description="Used swap space in MB")
    swap_percent: float = Field(..., description="Swap usage percentage (0-100)")


class DiskMetrics(BaseModel):
    total_gb: float = Field(..., description="Total disk space in GB")
    used_gb: float = Field(..., description="Used disk space in GB")
    available_gb: float = Field(..., description="Available disk space in GB")
    usage_percent: float = Field(..., description="Disk usage percentage (0-100)")
    read_speed_mbps: Optional[float] = Field(None, description="Disk read speed in MB/s")
    write_speed_mbps: Optional[float] = Field(None, description="Disk write speed in MB/s")


class PowerMetrics(BaseModel):
    total_mw: Optional[int] = Field(None, description="Total power consumption in mW")
    cpu_gpu_cv_mw: Optional[int] = Field(None, description="CPU+GPU+CV power in mW")
    soc_mw: Optional[int] = Field(None, description="SOC power in mW")


class SystemMetrics(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of metrics collection")
    cpu: CPUMetrics
    gpu: GPUMetrics
    ram: RAMMetrics
    disk: DiskMetrics
    power: Optional[PowerMetrics] = Field(None, description="Power consumption metrics")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    nvp_model: Optional[str] = Field(None, description="NVP model (e.g., MAXN)")


class AlertThreshold(BaseModel):
    cpu_percent: float = Field(default=85.0, description="CPU usage alert threshold (%)")
    gpu_percent: float = Field(default=85.0, description="GPU usage alert threshold (%)")
    ram_percent: float = Field(default=90.0, description="RAM usage alert threshold (%)")
    disk_percent: float = Field(default=90.0, description="Disk usage alert threshold (%)")
    cpu_temp_celsius: Optional[float] = Field(default=80.0, description="CPU temperature alert threshold (°C)")
    gpu_temp_celsius: Optional[float] = Field(default=80.0, description="GPU temperature alert threshold (°C)")


class Alert(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now, description="Alert generation time")
    alert_type: str = Field(..., description="Type of alert (cpu_usage, gpu_temp, etc.)")
    severity: str = Field(..., description="Alert severity: warning, critical")
    message: str = Field(..., description="Alert message")
    current_value: float = Field(..., description="Current metric value that triggered the alert")
    threshold_value: float = Field(..., description="Threshold value that was exceeded")


class MonitoringConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable/disable monitoring")
    update_interval_seconds: float = Field(default=2.0, description="Metrics update interval in seconds")
    alert_thresholds: AlertThreshold = Field(default_factory=AlertThreshold)
    alert_cooldown_seconds: float = Field(default=60.0, description="Cooldown period between duplicate alerts")
