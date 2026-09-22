from typing import List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


class StorageItem(BaseModel):
    name: str = Field(..., description="Folder/file name")
    path: str = Field(..., description="Relative path from base")
    type: str = Field(..., description="'directory' or 'file'")
    size_bytes: int = Field(..., description="Size in bytes")
    size_mb: float = Field(..., description="Size in MB")
    file_count: Optional[int] = Field(None, description="Number of files (for directories)")
    children: Optional[List['StorageItem']] = Field(None, description="Subdirectories/files")
    modified_time: Optional[str] = Field(None, description="Last modified time")


class StorageStats(BaseModel):
    total_size_bytes: int
    total_size_mb: float
    total_size_gb: float
    total_files: int
    total_directories: int


class DiskUsage(BaseModel):
    total_gb: float
    used_gb: float
    free_gb: float
    used_percent: float
    mount_point: str


class CleanupConfig(BaseModel):
    enabled: bool = False
    dry_run: bool = True
    schedule_mode: Literal["daily", "hourly"] = "daily"
    schedule_hour: int = Field(2, ge=0, le=23)
    schedule_minute: int = Field(0, ge=0, le=59)
    interval_hours: int = Field(6, ge=1, le=24)
    trigger_usage_percent: float = Field(70.0, ge=10.0, le=99.0)
    target_usage_percent: float = Field(60.0, ge=5.0, le=98.0)
    min_images_per_folder: int = Field(25, ge=20, le=30)
    keep_recent_dates_per_recipe: int = Field(3, ge=1, le=30)

    @model_validator(mode="after")
    def _check_thresholds(self) -> "CleanupConfig":
        if self.target_usage_percent >= self.trigger_usage_percent:
            raise ValueError(
                "target_usage_percent must be lower than trigger_usage_percent"
            )
        return self


class CleanupRunResult(BaseModel):
    ran_at: str
    trigger: Literal["scheduled", "manual", "threshold-skip"]
    dry_run: bool
    disk_used_percent_before: float
    disk_used_percent_after: float
    files_deleted: int = 0
    folders_deleted: int = 0
    freed_mb: float = 0.0
    leaf_folders_scanned: int = 0
    duration_seconds: float = 0.0
    message: str = ""


StorageItem.model_rebuild()
