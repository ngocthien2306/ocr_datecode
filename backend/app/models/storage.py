from typing import List, Optional
from pydantic import BaseModel, Field


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


StorageItem.model_rebuild()
