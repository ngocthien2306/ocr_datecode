import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from ...models.storage import StorageItem, StorageStats
from ...services.storage_service import StorageService

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize storage service with uploads path
UPLOADS_BASE = Path(__file__).parent.parent.parent.parent / "uploads" / "inference_results"
storage_service = StorageService(str(UPLOADS_BASE))


@router.get("/storage/tree", response_model=StorageItem)
async def get_storage_tree(
    path: str = Query(default="", description="Relative path from inference_results root"),
    max_depth: int = Query(default=3, ge=0, le=10, description="Maximum depth to scan (0=unlimited)")
):
    """
    Get folder tree structure with sizes

    Returns folder hierarchy with size information.
    Stops at leaf folders containing only image files.

    Example paths:
    - "" (root)
    - "694850c56beac57d01bdf65c"
    - "694850c56beac57d01bdf65c/2026-01-06"
    """
    try:
        tree = storage_service.get_folder_tree(path, max_depth=max_depth)

        if tree is None:
            raise HTTPException(
                status_code=404,
                detail=f"Path not found or inaccessible: {path}"
            )

        return tree

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting storage tree: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve storage tree: {str(e)}"
        )


@router.get("/storage/stats", response_model=StorageStats)
async def get_storage_stats(
    path: str = Query(default="", description="Relative path from inference_results root")
):
    """
    Get storage statistics for a path

    Returns total size, file count, and directory count
    """
    try:
        stats = storage_service.get_storage_stats(path)

        if stats is None:
            raise HTTPException(
                status_code=404,
                detail=f"Path not found or inaccessible: {path}"
            )

        return stats

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting storage stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve storage stats: {str(e)}"
        )


@router.delete("/storage/folder")
async def delete_folder(
    path: str = Query(..., description="Relative path to folder to delete")
):
    """
    Delete a folder and its contents

    **Warning**: This operation is irreversible!
    """
    try:
        # Validate path is not empty or root
        if not path or path == "/" or path == ".":
            raise HTTPException(
                status_code=400,
                detail="Cannot delete root folder"
            )

        success = storage_service.delete_folder(path)

        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to delete folder: {path}"
            )

        return {
            "success": True,
            "message": f"Folder deleted successfully: {path}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting folder: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete folder: {str(e)}"
        )


@router.get("/storage/info")
async def get_storage_info():
    """
    Get storage service information

    Returns base path and status
    """
    return {
        "base_path": str(storage_service.base_path),
        "exists": storage_service.base_path.exists(),
        "is_directory": storage_service.base_path.is_dir() if storage_service.base_path.exists() else False
    }
