import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query

from ...models.storage import (
    CleanupConfig,
    CleanupRunResult,
    DiskUsage,
    StorageItem,
    StorageStats,
)
from ...services import storage_cleanup_scheduler
from ...services.storage_service import StorageService

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize storage service with uploads path
UPLOADS_BASE = Path(__file__).parent.parent.parent.parent / "uploads" / "inference_results"
storage_service = StorageService(str(UPLOADS_BASE))

META_DIR = UPLOADS_BASE / "_meta"
CLEANUP_CONFIG_FILE = META_DIR / "cleanup_config.json"
CLEANUP_LAST_RUN_FILE = META_DIR / "cleanup_last_run.json"


@router.get("/storage/tree", response_model=StorageItem)
async def get_storage_tree(
    path: str = Query(default="", description="Relative path from inference_results root"),
    max_depth: int = Query(default=3, ge=0, le=10, description="Maximum depth to scan (0=unlimited)")
):
    """Get folder tree structure with sizes."""
    try:
        tree = storage_service.get_folder_tree(path, max_depth=max_depth)
        if tree is None:
            raise HTTPException(status_code=404, detail=f"Path not found or inaccessible: {path}")
        return tree
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting storage tree: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve storage tree: {str(e)}")


@router.get("/storage/stats", response_model=StorageStats)
async def get_storage_stats(
    path: str = Query(default="", description="Relative path from inference_results root")
):
    """Get storage statistics for a path."""
    try:
        stats = storage_service.get_storage_stats(path)
        if stats is None:
            raise HTTPException(status_code=404, detail=f"Path not found or inaccessible: {path}")
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting storage stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve storage stats: {str(e)}")


@router.delete("/storage/folder")
async def delete_folder(
    path: str = Query(..., description="Relative path to folder to delete")
):
    """Delete a folder and its contents. Irreversible."""
    try:
        if not path or path == "/" or path == ".":
            raise HTTPException(status_code=400, detail="Cannot delete root folder")
        success = storage_service.delete_folder(path)
        if not success:
            raise HTTPException(status_code=400, detail=f"Failed to delete folder: {path}")
        return {"success": True, "message": f"Folder deleted successfully: {path}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting folder: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete folder: {str(e)}")


@router.get("/storage/info")
async def get_storage_info():
    """Get storage service base path info."""
    return {
        "base_path": str(storage_service.base_path),
        "exists": storage_service.base_path.exists(),
        "is_directory": storage_service.base_path.is_dir() if storage_service.base_path.exists() else False,
    }


# ── Auto-cleanup ────────────────────────────────────────────────────────────
@router.get("/storage/disk-usage", response_model=DiskUsage)
async def get_disk_usage():
    """Return disk usage of the partition that hosts uploads/."""
    try:
        return storage_service.get_disk_usage()
    except Exception as e:
        logger.error(f"Error getting disk usage: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/storage/cleanup-config", response_model=CleanupConfig)
async def get_cleanup_config():
    """Read current auto-cleanup configuration."""
    return CleanupConfig(**storage_cleanup_scheduler.load_config())


@router.put("/storage/cleanup-config", response_model=CleanupConfig)
async def update_cleanup_config(payload: CleanupConfig):
    """Update auto-cleanup configuration and wake the scheduler."""
    cfg_dict = payload.model_dump()
    storage_cleanup_scheduler.save_config(cfg_dict)
    storage_cleanup_scheduler.reschedule()
    return CleanupConfig(**cfg_dict)


@router.post("/storage/cleanup/run-now", response_model=CleanupRunResult)
async def cleanup_run_now():
    """Trigger cleanup immediately using the saved config."""
    try:
        return await storage_cleanup_scheduler.run_now(trigger="manual")
    except Exception as e:
        logger.error(f"Error running cleanup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/storage/cleanup/last-run")
async def get_cleanup_last_run():
    """Return the most recent cleanup run result, if any."""
    return storage_cleanup_scheduler.load_last_run() or {}
