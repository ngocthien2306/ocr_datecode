import os
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from ..models.storage import StorageItem, StorageStats

logger = logging.getLogger(__name__)


class StorageService:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path).resolve()
        logger.info(f"StorageService initialized with base_path: {self.base_path}")

    def get_folder_tree(self, relative_path: str = "", max_depth: int = 10) -> Optional[StorageItem]:
        """
        Get folder tree structure with sizes

        Args:
            relative_path: Path relative to base_path (e.g., "inference_results")
            max_depth: Maximum depth to traverse (0 = unlimited)

        Returns:
            StorageItem with folder structure
        """
        target_path = self.base_path / relative_path

        # Security check: ensure path is within base_path
        if not self._is_safe_path(target_path):
            logger.warning(f"Attempted to access unsafe path: {target_path}")
            return None

        if not target_path.exists():
            logger.warning(f"Path does not exist: {target_path}")
            return None

        return self._scan_directory(target_path, relative_path, current_depth=0, max_depth=max_depth)

    def get_storage_stats(self, relative_path: str = "") -> Optional[StorageStats]:
        """
        Get storage statistics for a path

        Args:
            relative_path: Path relative to base_path

        Returns:
            StorageStats with total size and counts
        """
        target_path = self.base_path / relative_path

        if not self._is_safe_path(target_path):
            return None

        if not target_path.exists():
            return None

        total_size = 0
        file_count = 0
        dir_count = 0

        try:
            for root, dirs, files in os.walk(target_path):
                dir_count += len(dirs)
                for file in files:
                    file_path = Path(root) / file
                    try:
                        total_size += file_path.stat().st_size
                        file_count += 1
                    except (OSError, PermissionError):
                        continue

            return StorageStats(
                total_size_bytes=total_size,
                total_size_mb=total_size / (1024 * 1024),
                total_size_gb=total_size / (1024 * 1024 * 1024),
                total_files=file_count,
                total_directories=dir_count
            )

        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return None

    def _scan_directory(self, path: Path, relative_path: str, current_depth: int, max_depth: int) -> StorageItem:
        """
        Recursively scan directory and build tree structure

        Stops at folders that look like leaf image folders (numeric names like timestamps)
        """
        name = path.name if path.name else relative_path

        # Get directory size and file count
        dir_size, file_count = self._get_directory_size(path)

        # Get modified time
        try:
            modified_time = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        except:
            modified_time = None

        item = StorageItem(
            name=name,
            path=relative_path,
            type="directory",
            size_bytes=dir_size,
            size_mb=dir_size / (1024 * 1024),
            file_count=file_count,
            modified_time=modified_time,
            children=[]
        )

        # Check if we should stop here
        if max_depth > 0 and current_depth >= max_depth:
            return item

        # Check if this looks like a leaf folder (contains only image files)
        if self._is_leaf_folder(path):
            item.children = None  # Don't expand further
            return item

        # Scan subdirectories
        try:
            entries = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))

            for entry in entries:
                if entry.is_dir():
                    child_relative = str(Path(relative_path) / entry.name) if relative_path else entry.name
                    child_item = self._scan_directory(
                        entry,
                        child_relative,
                        current_depth + 1,
                        max_depth
                    )
                    item.children.append(child_item)

        except PermissionError:
            logger.warning(f"Permission denied: {path}")
            item.children = None

        return item

    def _get_directory_size(self, path: Path) -> tuple[int, int]:
        """
        Calculate total size and file count of directory

        Returns:
            (total_size_bytes, file_count)
        """
        total_size = 0
        file_count = 0

        try:
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = Path(root) / file
                    try:
                        total_size += file_path.stat().st_size
                        file_count += 1
                    except (OSError, PermissionError):
                        continue

        except Exception as e:
            logger.error(f"Error calculating directory size for {path}: {e}")

        return total_size, file_count

    def _is_leaf_folder(self, path: Path) -> bool:
        """
        Check if folder is a leaf folder (contains only files, no subdirectories)

        Typically inference result folders like:
        /694850c56beac57d01bdf65c/2026-01-06/40733814

        Returns True if folder contains only image files
        """
        try:
            entries = list(path.iterdir())

            # If no subdirectories, it's a leaf
            has_subdirs = any(entry.is_dir() for entry in entries)

            if not has_subdirs and len(entries) > 0:
                # Check if mostly image files
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
                files = [e for e in entries if e.is_file()]

                if files:
                    image_files = sum(1 for f in files if f.suffix.lower() in image_extensions)
                    # If > 80% are images, consider it a leaf
                    return image_files / len(files) > 0.8

            return False

        except Exception:
            return False

    def _is_safe_path(self, path: Path) -> bool:
        """
        Check if path is within base_path (prevent directory traversal)
        """
        try:
            resolved = path.resolve()
            return resolved.is_relative_to(self.base_path)
        except Exception:
            return False

    def delete_folder(self, relative_path: str) -> bool:
        """
        Delete a folder and its contents

        Args:
            relative_path: Path relative to base_path

        Returns:
            True if successful, False otherwise
        """
        import shutil

        target_path = self.base_path / relative_path

        if not self._is_safe_path(target_path):
            logger.warning(f"Attempted to delete unsafe path: {target_path}")
            return False

        if not target_path.exists():
            logger.warning(f"Path does not exist: {target_path}")
            return False

        # Don't allow deleting base_path itself
        if target_path == self.base_path:
            logger.warning(f"Cannot delete base path: {target_path}")
            return False

        try:
            shutil.rmtree(target_path)
            logger.info(f"Deleted folder: {target_path}")
            return True
        except Exception as e:
            logger.error(f"Error deleting folder {target_path}: {e}")
            return False
