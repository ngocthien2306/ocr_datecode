import os
import random
import re
import shutil
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from ..models.storage import (
    CleanupConfig,
    CleanupRunResult,
    DiskUsage,
    StorageItem,
    StorageStats,
)

logger = logging.getLogger(__name__)

DATE_FOLDER_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

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

    # ── Disk usage ──────────────────────────────────────────────────────────
    def get_disk_usage(self) -> DiskUsage:
        """Return disk usage of the partition that contains base_path."""
        # If base_path doesn't exist yet, walk up to first existing parent
        probe = self.base_path
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        usage = shutil.disk_usage(probe)
        gb = 1024 ** 3
        used_pct = (usage.used / usage.total) * 100.0 if usage.total else 0.0
        return DiskUsage(
            total_gb=usage.total / gb,
            used_gb=usage.used / gb,
            free_gb=usage.free / gb,
            used_percent=round(used_pct, 2),
            mount_point=str(probe),
        )

    # ── Auto cleanup ────────────────────────────────────────────────────────
    def _list_leaf_folders(self) -> List[Path]:
        """Find all leaf folders that look like image-result folders.

        Expected shape: base_path/{recipe_id}/{YYYY-MM-DD}/{timestamp}/<images>.
        Anything under a `_meta` folder is ignored.
        """
        leaves: List[Path] = []
        if not self.base_path.exists():
            return leaves
        for recipe_dir in self.base_path.iterdir():
            if not recipe_dir.is_dir() or recipe_dir.name.startswith("_"):
                continue
            for date_dir in recipe_dir.iterdir():
                if not date_dir.is_dir() or not DATE_FOLDER_RE.match(date_dir.name):
                    continue
                for ts_dir in date_dir.iterdir():
                    if ts_dir.is_dir() and self._is_leaf_folder(ts_dir):
                        leaves.append(ts_dir)
        return leaves

    def _images_in(self, folder: Path) -> List[Path]:
        try:
            return [p for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        except OSError:
            return []

    def run_auto_cleanup(
        self,
        cfg: CleanupConfig,
        trigger: str = "scheduled",
    ) -> CleanupRunResult:
        """Run cleanup according to the supplied config.

        Algorithm:
          - Skip entirely if disk usage < trigger threshold.
          - Pass 1: in every leaf folder, random-keep N images, delete the rest.
                    Stop early as soon as usage drops to target.
          - Pass 2: delete oldest leaf folders date-by-date, but always keep
                    `keep_recent_dates_per_recipe` newest date-folders per recipe.
        """
        started_at = time.time()
        before = self.get_disk_usage()
        ran_at_iso = datetime.now().isoformat(timespec="seconds")

        if before.used_percent < cfg.trigger_usage_percent:
            return CleanupRunResult(
                ran_at=ran_at_iso,
                trigger="threshold-skip",
                dry_run=cfg.dry_run,
                disk_used_percent_before=before.used_percent,
                disk_used_percent_after=before.used_percent,
                duration_seconds=round(time.time() - started_at, 3),
                message=(
                    f"Disk usage {before.used_percent:.1f}% is below trigger "
                    f"{cfg.trigger_usage_percent:.1f}% — no cleanup needed."
                ),
            )

        files_deleted = 0
        folders_deleted = 0
        freed_bytes = 0
        leaves = self._list_leaf_folders()
        leaf_count = len(leaves)

        def usage_now() -> float:
            return self.get_disk_usage().used_percent

        def stop_reached() -> bool:
            return usage_now() <= cfg.target_usage_percent

        # ── Pass 1: trim each leaf folder ──────────────────────────────────
        # Shuffle leaves so we don't always trim the same recipe first.
        shuffled = list(leaves)
        random.shuffle(shuffled)
        for leaf in shuffled:
            images = self._images_in(leaf)
            if len(images) <= cfg.min_images_per_folder:
                continue
            # Randomly choose which images to KEEP — rest are deleted.
            to_keep = set(random.sample(images, cfg.min_images_per_folder))
            to_delete = [p for p in images if p not in to_keep]
            for p in to_delete:
                try:
                    size = p.stat().st_size
                    if not cfg.dry_run:
                        p.unlink()
                    files_deleted += 1
                    freed_bytes += size
                except OSError as e:
                    logger.warning(f"[storage_cleanup] could not delete {p}: {e}")
            # Re-check disk every batch (avoid syscalls per file).
            if not cfg.dry_run and stop_reached():
                break

        # ── Pass 2: delete oldest date-folders per recipe ──────────────────
        if cfg.dry_run or not stop_reached():
            for recipe_dir in sorted(self.base_path.iterdir()):
                if not recipe_dir.is_dir() or recipe_dir.name.startswith("_"):
                    continue
                date_dirs = sorted(
                    [d for d in recipe_dir.iterdir()
                     if d.is_dir() and DATE_FOLDER_RE.match(d.name)],
                    key=lambda d: d.name,
                )
                # Keep N newest date-folders untouched.
                deletable = date_dirs[: max(0, len(date_dirs) - cfg.keep_recent_dates_per_recipe)]
                for d in deletable:
                    size = self._folder_size_bytes(d)
                    try:
                        if not cfg.dry_run:
                            shutil.rmtree(d)
                        folders_deleted += 1
                        freed_bytes += size
                        # Also account for inner files for reporting
                        files_deleted += sum(1 for _ in d.rglob("*") if _.is_file()) if cfg.dry_run else 0
                    except OSError as e:
                        logger.warning(f"[storage_cleanup] could not delete {d}: {e}")
                    if not cfg.dry_run and stop_reached():
                        break
                if not cfg.dry_run and stop_reached():
                    break

        after = self.get_disk_usage()
        return CleanupRunResult(
            ran_at=ran_at_iso,
            trigger=trigger,  # type: ignore[arg-type]
            dry_run=cfg.dry_run,
            disk_used_percent_before=before.used_percent,
            disk_used_percent_after=after.used_percent,
            files_deleted=files_deleted,
            folders_deleted=folders_deleted,
            freed_mb=round(freed_bytes / (1024 * 1024), 2),
            leaf_folders_scanned=leaf_count,
            duration_seconds=round(time.time() - started_at, 3),
            message=(
                f"{'[DRY-RUN] ' if cfg.dry_run else ''}"
                f"Trimmed {files_deleted} files, removed {folders_deleted} folders, "
                f"freed {freed_bytes / (1024 * 1024):.1f} MB. "
                f"Disk: {before.used_percent:.1f}% → {after.used_percent:.1f}%"
            ),
        )

    def _folder_size_bytes(self, path: Path) -> int:
        total = 0
        try:
            for p in path.rglob("*"):
                if p.is_file():
                    try:
                        total += p.stat().st_size
                    except OSError:
                        pass
        except OSError:
            pass
        return total
