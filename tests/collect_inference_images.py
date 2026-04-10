#!/usr/bin/env python3
"""
Script to collect inference result images with daily random sampling.
Strategy: Per-day per-camera sampling (30-50 images per day per camera)

Example:
- 1 organization with data from 2026-01-20 to 2026-01-28 (9 days)
- 2 cameras (40733814, 22376896)
- Result: 9 days × 2 cameras × (30-50 images) = 540-900 images per organization
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import random
from typing import Dict, List, Tuple
import json


class InferenceImageCollector:
    def __init__(
        self,
        source_root: str,
        target_folder: str,
        camera_ids: List[str],
        min_date: str = "2026-03-25",
        max_date: str = "2026-04-10",
        images_per_camera_per_day: Tuple[int, int] = (15, 25),
    ):
        self.source_root = Path(source_root)
        self.target_folder = Path(target_folder)
        self.camera_ids = camera_ids
        self.min_date = datetime.strptime(min_date, "%Y-%m-%d")
        self.max_date = datetime.strptime(max_date, "%Y-%m-%d")
        self.min_images, self.max_images = images_per_camera_per_day

        # Statistics
        self.stats = {
            "total_orgs_found": 0,
            "total_camera_folders": 0,
            "images_by_camera": defaultdict(int),
            "images_by_date": defaultdict(int),
            "images_by_org": defaultdict(int),
            "images_copied": 0,
        }

    def scan_and_analyze(self) -> Dict:
        """Scan source folders and analyze available images."""
        print("🔍 Scanning source folders...")

        # Structure: {org_id: {date: {camera_id: [images]}}}
        image_inventory = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        # Walk through all organization folders
        for org_folder in self.source_root.iterdir():
            if not org_folder.is_dir():
                continue

            org_id = org_folder.name

            # Walk through date folders
            for date_folder in org_folder.iterdir():
                if not date_folder.is_dir():
                    continue

                # Check if date is valid (format: YYYY-MM-DD)
                try:
                    folder_date = datetime.strptime(date_folder.name, "%Y-%m-%d")
                    if folder_date < self.min_date or folder_date > self.max_date:
                        continue
                except ValueError:
                    continue

                # Walk through camera folders
                for camera_folder in date_folder.iterdir():
                    if not camera_folder.is_dir():
                        continue

                    camera_id = camera_folder.name
                    if camera_id not in self.camera_ids:
                        continue

                    # Find all _org.jpg images
                    org_images = list(camera_folder.glob("*_org.jpg"))

                    if org_images:
                        self.stats["total_camera_folders"] += 1
                        self.stats["total_orgs_found"] += len(org_images)
                        self.stats["images_by_camera"][camera_id] += len(org_images)
                        self.stats["images_by_date"][date_folder.name] += len(org_images)
                        self.stats["images_by_org"][org_id] += len(org_images)

                        # Store images grouped by org, date, and camera
                        image_inventory[org_id][date_folder.name][camera_id].extend(org_images)

        return image_inventory

    def daily_random_sample(
        self,
        image_inventory: Dict[str, Dict[str, Dict[str, List[Path]]]]
    ) -> Dict[str, List[Path]]:
        """
        Perform daily random sampling:
        - For each organization, each date, each camera: sample 30-50 images
        - Strategy: Per day per camera sampling
        """
        print("\n🎲 Performing daily random sampling (per day, per camera)...")

        selected_images = defaultdict(list)
        sampling_details = []

        for org_id in sorted(image_inventory.keys()):
            date_groups = image_inventory[org_id]
            print(f"\n🏢 Organization: {org_id}")

            org_total = 0

            for date in sorted(date_groups.keys()):
                camera_groups = date_groups[date]
                print(f"\n  📅 Date: {date}")

                for camera_id in sorted(camera_groups.keys()):
                    images = camera_groups[camera_id]
                    available_count = len(images)

                    # Target: 30-50 images per day per camera
                    target_count = random.randint(self.min_images, self.max_images)
                    target_count = min(target_count, available_count)

                    if target_count > 0:
                        # Random sample
                        sampled = random.sample(images, target_count)
                        selected_images[f"{org_id}_{date}_{camera_id}"].extend(sampled)
                        org_total += target_count

                        print(f"    📷 Camera {camera_id}: {target_count}/{available_count} images")

                        sampling_details.append({
                            "org_id": org_id,
                            "date": date,
                            "camera_id": camera_id,
                            "available": available_count,
                            "selected": target_count,
                        })

            print(f"  ✓ Total for {org_id}: {org_total} images")

        return selected_images

    def copy_images(
        self,
        selected_images: Dict[str, List[Path]],
        dry_run: bool = True
    ):
        """Copy selected images to target folder."""

        if dry_run:
            print("\n🔍 DRY RUN MODE - No files will be copied")
        else:
            print("\n📦 Copying images...")
            self.target_folder.mkdir(parents=True, exist_ok=True)

        copied_count = 0
        file_mapping = []

        for group_key, images in selected_images.items():
            # group_key format: org_id_date_camera_id
            parts = group_key.split('_', 2)  # Split into 3 parts max
            if len(parts) >= 3:
                org_id = parts[0]
                date = parts[1]
                camera_id = '_'.join(parts[2:])  # Handle camera_id with underscores
            else:
                org_id = "unknown"
                date = "unknown"
                camera_id = group_key

            if not dry_run:
                print(f"\n📷 {org_id}/{date}/{camera_id}: {len(images)} images", end=" ")
            else:
                print(f"\n📷 {org_id}/{date}/{camera_id}: {len(images)} images")

            for img_path in images:
                # Create new filename: camera_id_date_original_filename
                # Remove _org.jpg suffix and add it back to keep consistency
                base_name = img_path.stem.replace('_org', '')
                new_filename = f"{camera_id}_{date}_{base_name}_org.jpg"
                target_path = self.target_folder / new_filename

                if not dry_run:
                    try:
                        shutil.copy2(img_path, target_path)
                        copied_count += 1
                    except Exception as e:
                        print(f"\n  ❌ Error copying {img_path.name}: {e}")
                        continue

                file_mapping.append({
                    "source": str(img_path),
                    "target": str(target_path),
                    "org_id": org_id,
                    "date": date,
                    "camera_id": camera_id,
                    "original_name": img_path.name,
                })

            if not dry_run:
                print("✓")

        self.stats["images_copied"] = copied_count if not dry_run else 0

        # Save mapping file
        if not dry_run:
            mapping_file = self.target_folder / "image_mapping.json"
            with open(mapping_file, "w") as f:
                json.dump(file_mapping, f, indent=2)
            print(f"\n💾 Mapping saved to: {mapping_file}")

        return file_mapping

    def print_summary(self, selected_images: Dict[str, List[Path]]):
        """Print summary statistics."""
        print("\n" + "="*60)
        print("📊 SUMMARY REPORT")
        print("="*60)

        print(f"\n🔍 Scanning Results:")
        print(f"  Total camera folders scanned: {self.stats['total_camera_folders']}")
        print(f"  Total _org.jpg images found: {self.stats['total_orgs_found']}")

        print(f"\n🏢 Images by Organization:")
        for org_id, count in sorted(self.stats['images_by_org'].items()):
            print(f"  {org_id}: {count} images available")

        print(f"\n📷 Images by Camera (Total):")
        for camera_id, count in sorted(self.stats['images_by_camera'].items()):
            print(f"  {camera_id}: {count} images available")

        print(f"\n📅 Images by Date (Total):")
        for date in sorted(self.stats['images_by_date'].keys()):
            count = self.stats['images_by_date'][date]
            print(f"  {date}: {count} images available")

        # Calculate total selected
        total_selected = sum(len(imgs) for imgs in selected_images.values())
        print(f"\n✅ Total Selected: {total_selected} images")

        if self.stats["images_copied"] > 0:
            print(f"✅ Successfully copied: {self.stats['images_copied']} images")

        print(f"\n📁 Target folder: {self.target_folder}")
        print("="*60)

    def run(self, dry_run: bool = True):
        """Main execution flow."""
        print("🚀 Starting Image Collection Process")
        print(f"📂 Source: {self.source_root}")
        print(f"📂 Target: {self.target_folder}")
        print(f"📷 Cameras: {', '.join(self.camera_ids)}")
        print(f"📅 Date range: {self.min_date.strftime('%Y-%m-%d')} to {self.max_date.strftime('%Y-%m-%d')}")
        print(f"🎯 Target per camera per day: {self.min_images}-{self.max_images} images")

        # Step 1: Scan and analyze
        image_inventory = self.scan_and_analyze()

        if not image_inventory:
            print("\n❌ No images found matching criteria!")
            return

        # Step 2: Daily random sampling (per day, per camera)
        selected_images = self.daily_random_sample(image_inventory)

        # Step 3: Copy images
        file_mapping = self.copy_images(selected_images, dry_run=dry_run)

        # Step 4: Print summary
        self.print_summary(selected_images)

        if dry_run:
            print("\n💡 This was a DRY RUN. To actually copy files, run with dry_run=False")

        return selected_images, file_mapping


def main():
    """Main entry point."""

    # Configuration
    SOURCE_ROOT = "/home/demo/Source/ocr_datecode/backend/uploads/inference_results"
    TARGET_FOLDER = "/home/demo/Source/ocr_datecode/data"
    CAMERA_IDS = ["40733814", "40767171"]
    MIN_DATE = "2026-03-25"
    MAX_DATE = "2026-04-10"
    IMAGES_PER_CAMERA_PER_DAY = (15, 25)  # (min, max) per day per camera

    # Create collector instance
    collector = InferenceImageCollector(
        source_root=SOURCE_ROOT,
        target_folder=TARGET_FOLDER,
        camera_ids=CAMERA_IDS,
        min_date=MIN_DATE,
        max_date=MAX_DATE,
        images_per_camera_per_day=IMAGES_PER_CAMERA_PER_DAY,
    )

    # Run with dry-run first to preview
    print("="*60)
    print("STEP 1: DRY RUN (Preview)")
    print("="*60)
    collector.run(dry_run=True)

    # Ask for confirmation
    print("\n" + "="*60)
    response = input("\n📝 Proceed with actual copy? (yes/no): ").strip().lower()

    if response in ['yes', 'y']:
        print("\n" + "="*60)
        print("STEP 2: ACTUAL COPY")
        print("="*60)

        # Reset stats for actual run
        collector.stats = {
            "total_orgs_found": 0,
            "total_camera_folders": 0,
            "images_by_camera": defaultdict(int),
            "images_by_date": defaultdict(int),
            "images_by_org": defaultdict(int),
            "images_copied": 0,
        }

        collector.run(dry_run=False)
        print("\n✅ Process completed successfully!")
    else:
        print("\n❌ Copy cancelled by user.")


if __name__ == "__main__":
    main()
