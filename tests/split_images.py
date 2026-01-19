#!/usr/bin/env python3
"""
Script to split images from subfolders into separate folders by image name
Example:
    data/folder/trigger_XXX/image_01.png -> data/folder/images1/trigger_XXX.png
    data/folder/trigger_XXX/image_02.png -> data/folder/images2/trigger_XXX.png
"""
import os
import shutil
from pathlib import Path

def split_images(base_folder):
    """Split images from subfolders into images1 and images2"""
    base_path = Path(base_folder)
    
    if not base_path.exists():
        print(f"❌ Folder not found: {base_folder}")
        return
    
    # Create output folders
    images1_folder = base_path / "images1"
    images2_folder = base_path / "images2"
    images1_folder.mkdir(exist_ok=True)
    images2_folder.mkdir(exist_ok=True)
    
    # Counter
    count_img1 = 0
    count_img2 = 0
    
    # Iterate through all subfolders
    for subfolder in sorted(base_path.iterdir()):
        if not subfolder.is_dir() or subfolder.name.startswith('images'):
            continue
        
        # Look for image files
        for img_file in subfolder.glob('*.*'):
            if img_file.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                continue
            
            # Determine destination based on filename
            if 'image_01' in img_file.name or img_file.name.endswith('_01.png'):
                dest_folder = images1_folder
                # Use subfolder name as new filename
                new_name = f"{subfolder.name}{img_file.suffix}"
                count_img1 += 1
            elif 'image_02' in img_file.name or img_file.name.endswith('_02.png'):
                dest_folder = images2_folder
                new_name = f"{subfolder.name}{img_file.suffix}"
                count_img2 += 1
            else:
                continue
            
            # Copy file
            dest_path = dest_folder / new_name
            shutil.copy2(img_file, dest_path)
            print(f"✓ {subfolder.name}/{img_file.name} -> {dest_folder.name}/{new_name}")
    
    print(f"\n✅ Done!")
    print(f"   {count_img1} images -> {images1_folder}")
    print(f"   {count_img2} images -> {images2_folder}")


if __name__ == '__main__':
    # Process both folders
    folders = [
        'data/Hộp tự in 2 vị trí',
        'data/Thùng in sẵn'
    ]
    
    for folder in folders:
        print(f"\n{'='*60}")
        print(f"Processing: {folder}")
        print('='*60)
        split_images(folder)