"""
Frame Saver Utility
Saves camera frames to disk with proper organization
"""

import cv2
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


def save_frame_to_disk(
    serial_number: str,
    frame: np.ndarray,
    base_dir: str = "/home/demo/Source/ocr_datecode/backend/uploads/save_frame",
    quality: int = 95
) -> Optional[str]:
    """
    Save camera frame to disk with organized directory structure

    Args:
        serial_number: Camera serial number
        frame: Frame image (numpy array)
        base_dir: Base directory for saving frames
        quality: JPEG quality (0-100)

    Returns:
        Relative path to saved file or None on error

    Directory structure:
        save_frame/{serial_number}/{YYYY_MM_DD}/{serial_number}_YYYY_MM_DD+HH_MM_SS_fff.jpg

    Example:
        save_frame/24166352/2026_01_15/24166352_2026_01_15+11_04_30_123.jpg
    """
    try:
        # Get current timestamp
        now = datetime.now()
        date_str = now.strftime("%Y_%m_%d")
        time_str = now.strftime("%Y_%m_%d+%H_%M_%S_%f")[:-3]  # Remove last 3 digits (microseconds → milliseconds)

        # Create directory structure: base_dir/{serial_number}/{YYYY_MM_DD}/
        save_dir = Path(base_dir) / serial_number / date_str
        save_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename: {serial_number}_YYYY_MM_DD+HH_MM_SS_fff.jpg
        filename = f"{serial_number}_{time_str}.jpg"
        full_path = save_dir / filename

        # Save frame with specified quality
        success = cv2.imwrite(str(full_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

        if not success:
            logger.error(f"Failed to save frame to {full_path}")
            return None

        # Return relative path from uploads/
        relative_path = f"save_frame/{serial_number}/{date_str}/{filename}"

        logger.debug(f"Saved frame: {relative_path}")
        return relative_path

    except Exception as e:
        logger.error(f"Error saving frame for {serial_number}: {e}")
        import traceback
        traceback.print_exc()
        return None
