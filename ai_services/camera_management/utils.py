"""
Utility functions for camera management
Pure functions without state dependencies
"""

import cv2
import subprocess
import logging
import base64
from typing import Optional, Tuple
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ============= GPIO/DI Utilities =============

def read_di_value(di_number: int) -> int:
    """
    Read Digital Input pin value (0 or 1)

    Args:
        di_number: DI pin number (0-3)

    Returns:
        Pin value (0 or 1), or 0 on error
    """
    try:
        result = subprocess.run(
            ["sudo", "dio_in", str(di_number)],
            capture_output=True,
            text=True,
            timeout=0.5
        )

        if result.returncode == 0:
            # Parse output format: "The id-X input gpio status = Y\nCompletion code = 0x00"
            output = result.stdout.strip()

            # Extract value from "status = Y" line
            for line in output.split('\n'):
                if 'status' in line and '=' in line:
                    # Extract the value after '='
                    value_str = line.split('=')[-1].strip()
                    try:
                        value = int(value_str)
                        return value
                    except ValueError:
                        logger.warning(f"Failed to parse DI{di_number} value: {value_str}")
                        return 0

            # Fallback: couldn't find status line
            logger.warning(f"Unexpected DI{di_number} output format: {output}")
            return 0
        else:
            logger.warning(f"Failed to read DI{di_number}: {result.stderr.strip()}")
            return 0

    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout reading DI{di_number}")
        return 0
    except Exception as e:
        logger.error(f"Error reading DI{di_number}: {e}")
        return 0


def check_trigger_edge(current: int, previous: Optional[int], activation: str) -> bool:
    """
    Check if current value matches trigger activation edge

    Args:
        current: Current DI value (0 or 1)
        previous: Previous DI value (0, 1, or None)
        activation: Trigger activation type (RisingEdge, FallingEdge, AnyEdge)

    Returns:
        True if edge detected, False otherwise
    """
    if previous is None:
        return False

    if activation == "RisingEdge":
        return previous == 0 and current == 1
    elif activation == "FallingEdge":
        return previous == 1 and current == 0
    elif activation == "AnyEdge":
        return previous != current
    else:
        logger.warning(f"Unknown trigger activation: {activation}")
        return False


# ============= Image Processing Utilities =============

def resize_for_display(frame_img, scale_factor: int = 3):
    """
    Resize frame by dividing dimensions

    Args:
        frame_img: Input frame (numpy array)
        scale_factor: Division factor for width and height

    Returns:
        Resized frame
    """
    h, w = frame_img.shape[:2]
    new_w = w // scale_factor
    new_h = h // scale_factor
    return cv2.resize(frame_img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def encode_image_to_base64(img, quality: int = 70) -> str:
    """
    Encode image to JPEG base64

    Args:
        img: Input image (numpy array)
        quality: JPEG quality (0-100)

    Returns:
        Base64 encoded string
    """
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')


def save_and_encode_frame(
    frame_img,
    serial_number: str,
    recipe_id: str,
    pass_fail: str,
    frame_idx: int,
    base_dir: str = "/home/demo/Source/ocr_datecode/backend/uploads/inference_results"
) -> Tuple[Optional[str], Optional[str]]:
    """
    Save frame DIRECTLY to permanent storage and create resized version for display

    Args:
        frame_img: Input frame (numpy array)
        serial_number: Camera serial number
        recipe_id: Recipe ID (for directory structure)
        pass_fail: Pass/fail status (for filename)
        frame_idx: Frame index
        base_dir: Base directory for permanent storage

    Returns:
        Tuple of (relative_image_path, image_base64) or (None, None) on error
        - relative_image_path: Relative path from uploads/ (e.g., "inference_results/recipe_id/2026-01-04/xxx.jpg")
        - image_base64: Base64 encoded resized image for display
    """
    try:
        # Create directory structure: base_dir/{recipe_id}/{YYYY-MM-DD}/
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        storage_dir = Path(base_dir) / recipe_id / today
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S%f")
        filename = f"{serial_number}_{timestamp}_{pass_fail.lower()}_f{frame_idx}.jpg"
        full_path = storage_dir / filename

        # Get original dimensions
        h, w = frame_img.shape[:2]

        # Save FULL resolution to permanent storage
        cv2.imwrite(str(full_path), frame_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Relative path for DB and API (from uploads/)
        relative_path = f"inference_results/{recipe_id}/{today}/{filename}"

        # Create RESIZED + COMPRESSED version for realtime display (divide by 3)
        display_img = resize_for_display(frame_img, scale_factor=3)
        image_base64 = encode_image_to_base64(display_img, quality=70)

        logger.info(
            f"Saved {pass_fail} frame: {relative_path} "
            f"(full: {w}x{h}, display: {display_img.shape[1]}x{display_img.shape[0]})"
        )

        return relative_path, image_base64

    except Exception as e:
        logger.error(f"Error saving/encoding frame: {e}")
        import traceback
        traceback.print_exc()
        return None, None
