"""
Utility functions for camera management
Pure functions without state dependencies
"""

import cv2
import subprocess
import logging
import base64
import numpy as np
import threading
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ============= GPIO/DI/DO Utilities =============

# Global lock to prevent concurrent DI/DO operations
# Advantech driver conflicts when multiple processes access it simultaneously
_gpio_lock = threading.Lock()

def read_di_value(di_number: int) -> int:
    """
    Read Digital Input pin value (0 or 1)

    Args:
        di_number: DI pin number (0-3)

    Returns:
        Pin value (0 or 1), or 0 on error

    Note:
        Uses global lock to prevent conflicts with DO operations.
        Advantech driver returns errors when DI read and DO write happen simultaneously.
    """
    with _gpio_lock:  # Prevent conflict with DO operations
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


def write_do_value(do_number: int, value: int) -> bool:
    """
    Set Digital Output pin value

    Args:
        do_number: DO pin number (0-7)
        value: Pin value (0 or 1)

    Returns:
        True if success, False on error

    Command format:
        sudo dio_out <do_number> <value>

    Example:
        sudo dio_out 2 1  # Set DO2 = HIGH
        sudo dio_out 2 0  # Set DO2 = LOW

    Note:
        Uses global lock to prevent conflicts with DI operations.
        Advantech driver returns errors when DI read and DO write happen simultaneously.
        Also returns returncode=255 with "Completion code = 0xFFFFFFFF"
        when trying to set same value twice. We check the output message instead.
    """
    with _gpio_lock:  # Prevent conflict with DI operations
        try:
            result = subprocess.run(
                ["sudo", "dio_out", str(do_number), str(value)],
                capture_output=True,
                text=True,
                timeout=1.0
            )

            # Check output for success indicator
            # Success: "Completion code = 0x00"
            # Failure: "Completion code = 0xFFFFFFFF"
            if "Completion code = 0x00" in result.stdout or "Completion code = 0x00" in result.stderr:
                logger.debug(f"DO{do_number} = {value}")
                return True
            elif "Completion code = 0xFFFFFFFF" in result.stdout or "Completion code = 0xFFFFFFFF" in result.stderr:
                # This happens when setting same value twice - treat as warning, not error
                logger.debug(
                    f"DO{do_number} already at {value} (Advantech driver quirk)"
                )
                return True  # Treat as success since pin is already at desired state
            elif result.returncode == 0:
                # Fallback: if returncode is 0, assume success
                logger.debug(f"DO{do_number} = {value} (returncode=0)")
                return True
            else:
                logger.error(
                    f"Failed to set DO{do_number}: returncode={result.returncode}, "
                    f"stdout={result.stdout.strip()}, stderr={result.stderr.strip()}"
                )
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout setting DO{do_number}")
            return False
        except Exception as e:
            logger.error(f"Error setting DO{do_number}: {e}")
            return False


def trigger_reject_pulse(do_number: int, pulse_ms: int = 100):
    """
    Trigger reject pulse on DO pin (ACTIVE LOW logic)

    Args:
        do_number: DO pin number (typically 2 for reject)
        pulse_ms: Pulse duration in milliseconds (default: 100ms)

    Raises:
        RuntimeError: If DO control fails

    Hardware Logic (ACTIVE LOW):
        - Idle state: DO = HIGH (1)
        - Active state: DO = LOW (0) → Solenoid activates → Product rejected

    Timing:
        t=0ms: DO = HIGH (ensure idle state)
        t=1ms: DO = LOW (activate solenoid - REJECT!)
        t=pulse_ms: DO = HIGH (deactivate - return to idle)

    Example:
        trigger_reject_pulse(2, 100)  # 100ms LOW pulse on DO2

    Note:
        Advantech DIO driver may return error if setting same value twice.
        We ensure HIGH state first to avoid this issue.
    """
    import time

    # Ensure HIGH state first (idle state)
    # Ignore error if already HIGH (Advantech driver quirk)
    write_do_value(do_number, 1)
    time.sleep(0.001)  # 1ms delay to ensure state change

    # Set LOW (ACTIVATE REJECT!)
    if not write_do_value(do_number, 0):
        raise RuntimeError(f"Failed to set DO{do_number} LOW (activate reject)")

    # Hold pulse (reject active)
    time.sleep(pulse_ms / 1000.0)

    # Set HIGH (deactivate - return to idle)
    if not write_do_value(do_number, 1):
        raise RuntimeError(f"Failed to set DO{do_number} HIGH (deactivate reject)")

    logger.info(f"DO{do_number} pulse complete ({pulse_ms}ms, active LOW)")


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


def draw_inference_bboxes(
    img: np.ndarray,
    transformed_bboxes: List[Dict[str, Any]],
    confidence: float,
    inliers: int,
    total_matches: int,
    crop_area: Optional[Dict[str, int]] = None
) -> np.ndarray:
    result_img = img.copy()
    height, width = img.shape[:2]

    # Calculate dynamic sizes based on image dimensions
    font_scale = max(0.5, min(2.0, width / 2000))  # Adaptive font size
    line_thickness = max(2, int(width / 800))       # Adaptive line thickness
    text_thickness = max(1, int(width / 1200))      # Adaptive text thickness

    # Define colors for different bbox types (BGR format)
    colors = {
        'template': (0, 255, 0),       # Green
        'text': (0, 165, 255),         # Orange
        'barcode': (255, 0, 255),      # Magenta
        'datecode': (255, 255, 0),     # Cyan
        'product': (255, 128, 0),      # Blue (for product region)
        'label': (128, 0, 255)         # Purple (for label region)
    }
    logger.debug(f"Drawing {len(transformed_bboxes)} bboxes")

    # Draw crop_area first (as background)
    if crop_area:
        cv2.rectangle(
            result_img,
            (crop_area['x1'], crop_area['y1']),
            (crop_area['x2'], crop_area['y2']),
            (0, 255, 255),  # Yellow
            line_thickness,
            cv2.LINE_AA
        )
        # Add label with background
        label_text = "CROP AREA"
        (label_w, label_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, text_thickness
        )
        label_x, label_y = crop_area['x1'] + 5, crop_area['y1'] + 5
        # Draw background rectangle
        cv2.rectangle(
            result_img,
            (label_x, label_y),
            (label_x + label_w + 6, label_y + label_h + baseline + 4),
            (0, 255, 255),
            -1
        )
        # Draw text
        cv2.putText(
            result_img,
            label_text,
            (label_x + 3, label_y + label_h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.7,
            (0, 0, 0),  # Black text on yellow background
            text_thickness,
            cv2.LINE_AA
        )

    # Draw each bbox
    for bbox in transformed_bboxes:
        bbox_type = bbox.get('type', 'text')
        points = bbox.get('points', [])
        text_label = bbox.get('text', '')

        if len(points) < 3:
            continue

        # Get color based on type
        color = colors.get(bbox_type, (255, 255, 255))

        # Override with RED if verification failed
        if bbox.get('verification_status') == 'fail':
            color = (0, 0, 255)  # Red (BGR)

        # Convert points to numpy array
        pts = np.array(points, dtype=np.int32)

        # Draw polygon with anti-aliasing
        cv2.polylines(result_img, [pts], isClosed=True, color=color,
                     thickness=line_thickness, lineType=cv2.LINE_AA)

        # Draw label with background box if available
        if text_label and len(points) > 0:
            # Get text size
            (text_w, text_h), baseline = cv2.getTextSize(
                text_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
            )

            # Calculate label position (above the bbox, left-aligned)
            label_x = int(points[0][0])
            label_y = int(points[0][1]) - text_h - 6

            # Ensure label stays within image bounds
            if label_y < 0:
                label_y = int(points[0][1]) + text_h + 6
            if label_x + text_w + 6 > width:
                label_x = width - text_w - 6
            if label_x < 0:
                label_x = 0

            # # Draw background rectangle with padding
            # cv2.rectangle(
            #     result_img,
            #     (label_x, label_y - text_h - 4),
            #     (label_x + text_w + 6, label_y + baseline),
            #     color,
            #     -1  # Filled rectangle
            # )

            # # Draw text in white on colored background
            # cv2.putText(
            #     result_img,
            #     text_label,
            #     (label_x + 3, label_y - 2),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     font_scale,
            #     (255, 255, 255),  # White text
            #     text_thickness,
            #     cv2.LINE_AA
            # )

    # Draw inference stats at top-left with background
    stats_text = f"Conf: {confidence:.1%} | Inliers: {inliers}/{total_matches}"
    (stats_w, stats_h), baseline = cv2.getTextSize(
        stats_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
    )

    # Choose color based on confidence
    stats_color = (0, 255, 0) if confidence > 0.5 else (0, 0, 255)

    # Draw background rectangle
    cv2.rectangle(
        result_img,
        (5, 5),
        (15 + stats_w, 10 + stats_h + baseline),
        stats_color,
        -1
    )

    # Draw stats text
    cv2.putText(
        result_img,
        stats_text,
        (10, 10 + stats_h),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),  # White text
        text_thickness,
        cv2.LINE_AA
    )

    return result_img


def draw_detected_obb_boxes(
    img: np.ndarray,
    detected_boxes: Optional[Dict[str, Any]],
    show_details: bool = True
) -> np.ndarray:
    """
    Draw detected OBB boxes from YOLO on the image.

    Args:
        img: Input image
        detected_boxes: Dict with 'product' and 'label' detected boxes
        show_details: Whether to show confidence and angles

    Returns:
        Image with drawn OBB boxes
    """
    if not detected_boxes:
        return img

    result_img = img.copy()
    height, width = img.shape[:2]

    # Calculate dynamic sizes
    font_scale = max(0.4, min(1.5, width / 2500))
    line_thickness = max(2, int(width / 1000))
    text_thickness = max(1, int(width / 1500))

    # Colors for detected boxes (brighter colors to distinguish from template regions)
    colors = {
        'product': (0, 255, 255),   # Yellow (detected product)
        'label': (255, 0, 255)      # Magenta (detected label)
    }

    # Draw each detected box
    for box_type in ['product', 'label']:
        if box_type not in detected_boxes:
            continue

        box_data = detected_boxes[box_type]
        if not box_data or not isinstance(box_data, dict):
            continue

        # Get corners
        corners = box_data.get('corners')
        if corners is None:
            continue

        # Convert to numpy array if needed
        if isinstance(corners, list):
            corners = np.array(corners, dtype=np.int32)
        else:
            corners = corners.astype(np.int32)

        # Get color
        color = colors.get(box_type, (255, 255, 255))

        # Draw OBB polygon with thicker line (dashed style)
        for i in range(4):
            pt1 = tuple(corners[i])
            pt2 = tuple(corners[(i + 1) % 4])
            cv2.line(result_img, pt1, pt2, color, line_thickness + 1, cv2.LINE_AA)

        # Draw corner points
        for corner in corners:
            cv2.circle(result_img, tuple(corner), max(3, line_thickness), color, -1)

        # Prepare label text
        score = box_data.get('score', 0.0)
        label_parts = [f"{box_type.upper()}"]

        if show_details:
            label_parts.append(f"{score:.2f}")

        label_text = " ".join(label_parts)

        # Calculate label position (top-left corner of box)
        label_x = int(corners[0][0])
        label_y = int(corners[0][1]) - 8

        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )

        # Ensure label stays within image bounds
        if label_y - text_h < 0:
            label_y = int(corners[0][1]) + text_h + 8
        if label_x + text_w + 8 > width:
            label_x = width - text_w - 8
        if label_x < 0:
            label_x = 0

        # Draw background rectangle for label
        cv2.rectangle(
            result_img,
            (label_x - 2, label_y - text_h - 4),
            (label_x + text_w + 2, label_y + baseline + 2),
            color,
            -1  # Filled
        )

        # Draw label text
        cv2.putText(
            result_img,
            label_text,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),  # Black text on colored background
            text_thickness,
            cv2.LINE_AA
        )

    return result_img


def encode_frame_for_display(
    frame_img: np.ndarray,
    transformed_bboxes: Optional[List[Dict[str, Any]]] = None,
    confidence: float = 0.0,
    inliers: int = 0,
    total_matches: int = 0,
    crop_area: Optional[Dict[str, int]] = None,
    product_verification: Optional[Dict[str, Any]] = None,
    scale_factor: int = 3,
    quality: int = 70
) -> Optional[str]:
    try:
        # Draw bboxes if provided
        img_to_encode = frame_img.copy()
        if transformed_bboxes and len(transformed_bboxes) > 0:
            img_to_encode = draw_inference_bboxes(
                img_to_encode,
                transformed_bboxes,
                confidence,
                inliers,
                total_matches,
                crop_area
            )

        # Draw detected OBB boxes from YOLO (if product verification was performed)
        if product_verification and not product_verification.get('skipped', False):
            detected_boxes = product_verification.get('detected_boxes')
            if detected_boxes:
                img_to_encode = draw_detected_obb_boxes(
                    img_to_encode,
                    detected_boxes,
                    show_details=True
                )

        # Resize for display
        display_img = resize_for_display(img_to_encode, scale_factor=scale_factor)

        # Encode to base64
        image_base64 = encode_image_to_base64(display_img, quality=quality)

        return image_base64

    except Exception as e:
        logger.error(f"Error encoding frame for display: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_and_encode_frame(
    frame_img,
    serial_number: str,
    recipe_id: str,
    pass_fail: str,
    frame_idx: int,
    base_dir: str = "/home/demo/Source/ocr_datecode/backend/uploads/inference_results",
    transformed_bboxes: Optional[List[Dict[str, Any]]] = None,
    confidence: float = 0.0,
    inliers: int = 0,
    total_matches: int = 0,
    crop_area: Optional[Dict[str, int]] = None,
    product_verification: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Save frame DIRECTLY to permanent storage and create resized version for display
    Optionally draw inference bboxes and crop_area on the image

    Args:
        frame_img: Input frame (numpy array)
        serial_number: Camera serial number
        recipe_id: Recipe ID (for directory structure)
        pass_fail: Pass/fail status (for filename)
        frame_idx: Frame index
        base_dir: Base directory for permanent storage
        transformed_bboxes: Optional list of bbox dicts from inference
        confidence: Confidence score (for drawing)
        inliers: Number of inliers (for drawing)
        total_matches: Total matches (for drawing)
        crop_area: Optional crop area dict to visualize

    Returns:
        Tuple of (relative_image_path, image_base64) or (None, None) on error
        - relative_image_path: Relative path from uploads/ (e.g., "inference_results/recipe_id/2026-01-04/xxx.jpg")
        - image_base64: Base64 encoded resized image for display
    """
    try:
        # Draw bboxes if provided (for inference frames)
        img_to_save = frame_img.copy()
        if transformed_bboxes and len(transformed_bboxes) > 0:
            img_to_save = draw_inference_bboxes(
                img_to_save,
                transformed_bboxes,
                confidence,
                inliers,
                total_matches,
                crop_area  # Pass crop_area to visualization
            )
            logger.info(f"Drew {len(transformed_bboxes)} bboxes on frame")

        # Draw detected OBB boxes from YOLO (if product verification was performed)
        if product_verification and not product_verification.get('skipped', False):
            detected_boxes = product_verification.get('detected_boxes')
            if detected_boxes:
                img_to_save = draw_detected_obb_boxes(
                    img_to_save,
                    detected_boxes,
                    show_details=True
                )
                logger.info(f"Drew detected OBB boxes (product + label) on frame")

        # Create directory structure: base_dir/{recipe_id}/{YYYY-MM-DD}/{camera_serial}/
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        storage_dir = Path(base_dir) / recipe_id / today / serial_number
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S%f")
        filename_viz = f"{pass_fail.lower()}_f{frame_idx}_{timestamp}_viz.jpg"
        filename_org = f"{pass_fail.lower()}_f{frame_idx}_{timestamp}_org.jpg"
        full_path_viz = storage_dir / filename_viz
        full_path_org = storage_dir / filename_org

        # Get original dimensions
        h, w = img_to_save.shape[:2]

        # Save FULL resolution to permanent storage (with bboxes drawn if applicable)
        cv2.imwrite(str(full_path_viz), img_to_save, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(full_path_org), frame_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Relative path for DB and API (from uploads/)
        relative_path = f"inference_results/{recipe_id}/{today}/{serial_number}/{filename_viz}"

        # Create RESIZED + COMPRESSED version for realtime display (divide by 3)
        display_img = resize_for_display(img_to_save, scale_factor=3)
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
