"""
Utility functions for camera management
Pure functions without state dependencies
"""

import cv2
import ctypes
import subprocess
import logging
import base64
import numpy as np
import threading
import atexit
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
from datetime import datetime, timezone


logger = logging.getLogger(__name__)


# ============= GPIO/DI/DO Utilities =============

# Global lock for ALL GPIO operations (libapmi is NOT thread-safe!)
# DI read and DO write must be serialized to prevent ret=-1 errors
_gpio_lock = threading.Lock()

# Per-pin locks for pulse sequences (prevent concurrent writes during pulse)
# Use RLock to allow reentrant calls from trigger_reject_pulse
_pulse_locks = {i: threading.RLock() for i in range(8)}

# ============= ASUS libapmi Native Library (Fast GPIO) =============
# Uses ctypes to call ASUS PE1100N native library directly
# Much faster than subprocess (~21ms vs ~37ms per call)

_libapmi = None
_apmi_dio_read_input = None
_apmi_dio_write_output = None
_apmi_dio_read_output = None
_use_native_gpio = False

# libapmi uses power-of-2 pin mapping: DO0->1, DO1->2, DO2->4, DO3->8
_LIBAPMI_PIN_MAP = {0: 1, 1: 2, 2: 4, 3: 8}

def _init_libapmi():
    """Initialize ASUS libapmi library for native GPIO access"""
    global _libapmi, _apmi_dio_read_input, _apmi_dio_write_output, _apmi_dio_read_output, _use_native_gpio

    try:
        _libapmi = ctypes.CDLL('/lib/libapmi.so')

        # Setup apmi_dio_read_input(int pin, int *value) -> int
        _apmi_dio_read_input = _libapmi.apmi_dio_read_input
        _apmi_dio_read_input.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        _apmi_dio_read_input.restype = ctypes.c_int

        # Setup apmi_dio_write_output(int pin, int value) -> int
        _apmi_dio_write_output = _libapmi.apmi_dio_write_output
        _apmi_dio_write_output.argtypes = [ctypes.c_int, ctypes.c_int]
        _apmi_dio_write_output.restype = ctypes.c_int

        # Setup apmi_dio_read_output(int pin, int *value) -> int
        _apmi_dio_read_output = _libapmi.apmi_dio_read_output
        _apmi_dio_read_output.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        _apmi_dio_read_output.restype = ctypes.c_int

        _use_native_gpio = True
        # print("ASUS libapmi initialized - using native GPIO (fast mode)")   
        logger.info("ASUS libapmi initialized - using native GPIO (fast mode)")

    except Exception as e:
        logger.warning(f"Failed to load libapmi, falling back to subprocess: {e}")
        _use_native_gpio = False

# Initialize on module load
_init_libapmi()

def read_di_value(di_number: int) -> int:
    """
    Read Digital Input pin value (0 or 1)

    Args:
        di_number: DI pin number (0-3)

    Returns:
        Pin value (0 or 1), or 0 on error

    Note:
        Uses global _gpio_lock because libapmi is not thread-safe.
    """
    with _gpio_lock:
        # Try native library first (faster)
        if _use_native_gpio and _apmi_dio_read_input is not None:
            try:
                value = ctypes.c_int()
                lib_pin = _LIBAPMI_PIN_MAP.get(di_number)
                if lib_pin is not None:
                    ret = _apmi_dio_read_input(lib_pin, ctypes.byref(value))
                    if ret == 0:
                        # print(f"DI{di_number} = {value.value} (native)")
                        return value.value
                    else:
                        print(f"Native DI{di_number} read failed: ret={ret}")
                        logger.warning(f"Native DI{di_number} read failed: ret={ret}")
            except Exception as e:
                logger.warning(f"Native DI{di_number} read error: {e}")

        # Fallback to subprocess
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


def _write_do_raw(do_number: int, value: int) -> bool:
    """
    Internal: Write DO value with only _gpio_lock (no pulse_lock).
    Used by trigger_reject_pulse which already holds pulse_lock.

    MUST be called while holding pulse_lock for the pin!
    """
    with _gpio_lock:
        # Try native library first (faster)
        if _use_native_gpio and _apmi_dio_write_output is not None:
            try:
                lib_pin = _LIBAPMI_PIN_MAP.get(do_number)
                if lib_pin is not None:
                    ret = _apmi_dio_write_output(lib_pin, value)
                    if ret == 0:
                        print(f"DO{do_number} = {value} (native)")
                        logger.debug(f"DO{do_number} = {value} (native)")
                        return True
                    else:
                        print(f"Native DO{do_number} write failed: ret={ret}")
                        logger.warning(f"Native DO{do_number} write failed: ret={ret}")
            except Exception as e:
                print(f"Native DO{do_number} write error: {e}")
                logger.warning(f"Native DO{do_number} write error: {e}")

        # Fallback to subprocess
        try:
            result = subprocess.run(
                ["sudo", "dio_out", str(do_number), str(value)],
                capture_output=True,
                text=True,
                timeout=1.0
            )

            if "Completion code = 0x00" in result.stdout or "Completion code = 0x00" in result.stderr:
                logger.debug(f"DO{do_number} = {value}")
                return True
            elif "Completion code = 0xFFFFFFFF" in result.stdout or "Completion code = 0xFFFFFFFF" in result.stderr:
                logger.debug(f"DO{do_number} already at {value}")
                return True
            elif result.returncode == 0:
                logger.debug(f"DO{do_number} = {value} (returncode=0)")
                return True
            else:
                logger.error(f"Failed to set DO{do_number}: {result.returncode}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout setting DO{do_number}")
            return False
        except Exception as e:
            logger.error(f"Error setting DO{do_number}: {e}")
            return False


def write_do_value(do_number: int, value: int) -> bool:
    """
    Set Digital Output pin value (public API).

    Args:
        do_number: DO pin number (0-7)
        value: Pin value (0 or 1)

    Returns:
        True if success, False on error

    Note:
        Uses per-pin pulse lock to prevent writes during active pulse.
        If a pulse is in progress on this pin, this call will block until complete.
    """
    pulse_lock = _pulse_locks.get(do_number)
    if pulse_lock is None:
        logger.error(f"Invalid DO pin: {do_number}")
        return False

    with pulse_lock:  # Wait for any active pulse on this pin
        return _write_do_raw(do_number, value)


def trigger_reject_pulse(do_number: int, pulse_ms: int = 100):
    """
    Trigger reject pulse on DO pin (ACTIVE LOW logic).

    Pulse sequence: HIGH -> LOW (pulse_ms) -> HIGH
    Uses per-pin lock to prevent concurrent writes during pulse.

    Args:
        do_number: DO pin number (0-3)
        pulse_ms: Pulse duration in milliseconds

    Raises:
        RuntimeError: If write fails
    """
    import time

    pulse_lock = _pulse_locks.get(do_number)
    if pulse_lock is None:
        raise RuntimeError(f"Invalid DO pin: {do_number}")

    with pulse_lock:
        # Set LOW (active)
        if not _write_do_raw(do_number, 0):
            raise RuntimeError(f"Failed to set DO{do_number} LOW")

        # Hold pulse - _gpio_lock is FREE during sleep, DI reads can proceed
        time.sleep(pulse_ms / 1000.0)

        # Set HIGH (inactive)
        if not _write_do_raw(do_number, 1):
            raise RuntimeError(f"Failed to set DO{do_number} HIGH")

    print(f"DO{do_number} pulse complete ({pulse_ms}ms)")
    logger.info(f"DO{do_number} pulse complete ({pulse_ms}ms)")


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


def draw_center_points(
    img: np.ndarray,
    center_alignment_check: Optional[Dict[str, Any]]
) -> np.ndarray:
    """
    Draw center points of template and product boxes.

    Args:
        img: Input image
        center_alignment_check: Dict with 'template_center' and 'product_center'

    Returns:
        Image with drawn center points
    """
    if not center_alignment_check or center_alignment_check.get('skipped', False):
        return img

    result_img = img.copy()
    height, width = img.shape[:2]

    # Calculate dynamic sizes
    font_scale = max(0.4, min(1.5, width / 2500))
    text_thickness = max(1, int(width / 1500))
    circle_radius = max(15, int(width / 300))  # Adaptive circle size

    # Get centers
    template_center = center_alignment_check.get('template_center')
    product_center = center_alignment_check.get('product_center')

    # Draw template center (cyan)
    if template_center:
        cv2.circle(
            result_img,
            (int(template_center[0]), int(template_center[1])),
            circle_radius,
            (255, 255, 0),  # Cyan
            -1  # Filled
        )
        cv2.putText(
            result_img, "Template",
            (int(template_center[0]) + circle_radius + 5, int(template_center[1]) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.8,
            (255, 255, 0),
            text_thickness,
            cv2.LINE_AA
        )

    # Draw product center (magenta)
    if product_center:
        cv2.circle(
            result_img,
            (int(product_center[0]), int(product_center[1])),
            circle_radius,
            (255, 0, 255),  # Magenta
            -1  # Filled
        )
        cv2.putText(
            result_img, "Product",
            (int(product_center[0]) + circle_radius + 5, int(product_center[1]) + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.8,
            (255, 0, 255),
            text_thickness,
            cv2.LINE_AA
        )

    # Draw x-axis distance between centers
    if template_center and product_center:
        x1, y1 = int(template_center[0]), int(template_center[1])
        x2, y2 = int(product_center[0]), int(product_center[1])

        # Calculate x-axis distance
        x_distance = abs(x2 - x1)

        # Draw horizontal line at the average y position
        y_line = (y1 + y2) // 2
        line_thickness = max(2, int(width / 1000))

        # Draw line from template x to product x
        cv2.line(
            result_img,
            (x1, y_line),
            (x2, y_line),
            (0, 255, 255),  # Yellow
            line_thickness,
            cv2.LINE_AA
        )

        # Draw vertical ticks at both ends
        tick_length = circle_radius // 2
        cv2.line(result_img, (x1, y_line - tick_length), (x1, y_line + tick_length), (0, 255, 255), line_thickness, cv2.LINE_AA)
        cv2.line(result_img, (x2, y_line - tick_length), (x2, y_line + tick_length), (0, 255, 255), line_thickness, cv2.LINE_AA)

        # Draw distance text in the middle
        mid_x = (x1 + x2) // 2
        distance_text = f"{x_distance}px"

        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(
            distance_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_thickness
        )

        # Draw background rectangle for text
        padding = 5
        cv2.rectangle(
            result_img,
            (mid_x - text_w // 2 - padding, y_line - text_h - padding - 5),
            (mid_x + text_w // 2 + padding, y_line - padding - 5),
            (0, 0, 0),  # Black background
            -1
        )

        # Draw distance text
        cv2.putText(
            result_img,
            distance_text,
            (mid_x - text_w // 2, y_line - padding - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 255),  # Red
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
        'label': (255, 0, 255),     # Magenta (detected label)
        'wrinkled': (0, 165, 255)   # Orange (detected wrinkled)
    }

    # Draw each detected box
    for box_type in ['product', 'label', 'wrinkled']:
        if box_type not in detected_boxes:
            continue

        box_data = detected_boxes[box_type]
        if not box_data:
            continue

        # Handle wrinkled as list of boxes, others as single dict
        boxes_to_draw = box_data if isinstance(box_data, list) else [box_data]

        for idx, single_box in enumerate(boxes_to_draw):
            if not isinstance(single_box, dict):
                continue

            # Get corners
            corners = single_box.get('corners')
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
            score = single_box.get('score', 0.0)
            label_parts = [f"{box_type.upper()}"]

            # For wrinkled with multiple boxes, add index
            if isinstance(box_data, list) and len(boxes_to_draw) > 1:
                label_parts[0] = f"{box_type.upper()}#{idx+1}"

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

            # Draw center points (template center and product center)
            center_alignment_check = product_verification.get('center_alignment_check')
            if center_alignment_check:
                img_to_encode = draw_center_points(
                    img_to_encode,
                    center_alignment_check
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


# ============= Async Image Saver =============

class AsyncImageSaver:
    """
    Singleton class for async image saving using ThreadPoolExecutor.

    Benefits:
    - Save images to disk without blocking the main inference pipeline
    - Encode base64 synchronously for real-time display
    - Multiple cameras save in parallel
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, max_workers: int = 4):
        """
        Initialize AsyncImageSaver.

        Args:
            max_workers: Maximum number of concurrent save operations
        """
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="AsyncImageSaver"
        )
        self._pending_count = 0
        self._count_lock = threading.Lock()
        self._shutdown = False

        # Register cleanup on exit
        atexit.register(self.shutdown)

        logger.info(f"AsyncImageSaver initialized with {max_workers} workers")

    @classmethod
    def get_instance(cls, max_workers: int = 4) -> 'AsyncImageSaver':
        """Get singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = AsyncImageSaver(max_workers=max_workers)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)"""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.shutdown(wait=True)
                cls._instance = None

    def save_images_async(
        self,
        img_viz: np.ndarray,
        img_org: np.ndarray,
        path_viz: Path,
        path_org: Path,
        quality: int = 95
    ) -> None:
        """
        Submit image save task to background thread.

        Args:
            img_viz: Visualization image (with bboxes)
            img_org: Original image (no bboxes)
            path_viz: Path for visualization image
            path_org: Path for original image
            quality: JPEG quality (1-100)
        """
        if self._shutdown:
            logger.warning("AsyncImageSaver is shutdown, saving synchronously")
            self._do_save(img_viz, img_org, path_viz, path_org, quality)
            return

        with self._count_lock:
            self._pending_count += 1

        self.executor.submit(
            self._do_save,
            img_viz, img_org, path_viz, path_org, quality
        )

    def _do_save(
        self,
        img_viz: np.ndarray,
        img_org: np.ndarray,
        path_viz: Path,
        path_org: Path,
        quality: int
    ) -> None:
        """
        Actually save images to disk (runs in background thread).

        Args:
            img_viz: Visualization image
            img_org: Original image
            path_viz: Path for viz image
            path_org: Path for org image
            quality: JPEG quality
        """
        import time
        t_start = time.perf_counter()
        try:
            cv2.imwrite(str(path_viz), img_viz, [cv2.IMWRITE_JPEG_QUALITY, quality])
            cv2.imwrite(str(path_org), img_org, [cv2.IMWRITE_JPEG_QUALITY, quality])
            t_elapsed = (time.perf_counter() - t_start) * 1000
            logger.debug(f"AsyncImageSaver: saved {path_viz.name} in {t_elapsed:.1f}ms (pending: {self._pending_count - 1})")
        except Exception as e:
            logger.error(f"AsyncImageSaver error saving images: {e}")
        finally:
            with self._count_lock:
                self._pending_count -= 1

    def get_pending_count(self) -> int:
        """Get number of pending save operations"""
        with self._count_lock:
            return self._pending_count

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the executor.

        Args:
            wait: Whether to wait for pending tasks to complete
        """
        if self._shutdown:
            return

        self._shutdown = True
        pending = self.get_pending_count()

        if pending > 0:
            logger.info(f"AsyncImageSaver shutting down, waiting for {pending} pending saves...")

        self.executor.shutdown(wait=wait)
        logger.info("AsyncImageSaver shutdown complete")


# ============= Background Result Emitter =============

class BackgroundResultEmitter:
    """
    Background worker for encoding images and emitting results.

    This allows the inference pipeline to complete immediately after verification,
    while encoding and emitting happens in background threads.

    Benefits:
    - Pipeline complete time excludes encoding time (~38ms for 3 cameras)
    - Encoding can run in parallel for multiple cameras
    - Real-time display still gets images, just slightly delayed
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, max_workers: int = 4):
        """
        Initialize BackgroundResultEmitter.

        Args:
            max_workers: Maximum concurrent encoding operations
        """
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ResultEmitter"
        )
        self._pending_count = 0
        self._count_lock = threading.Lock()
        self._shutdown = False
        self._emit_callback = None
        self._event_loop = None

        # Register cleanup on exit
        atexit.register(self.shutdown)

        logger.info(f"BackgroundResultEmitter initialized with {max_workers} workers")

    @classmethod
    def get_instance(cls, max_workers: int = 4) -> 'BackgroundResultEmitter':
        """Get singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = BackgroundResultEmitter(max_workers=max_workers)
        return cls._instance

    def set_emit_callback(self, callback, event_loop=None):
        """
        Set the emit callback function.

        Args:
            callback: Function to call with (event_type, data)
            event_loop: Asyncio event loop for async callbacks
        """
        self._emit_callback = callback
        self._event_loop = event_loop

    def emit_result_async(
        self,
        result_data: Dict[str, Any],
        encode_tasks: List[Dict[str, Any]],
        save_and_encode_func,
        encode_display_func
    ) -> None:
        """
        Queue result for background encoding and emit.

        Args:
            result_data: Base result dict (will be modified with encoded images)
            encode_tasks: List of encoding tasks with frame data
            save_and_encode_func: Function to save and encode FAIL frames
            encode_display_func: Function to encode PASS frames for display
        """
        if self._shutdown:
            logger.warning("BackgroundResultEmitter is shutdown, processing synchronously")
            self._process_and_emit(result_data, encode_tasks, save_and_encode_func, encode_display_func)
            return

        with self._count_lock:
            self._pending_count += 1

        self.executor.submit(
            self._process_and_emit,
            result_data, encode_tasks, save_and_encode_func, encode_display_func
        )

    def _process_and_emit(
        self,
        result_data: Dict[str, Any],
        encode_tasks: List[Dict[str, Any]],
        save_and_encode_func,
        encode_display_func
    ) -> None:
        """
        Process encoding tasks and emit result (runs in background thread).

        Args:
            result_data: Result dict to update and emit
            encode_tasks: List of encoding tasks
            save_and_encode_func: Function for FAIL frames
            encode_display_func: Function for PASS frames
        """
        import time
        t_start = time.perf_counter()

        try:
            # Encode all frames in parallel using nested ThreadPoolExecutor
            if len(encode_tasks) > 1:
                encoded_results = self._encode_parallel(
                    encode_tasks, save_and_encode_func, encode_display_func
                )
            else:
                encoded_results = self._encode_sequential(
                    encode_tasks, save_and_encode_func, encode_display_func
                )

            # Update result_data with encoded images
            for task, (image_path, image_base64) in zip(encode_tasks, encoded_results):
                camera_idx = task['camera_idx']
                frame_idx = task['frame_idx']

                if camera_idx < len(result_data.get('camera_results', [])):
                    camera_result = result_data['camera_results'][camera_idx]
                    if 'frames' in camera_result and frame_idx < len(camera_result['frames']):
                        camera_result['frames'][frame_idx]['image_path'] = image_path
                        camera_result['frames'][frame_idx]['image_base64'] = image_base64

            t_encode = (time.perf_counter() - t_start) * 1000

            # Emit result
            if self._emit_callback:
                if self._event_loop:
                    self._event_loop.call_soon_threadsafe(
                        lambda: self._event_loop.create_task(
                            self._emit_callback("inference_result", result_data)
                        )
                    )
                else:
                    self._emit_callback("inference_result", result_data)

            t_total = (time.perf_counter() - t_start) * 1000
            logger.info(
                f"BackgroundResultEmitter: encoded {len(encode_tasks)} frames in {t_encode:.1f}ms, "
                f"total={t_total:.1f}ms (pending: {self._pending_count - 1})"
            )

        except Exception as e:
            logger.error(f"BackgroundResultEmitter error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            with self._count_lock:
                self._pending_count -= 1

    def _encode_sequential(
        self,
        encode_tasks: List[Dict[str, Any]],
        save_and_encode_func,
        encode_display_func
    ) -> List[tuple]:
        """Encode frames sequentially"""
        results = []
        for task in encode_tasks:
            result = self._encode_single_frame(task, save_and_encode_func, encode_display_func)
            results.append(result)
        return results

    def _encode_parallel(
        self,
        encode_tasks: List[Dict[str, Any]],
        save_and_encode_func,
        encode_display_func
    ) -> List[tuple]:
        """Encode frames in parallel"""
        results = [None] * len(encode_tasks)

        with ThreadPoolExecutor(max_workers=len(encode_tasks)) as executor:
            futures = {
                executor.submit(
                    self._encode_single_frame, task, save_and_encode_func, encode_display_func
                ): idx
                for idx, task in enumerate(encode_tasks)
            }

            for future in futures:
                try:
                    idx = futures[future]
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Encode error: {e}")
                    results[futures[future]] = (None, None)

        return results

    def _encode_single_frame(
        self,
        task: Dict[str, Any],
        save_and_encode_func,
        encode_display_func
    ) -> tuple:
        """Encode a single frame"""
        pass_fail = task.get('pass_fail', 'PASS')

        if pass_fail in ['FAIL', 'ERROR']:
            return save_and_encode_func(
                frame_img=task['frame_img'],
                serial_number=task['serial_number'],
                recipe_id=task['recipe_id'],
                pass_fail=pass_fail,
                frame_idx=task['frame_idx'],
                transformed_bboxes=task.get('transformed_bboxes'),
                confidence=task.get('confidence', 0.0),
                inliers=task.get('inliers', 0),
                total_matches=task.get('total_matches', 0),
                crop_area=task.get('crop_area'),
                product_verification=task.get('product_verification')
            )
        else:
            image_base64 = encode_display_func(
                frame_img=task['frame_img'],
                transformed_bboxes=task.get('transformed_bboxes'),
                confidence=task.get('confidence', 0.0),
                inliers=task.get('inliers', 0),
                total_matches=task.get('total_matches', 0),
                crop_area=task.get('crop_area'),
                product_verification=task.get('product_verification')
            )
            return (None, image_base64)

    def get_pending_count(self) -> int:
        """Get number of pending emit operations"""
        with self._count_lock:
            return self._pending_count

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the emitter"""
        if self._shutdown:
            return

        self._shutdown = True
        pending = self.get_pending_count()

        if pending > 0:
            logger.info(f"BackgroundResultEmitter shutting down, waiting for {pending} pending emits...")

        self.executor.shutdown(wait=wait)
        logger.info("BackgroundResultEmitter shutdown complete")


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

            # Draw center points (template center and product center)
            center_alignment_check = product_verification.get('center_alignment_check')
            if center_alignment_check:
                img_to_save = draw_center_points(
                    img_to_save,
                    center_alignment_check
                )
                logger.info(f"Drew center points (template + product) on frame")

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

        # Relative path for DB and API (from uploads/)
        relative_path = f"inference_results/{recipe_id}/{today}/{serial_number}/{filename_viz}"

        # Create RESIZED + COMPRESSED version for realtime display (divide by 3)
        # This is SYNCHRONOUS because we need base64 for immediate response
        display_img = resize_for_display(img_to_save, scale_factor=3)
        image_base64 = encode_image_to_base64(display_img, quality=70)

        # Save FULL resolution to permanent storage ASYNCHRONOUSLY
        # This runs in background thread, doesn't block the pipeline
        async_saver = AsyncImageSaver.get_instance()
        async_saver.save_images_async(
            img_viz=img_to_save,
            img_org=frame_img,
            path_viz=full_path_viz,
            path_org=full_path_org,
            quality=95
        )

        logger.info(
            f"Queued {pass_fail} frame for async save: {relative_path} "
            f"(full: {w}x{h}, display: {display_img.shape[1]}x{display_img.shape[0]})"
        )

        return relative_path, image_base64

    except Exception as e:
        logger.error(f"Error saving/encoding frame: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ============= Reject Action Logging Utilities =============

# Dedicated logger for reject actions
_reject_logger = None
_reject_log_file = None

def _init_reject_logger():
    """Initialize dedicated reject action logger"""
    global _reject_logger, _reject_log_file

    if _reject_logger is not None:
        return _reject_logger

    # Create logs directory if not exists
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log file path with date
    today = datetime.now().strftime("%Y-%m-%d")
    _reject_log_file = log_dir / f"reject_actions_{today}.log"

    # Create logger
    _reject_logger = logging.getLogger("reject_actions")
    _reject_logger.setLevel(logging.INFO)
    _reject_logger.propagate = False  # Don't propagate to root logger

    # Remove existing handlers
    _reject_logger.handlers.clear()

    # Create file handler
    file_handler = logging.FileHandler(_reject_log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Create formatter with precise timing
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    _reject_logger.addHandler(file_handler)

    logger.info(f"Reject action logger initialized: {_reject_log_file}")

    return _reject_logger


def log_reject_start(reject_count: int, group_id: int, do_number: int,
                     T_capture_complete: float, T_reject: float,
                     delay_reject_ms: int, inference_time_ms: float,
                     alarm_number: int = -1):
    """
    Log reject action START with timing and counter

    Args:
        reject_count: Sequential reject counter (1, 2, 3, ...)
        group_id: Capture group ID
        do_number: Digital output pin number for reject
        T_capture_complete: Timestamp when capture completed (seconds)
        T_reject: Scheduled reject timestamp (seconds)
        delay_reject_ms: Configured delay from capture to reject (milliseconds)
        inference_time_ms: Inference duration (milliseconds)
        alarm_number: Digital output pin number for alarm (-1 = no alarm)
    """
    reject_log = _init_reject_logger()

    import time
    T_now = time.time()
    time_until_reject = (T_reject - T_now) * 1000  # ms

    alarm_info = f" + Alarm(DO{alarm_number})" if alarm_number >= 0 else ""
    reject_log.info(
        f"REJECT_START | #{reject_count:04d} | Group #{group_id} | Reject(DO{do_number}){alarm_info} | "
        f"Scheduled: T={T_reject:.3f}s (in {time_until_reject:.1f}ms) | "
        f"Delay: {delay_reject_ms}ms | Inference: {inference_time_ms:.1f}ms"
    )


def log_reject_end(reject_count: int, group_id: int, do_number: int,
                   pulse_duration_ms: float, actual_duration_ms: float,
                   alarm_number: int = -1):
    """
    Log reject action END with actual duration

    Args:
        reject_count: Sequential reject counter (same as start)
        group_id: Capture group ID
        do_number: Digital output pin number for reject
        pulse_duration_ms: Configured pulse duration (milliseconds)
        actual_duration_ms: Actual measured pulse duration (milliseconds)
        alarm_number: Digital output pin number for alarm (-1 = no alarm)
    """
    reject_log = _init_reject_logger()

    duration_diff = actual_duration_ms - pulse_duration_ms

    alarm_info = f" + Alarm(DO{alarm_number})" if alarm_number >= 0 else ""
    reject_log.info(
        f"REJECT_END   | #{reject_count:04d} | Group #{group_id} | Reject(DO{do_number}){alarm_info} | "
        f"Pulse: {pulse_duration_ms:.1f}ms | Actual: {actual_duration_ms:.1f}ms | "
        f"Diff: {duration_diff:+.1f}ms"
    )


def log_reject_cancelled(reject_count: int, group_id: int, reason: str = "PASS"):
    """
    Log reject action CANCELLED (when product passes)

    Args:
        reject_count: Sequential reject counter
        group_id: Capture group ID
        reason: Cancellation reason (default: "PASS")
    """
    reject_log = _init_reject_logger()

    reject_log.info(
        f"REJECT_CANCEL | #{reject_count:04d} | Group #{group_id} | "
        f"Reason: {reason}"
    )
