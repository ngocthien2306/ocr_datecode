"""
OCR Utility Functions
Common helpers for text recognition and preprocessing
"""
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class HostDeviceMem:
    """Simple helper data class for storing host and device memory pointers"""

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def preprocess_text_image(
    image: np.ndarray,
    target_height: int = 48,
    min_width: int = 320,
    max_width: int = 2000
) -> np.ndarray:
    """
    Preprocess image for text recognition model

    Args:
        image: Input image (BGR or grayscale)
        target_height: Target height (default: 48)
        min_width: Minimum width to pad to
        max_width: Maximum width to resize to

    Returns:
        Preprocessed tensor [1, 3, H, W]
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Resize to target height while maintaining aspect ratio
    h, w = gray.shape
    ratio = target_height / h
    new_w = int(w * ratio)

    # Ensure minimum width of at least 10px
    new_w = max(new_w, 10)

    # Resize maintaining aspect ratio
    resized = cv2.resize(gray, (new_w, target_height))

    # Only pad if width is less than min_width (for TensorRT engine constraints)
    if new_w < min_width:
        pad_width = min_width - new_w
        resized = cv2.copyMakeBorder(
            resized,
            0, 0, 0, pad_width,  # top, bottom, left, right
            cv2.BORDER_CONSTANT,
            value=0  # Pad with black
        )
        new_w = min_width

    # Resize if width exceeds max_width
    if new_w > max_width:
        resized = cv2.resize(resized, (max_width, target_height))
        new_w = max_width

    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    # Subtract mean and divide by std (standard normalization)
    mean = np.array([0.5], dtype=np.float32)
    std = np.array([0.5], dtype=np.float32)
    normalized = (normalized - mean) / std

    # Convert to 3 channels (RGB format)
    tensor = np.stack([normalized, normalized, normalized], axis=0).astype(np.float32)

    # Add batch dimension: [1, 3, H, W]
    tensor = np.expand_dims(tensor, axis=0).astype(np.float32)

    return tensor


def decode_ctc_greedy(preds: np.ndarray, char_list: list) -> str:
    """
    Decode CTC output to text using greedy (best path) decoding

    Args:
        preds: Model output [batch_size, sequence_length, num_classes]
        char_list: Character list including 'blank' at index 0

    Returns:
        Decoded text string
    """
    # Get best path (argmax)
    indices = np.argmax(preds, axis=2)[0]  # Take first batch

    # CTC decoding: remove duplicates and blanks
    chars = []
    prev_idx = -1

    for idx in indices:
        # Skip blank (index 0) and consecutive duplicates
        if idx != 0 and idx != prev_idx:
            if idx < len(char_list):
                chars.append(char_list[idx])
        prev_idx = idx

    return ''.join(chars)


def calculate_confidence(preds: np.ndarray) -> float:
    """
    Calculate average confidence from model predictions

    Args:
        preds: Model output [sequence_length, num_classes]

    Returns:
        Average confidence score (0-1)
    """
    # Compute softmax
    exp_x = np.exp(preds - np.max(preds, axis=-1, keepdims=True))
    softmax_preds = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    # Get max probability for each timestep
    max_probs = np.max(softmax_preds, axis=1)

    # Return average confidence
    return float(np.mean(max_probs))


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as [top-left, top-right, bottom-right, bottom-left].
    Works regardless of the original drawing order.
    """
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]   # TL: smallest x+y
    ordered[2] = pts[np.argmax(s)]   # BR: largest x+y
    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]  # TR: smallest x-y
    ordered[3] = pts[np.argmax(diff)]  # BL: largest x-y
    return ordered


def crop_text_region(frame_img: np.ndarray, points: list) -> np.ndarray:
    """
    Crop text region from frame using perspective transform

    Args:
        frame_img: Input frame (BGR)
        points: List of 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    Returns:
        Cropped and warped region (numpy array)
    """
    pts = np.array(points, dtype=np.float32)

    # Reorder points to TL, TR, BR, BL regardless of original drawing order
    pts = _order_points(pts)
    tl, tr, br, bl = pts

    # Compute destination size from actual edge lengths (not AABB)
    width_top = np.linalg.norm(tr - tl)
    width_bot = np.linalg.norm(br - bl)
    width = int(max(width_top, width_bot))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    height = int(max(height_left, height_right))

    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts, dst_pts)
    warped = cv2.warpPerspective(frame_img, M, (width, height))

    return warped


def compare_texts(text1: str, text2: str, case_sensitive: bool = False, strip: bool = True, space: bool = True) -> bool:
    """
    Compare two text strings with flexible matching

    Args:
        text1: First text string
        text2: Second text string
        case_sensitive: Whether to do case-sensitive comparison
        strip: Whether to strip whitespace
        space: Whether to remove spaces

    Returns:
        True if texts match, False otherwise
    """
    import re

    if strip:
        text1 = text1.strip()
        text2 = text2.strip()

    # Normalize special characters to space before removing spaces
    # OCR often confuses space with underscore, dash, punctuation, etc.
    special_chars_to_space = ['_', '-', '－', '—', '–', ',', '.', ':', ';', '--']  # underscore, hyphen, dashes, punctuation
    for char in special_chars_to_space:
        text1 = text1.replace(char, ' ')
        text2 = text2.replace(char, ' ')

    # Collapse multiple spaces into single space
    text1 = re.sub(r'\s+', ' ', text1)
    text2 = re.sub(r'\s+', ' ', text2)

    # Now remove all spaces
    text1 = text1.replace(" ", "")
    text2 = text2.replace(" ", "")

    # Remove trailing special characters
    text1 = re.sub(r'[^A-Za-z0-9]+$', '', text1)
    text2 = re.sub(r'[^A-Za-z0-9]+$', '', text2)

    if case_sensitive:
        text1 = text1.upper()
        text2 = text2.upper()

    # Compare character by character, treating O/0 as equivalent
    if len(text1) != len(text2):
        return False

    for c1, c2 in zip(text1, text2):
        if c1 == c2:
            continue
        # Treat O and 0 as equivalent
        if {c1, c2} == {'O', '0'}:
            continue
        return False

    return True


def draw_text_on_image(
    image: np.ndarray,
    text: str,
    position: tuple,
    font_scale: float = 0.8,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    background: bool = True
) -> np.ndarray:
    """
    Draw text on image with optional background

    Args:
        image: Input image
        text: Text to draw
        position: (x, y) position
        font_scale: Font scale
        color: Text color (B, G, R)
        thickness: Text thickness
        background: Whether to draw background rectangle

    Returns:
        Image with text drawn
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    if background:
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Draw background rectangle
        x, y = position
        cv2.rectangle(
            image,
            (x, y - text_h - 10),
            (x + text_w + 10, y + baseline),
            (0, 0, 0),
            -1
        )

    # Draw text
    cv2.putText(
        image,
        text,
        position,
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )

    return image


def draw_text_regions(
    image: np.ndarray,
    text_regions: list,
    color: tuple = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw text region bboxes on image

    Args:
        image: Input image
        text_regions: List of text region dicts with 'points' key
        color: Box color (B, G, R)
        thickness: Line thickness

    Returns:
        Image with boxes drawn
    """
    for region in text_regions:
        points = region.get('points', [])
        if len(points) >= 4:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(image, [pts], True, color, thickness)

    return image


def normalize_text_for_comparison(text: str) -> str:
    """
    Normalize text for robust comparison

    - Strip whitespace
    - Remove special characters
    - Convert to uppercase

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    import re

    # Strip whitespace
    text = text.strip()

    # Remove special characters except alphanumeric and space
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)

    # Convert to uppercase
    text = text.upper()

    # Remove extra spaces
    text = ' '.join(text.split())

    return text


