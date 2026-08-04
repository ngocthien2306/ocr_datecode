"""
Overlay drawing for saved/streamed inspection frames — template bboxes,
detected OBB boxes, centre points and the colour-match overlay.

Pure image-in/image-out: no I/O, no model access, nothing but OpenCV drawing,
so these stay safe to call from any thread.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

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
        'label': (128, 0, 255),        # Purple (for label region)
        'edge_left': (60, 145, 250),   # Orange  (edge search region, left)
        'edge_right': (200, 190, 45),  # Teal    (edge search region, right)
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
        # `dict.get` returns the value if the key exists even when that value is
        # None (the default is only used for MISSING keys). Pydantic rectangle
        # annotations leave `points: None` after deserialization, so coerce to [].
        points = bbox.get('points') or []
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

        # Draw type-tag for 'label' polygon (same style as the previous YOLO label box)
        if bbox_type == 'label' and len(points) > 0:
            tag_text = "LABEL"
            (tw, th), tbase = cv2.getTextSize(
                tag_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, text_thickness
            )
            tag_x = int(points[0][0])
            tag_y = int(points[0][1]) - 6
            if tag_y - th - 4 < 0:
                tag_y = int(points[0][1]) + th + 8
            if tag_x + tw + 8 > width:
                tag_x = width - tw - 8
            if tag_x < 0:
                tag_x = 0
            # Background filled rectangle
            cv2.rectangle(
                result_img,
                (tag_x, tag_y - th - 4),
                (tag_x + tw + 8, tag_y + tbase),
                color,
                -1
            )
            # White text on colored background
            cv2.putText(
                result_img,
                tag_text,
                (tag_x + 4, tag_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.8,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA
            )

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
    # NOTE: 'label' temporarily hidden — using template polygon (transformed via SuperPoint)
    # as the reference; no need to visualize the YOLO label box
    for box_type in ['product', 'wrinkled']:
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

            # Get color
            color = colors.get(box_type, (255, 255, 255))

            # Extract corners once — used by OBB drawing AND label anchor below
            corners = single_box.get('corners')
            if corners is not None:
                if isinstance(corners, list):
                    corners = np.array(corners, dtype=np.int32)
                else:
                    corners = corners.astype(np.int32)

            # Wrinkled box: ưu tiên vẽ contour mask (instance seg), fallback corners (OBB)
            contour = single_box.get('contour')
            if box_type == 'wrinkled' and contour is not None:
                contour_arr = np.array(contour, dtype=np.int32)
                # Semi-transparent filled mask
                overlay = result_img.copy()
                cv2.fillPoly(overlay, [contour_arr], color)
                cv2.addWeighted(overlay, 0.35, result_img, 0.65, 0, result_img)
                # Outline
                cv2.polylines(result_img, [contour_arr], True, color, line_thickness + 1, cv2.LINE_AA)
                # Derive corners from contour bounding rect when not provided — label anchor needs it
                if corners is None:
                    x_b, y_b, bw_b, bh_b = cv2.boundingRect(contour_arr)
                    corners = np.array(
                        [[x_b, y_b], [x_b + bw_b, y_b], [x_b + bw_b, y_b + bh_b], [x_b, y_b + bh_b]],
                        dtype=np.int32,
                    )
            else:
                # Fallback: vẽ OBB corners (product, label, hoặc wrinkled cũ)
                if corners is None:
                    continue

                for i in range(4):
                    pt1 = tuple(corners[i])
                    pt2 = tuple(corners[(i + 1) % 4])
                    cv2.line(result_img, pt1, pt2, color, line_thickness + 1, cv2.LINE_AA)

                for corner in corners:
                    cv2.circle(result_img, tuple(corner), max(3, line_thickness), color, -1)

            # Prepare label text
            score = single_box.get('score', 0.0)
            if box_type == 'wrinkled':
                # Bỏ chữ "WRINKLED", chỉ hiển thị #1, #2,... (kèm score nếu show_details)
                label_parts = [f"#{idx+1}"]
            else:
                label_parts = [f"{box_type.upper()}"]
                if isinstance(box_data, list) and len(boxes_to_draw) > 1:
                    label_parts[0] = f"{box_type.upper()}#{idx+1}"

            if show_details:
                label_parts.append(f"{score:.2f}")

            label_text = " ".join(label_parts)

            # Calculate label position (top-left corner of box)
            if corners is None:
                continue
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


def draw_color_match_overlay(
    img: np.ndarray,
    color_check: Dict[str, Any],
    detected_boxes: Optional[Dict[str, Any]],
) -> np.ndarray:
    """
    Paint a semi-transparent yellow overlay on pixels matching the color_check
    HSV range, restricted to the detected bottle bbox. Recomputes the HSV mask
    from the frame (the mask itself isn't persisted in product_verification).

    Args:
        img: Input frame (BGR)
        color_check: {h_range, s_range, v_range, ...} from product_verification.color_check
        detected_boxes: {product: {corners, ...}} — the bottle bbox to restrict overlay to

    Returns:
        Image with yellow overlay drawn on matching pixels.
    """
    if not color_check or not detected_boxes:
        return img
    product_box = (detected_boxes or {}).get('product')
    if not product_box or not isinstance(product_box, dict):
        return img
    corners = product_box.get('corners')
    if corners is None:
        return img
    try:
        corners_arr = np.array(corners, dtype=np.int32).reshape(-1, 2)
        H, W = img.shape[:2]
        x1 = max(0, int(corners_arr[:, 0].min()))
        y1 = max(0, int(corners_arr[:, 1].min()))
        x2 = min(W, int(corners_arr[:, 0].max()))
        y2 = min(H, int(corners_arr[:, 1].max()))
        if x2 <= x1 or y2 <= y1:
            return img

        h_lo, h_hi = color_check.get('h_range', [0, 180])
        s_lo, s_hi = color_check.get('s_range', [0, 255])
        v_lo, v_hi = color_check.get('v_range', [0, 255])

        roi = img[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (int(h_lo), int(s_lo), int(v_lo)),
                                (int(h_hi), int(s_hi), int(v_hi)))
        if not np.any(mask):
            return img

        # Build a yellow overlay only where mask is set, then alpha-blend.
        overlay = roi.copy()
        overlay[mask > 0] = (0, 255, 255)  # BGR yellow
        alpha = 0.45
        blended = cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0)
        # Only replace the masked pixels in the result — leave unmatched bottle
        # area as the original image.
        result = img.copy()
        roi_out = result[y1:y2, x1:x2]
        mask_bool = mask > 0
        roi_out[mask_bool] = blended[mask_bool]
        result[y1:y2, x1:x2] = roi_out
        return result
    except Exception as e:
        logger.warning(f"draw_color_match_overlay failed: {e}", exc_info=True)
        return img
