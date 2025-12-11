#!/usr/bin/env python3
"""
Test crop_area functionality
Kiểm tra:
1. Matching với crop_area
2. Matching không có crop_area
3. Visualize để so sánh coordinates
"""

import cv2
import json
import numpy as np
from template_matcher import TemplateMatcher, BoundingBox

def load_annotations(json_path):
    """Load annotations from JSON"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    template_image_path = data['_template_image']

    # Load template bboxes
    template_bboxes = []
    if template_image_path in data:
        for bbox_data in data[template_image_path]:
            bbox = BoundingBox.from_dict(bbox_data)
            template_bboxes.append(bbox)

    return template_image_path, template_bboxes

def visualize_bboxes(image, bboxes, title="Result", save_path=None):
    """Visualize bboxes on image"""
    result = image.copy()

    colors = {
        'template': (255, 80, 80),
        'text': (80, 255, 120),
        'datecode': (80, 150, 255),
        'barcode': (255, 220, 80),
        'crop_area': (255, 100, 255)
    }

    for bbox in bboxes:
        color = colors.get(bbox.bbox_type, (255, 255, 255))

        if bbox.shape == 'polygon' and bbox.points is not None:
            points = np.array(bbox.points, dtype=np.int32)
            cv2.polylines(result, [points], True, color, 2)
        else:
            x, y, w, h = bbox.rect.x(), bbox.rect.y(), bbox.rect.width(), bbox.rect.height()
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

        # Draw label
        label = f"{bbox.bbox_type}"
        if bbox.shape == 'polygon':
            label += " (P)"
        label_x = int(bbox.rect.x())
        label_y = int(bbox.rect.y() - 5)
        cv2.putText(result, label, (label_x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if save_path:
        cv2.imwrite(save_path, result)
        print(f"Saved: {save_path}")

    # Resize for display
    h, w = result.shape[:2]
    if h > 1200:
        scale = 1200 / h
        result = cv2.resize(result, None, fx=scale, fy=scale)

    cv2.imshow(title, result)

def test_with_crop_area():
    """Test matching with crop_area"""
    print("\n" + "="*60)
    print("TEST 1: Matching WITH crop_area")
    print("="*60)

    json_path = '/Users/ngocthien.ai/Source/Projects/ocr_datecode/images/annotations.json'
    template_image_path, template_bboxes = load_annotations(json_path)

    print(f"\nTemplate: {template_image_path}")
    print(f"Bboxes: {len(template_bboxes)}")
    for bbox in template_bboxes:
        print(f"  - {bbox.bbox_type}: {bbox.shape} @ ({bbox.rect.x()}, {bbox.rect.y()}, {bbox.rect.width()}, {bbox.rect.height()})")

    # Create matcher
    matcher = TemplateMatcher(template_image_path, template_bboxes)

    print(f"\nMatcher info:")
    print(f"  crop_area_bbox: {matcher.crop_area_bbox}")
    print(f"  crop_offset_x: {matcher.crop_offset_x}")
    print(f"  crop_offset_y: {matcher.crop_offset_y}")

    # Match same image
    target_image_path = template_image_path
    bboxes, confidence, result_image = matcher.match(target_image_path, method='feature', debug=True)

    print(f"\n✅ Match confidence: {confidence:.3f}")
    print(f"✅ Found {len(bboxes)} bboxes")

    for bbox in bboxes:
        print(f"  - {bbox.bbox_type}: {bbox.shape} @ ({bbox.rect.x()}, {bbox.rect.y()}, {bbox.rect.width()}, {bbox.rect.height()})")

    # Visualize
    visualize_bboxes(result_image, bboxes, "With crop_area", "results/test_with_crop.jpg")

def test_without_crop_area():
    """Test matching without crop_area"""
    print("\n" + "="*60)
    print("TEST 2: Matching WITHOUT crop_area")
    print("="*60)

    json_path = '/Users/ngocthien.ai/Source/Projects/ocr_datecode/images/annotations.json'
    template_image_path, all_bboxes = load_annotations(json_path)

    # Remove crop_area
    template_bboxes = [bbox for bbox in all_bboxes if bbox.bbox_type != 'crop_area']

    print(f"\nTemplate: {template_image_path}")
    print(f"Bboxes: {len(template_bboxes)} (removed crop_area)")
    for bbox in template_bboxes:
        print(f"  - {bbox.bbox_type}: {bbox.shape} @ ({bbox.rect.x()}, {bbox.rect.y()}, {bbox.rect.width()}, {bbox.rect.height()})")

    # Create matcher
    matcher = TemplateMatcher(template_image_path, template_bboxes)

    print(f"\nMatcher info:")
    print(f"  crop_area_bbox: {matcher.crop_area_bbox}")
    print(f"  crop_offset_x: {matcher.crop_offset_x}")
    print(f"  crop_offset_y: {matcher.crop_offset_y}")

    # Match same image
    target_image_path = template_image_path
    bboxes, confidence, result_image = matcher.match(target_image_path, method='feature', debug=True)

    print(f"\n✅ Match confidence: {confidence:.3f}")
    print(f"✅ Found {len(bboxes)} bboxes")

    for bbox in bboxes:
        print(f"  - {bbox.bbox_type}: {bbox.shape} @ ({bbox.rect.x()}, {bbox.rect.y()}, {bbox.rect.width()}, {bbox.rect.height()})")

    # Visualize
    visualize_bboxes(result_image, bboxes, "Without crop_area", "results/test_without_crop.jpg")

def test_visualize_template():
    """Visualize template bboxes"""
    print("\n" + "="*60)
    print("TEST 3: Visualize Template Bboxes")
    print("="*60)

    json_path = '/Users/ngocthien.ai/Source/Projects/ocr_datecode/images/annotations.json'
    template_image_path, template_bboxes = load_annotations(json_path)

    template_image = cv2.imread(template_image_path)

    visualize_bboxes(template_image, template_bboxes, "Template Bboxes", "results/test_template_bboxes.jpg")

if __name__ == '__main__':
    # Test 1: Visualize template
    test_visualize_template()

    # Test 2: With crop_area
    test_with_crop_area()

    # Test 3: Without crop_area
    test_without_crop_area()

    print("\n" + "="*60)
    print("Press any key to close...")
    print("="*60)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
