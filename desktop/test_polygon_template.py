"""
Test script để demo polygon template workflow
"""
import json
import cv2
from template_matcher import TemplateMatcher, BoundingBox
from PyQt5.QtCore import QRect


def create_polygon_template_annotations():
    """Tạo template annotations với polygon"""
    template_image_path = '../images/3.jpg'

    # Template bbox (rectangle như cũ)
    template_bbox = BoundingBox(
        rect=QRect(1209, 1016, 82, 95),
        bbox_type='template',
        shape='rectangle'
    )

    # Các bbox khác - một số rectangle, một số polygon
    datecode_polygon = BoundingBox(
        rect=None,
        bbox_type='datecode',
        shape='polygon',
        points=[
            [1179, 1270],
            [1510, 1270],
            [1510, 1330],
            [1179, 1330]
        ]
    )

    barcode_rect = BoundingBox(
        rect=QRect(1010, 1108, 556, 137),
        bbox_type='barcode',
        shape='rectangle'
    )

    text_polygon = BoundingBox(
        rect=None,
        bbox_type='text',
        shape='polygon',
        points=[
            [1039, 1323],
            [1435, 1323],
            [1435, 1389],
            [1039, 1389]
        ]
    )

    template_bboxes = [template_bbox, datecode_polygon, barcode_rect, text_polygon]

    return template_image_path, template_bboxes


def main():
    print("="*60)
    print("TEST: Polygon Template Matching")
    print("="*60)

    template_image_path, template_bboxes = create_polygon_template_annotations()

    print(f"\n📋 Template: {template_image_path}")
    print(f"   BBoxes: {len(template_bboxes)}")
    for bbox in template_bboxes:
        print(f"   - {bbox.bbox_type:10s} → {bbox.shape}")

    matcher = TemplateMatcher(template_image_path, template_bboxes)

    # Test với ảnh target
    target_image_path = '../images/2.jpg'
    print(f"\n🎯 Target: {target_image_path}")

    bboxes, confidence, target_image = matcher.match(
        target_image_path,
        method='feature',
        threshold=0.3,
        debug=True
    )

    if bboxes is None:
        print("❌ Template not found")
        return

    print(f"\n✅ Match found! Confidence: {confidence:.3f}")
    print(f"📦 Result bboxes:")

    for i, bbox in enumerate(bboxes):
        print(f"   [{i+1}] {bbox.bbox_type:10s} → {bbox.shape}")
        if bbox.shape == 'polygon':
            print(f"       Points: {bbox.points}")

    # Save result
    result_image = matcher.draw_bboxes(target_image, bboxes, draw_polygon=True)
    output_path = 'results/polygon_template_result.jpg'
    cv2.imwrite(output_path, result_image)
    print(f"\n💾 Saved: {output_path}")

    print("\n" + "="*60)
    print("EXPLANATION")
    print("="*60)
    print("""
Polygon Template Workflow:

1. Template có mixed shapes:
   - template: rectangle (for matching)
   - datecode: POLYGON (4 points)
   - barcode: rectangle
   - text: POLYGON (4 points)

2. Matching:
   - Vẫn dùng template rectangle để match
   - Tìm homography H

3. Transform:
   - Rectangle bbox → 4 corners → transform → output shape tùy H
   - Polygon bbox → 4 points → transform → output shape tùy H

4. Smart decision:
   - If H near similarity → output rectangle
   - If H has perspective → output polygon

Benefits:
✓ Support vẽ polygon trên template (cho vùng nghiêng)
✓ Polygon được transform chính xác qua homography
✓ Output tự động chọn shape phù hợp
    """)


if __name__ == '__main__':
    main()
