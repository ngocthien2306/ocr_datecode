"""
Demo: Template có TOÀN BỘ là polygons (kể cả các vùng không phải template)
"""
import json
import cv2
from template_matcher import TemplateMatcher, BoundingBox
from PyQt5.QtCore import QRect


def create_full_polygon_annotations():
    """Tạo annotations với ALL POLYGONS"""
    template_image_path = '../images/3.jpg'

    # Template bbox - vẫn cần rectangle để extract region cho matching
    template_bbox = BoundingBox(
        rect=QRect(1209, 1016, 82, 95),
        bbox_type='template',
        shape='rectangle'
    )

    # TOÀN BỘ các bbox còn lại đều là POLYGON
    datecode_polygon = BoundingBox(
        rect=None,
        bbox_type='datecode',
        shape='polygon',
        points=[
            [1179, 1270],  # Top-left
            [1510, 1272],  # Top-right (slightly rotated)
            [1508, 1330],  # Bottom-right
            [1177, 1328]   # Bottom-left
        ]
    )

    barcode_polygon = BoundingBox(
        rect=None,
        bbox_type='barcode',
        shape='polygon',
        points=[
            [1010, 1108],
            [1566, 1110],
            [1564, 1245],
            [1008, 1243]
        ]
    )

    text1_polygon = BoundingBox(
        rect=None,
        bbox_type='text',
        shape='polygon',
        points=[
            [1039, 1323],
            [1435, 1324],
            [1434, 1389],
            [1038, 1388]
        ]
    )

    text2_polygon = BoundingBox(
        rect=None,
        bbox_type='text',
        shape='polygon',
        points=[
            [1082, 828],
            [1534, 830],
            [1532, 933],
            [1080, 931]
        ]
    )

    template_bboxes = [
        template_bbox,
        datecode_polygon,
        barcode_polygon,
        text1_polygon,
        text2_polygon
    ]

    return template_image_path, template_bboxes


def save_as_json(template_image_path, template_bboxes):
    """Lưu ra file JSON để xem format"""
    data = {
        '_template_image': template_image_path,
        template_image_path: [bbox.to_dict() for bbox in template_bboxes]
    }

    output_path = 'polygon_template_example.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"📄 Saved template to: {output_path}\n")
    return data


def main():
    print("="*70)
    print("DEMO: Full Polygon Template (All non-template bboxes are polygons)")
    print("="*70)

    # Create polygon template
    template_image_path, template_bboxes = create_full_polygon_annotations()

    print(f"\n📋 Template Configuration:")
    print(f"   Image: {template_image_path}")
    print(f"   Total bboxes: {len(template_bboxes)}\n")

    for i, bbox in enumerate(template_bboxes):
        print(f"   [{i+1}] {bbox.bbox_type:10s} → {bbox.shape:10s}", end="")
        if bbox.shape == 'polygon':
            print(f" (4 points)")
        else:
            print(f" (for matching)")

    # Save to JSON for inspection
    json_data = save_as_json(template_image_path, template_bboxes)

    print("JSON Preview:")
    print("-" * 70)
    print(json.dumps(json_data, indent=2)[:500] + "...")
    print("-" * 70)

    # Create matcher
    matcher = TemplateMatcher(template_image_path, template_bboxes)

    # Test với ảnh target
    target_image_path = '../images/2.jpg'
    print(f"\n🎯 Matching with target: {target_image_path}\n")

    bboxes, confidence, target_image = matcher.match(
        target_image_path,
        method='feature',
        threshold=0.3,
        debug=True
    )

    if bboxes is None:
        print("\n❌ Template not found")
        return

    print(f"\n" + "="*70)
    print(f"✅ MATCH SUCCESS! Confidence: {confidence:.3f}")
    print("="*70)

    print(f"\n📦 Output bboxes (after transformation):\n")

    for i, bbox in enumerate(bboxes):
        print(f"   [{i+1}] {bbox.bbox_type:10s} → {bbox.shape:10s}", end="")

        if bbox.shape == 'polygon':
            print(f" 🔷")
            print(f"       Points: {bbox.points[:2]}... (4 total)")
        else:
            print(f" ▭")
            print(f"       Rect: {bbox.rect}")

    # Count shapes
    polygon_count = sum(1 for b in bboxes if b.shape == 'polygon')
    rect_count = sum(1 for b in bboxes if b.shape == 'rectangle')

    print(f"\n📊 Summary:")
    print(f"   Input:  4 polygons + 1 rectangle")
    print(f"   Output: {polygon_count} polygons + {rect_count} rectangles")

    if rect_count > 1:  # More than just template
        print(f"   → Image is STRAIGHT (polygons auto-converted to rectangles)")
    else:
        print(f"   → Image is TILTED (polygons preserved)")

    # Visualize
    result_image = matcher.draw_bboxes(target_image, bboxes, draw_polygon=True)
    output_path = 'results/full_polygon_template.jpg'
    cv2.imwrite(output_path, result_image)
    print(f"\n💾 Saved visualization: {output_path}")

    # Save crops
    print(f"\n📁 Saving perspective-corrected crops:")
    for i, bbox in enumerate(bboxes):
        if bbox.bbox_type != 'template':
            cropped = matcher.crop_region_with_perspective(target_image, bbox)
            crop_path = f'results/crop_{bbox.bbox_type}_{i}.jpg'
            cv2.imwrite(crop_path, cropped)
            print(f"   - {crop_path}")


if __name__ == '__main__':
    main()
