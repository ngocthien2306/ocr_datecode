"""
Test script để so sánh SIFT vs SuperPoint+LightGlue
"""
import json
import cv2
import time
from template_matcher import TemplateMatcher, BoundingBox


def load_annotations(annotations_file):
    with open(annotations_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    template_image_path = data.get('_template_image')
    template_bboxes_data = data.get(template_image_path, [])
    template_bboxes = [BoundingBox.from_dict(b) for b in template_bboxes_data]
    return template_image_path, template_bboxes


def test_method(matcher, target_image_path, method_name, threshold=0.3):
    """Test một method và đo thời gian"""
    print(f"\n{'='*60}")
    print(f"Testing: {method_name}")
    print(f"{'='*60}")

    start_time = time.time()
    bboxes, confidence, target_image = matcher.match(
        target_image_path,
        method=method_name,
        threshold=threshold
    )
    elapsed = time.time() - start_time

    if bboxes is None:
        print(f"❌ Not found. Confidence: {confidence:.3f}")
        print(f"⏱️  Time: {elapsed:.2f}s")
        return None

    print(f"✓ Found! Confidence: {confidence:.3f}")
    print(f"📦 Found {len(bboxes)} bboxes")
    print(f"⏱️  Time: {elapsed:.2f}s")

    # Count polygon vs rectangle
    polygon_count = sum(1 for b in bboxes if b.shape == 'polygon')
    rect_count = sum(1 for b in bboxes if b.shape == 'rectangle')
    print(f"   - {polygon_count} polygons, {rect_count} rectangles")

    # Save result
    result_image = matcher.draw_bboxes(target_image, bboxes, draw_polygon=True)
    output_path = f'results/result_{method_name}.jpg'
    cv2.imwrite(output_path, result_image)
    print(f"💾 Saved: {output_path}")

    return {'method': method_name, 'confidence': confidence, 'time': elapsed, 'bboxes': len(bboxes)}


def main():
    annotations_file = '../images/annotations.json'
    target_image_path = '../images/9.jpg'

    print("Loading template and annotations...")
    template_image_path, template_bboxes = load_annotations(annotations_file)
    matcher = TemplateMatcher(template_image_path, template_bboxes)

    print(f"Template: {template_image_path}")
    print(f"Template bboxes: {len(template_bboxes)}")
    print(f"Target: {target_image_path}")

    # Test các methods
    methods = ['feature', 'superpoint', 'auto']
    results = []

    for method in methods:
        result = test_method(matcher, target_image_path, method)
        if result:
            results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<15} {'Confidence':<12} {'Time (s)':<10} {'BBoxes'}")
    print(f"{'-'*60}")

    for r in results:
        print(f"{r['method']:<15} {r['confidence']:<12.3f} {r['time']:<10.2f} {r['bboxes']}")

    # Find best method
    if results:
        best = max(results, key=lambda x: x['confidence'])
        fastest = min(results, key=lambda x: x['time'])

        print(f"\n🏆 Best confidence: {best['method']} ({best['confidence']:.3f})")
        print(f"⚡ Fastest: {fastest['method']} ({fastest['time']:.2f}s)")


if __name__ == '__main__':
    main()
