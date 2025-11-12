import json
import cv2
from template_matcher import TemplateMatcher, BoundingBox


def load_annotations(annotations_file):
    with open(annotations_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    template_image_path = data.get('_template_image')
    if not template_image_path:
        raise ValueError("No template image found in annotations.json")

    template_bboxes_data = data.get(template_image_path, [])
    template_bboxes = [BoundingBox.from_dict(b) for b in template_bboxes_data]

    return template_image_path, template_bboxes


def main():
    annotations_file = '../images/annotations.json'
    target_image_path = '../images/7.jpg'

    template_image_path, template_bboxes = load_annotations(annotations_file)

    matcher = TemplateMatcher(template_image_path, template_bboxes)

    print("\n=== Testing different matching methods ===\n")

    methods = ['simple', 'multi_scale', 'feature', 'auto']

    for method in methods:
        print(f"Method: {method}")
        bboxes, confidence, target_image = matcher.match(
            target_image_path,
            method=method,
            threshold=0.3
        )

        if bboxes is None:
            print(f"  ❌ Not found. Confidence: {confidence:.3f}\n")
        else:
            print(f"  ✓ Found! Confidence: {confidence:.3f}")
            print(f"  Found {len(bboxes)} bboxes\n")

            result_image = matcher.draw_bboxes(target_image, bboxes)
            cv2.imwrite(f'result_{method}.jpg', result_image)

    heatmap_vis = matcher.visualize_matching(target_image_path)
    cv2.imwrite('heatmap_visualization.jpg', heatmap_vis)

    cv2.imshow('Template Region', matcher.template_region)
    cv2.imshow('Heatmap Visualization', heatmap_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
