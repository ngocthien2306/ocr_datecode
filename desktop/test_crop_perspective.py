import json
import cv2
from template_matcher import TemplateMatcher, BoundingBox


def load_annotations(annotations_file):
    with open(annotations_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    template_image_path = data.get('_template_image')
    template_bboxes_data = data.get(template_image_path, [])
    template_bboxes = [BoundingBox.from_dict(b) for b in template_bboxes_data]
    return template_image_path, template_bboxes


def main():
    annotations_file = '../images/annotations.json'
    target_image_path = '../images/1.jpg'

    template_image_path, template_bboxes = load_annotations(annotations_file)
    matcher = TemplateMatcher(template_image_path, template_bboxes)

    bboxes, confidence, target_image = matcher.match(
        target_image_path,
        method='feature',
        threshold=0.3,
        debug=True
    )

    if bboxes is None:
        print("Template not found")
        return

    print(f"Confidence: {confidence:.3f}")
    print(f"Found {len(bboxes)} bboxes\n")

    result_with_polygon = matcher.draw_bboxes(target_image, bboxes, draw_polygon=True)
    cv2.imwrite('../results/result_with_polygon.jpg', result_with_polygon)

    for i, bbox in enumerate(bboxes):
        print(f"BBox {i+1}: {bbox.bbox_type}")
        if bbox.polygon:
            print(f"  Polygon: {bbox.polygon}")

        cropped = matcher.crop_region_with_perspective(target_image, bbox)

        cv2.imwrite(f'results/cropped_{bbox.bbox_type}_{i}.jpg', cropped)
        print(f"  Saved: cropped_{bbox.bbox_type}_{i}.jpg\n")

    cv2.imshow('Result with Polygons', result_with_polygon)

    for i, bbox in enumerate(bboxes):
        cropped = matcher.crop_region_with_perspective(target_image, bbox)
        cv2.imshow(f'Cropped: {bbox.bbox_type} #{i}', cropped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
