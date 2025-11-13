import json
import cv2
from template_matcher import TemplateMatcher, BoundingBox
from text_recognizer import TextRecognizer


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

    # Initialize Text Recognizer (chỉ recognition, không detection)
    print("Initializing Text Recognizer...")
    recognizer = TextRecognizer(
        model_path='../languages/english/rec.onnx',
        dict_path='../languages/english/dict.txt',
        use_gpu=False
    )
    print()

    # Template matching
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

    print("\n" + "="*60)
    print("📝 TEXT RECOGNITION")
    print("="*60)

    # Collect all text/datecode regions for batch processing
    text_regions = []
    text_indices = []
    
    for i, bbox in enumerate(bboxes):
        print(f"\n{'─'*60}")
        print(f"BBox {i+1}: {bbox.bbox_type}")
        if bbox.polygon:
            print(f"  Polygon: {bbox.polygon}")

        # Crop với perspective correction
        cropped = matcher.crop_region_with_perspective(target_image, bbox)
        cv2.imwrite(f'results/cropped_{bbox.bbox_type}_{i}.jpg', cropped)
        print(f"  💾 Saved: cropped_{bbox.bbox_type}_{i}.jpg")
        
        # Collect text regions for batch OCR
        if bbox.bbox_type in ["text", "datecode"]:
            text_regions.append(cropped)
            text_indices.append(i)
    
    # Batch OCR processing
    if text_regions:
        print(f"\n{'='*60}")
        print(f"🚀 BATCH OCR PROCESSING ({len(text_regions)} regions)")
        print(f"{'='*60}")
        
        import time
        start = time.time()
        results = recognizer.recognize_batch(text_regions)
        batch_time = (time.time() - start) * 1000
        
        print(f"\n⏱️  Total batch time: {batch_time:.2f}ms")
        print(f"⚡ Average per region: {batch_time/len(text_regions):.2f}ms")
        
        for idx, (text, conf) in zip(text_indices, results):
            print(f"\nBBox {idx+1} ({bboxes[idx].bbox_type}):")
            print(f"  � Text: '{text}'")
            print(f"  🎯 Confidence: {conf:.3f}")

    print("\n" + "="*60)

    cv2.imshow('Result with Polygons', result_with_polygon)

    for i, bbox in enumerate(bboxes):
        cropped = matcher.crop_region_with_perspective(target_image, bbox)
        cv2.imshow(f'Cropped: {bbox.bbox_type} #{i}', cropped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
