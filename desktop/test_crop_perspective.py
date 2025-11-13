import json
import cv2
import time
from template_matcher import TemplateMatcher, BoundingBox
from text_recognizer import TextRecognizer


# ============================================================
# CONFIGURATION
# ============================================================
SHOW_TIMING = True   # Set to False to hide timing details
SHOW_DEBUG = False   # Set to True to show homography analysis


def load_annotations(annotations_file):
    with open(annotations_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    template_image_path = data.get('_template_image')
    template_bboxes_data = data.get(template_image_path, [])
    template_bboxes = [BoundingBox.from_dict(b) for b in template_bboxes_data]
    return template_image_path, template_bboxes


def main():
    import time
    annotations_file = '../images/annotations.json'
    target_image_path = '../images/2.jpg'

    # Timing dictionary
    timing_log = {}
    overall_start = time.time()

    # ========== STEP 1: Initialize Recognizer ==========
    init_start = time.time()
    print("Initializing Text Recognizer...")
    recognizer = TextRecognizer(
        model_path='../languages/english/rec.onnx',
        dict_path='../languages/english/dict.txt',
        use_gpu=False
    )
    timing_log['1_recognizer_init'] = (time.time() - init_start) * 1000
    if SHOW_TIMING:
        print(f"   ⏱️  Init time: {timing_log['1_recognizer_init']:.2f}ms")
    print()

    # ========== STEP 2: Load Annotations ==========
    load_start = time.time()
    template_image_path, template_bboxes = load_annotations(annotations_file)
    timing_log['2_load_annotations'] = (time.time() - load_start) * 1000
    if SHOW_TIMING:
        print(f"⏱️  Load annotations: {timing_log['2_load_annotations']:.2f}ms")

    # ========== STEP 3: Initialize Matcher ==========
    matcher_start = time.time()
    matcher = TemplateMatcher(template_image_path, template_bboxes)
    timing_log['3_matcher_init'] = (time.time() - matcher_start) * 1000
    if SHOW_TIMING:
        print(f"⏱️  Matcher init: {timing_log['3_matcher_init']:.2f}ms\n")

    # ========== STEP 4: Template Matching ==========
    print("="*60)
    print("🔍 TEMPLATE MATCHING")
    print("="*60)
    
    match_start = time.time()
    bboxes, confidence, target_image = matcher.match(
        target_image_path,
        method='feature',
        threshold=0.3,
        debug=SHOW_DEBUG
    )
    timing_log['4_template_matching'] = (time.time() - match_start) * 1000
    
    if bboxes is None:
        print("❌ Template not found")
        return

    print(f"\n✅ Match successful!")
    print(f"   Confidence: {confidence:.3f}")
    print(f"   Found {len(bboxes)} bboxes")
    if SHOW_TIMING:
        print(f"   ⏱️  Matching time: {timing_log['4_template_matching']:.2f}ms")

    # ========== STEP 5: Draw Results ==========
    draw_start = time.time()
    result_with_polygon = matcher.draw_bboxes(target_image, bboxes, draw_polygon=True)
    cv2.imwrite('../results/result_with_polygon.jpg', result_with_polygon)
    timing_log['5_draw_save'] = (time.time() - draw_start) * 1000
    if SHOW_TIMING:
        print(f"   ⏱️  Draw & save: {timing_log['5_draw_save']:.2f}ms")

    print("\n" + "="*60)
    print("📝 TEXT RECOGNITION")
    print("="*60)

    # ========== STEP 6: Crop Regions ==========
    crop_start = time.time()
    text_regions = []
    text_indices = []
    
    for i, bbox in enumerate(bboxes):
        print(f"\n{'─'*60}")
        print(f"BBox {i+1}: {bbox.bbox_type}")
        if bbox.polygon:
            print(f"  Polygon: {bbox.polygon}")

        cropped = matcher.crop_region_with_perspective(target_image, bbox)
        cv2.imwrite(f'results/cropped_{bbox.bbox_type}_{i}.jpg', cropped)
        print(f"  💾 Saved: cropped_{bbox.bbox_type}_{i}.jpg")
        
        if bbox.bbox_type in ["text", "datecode"]:
            text_regions.append(cropped)
            text_indices.append(i)
    
    timing_log['6_crop_regions'] = (time.time() - crop_start) * 1000
    if SHOW_TIMING:
        print(f"\n⏱️  Crop all regions: {timing_log['6_crop_regions']:.2f}ms")
        if len(bboxes) > 0:
            print(f"   Average per bbox: {timing_log['6_crop_regions']/len(bboxes):.2f}ms")

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
