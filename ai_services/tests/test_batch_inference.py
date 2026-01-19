"""
Test batch inference with multiple cameras
"""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from inference_service import SuperPointMatcherTRT

def test_batch_inference():
    """Test batch inference with 2-4 cameras"""

    engine_path = "/home/demo/Source/ocr_datecode/weights/pipeline_fp16_dynamic.engine"

    if not Path(engine_path).exists():
        print(f"❌ Engine not found: {engine_path}")
        return False

    temp_dir = Path("ocr_inference")
    temp_dir.mkdir(exist_ok=True)

    # Create 3 different templates (simulating 3 cameras)
    templates = []
    matchers = []

    for i in range(3):
        # Create test template
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.putText(img, f"Camera {i+1}", (200, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        template_path = temp_dir / f"test_cam{i+1}_template.jpg"
        cv2.imwrite(str(template_path), img)
        templates.append(template_path)

        # Create annotation
        import json
        ann_path = temp_dir / f"test_cam{i+1}_ann.json"
        ann_data = {
            "_template_image": str(template_path),
            str(template_path): [
                {
                    "type": "template",
                    "points": [[100, 100], [500, 100], [500, 350], [100, 350]]
                }
            ]
        }
        with open(ann_path, 'w') as f:
            json.dump(ann_data, f)

        # Initialize matcher
        matcher = SuperPointMatcherTRT(
            json_path=str(ann_path),
            engine_path=engine_path,
            scale=1.0,
            verbose=(i == 0)  # Only verbose for first
        )
        matchers.append(matcher)
        print(f"✅ Matcher {i+1} initialized")

    # Test different batch sizes
    print("\n📸 Testing batch inference...")

    for num_cameras in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"Testing {num_cameras} camera(s)")
        print(f"{'='*60}")

        # Prepare target images (same as templates for this test)
        target_imgs = []
        for i in range(num_cameras):
            target_img = cv2.imread(str(templates[i]))
            target_imgs.append(target_img)

        # Select matchers
        selected_matchers = matchers[:num_cameras]

        if num_cameras == 1:
            # Single camera - use match_array
            print("Using match_array (single camera)...")
            result = selected_matchers[0].match_array(
                target_img_array=target_imgs[0],
                score_threshold=0.3,
                ransac_threshold=5.0
            )

            print(f"  Success: {result['success']}")
            if result['success']:
                timings = result.get('timings', {})
                print(f"  Total: {timings.get('total', 0):.1f}ms")
                print(f"  TRT: {timings.get('trt_inference', 0):.1f}ms")
                print(f"  Confidence: {result.get('confidence', 0):.2%}")

        else:
            # Multiple cameras - use match_batch
            print(f"Using match_batch ({num_cameras} cameras)...")
            batch_result = selected_matchers[0].match_batch(
                target_imgs=target_imgs,
                templates=selected_matchers,
                score_threshold=0.3,
                ransac_threshold=5.0
            )

            print(f"  Batch success: {batch_result['success']}")

            if batch_result['success']:
                batch_timings = batch_result['batch_timings']
                print(f"\n  Batch timings:")
                print(f"    Total: {batch_timings['total']:.1f}ms")
                print(f"    TRT (shared): {batch_timings['trt_inference']:.1f}ms")
                print(f"    Preprocess: {batch_timings['preprocess']:.1f}ms")
                print(f"    Postprocess: {batch_timings['postprocess']:.1f}ms")

                print(f"\n  Per-camera results:")
                for idx, result in enumerate(batch_result['results']):
                    print(f"    Camera {idx+1}:")
                    print(f"      Success: {result['success']}")
                    if result['success']:
                        print(f"      Confidence: {result.get('confidence', 0):.2%}")
                        print(f"      Inliers: {result.get('inliers', 0)}/{result.get('total_matches', 0)}")
                        timings = result.get('timings', {})
                        print(f"      Timings: total={timings.get('total', 0):.1f}ms, "
                              f"trt={timings.get('trt_inference', 0):.1f}ms, "
                              f"pre={timings.get('preprocess', 0):.1f}ms, "
                              f"post={timings.get('postprocess', 0):.1f}ms")

                # Compare speedup
                sequential_time = num_cameras * 200  # Estimate 200ms per camera
                batch_time = batch_timings['total']
                speedup = sequential_time / batch_time
                print(f"\n  Performance:")
                print(f"    Sequential (est): {sequential_time:.1f}ms")
                print(f"    Batch: {batch_time:.1f}ms")
                print(f"    Speedup: {speedup:.2f}x ✨")

    print(f"\n{'='*60}")
    print("✅ Batch inference test complete!")
    print(f"{'='*60}")

    return True

if __name__ == "__main__":
    success = test_batch_inference()
    sys.exit(0 if success else 1)
