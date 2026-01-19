"""
Test dynamic batch inference: 2-8 batch size
"""
import cv2
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_dynamic_batch_engine():
    """Test inference with dynamic batch engine"""

    # Check if new engine exists
    engine_path = "/home/demo/Source/ocr_datecode/weights/pipeline_fp16_dynamic.engine"

    if not Path(engine_path).exists():
        print(f"❌ Engine not found: {engine_path}")
        print("   Run build script first: bash build_engine_batch6.sh")
        return False

    from inference_service import SuperPointMatcherTRT

    # Create dummy template and annotation for testing
    temp_dir = Path("ocr_inference")
    temp_dir.mkdir(exist_ok=True)

    # Create 3 dummy templates
    templates = []
    for i in range(3):
        # Create simple test image
        img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        cv2.putText(img, f"Template {i+1}", (200, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        template_path = temp_dir / f"test_template_{i+1}.jpg"
        cv2.imwrite(str(template_path), img)
        templates.append(template_path)

    # Create annotation JSON for first template
    import json
    ann_path = temp_dir / "test_annotations.json"
    ann_data = {
        "_template_image": str(templates[0]),
        str(templates[0]): [
            {
                "type": "template",
                "points": [[100, 100], [500, 100], [500, 350], [100, 350]]
            }
        ]
    }
    with open(ann_path, 'w') as f:
        json.dump(ann_data, f)

    print("🔧 Initializing matcher with batch 6 engine...")
    try:
        matcher = SuperPointMatcherTRT(
            json_path=str(ann_path),
            engine_path=engine_path,
            scale=1.0,
            verbose=True
        )
        print(f"   ✅ Matcher initialized")
        print(f"   Input shape: {matcher.input_shape}")

    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        return False

    # Test different batch sizes
    print("\n📸 Testing dynamic batch sizes...")

    h, w = matcher.input_shape[2:]
    test_results = []

    for num_cameras in [1, 2, 3, 4]:
        batch_size = num_cameras * 2  # templates + targets
        print(f"\n   Test {num_cameras} cameras (batch size {batch_size}):")

        try:
            batch_images = []

            # Load template-target pairs
            for i in range(num_cameras):
                template_idx = i % 3  # Cycle through 3 templates
                template_img = cv2.imread(str(templates[template_idx]))
                target_img = template_img.copy()

                # Convert to grayscale and normalize
                template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
                target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

                # Resize to engine size
                template_resized = cv2.resize(template_gray, (w, h))
                target_resized = cv2.resize(target_gray, (w, h))

                batch_images.append(template_resized)
                batch_images.append(target_resized)

            # Create batch tensor
            batch_input = np.array([img[None, ...] for img in batch_images], dtype=np.float32) / 255.0
            print(f"      Batch shape: {batch_input.shape}")

            # Run inference
            t0 = time.time()
            outputs = matcher._infer(batch_input)
            inference_time = (time.time() - t0) * 1000

            print(f"      ✅ Time: {inference_time:.1f}ms")
            test_results.append((num_cameras, inference_time))

        except Exception as e:
            print(f"      ❌ Failed: {e}")
            test_results.append((num_cameras, None))

    # Summary
    print(f"\n🎯 Results Summary:")
    print(f"   {'Cameras':<10} {'Sequential':<15} {'Batch':<15} {'Speedup':<10}")
    print(f"   {'-'*50}")

    for num_cameras, batch_time in test_results:
        sequential_time = num_cameras * 311  # 311ms per camera
        if batch_time:
            speedup = sequential_time / batch_time
            print(f"   {num_cameras:<10} {sequential_time:<15.1f}ms {batch_time:<15.1f}ms {speedup:<10.2f}x")
        else:
            print(f"   {num_cameras:<10} {sequential_time:<15.1f}ms {'FAILED':<15} {'-':<10}")

    return all(result[1] is not None for result in test_results)

if __name__ == "__main__":
    success = test_dynamic_batch_engine()
    sys.exit(0 if success else 1)
