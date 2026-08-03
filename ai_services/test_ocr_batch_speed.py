import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "camera_management"))

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

home = os.environ.get('HOME')


def get_real_images():
    """Get real cropped images from test_result folder"""
    test_dir = Path(f"{home}/Source/ocr_datecode/ai_services/test_result")
    existing_images = sorted(test_dir.glob("cropped_region_*.png"))

    images = []
    image_names = []
    for img_path in existing_images:
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append(img)
            image_names.append(img_path.name)

    return images, image_names


def test_sequential_ocr(recognizer, images):
    """Test sequential OCR (one by one)"""
    results = []
    t0 = time.time()

    for img in images:
        text, conf = recognizer.recognize(img, return_confidence=True)
        results.append((text, conf))

    elapsed = time.time() - t0
    return results, elapsed


def test_batch_ocr(recognizer, images):
    """Test batch OCR (all at once)"""
    t0 = time.time()
    results = recognizer.recognize_batch(images)
    elapsed = time.time() - t0
    return results, elapsed


def compare_results(seq_results, batch_results, image_names):
    """Compare sequential vs batch OCR results"""
    print("\n" + "=" * 95)
    print("RESULT COMPARISON: Sequential vs Batch")
    print("=" * 95)
    print(f"{'Image':<35} {'Sequential':<18} {'Conf':<6} {'Batch':<18} {'Conf':<6} {'Match'}")
    print("-" * 95)

    all_match = True
    for i, (seq, batch) in enumerate(zip(seq_results, batch_results)):
        seq_text, seq_conf = seq
        batch_text, batch_conf = batch
        match = seq_text == batch_text
        if not match:
            all_match = False

        name = image_names[i] if i < len(image_names) else f"Image {i}"
        # Truncate long names/texts
        name_short = name[:33] if len(name) > 33 else name
        seq_short = seq_text[:16] if len(seq_text) > 16 else seq_text
        batch_short = batch_text[:16] if len(batch_text) > 16 else batch_text

        status = "✅" if match else "❌"
        print(f"{name_short:<35} {seq_short:<18} {seq_conf:>5.3f} {batch_short:<18} {batch_conf:>5.3f} {status}")

    print("-" * 95)
    if all_match:
        print("✅ All results MATCH - Batch OCR produces identical results!")
    else:
        print("❌ Some results DIFFER - Check batch implementation!")

    return all_match


def test_speed_comparison(recognizer, batch_sizes=[1, 2, 3, 5, 7, 10]):
    """Test speed with different batch sizes"""
    print("\n" + "=" * 80)
    print("SPEED COMPARISON: Sequential vs Batch OCR")
    print("=" * 80)

    all_images, _ = get_real_images()
    if not all_images:
        print("No test images found!")
        return

    results_table = []

    for batch_size in batch_sizes:
        # Get subset of images
        images = all_images[:batch_size] if len(all_images) >= batch_size else all_images * (batch_size // len(all_images) + 1)
        images = images[:batch_size]

        # Test sequential (average of 3 runs)
        seq_times = []
        for _ in range(3):
            _, t = test_sequential_ocr(recognizer, images)
            seq_times.append(t)
        seq_time = np.mean(seq_times)

        # Test batch (average of 3 runs)
        batch_times = []
        for _ in range(3):
            _, t = test_batch_ocr(recognizer, images)
            batch_times.append(t)
        batch_time = np.mean(batch_times)

        speedup = seq_time / batch_time if batch_time > 0 else 0
        savings = (1 - batch_time / seq_time) * 100 if seq_time > 0 else 0

        results_table.append({
            'batch_size': batch_size,
            'sequential_ms': seq_time * 1000,
            'batch_ms': batch_time * 1000,
            'speedup': speedup,
            'savings': savings,
            'per_image_seq': (seq_time * 1000) / batch_size,
            'per_image_batch': (batch_time * 1000) / batch_size
        })

    # Print table
    print(f"\n{'Regions':<8} {'Sequential':<12} {'Batch':<12} {'Speedup':<10} {'Savings':<10} {'Per-img Seq':<12} {'Per-img Batch'}")
    print("-" * 90)

    for r in results_table:
        print(
            f"{r['batch_size']:<8} "
            f"{r['sequential_ms']:>8.1f}ms   "
            f"{r['batch_ms']:>8.1f}ms   "
            f"{r['speedup']:>6.2f}x    "
            f"{r['savings']:>6.1f}%    "
            f"{r['per_image_seq']:>8.1f}ms    "
            f"{r['per_image_batch']:>8.1f}ms"
        )

    # Summary
    print("\n" + "-" * 90)
    if results_table:
        avg_speedup = np.mean([r['speedup'] for r in results_table])
        best = max(results_table, key=lambda x: x['speedup'])
        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"Best speedup at {best['batch_size']} regions: {best['speedup']:.2f}x ({best['savings']:.1f}% faster)")

    return results_table


def test_accuracy_verification(recognizer):
    """Verify batch OCR produces same results as sequential"""
    print("\n" + "=" * 80)
    print("ACCURACY VERIFICATION")
    print("=" * 80)

    images, image_names = get_real_images()
    if not images:
        print("No test images found in test_result folder!")
        return False

    print(f"Testing with {len(images)} real images...")

    # Run both methods
    seq_results, seq_time = test_sequential_ocr(recognizer, images)
    batch_results, batch_time = test_batch_ocr(recognizer, images)

    print(f"\nSequential time: {seq_time*1000:.1f}ms ({seq_time*1000/len(images):.1f}ms per image)")
    print(f"Batch time:      {batch_time*1000:.1f}ms ({batch_time*1000/len(images):.1f}ms per image)")
    print(f"Speedup:         {seq_time/batch_time:.2f}x")

    # Compare results
    return compare_results(seq_results, batch_results, image_names)


def compare_onnx_vs_trt_results(onnx_results, trt_results, image_names):
    """Compare ONNX vs TensorRT OCR results"""
    print("\n" + "=" * 80)
    print("RESULT COMPARISON: ONNX vs TensorRT")
    print("=" * 80)
    print(f"{'Image':<35} {'ONNX Text':<18} {'Conf':<6} {'TRT Text':<18} {'Conf':<6} {'Match'}")
    print("-" * 95)

    all_match = True
    for i, (onnx, trt) in enumerate(zip(onnx_results, trt_results)):
        onnx_text, onnx_conf = onnx
        trt_text, trt_conf = trt
        match = onnx_text == trt_text
        if not match:
            all_match = False

        name = image_names[i] if i < len(image_names) else f"Image {i}"
        # Truncate long names/texts
        name_short = name[:33] if len(name) > 33 else name
        onnx_short = onnx_text[:16] if len(onnx_text) > 16 else onnx_text
        trt_short = trt_text[:16] if len(trt_text) > 16 else trt_text

        status = "✅" if match else "❌"
        print(f"{name_short:<35} {onnx_short:<18} {onnx_conf:>5.3f} {trt_short:<18} {trt_conf:>5.3f} {status}")

    print("-" * 95)
    if all_match:
        print("✅ All results MATCH - TensorRT produces identical results to ONNX!")
    else:
        print("❌ Some results DIFFER - Check implementations!")

    return all_match


def test_onnx_vs_trt_comparison(batch_sizes=[1, 3, 5, 7]):
    """Compare ONNX vs TensorRT speed"""
    from text_recognizer import TextRecognizer

    # Try to import TRT with proper path handling
    try:
        # Add camera_management to path for TRT imports
        cm_path = Path(__file__).parent / "camera_management"
        if str(cm_path) not in sys.path:
            sys.path.insert(0, str(cm_path))

        from camera_management.ocr.backends.default_trt import TextRecognizerTRT
        trt_available = True
    except Exception as e:
        print(f"⚠️ TensorRT not available: {e}")
        print("   Skipping TRT comparison")
        return

    print("\n" + "=" * 80)
    print("BACKEND COMPARISON: ONNX vs TensorRT")
    print("=" * 80)

    # Load models
    onnx_model_path = f"{home}/Source/ocr_datecode/languages/english/rec.onnx"
    trt_engine_path = f"{home}/Source/ocr_datecode/languages/english/rec.engine"
    dict_path = f"{home}/Source/ocr_datecode/languages/english/dict.txt"

    if not Path(trt_engine_path).exists():
        print(f"⚠️ TRT engine not found: {trt_engine_path}")
        print("   Please build TRT engine first")
        return

    print("\nLoading ONNX model...")
    onnx_recognizer = TextRecognizer(
        model_path=onnx_model_path,
        dict_path=dict_path,
        use_gpu=True
    )

    print("Loading TensorRT model...")
    try:
        trt_recognizer = TextRecognizerTRT(
            engine_path=trt_engine_path,
            dict_path=dict_path
        )
    except Exception as e:
        print(f"⚠️ Failed to load TRT model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Warmup
    print("\nWarming up models...")
    warmup_img = np.random.randint(0, 255, (32, 100, 3), dtype=np.uint8)
    for _ in range(5):
        onnx_recognizer.recognize(warmup_img)
        trt_recognizer.recognize(warmup_img)

    all_images, image_names = get_real_images()
    if not all_images:
        print("No test images found!")
        return

    # First, compare accuracy between ONNX and TensorRT
    print("\n" + "=" * 80)
    print("ACCURACY COMPARISON: ONNX vs TensorRT")
    print("=" * 80)
    print(f"Testing with {len(all_images)} real images...")

    # Run ONNX batch recognition
    onnx_results = onnx_recognizer.recognize_batch(all_images)

    # Run TensorRT batch recognition
    trt_results = trt_recognizer.recognize_batch(all_images)

    # Compare results
    accuracy_match = compare_onnx_vs_trt_results(onnx_results, trt_results, image_names)

    print(f"\nTesting with {len(all_images)} real images")
    print(f"Batch sizes: {batch_sizes}\n")

    # Results table
    results = []

    for batch_size in batch_sizes:
        images = all_images[:batch_size] if len(all_images) >= batch_size else all_images * (batch_size // len(all_images) + 1)
        images = images[:batch_size]

        # ONNX Sequential
        onnx_seq_times = []
        for _ in range(3):
            _, t = test_sequential_ocr(onnx_recognizer, images)
            onnx_seq_times.append(t)
        onnx_seq = np.mean(onnx_seq_times)

        # ONNX Batch
        onnx_batch_times = []
        for _ in range(3):
            _, t = test_batch_ocr(onnx_recognizer, images)
            onnx_batch_times.append(t)
        onnx_batch = np.mean(onnx_batch_times)

        # TRT Sequential (note: TRT recognize_batch is actually sequential)
        trt_seq_times = []
        for _ in range(3):
            _, t = test_sequential_ocr(trt_recognizer, images)
            trt_seq_times.append(t)
        trt_seq = np.mean(trt_seq_times)

        # TRT "Batch" (actually sequential internally)
        trt_batch_times = []
        for _ in range(3):
            _, t = test_batch_ocr(trt_recognizer, images)
            trt_batch_times.append(t)
        trt_batch = np.mean(trt_batch_times)

        results.append({
            'batch_size': batch_size,
            'onnx_seq': onnx_seq * 1000,
            'onnx_batch': onnx_batch * 1000,
            'trt_seq': trt_seq * 1000,
            'trt_batch': trt_batch * 1000,
            'onnx_speedup': onnx_seq / onnx_batch if onnx_batch > 0 else 0,
            'trt_vs_onnx_seq': onnx_seq / trt_seq if trt_seq > 0 else 0,
            'trt_vs_onnx_batch': onnx_batch / trt_batch if trt_batch > 0 else 0,
        })

    # Print table
    print(f"{'Size':<6} {'ONNX Seq':<12} {'ONNX Batch':<12} {'TRT Seq':<12} {'TRT Batch':<12} {'ONNX Batch':<12} {'TRT vs ONNX':<12}")
    print(f"{'':6} {'(ms)':<12} {'(ms)':<12} {'(ms)':<12} {'(ms)':<12} {'Speedup':<12} {'Seq Speedup':<12}")
    print("-" * 90)

    for r in results:
        print(
            f"{r['batch_size']:<6} "
            f"{r['onnx_seq']:>8.1f}ms   "
            f"{r['onnx_batch']:>8.1f}ms   "
            f"{r['trt_seq']:>8.1f}ms   "
            f"{r['trt_batch']:>8.1f}ms   "
            f"{r['onnx_speedup']:>6.2f}x      "
            f"{r['trt_vs_onnx_seq']:>6.2f}x"
        )

    # Summary
    print("\n" + "-" * 90)
    print("SUMMARY:")
    avg_onnx_speedup = np.mean([r['onnx_speedup'] for r in results])
    avg_trt_vs_onnx_seq = np.mean([r['trt_vs_onnx_seq'] for r in results])
    avg_trt_vs_onnx_batch = np.mean([r['trt_vs_onnx_batch'] for r in results])

    print(f"  ONNX Batch vs Sequential:     {avg_onnx_speedup:.2f}x faster")
    print(f"  TRT vs ONNX Sequential:       {avg_trt_vs_onnx_seq:.2f}x faster")
    print(f"  TRT vs ONNX Batch:            {avg_trt_vs_onnx_batch:.2f}x faster")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)

    # Accuracy note
    if 'accuracy_match' in locals():
        if accuracy_match:
            print("✅ Accuracy: ONNX and TensorRT produce identical results")
        else:
            print("⚠️ Accuracy: ONNX and TensorRT produce different results - verify before deployment!")

    # Find best option for each scenario
    print("\nBest backend by batch size:")
    for r in results:
        batch_size = r['batch_size']
        options = {
            'ONNX Sequential': r['onnx_seq'],
            'ONNX Batch': r['onnx_batch'],
            'TRT Sequential': r['trt_seq'],
            'TRT Batch': r['trt_batch']
        }
        best = min(options.items(), key=lambda x: x[1])
        print(f"  {batch_size} regions: {best[0]} ({best[1]:.1f}ms)")

    print("\nFor multi-camera batch (5+ regions):")
    if avg_trt_vs_onnx_batch > 1.2:
        print(f"  ✅ Use TRT - {avg_trt_vs_onnx_batch:.2f}x faster than ONNX Batch")
    else:
        print(f"  ✅ Use ONNX Batch - {avg_onnx_speedup:.2f}x speedup, easier to deploy")

    return accuracy_match if 'accuracy_match' in locals() else None


def main():
    from text_recognizer import TextRecognizer

    model_path = f"{home}/Source/ocr_datecode/languages/english/rec.onnx"
    dict_path = f"{home}/Source/ocr_datecode/languages/english/dict.txt"

    print("=" * 80)
    print("OCR INFERENCE SPEED COMPARISON")
    print("=" * 80)

    print("\n[TEST 1] ONNX: Sequential vs Batch")
    print("-" * 80)

    print("\nLoading ONNX model...")
    recognizer = TextRecognizer(
        model_path=model_path,
        dict_path=dict_path,
        use_gpu=True
    )

    # Warmup
    print("Warming up...")
    warmup_img = np.random.randint(0, 255, (32, 100, 3), dtype=np.uint8)
    for _ in range(5):
        recognizer.recognize(warmup_img)
    print("Warmup complete!\n")

    # Test 1: Accuracy verification
    accuracy_ok = test_accuracy_verification(recognizer)

    # Test 2: ONNX Speed comparison
    test_speed_comparison(recognizer, batch_sizes=[1, 3, 5, 7])

    # Test 3: ONNX vs TRT comparison
    print("\n[TEST 2] Backend Comparison: ONNX vs TensorRT")
    print("-" * 80)
    trt_accuracy_ok = test_onnx_vs_trt_comparison(batch_sizes=[1, 3, 5, 7])

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print("\n📊 ONNX Batch Verification:")
    if accuracy_ok:
        print("  ✅ ONNX Batch produces IDENTICAL results to Sequential")
        print("  ✅ Safe to use batch_verify_text_regions_multi_camera()")
    else:
        print("  ⚠️ ONNX Batch may produce different results - investigate before using")

    if trt_accuracy_ok is not None:
        print("\n📊 ONNX vs TensorRT Verification:")
        if trt_accuracy_ok:
            print("  ✅ TensorRT produces IDENTICAL results to ONNX")
            print("  ✅ Both backends can be used interchangeably")
        else:
            print("  ⚠️ TensorRT produces DIFFERENT results from ONNX")
            print("  ⚠️ Verify which backend is more accurate for your use case")

    print("\n🎯 Key Findings:")
    print("  • ONNX Batch: Good speedup for 3+ regions, easier deployment")
    print("  • TensorRT: Fastest for single region inference, requires engine build")
    print("  • Recommendation: Use ONNX Batch for multi-camera scenarios")


if __name__ == "__main__":
    main()
