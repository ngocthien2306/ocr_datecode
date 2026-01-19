"""
Benchmark script to compare ONNX Runtime vs TensorRT performance
"""
import cv2
import time
import numpy as np
from pathlib import Path
from desktop.text_recognizer import TextRecognizer

# Try import TensorRT
try:
    from desktop.text_recognizer_tensorrt import TextRecognizerTensorRT
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("⚠️  TensorRT not available")


def create_test_images(num_images=10):
    """Create test images with various widths"""
    images = []
    widths = np.linspace(320, 2000, num_images, dtype=int)
    
    for w in widths:
        # Create random grayscale image
        img = np.random.randint(0, 255, (48, w), dtype=np.uint8)
        # Convert to BGR for consistency
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        images.append(img_bgr)
    
    return images


def benchmark_recognizer(recognizer, images, name="Recognizer", warmup=3, iterations=10):
    """Benchmark a recognizer with given images"""
    print(f"\n{'='*60}")
    print(f"🔥 Benchmarking {name}")
    print(f"{'='*60}")
    
    # Warm-up
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        for img in images[:3]:  # Use first 3 images for warmup
            _ = recognizer.recognize(img)
    
    # Single image inference
    print(f"\n📊 Single Image Inference ({iterations} iterations per image)")
    single_times = []
    
    for img in images:
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            text, conf = recognizer.recognize(img)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        single_times.append(avg_time)
        
        w = img.shape[1]
        print(f"  Width {w:4d}px: {avg_time:6.2f}ms ± {std_time:5.2f}ms")
    
    avg_single = np.mean(single_times)
    print(f"\n  Average: {avg_single:.2f}ms")
    print(f"  Throughput: {1000/avg_single:.1f} FPS")
    
    # Batch inference
    print(f"\n📦 Batch Inference ({len(images)} images, {iterations} iterations)")
    batch_times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        results = recognizer.recognize_batch(images)
        elapsed = (time.perf_counter() - start) * 1000
        batch_times.append(elapsed)
    
    avg_batch = np.mean(batch_times)
    std_batch = np.std(batch_times)
    avg_per_image = avg_batch / len(images)
    
    print(f"  Total batch time: {avg_batch:.2f}ms ± {std_batch:.2f}ms")
    print(f"  Per image: {avg_per_image:.2f}ms")
    print(f"  Throughput: {1000/avg_per_image:.1f} FPS")
    
    return {
        'name': name,
        'single_avg': avg_single,
        'single_times': single_times,
        'batch_total': avg_batch,
        'batch_per_image': avg_per_image,
        'throughput': 1000 / avg_per_image
    }


def main():
    print("="*60)
    print("⚡ OCR Performance Benchmark")
    print("="*60)
    
    # Create test images
    print("\n📸 Creating test images...")
    num_images = 10
    images = create_test_images(num_images)
    print(f"   Created {num_images} test images")
    print(f"   Width range: {images[0].shape[1]}px - {images[-1].shape[1]}px")
    
    results = []
    
    # Benchmark ONNX Runtime
    print("\n" + "="*60)
    print("Testing ONNX Runtime (CPU)")
    print("="*60)
    
    try:
        onnx_recognizer = TextRecognizer(
            model_path='../languages/english/rec.onnx',
            dict_path='../languages/english/dict.txt',
            use_gpu=False
        )
        onnx_result = benchmark_recognizer(
            onnx_recognizer, 
            images, 
            name="ONNX Runtime (CPU)",
            warmup=2,
            iterations=5
        )
        results.append(onnx_result)
    except Exception as e:
        print(f"❌ ONNX Runtime benchmark failed: {e}")
    
    # Benchmark TensorRT
    if TENSORRT_AVAILABLE:
        print("\n" + "="*60)
        print("Testing TensorRT (GPU)")
        print("="*60)
        
        try:
            trt_recognizer = TextRecognizerTensorRT(
                engine_path='../languages/english/rec.engine',
                dict_path='../languages/english/dict.txt'
            )
            trt_result = benchmark_recognizer(
                trt_recognizer, 
                images, 
                name="TensorRT (GPU)",
                warmup=5,
                iterations=10
            )
            results.append(trt_result)
        except Exception as e:
            print(f"❌ TensorRT benchmark failed: {e}")
    else:
        print("\n⚠️  TensorRT not available - skipping TensorRT benchmark")
        print("   Install with: pip install nvidia-tensorrt pycuda")
    
    # Summary
    if len(results) >= 2:
        print("\n" + "="*60)
        print("📊 PERFORMANCE COMPARISON")
        print("="*60)
        
        onnx_res = results[0]
        trt_res = results[1]
        
        print(f"\n{'Metric':<25} {'ONNX (CPU)':<15} {'TensorRT (GPU)':<15} {'Speedup':<10}")
        print("-" * 70)
        
        # Single image
        speedup_single = onnx_res['single_avg'] / trt_res['single_avg']
        print(f"{'Single image (avg)':<25} {onnx_res['single_avg']:>10.2f}ms {trt_res['single_avg']:>14.2f}ms {speedup_single:>9.1f}x")
        
        # Batch per image
        speedup_batch = onnx_res['batch_per_image'] / trt_res['batch_per_image']
        print(f"{'Batch per image':<25} {onnx_res['batch_per_image']:>10.2f}ms {trt_res['batch_per_image']:>14.2f}ms {speedup_batch:>9.1f}x")
        
        # Throughput
        print(f"\n{'Throughput':<25} {onnx_res['throughput']:>10.1f} FPS {trt_res['throughput']:>13.1f} FPS")
        
        print("\n" + "="*60)
        print(f"✨ TensorRT is {speedup_batch:.1f}x faster than ONNX Runtime!")
        print("="*60)
    
    elif len(results) == 1:
        print("\n" + "="*60)
        print("📊 SINGLE BACKEND RESULTS")
        print("="*60)
        res = results[0]
        print(f"\nBackend: {res['name']}")
        print(f"Average inference: {res['single_avg']:.2f}ms")
        print(f"Batch per image: {res['batch_per_image']:.2f}ms")
        print(f"Throughput: {res['throughput']:.1f} FPS")
    
    print("\n✅ Benchmark complete!")


if __name__ == '__main__':
    main()
