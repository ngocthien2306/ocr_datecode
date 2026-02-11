#!/usr/bin/env python3
"""
Benchmark different batch sizes - TensorRT vs ONNX comparison
"""

import sys
import time
from test_trt_inference import TensorRTOCR
from simple_ocr_inference import SimpleOCR

def benchmark_batch_size(ocr, img_paths, batch_size):
    """
    Benchmark with specific batch size

    Args:
        ocr: TensorRT OCR instance
        img_paths: list of image paths
        batch_size: 1 for sequential, >1 for true batch
    """
    results = []
    start = time.time()

    if batch_size == 1:
        # Sequential processing (batch=1)
        for img_path in img_paths:
            result = ocr.recognize(img_path)
            results.append(result)
    else:
        # Process in chunks of batch_size
        for i in range(0, len(img_paths), batch_size):
            chunk = img_paths[i:i+batch_size]
            chunk_results = ocr.recognize_batch(chunk, use_true_batch=True)
            results.extend(chunk_results)

    elapsed = time.time() - start
    return results, elapsed

def benchmark_onnx(ocr_onnx, img_paths, batch_size):
    """
    Benchmark ONNX with specific batch size

    Args:
        ocr_onnx: ONNX OCR instance
        img_paths: list of image paths
        batch_size: batch size for processing
    """
    start = time.time()
    results = ocr_onnx.recognize(img_paths, batch_size=batch_size)
    elapsed = time.time() - start
    return results, elapsed

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python benchmark_batch_sizes.py <image1> [image2 ...]")
        print("\nRecommend testing with 8+ images to see batch effects")
        sys.exit(1)

    img_paths = sys.argv[1:]
    num_images = len(img_paths)

    if num_images < 8:
        print(f"Warning: Only {num_images} images. Recommend 8+ for accurate batch comparison\n")

    # Load engines
    batch_engine = './languages/english/openocr_rec_model_batch.engine'
    onnx_model = './languages/english/openocr_rec_model.onnx'

    print(f"Benchmarking with {num_images} images")
    print(f"TensorRT Engine: {batch_engine}")
    print(f"ONNX Model:      {onnx_model}")
    print("="*90)

    print("\n[Loading Models...]")
    ocr_trt = TensorRTOCR(batch_engine)
    ocr_onnx = SimpleOCR(onnx_model)

    # Test different batch sizes
    batch_sizes = [1, 4, 8]
    results_trt = {}
    results_onnx = {}

    # ========== TensorRT Benchmark ==========
    print("\n\n" + "="*90)
    print("TENSORRT BENCHMARK")
    print("="*90)

    for bs in batch_sizes:
        if bs > num_images and bs > 1:
            print(f"\n[Batch Size = {bs}] Skipped (not enough images)")
            continue

        print(f"\n[TensorRT - Batch Size = {bs}]", end="")
        if bs == 1:
            print(" (Sequential)")
        else:
            print(f" (True Batch)")

        # Run benchmark
        results, elapsed = benchmark_batch_size(ocr_trt, img_paths, bs)

        throughput = num_images / elapsed
        avg_per_image = elapsed / num_images

        results_trt[bs] = {
            'time': elapsed,
            'throughput': throughput,
            'avg_per_image': avg_per_image,
            'results': results
        }

        print(f"  Total time:       {elapsed*1000:.2f}ms")
        print(f"  Per image:        {avg_per_image*1000:.2f}ms")
        print(f"  Throughput:       {throughput:.2f} images/sec")

    # ========== ONNX Benchmark ==========
    print("\n\n" + "="*90)
    print("ONNX BENCHMARK")
    print("="*90)

    for bs in batch_sizes:
        if bs > num_images and bs > 1:
            print(f"\n[Batch Size = {bs}] Skipped (not enough images)")
            continue

        print(f"\n[ONNX - Batch Size = {bs}]", end="")
        if bs == 1:
            print(" (Sequential)")
        else:
            print(f" (Batch)")

        # Run benchmark
        results, elapsed = benchmark_onnx(ocr_onnx, img_paths, bs)

        throughput = num_images / elapsed
        avg_per_image = elapsed / num_images

        results_onnx[bs] = {
            'time': elapsed,
            'throughput': throughput,
            'avg_per_image': avg_per_image,
            'results': results
        }

        print(f"  Total time:       {elapsed*1000:.2f}ms")
        print(f"  Per image:        {avg_per_image*1000:.2f}ms")
        print(f"  Throughput:       {throughput:.2f} images/sec")

    # ========== Summary Comparison ==========
    print("\n\n" + "="*90)
    print("TENSORRT SUMMARY:")
    print("-"*90)
    print(f"{'Batch Size':<15} {'Total Time':<15} {'Per Image':<15} {'Throughput':<20} {'vs Batch=1'}")
    print("-"*90)

    baseline_trt = results_trt[1]['throughput']
    for bs in sorted(results_trt.keys()):
        data = results_trt[bs]
        speedup = data['throughput'] / baseline_trt
        speedup_str = f"{speedup:.2f}x" if bs > 1 else "baseline"
        print(f"{bs:<15} {data['time']*1000:<15.2f} {data['avg_per_image']*1000:<15.2f} "
              f"{data['throughput']:<20.2f} {speedup_str}")

    print("\n" + "="*90)
    print("ONNX SUMMARY:")
    print("-"*90)
    print(f"{'Batch Size':<15} {'Total Time':<15} {'Per Image':<15} {'Throughput':<20} {'vs Batch=1'}")
    print("-"*90)

    baseline_onnx = results_onnx[1]['throughput']
    for bs in sorted(results_onnx.keys()):
        data = results_onnx[bs]
        speedup = data['throughput'] / baseline_onnx
        speedup_str = f"{speedup:.2f}x" if bs > 1 else "baseline"
        print(f"{bs:<15} {data['time']*1000:<15.2f} {data['avg_per_image']*1000:<15.2f} "
              f"{data['throughput']:<20.2f} {speedup_str}")

    # ========== TensorRT vs ONNX Comparison ==========
    print("\n\n" + "="*90)
    print("TENSORRT vs ONNX COMPARISON:")
    print("-"*90)
    print(f"{'Batch Size':<15} {'TensorRT (imgs/s)':<25} {'ONNX (imgs/s)':<25} {'TRT Speedup'}")
    print("-"*90)

    for bs in sorted(results_trt.keys()):
        trt_throughput = results_trt[bs]['throughput']
        onnx_throughput = results_onnx[bs]['throughput']
        speedup = trt_throughput / onnx_throughput
        print(f"{bs:<15} {trt_throughput:<25.2f} {onnx_throughput:<25.2f} {speedup:.2f}x")

    # Show sample results
    print(f"\n\nSample Recognition Results (first 5):")
    sample_results = results_trt[1]['results'][:5]
    for i, result in enumerate(sample_results):
        img_name = img_paths[i].split('/')[-1]
        print(f"  {i+1}. {img_name:<25} -> '{result['text']:<30}' (conf: {result['confidence']:.3f})")

    # Final recommendation
    print("\n" + "="*90)
    print("FINAL RECOMMENDATION:")
    print("-"*90)

    # Find best config
    best_trt_bs = max(results_trt.keys(), key=lambda x: results_trt[x]['throughput'])
    best_trt_throughput = results_trt[best_trt_bs]['throughput']

    best_onnx_bs = max(results_onnx.keys(), key=lambda x: results_onnx[x]['throughput'])
    best_onnx_throughput = results_onnx[best_onnx_bs]['throughput']

    overall_speedup = best_trt_throughput / best_onnx_throughput

    print(f"\n  Best TensorRT config:  Batch Size = {best_trt_bs} → {best_trt_throughput:.2f} imgs/sec")
    print(f"  Best ONNX config:      Batch Size = {best_onnx_bs} → {best_onnx_throughput:.2f} imgs/sec")
    print(f"\n  → TensorRT is {overall_speedup:.2f}x faster than ONNX")
    print(f"  → Recommended: Use TensorRT with Batch Size = {best_trt_bs}")
