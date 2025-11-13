"""
Benchmark script to compare CPU vs CUDA template matching performance
Specifically for Jetson AGX Orin
"""
import cv2
import json
import time
import numpy as np
from template_matcher import TemplateMatcher, BoundingBox, OPENCV_CUDA_AVAILABLE


def load_annotations(annotations_file):
    with open(annotations_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    template_image_path = data.get('_template_image')
    template_bboxes_data = data.get(template_image_path, [])
    template_bboxes = [BoundingBox.from_dict(b) for b in template_bboxes_data]
    return template_image_path, template_bboxes


def benchmark_method(matcher, target_path, method, use_cuda, iterations=10, warmup=3):
    """Benchmark a single matching method"""
    target_image = cv2.imread(target_path)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    
    # Warm-up
    for _ in range(warmup):
        if method == 'feature':
            _ = matcher.match_feature_based(target_gray, use_cuda=use_cuda)
        elif method == 'orb':
            _ = matcher.match_orb(target_gray, use_cuda=use_cuda)
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        
        if method == 'feature':
            result = matcher.match_feature_based(target_gray, use_cuda=use_cuda)
        elif method == 'orb':
            result = matcher.match_orb(target_gray, use_cuda=use_cuda)
        
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'times': times
    }


def main():
    print("="*70)
    print("⚡ CUDA vs CPU Template Matching Benchmark")
    print("   Optimized for Jetson AGX Orin")
    print("="*70)
    
    # Check CUDA availability
    print(f"\n🔍 CUDA Status:")
    print(f"   OpenCV CUDA Available: {OPENCV_CUDA_AVAILABLE}")
    if OPENCV_CUDA_AVAILABLE:
        print(f"   CUDA Devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
        print(f"   Device Name: {cv2.cuda.printShortCudaDeviceInfo(0)}")
    print()
    
    # Load template and annotations
    annotations_file = '../images/annotations.json'
    target_image_path = '../images/2.jpg'
    
    print("📂 Loading template and annotations...")
    template_image_path, template_bboxes = load_annotations(annotations_file)
    matcher = TemplateMatcher(template_image_path, template_bboxes)
    print(f"   Template: {template_image_path}")
    print(f"   Target: {target_image_path}")
    print()
    
    iterations = 20
    warmup = 5
    
    results = {}
    
    # Benchmark SIFT (feature-based)
    print("="*70)
    print("🔬 Benchmarking SIFT Feature Matching")
    print("="*70)
    
    print(f"\n📊 CPU-based SIFT ({iterations} iterations, {warmup} warmup)...")
    cpu_sift = benchmark_method(
        matcher, target_image_path, 'feature', 
        use_cuda=False, iterations=iterations, warmup=warmup
    )
    results['sift_cpu'] = cpu_sift
    print(f"   Mean: {cpu_sift['mean']:.2f}ms ± {cpu_sift['std']:.2f}ms")
    print(f"   Min:  {cpu_sift['min']:.2f}ms")
    print(f"   Max:  {cpu_sift['max']:.2f}ms")
    
    if OPENCV_CUDA_AVAILABLE:
        print(f"\n⚡ CUDA-based SIFT ({iterations} iterations, {warmup} warmup)...")
        cuda_sift = benchmark_method(
            matcher, target_image_path, 'feature',
            use_cuda=True, iterations=iterations, warmup=warmup
        )
        results['sift_cuda'] = cuda_sift
        print(f"   Mean: {cuda_sift['mean']:.2f}ms ± {cuda_sift['std']:.2f}ms")
        print(f"   Min:  {cuda_sift['min']:.2f}ms")
        print(f"   Max:  {cuda_sift['max']:.2f}ms")
        
        speedup = cpu_sift['mean'] / cuda_sift['mean']
        print(f"\n   🚀 CUDA Speedup: {speedup:.2f}x")
    
    # Benchmark ORB
    print("\n" + "="*70)
    print("🔬 Benchmarking ORB Feature Matching")
    print("="*70)
    
    print(f"\n📊 CPU-based ORB ({iterations} iterations, {warmup} warmup)...")
    cpu_orb = benchmark_method(
        matcher, target_image_path, 'orb',
        use_cuda=False, iterations=iterations, warmup=warmup
    )
    results['orb_cpu'] = cpu_orb
    print(f"   Mean: {cpu_orb['mean']:.2f}ms ± {cpu_orb['std']:.2f}ms")
    print(f"   Min:  {cpu_orb['min']:.2f}ms")
    print(f"   Max:  {cpu_orb['max']:.2f}ms")
    
    if OPENCV_CUDA_AVAILABLE:
        print(f"\n⚡ CUDA-based ORB ({iterations} iterations, {warmup} warmup)...")
        cuda_orb = benchmark_method(
            matcher, target_image_path, 'orb',
            use_cuda=True, iterations=iterations, warmup=warmup
        )
        results['orb_cuda'] = cuda_orb
        print(f"   Mean: {cuda_orb['mean']:.2f}ms ± {cuda_orb['std']:.2f}ms")
        print(f"   Min:  {cuda_orb['min']:.2f}ms")
        print(f"   Max:  {cuda_orb['max']:.2f}ms")
        
        speedup = cpu_orb['mean'] / cuda_orb['mean']
        print(f"\n   🚀 CUDA Speedup: {speedup:.2f}x")
    
    # Summary
    print("\n" + "="*70)
    print("📊 PERFORMANCE SUMMARY")
    print("="*70)
    
    print(f"\n{'Method':<20} {'CPU (ms)':<15} {'CUDA (ms)':<15} {'Speedup':<10}")
    print("-" * 70)
    
    # SIFT comparison
    if OPENCV_CUDA_AVAILABLE and 'sift_cuda' in results:
        sift_speedup = results['sift_cpu']['mean'] / results['sift_cuda']['mean']
        print(f"{'SIFT':<20} {results['sift_cpu']['mean']:>10.2f} {results['sift_cuda']['mean']:>14.2f} {sift_speedup:>9.2f}x")
    else:
        print(f"{'SIFT':<20} {results['sift_cpu']['mean']:>10.2f} {'N/A':<15} {'N/A':<10}")
    
    # ORB comparison
    if OPENCV_CUDA_AVAILABLE and 'orb_cuda' in results:
        orb_speedup = results['orb_cpu']['mean'] / results['orb_cuda']['mean']
        print(f"{'ORB':<20} {results['orb_cpu']['mean']:>10.2f} {results['orb_cuda']['mean']:>14.2f} {orb_speedup:>9.2f}x")
    else:
        print(f"{'ORB':<20} {results['orb_cpu']['mean']:>10.2f} {'N/A':<15} {'N/A':<10}")
    
    # Recommendations
    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS")
    print("="*70)
    
    if OPENCV_CUDA_AVAILABLE:
        # Find fastest method
        methods = [
            ('SIFT CPU', results['sift_cpu']['mean']),
            ('SIFT CUDA', results.get('sift_cuda', {}).get('mean', float('inf'))),
            ('ORB CPU', results['orb_cpu']['mean']),
            ('ORB CUDA', results.get('orb_cuda', {}).get('mean', float('inf')))
        ]
        fastest = min(methods, key=lambda x: x[1])
        
        print(f"\n✅ Fastest method: {fastest[0]} ({fastest[1]:.2f}ms)")
        print(f"\nFor Jetson AGX Orin:")
        print(f"  • Use ORB CUDA for maximum speed")
        print(f"  • Use SIFT CUDA for better accuracy")
        print(f"  • Expected speedup: 2-5x over CPU")
    else:
        print(f"\n⚠️  OpenCV CUDA not available")
        print(f"   To enable CUDA acceleration on Jetson:")
        print(f"   1. Build OpenCV with CUDA support")
        print(f"   2. Or use JetPack SDK which includes CUDA-enabled OpenCV")
        print(f"\n   Current best: ORB CPU ({results['orb_cpu']['mean']:.2f}ms)")
    
    print("\n" + "="*70)
    print("✅ Benchmark complete!")
    print("="*70)


if __name__ == '__main__':
    main()
