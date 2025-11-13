# CUDA-Accelerated Template Matching Guide for Jetson AGX Orin

## 🚀 Overview

This guide shows how to enable CUDA acceleration for template matching on Jetson AGX Orin, providing **2-5x speedup** over CPU-based matching.

## 📋 Prerequisites

### Hardware
- NVIDIA Jetson AGX Orin (or other Jetson devices)
- At least 16GB RAM
- 32GB+ free storage

### Software
- JetPack 5.0+ (includes CUDA toolkit)
- Python 3.8+

## 🔧 Installation

### Option 1: Use Pre-built OpenCV from JetPack (Recommended)

JetPack already includes OpenCV with CUDA support:

```bash
# Check if CUDA-enabled OpenCV is available
python3 -c "import cv2; print('CUDA:', cv2.cuda.getCudaEnabledDeviceCount())"
```

If output shows `CUDA: 1` or higher, you're ready to go! Skip to Usage section.

### Option 2: Build OpenCV with CUDA Support

If pre-built OpenCV doesn't have CUDA, build from source:

#### 1. Install dependencies
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git pkg-config \
    libjpeg-dev libtiff5-dev libpng-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev libatlas-base-dev gfortran \
    python3-dev python3-pip
```

#### 2. Download OpenCV
```bash
cd ~
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
git checkout 4.8.0  # Or latest stable version
cd ../opencv_contrib
git checkout 4.8.0
```

#### 3. Configure build with CUDA
```bash
cd ~/opencv
mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D WITH_CUDA=ON \
    -D CUDA_ARCH_BIN="8.7" \
    -D CUDA_ARCH_PTX="" \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D WITH_LIBV4L=ON \
    -D BUILD_opencv_python3=ON \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
    ..
```

**Note:** `CUDA_ARCH_BIN="8.7"` is for Jetson AGX Orin. For other Jetson devices:
- Jetson Nano/TX1: `5.3`
- Jetson TX2: `6.2`
- Jetson Xavier: `7.2`
- Jetson Orin: `8.7`

#### 4. Build (takes 1-2 hours)
```bash
make -j$(nproc)
sudo make install
sudo ldconfig
```

#### 5. Verify installation
```bash
python3 -c "import cv2; print('OpenCV:', cv2.__version__); print('CUDA:', cv2.cuda.getCudaEnabledDeviceCount())"
```

Expected output:
```
OpenCV: 4.8.0
CUDA: 1
```

## 🎯 Usage

### Basic Usage

```python
from template_matcher import TemplateMatcher, BoundingBox

# Load template and annotations
template_path = 'template.jpg'
template_bboxes = [...]  # Your bboxes

matcher = TemplateMatcher(template_path, template_bboxes)

# Match with CUDA acceleration (default)
bboxes, conf, img = matcher.match(
    target_path='target.jpg',
    method='feature',  # or 'orb'
    threshold=0.7
)
```

### Explicit CUDA Control

```python
import cv2

# Check CUDA availability
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print("✅ CUDA available")
    
    # Force CUDA
    result = matcher.match_feature_based(target_gray, use_cuda=True)
    # or
    result = matcher.match_orb(target_gray, use_cuda=True)
else:
    print("❌ CUDA not available, using CPU")
    result = matcher.match_feature_based(target_gray, use_cuda=False)
```

### Configuration

Edit `config.py` to set matching method:

```python
# Template matching method
MATCHING_METHOD = 'orb'  # 'feature' (SIFT) or 'orb' (faster)
```

## 📊 Benchmarking

Run the benchmark to compare CPU vs CUDA performance:

```bash
cd desktop
python benchmark_cuda_matching.py
```

Expected output:
```
⚡ CUDA vs CPU Template Matching Benchmark
==================================================

Method              CPU (ms)        CUDA (ms)       Speedup   
--------------------------------------------------
SIFT                   250.00          100.00       2.50x
ORB                    150.00           40.00       3.75x

✅ Fastest method: ORB CUDA (40.00ms)
```

## ⚡ Performance Optimization Tips

### 1. Use ORB for Maximum Speed
```python
MATCHING_METHOD = 'orb'  # 2-3x faster than SIFT
```

### 2. Reduce Image Resolution
```python
# Resize target image before matching
target_small = cv2.resize(target, None, fx=0.5, fy=0.5)
```

### 3. Use Crop Area
```python
# Define crop_area bbox to search in smaller region
# This reduces computation significantly
```

### 4. Tune ORB Parameters
```python
# In template_matcher.py, adjust nfeatures
orb = cv2.cuda.ORB_create(
    nfeatures=1000,  # Reduce for speed (default: 2000)
    scaleFactor=1.2,
    nlevels=6        # Reduce for speed (default: 8)
)
```

### 5. Enable Power Mode
```bash
# Set Jetson to maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks
```

## 🔍 Troubleshooting

### Issue: "CUDA device count: 0"
**Solution:**
```bash
# Check CUDA installation
nvcc --version

# Verify GPU is detected
nvidia-smi  # Or jetson_stats
```

### Issue: "cv2.cuda module not found"
**Solution:** OpenCV was not built with CUDA. Rebuild from source (see Option 2 above).

### Issue: "CUDA error during matching"
**Solution:** Fallback to CPU is automatic. Check error message:
```python
# The code automatically falls back to CPU on error
# Check console output for specific error message
```

### Issue: Slower than CPU
**Solution:**
1. First inference is slow (CUDA initialization). Use warm-up:
   ```python
   _ = matcher.match_feature_based(img, use_cuda=True)  # Warm-up
   ```
2. Small images may not benefit from CUDA. Use larger images (>1MP).
3. Ensure power mode is set correctly: `sudo jetson_clocks`

## 📈 Expected Performance

| Method | CPU (ms) | CUDA (ms) | Speedup | Use Case |
|--------|----------|-----------|---------|----------|
| **SIFT** | 200-300 | 80-120 | 2.5x | High accuracy needed |
| **ORB** | 100-150 | 30-50 | 3-4x | **Best for production** |
| Simple | 50-100 | N/A | N/A | No rotation/scale |
| Multi-scale | 200-400 | N/A | N/A | Scale variation |

## 🎨 Integration with Full Pipeline

Example: CUDA matching + TensorRT OCR

```python
from template_matcher import TemplateMatcher
from text_recognizer_tensorrt import TextRecognizerTensorRT

# Initialize (both use CUDA/TensorRT)
matcher = TemplateMatcher(template_path, template_bboxes)
recognizer = TextRecognizerTensorRT(engine_path, dict_path)

# Template matching with CUDA
bboxes, conf, img = matcher.match(target_path, method='orb')

# OCR with TensorRT
for bbox in bboxes:
    if bbox.bbox_type == 'text':
        cropped = matcher.crop_region_with_perspective(img, bbox)
        text, conf = recognizer.recognize(cropped)
        print(f"Text: {text}")
```

**Total pipeline time:** ~50-100ms (vs 300-500ms on CPU)

## 🔗 Additional Resources

- [OpenCV CUDA Documentation](https://docs.opencv.org/4.x/d1/d1a/namespacecv_1_1cuda.html)
- [Jetson Developer Guide](https://developer.nvidia.com/embedded/jetson-agx-orin-developer-kit)
- [JetPack SDK](https://developer.nvidia.com/embedded/jetpack)
- [OpenCV Build Guide](https://docs.opencv.org/4.x/d6/d15/tutorial_building_tegra_cuda.html)

## 📝 Notes

- CUDA matching requires CUDA-enabled OpenCV (not available via pip)
- Code automatically falls back to CPU if CUDA is not available
- First CUDA inference is slow (initialization overhead)
- Best speedup on images >1MP with many features
- ORB CUDA is faster than SIFT CUDA but slightly less accurate
