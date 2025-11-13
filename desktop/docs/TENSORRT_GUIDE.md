# TensorRT Integration Guide

## 🚀 Overview

This guide explains how to use TensorRT for high-performance text recognition in the OCR pipeline.

## 📋 Prerequisites

### 1. Install TensorRT

**Ubuntu/Linux:**
```bash
# Download TensorRT from NVIDIA website
# https://developer.nvidia.com/tensorrt

# Install dependencies
pip install nvidia-tensorrt
pip install pycuda
```

**Docker (Recommended):**
```bash
docker pull nvcr.io/nvidia/tensorrt:23.08-py3
docker run --gpus all -it -v $(pwd):/workspace nvcr.io/nvidia/tensorrt:23.08-py3
```

### 2. Verify Installation

```bash
python -c "import tensorrt as trt; print(trt.__version__)"
python -c "import pycuda.driver as cuda; import pycuda.autoinit; print('PyCUDA OK')"
```

## 🔧 Convert ONNX to TensorRT Engine

### Step 1: Navigate to model directory
```bash
cd languages/english/
```

### Step 2: Convert using trtexec
```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=rec.onnx \
    --saveEngine=rec.engine \
    --fp16 \
    --workspace=4096 \
    --minShapes=x:1x3x48x320 \
    --optShapes=x:1x3x48x320 \
    --maxShapes=x:1x3x48x2000 \
    --verbose
```

**Parameters explained:**
- `--onnx`: Input ONNX model path
- `--saveEngine`: Output TensorRT engine path
- `--fp16`: Enable FP16 precision (faster, slightly less accurate)
- `--workspace`: GPU memory workspace in MB
- `--minShapes`: Minimum input shape (batch, channels, height, width)
- `--optShapes`: Optimal input shape (for performance tuning)
- `--maxShapes`: Maximum input shape (must cover all expected inputs)

### Step 3: Verify engine creation
```bash
ls -lh rec.engine
# Should show the engine file (typically 5-20MB)
```

## 📝 Usage

### Basic Usage

```python
from text_recognizer_tensorrt import TextRecognizerTensorRT
import cv2

# Initialize TensorRT recognizer
recognizer = TextRecognizerTensorRT(
    engine_path='../languages/english/rec.engine',
    dict_path='../languages/english/dict.txt'
)

# Recognize text from cropped image
image = cv2.imread('cropped_text.jpg')
text, confidence = recognizer.recognize(image)

print(f"Text: {text}")
print(f"Confidence: {confidence:.3f}")
```

### Batch Processing

```python
# Recognize multiple images
images = [cv2.imread(f'crop_{i}.jpg') for i in range(5)]
results = recognizer.recognize_batch(images)

for i, (text, conf) in enumerate(results):
    print(f"Image {i}: {text} (conf: {conf:.3f})")
```

### Using in test_crop_perspective.py

Edit the configuration at the top of the file:

```python
# ============================================================
# CONFIGURATION
# ============================================================
USE_TENSORRT = True   # Set to False to use ONNX Runtime
SHOW_TIMING = True    # Set to False to hide timing details
SHOW_DEBUG = False    # Set to True to show homography analysis
```

Then run:
```bash
python test_crop_perspective.py
```

## ⚡ Performance Comparison

| Backend | Avg Inference Time | Throughput |
|---------|-------------------|------------|
| **ONNX Runtime (CPU)** | ~50-100ms | ~10-20 FPS |
| **ONNX Runtime (GPU)** | ~10-20ms | ~50-100 FPS |
| **TensorRT (GPU, FP32)** | ~3-8ms | ~125-300 FPS |
| **TensorRT (GPU, FP16)** | ~2-5ms | ~200-500 FPS |

*Performance varies based on GPU, image size, and batch size*

## 🔍 Troubleshooting

### Issue: "TensorRT not available"
**Solution:** Make sure TensorRT and PyCUDA are installed:
```bash
pip install nvidia-tensorrt pycuda
```

### Issue: "CUDA_ERROR_NO_DEVICE"
**Solution:** Ensure you have a NVIDIA GPU and CUDA drivers installed:
```bash
nvidia-smi  # Should show GPU info
```

### Issue: "Engine file not found"
**Solution:** Convert ONNX to engine first (see Step 2 above)

### Issue: "Shape mismatch error"
**Solution:** Ensure input shapes are within min/max bounds:
- Minimum: 1x3x48x320
- Maximum: 1x3x48x2000

### Issue: "Out of memory"
**Solution:** Reduce `--workspace` value or use smaller batch sizes

## 📊 Dynamic Shapes

The TensorRT engine supports dynamic input widths:
- **Min width**: 320 pixels
- **Optimal width**: 320 pixels  
- **Max width**: 2000 pixels

The recognizer automatically handles different image widths within this range.

## 🎯 Best Practices

1. **Warm-up inference**: Run 1-2 inferences before benchmarking
   ```python
   _ = recognizer.recognize(sample_image)  # Warm-up
   ```

2. **Batch processing**: Process multiple images together for better throughput
   ```python
   results = recognizer.recognize_batch(images)  # Better than loop
   ```

3. **Reuse engine**: Initialize recognizer once, use many times
   ```python
   recognizer = TextRecognizerTensorRT(...)  # Once
   for img in images:
       text, conf = recognizer.recognize(img)  # Many times
   ```

4. **FP16 vs FP32**: Use FP16 for 2x speed with minimal accuracy loss

5. **Profile first**: Use `--verbose` during conversion to see performance metrics

## 🔄 Switching Between Backends

The code automatically falls back to ONNX Runtime if TensorRT is not available:

```python
# Automatic fallback
try:
    recognizer = TextRecognizerTensorRT(...)
except:
    recognizer = TextRecognizer(...)  # ONNX fallback
```

## 📚 Additional Resources

- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/)
- [ONNX to TensorRT Guide](https://github.com/onnx/onnx-tensorrt)

## 🐛 Debug Mode

Enable verbose logging for debugging:

```python
import tensorrt as trt

logger = trt.Logger(trt.Logger.VERBOSE)
runtime = trt.Runtime(logger)
```

This will show detailed engine building and inference logs.
