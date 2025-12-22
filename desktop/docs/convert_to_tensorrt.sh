#!/bin/bash

# Script to convert ONNX model to TensorRT engine
# Usage: ./convert_to_tensorrt.sh [model_dir]

set -e  # Exit on error

# Default model directory
MODEL_DIR="${1:-../languages/english}"

echo "=========================================="
echo "🚀 ONNX to TensorRT Converter"
echo "=========================================="
echo ""

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Error: Model directory not found: $MODEL_DIR"
    exit 1
fi

# Check if ONNX model exists
ONNX_MODEL="$MODEL_DIR/rec.onnx"
if [ ! -f "$ONNX_MODEL" ]; then
    echo "❌ Error: ONNX model not found: $ONNX_MODEL"
    exit 1
fi

# Output engine path
ENGINE_MODEL="$MODEL_DIR/rec.engine"

echo "📂 Model directory: $MODEL_DIR"
echo "📥 Input ONNX: $ONNX_MODEL"
echo "📤 Output Engine: $ENGINE_MODEL"
echo ""

# Check if trtexec is available
if ! command -v trtexec &> /dev/null; then
    echo "❌ Error: trtexec not found"
    echo "   Please install TensorRT or use Docker:"
    echo "   docker pull nvcr.io/nvidia/tensorrt:23.08-py3"
    exit 1
fi

# Check CUDA/GPU availability
if ! nvidia-smi &> /dev/null; then
    echo "⚠️  Warning: nvidia-smi not found. Make sure you have NVIDIA GPU and drivers installed."
fi

echo "🔧 Converting ONNX to TensorRT engine..."
echo "   This may take a few minutes..."
echo ""

# Run trtexec
trtexec \
    --onnx="$ONNX_MODEL" \
    --saveEngine="$ENGINE_MODEL" \
    --fp16 \
    --workspace=4096 \
    --minShapes=x:1x3x48x320 \
    --optShapes=x:1x3x48x320 \
    --maxShapes=x:1x3x48x2000 \
    --verbose

# Check if conversion succeeded
if [ -f "$ENGINE_MODEL" ]; then
    ENGINE_SIZE=$(du -h "$ENGINE_MODEL" | cut -f1)
    echo ""
    echo "=========================================="
    echo "✅ Conversion successful!"
    echo "=========================================="
    echo "📦 Engine file: $ENGINE_MODEL"
    echo "📊 Engine size: $ENGINE_SIZE"
    echo ""
    echo "🎯 Next steps:"
    echo "   1. Test the engine:"
    echo "      python test_crop_perspective.py"
    echo ""
    echo "   2. Benchmark performance:"
    echo "      python benchmark_ocr.py"
    echo ""
else
    echo ""
    echo "❌ Conversion failed!"
    echo "   Check the error messages above for details."
    exit 1
fi


# /usr/src/tensorrt/bin/trtexec --onnx=rec.onnx --saveEngine=rec.engine --fp16 --workspace=4096 --minShapes=x:1x3x48x50 --optShapes=x:1x3x48x320 --maxShapes=x:1x3x48x2000 --verbose