#!/bin/bash
# Build TensorRT engine with dynamic batch size (2-8)

set -e

ONNX_PATH="/home/demo/Source/ocr_datecode/weights/superpoint_lightglue_pipeline.onnx"
OUTPUT_PATH="/home/demo/Source/ocr_datecode/weights/pipeline_fp16_dynamic_480x640.engine"

echo "🔨 Building TensorRT engine with dynamic batch size (2-8)..."
echo "   ONNX: $ONNX_PATH"
echo "   Output: $OUTPUT_PATH"
echo ""

if [ ! -f "$ONNX_PATH" ]; then
    echo "❌ ONNX file not found: $ONNX_PATH"
    exit 1
fi

# Build engine with dynamic batch size
# min: 2 (1 template + 1 target)
# opt: 6 (3 templates + 3 targets) - optimized for this
# max: 8 (4 templates + 4 targets)
# /usr/src/tensorrt/bin/trtexec \
#     --onnx="$ONNX_PATH" \
#     --saveEngine="$OUTPUT_PATH" \
#     --fp16 \
#     --minShapes=images:2x1x1080x1920 \
#     --optShapes=images:6x1x1080x1920 \
#     --maxShapes=images:8x1x1080x1920 \
#     --memPoolSize=workspace:4096

/usr/src/tensorrt/bin/trtexec \
    --onnx="$ONNX_PATH" \
    --saveEngine="$OUTPUT_PATH" \
    --fp16 \
    --minShapes=images:2x1x480x640 \
    --optShapes=images:6x1x480x640 \
    --maxShapes=images:8x1x480x640 \
    --memPoolSize=workspace:4096

if [ -f "$OUTPUT_PATH" ]; then
    echo ""
    echo "✅ Engine built successfully!"
    echo "   Size: $(du -h $OUTPUT_PATH | cut -f1)"
    echo ""
    echo "Dynamic batch support:"
    echo "  - Min: 2 (1 camera)"
    echo "  - Opt: 6 (3 cameras) - best performance"
    echo "  - Max: 8 (4 cameras)"
    echo ""
    echo "Next steps:"
    echo "  1. Test engine: python test_batch_dynamic.py"
    echo "  2. Update inference_handler.py to use dynamic batch"
else
    echo ""
    echo "❌ Engine build failed!"
    exit 1
fi
