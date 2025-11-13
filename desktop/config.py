"""
Configuration file for OCR backend selection
"""

# ============================================================
# OCR BACKEND CONFIGURATION
# ============================================================

# Choose OCR backend: 'tensorrt', 'onnx_gpu', 'onnx_cpu', 'auto'
OCR_BACKEND = 'auto'  # 'auto' will try TensorRT first, fallback to ONNX

# Model paths
MODEL_CONFIG = {
    'tensorrt': {
        'engine_path': '../languages/english/rec.engine',
        'dict_path': '../languages/english/dict.txt',
    },
    'onnx': {
        'model_path': '../languages/english/rec.onnx',
        'dict_path': '../languages/english/dict.txt',
        'use_gpu': False,  # Set to True for ONNX GPU
    }
}

# ============================================================
# TEMPLATE MATCHING CONFIGURATION
# ============================================================

# Template matching method: 'auto', 'simple', 'multi_scale', 'feature', 'orb', 'superpoint'
MATCHING_METHOD = 'feature'

# Template matching threshold (0.0 - 1.0)
MATCHING_THRESHOLD = 0.3

# Enable homography analysis debug output
MATCHING_DEBUG = False

# ============================================================
# PERFORMANCE CONFIGURATION
# ============================================================

# Show timing information
SHOW_TIMING = True

# Batch size for OCR (TensorRT processes one-by-one due to dynamic width)
BATCH_SIZE = 10

# ============================================================
# OUTPUT CONFIGURATION
# ============================================================

# Output directory for results
OUTPUT_DIR = '../results'

# Save cropped regions
SAVE_CROPS = True

# Save visualization with bboxes
SAVE_VISUALIZATION = True

# Draw polygon shapes (vs rectangles only)
DRAW_POLYGONS = True

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_recognizer():
    """Get recognizer instance based on configuration"""
    from text_recognizer import TextRecognizer
    
    try:
        from text_recognizer_tensorrt import TextRecognizerTensorRT
        tensorrt_available = True
    except ImportError:
        tensorrt_available = False
    
    if OCR_BACKEND == 'tensorrt':
        if not tensorrt_available:
            raise ImportError("TensorRT not available. Install with: pip install nvidia-tensorrt pycuda")
        return TextRecognizerTensorRT(**MODEL_CONFIG['tensorrt']), 'TensorRT'
    
    elif OCR_BACKEND == 'onnx_gpu':
        config = MODEL_CONFIG['onnx'].copy()
        config['use_gpu'] = True
        return TextRecognizer(**config), 'ONNX Runtime (GPU)'
    
    elif OCR_BACKEND == 'onnx_cpu':
        config = MODEL_CONFIG['onnx'].copy()
        config['use_gpu'] = False
        return TextRecognizer(**config), 'ONNX Runtime (CPU)'
    
    elif OCR_BACKEND == 'auto':
        # Try TensorRT first, fallback to ONNX
        if tensorrt_available:
            try:
                return TextRecognizerTensorRT(**MODEL_CONFIG['tensorrt']), 'TensorRT (auto)'
            except Exception as e:
                print(f"⚠️  TensorRT failed: {e}")
                print("   Falling back to ONNX Runtime...")
        
        # Fallback to ONNX
        return TextRecognizer(**MODEL_CONFIG['onnx']), 'ONNX Runtime (auto)'
    
    else:
        raise ValueError(f"Unknown OCR backend: {OCR_BACKEND}")


def print_config():
    """Print current configuration"""
    print("="*60)
    print("⚙️  OCR CONFIGURATION")
    print("="*60)
    print(f"OCR Backend: {OCR_BACKEND}")
    print(f"Matching Method: {MATCHING_METHOD}")
    print(f"Matching Threshold: {MATCHING_THRESHOLD}")
    print(f"Show Timing: {SHOW_TIMING}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("="*60)


if __name__ == '__main__':
    print_config()
    
    # Test recognizer initialization
    print("\n🔧 Testing recognizer initialization...")
    try:
        recognizer, backend_name = get_recognizer()
        print(f"✅ Successfully initialized: {backend_name}")
    except Exception as e:
        print(f"❌ Failed to initialize recognizer: {e}")
