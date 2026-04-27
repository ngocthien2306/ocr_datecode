"""
Configuration file for OCR backend selection
"""

# ============================================================
# OCR BACKEND CONFIGURATION
# ============================================================

# Choose OCR backend: 'tensorrt', 'onnx_gpu', 'onnx_cpu', 'tesseract', 'auto'
OCR_BACKEND = 'auto'  # 'auto' will try TensorRT first, fallback to ONNX

# Model paths
MODEL_CONFIG = {
    'tensorrt': {
        'engine_path': '../languages/english/rec.engine',
        'dict_path': '../languages/english/dict.txt',
        'min_width': 320,  # Must match minShapes in TensorRT engine
        'max_width': 2000,  # Must match maxShapes in TensorRT engine
    },
    'onnx': {
        'model_path': '../languages/english/rec.onnx',
        'dict_path': '../languages/english/dict.txt',
        'use_gpu': False,  # Set to True for ONNX GPU
    },
    'tesseract': {
        'lang': 'eng',
        'psm': 7,            # 7 = single line (typical for LOT/datecode crops)
        'oem': 1,            # 1 = LSTM only
        'char_whitelist': '',  # e.g. '0123456789LOTPLACK ' to restrict chars
        # 'tessdata_path': '/opt/homebrew/share/tessdata',  # uncomment to override
        'library': 'auto',   # 'auto' | 'tesserocr' | 'pytesseract'
    },
    'easyocr': {
        'langs': ('en',),    # tuple of language codes, e.g. ('en', 'vi')
        'gpu': False,        # set True if CUDA is available
        'allowlist': None,   # e.g. '0123456789LOTPLACK ' to restrict chars
        # 'model_storage_directory': '~/.EasyOCR',  # default
    },
    'rapidocr': {
        # Our crops are already single-region text → skip det/cls, recognize only
        'use_det': False,
        'use_cls': False,
        'use_rec': True,
        # 'config_path': '/path/to/custom_config.yaml',  # optional
    },
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

    try:
        from text_recognizer_tesseract import TextRecognizerTesseract
        tesseract_available = True
    except ImportError:
        tesseract_available = False

    try:
        from text_recognizer_easyocr import TextRecognizerEasyOCR
        easyocr_available = True
    except ImportError:
        easyocr_available = False

    try:
        from text_recognizer_rapidocr import TextRecognizerRapidOCR
        rapidocr_available = True
    except ImportError:
        rapidocr_available = False

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

    elif OCR_BACKEND in ('tesseract', 'tesseract_pytesseract', 'tesseract_tesserocr'):
        if not tesseract_available:
            raise ImportError(
                "Tesseract recognizer module not available. "
                "Install one of: `pip install tesserocr` (faster) or "
                "`pip install pytesseract Pillow` (simpler)."
            )
        cfg = dict(MODEL_CONFIG['tesseract'])
        if OCR_BACKEND == 'tesseract_pytesseract':
            cfg['library'] = 'pytesseract'
        elif OCR_BACKEND == 'tesseract_tesserocr':
            cfg['library'] = 'tesserocr'
        rec = TextRecognizerTesseract(**cfg)
        return rec, f'Tesseract ({rec.library})'

    elif OCR_BACKEND in ('easyocr', 'easyocr_gpu', 'easyocr_cpu'):
        if not easyocr_available:
            raise ImportError(
                "easyocr not installed. `pip install easyocr` "
                "(downloads ~64MB of model weights on first run)."
            )
        cfg = dict(MODEL_CONFIG['easyocr'])
        if OCR_BACKEND == 'easyocr_gpu':
            cfg['gpu'] = True
        elif OCR_BACKEND == 'easyocr_cpu':
            cfg['gpu'] = False
        rec = TextRecognizerEasyOCR(**cfg)
        backend_name = 'EasyOCR (GPU)' if cfg.get('gpu') else 'EasyOCR (CPU)'
        return rec, backend_name

    elif OCR_BACKEND == 'rapidocr':
        if not rapidocr_available:
            raise ImportError(
                "rapidocr_onnxruntime not installed. "
                "`pip install rapidocr_onnxruntime`"
            )
        rec = TextRecognizerRapidOCR(**MODEL_CONFIG['rapidocr'])
        return rec, 'RapidOCR'

    elif OCR_BACKEND == 'auto':
        # Try TensorRT first, fallback to ONNX
        if tensorrt_available:
            try:
                return TextRecognizerTensorRT(**MODEL_CONFIG['tensorrt']), 'TensorRT (auto)'
            except Exception as e:
                print(f"⚠️  TensorRT failed: {e}")
                print("   Falling back to ONNX Runtime...")
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
