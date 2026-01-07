
#!/usr/bin/env python3
"""
Test OCR imports and initialization
Debug script to verify text recognizer availability
"""
import sys
from pathlib import Path

# Add camera_management to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("OCR Import Test")
print("="*80)

# Test 1: TensorRT availability
print("\n[1] Testing TensorRT availability...")
try:
    import tensorrt as trt
    print(f"✅ TensorRT imported: version {trt.__version__}")
    TENSORRT_OK = True
except ImportError as e:
    print(f"❌ TensorRT import failed: {e}")
    TENSORRT_OK = False

# Test 2: PyCUDA availability
print("\n[2] Testing PyCUDA availability...")
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    print(f"✅ PyCUDA imported")
    PYCUDA_OK = True
except ImportError as e:
    print(f"❌ PyCUDA import failed: {e}")
    PYCUDA_OK = False

# Test 3: Check files
print("\n[3] Checking OCR model files...")
ocr_engine_path = Path("/home/demo/Source/ocr_datecode/languages/english/rec.engine")
ocr_dict_path = Path("/home/demo/Source/ocr_datecode/languages/english/dict.txt")

if ocr_engine_path.exists():
    print(f"✅ Engine file exists: {ocr_engine_path}")
    print(f"   Size: {ocr_engine_path.stat().st_size / 1024 / 1024:.2f} MB")
else:
    print(f"❌ Engine file NOT found: {ocr_engine_path}")

if ocr_dict_path.exists():
    print(f"✅ Dict file exists: {ocr_dict_path}")
    with open(ocr_dict_path, 'r') as f:
        chars = f.readlines()
    print(f"   Characters: {len(chars)}")
else:
    print(f"❌ Dict file NOT found: {ocr_dict_path}")

# Test 4: Import text_recognizer_trt
print("\n[4] Testing text_recognizer_trt import...")
try:
    from camera_management.text_recognizer_trt import (
        get_text_recognizer,
        TENSORRT_AVAILABLE,
        TextRecognizerTRT
    )
    print(f"✅ text_recognizer_trt imported")
    print(f"   TENSORRT_AVAILABLE: {TENSORRT_AVAILABLE}")
except ImportError as e:
    print(f"❌ text_recognizer_trt import failed: {e}")
    import traceback
    traceback.print_exc()
    TENSORRT_AVAILABLE = False

# Test 5: Import ocr_utils
print("\n[5] Testing ocr_utils import...")
try:
    from camera_management.ocr_utils import (
        crop_text_region,
        compare_texts,
        preprocess_text_image,
        decode_ctc_greedy
    )
    print(f"✅ ocr_utils imported")
except ImportError as e:
    print(f"❌ ocr_utils import failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Try to initialize recognizer
print("\n[6] Testing TextRecognizerTRT initialization...")
if TENSORRT_AVAILABLE and TENSORRT_OK and PYCUDA_OK:
    if ocr_engine_path.exists() and ocr_dict_path.exists():
        try:
            recognizer = get_text_recognizer(
                engine_path=str(ocr_engine_path),
                dict_path=str(ocr_dict_path),
                min_width=320,
                max_width=2000
            )
            print(f"✅ TextRecognizerTRT initialized successfully")
            print(f"   Recognizer: {recognizer}")

            # Test recognition with dummy image
            import numpy as np
            import cv2

            # Create simple test image with text
            test_img = np.ones((60, 400, 3), dtype=np.uint8) * 255
            cv2.putText(test_img, "TEST", (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

            print("\n   Running test recognition...")
            result = recognizer.recognize(test_img, return_confidence=True)
            print(f"   ✅ Recognition result: '{result['text']}' (conf: {result['confidence']:.3f})")

        except Exception as e:
            print(f"❌ TextRecognizerTRT initialization failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️  Cannot initialize: Missing engine or dict files")
else:
    print(f"⚠️  Cannot initialize: TensorRT/PyCUDA not available")
    print(f"   TENSORRT_AVAILABLE: {TENSORRT_AVAILABLE}")
    print(f"   TENSORRT_OK: {TENSORRT_OK}")
    print(f"   PYCUDA_OK: {PYCUDA_OK}")

# Test 7: Check inference_handler integration
print("\n[7] Testing InferenceHandler integration...")
try:
    from camera_management.inference_handler import InferenceHandler, OCR_AVAILABLE
    print(f"✅ InferenceHandler imported")
    print(f"   OCR_AVAILABLE in inference_handler: {OCR_AVAILABLE}")

    # Try to create InferenceHandler
    handler = InferenceHandler()
    print(f"✅ InferenceHandler created")
    print(f"   text_recognizer: {handler.text_recognizer}")

except Exception as e:
    print(f"❌ InferenceHandler test failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"TensorRT:        {'✅' if TENSORRT_OK else '❌'}")
print(f"PyCUDA:          {'✅' if PYCUDA_OK else '❌'}")
print(f"Engine file:     {'✅' if ocr_engine_path.exists() else '❌'}")
print(f"Dict file:       {'✅' if ocr_dict_path.exists() else '❌'}")
print(f"OCR Available:   {'✅' if TENSORRT_AVAILABLE and TENSORRT_OK and PYCUDA_OK else '❌'}")
print("="*80)

if not (TENSORRT_OK and PYCUDA_OK):
    print("\n⚠️  ACTION REQUIRED:")
    print("   Install TensorRT and PyCUDA:")
    print("   pip install tensorrt pycuda")
    print()
elif not ocr_engine_path.exists():
    print("\n⚠️  ACTION REQUIRED:")
    print(f"   OCR engine file missing: {ocr_engine_path}")
    print("   Please build the TensorRT engine from ONNX model")
    print()
elif not ocr_dict_path.exists():
    print("\n⚠️  ACTION REQUIRED:")
    print(f"   Dictionary file missing: {ocr_dict_path}")
    print()
else:
    print("\n✅ All checks passed! OCR should be working.")
    print()
