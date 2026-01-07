#!/usr/bin/env python3
"""Quick OCR test with image"""
import cv2
from pathlib import Path

# Load image - thay đổi path này
img_path = "/home/demo/Source/ocr_datecode/ai_services/test_result/cropped_region_40733814_1.png"  # <-- Thay path của bạn ở đây
img = cv2.imread(img_path)

if img is None:
    print(f"❌ Cannot load image: {img_path}")
    # Create dummy image
    import numpy as np
    img = np.ones((60, 400, 3), dtype=np.uint8) * 255
    cv2.putText(img, "TEST123", (100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)
    print("✅ Created dummy image")
else:
    print(f"✅ Loaded image: {img.shape}")

# Initialize recognizer
from camera_management.text_recognizer_trt import get_text_recognizer

recognizer = get_text_recognizer(
    engine_path="/home/demo/Source/ocr_datecode/languages/english/rec.engine",
    dict_path="/home/demo/Source/ocr_datecode/languages/english/dict.txt"
)

# Recognize
text, confidence = recognizer.recognize(img)
print(f"\nResult: {text} (Confidence: {confidence:.3f})")
