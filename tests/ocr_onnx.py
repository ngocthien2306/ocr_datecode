from rapidocr_onnxruntime import RapidOCR
import cv2
import numpy as np
import time

# OCR with CUDA provider
ocr = RapidOCR(
    det_model_path="detection/v5/det.onnx",
    rec_model_path="languages/english/rec.onnx",
    rec_keys_path="languages/english/dict.txt"
)
image_path = "/Users/ngocthien.ai/Source/Projects/ocr_datecode/data/Code trên nắp chai/Image__2025-12-13__15-31-43.bmp"
start = time.time()
result, elapsed = ocr(image_path)
total_time = (time.time() - start) * 1000

print(f"Detection: {elapsed[0]:.2f} ms")
print(f"Recognition: {elapsed[1]:.2f} ms") 
print(f"Total: {total_time:.2f} ms")
print(f"Found {len(result)} texts")

# Visualize
img = cv2.imread(image_path)
for box, text, score in result:
    pts = np.array(box, dtype=np.int32)
    cv2.polylines(img, [pts], True, (0, 255, 0), 2)
    cv2.putText(img, f"{text}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

cv2.imwrite("ocr_result.jpg", img)
print("Saved!")