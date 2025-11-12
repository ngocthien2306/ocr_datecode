from paddleocr import PaddleOCR
import cv2
import time

# Initialize model
print("Initializing model...")
init_start = time.time()
ocr = PaddleOCR(
    ocr_version='PP-OCRv4',
    use_angle_cls=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    lang='en'
)
print(f"Init time: {time.time() - init_start:.3f}s\n")

# Warm-up
print("Warm-up...")
_ = ocr.predict("ocr.jpg")

# Measure inference speed (average of 3 runs)
print("\nMeasuring inference speed...")
times = []
for i in range(3):
    start = time.time()
    result = ocr.predict("ocr.jpg")
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"Run {i+1}: {elapsed:.3f}s")

avg_time = sum(times) / len(times)
fps = 1 / avg_time

print(f"\n{'='*40}")
print(f"Average inference time: {avg_time:.3f}s")
print(f"FPS: {fps:.2f}")
print(f"{'='*40}\n")

# Visualize results (using last result)
img = cv2.imread("ocr.jpg")
for page in result:
    for text, box in zip(page['rec_texts'], page['dt_polys']):
        print(text)
        pts = box.reshape((-1, 1, 2)).astype(int)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        cv2.putText(img, text, tuple(box[0]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

cv2.imwrite("result.jpg", img)
print("\n✅ Saved: result.jpg")