# Quick Start Guide

Hướng dẫn nhanh để bắt đầu với OCR Datecode trong 5 phút.

## 🚀 Setup nhanh (5 phút)

### 1. Clone repo
```bash
git clone https://github.com/ngocthien2306/ocr_datecode.git
cd ocr_datecode
```

### 2. Cài đặt dependencies
```bash
# Minimum setup
pip install opencv-python numpy onnxruntime PyQt5

# Full setup (recommended)
pip install opencv-python opencv-contrib-python numpy onnxruntime PyQt5
```

### 3. Chuẩn bị ảnh
```bash
# Copy ảnh template và target vào thư mục images/
cp /path/to/your/template.jpg images/template.jpg
cp /path/to/your/target.jpg images/1.jpg
```

---

## 📝 Workflow cơ bản

### Bước 1: Tạo Template Annotations

```bash
cd desktop
python main.py
```

1. Click **"Select Folder"** → chọn folder `../images`
2. Click vào `template.jpg` trong danh sách
3. Click **"Draw BBox"** → vẽ vùng template (rectangle)
4. Chọn type **"Template"** → OK
5. Vẽ thêm các vùng **"Text"** hoặc **"Datecode"**
6. Close app (auto-save vào `images/annotations.json`)

### Bước 2: Test Template Matching + OCR

```bash
cd desktop
python test_crop_perspective.py
```

**Kết quả:**
- Template matching confidence
- Các bbox đã transform
- OCR text với confidence scores
- Ảnh crop + kết quả visualization

---

## 🎯 Ví dụ Code Nhanh

### Ví dụ 1: OCR một vùng text đơn giản

```python
from text_recognizer import TextRecognizer
import cv2

# Init
recognizer = TextRecognizer(
    'languages/english/rec.onnx',
    'languages/english/dict.txt'
)

# OCR
img = cv2.imread('cropped_text.jpg')
text, conf = recognizer.recognize(img)
print(f"'{text}' (confidence: {conf:.3f})")
```

### Ví dụ 2: Template matching đơn giản

```python
from template_matcher import TemplateMatcher, BoundingBox
import json

# Load annotations
with open('images/annotations.json') as f:
    data = json.load(f)

template_path = data['_template_image']
bboxes = [BoundingBox.from_dict(b) for b in data[template_path]]

# Match
matcher = TemplateMatcher(template_path, bboxes)
result, conf, img = matcher.match('images/1.jpg', method='auto')

print(f"Matched: {conf:.3f}")
print(f"Found {len(result)} bboxes")
```

### Ví dụ 3: Full pipeline

```python
from template_matcher import TemplateMatcher, BoundingBox
from text_recognizer import TextRecognizer
import json

# 1. Load template
with open('images/annotations.json') as f:
    data = json.load(f)
template_path = data['_template_image']
template_bboxes = [BoundingBox.from_dict(b) for b in data[template_path]]

# 2. Template matching
matcher = TemplateMatcher(template_path, template_bboxes)
bboxes, conf, img = matcher.match('images/1.jpg')

# 3. Crop text regions
text_crops = []
for bbox in bboxes:
    if bbox.bbox_type == 'text':
        cropped = matcher.crop_region_with_perspective(img, bbox)
        text_crops.append(cropped)

# 4. Batch OCR
recognizer = TextRecognizer('languages/english/rec.onnx',
                           'languages/english/dict.txt')
results = recognizer.recognize_batch(text_crops)

for text, conf in results:
    print(f"📝 {text} ({conf:.3f})")
```

---

## 🔧 Troubleshooting

### Import Error: No module named 'cv2'
```bash
pip install opencv-python
```

### ONNX Runtime Error
```bash
pip install onnxruntime
# Hoặc GPU version:
pip install onnxruntime-gpu
```

### Template not found (confidence < 0.7)
- Thử giảm threshold: `threshold=0.3`
- Thử method khác: `method='feature'`
- Kiểm tra template bbox có đúng không

### OCR kết quả sai
- Kiểm tra ảnh crop
- Tăng chất lượng ảnh gốc
- Thử preprocess (denoise, contrast)

---

## 📚 Tài liệu thêm

- [Full README](../README.md)
- [Template Matching Design](desktop/docs/TEMPLATE_MATCHING_DESIGN.md)
- [API Documentation](../README.md#-hướng-dẫn-sử-dụng)

---

## 💡 Tips

✅ Dùng `crop_area` bbox để tăng tốc matching  
✅ Dùng `auto` method cho kết quả tốt nhất  
✅ Batch processing nhanh hơn single processing  
✅ GPU mode (nếu có) nhanh hơn CPU 3-5x  
✅ Polygon mode cho ảnh bị perspective/méo

---

**Ready to go! 🚀**
