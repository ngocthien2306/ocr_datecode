# OCR Datecode - Template-Based Text Recognition System
ssh -N -L 5173:localhost:5173 -L 8000:localhost:8000 -L 8001:localhost:8001 msi@100.116.14.127

❯ sau đó bạn hãy check xem logic mới khác với logic cũ như nào, lúc đầu tôi dùng yolo để segment sau đó dùng min/max để loại trừ các vùng, vậy cái này có làm được không? dùng predict mask để tính toán?

🔍 **Hệ thống OCR tự động sử dụng Template Matching + Deep Learning Recognition**

Công cụ này giúp tự động nhận diện và đọc text/datecode/barcode từ ảnh sản phẩm bằng cách:
1. Dùng **Template Matching** để tìm vị trí các vùng quan tâm (text, datecode, barcode)
2. Crop và perspective correction các vùng đó
3. Nhận diện text bằng **PP-OCRv5 ONNX model** (chỉ Recognition, không Detection)

---

## 📋 Tính năng chính

### 🎨 **Annotation Tool (PyQt5 Desktop App)**
- ✅ Vẽ bounding box (rectangle hoặc **polygon 4 điểm**)
- ✅ Phân loại bbox: `template`, `text`, `datecode`, `barcode`, `crop_area`
- ✅ Hỗ trợ polygon cho ảnh bị perspective/méo
- ✅ Auto-save annotations vào JSON
- ✅ UI trực quan với color-coding

### 🔍 **Template Matching**
- ✅ Nhiều phương pháp: `simple`, `multi_scale`, `feature` (SIFT), `superpoint` (Deep Learning)
- ✅ Auto mode: tự chọn method tốt nhất
- ✅ Hỗ trợ **crop_area** để tăng tốc matching
- ✅ **Smart detection**: tự động chọn rectangle/polygon dựa vào homography analysis
- ✅ Perspective correction cho polygon

### 📝 **Text Recognition**
- ✅ **Lightweight OCR** (chỉ Recognition, bỏ qua Detection)
- ✅ PP-OCRv5 English model (ONNX)
- ✅ **Batch processing** cho nhiều vùng text
- ✅ Confidence score cho mỗi kết quả
- ✅ Hỗ trợ CPU/GPU (CUDA)

---

## 🗂️ Cấu trúc dự án

```
ocr_datecode/
├── desktop/                          # Desktop annotation tool
│   ├── main.py                      # Entry point
│   ├── main_window.py               # Main UI
│   ├── image_viewer.py              # Image viewer với polygon support
│   ├── annotation_manager.py        # Quản lý annotations
│   ├── template_matcher.py          # Template matching engine ⭐
│   ├── text_recognizer.py           # Lightweight OCR engine ⭐
│   ├── test_crop_perspective.py     # Demo script ⭐
│   └── docs/                        # Documentation
│       ├── TEMPLATE_MATCHING_DESIGN.md
│       └── SUPERPOINT_README.md
├── detection/                        # Detection models (cho RapidOCR)
│   └── v3/det.onnx
├── languages/                        # Recognition models
│   └── english/
│       ├── rec.onnx                 # PP-OCRv5 recognition model ⭐
│       ├── dict.txt                 # Character dictionary (436 chars)
│       └── config.json              # Model metadata
├── images/                           # Ảnh và annotations
│   ├── annotations.json             # Annotations file
│   └── *.jpg
└── results/                          # Output results
```

---

## 🚀 Cài đặt

### 1. Yêu cầu hệ thống

- Python 3.8+
- OpenCV
- PyQt5 (cho annotation tool)
- ONNX Runtime

### 2. Cài đặt dependencies

```bash
# Core dependencies
pip install opencv-python opencv-contrib-python numpy

# ONNX Runtime (CPU)
pip install onnxruntime

# ONNX Runtime (GPU - nếu có CUDA)
pip install onnxruntime-gpu

# PyQt5 (cho annotation tool)
pip install PyQt5

# Optional: SuperPoint + LightGlue (deep learning matching)
pip install kornia
pip install git+https://github.com/cvg/LightGlue.git

# Optional: RapidOCR (nếu muốn dùng full pipeline)
pip install rapidocr-onnxruntime
```

---

## 📖 Hướng dẫn sử dụng

### **Bước 1: Annotation - Tạo Template**

Chạy annotation tool để đánh dấu các vùng trên ảnh template:

```bash
cd desktop
python main.py
```

**Các loại bbox:**
- 🔴 **template**: Vùng để template matching (bắt buộc có 1 cái)
- 🟢 **text**: Vùng text cần OCR
- 🔵 **datecode**: Vùng datecode
- 🟡 **barcode**: Vùng barcode
- 🟣 **crop_area**: Vùng crop để tăng tốc matching (optional)

**Lưu ý:**
- Chọn 1 ảnh làm template (ảnh chuẩn)
- Vẽ bbox `template` để đánh dấu vùng template matching
- Vẽ các bbox khác (`text`, `datecode`) cho các vùng cần OCR
- Annotations sẽ lưu vào `images/annotations.json`

**Polygon mode (cho ảnh bị méo/perspective):**
- Click vào các góc để vẽ polygon 4 điểm
- Hữu ích cho ảnh chụp nghiêng, bị perspective

---

### **Bước 2: Template Matching + OCR**

Chạy script demo để test:

```bash
cd desktop
python test_crop_perspective.py
```

**Script này sẽ:**
1. Load annotations từ JSON
2. Template matching trên ảnh target (`images/1.jpg`)
3. Crop các vùng text/datecode với perspective correction
4. **Batch OCR** tất cả vùng text
5. Hiển thị kết quả + timing

**Output mẫu:**

```
============================================================
🚀 BATCH OCR PROCESSING (3 regions)
============================================================

⏱️  Total batch time: 45.23ms
⚡ Average per region: 15.08ms

BBox 1 (text):
  📝 Text: 'PRODUCT NAME'
  🎯 Confidence: 0.952

BBox 3 (datecode):
  📝 Text: '2025-11-13'
  🎯 Confidence: 0.887
```

---

### **Bước 3: Sử dụng trong code của bạn**

#### **3.1. Template Matching**

```python
from template_matcher import TemplateMatcher, BoundingBox
import json

# Load annotations
with open('images/annotations.json', 'r') as f:
    data = json.load(f)

template_path = data['_template_image']
template_bboxes = [BoundingBox.from_dict(b) for b in data[template_path]]

# Initialize matcher
matcher = TemplateMatcher(template_path, template_bboxes)

# Match target image
bboxes, confidence, target_img = matcher.match(
    'images/target.jpg',
    method='auto',      # auto | simple | multi_scale | feature | superpoint
    threshold=0.7,
    debug=True
)

if bboxes:
    print(f"✅ Matched! Confidence: {confidence:.3f}")
    print(f"Found {len(bboxes)} bboxes")
```

#### **3.2. Text Recognition**

```python
from text_recognizer import TextRecognizer
import cv2

# Initialize recognizer
recognizer = TextRecognizer(
    model_path='languages/english/rec.onnx',
    dict_path='languages/english/dict.txt',
    use_gpu=False
)

# Recognize single image
cropped = cv2.imread('cropped_text.jpg')
text, confidence = recognizer.recognize(cropped)
print(f"Text: '{text}' (confidence: {confidence:.3f})")

# Batch recognition (faster!)
images = [img1, img2, img3]
results = recognizer.recognize_batch(images)
for text, conf in results:
    print(f"'{text}' ({conf:.3f})")
```

#### **3.3. Full Pipeline**

```python
from template_matcher import TemplateMatcher, BoundingBox
from text_recognizer import TextRecognizer
import json, cv2

# 1. Load template
with open('images/annotations.json', 'r') as f:
    data = json.load(f)
template_path = data['_template_image']
template_bboxes = [BoundingBox.from_dict(b) for b in data[template_path]]

# 2. Match template
matcher = TemplateMatcher(template_path, template_bboxes)
bboxes, conf, target_img = matcher.match('target.jpg', method='auto')

# 3. OCR text regions
recognizer = TextRecognizer('languages/english/rec.onnx', 
                           'languages/english/dict.txt')

text_regions = []
for bbox in bboxes:
    if bbox.bbox_type in ['text', 'datecode']:
        cropped = matcher.crop_region_with_perspective(target_img, bbox)
        text_regions.append(cropped)

# Batch OCR
results = recognizer.recognize_batch(text_regions)
for (text, conf) in results:
    print(f"📝 {text} (conf: {conf:.3f})")
```

---

## 🎯 Template Matching Methods

| Method | Speed | Accuracy | Scale | Rotation | Perspective |
|--------|-------|----------|-------|----------|-------------|
| `simple` | ⚡⚡⚡ | ⭐⭐ | ❌ | ❌ | ❌ |
| `multi_scale` | ⚡⚡ | ⭐⭐⭐ | ✅ | ❌ | ❌ |
| `feature` (SIFT) | ⚡ | ⭐⭐⭐⭐ | ✅ | ✅ | ✅ |
| `superpoint` | ⚡ | ⭐⭐⭐⭐⭐ | ✅ | ✅ | ✅ |
| `auto` | ⚡ | ⭐⭐⭐⭐⭐ | ✅ | ✅ | ✅ |

**Khuyến nghị:**
- Dùng `auto` cho kết quả tốt nhất (tự chọn method phù hợp)
- Dùng `simple` khi ảnh chuẩn, không scale/rotate
- Dùng `feature` hoặc `superpoint` khi ảnh bị méo/perspective

---

## 📊 Hiệu năng

**Tested on MacBook Air M1 (CPU mode):**

| Task | Time | Note |
|------|------|------|
| Template Matching (SIFT) | ~150ms | Full image 3024×4032 |
| Template Matching (with crop_area) | ~50ms | Cropped to 800×600 |
| Text Recognition (single) | ~15ms | PP-OCRv5, height=48 |
| Text Recognition (batch 3 items) | ~45ms | 15ms/image average |

**GPU Acceleration:**
- Thêm `use_gpu=True` khi khởi tạo `TextRecognizer`
- Yêu cầu: CUDA 11.x + `onnxruntime-gpu`
- Speed-up: ~3-5x faster

---

## 🔧 Advanced Features

### **Smart Rectangle/Polygon Detection**

Template matcher tự động quyết định dùng rectangle hay polygon dựa vào homography analysis:

```python
# Kiểm tra 5 điều kiện:
# 1. Perspective components (h₂₀, h₂₁) ≈ 0
# 2. Uniform scale (scale_x ≈ scale_y)
# 3. Rotation angle < 3°
# 4. No shear (determinant check)
# 5. Orthogonality (dot product ≈ 0)

# Nếu pass → dùng rectangle (ảnh thẳng)
# Nếu fail → dùng polygon (ảnh méo)
```

Set `debug=True` để xem chi tiết:

```python
matcher.match('target.jpg', debug=True)
```

Output:
```
🔍 Homography Transform Analysis:
  ✓ Perspective negligible: h20=0.000013, h21=0.000018
  ✓ Scale uniform: sx=1.069, sy=1.070, diff=0.002
  ✗ Rotation too large: -3.96°
  → Using POLYGON for text
```

### **Perspective Correction**

```python
# Crop với perspective correction (uốn thẳng ảnh bị méo)
cropped = matcher.crop_region_with_perspective(target_image, bbox)
```

---

## 📝 File Formats

### **annotations.json**

```json
{
  "_template_image": "/path/to/template.jpg",
  "/path/to/template.jpg": [
    {
      "type": "template",
      "shape": "rectangle",
      "x": 100,
      "y": 200,
      "width": 300,
      "height": 150
    },
    {
      "type": "text",
      "shape": "polygon",
      "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    }
  ]
}
```

**Supported types:**
- `template`: Template region (required)
- `text`: Text region
- `datecode`: Datecode region
- `barcode`: Barcode region
- `crop_area`: Crop area for faster matching

**Supported shapes:**
- `rectangle`: Standard bbox
- `polygon`: 4-point polygon (for perspective)

---

## 🛠️ Troubleshooting

### **1. ONNX Runtime Error: Invalid dimensions**

Lỗi: `Expected: 48, Got: 32`

**Fix:** Đảm bảo `TextRecognizer` dùng `target_height=48`:
```python
# Đã fix trong code mới nhất
recognizer.preprocess(image, target_height=48)
```

### **2. Template matching không tìm thấy**

**Giải pháp:**
- Giảm threshold: `threshold=0.3`
- Thử method khác: `method='feature'` hoặc `method='auto'`
- Thêm `crop_area` bbox để crop vùng search

### **3. OCR kết quả sai**

**Giải pháp:**
- Kiểm tra ảnh crop có đúng không
- Thử tăng resolution ảnh gốc
- Preprocess ảnh trước (denoise, contrast)
- Dùng dict.txt với từ điển riêng

### **4. SuperPoint not available**

**Fix:**
```bash
pip install kornia
pip install git+https://github.com/cvg/LightGlue.git
```

---

## 📚 Documentation

- [Template Matching Design](desktop/docs/TEMPLATE_MATCHING_DESIGN.md)
- [SuperPoint Installation Guide](desktop/docs/SUPERPOINT_README.md)
- [Polygon Template Example](desktop/docs/polygon_template_example.json)

---

## 🎓 Use Cases

✅ **QC/Quality Control**: Kiểm tra datecode, barcode trên sản phẩm  
✅ **Document Processing**: OCR form/document có template cố định  
✅ **Industrial Automation**: Đọc thông tin trên bao bì sản phẩm  
✅ **Warehouse Management**: Scan mã hàng tự động

---

## 📜 License

MIT License

---

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

- Repository: [ngocthien2306/ocr_datecode](https://github.com/ngocthien2306/ocr_datecode)
- Issues: [GitHub Issues](https://github.com/ngocthien2306/ocr_datecode/issues)

---

## 🙏 Credits

- **PP-OCRv5**: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- **SuperPoint + LightGlue**: [CVG LightGlue](https://github.com/cvg/LightGlue)
- **ONNX Runtime**: [Microsoft ONNX Runtime](https://github.com/microsoft/onnxruntime)

---

**Made with ❤️ by ngocthien2306**
