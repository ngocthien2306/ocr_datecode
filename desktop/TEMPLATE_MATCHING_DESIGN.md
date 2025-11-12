# Template Matching & Auto-Annotation Design

## 1. Tổng quan vấn đề

### Yêu cầu:
Bạn có một ảnh template với 3 bounding boxes:
- **1 bbox type `template`**: Vùng đặc trưng để matching
- **2 bbox khác** (text, datecode, barcode): Vùng cần tự động tìm trên ảnh mới

**Mục tiêu**: Với ảnh mới, tự động:
1. Tìm vùng template trong ảnh
2. Dựa vào vị trí tương đối, tự động vẽ 2 bbox còn lại

---

## 2. Giải pháp đề xuất

### 2.1. Template Matching với OpenCV

#### Workflow:
```
Ảnh Template                    Ảnh Target (mới)
┌─────────────────┐            ┌─────────────────┐
│  ┌───────┐     │            │                 │
│  │TEMPLATE│     │            │   ┌───────┐    │
│  │ BBOX   │     │  ────>     │   │FOUND  │    │
│  └───────┘     │  MATCHING  │   │HERE   │    │
│   ┌─┐  ┌──┐   │            │   └───────┘    │
│   │T│  │DC│   │            │    ┌─┐  ┌──┐   │
│   └─┘  └──┘   │            │    │T│  │DC│   │ (Auto)
└─────────────────┘            └─────────────────┘
```

#### Thuật toán:

**Bước 1: Extract Template Region**
```python
# Từ ảnh template, crop vùng bbox type="template"
template_bbox = find_template_bbox(template_image)
template_region = crop_image(template_image, template_bbox)
```

**Bước 2: Template Matching**
```python
import cv2

# Tìm vùng template trong ảnh mới
result = cv2.matchTemplate(target_image, template_region, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Vị trí tìm được
matched_position = max_loc  # (x, y)
```

**Bước 3: Calculate Offset & Transform**
```python
# Tính offset giữa template cũ và vị trí mới
offset_x = matched_position[0] - template_bbox.x
offset_y = matched_position[1] - template_bbox.y

# Transform tất cả bbox khác
for bbox in other_bboxes:
    new_bbox = {
        'x': bbox.x + offset_x,
        'y': bbox.y + offset_y,
        'width': bbox.width,
        'height': bbox.height,
        'type': bbox.type
    }
```

---

### 2.2. Xử lý trường hợp phức tạp

#### A. Ảnh bị xoay (Rotation)
```python
# Sử dụng ORB/SIFT features để detect rotation
import cv2

detector = cv2.ORB_create()
kp1, des1 = detector.detectAndCompute(template_region, None)
kp2, des2 = detector.detectAndCompute(target_image, None)

# Match features
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(des1, des2)

# Estimate homography (rotation + translation)
src_pts = [kp1[m.queryIdx].pt for m in matches]
dst_pts = [kp2[m.trainIdx].pt for m in matches]
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

# Transform bboxes với homography matrix
for bbox in other_bboxes:
    corners = get_bbox_corners(bbox)
    transformed_corners = cv2.perspectiveTransform(corners, H)
    new_bbox = corners_to_bbox(transformed_corners)
```

#### B. Ảnh scale khác nhau
```python
# Multi-scale template matching
scales = [0.8, 0.9, 1.0, 1.1, 1.2]
best_match = None
best_score = -1

for scale in scales:
    resized_template = cv2.resize(template_region, None, fx=scale, fy=scale)
    result = cv2.matchTemplate(target_image, resized_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val > best_score:
        best_score = max_val
        best_match = (max_loc, scale)

# Transform với scale factor
offset_x, offset_y = best_match[0]
scale_factor = best_match[1]

for bbox in other_bboxes:
    new_bbox = {
        'x': bbox.x * scale_factor + offset_x,
        'y': bbox.y * scale_factor + offset_y,
        'width': bbox.width * scale_factor,
        'height': bbox.height * scale_factor
    }
```

---

## 3. Implementation Plan

### 3.1. File Structure
```
desktop/
├── template_matcher.py      # Core matching logic
├── auto_annotate_dialog.py  # UI để chạy auto-annotation
└── main_window.py           # Thêm nút "Auto Annotate"
```

### 3.2. Core Class: TemplateMatcher

```python
class TemplateMatcher:
    def __init__(self, template_image_path, template_bboxes):
        """
        Args:
            template_image_path: Đường dẫn ảnh template
            template_bboxes: List các BoundingBox từ template
        """
        self.template_image = cv2.imread(template_image_path)
        self.template_bboxes = template_bboxes

        # Tìm bbox type="template"
        self.template_bbox = self._find_template_bbox()
        self.template_region = self._extract_template_region()

        # Các bbox khác cần auto-generate
        self.other_bboxes = [b for b in template_bboxes if b.bbox_type != 'template']

    def match_and_transform(self, target_image_path,
                           method='simple',  # 'simple', 'rotation', 'multi_scale'
                           threshold=0.8):
        """
        Tìm template trong ảnh target và transform các bbox

        Returns:
            List[BoundingBox]: Các bbox đã transform
            confidence: Độ tin cậy của matching (0-1)
        """
        target_image = cv2.imread(target_image_path)

        if method == 'simple':
            return self._simple_match(target_image, threshold)
        elif method == 'rotation':
            return self._rotation_match(target_image, threshold)
        elif method == 'multi_scale':
            return self._multiscale_match(target_image, threshold)

    def _simple_match(self, target_image, threshold):
        """Simple template matching với translation only"""
        result = cv2.matchTemplate(
            target_image,
            self.template_region,
            cv2.TM_CCOEFF_NORMED
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < threshold:
            return [], max_val

        # Calculate offset
        offset_x = max_loc[0] - self.template_bbox.rect.x()
        offset_y = max_loc[1] - self.template_bbox.rect.y()

        # Transform other bboxes
        transformed = []
        for bbox in self.other_bboxes:
            new_rect = QRect(
                bbox.rect.x() + offset_x,
                bbox.rect.y() + offset_y,
                bbox.rect.width(),
                bbox.rect.height()
            )
            transformed.append(BoundingBox(new_rect, bbox.bbox_type))

        return transformed, max_val
```

### 3.3. UI Integration

**Thêm nút trong toolbar:**
```python
self.auto_annotate_btn = QPushButton("Auto Annotate")
self.auto_annotate_btn.clicked.connect(self.show_auto_annotate_dialog)
self.auto_annotate_btn.setEnabled(False)  # Enable khi có template
```

**Dialog để chạy auto-annotation:**
```python
class AutoAnnotateDialog(QDialog):
    def __init__(self, annotation_manager, parent=None):
        """
        Features:
        - Select matching method (Simple/Rotation/Multi-scale)
        - Set confidence threshold
        - Preview results trước khi apply
        - Batch process nhiều ảnh
        - Progress bar
        """
```

---

## 4. Advanced Features

### 4.1. Confidence Visualization
```python
# Hiển thị heatmap của matching score
plt.imshow(result, cmap='hot')
plt.colorbar()
```

### 4.2. Multiple Template Support
```python
# Cho phép nhiều template bbox trong 1 ảnh
# Tự động chọn template tốt nhất dựa vào:
- Kích thước (bbox lớn hơn thường ổn định hơn)
- Độ phức tạp (nhiều feature hơn)
- Vị trí (corner/edge thường ổn định hơn center)
```

### 4.3. Verification & Correction
```python
# Sau khi auto-annotate, cho phép:
- Preview tất cả kết quả
- Accept/Reject từng ảnh
- Manual adjustment nếu cần
- Batch export
```

---

## 5. Performance Optimization

### 5.1. Speed Up
```python
# 1. Resize ảnh trước khi matching (nếu ảnh quá lớn)
scale_factor = 0.5
small_target = cv2.resize(target_image, None, fx=scale_factor, fy=scale_factor)
# ... match ...
# Scale ngược lại kết quả

# 2. Region of Interest (ROI)
# Nếu biết template thường ở khu vực nào, chỉ search trong ROI
roi = target_image[y1:y2, x1:x2]

# 3. Multi-threading cho batch processing
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(process_image, image_paths)
```

### 5.2. Accuracy Improvement
```python
# 1. Preprocessing
target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
target_normalized = cv2.normalize(target_gray, None, 0, 255, cv2.NORM_MINMAX)

# 2. Edge-based matching (robust hơn với lighting)
template_edges = cv2.Canny(template_region, 50, 150)
target_edges = cv2.Canny(target_image, 50, 150)
result = cv2.matchTemplate(target_edges, template_edges, cv2.TM_CCOEFF_NORMED)

# 3. Ensemble methods
results = []
for method in [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]:
    r = cv2.matchTemplate(target, template, method)
    results.append(r)
# Vote/Average kết quả
```

---

## 6. Error Handling

```python
class MatchingError(Exception):
    """Base class for matching errors"""
    pass

class NoTemplateFoundError(MatchingError):
    """Template bbox not found in template image"""
    pass

class LowConfidenceError(MatchingError):
    """Matching confidence below threshold"""
    pass

class InvalidTransformError(MatchingError):
    """Transformation results in invalid bboxes"""
    pass

# Usage
try:
    bboxes, confidence = matcher.match_and_transform(target_path)
    if confidence < 0.7:
        # Warning: Low confidence
        show_warning("Low matching confidence. Results may be inaccurate.")
except NoTemplateFoundError:
    show_error("No template bbox found. Please annotate template first.")
except LowConfidenceError:
    show_error("Cannot find template in target image.")
```

---

## 7. Testing Strategy

### 7.1. Unit Tests
```python
def test_simple_matching():
    # Test với ảnh giống hệt
    assert confidence > 0.95

def test_translation():
    # Test với ảnh shift 50px
    assert abs(offset_x - 50) < 2

def test_rotation():
    # Test với ảnh xoay 10 độ
    assert rotation_angle in [9, 10, 11]

def test_scale():
    # Test với ảnh scale 1.2x
    assert abs(scale_factor - 1.2) < 0.05
```

### 7.2. Integration Tests
```python
def test_end_to_end():
    # Load template
    # Match với 10 ảnh test
    # Verify accuracy > 90%
```

---

## 8. Deployment Considerations

### 8.1. Dependencies
```bash
pip install opencv-python opencv-contrib-python numpy
```

### 8.2. Configuration File
```json
{
    "matching": {
        "method": "simple",
        "threshold": 0.8,
        "max_rotation": 15,
        "scale_range": [0.8, 1.2],
        "preprocessing": {
            "grayscale": true,
            "normalize": true,
            "denoise": false
        }
    }
}
```

---

## 9. Future Improvements

1. **Deep Learning Approach**
   - Train CNN để detect template region (robust hơn với occlusion, lighting)
   - Use keypoint detection networks (SuperPoint, etc.)

2. **Active Learning**
   - User correction → retrain model
   - Adaptive threshold dựa trên history

3. **Template Library**
   - Lưu nhiều template cho các loại sản phẩm khác nhau
   - Auto-select template phù hợp

4. **OCR Integration**
   - Sau khi auto-annotate, tự động OCR vùng text/datecode
   - Validate kết quả (format check)

---

## 10. Recommended Approach

Để bắt đầu, tôi suggest triển khai theo thứ tự:

**Phase 1: Basic Implementation** (1-2 ngày)
- ✅ Simple template matching (translation only)
- ✅ UI với nút "Auto Annotate"
- ✅ Preview trước khi apply

**Phase 2: Robustness** (2-3 ngày)
- ✅ Multi-scale matching
- ✅ Confidence threshold tuning
- ✅ Error handling

**Phase 3: Advanced** (3-5 ngày)
- ✅ Rotation handling
- ✅ Batch processing
- ✅ Performance optimization

**Phase 4: Polish** (1-2 ngày)
- ✅ Config file
- ✅ Unit tests
- ✅ Documentation

---

Bạn muốn tôi implement ngay không, hay có điều chỉnh gì về thiết kế này?
