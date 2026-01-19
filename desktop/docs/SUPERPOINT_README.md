# SuperPoint + LightGlue Matcher

SuperPoint + LightGlue là phương pháp matching hiện đại sử dụng deep learning, thường cho kết quả tốt hơn SIFT đặc biệt với:
- Ảnh có góc độ khác nhau
- Ảnh có lighting thay đổi
- Ảnh có texture lặp lại

## Installation

### Option 1: Automatic (Recommended)
```bash
cd desktop
chmod +x install_superpoint.sh
./install_superpoint.sh
```

### Option 2: Manual
```bash
# Install PyTorch (nếu chưa có)
pip install torch torchvision --upgrade

# Install Kornia
pip install kornia

# Install LightGlue
pip install git+https://github.com/cvg/LightGlue.git
```

## Usage

### 1. Test với script
```bash
cd desktop
python test_superpoint.py
```

Script này sẽ:
- So sánh SIFT vs SuperPoint
- Đo thời gian thực thi
- Hiển thị confidence score
- Lưu kết quả vào `results/`

### 2. Sử dụng trong code

```python
from template_matcher import TemplateMatcher

matcher = TemplateMatcher(template_image_path, template_bboxes)

# Method 1: Chỉ dùng SuperPoint
bboxes, confidence, image = matcher.match(
    target_image_path,
    method='superpoint',
    threshold=0.3
)

# Method 2: Auto (thử tất cả methods và chọn tốt nhất)
bboxes, confidence, image = matcher.match(
    target_image_path,
    method='auto',  # Sẽ thử cả SIFT và SuperPoint
    threshold=0.3
)
```

## Comparison: SIFT vs SuperPoint

| Feature | SIFT | SuperPoint + LightGlue |
|---------|------|------------------------|
| Type | Classical CV | Deep Learning |
| Speed | Fast (~0.5-1s) | Slower (~2-5s) |
| Accuracy | Good | Excellent |
| Robustness | Moderate | High |
| Dependencies | OpenCV only | PyTorch + LightGlue |

## Performance Tips

1. **GPU acceleration**: SuperPoint tự động dùng CUDA nếu có
   ```python
   # Kiểm tra CUDA
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   ```

2. **Reduce keypoints** để tăng tốc:
   ```python
   # Trong template_matcher.py, line 201
   extractor = SuperPoint(max_num_keypoints=1024)  # Giảm từ 2048
   ```

3. **Fallback**: Nếu SuperPoint không cài được, code tự động fallback về SIFT

## Troubleshooting

### Error: "No module named 'lightglue'"
```bash
pip install git+https://github.com/cvg/LightGlue.git
```

### Error: "CUDA out of memory"
```python
# Giảm số keypoints hoặc force CPU
device = torch.device('cpu')  # Force CPU
```

### SuperPoint chạy chậm
- Nếu không có GPU, SuperPoint sẽ chậm hơn SIFT
- Cân nhắc dùng `method='auto'` để tự động chọn method tốt nhất
- Hoặc dùng `method='feature'` (SIFT) cho tốc độ

## Results Example

Sau khi chạy `test_superpoint.py`:

```
============================================================
Testing: feature
============================================================
✓ Found! Confidence: 0.847
📦 Found 4 bboxes
⏱️  Time: 0.52s
💾 Saved: results/result_feature.jpg

============================================================
Testing: superpoint
============================================================
✓ Found! Confidence: 0.923
📦 Found 4 bboxes
⏱️  Time: 3.14s
💾 Saved: results/result_superpoint.jpg

SUMMARY
============================================================
Method          Confidence   Time (s)   BBoxes
------------------------------------------------------------
feature         0.847        0.52       4
superpoint      0.923        3.14       4

🏆 Best confidence: superpoint (0.923)
⚡ Fastest: feature (0.52s)
```

## Notes

- SuperPoint tốt hơn cho ảnh khó (góc độ lệch, lighting thay đổi)
- SIFT nhanh hơn và đủ tốt cho ảnh đơn giản
- `method='auto'` sẽ thử cả 2 và chọn tốt nhất
