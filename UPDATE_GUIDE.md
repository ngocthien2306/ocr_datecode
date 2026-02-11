# Hướng Dẫn Update Code Sau Khi Revert

## 1. Đã làm gì trên máy này?

- ✅ Xóa 3 commits sau commit `a8a7175` (update ocr model)
- ✅ Force push lên remote để đồng bộ
- ⚠️ Code hiện tại: commit a8a7175 - có OpenOCR nhưng chưa fix CUDA conflict

## 2. Cách update trên máy KHÁC

```bash
cd /home/suntech/Source/ocr_datecode

# Bước 1: Lưu lại thay đổi local (nếu có)
git stash

# Bước 2: Fetch từ remote
git fetch origin

# Bước 3: Reset hard về remote (XÓA tất cả thay đổi local)
git reset --hard origin/release_v1

# Bước 4: Lấy lại thay đổi đã stash (nếu cần)
git stash pop
```

**⚠️ LƯU Ý:** `git reset --hard` sẽ XÓA TẤT CẢ thay đổi chưa commit!

## 3. Force dùng ONNX backend (tránh CUDA conflict)

### Cách 1: Set biến môi trường (Khuyến nghị)

Mở file service hoặc script khởi động và thêm:

```bash
export OCR_BACKEND=onnx
python ai_services/camera_management_service.py
```

### Cách 2: Sửa trong file `.env`

```bash
echo "OCR_BACKEND=onnx" >> /home/suntech/Source/ocr_datecode/.env
```

### Cách 3: Sửa trực tiếp trong code

File: `ai_services/camera_management/inference_handler.py`

Tìm dòng:
```python
self._ocr_backend_instance = OCRBackendFactory.create(OCRBackendType.AUTO)
```

Đổi thành:
```python
self._ocr_backend_instance = OCRBackendFactory.create(OCRBackendType.ONNX)
```

## 4. Kiểm tra backend đang dùng

Sau khi restart service, check logs:

```bash
tail -f /home/suntech/Source/ocr_datecode/ai_services/logs/camera_management.log | grep "OCR backend"
```

Kết quả mong đợi:
```
✅ OCR backend initialized: onnx_openocr
```

hoặc (nếu không có OpenOCR ONNX):
```
✅ OCR backend initialized: onnx_paddlev5
```

**KHÔNG ĐƯỢC:**
```
✅ OCR backend initialized: tensorrt_openocr  ❌ (Sẽ bị CUDA conflict)
```

## 5. Tại sao dùng ONNX thay vì TensorRT?

### Hiện tại (commit a8a7175):
- ✅ Có OpenOCR model (tốt hơn PaddleV5)
- ❌ OpenOCR TensorRT + SuperPoint TensorRT = CUDA context conflict
- ✅ OpenOCR ONNX chạy ổn định, không conflict

### Performance:
- **TensorRT OpenOCR**: ~127 imgs/sec (batch=4) ⚡
- **ONNX OpenOCR**: ~13 imgs/sec (batch=4) ✅ Ổn định
- **TensorRT PaddleV5**: ~40 imgs/sec
- **ONNX PaddleV5**: ~10 imgs/sec

→ ONNX OpenOCR vẫn nhanh hơn TensorRT/ONNX PaddleV5!

## 6. Troubleshooting

### Lỗi: "Your branch is behind..."

```bash
git reset --hard origin/release_v1
```

### Lỗi: "Cannot force push"

```bash
# Chỉ làm trên máy chính (đã có quyền)
git push origin release_v1 --force
```

### Lỗi: "CUDA context conflict" vẫn xuất hiện

Kiểm tra:
```bash
# 1. Kiểm tra backend
grep "OCR backend initialized" logs/camera_management.log

# 2. Nếu vẫn là tensorrt_openocr, force set ONNX
export OCR_BACKEND=onnx

# 3. Restart service
sudo systemctl restart camera-management
```

## 7. Kế hoạch tương lai

Để dùng TensorRT OpenOCR (nhanh hơn), cần:
1. Fix CUDA context conflict bằng shared CUDA context manager
2. Test kỹ trên production
3. Merge fix vào branch

Hiện tại: Dùng ONNX để đảm bảo ổn định ✅
