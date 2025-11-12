# Image Annotation Tool

Công cụ gán nhãn bounding box cho ảnh, hỗ trợ các loại: template, text, datecode, barcode.

## Cấu trúc dự án

```
desktop/
├── main.py                  # Entry point của ứng dụng
├── main_window.py           # Cửa sổ chính
├── image_viewer.py          # Widget hiển thị ảnh và vẽ bbox
├── annotation_manager.py    # Quản lý lưu/load annotations
├── type_dialog.py           # Dialog chọn type cho bbox
└── README.md               # File này
```

## Cài đặt

```bash
pip install PyQt5
```

## Cách sử dụng

### 1. Chạy ứng dụng

```bash
cd desktop
python main.py
```

### 2. Chọn folder chứa ảnh

- Click nút **"Select Folder"** để chọn folder chứa ảnh
- Hoặc ứng dụng sẽ tự động load folder `images/` nếu có

### 3. Làm việc với ảnh

- **Chọn ảnh**: Click vào tên ảnh trong danh sách bên trái
- Ảnh sẽ hiển thị bên phải
- Các ảnh đã có annotations sẽ có dấu ✓ màu xanh

### 4. Vẽ bounding box

1. Click nút **"Draw BBox"** để bật chế độ vẽ
2. Kéo chuột trên ảnh để vẽ rectangle
3. Khi thả chuột, một popup sẽ xuất hiện
4. Chọn loại bbox:
   - **Template**: Template matching region
   - **Text**: Text/OCR region
   - **Datecode**: Date code region
   - **Barcode**: Barcode/QR code region
5. Click **OK** để lưu bbox

**Phím tắt trong popup:**
- `1`, `2`, `3`, `4`: Chọn nhanh type
- `Enter`: OK
- `Esc`: Cancel

### 5. Quản lý annotations

- **Clear All BBoxes**: Xóa tất cả bbox của ảnh hiện tại
- **Save Annotations**: Lưu thủ công (tự động lưu khi vẽ mới)
- Annotations được lưu vào file `annotations.json` trong folder ảnh

### 6. Màu sắc bbox

- 🔴 **Đỏ**: Template
- 🟢 **Xanh lá**: Text
- 🔵 **Xanh dương**: Datecode
- 🟡 **Vàng**: Barcode

## Định dạng file annotations.json

```json
{
  "/path/to/image1.jpg": [
    {
      "x": 100,
      "y": 150,
      "width": 200,
      "height": 100,
      "type": "template"
    }
  ],
  "/path/to/image2.jpg": [
    {
      "x": 50,
      "y": 50,
      "width": 150,
      "height": 80,
      "type": "text"
    }
  ]
}
```

## Tính năng

✅ Hiển thị danh sách ảnh trong folder
✅ Xem ảnh chi tiết khi click
✅ Vẽ nhiều rectangle trên một ảnh
✅ Phân loại bbox (template, text, datecode, barcode)
✅ Lưu/load annotations tự động
✅ Hiển thị trạng thái ảnh đã annotation
✅ Zoom/scale ảnh tự động
✅ Auto-save khi vẽ mới hoặc chuyển ảnh

## Lưu ý

- Tất cả annotations được lưu với tọa độ của ảnh gốc (không bị ảnh hưởng bởi scaling)
- File `annotations.json` được tạo tự động trong folder chứa ảnh
- Ứng dụng tự động lưu khi đóng hoặc chuyển ảnh
