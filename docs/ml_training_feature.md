# ML Training Studio — Feature Overview

## Tổng quan

ML Training Studio là một overlay page cho phép người dùng tạo dataset, gán nhãn ký tự, huấn luyện mô hình phân loại OK/NG, và chạy dự đoán trực tiếp từ ảnh camera.

**Mở page:** click nút ML Training trên giao diện chính → overlay full-screen hiện ra.
**Khi mở:** backend tự snapshot `/public/images` → `/public/images_temp` để tạo bản camera buffer ổn định cho session.

---

## Cấu trúc file

```
frontend-ts/src/
├── components/ml-training/
│   ├── MLTrainingPage.tsx   # Container chính, sidebar project, tab routing
│   ├── ImageTab.tsx         # Tab quản lý ảnh
│   ├── LabelTab.tsx         # Tab gán nhãn (canvas Fabric.js)
│   └── TrainTab.tsx         # Tab huấn luyện & test dự đoán
├── services/
│   └── mlTraining.ts        # API client + types
└── styles/
    └── MLTraining.css       # Toàn bộ styles của feature

backend/app/api/endpoints/
└── ml_training.py           # Tất cả API endpoints
```

---

## MLTrainingPage (Container)

**File:** `MLTrainingPage.tsx`

### Layout
- **Header:** tiêu đề + badge tên project đang chọn + nút Close
- **Tabs bar:** Images · Label · Train (kèm badge số lượng)
- **Body:** sidebar project (trái) + nội dung tab (phải)

### Sidebar Projects
| Tính năng | Mô tả |
|-----------|-------|
| Danh sách projects | Click để chọn project active |
| Tạo project mới | Nhập tên + description (optional), Enter hoặc click Create |
| Xoá project | Nút ✕ trên mỗi project, confirm trước khi xoá |
| Stats | Hiển thị số ảnh và số ảnh đã labeled |

### State project
- `status: 'active' | 'training' | 'trained'` — hiển thị dưới dạng badge màu

---

## Tab 1 — Images

**File:** `ImageTab.tsx`

### Panel trái: Camera Buffer
- Hiển thị ảnh từ snapshot `/public/images_temp` (bản tĩnh lúc mở page)
- Checkbox chọn từng ảnh hoặc Select All
- Copy ảnh đã chọn sang project → nút **Copy N**
- Slider điều chỉnh số cột grid (1–8, mặc định 3)
- Nút Refresh tải lại danh sách

### Panel phải: Project Images
- Hiển thị ảnh đã có trong project
- Checkbox multi-select + Delete N (có confirm)
- Upload ảnh từ máy (multi-file, accept image/*)
- Slider điều chỉnh số cột grid (1–8, mặc định 3)
- Badge **annotated** trên thumbnail nếu ảnh đã có annotation

---

## Tab 2 — Label

**File:** `LabelTab.tsx` — dùng **Fabric.js v6** cho canvas tương tác

### Layout 3 cột
| Cột | Nội dung |
|-----|---------|
| Trái (image list) | Danh sách ảnh + toggle "Copy prev regions" |
| Giữa (canvas) | Toolbar + canvas Fabric.js + status bar |
| Phải (annotation panel) | Danh sách region & segment |

### Panel ảnh (trái)
- Click ảnh để load vào canvas
- Badge ✓ màu xanh góc phải thumbnail = đã annotated
- Toggle **Copy prev regions**: khi bật, chuyển sang ảnh chưa có annotation sẽ tự copy region boxes từ ảnh trước (không copy segments — user tự Segment)

### Toolbar canvas (giữa)
| Nút | Chức năng |
|-----|-----------|
| Select | Chế độ chọn/kéo object |
| Draw Region | Vẽ vùng chứa ký tự (hình chữ nhật màu amber) |
| Draw Char | Vẽ thủ công 1 char box vào region đang chọn |
| Auto Segment | Segment tự động vùng đang chọn (hiện khi có region) |
| 🔍+ / 🔍- / Reset | Zoom in/out/reset viewport |
| Stats | Hiển thị N OK · N NG · N unlabeled |
| Saving… / ✓ Saved | Trạng thái auto-save |
| Save | Lưu thủ công |

### Điều hướng canvas
| Thao tác | Kết quả |
|----------|---------|
| Scroll chuột | Zoom in/out tại vị trí con trỏ |
| Space + drag hoặc Middle click + drag | Pan canvas |
| Kéo region/segment rect | Di chuyển |
| Kéo góc/cạnh rect | Resize |
| Click char trên canvas | Focus vào item tương ứng ở panel phải |

### Region (vùng chứa ký tự)
- Vẽ bằng **Draw Region** → viền màu amber, selectable/movable/resizable
- Khi di chuyển/resize region box → tọa độ chuẩn hóa được cập nhật vào state
- Expand/collapse danh sách segments bằng chevron ▶
- Header region hiển thị: số segment, select "All…" (All OK / All NG / Reset), nút + (draw char), nút ✕ (xoá region + toàn bộ chars, có confirm)

### Segment (ký tự đơn)
- Tạo bằng **Auto Segment** hoặc vẽ thủ công **Draw Char**
- Mỗi segment có select box label: Unlabeled / OK / NG
- Select box sync 2 chiều với màu rect trên canvas (OK=xanh lá, NG=đỏ, unlabeled=xám)
- Click segment trong panel → cuộn & highlight item đó (nền xanh + border trái màu label)
- Xoá từng char bằng nút ✕

### Auto-save
- Debounce 1.5 giây sau mỗi thay đổi regions
- Bỏ qua lần load đầu từ server (`isInitLoadRef`)
- Hiển thị "Saving…" / "✓ Saved" trên toolbar

### Copy prev regions
- Khi tick checkbox, chuyển ảnh → region boxes (không có segments) từ ảnh trước được paste sang
- Chỉ áp dụng nếu ảnh mới **chưa có annotation**
- User tự click Segment hoặc Draw Char để thêm segments

---

## Tab 3 — Train

**File:** `TrainTab.tsx`

### Panel trái: Cấu hình huấn luyện
| Mục | Chi tiết |
|-----|---------|
| Dataset stats | Số crops OK / NG |
| Algorithm | Random Forest · SVM (RBF) · Neural Net (MLP) |
| Hyperparameters | RF: n_estimators · SVM: C · MLP: max_iter |
| Augmentation (NG) | Off / ×2 / ×3 / ×4 / ×5 — sinh NG giả từ OK |
| Start Training | Disabled nếu < 2 samples; poll mỗi 2s cho đến khi xong |
| History | 5 model gần nhất, trạng thái + test accuracy |

### Panel phải: Kết quả & Test
| Mục | Chi tiết |
|-----|---------|
| Labeled Crops | Grid preview tất cả char crops đã label (base64) |
| Training Results | Train accuracy, Test accuracy, số OK/NG samples |
| Confusion Matrix | Bảng 2×2 NG/OK |
| Classification Report | Text report chi tiết |
| Test Prediction | Upload ảnh → chạy model → hiển thị từng char với label + xác suất OK |

---

## API Endpoints

**Base URL:** `http://localhost:8000/api`
**File backend:** `backend/app/api/endpoints/ml_training.py`

### Projects
| Method | Path | Mô tả |
|--------|------|-------|
| GET | `/ml/projects` | Danh sách projects |
| POST | `/ml/projects` | Tạo project mới |
| GET | `/ml/projects/{id}` | Chi tiết project |
| PATCH | `/ml/projects/{id}` | Cập nhật tên/description |
| DELETE | `/ml/projects/{id}` | Xoá project + toàn bộ data |

### Camera Buffer / Images
| Method | Path | Mô tả |
|--------|------|-------|
| POST | `/ml/snapshot-images` | Copy `/public/images` → `/public/images_temp` |
| GET | `/ml/available-images` | Danh sách ảnh trong `images_temp` |
| GET | `/ml/projects/{id}/images` | Danh sách ảnh project (kèm `has_annotation`) |
| POST | `/ml/projects/{id}/images/copy` | Copy ảnh từ snapshot vào project |
| POST | `/ml/projects/{id}/images/upload` | Upload ảnh từ máy (multipart) |
| DELETE | `/ml/projects/{id}/images/{filename}` | Xoá ảnh khỏi project |
| GET | `/ml/projects/{id}/images/{filename}/meta` | Width/height ảnh (không base64) |

### Annotations
| Method | Path | Mô tả |
|--------|------|-------|
| GET | `/ml/projects/{id}/annotations/{filename}` | Lấy annotation (regions + segments) |
| PUT | `/ml/projects/{id}/annotations/{filename}` | Lưu annotation |

### Segmentation
| Method | Path | Mô tả |
|--------|------|-------|
| POST | `/ml/projects/{id}/segment` | Auto-segment 1 region → trả về danh sách CharSegment |

### Training & Models
| Method | Path | Mô tả |
|--------|------|-------|
| POST | `/ml/projects/{id}/train` | Bắt đầu training |
| GET | `/ml/projects/{id}/models` | Danh sách models |
| GET | `/ml/projects/{id}/models/{model_id}/status` | Trạng thái + metrics |
| GET | `/ml/projects/{id}/labeled-crops` | Tất cả char crops đã label |

### Prediction
| Method | Path | Mô tả |
|--------|------|-------|
| POST | `/ml/projects/{id}/predict` | Upload ảnh → chạy model → kết quả từng char |

### Static files (served trực tiếp)
| Mount | Thư mục | Dùng cho |
|-------|---------|---------|
| `/api/camera-images` | `/public/images_temp` | Ảnh camera buffer |
| `/api/ml-files` | `/public/ml_projects` | Ảnh & data project |

---

## Data Types chính

```ts
MLProject       { id, name, description, image_count, labeled_count, status }
AvailableImage  { filename, url }
ProjectImage    { filename, url, has_annotation }
AnnotationRegion { id, x, y, w, h, segments[] }   // tọa độ chuẩn hóa 0–1
CharSegment     { id, x, y, w, h, label: 'OK'|'NG'|null }
MLModel         { id, algorithm, params, metrics, status, error }
MLModelMetrics  { accuracy_train, accuracy_test, n_ok, n_ng, confusion_matrix, report }
TrainRequest    { algorithm, augment_factor, n_estimators?, max_iter?, C? }
PredictResult   { id, x, y, w, h, prob_ok, label, crop_b64 }
```

---

## Luồng sử dụng điển hình

```
1. Mở ML Training → snapshot camera buffer tự động
2. Tạo project mới (sidebar)
3. Tab Images → copy ảnh từ Camera Buffer vào project
4. Tab Label:
   a. Click ảnh → hiển thị trên canvas
   b. Draw Region → khoanh vùng chứa ký tự
   c. Click Auto Segment → tách từng ký tự
   d. Gán nhãn OK/NG cho từng char (select box hoặc "All…")
   e. Auto-save sau 1.5s; hoặc Save thủ công
   f. Bật "Copy prev regions" để paste layout sang ảnh kế tiếp
5. Tab Train:
   a. Chọn algorithm + params
   b. Start Training → poll kết quả
   c. Xem confusion matrix + report
   d. Test Prediction: upload ảnh → xem kết quả phân loại từng char
```
