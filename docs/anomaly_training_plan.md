# Anomaly Detection Training — Kế hoạch triển khai

> Nguồn gốc: mở rộng từ `PROJECT_PLAN_anomaly_6weeks.xlsx` (Epic E1/E2/E6 recipe integration). File này là bản kế hoạch kỹ thuật chi tiết, dùng để theo dõi triển khai qua nhiều phiên làm việc.

## 1. Mục tiêu

Xây dựng luồng huấn luyện **anomaly detection** (phát hiện nhãn nhăn/hư — thay thế `wrinkle_segmenter.py` hiện tại) độc lập với hệ OCR/char-classifier đang có:

1. UI cho phép: chọn **recipe** → xem **inference results** theo recipe đó → crop vùng `label` ra → preview → chọn ảnh **normal/abnormal** → import vào dataset của một **AI Anomaly project**.
2. Service **FastAPI riêng** để train/test/eval/export (ONNX + TensorRT) model anomaly (anomalib: PatchCore/Padim).
3. Mỗi **recipe** (ở cấp template, giống `wrinkle_area` hiện tại) chọn 1 **anomaly project + 1 model version cụ thể** để chạy live, thay cho `WrinkledSegmenterTRT`.

## 2. Kiến trúc

```
ocr_datecode/
├── backend/           (không đổi nhiều — chỉ thêm field recipe)
├── frontend-ts/        + module UI mới: components/anomaly-training/
├── ai_services/         + module inference mới: verification/anomaly_inference.py
└── anomaly_service/     ← MỚI: FastAPI service độc lập, venv riêng (torch + anomalib)
```

**Quyết định kiến trúc đã chốt:**

| Vấn đề | Quyết định |
|---|---|
| Service train/test/export | **Service FastAPI riêng** (`anomaly_service/`), không nhúng vào `backend` — tránh xung đột dependency nặng (torch/anomalib) với process phục vụ production. |
| Truy cập dữ liệu | `anomaly_service` đọc **trực tiếp MongoDB** (`recipes`, `inference_results`) + filesystem ảnh chung (`backend/uploads`, `backend/public`) — không qua REST của `backend`. |
| Nơi train + inference | **Không còn Jetson.** Cả train (`anomaly_service`) lẫn live inference (`ai_services`) đều chạy trên cùng máy GPU **RTX 5060 Ti 16GB**. Không có bước "deploy sang thiết bị khác". |
| Recipe binding | Recipe lưu `anomaly_project_id` + `anomaly_model_id` **cụ thể** (không phải "luôn dùng bản mới nhất") — đổi model là hành động rõ ràng trong RecipeFormModal. |
| Quan hệ với `ml-training` cũ | **Không đụng vào code cũ.** Chỉ tham khảo UX pattern (project sidebar, tab layout, live-log polling kiểu polling `since`, deep-link crop editor) để dựng UI mới — code viết mới hoàn toàn trong `anomaly-training/`. |
| Đối tượng crop | Vùng **`label`** trong `detected_regions` (bbox mà `WrinkledSegmenterTRT` đang dùng hôm nay), không phải crop ký tự như char-classifier. |
| Mask cho ảnh abnormal | **Không cần.** Chỉ label 2 nhãn Normal/Abnormal khi import — không vẽ mask pixel-level, không cần pixel-AUROC. |
| Dataset layout | Chuẩn anomalib Folder datamodule: `train/good/`, `test/good/`, `test/<defect_type>/`. Train set chỉ gồm ảnh "normal" (one-class); abnormal chỉ dùng cho test/eval (image-level AUROC/F1, không cần `ground_truth/` mask). |
| TensorRT engine | Build **ngay trên máy 5060 Ti** — cùng máy train và inference nên không có vấn đề portability giữa kiến trúc GPU khác nhau. Build tại thời điểm export (test ngay), `ai_services` load lại engine đã cache (giống pattern lazy-load TRT của SupCon hiện tại). |
| Auth cho `anomaly_service` | Dùng lại JWT secret của `backend` (không tự chế hệ auth mới). |

## 3. Roadmap theo tuần

### Tuần 1 — Scaffolding + dataset importer
- [ ] `anomaly_service/` — FastAPI skeleton, venv riêng, pin version anomalib.
- [ ] Spike: POC train PatchCore/Padim trên vài chục ảnh label test (xác nhận version anomalib hoạt động với torch/CUDA trên máy 5060 Ti).
- [ ] Mongo models `anomaly_projects` / `anomaly_models` (mirror cấu trúc `ml_projects` của `ml_training` nhưng field khác: `algorithm: patchcore|padim`, `dataset_stats`, `threshold`, `export_status`).
- [ ] Endpoint `GET /candidates` trong `anomaly_service`: lọc `inference_results` theo `recipe_id` + khoảng ngày, crop vùng `label` từ `image_path` gốc (viết lại logic kiểu `_crop_from_polygon` / `_resolve_inspection_image`, không import từ `backend`).
- [ ] Endpoint import: chọn ảnh preview (normal/abnormal) → copy vào `train/good` hoặc `test/<defect_type>` của project.
- [ ] FE scaffold: `frontend-ts/src/components/anomaly-training/` — route mới trong Dashboard, project list + modal "Import from Recipe" (gọi thẳng `anomaly_service`, không qua `backend`).

### Tuần 2 — Train/Test/Eval pipeline
- [ ] Training service trong `anomaly_service`: wrap anomalib `Engine`, config hyperparams theo model (PatchCore: backbone/layers/coreset_ratio; Padim: backbone/layers).
- [ ] Live-log train (polling kiểu `since`, viết lại nhẹ — không tái dùng `ml_training_logs.py` vì khác process).
- [ ] Train config UI + live log panel trong FE.
- [ ] Test/Eval API: chạy model trên test set → AUROC/F1, confusion matrix, heatmap overlay theo ảnh.
- [ ] Test/Eval UI: heatmap overlay, metrics panel, threshold slider.

### Tuần 3 — Export + runtime integration
- [ ] Export ONNX từ `anomaly_service`.
- [ ] Build + verify TensorRT engine ngay trên máy 5060 Ti (cùng máy sẽ chạy `ai_services`).
- [ ] `ai_services`: viết `verification/anomaly_inference.py` mới, thay lời gọi `WrinkledSegmenterTRT` trong `single_camera.py` / `product_verifier.py` — có cờ bật/tắt để rollback nhanh.
- [ ] `backend/app/models/recipe.py`: thêm `anomaly_project_id`, `anomaly_model_id`, `anomaly_threshold` vào `TemplateImage` (cạnh `wrinkle_area`).
- [ ] `RecipeFormModal`: dropdown chọn anomaly project → model version.
- [ ] Export UI: nút ONNX, trạng thái build + download.

### Tuần 4 — Song song (không phụ thuộc anomaly)
- [ ] E3: Cải thiện thuật toán detect cạnh chai (audit + failure cases + tuning).
- [ ] E4: OCR training add-on (fine-tune SMTR/SVTRv2, có thể làm song song nếu có người khác).

### Tuần 5–6 — Buffer
- [ ] Integration test E2E: import → train → eval → export → live.
- [ ] Regression: recipe cũ + ML Training Studio cũ không hỏng.
- [ ] Performance test trên máy GPU 5060 Ti (RAM/VRAM/latency khi anomaly inference chạy live song song với pipeline OCR).
- [ ] Bugfix round 1 + round 2 + docs + demo.

## 4. Tham chiếu code liên quan (đọc trước khi code, không sửa)

- `frontend-ts/src/components/ml-training/ImportFromInspectionsModal.tsx` — pattern recipe→candidates→preview→import (char-level), tham khảo layout/flow.
- `backend/app/api/endpoints/ml_training.py::get_inspection_candidates` — pattern crop từ inference_results theo recipe_id (dùng `region.type == "char"`; bản anomaly cần dùng `region.type == "label"`).
- `backend/app/services/ml_training_service.py` — pattern SupCon ONNX session lazy-load + TensorRT provider tuning cho Jetson Orin 8GB (tham khảo cách cache TRT engine).
- `ai_services/camera_management/verification/wrinkle_segmenter.py` + `product_verifier.py` (`class_names=['product','label','wrinkled']`) — nơi anomaly inference sẽ thay thế.
- `ai_services/camera_management/pipeline/single_camera.py` (dòng ~1043-1104) — nơi `wrinkle_area`/`wrinkle_min_area`/`wrinkle_max_area` được đọc từ template, cần thêm nhánh đọc `anomaly_project_id`/`anomaly_model_id`.
- `backend/app/models/recipe.py::TemplateImage` — nơi thêm field anomaly binding mới.
