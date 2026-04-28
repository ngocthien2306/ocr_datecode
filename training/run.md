# Run Commands

Tất cả lệnh chạy từ thư mục gốc: `/Users/ngocthien.ai/Source/Projects/ocr_datecode`

```bash
source /Users/ngocthien.ai/envs/event/bin/activate
```

---

## 1. Training — `scripts/train.py`

```bash
# Train với config mặc định
python training/scripts/train.py --config training/configs/ce_baseline.yaml

# Override backbone và learning rate
python training/scripts/train.py --config training/configs/ce_baseline.yaml \
    --override model.backbone=resnet18 train.lr=1e-4 \
    --name ce_resnet18

# Resume từ run đã có
python training/scripts/train.py --resume training/runs/<run_dir>
```

Configs có sẵn:
- `ce_baseline.yaml` — CE + linear head
- `ce_focal.yaml` — Focal loss
- `ce_polyloss.yaml` — Poly loss
- `ce_center.yaml` — CE + Center loss
- `supcon_128.yaml` — SupCon + projection head
- `arcface_128.yaml` — ArcFace head

---

## 2. Export ONNX — `scripts/export_onnx.py`

```bash
python training/scripts/export_onnx.py --run training/runs/<run_dir>

# Chỉ định output path
python training/scripts/export_onnx.py \
    --run training/runs/<run_dir> \
    --out training/runs/<run_dir>/model.onnx \
    --opset 17
```

---

## 3. Test ONNX — `scripts/test_onnx.py`

```bash
python training/scripts/test_onnx.py --run training/runs/<run_dir>
```

---

## 4. Embedding Similarity Test — `test_embedding.py`

So sánh similarity giữa template (OK) và target images theo từng ký tự.
Chỉ dùng được với head type: `projection`, `arcface`.

```bash
# Chạy toàn bộ dataset
python training/test_embedding.py \
    --run training/runs/supcon_128_efficientnet_b3_20260428-180622

# Chỉ test vài ký tự
python training/test_embedding.py \
    --run training/runs/supcon_128_efficientnet_b3_20260428-180622 \
    --chars A B 0 1

# Dùng nhiều template hơn (ổn định hơn)
python training/test_embedding.py \
    --run training/runs/supcon_128_efficientnet_b3_20260428-180622 \
    --n-template 3
```

Models dùng được:
```bash
# Projection (SupCon) — tốt nhất cho template matching
python training/test_embedding.py --run training/runs/supcon_128_convnext_tiny_20260428-131108
python training/test_embedding.py --run training/runs/supcon_128_efficientnet_b3_20260428-143115
python training/test_embedding.py --run training/runs/supcon_128_efficientnet_b3_20260428-180622

# ArcFace
python training/test_embedding.py --run training/runs/arcface_128_b0_20260428-190257
python training/test_embedding.py --run training/runs/arcface_128_b3_20260428-190614
python training/test_embedding.py --run training/runs/arcface_128_repvit_m1_0_20260428-191145
```

---

## 5. Visualize Embedding — `visualize_embedding.py`

Visualize cosine similarity của từng ảnh so với template. Output là file PNG.

```bash
# Chuẩn bị thư mục test (chỉ làm 1 lần)
mkdir -p training/test_data/ok training/test_data/ng

# Copy OK non-synth
ls /Users/ngocthien.ai/Downloads/output2/char_A/ok | grep -v synth | head -10 | \
    xargs -I{} cp "/Users/ngocthien.ai/Downloads/output2/char_A/ok/{}" training/test_data/ok/

# Copy NG (bao gồm synthetic)
ls /Users/ngocthien.ai/Downloads/output2/char_A/ng | head -10 | \
    xargs -I{} cp "/Users/ngocthien.ai/Downloads/output2/char_A/ng/{}" training/test_data/ng/

# Chạy visualize
python training/visualize_embedding.py \
    --run training/runs/supcon_128_efficientnet_b3_20260428-180622 \
    --ok-dir training/test_data/ok \
    --ng-dir training/test_data/ng \
    --out training/test_data/result.png

# Mở kết quả
open training/test_data/result.png
```

Cách đọc output:
- Ảnh đầu tiên trong `--ok-dir` = **template**
- `sim=0.xxx` = cosine similarity với template (cao = giống template)
- `OK→OK` / `NG→NG` = true label → predicted label
- Viền **xanh** = đúng, viền **đỏ** = sai
- Badge **`S`** = ảnh synthetic
- Threshold vàng trên bar chart = điểm phân tách tốt nhất

---

## 6. Viewer (PyQt5 GUI) — `viewer.py`

Xem kết quả inference trực tiếp trên ảnh thực, hỗ trợ tất cả head types.

```bash
python training/viewer.py
```

Thao tác:
1. **Load Model** → chọn thư mục run (chứa `config.yaml` + `model.onnx`)
2. **Load Folder** → chọn thư mục chứa ảnh cần test
3. Click ảnh trong danh sách → xem kết quả

Controls:
- `Threshold` — ngưỡng phân tách OK/NG
- `Temp` — temperature scale (chỉ ảnh hưởng `projection` và `arcface`)
- `Pad W/H` — padding thêm vào ký tự trước khi inference



---

## 7. Jetson Test — `visualize_embedding.py`

```bash
# Kiểm tra provider (phải thấy CUDAExecutionProvider)
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"

# Visualize + đo kết quả
python3 training/visualize_embedding.py \
    --run weights/supcon_128_repvit_m1_5_20260428-194406 \
    --ok-dir training/test_data/ok \
    --ng-dir training/test_data/ng \
    --out training/test_data/result.png

# Benchmark tốc độ (batch=1, 100 lần)
python3 training/visualize_embedding.py \
    --run weights/supcon_128_repvit_m1_5_20260428-194406 \
    --ok-dir training/test_data/ok \
    --ng-dir training/test_data/ng \
    --out training/test_data/result.png \
    --benchmark

# Tăng số lần đo
python3 training/visualize_embedding.py \
    --run weights/supcon_128_repvit_m1_5_20260428-194406 \
    --ok-dir training/test_data/ok \
    --ng-dir training/test_data/ng \
    --out training/test_data/result.png \
    --benchmark --bench-runs 500
```