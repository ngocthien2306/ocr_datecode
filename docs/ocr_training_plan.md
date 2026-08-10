# OCR Training Plan — SVTRv2/SMTR fine-tune service + FE + Recipe binding

> Trạng thái: **PLAN** (chưa code). Viết ngày 2026-08-10.
> Anh chị em cùng loại: `docs/anomaly_training_plan.md`, `docs/memory/anomaly-training.md`,
> `docs/memory/ml-training.md`, `docs/memory/recipe-system.md`.

Mục tiêu: cho phép operator **train lại model OCR (SMTR/SVTRv2)** trên chính data
sản xuất (crop vùng `text`/`datecode` từ `inference_results`), export ONNX → TensorRT,
rồi **chọn model đó trong Recipe** như một lựa chọn nằm cạnh 4 model mặc định.

---

## 0. Quyết định đã chốt (không cần hỏi lại)

| # | Vấn đề | Quyết định |
|---|--------|-----------|
| D1 | Service đặt ở đâu | Biến `ocr_service/` thành FastAPI service (`ocr_service/app/`), port **8002**, prefix `/api/ocr`. `ocr_service/OpenOCR/` giữ nguyên làm thư viện train (vendored). Conda env riêng **`ocr_train`**. |
| D2 | Nguồn ground-truth text | **Auto prefill + tab Label**: import tự điền `gt_text` (từ `expected` nếu match=true, từ `recognized` nếu match=false), gắn `status='need_review'`; operator sửa/confirm ở tab Label → `status='verified'`. **Chỉ ảnh `verified` mới vào tập train.** |
| D3 | Recipe trỏ model | Thêm **2 field recipe-level**: `ocr_project_id` + `ocr_model_id`; `ocr_model_type='CUSTOM'` khi chọn model mới. 1 dropdown duy nhất: 4 default trước → `<optgroup>` theo từng OCR project. |
| D4 | Phase 0 | **Chạy thật end-to-end trên máy này** (train → ONNX → engine → infer) trước khi viết service. |
| D5 | Base checkpoint | Train **luôn fine-tune từ một checkpoint**, không bao giờ from-scratch. Nguồn chọn được: 2 base built-in + **bất kỳ model đã train trong quá khứ, kể cả của project khác**. Xem mục 3.7. |
| D6 | `use_space_char` | **Mặc định BẬT (`True`)**, vẫn cho config trên UI. Bật lên thì service tự nong vocab của base checkpoint (`expand_ckpt_vocab`), không train lại từ đầu. Xem mục 3.8. |
| D7 | Chọn best checkpoint | `Metric.main_indicator` **bắt buộc** là `min_acc` cho mọi run, không dùng `acc`. Xem mục 3.9 — dùng `acc` sinh ra model hỏng head SMTR mà log vẫn xanh. |
| D8 | GPU arbitration | Train OCR **không cần dừng `ai_services`** (đo thật: 2.5 GB + 3.5 GB / 16 GB). Nhưng cần **flock semaphore dùng chung** giữa train OCR / train anomaly / build TRT. Xem mục 6b. |

### Hằng số kiến trúc bị khoá (không được đổi tuỳ tiện)

- **Kiến trúc train được**: chỉ **SVTRv2 + GTCDecoder(SMTRDecoder) + RCTCDecoder** — tức là
  nhánh `OCRModelType.SMTR` ở runtime. `SVTRV2_CTC` / `OPENOCR_REPSVTR` / `PADDLEV5`
  **không** được train từ UI này (khác config, khác dict, khác postprocess).
- **Dict cố định**: `EN_symbol_dict.txt` (đúng file `languages/english/EN_symbol_dict.txt`
  mà `SMTR_DICT_PATH` đang trỏ tới). Đổi dict = đổi số class của head = model không còn
  drop-in với `smtr_utils.build_postprocessors()`. Nếu sau này cần ký tự mới → là một
  đợt riêng, có migration.
- **`max_text_length = 25`**, `sub_str_len = 5`, `next_mode = True` — phải khớp giữa
  train config, `SMTRLabelDecode` lúc export và `build_postprocessors` lúc runtime.
- **Ảnh train 32×128** (`RecTVResize`), **ONNX export dummy 32×320 với width động**,
  **engine profile `min 1x3x32x32 / opt 4x3x32x320 / max 16x3x32x2000`** — giống
  spec `smtr_attn` trong `scripts/build_ocr_engines.py`.
- **Engine phải có đúng 2 output** (`gtc_logits`, `ctc_logits`): `smtr_trt.py:44`
  assert `len(self._out_names) == 2`. → **Export ONNX thẳng ra 2 output**
  (`ocr_service/export_smtr_onnx.py`, đã làm ở Phase 0), nên bước build TensorRT
  **không cần** `drop_outputs`. Script cũ `export_smtr_onnx_attn.py` (3 output, có
  `attn_maps`) chỉ giữ lại cho tooling char-bbox offline.

---

## 1. Hiểu biết nền (kết quả đọc code)

### 1.1 Train (từ `ocr_service/finetune_colab_2.ipynb`)

```
OpenOCR/tools/train_rec.py --c <config.yml>
```
- Config sinh động từ template: `Architecture: SVTRv2 / SVTRv2LNConvTwo33 encoder /
  GTCDecoder{gtc: SMTRDecoder, ctc: RCTCDecoder}`, `Loss: GTCLoss(ctc_weight=0.1)`,
  `PostProcess: GTCLabelDecode`, `Metric: RecGTCMetric`.
- Data: `SimpleDataSet` + `label_file_list=[rec_gt_train.txt]`, dòng dạng
  `train/<file>.jpg\t<label>` (TAB), `data_dir` là gốc.
- Transform train: `DecodeImagePIL → PARSeqAugPIL → RecTVResize[32,128] → GTCLabelEncode(SMTRLabelEncode)`;
  eval bỏ augment và dùng `ARLabelEncode`.
- Optimizer AdamW lr=1e-4, wd=0.05, `OneCycleLR` warmup 1.5 epoch, `use_amp: True`.
- Pretrained: `Global.pretrained_model = <best.pth>` (fine-tune, không train từ đầu).
- **2 patch bắt buộc khi dùng AMP/bf16** (notebook đã làm bằng `sed`):
  `openrec/postprocess/smtr_postprocess.py` và `ctc_postprocess.py`:
  `preds.detach().cpu().numpy()` → `preds.detach().cpu().float().numpy()`.
  → Service phải apply patch này **idempotent** lúc khởi động (hoặc commit sẵn vào
  bản OpenOCR vendored — ưu tiên cách này, sạch hơn).
- Kết quả tham chiếu (`svtrv2_finetune_output_24_6/train.log`): 640 train / 156 test,
  50 epoch, best acc **0.9679** @ epoch 30, ~7s/epoch trên T4.
- Output: `output/finetune/best.pth`, `latest.pth`, `epoch_N.pth`, `config.yml`.

### 1.2 Export ONNX (`ocr_service/export_smtr_onnx_attn.py`)

- Wrapper: `encoder(x)` → `decoder.forward_onnx_with_attn(feat)`.
- 3 output: `gtc_logits [B, L-1, V]`, `ctc_logits [B, T, V]`, `attn_maps [B, L-1, H', W']`.
- `dynamic_axes`: batch cho tất cả, width động cho `image`/`ctc_logits`/`attn_maps`.
- opset 14; merge external data về 1 file; verify bằng onnxruntime CPU.
- FP16 qua `onnxconverter_common.float16.convert_float_to_float16(keep_io_types=True)`
  → `rec_smtr_attn_fp16.onnx`.
- **Gotcha đã biết**: ORT `SimplifiedLayerNormFusion` crash trên graph fp16 →
  phải tạo session với `disabled_optimizers=['SimplifiedLayerNormFusion']`
  (fallback: `graph_optimization_level=ORT_ENABLE_BASIC`). Đã xử lý sẵn trong
  `infer_rec_onnx_attn.py::make_session`.

### 1.3 Export TensorRT (`scripts/build_ocr_engines.py` + `scripts/trt_build_utils.py`)

Spec `smtr_attn` đang dùng cho model production hiện tại:
```python
onnx="rec_smtr_attn_fp16.onnx", engine="rec_smtr_attn_fp16.engine",
input_name="image", min=(1,3,32,32), opt=(4,3,32,320), max=(16,3,32,2000),
fp16=True, drop_outputs=["attn_maps"]
```
`anomaly_service/app/services/trt_export.py` là bản copy độc lập của cùng API
(TensorRT 11.1.0 Python Builder, retry fp32 cho node lỗi dtype) — **ocr_service sẽ
ship bản copy tương tự**, thêm bước `drop_outputs`.

### 1.4 Runtime OCR trong `ai_services`

- `ocr/factory.py`: `OCRModelType` enum + `_REGISTRY` map `(model_type, backend)` →
  `(AdapterClass, model_path, dict_path, extra)`. Path là **hằng số hardcode** trỏ vào
  `languages/english/`.
- `inference_handler.py`: `_OCR_MODEL_MAP` (string recipe → enum),
  `set_ocr_model_type()` trả về `changed`, `_reinit_ocr_backend()` build lại backend
  và cập nhật `text_verification_service.text_recognizer`.
- `camera_manager.py:299` gọi `set_ocr_model_type(recipe_data.get('ocr_model_type'))`
  rồi submit `_reinit_ocr_backend` lên worker thread nếu đổi.
- ⇒ **OCR model là hot-swappable theo recipe** (khác anomaly/char-classifier vốn phải
  restart `ai_services`). Đây là điểm KHÁC với ghi chú #9 trong `docs/memory/MEMORY.md` —
  ghi chú đó nói về char classifier & anomaly, không áp dụng cho OCR backend.

### 1.5 Crop data từ recipe

`backend/scripts/crop_ocr_training_data.py` đã có sẵn toàn bộ logic cần thiết:
- `OCR_TYPES = {"text", "datecode"}`, lọc `detected_regions[].type`.
- `crop_obb()` = **perspective warp 4 điểm** (KHÁC anomaly_service dùng axis-aligned
  bbox + padding 4px). Phải dùng `crop_obb` để pixel giống hệt cái OCR nhìn thấy lúc chạy.
- `load_image()` = `UPLOADS_BASE / image_path.replace("_viz", "_org")`.
- Nhãn tham chiếu lấy từ `frame.text_verification.results[]` map theo
  `annotation_idx` ↔ `region.annotation_index`, các key: `expected`, `recognized`,
  `confidence`, `match`.

### 1.6 Khuôn mẫu `anomaly_service` sẽ bám theo

| File anomaly_service | Đối ứng ocr_service |
|---|---|
| `app/main.py` (lifespan + CORS + include_router) | y hệt, port 8002 |
| `app/core/config.py` (Settings + `BACKEND_UPLOADS_PATH` + `PROJECTS_DIR`) | y hệt |
| `app/core/security.py`, `app/api/dependencies/auth.py` | copy nguyên (dùng chung `SECRET_KEY` với backend) |
| `app/db/mongodb.py` | copy nguyên |
| `app/services/train_logs.py` (JSONL buffer + `since` cursor) | copy, đổi `_CAPTURED_LOGGERS` |
| `app/services/trt_export.py` | copy + thêm `drop_outputs` |
| `app/services/inspection_crop.py` | copy, thay `crop_from_polygon` → `crop_obb` |
| `app/services/dataset_fs.py` | viết mới (layout OpenOCR, không phải anomalib Folder) |
| `app/api/endpoints/train.py` (BackgroundTask + progress_cb + cancel flag + auto-export ONNX→TRT) | y hệt pattern |

---

## 2. Phase 0 — ✅ ĐÃ CHẠY XONG (2026-08-10)

### Kết quả

| Runtime | acc (normalized, batch=1) | Ghi chú |
|---|---|---|
| PyTorch `eval_rec` (metric của OpenOCR) | **0.9744** | best_epoch 11, 50 epoch trong **1m42s** trên 5060 Ti |
| ONNX fp16 (decode bằng `smtr_utils` của ai_services) | **0.9679** gtc / 0.9551 ctc | lệch 1 ảnh so với PyTorch |
| TensorRT fp16 `.engine` | **0.9679** gtc / 0.9551 ctc | **0/156 bất đồng với ONNX** |

Latency/ảnh: ONNX(CPU) 32.4 ms — TensorRT(GPU) 2.5 ms (batch 1) / 0.8 ms (batch 8).

Artifacts: `ocr_service/output/verify_run/{best.pth, export/rec_smtr.onnx,
export/rec_smtr_fp16.onnx (43.5 MB), export/rec_smtr_fp16.engine (54.6 MB)}`.
Shape khớp hệt model production: gtc `[B,27,96]`, ctc `[B,W/4,95]`.

### Việc đã làm (code đã nằm trong repo)

- Env conda **`ocr_train`** = clone của `anomaly_service` (torch 2.13+cu130, TRT
  11.1.0.106, ORT 1.23.2) + `lmdb pyclipper rapidfuzz tqdm pyyaml imgaug
  albumentations onnxscript pycuda`, **numpy hạ về 1.26.4**.
- Patch `OpenOCR/openrec/postprocess/{smtr,ctc}_postprocess.py`: `.float()` trước
  `.numpy()` (bf16 của AMP) — commit sẵn, không sed lúc runtime nữa.
- **Mới**: `SMTRDecoder.forward_onnx()` + `GTCDecoder.forward_onnx()` trong OpenOCR
  vendored — vòng decode chạy đủ `max_len-1` bước, không early-break, không thu
  `attn_maps`, trả tuple 2 output.
- **Mới**: `ocr_service/export_smtr_onnx.py` (export 2 output, fp32 + fp16).
- **Mới**: `ocr_service/verify_export.py` (chấm ONNX vs TRT bằng chính class runtime).
- **Sửa bug**: `ai_services/.../ocr/backends/smtr_onnx.py` — thiếu
  `disabled_optimizers=['SimplifiedLayerNormFusion']` nên **không load được bất kỳ
  ONNX fp16 SMTR nào**. Production không lộ vì AUTO luôn chọn TensorRT trước.
- `ocr_service/configs/svtrv2_verify.yml` — config train tham chiếu (path tuyệt đối).

### Phát hiện phải nhớ khi code service

1. **`torch.onnx.export` phải `dynamo=False`** — torch ≥2.9 mặc định dùng dynamo
   exporter; vòng decode là for-loop Python cần được tracer unroll vào graph.
   (Cũng cần cài `onnxscript`, torch import nó vô điều kiện.)
2. **`character_dict_path` trong config là đường dẫn tương đối theo cwd = OpenOCR/**.
   Export chạy từ chỗ khác → `_absolutize_dict_paths()` trong `export_smtr_onnx.py`.
   Service phải làm y hệt.
3. **Accuracy do OpenOCR in ra là accuracy ĐÃ CHUẨN HOÁ** (`RecMetric._normalize_text`:
   chỉ giữ chữ+số, lowercase, bỏ space). Exact-match thô chỉ **0.4359** — không phải
   model tệ, xem #4.
4. **Model không sinh được ký tự space** vì `use_space_char: False` trong khi nhãn có
   space (`'BB/MA 2029 FE 23'` → đọc ra `'BB/MA2029FE23'`). **Không phải lỗi**:
   `ocr_utils.compare_texts()` của production bỏ hết space + normalize dấu câu trước
   khi so, nên match vẫn đúng. Muốn model sinh space thì phải bật `use_space_char: True`
   → vocab 96→97 → **head của base checkpoint không load lại được** (encoder vẫn
   transfer). Giữ `False` như hiện tại.
5. **`preprocess_batch` pad width bằng −1 làm tụt accuracy**: batch=8 → 0.9167,
   batch=1 → 0.9679 (cùng model, cùng engine). Đây là hành vi có sẵn của production,
   không phải do bước export; nhưng khi service đo `acc_trt` thì **phải đo ở batch=1**
   để so sánh công bằng với accuracy PyTorch.
6. **numpy phải < 2 trong env train**: `imgaug` 0.4.0 (mà `PARSeqAugPIL` cần) gọi
   `np.sctypes`, đã bị xoá ở numpy 2. `opencv-python` 5.0.0 vẫn chạy tốt với numpy
   1.26 (chỉ pip metadata kêu), torch 2.13 cũng vậy.

### Các bước gốc đã thực thi (giữ lại để tái lập)

Mục tiêu: chứng minh trên chính máy này (RTX 5060 Ti 16GB) rằng
`data_ocr_merged → train → best.pth → ONNX fp16 → .engine → infer khớp` chạy được, và
**chốt được toàn bộ tham số** để nhét vào service.

**P0.1 — Env**
```bash
conda create -n ocr_train python=3.11 -y
conda activate ocr_train
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130   # khớp env anomaly_service (torch 2.13+cu130)
pip install -r ocr_service/OpenOCR/requirements.txt
pip install lmdb tqdm onnx onnxruntime onnxconverter-common tensorrt==11.1.0.106
```
> Env `anomaly_service` đã có torch cu130 + trt 11.1.0.106 + ort 1.23.2 — nhưng **không
> dùng chung**: OpenOCR pin `numpy` cũ, anomalib pin khác. Tách env.

**P0.2 — Patch OpenOCR vendored** (commit luôn, khỏi sed lúc runtime)
- `openrec/postprocess/smtr_postprocess.py`, `ctc_postprocess.py`: thêm `.float()`.

**P0.3 — Train thử**
- Data: `ocr_service/data_ocr_merged` (640 train / 156 test, đã có `rec_gt_*.txt`).
- Base ckpt: `ocr_service/svtrv2_finetune_output_24_6/best.pth`.
- Config: dựng đúng template trong notebook, `epoch_num=50, bs=32, lr=1e-4`.
- **Kỳ vọng**: acc test ≥ 0.95 (log tham chiếu: 0.9679). Thời gian ~10–20 phút.
- **Deliverable**: `ocr_service/weights/base/` chứa base ckpt + `output/verify_run/best.pth`.

**P0.4 — Export ONNX**
```bash
python ocr_service/export_smtr_onnx_attn.py --c <config.yml> \
  --o Global.pretrained_model=<best.pth> Global.export_dir=<out> Global.fp16=True
```
Check: 3 output, shape đúng, file fp16 chạy được với ORT (nhớ disable fusion).

**P0.5 — Build engine**
```bash
python -c "from trt_build_utils import build_engine; build_engine(
  '<fp16.onnx>', '<out.engine>', 'image',
  (1,3,32,32), (4,3,32,320), (16,3,32,2000), fp16=True)"   # chạy trong scripts/
```
Không cần `drop_outputs` — ONNX đã chỉ có 2 output. Build mất ~40s.

**P0.6 — Verify chéo**
- `python ocr_service/infer_rec_onnx_attn.py --img <vài crop test> --verbose` (ONNX).
- Script nhỏ load engine qua `TextRecognizerSMTRTRT` (dùng `ai_services/.../smtr_trt.py`)
  chạy cùng ảnh → so text/score với ONNX. Sai lệch chấp nhận: text giống 100%,
  score lệch < 0.02 (fp16).
- Chạy cả tập test 156 ảnh, in accuracy engine vs accuracy PyTorch eval.

**Cổng ra Phase 0**: bảng so sánh `PyTorch acc | ONNX acc | TRT acc` + thời gian
inference/ảnh. Chỉ khi khớp mới sang Phase 1.

---

## 3. Phase 1 — `ocr_service` backend (FastAPI, port 8002)

### 3.1 Cây thư mục

```
ocr_service/
  OpenOCR/                      # vendored, đã patch
  data/projects/{project_id}/
      dataset/
          train/ *.jpg
          test/  *.jpg
          rec_gt_train.txt      # sinh lại mỗi lần train từ DB (source of truth = Mongo)
          rec_gt_test.txt
      models/
          {model_id}.pth                     # best.pth copy về
          {model_id}_config.yml
          {model_id}_train.log.jsonl
          export/{model_id}/rec_smtr_attn.onnx
          export/{model_id}/rec_smtr_attn_fp16.onnx
          export/{model_id}/model.engine
          export/{model_id}/EN_symbol_dict.txt   # copy kèm để engine self-contained
  app/
    main.py                     # :8002, prefix /api/ocr
    core/config.py  core/security.py
    db/mongodb.py
    api/dependencies/auth.py
    api/endpoints/
        projects.py  candidates.py  import_dataset.py  dataset.py
        label.py     train.py       eval.py            export.py   test_model.py
    models/ocr.py
    repositories/ocr_repository.py
    services/
        dataset_fs.py  inspection_crop.py  ocr_training.py
        onnx_export.py trt_export.py       train_logs.py  ocr_inference.py
  requirements.txt  .env.sample  README.md
```

### 3.2 Mongo collections mới (cùng DB `ocr_datecode_db`)

**`ocr_projects`**
```
_id, name, description, created_at, updated_at, created_by,
total_count, verified_count, need_review_count,   # đếm dataset item
status: 'active'|'training'|'trained'
```

**`ocr_dataset_items`** — 1 doc / 1 crop
```
_id, project_id,
# provenance (null nếu upload/seed)
inspection_id, camera_serial, frame_idx, annotation_index, recipe_id, recipe_name,
region_type: 'text'|'datecode',
gt_text: str,                    # nhãn dùng để train
prefill_text: str,               # giá trị auto lúc import (để so sánh operator đã sửa gì)
expected_text, recognized_text, ocr_confidence, verify_match,
status: 'need_review'|'verified'|'rejected',
split: 'train'|'test',
image_path: str,                 # relative so với project_dir
source: 'import'|'upload'|'seed',
exclude_from_training: bool = False,
created_at, updated_at, verified_by, verified_at
```
Dedup key: `(inspection_id, camera_serial, frame_idx, annotation_index)`.
Index: `(project_id, status)`, `(project_id, split)`, unique partial trên dedup key.

**`ocr_models`**
```
_id, project_id,
params{epoch_num, batch_size, lr, image_h, image_w, max_text_length, test_split,
       exclude_item_ids, use_space_char,
       base{kind:'builtin'|'model', builtin?, project_id?, model_id?},
       expanded_from},          # set khi phải nong vocab (xem 3.8)
base_label: str,                # "Datecode 24/06" | "HEB · v3" — để hiện lịch sử phả hệ trên UI
use_space_char: bool,           # nhân ra ngoài params cho query/filter
vocab_size: int,                # 99 (no space) | 100 (space) — UI + runtime đều cần
metrics{acc, norm_edit_dis, gtc_acc, min_acc, best_epoch, n_train, n_test,
        acc_onnx, acc_trt, acc_exact_trt},
checkpoint_path, config_path, onnx_path, onnx_fp16_path, engine_path, dict_path,
status: 'pending'|'training'|'completed'|'failed'|'cancelled',
error, phase, progress, created_at, completed_at
```

### 3.3 Endpoints

| Method | Path | Ghi chú |
|---|---|---|
| GET/POST | `/api/ocr/projects` | list / create |
| GET/PATCH/DELETE | `/api/ocr/projects/{pid}` | |
| GET | `/api/ocr/projects/{pid}/dataset-stats` | total / verified / need_review / train / test / per-recipe |
| GET | `/api/ocr/candidates` | query `project_id, recipe_id?, date_from?, date_to?, region_type?, match_filter=all\|pass\|fail, limit` → crop_b64 + prefill + đã import chưa |
| POST | `/api/ocr/projects/{pid}/import` | body: selections[] → ghi file + insert item (`status=need_review`) |
| POST | `/api/ocr/projects/{pid}/import-folder` | seed nhanh từ `data_ocr_merged` (đọc `rec_gt_*.txt`, `status=verified`, `source=seed`) |
| POST | `/api/ocr/projects/{pid}/upload` | upload zip cùng format |
| GET | `/api/ocr/projects/{pid}/dataset/images` | phân trang + filter status/split, thumb_b64 |
| GET | `/api/ocr/projects/{pid}/dataset/image-ids` | ids-only cho "select all across pages" |
| PATCH | `/api/ocr/projects/{pid}/dataset/images/{iid}` | sửa `gt_text` / `status` / `split` |
| POST | `.../dataset/images/bulk-verify` | verify hàng loạt (không đổi text) |
| POST | `.../dataset/images/bulk-delete`, `bulk-exclude`, `bulk-split` | |
| POST | `/api/ocr/projects/{pid}/train` | body `OCRTrainRequest` → `{model_id, status}` |
| GET | `.../models`, `.../models/{mid}/status`, `.../models/{mid}/logs?since=` | |
| POST | `.../models/{mid}/cancel`, DELETE `.../models/{mid}` | |
| GET | `.../models/{mid}/test-results` | per-image: gt / pred_gtc / pred_ctc / score / correct + crop_b64 |
| POST | `.../models/{mid}/export-onnx`, `export-tensorrt` | idempotent, auto chạy sau train |
| GET | `.../models/{mid}/export/onnx`, `/export/tensorrt` | FileResponse download |
| POST | `.../models/{mid}/predict` | upload 1 ảnh → text + score, `engine=onnx|tensorrt` |
| GET | `/api/ocr/models/available` | **dùng cho RecipeFormModal**: list tất cả model `completed` + có `engine_path` tồn tại, group theo project |

### 3.4 `services/ocr_training.py` — luồng train

```
train_model(project_id, model_id, req, progress_cb, cancel_check, excluded_ids)
 1. materialize_dataset()   → query ocr_dataset_items {status:'verified', exclude:false}
                              → chia train/test theo req.test_split (stratify theo recipe_id)
                              → ghi rec_gt_train.txt / rec_gt_test.txt (path tương đối + \t + gt_text)
 2. render_config()         → sinh {model_id}_config.yml từ template (mục 1.1)
 3. subprocess.Popen([python, OpenOCR/tools/train_rec.py, '--c', cfg])
       cwd=OpenOCR, env=ocr_train
       - đọc stdout theo dòng → parse `epoch: [i/N]`, `acc:`, `best metric` → progress_cb
       - cancel_check() → proc.terminate() → raise TrainingCancelled
 4. copy output/best.pth → models/{model_id}.pth ; parse best metric từ log
 5. return metrics
```
**Vì sao subprocess chứ không import**: `train_rec.py` gọi `torchrun`/DDP init, set CUDA
device toàn cục và tự cấu hình logging root — chạy in-process trong uvicorn sẽ nhiễm
process chính và không kill sạch được khi cancel. Subprocess cho phép terminate thật.

Progress phases: `preparing_dataset` → `training (0-85%)` → `evaluating` → `exporting_onnx`
→ `exporting_tensorrt` → `completed`.

### 3.7 Base checkpoint — 2 built-in + mọi model đã train (kể cả project khác)

Train **luôn** là fine-tune từ một checkpoint có sẵn. Không có lựa chọn "from
scratch": 640–vài nghìn crop nhà máy không đủ để train SVTRv2 từ đầu.

**Built-in bases** (copy vào `ocr_service/weights/base/`, commit hoặc ship kèm
installer — KHÔNG để rải rác ở `ocr_service/` như hiện tại):

| id | file | Là gì | Khi nào dùng |
|---|---|---|---|
| `general` | `svtrv2_smtr_gtc_rctc.pth` (epoch 18, **global_step 114,500**) | Bản pretrained tổng quát upstream, chưa thấy data nhà máy | Sản phẩm/font mới hoàn toàn, hoặc muốn tránh bias của lần fine-tune trước |
| `datecode_2406` | `svtrv2_finetune_output_24_6/best.pth` (epoch 30, step 600, acc 0.968) | Đã fine-tune trên datecode nhà máy | **Mặc định** — hội tụ nhanh nhất cho data cùng loại |

Cả 2 đều: `use_space_char=False` (vocab 99/96/95), dict `EN_symbol_dict.txt`,
`max_text_length=25`, 21,022,849 tham số, cùng kiến trúc → hoán đổi được tự do.

**Model đã train**: bất kỳ `ocr_models` doc nào có `status='completed'` và
`checkpoint_path` còn trên đĩa — **không giới hạn trong project đang mở**. Cho
chọn cross-project là chủ ý: một project "Datecode chung" train trên nhiều
recipe có thể làm base tốt cho project hẹp hơn, và operator hay muốn "train tiếp
từ model tuần trước" sau khi bổ sung ảnh.

```python
class OCRBaseRef(BaseModel):
    kind: str                      # 'builtin' | 'model'
    builtin: Optional[str] = None  # 'general' | 'datecode_2406'   (khi kind='builtin')
    project_id: Optional[str] = None   # (khi kind='model') — có thể KHÁC project đang train
    model_id: Optional[str] = None
```

Endpoint riêng cho dropdown, trả cả built-in và model group theo project:

```
GET /api/ocr/base-checkpoints
→ {
    "builtin": [
      {"id":"datecode_2406","label":"Datecode 24/06 (đã fine-tune, acc 0.968)","recommended":true,
       "use_space_char":false,"vocab":99},
      {"id":"general","label":"SVTRv2 SMTR general (114.5k steps)","use_space_char":false,"vocab":99}
    ],
    "projects": [
      {"project_id":"...","project_name":"Datecode HEB","models":[
         {"model_id":"...","label":"v3 · acc 96.8% · 24/06","use_space_char":true,"vocab":100,
          "is_current_project":true}]}
    ]
  }
```

`ocr_models` phải lưu `use_space_char` + `vocab_size` để endpoint này trả về được
— UI cần cảnh báo khi base và request lệch nhau (xem 3.8), và runtime cần biết
để dựng đúng decoder.

Resolve trong `ocr_training.py`:
```
_resolve_base(ref) -> Path
  builtin → BASE_DIR / _BUILTIN[ref.builtin]
  model   → repo.get_model(ref.model_id).checkpoint_path
            (404 nếu không có; 409 nếu status != 'completed' hoặc file mất)
```

### 3.8 `use_space_char` — tham số train + tự nong vocab

`OCRTrainRequest.use_space_char: bool = True` — **mặc định bật**, hiện trên UI dưới
dạng checkbox để operator tắt được khi cần fine-tune tiếp từ một model no-space cũ.

Bật lên → nhãn giữ nguyên space, model sinh được space: **exact-match 0.4359 →
0.9231**. Và với `main_indicator: min_acc` thì **không tốn gì về accuracy** — 2 run
100 epoch cho metric trùng khít:

| run 100ep, `min_acc` | ctc_acc | gtc_acc | best_epoch |
|---|---|---|---|
| `use_space_char=False` | 0.9743 | 0.9679 | 4 |
| `use_space_char=True` | 0.9743 | 0.9679 | 18 |

`best_epoch` 4 vs 18 là lý do `min_acc` bắt buộc: run có space cần ~18 epoch mới
đạt đỉnh (phải học *chỗ* chèn space), trong khi `main_indicator: acc` "thấy đỉnh"
ngay epoch 2. Suy ra 2 điều cho UI: **đừng đặt `epoch_num` quá thấp khi bật space**
(tối thiểu ~30), và cảnh báo nếu operator chọn < 20 epoch với space bật.

**Điều kiện nong vocab dựa trên SHAPE, không dựa trên flag:**

```
need_expand = (request.use_space_char
               and base_ckpt['decoder.gtc_decoder.ques1_head.weight'].shape[0] == 96)
```
- Base `use_space_char=False` (vocab 99/96/95) + request True → **phải** chạy
  `expand_ckpt_vocab` trước, nếu không `load_state_dict` raise size mismatch
  (`strict=False` chỉ tha key thiếu/thừa, KHÔNG tha sai shape).
- Base đã train với space (vocab 100/97/96) + request True → không nong.
- Base có space + request **False** → **chặn ở API** (400). Thu nhỏ vocab là mất
  thông tin, và operator gần như luôn là bấm sai.

Ghi `expanded_from` vào `ocr_models.params` để về sau truy được nguồn gốc.

### 3.9 Chọn best checkpoint phải dùng `min_acc`, không dùng `acc`

`RecGTCMetric.get_metric()` trả `acc` = **chỉ head CTC**; head SMTR chỉ nằm ở
`gtc_acc` và trainer thì so bằng `cur_metric[main_indicator]`. Với
`main_indicator: acc`, best.pth có thể được lưu đúng lúc head SMTR chưa hồi —
đã dựng lại được ở Phase 0: run có space chọn **epoch 2** (ctc 0.974 / gtc 0.519),
model export ra có head GTC hỏng (normalized gtc acc chỉ 0.487) trong khi mọi
chỉ số trên log đều xanh.

→ Vendored OpenOCR đã thêm `min_acc = min(acc, gtc_acc)` vào `RecGTCMetric`, và
**config do service sinh ra phải luôn đặt `Metric.main_indicator: min_acc`**.
Áp dụng cho MỌI run (cả không space) — không chỉ khi bật space.

### 3.5 `services/onnx_export.py` + `trt_export.py`

- `onnx_export.export_smtr_onnx(ckpt, config, out_dir, fp16=True)` — port từ
  `export_smtr_onnx_attn.py`, bỏ ArgsParser, dùng trực tiếp `Config` của OpenOCR.
- `trt_export.build_engine_from_onnx(...)` — copy từ `anomaly_service`, **thêm
  `drop_outputs=['attn_maps']`** trước khi parse, profile `1x3x32x32 / 4x3x32x320 / 16x3x32x2000`.
- Auto-export sau khi train xong, best-effort, mỗi bước độc lập (giống
  `anomaly_service/app/api/endpoints/train.py:152-201`).

### 3.6 Eval

- Sau train: chạy `tools/eval_rec.py` (hoặc gọi lại post-process trực tiếp) trên tập test
  → acc / norm_edit_dis / gtc_acc.
- Thêm **eval trên engine** (`acc_trt`) để phát hiện lệch fp16 → lưu vào `metrics`.
  Đây là gate cho việc cho phép recipe chọn model: chỉ model có `engine_path` tồn tại
  **và** `acc_trt` không kém `acc` quá 2% mới hiện ở `/models/available`.

---

## 4. Phase 2 — Frontend `ocr-training/`

`frontend-ts/src/components/ocr-training/` + `frontend-ts/src/services/ocrTraining.ts`
(axios instance riêng, `OCR_API_BASE_URL = 'http://localhost:8002/api/ocr'`, cùng JWT —
copy y hệt `anomalyTraining.ts`).

Mount fullscreen từ `Dashboard.tsx`: thêm `'ocr-training'` vào type `Section`, thêm
nhánh render + link sidebar (`#ocr-training`), **kèm confirm dialog dừng AI service**
giống `ml-training` (train OCR chiếm gần hết 16GB VRAM, sẽ đá nhau với engine đang chạy).

### Tabs

1. **Projects** — sidebar list + create/rename/delete (copy `AnomalyTrainingPage`).
2. **Dataset / Import** — modal `ImportFromRecipeModal`: chọn recipe + khoảng ngày +
   filter `pass/fail/all` + `text/datecode` → grid crop có sẵn `expected` / `recognized`;
   nút "Import selected". Thêm nút **"Seed từ data_ocr_merged"** (gọi `import-folder`)
   để có data train ngay.
3. **Label** ⭐ (tab quan trọng nhất, không có ở anomaly) —
   - Grid crop lớn, mỗi ô: ảnh + input text (prefill), badge `need_review|verified`,
     hiển thị `expected` vs `recognized` để đối chiếu, phím tắt: `Enter`=verify+next,
     `Ctrl+D`=copy expected, `Del`=reject.
   - Filter: `need_review` / `verified` / `rejected` / theo recipe / theo độ dài text.
   - Bulk: "verify tất cả ảnh có match=true" (an toàn vì expected==recognized).
   - Counter luôn hiện `verified / total` — chính là điều kiện enable nút Train.
4. **Train** — form: base checkpoint (dropdown: base gốc | model đã train trước của project),
   `epoch_num`, `batch_size`, `lr`, `test_split`, `image_h/w`, `max_text_length` (readonly=25).
   → progress bar theo `phase`, log stream (poll `?since=`), nút Cancel, bảng lịch sử model.
5. **Eval** — bảng per-image: crop | GT | GTC pred | CTC pred | score | ✓/✗, filter "chỉ sai",
   metrics tổng (acc / NED / acc_onnx / acc_trt).
6. **Export** — trạng thái ONNX/engine, nút Export lại, Download, Verify engine
   (deserialize + check 2 output), hiển thị đường dẫn.
7. **Test** — upload 1 ảnh → chạy `predict` với `engine=onnx|tensorrt`, hiện text + score + ms.

---

## 5. Phase 3 — Recipe binding (chọn model trên Recipe)

### 5.1 Field mới: `ocr_project_id`, `ocr_model_id`

`ocr_model_type` giữ nguyên nhưng nhận thêm giá trị `'CUSTOM'`.

**Dropdown (RecipeFormModal.tsx:2233)**
```tsx
<select value={ocrSelectValue} onChange={handleOcrModelChange}>
  <option value="">-- Default --</option>
  <option value="SMTR">SMTR (large-x)</option>
  <option value="SVTRV2_CTC">SVTRV2_CTC (large)</option>
  <option value="OPENOCR_REPSVTR">OPENOCR_REPSVTR (medium)</option>
  <option value="PADDLEV5">PADDLEV5 (small)</option>
  {ocrProjects.map(p => (
    <optgroup key={p.id} label={p.name}>
      {p.models.map(m => (
        <option key={m.id} value={`CUSTOM:${p.id}:${m.id}`}>
          acc {(m.metrics.acc*100).toFixed(1)}% · {new Date(m.created_at).toLocaleDateString()}
        </option>
      ))}
    </optgroup>
  ))}
</select>
```
- `ocrSelectValue` = `ocr_model_type === 'CUSTOM' ? \`CUSTOM:${ocr_project_id}:${ocr_model_id}\` : ocr_model_type`.
- `handleOcrModelChange`: nếu value bắt đầu `CUSTOM:` → set 3 field; ngược lại set
  `ocr_model_type=value`, `ocr_project_id=''`, `ocr_model_id=''`.
- **Nếu service 8002 không chạy / không có model nào** → `ocrProjects=[]` → dropdown
  y hệt hiện tại. Fetch phải `catch` im lặng, không chặn mở modal.
- Cảnh báo inline nếu recipe đang trỏ `CUSTOM` mà model đó đã bị xoá / mất engine.

### 5.2 CHECKLIST 19 bước (bắt buộc, ×2 field — theo `docs/memory/recipe-system.md`)

| # | Chỗ sửa | `ocr_project_id` | `ocr_model_id` |
|---|---|---|---|
| 1 | `RecipeBase` — `backend/app/models/recipe.py` | ☐ | ☐ |
| 2 | `RecipeUpdate` — `backend/app/models/recipe.py` | ☐ | ☐ |
| 3 | `RecipeBase` — `backend/app/schemas/recipe.py` | ☐ | ☐ |
| 4 | `RecipeUpdate` — `backend/app/schemas/recipe.py` | ☐ | ☐ |
| 5 | `NULLABLE_STR_FIELDS` — `recipe_repository.py:43` **và** `:157` (2 tuple) | ☐ | ☐ |
| 6 | `recipe_to_response()` — `recipes.py:252` | ☐ | ☐ |
| 7 | `clone_recipe()` — `recipes.py:1457` | ☐ | ☐ |
| 8 | `load_recipe()` **metadata** — `recipes.py:1627` | ☐ | ☐ |
| 9 | `load_recipe()` **recipe_dict** — `recipes.py:1683` | ☐ | ☐ |
| 10 | `update_realtime()` recipe_dict — `recipes.py:2011` | ☐ | ☐ |
| 11 | `Recipe` interface — `types/index.ts:192` | ☐ | ☐ |
| 12 | `Receipt` interface — `types/index.ts:322` | ☐ | ☐ |
| 13 | `FormDataType` — `RecipeFormModal.tsx:71` | ☐ | ☐ |
| 14 | initial state — `RecipeFormModal.tsx:226` | ☐ | ☐ |
| 15 | edit-load — `RecipeFormModal.tsx:426` | ☐ | ☐ |
| 16 | create-reset — `RecipeFormModal.tsx:510` | ☐ | ☐ |
| 17 | `handleSubmit` payload | ☐ | ☐ |
| 18 | 3 `transformedReceipts` — `Receipts.tsx:204`, `:370`, + block search | ☐ | ☐ |
| 19 | `Camera.load_recipe()` — `ai_services/camera_management/camera.py:1607` | ☐ | ☐ |

⚠️ Ghi chú #2 trong MEMORY.md: `Receipts.tsx` có **3** block transform — hay bị sót block clone.

### 5.3 Giải đường dẫn engine — BE resolve, ai_services không cần biết Mongo

Ở **cả 3 chỗ** `load_recipe.metadata`, `load_recipe.recipe_dict`, `update_realtime.recipe_dict`,
khi `ocr_model_type == 'CUSTOM'`: backend query `ocr_models` (cùng DB) theo `ocr_model_id`,
rồi bơm thêm 2 key vào dict:
```python
'ocr_custom_engine_path': model['engine_path'],   # absolute
'ocr_custom_dict_path':   model['dict_path'],
```
Nếu model không tồn tại / file mất → log warning + fallback về `DEFAULT_OCR_MODEL_TYPE`,
**không** làm fail cả lệnh load recipe.

> Lý do chọn cách này: `ai_services` hiện không có kết nối Mongo cho recipe; mọi thứ đến
> qua WebSocket payload. Bơm path ở BE giữ nguyên biên giới đó.

### 5.4 Sửa `ai_services`

1. `ocr/factory.py`
   - Thêm `OCRModelType.CUSTOM = "custom"`.
   - `create()`: nếu `model_type == CUSTOM` → **bắt buộc** có `config` truyền vào
     (`OCRConfig(model_path=engine, dict_path=dict, model_type=CUSTOM)`),
     dùng `SMTRTRTBackend` (TRT) / `SMTRONNXBackend` (fallback ONNX nếu path là `.onnx`).
   - `check_availability()`: thêm key `tensorrt_custom` dựa trên `os.path.exists`.
2. `inference_handler.py`
   - `_OCR_MODEL_MAP['CUSTOM'] = OCRModelType.CUSTOM`.
   - Đổi `set_ocr_model_type(model_type_str)` → `set_ocr_model(model_type_str, engine_path=None, dict_path=None)`;
     lưu `self._ocr_custom_engine_path/_ocr_custom_dict_path`; trả `changed=True` khi
     **path đổi** dù `model_type` vẫn là `CUSTOM` (đổi giữa 2 model custom!).
   - `_init_ocr_backend()`: khi `CUSTOM` thì dựng `OCRConfig` từ path và truyền vào factory.
3. `camera_manager.py:299` — truyền thêm 2 path từ `recipe_data`.
4. `camera.py:1607` — lưu `self.ocr_project_id / self.ocr_model_id` (chỉ để log/debug).

**Hot-swap**: cơ chế `_reinit_ocr_backend()` hiện có đã đủ — đổi recipe sang model
custom khác sẽ rebuild backend trên worker thread, **không cần restart `ai_services`**.
(Khác với anomaly/char-classifier — cần ghi rõ điều này vào memory sau khi làm xong.)

---

## 6. Rủi ro & điểm cần canh

| # | Rủi ro | Xử lý |
|---|---|---|
| R1 | ~~Engine 3 output → `smtr_trt.py` assert fail~~ | ĐÃ GIẢI QUYẾT ở Phase 0: `export_smtr_onnx.py` xuất đúng 2 output. Vẫn nên assert số output sau khi build engine. |
| R2 | Dict/`max_text_length` lệch giữa train & runtime | Khoá hằng số; copy dict vào `export/{model_id}/`; runtime dùng `dict_path` từ recipe payload, không dùng hằng số factory |
| R3 | fp16 làm tụt accuracy | Bắt buộc đo `acc_trt`; chặn model lệch >2% khỏi `/models/available` |
| R4 | VRAM: train + ai_services cùng lúc trên 16GB | Confirm dialog dừng AI service như `ml-training`; batch mặc định 32 @32×128 (~4GB) |
| R5 | Nhãn sai do auto-prefill từ `expected` trên ảnh FAIL | Đã chốt D2: chỉ `verified` mới train; tab Label bắt buộc |
| R6 | ORT `SimplifiedLayerNormFusion` crash trên ONNX fp16 | Dùng `make_session()` với `disabled_optimizers` ở mọi chỗ load ONNX fp16 |
| R7 | Model bị xoá nhưng recipe vẫn trỏ | BE fallback default + warning; FE hiện badge "model missing" |
| R8 | `numpy` conflict (OpenOCR cần <2, opencv mới cần ≥2) | Env `ocr_train` riêng, pin `numpy==1.26.4` + `opencv-python<4.10` |
| R9 | Train chạy nền chết khi uvicorn `--reload` | Chạy service không reload ở production; `train_logs` đã persist JSONL nên log không mất |

---

## 6b. Ngân sách VRAM & điều phối GPU

Đo thật trên RTX 5060 Ti (**16,311 MiB**), 2026-08-10, `ai_services` đang chạy live:

| Consumer | VRAM | Nguồn |
|---|---|---|
| `ai_services/camera_management_service.py` live (SuperPoint TRT + YOLO OBB + OCR TRT + wrinkle/anomaly) | **3,514 MiB** | `nvidia-smi` per-process, ổn định |
| Train OCR, bs=32, 32×128, AMP | **~2,550 MiB** | phẳng suốt run |
| Train OCR + production song song | **6,058 MiB (37%)** | ✅ dư ~10 GB |
| Build TensorRT engine | **612 MiB** | Đo ở cả `workspace_gib=1` và `=4` → **giống nhau**. `workspace_gib` là *trần* của memory pool, không phải reservation; TensorRT chỉ cấp phần thật cần (`Max Scratch Memory: 221 MB`). Không cần hạ. |
| Train anomaly (PatchCore) | **có thể > 10 GB** | `anomaly-training.md`: `wide_resnet50_2` + 500 ảnh đã OOM trên GPU 20 GB bị tranh chấp; phải đổi `resnet18` |

### Hệ quả thiết kế

1. **OCR Studio KHÔNG cần entry-confirm "stop AI service"** như `ml-training`.
   Bằng chứng: toàn bộ run train ở Phase 0 chạy song song với
   `camera_management_service.py` live, không lần nào OOM. Đây là khác biệt có
   chủ ý so với `ml-training` và phải ghi rõ trong code comment, kèm số đo.
2. **Cần flock semaphore GPU dùng chung** — hiện repo **không có** cơ chế nào
   (grep `gpu_lock|FileLock|flock|semaphore`: chỉ có 1 flock ở
   `camera_management_service.py:623` để chống chạy 2 instance, không liên quan GPU).
   Hai operator bấm train ở 2 studio cùng lúc → OOM cả hai. Lock phải:
   - nằm ở đường dẫn cả `ocr_service` và `anomaly_service` thấy được;
   - được acquire bởi **train OCR, train anomaly, và build TRT**;
   - **xếp hàng** (blocking với timeout) chứ không fail ngay — người dùng bấm
     train rồi bỏ đi, fail im lặng tệ hơn chờ;
   - phản ánh trạng thái ra `phase` (`waiting_for_gpu`) để UI hiện được.
3. **Hạ `workspace_gib` của build engine** xuống 1 GB (đang 4). Log build cho thấy
   scratch thật chỉ 221 MB; 4 GiB là hạn mức pool, TensorRT có thể chạm tới.
4. **Gọi `release_torch_cuda_cache()` sau train/export** — port từ
   `anomaly_service/app/services/anomaly_training.py:270`. Allocator của PyTorch
   giữ block đã reserve; process khác không xin được dù tensor đã free. Đây là
   bug `anomaly_service` từng đụng, đừng lặp lại.
5. **Sửa comment sai ở `frontend-ts/src/components/dashboard/Dashboard.tsx:1236`**:
   nói *"anomaly training runs on a separate GPU workstation, not the camera
   service machine"*, trái với `anomaly-training.md` (chạy **cùng máy**). Comment
   sai này đang là căn cứ để bỏ qua mọi kiểm tra tranh chấp GPU cho anomaly.

## 7. Thứ tự thực hiện (tôi sẽ chờ lệnh từng bước)

| Bước | Nội dung | Output |
|---|---|---|
| ~~**B0**~~ | ✅ Phase 0: env + train thử + export + verify | xong 2026-08-10, xem mục 2 |
| **B1** | Scaffold `ocr_service/app` (config/db/auth/main/models/repository) + projects CRUD | service lên được `:8002/health` |
| **B2** | `candidates` + `import` + `import-folder` (seed `data_ocr_merged`) + `dataset` list | có data trong project |
| **B3** | `label` endpoints + `dataset_fs.materialize_dataset` | `rec_gt_*.txt` sinh đúng |
| **B4** | `ocr_training.py` + `train.py` endpoints + `train_logs` | train được từ API, có log/cancel |
| **B5** | `onnx_export` + `trt_export` + auto-export + `eval` + `predict` | có `.engine` + acc_trt |
| **B6** | FE `ocrTraining.ts` + `ocr-training/` (Projects/Dataset/Label) | label được trên UI |
| **B7** | FE Train/Eval/Export/Test tabs + Dashboard section | train được từ UI |
| **B8** | Recipe: 19-step checklist ×2 field + dropdown optgroup | recipe lưu được model custom |
| **B9** | `ai_services`: `OCRModelType.CUSTOM` + resolve path + hot-swap | chạy inference bằng model mới |
| **B10** | Ghi `docs/memory/ocr-training.md` + cập nhật `MEMORY.md` | memory cho session sau |

---

## 8. Những gì KHÔNG làm trong đợt này

- Không train `SVTRV2_CTC` / `OPENOCR_REPSVTR` / `PADDLEV5`.
- Không mở rộng dict / thêm ký tự mới.
- Không train model detect (`best-obb-text`) — chỉ recognition.
- Không đụng `template-level` config (`anomaly_config`, `color_config`, …).
- Không tự động rollout model mới cho mọi recipe — operator chọn thủ công từng recipe.
