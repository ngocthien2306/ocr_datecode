---
name: OCR Training (SVTRv2/SMTR fine-tune) — service + export + recipe binding
description: New ocr_service (fine-tune SVTRv2+SMTR on recipe crops, export ONNX/TensorRT) + how a trained model gets selected on a recipe. Phase 0 verified; service/FE not built yet.
type: project
---
# OCR Training

Kế hoạch đầy đủ: `docs/ocr_training_plan.md`. File này là phần "cái gì đã thực sự
chạy được + gotcha đã đụng" — đọc plan để biết roadmap, đọc file này để biết chi
tiết implementation mà session sau sẽ gặp.

## Trạng thái (2026-08-10)

**Phase 0 (verify pipeline train→ONNX→TensorRT) — XONG, đạt.** Service, FE, và
recipe binding **chưa làm**.

| Runtime | acc normalized (batch=1) | Ghi chú |
|---|---|---|
| PyTorch `eval_rec` | 0.9744 | 50 epoch / **1m42s** trên RTX 5060 Ti (Colab T4: ~7 phút) |
| ONNX fp16 | 0.9679 gtc / 0.9551 ctc | lệch 1 ảnh so với PyTorch |
| TensorRT fp16 | 0.9679 / 0.9551 | **0/156 bất đồng với ONNX** |

## Kiến trúc

```
ocr_service/            ← sẽ thành FastAPI :8002 /api/ocr (mirror anomaly_service)
    OpenOCR/            ← vendored fork (ngocthien2306/OpenOCR), ĐÃ PATCH, dùng để train
    weights/base/       ← 2 base checkpoint built-in
    Mongo dùng chung DB với backend; collections mới: ocr_projects /
    ocr_dataset_items / ocr_models
```
Env conda **`ocr_train`** (clone của `anomaly_service`) — xem `python_env.md`.

Chỉ train được kiến trúc **SVTRv2 + GTCDecoder(SMTRDecoder + RCTCDecoder)** =
nhánh `OCRModelType.SMTR` ở runtime. `SVTRV2_CTC`/`OPENOCR_REPSVTR`/`PADDLEV5`
không train từ UI này (khác config/dict/postprocess).

## Base checkpoint — luôn fine-tune, không bao giờ from-scratch

`ocr_service/weights/base/`:

| id | file | epoch/step | Là gì |
|---|---|---|---|
| `datecode_2406` | `svtrv2_datecode_2406.pth` | 30 / 600 | đã fine-tune datecode nhà máy, acc 0.968 — **mặc định** |
| `general` | `svtrv2_smtr_gtc_rctc.pth` | 18 / **114,500** | pretrained tổng quát upstream, chưa thấy data nhà máy |

Cả 2: `use_space_char=False` (vocab 99/96/95), dict `EN_symbol_dict.txt`,
`max_text_length=25`, 21,022,849 tham số, cùng kiến trúc → hoán đổi tự do.
Khác nhau 281/281 tensor (không phải bản copy của nhau).

Ngoài 2 base đó, base có thể là **bất kỳ `ocr_models` nào `status='completed'`,
kể cả của project KHÁC** — cross-project là chủ ý (project "datecode chung" làm
base cho project hẹp; operator hay muốn "train tiếp từ model tuần trước").
Endpoint `GET /api/ocr/base-checkpoints` trả built-in + model group theo project.

## Files đã tạo/sửa ở Phase 0

- `ocr_service/export_smtr_onnx.py` — **MỚI**, export **2 output** (`gtc_logits`,
  `ctc_logits`). Thay cho `export_smtr_onnx_attn.py` (3 output, có `attn_maps`,
  giờ chỉ dùng cho tooling char-bbox offline). Vì ONNX chỉ có 2 output nên bước
  build TensorRT **không cần** `drop_outputs` nữa.
- `ocr_service/verify_export.py` — **MỚI**, chấm ONNX vs TRT bằng chính class
  runtime của ai_services (`TextRecognizerSMTRONNX`/`SMTRTRT` + `smtr_utils`),
  không viết lại post-process → lệch ở đây là lệch production thật.
- `ocr_service/expand_ckpt_vocab.py` — **MỚI**, nong vocab +1 class cho space.
- `ocr_service/configs/svtrv2_verify*.yml` — config train tham chiếu (path tuyệt đối).
- OpenOCR vendored, 4 chỗ sửa (xem "Patch OpenOCR" bên dưới).
- `ai_services/.../ocr/backends/smtr_onnx.py` — **SỬA BUG** (xem bên dưới).

## Patch OpenOCR vendored (đã commit, không sed lúc runtime)

1. `openrec/postprocess/{smtr,ctc}_postprocess.py`: `.float()` trước `.numpy()` —
   `Global.use_amp=True` làm head xuất bf16, numpy không có dtype đó.
2. `openrec/modeling/decoders/smtr_decoder.py`: thêm `SMTRDecoder.forward_onnx()`
   — như `forward_test` nhưng **bỏ early-break** và bỏ thu `attn_maps`.
   `torch.onnx.export` trace điều kiện break trên dummy input, nên graph sẽ bị
   đóng băng ở đúng số bước mà cái tensor random đó cần. Loop luôn chạy đủ
   `max_len-1` = **27** bước → output `[B,27,V]` cố định.
3. `openrec/modeling/decoders/__init__.py`: thêm `GTCDecoder.forward_onnx()` trả
   **tuple** `(gtc, ctc)`. `forward()` trả dict; export flatten dict theo thứ tự
   key, dễ đảo — tuple ghim đúng thứ tự 2 binding của engine.
4. `openrec/metrics/rec_metric_gtc.py`: thêm key `min_acc` (xem gotcha #1).

## Gotcha (đều đã dựng lại được trên máy thật)

1. **`main_indicator: acc` chọn best checkpoint BỎ QUA head SMTR** — nghiêm trọng
   nhất. `RecGTCMetric.get_metric()` trả `acc` = **chỉ head CTC**; head SMTR chỉ
   nằm ở `gtc_acc`, mà trainer so bằng `cur_metric[main_indicator]`
   (`trainer.py:511`). Đã dựng lại: run 100 epoch có space lưu `best.pth` tại
   **epoch 2** (ctc 0.974 / gtc 0.519) → model export ra có head GTC hỏng
   (normalized gtc acc 0.487) trong khi **mọi dòng log đều xanh**.
   → Đã thêm `min_acc = min(acc, gtc_acc)` vào `RecGTCMetric`; config do service
   sinh ra **phải luôn** đặt `Metric.main_indicator: min_acc`, cho MỌI run.
2. **`torch.onnx.export` phải truyền `dynamo=False`** — torch ≥2.9 mặc định dùng
   dynamo exporter; vòng decode là for-loop Python cần tracer unroll vào graph.
   Cũng phải `pip install onnxscript` (torch import nó vô điều kiện).
3. **`character_dict_path` trong config là relative theo cwd = `OpenOCR/`** (vì
   `train_rec.py` được chạy từ đó). Export/eval chạy từ chỗ khác → dùng
   `_absolutize_dict_paths()` trong `export_smtr_onnx.py`.
4. **Accuracy do OpenOCR in ra là accuracy ĐÃ CHUẨN HOÁ**
   (`RecMetric._normalize_text`: chỉ giữ chữ+số, lowercase, bỏ space). Exact-match
   thô của cùng model chỉ 0.4359. Đừng so 2 con số này với nhau.
5. **`encode()` âm thầm BỎ ký tự không có trong dict** (`continue`, không loại
   mẫu) — nên với `use_space_char=False`, nhãn `'BB/MA 2029 FE 23'` được train
   thành `'BB/MA2029FE23'`. 58.3% nhãn train của `data_ocr_merged` có space.
6. **`preprocess_batch` pad width bằng −1 làm tụt accuracy**: cùng engine,
   batch=8 → 0.9167, batch=1 → 0.9679. Hành vi có sẵn của production, không phải
   do export. Khi service đo `acc_trt` để gate model thì **phải đo ở batch=1**.
7. **Bảng ký tự runtime lệch 1 so với model production hiện tại.**
   `smtr_utils.build_postprocessors()` gọi `use_space_char=True` → list 100 phần
   tử, trong khi model production train ở chế độ `False` (list 99). Index 95:
   model hiểu là `<s>`, runtime đọc thành `' '`. Vô hại thực tế (model gần như
   không sinh `<s>` giữa chuỗi) — nhưng nghĩa là **model train VỚI space mới
   thực sự khớp đúng decoder runtime**.
8. **numpy phải < 2 trong env train**: `imgaug` 0.4.0 (mà `PARSeqAugPIL` cần) gọi
   `np.sctypes`, đã bị xoá ở numpy 2. `opencv-python` 5.0.0 và torch 2.13 vẫn
   chạy tốt với numpy 1.26.4 (chỉ pip metadata kêu).

## `use_space_char` — MẶC ĐỊNH BẬT, có config trên UI

Bật lên: model sinh được space, **exact-match 0.4359 → 0.9231**, và **không tốn gì
về accuracy** — miễn là dùng `main_indicator: min_acc` (gotcha #1). Hai run 100
epoch với `min_acc` cho metric trùng khít:

| run 100ep, `min_acc` | ctc_acc | gtc_acc | best_epoch |
|---|---|---|---|
| `use_space_char=False` | 0.9743 | 0.9679 | 4 |
| `use_space_char=True` | 0.9743 | 0.9679 | **18** |

⚠️ Lịch sử kết luận (đừng lặp lại): đo với `main_indicator: acc` cho ra "space tốn
2pp accuracy" — **sai**. Đó là artifact của bug chọn checkpoint + nhiễu run-to-run
(cùng config no-space chạy 2 lần ra 0.9679 và 0.9551, tức 2 ảnh).

`best_epoch` 4 vs 18 → **đừng để `epoch_num` thấp khi bật space** (tối thiểu ~30);
UI nên cảnh báo nếu < 20 epoch mà space bật.

**Không phải train lại từ đầu.** `expand_ckpt_vocab.py` chèn 1 hàng vào 5 tensor:
```
decoder.gtc_decoder.char_embed.embedding.weight  (99,384) -> (100,384)   fill=mean
decoder.gtc_decoder.ques1_head.{weight,bias}     (96,..)  -> (97,..)     fill=zero
decoder.ctc_decoder.fc.{weight,bias}             (95,..)  -> (96,..)     fill=zero
```
→ **1,154 tham số mới (0.0055%)**, 21,022,849 (99.9945%) giữ nguyên từ base,
phần cũ trong cả 5 tensor bit-identical. Bằng chứng transfer: eval **epoch 1** đã
đạt acc 0.9166 (from-scratch phải ~0.00). Chỉ head SMTR tự hồi quy lún ~3 epoch
(gtc_acc 0.66 → 0.60 → 0.47 → 0.84 → 0.96) vì phải học *khi nào* chèn space; head
RCTC gần như không ảnh hưởng vì dự đoán song song.

**2 điều dễ sai:**
- `' '` chèn ở **index 95** cho head GTC, tức *giữa* ký tự cuối và `<s>` —
  append vào cuối sẽ âm thầm gán lại `<s>/<INF>/<INB>/<pad>`. Với head CTC thì
  `' '` ở cuối. Script đọc vị trí từ character list của post-processor, không hardcode.
- Điều kiện nong phải dựa trên **shape** (`ques1_head.weight.shape[0] == 96`),
  KHÔNG dựa trên flag — fine-tune tiếp từ model đã có space thì không được nong
  lần hai. `load_state_dict(strict=False)` **vẫn raise** khi sai shape (nó chỉ tha
  key thiếu/thừa).

## Bug đã sửa trong ai_services

`ocr/backends/smtr_onnx.py:37` thiếu `disabled_optimizers=['SimplifiedLayerNormFusion']`
→ **không load được bất kỳ ONNX fp16 SMTR nào** (`InsertedPrecisionFreeCast_*`
quanh RMSNorm làm ORT's SimplifiedLayerNormFusion crash lúc init session).
Production không lộ vì AUTO luôn chọn TensorRT trước, nhưng đường ONNX fallback
cho model custom sẽ cần. Đã vá + fallback `ORT_ENABLE_BASIC` cho ORT cũ.

## Space KHÔNG ảnh hưởng PASS/FAIL của production

`ocr_utils.compare_texts()` (`ocr_utils.py:228`) **xoá sạch space**, normalize
`_ - — – , . : ; '` thành space rồi xoá, cắt ký tự đặc biệt ở cuối, và coi
`O` ≡ `0`. Nên model không sinh space vẫn match đúng. Giá trị của space chỉ nằm ở:
`recognized` lưu DB/hiển thị FE trung thực với chữ in thật, và prefill nhãn ở tab
Label khớp với `expected` (vốn có space).

## Ngân sách VRAM (đo thật, RTX 5060 Ti 16,311 MiB)

| Consumer | VRAM |
|---|---|
| `ai_services` live (SuperPoint + YOLO OBB + OCR TRT + wrinkle/anomaly) | **3,514 MiB** |
| Train OCR bs=32, 32×128, AMP | **~2,550 MiB** |
| Cả hai song song | 6,058 MiB (37%) — không OOM |
| Train anomaly (PatchCore) | có thể **> 10 GB**, xem `anomaly-training.md` |

→ **OCR Studio KHÔNG cần entry-confirm "stop AI service"** như `ml-training`;
toàn bộ run Phase 0 chạy song song với `camera_management_service.py` live.

→ Nhưng **cần flock semaphore GPU dùng chung** cho train OCR / train anomaly /
build TRT. Repo hiện **không có** cơ chế nào (flock duy nhất ở
`camera_management_service.py:623` chỉ chống chạy 2 instance). Hai người bấm train
ở 2 studio cùng lúc là OOM cả hai.

→ `build_engine(workspace_gib=4)` xin thừa: log build báo `Max Scratch Memory: 221 MB`.
Hạ về 1 GB.

→ Nhớ gọi `release_torch_cuda_cache()` (port từ `anomaly_service`) sau train/export:
allocator PyTorch giữ block đã reserve, process khác không xin được.

→ Comment ở `Dashboard.tsx:1236` nói anomaly training chạy **máy khác** là SAI
(trái `anomaly-training.md`) — đang là căn cứ để bỏ qua kiểm tra tranh chấp GPU.

## Hằng số bị khoá

- Dict **`EN_symbol_dict.txt`** (94 entry; `languages/english/` và
  `OpenOCR/tools/utils/` giống hệt nhau — đã diff).
- `max_text_length=25` → decode **27** bước; `sub_str_len=5`; `next_mode=True`.
- Train 32×128 (`RecTVResize`); ONNX trace 32×320 width động; engine profile
  `min 1x3x32x32 / opt 4x3x32x320 / max 16x3x32x2000`, fp16. Build ~40s.
- Engine **phải đúng 2 output** — `smtr_trt.py:44` assert `len(out_names)==2`.

## Recipe binding (chưa làm)

Thêm 2 field recipe-level `ocr_project_id` + `ocr_model_id`, `ocr_model_type='CUSTOM'`.
**Phải chạy đủ CHECKLIST 19 bước** trong `recipe-system.md` cho cả 2 field.
Backend resolve `engine_path`/`dict_path` từ `ocr_models` rồi bơm vào
`load_recipe.metadata` + `recipe_dict` + `update_realtime.recipe_dict`, để
`ai_services` không cần query Mongo.

**OCR model là hot-swappable, KHÁC anomaly/char-classifier**: `camera_manager.py:299`
gọi `set_ocr_model_type()` rồi submit `_reinit_ocr_backend()` lên worker thread khi
đổi. Nên ghi chú #9 trong `MEMORY.md` ("đổi model phải restart ai_services")
**không áp dụng cho OCR backend**. Lưu ý: hàm này phải trả `changed=True` khi
**path đổi** dù `model_type` vẫn là `CUSTOM` (đổi giữa 2 model custom).
