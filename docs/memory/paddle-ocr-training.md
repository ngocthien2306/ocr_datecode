---
name: PaddleOCR training (PP-OCRv5/v6) — ocr_paddle_service + recipe CUSTOM_PADDLE
description: ocr_paddle_service on :8003 fine-tunes PP-OCRv5/v6 rec on the OCR Studio's labelled data, exports ONNX + TensorRT, and recipes run it as ocr_model_type CUSTOM_PADDLE. Built and verified end-to-end 2026-09-29; the export and preprocessing traps are the point of this file.
type: project
---
# PaddleOCR training

Sibling of [ocr-training.md](ocr-training.md) (SMTR, :8002). Same studio, same
labels. `OCRTrainingPage` owns a **Model** picker (SMTR / PaddleOCR, persisted in
localStorage `ocrTraining.modelFamily`) shown on Train/Eval/Export/Test; with
PaddleOCR those tabs render `PaddleTrainPanel` / `PaddleStudioTabs.tsx` against
:8003 (one CTC prediction per crop — no GTC/CTC split). Dataset/Label stay SMTR's.

## Architecture

```
ocr_paddle_service/        FastAPI :8003, prefix /api/ocr, conda env paddle_train
  app/                     train / models / logs / export / evaluate / predict only
  PaddleOCR/               git clone --depth 1 (gitignored), tools/train.py runs here
  weights/paddle_pretrained/<arch>_pretrained.pdparams   (setup script --download)
  tools/export_rec_onnx.py   checkpoint -> inference/ -> rec.onnx (+ parity check)
  tools/trt_build_eval.py    runs in `vision` env: build engine / score with ai_services' TextRecognizerTRT
  data/projects/{pid}/     gray copies, per-run label files, checkpoints, export/{mid}/
```
- Env `paddle_train`: python 3.10, `paddlepaddle-gpu==3.4.0` from the cu130 index
  **plus `--extra-index-url https://pypi.org/simple`** (it depends on
  `cuda-toolkit==13.0.2`, only on PyPI). Works on the RTX 5060 Ti (sm_120).
  Keep opencv (all three wheels) < 5 — albumentations pulls headless 5.0.
- Mongo: `ocr_projects` / `ocr_dataset_items` are **shared with ocr_service and
  read-only here**; this service owns only `paddle_ocr_models`. Split is computed
  exactly like ocr_service (same hash / pinned-test rule), so accuracies compare.
- Every heavy step is a subprocess (train, export, engine build, engine eval);
  the API process never holds a CUDA context. GPU flock is the shared
  `/tmp/ocr_datecode_gpu.lock`.
- Unit: `bash scripts/setup_ocr_paddle_service.sh` → `ocr-paddle-training.service`.

## Runtime binding

`ocr_model_type='CUSTOM_PADDLE'` + the existing `ocr_project_id`/`ocr_model_id`
(no new recipe field → the 19-step checklist does not apply). Backend
`recipes.py::_resolve_custom_ocr_paths` maps type → collection
(`CUSTOM`→`ocr_models`, `CUSTOM_PADDLE`→`paddle_ocr_models`); a Paddle model with
no engine falls back to its ONNX. ai_services: `OCRModelType.CUSTOM_PADDLE` runs
on the PADDLEV6 adapters (`TensorRTOCRBackend`/`ONNXOCRBackend`) with the
recipe's engine + dict; `inference_handler._CUSTOM_OCR_TYPES` covers both
custom types. Hot-swapped per recipe load, like CUSTOM.

## Gotchas (all reproduced)

1. **paddle2onnx 2.1.0 (latest) vs Paddle 3.4** — two separate bugs:
   - Paddle 3.4 emits `linear_v2` / `shape64` → "unsupported operators" abort.
     Patched to `matmul+add` / `shape` for the export trace only.
   - `layer_norm` epsilon is stored as `a_f64`; paddle2onnx reads it as f32 and
     writes garbage (5.8e-36, -0.023, 6.9e+31). Conversion "succeeds", ONNX
     disagrees with Paddle on 36/40 crops. Retag to `a_f32` before converting.
   `export_rec_onnx.py` does both and exits non-zero if ONNX vs Paddle max|diff|
   > 1e-3. Verified: re-exporting the official pretrained weights reproduces
   the official HF ONNX (max diff 2e-6).
2. **Production preprocessing ≠ PaddleOCR's default training input.**
   ai_services (`ocr_utils.preprocess_text_image`): grayscale ×3, H=48, keep
   aspect up to 2000 px, black pad only below 50 px. Upstream config: colour,
   squeezed into 320 px at heights 32/48/64. 41% of MINCED ONION crops are
   >320 px wide at H=48. Fix: gray PNG copies + `MultiScaleDataSet ds_width=True`
   at a single height 48 (max 768), `RecConAug` removed.
3. **Eval must also keep the aspect.** Fixed-width eval (pad to 768) reported
   acc 0.03 for a model scoring 0.95 on the line's preprocessing → best-epoch
   selection was noise. Now `RecResizeImg eval_mode` with shape [3,48,48],
   batch 1; train-eval acc then equals the ONNX/production number (0.935/0.935).
4. **Engines must be built by the `vision` env** (TensorRT 11.1.0.106 —
   ai_services' runtime), never in paddle_train. First build ~3.5 min, later ~30 s.
5. Keep the **full upstream dict** (18,708 lines for v6): the pretrained CTC head
   is sized to it. Output classes = dict + blank + space = 18,710; the export
   step asserts it, because ai_services silently pads a mismatch with `' '`.

## Measured (MINCED ONION, 635 train / 154 test, 15 epochs from pretrained)

| | normalized | exact |
|---|---|---|
| official PP-OCRv6_small (production preprocessing) | 0.60 | — |
| fine-tuned, ONNX, production preprocessing | 0.935–0.948 | 0.916–0.922 |
| fine-tuned, TRT via ai_services TextRecognizerTRT | 0.929–0.948 | 0.909–0.929 |

~2.5 GB VRAM at batch 32, ~15 s/epoch; TRT 1.4 ms/crop.

## Synthetic defects ("Làm giàu dữ liệu lỗi", 2026-09-29)

Why: a good OCR model reads THROUGH bad print (RecAug trains robustness), so
stroke loss / overprint PASS on the line. `app/services/synthetic_defects.py`
damages one char of verified crops and labels it `□` (in every v5/v6 dict,
never printed; `char_missing` drops the char instead). No ai_services change:
`compare_texts()` fails any read containing □ in place of a real char.
- Char positions from CTC timesteps of the project's newest Paddle model (or the
  official v6 ONNX); only crops it reads exactly are used (768/789).
- Damage < 35% of the char's ink is discarded (ambiguous); ≥ 90% erased is
  relabelled char_missing. Synthetic crops of TEST items are never trained on —
  they measure `syn_reject_*` = share of damaged crops the line would FAIL.
- Endpoints `GET/POST/DELETE /projects/{pid}/synthetic`; run opts in with
  `use_synthetic`. FE: `SyntheticDefectsPanel.tsx` inside the Paddle Train tab.

Measured on MINCED ONION (154 clean test / 142 damaged test crops, ONNX):

| model | clean acc | □ on clean | damaged caught |
|---|---|---|---|
| 15 ep, no synth | 93.5–94.8% | 0 | 44–48% |
| 50 ep, no synth | 96.8% | 0 | **37%** (longer training = more robust = catches less) |
| 15 ep + synth (602) | 92.2% | 2 (bottom of text clipped by the crop) | **72.5%** |

Per type with synth: char_missing 89, overprint 83, dot_dropout 71,
stroke_break 59, blot 53%. Synthetic ≠ real: confirm on real defective bottles
before trusting the catch rate; watch false FAIL on good bottles on the line.
