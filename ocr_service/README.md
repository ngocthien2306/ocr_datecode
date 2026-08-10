# ocr_service

Fine-tune the OCR recognition model (SVTRv2 + SMTR/GTC + RCTC) on real
text/datecode crops, export to ONNX → TensorRT, and let a recipe select the
result. Plan: [`docs/ocr_training_plan.md`](../docs/ocr_training_plan.md).
Findings + gotchas: [`docs/memory/ocr-training.md`](../docs/memory/ocr-training.md).

**Status:** Phase 0 (verify the train → ONNX → TensorRT chain on real data) is
done and passing. The FastAPI service, the frontend Studio, and the recipe
binding are not built yet.

## Setup

`OpenOCR/` is a separate git clone and is **gitignored** — the repo tracks the
pinned upstream commit plus our patches instead, so clone and patch it:

```bash
cd ocr_service
git clone https://github.com/ngocthien2306/OpenOCR.git
git -C OpenOCR checkout $(cat patches/openocr-pinned-commit.txt)
git -C OpenOCR apply ../patches/openocr-smtr-training.patch
```

The patch is 5 files / 87 lines and is not optional — see
[What the patch does](#what-the-patch-does).

Environment (conda; cloned from `anomaly_service` so torch/TensorRT already
match this machine's driver):

```bash
conda create -n ocr_train --clone anomaly_service -y
conda activate ocr_train
pip install lmdb pyclipper rapidfuzz tqdm pyyaml imgaug albumentations onnxscript pycuda
pip install "numpy==1.26.4"      # NOT optional, see below
```

`numpy` **must** stay below 2: `imgaug` 0.4.0 (which `PARSeqAugPIL` imports)
calls `np.sctypes`, removed in numpy 2. `opencv-python` 5.x and torch 2.13 both
run fine against numpy 1.26 despite their pip metadata asking for `>=2`.

Base checkpoints go in `weights/base/` (gitignored, 81 MB each):

| id | file | what it is |
|---|---|---|
| `datecode_2406` | `svtrv2_datecode_2406.pth` | already fine-tuned on factory datecodes — **default** |
| `general` | `svtrv2_smtr_gtc_rctc.pth` | upstream pretrained (114.5k steps), has not seen factory data |

## Run the service

```bash
cp .env.sample .env      # then set SECRET_KEY to MATCH backend/.env exactly
pip install -r requirements.txt
python -m app.main       # or: uvicorn app.main:app --port 8002 --reload
```

`GET /health` for a liveness check. Auth is the same Bearer token backend's
`/api/auth/login` issues — this service verifies it against the same
`SECRET_KEY` and the same `users` collection, and issues none of its own. A
mismatched `SECRET_KEY` shows up as a blanket 401 on every endpoint.

Ports: 8000 backend, 8001 `anomaly_service`, 8002 here.

Startup logs a warning (not an error) if `OpenOCR/` or the base checkpoints are
missing: the project, dataset and label endpoints work fine without them, and
refusing to boot would block label review on any machine that isn't the
training box. The train endpoint checks again and fails loudly there.

### Endpoints so far

| Method | Path | |
|---|---|---|
| GET | `/health` | |
| POST · GET | `/api/ocr/projects` | |
| GET · PATCH · DELETE | `/api/ocr/projects/{id}` | |
| GET | `/api/ocr/projects/{id}/dataset-stats` | `trainable_count` is what the Train tab gates on |
| GET | `/api/ocr/candidates/recipes` | recipes that actually have OCR regions on record |
| GET | `/api/ocr/candidates` | crop candidates; `match_filter=all\|pass\|fail` |
| POST | `/api/ocr/projects/{id}/import` | selected regions → `need_review` |
| POST | `/api/ocr/projects/{id}/import-folder` | seed from an OpenOCR dataset dir → `verified` |
| GET | `/api/ocr/projects/{id}/dataset/items` | paged, with thumbnails |
| GET | `/api/ocr/projects/{id}/dataset/item-ids` | ids only, for select-all across pages |
| GET | `/api/ocr/projects/{id}/dataset/items/{iid}/full` | full-res crop |
| PATCH | `/api/ocr/projects/{id}/dataset/items/{iid}` | edit `gt_text` / `status` / `split` |
| POST | `.../dataset/items/bulk-status` · `bulk-split` · `bulk-exclude` · `bulk-delete` | |

Collections owned here: `ocr_projects`, `ocr_dataset_items`, `ocr_models`.
`recipes`, `inference_results` and `users` are read-only shared with backend.

### Label lifecycle

```
import   → status=need_review   gt_text = prefill guess
label    → status=verified      operator confirmed or fixed the text
train    → reads verified only, and only where exclude_from_training is false
```

The prefill guess comes from the recipe's own verification result: a region that
matched read exactly what the recipe expected, so `expected` IS the ground
truth; a region that failed gets `recognized`, because on a real misprint what
OCR saw is closer to what is printed than what the recipe wanted. Neither is
good enough to train on unreviewed, which is why nothing is imported as
`verified`. In practice most of an import is `verify_match: true` and can be
promoted in one `bulk-status` call, leaving only the failures to read by hand.

An item with an empty `gt_text` cannot be verified (400). OpenOCR's label
encoder returns None for an empty string and drops the sample, so allowing it
would shrink the training set invisibly instead of erroring.

### Degenerate regions

~19% of text/datecode regions on FAIL frames are unusable: when template
alignment fails, the recipe's annotation quad still gets projected — to
coordinates like `x=-38418` on a 2448px frame. `quad_is_sane` rejects those, and
`/candidates` reports how many it dropped as `skipped_degenerate` so "this
recipe has little data" stays distinguishable from "this recipe's alignment is
failing".

## Train → export → verify

```bash
# 1. train (paths in the config are relative to ocr_service/)
cd OpenOCR && python tools/train_rec.py --c ../configs/svtrv2_finetune.yml

# 2. export ONNX (fp32 + fp16), two outputs: gtc_logits + ctc_logits
cd .. && python export_smtr_onnx.py \
    --c ./configs/svtrv2_finetune.yml \
    --ckpt ./output/finetune/best.pth \
    --out ./output/finetune/export

# 3. build the TensorRT engine
python -c "import sys; sys.path.insert(0,'../scripts')
from trt_build_utils import build_engine
build_engine('./output/finetune/export/rec_smtr_fp16.onnx',
             './output/finetune/export/rec_smtr_fp16.engine', 'image',
             (1,3,32,32), (4,3,32,320), (16,3,32,2000), fp16=True)"

# 4. score ONNX vs engine using ai_services' own runtime classes
python verify_export.py --data ./data_ocr_merged \
    --onnx ./output/finetune/export/rec_smtr_fp16.onnx \
    --engine ./output/finetune/export/rec_smtr_fp16.engine --batch 1
```

`verify_export.py` reports accuracy twice. **`normalized`** is comparable to what
`train_rec.py` prints (OpenOCR's metric keeps only letters+digits and lowercases);
**`exact`** is raw string equality. Do not compare the two.

Measure the engine's accuracy at **batch=1**: `preprocess_batch` pads every crop
in a batch to the widest one with −1, which costs real accuracy (0.968 → 0.917 at
batch 8 on the same engine). That's pre-existing production behaviour, not an
export problem, but it makes batched numbers useless for gating a model.

## Training with spaces

`use_space_char: True` (the default) needs one more class in the vocabulary than
the shipped base checkpoints have, so expand the checkpoint first — this keeps
99.9945% of the weights and adds 1,154 parameters, it does **not** retrain from
scratch:

```bash
python expand_ckpt_vocab.py --c ./configs/svtrv2_finetune.yml \
    --in ./weights/base/svtrv2_datecode_2406.pth \
    --out ./output/base_space/best_space.pth
```

Skip this when the base was already trained with spaces (its
`ques1_head.weight` is already 97 rows). `load_state_dict(strict=False)` does
**not** tolerate a shape mismatch — it only tolerates missing/extra keys — so
getting this wrong is a hard crash, not a silent degradation.

## What the patch does

1. `openrec/postprocess/{smtr,ctc}_postprocess.py` — `.float()` before
   `.numpy()`. With `Global.use_amp: True` the heads emit bf16, which numpy has
   no dtype for.
2. `openrec/modeling/decoders/smtr_decoder.py` — adds
   `SMTRDecoder.forward_onnx()`: `forward_test` minus the data-dependent early
   exit. `torch.onnx.export` traces that exit against the dummy input, which
   would freeze the graph at however many decode steps one random tensor needed.
3. `openrec/modeling/decoders/__init__.py` — adds `GTCDecoder.forward_onnx()`
   returning a tuple, so the two output bindings keep a fixed order.
4. `openrec/metrics/rec_metric_gtc.py` — adds a `min_acc` key.
   `RecGTCMetric.get_metric()` reports `acc` for the **CTC head alone**, so
   `main_indicator: acc` selects best-checkpoints blind to the SMTR head and can
   save a model whose GTC head is unusable while every log line looks healthy.

## VRAM

Measured on an RTX 5060 Ti (16,311 MiB) with `ai_services` live (3,514 MiB):

| batch | training peak | + ai_services | total |
|---|---|---|---|
| 16 | 1,664 MiB | 5,178 MiB | 32% |
| **32** | **2,592 MiB** | **6,106 MiB** | **37%** |
| 64 | 4,614 MiB | 8,128 MiB | 50% |
| 128 | 8,946 MiB | 12,460 MiB | 76% |

Building a TensorRT engine peaks at 612 MiB (same at `workspace_gib` 1 or 4 —
that setting is a pool ceiling, not a reservation).

So OCR training does **not** require stopping `ai_services`, unlike ML Training
Studio. Anomaly training is the one that can't share the card; see
`docs/ocr_training_plan.md` §6b for the GPU-arbitration plan.

Re-measure with `python measure_vram.py --c ./configs/svtrv2_finetune.yml
--batches 16 32 64 --epochs 10`. If two batch sizes report the same peak, the
runs were too short to sample — raise `--epochs`, don't trust the number.
