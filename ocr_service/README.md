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
| POST | `/api/ocr/projects/{id}/dataset/prepare` | validate + write `rec_gt_*.txt`; `dry_run=true` by default |
| GET | `/api/ocr/base-checkpoints` | built-in bases + every completed model, grouped by project |
| POST | `/api/ocr/projects/{id}/train` | start a run |
| GET | `/api/ocr/projects/{id}/models` · `.../{mid}/status` | |
| GET | `/api/ocr/projects/{id}/models/{mid}/logs?since=` | live log, `since` cursor |
| POST | `/api/ocr/projects/{id}/models/{mid}/cancel` | |
| DELETE | `/api/ocr/projects/{id}/models/{mid}` | drops the record and every file the run wrote |
| POST | `.../models/{mid}/export-onnx` · `export-tensorrt` | re-export (both run automatically after training) |
| GET | `.../models/{mid}/export/inspect` | engine bindings + profile; `runtime_compatible` |
| GET | `.../models/{mid}/export/{onnx\|onnx_fp16\|engine\|dict\|checkpoint}` | download |
| POST | `.../models/{mid}/evaluate?engine=tensorrt\|onnx` | score the export against the test split |
| POST | `.../models/{mid}/predict?engine=…` | read one uploaded crop |

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

### Preparing a run — and the silent-substitution trap

`POST .../dataset/prepare` validates the verified items and writes the two label
files. Call it with `dry_run=true` to see what a run would train on without
touching disk.

It exists mainly to catch one thing. `BaseRecLabelEncode.encode` returns None
when `len(text) > max_text_length` (measured on the raw text, before unknown
characters are stripped), and `SimpleDataSet.__getitem__` answers None by
fetching a **different sample** — a random one while training, `idx + 1` while
evaluating (`simple_dataset.py:163`). So an over-long label does not shrink the
dataset, it duplicates another image into its slot:

* the dataset size is unchanged, so nothing looks wrong;
* some other image is silently over-represented in training;
* during eval, the reported accuracy is computed over a set that is no longer
  your test set.

`data_ocr_merged` contains 12 such labels (e.g. `'Manufactured date:  MFG
2026/02'`, 31 characters), 10 in train and 2 in test — so every Phase 0 number
was measured on 154 distinct test images with 2 duplicated, not 156. Well inside
the noise band there, but the same defect on a dataset full of long labels would
be invisible and unbounded. `prepare` drops them and says so.

Unknown characters are reported but kept: `encode()` skips characters outside
the 94-entry dict, so the model learns the image as the stripped text. That is
usually a stray character worth fixing rather than a reason to drop the sample.

### Train/test split

An item's stored `split` wins — a folder seed carries its own meaningful split,
and `bulk-split` exists so an operator can pin hard crops into eval.
`test_split` only carves a slice when nothing is assigned to test at all, which
is the state a fresh inspection-import leaves.

The carve hashes the item id rather than slicing an ordered list, so an item's
assignment never changes as the dataset grows. With an index-based slice,
importing 100 more images reshuffles which images are held out, and the accuracy
difference between two runs would partly reflect a different eval set rather
than a better model. The cost is that the ratio is approximate on small
datasets (60 items at `test_split=0.2` gave 17, not 12); the response reports
the actual counts.

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

## Training through the API

`POST /projects/{id}/train` renders a config, widens the base checkpoint if the
run needs the space class, and drives `tools/train_rec.py` as a **subprocess**.
Not in-process: train_rec.py configures the root logger, claims a CUDA device
globally and initialises torch.distributed, so hosting it inside uvicorn would
pollute the API process and leave nothing clean to kill on cancel. A subprocess
gives a real `terminate()` and takes its CUDA context with it.

Phases reported on the model record: `queued` → `waiting_for_gpu` → `preparing`
→ `training` (0–85%) → `trained` → `completed`. Export owns the last 15%, so the
bar doesn't sit at 100% through a 40-second engine build.

Everything the run writes is under `data/projects/{id}/models/`. The generated
config sets `save_epoch_step: [100000, 100000]`, which suppresses `epoch_N.pth`
— each is 252 MB and the defaults wrote **2.7 GB per run**. `latest.pth` and the
run directory are removed on success too; only `{model_id}.pth` (84 MB), the
config and the JSONL log survive.

A run only reaches `status: completed` after ONNX export, the TensorRT build and
the engine accuracy check have all finished. Flipping it at the end of training
made a model look ready while its engine was still building, so the export
endpoints would 400 on something the UI showed as done.

### Scoring the export, not the checkpoint

The number that matters for putting a model on a recipe is the **engine's**
accuracy, not the checkpoint's: fp16 conversion and the export path can both
lose something. Auto-export measures it at batch=1 and stores `metrics.acc_trt`
(normalised) and `metrics.acc_exact_trt`, and warns in the log when the engine
falls more than 2pp below the checkpoint.

Evaluation derives its test set by re-running the same validation and split the
training run used, from the run's own recorded params — **not** by filtering
`ocr_dataset_items` on `split`. Filtering by split re-includes the labels
validation dropped, which training never saw and no model can reproduce; that
bug showed up first as an eval over 156 images against a training log that said
154.

Measured on a 20-epoch space-enabled run over 154 test images:

| | normalized (either head) | exact | ms/img |
|---|---|---|---|
| checkpoint (`min_acc`) | 0.9481 | — | — |
| ONNX fp16 | 0.9610 | 0.9416 | 49.7 |
| TensorRT fp16 | 0.9610 | 0.9416 | **2.6** |

ONNX and the engine agree exactly, which is the check that the export is
faithful. The engine reads slightly above the checkpoint's `min_acc` because
`min_acc` is the *worse* of the two heads while `either` is the better — they are
different questions, not a contradiction.

### GPU arbitration

Runs take a `flock` (`/tmp/ocr_datecode_gpu.lock`, override with
`OCR_GPU_LOCK_PATH`) and **queue** rather than fail — an operator starts a run
and walks away, so waiting beats erroring thirty seconds after they stopped
watching. A queued run sits in `waiting_for_gpu` and the start response reports
the current holder.

OCR training does not need `ai_services` stopped: 2.6 GB alongside its 3.5 GB
fits comfortably (see the VRAM table). What does not fit is two training runs at
once, which is what this lock prevents.

**Scope limit:** only `ocr_service` takes this lock today. `anomaly_service` has
no GPU arbitration of its own, so two operators training in the two studios
simultaneously can still OOM both runs. Fixing that means calling the same
context manager on anomaly's training path, pointed at the same file.

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
