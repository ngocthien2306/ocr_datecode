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
