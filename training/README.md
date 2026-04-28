# OCR DateCode — Training Framework

Loss + backbone comparison framework với YAML configs.

## Cấu trúc

```
training/
├── configs/                 # YAML experiment configs
│   ├── _common.yaml         # shared defaults (extends from)
│   ├── ce_baseline.yaml
│   ├── ce_focal.yaml
│   ├── ce_polyloss.yaml
│   ├── ce_center.yaml
│   ├── supcon_128.yaml
│   └── arcface_128.yaml
├── src/
│   ├── data.py              # Dataset, samplers, augment presets
│   ├── models.py            # timm backbone + heads (linear/projection/arcface)
│   ├── losses.py            # 6 losses
│   ├── trainer.py           # train loop
│   ├── evaluator.py         # validation + metrics + threshold sweep
│   └── utils.py             # config loader, seed, env, log mirror
├── scripts/
│   ├── train.py             # single-experiment entry point
│   └── compare.py           # multi-experiment sweep + aggregation
├── runs/                    # auto-created per experiment
└── requirements.txt
```

## Cài đặt

```bash
cd training
pip install -r requirements.txt
```

## Train 1 experiment

```bash
python scripts/train.py --config configs/ce_baseline.yaml
```

Override bất cứ field nào qua CLI (dotlist):
```bash
python scripts/train.py --config configs/ce_baseline.yaml \
    --override model.backbone=resnet18 train.lr=1e-4 train.epochs=20 \
    --name ce_resnet18
```

Output: `runs/<experiment_name>_<YYYYMMDD-HHMMSS>/`
- `config.yaml` — frozen config (audit)
- `env.json` — torch/timm/git versions
- `best.pt` — best checkpoint
- `model.onnx` — single-file ONNX
- `metrics.json` — final metrics + history
- `per_char_metrics.csv` — per-char OK_pass/NG_catch
- `val_predictions.npz` — predictions cho FN/FP analysis
- `train_log.txt` — full stdout mirror

## Compare nhiều experiment

### Mode 1: chỉ định N config files

```bash
python scripts/compare.py \
    --configs configs/ce_baseline.yaml \
              configs/ce_focal.yaml \
              configs/ce_polyloss.yaml \
              configs/ce_center.yaml \
              configs/supcon_128.yaml \
              configs/arcface_128.yaml \
    --output runs/loss_comparison_$(date +%Y%m%d)
```

### Mode 2: sweep 1 axis (Cartesian product)

```bash
# Sweep backbone với cùng loss CE
python scripts/compare.py \
    --base-config configs/ce_baseline.yaml \
    --sweep model.backbone=mobilenetv3_small_100,efficientnet_b0,mobileone_s1,fastvit_t8,repvit_m1_0 \
    --output runs/backbone_sweep
```

### Mode 3: sweep nhiều axis (multi-axis Cartesian)

```bash
# 3 backbone × 2 loss = 6 runs
python scripts/compare.py \
    --base-config configs/ce_baseline.yaml \
    --sweep model.backbone=efficientnet_b0,resnet18,mobileone_s1 \
    --sweep loss.type=ce,focal \
    --output runs/full_matrix
```

Output: `runs/<sweep_dir>/comparison.csv` + `comparison.md`.

## Recipes thường dùng

### Quick smoke test (3 epoch)

```bash
python scripts/train.py --config configs/ce_baseline.yaml \
    --override train.epochs=3 train.batch_size=32 \
    --name smoke_test
```

### Train với 1 model SOTA cụ thể

```bash
python scripts/train.py --config configs/ce_baseline.yaml \
    --override model.backbone=repvit_m1_0 \
    --name ce_repvit
```

### Compare 5 SOTA backbones (default lineup)

```bash
python scripts/compare.py \
    --base-config configs/ce_baseline.yaml \
    --sweep model.backbone=efficientnet_b0,mobileone_s1,fastvit_t8,repvit_m1_0,convnextv2_pico \
    --output runs/sota_5
```

### Compare 6 losses (cùng backbone)

```bash
python scripts/compare.py \
    --configs configs/ce_baseline.yaml \
              configs/ce_focal.yaml \
              configs/ce_polyloss.yaml \
              configs/ce_center.yaml \
              configs/supcon_128.yaml \
              configs/arcface_128.yaml \
    --output runs/loss_6
```

## Backbones tested at 64×64

| Backbone (timm) | Params | Status |
|---|---|---|
| `mobilenetv3_small_100` | 1.5M | ✅ |
| `efficientnet_b0` | 4.0M | ✅ baseline |
| `mobileone_s1` | 3.6M | ✅ Apple reparam |
| `fastvit_t8` | 3.3M | ✅ Apple hybrid |
| `repvit_m1_0` | 6.4M | ✅ 2024 SOTA |
| `ghostnetv2_100` | 4.9M | ✅ |
| `tiny_vit_5m_224` | 5.1M | ✅ distilled CLIP |
| `edgenext_small` | 5.3M | ✅ |
| `convnextv2_pico` | 8.6M | ✅ |
| `resnet18` | 11.2M | ✅ |
| `efficientnetv2_rw_t` | 12.6M | ✅ |
| `convnext_tiny` | 27.8M | ✅ |
| `efficientformerv2_s0` | — | ❌ pure ViT, requires ≥224 input |

Liệt kê thêm: `python -c "import timm; print(timm.list_models('repvit*'))"`

## Loss roster

| Loss | type | Task | Head | Khi nào dùng |
|---|---|---|---|---|
| Cross-Entropy + class weights + label smoothing | `ce` | binary | linear | Baseline |
| Focal Loss | `focal` | binary | linear | NG có hard examples |
| PolyLoss (Poly-1) | `poly` | binary | linear | Modern alt to CE/Focal |
| CE + Center Loss | `ce_center` | binary | linear | Pull same-class trong feature space |
| Supervised Contrastive | `supcon` | multi_128 | projection | Tách cluster (char×ok/ng) |
| ArcFace | `arcface` | multi_128 | arcface | Angular margin, structured embedding |

## Adding a new model / loss / config

### Thêm 1 backbone mới — KHÔNG cần đụng code

```bash
# CLI override
python scripts/train.py --config configs/ce_baseline.yaml \
    --override model.backbone=<bất_kỳ_timm_name> --name ce_<short>

# Hoặc tạo config mới
cp configs/ce_baseline.yaml configs/ce_my_backbone.yaml
# Sửa 2 dòng: experiment_name + model.backbone
```

### Thêm 1 loss mới

1. Thêm `class MyLoss(nn.Module)` trong `src/losses.py`
2. Thêm 1 `if loss_type == "my_loss":` branch trong `build_loss()`
3. Thêm field cần thiết trong `_common.yaml` (vd: `my_loss_alpha: 1.0`)
4. Tạo config `configs/my_loss.yaml` với `loss.type: my_loss`

### Thêm 1 head mới

1. Thêm `class MyHead(nn.Module)` trong `src/models.py`
2. Thêm branch trong `Model.__init__` và `Model.forward`
3. Thêm branch trong `build_model()`

## Audit & reproduce

Mỗi run lưu `config.yaml` (đã merge override) + `env.json` (git commit, package versions).
Reproduce 1 run đã chạy:

```bash
python scripts/train.py --config runs/<run_dir>/config.yaml --name reproduce_<orig>
```

Override sẽ được áp lại đúng vì config đã frozen.

## Lưu ý

- `data.task=binary` → 2 class (OK/NG); cần `head.type=linear`
- `data.task=multi_128` → ~132 class (66 chars × 2); chọn `head.type=projection` (SupCon) hoặc `arcface` hoặc `linear` (multi-class CE)
- Training metric losses (SupCon/ArcFace) tốn nhiều epoch hơn CE (50 vs 30)
- Two-view (`train.use_two_view=true`) gấp đôi memory; bắt buộc cho SupCon
- AMP (`train.amp=true`) chỉ work trên CUDA; auto-disabled trên CPU/MPS
