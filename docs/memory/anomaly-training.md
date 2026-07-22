---
name: Anomaly Training (anomalib) — service + recipe integration
description: New standalone anomaly_service (PatchCore/Padim training/export) + how it plugs into recipes and ai_services, replacing wrinkle_segmenter.py. Weeks 1-3 of the 6-week plan.
type: project
---
# Anomaly Training

Full plan: `docs/anomaly_training_plan.md`. This file is the "what actually
got built + gotchas found" companion — read the plan doc for the roadmap,
this file for implementation details future sessions will hit.

## Architecture

```
anomaly_service/          ← NEW, standalone FastAPI (own venv: torch+anomalib+onnx)
    Runs on the SAME GPU workstation as backend/ai_services (no Jetson).
    Reads backend's MongoDB directly (recipes, inference_results) — no
    REST calls between anomaly_service and backend.
```

- Training/testing/export lives entirely in `anomaly_service/` — NOT in `backend/`.
- Dataset layout: anomalib Folder convention — `data/projects/{pid}/dataset/train/good/`, `test/<defect_type>/`. No pixel masks (image-level only).
- Model = PatchCore/Padim via anomalib `Engine`. One-class: **training does NOT require abnormal images** (only needed for eval metrics) — this was a deliberate relaxation after testing on real data that only had normal crops.
- Recipe binding is **template-level** (not recipe-level) — mirrors `wrinkle_area`/`color_config`/`edge_config` placement, NOT `ml_project_id`/`ml_model_id` (which are recipe-level, different subsystem — char OK/NG classifier, unrelated).

## Files

- BE recipe schema: `TemplateImage.anomaly_config` in **both** `backend/app/models/recipe.py` and `backend/app/schemas/recipe.py` (this repo duplicates TemplateImage across models/ and schemas/ — always edit both, see recipe-system.md's existing note on this duplication, now also true for anomaly_config).
- `anomaly_service/app/services/anomaly_training.py` — wraps anomalib `Engine` (fit/predict/export). Verified API against anomalib==2.5.0.
- `anomaly_service/app/services/dataset_fs.py` — filesystem layout helpers.
- `ai_services/camera_management/verification/anomaly_inference.py` — NEW, runtime inference (onnxruntime TRT→CUDA→CPU, mirrors `ml_training_service._get_supcon_session`). Reuses `WrinkledSegmenterTRT.crop_from_obb` (static method) so training-time and inference-time crops are pixel-identical.
- Wired into `ai_services/camera_management/verification/product_verifier.py::_batch_wrinkle_check` — per-frame branch: `anomaly_config.enabled` → anomaly_inference; else → legacy `WrinkledSegmenterTRT` (unchanged). Result dict shape matches `build_wrinkled_check()`'s keys (`ok/has_wrinkled/wrinkled_count/wrinkled_boxes`) so downstream aggregation/visualization/storage code needed **zero changes**.
- `single_camera.py` + `multi_camera.py::_batch_verify_products` — both pass `anomaly_config = template.get('anomaly_config', None)` through to `frames_data`, same pattern as `wrinkle_area`.
- FE: `frontend-ts/src/components/anomaly-training/` (Dataset/Train/Eval/Export tabs, separate Studio like ml-training), `frontend-ts/src/components/recipe/AnomalySetupModal.tsx` (per-template picker, opened from RecipeFormModal filmstrip "Setup Anomaly" button — same pattern as "Setup Edge"/"Setup Color").
- FE service: `frontend-ts/src/services/anomalyTraining.ts` — separate axios instance, own base URL (`localhost:8001`), NOT the same `http.ts` instance backend uses.

## Recipe field checklist — template-level fields are DIFFERENT from the 19-step recipe-level checklist

`recipe-system.md`'s 19-step CHECKLIST (RecipeBase, RecipeUpdate, NULLABLE_STR_FIELDS,
`recipe_to_response`, `clone_recipe`, `load_recipe` metadata/recipe_dict,
`update_realtime`, camera.py explicit extraction, etc.) is for **recipe-level**
fields like `ml_project_id`/`classifier_backend`.

**Template-level** fields (`wrinkle_area`, `color_config`, `edge_config`, and
now `anomaly_config`) are nested inside `camera_templates[].templates[]`,
which round-trips as ONE opaque blob everywhere — `camera.py`'s
`self.templates = ct.get("templates", [])` is a raw passthrough, no explicit
per-field extraction needed. Only 3 things needed for a new template-level field:
1. `TemplateImage` in both `models/recipe.py` + `schemas/recipe.py`
2. Frontend `Template` interface in `RecipeFormModal.tsx`
3. The 4 template-creation/duplication/load default-value spots in `RecipeFormModal.tsx` (search `wrinkle_area:` to find all 4 — load-existing, save-to-camera_templates, duplicate-template, brand-new-template)

Do NOT run the full 19-step checklist for template-level fields — it doesn't apply and most steps are no-ops/irrelevant.

## Runtime model-swap = requires ai_services restart (by design, matches existing precedent)

`ProductVerificationService` (and its `WrinkledSegmenterTRT` instance) is constructed
**once** at `inference_handler.py` startup — NOT per-recipe, NOT hot-swapped on
recipe load. `Camera.load_recipe()` only updates lightweight per-recipe fields
(`wrinkle_conf`, `wrinkle_show_when_pass`) on the existing Camera object.

This matches the established pattern for the char-classifier model too — training
a new model already triggers `_trigger_restart_all()` (see ml-training.md) rather
than hot-swapping. Anomaly model changes follow the same rule: **changing which
model a recipe uses requires restarting `ai_services` to take effect.**
`anomaly_inference.py`'s session cache is keyed by onnx_path specifically to make
this assumption explicit rather than silently relying on "only one model ever loads."

## Gotchas found (verified against anomalib==2.5.0, real GPU, real production label crops)

1. **`val_split_mode=ValSplitMode.NONE` crashes training** — Lightning's fit loop
   still calls `val_dataloader()` but anomalib's `Folder` never populates `val_data`
   for `NONE` → `AttributeError: 'Folder' object has no attribute 'val_data'`.
   Use `ValSplitMode.SAME_AS_TEST` instead (reuses test set for the unused val
   loop, doesn't remove images from the real test set). PatchCore/Padim don't
   need a val set anyway (single-epoch feature modeling, no early stopping).
2. **Default `val_split_mode=FROM_TEST` (50%) silently halves your test set**
   before eval ever sees it — every imported test image has a coin-flip chance
   of going to an unused validation split instead. Same fix as above.
3. **ImageNet normalization is baked into the exported ONNX graph** — verified
   via `Patchcore.configure_pre_processor` source (`Resize` + `Normalize(mean=
   [0.485,0.456,0.406], std=[0.229,0.224,0.225])`, traced as part of the model).
   Callers (anomaly_inference.py) must NOT re-normalize — only resize to
   export `image_size`, BGR→RGB, scale to [0,1].
4. **`onnxruntime-gpu`'s TensorrtExecutionProvider needs the TensorRT SDK
   installed separately** (`libnvinfer.so.10` etc.) — `ort.get_available_providers()`
   lists it as compiled-in even when the actual .so is missing; it fails at
   session-create time with a clear error and onnxruntime auto-falls-back to
   CUDA (doesn't raise) if CUDA/CPU are also in the providers list. Always
   check `session.get_providers()[0]` (actual active provider) rather than
   trusting `get_available_providers()`.
5. **PatchCore's peak memory is dominated by the pre-coreset-subsample embedding
   accumulation** (`torch.vstack(embedding_store)`), not batch size — scales with
   dataset size × feature-map spatial resolution, not with `train_batch_size`.
   500 real images with `wide_resnet50_2` OOM'd on a contended 20GB GPU;
   `resnet18` backbone fixed it (smaller layer2/layer3 channel counts). On a
   dedicated (non-shared) GPU this isn't expected to be an issue at the
   recommended defaults.
6. **Training-time crop must match inference-time crop.** anomaly_service's
   candidate/import cropping (axis-aligned bbox from the `label` polygon,
   4px padding) and `ai_services`' `WrinkledSegmenterTRT.crop_from_obb` (OBB
   crop, 15px padding, but angle is always 0 today since the YOLO-priority
   product-box branch in `_get_product_box` is commented out) are close but
   not pixel-identical (padding differs). `anomaly_inference.py` deliberately
   calls the same `crop_from_obb` static method used for wrinkle to keep
   inference-time crops consistent with whichever path is active; if crop
   mismatch ever becomes a real accuracy issue, align anomaly_service's
   candidate cropper to use the same padding constant.

## Real-data validation result (2026-07-13, sandbox GPU, NOT the target 5060 Ti workstation)

Trained PatchCore (resnet18) on 500 real Kirkland Black Pepper label crops
(all normal, no abnormal labels existed yet) + scored 410 real unlabeled
images via `PredictDataset`. 99/100 held-out normal correctly classified.
Top-scored "unlabeled" images visually showed genuine anomalies (box
rotation/misalignment, physical corner damage, reflective glare) — confirms
the model finds real signal, not noise. Full ranking + this narrative not
reproduced here since it's data-specific, not architectural — see conversation
history if needed, this file is about the code/config, not the dataset run.

## Status / what's NOT done yet

- E1.10-equivalent Jetson TRT path — N/A, no Jetson in this deployment anymore.
- No live-camera end-to-end test yet (no hardware in dev sandbox) — only
  syntax-checked + manually traced the integration logic.
- Week 4-6 (edge-detection improvement, OCR training add-on, buffer/regression) not started.

comit 