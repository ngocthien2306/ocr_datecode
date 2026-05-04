---
name: ML Training Studio
description: Architecture & key flows of the ML Training tab — SupCon embedding + sklearn classifier pipeline (post-2026-05-04 refactor)
type: project
originSessionId: 62c49580-c229-497d-a0d0-965d7d677231
---
# ML Training Studio

## Pipeline (current — post 2026-05-04 refactor)

```
char crop → SupCon ONNX (efficientnet_b2, 128-dim L2-norm) → sklearn RF/SVM/MLP → OK/NG
```

Replaces the old handcrafted v2 pipeline (1016-dim handcrafted + per-char golden templates) entirely.

## Files
- BE service: `backend/app/services/ml_training_service.py`
- BE endpoints: `backend/app/api/endpoints/ml_training.py`
- AI service mirror: `ai_services/camera_management/verification/ml_classifier.py` (must stay in sync)
- FE: `frontend-ts/src/components/ml-training/` (Images, Label, Train tabs)
- FE service: `frontend-ts/src/services/mlTraining.ts`
- SupCon weights: `weights/supcon_128_efficientnet_b2_20260429-073504/model.onnx`

## Bundle format
```python
{'clf': sklearn_classifier, 'char_stats': {...}, 'algorithm': 'rf'|'svm'|'mlp'}
```
No `goldens`, no `feat_version`, no `feat_dim` (single embedding pipeline).

## Removed (do not reintroduce)
- `extract_features` / `extract_features_v2` / `extract_diff_features`
- `compute_golden` / `align_to_golden` / `preprocess_canonical` / `_compute_diff_map`
- `get_model_goldens` (BE service + endpoint + FE API + Goldens tab)
- `MLGoldenItem` type (FE)
- `feat_version`, `feat_dim`, `goldens` keys in bundle

## char_id status
Still kept on segments — used for char-balanced NG augmentation + display stats. NOT used in features. char_id missing → sample still trained, just shown under "No char_id" group.

## NG augmentation (kept)
Char-balanced: `target_per_char = factor × max(n_ok across chars)`. Each char tops up NG to target. OK never augmented.

## Algorithms (FE selectable)
RF / SVM / MLP / Centroid — all run on 128-dim SupCon embedding.

**Centroid** = global mean OK + global mean NG (Strategy B). Bundle saves raw arrays
(not class instance) to avoid joblib unpickle issues across BE/ai_services. See
`CentroidClassifier` (mimics sklearn API for fit/predict_proba parity) +
`_centroid_predict_proba()` for predict path. Temperature default 5.0 (UI knob 1-20).

## Tabs in Result panel
- Metrics
- Test Set
(No more Goldens tab.)

## Backward compat
**Legacy v2 bundles raise ValueError on load.** Old `.joblib` files must be retrained. DB `ml_models` records pointing to legacy files will fail predict/inference — drop manually via `db.ml_models.deleteMany({})` if stale records exist.

## Recipe integration
Recipe stores `ml_project_id` + `ml_model_id` to associate a trained model. RecipeFormModal Model tab → select project → select completed model. `defect_model` field (arcface/supcon) on recipe selects the EmbeddingClassifierService instance for char-classify path (separate from this ML Training pipeline).
