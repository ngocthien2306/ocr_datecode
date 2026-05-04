---
name: Recipe System — Data Flow & Architecture
description: How recipes propagate from FE save → DB → AI Service via load/latest/update-realtime/clone, plus the per-recipe classifier routing scheme
type: project
originSessionId: 62c49580-c229-497d-a0d0-965d7d677231
---
# Recipe System

## Recipe Fields (verified 2026-05-05)

### Core
`name, product_code, description, delay_reject, reject_pulse, reject_method, do_reject_number, do_alarm_number, normal_pulse_ms, cameras, camera_templates, camera_settings, model_thresholds, template_config, roi_config, is_active`

### ML / OCR
`ocr_model_type, ml_project_id, ml_model_id, defect_model, classifier_backend`

### Field meanings
- **`ocr_model_type`** `'SMTR' | 'SVTRV2_CTC' | 'OPENOCR_REPSVTR' | 'PADDLEV5'` — OCR backbone for text/datecode reading
- **`classifier_backend`** `'embedding' | 'ml'` (default `'embedding'`) — **Active Method** for char OK/NG verify
  - `'embedding'` → cosine sim between `template_crop` and `target_crop` via SupCon/ArcFace ONNX
  - `'ml'`        → trained sklearn/centroid bundle from ML Training Studio
- **`defect_model`** `'arcface' | 'supcon'` (default `'arcface'`) — picks which embedding ONNX (only used when `classifier_backend='embedding'`)
- **`ml_project_id` + `ml_model_id`** — point to bundle in `public/ml_projects/{pid}/models/{mid}.joblib`. Required when `classifier_backend='ml' AND recipe has char bboxes`

## Per-recipe classifier routing (no global flag — replaces old `CHAR_CLASSIFIER_BACKEND`)

| Recipe state | Behavior at inference |
|---|---|
| `classifier_backend='embedding'` (default) | `defect_model` picks embedding service → cosine sim |
| `classifier_backend='ml'` + valid `ml_project_id`/`ml_model_id` | `MLClassifierService.classify_batch` loads bundle (sklearn or centroid) → `predict_proba` |
| `classifier_backend='ml'` + missing ml_model | BE logs warning + skips char verify (`use_char_task=False`); OCR still runs |
| Recipe has only `text/datecode` bboxes (no `char`) | classifier never invoked (char_items list empty) — backend choice irrelevant |

FE blocks save with **ConfirmDialog popup** when `backend='ml' + recipeChars > 0 + missing ml_*` → button "Confirm" jumps to Model tab.

## Routing code points
- BE: `text_verifier._build_items_for_camera` reads `camera.classifier_backend` (set in `camera.py:1139`); branches `use_char_task` gate per backend
- BE: `text_verifier._run_char_batch` reads `char_items[0]['classifier_backend']`; routes to `embedding_classifier_services[defect_model]` or `ml_classifier_service`
- BE: `inference_handler.py` builds `embedding_classifier_services = {'arcface': ..., 'supcon': ...}` registry
- AI service: `MLClassifierService` handles 2 bundle shapes — sklearn `{'clf': ...}` or centroid `{'centroid_ok', 'centroid_ng', 'temperature'}`

## Data Flow paths (where new fields MUST appear)

### Save (Create/Update)
- FE `RecipeFormModal.handleSubmit` → `submitData = {...formData, camera_templates}`
- BE: `POST /api/recipes/` (create) → `RecipeCreate` Pydantic
- BE: `PUT /api/recipes/{id}` (update) → `RecipeUpdate` Pydantic
- DB: `recipes.create(...)` calls `_normalize_empty_strings` (NULLABLE_STR_FIELDS includes new fields)

### Load to AI Service — POST `/api/recipes/{id}/load`
**Two dicts populated** (recipes.py around line 1135 + 1156):
1. `metadata` dict → saved to `receipt_loads` collection (used by `/loads/latest` resume)
2. `recipe_dict` → `send_load_recipe(recipe_dict)` → WebSocket → CameraManagement service → `Camera.load_recipe(recipe_data)`

→ Both dicts MUST include any new field that AI Service needs.

### Resume after AI restart — GET `/api/recipes/loads/latest`
- Returns raw `metadata` from DB unchanged
- AI service: `camera_management_service.py:418` calls this; uses `metadata` directly as `recipe_data`
- → Field availability depends on `metadata` dict at save time (load_recipe step above)

### Realtime update — POST `/api/recipes/{id}/update-realtime`
- Reads recipe from DB → builds `recipe_dict` (recipes.py around line 1454) → merges `update_data` in-memory → `send_load_recipe(recipe_dict)` (re-load)
- → `recipe_dict` MUST include any new field

### Clone — POST `/api/recipes/{id}/clone`
- recipes.py around line 990: `RecipeCreate(name=..., ..., field=getattr(original_recipe, 'field', None))`
- → MUST add new field as kwarg here

### List/Display
- `GET /api/recipes/` + `/search` → `recipe_to_response()` → `RecipeResponse`
- FE `Receipts.tsx` has 3 transform blocks (`loadReceipts`, `handleSearch`, `clone handler`) that map BE response → frontend `Receipt` shape

### Edit
- FE `Receipts.tsx.handleEditReceipt(receipt)` → opens `RecipeFormModal` with `recipe` prop
- `RecipeFormModal` `useEffect` on (recipe, mode, isOpen) → `setFormData(...)` from `recipeAny`

## CHECKLIST — Adding a New Recipe Field

1. ✅ `RecipeBase` in `backend/app/models/recipe.py`
2. ✅ `RecipeUpdate` in `backend/app/models/recipe.py`
3. ✅ `RecipeBase` in `backend/app/schemas/recipe.py`
4. ✅ `RecipeUpdate` in `backend/app/schemas/recipe.py`
5. ✅ `NULLABLE_STR_FIELDS` in `backend/app/repositories/recipe_repository.py` (if string field)
6. ✅ `recipe_to_response()` in `recipes.py` endpoint (kwarg)
7. ✅ `clone_recipe()` in `recipes.py` endpoint — `RecipeCreate(... field=getattr(original_recipe, 'field', None))`
8. ✅ `load_recipe()` `metadata` dict in `recipes.py` (~line 1135)
9. ✅ `load_recipe()` `recipe_dict` in `recipes.py` (~line 1156)
10. ✅ `update_realtime()` `recipe_dict` in `recipes.py` (~line 1454)
11. ✅ `Recipe` interface in `frontend-ts/src/types/index.ts`
12. ✅ `Receipt` interface in `frontend-ts/src/types/index.ts`
13. ✅ `FormDataType` in `RecipeFormModal.tsx`
14. ✅ `setFormData` initial state in `RecipeFormModal.tsx`
15. ✅ `setFormData` edit-load (when recipe prop exists)
16. ✅ `setFormData` create-reset (mode='create')
17. ✅ submit payload assembly in `handleSubmit`
18. ✅ All 3 `transformedReceipts` mappings in `Receipts.tsx` (load + search + clone)
19. ✅ AI service `camera.py` `Camera.load_recipe()` → `self.field = recipe_data.get(...) or default` (if AI uses field at runtime)

→ **All 19 steps verified for `defect_model` and `classifier_backend` (2026-05-05).**

## File Locations
- BE models: `backend/app/models/recipe.py`
- BE schemas: `backend/app/schemas/recipe.py`
- BE endpoints: `backend/app/api/endpoints/recipes.py` (~1700 lines)
- BE repo: `backend/app/repositories/recipe_repository.py`
- AI service camera loader: `ai_services/camera_management/camera.py:1118+`
- AI service routing: `ai_services/camera_management/verification/text_verifier.py`
- AI service ML classifier: `ai_services/camera_management/verification/ml_classifier.py`
- AI service embedding: `ai_services/camera_management/verification/embedding_classifier.py`
- AI service inference handler: `ai_services/camera_management/inference_handler.py` (registry init)
- FE types: `frontend-ts/src/types/index.ts`
- FE list page: `frontend-ts/src/components/recipe/Receipts.tsx`
- FE form modal: `frontend-ts/src/components/recipe/RecipeFormModal.tsx`
- FE API service: `frontend-ts/src/services/recipes.ts`
