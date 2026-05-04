# Recipe System — Data Flow & Architecture

## Recipe Fields (as of 2026-04-23)
Standard: `name, product_code, description, delay_reject, reject_pulse, reject_method, do_reject_number, do_alarm_number, normal_pulse_ms, cameras, camera_templates, camera_settings, model_thresholds, template_config, roi_config, is_active`
New ML fields: `ocr_model_type, ml_project_id, ml_model_id, defect_model`

## Defect Model (per-recipe embedding picker)
- Field: `defect_model: 'arcface' | 'supcon'` (default `'arcface'`)
- Routes which `EmbeddingClassifierService` instance handles char OK/NG check
- BE registry built in `inference_handler.py`: `self.embedding_classifier_services = {'arcface': ..., 'supcon': ...}`
- `text_verifier._run_char_batch()` reads `defect_model` from each char_item (carried via `camera.defect_model` set by `camera.py:1138`)
- `CHAR_CLASSIFIER_BACKEND` constant in `text_verifier.py:50` still gates embedding-vs-ML routing globally — `defect_model` only picks **which embedding** within the embedding branch

## OCR Model Types
- `SMTR` (large-x)
- `SVTRV2_CTC` (large)
- `OPENOCR_REPSVTR` (medium)
- `PADDLEV5` (small)

## Data Flow: Save → Load → Edit

### 1. Save (Create/Update)
- FE `RecipeFormModal.tsx` → `handleSubmit()` → `onSubmit(submitData)` → API
- `submitData` = `{...formData, camera_templates: cameraTemplatesArray}`
- BE: `POST /api/recipes/` (create) or `PUT /api/recipes/{id}` (update)
- BE validates via `RecipeCreate` / `RecipeUpdate` Pydantic models

### 2. Load Recipe for AI Service
- **Direct load**: `POST /api/recipes/{recipe_id}/load`
  - Builds `recipe_dict` with ALL recipe fields → sends via WebSocket (`send_load_recipe`)
  - Also saves to `receipt_loads` collection as metadata snapshot
  - **PENDING**: `recipe_dict` at line ~1040 does NOT yet include `ocr_model_type, ml_project_id, ml_model_id`
  - **PENDING**: `metadata` dict at line ~1023 also does NOT include these fields

- **Resume after restart**: `GET /api/recipes/loads/latest`
  - Returns latest running load from `receipt_loads` collection
  - Data comes from the `metadata` snapshot saved during load
  - **PENDING**: If metadata didn't include OCR/ML fields at load time, latest won't have them either

- **Realtime update (no DB save)**: `POST /api/recipes/{recipe_id}/update-realtime`
  - Reads recipe from DB, merges `update_data` in-memory, stops+reloads via WebSocket
  - `recipe_dict` at line ~1333 does NOT yet include `ocr_model_type, ml_project_id, ml_model_id`
  - Currently handles: `delay_reject, reject_pulse, normal_pulse_ms, cameras, camera_templates`

### 3. List/Display
- `GET /api/recipes/` → `list_recipes()` → `recipe_to_response()` → `RecipeResponse`
- `GET /api/recipes/search` → `search_recipes()` → same
- FE `Receipts.tsx` → 3 transform blocks (loadReceipts, handleSearch, clone handler)

### 4. Edit
- FE `Receipts.tsx` → `handleEditReceipt(receipt)` → opens `RecipeFormModal` with `recipe` prop
- `RecipeFormModal.tsx` → `useEffect` on `(recipe, mode, isOpen)` → `setFormData(...)` from `recipeAny`
- **Key**: Uses `recipe as any` cast to bypass type checking

## NEXT TODO — OCR Model Mapping in Load
User wants OCR model type to actually affect which model the AI Service uses. Steps needed:

1. **`load_recipe()` (line ~1040)**: Add to `recipe_dict`:
   ```python
   'ocr_model_type': getattr(recipe, 'ocr_model_type', None),
   'ml_project_id': getattr(recipe, 'ml_project_id', None),
   'ml_model_id': getattr(recipe, 'ml_model_id', None),
   ```

2. **`load_recipe()` metadata (line ~1023)**: Add same 3 fields to `metadata` dict

3. **`update_realtime()` (line ~1333)**: Add same 3 fields to `recipe_dict`

4. **`loads/latest` endpoint**: No code change needed — it returns raw metadata from DB. Just need metadata to include the fields at save time (step 2).

5. **AI Service side**: Map `ocr_model_type` string to actual model class. ML training logic deferred to later.

## File Locations (quick reference)
- BE models: `backend/app/models/recipe.py`
- BE schemas: `backend/app/schemas/recipe.py`
- BE endpoints: `backend/app/api/endpoints/recipes.py`
- FE types: `frontend-ts/src/types/index.ts` (Recipe + Receipt interfaces)
- FE list page: `frontend-ts/src/components/recipe/Receipts.tsx`
- FE form modal: `frontend-ts/src/components/recipe/RecipeFormModal.tsx`
- FE API service: `frontend-ts/src/services/recipes.ts`

## Checklist: Adding a New Recipe Field
1. Add to `RecipeBase` in `backend/app/models/recipe.py`
2. Add to `RecipeUpdate` in `backend/app/models/recipe.py`
3. Add to `RecipeBase` in `backend/app/schemas/recipe.py`
4. Add to `RecipeUpdate` in `backend/app/schemas/recipe.py`
5. Add to `recipe_to_response()` in `recipes.py` endpoint
6. Add to `clone_recipe()` in `recipes.py` endpoint
7. Add to `load_recipe()` recipe_dict + metadata in `recipes.py`
8. Add to `update_realtime()` recipe_dict in `recipes.py`
9. Add to `Recipe` interface in `frontend-ts/src/types/index.ts`
10. Add to `Receipt` interface in `frontend-ts/src/types/index.ts`
11. Add to `FormDataType` in `RecipeFormModal.tsx`
12. Add to `setFormData` (edit mode + create reset) in `RecipeFormModal.tsx`
13. Add to all 3 `transformedReceipts` mappings in `Receipts.tsx`
