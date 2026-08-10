# OCR DateCode Project Memory

## Project Structure
- **Backend**: FastAPI + MongoDB (Pydantic v2 models)
- **Frontend**: React + TypeScript (Vite, `@/` alias)
- **AI Service**: Separate service, communicates via WebSocket from BE

## Key Files
- See [recipe-system.md](recipe-system.md) for recipe data flow & common pitfalls
- See [ml-training.md](ml-training.md) for ML Training Studio details
- See [color-check-feature.md](color-check-feature.md) — Check_Color (HSV color verification): FE Setup Color modal, AI service color_verifier, OR-aggregation across color cameras, normalized annotation coords gotcha
- See [anomaly-training.md](anomaly-training.md) — new standalone `anomaly_service` (anomalib PatchCore/Padim) + runtime integration replacing `wrinkle_segmenter.py`; anomalib gotchas (val_split_mode, ONNX normalization, TensorRT SDK), template-level vs recipe-level field checklist distinction
- See [ocr-training.md](ocr-training.md) — `ocr_service` on :8002 (fine-tune SVTRv2+SMTR on recipe text/datecode crops → ONNX → TensorRT), its studio UI, and recipe binding via `ocr_project_id`/`ocr_model_id` + `OCRModelType.CUSTOM`. Built and verified end-to-end. Read it for the traps: `main_indicator: acc` silently saves a model with a broken SMTR head, an over-long label makes OpenOCR substitute a *different* image, and pycuda's context is per-thread
- See [python_env.md](python_env.md) — Python venv path `/Users/ngocthien.ai/envs/event/`

## Common Pitfalls (IMPORTANT)
1. **Adding a new recipe field hits ~19 places** — see CHECKLIST in `recipe-system.md`. Most-missed: `clone_recipe`, `load_recipe` metadata + recipe_dict, `update_realtime` recipe_dict, AI service `camera.py:1118+`.
2. **FE recipe transform drops new fields**: `Receipts.tsx` has **2** manual `transformedReceipts` mappings (load + clone) — verified 2026-08-10 while adding `ocr_project_id`/`ocr_model_id`. It used to be 3; `handleSearch` now only sets `activeSearch` and searches server-side, so it has no mapping of its own.
3. **Types must be updated in 2 places**: Both `Recipe` and `Receipt` interfaces in `frontend-ts/src/types/index.ts`
4. **BE recipe models are duplicated**: `backend/app/models/recipe.py` AND `backend/app/schemas/recipe.py` — both need updating for new fields
5. **`recipe_to_response` uses explicit field mapping**: not auto-serialization. Forgot field → silently missing in API response.
6. **Per-recipe classifier routing**: `classifier_backend` ('embedding' | 'ml') replaces old global `CHAR_CLASSIFIER_BACKEND`. AI service routes per-recipe via `camera.classifier_backend`.
7. **Annotation coords are NORMALIZED [0,1]**: `TemplateEditorRefactored.normalize` divides by image bounds. Anything that bypasses SuperPoint (e.g. `color_verifier`) MUST multiply by `image_width`/`image_height` (or `frame.shape[:2]`) before using as pixel coords. See [color-check-feature.md](color-check-feature.md).
8. **Template-level fields (nested in `camera_templates[].templates[]`, e.g. `wrinkle_area`, `color_config`, `edge_config`, `anomaly_config`) do NOT need the 19-step recipe-field checklist** — that checklist is for recipe-level fields only. Template-level fields round-trip as one opaque blob (`camera.py`'s `self.templates = ct.get("templates", [])` is a raw passthrough). Only needs: `TemplateImage` in both `models/recipe.py` + `schemas/recipe.py`, FE `Template` interface, and the 4 default-value spots in `RecipeFormModal.tsx`. See [anomaly-training.md](anomaly-training.md).
9b. **EXCEPTION to #9 — the OCR backend IS hot-swappable per recipe.** `camera_manager.py` calls `inference_handler.set_ocr_model(ocr_model_type, engine_path, dict_path)` and, if it returns changed, submits `_reinit_ocr_backend()` to the worker thread. So #9 applies to the char classifier + anomaly model, NOT to OCR recognition models — including models trained in the OCR Studio. See [ocr-training.md](ocr-training.md).
9. **A trained model swap requires restarting `ai_services`** — models (char classifier, anomaly) are loaded once at process startup, not hot-swapped per recipe. This is deliberate/established (`_trigger_restart_all()` after ML training completes); anomaly integration follows the same rule. See [anomaly-training.md](anomaly-training.md).

## User Preferences
- Vietnamese communication preferred
- Wants thorough field mapping verification across BE/FE
- Prefers analyzing requirements before coding
