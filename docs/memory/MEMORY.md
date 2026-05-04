# OCR DateCode Project Memory

## Project Structure
- **Backend**: FastAPI + MongoDB (Pydantic v2 models)
- **Frontend**: React + TypeScript (Vite, `@/` alias)
- **AI Service**: Separate service, communicates via WebSocket from BE

## Key Files
- See [recipe-system.md](recipe-system.md) for recipe data flow & common pitfalls
- See [ml-training.md](ml-training.md) for ML Training Studio details
- See [python_env.md](python_env.md) — Python venv path `/Users/ngocthien.ai/envs/event/`

## Common Pitfalls (IMPORTANT)
1. **Adding a new recipe field hits ~19 places** — see CHECKLIST in `recipe-system.md`. Most-missed: `clone_recipe`, `load_recipe` metadata + recipe_dict, `update_realtime` recipe_dict, AI service `camera.py:1118+`.
2. **FE recipe transform drops new fields**: `Receipts.tsx` has 3 manual `transformedReceipts` mappings (load, search, clone). When adding new recipe fields, ALL 3 must be updated.
3. **Types must be updated in 2 places**: Both `Recipe` and `Receipt` interfaces in `frontend-ts/src/types/index.ts`
4. **BE recipe models are duplicated**: `backend/app/models/recipe.py` AND `backend/app/schemas/recipe.py` — both need updating for new fields
5. **`recipe_to_response` uses explicit field mapping**: not auto-serialization. Forgot field → silently missing in API response.
6. **Per-recipe classifier routing**: `classifier_backend` ('embedding' | 'ml') replaces old global `CHAR_CLASSIFIER_BACKEND`. AI service routes per-recipe via `camera.classifier_backend`.

## User Preferences
- Vietnamese communication preferred
- Wants thorough field mapping verification across BE/FE
- Prefers analyzing requirements before coding
