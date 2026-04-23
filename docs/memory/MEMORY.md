# OCR DateCode Project Memory

## Project Structure
- **Backend**: FastAPI + MongoDB (Pydantic v2 models)
- **Frontend**: React + TypeScript (Vite, `@/` alias)
- **AI Service**: Separate service, communicates via WebSocket from BE

## Key Files
- See [recipe-system.md](recipe-system.md) for recipe data flow & common pitfalls
- See [ml-training.md](ml-training.md) for ML Training Studio details

## Common Pitfalls (IMPORTANT)
1. **FE recipe transform drops new fields**: `Receipts.tsx` has 3 manual `transformedReceipts` mappings (load, search, clone). When adding new recipe fields, ALL 3 must be updated.
2. **Types must be updated in 2 places**: Both `Recipe` and `Receipt` interfaces in `frontend-ts/src/types/index.ts`
3. **BE recipe models are duplicated**: `backend/app/models/recipe.py` AND `backend/app/schemas/recipe.py` — both need updating for new fields
4. **clone_recipe was missing fields**: `reject_pulse` and `normal_pulse_ms` were missing. Always check clone when adding new fields.
5. **recipe_to_response must include new fields**: Uses explicit field mapping, not auto-serialization

## User Preferences
- Vietnamese communication preferred
- Wants thorough field mapping verification across BE/FE
- Prefers analyzing requirements before coding
