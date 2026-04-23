# ML Training Studio

## Architecture
- Page: `frontend-ts/src/components/ml-training/` (3 tabs: Images, Label, Train)
- BE service: `backend/app/services/ml_training_service.py`
- BE endpoints: `backend/app/api/endpoints/ml_training.py`
- FE service: `frontend-ts/src/services/mlTraining.ts`

## Key Features (completed)
- Project CRUD with image upload
- Label tab: draw rectangle annotations (OK/NG) on images
- Train tab: 2-column layout (60% Labeled Crops / 40% Results)
  - Left: Real Data + Synthetic tabs, OK/NG filter, lazy loading (IntersectionObserver)
  - Synthetic preview: POST /ml/preview-synthetic
  - Right: Model selectbox (history), Metrics tab + Test Set tab
  - OK Threshold: slider+number (0-100%), stored in model params, affects all evaluation
- Algorithms: RF, SVM, MLP
- Test set: sidecar JSON `{model_id}_test_set.json` saved during training

## API Endpoints
- `GET /ml/projects` → list projects
- `POST /ml/projects` → create project
- `GET /ml/projects/{id}/models` → list trained models
- `POST /ml/train` → train model (FormData: project_id, algorithm, augment_ng, augment_multiplier, threshold)
- `POST /ml/predict` → predict single crop (FormData: project_id, file, model_id?)
- `POST /ml/preview-synthetic` → preview augmented NG crops
- `GET /ml/models/{model_id}/test-set-crops` → get test set crops from sidecar JSON

## Integration with Recipe
- Recipe stores `ml_project_id` and `ml_model_id` to associate a trained model
- RecipeFormModal Model tab: select project → select completed model
- Models filtered to `status === 'completed'` in the dropdown
