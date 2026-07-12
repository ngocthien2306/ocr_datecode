# anomaly_service

Standalone FastAPI service for anomaly-detection training/testing/export
(anomalib: PatchCore/Padim). Separate process from `backend/`, own
dependencies (torch + anomalib get pinned in the Week 1 spike — see
`../docs/anomaly_training_plan.md`).

Runs on the same GPU workstation as `backend` and `ai_services` (no Jetson
involved). Reads `recipes` / `inference_results` directly from the same
MongoDB database backend uses — no REST calls between the two services.

## Setup

```bash
cd anomaly_service
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.sample .env   # then set SECRET_KEY to MATCH backend/.env exactly
```

## Run

```bash
python -m app.main
# or: uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

Health check: `GET /health`. Auth: send the same Bearer token issued by
backend's `/api/auth/login` — this service verifies it against the same
`SECRET_KEY`/`users` collection, it does not issue its own tokens.

## Status (Week 1 of docs/anomaly_training_plan.md)

- [x] Project CRUD (`/api/anomaly/projects`)
- [x] Candidates search — crop the `label` region out of past
      `inference_results` filtered by recipe (`GET /api/anomaly/candidates`)
- [x] Import into dataset (`POST /api/anomaly/projects/{id}/import`) —
      anomalib Folder layout under `data/projects/{id}/dataset/`
- [ ] Train/Test/Eval (anomalib Engine) — Week 2
- [ ] Export ONNX + TensorRT — Week 3
- [ ] Live runtime integration in `ai_services` — Week 3
