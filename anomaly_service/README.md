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
# torch first, matching this machine's CUDA driver (verified with cu121):
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
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
- [x] Train (PatchCore/Padim via anomalib Engine) + live log — end-to-end
      verified on GPU (fit → predict → metrics), including a real training
      bug fix (`val_split_mode` must be `SAME_AS_TEST`, not `NONE` —
      anomalib's Folder datamodule never populates `val_data` for `NONE`
      and Lightning's fit loop still requires it, so training crashes;
      `SAME_AS_TEST` reuses the test set for the unused val loop instead of
      siphoning images out of it)
- [x] Test/Eval (image AUROC/F1, per-image scores, threshold recompute) —
      verified against the full imported test set (not a silently-halved one)
- [x] Export ONNX + TensorRT-verify (build/cache engine on this machine) —
      verified with real onnxruntime inference. Note: the TensorRT provider
      needs the TensorRT SDK's shared libraries (`libnvinfer.so.10` etc.)
      installed on the machine, separate from the `onnxruntime-gpu` pip
      package — `ai_services` already depends on this today for
      `wrinkle_segmenter.py`, so the target workstation should already have
      it; `/verify-tensorrt` reports `active_provider` so a silent fallback
      to CUDA is visible rather than silently accepted as "TensorRT works".
- [ ] Live runtime integration in `ai_services` — Week 3
