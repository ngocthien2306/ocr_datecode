"""
Live template realignment for the realtime screen — CPU-only, backend-local.

Everything the AI service does to line a frame up with its template (cap
rotation, cap crop, text window, SuperPoint + RANSAC) re-done here so
"Get Live Template" returns an image and regions from the SAME frame.

Must never import ai_services / TensorRT / PyCUDA, and must never use a CUDA
execution provider — see onnx_models.py and
docs/DEBUG_SERVICE_CRASH_RUNBOOK.md §6.

Import `realign` from `.realign` explicitly; this __init__ stays empty on
purpose so importing the package loads nothing heavy.
"""
