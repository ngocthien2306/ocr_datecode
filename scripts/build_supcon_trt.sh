#!/usr/bin/env bash
#
# Pre-build TensorRT engine cache for the SupCon embedder used by the AI
# service (ai_services/camera_management/verification/ml_classifier.py).
#
# Why a separate script?
#   The TRT engine build can take several minutes. If left to lazy-init at
#   first inference, the first WebSocket request stalls. The runtime code in
#   ml_classifier.py only enables TensorrtExecutionProvider when the cache
#   directory is already populated; this script populates it offline.
#
# Usage:
#   bash scripts/build_supcon_trt.sh              # build (skip if cache exists)
#   bash scripts/build_supcon_trt.sh --rebuild    # wipe cache and rebuild
#   ML_TRT_FP16=0 bash scripts/build_supcon_trt.sh   # build FP32 engine
#
# Env vars:
#   ML_TRT_FP16   "1" (default) → FP16; "0"/"false" → FP32
#   PYTHON_BIN    override python binary (default: $VIRTUAL_ENV/bin/python or python3)
#
set -euo pipefail

# ── Resolve paths ──────────────────────────────────────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
SUPCON_DIR="$REPO_ROOT/weights/supcon_128_efficientnet_b2_20260429-073504"
ONNX_PATH="$SUPCON_DIR/model.onnx"
CACHE_DIR="$REPO_ROOT/cache/supcon_trt_ai"

# Must match _TRT_PROFILE_{MIN,OPT,MAX} in ml_classifier.py
TRT_PROFILE_MIN="input:1x3x64x64"
TRT_PROFILE_OPT="input:48x3x64x64"
TRT_PROFILE_MAX="input:128x3x64x64"

REBUILD=0
for arg in "$@"; do
    case "$arg" in
        --rebuild) REBUILD=1 ;;
        -h|--help)
            sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $arg" >&2; exit 2 ;;
    esac
done

# ── Sanity checks ──────────────────────────────────────────────────────────
if [[ ! -f "$ONNX_PATH" ]]; then
    echo "ERROR: SupCon ONNX not found at $ONNX_PATH" >&2
    exit 1
fi

if [[ "$REBUILD" -eq 1 && -d "$CACHE_DIR" ]]; then
    echo "[build_supcon_trt] --rebuild: wiping $CACHE_DIR"
    rm -rf "$CACHE_DIR"
fi

mkdir -p "$CACHE_DIR"

# Reuse a populated cache (engine binary + profile) without rebuilding.
if [[ "$REBUILD" -eq 0 ]] && find "$CACHE_DIR" -mindepth 1 -name '*.engine' -print -quit | grep -q .; then
    echo "[build_supcon_trt] cache already populated at $CACHE_DIR (use --rebuild to force)"
    exit 0
fi

# ── Pick python ────────────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
        PYTHON_BIN="$VIRTUAL_ENV/bin/python"
    else
        PYTHON_BIN="$(command -v python3 || command -v python)"
    fi
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: python not found (PYTHON_BIN='$PYTHON_BIN')" >&2
    exit 1
fi

echo "[build_supcon_trt] python   = $PYTHON_BIN"
echo "[build_supcon_trt] onnx     = $ONNX_PATH"
echo "[build_supcon_trt] cache    = $CACHE_DIR"
echo "[build_supcon_trt] fp16     = ${ML_TRT_FP16:-1}"
echo "[build_supcon_trt] profile  = min=$TRT_PROFILE_MIN opt=$TRT_PROFILE_OPT max=$TRT_PROFILE_MAX"
echo "[build_supcon_trt] building TRT engine — this may take several minutes…"

# ── Build via onnxruntime TRT provider ─────────────────────────────────────
ONNX_PATH="$ONNX_PATH" \
CACHE_DIR="$CACHE_DIR" \
TRT_PROFILE_MIN="$TRT_PROFILE_MIN" \
TRT_PROFILE_OPT="$TRT_PROFILE_OPT" \
TRT_PROFILE_MAX="$TRT_PROFILE_MAX" \
ML_TRT_FP16="${ML_TRT_FP16:-1}" \
"$PYTHON_BIN" - <<'PYEOF'
import os
import sys
import time
import numpy as np
import onnxruntime as ort

onnx_path = os.environ["ONNX_PATH"]
cache_dir = os.environ["CACHE_DIR"]
trt_fp16  = os.environ.get("ML_TRT_FP16", "1") not in ("0", "false", "False")

available = set(ort.get_available_providers())
if "TensorrtExecutionProvider" not in available:
    print(f"[build_supcon_trt] ERROR: TensorrtExecutionProvider not available. "
          f"Have: {sorted(available)}", file=sys.stderr)
    sys.exit(1)

sess_options = ort.SessionOptions()
sess_options.enable_mem_pattern = False
sess_options.enable_cpu_mem_arena = False
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
sess_options.log_severity_level = 2  # show TRT build progress (1=verbose, 2=info, 3=warning)

providers = [
    ("TensorrtExecutionProvider", {
        "trt_engine_cache_enable":  True,
        "trt_engine_cache_path":    cache_dir,
        "trt_fp16_enable":          trt_fp16,
        "trt_max_workspace_size":   256 * 1024 * 1024,
        "trt_profile_min_shapes":   os.environ["TRT_PROFILE_MIN"],
        "trt_profile_opt_shapes":   os.environ["TRT_PROFILE_OPT"],
        "trt_profile_max_shapes":   os.environ["TRT_PROFILE_MAX"],
    }),
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]

t0 = time.perf_counter()
sess = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
build_ms = (time.perf_counter() - t0) * 1000

active = sess.get_providers()[0]
print(f"[build_supcon_trt] session created on {active} in {build_ms:.0f}ms")

if active != "TensorrtExecutionProvider":
    print(f"[build_supcon_trt] WARNING: TRT not active (got {active}); "
          "engine was not built", file=sys.stderr)
    sys.exit(2)

# Warm up at min / opt / max batch sizes so all profile shapes are realized
# before the runtime sees its first request.
input_name = sess.get_inputs()[0].name
for bs in (1, 48, 128):
    x = np.random.randn(bs, 3, 64, 64).astype(np.float32)
    t = time.perf_counter()
    out = sess.run(None, {input_name: x})[0]
    print(f"[build_supcon_trt] warmup batch={bs:>3} → out{out.shape} "
          f"in {(time.perf_counter()-t)*1000:.1f}ms")

print("[build_supcon_trt] done")
PYEOF

# ── Summary ────────────────────────────────────────────────────────────────
echo
echo "[build_supcon_trt] cache contents:"
ls -lh "$CACHE_DIR"
