#!/bin/bash
# Install ONLY the OCR Training API unit (ocr_service, :8002).
#
#   bash scripts/setup_ocr_training_service.sh              # install + start
#   bash scripts/setup_ocr_training_service.sh --uninstall  # remove it
#
# Deliberately separate from setup_systemd_services.sh: that script deletes and
# recreates ocr-backend + ocr-all and then restarts the whole stack (camera check
# included), which is far too disruptive to run just to add one service. This
# touches nothing but ocr-training.service.
#
# ocr_service is also NOT in start_services.sh on purpose. It shares only
# MongoDB with the camera pipeline, and a training run should not be taken down
# by a camera restart.

set -e

MODE="install"
[ "${1:-}" = "--uninstall" ] && MODE="uninstall"

# Resolve the SERVICE user, not the effective one — same reasoning as
# setup_systemd_services.sh: this script sudo's for the few root steps, so under
# `sudo bash ...` $HOME/whoami would become root's and point everything at
# /root/Source/ocr_datecode.
USER_NAME="${SUDO_USER:-$(whoami)}"
USER_HOME=$(getent passwd "$USER_NAME" | cut -d: -f6)
if [ -z "$USER_HOME" ] || [ ! -d "$USER_HOME" ]; then
    echo "❌ Could not resolve home directory for user '${USER_NAME}'"
    exit 1
fi
PROJECT_DIR="${USER_HOME}/Source/ocr_datecode"
LOG_DIR="${PROJECT_DIR}/logs"
CONDA_SH="${USER_HOME}/miniconda3/etc/profile.d/conda.sh"
OCR_DIR="${PROJECT_DIR}/ocr_service"
UNIT=/etc/systemd/system/ocr-training.service

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " ocr-training.service — ${MODE}"
echo " User    : $USER_NAME"
echo " Project : $PROJECT_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$MODE" = "uninstall" ]; then
    sudo systemctl stop    ocr-training.service 2>/dev/null || true
    sudo systemctl disable ocr-training.service 2>/dev/null || true
    sudo rm -f "$UNIT"
    sudo systemctl reset-failed ocr-training.service 2>/dev/null || true
    sudo systemctl daemon-reload
    echo "✅ Removed ocr-training.service (nothing else touched)"
    exit 0
fi

# ── Validate BEFORE writing anything ─────────────────────────────────────────
echo "Validating..."
OCR_PY="${USER_HOME}/miniconda3/envs/ocr_train/bin/python"
if [ ! -x "$OCR_PY" ]; then
    echo "❌ ocr_train env not found at ${OCR_PY}"
    echo "   Create it:  conda create -n ocr_train --clone anomaly_service -y"
    echo "   then:       pip install lmdb pyclipper rapidfuzz tqdm pyyaml imgaug \\"
    echo "                           albumentations onnxscript pycuda 'numpy==1.26.4'"
    exit 1
fi
[ -f "$CONDA_SH" ]        || { echo "❌ conda.sh not found at ${CONDA_SH}"; exit 1; }
[ -d "${OCR_DIR}/app" ]   || { echo "❌ ${OCR_DIR}/app missing — wrong PROJECT_DIR?"; exit 1; }
if [ ! -f "${OCR_DIR}/.env" ]; then
    echo "❌ ${OCR_DIR}/.env missing — cp .env.sample .env and set SECRET_KEY to"
    echo "   MATCH backend/.env exactly, or every endpoint answers 401."
    exit 1
fi
# uvicorn lives in the env, not the base interpreter — check the actual import.
"$OCR_PY" -c "import uvicorn, fastapi, motor" 2>/dev/null \
    || { echo "❌ ocr_train env is missing uvicorn/fastapi/motor"; exit 1; }
echo "   ✅ ocr_train env, ocr_service/app, .env all present"

# Not fatal: the label-review half of the studio works without OpenOCR or the
# base checkpoints, and refusing to install would block that.
[ -d "${OCR_DIR}/OpenOCR" ] \
    || echo "   ⚠️  ${OCR_DIR}/OpenOCR missing — TRAINING will fail (see ocr_service/README.md)"
ls "${OCR_DIR}"/weights/base/*.pth >/dev/null 2>&1 \
    || echo "   ⚠️  no base checkpoints in ${OCR_DIR}/weights/base — training will fail"
echo ""

mkdir -p "$LOG_DIR"

echo "Writing ${UNIT}..."
sudo tee "$UNIT" > /dev/null << EOF
[Unit]
Description=OCR Datecode OCR Training API (:8002)
After=network.target mongod.service
Wants=mongod.service

[Service]
User=${USER_NAME}
Group=${USER_NAME}
WorkingDirectory=${OCR_DIR}
# uvicorn directly, NOT \`python -m app.main\`: that entry point passes
# reload=settings.DEBUG and .env ships DEBUG=True, so a reload watcher under
# systemd would restart the process on any file touch and kill a training run
# that was minutes in.
#
# WorkingDirectory matters more than it looks: the repo ROOT also has an
# app/main.py (an unrelated PyQt5 tool), so a wrong cwd imports the wrong app.
ExecStart=/bin/bash -lc 'source ${CONDA_SH} && conda activate ocr_train && exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8002'
Restart=always
RestartSec=5
# Long enough for the OpenOCR trainer subprocess and a TensorRT context to
# unwind on stop.
TimeoutStopSec=60
#
# No KillMode override, unlike ocr-backend which needs 'mixed'. Training spawns
# train_rec.py with a plain subprocess.Popen: on Linux that child is NOT tied to
# its parent's lifetime, and its CUDA memory belongs to it, not to us. Verified:
# kill the parent and the child keeps running while the GPU flock IS released
# (Popen defaults to close_fds=True, so the child never inherited the lock fd).
# That combination is the dangerous one — the next run sees a free lock and
# starts a SECOND trainer alongside the orphan. The default control-group kill
# takes the trainer down with the unit, which is what we want.
#
# PrivateTmp is intentionally absent: the cross-service GPU lock is
# /tmp/ocr_datecode_gpu.lock and must be the same file anomaly_service and any
# manual run see. PrivateTmp=yes would give this unit a private /tmp and
# silently disable the mutual exclusion.
#
# StartLimitIntervalSec/StartLimitBurst are also absent. ocr-backend.service has
# them in [Service], where systemd ignores them (systemd-analyze verify:
# "Unknown key name"), so they have never limited anything there. Putting them
# in [Unit] would make them real and let this unit give up after N restarts —
# for an operator-facing API, silently staying down is worse than retrying.
StandardOutput=append:${LOG_DIR}/ocr_service.log
StandardError=append:${LOG_DIR}/ocr_service.log
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ocr-training.service >/dev/null
sudo systemctl restart ocr-training.service
echo ""

# Give uvicorn a moment: importing torch/cv2 in this env is not instant.
for _ in $(seq 1 20); do
    curl -sf -o /dev/null http://127.0.0.1:8002/health && break
    sleep 1
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if curl -sf -o /dev/null http://127.0.0.1:8002/health; then
    echo "  ✅ ocr-training.service — healthy on :8002"
else
    echo "  ❌ not answering /health"
    echo "     sudo journalctl -u ocr-training -n 50 --no-pager"
    echo "     tail -50 ${LOG_DIR}/ocr_service.log"
fi
echo ""
echo "  sudo systemctl restart ocr-training"
echo "  sudo journalctl -u ocr-training -f"
echo "  tail -f ${LOG_DIR}/ocr_service.log"
echo ""
echo "  bash scripts/setup_ocr_training_service.sh --uninstall   # remove"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
