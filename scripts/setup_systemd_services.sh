#!/bin/bash
# Setup ocr-all.service — single service that runs start_services.sh
#
# Run once on Jetson: bash scripts/setup_systemd_services.sh

set -e

# Resolve the SERVICE user, not the effective one. This script sudo's for the
# few steps that need root (tee into /etc/systemd/system, systemctl), so it is
# meant to run WITHOUT sudo -- but `sudo bash scripts/setup_systemd_services.sh`
# is the reflex, and under sudo $HOME/whoami become root's. That pointed
# PROJECT_DIR at /root/Source/ocr_datecode and the vision-env probe at
# /root/miniconda3, so the run aborted after already deleting every ocr-* unit.
# SUDO_USER gives the real invoker back; getent reads the passwd entry rather
# than trusting $HOME.
USER_NAME="${SUDO_USER:-$(whoami)}"
USER_HOME=$(getent passwd "$USER_NAME" | cut -d: -f6)
if [ -z "$USER_HOME" ] || [ ! -d "$USER_HOME" ]; then
    echo "❌ Could not resolve home directory for user '${USER_NAME}'"
    exit 1
fi
PROJECT_DIR="${USER_HOME}/Source/ocr_datecode"
LOG_DIR="${PROJECT_DIR}/logs"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " OCR Datecode — systemd setup"
echo " User    : $USER_NAME"
echo " Project : $PROJECT_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

mkdir -p "$LOG_DIR"

# NOTE: GigE network tuning (rmem_max + eth1 MTU) is intentionally NOT done here.
# eth1 is a USB 10/100 adapter (r8152) on this machine, so jumbo frames (MTU 9000)
# are harmful — set MTU/buffers by hand per-machine if ever needed.

# ── Validate the environment BEFORE touching anything ────────────────────────
# This probe used to sit BELOW the removal loop. When it failed (e.g. the script
# was run under sudo, so it looked in /root), the units were already deleted and
# the machine was left with no ocr-* services at all. Validate first, mutate
# second.
#
# NOT `which python3` -- that bakes in whatever interpreter is active in the
# shell you happen to run this setup script from (e.g. base conda env if you
# forgot to `conda activate vision` first), which silently produces a
# ModuleNotFoundError under systemd for fastapi/pypylon/torch/etc since those
# only live in the `vision` env. Pin to the vision env explicitly, and run it
# through `conda activate` (not just the raw binary path) so the CUDA
# LD_LIBRARY_PATH shim from vision's activate.d hook is also applied --
# invoking the binary path directly skips that hook entirely.
echo "Validating environment..."
CONDA_SH="${USER_HOME}/miniconda3/etc/profile.d/conda.sh"
VISION_PY="${USER_HOME}/miniconda3/envs/vision/bin/python3"
if [ ! -x "$VISION_PY" ]; then
    echo "❌ vision conda env not found at ${VISION_PY} -- create it first (see ai_services setup docs)"
    exit 1
fi
if [ ! -f "$CONDA_SH" ]; then
    echo "❌ conda.sh not found at ${CONDA_SH}"
    exit 1
fi
if [ ! -x "${PROJECT_DIR}/start_services.sh" ]; then
    echo "❌ start_services.sh missing or not executable at ${PROJECT_DIR}/start_services.sh"
    exit 1
fi
echo "   ✅ vision env + project scripts found"

# ocr_service runs in its own env (torch + OpenOCR + TensorRT). Missing it is
# not fatal for the rest of the stack, so warn and skip that unit rather than
# aborting a setup that would otherwise succeed.
OCR_PY="${USER_HOME}/miniconda3/envs/ocr_train/bin/python"
INSTALL_OCR_TRAINING=1
if [ ! -x "$OCR_PY" ]; then
    echo "   ⚠️  ocr_train env not found at ${OCR_PY} — skipping ocr-training.service"
    echo "      (create it: conda create -n ocr_train --clone anomaly_service)"
    INSTALL_OCR_TRAINING=0
elif [ ! -d "${PROJECT_DIR}/ocr_service/app" ]; then
    echo "   ⚠️  ${PROJECT_DIR}/ocr_service/app missing — skipping ocr-training.service"
    INSTALL_OCR_TRAINING=0
else
    echo "   ✅ ocr_train env + ocr_service found"
fi
echo ""

# ── Remove old units ─────────────────────────────────────────────────────────
# NOTE: ocr-ai-services MUST be in this list. An older install shipped it with a
# hardcoded `User=demo`; on a `suntech` machine systemd fails at step USER
# (217/USER) before python even starts, and Restart=always turns that into a
# fork-and-die loop every 5s (seen at 35k+ restarts). It never actually ran the
# camera service — that one is launched by start_services.sh as a plain process.
# The loop also floods the journal, which rate-limits and swallows the real
# crash logs you need. `reset-failed` clears the accumulated failure counter.
echo "Removing old service units..."
for svc in ocr-all ocr-backend ocr-ai ocr-ai-services ocr-frontend ocr-camera-check ocr-firefox ocr-training; do
    sudo systemctl stop    "${svc}.service" 2>/dev/null || true
    sudo systemctl stop    "${svc}.target"  2>/dev/null || true
    sudo systemctl disable "${svc}.service" 2>/dev/null || true
    sudo systemctl disable "${svc}.target"  2>/dev/null || true
    sudo rm -f "/etc/systemd/system/${svc}.service" "/etc/systemd/system/${svc}.target"
    sudo systemctl reset-failed "${svc}.service" 2>/dev/null || true
done
echo "   ✅ Done"
echo ""

YARN=$(which yarn 2>/dev/null || echo "")

# ── Create backend crash-recovery service ─────────────────────────────────────
# NOTE: Camera Management does NOT get its own systemd service. It runs as a
# plain background process inside start_services.sh (sequential, after the camera
# health-check). Crash recovery for it is handled by the backend supervisor
# (kill + respawn via app/services/camera_service_supervisor.py).
echo "Creating backend crash-recovery service..."

sudo tee /etc/systemd/system/ocr-backend.service > /dev/null << EOF
[Unit]
Description=OCR Datecode Backend API
After=network.target mongod.service
Wants=mongod.service

[Service]
User=${USER_NAME}
WorkingDirectory=${PROJECT_DIR}/backend
ExecStart=/bin/bash -lc 'source ${CONDA_SH} && conda activate vision && exec python3 -m uvicorn app.main:app --port 8000 --host 0.0.0.0'
Restart=always
RestartSec=5
# The backend respawns the AI camera service via subprocess.Popen
# (camera_service_supervisor → service_tools.start_service). start_new_session=True
# detaches the session but NOT the cgroup, so the default KillMode=control-group
# would SIGKILL that camera process every time this unit stops — meaning every
# backend restart dragged the AI service down with it. `mixed` signals only the
# main process, leaving the respawned camera alive.
KillMode=mixed
StartLimitIntervalSec=600
StartLimitBurst=10
StandardOutput=append:${LOG_DIR}/backend.log
StandardError=append:${LOG_DIR}/backend.log
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF
echo "   ✅ ocr-backend.service (Restart=always)"
echo ""

# ── OCR Training API (ocr_service, :8002) ────────────────────────────────────
# Deliberately its own unit rather than a line in start_services.sh: it is
# independent of the camera pipeline (only shares MongoDB), and an operator
# training a model should not have their run taken down by a camera restart.
if [ "$INSTALL_OCR_TRAINING" = "1" ]; then
echo "Creating OCR training service..."

sudo tee /etc/systemd/system/ocr-training.service > /dev/null << EOF
[Unit]
Description=OCR Datecode OCR Training API (:8002)
After=network.target mongod.service
Wants=mongod.service

[Service]
User=${USER_NAME}
WorkingDirectory=${PROJECT_DIR}/ocr_service
# uvicorn directly, NOT \`python -m app.main\`: that entry point passes
# reload=settings.DEBUG and .env ships DEBUG=True, so a file touch would restart
# the process and kill a training run that was minutes in.
ExecStart=/bin/bash -lc 'source ${CONDA_SH} && conda activate ocr_train && exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8002'
Restart=always
RestartSec=5
# Long enough for the OpenOCR trainer subprocess and a TensorRT context to
# unwind on stop.
TimeoutStopSec=60
# No KillMode override (default control-group), unlike ocr-backend which needs
# 'mixed'. Training spawns train_rec.py with a plain subprocess.Popen, so the
# trainer lives in this cgroup and stopping the unit takes it with it — wanted,
# since an orphaned trainer would hold ~2.6 GB of VRAM and the GPU flock.
#
# PrivateTmp is intentionally left off: the cross-service GPU lock is
# /tmp/ocr_datecode_gpu.lock and must be the same file anomaly_service and any
# manual run see. PrivateTmp=yes would silently disable that mutual exclusion.
# StartLimitIntervalSec/StartLimitBurst are deliberately absent here. In
# [Service] systemd ignores them (as it does in ocr-backend.service above —
# systemd-analyze verify reports "Unknown key name"), and moving them to [Unit]
# would make this unit give up after 10 restarts, leaving the studio silently
# down. Restart=always with no limit is what is wanted.
StandardOutput=append:${LOG_DIR}/ocr_service.log
StandardError=append:${LOG_DIR}/ocr_service.log
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF
echo "   ✅ ocr-training.service (Restart=always, :8002)"
echo ""
fi

echo "Creating ocr-all.service..."

sudo tee /etc/systemd/system/ocr-all.service > /dev/null << EOF
[Unit]
Description=OCR Datecode All Services
After=network.target mongodb.service ocr-backend.service
# Wants= ocr-backend so that \`systemctl start ocr-all\` (and boot) pulls the
# backend up via systemd (as root — no sudo needed inside start_services.sh).
Wants=mongodb.service ocr-backend.service

[Service]
Type=oneshot
RemainAfterExit=yes
User=${USER_NAME}
Group=${USER_NAME}
WorkingDirectory=${PROJECT_DIR}
ExecStart=${PROJECT_DIR}/start_services.sh
ExecStop=${PROJECT_DIR}/stop_services.sh
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

echo "   ✅ ocr-all.service created (Type=oneshot)"
echo ""

# ── Enable & start ───────────────────────────────────────────────────────────
sudo systemctl daemon-reload
# Only backend + all auto-start on boot. Camera management has no systemd unit —
# start_services.sh launches it as a background process after the camera check.
sudo systemctl enable ocr-backend.service ocr-all.service
echo "   ✅ Enabled ocr-backend + ocr-all (auto-start on boot)"
if [ "$INSTALL_OCR_TRAINING" = "1" ]; then
    sudo systemctl enable ocr-training.service
    sudo systemctl restart ocr-training.service
    if sudo systemctl is-active --quiet ocr-training.service; then
        echo "   ✅ ocr-training.service — running on :8002"
    else
        echo "   ❌ ocr-training.service — failed (sudo journalctl -u ocr-training -n 50 --no-pager)"
    fi
fi
echo ""

echo "Starting ocr-all (runs camera check first)..."
sudo systemctl start ocr-all.service

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if sudo systemctl is-active --quiet ocr-all.service; then
    echo "  ✅ ocr-all.service — running"
else
    echo "  ❌ ocr-all.service — failed (sudo journalctl -u ocr-all -n 50 --no-pager)"
fi
echo ""
echo "  sudo systemctl restart ocr-all       # restart everything"
echo "  sudo journalctl -u ocr-all -f        # live logs"
if [ "$INSTALL_OCR_TRAINING" = "1" ]; then
echo "  sudo systemctl restart ocr-training  # OCR Training API (:8002)"
echo "  sudo journalctl -u ocr-training -f   # its logs"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
