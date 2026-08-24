#!/bin/bash
# Setup ocr-all.service — single service that runs start_services.sh
#
# Run once on Jetson: bash scripts/setup_systemd_services.sh

set -e

USER_NAME=$(whoami)
USER_HOME="$HOME"
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

# ── Remove old units ─────────────────────────────────────────────────────────
echo "Removing old service units..."
for svc in ocr-all ocr-backend ocr-ai ocr-frontend ocr-camera-check ocr-firefox; do
    sudo systemctl stop    "${svc}.service" 2>/dev/null || true
    sudo systemctl stop    "${svc}.target"  2>/dev/null || true
    sudo systemctl disable "${svc}.service" 2>/dev/null || true
    sudo systemctl disable "${svc}.target"  2>/dev/null || true
    sudo rm -f "/etc/systemd/system/${svc}.service" "/etc/systemd/system/${svc}.target"
done
echo "   ✅ Done"
echo ""

# ── Create ocr-all.service ───────────────────────────────────────────────────
# NOT `which python3` -- that bakes in whatever interpreter is active in the
# shell you happen to run this setup script from (e.g. base conda env if you
# forgot to `conda activate vision` first), which silently produces a
# ModuleNotFoundError under systemd for fastapi/pypylon/torch/etc since those
# only live in the `vision` env. Pin to the vision env explicitly, and run it
# through `conda activate` (not just the raw binary path) so the CUDA
# LD_LIBRARY_PATH shim from vision's activate.d hook is also applied --
# invoking the binary path directly skips that hook entirely.
CONDA_SH="${USER_HOME}/miniconda3/etc/profile.d/conda.sh"
VISION_PY="${USER_HOME}/miniconda3/envs/vision/bin/python3"
if [ ! -x "$VISION_PY" ]; then
    echo "❌ vision conda env not found at ${VISION_PY} -- create it first (see ai_services setup docs)"
    exit 1
fi
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
echo "  sudo systemctl restart ocr-all   # restart everything"
echo "  sudo journalctl -u ocr-all -f    # live logs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
