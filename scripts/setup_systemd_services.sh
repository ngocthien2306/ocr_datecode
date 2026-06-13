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

# ── Network tuning for GigE cameras ─────────────────────────────────────────
echo "Tuning network for GigE camera performance..."

sudo sysctl -w net.core.rmem_max=33554432 net.core.rmem_default=8388608 2>/dev/null || true
if ! grep -q "net.core.rmem_max" /etc/sysctl.d/60-gige-camera.conf 2>/dev/null; then
    sudo tee /etc/sysctl.d/60-gige-camera.conf > /dev/null << 'EOF'
# GigE camera receive buffer tuning (ocr_datecode)
net.core.rmem_max=33554432
net.core.rmem_default=8388608
EOF
    echo "   ✅ /etc/sysctl.d/60-gige-camera.conf written (persistent)"
fi

CURRENT_MTU=$(ip link show eth1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="mtu") print $(i+1)}' | head -1)
if [ "${CURRENT_MTU:-0}" -lt 9000 ] 2>/dev/null; then
    sudo ip link set eth1 mtu 9000 2>/dev/null \
        && echo "   ✅ eth1 MTU set to 9000 (Jumbo Frames)" \
        || echo "   ⚠️  Could not set eth1 MTU — camera.py adapts automatically"
else
    echo "   ✅ eth1 MTU already ${CURRENT_MTU}"
fi
echo ""

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
PYTHON3=$(which python3)
YARN=$(which yarn 2>/dev/null || echo "")

# ── Create individual crash-recovery services ────────────────────────────────
echo "Creating crash-recovery services..."

sudo tee /etc/systemd/system/ocr-backend.service > /dev/null << EOF
[Unit]
Description=OCR Datecode Backend API
After=network.target mongod.service
Wants=mongod.service

[Service]
User=${USER_NAME}
WorkingDirectory=${PROJECT_DIR}/backend
ExecStart=${PYTHON3} -m uvicorn app.main:app --port 8000 --host 0.0.0.0
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

sudo tee /etc/systemd/system/ocr-ai.service > /dev/null << EOF
[Unit]
Description=OCR Datecode AI Camera Service
After=network.target ocr-backend.service
Wants=ocr-backend.service

[Service]
User=${USER_NAME}
WorkingDirectory=${PROJECT_DIR}/ai_services
ExecStart=${PYTHON3} camera_management_service.py
Restart=always
RestartSec=5
StartLimitIntervalSec=600
StartLimitBurst=10
StandardOutput=append:${LOG_DIR}/ai_camera.log
StandardError=append:${LOG_DIR}/ai_camera.log
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF
echo "   ✅ ocr-ai.service      (Restart=always)"
echo ""

echo "Creating ocr-all.service..."

sudo tee /etc/systemd/system/ocr-all.service > /dev/null << EOF
[Unit]
Description=OCR Datecode All Services
After=network.target mongodb.service
Wants=mongodb.service

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
sudo systemctl enable ocr-backend.service ocr-ai.service ocr-all.service
echo "   ✅ Enabled (auto-start on boot)"
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
