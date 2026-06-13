#!/bin/bash
# Migrate OCR Datecode systemd services to individual per-service units.
#
# BEFORE: ocr-all.service wraps start_services.sh (Type=forking, no per-process restart)
# AFTER:  ocr-backend / ocr-ai / ocr-frontend each have Restart=always
#         ocr-all.target groups them for convenience (start/stop/restart all at once)
#
# Run once on Jetson: bash scripts/setup_systemd_services.sh

set -e

# ── Detect environment ───────────────────────────────────────────────────────
USER_NAME=$(whoami)
USER_HOME="$HOME"
PROJECT_DIR="${USER_HOME}/Source/ocr_datecode"
LOG_DIR="${PROJECT_DIR}/logs"

PYTHON3=$(which python3)
YARN=$(which yarn 2>/dev/null || echo "")

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " OCR Datecode — systemd service migration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " User      : $USER_NAME"
echo " Project   : $PROJECT_DIR"
echo " Python3   : $PYTHON3"
echo " Yarn      : ${YARN:-not found}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

mkdir -p "$LOG_DIR"

# ── STEP 1: Tear down old services ───────────────────────────────────────────
echo "[1/4] Removing old service units..."

OLD_SERVICES=(ocr-all ocr-backend ocr-ai ocr-frontend)

for svc in "${OLD_SERVICES[@]}"; do
    unit_file="/etc/systemd/system/${svc}.service"
    target_file="/etc/systemd/system/${svc}.target"

    # Stop if active (ignore errors — unit may not exist)
    sudo systemctl stop "${svc}.service"  2>/dev/null || true
    sudo systemctl stop "${svc}.target"   2>/dev/null || true

    # Disable (ignore errors)
    sudo systemctl disable "${svc}.service" 2>/dev/null || true
    sudo systemctl disable "${svc}.target"  2>/dev/null || true

    # Remove unit files
    if [ -f "$unit_file" ]; then
        sudo rm -f "$unit_file"
        echo "   removed $unit_file"
    fi
    if [ -f "$target_file" ]; then
        sudo rm -f "$target_file"
        echo "   removed $target_file"
    fi
done

echo "   ✅ Old units removed"
echo ""

# ── STEP 2: Create individual service units ───────────────────────────────────
echo "[2/4] Creating new service units..."

# ── 2a. Backend (FastAPI / uvicorn) ─────────────────────────────────────────
sudo tee /etc/systemd/system/ocr-backend.service > /dev/null << EOF
[Unit]
Description=OCR Datecode Backend API
After=network.target mongod.service
Wants=mongod.service
PartOf=ocr-all.target

[Service]
User=${USER_NAME}
WorkingDirectory=${PROJECT_DIR}/backend
ExecStart=${PYTHON3} -m uvicorn app.main:app --port 8000 --host 0.0.0.0
Restart=always
RestartSec=5
# Allow max 5 rapid restarts in 5 minutes before giving up
StartLimitIntervalSec=300
StartLimitBurst=5
StandardOutput=append:${LOG_DIR}/backend.log
StandardError=append:${LOG_DIR}/backend.log
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=ocr-all.target
EOF
echo "   ✅ ocr-backend.service"

# ── 2b. AI Camera Service ────────────────────────────────────────────────────
sudo tee /etc/systemd/system/ocr-ai.service > /dev/null << EOF
[Unit]
Description=OCR Datecode AI Camera Service
After=network.target ocr-backend.service
Wants=ocr-backend.service
PartOf=ocr-all.target

[Service]
User=${USER_NAME}
WorkingDirectory=${PROJECT_DIR}/ai_services
ExecStart=${PYTHON3} camera_management_service.py
Restart=always
RestartSec=10
# Allow max 5 rapid restarts in 5 minutes before giving up
StartLimitIntervalSec=300
StartLimitBurst=5
StandardOutput=append:${LOG_DIR}/ai_camera.log
StandardError=append:${LOG_DIR}/ai_camera.log
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=ocr-all.target
EOF
echo "   ✅ ocr-ai.service"

# ── 2c. Frontend (Vite / yarn) ───────────────────────────────────────────────
if [ -n "$YARN" ]; then
sudo tee /etc/systemd/system/ocr-frontend.service > /dev/null << EOF
[Unit]
Description=OCR Datecode Frontend (Vite)
After=network.target
PartOf=ocr-all.target

[Service]
User=${USER_NAME}
WorkingDirectory=${PROJECT_DIR}/frontend-ts
ExecStart=${YARN} dev
Restart=always
RestartSec=5
StartLimitIntervalSec=300
StartLimitBurst=5
StandardOutput=append:${LOG_DIR}/frontend.log
StandardError=append:${LOG_DIR}/frontend.log
Environment=NODE_ENV=development

[Install]
WantedBy=ocr-all.target
EOF
    echo "   ✅ ocr-frontend.service"
else
    echo "   ⚠️  yarn not found — skipping ocr-frontend.service"
fi

# ── 2d. ocr-all.target (convenience group) ───────────────────────────────────
#
# With PartOf=ocr-all.target in each service:
#   systemctl start ocr-all.target   → starts all 3 (via Wants below)
#   systemctl stop  ocr-all.target   → stops  all 3 (PartOf propagates stop)
#   systemctl restart ocr-all.target → restarts all 3
#
sudo tee /etc/systemd/system/ocr-all.target > /dev/null << EOF
[Unit]
Description=OCR Datecode — All Services
Wants=ocr-backend.service ocr-ai.service ocr-frontend.service
After=ocr-backend.service ocr-ai.service ocr-frontend.service

[Install]
WantedBy=multi-user.target
EOF
echo "   ✅ ocr-all.target"
echo ""

# ── STEP 3: Enable ────────────────────────────────────────────────────────────
echo "[3/4] Reloading systemd and enabling services..."
sudo systemctl daemon-reload

sudo systemctl enable ocr-backend.service
sudo systemctl enable ocr-ai.service
[ -n "$YARN" ] && sudo systemctl enable ocr-frontend.service
sudo systemctl enable ocr-all.target

echo "   ✅ Services enabled (auto-start on boot)"
echo ""

# ── STEP 4: Start ─────────────────────────────────────────────────────────────
echo "[4/4] Starting services..."
sudo systemctl start ocr-backend.service
echo "   ocr-backend started, waiting 4s for API to be ready..."
sleep 4
sudo systemctl start ocr-ai.service
[ -n "$YARN" ] && sudo systemctl start ocr-frontend.service
echo ""

# ── Status report ─────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

check_service() {
    local name=$1
    if sudo systemctl is-active --quiet "${name}"; then
        echo "  ✅ ${name}"
    else
        echo "  ❌ ${name}  ← check: sudo journalctl -u ${name} -n 30 --no-pager"
    fi
}

check_service ocr-backend.service
check_service ocr-ai.service
[ -n "$YARN" ] && check_service ocr-frontend.service

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Useful commands"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  # Restart everything:"
echo "  sudo systemctl restart ocr-all.target"
echo ""
echo "  # Restart one service:"
echo "  sudo systemctl restart ocr-ai"
echo "  sudo systemctl restart ocr-backend"
echo ""
echo "  # Live logs:"
echo "  sudo journalctl -u ocr-backend -f"
echo "  sudo journalctl -u ocr-ai -f"
echo "  tail -f ${LOG_DIR}/ai_camera.log"
echo "  tail -f ${LOG_DIR}/backend.log"
echo ""
echo "  # Stop everything:"
echo "  sudo systemctl stop ocr-all.target"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
