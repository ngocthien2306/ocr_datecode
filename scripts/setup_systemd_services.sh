#!/bin/bash
# Setup OCR Datecode systemd services — Hybrid design:
#
#   ocr-all.service (Type=oneshot, orchestrator)
#     Runs start_services.sh which:
#       1. Camera check (GUI popup when interactive)
#       2. systemctl start ocr-backend / ocr-ai / ocr-frontend  (ordered)
#       3. Waits for services to respond
#       4. Opens Firefox kiosk
#
#   ocr-backend / ocr-ai / ocr-frontend (Restart=always)
#     Auto-restart on crash — independent of ocr-all.service
#     NOT started on boot directly; started by start_services.sh via ocr-all
#
# Result:
#   Boot           → ocr-all.service → ordered startup → Firefox when ready
#   BE crash       → ocr-backend.service auto-restarts in 5s
#   AI crash       → ocr-ai.service   auto-restarts in 5s
#   Full restart   → systemctl restart ocr-all → camera check → ordered startup
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
echo " OCR Datecode — systemd setup (hybrid)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " User      : $USER_NAME"
echo " Project   : $PROJECT_DIR"
echo " Python3   : $PYTHON3"
echo " Yarn      : ${YARN:-not found}"
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
        || echo "   ⚠️  Could not set eth1 MTU — camera.py will adapt to current MTU automatically"
else
    echo "   ✅ eth1 MTU already ${CURRENT_MTU} (Jumbo Frames OK)"
fi
echo ""

# ── STEP 1: Remove old units ─────────────────────────────────────────────────
echo "[1/4] Removing old service units..."

OLD_UNITS=(ocr-all ocr-backend ocr-ai ocr-frontend ocr-camera-check ocr-firefox)
for svc in "${OLD_UNITS[@]}"; do
    sudo systemctl stop    "${svc}.service" 2>/dev/null || true
    sudo systemctl stop    "${svc}.target"  2>/dev/null || true
    sudo systemctl disable "${svc}.service" 2>/dev/null || true
    sudo systemctl disable "${svc}.target"  2>/dev/null || true
    [ -f "/etc/systemd/system/${svc}.service" ] && sudo rm -f "/etc/systemd/system/${svc}.service" && echo "   removed ${svc}.service"
    [ -f "/etc/systemd/system/${svc}.target"  ] && sudo rm -f "/etc/systemd/system/${svc}.target"  && echo "   removed ${svc}.target"
done
echo "   ✅ Old units removed"
echo ""

# ── STEP 2: Create individual services (crash recovery) ──────────────────────
echo "[2/4] Creating service units..."

# ── 2a. Backend ──────────────────────────────────────────────────────────────
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
echo "   ✅ ocr-backend.service  (Restart=always)"

# ── 2b. AI Camera Service ────────────────────────────────────────────────────
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
echo "   ✅ ocr-ai.service       (Restart=always)"

# ── 2c. Frontend ─────────────────────────────────────────────────────────────
if [ -n "$YARN" ]; then
sudo tee /etc/systemd/system/ocr-frontend.service > /dev/null << EOF
[Unit]
Description=OCR Datecode Frontend (Vite)
After=network.target

[Service]
User=${USER_NAME}
WorkingDirectory=${PROJECT_DIR}/frontend-ts
ExecStart=${YARN} dev
Restart=always
RestartSec=5
StartLimitIntervalSec=600
StartLimitBurst=10
StandardOutput=append:${LOG_DIR}/frontend.log
StandardError=append:${LOG_DIR}/frontend.log
Environment=NODE_ENV=development

[Install]
WantedBy=multi-user.target
EOF
    echo "   ✅ ocr-frontend.service (Restart=always)"
else
    echo "   ⚠️  yarn not found — skipping ocr-frontend.service"
fi

# ── 2d. ocr-all (orchestrator — ordered startup + Firefox) ───────────────────
#
# Type=oneshot + RemainAfterExit=yes:
#   start_services.sh runs, starts everything in order, exits 0.
#   systemd marks ocr-all as "active (exited)" — this is correct.
#   Individual services (ocr-backend/ai/frontend) keep running with Restart=always.
#   If ocr-all is restarted → start_services.sh runs again (camera check + Firefox).
#
sudo tee /etc/systemd/system/ocr-all.service > /dev/null << EOF
[Unit]
Description=OCR Datecode All Services (ordered startup)
After=network.target mongod.service
Wants=mongod.service

[Service]
Type=oneshot
RemainAfterExit=yes
User=${USER_NAME}
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
echo "   ✅ ocr-all.service      (Type=oneshot, orchestrator)"
echo ""

# ── STEP 3: Enable ────────────────────────────────────────────────────────────
echo "[3/4] Enabling services..."
sudo systemctl daemon-reload

# Individual services enabled for auto-restart after boot
sudo systemctl enable ocr-backend.service
sudo systemctl enable ocr-ai.service
[ -n "$YARN" ] && sudo systemctl enable ocr-frontend.service

# ocr-all handles boot-time ordered startup
sudo systemctl enable ocr-all.service

echo "   ✅ All services enabled"
echo ""

# ── STEP 4: Start ─────────────────────────────────────────────────────────────
echo "[4/4] Starting via ocr-all (ordered startup)..."
echo "   This runs camera check first — may take up to 3 min if eth1 needs reset."
echo ""
sudo systemctl start ocr-all.service
echo ""

# ── Status ────────────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
_check() {
    local name=$1
    if sudo systemctl is-active --quiet "${name}"; then
        echo "  ✅ ${name}"
    else
        echo "  ❌ ${name}  ← sudo journalctl -u ${name} -n 30 --no-pager"
    fi
}
_check ocr-all.service
_check ocr-backend.service
_check ocr-ai.service
[ -n "$YARN" ] && _check ocr-frontend.service

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Commands"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  # Full restart (camera check + ordered startup + Firefox):"
echo "  sudo systemctl restart ocr-all"
echo ""
echo "  # Restart one crashed service only (no camera check):"
echo "  sudo systemctl restart ocr-ai"
echo "  sudo systemctl restart ocr-backend"
echo ""
echo "  # Live logs:"
echo "  sudo journalctl -u ocr-all -f"
echo "  tail -f ${LOG_DIR}/ai_camera.log"
echo "  tail -f ${LOG_DIR}/backend.log"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
