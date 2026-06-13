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

# ── Network tuning for GigE cameras ─────────────────────────────────────────
echo "Tuning network for GigE camera performance..."

# Enlarge OS socket receive buffer — prevents "incompletely grabbed" errors
# when GigE camera sends bursts faster than default kernel buffer can absorb.
sudo sysctl -w net.core.rmem_max=33554432 net.core.rmem_default=8388608 2>/dev/null || true
if ! grep -q "net.core.rmem_max" /etc/sysctl.d/60-gige-camera.conf 2>/dev/null; then
    sudo tee /etc/sysctl.d/60-gige-camera.conf > /dev/null << 'EOF'
# GigE camera receive buffer tuning (ocr_datecode)
net.core.rmem_max=33554432
net.core.rmem_default=8388608
EOF
    echo "   ✅ /etc/sysctl.d/60-gige-camera.conf written (persistent)"
fi

# Jumbo Frames on eth1 — 9000 MTU reduces IP fragmentation from 8192-byte GigE packets
CURRENT_MTU=$(ip link show eth1 2>/dev/null | awk '/mtu/{for(i=1;i<=NF;i++) if($i=="mtu") print $(i+1)}')
if [ "${CURRENT_MTU:-0}" -lt 9000 ] 2>/dev/null; then
    sudo ip link set eth1 mtu 9000 2>/dev/null \
        && echo "   ✅ eth1 MTU set to 9000 (Jumbo Frames)" \
        || echo "   ⚠️  Could not set eth1 MTU (not critical, camera.py adapts automatically)"
else
    echo "   ✅ eth1 MTU already ${CURRENT_MTU} (Jumbo Frames OK)"
fi
echo ""

# ── STEP 1: Tear down old services ───────────────────────────────────────────
echo "[1/4] Removing old service units..."

OLD_SERVICES=(ocr-all ocr-backend ocr-ai ocr-frontend ocr-camera-check ocr-firefox)

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

# ── 2a. Camera Health Check (oneshot, boot-time only) ───────────────────────
#
# Runs ONCE at boot to verify camera is reachable and reset eth1 if needed.
# Type=oneshot + RemainAfterExit=yes means it stays "active" after exit —
# so when ocr-ai crashes and restarts, this service is already done and does
# NOT re-run (no 220s delay on crash recovery).
#
# Uses --no-reboot so systemd gets a clean exit code instead of a kernel reboot.
# ocr-ai uses Wants= (not Requires=) so it still starts even if check fails.
#
CAMERA_CHECK_SCRIPT=""
if [ "$USER_HOME" = "/home/suntech" ]; then
    CAMERA_CHECK_SCRIPT="${PROJECT_DIR}/camera_check_eth1.py"
elif [ "$USER_HOME" = "/home/demo" ]; then
    CAMERA_CHECK_SCRIPT="${PROJECT_DIR}/camera_check_all.py"
fi

if [ -n "$CAMERA_CHECK_SCRIPT" ] && [ -f "$CAMERA_CHECK_SCRIPT" ]; then
sudo tee /etc/systemd/system/ocr-camera-check.service > /dev/null << EOF
[Unit]
Description=OCR Camera Health Check (boot-time only)
After=network.target
Before=ocr-ai.service

[Service]
Type=oneshot
RemainAfterExit=yes
User=${USER_NAME}
WorkingDirectory=${PROJECT_DIR}
ExecStart=${PYTHON3} ${CAMERA_CHECK_SCRIPT} --no-reboot
StandardOutput=append:${LOG_DIR}/camera_check.log
StandardError=append:${LOG_DIR}/camera_check.log

[Install]
WantedBy=ocr-all.target
EOF
    echo "   ✅ ocr-camera-check.service (script: $(basename $CAMERA_CHECK_SCRIPT))"
    CAMERA_CHECK_ENABLED=1
else
    echo "   ⚠️  Not suntech/demo user — skipping ocr-camera-check.service"
    CAMERA_CHECK_ENABLED=0
fi

# ── 2b. Backend (FastAPI / uvicorn) ─────────────────────────────────────────
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
# Allow up to 10 restarts in 10 minutes before systemd gives up
StartLimitIntervalSec=600
StartLimitBurst=10
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
After=network.target ocr-backend.service ocr-camera-check.service
Wants=ocr-backend.service ocr-camera-check.service
PartOf=ocr-all.target

[Service]
User=${USER_NAME}
WorkingDirectory=${PROJECT_DIR}/ai_services
ExecStart=${PYTHON3} camera_management_service.py
Restart=always
RestartSec=5
# Camera failure is handled internally (recovery loop, no self-restart).
# systemd only restarts on actual process crash — allow up to 10 in 10 minutes.
StartLimitIntervalSec=600
StartLimitBurst=10
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
StartLimitIntervalSec=600
StartLimitBurst=10
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

# ── 2d. Firefox (kiosk browser) ─────────────────────────────────────────────
sudo tee /etc/systemd/system/ocr-firefox.service > /dev/null << EOF
[Unit]
Description=OCR Datecode Firefox Kiosk
After=ocr-frontend.service
Wants=ocr-frontend.service
PartOf=ocr-all.target

[Service]
User=${USER_NAME}
Environment=DISPLAY=:0
Environment=XAUTHORITY=/home/${USER_NAME}/.Xauthority
# Kill any existing Firefox first, then wait for frontend to respond
ExecStartPre=-/usr/bin/pkill -u ${USER_NAME} firefox
ExecStartPre=/bin/sh -c 'for i in \$(seq 30); do curl -sf http://localhost:5173 >/dev/null 2>&1 && break; sleep 1; done'
ExecStart=/usr/bin/firefox --kiosk http://localhost:5173
# on-failure: don't restart if user closes Firefox normally (exit 0)
Restart=on-failure
RestartSec=5
StandardOutput=append:${LOG_DIR}/firefox.log
StandardError=append:${LOG_DIR}/firefox.log

[Install]
WantedBy=ocr-all.target
EOF
echo "   ✅ ocr-firefox.service"

# ── 2e. ocr-all.target (convenience group) ───────────────────────────────────
#
# With PartOf=ocr-all.target in each service:
#   systemctl start ocr-all.target   → starts all 3 (via Wants below)
#   systemctl stop  ocr-all.target   → stops  all 3 (PartOf propagates stop)
#   systemctl restart ocr-all.target → restarts all 3
#
sudo tee /etc/systemd/system/ocr-all.target > /dev/null << EOF
[Unit]
Description=OCR Datecode — All Services
Wants=ocr-backend.service ocr-ai.service ocr-frontend.service ocr-firefox.service
After=ocr-backend.service ocr-ai.service ocr-frontend.service ocr-firefox.service

[Install]
WantedBy=multi-user.target
EOF
echo "   ✅ ocr-all.target"
echo ""

# ── STEP 3: Enable ────────────────────────────────────────────────────────────
echo "[3/4] Reloading systemd and enabling services..."
sudo systemctl daemon-reload

[ "${CAMERA_CHECK_ENABLED:-0}" = "1" ] && sudo systemctl enable ocr-camera-check.service
sudo systemctl enable ocr-backend.service
sudo systemctl enable ocr-ai.service
[ -n "$YARN" ] && sudo systemctl enable ocr-frontend.service
sudo systemctl enable ocr-firefox.service
sudo systemctl enable ocr-all.target

echo "   ✅ Services enabled (auto-start on boot)"
echo ""

# ── STEP 4: Start ─────────────────────────────────────────────────────────────
echo "[4/4] Starting services..."
if [ "${CAMERA_CHECK_ENABLED:-0}" = "1" ]; then
    echo "   Running camera health check (may take up to 3 min if eth1 needs reset)..."
    sudo systemctl start ocr-camera-check.service || true
fi
sudo systemctl start ocr-backend.service
echo "   ocr-backend started, waiting 4s for API to be ready..."
sleep 4
sudo systemctl start ocr-ai.service
[ -n "$YARN" ] && sudo systemctl start ocr-frontend.service
sudo systemctl start ocr-firefox.service
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

[ "${CAMERA_CHECK_ENABLED:-0}" = "1" ] && check_service ocr-camera-check.service
check_service ocr-backend.service
check_service ocr-ai.service
[ -n "$YARN" ] && check_service ocr-frontend.service
check_service ocr-firefox.service

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
