#!/bin/bash
# Setup systemd auto-restart for OCR Datecode services
# Run ONCE on Jetson: bash scripts/setup_systemd_services.sh

set -e

USER_NAME=$(whoami)
USER_HOME="$HOME"
PROJECT_DIR="${USER_HOME}/Source/ocr_datecode"
LOG_DIR="${PROJECT_DIR}/logs"

echo "Setting up systemd services for user: $USER_NAME"
echo "Project dir: $PROJECT_DIR"
echo ""

# Detect python3 path
PYTHON3=$(which python3)
echo "Python3: $PYTHON3"

mkdir -p "$LOG_DIR"

# ── 1. Backend (FastAPI / uvicorn) ──────────────────────────────────────────
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
StandardOutput=append:${LOG_DIR}/backend.log
StandardError=append:${LOG_DIR}/backend.log
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF
echo "✅ ocr-backend.service created"

# ── 2. AI Camera Service ─────────────────────────────────────────────────────
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
RestartSec=10
StandardOutput=append:${LOG_DIR}/ai_camera.log
StandardError=append:${LOG_DIR}/ai_camera.log
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF
echo "✅ ocr-ai.service created"

# ── 3. Frontend (Vite) ───────────────────────────────────────────────────────
YARN=$(which yarn 2>/dev/null || echo "")
if [ -n "$YARN" ]; then
sudo tee /etc/systemd/system/ocr-frontend.service > /dev/null << EOF
[Unit]
Description=OCR Datecode Frontend
After=network.target

[Service]
User=${USER_NAME}
WorkingDirectory=${PROJECT_DIR}/frontend-ts
ExecStart=${YARN} dev
Restart=always
RestartSec=5
StandardOutput=append:${LOG_DIR}/frontend.log
StandardError=append:${LOG_DIR}/frontend.log
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF
echo "✅ ocr-frontend.service created"
else
    echo "⚠️  yarn not found, skipping ocr-frontend.service"
fi

# ── Reload & enable ──────────────────────────────────────────────────────────
echo ""
echo "Reloading systemd..."
sudo systemctl daemon-reload

echo "Enabling services (auto-start on boot)..."
sudo systemctl enable ocr-backend ocr-ai
[ -n "$YARN" ] && sudo systemctl enable ocr-frontend

echo ""
echo "Starting services..."
sudo systemctl start ocr-backend
sleep 3
sudo systemctl start ocr-ai
[ -n "$YARN" ] && sudo systemctl start ocr-frontend

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Status:"
sudo systemctl is-active ocr-backend  && echo "  ocr-backend  : ✅ running" || echo "  ocr-backend  : ❌ failed"
sudo systemctl is-active ocr-ai       && echo "  ocr-ai       : ✅ running" || echo "  ocr-ai       : ❌ failed"
[ -n "$YARN" ] && \
sudo systemctl is-active ocr-frontend && echo "  ocr-frontend : ✅ running" || echo "  ocr-frontend : ❌ failed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status ocr-backend"
echo "  sudo systemctl status ocr-ai"
echo "  sudo journalctl -u ocr-backend -f"
echo "  sudo journalctl -u ocr-ai -f"
echo "  tail -f ${LOG_DIR}/backend.log"
echo "  tail -f ${LOG_DIR}/ai_camera.log"
