#!/bin/bash

# Script to start all OCR Datecode services in background
set -e

echo "🚀 Starting OCR Datecode Services in Background..."

# Detect user home directory
USER_HOME="$HOME"

# Base directories
BACKEND_DIR="${USER_HOME}/Source/ocr_datecode/backend"
FRONTEND_DIR="${USER_HOME}/Source/ocr_datecode/frontend-ts"
AI_SERVICES_DIR="${USER_HOME}/Source/ocr_datecode/ai_services"
# Logs from this script (stdout/stderr of background processes) are ephemeral —
# wiped on every start. Persistent app logs live under logs/{backend,...}/.
LOGS_ROOT="${USER_HOME}/Source/ocr_datecode/logs"
LOG_DIR="${LOGS_ROOT}/start_services"

mkdir -p "$LOGS_ROOT"
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

# === STEP 0: Camera Check ===
if [ "$USER_HOME" = "/home/suntech" ] || [ "$USER_HOME" = "/home/demo" ]; then
    SCRIPT_DIR="${USER_HOME}/Source/ocr_datecode"
    TMPFILE=$(mktemp /tmp/cam_check_exit_XXXX)

    # Ensure display is available for terminal window
    export DISPLAY="${DISPLAY:-:0}"
    export XAUTHORITY="${USER_HOME}/.Xauthority"

    if [ "$USER_HOME" = "/home/demo" ]; then
        CAM_SCRIPT="camera_check_all.py"
        CAM_TITLE="Camera Health Check - ALL"
        CAM_ARGS=""
    else
        CAM_SCRIPT="camera_check_eth1.py"
        CAM_TITLE="Camera Health Check - eth1"
        # --no-reboot: on failure, exit 1 instead of rebooting the whole box.
        # ocr-all.service (Restart=on-failure) re-runs this script → popup again.
        CAM_ARGS="--no-reboot"
    fi

    TERMINAL_CMD="python3 '${SCRIPT_DIR}/${CAM_SCRIPT}' ${CAM_ARGS}; \
EXIT=\$?; echo \$EXIT > '${TMPFILE}'; \
if [ \$EXIT -eq 0 ]; then \
    echo ''; echo '>>> Camera OK! Services will start in 3 seconds...'; sleep 3; \
else \
    echo ''; echo '>>> Camera check FAILED. Services will NOT start (will retry).'; sleep 6; \
fi"

    echo "📷 Opening camera health check terminal..."
    # Show the GUI popup (gnome-terminal/xterm) FIRST — works under systemd too as
    # long as DISPLAY=:0 + XAUTHORITY are exported (done above). If the GUI can't
    # open (no X access), fall back to an inline run so the service still proceeds.
    # NB: capture the exit code via `if` so `set -e` does NOT abort the script when
    # the camera check fails — we need to reach the CAMERA_EXIT check below.
    if command -v gnome-terminal &>/dev/null && \
       gnome-terminal --wait --title="$CAM_TITLE" -- bash -c "$TERMINAL_CMD"; then
        :  # popup ran; it wrote the exit code into TMPFILE
    elif command -v xterm &>/dev/null && \
         xterm -title "$CAM_TITLE" -geometry 120x35 -e bash -c "$TERMINAL_CMD"; then
        :  # popup ran; it wrote the exit code into TMPFILE
    else
        echo "⚠️  No usable GUI terminal — running camera check inline..."
        if python3 "${SCRIPT_DIR}/${CAM_SCRIPT}" ${CAM_ARGS}; then echo 0 > "${TMPFILE}"; else echo $? > "${TMPFILE}"; fi
    fi

    CAMERA_EXIT=$(cat "${TMPFILE}" 2>/dev/null || echo "1")
    rm -f "${TMPFILE}"

    if [ "$CAMERA_EXIT" != "0" ]; then
        echo "❌ Camera check failed. Aborting service startup."
        exit 1
    fi

    echo "✅ Camera check passed. Starting services..."
    echo ""
fi

# Check if directories exist
for dir in "$BACKEND_DIR" "$FRONTEND_DIR" "$AI_SERVICES_DIR"; do
    if [ ! -d "$dir" ]; then
        echo "❌ Error: Directory not found: $dir"
        exit 1
    fi
done

# Function to kill existing processes on ports
kill_port() {
    local port=$1
    echo "🔍 Checking port $port..."
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo "⚠️  Port $port is in use. Killing existing process..."
        fuser -k $port/tcp 2>/dev/null || true
        sleep 1
    fi
}

# Helper: true if a systemd service unit is installed (enabled, disabled or static)
_systemd_managed() {
    systemctl list-unit-files "$1" --no-legend 2>/dev/null | grep -qE "enabled|static|disabled"
}

# Kill existing processes
# NOTE: port 8000 (backend) is intentionally NOT killed here — the backend runs
# as its own systemd unit (ocr-backend.service, Restart=always). Killing it would
# fight systemd and cause an unnecessary restart/outage.
kill_port 5173
# Camera management runs as a plain background process (started below, after the
# camera check). Kill any leftover instance so we start fresh.
pkill -f "camera_management_service.py" 2>/dev/null || true
pkill ngrok 2>/dev/null || true

# Firefox: graceful shutdown if any leftover instance is alive. Avoid SIGKILL
# here — it leaves session state marked as crashed and triggers the
# "Open in Troubleshoot Mode?" dialog on next launch.
if pgrep -u "$USER" firefox >/dev/null 2>&1; then
    pkill -u "$USER" firefox 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        pgrep -u "$USER" firefox >/dev/null 2>&1 || break
        sleep 1
    done
    if pgrep -u "$USER" firefox >/dev/null 2>&1; then
        pkill -9 -u "$USER" firefox 2>/dev/null || true
        sleep 1
    fi
fi

sleep 2

# 0.5 Update code from git before starting services.
# Non-fatal: a failed pull (no network, local changes, conflicts) must NOT
# block service startup, so we never let it trip `set -e`.
echo "⬇️  Updating code (git pull)..."
PROJECT_DIR="${USER_HOME}/Source/ocr_datecode"
if cd "$PROJECT_DIR"; then
    GIT_TERMINAL_PROMPT=0 git pull --ff-only > "$LOG_DIR/git_pull.log" 2>&1 \
        && echo "   ✅ Code updated" \
        || echo "   ⚠️  git pull skipped/failed (see $LOG_DIR/git_pull.log) — starting with current code"
fi

# 1. Start Backend (FastAPI)
echo "📦 Starting Backend API (port 8000)..."
if _systemd_managed "ocr-backend.service"; then
    # Backend is its own enabled systemd unit (Restart=always) and auto-starts on
    # boot. We do NOT start it here — that would need `sudo systemctl` (interactive
    # password under systemd) and would fight the unit. We just wait for /health
    # further below. Nudge it active without sudo if it happens to be down.
    systemctl is-active --quiet ocr-backend || systemctl --no-block start ocr-backend 2>/dev/null || true
    BACKEND_PID=$(systemctl show -p MainPID ocr-backend 2>/dev/null | cut -d= -f2)
    echo "   Managed by systemd (PID: ${BACKEND_PID:-?}) — waiting for health below"
else
    cd "$BACKEND_DIR"
    nohup python3 -m uvicorn app.main:app --port 8000 --host 0.0.0.0 \
        >> "$LOG_DIR/backend.log" 2>&1 &
    BACKEND_PID=$!
    echo "   PID: $BACKEND_PID"
fi
sleep 3

# 2. Start Frontend (Vite)
echo "🎨 Starting Frontend (port 5173)..."
cd "$FRONTEND_DIR"
nohup yarn dev >> "$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo "   PID: $FRONTEND_PID"
sleep 3

# 3. Start AI Services (Camera Management)
# Runs as a plain background process (NOT a systemd service) so it stays
# sequential after the camera check above. Crash recovery is handled by the
# backend supervisor (kill + respawn via app/services/camera_service_supervisor.py).
echo "📷 Starting AI Camera Services..."
cd "$AI_SERVICES_DIR"
nohup python camera_management_service.py >> "$LOG_DIR/ai_camera.log" 2>&1 &
AI_PID=$!
echo "   PID: $AI_PID"
sleep 2

# 4. Start ngrok for Backend API
echo "🌐 Starting ngrok for Backend API..."
nohup ngrok http --url=suntech-vision-api.ngrok.app 8000 \
    > "$LOG_DIR/ngrok_api.log" 2>&1 &
NGROK_API_PID=$!
echo "   PID: $NGROK_API_PID"
sleep 2

# 5. Start ngrok for Frontend
echo "🌐 Starting ngrok for Frontend..."
nohup ngrok http --url=suntech-vision.ngrok.app 5173 \
    > "$LOG_DIR/ngrok_frontend.log" 2>&1 &
NGROK_FRONTEND_PID=$!
echo "   PID: $NGROK_FRONTEND_PID"

sleep 2

# 6. Wait for all services to be ready, then open Firefox
echo "⏳ Waiting for services to be ready before opening browser..."

# Wait for backend API (max 60s)
echo "   Backend API..."
for _i in $(seq 1 60); do
    curl -sf http://localhost:8000/health -o /dev/null 2>/dev/null && break
    curl -sf http://localhost:8000/docs   -o /dev/null 2>/dev/null && break
    sleep 1
done

# Wait for AI camera service websocket to connect (max 30s)
# Proxy through backend: if /api/cameras returns 200, AI service is up
echo "   AI camera service..."
for _i in $(seq 1 30); do
    curl -sf http://localhost:8000/api/cameras -o /dev/null 2>/dev/null && break
    sleep 1
done

# Wait for frontend (max 30s)
echo "   Frontend..."
for _i in $(seq 1 30); do
    curl -sf http://localhost:5173 -o /dev/null 2>/dev/null && break
    sleep 1
done

echo "🦊 Opening Firefox..."

# Get the current display and xauthority
CURRENT_DISPLAY=$(who | grep "$USER" | awk '{print $2}' | head -1)
CURRENT_DISPLAY=":${CURRENT_DISPLAY##*:}"
[ -z "$CURRENT_DISPLAY" ] && CURRENT_DISPLAY=":0"
export DISPLAY="$CURRENT_DISPLAY"
export XAUTHORITY="$HOME/.Xauthority"

FIREFOX_PID=""
if command -v firefox &> /dev/null; then
    # Kill any existing Firefox before opening fresh kiosk session
    if pgrep -u "$USER" firefox > /dev/null 2>&1; then
        echo "   Closing existing Firefox..."
        pkill -u "$USER" firefox 2>/dev/null || true
        for _i in 1 2 3 4 5; do
            pgrep -u "$USER" firefox > /dev/null 2>&1 || break
            sleep 1
        done
    fi
    # firefox --kiosk http://localhost:5173 >> "$LOG_DIR/firefox.log" 2>&1 &
    firefox http://localhost:5173 >> "$LOG_DIR/firefox.log" 2>&1 &
    FIREFOX_PID=$!
    echo "   PID: $FIREFOX_PID (DISPLAY=$DISPLAY)"
else
    echo "⚠️  Firefox not found. Please install Firefox."
fi

# Save PIDs to file for easy stopping later
cat > "$LOG_DIR/pids.txt" << EOF
BACKEND_PID=$BACKEND_PID
FRONTEND_PID=$FRONTEND_PID
AI_PID=$AI_PID
NGROK_API_PID=$NGROK_API_PID
NGROK_FRONTEND_PID=$NGROK_FRONTEND_PID
FIREFOX_PID=$FIREFOX_PID
EOF

echo ""
echo "✅ All services started in background!"
echo ""
echo "📊 Services Dashboard:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🔹 Backend API       (PID: $BACKEND_PID)"
echo "     Local:  http://localhost:8000"
echo "     Public: https://suntech-vision-api.ngrok.app"
echo "     Docs:   http://localhost:8000/docs"
echo "     Log:    $LOG_DIR/backend.log"
echo ""
echo "  🔹 Frontend          (PID: $FRONTEND_PID)"
echo "     Local:  http://localhost:5173"
echo "     Public: https://suntech-vision.ngrok.app"
echo "     Log:    $LOG_DIR/frontend.log"
echo ""
echo "  🔹 AI Camera Service (PID: $AI_PID)"
echo "     Log:    $LOG_DIR/ai_camera.log"
echo ""
echo "  🔹 ngrok API         (PID: $NGROK_API_PID)"
echo "     Log:    $LOG_DIR/ngrok_api.log"
echo ""
echo "  🔹 ngrok Frontend    (PID: $NGROK_FRONTEND_PID)"
echo "     Log:    $LOG_DIR/ngrok_frontend.log"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "💡 Commands:"
echo "  - View logs:     tail -f $LOG_DIR/backend.log"
echo "  - Check status:  ./check_services.sh"
echo "  - Stop all:      ./stop_services.sh"
echo ""