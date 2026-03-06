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
LOG_DIR="${USER_HOME}/Source/ocr_datecode/logs"

# Create log directory
mkdir -p "$LOG_DIR"

# === STEP 0: Camera Check (only for suntech machine) ===
if [ "$USER_HOME" = "/home/suntech" ]; then
    SCRIPT_DIR="${USER_HOME}/Source/ocr_datecode"
    TMPFILE=$(mktemp /tmp/cam_check_exit_XXXX)

    # Ensure display is available for terminal window
    export DISPLAY="${DISPLAY:-:0}"
    export XAUTHORITY="${USER_HOME}/.Xauthority"

    TERMINAL_CMD="python3 '${SCRIPT_DIR}/camera_check_eth1.py'; \
EXIT=\$?; echo \$EXIT > '${TMPFILE}'; \
if [ \$EXIT -eq 0 ]; then \
    echo ''; echo '>>> Camera OK! Services will start in 3 seconds...'; sleep 3; \
else \
    echo ''; echo '>>> Camera check FAILED. Press Enter to close.'; read; \
fi"

    echo "📷 Opening camera health check terminal..."
    if command -v gnome-terminal &>/dev/null; then
        gnome-terminal --wait --title="Camera Health Check - eth1" -- bash -c "$TERMINAL_CMD"
    elif command -v xterm &>/dev/null; then
        xterm -title "Camera Health Check - eth1" -geometry 120x35 -e bash -c "$TERMINAL_CMD"
    else
        echo "⚠️  No GUI terminal found, running inline..."
        python3 "${SCRIPT_DIR}/camera_check_eth1.py"
        echo $? > "${TMPFILE}"
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

# Kill existing processes
kill_port 8000
kill_port 5173
pkill -f "camera_management_service.py" 2>/dev/null || true
pkill ngrok 2>/dev/null || true

sleep 2

# 1. Start Backend (FastAPI)
echo "📦 Starting Backend API (port 8000)..."
cd "$BACKEND_DIR"
nohup python3 -m uvicorn app.main:app --reload --port 8000 --host 0.0.0.0 \
    > "$LOG_DIR/backend.log" 2>&1 &
BACKEND_PID=$!
echo "   PID: $BACKEND_PID"
sleep 3

# 2. Start Frontend (Vite)
echo "🎨 Starting Frontend (port 5173)..."
cd "$FRONTEND_DIR"
nohup yarn dev > "$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo "   PID: $FRONTEND_PID"
sleep 3

# 3. Start AI Services (Camera Management)
echo "📷 Starting AI Camera Services..."
cd "$AI_SERVICES_DIR"
nohup python camera_management_service.py > "$LOG_DIR/ai_camera.log" 2>&1 &
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

# 6. Auto-open Firefox after services are ready
echo "🦊 Opening Firefox..."
sleep 5  # Wait for frontend to be fully ready

# Get the current display and xauthority
CURRENT_DISPLAY=$(who | grep "$USER" | awk '{print $2}' | head -1)
CURRENT_DISPLAY=":${CURRENT_DISPLAY##*:}"
[ -z "$CURRENT_DISPLAY" ] && CURRENT_DISPLAY=":0"

export DISPLAY="$CURRENT_DISPLAY"
export XAUTHORITY="$HOME/.Xauthority"

# Open Firefox
if command -v firefox &> /dev/null; then
    firefox http://localhost:5173 > "$LOG_DIR/firefox.log" 2>&1 &
    FIREFOX_PID=$!
    echo "   PID: $FIREFOX_PID (DISPLAY=$DISPLAY)"
    
    echo "FIREFOX_PID=$FIREFOX_PID" >> "$LOG_DIR/pids.txt"
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