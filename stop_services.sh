#!/bin/bash

echo "🛑 Stopping OCR Datecode Services..."

LOG_DIR="${HOME}/Source/ocr_datecode/logs"
PID_FILE="$LOG_DIR/pids.txt"

# Stop by PIDs if available
if [ -f "$PID_FILE" ]; then
    echo "📋 Reading PIDs from $PID_FILE..."
    source "$PID_FILE"
    
    for pid_var in BACKEND_PID FRONTEND_PID AI_PID NGROK_API_PID NGROK_FRONTEND_PID FIREFOX_PID; do
        pid_value=$(eval echo \$$pid_var)
        if [ -n "$pid_value" ] && kill -0 "$pid_value" 2>/dev/null; then
            echo "   Killing $pid_var ($pid_value)..."
            kill -9 "$pid_value" 2>/dev/null || true
        fi
    done
    
    rm -f "$PID_FILE"
fi

# Kill by port (fallback)
echo "📦 Stopping Backend API (port 8000)..."
fuser -k 8000/tcp 2>/dev/null || echo "   No process on port 8000"

echo "🎨 Stopping Frontend (port 5173)..."
fuser -k 5173/tcp 2>/dev/null || echo "   No process on port 5173"

# Kill by process name (fallback)
echo "📷 Stopping AI Camera Services..."
pkill -f "camera_management_service.py" || echo "   No camera service running"

echo "🌐 Stopping ngrok tunnels..."
pkill ngrok || echo "   No ngrok processes running"

echo "🦊 Stopping Firefox (localhost:5173)..."
pkill -f "firefox.*localhost:5173" 2>/dev/null || true
# Fallback: kill any remaining firefox process owned by current user
pkill -u "$USER" firefox 2>/dev/null || echo "   No firefox processes running"

# Wait for Firefox to release its profile lock cleanly
sleep 1

# Remove stale Firefox profile lock files (in case Firefox was hung)
for lock_file in "$HOME"/.mozilla/firefox/*.default*/lock "$HOME"/.mozilla/firefox/*.default*/.parentlock; do
    [ -e "$lock_file" ] && rm -f "$lock_file" 2>/dev/null || true
done

echo ""
echo "✅ All services stopped!"
echo "📁 Logs preserved in: $LOG_DIR/"
