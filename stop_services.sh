#!/bin/bash

echo "🛑 Stopping OCR Datecode Services..."

LOG_DIR="${HOME}/Source/ocr_datecode/logs"
PID_FILE="$LOG_DIR/pids.txt"

# Stop by PIDs if available
if [ -f "$PID_FILE" ]; then
    echo "📋 Reading PIDs from $PID_FILE..."
    source "$PID_FILE"
    
    # Backend / frontend / AI / ngrok: hard kill is fine — they don't show
    # a "previous session crashed" dialog on next start.
    for pid_var in BACKEND_PID FRONTEND_PID AI_PID NGROK_API_PID NGROK_FRONTEND_PID; do
        pid_value=$(eval echo \$$pid_var)
        if [ -n "$pid_value" ] && kill -0 "$pid_value" 2>/dev/null; then
            echo "   Killing $pid_var ($pid_value)..."
            kill -9 "$pid_value" 2>/dev/null || true
        fi
    done

    # Firefox: graceful shutdown — SIGTERM, wait, only force-kill as fallback.
    # SIGKILL leaves Firefox marking the session as crashed, which triggers
    # the "Open Firefox in Troubleshoot Mode?" dialog on next launch.
    if [ -n "$FIREFOX_PID" ] && kill -0 "$FIREFOX_PID" 2>/dev/null; then
        echo "   Asking Firefox to quit gracefully (PID $FIREFOX_PID)..."
        kill -TERM "$FIREFOX_PID" 2>/dev/null || true
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            kill -0 "$FIREFOX_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$FIREFOX_PID" 2>/dev/null; then
            echo "   Firefox did not exit in 10s, force killing..."
            kill -9 "$FIREFOX_PID" 2>/dev/null || true
        fi
    fi
    
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

echo "🦊 Stopping Firefox (graceful)..."
# Try graceful shutdown first (SIGTERM = default for pkill).
if pkill -u "$USER" firefox 2>/dev/null; then
    # Wait up to 10s for Firefox to exit cleanly.
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        pgrep -u "$USER" firefox >/dev/null 2>&1 || break
        sleep 1
    done
    # Last resort if it's still alive.
    if pgrep -u "$USER" firefox >/dev/null 2>&1; then
        echo "   Firefox still alive after 10s, force killing..."
        pkill -9 -u "$USER" firefox 2>/dev/null || true
    fi
else
    echo "   No firefox processes running"
fi

echo ""
echo "✅ All services stopped!"
echo "📁 Logs preserved in: $LOG_DIR/"
