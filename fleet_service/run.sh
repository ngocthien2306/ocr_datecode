#!/bin/bash
# Fleet service → :8200. Chạy trên máy dev, không phải trên Jetson.
set -e
cd "$(dirname "$0")"
[ -f .env ] || { echo "❌ Thiếu .env — cp .env.sample .env rồi điền FLEET_EDGE_PASSWORD"; exit 1; }
PORT="${PORT:-8200}"
echo "🛰  Fleet Service → http://localhost:${PORT}"
echo "📚 API Docs      → http://localhost:${PORT}/docs"
exec python3 -m uvicorn fleet_app.main:app --host 0.0.0.0 --port "${PORT}" "$@"
