#!/bin/bash
# Chạy AI Agent Service (mặc định :8100). Backend chính ở :8000 không bị đụng.
set -e

cd "$(dirname "$0")"

PORT="${PORT:-8100}"

if [ ! -f .env ]; then
    echo "❌ Thiếu file .env — copy từ .env.sample rồi điền SECRET_KEY + OPENAI_API_KEY"
    echo "   cp .env.sample .env"
    exit 1
fi

if ! nc -z localhost 27017 2>/dev/null; then
    echo "❌ MongoDB chưa chạy trên localhost:27017"
    exit 1
fi

echo "🤖 Agent Service  → http://localhost:${PORT}"
echo "📚 API Docs       → http://localhost:${PORT}/docs"
echo ""

exec python3 -m uvicorn agent_app.main:app --host 0.0.0.0 --port "${PORT}" "$@"
