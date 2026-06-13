#!/bin/bash

echo "🚀 Starting OCR Datecode API Server..."
echo ""

# Check if MongoDB is running
echo "🔍 Checking MongoDB connection..."
if ! nc -z localhost 27017 2>/dev/null; then
    echo "❌ MongoDB is not running on localhost:27017"
    echo "   Please start MongoDB first:"
    echo "   - Using Docker: docker run -d -p 27017:27017 --name mongodb mongo:latest"
    echo "   - Or start local MongoDB: mongod"
    exit 1
fi

echo "✅ MongoDB is running"
echo ""

# Start FastAPI server
echo "🌐 Starting FastAPI server on http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""

# NOTE: NO --reload. With --reload, uvicorn watches the project tree and
# restarts the process on any file write — including logs written under the
# repo — which makes the backend "tự tắt"/restart unexpectedly. Use plain run.
# (For dev hot-reload, run uvicorn with --reload manually and --reload-dir app.)
uvicorn app.main:app --host 0.0.0.0 --port 8000
