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

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
