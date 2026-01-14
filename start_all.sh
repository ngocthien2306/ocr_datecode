#!/bin/bash

# Script to start all OCR Datecode services in separate terminals (Ubuntu/Jetson Orin)
# Each command runs in a new terminal window with 2 second delay between launches

echo "🚀 Starting OCR Datecode Services on Ubuntu/Jetson Orin..."

# Detect user home directory (change if needed)
USER_HOME="/home/demo"

# 1. Start Backend (FastAPI)
echo "📦 Starting Backend API (port 8000)..."
gnome-terminal --title="Backend API" --window -- bash -c "cd ${USER_HOME}/Source/ocr_datecode/backend && uvicorn app.main:app --reload --port 8000 --host 0.0.0.0; exec bash"
sleep 2

# 2. Start Frontend (Vite)
echo "🎨 Starting Frontend (port 5173)..."
gnome-terminal --title="Frontend Vite" --window -- bash -c "cd ${USER_HOME}/Source/ocr_datecode/frontend-ts && yarn dev; exec bash"
sleep 2

# 3. Start AI Services (Camera Management)
echo "📷 Starting AI Camera Services..."
gnome-terminal --title="AI Camera Services" --window -- bash -c "cd ${USER_HOME}/Source/ocr_datecode/ai_services && python camera_management_service.py; exec bash"
sleep 2

# 4. Start ngrok for Backend API
echo "🌐 Starting ngrok for Backend API..."
gnome-terminal --title="ngrok API" --window -- bash -c "ngrok http --url=suntech-vision-api.ngrok.app 8000; exec bash"
sleep 2

# 5. Start ngrok for Frontend
echo "🌐 Starting ngrok for Frontend..."
gnome-terminal --title="ngrok Frontend" --window -- bash -c "ngrok http --url=suntech-vision.ngrok.app 5173; exec bash"

echo ""
echo "✅ All services started in separate terminal windows!"
echo ""
echo "Services:"
echo "  - Backend API: http://localhost:8000 → https://suntech-vision-api.ngrok.app"
echo "  - Frontend: http://localhost:5173 → https://suntech-vision.ngrok.app"
echo "  - AI Camera Services: Running in background"
echo ""
echo "To stop all services, close the terminal windows or use Ctrl+C in each terminal."
