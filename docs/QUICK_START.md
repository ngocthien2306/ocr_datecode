# Quick Start Guide

## Prerequisites

- Python 3.8+
- Node.js 18+
- MongoDB running
- Basler cameras connected (optional for testing)

## 1. Backend Setup

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env: MongoDB URL, JWT secret, etc.

# Start backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Backend runs at**: `http://localhost:8000`

## 2. Frontend Setup

```bash
cd frontend-ts

# Install dependencies
npm install

# Start dev server
npm run dev
```

**Frontend runs at**: `http://localhost:5173`

## 3. Camera Management Service

```bash
cd ai_services

# Set environment variables
export API_BASE=http://localhost:8000
export WS_HOST=localhost
export WS_PORT=8000

# Start camera service
python3 camera_management_service.py
```

## 4. Login

- **URL**: `http://localhost:5173`
- **Default admin**:
  - Username: `admin`
  - Password: (check database or create via API)

## 5. Basic Workflow

1. **Add Cameras** (Camera Management page)
   - Add Basler cameras with serial numbers
   - Connect cameras

2. **Create Recipe** (Recipes page)
   - Add basic info (name, product code)
   - Select cameras
   - Configure templates (upload images or capture from camera)
   - Draw regions: template, text, barcode, datecode

3. **Load Recipe** (Recipes page)
   - Click LOAD button
   - Cameras switch to hardware_trigger mode
   - Wait for DI trigger signal

4. **View Results** (Inference Results page)
   - See real-time results as products are inspected
   - Filter by recipe, PASS/FAIL, date
   - View details with captured images

5. **Stop Recipe** (Recipes page)
   - Click STOP button (red)
   - Cameras return to idle mode

## Architecture

```
Frontend (React TS) ←REST/SocketIO→ Backend (FastAPI) ←WebSocket→ Camera Service
                                         ↓
                                    Shared Memory ← Camera Processes
```

## Troubleshooting

### Backend won't start
- Check MongoDB connection in `.env`
- Check port 8000 is available

### Frontend won't start
- Run `npm install` again
- Check port 5173 is available

### Camera service won't connect
- Check `WS_HOST` and `WS_PORT` match backend
- Check backend WebSocket endpoint `/ws/camera-management`

### No real-time updates
- Check browser console for SocketIO connection
- Backend logs should show: "Client connected"

## Advanced Configuration

See:
- `/docs/IMPLEMENTATION_SUMMARY.md` - Full architecture details
- `/docs/TEST_GUIDE_PHASE_1_2.md` - Testing guide
- `/docs/PHASE_5_REMAINING_TASKS.md` - Remaining tasks
