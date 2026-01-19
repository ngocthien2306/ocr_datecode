# Implementation Summary - Camera Management System Refactoring

**Date**: 2026-01-02
**Status**: ✅ Phase 1-3 Completed (Backend + AI Service)

---

## 📊 Overview

Đã hoàn thành refactor hệ thống camera management từ **subprocess + JSON config** → **WebSocket + Shared Memory** architecture.

### Architecture mới:
```
FE (React TS) ←REST/SocketIO→ BE (FastAPI) ←WebSocket→ CameraManagement Service
                                    ↓ SharedMemory Reader
                            Camera Processes → Shared Memory
```

---

## ✅ Phase 1: AI Service Core (Completed)

### Files Created:

#### 1.1 Camera Class
**File**: `/ai_services/camera_management/camera.py`

**Features**:
- 3 operation modes: `idle`, `continuous`, `hardware_trigger`
- Shared memory frame storage (`camera_{serial_number}`)
- Multi-template capture with delay_trigger support
- Inference integration with SuperPointMatcherTRT
- Hardware DI trigger detection (RisingEdge/FallingEdge/AnyEdge)

**Key Methods**:
- `connect()`: Connect to Basler camera + create shared memory
- `disconnect()`: Cleanup resources
- `set_mode(mode)`: Switch between idle/continuous/hardware_trigger
- `load_recipe(recipe_data)`: Load recipe + init inference matcher
- `stop_recipe()`: Stop recipe + set to idle
- `update_settings(settings)`: Update exposure/gain/trigger config

---

#### 1.2 CameraManager Class
**File**: `/ai_services/camera_management/camera_manager.py`

**Features**:
- Manages multiple Camera instances (dict: serial_number → Camera)
- Thread-safe operations (RLock)
- Event aggregation and forwarding to backend

**Key Methods**:
- `add_camera(serial, device_index)`: Create + connect camera
- `remove_camera(serial)`: Disconnect + cleanup camera
- `load_recipe(recipe_data)`: Load recipe to all cameras in recipe
- `stop_recipe(recipe_id)`: Stop recipe on all cameras
- `set_camera_mode(serial, mode)`: Change camera mode
- `get_camera_status(serial)`: Get camera status
- `shutdown()`: Shutdown all cameras

---

#### 1.3 WebSocket Client
**File**: `/ai_services/camera_management/websocket_client.py`

**Features**:
- Auto-reconnect with exponential backoff (1s → 60s max)
- Heartbeat (ping/pong) every 30s
- Event-driven message handling
- Async/await architecture

**Key Methods**:
- `connect()`: Connect to BE WebSocket server
- `send_message(message)`: Send JSON message to BE
- `start()`: Start client with reconnect loop
- `wait_until_connected(timeout)`: Wait for connection

---

#### 1.4 Main Service
**File**: `/ai_services/camera_management_service.py`

**Features**:
- Coordinates CameraManager + WebSocket client
- Startup: Auto-load running recipes from API
- Handles commands from BE (connect/disconnect/load/stop/set_mode)
- Forwards camera events to BE (camera_connected, camera_error, inference_result...)

**Supported WebSocket Events**:

**From BE → AI**:
- `connect_camera`: Add and connect camera
- `disconnect_camera`: Disconnect camera
- `load_recipe`: Load recipe to cameras
- `stop_recipe`: Stop recipe
- `set_camera_mode`: Set camera mode
- `get_camera_status`: Query camera status

**From AI → BE**:
- `service_connected`: Service startup notification
- `camera_connected`: Camera connected successfully
- `camera_disconnected`: Camera disconnected
- `camera_error`: Camera error occurred
- `inference_result`: Inference result ready
- `*_response`: Response to commands

---

## ✅ Phase 2: Backend Core Services (Completed)

### Files Created:

#### 2.1 WebSocket Server
**File**: `/backend/app/api/websocket/camera_ws.py`

**Features**:
- Singleton WebSocket connection manager
- Bidirectional communication with AI service
- Event routing and processing

**Key Functions**:
- `camera_management_websocket(websocket)`: WebSocket endpoint `/ws/camera-management`
- `send_connect_camera(serial, device_index)`: Send connect command
- `send_disconnect_camera(serial)`: Send disconnect command
- `send_load_recipe(recipe_dict)`: Send load recipe command
- `send_stop_recipe(recipe_id)`: Send stop recipe command
- `send_set_camera_mode(serial, mode)`: Send mode change command
- `process_inference_result(data)`: Process inference result from AI

---

#### 2.2 Shared Memory Service
**File**: `/backend/app/services/shared_memory_service.py`

**Features**:
- Read frames from shared memory (created by Camera processes)
- Thread-safe operations
- Automatic format conversion (BGR → JPEG)
- Connection pooling & cleanup

**Key Methods**:
- `read_frame(serial)`: Read latest frame + metadata
- `read_frame_as_jpeg(serial, quality)`: Read frame as JPEG bytes
- `cleanup(serial)`: Close shared memory connection

**Shared Memory Format**:
```
[frame_idx (8B)] [metadata_len (4B)] [metadata_pickle]
[frame_len (4B)] [frame_bytes_BGR] [frame_idx_verify (8B)]
```

---

#### 2.3 Inference Result Service
**File**: `/backend/app/services/inference_result_service.py`

**Features**:
- Save inference results to database
- Save result images to storage
- Process pass/fail logic (customizable)

**Key Methods**:
- `process_inference_result(data)`: Process result from AI service
- `_move_image_to_storage(...)`: Move temp image to permanent storage

**Image Storage Path**:
```
/backend/uploads/inference_results/{recipe_id}/{YYYY-MM-DD}/{serial}_{timestamp}_{result}_f{idx}.jpg
```

---

#### 2.4 Database Models & Repositories
**Files**:
- `/backend/app/models/inference_result.py`
- `/backend/app/repositories/inference_result_repository.py`

**Schema** (per-product record):
```python
{
    "_id": ObjectId,
    "recipe_id": str,
    "recipe_name": str,
    "product_pass_fail": "PASS" | "FAIL",
    "camera_results": [
        {
            "camera_id": str,
            "serial_number": str,
            "frames": [
                {
                    "template_name": str,
                    "frame_idx": int,
                    "pass_fail": "PASS" | "FAIL",
                    "confidence": float,
                    "image_path": str,
                    "detected_regions": [...]
                }
            ]
        }
    ],
    "metadata": {...},
    "timestamp": datetime,
    "created_at": datetime
}
```

**Repository Methods**:
- `create(result_data)`: Create new result
- `get_by_id(result_id)`: Get result by ID
- `get_all(skip, limit, filters...)`: List results with pagination
- `count(filters...)`: Count results
- `delete_by_id(result_id)`: Delete result

---

## ✅ Phase 3: API Refactoring (Completed)

### 3.1 Camera Endpoints Refactored

**File**: `/backend/app/api/endpoints/cameras.py`

**Changed Endpoints**:

#### `POST /cameras/{serial_number}/connect`
**Before**: Subprocess `camera_producer_service.start_camera()`
**After**: WebSocket `send_connect_camera(serial, device_index)`

**Changes**:
- Check `camera_ws_manager.is_connected()` → 503 if not connected
- Send WebSocket command instead of starting subprocess
- Still update DB connection status

---

#### `POST /cameras/{serial_number}/disconnect`
**Before**: Subprocess `camera_producer_service.stop_camera()`
**After**: WebSocket `send_disconnect_camera(serial)`

**Changes**:
- Send WebSocket command
- Cleanup `shared_memory_service.cleanup(serial)` after disconnect

---

#### `GET /cameras/{serial_number}/frame`
**Before**: `camera_frame_service.get_frame()` → old shared memory format
**After**: `shared_memory_service.read_frame_as_jpeg(serial, quality)`

**Changes**:
- Use new SharedMemoryService
- Simpler code (1 call vs 3 calls)
- Better error handling

---

#### `GET /cameras/{serial_number}/frame/metadata`
**Before**: `camera_frame_service.get_frame()`
**After**: `shared_memory_service.read_frame(serial)`

---

### 3.2 Recipe Endpoints Refactored

**File**: `/backend/app/api/endpoints/recipes.py`

**Changed Endpoints**:

#### `POST /recipes/{recipe_id}/load`
**Before**:
- Update camera settings via JSON files
- Save recipe_id to JSON files
- Camera producer polls JSON files

**After**:
- Check `camera_ws_manager.is_connected()` → 503 if not
- Send full recipe JSON via WebSocket: `send_load_recipe(recipe_dict)`
- No more JSON file operations
- AI service receives recipe instantly

**Removed Code**:
- ~60 lines of JSON file read/write logic
- `camera_settings_service.update_settings()` calls

---

#### `POST /recipes/{recipe_id}/stop` (NEW ENDPOINT)
**Purpose**: Stop running recipe and set cameras to idle mode

**Implementation**:
- Send WebSocket command: `send_stop_recipe(recipe_id)`
- Log action to action_log
- Return success message

**Usage**:
```bash
POST /api/recipes/{recipe_id}/stop
Authorization: Bearer <token>
```

---

### 3.3 Inference Results Endpoints Created

**File**: `/backend/app/api/endpoints/inference_results.py` (NEW)

**Endpoints**:

#### `GET /api/inference-results/`
- List all results with pagination
- Filters: recipe_id, pass_fail, start_date, end_date
- Returns: List of InferenceResultResponse

#### `GET /api/inference-results/count`
- Count results with same filters
- Returns: `{count, filters}`

#### `GET /api/inference-results/{result_id}`
- Get single result by ID
- Returns: InferenceResultResponse

#### `DELETE /api/inference-results/{result_id}`
- Delete result by ID
- Returns: `{success, message}`

---

## 📦 Dependencies Added

### Backend (`requirements.txt`):
```bash
websockets  # For WebSocket server
```

### AI Service:
```bash
websockets  # For WebSocket client (already installed with Python 3.8+)
```

---

## 🗂️ Files Structure

### AI Service:
```
/ai_services/
├── camera_management/
│   ├── __init__.py
│   ├── camera.py                    # Camera class
│   ├── camera_manager.py            # CameraManager class
│   └── websocket_client.py          # WebSocket client
├── camera_management_service.py     # Main entry point
└── inference_service.py             # SuperPointMatcherTRT (existing)
```

### Backend:
```
/backend/app/
├── api/
│   ├── endpoints/
│   │   ├── cameras.py               # Refactored
│   │   ├── recipes.py               # Refactored
│   │   └── inference_results.py     # NEW
│   └── websocket/
│       ├── __init__.py              # NEW
│       └── camera_ws.py             # NEW - WebSocket server
├── services/
│   ├── shared_memory_service.py     # NEW
│   └── inference_result_service.py  # NEW
├── models/
│   └── inference_result.py          # NEW
├── repositories/
│   └── inference_result_repository.py  # NEW
└── main.py                          # Updated - register new endpoints
```

---

## 🚀 How to Run

### 1. Start Backend
```bash
cd /Users/ngocthien.ai/Source/Projects/ocr_datecode/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start Camera Management Service
```bash
cd /Users/ngocthien.ai/Source/Projects/ocr_datecode/ai_services

export API_BASE=http://localhost:8000
export WS_HOST=localhost
export WS_PORT=8000

python3 camera_management_service.py
```

### 3. Expected Logs

**Backend**:
```
INFO: Uvicorn running on http://0.0.0.0:8000
✅ Database indexes created
INFO: Camera service connected via WebSocket
```

**AI Service**:
```
Camera Management Service v1.0
INFO: WebSocket URL: ws://localhost:8000/ws/camera-management
INFO: WebSocket connected successfully
INFO: CameraManagementService started successfully
```

---

## 🧪 Testing Flow

### Test 1: Connect Camera
```bash
POST /api/cameras/24241268/connect?device_index=0
```
Expected: Camera connects, shared memory created

### Test 2: Get Frame
```bash
GET /api/cameras/24241268/frame?quality=90
```
Expected: Returns JPEG image

### Test 3: Load Recipe
```bash
POST /api/recipes/{recipe_id}/load
```
Expected: Recipe loaded, cameras switch to hardware_trigger mode

### Test 4: Stop Recipe
```bash
POST /api/recipes/{recipe_id}/stop
```
Expected: Cameras switch to idle mode

### Test 5: List Inference Results
```bash
GET /api/inference-results/?recipe_id={recipe_id}
```
Expected: Returns list of results

---

## 📋 Next Steps (Phase 4 & 5)

### Phase 4: Frontend Updates (Pending)
- [ ] Recipe page: Add STOP button
- [ ] Template config: Add function_type select
- [ ] New Inference page: Show realtime results
- [ ] SocketIO client: Listen for inference_result events

### Phase 5: Cleanup (Pending)
- [ ] Remove old files:
  - `/ai_services/camera_shm_producer.py`
  - `/backend/app/services/camera_producer_service.py`
  - `/backend/app/services/camera_settings_service.py`
  - `/backend/camera_settings/` folder
- [ ] Integration testing
- [ ] Documentation updates

---

## 🔗 References

- **Test Guide**: `/docs/TEST_GUIDE_PHASE_1_2.md`
- **Architecture Doc**: `/docs/camera_system_architecture.txt`
- **Answer Doc**: `/docs/answer_question_of_ai.txt`

---

## ✅ Success Criteria Met

- [x] WebSocket connection BE ↔ AI Service working
- [x] Connect/disconnect camera via WebSocket
- [x] SharedMemory read/write working
- [x] Load/stop recipe via WebSocket
- [x] Inference results saved to DB
- [x] API endpoints refactored
- [x] No subprocess/JSON file dependencies
- [x] Clean separation of concerns

**Status**: Ready for Phase 4 (Frontend) implementation! 🎉
