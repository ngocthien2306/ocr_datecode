# Test Guide - Phase 1 & 2

## 📋 Tổng quan

Test flow cho Phase 1 (AI Service) và Phase 2 (Backend Core Services):
- WebSocket connection giữa BE ↔ AI Service
- Camera connect/disconnect commands
- Shared Memory đọc/ghi frame
- Inference result processing

---

## 🔧 Prerequisites

### 1. Backend Dependencies

```bash
cd /Users/ngocthien.ai/Source/Projects/ocr_datecode/backend

# Cài đặt WebSocket dependencies (nếu chưa có)
pip install websockets
```

### 2. AI Service Dependencies

```bash
cd /Users/ngocthien.ai/Source/Projects/ocr_datecode/ai_services

# Đã có sẵn:
# - pypylon (Basler camera SDK)
# - cv2 (OpenCV)
# - numpy
# - multiprocessing.shared_memory (Python 3.8+)

# Kiểm tra WebSocket client
pip show websockets || pip install websockets
```

### 3. Environment Variables

```bash
# Backend (.env hoặc export)
export API_BASE=http://localhost:8000

# AI Service (.bashrc hoặc export trước khi chạy)
export API_BASE=http://localhost:8000
export WS_HOST=localhost
export WS_PORT=8000
```

---

## 🚀 Test Flow

### ✅ Test 1: Backend WebSocket Server

**Mục tiêu**: Kiểm tra backend WebSocket endpoint hoạt động

**Bước 1**: Start backend server

```bash
cd /Users/ngocthien.ai/Source/Projects/ocr_datecode/backend

# Start FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Bước 2**: Verify endpoint

Mở browser hoặc dùng curl:
```bash
curl http://localhost:8000/
# Expected: {"app":"OCR Datecode API","version":"1.0.0","status":"running"}

curl http://localhost:8000/health
# Expected: {"status":"healthy"}
```

**Bước 3**: Check logs

```bash
# Logs sẽ hiển thị:
# INFO:     Started server process
# INFO:     Waiting for application startup.
# ✅ Database indexes created
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Expected Result**: ✅ Backend server running, WebSocket endpoint `/ws/camera-management` available

---

### ✅ Test 2: Camera Management Service Startup

**Mục tiêu**: Kiểm tra AI service kết nối WebSocket với backend

**Bước 1**: Make script executable

```bash
chmod +x /Users/ngocthien.ai/Source/Projects/ocr_datecode/ai_services/camera_management_service.py
```

**Bước 2**: Start camera management service

```bash
cd /Users/ngocthien.ai/Source/Projects/ocr_datecode/ai_services

# Set environment
export API_BASE=http://localhost:8000
export WS_HOST=localhost
export WS_PORT=8000

# Run service
python3 camera_management_service.py
```

**Expected Logs (AI Service)**:
```
================================================================================
Camera Management Service v1.0
================================================================================
INFO - CameraManagementService initialized
INFO - API Base: http://localhost:8000
INFO - WebSocket URL: ws://localhost:8000/ws/camera-management
INFO - Starting CameraManagementService...
INFO - Connecting to WebSocket server: ws://localhost:8000/ws/camera-management
INFO - WebSocket connected successfully
INFO - Sent message: service_connected
INFO - WebSocket connected, checking for running recipes...
INFO - Checking for running recipes...
INFO - No running recipes found (hoặc danh sách recipes nếu có)
INFO - CameraManagementService started successfully
```

**Expected Logs (Backend)**:
```
INFO - Camera service connected via WebSocket
INFO - Camera service event: service_connected
INFO - Camera service connected: {'service': 'camera_management'}
```

**Expected Result**: ✅ WebSocket connection established, no errors

---

### ✅ Test 3: Test Connect Camera Command

**Mục tiêu**: Test gửi command từ BE → AI service để connect camera

**Bước 1**: Tạo test script để gửi command

```bash
# Tạo file test script
cat > /Users/ngocthien.ai/Source/Projects/ocr_datecode/backend/test_ws_commands.py << 'EOF'
#!/usr/bin/env python3
"""
Test WebSocket commands to Camera Management Service
"""
import asyncio
import json
from app.api.websocket.camera_ws import camera_ws_manager

async def test_connect_camera():
    """Test connect camera command"""
    print("Testing connect camera command...")

    # Wait for service to connect
    await asyncio.sleep(2)

    # Check if service is connected
    if not camera_ws_manager.is_connected():
        print("❌ Camera service not connected!")
        return

    print("✅ Camera service connected")

    # Send connect camera command
    success = await camera_ws_manager.send_to_camera_service({
        "event": "connect_camera",
        "data": {
            "serial_number": "24241268",  # Replace with your camera serial
            "device_index": 0
        }
    })

    if success:
        print("✅ Connect camera command sent")
    else:
        print("❌ Failed to send command")

    # Wait for response
    await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(test_connect_camera())
EOF

chmod +x /Users/ngocthien.ai/Source/Projects/ocr_datecode/backend/test_ws_commands.py
```

**Bước 2**: Run test (trong terminal riêng, với backend đang chạy)

```bash
cd /Users/ngocthien.ai/Source/Projects/ocr_datecode/backend

# Run test
python3 test_ws_commands.py
```

**Expected Logs (AI Service)**:
```
INFO - Handling WS message: connect_camera
INFO - Camera instance created: 24241268 (device_index=0)
INFO - Opened: Basler acA2440-35uc (SN: 24241268)
INFO - Camera connected: ... resolution: 2464x2056, shared memory: camera_24241268
INFO - Sent message: connect_camera_response
```

**Expected Logs (Backend)**:
```
INFO - Sent to camera service: connect_camera
INFO - Received from camera service: connect_camera_response
INFO - Response received: connect_camera_response
```

**Expected Result**: ✅ Camera connected, shared memory created

---

### ✅ Test 4: Test Shared Memory Read

**Mục tiêu**: Test backend đọc frame từ shared memory

**Bước 1**: Verify camera đang chạy và ghi frame

```bash
# Check shared memory exists
ls -lh /dev/shm/ | grep camera_24241268
# Expected: File camera_24241268 với size > 0
```

**Bước 2**: Test read frame từ Python

```bash
cd /Users/ngocthien.ai/Source/Projects/ocr_datecode/backend

# Tạo test script
cat > test_shared_memory.py << 'EOF'
#!/usr/bin/env python3
"""Test SharedMemoryService"""
import sys
sys.path.insert(0, '/Users/ngocthien.ai/Source/Projects/ocr_datecode/backend')

from app.services.shared_memory_service import shared_memory_service
import cv2

# Test read frame
print("Testing SharedMemoryService.read_frame()...")

serial_number = "24241268"  # Replace with your camera serial
result = shared_memory_service.read_frame(serial_number)

if result:
    frame, metadata = result
    print(f"✅ Frame read successfully!")
    print(f"   Shape: {frame.shape}")
    print(f"   Dtype: {frame.dtype}")
    print(f"   Frame index: {metadata.get('frame_idx')}")
    print(f"   Timestamp: {metadata.get('timestamp')}")

    # Test JPEG encode
    jpeg_bytes = shared_memory_service.read_frame_as_jpeg(serial_number)
    if jpeg_bytes:
        print(f"✅ JPEG encoding successful: {len(jpeg_bytes)} bytes")
    else:
        print("❌ JPEG encoding failed")
else:
    print("❌ Failed to read frame")
EOF

chmod +x test_shared_memory.py
python3 test_shared_memory.py
```

**Expected Output**:
```
Testing SharedMemoryService.read_frame()...
✅ Frame read successfully!
   Shape: (2056, 2464, 3)
   Dtype: uint8
   Frame index: 123
   Timestamp: 1735819234.567
✅ JPEG encoding successful: 245678 bytes
```

**Expected Result**: ✅ Backend có thể đọc frame từ shared memory

---

### ✅ Test 5: Test Disconnect Camera

**Bước 1**: Tạo test disconnect script

```bash
cat > /Users/ngocthien.ai/Source/Projects/ocr_datecode/backend/test_disconnect.py << 'EOF'
#!/usr/bin/env python3
"""Test disconnect camera"""
import asyncio
from app.api.websocket.camera_ws import camera_ws_manager

async def test_disconnect():
    await asyncio.sleep(2)

    success = await camera_ws_manager.send_to_camera_service({
        "event": "disconnect_camera",
        "data": {
            "serial_number": "24241268"
        }
    })

    if success:
        print("✅ Disconnect command sent")

    await asyncio.sleep(3)

if __name__ == "__main__":
    asyncio.run(test_disconnect())
EOF

python3 test_disconnect.py
```

**Expected Logs (AI Service)**:
```
INFO - Handling WS message: disconnect_camera
INFO - Camera 24241268 removed successfully
INFO - Camera disconnected: 24241268
INFO - Sent message: disconnect_camera_response
```

**Expected Result**: ✅ Camera disconnected, shared memory cleaned up

---

## 🐛 Troubleshooting

### Issue 1: `ModuleNotFoundError: No module named 'websockets'`

```bash
pip install websockets
```

### Issue 2: WebSocket connection refused

**Check**:
- Backend running? `curl http://localhost:8000/health`
- Port 8000 available? `netstat -an | grep 8000`
- Firewall blocking?

### Issue 3: Camera not found

**Check**:
- Camera connected? `lsusb` (Linux) hoặc check Pylon Viewer
- Device index correct? Try device_index=0, 1, 2...
- Permissions? May need `sudo` for camera access

### Issue 4: Shared memory permission denied

```bash
# Check permissions
ls -l /dev/shm/camera_*

# Fix permissions (if needed)
sudo chmod 666 /dev/shm/camera_*
```

### Issue 5: Import errors trong AI service

```bash
cd /Users/ngocthien.ai/Source/Projects/ocr_datecode/ai_services

# Check Python path
python3 -c "import sys; print('\n'.join(sys.path))"

# Verify imports
python3 -c "from camera_management.camera import Camera; print('OK')"
```

---

## ✅ Success Criteria

Để Phase 1+2 pass tests, cần:

1. ✅ Backend WebSocket server running
2. ✅ AI Service connects to backend WebSocket
3. ✅ Connect camera command → Camera opens successfully
4. ✅ Backend reads frame from shared memory
5. ✅ Disconnect camera command → Cleanup successful
6. ✅ No errors in logs (trừ expected warnings như "No running recipes")

---

## 📝 Next Steps

Sau khi Phase 1+2 tests pass:
- **Phase 3**: Refactor API endpoints (cameras.py, recipes.py)
- **Phase 4**: Frontend updates
- **Phase 5**: Full integration test

---

## 🔗 Quick Reference

**Backend logs**: `/Users/ngocthien.ai/Source/Projects/ocr_datecode/backend/logs/backend.log`

**AI Service logs**: `/Users/ngocthien.ai/Source/Projects/ocr_datecode/backend/logs/camera_management.log`

**Shared memory**: `/dev/shm/camera_{serial_number}`

**WebSocket endpoint**: `ws://localhost:8000/ws/camera-management`
