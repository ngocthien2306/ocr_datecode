# Phase 5: Remaining Cleanup Tasks

## Status

**Partially completed** - Some files have been deprecated, but additional refactoring is needed.

## Completed

- [X] ✅ Deprecated `/ai_services/camera_shm_producer.py` → renamed to `.deprecated`
- [X] ✅ Deprecated `/backend/camera_settings/` → renamed to `.deprecated`

## Remaining Tasks

### 1. Refactor Camera Status and Settings Endpoints

The following endpoints in `/backend/app/api/endpoints/cameras.py` still use old services:

#### Endpoint: `GET /cameras/{serial_number}/status`

- **Line**: 598
- **Current**: Uses `camera_producer_service.get_camera_status()`
- **TODO**: Refactor to use WebSocket command `get_camera_status` or query from CameraManager directly

#### Endpoint: `POST /cameras/{serial_number}/settings`

- **Lines**: 608-657
- **Current**: Uses `camera_settings_service.update_settings()`
- **TODO**: Refactor to use WebSocket command `send_set_camera_mode` or create new command for settings update

#### Endpoint: `GET /cameras/{serial_number}/settings`

- **Lines**: 660-694
- **Current**: Uses `camera_settings_service.get_settings()`
- **TODO**: Refactor to use WebSocket query or retrieve from database

### 2. Remove Old Service Files

Once the above endpoints are refactored, delete these files:

```bash
rm /Users/ngocthien.ai/Source/Projects/ocr_datecode/backend/app/services/camera_producer_service.py
rm /Users/ngocthien.ai/Source/Projects/ocr_datecode/backend/app/services/camera_settings_service.py
```

And update `/backend/app/services/__init__.py`:

```python
# Remove these imports:
# from .camera_producer_service import camera_producer_service
# __all__ list update
```

### 3. Update Imports in Endpoints

**File**: `/backend/app/api/endpoints/cameras.py`

- Remove line 22: `from app.services import camera_frame_service, camera_producer_service, shared_memory_service`
- Remove line 23: `from app.services.camera_settings_service import camera_settings_service`
- Add WebSocket commands as needed

**File**: `/backend/app/api/endpoints/recipes.py`

- Remove line 600: `from app.services.camera_settings_service import camera_settings_service`

### 4. Recommended Approach

#### Option A: Add WebSocket Commands (Recommended)

Add these commands to `/backend/app/api/websocket/camera_ws.py`:

```python
async def send_get_camera_status(serial_number: str) -> dict:
    """Get camera status via WebSocket"""
    # Implementation

async def send_update_camera_settings(serial_number: str, settings: dict) -> dict:
    """Update camera settings via WebSocket"""
    # Implementation
```

And implement handlers in `/ai_services/camera_management_service.py`:

- `get_camera_status` command handler
- `update_camera_settings` command handler

#### Option B: Query from Database (Alternative)

Store camera settings in the database and query from there instead of JSON files.

## Testing Checklist

Before deleting the old service files, test:

- [ ] Camera status retrieval works
- [ ] Camera settings update works
- [ ] Camera settings retrieval works
- [ ] No references to `camera_producer_service` remain
- [ ] No references to `camera_settings_service` remain
- [ ] No references to `camera_settings/` folder remain

## Notes

- The old files have been renamed to `.deprecated` to prevent accidental use
- They can be restored if needed during development
- Once refactoring is complete and tested, permanently delete the `.deprecated` files
