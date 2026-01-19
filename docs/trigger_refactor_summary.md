# Camera Trigger Refactor - Implementation Summary

## Overview

Refactored camera trigger system from per-camera DI polling to centralized CameraManager polling. This eliminates race conditions and enables synchronized multi-camera capture.

**Implementation Date:** 2026-01-03
**Phase:** Phase 1 - Software Trigger only (Continuous and Hardware Trigger modes are TODO)

---

## Architecture Changes

### OLD Architecture (Problematic)
```
Camera 1 → Poll DI0 independently → Trigger → Capture → Race condition!
Camera 2 → Poll DI0 independently → Trigger → Capture → Race condition!
Camera 3 → Poll DI0 independently → Trigger → Capture → Race condition!
```

**Problems:**
- Race conditions during simultaneous triggers
- Buffer conflicts with `GrabStrategy_LatestImageOnly`
- Frame overwrites during multi-frame capture
- Continuous polling during trigger processing

### NEW Architecture (Centralized)
```
CameraManager → Poll DI0 @ 100Hz → Detect edge
  ↓
  Trigger all 3 cameras simultaneously (ThreadPoolExecutor)
  ↓
  Camera 1: delay → ExecuteSoftwareTrigger() → frame 0 → delay → frame 1 → delay → frame 2
  Camera 2: delay → ExecuteSoftwareTrigger() → frame 0 → delay → frame 1 → delay → frame 2
  Camera 3: delay → ExecuteSoftwareTrigger() → frame 0 → delay → frame 1 → delay → frame 2
  ↓
  Wait for all cameras to finish
  ↓
  If any fails: emit error (Option B)
  If all succeed: run inference on Camera 1 frame 0
  ↓
  Emit result
```

**Benefits:**
- No race conditions - single polling thread
- Synchronized multi-camera capture
- Proper error handling across cameras
- Uses `GrabStrategy_OneByOne` for reliable multi-frame capture

---

## Key Implementation Details

### 1. Trigger Modes

**Software Trigger Mode** (✅ Implemented - Phase 1)
- Jetson DI polling in CameraManager (100Hz)
- Calls `camera.ExecuteSoftwareTrigger()` via Pylon API
- Each camera configured with:
  - `trigger_selector`: FrameStart / ExposureStart / FrameBurstStart
  - `trigger_activation`: RisingEdge / FallingEdge / AnyEdge
  - `di_number`: 0-3 (which Jetson DI pin to poll)
  - `delay_trigger`: Delay BEFORE each frame (ms)

**Continuous Mode** (🔄 TODO)
- Free-running frame capture
- No trigger events

**Hardware Trigger Mode** (🔄 TODO)
- Camera Line inputs (Line0-Line3)
- Camera triggers autonomously via hardware pins

### 2. DI Polling Strategy

**Option A (Implemented):** Poll only DI pins that are actually used
- Build `di_camera_map: Dict[int, List[Tuple[Camera, str]]]` from recipe
- Only poll DI pins with cameras mapped to them
- More efficient - no unnecessary system calls

### 3. Multi-Frame Capture Timing

**IMPORTANT:** Delay BEFORE each frame (including first frame)

```
Trigger Event Detected
  ↓ delay_trigger (100ms)
  Frame 0 → Capture → Write to SHM
  ↓ delay_trigger (100ms)
  Frame 1 → Capture → Write to SHM
  ↓ delay_trigger (100ms)
  Frame 2 → Capture → Write to SHM
```

**Shared Memory Strategy (Option B):** Write all frames
- Each frame written to SHM with metadata
- Frontend Live View always has latest frame
- Metadata includes: timestamp, frame_idx, trigger_event flag

### 4. Error Handling

**Option B (Implemented):** Emit error if ANY camera fails
- If Camera 1 OK, Camera 2 TIMEOUT, Camera 3 OK
- → Emit `trigger_error` event
- → Do NOT run inference
- → Frontend displays error

**Reasoning:** Ensures complete data for inference. Partial results could lead to false positives/negatives.

### 5. Inference Logic

**Phase 1 (Simple - Implemented):**
- Run inference on first camera's first frame only
- Placeholder logic - emits PASS result
- TODO: Implement actual SuperPointMatcherTRT inference

**Phase 2 (Future - TODO):**
- Use all frames from all cameras
- Cross-camera analysis
- OCR on regions detected by SuperPointMatcherTRT

---

## Files Modified

### Backend Schema
**File:** `/backend/app/schemas/recipe.py`

**Changes:**
- Reordered `TriggerConfiguration` fields
- Added `trigger_mode` field to `CameraConfiguration`
- Updated field descriptions

```python
class TriggerConfiguration(BaseModel):
    trigger_selector: str = Field(default="FrameStart", ...)
    trigger_activation: str = Field(default="RisingEdge", ...)
    di_number: int = Field(default=0, ge=0, le=3, ...)  # Software Trigger
    trigger_source: str = Field(default="Line0", ...)   # Hardware Trigger

class CameraConfiguration(BaseModel):
    trigger_mode: str = Field(default="continuous", ...)  # NEW
    trigger_config: TriggerConfiguration = Field(default_factory=TriggerConfiguration)
```

### Frontend UI
**File:** `/frontend-ts/src/components/recipe/RecipeFormModal.tsx`

**Changes:**
- Updated `RecipeCamera` interface
- Changed trigger mode from nested to top-level field
- Conditional UI for Software Trigger mode
- Added TODO warning for Hardware Trigger mode (disabled)

**UI Fields for Software Trigger:**
- Trigger Selector dropdown (FrameStart, ExposureStart, FrameBurstStart)
- Digital Input (DI) Number dropdown (DI0-DI3)
- Trigger Activation dropdown (RisingEdge, FallingEdge, AnyEdge)

### Camera Class
**File:** `/ai_services/camera_management/camera.py`

**Major Changes:**

1. **Updated `CameraMode` enum:**
   - Added `SOFTWARE_TRIGGER` mode
   - Removed `HARDWARE_TRIGGER` (TODO)

2. **New attributes in `__init__`:**
   ```python
   self.trigger_mode = "continuous"
   self.trigger_selector = "FrameStart"
   self.trigger_activation = "RisingEdge"
   self.di_number = 0
   self.trigger_source = "Line0"
   self.delay_trigger = 100  # ms
   self.captured_frames = []
   ```

3. **New methods:**
   - `configure_software_trigger()` - Configure Pylon camera for software trigger
   - `execute_software_trigger()` - Execute trigger and capture N frames
   - Updated `update_settings()` - Handle new trigger fields
   - Updated `load_recipe()` - Configure trigger mode
   - Simplified `run()` loop - No more DI polling

4. **Removed methods (~230 lines):**
   - `_read_di_value()` - Moved to CameraManager
   - `_check_trigger_edge()` - Moved to CameraManager
   - `_handle_trigger_event()` - Replaced by `execute_software_trigger()`

5. **Fixed deprecation warnings:**
   - Changed `datetime.utcnow()` → `datetime.now(timezone.utc)`

### CameraManager Class
**File:** `/ai_services/camera_management/camera_manager.py`

**Major Changes:**

1. **New attributes in `__init__`:**
   ```python
   # DI Polling
   self._trigger_polling = False
   self._trigger_thread: Optional[threading.Thread] = None
   self._trigger_lock = threading.Lock()

   # DI → Cameras mapping
   self.di_camera_map: Dict[int, List[Tuple[Camera, str]]] = {0: [], 1: [], 2: [], 3: []}

   # Previous DI values for edge detection
   self.previous_di_values: Dict[int, Optional[int]] = {0: None, 1: None, 2: None, 3: None}
   ```

2. **New methods:**
   - `_read_di_value(di_number)` - Read Jetson DI pin via subprocess
   - `_check_trigger_edge(current, previous, activation)` - Detect edge
   - `_build_di_camera_map()` - Build DI → Camera mapping
   - `start_trigger_polling()` - Start 100Hz polling thread
   - `stop_trigger_polling()` - Stop polling thread
   - `_trigger_polling_loop()` - Main polling loop
   - `trigger_cameras_group(cameras)` - Trigger multiple cameras + inference

3. **Updated methods:**
   - `load_recipe()` - Build DI map and start polling
   - `stop_recipe()` - Rebuild DI map and stop polling if needed
   - `simulate_trigger()` - Use new `trigger_cameras_group()` architecture
   - `shutdown()` - Stop polling before shutdown

---

## Configuration Examples

### Example 1: 3 Cameras, Same DI Pin

All cameras trigger together on DI0 rising edge:

```json
{
  "cameras": [
    {
      "serial_number": "CAM001",
      "trigger_mode": "software_trigger",
      "delay_trigger": 100,
      "trigger_config": {
        "trigger_selector": "FrameStart",
        "trigger_activation": "RisingEdge",
        "di_number": 0
      }
    },
    {
      "serial_number": "CAM002",
      "trigger_mode": "software_trigger",
      "delay_trigger": 100,
      "trigger_config": {
        "trigger_selector": "FrameStart",
        "trigger_activation": "RisingEdge",
        "di_number": 0
      }
    },
    {
      "serial_number": "CAM003",
      "trigger_mode": "software_trigger",
      "delay_trigger": 100,
      "trigger_config": {
        "trigger_selector": "FrameStart",
        "trigger_activation": "RisingEdge",
        "di_number": 0
      }
    }
  ]
}
```

**Result:** DI0 rising edge → All 3 cameras trigger simultaneously

### Example 2: Different DI Pins

Cameras can use different DI pins (flexibility for future):

```json
{
  "cameras": [
    {
      "serial_number": "CAM001",
      "trigger_mode": "software_trigger",
      "trigger_config": {
        "di_number": 0,  // Different DI pin
        "trigger_activation": "RisingEdge"
      }
    },
    {
      "serial_number": "CAM002",
      "trigger_mode": "software_trigger",
      "trigger_config": {
        "di_number": 1,  // Different DI pin
        "trigger_activation": "FallingEdge"
      }
    }
  ]
}
```

**Result:**
- DI0 rising edge → CAM001 triggers
- DI1 falling edge → CAM002 triggers

---

## Testing Checklist

### Basic Tests
- [ ] Load recipe with 3 cameras on DI0
- [ ] Verify DI camera map built correctly
- [ ] Verify trigger polling started (100Hz)
- [ ] Simulate trigger - all 3 cameras capture
- [ ] Verify all frames written to shared memory
- [ ] Verify inference runs on first camera's first frame
- [ ] Verify inference result emitted
- [ ] Stop recipe - verify polling stopped

### Edge Detection Tests
- [ ] Test RisingEdge (0→1)
- [ ] Test FallingEdge (1→0)
- [ ] Test AnyEdge (both)
- [ ] Test no trigger on same value (1→1, 0→0)

### Error Handling Tests
- [ ] Simulate Camera 1 OK, Camera 2 timeout
- [ ] Verify error event emitted
- [ ] Verify no inference run
- [ ] Disconnect camera mid-trigger
- [ ] Verify error handling

### Multi-Template Tests
- [ ] Recipe with 1 template - verify 1 frame captured
- [ ] Recipe with 3 templates - verify 3 frames captured
- [ ] Verify delay BEFORE each frame
- [ ] Verify all frames written to SHM

### Mixed Mode Tests (TODO - Future)
- [ ] Recipe with Camera 1 Software Trigger, Camera 2 Continuous
- [ ] Verify only Software Trigger camera in DI map

---

## Known Limitations / TODO

### Phase 1 Limitations
1. **Continuous Mode not implemented**
   - UI shows mode but backend doesn't handle it
   - Camera.run() has placeholder code

2. **Hardware Trigger Mode not implemented**
   - UI shows mode as disabled with TODO warning
   - Backend schema has fields but no logic

3. **Simple inference only**
   - Only uses first camera's first frame
   - Full multi-camera inference planned for Phase 2

4. **No actual inference logic**
   - Placeholder code emits "PASS" result
   - TODO: Integrate SuperPointMatcherTRT

### Future Enhancements
1. Implement Continuous Mode
2. Implement Hardware Trigger Mode (Camera Line inputs)
3. Support mixed trigger modes per recipe
4. Implement full multi-camera inference
5. Add OCR logic for detected regions
6. Performance optimization for high-frequency triggers
7. Add trigger statistics (count, timing, errors)

---

## Troubleshooting

### Issue: Trigger polling not starting
**Check:**
- Is recipe loaded?
- Are cameras in `software_trigger` mode?
- Check logs for "Trigger polling started (100Hz)"

### Issue: Cameras not triggering
**Check:**
- Is DI camera map built correctly? (Check logs)
- Is DI pin value changing? (Use `sudo dio_in 0` manually)
- Are cameras in SOFTWARE_TRIGGER mode?
- Check trigger activation matches edge (RisingEdge vs FallingEdge)

### Issue: Some cameras fail to capture
**Check:**
- Camera connection OK?
- Camera in grabbing state?
- Check timeout settings (10s default)
- Check logs for specific camera errors

### Issue: Frames overwritten
**Should not happen with new architecture**
- Verify using `GrabStrategy_OneByOne` for Software Trigger
- Check Camera.configure_software_trigger() logs

---

## Performance Notes

### DI Polling Frequency
- **Current:** 100Hz (10ms interval)
- **Overhead:** Minimal - only polls used DI pins
- **Latency:** ~10ms from edge to trigger detection

### Multi-Camera Trigger
- **Method:** ThreadPoolExecutor (parallel)
- **Timeout:** 10s per camera
- **Expected time:** ~(delay_trigger * num_templates) per camera

### Example Timing
**3 cameras, 3 templates each, 100ms delay:**
```
DI edge detected at T=0ms
  ↓ Parallel trigger (all cameras start simultaneously)

Camera 1: 0ms → 100ms → frame 0 → 200ms → frame 1 → 300ms → frame 2 (done at 300ms)
Camera 2: 0ms → 100ms → frame 0 → 200ms → frame 1 → 300ms → frame 2 (done at 300ms)
Camera 3: 0ms → 100ms → frame 0 → 200ms → frame 1 → 300ms → frame 2 (done at 300ms)

  ↓ All complete at T=~300ms
  ↓ Run inference
  ↓ Emit result at T=~320ms
```

**Total latency:** ~320ms from trigger to result

---

## Code References

### Camera Configuration
- Schema: `/backend/app/schemas/recipe.py:43-61` (CameraConfiguration)
- Frontend: `/frontend-ts/src/components/recipe/RecipeFormModal.tsx`

### Trigger Execution
- Camera: `/ai_services/camera_management/camera.py:472-537` (execute_software_trigger)
- CameraManager: `/ai_services/camera_management/camera_manager.py:700-780` (trigger_cameras_group)

### DI Polling
- Polling loop: `/ai_services/camera_management/camera_manager.py:650-698` (_trigger_polling_loop)
- Edge detection: `/ai_services/camera_management/camera_manager.py:559-582` (_check_trigger_edge)

### Recipe Loading
- CameraManager: `/ai_services/camera_management/camera_manager.py:203-270` (load_recipe)
- Camera: `/ai_services/camera_management/camera.py:418-469` (load_recipe)

---

## Contact / Questions

For questions about this implementation, refer to:
- Requirements doc: `/docs/refactor_trigger_logic.txt`
- This summary: `/docs/trigger_refactor_summary.md`
- Code comments in modified files
