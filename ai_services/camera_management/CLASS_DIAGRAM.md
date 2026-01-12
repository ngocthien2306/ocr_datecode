# Class Diagram - Camera Management System

## 🏗️ Architecture Overview

```mermaid
graph TB
    subgraph "Main Service"
        CMS[CameraManagementService]
    end

    subgraph "Core Manager"
        CM[CameraManager]
    end

    subgraph "Handlers"
        TH[TriggerHandler]
        IH[InferenceHandler]
        RS[RejectScheduler]
    end

    subgraph "Camera Pool"
        C1[Camera1]
        C2[Camera2]
        C3[Camera3]
    end

    subgraph "Hardware"
        DI[DI Pins]
        DO[DO Pins]
        GPU[GPU/CUDA]
    end

    CMS --> CM
    CM --> TH
    CM --> IH
    CM --> RS
    CM --> C1
    CM --> C2
    CM --> C3
    TH --> DI
    RS --> DO
    IH --> GPU
```

---

## 🔷 Detailed Class Diagram

```mermaid
classDiagram
    class CameraManagementService {
        -CameraManager camera_manager
        -WebSocketClient ws_client
        -asyncio.EventLoop event_loop
        +start()
        +stop()
        +_handle_camera_event(event_type, data)
        +_check_running_recipes()
    }

    class CameraManager {
        -Dict~str,Camera~ cameras
        -TriggerHandler trigger_handler
        -InferenceHandler inference_handler
        -RejectScheduler reject_scheduler
        -bool inference_enabled
        -threading.RLock _lock
        +add_camera(serial_number, pixel_format) Camera
        +remove_camera(serial_number) bool
        +connect_camera(serial_number) bool
        +disconnect_camera(serial_number) bool
        +load_recipe(recipe_data) Dict
        +process_inference(cameras, results)
        +set_inference_mode(enabled) Dict
        +shutdown()
        -_emit_event(event_type, data)
    }

    class TriggerHandler {
        -CameraManager camera_manager
        -Dict~int,List~ _di_camera_map
        -Dict~int,int~ _di_states
        -Dict~int,CaptureGroup~ _capture_groups
        -int _timer_counter
        -threading.Lock _group_lock
        -DIPollingThread _polling_thread
        -Dict _stats
        +build_di_camera_map()
        +start_polling()
        +stop_polling()
        +trigger_cameras_group(cameras)
        -_polling_loop()
        -_capture_and_register(camera, group_id)
        -_group_timeout(group_id)
    }

    class InferenceHandler {
        -Dict~str,Matcher~ camera_matchers
        -RejectScheduler reject_scheduler
        -ThreadPoolExecutor _inference_executor
        -Dict~int,Future~ _active_jobs
        -int _job_counter
        -Dict _inference_stats
        -TextRecognizer text_recognizer
        +init_matchers(cameras) int
        +clear_matchers()
        +process_inference_async(cameras, results, emit_callback, group_id, T_capture_complete) int
        +shutdown()
        -_do_inference_sync(job_id, cameras, results, emit_callback, group_id, T_capture_complete)
        -_cleanup_job(job_id, future)
        -_handle_reject_decision(group_id, overall_pass_fail, T_capture_complete, inference_time, cameras)
        -_build_inference_result(cameras, results, camera_inference_results, overall_pass_fail) Dict
        +verify_text_regions(frame_img, transformed_bboxes, expected_texts, camera) Dict
    }

    class RejectScheduler {
        -List~RejectEntry~ _reject_queue
        -Dict~int,RejectEntry~ _scheduled_rejects
        -threading.Lock _queue_lock
        -bool _running
        -threading.Thread _scheduler_thread
        -threading.Thread _monitoring_thread
        -Dict _stats
        +start()
        +stop()
        +schedule_reject(group_id, T_capture_complete, inference_time, delay_reject, do_number) bool
        +cancel_reject(group_id) bool
        +get_stats() Dict
        -_scheduler_loop()
        -_execute_reject(entry)
        -_monitoring_loop()
    }

    class Camera {
        -str serial_number
        -str model
        -str pixel_format
        -InstantCamera camera
        -int exposure_time
        -float gain
        -int delay_trigger
        -int delay_reject
        -int do_reject_number
        -str trigger_mode
        -str trigger_activation
        -int di_number
        -str recipe_id
        -str recipe_name
        -List~Dict~ templates
        -str function_type
        -Dict~int,str~ expected_texts
        +connect() bool
        +disconnect() bool
        +start_grabbing()
        +stop_grabbing()
        +load_recipe(recipe_data) bool
        +update_settings(settings)
        +configure_software_trigger() bool
        +execute_software_trigger_immediate() Dict
        +get_status() Dict
    }

    class RejectEntry {
        <<dataclass>>
        +float T_reject
        +int group_id
        +int do_number
        +float inference_time
        +float scheduled_at
        +bool cancelled
    }

    class CaptureGroup {
        <<TypedDict>>
        +List~Camera~ cameras
        +Dict~str,Dict~ results
        +int expected_count
        +int completed_count
        +float created_at
        +threading.Lock group_lock
        +threading.Timer timeout_timer
    }

    class Utils {
        <<module>>
        +read_di_value(di_number) int
        +write_do_value(do_number, value) bool
        +trigger_reject_pulse(do_number, pulse_ms)
        +check_trigger_edge(current, previous, activation) bool
        +resize_for_display(frame_img, scale_factor) ndarray
        +encode_image_to_base64(img, quality) str
        +draw_inference_bboxes(img, transformed_bboxes, confidence, inliers, total_matches) ndarray
        +save_and_encode_frame(...) Tuple
    }

    %% Relationships
    CameraManagementService --> CameraManager : manages
    CameraManager --> TriggerHandler : uses
    CameraManager --> InferenceHandler : uses
    CameraManager --> RejectScheduler : uses
    CameraManager --> Camera : manages pool
    InferenceHandler --> RejectScheduler : schedules/cancels
    TriggerHandler --> Camera : triggers
    TriggerHandler --> CaptureGroup : creates
    RejectScheduler --> RejectEntry : manages queue
    RejectScheduler --> Utils : calls DO control
    TriggerHandler --> Utils : calls DI read
    InferenceHandler --> Camera : reads config
```

---

## 📊 Sequence Diagram: Complete Flow

```mermaid
sequenceDiagram
    participant DI as DI Pin (Hardware)
    participant TH as TriggerHandler
    participant CAM as Camera Pool
    participant CM as CameraManager
    participant IH as InferenceHandler
    participant RS as RejectScheduler
    participant DO as DO Pin (Hardware)

    Note over DI,DO: Product passes sensor

    DI->>TH: Edge detected (0→1)
    activate TH
    TH->>TH: Record T_sensor
    TH->>TH: trigger_cameras_group()
    TH->>CAM: Schedule capture timers (3 cameras)
    deactivate TH

    Note over CAM: Timers fire based on delay_trigger

    CAM->>CAM: Camera1 capture (delay=1.5s)
    CAM->>CAM: Camera2 capture (delay=1.5s)
    CAM->>CAM: Camera3 capture (delay=1.6s)

    activate TH
    TH->>TH: All cameras done
    TH->>TH: Record T_capture_complete
    TH->>CM: Emit "frames_captured"
    deactivate TH

    activate CM
    CM->>IH: process_inference_async()
    deactivate CM

    activate IH
    IH->>IH: Submit job to ThreadPool
    IH-->>CM: Return job_id (non-blocking)
    deactivate IH

    Note over IH: Worker thread starts

    activate IH
    IH->>IH: Record T_inference_start
    IH->>IH: Run TensorRT inference (400-600ms)
    IH->>IH: Record T_inference_done
    IH->>IH: Calculate inference_time

    alt Result is FAIL
        IH->>RS: schedule_reject(group_id, T_capture_complete, inference_time, delay_reject, do_number)
        activate RS
        RS->>RS: Calculate T_reject & delay_needed
        RS->>RS: Add RejectEntry to priority queue
        RS-->>IH: Success
        deactivate RS
    else Result is PASS
        IH->>RS: cancel_reject(group_id)
        activate RS
        RS->>RS: Mark entry as cancelled
        RS-->>IH: Cancelled
        deactivate RS
    end

    IH->>CM: Emit "inference_result"
    deactivate IH

    Note over RS: Scheduler thread continuously checking

    activate RS
    RS->>RS: Check queue every 10ms
    RS->>RS: T_now >= T_reject?
    RS->>DO: trigger_reject_pulse(do_number, 100ms)
    activate DO
    DO->>DO: Set DO HIGH
    DO->>DO: Sleep 100ms
    DO->>DO: Set DO LOW
    DO-->>RS: Pulse complete
    deactivate DO
    deactivate RS

    Note over DO: Product rejected at exact timing! 🔴
```

---

## 🔄 State Diagram: Inference Job Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Submitted: process_inference_async()

    Submitted --> Queued: Add to ThreadPool queue

    Queued --> Running: Worker picks up job

    Running --> Inferencing: _do_inference_sync()

    Inferencing --> Processing: TensorRT batch inference

    Processing --> TextVerification: If function_type = Check_Type_Product

    TextVerification --> DecisionLogic: Calculate result

    Processing --> DecisionLogic: If standard inference

    DecisionLogic --> ScheduleReject: If FAIL/ERROR
    DecisionLogic --> CancelReject: If PASS

    ScheduleReject --> EmitResult: schedule_reject()
    CancelReject --> EmitResult: cancel_reject()

    EmitResult --> Completed: Emit to backend

    Completed --> [*]: Cleanup job

    Running --> Failed: Exception occurred
    Failed --> [*]: Log error, cleanup
```

---

## 🔧 Component Interaction Matrix

| Component | Triggers | Inference | Reject | Camera | Utils |
|-----------|----------|-----------|--------|--------|-------|
| **TriggerHandler** | - | ✓ Emit event | - | ✓ Capture | ✓ DI read |
| **InferenceHandler** | - | - | ✓ Schedule/Cancel | ✓ Read config | - |
| **RejectScheduler** | - | - | - | - | ✓ DO write |
| **CameraManager** | ✓ Build map | ✓ Process | ✓ Init/Shutdown | ✓ Manage pool | - |
| **Camera** | ✓ Execute trigger | - | - | - | - |

---

## 📦 Module Dependencies

```mermaid
graph LR
    subgraph "camera_management package"
        CMS[camera_management_service.py]
        CM[camera_manager.py]
        TH[trigger_handler.py]
        IH[inference_handler.py]
        RS[reject_scheduler.py]
        CAM[camera.py]
        UTIL[utils.py]
    end

    subgraph "External Dependencies"
        PYLON[pypylon - Basler SDK]
        TORCH[PyTorch/TensorRT]
        CV2[OpenCV]
        WS[WebSocket Client]
        SYS[System - dio_in/dio_out]
    end

    CMS --> CM
    CM --> TH
    CM --> IH
    CM --> RS
    CM --> CAM
    TH --> CAM
    TH --> UTIL
    IH --> RS
    IH --> UTIL
    RS --> UTIL
    CAM --> PYLON
    IH --> TORCH
    UTIL --> CV2
    UTIL --> SYS
    CMS --> WS
```

---

## 🎯 Threading Architecture

```mermaid
graph TB
    subgraph "Main Thread (CUDA Context)"
        MT[Main Thread]
        MT --> |Manages| CM[CameraManager]
        MT --> |Handles| Events[Event Callbacks]
    end

    subgraph "DI Polling Thread (100Hz)"
        PT[DIPollingThread]
        PT --> |Reads| DI[DI Pins]
        PT --> |Creates| Timers1[Capture Timers]
    end

    subgraph "Inference Workers (ThreadPool)"
        IW1[InferenceWorker-0]
        IW2[InferenceWorker-1]
        IW1 --> |Runs| TRT1[TensorRT Inference]
        IW2 --> |Runs| TRT2[TensorRT Inference]
    end

    subgraph "Reject Scheduler Thread"
        RST[Scheduler Thread]
        RMT[Monitoring Thread]
        RST --> |Checks| Queue[Priority Queue]
        RST --> |Triggers| DO[DO Pins]
        RMT --> |Logs| Stats[Statistics]
    end

    subgraph "Camera Capture Timers"
        CT1[Timer Group #1]
        CT2[Timer Group #2]
        CTN[Timer Group #N]
        CT1 --> |Fires| Capture1[Camera Capture]
        CT2 --> |Fires| Capture2[Camera Capture]
        CTN --> |Fires| CaptureN[Camera Capture]
    end

    CM --> PT
    CM --> IW1
    CM --> IW2
    CM --> RST
    CM --> RMT
    PT --> CT1
    PT --> CT2
    PT --> CTN
```

**Thread Summary:**
- **Main Thread (1):** Event loop, CUDA context, camera management
- **DI Polling Thread (1):** 100Hz polling, edge detection
- **Inference Workers (2):** Parallel inference execution
- **Reject Scheduler (1):** Priority queue checking, DO control
- **Reject Monitor (1):** Statistics logging
- **Capture Timers (N):** Dynamic, one per camera per trigger

**Total Threads:** 6 permanent + N dynamic (capture timers)

---

## 🔐 Thread Safety Mechanisms

```mermaid
graph TB
    subgraph "Shared Resources"
        SR1[_di_camera_map]
        SR2[_capture_groups]
        SR3[_active_jobs]
        SR4[_reject_queue]
        SR5[_scheduled_rejects]
        SR6[cameras dict]
    end

    subgraph "Locks"
        L1[_group_lock]
        L2[_job_lock]
        L3[_queue_lock]
        L4[_stats_lock]
        L5[_lock - RLock]
    end

    SR2 --> L1
    SR3 --> L2
    SR4 --> L3
    SR5 --> L3
    SR6 --> L5

    L1 --> |Protects| SR2
    L2 --> |Protects| SR3
    L3 --> |Protects| SR4
    L3 --> |Protects| SR5
    L5 --> |Protects| SR6
```

**Lock Hierarchy:**
1. `CameraManager._lock` (RLock) - Highest level, camera pool operations
2. `TriggerHandler._group_lock` - Capture group operations
3. `InferenceHandler._job_lock` - Job tracking
4. `RejectScheduler._queue_lock` - Reject queue operations
5. Per-group locks - Individual capture group coordination

**No Deadlocks:** Locks are acquired in consistent order, released promptly.

---

## 📐 Data Flow Architecture

```mermaid
flowchart TD
    A[DI Edge Event] -->|T_sensor| B[Create Capture Group]
    B --> C[Schedule 3 Camera Timers]
    C --> D1[Camera1 Capture<br/>delay=1.5s]
    C --> D2[Camera2 Capture<br/>delay=1.5s]
    C --> D3[Camera3 Capture<br/>delay=1.6s]

    D1 --> E[All Cameras Complete]
    D2 --> E
    D3 --> E

    E -->|T_capture_complete| F[Emit frames_captured Event]

    F --> G[Submit Async Inference Job]

    G --> H{Queue Depth?}
    H -->|< 2| I[Start Immediately]
    H -->|≥ 2| J[Queue for Next Worker]

    I --> K[Run TensorRT Inference]
    J --> K

    K -->|inference_time| L{Result?}

    L -->|FAIL/ERROR| M[Calculate Reject Timing]
    L -->|PASS| N[Cancel Reject if Scheduled]

    M --> O[Schedule Reject Timer]
    O --> P[Add to Priority Queue]

    P --> Q{T_now ≥ T_reject?}
    Q -->|No| R[Sleep until ready]
    R --> Q
    Q -->|Yes| S[Trigger DO Pulse]

    S --> T[DO2 = HIGH]
    T --> U[Sleep 100ms]
    U --> V[DO2 = LOW]
    V --> W[Product Rejected! 🔴]

    N --> X[Product Continues ✓]
```

---

## 🎨 Design Patterns Used

### **1. Observer Pattern**
```
CameraManager (Subject)
    ↓ _emit_event()
CameraManagementService (Observer)
    ↓ _handle_camera_event()
```

### **2. Producer-Consumer Pattern**
```
TriggerHandler (Producer)
    → frames_captured events
InferenceHandler (Consumer)
    → processes events in ThreadPool
```

### **3. Priority Queue Pattern**
```
RejectScheduler
    → heapq-based priority queue
    → Sorted by T_reject (earliest first)
```

### **4. Thread Pool Pattern**
```
InferenceHandler
    → ThreadPoolExecutor (2 workers)
    → Async job submission
```

### **5. Strategy Pattern**
```
Camera.function_type
    → "OCR" strategy
    → "Check_Type_Product" strategy
    → "Default" strategy
```

---

## 📊 Memory Footprint Estimate

| Component | Memory Usage | Notes |
|-----------|--------------|-------|
| Camera (Pylon) | ~50MB each | 3 cameras = 150MB |
| TensorRT Engine | ~200MB | Shared across jobs |
| Frame Buffers | ~10MB each | 3 cameras × N frames |
| Text Recognizer | ~100MB | ONNX model |
| ThreadPool | ~20MB | 2 workers |
| Reject Queue | ~1KB | Minimal overhead |
| **Total Estimate** | **~500-600MB** | Under normal load |

---

## 🔍 Key Metrics

| Metric | Value | Target |
|--------|-------|--------|
| DI Polling Rate | 100Hz | ±10% |
| Inference Latency | 400-600ms | <800ms |
| Reject Timing Accuracy | ±50ms | ±100ms acceptable |
| Throughput | 10 products/min | Scalable to 20+ |
| Memory Usage | 500-600MB | <1GB |
| CPU Usage | 20-40% | <60% |
| GPU Usage | 30-50% | <80% |

---

## ⚠️ Critical Issue Discovered

### **🐛 BUG: Incorrect Reject Timing Calculation**

**Current Implementation (WRONG):**
```python
# In reject_scheduler.py:
T_reject = T_capture_complete + (delay_reject / 1000.0)
```

**Issue:**
- `delay_reject` is measured from **SENSOR** to reject station
- NOT from camera to reject station!
- Products arrive at reject BEFORE reject pulse fires

**Correct Implementation:**
```python
# Should be:
T_reject = T_sensor + (delay_reject / 1000.0)
delay_needed = T_reject - T_inference_done
```

**Impact:**
- Products pass reject station before DO pulse
- 100% reject miss rate!

**Fix Required:**
1. Track `T_sensor` in `TriggerHandler.trigger_cameras_group()`
2. Pass `T_sensor` through event chain
3. Update `RejectScheduler.schedule_reject()` to use `T_sensor`

---

**Document Version:** 1.0
**Last Updated:** 2026-01-12
**Diagrams Generated:** Mermaid.js compatible
