# Async Inference & Reject Mechanism - Technical Summary

**Date:** 2026-01-12
**System:** Camera Management Service
**Features:** Async Inference + Automatic Reject Mechanism

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Pipeline với 3 Cameras & 10 Sản phẩm](#pipeline-example)
3. [Architecture Overview](#architecture)
4. [Components Detail](#components)
5. [Timing Calculation](#timing)
6. [Statistics & Monitoring](#statistics)
7. [Configuration](#configuration)
8. [Testing Guide](#testing)

---

## 🎯 Overview

### **Tính năng chính:**

1. **Async Inference**
   - ThreadPoolExecutor với 2 workers
   - Non-blocking inference execution
   - Parallel processing để tăng throughput

2. **Reject Mechanism**
   - Tự động reject sản phẩm FAIL/ERROR
   - Priority queue scheduler với timing chính xác
   - Digital Output (DO) control cho solenoid valve

### **Vấn đề được giải quyết:**

**TRƯỚC (Synchronous):**
```
Trigger #1 → Capture → Inference (500ms) → Block!
                         ↓
Trigger #2 → Capture → Wait... → Inference (500ms) → Block!
                                    ↓
→ DELAY TÍCH LŨY! Queue buildup!
```

**SAU (Asynchronous + Reject):**
```
Trigger #1 → Capture → Submit Job #1 → Return ngay
Trigger #2 → Capture → Submit Job #2 → Return ngay
                         ↓
              Worker-0: Job #1 (parallel)
              Worker-1: Job #2 (parallel)
                         ↓
              FAIL → Schedule Reject Timer
              PASS → Cancel Reject Timer
                         ↓
              Timer fires → DO2 pulse → Reject! 🔴
```

---

## 🏭 Pipeline với 3 Cameras & 10 Sản phẩm {#pipeline-example}

### **Cấu hình hệ thống:**

```
┌─────────────────────────────────────────────────────────────────────┐
│ Physical Layout:                                                     │
│                                                                      │
│  [Sensor] ──1.5s──> [Camera1] ──┐                                  │
│             1.5s──> [Camera2] ──┼── 2.5s ──> [Reject Station]      │
│             1.6s──> [Camera3] ──┘                                   │
│                                                                      │
│  Total distance: Sensor → Reject = 4.0s (delay_reject = 4000ms)    │
└─────────────────────────────────────────────────────────────────────┘

Camera Delays:
  - Camera1: delay_trigger = 1500ms
  - Camera2: delay_trigger = 1500ms
  - Camera3: delay_trigger = 1600ms

Reject Config:
  - delay_reject = 4000ms (sensor → reject station)
  - do_reject_number = 2 (DO2)
  - pulse_duration = 100ms

Inference:
  - Workers: 2 (parallel)
  - Avg time: 400-600ms per job
```

---

## 📊 Timeline: 10 Sản phẩm qua hệ thống

### **Phase 1: Sensor Triggers (0-2s)**

```
t=0.0s:  Product #1 → DI0 edge (0→1) → Group #1 created
t=0.2s:  Product #2 → DI0 edge (0→1) → Group #2 created
t=0.4s:  Product #3 → DI0 edge (0→1) → Group #3 created
t=0.6s:  Product #4 → DI0 edge (0→1) → Group #4 created
t=0.8s:  Product #5 → DI0 edge (0→1) → Group #5 created
t=1.0s:  Product #6 → DI0 edge (0→1) → Group #6 created
t=1.2s:  Product #7 → DI0 edge (0→1) → Group #7 created
t=1.4s:  Product #8 → DI0 edge (0→1) → Group #8 created
t=1.6s:  Product #9 → DI0 edge (0→1) → Group #9 created
t=1.8s:  Product #10 → DI0 edge (0→1) → Group #10 created
```

**DI Polling:** 100Hz (10ms interval)
**Action:** `trigger_cameras_group()` schedules 3 timers per group

---

### **Phase 2: Camera Captures (1.5s-3.4s)**

```
Group #1 (Product #1, triggered @ t=0.0s):
  t=1.5s: Camera1 & Camera2 capture (delay=1500ms)
  t=1.6s: Camera3 capture (delay=1600ms)
          → T_capture_complete = 1.6s
          → Emit "frames_captured" event

Group #2 (Product #2, triggered @ t=0.2s):
  t=1.7s: Camera1 & Camera2 capture
  t=1.8s: Camera3 capture
          → T_capture_complete = 1.8s

Group #3 (Product #3, triggered @ t=0.4s):
  t=1.9s: Camera1 & Camera2 capture
  t=2.0s: Camera3 capture
          → T_capture_complete = 2.0s

Group #4 (Product #4, triggered @ t=0.6s):
  t=2.1s: Camera1 & Camera2 capture
  t=2.2s: Camera3 capture
          → T_capture_complete = 2.2s

Group #5 (Product #5, triggered @ t=0.8s):
  t=2.3s: Camera1 & Camera2 capture
  t=2.4s: Camera3 capture
          → T_capture_complete = 2.4s

Group #6 (Product #6, triggered @ t=1.0s):
  t=2.5s: Camera1 & Camera2 capture
  t=2.6s: Camera3 capture
          → T_capture_complete = 2.6s

Group #7 (Product #7, triggered @ t=1.2s):
  t=2.7s: Camera1 & Camera2 capture
  t=2.8s: Camera3 capture
          → T_capture_complete = 2.8s

Group #8 (Product #8, triggered @ t=1.4s):
  t=2.9s: Camera1 & Camera2 capture
  t=3.0s: Camera3 capture
          → T_capture_complete = 3.0s

Group #9 (Product #9, triggered @ t=1.6s):
  t=3.1s: Camera1 & Camera2 capture
  t=3.2s: Camera3 capture
          → T_capture_complete = 3.2s

Group #10 (Product #10, triggered @ t=1.8s):
  t=3.3s: Camera1 & Camera2 capture
  t=3.4s: Camera3 capture
          → T_capture_complete = 3.4s
```

**Camera Action:** `execute_software_trigger_immediate()` per camera
**Output:** 3 frames per product (1 per camera)

---

### **Phase 3: Async Inference Submission (1.6s-3.4s)**

```
t=1.6s: Group #1 capture done → Submit Job #1 (queue depth: 0)
t=1.8s: Group #2 capture done → Submit Job #2 (queue depth: 1)
t=2.0s: Group #3 capture done → Submit Job #3 (queue depth: 2)
t=2.2s: Group #4 capture done → Submit Job #4 (queue depth: 3)
t=2.4s: Group #5 capture done → Submit Job #5 (queue depth: 4)
t=2.6s: Group #6 capture done → Submit Job #6 (queue depth: 5)
t=2.8s: Group #7 capture done → Submit Job #7 (queue depth: 6)
t=3.0s: Group #8 capture done → Submit Job #8 (queue depth: 7)
t=3.2s: Group #9 capture done → Submit Job #9 (queue depth: 8)
t=3.4s: Group #10 capture done → Submit Job #10 (queue depth: 9)
```

**Non-blocking:** `process_inference_async()` returns immediately
**Max Queue Depth:** 9 jobs waiting

---

### **Phase 4: Parallel Inference Execution (1.6s-5.4s)**

**2 Workers processing in parallel:**

```
Worker-0 Timeline:
  t=1.6s-2.1s:  Job #1 (0.5s) → Result: FAIL
  t=2.1s-2.6s:  Job #3 (0.5s) → Result: PASS
  t=2.6s-3.1s:  Job #5 (0.5s) → Result: FAIL
  t=3.1s-3.6s:  Job #7 (0.5s) → Result: PASS
  t=3.6s-4.1s:  Job #9 (0.5s) → Result: FAIL

Worker-1 Timeline:
  t=1.8s-2.3s:  Job #2 (0.5s) → Result: PASS
  t=2.3s-2.8s:  Job #4 (0.5s) → Result: FAIL
  t=2.8s-3.3s:  Job #6 (0.5s) → Result: PASS
  t=3.3s-3.8s:  Job #8 (0.5s) → Result: FAIL
  t=3.8s-4.3s:  Job #10 (0.5s) → Result: FAIL
```

**Summary:**
- FAIL: Products #1, #4, #5, #8, #9, #10 (6 products)
- PASS: Products #2, #3, #6, #7 (4 products)

---

### **Phase 5: Reject Scheduling (2.1s-4.3s)**

**Calculation for each FAIL product:**

```python
T_reject = T_capture_complete + (delay_reject / 1000.0)
delay_needed = (delay_reject / 1000.0) - inference_time
```

**Product #1 (FAIL @ t=2.1s):**
```
T_capture_complete = 1.6s
inference_time = 0.5s
→ T_reject = 1.6 + 4.0 = 5.6s
→ delay_needed = 4.0 - 0.5 = 3.5s
→ Schedule Timer(3.5s) → Fire @ t=5.6s ✓
```

**Product #2 (PASS @ t=2.3s):**
```
→ Cancel reject (if scheduled)
→ No DO output
```

**Product #3 (PASS @ t=2.6s):**
```
→ Cancel reject
```

**Product #4 (FAIL @ t=2.8s):**
```
T_capture_complete = 2.2s
inference_time = 0.5s
→ T_reject = 2.2 + 4.0 = 6.2s
→ Schedule Timer(3.4s) → Fire @ t=6.2s ✓
```

**Product #5 (FAIL @ t=3.1s):**
```
T_capture_complete = 2.4s
→ T_reject = 6.4s
→ Schedule → Fire @ t=6.4s ✓
```

**Product #6 (PASS @ t=3.3s):**
```
→ Cancel reject
```

**Product #7 (PASS @ t=3.6s):**
```
→ Cancel reject
```

**Product #8 (FAIL @ t=3.8s):**
```
T_capture_complete = 3.0s
→ T_reject = 7.0s
→ Schedule → Fire @ t=7.0s ✓
```

**Product #9 (FAIL @ t=4.1s):**
```
T_capture_complete = 3.2s
→ T_reject = 7.2s
→ Schedule → Fire @ t=7.2s ✓
```

**Product #10 (FAIL @ t=4.3s):**
```
T_capture_complete = 3.4s
→ T_reject = 7.4s
→ Schedule → Fire @ t=7.4s ✓
```

**Reject Queue @ t=4.3s:**
```
Priority Queue (sorted by T_reject):
  [Entry(T_reject=5.6s, group_id=1, do=2),
   Entry(T_reject=6.2s, group_id=4, do=2),
   Entry(T_reject=6.4s, group_id=5, do=2),
   Entry(T_reject=7.0s, group_id=8, do=2),
   Entry(T_reject=7.2s, group_id=9, do=2),
   Entry(T_reject=7.4s, group_id=10, do=2)]
```

---

### **Phase 6: Reject Execution (5.6s-7.5s)**

**Scheduler Thread checking every 10ms:**

```
t=5.6s:  Timer fires for Product #1
         → trigger_reject_pulse(do_number=2, pulse_ms=100)
         → DO2 = HIGH
         → sleep(100ms)
         → DO2 = LOW
         → Product #1 rejected! 🔴
         Log: "[Group #1] 🔴 REJECTING! DO2 pulse (100ms)"

t=6.2s:  Product #4 rejected! 🔴

t=6.4s:  Product #5 rejected! 🔴

t=7.0s:  Product #8 rejected! 🔴

t=7.2s:  Product #9 rejected! 🔴

t=7.4s:  Product #10 rejected! 🔴
```

**Physical Action:**
- DO2 HIGH → Solenoid valve opens
- Air blast pushes product off conveyor
- Product falls into reject bin

---

### **Phase 7: Products at Reject Station**

```
Products arriving at reject station (sensor + 4.0s):

t=4.0s:  Product #1 arrives → DO2 pulse @ t=5.6s → ❌ TOO LATE! (1.6s late)
         → Issue: Inference too slow! Product already passed.

CORRECTED TIMELINE (Realistic):
Product #1 triggered @ t=0.0s
Product #1 arrives @ reject @ t=4.0s (0.0 + 4.0)
Inference done @ t=2.1s (1.6 + 0.5)
DO pulse scheduled @ t=5.6s (1.6 + 4.0)
→ Problem: Product arrives @ t=4.0 but reject @ t=5.6!

ERROR IN LOGIC! Let me recalculate...
```

**🔧 CORRECTED CALCULATION:**

The issue is: `T_reject` should be calculated from **sensor trigger time**, not capture complete time!

```python
# CORRECT formula:
T_reject = T_sensor + (delay_reject / 1000.0)

# Example for Product #1:
T_sensor = 0.0s (when DI triggered)
T_capture_complete = 1.6s
T_inference_done = 2.1s
inference_time = 0.5s

→ T_product_at_reject = T_sensor + 4.0 = 0.0 + 4.0 = 4.0s

→ delay_needed = T_product_at_reject - T_inference_done
               = 4.0 - 2.1
               = 1.9s

→ Schedule Timer(1.9s) from NOW (t=2.1s)
→ Timer fires @ t=2.1 + 1.9 = 4.0s ✓ CORRECT!
```

**⚠️ CRITICAL FIX NEEDED:**

Current implementation uses:
```python
T_reject = T_capture_complete + delay_reject
```

Should be:
```python
T_reject = T_sensor + delay_reject
delay_needed = T_reject - T_inference_done
```

**Let me recalculate with CORRECT formula:**

---

### **🔄 CORRECTED Phase 5-7: With T_sensor**

Assuming we track `T_sensor` for each group:

```
Product #1:
  T_sensor = 0.0s
  T_capture_complete = 1.6s
  T_inference_done = 2.1s

  → T_product_at_reject = 0.0 + 4.0 = 4.0s
  → delay_needed = 4.0 - 2.1 = 1.9s
  → Schedule Timer(1.9s) @ t=2.1s
  → Fire @ t=4.0s ✓
  → DO2 pulse → Product #1 rejected exactly when it arrives! 🔴

Product #2:
  T_sensor = 0.2s
  T_inference_done = 2.3s (PASS)
  → Cancel reject

Product #3:
  T_sensor = 0.4s
  T_inference_done = 2.6s (PASS)
  → Cancel reject

Product #4:
  T_sensor = 0.6s
  T_inference_done = 2.8s
  → T_product_at_reject = 0.6 + 4.0 = 4.6s
  → delay_needed = 4.6 - 2.8 = 1.8s
  → Fire @ t=4.6s ✓

Product #5:
  T_sensor = 0.8s
  T_inference_done = 3.1s
  → Fire @ t=4.8s (0.8 + 4.0) ✓

Product #6-7: PASS → Cancel

Product #8:
  T_sensor = 1.4s
  T_inference_done = 3.8s
  → Fire @ t=5.4s (1.4 + 4.0) ✓

Product #9:
  T_sensor = 1.6s
  T_inference_done = 4.1s
  → Fire @ t=5.6s (1.6 + 4.0) ✓

Product #10:
  T_sensor = 1.8s
  T_inference_done = 4.3s
  → Fire @ t=5.8s (1.8 + 4.0) ✓
```

**Reject Execution:**
```
t=4.0s: DO2 pulse (100ms) → Product #1 rejected 🔴
t=4.6s: DO2 pulse (100ms) → Product #4 rejected 🔴
t=4.8s: DO2 pulse (100ms) → Product #5 rejected 🔴
t=5.4s: DO2 pulse (100ms) → Product #8 rejected 🔴
t=5.6s: DO2 pulse (100ms) → Product #9 rejected 🔴
t=5.8s: DO2 pulse (100ms) → Product #10 rejected 🔴
```

---

## 🏗️ Architecture Overview {#architecture}

### **System Components:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    CameraManagementService                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │               CameraManager (Main Orchestrator)           │  │
│  │                                                            │  │
│  │  ┌───────────────┐  ┌──────────────┐  ┌───────────────┐ │  │
│  │  │TriggerHandler │  │RejectScheduler│  │InferenceHandler│ │
│  │  │               │  │               │  │                │ │  │
│  │  │- DI Polling   │  │- Priority Q   │  │- ThreadPool    │ │
│  │  │- 100Hz        │  │- DO Control   │  │- 2 Workers     │ │
│  │  │- Edge Detect  │  │- Timer Mgmt   │  │- Async Jobs    │ │
│  │  └───────────────┘  └──────────────┘  └────────────────┘ │  │
│  │         ↓                   ↓                   ↓         │  │
│  │  ┌──────────────────────────────────────────────────────┐ │  │
│  │  │              Camera Pool (3 cameras)                  │ │
│  │  │  Camera1  |  Camera2  |  Camera3                     │ │
│  │  │  (Basler GigE - Serial: 12345, 12346, 12347)        │ │
│  │  └──────────────────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
          ↓                          ↓                   ↓
    [DI0 Pin]              [DO2 Pin - Solenoid]    [GPU - CUDA]
```

### **Data Flow:**

```
1. DI Edge Detection (100Hz polling)
   ↓
2. Camera Trigger Group Creation
   ↓
3. Multi-Camera Capture (with delays)
   ↓
4. Async Inference Submission (non-blocking)
   ↓
5. Parallel Inference Execution (2 workers)
   ↓
6. Reject Decision Logic
   ├─ FAIL → Schedule Reject Timer
   └─ PASS → Cancel Reject Timer
   ↓
7. Reject Scheduler (priority queue)
   ↓
8. DO Pulse Trigger (solenoid activation)
   ↓
9. Physical Product Rejection
```

---

## 🧩 Components Detail {#components}

### **1. TriggerHandler**

**File:** `trigger_handler.py`
**Responsibilities:**
- DI polling @ 100Hz
- Edge detection (0→1, 1→0, any)
- Camera trigger group management
- Timeout handling

**Key Methods:**
```python
class TriggerHandler:
    def start_polling()
        # Start DIPollingThread @ 100Hz

    def build_di_camera_map()
        # Map DI pins → camera lists

    def trigger_cameras_group(cameras)
        # Create group, schedule capture timers

    def _capture_and_register(camera, group_id)
        # Execute capture, track T_capture_complete
```

**Statistics:**
```python
self._stats = {
    'total_triggers': 0,
    'total_groups_created': 0,
    'total_groups_completed': 0,
    'total_groups_timeout': 0,
    'total_inferences': 0
}
```

---

### **2. InferenceHandler**

**File:** `inference_handler.py`
**Responsibilities:**
- Async inference job submission
- TensorRT batch inference
- OCR text verification
- Reject decision logic

**Key Methods:**
```python
class InferenceHandler:
    def __init__(reject_scheduler)
        # Initialize ThreadPoolExecutor (2 workers)

    def process_inference_async(cameras, results, T_capture_complete)
        # Submit job, return immediately

    def _do_inference_sync(job_id, cameras, results)
        # Worker function: run inference
        # Calculate inference_time
        # Call _handle_reject_decision()

    def _handle_reject_decision(group_id, overall_pass_fail, T_capture_complete, inference_time)
        # FAIL → schedule_reject()
        # PASS → cancel_reject()
```

**Statistics:**
```python
self._inference_stats = {
    'total_submitted': 0,
    'total_completed': 0,
    'total_failed': 0,
    'max_queue_depth': 0
}
```

---

### **3. RejectScheduler**

**File:** `reject_scheduler.py` (NEW)
**Responsibilities:**
- Priority queue management
- Timer scheduling
- DO pulse control
- Statistics tracking

**Key Methods:**
```python
class RejectScheduler:
    def start()
        # Start scheduler thread + monitoring thread

    def schedule_reject(group_id, T_capture_complete, inference_time, delay_reject, do_number)
        # Calculate delay_needed
        # Add to priority queue
        # Return success/failure

    def cancel_reject(group_id)
        # Mark entry as cancelled

    def _scheduler_loop()
        # Continuous loop checking queue every 10ms
        # Pop ready entries, trigger DO pulse

    def _execute_reject(entry)
        # Call trigger_reject_pulse()
```

**Data Structures:**
```python
@dataclass
class RejectEntry:
    T_reject: float       # Absolute time to fire
    group_id: int
    do_number: int
    inference_time: float
    scheduled_at: float
    cancelled: bool = False
```

**Statistics:**
```python
self._stats = {
    'total_scheduled': 0,
    'total_rejected': 0,    # Actually triggered
    'total_cancelled': 0,   # PASS case
    'total_missed': 0,      # Too late
    'max_queue_depth': 0
}
```

---

### **4. Camera**

**File:** `camera.py`
**Responsibilities:**
- Basler camera control (Pylon SDK)
- Software trigger execution
- Frame capture & storage
- Recipe configuration

**Key Attributes:**
```python
class Camera:
    serial_number: str
    delay_trigger: int     # ms, delay from sensor to camera
    delay_reject: int      # ms, delay from sensor to reject (recipe-level)
    do_reject_number: int  # DO pin number (recipe-level)
    function_type: str     # OCR, Check_Type_Product, etc.
    templates: List[Dict]
```

**Key Methods:**
```python
def load_recipe(recipe_data)
    # Load delay_reject, do_reject_number from recipe
    # Load camera-specific config
    # Load templates & annotations

def execute_software_trigger_immediate()
    # NO DELAY inside (timer already handled it)
    # ExecuteSoftwareTrigger()
    # RetrieveResult()
    # Return frames
```

---

### **5. Utils Module**

**File:** `utils.py`
**New Functions:**

```python
def write_do_value(do_number: int, value: int) -> bool
    """
    Set DO pin HIGH/LOW
    Command: sudo dio_out <do_number> <value>
    Returns: True if success
    """

def trigger_reject_pulse(do_number: int, pulse_ms: int = 100)
    """
    Trigger reject pulse sequence:
    1. write_do_value(do_number, 1)  # HIGH
    2. sleep(pulse_ms / 1000.0)
    3. write_do_value(do_number, 0)  # LOW

    Raises: RuntimeError if DO control fails
    """
```

---

## ⏱️ Timing Calculation {#timing}

### **Critical Timing Variables:**

```python
T_sensor            # Time when DI edge detected (0→1)
T_capture_complete  # Time when all cameras finished capturing
T_inference_start   # Time when inference worker started
T_inference_done    # Time when inference completed
T_product_at_reject # Time when product arrives at reject station
T_reject            # Time to trigger DO pulse

inference_time = T_inference_done - T_inference_start
```

### **Reject Timing Formula:**

**⚠️ IMPORTANT: Use T_sensor, not T_capture_complete!**

```python
# Correct calculation:
T_product_at_reject = T_sensor + (delay_reject / 1000.0)

delay_needed = T_product_at_reject - T_inference_done
             = (T_sensor + delay_reject/1000) - T_inference_done

T_reject = T_inference_done + delay_needed
         = T_product_at_reject
         = T_sensor + (delay_reject / 1000.0)
```

### **Edge Cases:**

**1. Inference Too Slow:**
```python
if delay_needed < 0:
    # Product already passed reject station!
    logger.error(f"INFERENCE TOO SLOW! Product passed {-delay_needed:.3f}s ago")
    stats['total_missed'] += 1
    return False  # Don't schedule
```

**2. Acceptable Delay Range:**
```python
# With delay_reject = 4000ms, inference should be < 3.5s
# Typical: 400-600ms → delay_needed = 3.4-3.6s → Safe! ✓
```

---

## 📈 Statistics & Monitoring {#statistics}

### **1. Trigger Statistics:**

```python
# Log every 10s:
📊 [STATS] Triggers=50, Groups=50/50 (100.0%), Inferences=50, Active=0g/0t

Fields:
- Triggers: Total DI edge detections
- Groups: Completed/Created (success rate)
- Inferences: Total inference jobs submitted
- Active: Active groups / Active timers
```

### **2. Inference Statistics:**

```python
📊 [INFERENCE STATS] Total: 50, Completed: 48, Failed: 2, Max Queue: 3

Fields:
- Total: Jobs submitted
- Completed: Successfully finished
- Failed: Jobs with exceptions
- Max Queue: Peak queue depth
```

### **3. Reject Statistics:**

```python
📊 [REJECT STATS] Scheduled: 25, Rejected: 24, Cancelled: 25, Missed: 1, Max Queue: 3, Active: 0

Fields:
- Scheduled: Total reject timers scheduled
- Rejected: DO pulses actually triggered
- Cancelled: PASS products (reject cancelled)
- Missed: Inference too slow (product passed)
- Max Queue: Peak reject queue depth
- Active: Current queue size
```

### **4. Per-Job Logs:**

```
[Job #1] Submitting inference for 3 camera(s) (queue depth: 0)
[Job #1] Starting inference in InferenceWorker-0 (group_id: 1)
[Job #1] Running BATCH inference on 3 cameras
[Job #1] Inference complete: FAIL, time: 0.450s
[Group #1] Reject scheduled @ T=4.000s (in 1.900s), DO2
[Group #1] 🕐 Reject scheduled (delay_reject=4000ms, DO2)
...
[Group #1] 🔴 REJECTING! DO2 pulse (100ms)
DO2 pulse complete (100ms)
[Group #1] Reject pulse complete
```

---

## ⚙️ Configuration {#configuration}

### **1. Recipe Level (MongoDB):**

```json
{
  "_id": "694850c56beac57d01bdf65c",
  "name": "Black Pepper 400 gram",
  "delay_reject": 4000,       // ms, sensor → reject station
  "do_reject_number": 2,      // DO pin number (DO2)
  "cameras": [
    {
      "serial_number": "24919818",
      "delay_trigger": 1500,  // ms, sensor → camera
      "function_type": "Check_Type_Product",
      ...
    },
    ...
  ]
}
```

### **2. Code Level:**

```python
# inference_handler.py
ThreadPoolExecutor(
    max_workers=2,  # Adjust based on CPU/GPU resources
    thread_name_prefix="InferenceWorker"
)

# reject_scheduler.py
pulse_ms = 100           # DO pulse duration (default)
_stats_interval = 10     # Log stats every N seconds

# trigger_handler.py
polling_frequency = 100  # Hz (10ms interval)
```

### **3. Environment Variables:**

```bash
# OCR Backend selection
export OCR_BACKEND="onnx"  # or "tensorrt"

# Log level
export LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR
```

---

## 🧪 Testing Guide {#testing}

### **1. Syntax Verification:**

```bash
cd /home/demo/Source/ocr_datecode/ai_services/camera_management

python3 -m py_compile reject_scheduler.py
python3 -m py_compile camera_manager.py
python3 -m py_compile inference_handler.py
python3 -m py_compile utils.py
```

### **2. Manual DO Testing:**

```bash
# Test DO2 control
sudo dio_out 2 1  # Set HIGH
sleep 0.1
sudo dio_out 2 0  # Set LOW

# Check all DO status
sudo dio_out

# Check all DI status
sudo dio_in
```

### **3. System Integration Test:**

```bash
# Start camera management service
python3 camera_management_service.py

# Load recipe via API
curl -X POST http://localhost:8000/api/camera-manager/recipes/load \
  -H "Content-Type: application/json" \
  -d '{"recipe_id": "694850c56beac57d01bdf65c"}'

# Monitor logs
tail -f backend/logs/camera_management.log | grep -E "(Job #|Reject|STATS)"
```

### **4. Expected Log Pattern:**

```
# Successful FAIL → Reject flow:
[Group #1] ✅ All 3 cameras completed! Triggering batch inference.
Frames captured (Group #1), submitting async inference for 3 camera(s)...
[Job #1] Submitting inference for 3 camera(s) (queue depth: 0)
[Job #1] Starting inference in InferenceWorker-0 (group_id: 1)
[Job #1] Running BATCH inference on 3 cameras
Camera 24919818 result: FAIL, confidence: 45.00%, inliers: 8/50
[Job #1] Inference complete: FAIL, time: 0.450s
[Group #1] Reject scheduled (delay_reject=4000ms, DO2)
[Group #1] 🕐 Reject scheduled @ T=4.000s (in 1.900s), DO2
...
[Group #1] 🔴 REJECTING! DO2 pulse (100ms)
DO2 = 1
DO2 = 0
DO2 pulse complete (100ms)
[Group #1] Reject pulse complete
```

### **5. Performance Benchmarks:**

```python
# Expected metrics:
- DI Polling Rate: 100Hz (measured)
- Camera Capture Time: 50-100ms per camera
- Inference Time: 400-600ms per job (3 cameras)
- Reject Timing Accuracy: ±50ms
- Throughput: 10 products/minute (with 2 workers)
```

---

## 🔧 Troubleshooting

### **Common Issues:**

**1. "Reject Too Late" errors:**
```
[Group #5] ❌ REJECT TOO LATE! inference_time=0.800s > delay_reject=0.500s
```
**Solution:**
- Increase `delay_reject` in recipe
- Optimize inference (reduce model size, improve GPU)
- Add more workers (increase `max_workers`)

**2. DO Control Failures:**
```
Failed to set DO2: Permission denied
```
**Solution:**
```bash
# Add user to gpio group
sudo usermod -a -G gpio $USER

# Or run service with sudo
sudo python3 camera_management_service.py
```

**3. High Queue Depth:**
```
[Job #10] Submitting inference for 3 camera(s) (queue depth: 9)
```
**Solution:**
- Increase workers: `max_workers=4`
- Reduce trigger frequency
- Optimize inference speed

---

## 📚 References

- **Basler Pylon SDK:** https://docs.baslerweb.com/pylonapi
- **TensorRT:** https://docs.nvidia.com/deeplearning/tensorrt/
- **Advantech DIO Commands:** `dio_in`, `dio_out` (system utilities)

---

**Document Version:** 1.0
**Last Updated:** 2026-01-12
**Author:** Claude Sonnet 4.5 + Human Collaboration
