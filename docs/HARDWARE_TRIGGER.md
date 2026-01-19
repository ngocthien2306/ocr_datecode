# Hardware Trigger Documentation

## Tổng quan

Hệ thống hỗ trợ 3 chế độ hoạt động cho camera:

1. **Continuous Mode** (Free-running): Camera capture liên tục
2. **Software Trigger**: Camera chờ trigger command từ software
3. **Hardware Trigger**: Camera capture khi nhận signal từ Digital Input (DI)

## Digital I/O Specifications

- **Digital Inputs (DI)**: 4 ports (DI 0, DI 1, DI 2, DI 3)
- **Digital Outputs (DO)**: 4 ports (DO 0, DO 1, DO 2, DO 3)
- **Voltage Levels**: 0 (LOW) hoặc 1 (HIGH)

## Hardware Trigger Configuration

### 1. Trigger Modes

- **Continuous**: Camera grab frames liên tục (default)
- **Software**: Camera chờ software trigger (reserved for future)
- **Hardware**: Camera chờ signal từ DI để trigger

### 2. Trigger Activation

Khi chọn Hardware Trigger, bạn có thể chọn trigger edge:

- **Rising Edge (0→1)**: Trigger khi DI chuyển từ LOW sang HIGH
- **Falling Edge (1→0)**: Trigger khi DI chuyển từ HIGH sang LOW
- **Any Edge**: Trigger khi DI thay đổi (cả 0→1 và 1→0)

### 3. DI Number Selection

Chọn DI port để monitor (0-3)

## Setup Instructions

### Bước 1: Configure Sudoers (Khuyến nghị)

Để camera producer không cần nhập password mỗi lần đọc DI:

```bash
cd /home/demo/Source/ocr_datecode
sudo bash scripts/setup_dio_sudoers.sh
```

Script này sẽ:
- Tạo sudoers rule cho `dio_in` và `dio_out`
- Cho phép user `demo` chạy lệnh không cần password
- Test xem config có hoạt động không

### Bước 2: Test Digital I/O

#### Test đọc DI:
```bash
# Đọc tất cả DI
sudo dio_in

# Đọc DI cụ thể
sudo dio_in 0
sudo dio_in 1
```

#### Test ghi DO:
```bash
# Set DO 0 = HIGH
sudo dio_out 0 1

# Set DO 0 = LOW
sudo dio_out 0 0

# Xem tất cả DO
sudo dio_out
```

#### Test bằng Python script:
```bash
cd /home/demo/Source/ocr_datecode
python3 test_dio.py
```

Chọn options:
1. Read all DI values
2. Write DO values
3. Test edge detection logic
4. Hardware trigger simulation

### Bước 3: Configure Recipe

1. Mở Recipe Form → Tab "Camera Configuration"
2. Chọn camera cần config
3. Tìm section **"Trigger Configuration"**
4. Chọn **Trigger Mode** = "Hardware Trigger (DI)"
5. Chọn **Digital Input Number** (0-3)
6. Chọn **Trigger Activation** (Rising/Falling/Any Edge)
7. Save recipe

### Bước 4: Load Recipe

1. Vào Receipts page
2. Click "Load Recipe" cho recipe đã config
3. Backend sẽ apply trigger config vào camera
4. Camera producer sẽ hot-reload settings và chuyển sang hardware trigger mode

## Working Flow

```
User creates Recipe
  → Select Trigger Mode = "Hardware"
  → Config DI = 0, Activation = "RisingEdge"
  → Save Recipe

User clicks "Load Recipe"
  → Backend applies trigger config to camera settings
  → Camera producer hot-reloads settings
  → Producer switches to hardware trigger mode
  → Producer polls DI 0
  → When DI 0: 0→1 → Capture image
```

## Camera Producer Behavior

### Continuous Mode
```python
while grabbing:
    grab_frame()  # Always grab
```

### Hardware Trigger Mode
```python
previous_di = read_di_value(di_number)

while grabbing:
    current_di = read_di_value(di_number)

    if check_trigger_edge(current_di, previous_di, activation):
        grab_frame()  # Trigger detected!
        logger.info(f"⚡ TRIGGER: DI {di_number}: {previous_di}→{current_di}")

    previous_di = current_di
    sleep(0.01)  # 10ms polling interval
```

## Digital Output (DO) Configuration

Recipe cũng có field **DO Reject Number** trong tab "Basic Information":
- Chọn DO port (0-3) để control reject mechanism
- Hiện tại chỉ lưu config, chưa auto-trigger
- Có thể dùng manual hoặc trong logic xử lý sau

## Testing Hardware Trigger

### Simulation Test (Python)
```bash
python3 test_dio.py
# Chọn option 4: Hardware trigger simulation
# Script sẽ monitor DI 0 và detect trigger
```

### Manual Test với Hardware

1. Start camera producer với hardware trigger config
2. Dùng switch/button để toggle DI
3. Hoặc dùng DO để simulate:
```bash
# Toggle DO 0 để simulate DI signal
sudo dio_out 0 1
sudo dio_out 0 0
```

4. Check camera producer log:
```
🔧 [HARDWARE TRIGGER] Monitoring DI 0, activation: RisingEdge
⚡ [TRIGGER] Hardware trigger detected on DI 0: 0→1
```

## Troubleshooting

### Camera không trigger khi toggle DI

1. **Check DI connection**: `sudo dio_in 0` - Verify giá trị thay đổi
2. **Check trigger config**: Xem log camera producer
3. **Check activation mode**: RisingEdge cần 0→1, FallingEdge cần 1→0
4. **Check polling**: Camera producer phải running

### Sudo password prompts

1. Run sudoers setup script:
```bash
sudo bash scripts/setup_dio_sudoers.sh
```

2. Verify:
```bash
sudo -l | grep dio
```

Should show:
```
NOPASSWD: /usr/bin/dio_in, /usr/bin/dio_out
```

### Performance Issues

- Hardware trigger mode polls DI every 10ms
- If performance issue, increase sleep time in producer code
- Default: 10ms = 100 Hz polling rate

## API Reference

### Python Functions

#### `read_di_value(di_number, sudo_password="1")`
Đọc giá trị Digital Input

**Parameters:**
- `di_number` (int): DI number (0-3)
- `sudo_password` (str): Password for sudo (default: "1")

**Returns:**
- `int`: 0 or 1

**Example:**
```python
value = read_di_value(0)
print(f"DI 0 = {value}")
```

#### `write_do_value(do_number, value, sudo_password="1")`
Ghi giá trị Digital Output

**Parameters:**
- `do_number` (int): DO number (0-3)
- `value` (int): 0 or 1
- `sudo_password` (str): Password for sudo (default: "1")

**Returns:**
- `bool`: True if successful

**Example:**
```python
success = write_do_value(0, 1)  # Set DO 0 = HIGH
```

#### `check_trigger_edge(current, previous, activation)`
Check trigger condition

**Parameters:**
- `current` (int): Current DI value
- `previous` (int): Previous DI value
- `activation` (str): "RisingEdge", "FallingEdge", or "AnyEdge"

**Returns:**
- `bool`: True if trigger condition met

**Example:**
```python
if check_trigger_edge(1, 0, "RisingEdge"):
    print("Trigger detected!")
```

## Notes

- Password mặc định: "1"
- DI/DO range: 0-3 (4 ports each)
- Polling interval: 10ms (configurable)
- Hot-reload: Camera producer auto-reload settings mỗi 2 giây

## Advanced: Custom Password

Nếu sudo password không phải "1", update trong code:

**File**: `ai_services/camera_shm_producer.py`

```python
# Change default password
read_di_value(di_number, sudo_password="your_password")
write_do_value(do_number, value, sudo_password="your_password")
```

Hoặc set environment variable và đọc từ config file.
