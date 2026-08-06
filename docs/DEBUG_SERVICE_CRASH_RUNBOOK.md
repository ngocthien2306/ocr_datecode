# Runbook — Điều tra service crash trên Jetson

> Viết sau sự cố backend chết SIGABRT ngày **2026-08-05**. Mục đích: lần sau gặp
> "service tự chết / overlay Service Down nhảy lên" thì lần theo đúng thứ tự này,
> không phải mò lại từ đầu.

---

## 1. Khi nào dùng tài liệu này

- Backend hoặc AI camera service tự tắt, tự restart
- FE hiện overlay `⚠ Backend Service` / `⚠ Camera Service`
- Dây chuyền dừng inference mà không rõ lý do
- `systemctl status` báo `failed` / `Restart` liên tục

---

## 2. Bản đồ triển khai — phải nắm trước khi debug

Sai lầm tốn thời gian nhất là **debug nhầm tiến trình**. Trên Jetson:

| Thành phần                | Chạy bằng gì                                                                                               | Log ở đâu                                                                                                                                         |
| --------------------------- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Backend** (uvicorn) | `ocr-backend.service` (systemd, `Restart=always`) — hoặc `nohup` inline nếu :8000 chưa listen       | app:`logs/backend/YYYY-MM-DD.log`stdout: `logs/start_services/backend.log`                                                                       |
| **AI camera service** | `nohup python camera_management_service.py` từ `start_services.sh:187` — **KHÔNG phải systemd** | app:`logs/camera_management/YYYY-MM-DD.log`stdout: `logs/start_services/ai_camera.log`khi BE respawn: `ai_services/logs/camera_management.log` |
| **Frontend** (Vite)   | `nohup yarn dev`                                                                                            | `logs/start_services/frontend.log`                                                                                                                 |
| `ocr-all.service`         | `Type=oneshot`, chỉ chạy `start_services.sh`                                                            | journal                                                                                                                                              |

### Ba cái bẫy

1. **`logs/start_services/` bị xóa sạch mỗi lần khởi động** — `start_services.sh:21` có
   `rm -rf "$LOG_DIR"`. Traceback của cú crash sẽ **mất** nếu máy đã reboot hoặc script chạy lại.
   👉 **Việc đầu tiên khi vào máy: copy thư mục đó ra chỗ khác.**
2. **AI service do backend respawn ghi log vào file KHÁC** (`ai_services/logs/camera_management.log`).
   Nhìn nhầm file sẽ tưởng service không chạy.
3. **`ocr-ai-services.service` là unit chết, không phải AI service thật.** Nếu thấy nó restart hàng
   chục nghìn lần thì đó là bẫy (xem §6), đừng đi theo.

---

## 3. Quy trình 6 bước

### Bước 0 — Chụp hiện trường NGAY

```bash
cd ~/Source/ocr_datecode
mkdir -p /tmp/crash_$(date +%F_%H%M) && D=$_
cp -r logs/start_services "$D"/ 2>/dev/null
cp logs/backend/$(date +%F).log "$D"/be_app.log 2>/dev/null
cp logs/camera_management/$(date +%F).log "$D"/ai_app.log 2>/dev/null
echo "Saved to $D"
```

### Bước 1 — Xác định AI CHẾT và CHẾT KIỂU GÌ

```bash
ps -eo pid,etime,rss,cmd | grep -E "uvicorn|camera_management_service" | grep -v grep
systemctl status ocr-backend --no-pager
systemctl show ocr-backend -p NRestarts -p ExecMainStatus -p Result
sudo journalctl -u ocr-backend --since "7 days ago" --no-pager \
  | grep -E "ABRT|KILL|Failed with result|Scheduled restart"
```

> ⚠️ `NRestarts` / `Result` / `ExecMainStatus` mô tả **lần chạy hiện tại**, không phải lần crash.
> `Result=success` không có nghĩa là chưa từng crash.

**Đọc mã thoát — đây là ngã ba quyết định hướng điều tra:**

| Mã           | Tín hiệu | Kết luận                                                         | Đi tiếp bước |
| ------------- | ---------- | ------------------------------------------------------------------ | ---------------- |
| `9/KILL`    | SIGKILL    | **OOM killer** — hết RAM thật                             | 4                |
| `6/ABRT`    | SIGABRT    | Process**tự gọi `abort()`** từ code native              | 3                |
| `11/SEGV`   | SIGSEGV    | Truy cập bộ nhớ sai                                             | 3                |
| `7/BUS`     | SIGBUS     | Đọc mmap/shared memory đã bị hủy hoặc thu nhỏ              | 3                |
| `1/FAILURE` | exit 1     | Lỗi Python thường — đọc traceback trong log app              | 2                |
| `217/USER`  | —         | **User trong unit không tồn tại**, python chưa hề chạy | 6                |
| `203/EXEC`  | —         | Đường dẫn`ExecStart` sai                                     | 6                |

### Bước 2 — Log ứng dụng quanh thời điểm chết

```bash
cd ~/Source/ocr_datecode
sed -n '/08:5[0-9]:/p' logs/backend/$(date +%F).log | tail -80
sed -n '/08:5[0-9]:/p' logs/camera_management/$(date +%F).log | tail -80

# dấu vết supervisor: BE có tự phát hiện & respawn AI service không
grep -nE "Camera service down|self-healed|PROCESS is dead|forcing restart|kill_respawn|restart (succeeded|FAILED)|STALE-SHM|frame_idx stuck|Watchdog" \
  logs/backend/$(date +%F).log | tail -40
```

Ý nghĩa các dòng supervisor:

| Dòng                                           | Nghĩa                                                    |
| ----------------------------------------------- | --------------------------------------------------------- |
| `self-healed after Ns`                        | AI tự kết nối lại, BE không đụng gì — tốt nhất |
| `PROCESS is dead — skipping grace`           | Tiến trình chết hẳn, restart ngay không chờ         |
| `kill_respawn: start → ✅`                   | BE hồi sinh AI thành công                              |
| `STALE-SHM RECOVERY triggered`                | Shared memory hỏng — BE tự kill+respawn camera         |
| `Skipping restart — last restart was Ns ago` | Đang trong cửa sổ debounce 60s                         |

### Bước 3 — Signal crash: lấy core dump (đường ngắn nhất)

Với `6/ABRT`, `11/SEGV`, `7/BUS`, apport thường đã ghi sẵn crash report.

```bash
sudo ls -lt /var/crash/
sudo apport-unpack /var/crash/<file>.crash /tmp/cd
cat /tmp/cd/Signal /tmp/cd/ProcCmdline
grep -E "VmRSS|VmPeak|VmSwap|VmHWM|Threads|State" /tmp/cd/ProcStatus
```

**Backtrace — bước quyết định:**

```bash
sudo apt install -y gdb
gdb /usr/bin/python3.10 /tmp/cd/CoreDump -batch \
  -ex "bt" -ex "thread apply all bt" 2>&1 | tee /tmp/bt.txt | head -120
```

Không cần `python3.10-dbg`. Tìm frame **ngay trên `__GI_abort`**:

| Frame thấy trong backtrace                               | Kết luận                                          |
| --------------------------------------------------------- | --------------------------------------------------- |
| `pycuda...tls_destructor` + `__nptl_deallocate_tsd`   | **PyCUDA abort khi thread chết** (§5)       |
| `std::terminate` / `__cxa_throw` / `std::bad_alloc` | C++ exception không bắt / hết bộ nhớ tầng C++ |
| `free` / `malloc` / `_int_free` trong `libc`      | Heap corruption                                     |
| `libonnxruntime` / `Ort::`                            | Lỗi trong ONNX Runtime                             |
| `libcuda` / `libnvinfer`                              | Lỗi driver GPU / TensorRT                          |
| `cv::`                                                  | Assertion của OpenCV                               |

#### Hai cái bẫy khi đọc core dump

- **Timestamp lệch rất xa.** Core dump lần này nặng **635 MB**; apport mất **87 giây** để ghi.
  Process abort lúc `08:54:39` (ghi trong `.crash`) nhưng systemd chỉ báo `Main process exited`
  lúc `08:56:06`. **Luôn tin timestamp trong file `.crash`**, không tin dòng của systemd.
- **`VmPeak` không phải RAM thật.** Lần này `VmPeak = 13.4 GB` nhưng chỉ là địa chỉ ảo do CUDA
  reserve. RAM vật lý đỉnh là `VmHWM` (900 MB). Con số đáng lo là **`VmSwap`**.

### Bước 4 — Loại trừ áp lực bộ nhớ

```bash
sudo dmesg -T | grep -iE "out of memory|oom-kill|killed process"
sudo journalctl -k -b -1 --no-pager | grep -iE "oom|killed process"
free -h; swapon --show
ps -eo pid,rss,comm --sort=-rss | head -10
```

> OOM killer **luôn** gửi SIGKILL (`9/KILL`). Nếu mã thoát là `6/ABRT` thì **không phải** OOM killer —
> nhưng vẫn có thể là hết bộ nhớ theo đường khác (`std::bad_alloc` → terminate → abort), phân biệt
> bằng backtrace ở bước 3.

### Bước 5 — Loại trừ hạ tầng

```bash
uptime; last -x reboot shutdown | head
sudo dmesg -T | grep -iE "thermal|throttl|undervolt|soctherm"
tegrastats --interval 1000
journalctl --list-boots
```

Nếu `journalctl --list-boots` chỉ có boot hiện tại → journal **không** lưu vĩnh viễn, bật lên:

```bash
sudo mkdir -p /var/log/journal && sudo systemctl restart systemd-journald
```

### Bước 6 — Lỗi cấu hình unit (`217/USER`, `203/EXEC`)

Không phải bug code. Kiểm tra:

```bash
systemctl cat <unit> | grep -E "User=|Group=|ExecStart="
id <user_trong_unit>          # "no such user" = nguyên nhân
```

---

## 4. Nhiễu che mất bằng chứng

Nếu một unit hỏng đang restart liên tục, journald sẽ **rate-limit** và nuốt mất log crash thật:

```bash
sudo journalctl --since "..." --until "..." --no-pager | grep -i "rate-limit\|Suppressed"
```

Có dòng đó → journal không còn tin được cho khoảng thời gian đó, phải dựa vào `/var/crash` và log app.

Lọc nhiễu khi đọc journal:

```bash
sudo journalctl --since "2026-08-05 08:54:00" --until "2026-08-05 08:57:30" \
  -o short-precise --no-pager \
  | grep -vE "ocr-ai-services|AICWFDBG|NetworkManager|wpa_supplicant|rtkit-daemon|cpufreq:"
```

---

## 5. Case study — 2026-08-05

### Timeline

| Thời điểm       | Sự kiện                                                                     |
| ------------------ | ----------------------------------------------------------------------------- |
| 08:53:32           | `frame_idx stuck at 6377` → STALE-SHM RECOVERY → kill+respawn camera      |
| 08:53:41           | Watchdog thấy AI down (do chính recovery vừa giết) → chờ 15s self-heal  |
| 08:53:57           | Hết grace → forcing restart                                                 |
| 08:54:01           | BE respawn camera thành công —**tiến trình con của backend**      |
| **08:54:39** | **Backend abort (SIGABRT)** — apport bắt đầu ghi core 635 MB        |
| 08:54:39→08:56:06 | Backend treo cứng 87 giây trong lúc ghi core                               |
| 08:56:06           | systemd mới thấy`Main process exited, code=killed, status=6/ABRT`         |
| 08:56:12–17       | Unit stop → systemd SIGKILL cả cgroup →**camera service chết theo** |
| 08:56:17           | Backend restart                                                               |
| 08:56:31           | Watchdog backend mới:`Camera service PROCESS is dead`                      |
| 08:56:34           | Respawn camera thành công                                                   |
| 08:56:36           | AI service kết nối lại, overlay tắt                                       |

Tổng downtime ~2 phút, trong đó **87 giây chỉ để ghi core dump**.

### Nguyên nhân gốc — PyCUDA TLS destructor

Backtrace:

```
#3  __GI_abort ()
#4  pycudaboost::thread_specific_ptr<pycuda::context_stack>::delete_data::operator()(void*)
#6  tls_destructor.part ()          ← pycuda/_driver...so
#7  __GI___nptl_deallocate_tsd ()   ← thread đang KẾT THÚC, dọn TLS
#8  start_thread
```

Chuỗi:

1. `recipes.py` (cũ) có endpoint `POST /cv-preview` làm `sys.path.insert(repo_root)` rồi
   `from ai_services.camera_management.verification.embedding_classifier import ...`
2. Import đó buộc Python chạy `camera_management/__init__.py` → `verification/__init__.py`
   → `wrinkle_segmenter.py` → **`import pycuda.driver`**
3. PyCUDA đăng ký một **pthread TLS destructor**. Từ đó **mọi thread backend khi kết thúc** đều chạy
   nó để dọn `pycuda::context_stack`. Dọn hỏng → ném C++ exception **bên trong destructor** →
   `std::terminate()` → `abort()`
4. Endpoint là `def` (sync) nên FastAPI chạy trong **anyio threadpool** — đúng loại thread hay bị thu hồi

Vì sao trông ngẫu nhiên: PyCUDA chỉ vào backend **kể từ lần đầu có người mở tab Model** của form
recipe. Trước đó backend chạy nhiều giờ hoàn toàn bình thường.

### Ba vấn đề phát hiện kèm

| # | Vấn đề                                                                                                                                                                                                                        | Xử lý                                                                                                                        |
| - | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 1 | Backend import`ai_services.*` → kéo PyCUDA/TensorRT vào                                                                                                                                                                     | Xóa hẳn tính năng cv-preview (BE + FE + CSS). Dropdown CV Method đã khóa ở v4 từ trước nên panel preview vô dụng |
| 2 | `ocr-backend.service` dùng `KillMode` mặc định (`control-group`) → mỗi lần BE restart là SIGKILL luôn camera service do BE spawn (`start_new_session=True` tách session nhưng **không thoát cgroup**) | Thêm`KillMode=mixed`                                                                                                        |
| 3 | `ocr-ai-services.service` hardcode `User=demo` trên máy user `suntech` → `217/USER` → loop fork-rồi-chết mỗi 5s, **35.515 lần**, ngập journal                                                             | Thêm vào vòng gỡ unit trong`setup_systemd_services.sh` + `reset-failed`                                                |

~~⚠️ **Còn tồn**: `scripts/install_services.sh:35` vẫn cp `ocr-ai-services.service` vào
`/etc/systemd/system/`.~~ → **Đã xử lý 2026-08-06**: gỡ khỏi mảng `SERVICES`, kèm comment cảnh báo
không thêm lại. Text hướng dẫn trong `install_services.sh` / `setup_all.sh` / `install.sh` cũng đã
bỏ các lệnh `systemctl start|stop ocr-ai-services` (chúng trỏ tới unit không còn tồn tại).

---

## 5b. Áp dụng cho `release_v2` — 2026-08-06

`release_v2` là nhánh máy dev (`/home/msi`, conda env `vision`), **chưa từng nhận** các commit fix
của `release_v1`, nên vẫn dính nguyên lỗi. Đã port sang, nhưng **cách xử lý khác v1 ở một điểm**.

### Vì sao khác

`release_v1` xoá sạch mọi endpoint có import `ai_services`. `release_v2` có thêm 3 endpoint mà v1
**chưa từng có** — chúng phục vụ ColorSetupModal / EdgeSetupModal (tinh chỉnh `color_config` /
`edge_config`), là tính năng đang dùng:

| Endpoint | Module cần | Có CUDA không |
| --- | --- | --- |
| `POST /templates/detect-bottle-preview` | `color_verifier.py` | **Không** — chỉ cv2 + numpy |
| `POST /templates/detect-walls-preview` | `image_proc_detector.py` | **Không** — cv2 + numpy + scipy |
| `POST /templates/detect-cap-edges-preview` | `image_proc_detector.py` | **Không** |

Điểm mấu chốt: **hai module này chưa bao giờ cần CUDA**. PyCUDA lọt vào chỉ vì Python phải chạy
`camera_management/__init__.py` → `verification/__init__.py` → `wrinkle_segmenter.py` trên đường
import. Tức là "giữ preview" và "backend sạch CUDA" **không hề mâu thuẫn**.

### Cách làm — nạp thẳng file, không đụng package

`backend/app/services/cv_detectors.py` đăng ký một **package tổng hợp** (`_ocr_cv_detectors`) có
`__path__` trỏ vào thư mục `verification/`, rồi nạp 2 file bằng `spec_from_file_location`.
Không `__init__.py` nào được chạy.

Vì sao phải là package tổng hợp chứ không phải `spec_from_file_location` trần:
`color_verifier.py:477` có `from .image_proc_detector import ...` **chạy lúc runtime** — relative
import cần một package cha. Không có `__path__` thì nó ném lỗi, và đáng sợ hơn: chỗ đó nằm trong
`try/except` chỉ `logger.warning` rồi bỏ qua, nên sẽ **hỏng âm thầm** (mất cap-axis correction) chứ
không báo gì. Đã test đúng câu import đó chạy được.

(`color_verifier.py:30` có `from ..camera import Camera` nhưng nằm dưới `if TYPE_CHECKING:` → không
chạy lúc runtime, không cần xử lý.)

### Những gì đã đổi trên `release_v2`

| # | Thay đổi | File |
| - | --- | --- |
| 1 | Xoá hẳn `POST /cv-preview` + panel FE + CSS (giống v1; dropdown CV Method khoá ở v4 nên panel vô dụng) | `recipes.py`, `RecipeFormModal.tsx`, `RecipeFormModal.css` |
| 2 | 3 endpoint preview còn lại: bỏ `sys.path.insert` + `from camera_management...`, chuyển sang loader | `recipes.py`, `cv_detectors.py` (mới) |
| 3 | Char OCR trong backend ép **CPU-only** (bỏ `CUDAExecutionProvider`) — chạy trên thread của `run_in_executor`, đúng loại thread gây abort | `ml_char_ocr_service.py` |
| 4 | Thêm `KillMode=mixed` + đưa `ocr-ai-services` vào vòng gỡ unit + `reset-failed` (port từ v1, **giữ nguyên** phần conda `vision` riêng của v2) | `setup_systemd_services.sh` |
| 5 | Gỡ `ocr-ai-services.service` khỏi `install_services.sh` | `install_services.sh`, `setup_all.sh`, `install.sh` |

### Kiểm chứng đã chạy

```bash
cd backend && python3 -c "
import sys; sys.path.insert(0,'.')
from app.main import app
bad=[m for m in sys.modules if m.split('.')[0] in ('pycuda','tensorrt','camera_management','ai_services')]
print('CUDA/ai_services modules:', bad or 'NONE')"
# -> NONE   (172 routes đăng ký)
```

Ngoài ra: gọi thẳng cả 3 endpoint với ảnh template thật → chạy hết code detector, trả JSON đúng
shape, `cuda_modules_present()` vẫn rỗng. FE `tsc --noEmit`: 30 lỗi trước và 30 lỗi sau, **trùng
khớp từng dòng** (đều là lỗi có sẵn của nhánh, không phải do sửa lần này).

---

## 6. Nguyên tắc phòng ngừa

1. **Backend TUYỆT ĐỐI không import package `ai_services.*`.** `__init__.py` của
   `camera_management` và `verification` đều eager-import TensorRT/PyCUDA. Cần dùng chung thuật toán
   thì có 2 đường:
   - File thuần `cv2`/`numpy` → **nạp thẳng file** qua `app/services/cv_detectors.py` (xem §5b).
     Không copy, không lệch bản, không chạy `__init__.py` nào.
   - Nếu buộc phải copy sang `backend/app/services/` thì nhớ nó sẽ trôi khỏi bản gốc theo thời gian.
2. **Không để thư viện CUDA vào tiến trình có threadpool động.** PyCUDA + thread ngắn hạn = abort.
   Áp dụng cho **cả `onnxruntime` với `CUDAExecutionProvider`**, không riêng PyCUDA — đó là lý do
   `ml_char_ocr_service.py` bị ép CPU-only.
3. **Kiểm tra định kỳ**: `sudo grep -c pycuda /proc/$(pgrep -f "uvicorn app.main")/maps` phải bằng `0`.
   Trong Python: `from app.services.cv_detectors import cuda_modules_present` → phải trả về `[]`.
4. **Đừng ghi core dump khổng lồ trên máy production.** 635 MB = thêm ~90 giây downtime mỗi lần
   crash. Sau khi mổ xong thì tắt apport.
5. **Đừng để unit hỏng loop** — nó ngập journal và làm mất log crash thật.
6. **Cân nhắc rotate thay vì `rm -rf` `logs/start_services/`** trong `start_services.sh:21`, giữ 2–3
   lần chạy gần nhất để không mất bằng chứng.

---

## 7. Phụ lục — script gom log một phát

```bash
cd ~/Source/ocr_datecode
OUT=/tmp/crash_$(date +%F_%H%M).txt
FROM="2026-08-05 08:54:00"; TO="2026-08-05 08:57:30"   # sửa khung giờ
{
echo "########## 1. CRASH REPORTS ##########"
sudo ls -lt /var/crash/ 2>/dev/null

echo; echo "########## 2. TRẠNG THÁI UNIT ##########"
systemctl status ocr-backend --no-pager
systemctl show ocr-backend -p NRestarts -p ExecMainStatus -p Result -p KillMode
ps -eo pid,etime,rss,cmd | grep -E "uvicorn|camera_management_service" | grep -v grep

echo; echo "########## 3. JOURNAL (đã lọc nhiễu) ##########"
sudo journalctl --since "$FROM" --until "$TO" -o short-precise --no-pager \
  | grep -vE "ocr-ai-services|AICWFDBG|NetworkManager|wpa_supplicant|rtkit-daemon|cpufreq:"

echo; echo "########## 4. LOG APP BACKEND ##########"
tail -150 logs/backend/$(date +%F).log

echo; echo "########## 5. LOG APP AI SERVICE ##########"
tail -100 logs/camera_management/$(date +%F).log

echo; echo "########## 6. AI DO BACKEND RESPAWN ##########"
tail -60 ai_services/logs/camera_management.log 2>/dev/null

echo; echo "########## 7. DẤU VẾT SUPERVISOR ##########"
grep -anE "Camera service down|self-healed|PROCESS is dead|forcing restart|kill_respawn|restart (succeeded|FAILED)|STALE-SHM|frame_idx stuck|Watchdog" \
  logs/backend/$(date +%F).log | tail -40

echo; echo "########## 8. BỘ NHỚ + KERNEL ##########"
free -h; swapon --show
ps -eo pid,rss,comm --sort=-rss | head -10
sudo dmesg -T | grep -iE "oom|killed process|thermal|throttl" | tail -20
} > "$OUT" 2>&1
echo "→ $OUT"
```

---

## 8. Tham chiếu code

| Vị trí                                                     | Nội dung                                                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| `backend/app/services/cv_detectors.py`                     | Nạp `image_proc_detector.py` / `color_verifier.py` **không kéo CUDA** (§5b)                |
| `backend/app/services/camera_service_supervisor.py`        | Watchdog 8s, grace 15s, debounce 60s, kill+respawn, stale-shm recovery                      |
| `backend/app/agent/tools/service_tools.py:162`             | `subprocess.Popen(start_new_session=True)` — chỗ spawn camera (liên quan `KillMode`) |
| `backend/app/services/shared_memory_service.py`            | Đọc ring buffer + seqlock chống torn read +`check_staleness`                           |
| `frontend-ts/src/components/shared/ServiceDownOverlay.tsx` | Overlay Service Down, self-heal 5s, tự reload trang khi hồi phục                         |
| `start_services.sh:21`                                     | `rm -rf` thư mục log — nguồn cơn mất bằng chứng                                   |
| `start_services.sh:187`                                    | Nơi AI camera service thật được khởi động                                           |
| `scripts/setup_systemd_services.sh`                        | Script sinh unit — nguồn sự thật duy nhất cho systemd                                  |
