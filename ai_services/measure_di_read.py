#!/usr/bin/env python3
"""
Đo chi phí đọc Digital Input (DI) qua libapmi để xác minh giả thuyết
"mỗi lần đọc GPIO ~21ms khiến vòng polling 100Hz thực chất chỉ ~45Hz".

Đo 3 thứ:
  1. RAW native call  : apmi_dio_read_input() thuần, không lock  -> chi phí phần cứng thật
  2. WRAPPED read     : read_di_value() của project (có _gpio_lock) -> chi phí thực tế trong service
  3. LOOP thực tế     : mô phỏng đúng _polling_loop (read + sleep(max(0,10ms-elapsed))) -> Hz đạt được

CẢNH BÁO AN TOÀN:
  - Script CHỈ ĐỌC DI, không ghi DO nên KHÔNG kích relay/reject.
  - Nhưng libapmi KHÔNG thread-safe và _gpio_lock chỉ có tác dụng trong 1 process.
    Nếu service OCR đang chạy và cũng đang poll DI, chạy song song có thể gây
    vài lần đọc lỗi (ret!=0) ở CẢ hai bên. Nên chạy trong cửa sổ dừng line,
    hoặc dừng service (sudo systemctl stop ocr-all) trước khi đo cho số liệu sạch.

Cách chạy (từ thư mục ai_services):
    python3 measure_di_read.py                # đo DI0, 2000 lần
    python3 measure_di_read.py --pin 0 --iters 3000
    python3 measure_di_read.py --wrapped      # đo thêm read_di_value() (import project)
    python3 measure_di_read.py --subprocess   # đo thêm fallback 'sudo dio_in'
"""
import argparse
import ctypes
import statistics
import subprocess
import time

# libapmi dùng mã hoá pin theo luỹ thừa 2: DI0->1, DI1->2, DI2->4, DI3->8
_LIBAPMI_PIN_MAP = {0: 1, 1: 2, 2: 4, 3: 8}


def load_libapmi():
    """Nạp libapmi và cấu hình prototype giống utils._init_libapmi()."""
    lib = ctypes.CDLL("/lib/libapmi.so")
    fn = lib.apmi_dio_read_input
    fn.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    fn.restype = ctypes.c_int
    return fn


def summarize(name, samples_ms, errors):
    """In thống kê cho một tập mẫu latency (ms)."""
    s = sorted(samples_ms)
    n = len(s)
    if n == 0:
        print(f"[{name}] không có mẫu")
        return
    def pct(p):
        return s[min(n - 1, int(n * p))]
    mean = statistics.fmean(s)
    stdev = statistics.pstdev(s) if n > 1 else 0.0
    print(f"\n===== {name} (n={n}, errors={errors}) =====")
    print(f"  min   = {s[0]:8.3f} ms")
    print(f"  p50   = {pct(0.50):8.3f} ms")
    print(f"  p90   = {pct(0.90):8.3f} ms")
    print(f"  p99   = {pct(0.99):8.3f} ms")
    print(f"  max   = {s[-1]:8.3f} ms")
    print(f"  mean  = {mean:8.3f} ms   stdev={stdev:.3f} ms")
    print(f"  => tần số tối đa nếu lặp liên tục: {1000.0/mean:6.1f} Hz")
    # Histogram thô để thấy phân bố (bin 2ms tới 40ms, rồi gộp)
    bins = {}
    for x in s:
        b = min(40, int(x // 2) * 2)
        bins[b] = bins.get(b, 0) + 1
    print("  histogram (bin 2ms, '40+' gộp phần đuôi):")
    for b in sorted(bins):
        label = f"{b:2d}-{b+2:<2d}" if b < 40 else " 40+  "
        bar = "#" * min(60, bins[b] * 60 // n)
        print(f"    {label} | {bar} ({bins[b]})")


def measure_raw(read_fn, lib_pin, iters, warmup):
    """Đo apmi_dio_read_input() thuần."""
    val = ctypes.c_int()
    # warmup (bỏ vài lần đầu vì có thể lazy-init driver)
    first = None
    for i in range(warmup):
        t0 = time.perf_counter()
        read_fn(lib_pin, ctypes.byref(val))
        if i == 0:
            first = (time.perf_counter() - t0) * 1000.0
    samples, errors = [], 0
    for _ in range(iters):
        t0 = time.perf_counter()
        ret = read_fn(lib_pin, ctypes.byref(val))
        dt = (time.perf_counter() - t0) * 1000.0
        if ret != 0:
            errors += 1
        samples.append(dt)
    if first is not None:
        print(f"[RAW] chi phí lần gọi ĐẦU TIÊN (gồm init): {first:.3f} ms")
    return samples, errors


def measure_loop(read_fn, lib_pin, iters, poll_interval_ms=10.0):
    """Mô phỏng đúng _polling_loop: read + sleep(max(0, poll_interval - elapsed))."""
    val = ctypes.c_int()
    period_target = poll_interval_ms / 1000.0
    periods_ms = []
    prev = time.perf_counter()
    for _ in range(iters):
        loop_start = time.perf_counter()
        read_fn(lib_pin, ctypes.byref(val))
        elapsed = time.perf_counter() - loop_start
        sleep_t = max(0.0, period_target - elapsed)
        time.sleep(sleep_t)
        now = time.perf_counter()
        periods_ms.append((now - prev) * 1000.0)
        prev = now
    return periods_ms


def measure_subprocess(pin, iters):
    """Đo fallback 'sudo dio_in <pin>' (đường chậm dự phòng)."""
    samples, errors = [], 0
    for _ in range(iters):
        t0 = time.perf_counter()
        r = subprocess.run(["sudo", "dio_in", str(pin)],
                           capture_output=True, text=True, timeout=1.0)
        dt = (time.perf_counter() - t0) * 1000.0
        if r.returncode != 0:
            errors += 1
        samples.append(dt)
    return samples, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", type=int, default=0, help="DI pin (0-3), mặc định 0")
    ap.add_argument("--iters", type=int, default=2000, help="số lần đo")
    ap.add_argument("--warmup", type=int, default=10, help="số lần bỏ đầu")
    ap.add_argument("--wrapped", action="store_true", help="đo thêm read_di_value() của project")
    ap.add_argument("--subprocess", action="store_true", help="đo thêm fallback sudo dio_in (chậm)")
    args = ap.parse_args()

    lib_pin = _LIBAPMI_PIN_MAP.get(args.pin)
    if lib_pin is None:
        raise SystemExit(f"pin không hợp lệ: {args.pin}")

    print(f"Đo DI{args.pin} (lib_pin={lib_pin}), iters={args.iters}, warmup={args.warmup}")
    print("LƯU Ý: nếu service OCR đang chạy, số liệu có thể nhiễu do tranh chấp libapmi.\n")

    read_fn = load_libapmi()

    # 1) RAW native
    raw, raw_err = measure_raw(read_fn, lib_pin, args.iters, args.warmup)
    summarize("RAW native apmi_dio_read_input()", raw, raw_err)

    # 3) LOOP thực tế (dùng cùng read_fn, mô phỏng poll 100Hz)
    periods = measure_loop(read_fn, lib_pin, min(args.iters, 1000))
    summarize("LOOP period thực tế (mục tiêu 10ms/100Hz)", periods, 0)
    print(f"  => TẦN SỐ POLLING THỰC = {1000.0/statistics.fmean(periods):.1f} Hz "
          f"(mục tiêu 100 Hz)")

    # 2) WRAPPED read_di_value() (tuỳ chọn, cần import project)
    if args.wrapped:
        import os, sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from camera_management.utils import read_di_value
        wr, wr_err = [], 0
        for _ in range(args.iters):
            t0 = time.perf_counter()
            v = read_di_value(args.pin)
            wr.append((time.perf_counter() - t0) * 1000.0)
        summarize("WRAPPED read_di_value() (có _gpio_lock)", wr, wr_err)

    # subprocess fallback (tuỳ chọn)
    if args.subprocess:
        sp, sp_err = measure_subprocess(args.pin, min(args.iters, 200))
        summarize("SUBPROCESS 'sudo dio_in'", sp, sp_err)


if __name__ == "__main__":
    main()
