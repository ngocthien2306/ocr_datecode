import time
import ctypes
import subprocess

lib = ctypes.CDLL('/lib/libapmi.so')

apmi_read_do = lib.apmi_dio_read_output
apmi_read_do.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
apmi_read_do.restype = ctypes.c_int

apmi_write_do = lib.apmi_dio_write_output
apmi_write_do.argtypes = [ctypes.c_int, ctypes.c_int]
apmi_write_do.restype = ctypes.c_int

apmi_read_di = lib.apmi_dio_read_input
apmi_read_di.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
apmi_read_di.restype = ctypes.c_int

PIN_MAP = {0: 1, 1: 2, 2: 4, 3: 8}

print("=== SO SANH DIO vs APMI ===\n")

print("DO Write:")
print(f"{'Pin':<6}{'DIO(ms)':<12}{'APMI(ms)':<12}{'Speedup':<10}")
print("-" * 40)
for pin in [0, 1, 2, 3]:
    t0 = time.perf_counter()
    subprocess.run(['sudo', 'dio_out', str(pin), '1'], capture_output=True, timeout=1)
    t1 = time.perf_counter()
    dio_ms = (t1 - t0) * 1000

    t0 = time.perf_counter()
    apmi_write_do(PIN_MAP[pin], 1)
    t1 = time.perf_counter()
    apmi_ms = (t1 - t0) * 1000

    speedup = dio_ms / apmi_ms if apmi_ms > 0 else 0
    print(f"DO{pin:<4}{dio_ms:<12.1f}{apmi_ms:<12.1f}{speedup:.1f}x")

print("\nDI Read:")
print(f"{'Pin':<6}{'DIO(ms)':<12}{'APMI(ms)':<12}{'Speedup':<10}")
print("-" * 40)
for pin in [0, 1, 2, 3]:
    t0 = time.perf_counter()
    subprocess.run(['sudo', 'dio_in', str(pin)], capture_output=True, timeout=1)
    t1 = time.perf_counter()
    dio_ms = (t1 - t0) * 1000

    val = ctypes.c_int()
    t0 = time.perf_counter()
    apmi_read_di(PIN_MAP[pin], ctypes.byref(val))
    t1 = time.perf_counter()
    apmi_ms = (t1 - t0) * 1000

    speedup = dio_ms / apmi_ms if apmi_ms > 0 else 0
    print(f"DI{pin:<4}{dio_ms:<12.1f}{apmi_ms:<12.1f}{speedup:.1f}x")
