import time
from pymodbus.client import ModbusTcpClient

PLC_IP = "192.168.1.5"
PLC_PORT = 502
D0_ADDRESS = 0

client = ModbusTcpClient(host=PLC_IP, port=PLC_PORT, timeout=3)

if client.connect():
    print("Connected to PLC\n")

    write_times = []

    for i in range(100):
        # D0 = 1
        t0 = time.perf_counter()
        result = client.write_register(D0_ADDRESS, 1)
        dt = (time.perf_counter() - t0) * 1000  # ms
        write_times.append(dt)
        print(f"[{i+1}/100] D0=1: {'OK' if not result.isError() else result}  | {dt:.2f} ms")
        time.sleep(0.5)

        # D0 = 0
        t0 = time.perf_counter()
        result = client.write_register(D0_ADDRESS, 0)
        dt = (time.perf_counter() - t0) * 1000
        write_times.append(dt)
        print(f"[{i+1}/100] D0=0: {'OK' if not result.isError() else result}  | {dt:.2f} ms")
        time.sleep(0.5)

    client.close()

    # Thống kê
    print("\n===== WRITE LATENCY STATS =====")
    print(f"  Samples : {len(write_times)}")
    print(f"  Min     : {min(write_times):.2f} ms")
    print(f"  Max     : {max(write_times):.2f} ms")
    print(f"  Avg     : {sum(write_times)/len(write_times):.2f} ms")
    print(f"  Total   : {sum(write_times):.2f} ms")

else:
    print("Cannot connect to PLC")
