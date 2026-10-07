"""
Quick test: đọc bit X0 (cảm biến) qua Modbus TCP trên PLC Delta.

Địa chỉ theo chuẩn Delta PLC:
  X0 -> coil address 1024 (0x0400), read-only (input vật lý)
  M10 -> coil address 2058 (0x0800 + 10), read/write (internal relay)

Chạy: python tests/test_plc_x0_read.py
"""
import time
from pymodbus.client import ModbusTcpClient

PLC_IP = "192.168.1.5"
PLC_PORT = 502

X0_ADDRESS = 1024   # cảm biến vào chân X0
M10_ADDRESS = 2058  # bit báo NG cho PLC (Y1 xy-lanh PLC tự xử lý)

POLL_INTERVAL_S = 0.02  # 20ms


def read_x0(client: ModbusTcpClient) -> bool | None:
    # X là input vật lý (read-only) -> function code 02 (read_discrete_inputs).
    # FC01 (read_coils) trả ILLEGAL DATA ADDRESS trên PLC này cho vùng X.
    result = client.read_discrete_inputs(X0_ADDRESS, count=1)
    if result.isError():
        return None
    return bool(result.bits[0])


def write_m10(client: ModbusTcpClient, value: bool) -> bool:
    result = client.write_coil(M10_ADDRESS, value)
    if result.isError():
        print(f"✗ Lỗi ghi M10 @ {M10_ADDRESS}: {result}")
        return False
    return True


def main():
    client = ModbusTcpClient(host=PLC_IP, port=PLC_PORT, timeout=3)

    if not client.connect():
        print(f"✗ Không kết nối được PLC {PLC_IP}:{PLC_PORT}")
        return

    print(f"✓ Đã kết nối PLC {PLC_IP}:{PLC_PORT}")
    print(f"  Đang poll X0 (coil {X0_ADDRESS}) mỗi {POLL_INTERVAL_S*1000:.0f}ms. Ctrl+C để dừng.\n")

    last_value = None
    error_streak = 0
    try:
        while True:
            value = read_x0(client)

            if value is None:
                error_streak += 1
                if error_streak == 1 or error_streak % 100 == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] ✗ Lỗi đọc X0 @ {X0_ADDRESS} "
                          f"(liên tiếp {error_streak} lần)")
            else:
                error_streak = 0
                if last_value is None:
                    print(f"[{time.strftime('%H:%M:%S')}] X0 trạng thái ban đầu = {int(value)}")
                elif value != last_value:
                    if value:
                        print(f"[{time.strftime('%H:%M:%S')}] X0: 0 → 1  → (sẽ) chụp ảnh + chạy AI")
                        # TODO: gọi camera.execute_software_trigger_immediate() + inference thật ở đây
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] X0: 1 → 0")
                last_value = value

            time.sleep(POLL_INTERVAL_S)

    except KeyboardInterrupt:
        print("\n⏹ Dừng test")
    finally:
        client.close()


if __name__ == "__main__":
    main()
