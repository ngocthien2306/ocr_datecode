#!/usr/bin/env python3
"""
Giả lập tình huống (B): "camera connected nhưng không có frame trong shared memory"
để test cơ chế tự kill + restart (handle_missing_frame -> restart_backend_and_camera).

Cách hoạt động: ghi 0 vào ô `frame_count` (offset 4, <I) trong header ring buffer
của segment `camera_<serial>`. read_frame()/read_latest_frames() bên BE đọc lại
frame_count mỗi lần gọi và trả None ngay khi == 0  ->  endpoint get-frame rơi vào
nhánh None  ->  handle_missing_frame thấy is_connected=True + camera service connected
-> restart cả backend + camera.

LƯU Ý:
  - Chạy trên ĐÚNG máy đang chạy camera service + backend (nơi /dev/shm sống).
  - Camera nên ở mode SOFTWARE_TRIGGER/IDLE và KHÔNG có thùng chạy qua, nếu không
    writer (CONTINUOUS hoặc trigger) sẽ ghi frame_count > 0 lại ngay.
  - Thao tác này sẽ THẬT SỰ restart backend + camera production. Chỉ chạy khi an toàn.

Dùng:
  python3 sim_no_frame.py <serial_number>            # xem frame_count hiện tại + set về 0
  python3 sim_no_frame.py <serial_number> --check     # chỉ xem, không sửa
"""
import struct
import sys
from multiprocessing import shared_memory, resource_tracker

FRAME_COUNT_OFFSET = 4  # header: write_idx(<I)@0, frame_count(<I)@4


def _detach_from_resource_tracker(shm):
    """
    QUAN TRỌNG: mặc định CPython đăng ký MỌI SharedMemory (kể cả create=False) với
    resource_tracker và sẽ tự gọi unlink() lúc process này thoát -> XOÁ segment của
    process khác (BE/camera service). Gỡ đăng ký để script chỉ ĐỌC/GHI chứ không
    huỷ segment.  (CPython bug: https://github.com/python/cpython/issues/82300)
    """
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    serial = sys.argv[1]
    check_only = "--check" in sys.argv[2:]
    shm_name = f"camera_{serial}"

    try:
        shm = shared_memory.SharedMemory(name=shm_name, create=False)
        _detach_from_resource_tracker(shm)  # KHÔNG để script này unlink segment khi thoát
    except FileNotFoundError:
        print(f"[!] Không tìm thấy segment '{shm_name}' trong /dev/shm.")
        print("    => Camera có thể chưa connect, hoặc segment đã bị unlink.")
        print("    (Bản thân việc này cũng là một dạng tình huống (B) nếu camera đang connected.)")
        sys.exit(2)

    try:
        write_idx = struct.unpack_from("<I", shm.buf, 0)[0]
        frame_count = struct.unpack_from("<I", shm.buf, FRAME_COUNT_OFFSET)[0]
        print(f"[i] {shm_name}: write_idx={write_idx}, frame_count={frame_count}")

        if check_only:
            return

        if frame_count == 0:
            print("[i] frame_count đã = 0 sẵn (camera chưa grab frame nào) — sẵn sàng test.")
        else:
            struct.pack_into("<I", shm.buf, FRAME_COUNT_OFFSET, 0)
            after = struct.unpack_from("<I", shm.buf, FRAME_COUNT_OFFSET)[0]
            print(f"[✓] Đã set frame_count: {frame_count} -> {after}")
        print("\nGiờ gọi API get-frame (qua UI 'Get frames' hoặc curl) để kích hoạt restart.")
        print("Nếu writer ghi đè lại frame_count>0 trước khi kịp gọi API, hãy đảm bảo")
        print("camera không có thùng chạy qua / không ở CONTINUOUS mode rồi chạy lại.")
    finally:
        shm.close()


if __name__ == "__main__":
    main()
