"""
Chẩn đoán xung X0: tách bạch nguyên nhân PHẦN MỀM vs PHẦN CỨNG.

Khác với trigger_handler (poll 100Hz cố định), script này:
  1. Poll nhanh nhất có thể (không sleep) -> độ phân giải đo = độ trễ 1 lần đọc Modbus
  2. Đo & in ĐỘ TRỄ TỪNG LẦN ĐỌC Modbus  -> nếu mạng/PLC chậm sẽ thấy ngay
  3. Đo & in JITTER VÒNG LẶP             -> nếu CPU/OS làm trễ sẽ thấy ngay
  4. Vẫn tính ngầm SAI SỐ ĐO mỗi xung (không in từng dòng) để phần tổng kết
     so sánh và kết luận phần mềm hay phần cứng

Cách đọc kết quả:
  - Nếu read-latency & jitter ổn định (vài ms) mà pulse_width vẫn nhảy loạn
    (11ms, 240ms, 1000ms...) -> tín hiệu THẬT ở cảm biến loạn => PHẦN CỨNG/CƠ KHÍ.
  - Nếu read-latency/jitter có spike lớn trùng thời điểm xung loạn
    -> đường đo (Modbus/CPU) có vấn đề => PHẦN MỀM/MẠNG.

Chạy:  python tests/pulse_width_diag.py            # poll tốc độ tối đa
       python tests/pulse_width_diag.py --csv out.csv   # lưu thêm CSV
Ctrl+C để dừng, sẽ in bảng tổng kết.
"""
import argparse
import csv
import statistics
import sys
import time
from datetime import datetime

from pymodbus.client import ModbusTcpClient

# Cùng thông số với plc_controller.py / trigger_handler.py
PLC_IP = "192.168.1.5"
PLC_PORT = 502
PLC_TIMEOUT = 3.0
X0_ADDRESS = 1024          # X0 trên Delta PLC (FC02, read-only)
SUMMARY_INTERVAL_S = 30.0  # in tổng kết định kỳ


def pctl(values, p):
    """Percentile đơn giản trên list đã có dữ liệu."""
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


def fmt_stats(name, values, unit="ms"):
    if not values:
        return f"  {name}: (chưa có dữ liệu)"
    return (f"  {name}: n={len(values)} | min={min(values):.2f} | "
            f"avg={statistics.mean(values):.2f} | p95={pctl(values, 95):.2f} | "
            f"max={max(values):.2f} {unit}")


def main():
    parser = argparse.ArgumentParser(description="Chẩn đoán xung X0 qua PLC Modbus")
    parser.add_argument("--csv", help="Ghi từng xung ra file CSV")
    parser.add_argument("--sleep-ms", type=float, default=10,
                        help="Sleep giữa 2 lần poll (mặc định 0 = poll tối đa; "
                             "đặt 10 để mô phỏng đúng 100Hz như code chính)")
    args = parser.parse_args()

    csv_writer = None
    csv_file = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["wall_time", "pulse_width_ms", "read_latency_at_edge_ms"])

    client = ModbusTcpClient(host=PLC_IP, port=PLC_PORT, timeout=PLC_TIMEOUT)
    if not client.connect():
        print(f"✗ Không kết nối được PLC {PLC_IP}:{PLC_PORT}")
        sys.exit(1)
    print(f"✓ Đã kết nối PLC {PLC_IP}:{PLC_PORT} — poll X0 (addr {X0_ADDRESS})")
    if args.sleep_ms > 0:
        print(f"  Chế độ mô phỏng code chính: sleep {args.sleep_ms:.0f}ms/vòng")
    else:
        print("  Chế độ tốc độ tối đa: không sleep, độ phân giải = độ trễ 1 lần đọc")
    print("  Ctrl+C để dừng và xem tổng kết.\n")
    print(f"{'Thời gian':<12} | {'Xung (ms)':>10} | {'Đọc Modbus':>10} | Ghi chú")
    print("-" * 60)

    # Thống kê tích luỹ
    read_latencies = []      # ms, độ trễ từng lần đọc Modbus
    loop_intervals = []      # ms, khoảng cách giữa 2 lần đọc liên tiếp
    pulse_widths = []        # ms
    uncertainties = []       # ms, sai số đo của từng xung
    read_errors = 0
    total_reads = 0

    last_value = None
    last_poll_t = None       # perf_counter của lần đọc trước (để tính sai số cạnh)
    rise_t = None            # perf_counter tại cạnh lên
    rise_uncert = 0.0        # sai số cạnh lên
    last_summary = time.perf_counter()

    try:
        while True:
            t0 = time.perf_counter()
            try:
                result = client.read_discrete_inputs(X0_ADDRESS, count=1)
                value = None if result.isError() else bool(result.bits[0])
            except Exception:
                value = None
            t1 = time.perf_counter()

            total_reads += 1
            read_ms = (t1 - t0) * 1000.0
            if value is None:
                read_errors += 1
                if read_errors == 1 or read_errors % 50 == 0:
                    print(f"{datetime.now():%H:%M:%S.%f}"[:-3] +
                          f" | {'—':>10} | {read_ms:>9.1f}ms"
                          f" | ✗ LỖI ĐỌC Modbus (lần {read_errors})")
            else:
                read_latencies.append(read_ms)
                if last_poll_t is not None:
                    loop_intervals.append((t0 - last_poll_t) * 1000.0)

                if last_value is not None and value != last_value:
                    # Sai số cạnh = khoảng giữa lần đọc trước và lần đọc này:
                    # cạnh thật nằm đâu đó trong khoảng đó.
                    edge_uncert = ((t1 - last_poll_t) * 1000.0
                                   if last_poll_t is not None else read_ms)
                    if value:  # cạnh lên
                        rise_t = t0
                        rise_uncert = edge_uncert
                    elif rise_t is not None:  # cạnh xuống -> đủ 1 xung
                        width_ms = (t0 - rise_t) * 1000.0
                        uncert_ms = rise_uncert + edge_uncert
                        pulse_widths.append(width_ms)
                        uncertainties.append(uncert_ms)

                        note = ""
                        if width_ms < 60:
                            note = "⚠ XUNG QUÁ NGẮN"
                        elif width_ms > 300:
                            note = "⚠ XUNG QUÁ DÀI (kẹt/dồn chai?)"
                        wall = f"{datetime.now():%H:%M:%S.%f}"[:-3]
                        print(f"{wall:<12} | {width_ms:>10.1f} | {read_ms:>9.1f}ms | {note}")
                        if csv_writer:
                            csv_writer.writerow(
                                [wall, f"{width_ms:.3f}", f"{read_ms:.3f}"])
                last_value = value

            last_poll_t = t0

            # Tổng kết định kỳ
            if (t1 - last_summary) >= SUMMARY_INTERVAL_S:
                last_summary = t1
                print(f"\n===== TỔNG KẾT {SUMMARY_INTERVAL_S:.0f}s "
                      f"({datetime.now():%H:%M:%S}) =====")
                print(fmt_stats("Độ trễ đọc Modbus ", read_latencies[-5000:]))
                print(fmt_stats("Chu kỳ vòng lặp   ", loop_intervals[-5000:]))
                print(fmt_stats("Bề rộng xung      ", pulse_widths))
                print(f"  Lỗi đọc Modbus: {read_errors}/{total_reads}")
                if pulse_widths and uncertainties:
                    sd = statistics.pstdev(pulse_widths) if len(pulse_widths) > 1 else 0.0
                    max_u = max(uncertainties)
                    verdict = ("=> Dao động xung LỚN HƠN NHIỀU sai số đo "
                               "=> tín hiệu thật loạn => PHẦN CỨNG/CƠ KHÍ"
                               if sd > 3 * max_u else
                               "=> Dao động xung cỡ sai số đo => chưa kết luận được, "
                               "cần chạy lâu hơn")
                    print(f"  SD xung = {sd:.1f}ms, sai số đo max = ±{max_u:.1f}ms  {verdict}")
                print("=" * 50 + "\n")

            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)

    except KeyboardInterrupt:
        pass
    finally:
        client.close()
        if csv_file:
            csv_file.close()

    # Tổng kết cuối
    print(f"\n\n========== KẾT QUẢ CUỐI ({total_reads} lần đọc) ==========")
    print(fmt_stats("Độ trễ đọc Modbus ", read_latencies))
    print(fmt_stats("Chu kỳ vòng lặp   ", loop_intervals))
    print(fmt_stats("Bề rộng xung      ", pulse_widths))
    print(f"  Lỗi đọc Modbus: {read_errors}/{total_reads}")
    if len(pulse_widths) > 1:
        sd = statistics.pstdev(pulse_widths)
        max_u = max(uncertainties) if uncertainties else 0.0
        print(f"\n  SD bề rộng xung : {sd:.1f}ms")
        print(f"  Sai số đo tối đa: ±{max_u:.1f}ms  (phần 'lỗi' tối đa do phần mềm)")
        if sd > 3 * max_u:
            print("  KẾT LUẬN: dao động xung >> sai số phần mềm "
                  "=> XUNG THẬT KHÔNG ĐỀU => kiểm tra CẢM BIẾN / CƠ KHÍ / BĂNG TẢI.")
        else:
            print("  KẾT LUẬN: dao động xung nằm trong sai số đo "
                  "=> phần mềm/Modbus chi phối kết quả, xung thật có thể vẫn đều.")
    if args.csv:
        print(f"  CSV: {args.csv}")


if __name__ == "__main__":
    main()
