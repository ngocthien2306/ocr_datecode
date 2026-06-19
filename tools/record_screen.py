"""Ghi màn hình trên Jetson bằng GStreamer + NVENC (hw-accelerated).

Cách dùng:
    python tools/record_screen.py                  # ghi đến khi Ctrl+C
    python tools/record_screen.py --duration 30    # ghi 30 giây
    python tools/record_screen.py --fps 25 --bitrate 6000000
    python tools/record_screen.py --output /tmp/myrec.mp4
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def build_pipeline(output_file: Path, fps: int, bitrate: int) -> list[str]:
    pipeline = (
        f"ximagesrc use-damage=0 ! "
        f"video/x-raw,framerate={fps}/1 ! "
        f"videoconvert ! "
        f"nvvidconv ! "
        f"video/x-raw(memory:NVMM),format=NV12 ! "
        f"nvv4l2h264enc bitrate={bitrate} preset-level=1 insert-sps-pps=1 ! "
        f"h264parse ! "
        f"mp4mux ! "
        f"filesink location={output_file}"
    )
    return ["gst-launch-1.0", "-e"] + pipeline.split()


def main() -> int:
    parser = argparse.ArgumentParser(description="Jetson screen recorder")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Đường dẫn file MP4 đầu ra (mặc định: logs/recordings/screen_<timestamp>.mp4)")
    parser.add_argument("--duration", "-d", type=int, default=None,
                        help="Thời lượng ghi (giây). Bỏ trống = ghi đến khi Ctrl+C")
    parser.add_argument("--fps", type=int, default=30, help="Framerate (mặc định 30)")
    parser.add_argument("--bitrate", type=int, default=8_000_000,
                        help="Bitrate H.264 bps (mặc định 8 Mbps)")
    parser.add_argument("--display", type=str, default=None,
                        help="Giá trị DISPLAY (mặc định lấy từ env, fallback ':0')")
    args = parser.parse_args()

    if shutil.which("gst-launch-1.0") is None:
        print("Lỗi: không tìm thấy gst-launch-1.0. Cài: sudo apt install gstreamer1.0-tools",
              file=sys.stderr)
        return 1

    env = os.environ.copy()
    env["DISPLAY"] = args.display or env.get("DISPLAY", ":0")

    if args.output:
        out_file = Path(args.output).expanduser().resolve()
    else:
        out_dir = Path("logs/recordings")
        out_file = out_dir / f"screen_{datetime.now():%Y%m%d_%H%M%S}.mp4"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_pipeline(out_file, args.fps, args.bitrate)
    print(f"DISPLAY = {env['DISPLAY']}")
    print(f"Output  = {out_file}")
    print(f"FPS={args.fps}  Bitrate={args.bitrate/1e6:.1f} Mbps")
    print("Bắt đầu ghi... (Ctrl+C để dừng)\n")

    proc = subprocess.Popen(cmd, env=env)

    try:
        if args.duration:
            proc.wait(timeout=args.duration)
        else:
            proc.wait()
    except KeyboardInterrupt:
        print("\nNhận Ctrl+C, đang đóng file MP4...")
    except subprocess.TimeoutExpired:
        print(f"\nĐã đủ {args.duration}s, đang đóng file MP4...")
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    if out_file.exists() and out_file.stat().st_size > 0:
        size_mb = out_file.stat().st_size / (1024 * 1024)
        print(f"Đã lưu: {out_file}  ({size_mb:.1f} MB)")
        return 0
    print("Cảnh báo: file rỗng hoặc không tồn tại.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
