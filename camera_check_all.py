#!/usr/bin/env python3
# camera_check_all.py
# Quét xem có đủ 2 camera không, nếu thiếu thì reset interface tương ứng và retry.

import subprocess
import time
import sys
import logging
import os
import pypylon.pylon as py

SUDO_PASSWORD = "1"
MAX_ATTEMPTS = 10
WAIT_AFTER_RESET = 20  # seconds
GRAB_TIMEOUT_MS = 5000

# Interface → subnet prefix mapping
IFACE_SUBNETS = {
    "eth1": "192.168.120",
    "eth2": "192.168.100",
}
EXPECTED_COUNT = 2

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_FILE = os.path.join(LOG_DIR, "camera_check_all.log")


def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ],
    )


def sudo_run(cmd):
    subprocess.run(
        ["sudo", "-S"] + cmd,
        input=SUDO_PASSWORD + "\n",
        text=True,
        stderr=subprocess.DEVNULL,
    )


def enumerate_gige_cameras():
    """Trả về list tất cả GigE cameras tìm thấy."""
    tl = py.TlFactory.GetInstance()
    di = py.DeviceInfo()
    di.SetDeviceClass("BaslerGigE")
    return list(tl.EnumerateDevices([di]))


def cameras_by_iface(devices):
    """Phân loại camera theo interface dựa trên subnet."""
    result = {iface: [] for iface in IFACE_SUBNETS}
    for dev in devices:
        ip = dev.GetIpAddress()
        for iface, prefix in IFACE_SUBNETS.items():
            if ip.startswith(prefix):
                result[iface].append(dev)
                break
    return result


def missing_ifaces(by_iface):
    """Trả về list interface bị thiếu camera."""
    return [iface for iface, devs in by_iface.items() if len(devs) == 0]


def try_open_and_grab(device_info):
    """Thử mở camera và grab 1 frame. Trả về True nếu thành công."""
    camera = None
    grab_result = None
    try:
        camera = py.InstantCamera(py.TlFactory.GetInstance().CreateDevice(device_info))
        camera.Open()
        try:
            if camera.HeartbeatTimeout.IsWritable():
                camera.HeartbeatTimeout.Value = 1000
        except Exception:
            pass
        camera.StartGrabbing(py.GrabStrategy_LatestImageOnly)
        grab_result = camera.RetrieveResult(GRAB_TIMEOUT_MS, py.TimeoutHandling_ThrowException)
        if grab_result.GrabSucceeded():
            logging.info(f"    [OK] Grab OK: {grab_result.Width}x{grab_result.Height}")
            return True
        else:
            logging.warning(f"    [FAIL] Grab failed: {grab_result.ErrorDescription}")
            return False
    except Exception as e:
        logging.error(f"    [ERROR] {e}")
        return False
    finally:
        try:
            if grab_result is not None:
                grab_result.Release()
        except Exception:
            pass
        try:
            if camera is not None:
                if camera.IsGrabbing():
                    camera.StopGrabbing()
                if camera.IsOpen():
                    camera.Close()
        except Exception:
            pass


def reset_interface(iface):
    logging.info(f"  Resetting {iface}...")
    sudo_run(["ip", "link", "set", iface, "down"])
    time.sleep(2)
    sudo_run(["ip", "link", "set", iface, "up"])
    logging.info(f"  Waiting {WAIT_AFTER_RESET}s for {iface} to come up...")
    time.sleep(WAIT_AFTER_RESET)
    sudo_run(["ip", "neigh", "flush", "dev", iface])
    time.sleep(2)


def main():
    setup_logger()
    logging.info("=" * 50)
    logging.info("Camera check ALL started")
    logging.info(f"Expected: {EXPECTED_COUNT} camera(s)")
    logging.info(f"Interfaces: {list(IFACE_SUBNETS.keys())}")
    logging.info("=" * 50)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        logging.info(f"[Attempt {attempt}/{MAX_ATTEMPTS}] Scanning GigE cameras...")
        devices = enumerate_gige_cameras()
        by_iface = cameras_by_iface(devices)

        # Log hiện trạng từng interface
        total = 0
        for iface, devs in by_iface.items():
            total += len(devs)
            for dev in devs:
                logging.info(f"  [{iface}] {dev.GetModelName()} - {dev.GetIpAddress()}")
            if not devs:
                logging.warning(f"  [{iface}] --- no camera ---")

        logging.info(f"Total found: {total}/{EXPECTED_COUNT}")

        if total >= EXPECTED_COUNT:
            # Thử grab frame từng camera, ghi lại interface nào bị lỗi
            failed_ifaces = set()
            for iface, devs in by_iface.items():
                for dev in devs:
                    logging.info(f"  Grab test [{iface}] {dev.GetModelName()} - {dev.GetIpAddress()}")
                    if not try_open_and_grab(dev):
                        failed_ifaces.add(iface)

            if not failed_ifaces:
                logging.info("[DONE] Đủ camera và grab OK. Proceeding...")
                sys.exit(0)

            logging.warning(f"Grab thất bại trên: {list(failed_ifaces)}. Tiến hành reset...")
            for iface in failed_ifaces:
                reset_interface(iface)
            continue

        # Reset các interface bị thiếu camera
        lost = missing_ifaces(by_iface)
        logging.warning(f"Thiếu camera trên: {lost}. Tiến hành reset...")
        for iface in lost:
            reset_interface(iface)

    logging.error(f"[FATAL] Vẫn không đủ {EXPECTED_COUNT} camera sau {MAX_ATTEMPTS} lần thử. Tiến hành reboot...")
    for i in range(5, 0, -1):
        logging.info(f"  Reboot trong {i} giây...")
        time.sleep(1)
    logging.info("Rebooting now...")
    subprocess.run(
        ["sudo", "-S", "reboot"],
        input=SUDO_PASSWORD + "\n",
        text=True,
        stderr=subprocess.DEVNULL,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
