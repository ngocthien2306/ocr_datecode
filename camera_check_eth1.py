#!/usr/bin/env python3
# camera_check_eth1.py
import subprocess
import time
import sys
import logging
import os
import pypylon.pylon as py

INTERFACE = "eth1"
USER_HOME = "/home/suntech"
LOG_DIR = os.path.join(USER_HOME, "Source/ocr_datecode/logs")
LOG_FILE = os.path.join(LOG_DIR, "camera_check.log")


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


def reset_interface(iface):
    logging.info(f"Resetting {iface}...")
    subprocess.run(["sudo", "ip", "link", "set", iface, "down"])
    time.sleep(10)
    subprocess.run(["sudo", "ip", "link", "set", iface, "up"])
    logging.info(f"Waiting 20s for {iface} to come up...")
    time.sleep(20)


def find_gige_cameras():
    tl_factory = py.TlFactory.GetInstance()
    di = py.DeviceInfo()
    di.SetDeviceClass("BaslerGigE")
    devices = tl_factory.EnumerateDevices([di])
    return devices


def try_open_and_grab(device_info):
    try:
        camera = py.InstantCamera(py.TlFactory.GetInstance().CreateDevice(device_info))
        camera.Open()
        camera.StartGrabbing(py.GrabStrategy_LatestImageOnly)
        grab_result = camera.RetrieveResult(5000, py.TimeoutHandling_ThrowException)
        if grab_result.GrabSucceeded():
            logging.info(f"[OK] Grab successful! Size: {grab_result.Width}x{grab_result.Height}")
            grab_result.Release()
            camera.StopGrabbing()
            camera.Close()
            return True
        else:
            logging.warning(f"[FAIL] Grab failed: {grab_result.ErrorDescription}")
            grab_result.Release()
            camera.StopGrabbing()
            camera.Close()
            return False
    except Exception as e:
        logging.error(f"[ERROR] Failed to open camera: {e}")
        return False


def main():
    setup_logger()
    logging.info("=" * 50)
    logging.info("Camera check started")
    logging.info(f"Interface: {INTERFACE}")
    logging.info(f"Log file: {LOG_FILE}")
    logging.info("=" * 50)

    attempt = 0
    while True:
        attempt += 1
        logging.info(f"[Attempt {attempt}] Scanning GigE cameras on {INTERFACE}...")
        devices = find_gige_cameras()
        logging.info(f"Found {len(devices)} camera(s)")

        if len(devices) == 0:
            logging.warning(f"No cameras found. Resetting {INTERFACE}...")
            reset_interface(INTERFACE)
            continue

        success = False
        for i, dev in enumerate(devices):
            logging.info(f"Trying camera [{i}]: {dev.GetModelName()} - {dev.GetIpAddress()}")
            if try_open_and_grab(dev):
                success = True
                break

        if success:
            logging.info("[DONE] Camera is working. Proceeding to start services...")
            sys.exit(0)
        else:
            logging.warning(f"Grab failed. Resetting {INTERFACE} and retrying...")
            reset_interface(INTERFACE)


if __name__ == "__main__":
    main()
