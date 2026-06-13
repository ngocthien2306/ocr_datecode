#!/usr/bin/env python3
# camera_check_eth1.py
import subprocess
import time
import sys
import logging
import os
from datetime import datetime
import pypylon.pylon as py

INTERFACE = "eth1"
# Log to centralized {repo}/logs/camera_check/{YYYY-MM-DD}.log
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "camera_check")
SUDO_PASSWORD = "1"

MAX_ATTEMPTS = 10
GRAB_TIMEOUT_MS = 5000


def sudo_run(cmd):
    """Run a sudo command with auto password via stdin."""
    subprocess.run(
        ["sudo", "-S"] + cmd,
        input=SUDO_PASSWORD + "\n",
        text=True,
        stderr=subprocess.DEVNULL,  # suppress password prompt on stderr
    )


def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def reset_interface(iface):
    logging.info(f"Resetting {iface}...")
    sudo_run(["ip", "link", "set", iface, "down"])
    time.sleep(2)
    sudo_run(["ip", "link", "set", iface, "up"])
    logging.info(f"Waiting 20s for {iface} to come up...")
    time.sleep(20)
    # Flush stale ARP entries so camera IP resolves fresh
    sudo_run(["ip", "neigh", "flush", "dev", iface])
    time.sleep(2)


def find_gige_cameras(iface=None):
    tl_factory = py.TlFactory.GetInstance()
    di = py.DeviceInfo()
    di.SetDeviceClass("BaslerGigE")
    devices = tl_factory.EnumerateDevices([di])
    if iface:
        # Filter by subnet/interface — keeps only cameras reachable via this iface
        iface_ip = get_interface_ip(iface)
        if iface_ip:
            prefix = ".".join(iface_ip.split(".")[:3])  # e.g. "192.168.100"
            devices = [d for d in devices if d.GetIpAddress().startswith(prefix)]
    return devices


def get_interface_ip(iface):
    try:
        result = subprocess.run(
            ["ip", "-4", "addr", "show", iface],
            capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("inet "):
                return line.split()[1].split("/")[0]
    except Exception:
        pass
    return None


def try_open_and_grab(device_info):
    camera = None
    grab_result = None
    try:
        camera = py.InstantCamera(py.TlFactory.GetInstance().CreateDevice(device_info))
        camera.Open()

        # Reduce heartbeat timeout so camera releases stale sessions faster
        try:
            if camera.HeartbeatTimeout.IsWritable():
                camera.HeartbeatTimeout.Value = 1000
        except Exception:
            pass  # Not all cameras expose this param

        camera.StartGrabbing(py.GrabStrategy_LatestImageOnly)
        grab_result = camera.RetrieveResult(GRAB_TIMEOUT_MS, py.TimeoutHandling_ThrowException)

        if grab_result.GrabSucceeded():
            logging.info(f"[OK] Grab successful! Size: {grab_result.Width}x{grab_result.Height}")
            return True
        else:
            logging.warning(f"[FAIL] Grab failed: {grab_result.ErrorDescription}")
            return False

    except Exception as e:
        logging.error(f"[ERROR] Failed to open camera: {e}")
        return False

    finally:
        # Always cleanup regardless of success/failure/exception
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


def main():
    setup_logger()
    logging.info("=" * 50)
    logging.info("Camera check started")
    logging.info(f"Interface: {INTERFACE}")
    logging.info(f"Max attempts: {MAX_ATTEMPTS}")
    # logging.info(f"Log file: {LOG_FILE}")
    logging.info("=" * 50)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        logging.info(f"[Attempt {attempt}/{MAX_ATTEMPTS}] Scanning GigE cameras on {INTERFACE}...")
        devices = find_gige_cameras(iface=INTERFACE)
        logging.info(f"Found {len(devices)} camera(s)")

        if len(devices) == 0:
            logging.warning(f"No cameras found on {INTERFACE}. Resetting interface...")
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

        logging.warning(f"Grab failed. Resetting {INTERFACE} and retrying...")
        reset_interface(INTERFACE)

    logging.error(f"[FATAL] Camera still not working after {MAX_ATTEMPTS} attempts. Tiến hành reboot...")
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