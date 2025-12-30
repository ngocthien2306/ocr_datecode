#!/usr/bin/env python3

from multiprocessing import shared_memory, Queue
import cv2
from pypylon import pylon
import numpy as np
import time
import pickle
import logging
import struct
import ctypes
import json
import os
from pathlib import Path

SETTINGS_DIR = Path("camera_settings")
LOGS_DIR = Path("logs")

# Create logs directory
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / 'camera_producer.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


def load_camera_settings(serial_number: str) -> dict:
    """Load camera settings from JSON file"""
    settings_file = SETTINGS_DIR / f"{serial_number}.json"

    if settings_file.exists():
        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)
                logger.info(f"✅ [SETTINGS LOADED] Camera {serial_number}: {settings}")
                return settings
        except Exception as e:
            logger.error(f"❌ [SETTINGS ERROR] Failed to load settings for {serial_number}: {e}")

    # Return default settings if file doesn't exist
    default_settings = {
        "exposure_time": 10000,
        "gain": 1.0
    }
    logger.info(f"⚠️  [DEFAULT SETTINGS] Camera {serial_number}: Using default settings {default_settings}")
    return default_settings


def apply_camera_settings(camera, settings: dict, serial_number: str = "Unknown"):
    """Apply settings to camera"""
    applied = []

    try:
        if "exposure_time" in settings:
            exposure_time = int(settings["exposure_time"])
            camera.ExposureTime.SetValue(exposure_time)
            actual_exposure = camera.ExposureTime.GetValue()
            logger.info(f"✅ [EXPOSURE APPLIED] Camera {serial_number}: {exposure_time} μs (actual: {actual_exposure} μs)")
            applied.append(f"exposure={actual_exposure}μs")
    except Exception as e:
        logger.error(f"❌ [EXPOSURE ERROR] Camera {serial_number}: {e}")

    try:
        if "gain" in settings:
            gain = float(settings["gain"])
            camera.Gain.SetValue(gain)
            actual_gain = camera.Gain.GetValue()
            logger.info(f"✅ [GAIN APPLIED] Camera {serial_number}: {gain} (actual: {actual_gain})")
            applied.append(f"gain={actual_gain}")
    except Exception as e:
        logger.error(f"❌ [GAIN ERROR] Camera {serial_number}: {e}")

    if applied:
        logger.info(f"🎥 [SETTINGS SUMMARY] Camera {serial_number}: {', '.join(applied)}")


def camera_producer(device_index=0):
    try:
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()

        if not devices or device_index >= len(devices):
            logger.error(f"Camera {device_index} not available")
            return

        device = devices[device_index]
        serial_number = device.GetSerialNumber()
        model_name = device.GetModelName()

        camera = pylon.InstantCamera(tlFactory.CreateDevice(device))
        camera.Open()

        logger.info(f"Opened: {model_name} (SN: {serial_number})")

        # Load and apply initial settings
        logger.info(f"🚀 [CAMERA INIT] Starting camera producer for {model_name} (SN: {serial_number})")
        settings = load_camera_settings(serial_number)
        apply_camera_settings(camera, settings, serial_number)
        last_settings_check = time.time()
        settings_file = SETTINGS_DIR / f"{serial_number}.json"
        last_mtime = settings_file.stat().st_mtime if settings_file.exists() else 0
        logger.info(f"📁 [SETTINGS FILE] Watching: {settings_file}")

        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        actual_width = camera.Width.GetValue()
        actual_height = camera.Height.GetValue()
        max_frame_size = actual_width * actual_height * 3  # BGR = 3 channels
        logger.info(f"Camera resolution: {actual_width}x{actual_height}, max frame size: {max_frame_size} bytes")

        shm_name = f"camera_{serial_number}"

        try:
            shm_size = max_frame_size + 4096
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=shm_size)
            logger.info(f"Created shared memory: {shm_name} (size: {shm_size} bytes)")
        except:
            shm = shared_memory.SharedMemory(name=shm_name)
            logger.info(f"Using existing shared memory: {shm_name}")

        frame_idx = 0

        logger.info(f"Streaming to shared memory '{shm_name}'...")

        while camera.IsGrabbing():
            # Check for settings updates every 2 seconds
            current_time = time.time()
            if current_time - last_settings_check > 2.0:
                last_settings_check = current_time
                if settings_file.exists():
                    try:
                        current_mtime = settings_file.stat().st_mtime
                        if current_mtime > last_mtime:
                            # Settings file changed, reload
                            logger.info(f"🔄 [HOT RELOAD] Detected settings file change for camera {serial_number}")
                            new_settings = load_camera_settings(serial_number)
                            apply_camera_settings(camera, new_settings, serial_number)
                            last_mtime = current_mtime
                            logger.info(f"✅ [HOT RELOAD SUCCESS] Camera {serial_number} settings updated!")
                    except Exception as e:
                        logger.error(f"❌ [HOT RELOAD ERROR] Camera {serial_number}: {e}")

            grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)

            if grabResult and grabResult.GrabSucceeded():
                image = converter.Convert(grabResult)
                img_array = image.GetArray()

                frame_idx += 1

                metadata = {
                    'serial_number': serial_number,
                    'model_name': model_name,
                    'timestamp': time.time(),
                    'frame_idx': frame_idx,
                    'shape': img_array.shape,
                    'dtype': str(img_array.dtype)
                }

                metadata_bytes = pickle.dumps(metadata)
                metadata_len = len(metadata_bytes)

                frame_bytes = img_array.tobytes()
                frame_len = len(frame_bytes)

                # Tạo numpy array view của shared memory buffer để dễ ghi dữ liệu
                shm_array = np.ndarray((len(shm.buf),), dtype=np.uint8, buffer=shm.buf)

                offset = 0

                # Ghi metadata length (4 bytes)
                struct.pack_into('<I', shm.buf, offset, metadata_len)
                offset += 4

                # Ghi metadata bytes - dùng numpy array view
                shm_array[offset:offset+metadata_len] = np.frombuffer(metadata_bytes, dtype=np.uint8)
                offset += metadata_len

                # Ghi frame length (4 bytes)
                struct.pack_into('<I', shm.buf, offset, frame_len)
                offset += 4

                # Ghi frame bytes - dùng numpy array view
                shm_array[offset:offset+frame_len] = np.frombuffer(frame_bytes, dtype=np.uint8)

                if frame_idx % 30 == 0:
                    pass
                    # logger.info(f"Frame #{frame_idx} written to shared memory")

                grabResult.Release()
                time.sleep(0.033)

            else:
                if grabResult:
                    grabResult.Release()
                time.sleep(0.01)

        camera.StopGrabbing()
        camera.Close()
        shm.close()
        shm.unlink()

    except KeyboardInterrupt:
        logger.info("Stopped by user")
        try:
            shm.close()
            shm.unlink()
        except:
            pass
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    camera_producer(device_index=0)
