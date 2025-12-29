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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

        try:
            camera.ExposureTime.SetValue(10000)
        except:
            pass

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
                    logger.info(f"Frame #{frame_idx} written to shared memory")

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
