#!/usr/bin/env python3

from multiprocessing import shared_memory, Queue
import cv2
from pypylon import pylon
import numpy as np
import time
import pickle
import logging

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

        max_frame_size = 2448 * 2048 * 3
        shm_name = f"camera_{serial_number}"

        try:
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=max_frame_size + 1024)
            logger.info(f"Created shared memory: {shm_name}")
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

                offset = 0
                shm.buf[offset:offset+4] = metadata_len.to_bytes(4, 'little')
                offset += 4
                shm.buf[offset:offset+metadata_len] = metadata_bytes
                offset += metadata_len
                shm.buf[offset:offset+4] = frame_len.to_bytes(4, 'little')
                offset += 4
                shm.buf[offset:offset+frame_len] = frame_bytes

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
