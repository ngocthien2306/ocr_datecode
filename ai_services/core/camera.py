"""Basler Camera Wrapper - Simple version for testing"""

import cv2
import numpy as np
import time
from pypylon import pylon
import logging

logger = logging.getLogger(__name__)


class BaslerCamera:
    def __init__(self, camera_id: str, config: dict):
        self.camera_id = camera_id
        self.config = config
        self.camera = None
        self.converter = None
        self.is_connected = False
        self.frame_count = 0

    def connect(self) -> bool:
        try:
            logger.info(f"[{self.camera_id}] Connecting...")

            tlFactory = pylon.TlFactory.GetInstance()
            devices = tlFactory.EnumerateDevices()

            if len(devices) == 0:
                logger.error(f"[{self.camera_id}] No cameras found")
                return False

            # Find by serial or use first
            target_serial = self.config.get('serial_number', '')
            device_info = None

            for device in devices:
                if device.GetSerialNumber() == target_serial:
                    device_info = device
                    break

            if device_info is None:
                device_info = devices[0]
                logger.warning(f"[{self.camera_id}] Serial {target_serial} not found, using first camera")

            self.camera = pylon.InstantCamera(tlFactory.CreateDevice(device_info))
            self.camera.Open()

            # Setup converter
            self.converter = pylon.ImageFormatConverter()
            pixel_format = self.config.get('pixel_format', 'Mono8')

            if pixel_format == 'Mono8':
                self.converter.OutputPixelFormat = pylon.PixelType_Mono8
            else:
                self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed

            logger.info(f"[{self.camera_id}] Connected: {self.camera.GetDeviceInfo().GetModelName()}")

            self._apply_config()
            self.is_connected = True
            return True

        except Exception as e:
            logger.error(f"[{self.camera_id}] Connection error: {e}")
            return False

    def _apply_config(self):
        if not self.camera or not self.camera.IsOpen():
            return

        # Exposure
        exposure = self.config.get('exposure_time', 10000)
        try:
            self.camera.ExposureTime.SetValue(exposure)
        except:
            try:
                self.camera.ExposureTimeAbs.SetValue(exposure)
            except:
                pass

        # Gain
        gain = self.config.get('gain', 0.0)
        try:
            self.camera.Gain.SetValue(gain)
        except:
            try:
                self.camera.GainRaw.SetValue(int(gain))
            except:
                pass

        # Trigger
        trigger_mode = self.config.get('trigger_mode', True)
        trigger_source = self.config.get('trigger_source', 'Software')

        if trigger_mode:
            self.camera.TriggerSelector.SetValue('FrameStart')
            self.camera.TriggerMode.SetValue('On')
            self.camera.TriggerSource.SetValue(trigger_source)
            logger.info(f"[{self.camera_id}] Trigger: {trigger_source}")
        else:
            self.camera.TriggerMode.SetValue('Off')

    def start_grabbing(self) -> bool:
        try:
            if not self.camera or not self.camera.IsOpen():
                return False

            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.frame_count = 0
            logger.info(f"[{self.camera_id}] Started grabbing")
            return True
        except Exception as e:
            logger.error(f"[{self.camera_id}] Start error: {e}")
            return False

    def stop_grabbing(self):
        try:
            if self.camera and self.camera.IsGrabbing():
                self.camera.StopGrabbing()
        except:
            pass

    def execute_software_trigger(self) -> bool:
        try:
            if self.camera and self.camera.IsOpen():
                if self.camera.WaitForFrameTriggerReady(1000, pylon.TimeoutHandling_Return):
                    self.camera.ExecuteSoftwareTrigger()
                    return True
            return False
        except:
            return False

    def grab_frame(self, timeout_ms: int = 5000):
        try:
            if not self.camera or not self.camera.IsGrabbing():
                return None

            grabResult = self.camera.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)

            if grabResult.GrabSucceeded():
                image = self.converter.Convert(grabResult)
                img_array = image.GetArray()

                self.frame_count += 1

                metadata = {
                    'camera_id': self.camera_id,
                    'width': grabResult.Width,
                    'height': grabResult.Height,
                    'frame_number': self.frame_count,
                    'timestamp': time.time(),
                }

                grabResult.Release()
                return img_array, metadata
            else:
                grabResult.Release()
                return None

        except Exception as e:
            logger.error(f"[{self.camera_id}] Grab error: {e}")
            return None

    def disconnect(self):
        try:
            self.stop_grabbing()
            if self.camera and self.camera.IsOpen():
                self.camera.Close()
            self.is_connected = False
        except:
            pass

    def __del__(self):
        self.disconnect()
