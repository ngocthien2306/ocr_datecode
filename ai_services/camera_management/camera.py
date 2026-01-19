"""
Camera Module
Manages individual camera instance with shared memory and inference
"""

from multiprocessing import shared_memory, Process
from pypylon import pylon
import cv2
import numpy as np
import time
import pickle
import struct
import logging
from enum import Enum
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
from datetime import datetime, timezone
import shutil
import json

# Inference removed - now handled by CameraManager

logger = logging.getLogger(__name__)


class CameraMode(Enum):
    """Camera operation modes"""
    IDLE = "idle"  # Not grabbing frames
    CONTINUOUS = "continuous"  # Grab frames continuously - TODO
    SOFTWARE_TRIGGER = "software_trigger"  # Software trigger mode (DI polling in CameraManager)
    # HARDWARE_TRIGGER = "hardware_trigger"  # TODO: Hardware trigger with camera Line input


class Camera:
    """
    Manages a single Basler camera instance

    Features:
    - 3 operation modes: idle, continuous, hardware_trigger
    - Shared memory frame storage
    - Multi-template capture with delay
    - Inference with SuperPointMatcherTRT
    """

    def __init__(
        self,
        serial_number: str,
        pixel_format: str = "BGR8",
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        event_loop: Optional[Any] = None
    ):
        """
        Initialize Camera instance

        Args:
            serial_number: Camera serial number
            pixel_format: Pixel format (Mono8, RGB8, etc.) from DB config
            event_callback: Callback for events (event_type, data)
            event_loop: Event loop for async callbacks from thread
        """
        self.serial_number = serial_number
        self.model_name: Optional[str] = None  # Will be set when camera connects
        self.event_callback = event_callback
        self.event_loop = event_loop

        # Camera state
        self.mode = CameraMode.IDLE
        self.camera: Optional[pylon.InstantCamera] = None
        self.shm: Optional[shared_memory.SharedMemory] = None
        self.shm_name = f"camera_{serial_number}"

        # Camera settings (from DB config or defaults)
        self.exposure_time = 500  # μs (default 500μs = 0.5ms)
        self.gain = 1.0
        self.pixel_format = pixel_format  # From DB config
        self.delay_trigger = 100  # ms (delay from sensor to first frame)
        self.delay_interval = 500  # ms (delay between frames for multi-template)

        # Reject config (from recipe)
        self.delay_reject = 4000  # ms (default: 4s from camera to reject station)
        self.do_reject_number = 2  # DO pin number for reject (default: DO2)

        # Trigger config (per-camera)
        self.trigger_mode = "continuous"  # "continuous", "software_trigger", "hardware_trigger"
        self.trigger_selector = "FrameStart"  # "FrameStart", "ExposureStart", "FrameBurstStart"
        self.trigger_activation = "RisingEdge"  # "RisingEdge", "FallingEdge", "AnyEdge"
        self.di_number = 0  # Digital Input number (0-3) for software trigger
        self.trigger_source = "Line0"  # Camera Line input for hardware trigger (TODO)

        # Recipe & templates
        self.recipe_id: Optional[str] = None
        self.recipe_name: Optional[str] = None
        self.templates: List[Dict[str, Any]] = []
        self.function_type: str = "OCR"  # Function type: OCR, Check_Type_Product, etc.
        self.expected_texts: Dict[int, Dict[int, str]] = {}  # Map template_idx -> {region_idx -> expected_text}
        self.matching_threshold: float = 0.85  # Template matching similarity threshold
        self.recognition_threshold: float = 0.5  # OCR recognition confidence threshold

        # Frame tracking
        self.frame_idx = 0
        self.captured_frames = []  # Store frames after software trigger

        # Process control
        self._running = False
        self._process: Optional[Process] = None

        logger.info(f"Camera instance created: {serial_number}, pixel_format={pixel_format}")

    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to callback (handles both sync and async callbacks from thread)"""
        if self.event_callback:
            try:
                import asyncio
                import inspect

                event_data = {
                    "serial_number": self.serial_number,
                    **data
                }

                # Check if callback is async
                if inspect.iscoroutinefunction(self.event_callback):
                    # Async callback - need to schedule in event loop
                    if self.event_loop:
                        # Use provided event loop
                        asyncio.run_coroutine_threadsafe(
                            self.event_callback(event_type, event_data),
                            self.event_loop
                        )
                    else:
                        logger.warning(f"Async callback but no event loop provided for {event_type}")
                else:
                    # Sync callback
                    self.event_callback(event_type, event_data)

            except Exception as e:
                logger.error(f"Error emitting event {event_type}: {e}")

    def connect(self) -> bool:
        """Connect to Basler camera by serial number"""
        try:
            tlFactory = pylon.TlFactory.GetInstance()
            devices = tlFactory.EnumerateDevices()

            if not devices:
                logger.error("No Basler cameras found")
                self._emit_event("camera_error", {
                    "error": "No cameras detected"
                })
                return False

            # Find device by serial number
            target_device = None
            available_serials = []

            for device in devices:
                serial = device.GetSerialNumber()
                available_serials.append(serial)

                if serial == self.serial_number:
                    target_device = device
                    break

            if not target_device:
                logger.error(f"Camera with serial {self.serial_number} not found")
                logger.error(f"Available cameras: {available_serials}")
                self._emit_event("camera_error", {
                    "error": f"Serial {self.serial_number} not found",
                    "available_cameras": available_serials
                })
                return False

            self.model_name = target_device.GetModelName()

            self.camera = pylon.InstantCamera(tlFactory.CreateDevice(target_device))
            self.camera.Open()

            # Apply initial settings
            self._apply_settings()

            # Setup shared memory
            actual_width = self.camera.Width.GetValue()
            actual_height = self.camera.Height.GetValue()
            max_frame_size = actual_width * actual_height * 3  # BGR
            shm_size = max_frame_size + 4096  # Extra for metadata

            # Cleanup existing shared memory
            try:
                existing_shm = shared_memory.SharedMemory(name=self.shm_name)
                existing_shm.close()
                existing_shm.unlink()
            except FileNotFoundError:
                pass

            # Create new shared memory
            self.shm = shared_memory.SharedMemory(
                name=self.shm_name,
                create=True,
                size=shm_size
            )

            logger.info(
                f"Camera connected: {self.model_name} (SN: {self.serial_number}), "
                f"resolution: {actual_width}x{actual_height}, "
                f"shared memory: {self.shm_name}"
            )

            self._emit_event("camera_connected", {
                "serial_number": self.serial_number,
                "model_name": self.model_name,
                "resolution": [actual_width, actual_height]
            })

            return True

        except Exception as e:
            logger.error(f"Failed to connect camera {self.serial_number}: {e}")
            self._emit_event("camera_error", {"error": str(e)})
            return False

    def disconnect(self):
        """Disconnect camera and cleanup resources"""
        try:
            if self.camera and self.camera.IsGrabbing():
                self.camera.StopGrabbing()

            if self.camera and self.camera.IsOpen():
                self.camera.Close()

            if self.shm:
                self.shm.close()
                self.shm.unlink()

            logger.info(f"Camera disconnected: {self.serial_number}")
            self._emit_event("camera_disconnected", {})

        except Exception as e:
            logger.error(f"Error disconnecting camera {self.serial_number}: {e}")

    def _apply_settings(self, apply_pixel_format: bool = True):
        """
        Apply camera settings

        Args:
            apply_pixel_format: Whether to apply pixel format (only True during initial connect)
        """
        if not self.camera:
            return

        try:
            # Pixel Format (can ONLY be set when camera is NOT grabbing)
            # Only apply during initial connect, not during runtime settings update
            if apply_pixel_format and self.pixel_format and not self.camera.IsGrabbing():
                self.camera.PixelFormat.SetValue(self.pixel_format)
                actual_pixel_format = self.camera.PixelFormat.GetValue()
                logger.info(f"[{self.serial_number}] PixelFormat: {actual_pixel_format}")

            # Exposure (can be changed during grabbing) - try different node names
            try:
                self.camera.ExposureTime.SetValue(int(self.exposure_time))
                actual_exposure = self.camera.ExposureTime.GetValue()
                logger.info(f"[{self.serial_number}] Exposure: {actual_exposure} μs")
            except:
                try:
                    self.camera.ExposureTimeAbs.SetValue(float(self.exposure_time))
                    actual_exposure = self.camera.ExposureTimeAbs.GetValue()
                    logger.info(f"[{self.serial_number}] ExposureAbs: {actual_exposure} μs")
                except Exception as e:
                    logger.error(f"[{self.serial_number}] Failed to set exposure time: {e}")

            # Gain (can be changed during grabbing) - try different node names
            try:
                self.camera.Gain.SetValue(float(self.gain))
                actual_gain = self.camera.Gain.GetValue()
                logger.info(f"[{self.serial_number}] Gain: {actual_gain}")
            except:
                try:
                    self.camera.GainRaw.SetValue(int(self.gain))
                    actual_gain = self.camera.GainRaw.GetValue()
                    logger.info(f"[{self.serial_number}] GainRaw: {actual_gain}")
                except Exception as e:
                    logger.error(f"[{self.serial_number}] Failed to set gain: {e}")

        except Exception as e:
            logger.error(f"Error applying settings: {e}")

    def set_mode(self, mode: CameraMode):
        """Set camera operation mode"""
        old_mode = self.mode
        self.mode = mode

        logger.info(f"[{self.serial_number}] Mode changed: {old_mode.value} → {mode.value}")

        # Start camera loop if switching from IDLE to active mode
        if old_mode == CameraMode.IDLE and mode != CameraMode.IDLE:
            if not self._running:
                self._start_loop()

    def _start_loop(self):
        """Start camera loop in background thread"""
        import threading
        if self._running:
            logger.warning(f"[{self.serial_number}] Camera loop already running")
            return

        thread = threading.Thread(target=self.run, daemon=True, name=f"Camera_{self.serial_number}")
        thread.start()
        logger.info(f"[{self.serial_number}] Camera loop thread started")

    def set_exposure_time(self, exposure_time: float):
        """
        Set camera exposure time

        Args:
            exposure_time: Exposure time in microseconds
        """
        self.exposure_time = exposure_time

        # Apply to camera if connected
        if self.camera and self.camera.IsOpen():
            try:
                self.camera.ExposureTime.SetValue(int(exposure_time))
                logger.info(f"[{self.serial_number}] Exposure time set to {exposure_time}μs")
            except:
                try:
                    self.camera.ExposureTimeAbs.SetValue(float(exposure_time))
                    logger.info(f"[{self.serial_number}] Exposure time set to {exposure_time}μs")
                except Exception as e:
                    logger.error(f"[{self.serial_number}] Failed to set exposure time: {e}")

    def set_gain(self, gain: float):
        """
        Set camera gain

        Args:
            gain: Gain value
        """
        self.gain = gain

        # Apply to camera if connected
        if self.camera and self.camera.IsOpen():
            try:
                self.camera.Gain.SetValue(float(gain))
                logger.info(f"[{self.serial_number}] Gain set to {gain}")
            except:
                try:
                    self.camera.GainRaw.SetValue(int(gain))
                    logger.info(f"[{self.serial_number}] Gain set to {gain}")
                except Exception as e:
                    logger.error(f"[{self.serial_number}] Failed to set gain: {e}")

    def update_settings(self, settings: Dict[str, Any]):
        """Update camera settings from recipe"""
        if "exposure_time" in settings:
            self.set_exposure_time(settings["exposure_time"])
        if "gain" in settings:
            self.set_gain(settings["gain"])
        if "pixel_format" in settings:
            self.pixel_format = settings["pixel_format"]
        if "delay_trigger" in settings:
            self.delay_trigger = settings["delay_trigger"]
        if "delay_interval" in settings:
            self.delay_interval = settings["delay_interval"]

        # Trigger mode
        if "trigger_mode" in settings:
            self.trigger_mode = settings["trigger_mode"]

        # Trigger config
        trigger_config = settings.get("trigger_config", {})
        if "trigger_selector" in trigger_config:
            self.trigger_selector = trigger_config["trigger_selector"]
        if "trigger_activation" in trigger_config:
            self.trigger_activation = trigger_config["trigger_activation"]
        if "di_number" in trigger_config:
            self.di_number = trigger_config["di_number"]
        if "trigger_source" in trigger_config:
            self.trigger_source = trigger_config["trigger_source"]

        # Apply pixel format if needed (skip if grabbing)
        if self.camera and "pixel_format" in settings:
            self._apply_settings(apply_pixel_format=True)

        logger.info(f"[{self.serial_number}] Settings updated")

    def configure_software_trigger(self) -> bool:
        """
        Configure camera for software trigger mode

        Returns:
            True if successful
        """
        if not self.camera or not self.camera.IsOpen():
            logger.error(f"[{self.serial_number}] Camera not open")
            return False

        try:
            # Stop grabbing temporarily
            was_grabbing = self.camera.IsGrabbing()
            if was_grabbing:
                self.camera.StopGrabbing()

            # Configure software trigger
            self.camera.TriggerSelector.SetValue(self.trigger_selector)
            self.camera.TriggerMode.SetValue("On")
            self.camera.TriggerSource.SetValue("Software")

            # Set TriggerActivation if supported (some camera models don't have this feature)
            try:
                # Check if TriggerActivation parameter exists and is accessible
                if hasattr(self.camera, 'TriggerActivation'):
                    # For GenICam parameters, check GetAccessMode()
                    try:
                        from pypylon import genicam
                        access_mode = self.camera.TriggerActivation.GetAccessMode()
                        if access_mode == genicam.RW or access_mode == genicam.WO:
                            self.camera.TriggerActivation.SetValue(self.trigger_activation)
                            logger.info(
                                f"[{self.serial_number}] Software trigger configured: "
                                f"Selector={self.trigger_selector}, Activation={self.trigger_activation}"
                            )
                        else:
                            logger.warning(
                                f"[{self.serial_number}] TriggerActivation not writable "
                                f"(AccessMode={access_mode}, model: {self.model_name})"
                            )
                    except Exception as e:
                        logger.warning(
                            f"[{self.serial_number}] Could not set TriggerActivation: {e} "
                            f"(model: {self.model_name})"
                        )
                else:
                    logger.warning(
                        f"[{self.serial_number}] TriggerActivation not available on camera model: {self.model_name}"
                    )
            except Exception as e:
                logger.warning(
                    f"[{self.serial_number}] Error checking TriggerActivation: {e}"
                )

            # Resume grabbing with OneByOne strategy
            if was_grabbing or self.mode == CameraMode.SOFTWARE_TRIGGER:
                self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
                logger.info(f"[{self.serial_number}] Camera grabbing started (OneByOne)")

            return True

        except Exception as e:
            logger.error(f"[{self.serial_number}] Error configuring software trigger: {e}")
            import traceback
            traceback.print_exc()
            return False

    def execute_software_trigger(self) -> Dict[str, Any]:
        """
        Execute software trigger and capture N frames (N = number of templates)

        Process:
        1. Delay before first frame
        2. For each template:
           - ExecuteSoftwareTrigger()
           - Retrieve frame
           - Write to shared memory
           - Delay before next frame

        Returns:
            {
                'success': bool,
                'frames': List[np.ndarray],
                'frame_count': int,
                'error': str (if failed)
            }
        """
        if not self.camera or not self.camera.IsGrabbing():
            return {'success': False, 'error': 'Camera not grabbing'}

        if not self.templates:
            return {'success': False, 'error': 'No templates loaded'}

        self.captured_frames = []

        try:
            logger.info(
                f"[{self.serial_number}] Executing SW trigger for {len(self.templates)} frames, "
                f"delay={self.delay_trigger}ms"
            )

            for idx, template in enumerate(self.templates):
                # IMPORTANT: Delay BEFORE each frame (including first)
                time.sleep(self.delay_trigger / 1000.0)

                # Execute software trigger
                self.camera.ExecuteSoftwareTrigger()

                # Wait and retrieve frame
                grab_result = self.camera.RetrieveResult(
                    2000,  # 2s timeout
                    pylon.TimeoutHandling_ThrowException
                )

                if not grab_result or not grab_result.GrabSucceeded():
                    if grab_result:
                        grab_result.Release()
                    return {
                        'success': False,
                        'error': f'Failed to grab frame {idx}'
                    }

                img_array = grab_result.Array

                # Convert Mono8 to BGR
                if len(img_array.shape) == 2:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

                # Write to shared memory (Option B - write all frames)
                metadata = {
                    "timestamp": time.time(),
                    "mode": self.mode.value,
                    "shape": img_array.shape,
                    "dtype": str(img_array.dtype),
                    "frame_idx": idx,
                    "trigger_event": True,
                    "template_name": template.get('name', f'Template {idx+1}')
                }
                self._write_frame_to_shm(img_array, metadata)

                # Store frame
                self.captured_frames.append(img_array.copy())

                grab_result.Release()

                logger.info(f"[{self.serial_number}] Captured frame {idx+1}/{len(self.templates)}")

            return {
                'success': True,
                'frames': self.captured_frames,
                'frame_count': len(self.captured_frames)
            }

        except Exception as e:
            logger.error(f"[{self.serial_number}] Error executing software trigger: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }

    def execute_software_trigger_immediate(self) -> Dict[str, Any]:
        """
        Execute software trigger IMMEDIATELY without delay_trigger

        Differences from execute_software_trigger():
        - OLD: time.sleep(delay_trigger) before each frame
        - NEW: NO delay_trigger, but APPLY delay_interval between frames

        This is called from Timer callback after delay_trigger has already passed

        For multi-template capture:
        - Frame 0: Capture immediately (no delay)
        - Frame 1+: Apply delay_interval before capture

        Returns:
            Dict with success status and captured frames
        """
        if not self.camera or not self.camera.IsGrabbing():
            return {'success': False, 'error': 'Camera not grabbing'}

        if not self.templates:
            return {'success': False, 'error': 'No templates loaded'}

        self.captured_frames = []
        num_templates = len(self.templates)

        try:
            if num_templates > 1:
                logger.info(
                    f"[{self.serial_number}] Capturing {num_templates} frames "
                    f"with delay_interval={self.delay_interval}ms between frames (REVERSED order)"
                )
            else:
                logger.info(
                    f"[{self.serial_number}] Capturing 1 frame IMMEDIATELY"
                )

            # IMPORTANT: Reverse template order to match physical product movement
            # Product moves with Template 2 appearance first, then Template 1
            # We capture in reverse order but store in a dict to maintain template_idx mapping
            reversed_templates = list(reversed(list(enumerate(self.templates))))
            frames_dict = {}  # {template_idx: frame_array}

            for capture_idx, (template_idx, template) in enumerate(reversed_templates):
                # Apply delay_interval between frames (skip for first frame)
                if capture_idx > 0:
                    time.sleep(self.delay_interval / 1000.0)
                    logger.debug(
                        f"[{self.serial_number}] Delayed {self.delay_interval}ms "
                        f"before capture {capture_idx+1}"
                    )

                # Execute software trigger
                self.camera.ExecuteSoftwareTrigger()

                # Wait and retrieve frame
                grab_result = self.camera.RetrieveResult(
                    2000,  # 2s timeout
                    pylon.TimeoutHandling_ThrowException
                )

                if not grab_result or not grab_result.GrabSucceeded():
                    if grab_result:
                        grab_result.Release()
                    return {
                        'success': False,
                        'error': f'Failed to grab frame {capture_idx}'
                    }

                img_array = grab_result.Array

                # Convert Mono8 to BGR
                if len(img_array.shape) == 2:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

                # Write to shared memory
                # Use template_idx (original index) for proper verification mapping
                metadata = {
                    "timestamp": time.time(),
                    "mode": self.mode.value,
                    "shape": img_array.shape,
                    "dtype": str(img_array.dtype),
                    "frame_idx": template_idx,  # Use original template index
                    "trigger_event": True,
                    "template_name": template.get('name', f'Template {template_idx+1}')
                }
                self._write_frame_to_shm(img_array, metadata)

                # Store frame in dict with template_idx as key
                frames_dict[template_idx] = img_array.copy()

                grab_result.Release()

                logger.debug(
                    f"[{self.serial_number}] Captured template {template_idx} "
                    f"(capture order: {capture_idx+1}/{num_templates})"
                )

            # Rebuild captured_frames in ORIGINAL template order (0, 1, 2, ...)
            # This ensures inference gets frames in correct order matching expected_texts
            self.captured_frames = [frames_dict[i] for i in range(num_templates)]

            logger.info(
                f"[{self.serial_number}] ✅ Captured {len(self.captured_frames)} frames (reordered to match templates)"
            )

            return {
                'success': True,
                'frames': self.captured_frames,
                'frame_count': len(self.captured_frames)
            }

        except Exception as e:
            logger.error(f"[{self.serial_number}] Error executing immediate trigger: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }

    def load_recipe(self, recipe_data: Dict[str, Any]) -> bool:
        """
        Load recipe and initialize inference

        Args:
            recipe_data: Full recipe JSON

        Returns:
            True if successful
        """
        try:
            print(recipe_data)
            self.recipe_id = recipe_data.get("_id") or recipe_data.get("id")
            self.recipe_name = recipe_data.get("name", "Unknown")

            # Load reject config from recipe (recipe level, not camera level)
            self.delay_reject = recipe_data.get("delay_reject", 4000)  # ms, default 4s
            self.do_reject_number = recipe_data.get("do_reject_number", 2)  # DO2 default

            # Load model thresholds from recipe
            model_thresholds = recipe_data.get("model_thresholds", {})
            self.matching_threshold = model_thresholds.get("matching_threshold", 0.85)
            self.recognition_threshold = model_thresholds.get("recognition_threshold", 0.5)
            logger.info(
                f"[{self.serial_number}] Loaded thresholds: "
                f"matching={self.matching_threshold}, recognition={self.recognition_threshold}"
            )

            # Find camera config in recipe
            cameras = recipe_data.get("cameras", [])
            camera_config = None
            for cam in cameras:
                if cam.get("serial_number") == self.serial_number:
                    camera_config = cam
                    break

            if not camera_config:
                logger.error(f"Camera {self.serial_number} not found in recipe")
                return False

            # Store function_type from camera config
            self.function_type = camera_config.get('function_type', 'OCR')

            # Log camera config for debugging
            logger.info(f"[{self.serial_number}] Camera config from recipe:")
            logger.info(f"  - pixel_format: {camera_config.get('pixel_format', 'NOT SET')}")
            logger.info(f"  - function_type: {self.function_type}")
            logger.info(f"  - exposure_time: {camera_config.get('exposure_time', 'NOT SET')}")
            logger.info(f"  - delay_trigger: {camera_config.get('delay_trigger', 'NOT SET')}")
            logger.info(f"  - delay_reject: {self.delay_reject}ms (recipe level)")
            logger.info(f"  - do_reject_number: DO{self.do_reject_number} (recipe level)")

            # Update settings
            self.update_settings(camera_config)

            # Load templates and parse expected texts
            camera_templates = recipe_data.get("camera_templates", [])
            camera_id = camera_config.get("camera_id")

            for ct in camera_templates:
                if ct.get("camera_id") == camera_id:
                    self.templates = ct.get("templates", [])

                    # Parse expected texts from ALL templates
                    # IMPORTANT: region_idx here matches the annotation index in the original annotations list
                    # This must match how transformed_bboxes are indexed during inference
                    self.expected_texts = {}
                    for template_idx, template in enumerate(self.templates):
                        annotations = template.get("annotations", [])
                        template_expected_texts = {}

                        for ann_idx, ann in enumerate(annotations):
                            if ann.get('type') == 'text':
                                expected_text = ann.get('text', '')
                                if expected_text:
                                    # Use ann_idx (annotation index) as the key
                                    template_expected_texts[ann_idx] = expected_text
                                    logger.info(
                                        f"  - Template {template_idx}, Annotation {ann_idx} (text): '{expected_text}'"
                                    )

                        if template_expected_texts:
                            self.expected_texts[template_idx] = template_expected_texts

                    logger.info(
                        f"[{self.serial_number}] Loaded expected_texts for {len(self.expected_texts)} templates"
                    )
                    break

            if not self.templates:
                logger.warning(f"No templates found for camera {self.serial_number}")
                return False

            logger.info(
                f"[{self.serial_number}] Recipe loaded: {self.recipe_name}, "
                f"templates: {len(self.templates)}, "
                f"trigger_mode: {self.trigger_mode}"
            )

            # Configure trigger mode based on settings
            if self.trigger_mode == "software_trigger":
                success = self.configure_software_trigger()
                if success:
                    self.set_mode(CameraMode.SOFTWARE_TRIGGER)
                    logger.info(f"[{self.serial_number}] Software trigger mode configured")
                else:
                    logger.error(f"[{self.serial_number}] Failed to configure software trigger")
                    return False

            elif self.trigger_mode == "continuous":
                # TODO: Implement continuous mode
                logger.warning(f"[{self.serial_number}] Continuous mode not implemented yet")
                self.set_mode(CameraMode.CONTINUOUS)

            elif self.trigger_mode == "hardware_trigger":
                # TODO: Implement hardware trigger mode
                logger.warning(f"[{self.serial_number}] Hardware trigger mode not implemented yet")
                return False

            else:
                logger.error(f"[{self.serial_number}] Unknown trigger mode: {self.trigger_mode}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error loading recipe: {e}")
            import traceback
            traceback.print_exc()
            return False

    def stop_recipe(self):
        """Stop recipe and switch back to continuous mode"""
        self.recipe_id = None
        self.recipe_name = None
        self.templates = []
        self.set_mode(CameraMode.CONTINUOUS)

        logger.info(f"[{self.serial_number}] Recipe stopped, mode set to CONTINUOUS")

    # NOTE: Old trigger methods removed (_read_di_value, _check_trigger_edge, _handle_trigger_event)
    # Trigger logic moved to CameraManager for centralized multi-camera triggering

    def _save_inference_image(self, img: np.ndarray, frame_idx: int, result: str) -> Optional[str]:
        """
        Save inference result image to disk

        Args:
            img: Annotated image (BGR format)
            frame_idx: Frame index
            result: PASS or FAIL

        Returns:
            Relative path to saved image
        """
        try:
            # Create directory structure
            base_dir = Path("/home/demo/Source/ocr_datecode/backend/uploads/inference_results")
            recipe_dir = base_dir / self.recipe_id if self.recipe_id else base_dir / "unknown"
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            save_dir = recipe_dir / today
            save_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename
            timestamp = datetime.now(timezone.utc).strftime("%H%M%S%f")
            filename = f"{self.serial_number}_{timestamp}_{result.lower()}_f{frame_idx}.jpg"

            # Save image
            save_path = save_dir / filename
            cv2.imwrite(str(save_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Return relative path from uploads directory
            relative_path = f"inference_results/{self.recipe_id or 'unknown'}/{today}/{filename}"

            logger.info(f"[{self.serial_number}] Image saved: {relative_path}")

            return relative_path

        except Exception as e:
            logger.error(f"[{self.serial_number}] Error saving inference image: {e}")
            return None

    def _write_frame_to_shm(self, img_array: np.ndarray, metadata: Dict[str, Any]):
        """Write frame and metadata to shared memory"""
        if not self.shm:
            return

        try:
            metadata_bytes = pickle.dumps(metadata)
            metadata_len = len(metadata_bytes)

            frame_bytes = img_array.tobytes()
            frame_len = len(frame_bytes)

            offset = 0

            # Frame version (8 bytes)
            struct.pack_into("<Q", self.shm.buf, offset, self.frame_idx)
            offset += 8

            # Metadata length (4 bytes)
            struct.pack_into("<I", self.shm.buf, offset, metadata_len)
            offset += 4

            # Metadata bytes
            self.shm.buf[offset:offset+metadata_len] = metadata_bytes
            offset += metadata_len

            # Frame length (4 bytes)
            struct.pack_into("<I", self.shm.buf, offset, frame_len)
            offset += 4

            # Frame bytes
            self.shm.buf[offset:offset+frame_len] = frame_bytes
            offset += frame_len

            # Frame version verify (8 bytes)
            struct.pack_into("<Q", self.shm.buf, offset, self.frame_idx)

        except Exception as e:
            logger.error(f"Error writing to shared memory: {e}")

    def run(self):
        """
        Main camera loop - SIMPLIFIED

        Handles:
        - IDLE mode: Sleep
        - CONTINUOUS mode: Grab and write to SHM (TODO - not implemented)
        - SOFTWARE_TRIGGER mode: Keep camera ready, waiting for ExecuteSoftwareTrigger()

        Note: DI polling and trigger detection moved to CameraManager
        """
        if not self.camera:
            logger.error(f"[{self.serial_number}] Camera not connected")
            return

        self._running = True
        logger.info(f"[{self.serial_number}] Starting camera loop (mode: {self.mode.value})")

        try:
            # Start grabbing based on mode
            if self.mode == CameraMode.SOFTWARE_TRIGGER:
                # Already started in configure_software_trigger() with OneByOne strategy
                if not self.camera.IsGrabbing():
                    self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
                    logger.info(f"[{self.serial_number}] Camera grabbing started (OneByOne)")
            else:
                # For other modes, use LatestImageOnly
                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            while self._running:
                if self.mode == CameraMode.IDLE:
                    # Idle mode: just sleep
                    time.sleep(0.1)
                    continue

                elif self.mode == CameraMode.CONTINUOUS:
                    # Continuous mode: grab frames and write to shared memory
                    try:
                        # Retrieve frame with timeout (30 FPS = ~33ms)
                        grab_result = self.camera.RetrieveResult(100, pylon.TimeoutHandling_Return)

                        if grab_result and grab_result.GrabSucceeded():
                            # Get image array
                            img_array = grab_result.Array

                            # Convert Mono8 to BGR if needed
                            if len(img_array.shape) == 2:
                                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

                            # Write to shared memory
                            metadata = {
                                "timestamp": time.time(),
                                "mode": self.mode.value,
                                "shape": img_array.shape,
                                "dtype": str(img_array.dtype),
                                "frame_idx": self.frame_idx,
                                "trigger_event": False
                            }
                            self._write_frame_to_shm(img_array, metadata)

                            # Increment frame counter
                            self.frame_idx += 1

                            # Release grab result
                            grab_result.Release()
                        else:
                            # No frame available, sleep briefly
                            time.sleep(0.001)

                    except Exception as e:
                        logger.error(f"[{self.serial_number}] Error in continuous mode: {e}")
                        time.sleep(0.01)

                elif self.mode == CameraMode.SOFTWARE_TRIGGER:
                    # Just keep camera ready, waiting for ExecuteSoftwareTrigger()
                    # No polling here - CameraManager handles trigger detection
                    time.sleep(0.1)

        except Exception as e:
            logger.error(f"[{self.serial_number}] Error in camera loop: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Stop grabbing
            if self.camera and self.camera.IsGrabbing():
                self.camera.StopGrabbing()

            self._running = False
            logger.info(f"[{self.serial_number}] Camera loop stopped")
