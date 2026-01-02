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
import subprocess
import shutil
import json

# Import inference service
try:
    import sys
    ai_services_path = Path(__file__).parent.parent
    if str(ai_services_path) not in sys.path:
        sys.path.insert(0, str(ai_services_path))
    from inference_service import SuperPointMatcherTRT
    INFERENCE_AVAILABLE = True
except Exception as e:
    logging.warning(f"Inference service not available: {e}")
    INFERENCE_AVAILABLE = False
    SuperPointMatcherTRT = None

logger = logging.getLogger(__name__)


class CameraMode(Enum):
    """Camera operation modes"""
    IDLE = "idle"  # Not grabbing frames
    CONTINUOUS = "continuous"  # Grab frames continuously
    HARDWARE_TRIGGER = "hardware_trigger"  # Grab on hardware trigger with inference


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
        device_index: int,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ):
        """
        Initialize Camera instance

        Args:
            serial_number: Camera serial number
            device_index: Pylon device index
            event_callback: Callback for events (event_type, data)
        """
        self.serial_number = serial_number
        self.device_index = device_index
        self.event_callback = event_callback

        # Camera state
        self.mode = CameraMode.IDLE
        self.camera: Optional[pylon.InstantCamera] = None
        self.shm: Optional[shared_memory.SharedMemory] = None
        self.shm_name = f"camera_{serial_number}"

        # Camera settings
        self.exposure_time = 10000  # μs
        self.gain = 1.0
        self.pixel_format = "Mono8"

        # Trigger config
        self.trigger_activation = "RisingEdge"
        self.di_number = 0
        self.delay_trigger = 100  # ms

        # Recipe & templates
        self.recipe_id: Optional[str] = None
        self.recipe_name: Optional[str] = None
        self.templates: List[Dict[str, Any]] = []
        self.inference_matcher: Optional[Any] = None

        # Frame tracking
        self.frame_idx = 0
        self.previous_di_value: Optional[int] = None

        # Process control
        self._running = False
        self._process: Optional[Process] = None

        logger.info(f"Camera instance created: {serial_number} (device_index={device_index})")

    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to callback"""
        if self.event_callback:
            try:
                self.event_callback(event_type, {
                    "serial_number": self.serial_number,
                    **data
                })
            except Exception as e:
                logger.error(f"Error emitting event {event_type}: {e}")

    def connect(self) -> bool:
        """Connect to Basler camera"""
        try:
            tlFactory = pylon.TlFactory.GetInstance()
            devices = tlFactory.EnumerateDevices()

            if not devices or self.device_index >= len(devices):
                logger.error(f"Camera device {self.device_index} not found")
                self._emit_event("camera_error", {
                    "error": f"Device index {self.device_index} not found"
                })
                return False

            device = devices[self.device_index]
            actual_serial = device.GetSerialNumber()
            model_name = device.GetModelName()

            if actual_serial != self.serial_number:
                logger.warning(
                    f"Serial number mismatch: expected {self.serial_number}, "
                    f"got {actual_serial}"
                )

            self.camera = pylon.InstantCamera(tlFactory.CreateDevice(device))
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
                f"Camera connected: {model_name} (SN: {self.serial_number}), "
                f"resolution: {actual_width}x{actual_height}, "
                f"shared memory: {self.shm_name}"
            )

            self._emit_event("camera_connected", {
                "model_name": model_name,
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

    def _apply_settings(self):
        """Apply camera settings"""
        if not self.camera:
            return

        try:
            # Exposure
            self.camera.ExposureTime.SetValue(int(self.exposure_time))
            actual_exposure = self.camera.ExposureTime.GetValue()
            logger.info(f"[{self.serial_number}] Exposure: {actual_exposure} μs")

            # Gain
            self.camera.Gain.SetValue(float(self.gain))
            actual_gain = self.camera.Gain.GetValue()
            logger.info(f"[{self.serial_number}] Gain: {actual_gain}")

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

        # Initialize hardware trigger if needed
        if mode == CameraMode.HARDWARE_TRIGGER and self.previous_di_value is None:
            self.previous_di_value = self._read_di_value(self.di_number)

    def _start_loop(self):
        """Start camera loop in background thread"""
        import threading
        if self._running:
            logger.warning(f"[{self.serial_number}] Camera loop already running")
            return

        thread = threading.Thread(target=self.run, daemon=True, name=f"Camera_{self.serial_number}")
        thread.start()
        logger.info(f"[{self.serial_number}] Camera loop thread started")

    def update_settings(self, settings: Dict[str, Any]):
        """Update camera settings from recipe"""
        if "exposure_time" in settings:
            self.exposure_time = settings["exposure_time"]
        if "gain" in settings:
            self.gain = settings["gain"]
        if "pixel_format" in settings:
            self.pixel_format = settings["pixel_format"]
        if "delay_trigger" in settings:
            self.delay_trigger = settings["delay_trigger"]

        # Trigger config
        trigger_config = settings.get("trigger_config", {})
        if "trigger_activation" in trigger_config:
            self.trigger_activation = trigger_config["trigger_activation"]
        if "di_number" in trigger_config:
            self.di_number = trigger_config["di_number"]

        # Apply to camera if connected
        if self.camera:
            self._apply_settings()

        logger.info(f"[{self.serial_number}] Settings updated")

    def load_recipe(self, recipe_data: Dict[str, Any]) -> bool:
        """
        Load recipe and initialize inference

        Args:
            recipe_data: Full recipe JSON

        Returns:
            True if successful
        """
        try:
            self.recipe_id = recipe_data.get("_id") or recipe_data.get("id")
            self.recipe_name = recipe_data.get("name", "Unknown")

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

            # Update settings
            self.update_settings(camera_config)

            # Load templates
            camera_templates = recipe_data.get("camera_templates", [])
            camera_id = camera_config.get("camera_id")

            for ct in camera_templates:
                if ct.get("camera_id") == camera_id:
                    self.templates = ct.get("templates", [])
                    break

            if not self.templates:
                logger.warning(f"No templates found for camera {self.serial_number}")
                return False

            # Initialize inference matcher
            if INFERENCE_AVAILABLE and SuperPointMatcherTRT:
                self.inference_matcher = self._init_inference_matcher(
                    recipe_data,
                    camera_id
                )

            logger.info(
                f"[{self.serial_number}] Recipe loaded: {self.recipe_name}, "
                f"templates: {len(self.templates)}"
            )

            # Switch to hardware trigger mode
            self.set_mode(CameraMode.HARDWARE_TRIGGER)

            return True

        except Exception as e:
            logger.error(f"Error loading recipe: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _init_inference_matcher(
        self,
        recipe_data: Dict[str, Any],
        camera_id: str,
        engine_path: str = "/home/demo/Source/ocr_datecode/weights/pipeline_fp16_small.engine"
    ):
        """Initialize TensorRT inference matcher"""
        try:
            # Get first template (will support multi-template later)
            template_data = self.templates[0]
            image_url = template_data.get("image_url")

            if not image_url:
                logger.error("No template image URL")
                return None

            # Copy template to temp directory
            filename = image_url.split("/")[-1]
            backend_dir = Path(__file__).parent.parent.parent / "backend"
            source_path = backend_dir / "uploads" / "templates" / filename

            if not source_path.exists():
                logger.error(f"Template not found: {source_path}")
                return None

            temp_dir = Path("ocr_inference")
            temp_dir.mkdir(exist_ok=True)
            template_path = temp_dir / f"template_{self.serial_number}.jpg"
            shutil.copy(source_path, template_path)

            # Parse annotations
            annotations = template_data.get("annotations", [])
            template_bbox = None
            other_bboxes = []

            template_img = cv2.imread(str(template_path))
            img_h, img_w = template_img.shape[:2]

            for ann in annotations:
                ann_type = ann.get("type", "")

                if ann_type == "template":
                    x, y = ann.get("x", 0), ann.get("y", 0)
                    w, h = ann.get("width", 0), ann.get("height", 0)

                    x1, y1 = int(x * img_w), int(y * img_h)
                    x2, y2 = int((x + w) * img_w), int((y + h) * img_h)

                    template_bbox = {
                        "type": "template",
                        "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    }

                elif ann_type == "text" and ann.get("points"):
                    pixel_points = [
                        [int(pt[0] * img_w), int(pt[1] * img_h)]
                        for pt in ann.get("points", [])
                    ]
                    other_bboxes.append({
                        "type": ann_type,
                        "text": ann.get("text", ""),
                        "points": pixel_points
                    })

            if not template_bbox:
                logger.error("No template bbox in annotations")
                return None

            # Create annotation file
            ann_json_path = temp_dir / f"annotations_{self.serial_number}.json"
            ann_data = {
                "_template_image": str(template_path),
                str(template_path): [template_bbox] + other_bboxes
            }

            with open(ann_json_path, "w") as f:
                json.dump(ann_data, f, indent=2)

            # Initialize matcher
            matcher = SuperPointMatcherTRT(
                json_path=str(ann_json_path),
                engine_path=engine_path,
                scale=1.0,
                verbose=True
            )

            logger.info(f"[{self.serial_number}] Inference matcher initialized")
            return matcher

        except Exception as e:
            logger.error(f"Error initializing matcher: {e}")
            import traceback
            traceback.print_exc()
            return None

    def stop_recipe(self):
        """Stop recipe and switch back to continuous mode"""
        self.recipe_id = None
        self.recipe_name = None
        self.templates = []
        self.inference_matcher = None
        self.set_mode(CameraMode.CONTINUOUS)

        logger.info(f"[{self.serial_number}] Recipe stopped, mode set to CONTINUOUS")

    def _read_di_value(self, di_number: int) -> int:
        """Read Digital Input value from hardware"""
        try:
            result = subprocess.run(
                ["sudo", "-n", "dio_in", str(di_number)],
                capture_output=True,
                text=True,
                timeout=1
            )

            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "status" in line and "=" in line:
                        value_str = line.split("=")[-1].strip()
                        return int(value_str)
            return 0

        except Exception as e:
            logger.error(f"Error reading DI {di_number}: {e}")
            return 0

    def _check_trigger_edge(self, current: int, previous: int) -> bool:
        """Check if trigger edge condition is met"""
        if self.trigger_activation == "RisingEdge":
            return previous == 0 and current == 1
        elif self.trigger_activation == "FallingEdge":
            return previous == 1 and current == 0
        elif self.trigger_activation == "AnyEdge":
            return previous != current
        return False

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
        Main camera loop

        Handles:
        - Continuous frame grabbing (CONTINUOUS mode)
        - Hardware trigger polling (HARDWARE_TRIGGER mode)
        - Writing frames to shared memory
        """
        if not self.camera:
            logger.error(f"[{self.serial_number}] Camera not connected")
            return

        self._running = True
        logger.info(f"[{self.serial_number}] Starting camera loop (mode: {self.mode.value})")

        try:
            # Start grabbing
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            while self._running:
                # Mode-specific behavior
                if self.mode == CameraMode.IDLE:
                    # Idle mode: just sleep
                    time.sleep(0.1)
                    continue

                elif self.mode == CameraMode.CONTINUOUS:
                    # Continuous mode: grab and write to shared memory
                    if self.camera.IsGrabbing():
                        grab_result = self.camera.RetrieveResult(100, pylon.TimeoutHandling_Return)

                        if grab_result and grab_result.GrabSucceeded():
                            img_array = grab_result.Array

                            # Convert Mono8 to BGR for consistency
                            if len(img_array.shape) == 2:
                                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

                            # Write to shared memory
                            metadata = {
                                "timestamp": time.time(),
                                "mode": self.mode.value,
                                "shape": img_array.shape,
                                "dtype": str(img_array.dtype)
                            }

                            self._write_frame_to_shm(img_array, metadata)
                            self.frame_idx += 1

                        grab_result.Release()

                    time.sleep(0.033)  # ~30 FPS

                elif self.mode == CameraMode.HARDWARE_TRIGGER:
                    # Hardware trigger mode: poll DI and trigger on edge
                    current_di = self._read_di_value(self.di_number)

                    if self.previous_di_value is not None:
                        if self._check_trigger_edge(current_di, self.previous_di_value):
                            logger.info(f"[{self.serial_number}] Trigger detected!")
                            self._handle_trigger_event()

                    self.previous_di_value = current_di
                    time.sleep(0.01)  # 100Hz polling

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
