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
        pixel_format: str = "Mono8",
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ):
        """
        Initialize Camera instance

        Args:
            serial_number: Camera serial number
            pixel_format: Pixel format (Mono8, RGB8, etc.) from DB config
            event_callback: Callback for events (event_type, data)
        """
        self.serial_number = serial_number
        self.event_callback = event_callback

        # Camera state
        self.mode = CameraMode.IDLE
        self.camera: Optional[pylon.InstantCamera] = None
        self.shm: Optional[shared_memory.SharedMemory] = None
        self.shm_name = f"camera_{serial_number}"

        # Camera settings (from DB config or defaults)
        self.exposure_time = 10000  # μs
        self.gain = 1.0
        self.pixel_format = pixel_format  # From DB config

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

        logger.info(f"Camera instance created: {serial_number}, pixel_format={pixel_format}")

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

            model_name = target_device.GetModelName()

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
            # Pixel Format (must be set after Open but before StartGrabbing)
            if self.pixel_format:
                self.camera.PixelFormat.SetValue(self.pixel_format)
                actual_pixel_format = self.camera.PixelFormat.GetValue()
                logger.info(f"[{self.serial_number}] PixelFormat: {actual_pixel_format}")

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

            # Log camera config for debugging
            logger.info(f"[{self.serial_number}] Camera config from recipe: pixel_format={camera_config.get('pixel_format', 'NOT SET')}")

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

    def _handle_trigger_event(self):
        """
        Handle hardware trigger event

        Process:
        1. Capture multiple frames (for multi-template recipes)
        2. Run inference on each frame
        3. Aggregate results (PASS only if all templates match)
        4. Emit inference result event
        """
        if not self.camera or not self.camera.IsGrabbing():
            logger.error(f"[{self.serial_number}] Camera not grabbing, cannot handle trigger")
            return

        if not self.recipe_id:
            logger.warning(f"[{self.serial_number}] No recipe loaded, ignoring trigger")
            return

        try:
            logger.info(f"[{self.serial_number}] ⚡ Trigger event - capturing {len(self.templates)} frames")

            captured_frames = []

            # Capture frames for each template with delay
            for idx, template in enumerate(self.templates):
                # Wait for delay_trigger if not first frame
                if idx > 0:
                    time.sleep(self.delay_trigger / 1000.0)  # Convert ms to seconds

                # Grab frame
                grab_result = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)

                if grab_result and grab_result.GrabSucceeded():
                    img_array = grab_result.Array

                    # Convert Mono8 to BGR
                    if len(img_array.shape) == 2:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

                    captured_frames.append({
                        'template_idx': idx,
                        'template_name': template.get('name', f'Template {idx+1}'),
                        'image': img_array.copy(),
                        'timestamp': time.time()
                    })

                    logger.info(f"[{self.serial_number}] Captured frame {idx+1}/{len(self.templates)}")

                grab_result.Release()

            # Run inference on all frames
            if not self.inference_matcher or not INFERENCE_AVAILABLE:
                logger.warning(f"[{self.serial_number}] Inference not available, skipping")
                return

            inference_results = []
            overall_pass = True

            for frame_data in captured_frames:
                img = frame_data['image']

                # Run inference directly on numpy array
                try:
                    result = self.inference_matcher.match_array(img)

                    # Determine PASS/FAIL based on template match
                    # Use inliers count and confidence score
                    inliers = int(result.get('inliers', 0))  # Convert numpy.uint64 to Python int
                    confidence = float(result.get('confidence', 0.0))  # Convert numpy.float64 to Python float
                    total_matches = int(result.get('total_matches', 0))

                    frame_pass = inliers >= 30 and confidence >= 0.3  # Match if >= 30 inliers and confidence >= 0.3

                    # Draw bounding boxes on image for annotation
                    annotated_img = img.copy()
                    transformed_bboxes = result.get('transformed_bboxes', [])

                    # Draw bboxes
                    for bbox in transformed_bboxes:
                        points = bbox.get('points', [])
                        bbox_type = bbox.get('type', '')

                        if len(points) >= 4:
                            pts = np.array(points, dtype=np.int32)

                            # Color based on type: template=green, text=blue, others=yellow
                            if bbox_type == 'template':
                                color = (0, 255, 0) if frame_pass else (0, 0, 255)  # Green if PASS, Red if FAIL
                            elif bbox_type == 'text':
                                color = (255, 128, 0)  # Orange
                            else:
                                color = (255, 255, 0)  # Yellow

                            cv2.polylines(annotated_img, [pts], True, color, 3)

                    # Add PASS/FAIL text
                    text = f"{'PASS' if frame_pass else 'FAIL'} | Conf: {confidence:.2f} | Inliers: {inliers}"
                    cv2.putText(annotated_img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                               (0, 255, 0) if frame_pass else (0, 0, 255), 3)

                    # Save image for FAIL results
                    image_path = None
                    image_base64 = None

                    if not frame_pass:  # Only save FAIL images
                        # Save full resolution to disk (for historical)
                        image_path = self._save_inference_image(
                            annotated_img,
                            frame_data['template_idx'],
                            'FAIL'
                        )

                        # Create resized base64 for realtime preview (1/3 resolution)
                        h, w = annotated_img.shape[:2]
                        preview_img = cv2.resize(annotated_img, (w//3, h//3), interpolation=cv2.INTER_AREA)

                        # Encode to JPEG with quality 85
                        _, buffer = cv2.imencode('.jpg', preview_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        import base64
                        image_base64 = base64.b64encode(buffer).decode('utf-8')

                    # Remove numpy arrays from result for JSON serialization
                    result_clean = {
                        'success': bool(result.get('success', False)),
                        'confidence': confidence,
                        'inliers': inliers,
                        'total_matches': total_matches,
                        'timings': result.get('timings', {})
                    }

                except Exception as e:
                    logger.error(f"[{self.serial_number}] Inference error: {e}")
                    import traceback
                    traceback.print_exc()
                    result_clean = {'error': str(e)}
                    frame_pass = False
                    inliers = 0
                    confidence = 0.0
                    image_path = None
                    image_base64 = None

                overall_pass = overall_pass and frame_pass

                inference_results.append({
                    'template_name': frame_data['template_name'],
                    'frame_idx': frame_data['template_idx'],  # Renamed to match Pydantic schema
                    'pass_fail': 'PASS' if frame_pass else 'FAIL',
                    'confidence': confidence,
                    'image_path': image_path,  # Relative path (for historical)
                    'image_base64': image_base64,  # Base64 preview (for realtime, FAIL only)
                    'detected_regions': None,  # Optional field
                    'metadata': {
                        'inliers': inliers,
                        'total_matches': total_matches,
                        'timings': result_clean.get('timings', {}),
                        'timestamp': frame_data['timestamp']
                    }
                })

                logger.info(
                    f"[{self.serial_number}] Template '{frame_data['template_name']}': "
                    f"{'PASS' if frame_pass else 'FAIL'} (inliers: {inliers}, confidence: {confidence:.2f})"
                )

            # Emit inference result event
            self._emit_event('inference_result', {
                'recipe_id': self.recipe_id,
                'recipe_name': self.recipe_name,
                'product_pass_fail': 'PASS' if overall_pass else 'FAIL',
                'camera_results': [{
                    'camera_id': self.serial_number,
                    'serial_number': self.serial_number,  # Add required field
                    'frames': inference_results
                }],
                'metadata': {
                    'frame_count': len(captured_frames),
                    'overall_pass': overall_pass,
                    'timestamp': time.time()
                }
            })

            logger.info(
                f"[{self.serial_number}] ✅ Inference complete: "
                f"{'PASS' if overall_pass else 'FAIL'} ({len(captured_frames)} frames)"
            )

        except Exception as e:
            logger.error(f"[{self.serial_number}] Error handling trigger: {e}")
            import traceback
            traceback.print_exc()

            # Emit error event
            self._emit_event('inference_error', {
                'recipe_id': self.recipe_id,
                'error': str(e)
            })

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
            from datetime import datetime

            # Create directory structure
            base_dir = Path("/home/demo/Source/ocr_datecode/backend/uploads/inference_results")
            recipe_dir = base_dir / self.recipe_id if self.recipe_id else base_dir / "unknown"
            today = datetime.utcnow().strftime("%Y-%m-%d")
            save_dir = recipe_dir / today
            save_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename
            timestamp = datetime.utcnow().strftime("%H%M%S%f")
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
