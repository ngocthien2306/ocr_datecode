"""
Camera Module
Manages individual camera instance with shared memory and inference
"""

from multiprocessing import shared_memory, Process, Value
import os
import threading
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


class RingBufferSharedMemory:
    """
    Ring buffer implementation for storing multiple frames in shared memory

    Memory Layout:
    - Header (64 bytes): Buffer management metadata
    - Frame Slots (5 slots): Each slot stores one complete frame

    Features:
    - Lock-free write (single writer - camera thread)
    - Multiple concurrent readers (API threads)
    - Atomic index updates using multiprocessing.Value
    - Fixed 5-frame circular buffer
    """

    BUFFER_SIZE = 5  # Number of frames to store
    HEADER_SIZE = 64  # Bytes for buffer header

    def __init__(self, serial_number: str, max_frame_size: int):
        """
        Initialize ring buffer shared memory

        Args:
            serial_number: Camera serial number (for shm name)
            max_frame_size: Maximum size of single frame in bytes (width × height × 3)
        """
        self.serial_number = serial_number
        self.shm_name = f"camera_{serial_number}"

        # Calculate slot size (frame + metadata overhead)
        # Each slot: frame_idx(8) + timestamp(8) + metadata_len(4) + metadata(~512) + frame_len(4) + frame_bytes
        self.slot_size = max_frame_size + 1024  # Extra 1KB for metadata

        # Total shared memory size
        self.shm_size = self.HEADER_SIZE + (self.slot_size * self.BUFFER_SIZE)

        # Cleanup existing shared memory
        try:
            existing_shm = shared_memory.SharedMemory(name=self.shm_name)
            existing_shm.close()
            existing_shm.unlink()
            logger.info(f"Cleaned up existing shared memory: {self.shm_name}")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Error cleaning up existing shared memory: {e}")

        # Create new shared memory
        self.shm = shared_memory.SharedMemory(
            name=self.shm_name,
            create=True,
            size=self.shm_size
        )

        # Initialize header
        self._init_header()

        logger.info(
            f"Ring buffer created: {self.shm_name}, "
            f"size={self.shm_size / 1024 / 1024:.2f}MB, "
            f"slots={self.BUFFER_SIZE}, "
            f"slot_size={self.slot_size / 1024:.2f}KB"
        )

    def _init_header(self):
        """Initialize buffer header with default values"""
        offset = 0

        # write_idx (4 bytes) - Index where next frame will be written
        struct.pack_into("<I", self.shm.buf, offset, 0)
        offset += 4

        # frame_count (4 bytes) - Total frames written (capped at BUFFER_SIZE)
        struct.pack_into("<I", self.shm.buf, offset, 0)
        offset += 4

        # buffer_size (4 bytes) - Maximum frames in buffer
        struct.pack_into("<I", self.shm.buf, offset, self.BUFFER_SIZE)
        offset += 4

        # frame_counter (8 bytes) - Monotonic counter for all frames written
        struct.pack_into("<Q", self.shm.buf, offset, 0)
        offset += 8

        # slot_size (4 bytes) - Size of each frame slot in bytes
        struct.pack_into("<I", self.shm.buf, offset, self.slot_size)
        offset += 4

        # reserved (40 bytes) - For future use
        # Total header: 64 bytes

    def write_frame(self, img_array: np.ndarray, metadata: Dict[str, Any]):
        """
        Write frame to next slot in ring buffer

        Args:
            img_array: Frame image array (BGR)
            metadata: Frame metadata dict
        """
        try:
            # Read current write index from header
            write_idx = struct.unpack_from("<I", self.shm.buf, 0)[0]
            frame_counter = struct.unpack_from("<Q", self.shm.buf, 12)[0]

            # Calculate slot offset
            slot_offset = self.HEADER_SIZE + (write_idx * self.slot_size)

            # Prepare frame data
            metadata_bytes = pickle.dumps(metadata)
            metadata_len = len(metadata_bytes)
            frame_bytes = img_array.tobytes()
            frame_len = len(frame_bytes)

            # Write to slot
            offset = slot_offset

            # frame_idx (8 bytes) - Monotonic frame counter
            struct.pack_into("<Q", self.shm.buf, offset, frame_counter)
            offset += 8

            # timestamp (8 bytes) - Unix timestamp in nanoseconds
            timestamp_ns = time.time_ns()
            struct.pack_into("<Q", self.shm.buf, offset, timestamp_ns)
            offset += 8

            # metadata_len (4 bytes)
            struct.pack_into("<I", self.shm.buf, offset, metadata_len)
            offset += 4

            # metadata bytes
            self.shm.buf[offset:offset+metadata_len] = metadata_bytes
            offset += metadata_len

            # frame_len (4 bytes)
            struct.pack_into("<I", self.shm.buf, offset, frame_len)
            offset += 4

            # frame bytes
            self.shm.buf[offset:offset+frame_len] = frame_bytes

            # Update header (atomic-like operation - single writer so safe)
            next_write_idx = (write_idx + 1) % self.BUFFER_SIZE
            struct.pack_into("<I", self.shm.buf, 0, next_write_idx)

            # Update frame count (capped at BUFFER_SIZE)
            frame_count = struct.unpack_from("<I", self.shm.buf, 4)[0]
            frame_count = min(frame_count + 1, self.BUFFER_SIZE)
            struct.pack_into("<I", self.shm.buf, 4, frame_count)

            # Update frame counter (monotonic)
            struct.pack_into("<Q", self.shm.buf, 12, frame_counter + 1)

        except Exception as e:
            logger.error(f"Error writing to ring buffer: {e}")

    def get_buffer_info(self) -> Dict[str, int]:
        """Get current buffer status"""
        return {
            'write_idx': struct.unpack_from("<I", self.shm.buf, 0)[0],
            'frame_count': struct.unpack_from("<I", self.shm.buf, 4)[0],
            'buffer_size': struct.unpack_from("<I", self.shm.buf, 8)[0],
            'frame_counter': struct.unpack_from("<Q", self.shm.buf, 12)[0]
        }

    def cleanup(self):
        """Cleanup shared memory"""
        if self.shm:
            try:
                self.shm.close()
                self.shm.unlink()
                logger.info(f"Ring buffer cleaned up: {self.shm_name}")
            except Exception as e:
                logger.error(f"Error cleaning up ring buffer: {e}")


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
        self.do_alarm_number = -1  # DO pin number for alarm (default: -1 = disabled)
        self.allow_late_reject = False  # Fire reject immediately when inference > delay_reject (carton/alarm systems)

        # Trigger config (per-camera)
        self.trigger_mode = "continuous"  # "continuous", "software_trigger", "hardware_trigger"
        self.trigger_selector = "FrameStart"  # "FrameStart", "ExposureStart", "FrameBurstStart"
        self.trigger_activation = "RisingEdge"  # "RisingEdge", "FallingEdge", "AnyEdge"
        self.di_number = 0  # Digital Input number (0-3) for software trigger
        self.trigger_source = "Line0"  # Camera Line input for hardware trigger (TODO)

        # HW TriggerDelay: camera tự đợi delay_trigger (µs-precision) sau
        # ExecuteSoftwareTrigger thay vì threading.Timer phía Python (vốn bị
        # GIL làm trễ 30-350ms khi inference nặng). configure_trigger_delay()
        # bật cờ này khi camera hỗ trợ.
        # _hw_delay_mode: "timer1" (Timer1 phần cứng: SoftwareSignal1 → Timer1
        #                  đếm delay → Timer1End trigger FrameStart) | None
        # Node TriggerDelay KHÔNG được dùng để arm — xem configure_trigger_delay().
        self.hw_trigger_delay_active = False
        self._hw_delay_mode: Optional[str] = None
        # Khoá capture: giữ từ lúc fire_software_trigger tới khi RetrieveResult
        # xong. Nếu xung DI kế đến khi capture trước chưa xong (2 chai sát hơn
        # delay_trigger) thì KHÔNG bắn pulse mới — pulse thừa sẽ sinh frame mồ
        # côi nằm lại grab queue, khiến mọi retrieve sau lấy nhầm frame của
        # chai TRƯỚC (off-by-one vĩnh viễn → "chụp lệch" dù Timer1 chính xác).
        self._hw_capture_lock = threading.Lock()
        self._hw_fire_ts = 0.0  # time.monotonic() của lần fire gần nhất

        # Recipe & templates
        self.recipe_id: Optional[str] = None
        self.recipe_name: Optional[str] = None
        self.templates: List[Dict[str, Any]] = []
        self.function_type: str = "OCR"  # Function type: OCR, Check_Type_Product, etc.
        self.expected_texts: Dict[int, Dict[int, str]] = {}  # Map template_idx -> {region_idx -> expected_text}
        self.matching_threshold: float = 0.85  # Template matching similarity threshold
        self.recognition_threshold: float = 0.5  # OCR recognition confidence threshold
        self.wrinkle_conf: float = 0.25  # Wrinkle segmentation model confidence threshold (recipe-level)
        self.wrinkle_show_when_pass: bool = True  # Draw wrinkle contour even when frame PASSes (debug)
        self.matching_conf: float = 0.20  # SuperPoint matching inlier_ratio threshold; below → skip verify
        self.mask_overlap_threshold: float = 0.6  # Wrinkle region inside 'mask' annotation by >= this fraction is excluded
        self.match_erosion_enabled: bool = False  # Apply horizontal erosion before SuperPoint matching
        self.match_erosion_kernel_w: int = 80  # Erosion kernel width in pixels
        self.match_erosion_kernel_h: int = 1   # Erosion kernel height in pixels (1=pure horizontal, 15=fills letter gaps)
        self.match_erosion_iterations: int = 1  # Number of erosion iterations

        # Product (bottle) edge detection method
        # "yolo_obb"     = YOLO OBB model (default, current)
        # "yolo_segment" = Image processing (Sobel + outer-anchored detection).
        #                  NOTE: tên giữ là "yolo_segment" cho consistency với UI,
        #                  KHÔNG dùng YOLO model thực sự. Có thể tắt OBB model
        #                  khi tất cả camera đều dùng yolo_segment.
        self.product_detection_method: str = "yolo_obb"
        # Only used when product_detection_method='yolo_segment':
        # "outer" = product box = bottle silhouette (default, matches YOLO convention)
        # "inner" = product box = sát label (tighter)
        self.product_box_wall_type: str = "outer"
        # Save PASS-frame images to disk (default ON). 200-most-recent ring buffer
        # per camera per recipe (see PassImagePruner). Used to re-test missed defects.
        self.save_pass_images: bool = True
        # Cap rotation method for /frames/rotate + Check_Color OCR sub-mode:
        # 'yolo_obb' = trained best_bottle_m engine | 'yolo_segment' = pure CV
        self.cap_rotation_method: str = "yolo_obb"
        # Cap crop method: detect cap circle and feed cap-only crop to SuperPoint
        # 'none' (default, off) | 'yolo_obb' | 'yolo_segment'
        self.cap_crop_method: str = "none"
        # Method for matching template ↔ target crop (transforms annotation
        # bboxes from template-coords to frame-coords): 'superpoint' (default,
        # TRT model) | 'shape_outline' (ECC on Sobel gradient — ~30ms, ideal
        # for cap-OCR after cap_rotation + cap_crop)
        self.crop_match_method: str = "superpoint"
        # When True (and function_type=Check_Color), pipeline tries BOTH
        # rotation candidates (angle, angle+180) and picks higher match
        # confidence. Fixes ambiguous flip detection in cap-OCR mode.
        self.dual_rotation_check: bool = False

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

            # Configure GigE buffer and network settings (must be done after Open and before StartGrabbing)
            # self._configure_gige_buffer_settings()

            # Setup ring buffer shared memory
            actual_width = self.camera.Width.GetValue()
            actual_height = self.camera.Height.GetValue()
            max_frame_size = actual_width * actual_height * 3  # BGR

            # Create ring buffer (5 frames)
            self.ring_buffer = RingBufferSharedMemory(
                serial_number=self.serial_number,
                max_frame_size=max_frame_size
            )

            logger.info(
                f"Camera connected: {self.model_name} (SN: {self.serial_number}), "
                f"resolution: {actual_width}x{actual_height}, "
                f"ring buffer: {self.ring_buffer.shm_size / 1024 / 1024:.2f}MB ({RingBufferSharedMemory.BUFFER_SIZE} frames)"
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

            if hasattr(self, 'ring_buffer') and self.ring_buffer:
                self.ring_buffer.cleanup()

            logger.info(f"Camera disconnected: {self.serial_number}")
            self._emit_event("camera_disconnected", {})

        except Exception as e:
            logger.error(f"Error disconnecting camera {self.serial_number}: {e}")

    def _attempt_reconnect(self) -> bool:
        """
        Attempt to reconnect camera when connection is lost

        Returns:
            True if reconnection successful, False otherwise
        """
        try:
            logger.warning(f"[{self.serial_number}] Attempting to reconnect camera...")

            # Close existing connection if any
            try:
                if self.camera:
                    if self.camera.IsGrabbing():
                        self.camera.StopGrabbing()
                    if self.camera.IsOpen():
                        self.camera.Close()
            except:
                pass  # Ignore errors during cleanup

            # Wait a bit before reconnecting
            time.sleep(0.5)

            # Attempt to reconnect
            if not self.connect():
                logger.error(f"[{self.serial_number}] Failed to reconnect camera")
                return False

            # Reconfigure camera mode if needed
            if self.mode != CameraMode.IDLE:
                logger.info(f"[{self.serial_number}] Restarting grabbing after reconnect...")
                try:
                    # MaxNumBuffer is already configured during connect() in _configure_gige_buffer_settings()
                    if self.mode == CameraMode.SOFTWARE_TRIGGER:
                        self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
                    else:
                        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

                    logger.info(f"[{self.serial_number}] ✅ Camera reconnected successfully")
                    self._emit_event("camera_reconnected", {
                        "serial_number": self.serial_number,
                        "mode": self.mode.value
                    })
                    return True
                except Exception as e:
                    logger.error(f"[{self.serial_number}] Failed to restart grabbing after reconnect: {e}")
                    return False

            return True

        except Exception as e:
            logger.error(f"[{self.serial_number}] Error during reconnect attempt: {e}")
            return False

    def _convert_to_bgr(self, img_array, pixel_format=None):
        """
        Convert image array to BGR format based on pixel format

        Args:
            img_array: Raw image array from camera
            pixel_format: Current pixel format (if None, auto-detect from shape)

        Returns:
            BGR image array
        """
        # Get actual pixel format from camera if possible
        actual_format = None
        try:
            if self.camera and hasattr(self.camera, 'PixelFormat'):
                actual_format = self.camera.PixelFormat.GetValue()
        except:
            pass

        current_format = pixel_format or actual_format or self.pixel_format

        # If already 3-channel (BGR/RGB), return as-is
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            return img_array

        # Detect Bayer pattern from actual pixel format
        if current_format and "Bayer" in current_format:
            # BayerRG8/BayerRG12 -> BGR using debayering
            if "RG" in current_format:
                return cv2.cvtColor(img_array, cv2.COLOR_BAYER_RG2RGB)
            elif "BG" in current_format:
                return cv2.cvtColor(img_array, cv2.COLOR_BAYER_BG2RGB)
            elif "GR" in current_format:
                return cv2.cvtColor(img_array, cv2.COLOR_BAYER_GR2RGB)
            elif "GB" in current_format:
                return cv2.cvtColor(img_array, cv2.COLOR_BAYER_GB2RGB)
        # Grayscale (Mono8, Mono12) -> BGR
        if len(img_array.shape) == 2:
            return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        return img_array

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
                try:
                    self.camera.PixelFormat.SetValue(self.pixel_format)
                    actual_pixel_format = self.camera.PixelFormat.GetValue()
                    logger.info(f"[{self.serial_number}] PixelFormat set: {actual_pixel_format}")
                except Exception as pixel_error:
                    # USB format -> GigE format mapping
                    logger.warning(f"[{self.serial_number}] Failed to set {self.pixel_format}: {pixel_error}")

                    # Mapping: USB/UI format -> GigE equivalent format
                    format_mapping = {
                        "BGR8": "BayerRG8",
                        "RGB8": "BayerRG8",
                        "Mono8": "Mono8",
                        "Mono12": "Mono12",
                        "YUV422": "YUV422Packed",
                        "BayerRG12": "BayerRG12Packed"
                    }

                    fallback_format = format_mapping.get(self.pixel_format)

                    applied = False
                    if fallback_format:
                        try:
                            self.camera.PixelFormat.SetValue(fallback_format)
                            actual_pixel_format = self.camera.PixelFormat.GetValue()
                            logger.info(f"[{self.serial_number}] PixelFormat mapped: {self.pixel_format} -> {actual_pixel_format}")
                            applied = True
                        except Exception as e:
                            logger.error(f"[{self.serial_number}] Failed to set mapped format {fallback_format}: {e}")
                    else:
                        logger.error(f"[{self.serial_number}] No GigE mapping found for {self.pixel_format}")

                    if not applied:
                        # Model không hỗ trợ format của recipe (vd recipe BGR8 trên
                        # camera mono). Giữ format camera đang chạy và đồng bộ lại
                        # self.pixel_format để _convert_to_bgr() không debayer nhầm.
                        try:
                            actual_pixel_format = self.camera.PixelFormat.GetValue()
                            logger.warning(
                                f"[{self.serial_number}] Giữ PixelFormat hiện tại "
                                f"'{actual_pixel_format}' (model {self.model_name} không "
                                f"hỗ trợ '{self.pixel_format}')"
                            )
                            self.pixel_format = actual_pixel_format
                        except Exception as e:
                            logger.error(f"[{self.serial_number}] Không đọc được PixelFormat: {e}")

            # Exposure / Gain (can be changed during grabbing) - node name and
            # valid range đều khác nhau giữa các model → resolve + clamp
            self._set_numeric_node(
                ("ExposureTime", "ExposureTimeAbs"), self.exposure_time, "Exposure"
            )
            self._set_numeric_node(("Gain", "GainRaw"), self.gain, "Gain")

        except Exception as e:
            logger.error(f"Error applying settings: {e}")

    def _configure_gige_buffer_settings(self):
        """
        Configure GigE-specific buffer and network settings for reliable image acquisition

        This method addresses the buffer underrun issues by:
        1. Increasing buffer pool size
        2. Optimizing GigE packet size
        3. Configuring inter-packet delay
        4. Setting bandwidth reserve
        """
        if not self.camera or not self.camera.IsOpen():
            return

        try:
            # Check if this is a GigE camera by checking for GigE-specific parameters
            is_gige = False
            try:
                if hasattr(self.camera, 'GevSCPSPacketSize'):
                    is_gige = True
            except:
                pass

            if not is_gige:
                logger.info(f"[{self.serial_number}] Not a GigE camera, skipping GigE-specific configuration")
                return

            logger.info(f"[{self.serial_number}] Configuring GigE buffer and network settings...")

            # 1. Increase buffer count (default is 5-10, increase to 30 for GigE)
            try:
                if hasattr(self.camera, 'MaxNumBuffer'):
                    # Note: MaxNumBuffer must be set BEFORE StartGrabbing
                    self.camera.MaxNumBuffer.SetValue(30)
                    logger.info(f"[{self.serial_number}] ✓ MaxNumBuffer set to 30")
            except Exception as e:
                logger.warning(f"[{self.serial_number}] Could not set MaxNumBuffer: {e}")

            # 2. Optimize GigE packet size
            # 8192 bytes for Jumbo Frames (MTU 9000), 1500 for standard Ethernet
            try:
                # Try to set to 8192 first (Jumbo Frames)
                try:
                    max_packet_size = self.camera.GevSCPSPacketSize.GetMax()
                    packet_size = min(8192, max_packet_size)
                    self.camera.GevSCPSPacketSize.SetValue(packet_size)
                    logger.info(f"[{self.serial_number}] ✓ GevSCPSPacketSize set to {packet_size} bytes")
                    if packet_size < 8192:
                        logger.warning(
                            f"[{self.serial_number}] Packet size limited to {packet_size}. "
                            f"Enable Jumbo Frames (MTU 9000) for better performance"
                        )
                except Exception as e:
                    # Fallback to 1500 (standard Ethernet MTU)
                    self.camera.GevSCPSPacketSize.SetValue(1500)
                    logger.info(f"[{self.serial_number}] ✓ GevSCPSPacketSize set to 1500 bytes (standard MTU)")
            except Exception as e:
                logger.warning(f"[{self.serial_number}] Could not set GevSCPSPacketSize: {e}")

            # 3. Configure inter-packet delay
            # 0 = maximum speed, increase if network congestion occurs
            try:
                self.camera.GevSCPD.SetValue(0)
                logger.info(f"[{self.serial_number}] ✓ GevSCPD (inter-packet delay) set to 0 μs")
            except Exception as e:
                logger.warning(f"[{self.serial_number}] Could not set GevSCPD: {e}")

            # 4. Enable bandwidth reserve (helps with burst traffic)
            try:
                if hasattr(self.camera, 'GevSCBWR'):
                    self.camera.GevSCBWR.SetValue(10)  # 10% reserve
                    logger.info(f"[{self.serial_number}] ✓ GevSCBWR (bandwidth reserve) set to 10%")
            except Exception as e:
                logger.warning(f"[{self.serial_number}] Could not set GevSCBWR: {e}")

            # 5. Enable resend mechanism for lost packets (if available)
            try:
                if hasattr(self.camera, 'GevSCPSFireTestPacket'):
                    self.camera.GevSCPSFireTestPacket.SetValue(False)
                    logger.info(f"[{self.serial_number}] ✓ GevSCPSFireTestPacket disabled")
            except Exception as e:
                logger.debug(f"[{self.serial_number}] GevSCPSFireTestPacket not available: {e}")

            logger.info(f"[{self.serial_number}] ✅ GigE buffer and network settings configured successfully")

        except Exception as e:
            logger.error(f"[{self.serial_number}] Error configuring GigE buffer settings: {e}")
            import traceback
            traceback.print_exc()

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
            self._set_numeric_node(
                ("ExposureTime", "ExposureTimeAbs"), exposure_time, "Exposure time"
            )

    def set_gain(self, gain: float):
        """
        Set camera gain

        Args:
            gain: Gain value
        """
        self.gain = gain

        # Apply to camera if connected
        if self.camera and self.camera.IsOpen():
            self._set_numeric_node(("Gain", "GainRaw"), gain, "Gain")

    def update_settings(self, settings: Dict[str, Any]):
        """Update camera settings from recipe.

        Không bao giờ ném exception: một node phần cứng thiếu/không ghi được chỉ
        làm mất TÍNH NĂNG đó, không được làm hỏng load_recipe() — nếu hỏng thì
        camera bị loại khỏi capture group và cả nhóm chạy thiếu camera.
        """
        try:
            self._update_settings_impl(settings)
        except Exception as e:
            logger.error(
                f"[{self.serial_number}] update_settings gặp lỗi phần cứng ({e}) "
                f"— tiếp tục với cấu hình hiện có (model: {self.model_name})"
            )

    def _update_settings_impl(self, settings: Dict[str, Any]):
        if "exposure_time" in settings:
            self.set_exposure_time(settings["exposure_time"])
        if "gain" in settings:
            self.set_gain(settings["gain"])
        if "pixel_format" in settings:
            self.pixel_format = settings["pixel_format"]
        if "delay_trigger" in settings:
            self.delay_trigger = settings["delay_trigger"]
            # HW TriggerDelay bám theo delay mới (node writable cả khi đang grab;
            # nếu fail → tự fallback timer legacy bên trong)
            if self.camera and self.camera.IsOpen():
                self.configure_trigger_delay()
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

        # Apply pixel format if needed - requires reconnect if grabbing
        if self.camera and "pixel_format" in settings:
            if self.camera.IsGrabbing():
                logger.warning(f"[{self.serial_number}] Pixel format change requires reconnect (camera is grabbing)")
                # Stop grabbing temporarily
                old_mode = self.mode
                self.camera.StopGrabbing()

                # Apply new pixel format
                self._apply_settings(apply_pixel_format=True)

                # Restart grabbing if was in continuous mode
                if old_mode == CameraMode.CONTINUOUS:
                    self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                    logger.info(f"[{self.serial_number}] Restarted grabbing after pixel format change")
            else:
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

            # Configure software trigger.
            # TriggerSelector nhận enum khác nhau theo model (FrameStart /
            # AcquisitionStart) — chọn sai chỉ nên rơi về mặc định của camera,
            # không được làm hỏng cả recipe.
            try:
                self.camera.TriggerSelector.SetValue(self.trigger_selector)
            except Exception as e:
                logger.warning(
                    f"[{self.serial_number}] TriggerSelector='{self.trigger_selector}' "
                    f"không hợp lệ ({e}) — dùng selector mặc định của model "
                    f"{self.model_name}"
                )
                try:
                    self.trigger_selector = self.camera.TriggerSelector.GetValue()
                except Exception:
                    pass

            # Hai node này là bắt buộc: không có software trigger thì camera
            # không thể tham gia capture group → để exception thoát ra ngoài.
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

            # HW TriggerDelay (đặt trước khi StartGrabbing — node chắc chắn writable)
            self.configure_trigger_delay()

            # Resume grabbing with OneByOne strategy
            if was_grabbing or self.mode == CameraMode.SOFTWARE_TRIGGER:
                # MaxNumBuffer is already configured in _configure_gige_buffer_settings()
                self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
                logger.info(f"[{self.serial_number}] Camera grabbing started (OneByOne)")

            return True

        except Exception as e:
            logger.error(f"[{self.serial_number}] Error configuring software trigger: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_node(self, *node_names, writable: bool = True):
        """Tìm node đầu tiên tồn tại (và writable nếu yêu cầu) trong danh sách tên.

        KHÔNG BAO GIỜ raise: pypylon ném LogicalErrorException('Node not existing')
        ngay tại bước truy cập thuộc tính chứ không trả None như getattr thường,
        nên mọi lookup phải nằm trong try. Đây là điểm khiến camera model khác
        (vd acA1600 chỉ có 'TriggerDelayAbs') làm hỏng cả quá trình load recipe.
        """
        if not self.camera:
            return None
        for node_name in node_names:
            try:
                node = getattr(self.camera, node_name)
            except Exception:
                continue  # node không tồn tại trên model này
            if node is None:
                continue
            if not writable:
                return node
            try:
                from pypylon import genicam
                if node.GetAccessMode() in (genicam.RW, genicam.WO):
                    return node
            except Exception:
                continue
        return None

    def _set_numeric_node(self, node_names, value: float, label: str) -> bool:
        """Ghi giá trị số vào node đầu tiên khả dụng, tự clamp về [Min, Max].

        Mỗi model có dải hợp lệ khác nhau (vd GainRaw của acA1600-60gm max=3
        trong khi Gain của a2A1920 tính theo dB tới 24) — clamp + cảnh báo thay
        vì để OutOfRangeException làm hỏng recipe.

        Returns: True nếu ghi được (kể cả khi đã clamp).
        """
        node = self._get_node(*node_names)
        if node is None:
            logger.warning(
                f"[{self.serial_number}] {label}: không có node "
                f"{'/'.join(node_names)} khả dụng (model: {self.model_name}) — bỏ qua"
            )
            return False

        try:
            target = float(value)
            try:
                node_min, node_max = float(node.GetMin()), float(node.GetMax())
                clamped = min(max(target, node_min), node_max)
            except Exception:
                node_min = node_max = None
                clamped = target

            # Node kiểu Integer chỉ nhận int (và phải đúng bội của Inc nếu có)
            is_int = "Integer" in type(node).__name__
            if is_int:
                clamped = int(clamped)
                try:
                    inc = int(node.GetInc())
                    if inc > 1 and node_min is not None:
                        clamped = int(node_min) + ((clamped - int(node_min)) // inc) * inc
                except Exception:
                    pass

            node.SetValue(clamped)

            if node_max is not None and abs(clamped - target) > 1e-6:
                logger.warning(
                    f"[{self.serial_number}] {label}={target} ngoài dải "
                    f"[{node_min}, {node_max}] của model {self.model_name} "
                    f"— đã clamp về {clamped}"
                )
            else:
                logger.info(f"[{self.serial_number}] {label} set to {clamped}")
            return True
        except Exception as e:
            logger.error(f"[{self.serial_number}] Failed to set {label}: {e}")
            return False

    def _get_trigger_delay_node(self):
        """Node TriggerDelay theo model: ace2/USB = 'TriggerDelay' (µs, float),
        ace classic GigE = 'TriggerDelayAbs' (µs, float). None nếu không có."""
        return self._get_node("TriggerDelay", "TriggerDelayAbs")

    @staticmethod
    def _genicam_unit_per_ms(unit: str) -> float:
        """Hệ số quy đổi 1ms → đơn vị của node (SFNC mặc định µs)."""
        u = (unit or "").strip().lower()
        if u in ("us", "µs", "usec", ""):
            return 1000.0
        if u in ("ms", "msec"):
            return 1.0
        if u == "s":
            return 0.001
        if u in ("ns", "nsec"):
            return 1_000_000.0
        return 1000.0

    def _disarm_hw_delay(self, delay_node=None):
        """Về trạng thái legacy an toàn: TriggerDelay=0, TriggerSource=Software."""
        self.hw_trigger_delay_active = False
        self._hw_delay_mode = None
        try:
            if delay_node is not None:
                delay_node.SetValue(0.0)
        except Exception:
            pass
        try:
            # Quan trọng khi trước đó armed timer1 (TriggerSource=Timer1End):
            # phải trả về Software, nếu không ExecuteSoftwareTrigger thành vô hiệu.
            self.camera.TriggerSelector.SetValue(self.trigger_selector)
            self.camera.TriggerSource.SetValue("Software")
        except Exception as e:
            logger.error(
                f"[{self.serial_number}] KHÔNG khôi phục được TriggerSource=Software: {e} "
                f"— capture có thể kẹt, cần reconfigure camera!"
            )

    def _arm_timer1_scheme(self) -> bool:
        """
        Fallback khi TriggerDelay max quá nhỏ (ace2: 10ms): dùng khối Timer1
        phần cứng — SoftwareSignal1 → Timer1 đếm delay_trigger → Timer1End
        trigger FrameStart. Hiệu quả y hệt TriggerDelay: camera tự đợi bằng
        phần cứng, fire chỉ là 1 control write tại DI edge.
        """
        cam = self.camera
        try:
            cam.TimerSelector.SetValue("Timer1")
            # ace2/USB = 'TimerDuration', ace classic GigE = 'TimerDurationAbs'
            dur = self._get_node("TimerDuration", "TimerDurationAbs")
            if dur is None:
                logger.info(
                    f"[{self.serial_number}] Không có node TimerDuration "
                    f"(model {self.model_name}) — dùng timer (legacy)"
                )
                return False
            unit = ""
            try:
                unit = (dur.GetUnit() or "").strip()
            except Exception:
                pass
            per_ms = self._genicam_unit_per_ms(unit)
            dmax = float(dur.GetMax())
            target = float(self.delay_trigger) * per_ms
            logger.info(
                f"[{self.serial_number}] Timer1Duration node: unit='{unit or 'µs?'}' "
                f"max={dmax} (≈{dmax/per_ms:.0f}ms) target={target} ({self.delay_trigger}ms)"
            )
            if target > dmax:
                logger.warning(
                    f"[{self.serial_number}] delay_trigger vượt Timer1 max≈{dmax/per_ms:.0f}ms "
                    f"— dùng timer (legacy)"
                )
                return False
            dur.SetValue(target)
            cam.TimerTriggerSource.SetValue("SoftwareSignal1")
            try:
                cam.TimerTriggerActivation.SetValue("RisingEdge")
            except Exception:
                pass  # một số model không có node này — mặc định RisingEdge
            cam.TriggerSelector.SetValue(self.trigger_selector)
            cam.TriggerSource.SetValue("Timer1End")
            cam.SoftwareSignalSelector.SetValue("SoftwareSignal1")
            self._hw_delay_mode = "timer1"
            self.hw_trigger_delay_active = True
            logger.info(
                f"[{self.serial_number}] ✅ HW delay armed qua Timer1: "
                f"{self.delay_trigger}ms in-camera (SoftwareSignal1→Timer1→FrameStart) "
                f"— hết trễ GIL"
            )
            return True
        except Exception as e:
            logger.warning(
                f"[{self.serial_number}] Không arm được Timer1 scheme ({e}) — dùng timer (legacy)"
            )
            return False

    def configure_trigger_delay(self) -> bool:
        """
        Đưa delay_trigger (ms) xuống camera để camera tự đợi bằng phần cứng.
        CHỈ dùng Timer1 scheme.

        Node TriggerDelay/TriggerDelayAbs KHÔNG được dùng để arm: trên ace
        classic (acA1600-60gm) node nhận giá trị và báo thành công nhưng camera
        LỜ HOÀN TOÀN delay khi TriggerSource=Software — đo log 2026-07-20 cho
        thấy latency ~160ms không đổi dù đặt 200/320/360/370ms, trong khi ace2
        đi đường Timer1 bám đúng delay. Node "ghi được" không đồng nghĩa "có tác
        dụng", nên nó chỉ được ZERO HOÁ để không cộng thêm delay ẩn lên Timer1
        hoặc lên timer legacy.

        Khi thành công (hw_trigger_delay_active=True), trigger_handler sẽ
        fire_software_trigger() NGAY tại DI edge — loại threading.Timer khỏi
        đường timing (hết trễ GIL).

        Kill-switch: env OCR_HW_TRIGGER_DELAY=0 → luôn dùng timer cũ.
        Multi-template: disarm (delay phần cứng áp lên MỌI trigger → frame 2+ lệch).
        Fallback: mọi lỗi → disarm về trạng thái legacy nguyên vẹn.
        """
        if not self.camera or not self.camera.IsOpen():
            self.hw_trigger_delay_active = False
            self._hw_delay_mode = None
            return False

        node = self._get_trigger_delay_node()

        if os.environ.get("OCR_HW_TRIGGER_DELAY", "1") != "1":
            logger.info(f"[{self.serial_number}] HW TriggerDelay disabled by env")
            self._disarm_hw_delay(node)
            return False

        if len(self.templates) > 1:
            logger.info(
                f"[{self.serial_number}] {len(self.templates)} templates — HW delay "
                f"không áp dụng (multi-template), dùng timer (legacy)"
            )
            self._disarm_hw_delay(node)
            return False

        # Zero hoá node TriggerDelay: không tin nó để arm, nhưng nếu nó còn giữ
        # giá trị cũ trên model CÓ áp dụng thật thì delay sẽ cộng dồn lên Timer1
        # / timer legacy → chụp trễ gấp đôi.
        if node is not None:
            try:
                node.SetValue(0.0)
            except Exception as e:
                logger.warning(
                    f"[{self.serial_number}] Không zero được TriggerDelay node: {e}"
                )

        # Timer1 phần cứng — đường HW delay DUY NHẤT được tin
        if self._arm_timer1_scheme():
            return True

        logger.info(
            f"[{self.serial_number}] Không arm được Timer1 (model {self.model_name}) "
            f"— dùng timer legacy cho delay {self.delay_trigger}ms"
        )
        self._disarm_hw_delay(node)
        return False

    def hw_delay_capture_ready(self) -> bool:
        """HW-delay path chỉ dùng được khi: đã arm + đúng 1 template.
        Multi-template bị loại vì TriggerDelay áp lên MỌI ExecuteSoftwareTrigger
        → frame 2+ sẽ lệch thêm delay_trigger."""
        return (
            self.hw_trigger_delay_active
            and len(self.templates) == 1
            and self.camera is not None
            and self.camera.IsOpen()
            and self.camera.IsGrabbing()
        )

    def _release_hw_capture_lock(self) -> None:
        """Nhả khoá capture (an toàn khi đã nhả rồi — threading.Lock cho phép
        release từ thread khác thread acquire)."""
        try:
            self._hw_capture_lock.release()
        except RuntimeError:
            pass

    def _drain_stale_frames(self) -> int:
        """Vét frame tồn trong grab queue TRƯỚC khi bắn trigger mới.

        Frame tồn sinh ra khi một trigger đã bắn nhưng retrieve lỗi/bỏ lượt
        (vd 2 xung DI sát nhau → 'There is already a thread waiting for a
        result'). Không vét thì RetrieveResult kế tiếp trả về NGAY frame cũ
        (chụp theo edge của chai trước) → mọi job sau dùng ảnh sai thời điểm.
        Gọi khi đang giữ _hw_capture_lock (không có retrieve nào đang chờ)."""
        drained = 0
        try:
            while True:
                res = self.camera.RetrieveResult(0, pylon.TimeoutHandling_Return)
                if res is None:
                    break
                valid = res.IsValid()
                res.Release()
                if not valid:
                    break
                drained += 1
        except Exception as e:
            logger.debug(f"[{self.serial_number}] drain queue dừng: {e}")
        if drained:
            logger.warning(
                f"[{self.serial_number}] Đã vét {drained} frame tồn khỏi grab "
                f"queue trước trigger mới (tự sửa off-by-one/ảnh lệch)"
            )
        return drained

    def fire_software_trigger(self) -> bool:
        """Bắn trigger ngay (~1-3ms control write, nhả GIL). Gọi tại DI edge.
        Mode timer1: bắn SoftwareSignal1 → Timer1 đếm delay → FrameStart.
        Mode khác: ExecuteSoftwareTrigger (exposure sau TriggerDelay nếu armed).

        Giữ _hw_capture_lock từ đây tới khi RetrieveResult xong (nhả trong
        retrieve_hw_delayed_frame / execute_software_trigger_immediate).
        Trả False nếu capture trước chưa xong — bỏ lượt thay vì bắn pulse
        thừa làm hỏng đồng bộ frame↔chai của mọi lần chụp sau."""
        if not self._hw_capture_lock.acquire(blocking=False):
            age = time.monotonic() - self._hw_fire_ts
            if age < (self.delay_trigger + 5000.0) / 1000.0:
                logger.warning(
                    f"[{self.serial_number}] Bỏ lượt trigger: capture trước chưa "
                    f"xong (fire cách đây {age*1000:.0f}ms — xung DI sát hơn "
                    f"delay_trigger {self.delay_trigger:.0f}ms)"
                )
                return False
            # Retrieve trước chết bất thường (quá delay + 5s) — cướp lại khoá
            logger.error(
                f"[{self.serial_number}] Khoá capture kẹt {age:.1f}s — force release"
            )
            self._release_hw_capture_lock()
            if not self._hw_capture_lock.acquire(blocking=False):
                return False
        try:
            self._drain_stale_frames()
            if self._hw_delay_mode == "timer1":
                self.camera.SoftwareSignalPulse.Execute()
            else:
                self.camera.ExecuteSoftwareTrigger()
            self._hw_fire_ts = time.monotonic()
            return True
        except Exception as e:
            logger.error(f"[{self.serial_number}] fire_software_trigger failed: {e}")
            self._release_hw_capture_lock()
            return False

    def retrieve_hw_delayed_frame(self) -> Dict[str, Any]:
        """
        Lấy frame đã được trigger bằng fire_software_trigger() (HW-delay path,
        single-template). RetrieveResult chỉ chờ (C call, nhả GIL) — không có
        yêu cầu đúng-giờ nào phía Python nữa.

        Return shape khớp execute_software_trigger_immediate().
        """
        try:
            timeout_ms = 2000 + int(self.delay_trigger)
            try:
                grab_result = self.camera.RetrieveResult(
                    timeout_ms, pylon.TimeoutHandling_ThrowException
                )
            finally:
                # Capture sequence kết thúc tại đây (thành công hay lỗi) —
                # nhả khoá để xung DI kế được bắn.
                self._release_hw_capture_lock()
            if not grab_result or not grab_result.GrabSucceeded():
                if grab_result:
                    grab_result.Release()
                return {'success': False, 'error': 'Failed to grab HW-delayed frame'}

            img_array = self._convert_to_bgr(grab_result.Array)
            metadata = {
                "timestamp": time.time(),
                "mode": self.mode.value,
                "shape": img_array.shape,
                "dtype": str(img_array.dtype),
                "frame_idx": 0,
                "trigger_event": True,
                "template_name": self.templates[0].get('name', 'Template 1'),
            }
            self._write_frame_to_shm(img_array, metadata)
            self.captured_frames = [img_array.copy()]
            grab_result.Release()

            logger.info(
                f"[{self.serial_number}] ✅ Captured 1 frame (HW TriggerDelay "
                f"{self.delay_trigger}ms in-camera)"
            )
            return {
                'success': True,
                'frames': self.captured_frames,
                'frame_count': 1,
            }
        except Exception as e:
            logger.error(f"[{self.serial_number}] Error retrieving HW-delayed frame: {e}")
            return {'success': False, 'error': str(e)}

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
        # Check camera connection and grabbing status
        if not self.camera:
            return {'success': False, 'error': 'Camera not initialized'}

        if not self.camera.IsOpen():
            logger.error(f"[{self.serial_number}] Camera control channel not open")
            return {'success': False, 'error': 'Camera control channel not open'}

        if not self.camera.IsGrabbing():
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

                # Check camera connection before triggering
                if not self.camera.IsOpen():
                    logger.warning(
                        f"[{self.serial_number}] Camera control channel closed during capture "
                        f"(frame {idx+1}/{len(self.templates)}), attempting reconnect..."
                    )

                    # Attempt to reconnect once
                    if not self._attempt_reconnect():
                        return {
                            'success': False,
                            'error': f'Camera connection lost at frame {idx}, reconnect failed'
                        }

                    logger.info(f"[{self.serial_number}] Reconnect successful, resuming capture...")

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

                # Convert to BGR (handles Mono8, BayerRG8, etc.)
                img_array = self._convert_to_bgr(img_array)

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
        # Check camera connection and grabbing status
        if not self.camera:
            return {'success': False, 'error': 'Camera not initialized'}

        if not self.camera.IsOpen():
            logger.error(f"[{self.serial_number}] Camera control channel not open")
            return {'success': False, 'error': 'Camera control channel not open'}

        if not self.camera.IsGrabbing():
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

                # Check camera connection before triggering
                if not self.camera.IsOpen():
                    logger.warning(
                        f"[{self.serial_number}] Camera control channel closed during capture "
                        f"(frame {capture_idx+1}/{num_templates}), attempting reconnect..."
                    )

                    # Attempt to reconnect once
                    if not self._attempt_reconnect():
                        return {
                            'success': False,
                            'error': f'Camera connection lost at frame {capture_idx}, reconnect failed'
                        }

                    logger.info(f"[{self.serial_number}] Reconnect successful, resuming capture...")

                # Execute software trigger (fire_software_trigger tự chọn
                # SoftwareSignalPulse khi armed timer1 — ExecuteSoftwareTrigger
                # sẽ vô hiệu vì TriggerSource=Timer1End)
                if not self.fire_software_trigger():
                    return {
                        'success': False,
                        'error': f'Software trigger failed at frame {capture_idx}'
                    }

                # Wait and retrieve frame (+delay_trigger khi HW delay armed —
                # exposure xảy ra sau khi camera tự đếm xong delay)
                retrieve_timeout = 2000 + (
                    int(self.delay_trigger) if self.hw_trigger_delay_active else 0
                )
                try:
                    grab_result = self.camera.RetrieveResult(
                        retrieve_timeout,
                        pylon.TimeoutHandling_ThrowException
                    )
                finally:
                    # fire_software_trigger giữ khoá capture — nhả ngay khi
                    # RetrieveResult kết thúc (kể cả timeout/exception) để
                    # không chặn xung DI kế / vòng lặp template kế.
                    self._release_hw_capture_lock()

                if not grab_result or not grab_result.GrabSucceeded():
                    if grab_result:
                        grab_result.Release()
                    return {
                        'success': False,
                        'error': f'Failed to grab frame {capture_idx}'
                    }

                img_array = grab_result.Array

                # Convert to BGR (handles Mono8, BayerRG8, etc.)
                img_array = self._convert_to_bgr(img_array)

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

            self.recipe_id = recipe_data.get("_id") or recipe_data.get("id")
            self.recipe_name = recipe_data.get("name", "Unknown")

            # Load reject config from recipe (recipe level, not camera level)
            self.delay_reject = recipe_data.get("delay_reject", 4000)  # ms, default 4s
            self.do_reject_number = recipe_data.get("do_reject_number", 2)  # DO2 default
            self.do_alarm_number = recipe_data.get("do_alarm_number", -1)  # -1 = disabled
            self.allow_late_reject = bool(recipe_data.get("allow_late_reject", False))

            # Load model thresholds from recipe
            model_thresholds = recipe_data.get("model_thresholds", {})
            self.matching_threshold = model_thresholds.get("matching_threshold", 0.85)
            self.recognition_threshold = model_thresholds.get("recognition_threshold", 0.5)
            self.ocr_model_type = recipe_data.get("ocr_model_type")
            # Recorded for logs/debug only — the OCR backend is process-wide
            # (InferenceHandler), not per-camera, so nothing here drives it.
            self.ocr_project_id = recipe_data.get("ocr_project_id")
            self.ocr_model_id = recipe_data.get("ocr_model_id")
            self.ml_project_id = recipe_data.get("ml_project_id")
            self.ml_model_id = recipe_data.get("ml_model_id")
            self.defect_model = recipe_data.get("defect_model") or "arcface"
            self.classifier_backend = recipe_data.get("classifier_backend") or "embedding"
            # CV pipeline variant when classifier_backend='embedding': 'legacy' | 'v4' | 'shape_v7'
            self.cv_method = (recipe_data.get("cv_method") or "v4").lower()
            # Template bank — adaptive multi-template per (recipe, camera, ann_idx)
            tbe = recipe_data.get("template_bank_enabled")
            self.template_bank_enabled = bool(tbe) if tbe is not None else False
            tbs = recipe_data.get("template_bank_size")
            self.template_bank_size = int(tbs) if tbs is not None else 10
            # Char denoise — largest-CC filter before centroid alignment (robust IoU)
            cde = recipe_data.get("char_denoise_enabled")
            self.char_denoise_enabled = bool(cde) if cde is not None else False
            # Version key: changes when recipe is re-saved → bank dynamic gets wiped on next load.
            # Use recipe.updated_at if present; else fall back to recipe.id (stable, no wipe).
            self.template_version_key = str(
                recipe_data.get("updated_at")
                or recipe_data.get("_id")
                or recipe_data.get("id")
                or ""
            )
            wc = recipe_data.get("wrinkle_conf")
            self.wrinkle_conf = float(wc) if wc is not None else 0.25
            wsp = recipe_data.get("wrinkle_show_when_pass")
            self.wrinkle_show_when_pass = bool(wsp) if wsp is not None else True
            mc = recipe_data.get("matching_conf")
            self.matching_conf = float(mc) if mc is not None else 0.20
            mot = recipe_data.get("mask_overlap_threshold")
            self.mask_overlap_threshold = float(mot) if mot is not None else 0.6
            me = recipe_data.get("match_erosion_enabled")
            self.match_erosion_enabled = bool(me) if me is not None else False
            mew = recipe_data.get("match_erosion_kernel_w")
            self.match_erosion_kernel_w = int(mew) if mew is not None else 80
            meh = recipe_data.get("match_erosion_kernel_h")
            self.match_erosion_kernel_h = int(meh) if meh is not None else 1
            mei = recipe_data.get("match_erosion_iterations")
            self.match_erosion_iterations = int(mei) if mei is not None else 1
            # Product detection method: "yolo_obb" (default) | "yolo_segment" (image-proc)
            pdm = recipe_data.get("product_detection_method")
            self.product_detection_method = pdm if pdm in ("yolo_obb", "yolo_segment") else "yolo_obb"
            # Product box wall type (chỉ áp dụng cho yolo_segment): "outer" | "inner"
            pwt = recipe_data.get("product_box_wall_type")
            self.product_box_wall_type = pwt if pwt in ("outer", "inner") else "outer"
            # Save PASS images to disk (default ON unless explicitly disabled)
            spi = recipe_data.get("save_pass_images")
            self.save_pass_images = True if spi is None else bool(spi)
            # Cap rotation method: 'yolo_obb' (default, TRT engine) | 'yolo_segment' (pure CV)
            crm = recipe_data.get("cap_rotation_method")
            self.cap_rotation_method = crm if crm in ("yolo_obb", "yolo_segment") else "yolo_obb"
            ccm = recipe_data.get("cap_crop_method")
            self.cap_crop_method = ccm if ccm in ("none", "yolo_obb", "yolo_segment") else "none"
            # Crop match method: 'superpoint' (default, TRT) | 'shape_outline' (ECC)
            cmm = recipe_data.get("crop_match_method")
            self.crop_match_method = cmm if cmm in ("superpoint", "shape_outline") else "superpoint"
            # Dual rotation check (only Check_Color)
            self.dual_rotation_check = bool(recipe_data.get("dual_rotation_check", False))
            logger.info(
                f"[{self.serial_number}] Loaded thresholds: "
                f"matching={self.matching_threshold}, recognition={self.recognition_threshold}"
            )
            logger.info(
                f"[{self.serial_number}] OCR model: {self.ocr_model_type or 'default'}, "
                f"ML project: {self.ml_project_id or 'none'}, ML model: {self.ml_model_id or 'none'}"
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
            alarm_status = f"DO{self.do_alarm_number}" if self.do_alarm_number >= 0 else "DISABLED"
            logger.info(f"  - do_alarm_number: {alarm_status} (recipe level)")
            logger.info(f"  - allow_late_reject: {self.allow_late_reject} (recipe level)")

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
                            if ann.get('type') in ['text', 'datecode']:
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

            # Re-arm HW delay theo số template mới: 1 template → arm;
            # multi-template → disarm (delay phần cứng áp lên mọi trigger).
            if self.camera and self.camera.IsOpen():
                self.configure_trigger_delay()

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
            home = os.environ.get('HOME')

            base_dir = Path(f"{home}/Source/ocr_datecode/backend/uploads/inference_results")
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
        """Write frame and metadata to ring buffer shared memory"""
        if not hasattr(self, 'ring_buffer') or not self.ring_buffer:
            return

        try:
            # Write to ring buffer (automatically handles slot rotation)
            self.ring_buffer.write_frame(img_array, metadata)

        except Exception as e:
            logger.error(f"Error writing to ring buffer: {e}")

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
            # Note: MaxNumBuffer is already configured in _configure_gige_buffer_settings()
            # which is called during connect(). No need to configure again here.

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

                            # Convert to BGR (handles Mono8, BayerRG8, etc.)
                            img_array = self._convert_to_bgr(img_array)

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
