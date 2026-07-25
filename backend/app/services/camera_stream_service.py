"""
Camera Stream Service
Reads frames from shared memory and streams via SocketIO
"""

import asyncio
import logging
import base64
import time
import cv2
import numpy as np
from multiprocessing import shared_memory
import pickle
import struct
from typing import Dict, Any, Optional, Set

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Enable debug logging


class CameraStreamService:
    """
    Service to stream camera frames from shared memory

    Features:
    - Read frames from /dev/shm/camera_{serial}
    - Downscale and encode as JPEG
    - Stream via SocketIO to subscribed clients

    Same unsynchronized-writer caveats as SharedMemoryService (see its class
    docstring): reads use a seqlock retry to avoid feeding torn bytes into
    pickle.loads(), and _stream_loop tracks frame_idx progression to detect
    an orphaned (unlink()ed + recreated) segment, since unlike SharedMemoryService
    this class holds its own shm handle for the lifetime of the subscription
    rather than re-fetching it from a shared cache.
    """

    _MAX_TORN_READ_RETRIES = 3
    _TORN_READ_RETRY_DELAY = 0.002  # seconds
    _STALE_FRAME_MAX_AGE = 5.0  # seconds

    def __init__(self):
        """Initialize stream service"""
        self.active_streams: Dict[str, asyncio.Task] = {}  # {serial_number: task}
        self.stream_subscribers: Dict[str, Set[str]] = {}  # {serial_number: set(sid)}
        self.save_enabled: Dict[str, bool] = {}  # {serial_number: save_enabled}

    def add_subscriber(self, serial_number: str, sid: str):
        """Add client subscription to camera stream"""
        if serial_number not in self.stream_subscribers:
            self.stream_subscribers[serial_number] = set()

        self.stream_subscribers[serial_number].add(sid)
        logger.info(f"Client {sid} subscribed to camera {serial_number}")

    def remove_subscriber(self, serial_number: str, sid: str):
        """Remove client subscription"""
        if serial_number in self.stream_subscribers:
            self.stream_subscribers[serial_number].discard(sid)

            # If no more subscribers, stop streaming
            if not self.stream_subscribers[serial_number]:
                del self.stream_subscribers[serial_number]
                logger.info(f"No more subscribers for {serial_number}")

    def has_subscribers(self, serial_number: str) -> bool:
        """Check if camera has any subscribers"""
        return serial_number in self.stream_subscribers and len(self.stream_subscribers[serial_number]) > 0

    def get_subscribers(self, serial_number: str) -> Set[str]:
        """Get list of subscriber SIDs for a camera"""
        return self.stream_subscribers.get(serial_number, set())

    async def start_streaming(self, serial_number: str, frame_rate: int = 10, save_enabled: bool = False):
        """
        Start streaming frames from shared memory

        Args:
            serial_number: Camera serial number
            frame_rate: Target frame rate (FPS)
            save_enabled: Save frames to disk if True
        """
        # Check if already streaming
        if serial_number in self.active_streams:
            logger.warning(f"Stream already active for {serial_number}")
            return

        # Store save_enabled flag
        self.save_enabled[serial_number] = save_enabled

        # Create streaming task
        task = asyncio.create_task(
            self._stream_loop(serial_number, frame_rate)
        )
        self.active_streams[serial_number] = task
        logger.info(f"Started streaming for {serial_number} at {frame_rate} FPS (save_enabled={save_enabled})")

    async def stop_streaming(self, serial_number: str):
        """Stop streaming for a camera"""
        if serial_number in self.active_streams:
            task = self.active_streams[serial_number]
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            del self.active_streams[serial_number]

            # Clear save_enabled flag
            if serial_number in self.save_enabled:
                del self.save_enabled[serial_number]

            logger.info(f"Stopped streaming for {serial_number}")

    async def _stream_loop(self, serial_number: str, frame_rate: int):
        """
        Main streaming loop - reads from shared memory and emits frames

        Args:
            serial_number: Camera serial number
            frame_rate: Target FPS
        """
        shm_name = f"camera_{serial_number}"
        interval = 1.0 / frame_rate

        shm = None
        last_frame_idx: Optional[int] = None
        last_advance_ts = time.monotonic()

        try:
            # Open shared memory
            shm = shared_memory.SharedMemory(name=shm_name)
            logger.info(f"Opened shared memory: {shm_name}")

            while True:
                # Check if still has subscribers
                if not self.has_subscribers(serial_number):
                    logger.info(f"No subscribers, stopping stream for {serial_number}")
                    break

                # Reattach if a previous stale-shm recovery closed our handle.
                if shm is None:
                    try:
                        shm = shared_memory.SharedMemory(name=shm_name)
                        logger.info(f"Re-opened shared memory: {shm_name}")
                        last_frame_idx = None
                        last_advance_ts = time.monotonic()
                    except FileNotFoundError:
                        await asyncio.sleep(interval)
                        continue

                try:
                    # Read frame from shared memory
                    frame_data = self._read_frame_from_shm(shm)

                    if frame_data is not None:
                        frame_idx = frame_data['metadata'].get('frame_idx', 0)
                        now = time.monotonic()

                        if frame_idx != last_frame_idx:
                            last_frame_idx = frame_idx
                            last_advance_ts = now
                        elif (now - last_advance_ts) >= self._STALE_FRAME_MAX_AGE:
                            # frame_idx hasn't moved despite successful reads —
                            # this handle is very likely reading an orphaned
                            # (unlink()ed + recreated elsewhere) segment.
                            logger.warning(
                                f"Stream {serial_number} stuck at frame_idx={frame_idx} for "
                                f"{now - last_advance_ts:.0f}s — recovering stale shm"
                            )
                            await self._recover_stale_stream(serial_number)
                            try:
                                shm.close()
                            except Exception:
                                pass
                            shm = None
                            last_frame_idx = None
                            await asyncio.sleep(interval)
                            continue

                        # Save to disk if enabled
                        if self.save_enabled.get(serial_number, False):
                            from app.utils.frame_saver import save_frame_to_disk
                            saved_path = save_frame_to_disk(serial_number, frame_data['image'], quality=95)
                            if saved_path:
                                logger.debug(f"Saved frame to disk: {saved_path}")

                        # Downscale and encode
                        frame_base64 = self._encode_frame(frame_data['image'])

                        if frame_base64:
                            # Emit to SocketIO
                            from app.services.socketio_service import emit_camera_frame

                            await emit_camera_frame({
                                'serial_number': serial_number,
                                'frame_base64': frame_base64,
                                'timestamp': frame_data['metadata'].get('timestamp'),
                                'frame_idx': frame_idx
                            })

                            # logger.debug(f"Emitted frame {frame_idx} for {serial_number}, base64_len={len(frame_base64)}")
                        else:
                            logger.warning(f"Failed to encode frame for {serial_number}")
                    else:
                        logger.debug(f"No frame data available for {serial_number}")

                except Exception as e:
                    logger.error(f"Error reading frame from {serial_number}: {e}")
                    import traceback
                    traceback.print_exc()

                # Control frame rate
                await asyncio.sleep(interval)

        except FileNotFoundError:
            logger.error(f"Shared memory not found: {shm_name}")

        except asyncio.CancelledError:
            logger.info(f"Stream cancelled for {serial_number}")
            raise

        except Exception as e:
            logger.error(f"Error in stream loop for {serial_number}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            if shm:
                try:
                    shm.close()
                except Exception as e:
                    logger.error(f"Error closing shared memory: {e}")

    async def _recover_stale_stream(self, serial_number: str):
        """Trigger the shared stale-shm recovery path (drop cache + respawn camera)."""
        try:
            from app.services.camera_service_supervisor import recover_stale_shm
            await recover_stale_shm(
                serial_number,
                f"Camera {serial_number} live stream frame_idx stuck (stale shm)"
            )
        except Exception as e:
            logger.error(f"Failed to trigger stale-shm recovery for stream {serial_number}: {e}")

    def _read_frame_from_shm(self, shm: shared_memory.SharedMemory) -> Optional[Dict[str, Any]]:
        """
        Read latest frame from ring buffer shared memory, retrying on torn reads.
        See SharedMemoryService._read_frame_from_slot for the seqlock rationale:
        the AI writer can overwrite this exact slot mid-read with no lock.
        """
        for attempt in range(self._MAX_TORN_READ_RETRIES):
            result = self._read_frame_from_shm_once(shm)
            if result is not None:
                return result
            if attempt < self._MAX_TORN_READ_RETRIES - 1:
                time.sleep(self._TORN_READ_RETRY_DELAY)
        return None

    def _read_frame_from_shm_once(self, shm: shared_memory.SharedMemory) -> Optional[Dict[str, Any]]:
        """
        Single read attempt of the latest ring buffer slot.

        Ring Buffer Format:
        - Header (64 bytes): [write_idx, frame_count, buffer_size, frame_counter, slot_size, ...]
        - Frame Slots (5 slots): Each slot contains one frame with metadata

        Returns:
            Dict with 'image' (numpy array) and 'metadata', or None on failure
            or a detected torn read (caller retries).
        """
        try:
            # Constants
            HEADER_SIZE = 64
            BUFFER_SIZE = 5

            # Read ring buffer header
            write_idx = struct.unpack_from("<I", shm.buf, 0)[0]
            frame_count = struct.unpack_from("<I", shm.buf, 4)[0]
            slot_size = struct.unpack_from("<I", shm.buf, 20)[0]

            # Check if buffer has any frames
            if frame_count == 0:
                return None

            # Calculate slot index of latest frame (last written)
            latest_slot_idx = (write_idx - 1) % BUFFER_SIZE

            # Calculate slot offset
            slot_offset = HEADER_SIZE + (latest_slot_idx * slot_size)
            offset = slot_offset

            # Read frame_idx (8 bytes)
            frame_idx_before = struct.unpack_from("<Q", shm.buf, offset)[0]
            offset += 8

            # Read timestamp (8 bytes)
            timestamp_ns = struct.unpack_from("<Q", shm.buf, offset)[0]
            offset += 8

            # Read metadata length (4 bytes)
            metadata_len = struct.unpack_from("<I", shm.buf, offset)[0]
            offset += 4

            # Validate metadata_len to prevent reading garbage
            if metadata_len > 10000:  # Metadata should be < 10KB
                logger.warning(f"Invalid metadata_len={metadata_len}, slot may be empty")
                return None

            # Read metadata bytes (raw copy only — do NOT unpickle yet)
            metadata_bytes = bytes(shm.buf[offset:offset+metadata_len])
            offset += metadata_len

            # Read frame length (4 bytes)
            frame_len = struct.unpack_from("<I", shm.buf, offset)[0]
            offset += 4

            # Read frame bytes (raw copy)
            frame_bytes = bytes(shm.buf[offset:offset+frame_len])

            # Seqlock validation: bail before pickle.loads if the writer
            # overwrote this slot while we were copying the bytes above.
            frame_idx_after = struct.unpack_from("<Q", shm.buf, slot_offset)[0]
            if frame_idx_after != frame_idx_before:
                logger.debug(
                    f"Torn read on latest slot: frame_idx {frame_idx_before} -> {frame_idx_after}"
                )
                return None

            # Bytes are consistent — safe to unpickle now.
            metadata = pickle.loads(metadata_bytes)

            # Add ring buffer specific metadata
            metadata['frame_idx'] = frame_idx_before
            metadata['timestamp_ns'] = timestamp_ns

            # Reconstruct frame array
            shape = metadata.get("shape")
            dtype = metadata.get("dtype")

            if not shape or not dtype:
                logger.error("Missing shape or dtype in metadata")
                return None

            img_array = np.frombuffer(frame_bytes, dtype=dtype).reshape(shape)

            return {
                'image': img_array,
                'metadata': metadata
            }

        except Exception as e:
            logger.warning(f"Error reading from ring buffer: {e}")
            return None

    def _encode_frame(self, img_array: np.ndarray, quality: int = 65) -> Optional[str]:
        """
        Encode frame as base64 JPEG

        Args:
            img_array: BGR image
            quality: JPEG quality (1-100)

        Returns:
            Base64 encoded string
        """
        try:
            # Downscale to 1/3 resolution
            h, w = img_array.shape[:2]
            small_img = cv2.resize(img_array, (w // 3, h // 3), interpolation=cv2.INTER_AREA)

            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', small_img, [cv2.IMWRITE_JPEG_QUALITY, quality])

            # Convert to base64
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            return frame_base64

        except Exception as e:
            logger.error(f"Error encoding frame: {e}")
            return None


# Global singleton instance
camera_stream_service = CameraStreamService()
