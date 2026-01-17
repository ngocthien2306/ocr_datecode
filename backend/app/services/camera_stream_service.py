"""
Camera Stream Service
Reads frames from shared memory and streams via SocketIO
"""

import asyncio
import logging
import base64
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
    """

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

        try:
            # Open shared memory
            shm = shared_memory.SharedMemory(name=shm_name)
            logger.info(f"Opened shared memory: {shm_name}")

            while True:
                # Check if still has subscribers
                if not self.has_subscribers(serial_number):
                    logger.info(f"No subscribers, stopping stream for {serial_number}")
                    break

                try:
                    # Read frame from shared memory
                    frame_data = self._read_frame_from_shm(shm)

                    if frame_data is not None:
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

                            frame_idx = frame_data['metadata'].get('frame_idx', 0)
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

    def _read_frame_from_shm(self, shm: shared_memory.SharedMemory) -> Optional[Dict[str, Any]]:
        """
        Read frame from shared memory

        Format written by camera.py:
        [8 bytes: frame_idx] [4 bytes: metadata_len] [metadata] [4 bytes: frame_len] [frame] [8 bytes: frame_idx verify]

        Returns:
            Dict with 'image' (numpy array) and 'metadata'
        """
        try:
            offset = 0

            # Read frame version (8 bytes)
            frame_idx = struct.unpack('<Q', bytes(shm.buf[offset:offset+8]))[0]
            offset += 8

            # Read metadata size (4 bytes)
            metadata_size = struct.unpack('<I', bytes(shm.buf[offset:offset+4]))[0]
            offset += 4

            if metadata_size == 0 or metadata_size > 1000000:  # Sanity check
                return None

            # Read metadata
            metadata_bytes = bytes(shm.buf[offset:offset + metadata_size])
            metadata = pickle.loads(metadata_bytes)
            offset += metadata_size

            # Read frame length (4 bytes)
            frame_len = struct.unpack('<I', bytes(shm.buf[offset:offset+4]))[0]
            offset += 4

            # Read image shape from metadata
            shape = metadata.get('shape')
            if not shape or len(shape) != 3:
                return None

            # Calculate expected image size
            h, w, c = shape
            expected_size = h * w * c

            if frame_len != expected_size:
                logger.warning(f"Frame length mismatch: {frame_len} != {expected_size}")

            # Read image data
            image_bytes = bytes(shm.buf[offset:offset + frame_len])
            offset += frame_len

            # Read frame version verify (8 bytes)
            frame_idx_verify = struct.unpack('<Q', bytes(shm.buf[offset:offset+8]))[0]

            # Verify frame consistency
            if frame_idx != frame_idx_verify:
                logger.warning(f"Frame index mismatch: {frame_idx} != {frame_idx_verify} (frame being written)")
                return None

            # Reconstruct numpy array
            img_array = np.frombuffer(image_bytes, dtype=np.uint8)
            img_array = img_array.reshape(shape)

            return {
                'image': img_array,
                'metadata': metadata
            }

        except Exception as e:
            logger.warning(f"Error reading from shared memory: {e}")
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
