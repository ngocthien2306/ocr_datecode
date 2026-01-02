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

    async def start_streaming(self, serial_number: str, frame_rate: int = 10):
        """
        Start streaming frames from shared memory

        Args:
            serial_number: Camera serial number
            frame_rate: Target frame rate (FPS)
        """
        # Check if already streaming
        if serial_number in self.active_streams:
            logger.warning(f"Stream already active for {serial_number}")
            return

        # Create streaming task
        task = asyncio.create_task(
            self._stream_loop(serial_number, frame_rate)
        )
        self.active_streams[serial_number] = task
        logger.info(f"Started streaming for {serial_number} at {frame_rate} FPS")

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
                        # Downscale and encode
                        frame_base64 = self._encode_frame(frame_data['image'])

                        if frame_base64:
                            # Emit to SocketIO
                            from app.services.socketio_service import emit_camera_frame

                            await emit_camera_frame({
                                'serial_number': serial_number,
                                'frame_base64': frame_base64,
                                'timestamp': frame_data['metadata'].get('timestamp'),
                                'frame_idx': frame_data['metadata'].get('frame_idx', 0)
                            })

                except Exception as e:
                    logger.error(f"Error reading frame from {serial_number}: {e}")

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

        Returns:
            Dict with 'image' (numpy array) and 'metadata'
        """
        try:
            # Read header (metadata size)
            metadata_size = struct.unpack('I', bytes(shm.buf[:4]))[0]

            if metadata_size == 0 or metadata_size > 1000000:  # Sanity check
                return None

            # Read metadata
            metadata_bytes = bytes(shm.buf[4:4 + metadata_size])
            metadata = pickle.loads(metadata_bytes)

            # Read image shape
            shape = metadata.get('shape')
            if not shape or len(shape) != 3:
                return None

            # Calculate image size
            h, w, c = shape
            image_size = h * w * c

            # Read image data
            image_offset = 4 + metadata_size
            image_bytes = bytes(shm.buf[image_offset:image_offset + image_size])

            # Reconstruct numpy array
            img_array = np.frombuffer(image_bytes, dtype=np.uint8)
            img_array = img_array.reshape(shape)

            return {
                'image': img_array,
                'metadata': metadata
            }

        except Exception as e:
            logger.debug(f"Error reading from shared memory: {e}")
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
