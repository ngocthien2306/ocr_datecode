"""
Shared Memory Service
Reads frames from shared memory created by Camera processes
"""

from multiprocessing import shared_memory
import pickle
import struct
import numpy as np
import cv2
import logging
from typing import Optional, Tuple, Dict, Any
import threading

logger = logging.getLogger(__name__)


class SharedMemoryService:
    """
    Service to read frames from shared memory

    Features:
    - Read latest frame from camera shared memory
    - Thread-safe operations
    - Automatic format conversion (BGR → JPEG)
    - Metadata extraction
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._connections: Dict[str, shared_memory.SharedMemory] = {}

        logger.info("SharedMemoryService initialized")

    def _get_shm(self, serial_number: str) -> Optional[shared_memory.SharedMemory]:
        """
        Get or create shared memory connection

        Args:
            serial_number: Camera serial number

        Returns:
            SharedMemory instance or None
        """
        shm_name = f"camera_{serial_number}"

        # Check if already connected
        if shm_name in self._connections:
            return self._connections[shm_name]

        # Try to connect
        try:
            shm = shared_memory.SharedMemory(name=shm_name, create=False)
            self._connections[shm_name] = shm
            logger.info(f"Connected to shared memory: {shm_name}")
            return shm

        except FileNotFoundError:
            logger.warning(f"Shared memory not found: {shm_name}")
            return None

        except Exception as e:
            logger.error(f"Error connecting to shared memory {shm_name}: {e}")
            return None

    def read_frame(
        self,
        serial_number: str
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Read latest frame and metadata from shared memory

        Args:
            serial_number: Camera serial number

        Returns:
            Tuple of (frame_array, metadata) or None if failed

        Frame format:
        [frame_idx (8B)] [metadata_len (4B)] [metadata_pickle]
        [frame_len (4B)] [frame_bytes_BGR] [frame_idx_verify (8B)]
        """
        with self._lock:
            shm = self._get_shm(serial_number)

            if not shm:
                return None

            try:
                offset = 0

                # Read frame version (8 bytes)
                frame_idx_start = struct.unpack_from("<Q", shm.buf, offset)[0]
                offset += 8

                # Read metadata length (4 bytes)
                metadata_len = struct.unpack_from("<I", shm.buf, offset)[0]
                offset += 4

                # Read metadata bytes
                metadata_bytes = bytes(shm.buf[offset:offset+metadata_len])
                metadata = pickle.loads(metadata_bytes)
                offset += metadata_len

                # Read frame length (4 bytes)
                frame_len = struct.unpack_from("<I", shm.buf, offset)[0]
                offset += 4

                # Read frame bytes
                frame_bytes = bytes(shm.buf[offset:offset+frame_len])
                offset += frame_len

                # Read frame version verify (8 bytes)
                frame_idx_end = struct.unpack_from("<Q", shm.buf, offset)[0]

                # Verify frame integrity
                if frame_idx_start != frame_idx_end:
                    logger.warning(
                        f"Frame integrity check failed for {serial_number}: "
                        f"start={frame_idx_start}, end={frame_idx_end}"
                    )
                    return None

                # Reconstruct frame array
                shape = metadata.get("shape")
                dtype = metadata.get("dtype")

                if not shape or not dtype:
                    logger.error("Missing shape or dtype in metadata")
                    return None

                frame_array = np.frombuffer(frame_bytes, dtype=dtype).reshape(shape)

                logger.debug(
                    f"Read frame from {serial_number}: "
                    f"idx={frame_idx_start}, shape={shape}"
                )

                return frame_array, metadata

            except Exception as e:
                logger.error(f"Error reading frame from {serial_number}: {e}")
                # Remove failed connection
                if f"camera_{serial_number}" in self._connections:
                    try:
                        self._connections[f"camera_{serial_number}"].close()
                    except:
                        pass
                    del self._connections[f"camera_{serial_number}"]

                return None

    def read_frame_as_jpeg(
        self,
        serial_number: str,
        quality: int = 90
    ) -> Optional[bytes]:
        """
        Read frame and encode as JPEG

        Args:
            serial_number: Camera serial number
            quality: JPEG quality (0-100)

        Returns:
            JPEG bytes or None
        """
        result = self.read_frame(serial_number)

        if not result:
            return None

        frame_array, metadata = result

        try:
            # Encode as JPEG
            success, jpeg_buffer = cv2.imencode(
                ".jpg",
                frame_array,
                [cv2.IMWRITE_JPEG_QUALITY, quality]
            )

            if not success:
                logger.error(f"Failed to encode frame as JPEG for {serial_number}")
                return None

            return jpeg_buffer.tobytes()

        except Exception as e:
            logger.error(f"Error encoding frame as JPEG: {e}")
            return None

    def cleanup(self, serial_number: Optional[str] = None):
        """
        Cleanup shared memory connections

        Args:
            serial_number: If specified, cleanup only this camera.
                          Otherwise cleanup all.
        """
        with self._lock:
            if serial_number:
                shm_name = f"camera_{serial_number}"
                if shm_name in self._connections:
                    try:
                        self._connections[shm_name].close()
                        del self._connections[shm_name]
                        logger.info(f"Closed shared memory connection: {shm_name}")
                    except Exception as e:
                        logger.error(f"Error closing shared memory {shm_name}: {e}")

            else:
                # Cleanup all connections
                for shm_name, shm in list(self._connections.items()):
                    try:
                        shm.close()
                        logger.info(f"Closed shared memory connection: {shm_name}")
                    except Exception as e:
                        logger.error(f"Error closing shared memory {shm_name}: {e}")

                self._connections.clear()


# Singleton instance
shared_memory_service = SharedMemoryService()
