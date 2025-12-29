"""
Camera Service
Service để đọc frame từ shared memory
"""
from multiprocessing import shared_memory
import pickle
import numpy as np
import cv2
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class CameraFrameService:
    """Service để đọc frame từ shared memory"""

    def __init__(self):
        self._shm_connections: Dict[str, shared_memory.SharedMemory] = {}

    def get_frame(self, serial_number: str) -> Optional[Dict]:
        """
        Đọc frame mới nhất từ shared memory

        Args:
            serial_number: Serial number của camera

        Returns:
            Dict chứa frame và metadata hoặc None nếu không tìm thấy
        """
        shm_name = f"camera_{serial_number}"

        try:
            # Kết nối hoặc tái sử dụng kết nối shared memory
            if shm_name not in self._shm_connections:
                self._shm_connections[shm_name] = shared_memory.SharedMemory(name=shm_name)

            shm = self._shm_connections[shm_name]

            # Đọc metadata length (4 bytes đầu tiên)
            offset = 0
            metadata_len = int.from_bytes(bytes(shm.buf[offset:offset+4]), 'little')
            offset += 4

            # Đọc metadata
            metadata_bytes = bytes(shm.buf[offset:offset+metadata_len])
            metadata = pickle.loads(metadata_bytes)
            offset += metadata_len

            # Đọc frame length
            frame_len = int.from_bytes(bytes(shm.buf[offset:offset+4]), 'little')
            offset += 4

            # Đọc frame data
            frame_bytes = bytes(shm.buf[offset:offset+frame_len])

            # Chuyển bytes thành numpy array
            shape = metadata['shape']
            dtype = np.dtype(metadata['dtype'])
            frame = np.frombuffer(frame_bytes, dtype=dtype).reshape(shape)

            return {
                'frame': frame,
                'metadata': metadata
            }

        except FileNotFoundError:
            logger.warning(f"Shared memory '{shm_name}' not found. Camera producer may not be running.")
            # Xóa kết nối đã lưu nếu có
            if shm_name in self._shm_connections:
                try:
                    self._shm_connections[shm_name].close()
                except:
                    pass
                del self._shm_connections[shm_name]
            return None
        except Exception as e:
            logger.error(f"Error reading frame from shared memory: {e}")
            return None

    def encode_frame_jpeg(self, frame: np.ndarray, quality: int = 85) -> Optional[bytes]:
        """
        Encode frame thành JPEG bytes

        Args:
            frame: Numpy array của frame
            quality: JPEG quality (0-100)

        Returns:
            JPEG bytes hoặc None nếu lỗi
        """
        try:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            return buffer.tobytes()
        except Exception as e:
            logger.error(f"Error encoding frame: {e}")
            return None

    def close_all(self):
        """Đóng tất cả kết nối shared memory"""
        for shm_name, shm in self._shm_connections.items():
            try:
                shm.close()
            except Exception as e:
                logger.error(f"Error closing shared memory {shm_name}: {e}")
        self._shm_connections.clear()


# Singleton instance
camera_frame_service = CameraFrameService()
