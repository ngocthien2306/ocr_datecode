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
                logger.info(f"[READ] Creating new connection to shared memory '{shm_name}'")
                self._shm_connections[shm_name] = shared_memory.SharedMemory(name=shm_name)
            else:
                logger.debug(f"[READ] Reusing existing connection to shared memory '{shm_name}'")

            shm = self._shm_connections[shm_name]

            # Retry logic để handle race condition
            max_retries = 3
            for retry in range(max_retries):
                try:
                    offset = 0

                    # Đọc frame version đầu (8 bytes)
                    version_start = int.from_bytes(bytes(shm.buf[offset:offset+8]), 'little')
                    offset += 8

                    # Đọc metadata length (4 bytes)
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
                    offset += frame_len

                    # Đọc frame version cuối (8 bytes)
                    version_end = int.from_bytes(bytes(shm.buf[offset:offset+8]), 'little')

                    # Verify version consistency
                    if version_start != version_end:
                        if retry < max_retries - 1:
                            logger.warning(f"[READ] Version mismatch (start={version_start}, end={version_end}), retrying...")
                            import time
                            time.sleep(0.01)
                            continue
                        else:
                            logger.error(f"[READ] Version mismatch after {max_retries} retries")
                            return None

                    # Chuyển bytes thành numpy array
                    shape = metadata['shape']
                    dtype = np.dtype(metadata['dtype'])
                    frame = np.frombuffer(frame_bytes, dtype=dtype).reshape(shape)

                    return {
                        'frame': frame,
                        'metadata': metadata
                    }

                except Exception as e:
                    if retry < max_retries - 1:
                        logger.warning(f"[READ] Error on attempt {retry + 1}: {e}, retrying...")
                        import time
                        time.sleep(0.01)
                        continue
                    else:
                        raise

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
            # Clear stale connection và retry một lần
            if shm_name in self._shm_connections:
                logger.info(f"[READ] Clearing potentially stale connection for '{shm_name}' and retrying...")
                try:
                    self._shm_connections[shm_name].close()
                except:
                    pass
                del self._shm_connections[shm_name]
                # Retry once with fresh connection
                try:
                    return self.get_frame(serial_number)
                except:
                    pass
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
