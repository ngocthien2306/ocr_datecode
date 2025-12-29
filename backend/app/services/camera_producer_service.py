"""
Camera Producer Service
Service to manage camera producer processes
"""
import subprocess
import psutil
import logging
from typing import Optional, Dict
import os

logger = logging.getLogger(__name__)


class CameraProducerService:
    """Service to manage camera producer processes"""

    def __init__(self):
        self._processes: Dict[str, subprocess.Popen] = {}
        self._camera_script_path = "/home/demo/Source/ocr_datecode/ai_services/camera_shm_producer.py"

    def start_camera(self, serial_number: str, device_index: int = 0) -> bool:
        """
        Start camera producer process

        Args:
            serial_number: Serial number of camera
            device_index: Camera device index (default 0)

        Returns:
            True if started successfully, False otherwise
        """
        if serial_number in self._processes:
            if self._processes[serial_number].poll() is None:
                logger.warning(f"Camera {serial_number} producer already running")
                return True

        try:
            # Start camera producer process
            process = subprocess.Popen(
                ["python3", self._camera_script_path],
                cwd=os.path.dirname(self._camera_script_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            self._processes[serial_number] = process
            logger.info(f"Started camera producer for {serial_number} (PID: {process.pid})")
            return True

        except Exception as e:
            logger.error(f"Failed to start camera producer for {serial_number}: {e}")
            return False

    def stop_camera(self, serial_number: str) -> bool:
        """
        Stop camera producer process

        Args:
            serial_number: Serial number of camera

        Returns:
            True if stopped successfully, False otherwise
        """
        if serial_number not in self._processes:
            logger.warning(f"Camera {serial_number} producer not found")
            return False

        try:
            process = self._processes[serial_number]

            if process.poll() is not None:
                # Process already terminated
                del self._processes[serial_number]
                return True

            # Terminate process
            process.terminate()

            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if timeout
                process.kill()
                process.wait()

            del self._processes[serial_number]
            logger.info(f"Stopped camera producer for {serial_number}")
            return True

        except Exception as e:
            logger.error(f"Failed to stop camera producer for {serial_number}: {e}")
            return False

    def get_camera_status(self, serial_number: str) -> Dict:
        """
        Get camera producer status

        Args:
            serial_number: Serial number of camera

        Returns:
            Dict with status information
        """
        if serial_number not in self._processes:
            return {
                "running": False,
                "pid": None
            }

        process = self._processes[serial_number]
        is_running = process.poll() is None

        if not is_running:
            del self._processes[serial_number]

        return {
            "running": is_running,
            "pid": process.pid if is_running else None
        }

    def stop_all_cameras(self):
        """Stop all camera producer processes"""
        serial_numbers = list(self._processes.keys())
        for serial_number in serial_numbers:
            self.stop_camera(serial_number)


# Singleton instance
camera_producer_service = CameraProducerService()
