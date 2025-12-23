"""Camera Manager with Multiprocessing - Auto detect cameras"""

from multiprocessing import Process, Queue, Event
import time
import logging
import traceback
from pypylon import pylon

from camera import BaslerCamera

logger = logging.getLogger(__name__)


def camera_worker(camera_id, config, cmd_queue, result_queue, frame_queue, stop_event):
    """Camera process worker"""
    logger.info(f"[{camera_id}] Process started")

    camera = None
    try:
        camera = BaslerCamera(camera_id, config)

        if not camera.connect():
            result_queue.put({'camera_id': camera_id, 'type': 'error', 'message': 'Connection failed'})
            return

        result_queue.put({'camera_id': camera_id, 'type': 'connected'})

        if not camera.start_grabbing():
            result_queue.put({'camera_id': camera_id, 'type': 'error', 'message': 'Start grabbing failed'})
            return

        while not stop_event.is_set():
            # Check commands
            if not cmd_queue.empty():
                try:
                    cmd = cmd_queue.get_nowait()
                    if cmd.get('type') == 'software_trigger':
                        camera.execute_software_trigger()
                except:
                    pass

            # Grab frame
            result = camera.grab_frame(timeout_ms=100)
            if result:
                img_array, metadata = result

                frame_data = {
                    'camera_id': camera_id,
                    'image': img_array,
                    'metadata': metadata,
                }

                if not frame_queue.full():
                    frame_queue.put_nowait(frame_data)

            time.sleep(0.001)

    except Exception as e:
        logger.error(f"[{camera_id}] Error: {e}")
        traceback.print_exc()
    finally:
        if camera:
            camera.disconnect()
        logger.info(f"[{camera_id}] Process stopped")


class CameraManager:
    def __init__(self):
        self.processes = {}
        self.cmd_queues = {}
        self.stop_events = {}
        self.result_queue = Queue()
        self.frame_queue = Queue(maxsize=100)
        self.camera_configs = {}

    def enumerate_and_start_cameras(self) -> int:
        """Enumerate all connected cameras and start workers"""
        try:
            logger.info("Enumerating cameras...")

            tlFactory = pylon.TlFactory.GetInstance()
            devices = tlFactory.EnumerateDevices()

            if len(devices) == 0:
                logger.warning("No cameras found")
                return 0

            logger.info(f"Found {len(devices)} camera(s)")

            self.stop_all_cameras()

            for idx, device in enumerate(devices):
                serial = device.GetSerialNumber()
                model = device.GetModelName()
                camera_id = f"CAM_{serial}"

                logger.info(f"  [{idx+1}] {model} (S/N: {serial})")

                config = {
                    'camera_id': camera_id,
                    'serial_number': serial,
                    'model_name': model,
                    'exposure_time': 10000,  # Default 10ms
                    'gain': 0.0,
                    'pixel_format': 'Mono8',
                    'trigger_mode': False,  # Free running
                    'trigger_source': 'Software',
                }

                self.camera_configs[camera_id] = config
                self.start_camera(camera_id, config)

            logger.info(f"Started {len(self.camera_configs)} camera worker(s)")
            return len(self.camera_configs)

        except Exception as e:
            logger.error(f"Enumerate cameras error: {e}")
            traceback.print_exc()
            return 0

    def start_camera(self, camera_id, config):
        try:
            if camera_id in self.processes and self.processes[camera_id].is_alive():
                logger.warning(f"[{camera_id}] Already running")
                return False

            cmd_queue = Queue()
            stop_event = Event()

            process = Process(
                target=camera_worker,
                args=(camera_id, config, cmd_queue, self.result_queue, self.frame_queue, stop_event),
                daemon=True
            )
            process.start()

            self.processes[camera_id] = process
            self.cmd_queues[camera_id] = cmd_queue
            self.stop_events[camera_id] = stop_event

            logger.info(f"[{camera_id}] Started (PID: {process.pid})")
            return True
        except Exception as e:
            logger.error(f"[{camera_id}] Start error: {e}")
            return False

    def stop_camera(self, camera_id):
        if camera_id not in self.processes:
            return

        self.stop_events[camera_id].set()
        self.processes[camera_id].join(timeout=5)

        if self.processes[camera_id].is_alive():
            self.processes[camera_id].terminate()

        del self.processes[camera_id]
        del self.cmd_queues[camera_id]
        del self.stop_events[camera_id]
        logger.info(f"[{camera_id}] Stopped")

    def stop_all_cameras(self):
        for camera_id in list(self.processes.keys()):
            self.stop_camera(camera_id)

    def software_trigger(self, camera_id):
        if camera_id in self.cmd_queues:
            self.cmd_queues[camera_id].put({'type': 'software_trigger'})

    def get_frame(self, timeout=1.0):
        try:
            return self.frame_queue.get(timeout=timeout)
        except:
            return None

    def get_result(self, timeout=1.0):
        try:
            return self.result_queue.get(timeout=timeout)
        except:
            return None

    def get_running_cameras(self):
        return [cam_id for cam_id, proc in self.processes.items() if proc.is_alive()]

    def cleanup(self):
        self.stop_all_cameras()

    def __del__(self):
        self.cleanup()
