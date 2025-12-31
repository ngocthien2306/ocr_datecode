#!/usr/bin/env python3

from multiprocessing import shared_memory, Queue
import cv2
from pypylon import pylon
import numpy as np
import time
import pickle
import logging
import struct
import ctypes
import json
import os
import subprocess
from pathlib import Path

# Import for inference (only SuperPointMatcherTRT, not the singleton service)
try:
    import sys
    from pathlib import Path as PathLib
    ai_services_path = PathLib(__file__).parent
    if str(ai_services_path) not in sys.path:
        sys.path.insert(0, str(ai_services_path))

    from inference_service import SuperPointMatcherTRT
    import requests
    import shutil
    INFERENCE_AVAILABLE = True
    logging.info(f"✅ [INFERENCE] SuperPointMatcherTRT imported successfully")
except Exception as e:
    logging.warning(f"Inference service not available: {e}")
    INFERENCE_AVAILABLE = False
    SuperPointMatcherTRT = None

SETTINGS_DIR = Path("/home/demo/Source/ocr_datecode/backend/camera_settings")
LOGS_DIR = Path("/home/demo/Source/ocr_datecode/backend/logs")

# Create logs directory
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / 'camera_producer.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


def read_di_value(di_number: int, use_password: bool = False, sudo_password: str = "1") -> int:
    """
    Read Digital Input value from hardware

    Args:
        di_number: DI number (0-3)
        use_password: If True, use password (False if sudoers configured)
        sudo_password: Password for sudo command (default: "1")

    Returns:
        0 or 1 (DI state)
    """
    try:
        if use_password:
            # Method 1: With password (if sudoers NOT configured)
            process = subprocess.Popen(
                ['sudo', '-S', 'dio_in', str(di_number)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(input=f"{sudo_password}\n", timeout=2)
        else:
            # Method 2: Without password (if sudoers configured) - FASTER!
            result = subprocess.run(
                ['sudo', '-n', 'dio_in', str(di_number)],
                capture_output=True,
                text=True,
                timeout=1
            )
            stdout = result.stdout
            stderr = result.stderr
            process = result

        if process.returncode == 0:
            # Parse output format: "The id-0 input gpio status = 0"
            for line in stdout.split('\n'):
                if 'status' in line and '=' in line:
                    value_str = line.split('=')[-1].strip()
                    try:
                        return int(value_str)
                    except ValueError:
                        pass
            logger.warning(f"Could not parse DI {di_number} output: {stdout}")
            return 0
        else:
            logger.warning(f"Failed to read DI {di_number}: {stderr}")
            return 0
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout reading DI {di_number}")
        return 0
    except Exception as e:
        logger.error(f"Error reading DI {di_number}: {e}")
        return 0


def write_do_value(do_number: int, value: int, use_password: bool = False, sudo_password: str = "1") -> bool:
    """
    Write Digital Output value to hardware

    Args:
        do_number: DO number (0-3)
        value: Output value (0 or 1)
        use_password: If True, use password (False if sudoers configured)
        sudo_password: Password for sudo command (default: "1")

    Returns:
        True if successful, False otherwise
    """
    try:
        if use_password:
            # Method 1: With password (if sudoers NOT configured)
            process = subprocess.Popen(
                ['sudo', '-S', 'dio_out', str(do_number), str(value)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(input=f"{sudo_password}\n", timeout=2)
            returncode = process.returncode
        else:
            # Method 2: Without password (if sudoers configured) - FASTER!
            result = subprocess.run(
                ['sudo', '-n', 'dio_out', str(do_number), str(value)],
                capture_output=True,
                text=True,
                timeout=1
            )
            returncode = result.returncode
            stderr = result.stderr

        if returncode == 0:
            logger.info(f"✅ [DO] Set DO {do_number} = {value}")
            return True
        else:
            logger.error(f"❌ [DO] Failed to set DO {do_number}: {stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ [DO] Error writing DO {do_number}: {e}")
        return False


def check_trigger_edge(current_value: int, previous_value: int, activation: str) -> bool:
    """
    Check if trigger edge condition is met

    Args:
        current_value: Current DI value
        previous_value: Previous DI value
        activation: Trigger activation type (RisingEdge, FallingEdge, AnyEdge)

    Returns:
        True if trigger condition is met
    """
    if activation == "RisingEdge":
        return previous_value == 0 and current_value == 1
    elif activation == "FallingEdge":
        return previous_value == 1 and current_value == 0
    elif activation == "AnyEdge":
        return previous_value != current_value
    return False


def load_camera_settings(serial_number: str) -> dict:
    """Load camera settings from JSON file"""
    settings_file = SETTINGS_DIR / f"{serial_number}.json"

    if settings_file.exists():
        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)
                logger.info(f"✅ [SETTINGS LOADED] Camera {serial_number}: {settings}")
                return settings
        except Exception as e:
            logger.error(f"❌ [SETTINGS ERROR] Failed to load settings for {serial_number}: {e}")

    # Return default settings if file doesn't exist
    default_settings = {
        "exposure_time": 10000,
        "gain": 1.0,
        "trigger_config": {
            "mode": "continuous",
            "trigger_source": "Software",
            "trigger_selector": "FrameStart",
            "trigger_activation": "RisingEdge",
            "di_number": 0
        }
    }
    logger.info(f"⚠️  [DEFAULT SETTINGS] Camera {serial_number}: Using default settings {default_settings}")
    return default_settings


def apply_camera_settings(camera, settings: dict, serial_number: str = "Unknown"):
    """Apply settings to camera"""
    applied = []

    try:
        if "exposure_time" in settings:
            exposure_time = int(settings["exposure_time"])
            camera.ExposureTime.SetValue(exposure_time)
            actual_exposure = camera.ExposureTime.GetValue()
            logger.info(f"✅ [EXPOSURE APPLIED] Camera {serial_number}: {exposure_time} μs (actual: {actual_exposure} μs)")
            applied.append(f"exposure={actual_exposure}μs")
    except Exception as e:
        logger.error(f"❌ [EXPOSURE ERROR] Camera {serial_number}: {e}")

    try:
        if "gain" in settings:
            gain = float(settings["gain"])
            camera.Gain.SetValue(gain)
            actual_gain = camera.Gain.GetValue()
            logger.info(f"✅ [GAIN APPLIED] Camera {serial_number}: {gain} (actual: {actual_gain})")
            applied.append(f"gain={actual_gain}")
    except Exception as e:
        logger.error(f"❌ [GAIN ERROR] Camera {serial_number}: {e}")

    if applied:
        logger.info(f"🎥 [SETTINGS SUMMARY] Camera {serial_number}: {', '.join(applied)}")


def load_recipe_and_init_matcher(recipe_id: str, serial_number: str, engine_path: str = "/home/demo/Source/ocr_datecode/weights/pipeline_fp16_small.engine"):
    """
    Load recipe from API and initialize TensorRT matcher (called once when recipe changes)

    Returns:
        (matcher, recipe_name) if successful, (None, None) otherwise
    """
    if not INFERENCE_AVAILABLE or not SuperPointMatcherTRT:
        logger.warning(f"❌ [INFERENCE] SuperPointMatcherTRT not available")
        return None, None

    try:
        # Call API to get recipe load metadata (contains full template data)
        API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
        url = f"{API_BASE}/api/recipes/loads/latest"
        logger.info(f"🌐 [API CALL] Fetching latest recipe load from {url}")

        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # Extract metadata from recipe load event
        metadata = data.get('metadata', {})
        loaded_recipe_id = metadata.get('recipe_id') or data.get('recipe_id')
        recipe_name = metadata.get('name', 'Unknown')

        # Verify this is the recipe we want
        if loaded_recipe_id != recipe_id:
            logger.warning(f"⚠️ [INFERENCE] Latest recipe ({loaded_recipe_id}) does not match requested ({recipe_id})")
            # Continue anyway - user just loaded this recipe, metadata should be correct

        logger.info(f"✅ [API] Got recipe: {recipe_name} (ID: {loaded_recipe_id})")

        # Find camera config and template in metadata
        cameras = metadata.get('cameras', [])
        camera_config = None
        for cam in cameras:
            if cam.get('serial_number') == serial_number:
                camera_config = cam
                break

        if not camera_config:
            logger.error(f"❌ [INFERENCE] Camera {serial_number} not found in recipe metadata")
            return None, None

        # Find template for this camera
        camera_templates = metadata.get('camera_templates', [])
        template_data = None
        for ct in camera_templates:
            if ct.get('camera_id') == camera_config.get('camera_id'):
                templates = ct.get('templates', [])
                if templates:
                    template_data = templates[0]
                    break

        if not template_data:
            logger.error(f"❌ [INFERENCE] No template found for camera {serial_number}")
            return None, None

        # Get template image from local filesystem
        image_url = template_data.get('image_url')
        if not image_url:
            logger.error(f"❌ [INFERENCE] No template image URL")
            return None, None

        filename = image_url.split('/')[-1]
        backend_dir = Path(__file__).parent.parent / "backend"
        source_template_path = backend_dir / "uploads" / "templates" / filename

        if not source_template_path.exists():
            logger.error(f"❌ [INFERENCE] Template not found: {source_template_path}")
            return None, None

        # Copy to temp directory
        temp_dir = Path("ocr_inference")
        temp_dir.mkdir(exist_ok=True)
        template_path = temp_dir / f"template_{serial_number}.jpg"
        shutil.copy(source_template_path, template_path)
        logger.info(f"📁 [INFERENCE] Template copied to {template_path}")

        # Parse annotations
        annotations = template_data.get('annotations', [])
        template_bbox = None
        other_bboxes = []

        for ann in annotations:
            ann_type = ann.get('type', '')
            if ann_type == 'template':
                x = ann.get('x', 0)
                y = ann.get('y', 0)
                w = ann.get('width', 0)
                h = ann.get('height', 0)

                template_img = cv2.imread(str(template_path))
                img_h, img_w = template_img.shape[:2]

                x1 = int(x * img_w)
                y1 = int(y * img_h)
                x2 = int((x + w) * img_w)
                y2 = int((y + h) * img_h)

                template_bbox = {
                    'type': 'template',
                    'points': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                }
            elif ann_type == 'text' and ann.get('points'):
                points = ann.get('points', [])
                template_img = cv2.imread(str(template_path))
                img_h, img_w = template_img.shape[:2]

                pixel_points = []
                for pt in points:
                    px = int(pt[0] * img_w)
                    py = int(pt[1] * img_h)
                    pixel_points.append([px, py])

                other_bboxes.append({
                    'type': ann_type,
                    'text': ann.get('text', ''),
                    'points': pixel_points
                })

        if not template_bbox:
            logger.error(f"❌ [INFERENCE] No template bbox in annotations")
            return None, None

        # Create annotation file for TensorRT matcher
        ann_json_path = temp_dir / f"annotations_{serial_number}.json"
        ann_data = {
            '_template_image': str(template_path),
            str(template_path): [template_bbox] + other_bboxes
        }

        with open(ann_json_path, 'w') as f:
            json.dump(ann_data, f, indent=2)

        # Initialize TensorRT matcher
        logger.info(f"🔥 [INFERENCE] Initializing TensorRT matcher with {len(other_bboxes)} regions...")
        matcher = SuperPointMatcherTRT(
            json_path=str(ann_json_path),
            engine_path=engine_path,
            scale=1.0,
            verbose=True
        )

        logger.info(f"✅ [INFERENCE] Matcher initialized for recipe: {recipe_name}")
        return matcher, recipe_name

    except Exception as e:
        logger.error(f"❌ [INFERENCE] Failed to load recipe: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None


def camera_producer(device_index=0):
    try:
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()

        if not devices or device_index >= len(devices):
            logger.error(f"Camera {device_index} not available")
            return

        device = devices[device_index]
        serial_number = device.GetSerialNumber()
        model_name = device.GetModelName()

        camera = pylon.InstantCamera(tlFactory.CreateDevice(device))
        camera.Open()

        logger.info(f"Opened: {model_name} (SN: {serial_number})")

        # Load and apply initial settings
        logger.info(f"🚀 [CAMERA INIT] Starting camera producer for {model_name} (SN: {serial_number})")
        settings = load_camera_settings(serial_number)
        apply_camera_settings(camera, settings, serial_number)
        last_settings_check = time.time()
        settings_file = SETTINGS_DIR / f"{serial_number}.json"
        last_mtime = settings_file.stat().st_mtime if settings_file.exists() else 0
        logger.info(f"📁 [SETTINGS FILE] Watching: {settings_file}")

        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        actual_width = camera.Width.GetValue()
        actual_height = camera.Height.GetValue()
        max_frame_size = actual_width * actual_height * 3  # BGR = 3 channels
        logger.info(f"Camera resolution: {actual_width}x{actual_height}, max frame size: {max_frame_size} bytes")

        shm_name = f"camera_{serial_number}"

        # Cleanup any existing shared memory first
        try:
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            existing_shm.close()
            existing_shm.unlink()
            logger.info(f"Cleaned up existing shared memory: {shm_name}")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Error cleaning up existing shared memory: {e}")

        try:
            shm_size = max_frame_size + 4096
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=shm_size)
            logger.info(f"Created shared memory: {shm_name} (size: {shm_size} bytes)")
        except Exception as e:
            logger.error(f"Failed to create shared memory: {e}")
            raise

        frame_idx = 0

        # Get trigger configuration
        trigger_config = settings.get('trigger_config', {})
        trigger_mode = trigger_config.get('mode', 'continuous')
        trigger_activation = trigger_config.get('trigger_activation', 'RisingEdge')
        di_number = trigger_config.get('di_number', 0)

        # For hardware trigger mode
        previous_di_value = None
        if trigger_mode == 'hardware':
            previous_di_value = read_di_value(di_number)
            logger.info(f"🔧 [HARDWARE TRIGGER] Monitoring DI {di_number}, activation: {trigger_activation}, initial value: {previous_di_value}")

        # Inference state (managed locally in camera producer)
        current_recipe_id = settings.get('recipe_id')
        current_recipe_name = settings.get('recipe_name')
        inference_matcher = None

        # Initialize matcher if recipe_id exists at startup
        if current_recipe_id and trigger_mode == 'hardware':
            logger.info(f"🔍 [INFERENCE] Recipe detected at startup: {current_recipe_name} ({current_recipe_id})")
            inference_matcher, loaded_recipe_name = load_recipe_and_init_matcher(current_recipe_id, serial_number)
            if inference_matcher:
                current_recipe_name = loaded_recipe_name
                logger.info(f"✅ [INFERENCE] Matcher initialized at startup")

        logger.info(f"🎬 [TRIGGER MODE] Camera {serial_number} running in '{trigger_mode}' mode")
        logger.info(f"Streaming to shared memory '{shm_name}'...")

        while camera.IsGrabbing():
            # Check for settings updates every 2 seconds
            current_time = time.time()
            if current_time - last_settings_check > 2.0:
                last_settings_check = current_time
                if settings_file.exists():
                    try:
                        current_mtime = settings_file.stat().st_mtime
                        if current_mtime > last_mtime:
                            # Settings file changed, reload
                            logger.info(f"🔄 [HOT RELOAD] Detected settings file change for camera {serial_number}")
                            new_settings = load_camera_settings(serial_number)
                            apply_camera_settings(camera, new_settings, serial_number)

                            # Check if recipe_id changed
                            new_recipe_id = new_settings.get('recipe_id')
                            if new_recipe_id and new_recipe_id != current_recipe_id:
                                logger.info(f"🔍 [RECIPE CHANGE] New recipe detected: {new_recipe_id} (was: {current_recipe_id})")
                                # Load recipe and init matcher (only call API once)
                                inference_matcher, loaded_recipe_name = load_recipe_and_init_matcher(new_recipe_id, serial_number)
                                if inference_matcher:
                                    current_recipe_id = new_recipe_id
                                    current_recipe_name = loaded_recipe_name
                                    logger.info(f"✅ [INFERENCE] Matcher updated for new recipe: {current_recipe_name}")
                                else:
                                    logger.error(f"❌ [INFERENCE] Failed to load new recipe")

                            # Update trigger config
                            trigger_config = new_settings.get('trigger_config', {})
                            new_trigger_mode = trigger_config.get('mode', 'continuous')
                            if new_trigger_mode != trigger_mode:
                                logger.info(f"🔄 [MODE CHANGE] Switching from '{trigger_mode}' to '{new_trigger_mode}'")
                                trigger_mode = new_trigger_mode
                                trigger_activation = trigger_config.get('trigger_activation', 'RisingEdge')
                                di_number = trigger_config.get('di_number', 0)

                                if trigger_mode == 'hardware':
                                    previous_di_value = read_di_value(di_number)
                                    logger.info(f"🔧 [HARDWARE TRIGGER] Monitoring DI {di_number}, activation: {trigger_activation}")

                            last_mtime = current_mtime
                            logger.info(f"✅ [HOT RELOAD SUCCESS] Camera {serial_number} settings updated!")
                    except Exception as e:
                        logger.error(f"❌ [HOT RELOAD ERROR] Camera {serial_number}: {e}")

            # Check if we should capture based on trigger mode
            should_capture = False

            if trigger_mode == 'continuous':
                # Continuous mode: always capture
                should_capture = True

            elif trigger_mode == 'hardware':
                # Hardware trigger: check DI for edge HOẶC trigger file - chỉ chụp 1 frame khi có edge

                # Check trigger file trước (cho test API)
                trigger_file = Path(f"/tmp/camera_trigger_{serial_number}")
                if trigger_file.exists():
                    should_capture = True
                    trigger_file.unlink()
                    logger.info(f"⚡ [TRIGGER] Software trigger file detected - capturing frame")
                else:
                    # Check hardware DI
                    current_di_value = read_di_value(di_number)
                    if previous_di_value is not None:
                        if check_trigger_edge(current_di_value, previous_di_value, trigger_activation):
                            should_capture = True
                            logger.info(f"⚡ [TRIGGER] Hardware trigger detected on DI {di_number}: {previous_di_value}→{current_di_value}")
                    previous_di_value = current_di_value

                # Nếu không có trigger, đợi và skip frame này
                if not should_capture:
                    time.sleep(0.01)
                    continue

            elif trigger_mode == 'software':
                # Software trigger: check for trigger file or flag (to be implemented)
                # For now, skip capturing in software trigger mode
                time.sleep(0.01)
                continue

            if not should_capture:
                continue

            # Debug: Log that we're attempting to grab
            if frame_idx == 0:
                logger.info(f"🎥 [GRAB START] Attempting first frame grab...")

            grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)

            if grabResult and grabResult.GrabSucceeded():
                image = converter.Convert(grabResult)
                img_array = image.GetArray()

                frame_idx += 1

                metadata = {
                    'serial_number': serial_number,
                    'model_name': model_name,
                    'timestamp': time.time(),
                    'frame_idx': frame_idx,
                    'shape': img_array.shape,
                    'dtype': str(img_array.dtype)
                }

                metadata_bytes = pickle.dumps(metadata)
                metadata_len = len(metadata_bytes)

                frame_bytes = img_array.tobytes()
                frame_len = len(frame_bytes)

                offset = 0

                # Ghi frame version/sequence number (8 bytes) - để detect race condition
                struct.pack_into('<Q', shm.buf, offset, frame_idx)
                offset += 8

                # Ghi metadata length (4 bytes)
                struct.pack_into('<I', shm.buf, offset, metadata_len)
                offset += 4

                # Ghi metadata bytes - trực tiếp vào buffer
                shm.buf[offset:offset+metadata_len] = metadata_bytes
                offset += metadata_len

                # Ghi frame length (4 bytes)
                struct.pack_into('<I', shm.buf, offset, frame_len)
                offset += 4

                # Ghi frame bytes - trực tiếp vào buffer
                shm.buf[offset:offset+frame_len] = frame_bytes
                offset += frame_len

                # Ghi frame version ở cuối (8 bytes) - để verify data integrity
                struct.pack_into('<Q', shm.buf, offset, frame_idx)

                if frame_idx == 1 or frame_idx % 100 == 0:
                    logger.info(f"✅ [FRAME WRITTEN] Frame #{frame_idx} written to shared memory (size: {frame_len} bytes)")

                # Run inference if matcher initialized and in hardware trigger mode
                if inference_matcher and trigger_mode == 'hardware' and should_capture:
                    try:
                        logger.info(f"🔍 [INFERENCE] Running TensorRT matching for frame #{frame_idx}...")

                        # Save frame temporarily
                        temp_dir = Path("ocr_inference")
                        temp_frame_path = temp_dir / f"trigger_frame_{serial_number}_{frame_idx}.jpg"
                        cv2.imwrite(str(temp_frame_path), img_array)

                        # Run TensorRT matching
                        result = inference_matcher.match(
                            target_path=str(temp_frame_path),
                            score_threshold=0.3,
                            ransac_threshold=5.0
                        )

                        if result['success']:
                            # Visualize result
                            output_path = temp_dir / f"result_{serial_number}_{frame_idx}.jpg"

                            # Draw matches and bounding boxes on frame
                            vis_img = img_array.copy()
                            if result.get('transformed_bboxes'):
                                for bbox_data in result['transformed_bboxes']:
                                    pts = np.array(bbox_data['points'], dtype=np.int32)
                                    cv2.polylines(vis_img, [pts], True, (0, 255, 0), 2)

                                    # Draw text label
                                    text = bbox_data.get('text', bbox_data.get('type', ''))
                                    if text:
                                        cv2.putText(vis_img, text, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            cv2.imwrite(str(output_path), vis_img)

                            logger.info(f"✅ [INFERENCE] Success! Recipe: {current_recipe_name}, Confidence: {result['confidence']:.1%}, Output: {output_path}")
                        else:
                            logger.warning(f"⚠️ [INFERENCE] Matching failed: {result.get('error')}")

                    except Exception as e:
                        logger.error(f"❌ [INFERENCE] Error: {e}")
                        import traceback
                        logger.error(traceback.format_exc())

                grabResult.Release()
                time.sleep(0.033)

            else:
                logger.warning(f"⚠️ [GRAB FAILED] Frame grab failed or timed out")
                if grabResult:
                    grabResult.Release()
                time.sleep(0.01)

        camera.StopGrabbing()
        camera.Close()
        shm.close()
        shm.unlink()

    except KeyboardInterrupt:
        logger.info("Stopped by user")
        try:
            shm.close()
            shm.unlink()
        except:
            pass
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys

    # Get device_index from command line argument, default to 0
    device_index = 0
    if len(sys.argv) > 1:
        try:
            device_index = int(sys.argv[1])
        except ValueError:
            logger.error(f"Invalid device_index argument: {sys.argv[1]}, using default 0")

    logger.info(f"Starting camera producer with device_index={device_index}")
    camera_producer(device_index=device_index)
