"""
Turning inspection frames into things other systems consume: display-sized
JPEG/base64 for the UI, and permanent on-disk copies (original + annotated).

Also holds the two background workers that keep that work off the inference
path — AsyncImageSaver (disk writes) and BackgroundResultEmitter (encode +
socket emit).
"""

import atexit
import base64
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .visualization import (
    draw_center_points,
    draw_color_match_overlay,
    draw_detected_obb_boxes,
    draw_inference_bboxes,
)

logger = logging.getLogger(__name__)

# ============= Image Processing Utilities =============

def resize_for_display(frame_img, scale_factor: int = 3):
    """
    Resize frame by dividing dimensions

    Args:
        frame_img: Input frame (numpy array)
        scale_factor: Division factor for width and height

    Returns:
        Resized frame
    """
    h, w = frame_img.shape[:2]
    new_w = w // scale_factor
    new_h = h // scale_factor
    return cv2.resize(frame_img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def encode_image_to_base64(img, quality: int = 70) -> str:
    """
    Encode image to JPEG base64

    Args:
        img: Input image (numpy array)
        quality: JPEG quality (0-100)

    Returns:
        Base64 encoded string
    """
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')


def encode_frame_for_display(
    frame_img: np.ndarray,
    transformed_bboxes: Optional[List[Dict[str, Any]]] = None,
    confidence: float = 0.0,
    inliers: int = 0,
    total_matches: int = 0,
    crop_area: Optional[Dict[str, int]] = None,
    product_verification: Optional[Dict[str, Any]] = None,
    scale_factor: int = 3,
    quality: int = 70
) -> Optional[str]:
    try:
        # Draw bboxes if provided
        img_to_encode = frame_img.copy()
        if transformed_bboxes and len(transformed_bboxes) > 0:
            img_to_encode = draw_inference_bboxes(
                img_to_encode,
                transformed_bboxes,
                confidence,
                inliers,
                total_matches,
                crop_area
            )

        # Draw detected OBB boxes from YOLO (if product verification was performed)
        if product_verification and not product_verification.get('skipped', False):
            detected_boxes = product_verification.get('detected_boxes')

            # Color match overlay (Check_Color path): paint yellow on matching
            # HSV pixels inside the bottle bbox BEFORE drawing OBB outlines so
            # the bbox edges remain crisp on top.
            color_check = product_verification.get('color_check')
            if color_check and detected_boxes:
                img_to_encode = draw_color_match_overlay(
                    img_to_encode, color_check, detected_boxes
                )

            if detected_boxes:
                try:
                    img_to_encode = draw_detected_obb_boxes(
                        img_to_encode,
                        detected_boxes,
                        show_details=True
                    )
                except Exception as draw_err:
                    logger.warning(f"draw_detected_obb_boxes failed, skipping overlay: {draw_err}", exc_info=True)

            # Draw center points (template center and product center)
            center_alignment_check = product_verification.get('center_alignment_check')
            if center_alignment_check:
                img_to_encode = draw_center_points(
                    img_to_encode,
                    center_alignment_check
                )

        # Resize for display
        display_img = resize_for_display(img_to_encode, scale_factor=scale_factor)

        # Encode to base64
        image_base64 = encode_image_to_base64(display_img, quality=quality)

        return image_base64

    except Exception as e:
        logger.error(f"Error encoding frame for display: {e}", exc_info=True)
        return None


# ============= Async Image Saver =============

class AsyncImageSaver:
    """
    Singleton class for async image saving using ThreadPoolExecutor.

    Benefits:
    - Save images to disk without blocking the main inference pipeline
    - Encode base64 synchronously for real-time display
    - Multiple cameras save in parallel
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, max_workers: int = 4):
        """
        Initialize AsyncImageSaver.

        Args:
            max_workers: Maximum number of concurrent save operations
        """
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="AsyncImageSaver"
        )
        self._pending_count = 0
        self._count_lock = threading.Lock()
        self._shutdown = False

        # Register cleanup on exit
        atexit.register(self.shutdown)

        logger.info(f"AsyncImageSaver initialized with {max_workers} workers")

    @classmethod
    def get_instance(cls, max_workers: int = 4) -> 'AsyncImageSaver':
        """Get singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = AsyncImageSaver(max_workers=max_workers)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)"""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.shutdown(wait=True)
                cls._instance = None

    def save_images_async(
        self,
        img_viz: np.ndarray,
        img_org: np.ndarray,
        path_viz: Path,
        path_org: Path,
        quality: int = 95
    ) -> None:
        """
        Submit image save task to background thread.

        Args:
            img_viz: Visualization image (with bboxes)
            img_org: Original image (no bboxes)
            path_viz: Path for visualization image
            path_org: Path for original image
            quality: JPEG quality (1-100)
        """
        if self._shutdown:
            logger.warning("AsyncImageSaver is shutdown, saving synchronously")
            self._do_save(img_viz, img_org, path_viz, path_org, quality)
            return

        with self._count_lock:
            self._pending_count += 1

        self.executor.submit(
            self._do_save,
            img_viz, img_org, path_viz, path_org, quality
        )

    def _do_save(
        self,
        img_viz: np.ndarray,
        img_org: np.ndarray,
        path_viz: Path,
        path_org: Path,
        quality: int
    ) -> None:
        """
        Actually save images to disk (runs in background thread).

        Args:
            img_viz: Visualization image
            img_org: Original image
            path_viz: Path for viz image
            path_org: Path for org image
            quality: JPEG quality
        """
        import time
        t_start = time.perf_counter()
        try:
            cv2.imwrite(str(path_viz), img_viz, [cv2.IMWRITE_JPEG_QUALITY, quality])
            cv2.imwrite(str(path_org), img_org, [cv2.IMWRITE_JPEG_QUALITY, quality])
            t_elapsed = (time.perf_counter() - t_start) * 1000
            logger.debug(f"AsyncImageSaver: saved {path_viz.name} in {t_elapsed:.1f}ms (pending: {self._pending_count - 1})")
        except Exception as e:
            logger.error(f"AsyncImageSaver error saving images: {e}")
        finally:
            with self._count_lock:
                self._pending_count -= 1

    def get_pending_count(self) -> int:
        """Get number of pending save operations"""
        with self._count_lock:
            return self._pending_count

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the executor.

        Args:
            wait: Whether to wait for pending tasks to complete
        """
        if self._shutdown:
            return

        self._shutdown = True
        pending = self.get_pending_count()

        if pending > 0:
            logger.info(f"AsyncImageSaver shutting down, waiting for {pending} pending saves...")

        self.executor.shutdown(wait=wait)
        logger.info("AsyncImageSaver shutdown complete")


# ============= Background Result Emitter =============

class BackgroundResultEmitter:
    """
    Background worker for encoding images and emitting results.

    This allows the inference pipeline to complete immediately after verification,
    while encoding and emitting happens in background threads.

    Benefits:
    - Pipeline complete time excludes encoding time (~38ms for 3 cameras)
    - Encoding can run in parallel for multiple cameras
    - Real-time display still gets images, just slightly delayed
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, max_workers: int = 4):
        """
        Initialize BackgroundResultEmitter.

        Args:
            max_workers: Maximum concurrent encoding operations
        """
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ResultEmitter"
        )
        self._pending_count = 0
        self._count_lock = threading.Lock()
        self._shutdown = False
        self._emit_callback = None
        self._event_loop = None

        # Register cleanup on exit
        atexit.register(self.shutdown)

        logger.info(f"BackgroundResultEmitter initialized with {max_workers} workers")

    @classmethod
    def get_instance(cls, max_workers: int = 4) -> 'BackgroundResultEmitter':
        """Get singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = BackgroundResultEmitter(max_workers=max_workers)
        return cls._instance

    def set_emit_callback(self, callback, event_loop=None):
        """
        Set the emit callback function.

        Args:
            callback: Function to call with (event_type, data)
            event_loop: Asyncio event loop for async callbacks
        """
        self._emit_callback = callback
        self._event_loop = event_loop

    def emit_result_async(
        self,
        result_data: Dict[str, Any],
        encode_tasks: List[Dict[str, Any]],
        save_and_encode_func,
        encode_display_func
    ) -> None:
        """
        Queue result for background encoding and emit.

        Args:
            result_data: Base result dict (will be modified with encoded images)
            encode_tasks: List of encoding tasks with frame data
            save_and_encode_func: Function to save and encode FAIL frames
            encode_display_func: Function to encode PASS frames for display
        """
        if self._shutdown:
            logger.warning("BackgroundResultEmitter is shutdown, processing synchronously")
            self._process_and_emit(result_data, encode_tasks, save_and_encode_func, encode_display_func)
            return

        with self._count_lock:
            self._pending_count += 1

        self.executor.submit(
            self._process_and_emit,
            result_data, encode_tasks, save_and_encode_func, encode_display_func
        )

    def _process_and_emit(
        self,
        result_data: Dict[str, Any],
        encode_tasks: List[Dict[str, Any]],
        save_and_encode_func,
        encode_display_func
    ) -> None:
        """
        Process encoding tasks and emit result (runs in background thread).

        Args:
            result_data: Result dict to update and emit
            encode_tasks: List of encoding tasks
            save_and_encode_func: Function for FAIL frames
            encode_display_func: Function for PASS frames
        """
        import time
        t_start = time.perf_counter()

        try:
            # Encode all frames in parallel using nested ThreadPoolExecutor
            if len(encode_tasks) > 1:
                encoded_results = self._encode_parallel(
                    encode_tasks, save_and_encode_func, encode_display_func
                )
            else:
                encoded_results = self._encode_sequential(
                    encode_tasks, save_and_encode_func, encode_display_func
                )

            # Update result_data with encoded images
            for task, (image_path, image_base64) in zip(encode_tasks, encoded_results):
                camera_idx = task['camera_idx']
                frame_idx = task['frame_idx']

                if camera_idx < len(result_data.get('camera_results', [])):
                    camera_result = result_data['camera_results'][camera_idx]
                    if 'frames' in camera_result and frame_idx < len(camera_result['frames']):
                        camera_result['frames'][frame_idx]['image_path'] = image_path
                        camera_result['frames'][frame_idx]['image_base64'] = image_base64

            t_encode = (time.perf_counter() - t_start) * 1000

            # Emit result
            if self._emit_callback:
                if self._event_loop:
                    self._event_loop.call_soon_threadsafe(
                        lambda: self._event_loop.create_task(
                            self._emit_callback("inference_result", result_data)
                        )
                    )
                else:
                    self._emit_callback("inference_result", result_data)

            t_total = (time.perf_counter() - t_start) * 1000
            logger.info(
                f"BackgroundResultEmitter: encoded {len(encode_tasks)} frames in {t_encode:.1f}ms, "
                f"total={t_total:.1f}ms (pending: {self._pending_count - 1})"
            )

        except Exception as e:
            logger.error(f"BackgroundResultEmitter error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            with self._count_lock:
                self._pending_count -= 1

    def _encode_sequential(
        self,
        encode_tasks: List[Dict[str, Any]],
        save_and_encode_func,
        encode_display_func
    ) -> List[tuple]:
        """Encode frames sequentially"""
        results = []
        for task in encode_tasks:
            result = self._encode_single_frame(task, save_and_encode_func, encode_display_func)
            results.append(result)
        return results

    def _encode_parallel(
        self,
        encode_tasks: List[Dict[str, Any]],
        save_and_encode_func,
        encode_display_func
    ) -> List[tuple]:
        """Encode frames in parallel"""
        results = [None] * len(encode_tasks)

        with ThreadPoolExecutor(max_workers=len(encode_tasks)) as executor:
            futures = {
                executor.submit(
                    self._encode_single_frame, task, save_and_encode_func, encode_display_func
                ): idx
                for idx, task in enumerate(encode_tasks)
            }

            for future in futures:
                try:
                    idx = futures[future]
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Encode error: {e}")
                    results[futures[future]] = (None, None)

        return results

    def _encode_single_frame(
        self,
        task: Dict[str, Any],
        save_and_encode_func,
        encode_display_func
    ) -> tuple:
        """Encode a single frame"""
        pass_fail = task.get('pass_fail', 'PASS')

        if pass_fail in ['FAIL', 'ERROR']:
            return save_and_encode_func(
                frame_img=task['frame_img'],
                serial_number=task['serial_number'],
                recipe_id=task['recipe_id'],
                pass_fail=pass_fail,
                frame_idx=task['frame_idx'],
                transformed_bboxes=task.get('transformed_bboxes'),
                confidence=task.get('confidence', 0.0),
                inliers=task.get('inliers', 0),
                total_matches=task.get('total_matches', 0),
                crop_area=task.get('crop_area'),
                product_verification=task.get('product_verification')
            )
        else:
            image_base64 = encode_display_func(
                frame_img=task['frame_img'],
                transformed_bboxes=task.get('transformed_bboxes'),
                confidence=task.get('confidence', 0.0),
                inliers=task.get('inliers', 0),
                total_matches=task.get('total_matches', 0),
                crop_area=task.get('crop_area'),
                product_verification=task.get('product_verification')
            )
            return (None, image_base64)

    def get_pending_count(self) -> int:
        """Get number of pending emit operations"""
        with self._count_lock:
            return self._pending_count

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the emitter"""
        if self._shutdown:
            return

        self._shutdown = True
        pending = self.get_pending_count()

        if pending > 0:
            logger.info(f"BackgroundResultEmitter shutting down, waiting for {pending} pending emits...")

        self.executor.shutdown(wait=wait)
        logger.info("BackgroundResultEmitter shutdown complete")



def save_and_encode_frame(
    frame_img,
    serial_number: str,
    recipe_id: str,
    pass_fail: str,
    frame_idx: int,
    base_dir: str = "/home/demo/Source/ocr_datecode/backend/uploads/inference_results",
    transformed_bboxes: Optional[List[Dict[str, Any]]] = None,
    confidence: float = 0.0,
    inliers: int = 0,
    total_matches: int = 0,
    crop_area: Optional[Dict[str, int]] = None,
    product_verification: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Save frame DIRECTLY to permanent storage and create resized version for display
    Optionally draw inference bboxes and crop_area on the image

    Args:
        frame_img: Input frame (numpy array)
        serial_number: Camera serial number
        recipe_id: Recipe ID (for directory structure)
        pass_fail: Pass/fail status (for filename)
        frame_idx: Frame index
        base_dir: Base directory for permanent storage
        transformed_bboxes: Optional list of bbox dicts from inference
        confidence: Confidence score (for drawing)
        inliers: Number of inliers (for drawing)
        total_matches: Total matches (for drawing)
        crop_area: Optional crop area dict to visualize

    Returns:
        Tuple of (relative_image_path, image_base64) or (None, None) on error
        - relative_image_path: Relative path from uploads/ (e.g., "inference_results/recipe_id/2026-01-04/xxx.jpg")
        - image_base64: Base64 encoded resized image for display
    """
    try:
        # Draw bboxes if provided (for inference frames)
        img_to_save = frame_img.copy()
        if transformed_bboxes and len(transformed_bboxes) > 0:
            img_to_save = draw_inference_bboxes(
                img_to_save,
                transformed_bboxes,
                confidence,
                inliers,
                total_matches,
                crop_area  # Pass crop_area to visualization
            )
            logger.info(f"Drew {len(transformed_bboxes)} bboxes on frame")

        # Draw detected OBB boxes from YOLO (if product verification was performed)
        if product_verification and not product_verification.get('skipped', False):
            detected_boxes = product_verification.get('detected_boxes')

            # Color match overlay (Check_Color path): paint yellow on matching
            # HSV pixels inside the bottle bbox BEFORE drawing OBB outlines so
            # the bbox stays crisp on top of the overlay.
            color_check = product_verification.get('color_check')
            if color_check and detected_boxes:
                img_to_save = draw_color_match_overlay(
                    img_to_save, color_check, detected_boxes
                )
                logger.info(
                    f"Drew color-match overlay (matching={color_check.get('matching_pixels')}/"
                    f"{color_check.get('pixel_threshold')}) on frame"
                )

            if detected_boxes:
                img_to_save = draw_detected_obb_boxes(
                    img_to_save,
                    detected_boxes,
                    show_details=True
                )
                logger.info(f"Drew detected OBB boxes (product + label) on frame")

            # Draw center points (template center and product center)
            center_alignment_check = product_verification.get('center_alignment_check')
            if center_alignment_check:
                img_to_save = draw_center_points(
                    img_to_save,
                    center_alignment_check
                )
                logger.info(f"Drew center points (template + product) on frame")

        # Create directory structure: base_dir/{recipe_id}/{YYYY-MM-DD}/{camera_serial}/
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        storage_dir = Path(base_dir) / recipe_id / today / serial_number
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S%f")
        filename_viz = f"{pass_fail.lower()}_f{frame_idx}_{timestamp}_viz.jpg"
        filename_org = f"{pass_fail.lower()}_f{frame_idx}_{timestamp}_org.jpg"
        full_path_viz = storage_dir / filename_viz
        full_path_org = storage_dir / filename_org

        # Get original dimensions
        h, w = img_to_save.shape[:2]

        # Relative path for DB and API (from uploads/)
        relative_path = f"inference_results/{recipe_id}/{today}/{serial_number}/{filename_viz}"

        # Create RESIZED + COMPRESSED version for realtime display (divide by 3)
        # This is SYNCHRONOUS because we need base64 for immediate response
        display_img = resize_for_display(img_to_save, scale_factor=3)
        image_base64 = encode_image_to_base64(display_img, quality=70)

        # Save FULL resolution to permanent storage ASYNCHRONOUSLY
        # This runs in background thread, doesn't block the pipeline
        async_saver = AsyncImageSaver.get_instance()
        async_saver.save_images_async(
            img_viz=img_to_save,
            img_org=frame_img,
            path_viz=full_path_viz,
            path_org=full_path_org,
            quality=95
        )

        logger.info(
            f"Queued {pass_fail} frame for async save: {relative_path} "
            f"(full: {w}x{h}, display: {display_img.shape[1]}x{display_img.shape[0]})"
        )

        return relative_path, image_base64

    except Exception as e:
        logger.error(f"Error saving/encoding frame: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ============= Reject Action Logging Utilities =============
