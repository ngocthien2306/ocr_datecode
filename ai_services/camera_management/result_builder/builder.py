"""
Inference Result Builder

Builder pattern for constructing inference results.
Provides fluent interface for building complex result structures.
"""

import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from ..camera import Camera
    import numpy as np

logger = logging.getLogger(__name__)

home = os.environ.get('HOME')


@dataclass
class FrameResult:
    """Represents a single frame's inference result"""
    frame_idx: int
    template_name: str
    pass_fail: str
    confidence: float
    inliers: int = 0
    total_matches: int = 0
    detected_regions: Optional[List[Dict]] = None
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    timings: Optional[Dict] = None
    text_verification: Optional[Dict] = None
    template_verification: Optional[Dict] = None
    product_verification: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_name": self.template_name,
            "frame_idx": self.frame_idx,
            "pass_fail": self.pass_fail,
            "confidence": self.confidence,
            "detected_regions": self.detected_regions,
            "image_path": self.image_path,
            "image_base64": self.image_base64,
            "timings": self.timings,
            "text_verification": self.text_verification,
            "template_verification": self.template_verification,
            "product_verification": self.product_verification
        }


@dataclass
class CameraResult:
    """Represents a camera's inference results"""
    serial_number: str
    delay_trigger: int
    frames: List[FrameResult] = field(default_factory=list)
    pass_fail: str = "PASS"  # Overall camera pass/fail status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.serial_number,
            "serial_number": self.serial_number,
            "delay_trigger": self.delay_trigger,
            "pass_fail": self.pass_fail,
            "frames": [f.to_dict() for f in self.frames]
        }


class FrameResultBuilder:
    """Builder for frame-level results"""

    def __init__(self, frame_idx: int, template_name: str = ""):
        self._frame_idx = frame_idx
        self._template_name = template_name or f"Template {frame_idx + 1}"
        self._pass_fail = "PASS"
        self._confidence = 0.0
        self._inliers = 0
        self._total_matches = 0
        self._detected_regions = None
        self._image_path = None
        self._image_base64 = None
        self._timings = None
        self._text_verification = None
        self._template_verification = None
        self._product_verification = None
        self._crop_area = None

    def with_inference_data(
        self,
        pass_fail: str,
        confidence: float,
        inliers: int = 0,
        total_matches: int = 0
    ) -> 'FrameResultBuilder':
        """Set basic inference data"""
        self._pass_fail = pass_fail
        self._confidence = confidence
        self._inliers = inliers
        self._total_matches = total_matches
        return self

    def with_regions(self, regions: Optional[List[Dict]]) -> 'FrameResultBuilder':
        """Set detected regions"""
        self._detected_regions = regions
        return self

    def with_timings(self, timings: Optional[Dict]) -> 'FrameResultBuilder':
        """Set timing information"""
        self._timings = timings
        return self

    def with_text_verification(self, verification: Optional[Dict]) -> 'FrameResultBuilder':
        """Set text verification result"""
        self._text_verification = verification
        return self

    def with_template_verification(self, verification: Optional[Dict]) -> 'FrameResultBuilder':
        """Set template verification result"""
        self._template_verification = verification
        return self

    def with_product_verification(self, verification: Optional[Dict]) -> 'FrameResultBuilder':
        """Set product verification result"""
        self._product_verification = verification
        return self

    def with_crop_area(self, crop_area: Optional[Dict]) -> 'FrameResultBuilder':
        """Set crop area for image encoding"""
        self._crop_area = crop_area
        return self

    def with_encoded_image(
        self,
        frame_img: 'np.ndarray',
        serial_number: str,
        recipe_id: int,
        save_and_encode_func,
        encode_display_func
    ) -> 'FrameResultBuilder':
        """
        Encode frame image for display.

        Args:
            frame_img: Frame numpy array
            serial_number: Camera serial number
            recipe_id: Recipe ID
            save_and_encode_func: Function to save and encode frame
            encode_display_func: Function to encode for display only
        """
        if self._pass_fail in ["FAIL", "ERROR"]:
            # Save to disk + encode
            base_dir = f"{home}/Source/ocr_datecode/backend/uploads/inference_results"
            self._image_path, self._image_base64 = save_and_encode_func(
                frame_img=frame_img,
                serial_number=serial_number,
                recipe_id=recipe_id,
                pass_fail=self._pass_fail,
                base_dir=base_dir,
                frame_idx=self._frame_idx,
                transformed_bboxes=self._detected_regions,
                confidence=self._confidence,
                inliers=self._inliers,
                total_matches=self._total_matches,
                crop_area=self._crop_area,
                product_verification=self._product_verification
            )
        else:
            # Encode for display only
            self._image_base64 = encode_display_func(
                frame_img=frame_img,
                transformed_bboxes=self._detected_regions,
                confidence=self._confidence,
                inliers=self._inliers,
                total_matches=self._total_matches,
                crop_area=self._crop_area,
                product_verification=self._product_verification
            )
        return self

    def set_encoded_image(
        self,
        image_path: Optional[str],
        image_base64: Optional[str]
    ) -> 'FrameResultBuilder':
        """
        Directly set encoded image (for parallel encoding).

        Args:
            image_path: Path to saved image (for FAIL frames)
            image_base64: Base64 encoded image for display
        """
        self._image_path = image_path
        self._image_base64 = image_base64
        return self

    def build(self) -> FrameResult:
        """Build the frame result"""
        return FrameResult(
            frame_idx=self._frame_idx,
            template_name=self._template_name,
            pass_fail=self._pass_fail,
            confidence=self._confidence,
            inliers=self._inliers,
            total_matches=self._total_matches,
            detected_regions=self._detected_regions,
            image_path=self._image_path,
            image_base64=self._image_base64,
            timings=self._timings,
            text_verification=self._text_verification,
            template_verification=self._template_verification,
            product_verification=self._product_verification
        )


class CameraResultBuilder:
    """Builder for camera-level results"""

    def __init__(self, serial_number: str, delay_trigger: int = 0):
        self._serial_number = serial_number
        self._delay_trigger = delay_trigger
        self._frames: List[FrameResult] = []

    def add_frame(self, frame: FrameResult) -> 'CameraResultBuilder':
        """Add a frame result"""
        self._frames.append(frame)
        return self

    def add_frames(self, frames: List[FrameResult]) -> 'CameraResultBuilder':
        """Add multiple frame results"""
        self._frames.extend(frames)
        return self

    def build(self) -> CameraResult:
        """Build the camera result"""
        # Determine camera pass/fail based on frames
        # If any frame is FAIL or ERROR, camera is FAIL
        camera_pass_fail = "PASS"
        for frame in self._frames:
            if frame.pass_fail in ["FAIL", "ERROR"]:
                camera_pass_fail = "FAIL"
                break

        return CameraResult(
            serial_number=self._serial_number,
            delay_trigger=self._delay_trigger,
            frames=self._frames,
            pass_fail=camera_pass_fail
        )


class InferenceResultBuilder:
    """
    Builder for complete inference results.

    Provides fluent interface for building the complex result structure
    expected by the backend.
    """

    def __init__(self, recipe_id: int, recipe_name: str):
        self._recipe_id = recipe_id
        self._recipe_name = recipe_name
        self._overall_pass_fail = "PASS"
        self._camera_results: List[CameraResult] = []
        self._total_cameras = 0
        self._total_frames = 0
        self._all_confidences: List[float] = []
        self._all_inliers: List[int] = []
        self._all_total_matches: List[int] = []
        self._overall_timings: Dict = {}
        self._per_camera_stats: List[Dict] = []
        # Statistics tracking (will be set from InferenceHandler)
        self._statistics: Dict[str, Any] = {
            'total_triggers': 0,
            'total_inferences': 0,
            'total_pass': 0,
            'total_fail': 0,
            'total_reject': 0,
            'per_camera': []  # Cumulative per-camera counts from InferenceHandler
        }

    def with_overall_result(self, pass_fail: str) -> 'InferenceResultBuilder':
        """Set overall pass/fail result"""
        self._overall_pass_fail = pass_fail
        return self

    def add_camera_result(self, camera_result: CameraResult) -> 'InferenceResultBuilder':
        """Add a camera result"""
        self._camera_results.append(camera_result)
        self._total_cameras += 1
        self._total_frames += len(camera_result.frames)

        # Aggregate stats
        for frame in camera_result.frames:
            if frame.confidence > 0:
                self._all_confidences.append(frame.confidence)
            self._all_inliers.append(frame.inliers)
            self._all_total_matches.append(frame.total_matches)

            if not self._overall_timings and frame.timings:
                self._overall_timings = frame.timings

        return self

    def add_camera_stats(
        self,
        serial_number: str,
        avg_confidence: float,
        total_inliers: int,
        total_matches: int,
        timings: Dict,
        pass_count: int,
        fail_count: int,
        error_count: int
    ) -> 'InferenceResultBuilder':
        """Add per-camera statistics"""
        # Get cumulative stats from statistics (if available)
        cumulative_stats_for_camera = None
        if 'per_camera' in self._statistics:
            cumulative_stats_for_camera = next(
                (s for s in self._statistics['per_camera'] if s['serial_number'] == serial_number),
                None
            )

        # Determine camera pass/fail (use cumulative if available)
        if cumulative_stats_for_camera:
            camera_pass_fail = cumulative_stats_for_camera['camera_pass_fail']
        else:
            camera_pass_fail = "PASS" if (fail_count == 0 and error_count == 0) else "FAIL"

        camera_stat = {
            "serial_number": serial_number,
            "pass_fail": camera_pass_fail,
            "confidence": avg_confidence,
            "inliers": total_inliers,
            "total_matches": total_matches,
            "timings": timings,
            "frame_stats": {
                "total_frames": pass_count + fail_count + error_count,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "error_count": error_count,
                "avg_confidence": avg_confidence
            }
        }

        # Add cumulative stats if available
        if cumulative_stats_for_camera:
            camera_stat["cumulative_stats"] = {
                "pass_count": cumulative_stats_for_camera['pass_count'],
                "fail_count": cumulative_stats_for_camera['fail_count'],
                "error_count": cumulative_stats_for_camera['error_count']
            }

        self._per_camera_stats.append(camera_stat)

        return self

    def with_statistics(
        self,
        total_triggers: int = 0,
        total_inferences: int = 0,
        total_pass: int = 0,
        total_fail: int = 0,
        total_reject: int = 0
    ) -> 'InferenceResultBuilder':
        """Set global statistics from trigger handler"""
        # Preserve per_camera list if it exists
        per_camera = self._statistics.get('per_camera', [])

        self._statistics = {
            'total_triggers': total_triggers,
            'total_inferences': total_inferences,
            'total_pass': total_pass,
            'total_fail': total_fail,
            'total_reject': total_reject,
            'per_camera': per_camera
        }
        return self

    def build(self) -> Dict[str, Any]:
        """Build the complete inference result"""
        avg_confidence = (
            sum(self._all_confidences) / len(self._all_confidences)
            if self._all_confidences else 0.0
        )

        return {
            "recipe_id": self._recipe_id,
            "recipe_name": self._recipe_name,
            "product_pass_fail": self._overall_pass_fail,
            "camera_results": [cr.to_dict() for cr in self._camera_results],
            "statistics": self._statistics,
            "metadata": {
                "total_cameras": self._total_cameras,
                "total_frames": self._total_frames,
                "inference_stats": {
                    "avg_confidence": avg_confidence,
                    "total_inliers": sum(self._all_inliers),
                    "total_matches": sum(self._all_total_matches),
                    "per_camera_stats": self._per_camera_stats,
                    "overall_timings": self._overall_timings
                }
            }
        }

    @classmethod
    def from_cameras(
        cls,
        cameras: List['Camera'],
        results: Dict[str, Any],
        camera_inference_results: Dict[str, Dict[str, Any]],
        overall_pass_fail: str,
        camera_matchers: Dict[str, Any],
        save_and_encode_func,
        encode_display_func,
        statistics: Optional[Dict[str, Any]] = None,
        parallel_encode: bool = True
    ) -> Dict[str, Any]:
        """
        Build inference result from camera data.

        This is a convenience method that replicates the original
        _build_inference_result logic using the builder pattern.

        Args:
            cameras: List of Camera objects
            results: Capture results dict
            camera_inference_results: Inference results per camera
            overall_pass_fail: Overall pass/fail status
            camera_matchers: Dict of matchers per camera
            save_and_encode_func: Function to save and encode frames
            encode_display_func: Function to encode for display
            statistics: Optional statistics dict from trigger handler
            parallel_encode: Whether to encode images in parallel (default: True)

        Returns:
            Complete inference result dict
        """
        if not cameras:
            return {}

        t_start = time.perf_counter()

        first_camera = cameras[0]
        builder = cls(
            recipe_id=first_camera.recipe_id,
            recipe_name=first_camera.recipe_name
        ).with_overall_result(overall_pass_fail)

        # Apply statistics if provided
        if statistics:
            # Directly set statistics to preserve per_camera from InferenceHandler
            builder._statistics = statistics.copy()

        # Collect all frame builders and encode tasks
        camera_frame_data = []  # List of (camera, camera_builder, [(frame_builder, encode_task), ...])

        for camera in cameras:
            serial_number = camera.serial_number
            camera_frames = results.get(serial_number, {}).get('frames', [])
            camera_inference = camera_inference_results.get(serial_number, {})

            # Get matcher for crop_area
            matcher = camera_matchers.get(serial_number)

            # Build camera result
            camera_builder = CameraResultBuilder(
                serial_number=serial_number,
                delay_trigger=camera.delay_trigger
            )

            # Determine frame processing
            is_multi_frame = 'frames' in camera_inference

            if is_multi_frame:
                num_frames = len(camera_inference['frames'])
            else:
                num_frames = min(len(camera.templates), len(camera_frames))

            frame_builders_with_tasks = []

            # Build frame results (without encoding)
            for idx in range(num_frames):
                if idx >= len(camera_frames):
                    break

                frame_img = camera_frames[idx]
                template_name = (
                    camera.templates[idx].get('name', f'Template {idx+1}')
                    if idx < len(camera.templates) else f'Template {idx+1}'
                )

                frame_builder = FrameResultBuilder(idx, template_name)

                # Get crop_area for this frame
                if isinstance(matcher, list) and idx < len(matcher):
                    crop_area = getattr(matcher[idx], 'crop_area', None)
                elif not isinstance(matcher, list):
                    crop_area = getattr(matcher, 'crop_area', None) if matcher else None
                else:
                    crop_area = None

                frame_builder.with_crop_area(crop_area)

                # Get inference data
                if is_multi_frame:
                    frame_data = camera_inference['frames'][idx]
                    frame_builder.with_inference_data(
                        pass_fail=frame_data['result'],
                        confidence=frame_data['confidence'],
                        inliers=frame_data['inliers'],
                        total_matches=frame_data['total_matches']
                    ).with_regions(
                        frame_data.get('transformed_bboxes')
                    ).with_timings(
                        frame_data.get('timings')
                    ).with_text_verification(
                        frame_data.get('text_verification')
                    ).with_template_verification(
                        frame_data.get('template_verification')
                    ).with_product_verification(
                        frame_data.get('product_verification')
                    )
                elif camera_inference and idx == 0:
                    frame_builder.with_inference_data(
                        pass_fail=camera_inference.get('result', 'PASS'),
                        confidence=camera_inference.get('confidence', 0.0),
                        inliers=camera_inference.get('inliers', 0),
                        total_matches=camera_inference.get('total_matches', 0)
                    ).with_regions(
                        camera_inference.get('transformed_bboxes')
                    ).with_timings(
                        camera_inference.get('timings')
                    ).with_text_verification(
                        camera_inference.get('text_verification')
                    ).with_template_verification(
                        camera_inference.get('template_verification')
                    ).with_product_verification(
                        camera_inference.get('product_verification')
                    )
                else:
                    frame_builder.with_inference_data("PASS", 0.0)

                # Create encode task
                encode_task = {
                    'frame_img': frame_img,
                    'serial_number': serial_number,
                    'recipe_id': camera.recipe_id,
                    'pass_fail': frame_builder._pass_fail,
                    'frame_idx': idx,
                    'transformed_bboxes': frame_builder._detected_regions,
                    'confidence': frame_builder._confidence,
                    'inliers': frame_builder._inliers,
                    'total_matches': frame_builder._total_matches,
                    'crop_area': crop_area,
                    'product_verification': frame_builder._product_verification
                }

                frame_builders_with_tasks.append((frame_builder, encode_task))

            camera_frame_data.append((camera, camera_builder, frame_builders_with_tasks))

        # Collect all encode tasks
        all_encode_tasks = []
        task_to_builder = {}  # Map task index to frame_builder

        for camera, camera_builder, frame_builders_with_tasks in camera_frame_data:
            for frame_builder, encode_task in frame_builders_with_tasks:
                task_idx = len(all_encode_tasks)
                all_encode_tasks.append(encode_task)
                task_to_builder[task_idx] = frame_builder

        # Encode images (parallel or sequential)
        t_encode_start = time.perf_counter()

        if parallel_encode and len(all_encode_tasks) > 1:
            encoded_results = cls._encode_parallel(
                all_encode_tasks, save_and_encode_func, encode_display_func
            )
        else:
            encoded_results = cls._encode_sequential(
                all_encode_tasks, save_and_encode_func, encode_display_func
            )

        t_encode = (time.perf_counter() - t_encode_start) * 1000

        # Apply encoded images to frame builders
        for task_idx, (image_path, image_base64) in enumerate(encoded_results):
            frame_builder = task_to_builder[task_idx]
            frame_builder.set_encoded_image(image_path, image_base64)

        # Build final results
        for camera, camera_builder, frame_builders_with_tasks in camera_frame_data:
            for frame_builder, _ in frame_builders_with_tasks:
                camera_builder.add_frame(frame_builder.build())

            camera_result = camera_builder.build()
            builder.add_camera_result(camera_result)

            # Calculate and add camera stats
            frames = camera_result.frames
            pass_count = sum(1 for f in frames if f.pass_fail == "PASS")
            fail_count = sum(1 for f in frames if f.pass_fail == "FAIL")
            error_count = sum(1 for f in frames if f.pass_fail == "ERROR")
            confidences = [f.confidence for f in frames if f.confidence > 0]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            builder.add_camera_stats(
                serial_number=camera.serial_number,
                avg_confidence=avg_conf,
                total_inliers=sum(f.inliers for f in frames),
                total_matches=sum(f.total_matches for f in frames),
                timings=frames[0].timings if frames and frames[0].timings else {},
                pass_count=pass_count,
                fail_count=fail_count,
                error_count=error_count
            )

        t_total = (time.perf_counter() - t_start) * 1000
        mode = "parallel" if parallel_encode and len(all_encode_tasks) > 1 else "sequential"
        logger.info(
            f"Result builder ({mode}): {len(all_encode_tasks)} frames encoded in {t_encode:.1f}ms, "
            f"total={t_total:.1f}ms"
        )

        return builder.build()

    @classmethod
    def _encode_sequential(
        cls,
        encode_tasks: List[Dict[str, Any]],
        save_and_encode_func,
        encode_display_func
    ) -> List[tuple]:
        """Encode frames sequentially"""
        results = []
        for task in encode_tasks:
            result = cls._encode_single_frame(task, save_and_encode_func, encode_display_func)
            results.append(result)
        return results

    @classmethod
    def _encode_parallel(
        cls,
        encode_tasks: List[Dict[str, Any]],
        save_and_encode_func,
        encode_display_func,
        max_workers: int = 4
    ) -> List[tuple]:
        """Encode frames in parallel using ThreadPoolExecutor"""
        results = [None] * len(encode_tasks)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    cls._encode_single_frame, task, save_and_encode_func, encode_display_func
                ): idx
                for idx, task in enumerate(encode_tasks)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Encode error for task {idx}: {e}")
                    results[idx] = (None, None)

        return results

    @classmethod
    def _encode_single_frame(
        cls,
        task: Dict[str, Any],
        save_and_encode_func,
        encode_display_func
    ) -> tuple:
        """Encode a single frame"""
        pass_fail = task.get('pass_fail', 'PASS')

        if pass_fail in ['FAIL', 'ERROR']:
            base_dir = f"{home}/Source/ocr_datecode/backend/uploads/inference_results"
            return save_and_encode_func(
                frame_img=task['frame_img'],
                serial_number=task['serial_number'],
                recipe_id=task['recipe_id'],
                pass_fail=pass_fail,
                base_dir=base_dir,
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
