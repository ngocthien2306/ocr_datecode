"""
Inference Result Builder

Builder pattern for constructing inference results.
Provides fluent interface for building complex result structures.
"""

import os
import logging
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.serial_number,
            "serial_number": self.serial_number,
            "delay_trigger": self.delay_trigger,
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
                crop_area=self._crop_area
            )
        else:
            # Encode for display only
            self._image_base64 = encode_display_func(
                frame_img=frame_img,
                transformed_bboxes=self._detected_regions,
                confidence=self._confidence,
                inliers=self._inliers,
                total_matches=self._total_matches,
                crop_area=self._crop_area
            )
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
        return CameraResult(
            serial_number=self._serial_number,
            delay_trigger=self._delay_trigger,
            frames=self._frames
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
        self._per_camera_stats.append({
            "serial_number": serial_number,
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
        })
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
        encode_display_func
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

        Returns:
            Complete inference result dict
        """
        if not cameras:
            return {}

        first_camera = cameras[0]
        builder = cls(
            recipe_id=first_camera.recipe_id,
            recipe_name=first_camera.recipe_name
        ).with_overall_result(overall_pass_fail)

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

            # Build frame results
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

                # Encode image
                frame_builder.with_encoded_image(
                    frame_img=frame_img,
                    serial_number=serial_number,
                    recipe_id=camera.recipe_id,
                    save_and_encode_func=save_and_encode_func,
                    encode_display_func=encode_display_func
                )

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
                serial_number=serial_number,
                avg_confidence=avg_conf,
                total_inliers=sum(f.inliers for f in frames),
                total_matches=sum(f.total_matches for f in frames),
                timings=frames[0].timings if frames and frames[0].timings else {},
                pass_count=pass_count,
                fail_count=fail_count,
                error_count=error_count
            )

        return builder.build()
