"""
Inference Pipeline Base - Template Method Pattern

Defines the abstract pipeline structure for inference processing.
Concrete implementations handle specific scenarios (single/multi camera).
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..camera import Camera
    import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """
    Context object passed through pipeline stages.
    Holds all data needed for inference processing.
    """
    # Input data
    job_id: int
    cameras: List['Camera']
    results: Dict[str, Any]  # Captured frames
    camera_matchers: Dict[str, Any]
    group_id: Optional[int] = None
    T_capture_complete: Optional[float] = None

    # Processing state
    cameras_to_process: List['Camera'] = field(default_factory=list)
    camera_inference_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    overall_pass_fail: str = "PASS"

    # Services (injected)
    text_verification_service: Any = None
    template_verification_service: Any = None
    product_verification_service: Any = None
    obb_rotation_service: Any = None
    cv_rotation_service: Any = None

    # Callbacks
    emit_callback: Optional[Callable] = None

    # Timing
    T_inference_start: float = 0.0

    # Statistics from TriggerHandler (passed through for reporting)
    statistics: Dict[str, Any] = field(default_factory=lambda: {
        'total_triggers': 0,
        'total_inferences': 0,
        'total_pass': 0,
        'total_fail': 0,
        'total_reject': 0
    })


class InferencePipelineTemplate(ABC):
    """
    Abstract base class implementing Template Method pattern for inference.

    The pipeline executes in these stages:
    1. prepare() - Setup and validation
    2. preprocess() - Prepare frames for inference
    3. run_inference() - Execute model inference
    4. verify_results() - Text/template verification
    5. postprocess() - Build final results
    6. finalize() - Cleanup and emit results

    Subclasses must implement the abstract methods to handle
    specific scenarios (single camera, multi-camera batch, etc.)
    """

    def process(self, context: PipelineContext) -> Dict[str, Any]:
        """
        Template method - defines the pipeline execution order.
        Do not override this method in subclasses.

        Args:
            context: Pipeline context with all necessary data

        Returns:
            Final inference result dict
        """
        import time
        context.T_inference_start = time.time()

        try:
            # Stage timings — flush at end so we can spot the slow phase quickly.
            # Replaces ad-hoc instrumentation in subclasses.
            t_stages: Dict[str, float] = {}

            # Step 1: Prepare
            t = time.perf_counter()
            if not self.prepare(context):
                return self._build_error_result(context, "Preparation failed")
            t_stages['prepare'] = (time.perf_counter() - t) * 1000

            # Step 2: Preprocess frames
            t = time.perf_counter()
            preprocessed = self.preprocess(context)
            t_stages['preprocess'] = (time.perf_counter() - t) * 1000
            if preprocessed is None:
                return self._build_error_result(context, "Preprocessing failed")

            # Step 3: Run inference
            t = time.perf_counter()
            inference_results = self.run_inference(context, preprocessed)
            t_stages['run_inference'] = (time.perf_counter() - t) * 1000
            if inference_results is None:
                return self._build_error_result(context, "Inference failed")

            # Step 4: Verify results (text, template)
            t = time.perf_counter()
            verified_results = self.verify_results(context, inference_results)
            t_stages['verify_results'] = (time.perf_counter() - t) * 1000

            # Step 5: Postprocess - build final results
            t = time.perf_counter()
            final_result = self.postprocess(context, verified_results)
            t_stages['postprocess'] = (time.perf_counter() - t) * 1000

            # Stash for finalize() to log
            context.results.setdefault('_stage_timings', {})
            context.results['_stage_timings'][context.job_id] = t_stages

            # Step 6: Finalize - emit results, schedule rejects, etc.
            self.finalize(context, final_result)

            return final_result

        except Exception as e:
            logger.error(f"[Job #{context.job_id}] Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            return self._build_error_result(context, str(e))

    # ========== Abstract methods - must be implemented by subclasses ==========

    @abstractmethod
    def prepare(self, context: PipelineContext) -> bool:
        """
        Prepare for inference - validate inputs, filter cameras.

        Args:
            context: Pipeline context

        Returns:
            True if preparation successful, False otherwise
        """
        pass

    @abstractmethod
    def preprocess(self, context: PipelineContext) -> Optional[Dict[str, Any]]:
        """
        Preprocess frames for inference.

        Args:
            context: Pipeline context

        Returns:
            Dict with preprocessed data or None if failed
        """
        pass

    @abstractmethod
    def run_inference(
        self,
        context: PipelineContext,
        preprocessed: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Run model inference on preprocessed data.

        Args:
            context: Pipeline context
            preprocessed: Data from preprocess step

        Returns:
            Dict with inference results or None if failed
        """
        pass

    @abstractmethod
    def verify_results(
        self,
        context: PipelineContext,
        inference_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify inference results (text OCR, template matching).

        Args:
            context: Pipeline context
            inference_results: Results from inference step

        Returns:
            Dict with verified results
        """
        pass

    @abstractmethod
    def postprocess(
        self,
        context: PipelineContext,
        verified_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build final result structure.

        Args:
            context: Pipeline context
            verified_results: Results from verification step

        Returns:
            Final inference result dict
        """
        pass

    # ========== Hook methods - can be overridden ==========

    def finalize(self, context: PipelineContext, result: Dict[str, Any]) -> None:
        """
        Finalize pipeline - emit results, log timing.
        Default implementation emits result via callback.

        Args:
            context: Pipeline context
            result: Final result to emit
        """
        import time

        # Calculate total time
        total_time = time.time() - context.T_inference_start

        # Stage breakdown if available (set in process())
        stage_timings = (context.results.get('_stage_timings') or {}).get(context.job_id, {})
        breakdown_str = ""
        if stage_timings:
            parts = [f"{k}={v:.0f}ms" for k, v in stage_timings.items()]
            breakdown_str = "  STAGES: " + " | ".join(parts)

        logger.info(
            f"[Job #{context.job_id}] Pipeline complete: "
            f"{result.get('product_pass_fail', 'UNKNOWN')}, time: {total_time:.3f}s"
            f"{breakdown_str}"
        )

        # Emit result
        if context.emit_callback:
            context.emit_callback("inference_result", result)

    def _build_error_result(
        self,
        context: PipelineContext,
        error_msg: str
    ) -> Dict[str, Any]:
        """
        Build error result when pipeline fails.

        Args:
            context: Pipeline context
            error_msg: Error message

        Returns:
            Error result dict
        """
        if not context.cameras:
            return {
                "recipe_id": 0,
                "recipe_name": "Unknown",
                "product_pass_fail": "ERROR",
                "error": error_msg,
                "camera_results": [],
                "metadata": {}
            }

        first_camera = context.cameras[0]
        return {
            "recipe_id": first_camera.recipe_id,
            "recipe_name": first_camera.recipe_name,
            "product_pass_fail": "ERROR",
            "error": error_msg,
            "camera_results": [],
            "metadata": {
                "total_cameras": len(context.cameras),
                "total_frames": 0,
                "inference_stats": {}
            }
        }
