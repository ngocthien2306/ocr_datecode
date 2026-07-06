"""
Inference Result Service
Handles inference results from AI service
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import shutil

from app.models.inference_result import InferenceResultCreate, InferenceResultResponse
from app.repositories.inference_result_repository import InferenceResultRepository

logger = logging.getLogger(__name__)
home = os.environ.get('HOME')

class InferenceResultService:
    """
    Service to process inference results

    Features:
    - Save results to database
    - Save images to storage
    - Emit events to frontend via SocketIO
    """

    def __init__(
        self,
        result_repository: InferenceResultRepository,
        upload_base_path: str = f"{home}/Source/ocr_datecode/backend/uploads"
    ):
        """
        Initialize service

        Args:
            result_repository: Repository for database operations
            upload_base_path: Base path for file uploads
        """
        self.result_repo = result_repository
        self.upload_base_path = Path(upload_base_path)
        self.inference_results_path = self.upload_base_path / "inference_results"

        # Create directory structure
        self.inference_results_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"InferenceResultService initialized")
        logger.info(f"Upload base path: {self.upload_base_path}")

    async def process_inference_result(self, data: Dict[str, Any]) -> Optional[InferenceResultResponse]:
        """
        Process inference result from AI service

        Args:
            data: Inference result data from WebSocket

        Returns:
            Saved result or None if failed
        """
        try:
            # Extract data
            recipe_id = data.get("recipe_id")
            recipe_name = data.get("recipe_name", "Unknown")
            camera_results = data.get("camera_results", [])

            if not recipe_id:
                logger.error("Missing recipe_id in inference result")
                return None

            # Process images for SocketIO emit
            # Camera service already saved to permanent storage, no file operations needed!
            base64_data = {}  # Store base64 separately for emit

            for camera_idx, camera_result in enumerate(camera_results):
                frames = camera_result.get("frames", [])

                for frame_idx, frame in enumerate(frames):
                    # Save base64 for SocketIO (will be added back after DB save)
                    if frame.get("image_base64"):
                        base64_data[(camera_idx, frame_idx)] = frame.get("image_base64")

                    # Note: image_path is already the correct permanent path from camera service
                    # No file move operation needed!

            # Determine overall product pass/fail
            # Simple logic: if any frame fails, product fails
            product_pass_fail = "PASS"
            for camera_result in camera_results:
                for frame in camera_result.get("frames", []):
                    if frame.get("pass_fail") == "FAIL":
                        product_pass_fail = "FAIL"
                        break
                if product_pass_fail == "FAIL":
                    break

            # Respect an explicit ERROR from the AI service (inference
            # exception, timeout, no matchers, ...). These results have an
            # empty camera_results list, which the frame loop above would
            # otherwise misclassify as PASS.
            metadata = data.get("metadata") or {}
            if data.get("product_pass_fail") == "ERROR" or data.get("error"):
                product_pass_fail = "ERROR"
                if data.get("error"):
                    metadata = {**metadata, "error": data.get("error")}

            # Create result document
            result_data = InferenceResultCreate(
                recipe_id=recipe_id,
                recipe_name=recipe_name,
                product_pass_fail=product_pass_fail,
                camera_results=camera_results,
                statistics=data.get("statistics"),
                metadata=metadata
            )

            # Save to database
            result = await self.result_repo.create(result_data)

            logger.info(
                f"Inference result saved: ID={result.id}, "
                f"recipe={recipe_name}, result={product_pass_fail}"
            )

            # Emit SocketIO event to frontend
            try:
                from app.services.socketio_service import emit_inference_result
                from app.core.config import settings

                result_dict = result.model_dump()
                # Convert ObjectId to string for JSON serialization
                result_dict['id'] = str(result_dict['id'])
                # Convert datetime to ISO string
                if 'timestamp' in result_dict and result_dict['timestamp']:
                    result_dict['timestamp'] = result_dict['timestamp'].isoformat()
                if 'created_at' in result_dict and result_dict['created_at']:
                    result_dict['created_at'] = result_dict['created_at'].isoformat()

                # Add base64 back for SocketIO (not saved in DB)
                for camera_idx, camera_result in enumerate(result_dict.get('camera_results', [])):
                    for frame_idx, frame in enumerate(camera_result.get('frames', [])):
                        # Add base64 if exists
                        if (camera_idx, frame_idx) in base64_data:
                            frame['image_base64'] = base64_data[(camera_idx, frame_idx)]

                        # Add full URL to image paths for frontend fallback
                        if frame.get('image_path'):
                            base_url = settings.API_BASE_URL if hasattr(settings, 'API_BASE_URL') else "http://localhost:8000"
                            frame['image_url'] = f"{base_url}/api/uploads/{frame['image_path']}"

                await emit_inference_result(result_dict)
            except Exception as e:
                logger.error(f"Error emitting SocketIO event: {e}")
                import traceback
                traceback.print_exc()

            return result

        except Exception as e:
            logger.error(f"Error processing inference result: {e}")
            import traceback
            traceback.print_exc()
            return None

    # NOTE: _move_image_to_storage() method removed
    # Camera service now saves directly to permanent storage
    # No file move operation needed in backend!


# Factory function to create service instance
def create_inference_result_service(db) -> InferenceResultService:
    """Create InferenceResultService instance"""
    result_repo = InferenceResultRepository(db)
    return InferenceResultService(result_repo)
