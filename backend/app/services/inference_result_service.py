"""
Inference Result Service
Handles inference results from AI service
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import shutil

from app.models.inference_result import InferenceResultCreate, InferenceResultResponse
from app.repositories.inference_result_repository import InferenceResultRepository

logger = logging.getLogger(__name__)


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
        upload_base_path: str = "/home/demo/Source/ocr_datecode/backend/uploads"
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

            # Process images and update paths
            for camera_result in camera_results:
                serial_number = camera_result.get("serial_number")
                frames = camera_result.get("frames", [])

                for frame in frames:
                    # Move image from temp to permanent storage
                    temp_image_path = frame.get("temp_image_path")
                    if temp_image_path:
                        new_path = self._move_image_to_storage(
                            temp_image_path,
                            recipe_id,
                            serial_number,
                            frame.get("frame_idx", 0),
                            frame.get("pass_fail", "unknown")
                        )
                        if new_path:
                            frame["image_path"] = new_path
                            # Remove temp path from data
                            del frame["temp_image_path"]

            # Determine overall product pass/fail
            # TODO: User will implement custom logic
            # For now, simple logic: if any frame fails, product fails
            product_pass_fail = "PASS"
            for camera_result in camera_results:
                for frame in camera_result.get("frames", []):
                    if frame.get("pass_fail") == "FAIL":
                        product_pass_fail = "FAIL"
                        break
                if product_pass_fail == "FAIL":
                    break

            # Create result document
            result_data = InferenceResultCreate(
                recipe_id=recipe_id,
                recipe_name=recipe_name,
                product_pass_fail=product_pass_fail,
                camera_results=camera_results,
                metadata=data.get("metadata")
            )

            # Save to database
            result = await self.result_repo.create(result_data)

            logger.info(
                f"Inference result saved: ID={result.id}, "
                f"recipe={recipe_name}, result={product_pass_fail}"
            )

            # TODO: Emit SocketIO event to frontend (will be implemented when adding SocketIO)

            return result

        except Exception as e:
            logger.error(f"Error processing inference result: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _move_image_to_storage(
        self,
        temp_path: str,
        recipe_id: str,
        serial_number: str,
        frame_idx: int,
        result: str
    ) -> Optional[str]:
        """
        Move image from temp to permanent storage

        Args:
            temp_path: Temporary image path
            recipe_id: Recipe ID
            serial_number: Camera serial number
            frame_idx: Frame index
            result: Pass or Fail

        Returns:
            Relative path to stored image or None
        """
        try:
            temp_file = Path(temp_path)

            if not temp_file.exists():
                logger.warning(f"Temp image not found: {temp_path}")
                return None

            # Create directory structure: /recipe_id/YYYY-MM-DD/
            today = datetime.utcnow().strftime("%Y-%m-%d")
            storage_dir = self.inference_results_path / recipe_id / today
            storage_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename: serial_timestamp_result.jpg
            timestamp = datetime.utcnow().strftime("%H%M%S%f")
            filename = f"{serial_number}_{timestamp}_{result.lower()}_f{frame_idx}.jpg"

            # Copy file
            dest_path = storage_dir / filename
            shutil.copy(temp_file, dest_path)

            # Return relative path from uploads directory
            relative_path = f"inference_results/{recipe_id}/{today}/{filename}"

            logger.info(f"Image saved: {relative_path}")

            return relative_path

        except Exception as e:
            logger.error(f"Error moving image to storage: {e}")
            return None


# Factory function to create service instance
def create_inference_result_service(db) -> InferenceResultService:
    """Create InferenceResultService instance"""
    result_repo = InferenceResultRepository(db)
    return InferenceResultService(result_repo)
