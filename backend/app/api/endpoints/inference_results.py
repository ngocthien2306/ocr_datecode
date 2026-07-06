"""
Inference Results API Endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional
from datetime import datetime
import logging

from app.db.mongodb import get_database
from app.repositories.inference_result_repository import InferenceResultRepository
from app.models.inference_result import InferenceResultResponse
from app.models.statistics import SummaryStatisticsResponse, TimeseriesStatisticsResponse
from app.api.dependencies.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/inference-results", tags=["Inference Results"])


@router.get(
    "/",
    response_model=List[InferenceResultResponse],
    summary="Get all inference results"
)
async def get_inference_results(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    recipe_id: Optional[str] = None,
    pass_fail: Optional[str] = Query(None, regex="^(PASS|FAIL)$"),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    fail_reasons: Optional[str] = Query(None, description="Comma-separated: text,char,template,wrinkled,center"),
    wrinkled_min_area: Optional[float] = Query(None, ge=0, description="Min wrinkle area (px²)"),
    center_direction: Optional[str] = Query(None, regex="^(left|right|any)$", description="Center offset direction"),
    db=Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    """
    Get all inference results with pagination and filters.

    - **fail_reasons**: Comma-separated list — `text`, `char`, `template`, `wrinkled`, `center` (AND logic)
    - **wrinkled_min_area**: Only match wrinkled boxes with area >= value (requires `wrinkled` in fail_reasons)
    - **center_direction**: `left`, `right`, or `any` (requires `center` in fail_reasons)
    """
    repo = InferenceResultRepository(db)

    fail_reason_list = [r.strip() for r in fail_reasons.split(",")] if fail_reasons else None

    results = await repo.get_all(
        skip=skip,
        limit=limit,
        recipe_id=recipe_id,
        pass_fail=pass_fail,
        start_date=start_date,
        end_date=end_date,
        fail_reasons=fail_reason_list,
        wrinkled_min_area=wrinkled_min_area,
        center_direction=center_direction,
    )

    return results


@router.get(
    "/count",
    summary="Get inference results count"
)
async def get_inference_results_count(
    recipe_id: Optional[str] = None,
    pass_fail: Optional[str] = Query(None, regex="^(PASS|FAIL)$"),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    fail_reasons: Optional[str] = Query(None, description="Comma-separated: text,char,template,wrinkled,center"),
    wrinkled_min_area: Optional[float] = Query(None, ge=0),
    center_direction: Optional[str] = Query(None, regex="^(left|right|any)$"),
    db=Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    """
    Get total count of inference results with filters.

    Returns: Count object
    """
    repo = InferenceResultRepository(db)

    fail_reason_list = [r.strip() for r in fail_reasons.split(",")] if fail_reasons else None

    count = await repo.count(
        recipe_id=recipe_id,
        pass_fail=pass_fail,
        start_date=start_date,
        end_date=end_date,
        fail_reasons=fail_reason_list,
        wrinkled_min_area=wrinkled_min_area,
        center_direction=center_direction,
    )

    return {
        "count": count,
        "filters": {
            "recipe_id": recipe_id,
            "pass_fail": pass_fail,
            "start_date": start_date,
            "end_date": end_date,
            "fail_reasons": fail_reason_list,
            "wrinkled_min_area": wrinkled_min_area,
            "center_direction": center_direction,
        }
    }


@router.get(
    "/statistics/summary",
    response_model=SummaryStatisticsResponse,
    summary="Get summary statistics"
)
async def get_summary_statistics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    recipe_id: Optional[str] = None,
    db=Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    """
    Get summary statistics with breakdown by camera and recipe.

    - **start_date**: Start date (VN timezone, ISO format)
    - **end_date**: End date (VN timezone, ISO format)
    - **recipe_id**: Optional filter by recipe ID

    Returns summary with:
    - Overall total, pass, fail counts and pass rate
    - Breakdown by camera (camera_id, serial_number, counts, pass rate)
    - Breakdown by recipe (recipe_id, recipe_name, counts, pass rate)
    """
    # Query directly from database (caching removed)
    repo = InferenceResultRepository(db)
    result = await repo.get_summary_statistics(
        start_date=start_date,
        end_date=end_date,
        recipe_id=recipe_id
    )

    logger.info(f"📊 Summary statistics: {result.total} total, {result.pass_rate}% pass rate")
    return result


@router.get(
    "/statistics/timeseries",
    response_model=TimeseriesStatisticsResponse,
    summary="Get timeseries statistics"
)
async def get_timeseries_statistics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    granularity: str = Query("day", regex="^(minute|hour|day)$"),
    recipe_ids: Optional[str] = None,
    db=Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    """
    Get timeseries statistics by recipe.

    - **start_date**: Start date (VN timezone, ISO format)
    - **end_date**: End date (VN timezone, ISO format)
    - **granularity**: Time granularity - "minute", "hour" or "day" (default: "day")
    - **recipe_ids**: Optional filter by recipes (comma-separated IDs, e.g. "recipe1,recipe2")

    Returns timeseries data with:
    - Timestamp (VN timezone)
    - Total, pass, fail counts and pass rate for each time point
    - Breakdown by recipe
    """
    # Parse comma-separated IDs
    recipe_id_list = [id.strip() for id in recipe_ids.split(",")] if recipe_ids else None

    # Validate recipe count (max 30)
    if recipe_id_list and len(recipe_id_list) > 30:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 30 recipes can be selected"
        )

    # Query directly from database (caching removed)
    repo = InferenceResultRepository(db)
    result = await repo.get_timeseries_statistics(
        start_date=start_date,
        end_date=end_date,
        granularity=granularity,  # type: ignore
        recipe_ids=recipe_id_list
    )

    logger.info(f"📈 Timeseries statistics: {len(result.data)} data points ({granularity})")
    return result


@router.get(
    "/{result_id}",
    response_model=InferenceResultResponse,
    summary="Get inference result by ID"
)
async def get_inference_result(
    result_id: str,
    db=Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    """
    Get a specific inference result by ID.

    Returns: Inference result details
    """
    repo = InferenceResultRepository(db)
    result = await repo.get_by_id(result_id)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inference result with ID '{result_id}' not found"
        )

    return result


@router.delete(
    "/{result_id}",
    summary="Delete inference result"
)
async def delete_inference_result(
    result_id: str,
    db=Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    """
    Delete an inference result by ID.

    Returns: Success status
    """
    logger.info(f"DELETE request for result_id: {result_id}")
    repo = InferenceResultRepository(db)

    # Check if exists
    existing = await repo.get_by_id(result_id)
    if not existing:
        logger.warning(f"Result not found for deletion: {result_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inference result with ID '{result_id}' not found"
        )

    # Delete
    success = await repo.delete_by_id(result_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete inference result"
        )

    return {
        "success": True,
        "message": f"Inference result '{result_id}' deleted successfully"
    }
