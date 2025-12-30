from fastapi import APIRouter, Depends, Query, HTTPException
from typing import List, Optional
from datetime import datetime
from app.repositories.action_log_repository import ActionLogRepository
from app.models.action_log import ActionLogResponse
from app.api.dependencies.auth import get_current_user
from app.api.dependencies.action_log import get_action_log_repository
from app.models.user import UserInDB


router = APIRouter()


@router.get("/", response_model=List[ActionLogResponse])
async def get_action_logs(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    user_id: Optional[str] = None,
    action_type: Optional[str] = None,
    resource_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: UserInDB = Depends(get_current_user),
    action_log_repo: ActionLogRepository = Depends(get_action_log_repository)
):
    """
    Get action logs with optional filtering.
    Requires authentication.
    """
    # Only admins can view all action logs
    if current_user.role != "admin":
        # Non-admin users can only see their own action logs
        user_id = current_user.id

    action_logs = await action_log_repo.get_action_logs(
        skip=skip,
        limit=limit,
        user_id=user_id,
        action_type=action_type,
        resource_type=resource_type,
        start_date=start_date,
        end_date=end_date
    )

    return action_logs


@router.get("/recent", response_model=List[ActionLogResponse])
async def get_recent_action_logs(
    limit: int = Query(20, ge=1, le=100),
    current_user: UserInDB = Depends(get_current_user),
    action_log_repo: ActionLogRepository = Depends(get_action_log_repository)
):
    """
    Get the most recent action logs.
    Requires authentication.
    """
    action_logs = await action_log_repo.get_recent_action_logs(limit=limit)

    # Filter for non-admin users
    if current_user.role != "admin":
        action_logs = [log for log in action_logs if log.user_id == current_user.id]

    return action_logs


@router.get("/user/{user_id}", response_model=List[ActionLogResponse])
async def get_user_action_logs(
    user_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    current_user: UserInDB = Depends(get_current_user),
    action_log_repo: ActionLogRepository = Depends(get_action_log_repository)
):
    """
    Get action logs for a specific user.
    Only admins can view other users' action logs.
    """
    # Only admins can view other users' action logs
    if current_user.role != "admin" and current_user.id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to view other users' action logs"
        )

    action_logs = await action_log_repo.get_user_action_logs(
        user_id=user_id,
        skip=skip,
        limit=limit
    )

    return action_logs


@router.get("/{action_log_id}", response_model=ActionLogResponse)
async def get_action_log(
    action_log_id: str,
    current_user: UserInDB = Depends(get_current_user),
    action_log_repo: ActionLogRepository = Depends(get_action_log_repository)
):
    """
    Get a specific action log by ID.
    Requires authentication.
    """
    action_log = await action_log_repo.get_action_log_by_id(action_log_id)

    if not action_log:
        raise HTTPException(status_code=404, detail="Action log not found")

    # Only admins can view other users' action logs
    if current_user.role != "admin" and action_log.user_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to view this action log"
        )

    return action_log