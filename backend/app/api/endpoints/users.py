from fastapi import APIRouter, Depends, HTTPException, status, Request
from typing import List
from app.db.mongodb import get_database
from app.repositories.user_repository import UserRepository
from app.repositories.action_log_repository import ActionLogRepository
from app.models.user import UserCreate, UserUpdate, UserResponse, UserChangePassword, UserInDB
from app.models.action_log import ActionLogCreate, ActionType
from app.api.dependencies.auth import get_current_user, require_admin, require_supervisor
from app.api.dependencies.action_log import get_action_log_repository

router = APIRouter()


@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_create: UserCreate,
    request: Request,
    db=Depends(get_database),
    current_user: UserInDB = Depends(require_admin),
    action_log_repo: ActionLogRepository = Depends(get_action_log_repository)
):
    """
    Create new user - ADMIN only

    - **username**: Unique username (3-50 chars)
    - **email**: Valid email address
    - **full_name**: User's full name
    - **role**: operator, supervisor, or admin
    - **password**: Password (min 6 chars)
    """
    user_repo = UserRepository(db)

    # Check if username already exists
    existing_user = await user_repo.get_user_by_username(user_create.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )

    # Create user
    user = await user_repo.create_user(user_create)

    # Log the action
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    action_log = ActionLogCreate(
        user_id=current_user.id,
        username=current_user.username,
        action_type=ActionType.CREATE_USER,
        resource_type="user",
        resource_id=user.id,
        description=f"Created user '{user.username}' with role '{user.role}'",
        new_value={
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
            "is_active": user.is_active
        },
        ip_address=client_ip,
        user_agent=user_agent
    )
    await action_log_repo.create_action_log(action_log)

    return UserResponse(**user.model_dump())


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: UserInDB = Depends(get_current_user)):
    """Get current logged-in user information"""
    return UserResponse(**current_user.model_dump())


@router.put("/me", response_model=UserResponse)
async def update_current_user_profile(
    user_update: UserUpdate,
    db=Depends(get_database),
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Update current user's own profile

    Users can update their own:
    - **email**: Email address
    - **full_name**: Full name
    - **phone_number**: Phone number
    - **avatar_url**: Avatar image URL

    Note: Users cannot change their own role or is_active status
    """
    # Prevent users from changing their own role and active status
    update_data = user_update.model_dump(exclude_unset=True)
    if 'role' in update_data:
        del update_data['role']
    if 'is_active' in update_data:
        del update_data['is_active']

    # Create filtered update object
    filtered_update = UserUpdate(**update_data)

    user_repo = UserRepository(db)
    updated_user = await user_repo.update_user(current_user.id, filtered_update)

    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Failed to update profile"
        )

    return UserResponse(**updated_user.model_dump())


@router.get("/", response_model=List[UserResponse])
async def get_all_users(
    skip: int = 0,
    limit: int = 100,
    db=Depends(get_database),
    current_user: UserInDB = Depends(require_supervisor)
):
    """
    Get all users - SUPERVISOR and ADMIN only

    - **skip**: Number of records to skip (pagination)
    - **limit**: Maximum number of records to return
    """
    user_repo = UserRepository(db)
    users = await user_repo.get_all_users(skip=skip, limit=limit)

    return [UserResponse(**user.model_dump()) for user in users]


@router.get("/simple", response_model=List[dict])
async def get_simple_users(
    skip: int = 0,
    limit: int = 100,
    db=Depends(get_database),
    current_user: UserInDB = Depends(require_supervisor)
):
    """
    Return a lightweight list of users for selection controls.

    Permissions: Supervisor and Admin only.
    Returns items with `id`, `username`, and `full_name`.
    """
    user_repo = UserRepository(db)
    users = await user_repo.get_all_users(skip=skip, limit=limit)
    return [
        {"id": u.id, "username": u.username, "full_name": u.full_name}
        for u in users
    ]


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    db=Depends(get_database),
    current_user: UserInDB = Depends(require_supervisor)
):
    """Get user by ID - SUPERVISOR and ADMIN only"""
    user_repo = UserRepository(db)
    user = await user_repo.get_user_by_id(user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return UserResponse(**user.model_dump())


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    user_update: UserUpdate,
    request: Request,
    db=Depends(get_database),
    current_user: UserInDB = Depends(require_admin),
    action_log_repo: ActionLogRepository = Depends(get_action_log_repository)
):
    """
    Update user - ADMIN only

    - **email**: New email (optional)
    - **full_name**: New full name (optional)
    - **role**: New role (optional)
    - **is_active**: Active status (optional)
    """
    user_repo = UserRepository(db)

    # Get the user before update for logging
    old_user = await user_repo.get_user_by_id(user_id)
    if not old_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    user = await user_repo.update_user(user_id, user_update)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Log the action
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    # Prepare old and new values for logging
    old_value = {
        "username": old_user.username,
        "email": old_user.email,
        "full_name": old_user.full_name,
        "role": old_user.role,
        "is_active": old_user.is_active
    }

    new_value = {}
    update_dict = user_update.model_dump(exclude_unset=True)
    for field in update_dict:
        if hasattr(user, field):
            new_value[field] = getattr(user, field)

    action_log = ActionLogCreate(
        user_id=current_user.id,
        username=current_user.username,
        action_type=ActionType.UPDATE_USER,
        resource_type="user",
        resource_id=user.id,
        description=f"Updated user '{user.username}'",
        old_value=old_value,
        new_value=new_value,
        ip_address=client_ip,
        user_agent=user_agent
    )
    await action_log_repo.create_action_log(action_log)

    return UserResponse(**user.model_dump())


@router.post("/change-password")
async def change_own_password(
    password_change: UserChangePassword,
    db=Depends(get_database),
    current_user: UserInDB = Depends(get_current_user)
):
    """Change current user's password"""
    user_repo = UserRepository(db)

    # Verify old password
    from app.core.security import verify_password
    if not verify_password(password_change.old_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )

    # Update password
    success = await user_repo.update_password(current_user.id, password_change.new_password)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update password"
        )

    return {"message": "Password updated successfully"}


@router.post("/{user_id}/reset-password")
async def reset_user_password(
    user_id: str,
    new_password: str,
    request: Request,
    db=Depends(get_database),
    current_user: UserInDB = Depends(require_admin),
    action_log_repo: ActionLogRepository = Depends(get_action_log_repository)
):
    """
    Reset user password - ADMIN only

    - **user_id**: ID of user to reset password for
    - **new_password**: New password
    """
    user_repo = UserRepository(db)

    # Check user exists
    user = await user_repo.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Update password
    success = await user_repo.update_password(user_id, new_password)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset password"
        )

    # Log the action
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    action_log = ActionLogCreate(
        user_id=current_user.id,
        username=current_user.username,
        action_type=ActionType.RESET_USER_PASSWORD,
        resource_type="user",
        resource_id=user.id,
        description=f"Reset password for user '{user.username}'",
        ip_address=client_ip,
        user_agent=user_agent
    )
    await action_log_repo.create_action_log(action_log)

    return {"message": f"Password reset successfully for user: {user.username}"}


@router.delete("/{user_id}")
async def delete_user(
    user_id: str,
    request: Request,
    db=Depends(get_database),
    current_user: UserInDB = Depends(require_admin),
    action_log_repo: ActionLogRepository = Depends(get_action_log_repository)
):
    """Delete user - ADMIN only"""
    user_repo = UserRepository(db)

    # Get user info before deletion for logging
    user = await user_repo.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Cannot delete yourself
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )

    success = await user_repo.delete_user(user_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Log the action
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    action_log = ActionLogCreate(
        user_id=current_user.id,
        username=current_user.username,
        action_type=ActionType.DELETE_USER,
        resource_type="user",
        resource_id=user_id,
        description=f"Deleted user '{user.username}' with role '{user.role}'",
        old_value={
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
            "is_active": user.is_active
        },
        ip_address=client_ip,
        user_agent=user_agent
    )
    await action_log_repo.create_action_log(action_log)

    return {"message": "User deleted successfully"}
