from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from app.db.mongodb import get_database
from app.repositories.user_repository import UserRepository
from app.models.user import UserCreate, UserUpdate, UserResponse, UserChangePassword, UserInDB
from app.api.dependencies.auth import get_current_user, require_admin, require_supervisor

router = APIRouter()


@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_create: UserCreate,
    db=Depends(get_database),
    current_user: UserInDB = Depends(require_admin)
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
    db=Depends(get_database),
    current_user: UserInDB = Depends(require_admin)
):
    """
    Update user - ADMIN only

    - **email**: New email (optional)
    - **full_name**: New full name (optional)
    - **role**: New role (optional)
    - **is_active**: Active status (optional)
    """
    user_repo = UserRepository(db)
    user = await user_repo.update_user(user_id, user_update)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

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
    db=Depends(get_database),
    current_user: UserInDB = Depends(require_admin)
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

    return {"message": f"Password reset successfully for user: {user.username}"}


@router.delete("/{user_id}")
async def delete_user(
    user_id: str,
    db=Depends(get_database),
    current_user: UserInDB = Depends(require_admin)
):
    """Delete user - ADMIN only"""
    user_repo = UserRepository(db)

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

    return {"message": "User deleted successfully"}
