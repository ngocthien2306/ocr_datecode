from typing import Optional
from fastapi import Depends, HTTPException, status, Query
from fastapi.security import OAuth2PasswordBearer
from app.core.security import decode_access_token
from app.db.mongodb import get_database
from app.repositories.user_repository import UserRepository
from app.models.user import UserInDB, UserRole

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db=Depends(get_database)
) -> UserInDB:
    """Get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception

    user_repo = UserRepository(db)
    user = await user_repo.get_user_by_username(username)

    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )

    return user


async def get_current_active_user(
    current_user: UserInDB = Depends(get_current_user)
) -> UserInDB:
    """Verify user is active"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# Role-based access control
class RoleChecker:
    def __init__(self, allowed_roles: list[UserRole]):
        self.allowed_roles = allowed_roles

    def __call__(self, current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
        if current_user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation not permitted. Required roles: {[role.value for role in self.allowed_roles]}"
            )
        return current_user


# Specific role checkers
require_operator = RoleChecker([UserRole.OPERATOR, UserRole.SUPERVISOR, UserRole.ADMIN])
require_supervisor = RoleChecker([UserRole.SUPERVISOR, UserRole.ADMIN])
require_admin = RoleChecker([UserRole.ADMIN])


async def get_current_user_from_query(
    token: Optional[str] = Query(None),
    db=Depends(get_database)
) -> UserInDB:
    """
    Get current authenticated user from query parameter token.
    This is used for endpoints that need to be accessed via <img> tags
    where custom headers cannot be set.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if token is None:
        raise credentials_exception

    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception

    user_repo = UserRepository(db)
    user = await user_repo.get_user_by_username(username)

    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )

    return user
