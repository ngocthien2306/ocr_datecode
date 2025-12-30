from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from app.db.mongodb import get_database
from app.repositories.user_repository import UserRepository
from app.repositories.action_log_repository import ActionLogRepository
from app.core.security import verify_password, create_access_token
from app.schemas.auth import LoginResponse, Token
from app.models.user import UserResponse, UserInDB
from app.models.action_log import ActionLogCreate, ActionType
from app.api.dependencies.auth import get_current_user
from app.api.dependencies.action_log import get_action_log_repository

router = APIRouter()


@router.post("/login", response_model=LoginResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    request: Request = None,
    db=Depends(get_database),
    action_log_repo: ActionLogRepository = Depends(get_action_log_repository)
):
    """
    Login endpoint - authenticate user and return JWT token

    - **username**: Username
    - **password**: Password
    """
    user_repo = UserRepository(db)
    user = await user_repo.get_user_by_username(form_data.username)

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )

    # Update last login
    await user_repo.update_last_login(user.id)

    # Create access token
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role}
    )

    # Prepare user response
    user_dict = user.model_dump()
    user_dict.pop("hashed_password", None)

    # Log the login action
    client_ip = request.client.host if request and request.client else None
    user_agent = request.headers.get("user-agent") if request else None

    action_log = ActionLogCreate(
        user_id=user.id,
        username=user.username,
        action_type=ActionType.LOGIN,
        resource_type="auth",
        description=f"User '{user.username}' logged in",
        ip_address=client_ip,
        user_agent=user_agent
    )
    await action_log_repo.create_action_log(action_log)

    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        user=user_dict
    )


@router.post("/logout")
async def logout(
    request: Request = None,
    current_user: UserInDB = Depends(get_current_user),
    action_log_repo: ActionLogRepository = Depends(get_action_log_repository)
):
    """
    Logout endpoint - log the logout action

    Note: JWT tokens are stateless, so this endpoint mainly logs the logout action.
    Client should discard the token on their side.
    """
    # Log the logout action
    client_ip = request.client.host if request and request.client else None
    user_agent = request.headers.get("user-agent") if request else None

    action_log = ActionLogCreate(
        user_id=current_user.id,
        username=current_user.username,
        action_type=ActionType.LOGOUT,
        resource_type="auth",
        description=f"User '{current_user.username}' logged out",
        ip_address=client_ip,
        user_agent=user_agent
    )
    await action_log_repo.create_action_log(action_log)

    return {"message": "Logged out successfully"}
