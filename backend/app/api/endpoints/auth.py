from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from app.db.mongodb import get_database
from app.repositories.user_repository import UserRepository
from app.core.security import verify_password, create_access_token
from app.schemas.auth import LoginResponse, Token
from app.models.user import UserResponse

router = APIRouter()


@router.post("/login", response_model=LoginResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db=Depends(get_database)
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

    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        user=user_dict
    )
