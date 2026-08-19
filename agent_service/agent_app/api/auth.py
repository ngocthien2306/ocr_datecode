"""
Đăng nhập ngay tại agent service.

Cùng đường dẫn, cùng payload, cùng response shape với backend
(`POST /api/auth/login`, form urlencoded) nên client đang trỏ vào :8000 chỉ cần
đổi host là chạy. Token cấp ra dùng được cho cả hai service vì chung SECRET_KEY.

Khác backend một điểm: KHÔNG ghi action log. Bảng `action_logs` là của backend,
agent service chỉ đọc dữ liệu chứ không ghi vào miền của nó.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from agent_app.api.deps import get_current_user
from agent_app.core.security import create_access_token, verify_password
from agent_app.db.mongodb import get_database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Auth"])


class UserOut(BaseModel):
    id: str
    username: str
    role: str
    full_name: str = ""


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


@router.post("/login", response_model=LoginResponse, summary="Đăng nhập, nhận JWT")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    db = get_database()
    user = await db["users"].find_one({"username": form_data.username})

    # Thông báo lỗi giống hệt nhau cho "sai user" và "sai mật khẩu" — không để
    # lộ username nào có tồn tại.
    if not user or not verify_password(form_data.password, user.get("hashed_password", "")):
        logger.warning("Đăng nhập thất bại cho username=%r", form_data.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )

    # Claim phải khớp backend, nếu không token bên này backend sẽ không nhận.
    token = create_access_token({"sub": user["username"], "role": user.get("role", "operator")})

    logger.info("Đăng nhập thành công: %s", user["username"])

    return LoginResponse(
        access_token=token,
        user=UserOut(
            id=str(user["_id"]),
            username=user["username"],
            role=user.get("role", "operator"),
            full_name=user.get("full_name", "") or "",
        ),
    )


@router.get("/me", response_model=UserOut, summary="Thông tin user của token hiện tại")
async def me(current_user: Dict[str, Any] = Depends(get_current_user)):
    return UserOut(**current_user)
