"""
Mật khẩu + JWT.

Agent service tự cấp được token (không phải gọi backend :8000), vì nó đã có
sẵn hai thứ cần thiết: SECRET_KEY chung và collection `users` chung.

LƯU Ý VẬN HÀNH: token cấp ở đây dùng được luôn trên backend production, vì cùng
SECRET_KEY. Nghĩa là :8100 là một cửa xác thực thứ hai — cần bảo vệ ngang bằng
:8000 (đừng expose ra mạng ngoài nếu backend không được expose).

Claim phải khớp backend/app/core/security.py: {"sub": username, "role": role, "exp"}.
Sai claim thì token của bên này không dùng được bên kia.
"""

from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from agent_app.core.config import settings

# Khớp backend: bcrypt.
_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _pwd.verify(plain, hashed)
    except Exception:
        # Hash hỏng / thuật toán lạ — coi như sai mật khẩu, không để 500.
        return False


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = dict(data)
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    """Decode JWT access token. Trả None nếu token sai/hết hạn."""
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        return None
