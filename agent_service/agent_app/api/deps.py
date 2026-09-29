"""
Auth dependency.

Tự verify JWT do backend cấp (chung SECRET_KEY) rồi tra user trực tiếp trong
MongoDB — không cần gọi backend, nên agent service vẫn auth được kể cả khi
backend đang restart.

Trả về dict thô thay vì model UserInDB của backend để không phải kéo theo
toàn bộ app.models / app.repositories.
"""

from typing import Any, Dict

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from agent_app.core.security import decode_access_token
from agent_app.db.mongodb import get_database

# tokenUrl trỏ về backend vì token do backend phát hành (chỉ để hiển thị ở /docs)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

_CREDENTIALS_ERROR = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


async def get_current_user_from_query(token: str = "") -> Dict[str, Any]:
    """Xác thực bằng token trong QUERY STRING, dùng cho thẻ <img>.

    Thẻ <img> không gắn được header Authorization, nên ảnh trong Line Station
    trả 401 và ô ảnh trống trơn. Đây là cùng cách backend đã làm cho
    `/api/cameras/{serial}/frame`, không phải một lối tắt mới: token vẫn phải
    hợp lệ, vẫn tra user trong DB, chỉ khác chỗ đọc token.

    Đánh đổi đã biết: token nằm trong URL nên nó vào log truy cập của uvicorn.
    Chấp nhận được ở đây vì đường này chỉ chạy trong tailnet nội bộ và chỉ trả
    ảnh sản phẩm — nhưng KHÔNG được dùng cho đường nào ghi dữ liệu.
    """
    if not token:
        raise _CREDENTIALS_ERROR
    return await _user_from_token(token)


async def _user_from_token(token: str) -> Dict[str, Any]:
    payload = decode_access_token(token)
    if payload is None:
        raise _CREDENTIALS_ERROR

    username = payload.get("sub")
    if not username:
        raise _CREDENTIALS_ERROR

    db = get_database()
    user = await db["users"].find_one({"username": username})
    if user is None:
        raise _CREDENTIALS_ERROR

    if not user.get("is_active", True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")

    return {
        "id": str(user["_id"]),
        "username": user["username"],
        "role": user.get("role", "operator"),
        "full_name": user.get("full_name", ""),
    }


async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    return await _user_from_token(token)
