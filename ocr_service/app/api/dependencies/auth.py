from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from app.core.security import decode_access_token
from app.db.mongodb import get_database

# tokenUrl points at backend's login route — this service issues no tokens
# of its own, it only verifies ones backend already issued.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


class CurrentUser(BaseModel):
    """Minimal user shape — just enough to gate access. Not the full
    backend UserInDB model; avoids cross-service coupling on that schema."""
    username: str
    is_active: bool = True


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db=Depends(get_database),
) -> CurrentUser:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise credentials_exception

    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    username = payload.get("sub")
    if username is None:
        raise credentials_exception

    # Same Mongo DB as backend — read-only lookup against the shared
    # 'users' collection, no separate user store for this service.
    user_doc = await db.get_collection("users").find_one({"username": username})
    if user_doc is None:
        raise credentials_exception
    if not user_doc.get("is_active", True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")

    return CurrentUser(username=username, is_active=True)
