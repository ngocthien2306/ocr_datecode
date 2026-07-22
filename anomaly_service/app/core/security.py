from typing import Optional

from jose import JWTError, jwt

from app.core.config import settings


def decode_access_token(token: str) -> Optional[dict]:
    """Decode a JWT issued by backend's /api/auth/login.

    Mirrors backend/app/core/security.py::decode_access_token exactly —
    same SECRET_KEY/ALGORITHM — so a token from the main app is valid here
    too, with no separate login for this service.
    """
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        return None
