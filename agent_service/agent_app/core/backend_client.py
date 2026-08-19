"""
Client HTTP tới backend đang chạy (port 8000).

Lý do tồn tại: sau khi tách process, agent service KHÔNG còn nhìn thấy
`camera_ws_manager` — đó là singleton in-memory sống trong process backend.
Trạng thái "camera service đã kết nối WebSocket chưa" vì vậy phải hỏi backend
qua HTTP.

Dùng client sync vì tool function của LangGraph là sync (chạy trong threadpool).
"""

import logging

import httpx

from agent_app.core.config import settings

logger = logging.getLogger(__name__)


def get_camera_ws_connected() -> bool | None:
    """
    Hỏi backend xem camera_management_service đã nối WebSocket chưa.

    Returns:
        True/False theo backend, hoặc None nếu không gọi được backend
        (backend chết / sai URL) — caller phải phân biệt "chưa kết nối"
        với "không biết".
    """
    url = f"{settings.BACKEND_URL}/api/system/camera-ws-status"
    try:
        with httpx.Client(timeout=settings.BACKEND_TIMEOUT) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return bool(resp.json().get("connected", False))
    except Exception as e:
        logger.warning("Không lấy được camera-ws-status từ backend (%s): %s", url, e)
        return None


def is_backend_reachable() -> bool:
    try:
        with httpx.Client(timeout=settings.BACKEND_TIMEOUT) as client:
            return client.get(f"{settings.BACKEND_URL}/health").status_code == 200
    except Exception:
        return False
