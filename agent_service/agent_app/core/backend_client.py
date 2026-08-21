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


def get_system_metrics() -> dict:
    """
    Số liệu phần cứng (CPU/GPU/RAM/disk/power) từ backend cùng máy.

    Endpoint `/api/jetson-monitoring/metrics` KHÔNG yêu cầu auth, nên không cần
    token ở đây.

    Trả về dict có `state` thay vì None, vì ba tình huống dưới đây phải phân biệt
    được — gộp cả ba thành "không có dữ liệu" là cách nhanh nhất để giấu mất một
    sự cố thật:

      unreachable — backend chết hoặc sai URL
      not_ready   — backend sống nhưng VÒNG GIÁM SÁT đã chết (HTTP 503). Đây là
                    lỗi có thật và tái diễn: `ocr-backend` không khai báo
                    After=jtop.service nên lúc boot nó thường thắng cuộc đua,
                    jtop() ném JtopException, và code đặt is_monitoring=False rồi
                    thoát nên không bao giờ tự phục hồi. Hồi sinh bằng
                    POST /api/jetson-monitoring/start, không cần restart backend.
      ok          — có số liệu
    """
    url = f"{settings.BACKEND_URL}/api/jetson-monitoring/metrics"
    try:
        with httpx.Client(timeout=settings.BACKEND_TIMEOUT) as client:
            resp = client.get(url)
            if resp.status_code == 503:
                return {"state": "not_ready", "data": None,
                        "detail": "Vòng giám sát chưa chạy trong tiến trình backend."}
            resp.raise_for_status()
            return {"state": "ok", "data": resp.json().get("data") or {}, "detail": None}
    except Exception as e:
        logger.warning("Không lấy được jetson-monitoring/metrics (%s): %s", url, e)
        return {"state": "unreachable", "data": None, "detail": str(e)}


def get_system_alerts(limit: int = 20) -> dict:
    """Cảnh báo do backend sinh ra khi số liệu vượt ngưỡng đã cấu hình."""
    url = f"{settings.BACKEND_URL}/api/jetson-monitoring/alerts"
    try:
        with httpx.Client(timeout=settings.BACKEND_TIMEOUT) as client:
            resp = client.get(url, params={"limit": limit})
            resp.raise_for_status()
            return {"state": "ok", "data": resp.json().get("data") or [], "detail": None}
    except Exception as e:
        logger.warning("Không lấy được jetson-monitoring/alerts (%s): %s", url, e)
        return {"state": "unreachable", "data": None, "detail": str(e)}
