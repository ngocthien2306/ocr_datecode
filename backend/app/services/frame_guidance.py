"""
Frame-unavailable guidance
==========================

When a get-frame call comes back empty the operator standing at the line needs
to know what to DO, not that a 404 happened. This module turns the camera's
liveness state into a Vietnamese explanation plus the three remedies, in the
order they should be attempted:

  1. Ngắt/kết nối lại camera (tab Camera)      — cheapest, fixes most cases
  2. Load lại recipe                            — makes the camera grab again
  3. Khẩn cấp: restart toàn bộ dịch vụ          — stops the line, last resort

The payload is returned as the `detail` of the HTTP error so the frontend can
render a proper dialog instead of a toast with a status code in it.
"""

from typing import Any, Dict, Optional

# Machine-readable action ids — the FE maps these to buttons/navigation.
ACTION_RECONNECT_CAMERA = "reconnect_camera"
ACTION_RELOAD_RECIPE = "reload_recipe"
ACTION_RESTART_ALL = "restart_all"

CODE_FRAME_UNAVAILABLE = "FRAME_UNAVAILABLE"


def _reason_for(health: Optional[Dict[str, Any]]) -> str:
    """One sentence, in Vietnamese, explaining why no frame came back."""
    if not health or not health.get("supported"):
        return (
            "Không đọc được hình từ bộ nhớ chia sẻ của camera. "
            "Có thể camera chưa kết nối, hoặc dịch vụ camera vừa khởi động lại."
        )

    if not health.get("alive"):
        return (
            "Tiến trình camera đã ngừng phản hồi "
            f"({health.get('frozen_for', 0):.0f} giây không có tín hiệu). "
            "Dịch vụ camera đang được khôi phục tự động."
        )

    grab_mode = health.get("grab_mode")
    if grab_mode == "idle":
        return (
            "Camera đang ở chế độ nghỉ nên chưa sinh hình. "
            "Thường là do chưa có recipe nào được load, hoặc recipe vừa bị dừng."
        )
    if grab_mode == "software_trigger":
        return (
            "Camera đang chờ tín hiệu kích hoạt và chưa có sản phẩm nào đi qua "
            "nên chưa có hình mới. Đây là trạng thái bình thường khi dây chuyền đang rảnh."
        )
    return (
        "Camera đang chạy nhưng bộ đệm hình còn trống. "
        "Thử lại sau vài giây, hoặc làm theo các bước bên dưới."
    )


def frame_unavailable_detail(
    serial_number: str,
    health: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the structured `detail` payload for a failed get-frame call.

    Returns a dict — FastAPI serialises it as-is, so the FE gets an object at
    `error.response.data.detail` rather than a bare string.
    """
    return {
        "code": CODE_FRAME_UNAVAILABLE,
        "title": "Không lấy được hình từ camera",
        "reason": _reason_for(health),
        "serial_number": serial_number,
        "grab_mode": (health or {}).get("grab_mode", "unknown"),
        "remedies": [
            {
                "step": 1,
                "action": ACTION_RECONNECT_CAMERA,
                "label": "Ngắt kết nối rồi kết nối lại camera",
                "detail": (
                    f"Mở tab Camera, bấm Disconnect cho camera {serial_number}, "
                    "chờ vài giây rồi bấm Connect lại. Sau đó quay lại đây và chụp hình."
                ),
            },
            {
                "step": 2,
                "action": ACTION_RELOAD_RECIPE,
                "label": "Load lại recipe rồi chụp hình",
                "detail": (
                    "Mở tab Recipe và Load lại recipe đang dùng. Camera chỉ sinh hình "
                    "khi recipe đang chạy — recipe vừa dừng thì bộ đệm sẽ không có hình mới."
                ),
            },
            {
                "step": 3,
                "action": ACTION_RESTART_ALL,
                "label": "Khẩn cấp: khởi động lại toàn bộ dịch vụ",
                "detail": (
                    "Chỉ dùng khi hai cách trên không hiệu quả. Toàn bộ hệ thống sẽ khởi "
                    "động lại và dây chuyền ngừng kiểm tra khoảng 1–2 phút."
                ),
                "danger": True,
            },
        ],
    }
