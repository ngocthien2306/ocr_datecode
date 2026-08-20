"""
Số liệu phần cứng của máy: CPU, GPU, RAM, đĩa, điện năng, uptime.

Dữ liệu lấy từ backend cùng máy (`/api/jetson-monitoring/*`) chứ không đọc thẳng
`psutil` ở đây. Lý do: trên Jetson, nhiệt độ và điện năng chỉ đọc được qua `jtop`,
mà `jtop` chỉ cho MỘT tiến trình giữ kết nối tại một thời điểm. Backend đã giữ
kết nối đó rồi; agent service mở thêm một kết nối nữa là tranh nhau. Hỏi qua HTTP
thì không tranh, và số liệu agent nói ra trùng đúng số liệu UI đang hiện.

`psutil` trong `service_tools.py` là chuyện khác: ở đó nó đo CPU/RAM của MỘT TIẾN
TRÌNH cụ thể, không phải của cả máy.
"""

import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from agent_app.core import backend_client
from agent_app.tools.base_tool import BaseTool, ToolMetadata, ToolRegistry

logger = logging.getLogger(__name__)

_MB_PER_GB = 1024.0


def _gb(mb: Optional[float]) -> Optional[float]:
    return None if mb is None else round(mb / _MB_PER_GB, 2)


def _uptime_text(seconds: Optional[float]) -> Optional[str]:
    if not seconds:
        return None
    s = int(seconds)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m = s // 60
    if d:
        return f"{d} ngày {h} giờ"
    if h:
        return f"{h} giờ {m} phút"
    return f"{m} phút"


def _not_available(state: str, detail: Optional[str]) -> Dict[str, Any]:
    """
    Ba lý do "không có số liệu" phải phân biệt được.

    Gộp chúng thành một câu "không có dữ liệu" là cách nhanh nhất để giấu mất một
    sự cố thật: `not_ready` nghĩa là vòng giám sát đã chết trong tiến trình backend
    — máy vẫn chạy, chỉ là không ai đo nữa. Nói "không có dữ liệu" khiến người đọc
    tưởng máy im, và không ai đi sửa.
    """
    if state == "not_ready":
        return {
            "success": False,
            "reason": "monitoring_loop_dead",
            "message": (
                "Backend đang chạy nhưng vòng giám sát phần cứng đã dừng, nên "
                "hiện không có số liệu. Đây là lỗi đã biết: `ocr-backend` không "
                "khai báo `After=jtop.service`, lúc boot nó thường khởi động "
                "trước jtop, `jtop()` ném lỗi và vòng lặp tắt vĩnh viễn."
            ),
            "how_to_fix": (
                "Hồi sinh ngay, KHÔNG cần restart backend: "
                "POST /api/jetson-monitoring/start. Chống tái phát sau reboot: "
                "thêm drop-in After=/Wants=jtop.service cho ocr-backend.service."
            ),
            "detail": detail,
        }
    return {
        "success": False,
        "reason": "backend_unreachable",
        "message": "Không gọi được backend trên máy này, nên không biết tình trạng phần cứng.",
        "detail": detail,
    }


def get_system_metrics() -> Dict[str, Any]:
    """Đọc CPU / GPU / RAM / đĩa / điện năng / uptime của chính máy này."""
    res = backend_client.get_system_metrics()
    if res["state"] != "ok":
        return _not_available(res["state"], res.get("detail"))

    m = res["data"] or {}
    cpu = m.get("cpu") or {}
    gpu = m.get("gpu") or {}
    ram = m.get("ram") or {}
    disk = m.get("disk") or {}
    power = m.get("power")

    # Nhiệt độ và điện năng có thể là None trên máy x86 (không có cảm biến kiểu
    # Jetson, `jtop_available: false`). PHẢI giữ nguyên None chứ không đổi thành
    # 0 — "0°C" đọc như máy đang cực mát, tức là sai theo hướng nguy hiểm nhất.
    cpu_temp = cpu.get("temperature_celsius")
    gpu_temp = gpu.get("temperature_celsius")
    has_thermal = cpu_temp is not None or gpu_temp is not None

    out: Dict[str, Any] = {
        "success": True,
        "measured_at": m.get("timestamp"),
        "cpu": {
            "usage_percent": cpu.get("usage_percent"),
            "temperature_celsius": cpu_temp,
            "frequency_mhz": cpu.get("frequency_mhz"),
            "core_count": len(cpu.get("per_core_usage") or {}) or None,
        },
        "gpu": {
            "usage_percent": gpu.get("usage_percent"),
            "temperature_celsius": gpu_temp,
            "frequency_mhz": gpu.get("frequency_mhz"),
        },
        # Kèm cả số tuyệt đối chứ không chỉ phần trăm: "RAM 87%" trên máy 7 GB
        # và trên máy 32 GB là hai tình huống hoàn toàn khác nhau, mà phần trăm
        # thì giống hệt.
        "ram": {
            "usage_percent": ram.get("usage_percent"),
            "used_gb": _gb(ram.get("used_mb")),
            "available_gb": _gb(ram.get("available_mb")),
            "total_gb": _gb(ram.get("total_mb")),
            "swap_used_gb": _gb(ram.get("swap_used_mb")),
            "swap_total_gb": _gb(ram.get("swap_total_mb")),
        },
        "disk": {
            "usage_percent": disk.get("usage_percent"),
            "used_gb": round(disk.get("used_gb"), 1) if disk.get("used_gb") is not None else None,
            "available_gb": round(disk.get("available_gb"), 1) if disk.get("available_gb") is not None else None,
            "total_gb": round(disk.get("total_gb"), 1) if disk.get("total_gb") is not None else None,
        },
        "power": (
            {"total_w": round(power.get("total_mw", 0) / 1000.0, 2)} if power else None
        ),
        "uptime": _uptime_text(m.get("uptime_seconds")),
        "power_mode": m.get("nvp_model"),
        # Để mô hình biết trường None là "phần cứng không đo được", chứ không
        # phải "đo được và bằng không".
        "sensors": {
            "thermal": "có" if has_thermal else "máy này không có cảm biến nhiệt đọc được",
            "power": "có" if power else "máy này không đọc được điện năng",
        },
        # Cảnh báo đi kèm luôn, không đợi mô hình nhớ gọi thêm tool.
        #
        # Vì sao: hỏi "máy có cảnh báo phần cứng nào không" thì orchestrator tách
        # câu, đẩy phần "cảnh báo" sang agent equipment_health (xung reject,
        # trigger, cảm biến) và get_system_alerts không bao giờ được gọi — nên
        # câu trả lời bỏ sót đúng thứ đang nghiêm trọng nhất: đĩa 97,2%. Siết mô
        # tả tool không chữa được, vì quyết định tách câu xảy ra ở tầng trên.
        # Gắn thẳng vào đây thì mọi đường hỏi về phần cứng đều thấy.
        "active_alerts": _alert_summary(),
    }
    return out


def _alert_summary() -> Dict[str, Any]:
    """
    Gộp cảnh báo theo loại, chỉ giữ giá trị mới nhất mỗi loại.

    Backend ghi lại cùng một cảnh báo mỗi phút khi điều kiện còn đúng — đĩa đầy
    thì sinh một bản ghi `disk_usage` mỗi 60 giây. Đưa nguyên danh sách vào mô
    hình thì 20 dòng cảnh báo hoá ra chỉ là MỘT vấn đề lặp lại, và mô hình sẽ
    đọc thành "có 20 cảnh báo".
    """
    res = backend_client.get_system_alerts(limit=100)
    if res["state"] != "ok":
        return {"available": False}

    by_type: Dict[str, Dict[str, Any]] = {}
    for a in res["data"] or []:
        t = a.get("alert_type") or "unknown"
        cur = by_type.setdefault(t, {"count": 0, "severity": a.get("severity"),
                                     "latest_value": None, "latest_at": None,
                                     "message": a.get("message")})
        cur["count"] += 1
        ts = a.get("timestamp")
        if cur["latest_at"] is None or (ts or "") >= cur["latest_at"]:
            cur["latest_at"] = ts
            cur["latest_value"] = a.get("current_value")
            cur["severity"] = a.get("severity")
            cur["message"] = a.get("message")

    return {
        "available": True,
        "distinct_problems": len(by_type),
        "by_type": by_type,
        "note": (
            "Mỗi loại là MỘT vấn đề, dù backend ghi lại nhiều lần khi điều kiện "
            "còn đúng. `count` là số lần ghi, không phải số vấn đề."
        ),
    }


class SystemAlertsArgs(BaseModel):
    limit: int = Field(
        default=20,
        description="Số cảnh báo gần nhất cần lấy (1–100).",
    )


def get_system_alerts(limit: int = 20, **_ignored) -> Dict[str, Any]:
    """Cảnh báo do backend sinh khi CPU/RAM/đĩa/nhiệt độ vượt ngưỡng đã cấu hình."""
    limit = max(1, min(int(limit or 20), 100))
    res = backend_client.get_system_alerts(limit)
    if res["state"] != "ok":
        return _not_available(res["state"], res.get("detail"))

    alerts = res["data"] or []
    return {
        "success": True,
        "count": len(alerts),
        "alerts": alerts,
        # Ngưỡng do backend cấu hình, không phải do tool tự đặt. Tool tự nghĩ ra
        # ngưỡng thì con số cảnh báo ở đây sẽ lệch với con số trên UI.
        "note": "Ngưỡng lấy theo cấu hình của backend (/api/jetson-monitoring/config).",
    }


get_system_metrics_tool = BaseTool.create_tool(
    func=get_system_metrics,
    metadata=ToolMetadata(
        name="get_system_metrics",
        description=(
            "Tình trạng PHẦN CỨNG của máy đang chạy: mức dùng và nhiệt độ CPU/GPU, "
            "RAM còn trống, dung lượng đĩa, điện năng tiêu thụ, thời gian đã chạy. "
            "Dùng khi được hỏi máy nóng không, còn bao nhiêu RAM, đĩa sắp đầy chưa, "
            "máy chạy được bao lâu rồi. KHÁC check_service_status — cái đó nói về "
            "một tiến trình service, còn cái này nói về cả cỗ máy."
        ),
        # Category "system" cố ý KHÔNG nằm trong _CACHEABLE_CATEGORIES: số liệu
        # làm mới mỗi 2 giây, mà cache 45 giây thì câu "máy còn nóng không" sau
        # khi vừa hạ tải sẽ trả lại đúng con số nóng cũ.
        category="system",
        requires_approval=False,
    ),
)

get_system_alerts_tool = BaseTool.create_tool(
    func=get_system_alerts,
    metadata=ToolMetadata(
        name="get_system_alerts",
        description=(
            "Cảnh báo của MÁY TÍNH khi vượt ngưỡng: CPU/GPU quá nóng, RAM sắp cạn, "
            "đĩa gần đầy, tải cao kéo dài. "
            "Hỏi 'máy có cảnh báo gì không', 'có gì bất thường không', 'vì sao máy "
            "chậm' thì PHẢI gọi tool này. "
            "ĐỪNG nhầm với cảnh báo THIẾT BỊ SẢN XUẤT (xung reject, trigger, cảm "
            "biến) — nhóm đó thuộc agent equipment_health. Câu hỏi chung chung về "
            "'cảnh báo' thì lấy CẢ HAI, vì người hỏi thường không phân biệt."
        ),
        category="system",
        requires_approval=False,
    ),
    args_schema=SystemAlertsArgs,
)

ToolRegistry.register(get_system_metrics_tool)
ToolRegistry.register(get_system_alerts_tool)

logger.info("✅ System tools registered")
