"""
Kênh phát tiến trình để stream về giao diện.

Vì sao stream tiến trình chứ không stream token: câu trả lời của agent chỉ xuất hiện
Ở CUỐI, sau khi tool đã chạy xong — mà tool chiếm phần lớn thời gian (đo thực tế:
5–27 giây). Stream token chỉ làm mượt được khoảng một giây cuối, còn hơn 20 giây đầu
vẫn là màn hình trắng. Thứ người dùng cần biết trong 20 giây đó là "hệ thống đang làm
gì", nên đó mới là thứ đáng gửi.

Hàng đợi là `queue.Queue` chứ không phải `asyncio.Queue`: node của LangGraph là hàm
đồng bộ và được chạy trong thread executor, nên chỗ PHÁT nằm ở thread khác với chỗ
ĐỌC. `asyncio.Queue` không an toàn giữa các thread; `queue.Queue` thì có.

`ContextVar` giữ chính đối tượng queue, và điều đó quan trọng: các node chạy trong
context được COPY nên `set()` ở trong đó không lan ra ngoài, nhưng `get()` vẫn trả về
đúng đối tượng queue đã đặt trước khi tạo task — nên `put()` vào nó thì bên ngoài đọc
được. Cùng lý do `core/collector.py` sửa dict tại chỗ thay vì `set()` lại.
"""

import logging
import queue
import time
from contextvars import ContextVar
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_channel: ContextVar[Optional["queue.Queue"]] = ContextVar("progress_channel", default=None)

# Nhãn hiển thị của từng agent. Người vận hành không cần biết tên module.
AGENT_LABEL = {
    "historical_analytics": "số liệu sản xuất",
    "log_analysis": "log & audit",
    "equipment_health": "thiết bị",
    "service_management": "camera service",
}

# Nhãn cho các tool tốn thời gian nhất. Tool không có trong bảng thì hiện tên tool —
# thà hiện tên kỹ thuật hơn là hiện một dòng chung chung không nói gì.
TOOL_LABEL = {
    "get_pass_fail_stats": "đang tính pass/fail",
    "get_production_summary": "đang tổng hợp sản lượng",
    "explain_failures": "đang mổ các sản phẩm fail",
    "compare_periods": "đang so sánh hai kỳ",
    "get_downtime": "đang tìm các lần dừng máy",
    "get_target_progress": "đang đối chiếu chỉ tiêu",
    "get_shift_handover": "đang dựng bản giao ca",
    "get_recipe_load_history": "đang tra lịch sử load recipe",
    "generate_report": "đang tạo file báo cáo",
    "search_logs": "đang quét file log",
    "summarize_log_errors": "đang gom nhóm lỗi trong log",
    "get_audit_logs": "đang tra audit log",
    "read_log_tail": "đang đọc cuối file log",
    "get_log_storage_report": "đang đo dung lượng log",
    "check_reject_timing": "đang đo xung reject",
    "check_trigger_health": "đang kiểm tra trigger",
    "check_sensor_pulse": "đang kiểm tra cảm biến",
    "check_subsystem_health": "đang kiểm tra các module",
    "check_service_status": "đang kiểm tra service",
}


def open_channel() -> "queue.Queue":
    """Mở kênh cho request hiện tại. Phải gọi TRƯỚC khi tạo task chạy agent."""
    q: "queue.Queue" = queue.Queue()
    _channel.set(q)
    return q


def close_channel() -> None:
    _channel.set(None)


def emit(kind: str, text: str = "", **extra: Any) -> None:
    """
    Phát một sự kiện tiến trình. Không có kênh thì bỏ qua, không lỗi.

    Bỏ qua im lặng là có chủ ý: endpoint /chat thường (không stream) không mở kênh,
    và các điểm phát nằm rải trong tool nên không được phép phụ thuộc vào việc có
    stream hay không.
    """
    q = _channel.get()
    if q is None:
        return
    try:
        q.put_nowait({"kind": kind, "text": text, "ts": time.time(), **extra})
    except Exception:  # queue.Full — không đáng làm hỏng câu trả lời
        pass


def agent_started(agent_id: str) -> None:
    emit("agent", f"đang hỏi {AGENT_LABEL.get(agent_id, agent_id)}", agent=agent_id)


def tool_started(name: str) -> None:
    emit("tool", TOOL_LABEL.get(name, f"đang chạy {name}"), tool=name)


def tool_finished(name: str, seconds: float, ok: bool = True) -> None:
    emit("tool_done", "", tool=name, seconds=round(seconds, 2), ok=ok)


def timed(func, name: str):
    """
    Bọc một hàm tool để phát sự kiện bắt đầu/kết thúc kèm thời gian chạy.

    Thời gian chạy được gửi về để giao diện hiện "đã xong (2,1s)" — và để chính ta
    biết tool nào chậm mà không phải đọc log.
    """
    def wrapped(*args, **kwargs):
        tool_started(name)
        t0 = time.monotonic()
        ok = True
        try:
            return func(*args, **kwargs)
        except Exception:
            ok = False
            raise
        finally:
            tool_finished(name, time.monotonic() - t0, ok)

    wrapped.__name__ = getattr(func, "__name__", name)
    wrapped.__doc__ = getattr(func, "__doc__", None)
    return wrapped


def drain(q: "queue.Queue", limit: int = 64) -> list:
    """Lấy hết sự kiện đang chờ. `limit` chặn trường hợp phát nhanh hơn đọc."""
    out = []
    while len(out) < limit:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            break
    return out
