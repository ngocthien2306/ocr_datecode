"""
Chỗ hứng attachment khi một lượt chat đi qua NHIỀU agent.

Trước đây orchestrator chọn đúng một agent rồi giao đứt, nên attachment
(`ui_charts`, `ui_kpis`, …) chỉ cần chuyển tiếp một lần qua `context`. Khi
orchestrator gọi agent con như tool, một câu hỏi có thể chạy hai ba agent, và mỗi
agent lại trả về `context` riêng của nó. Hợp nhất các dict đó bằng `{**a, **b}`
thì cái sau ghi đè cái trước — biểu đồ của agent thiết bị làm mất biểu đồ của agent
sản xuất, im lặng.

Nên dùng một chỗ hứng chung theo request: agent con nào chạy cũng NỐI THÊM vào,
không ghi đè. Cùng cách làm với `core/i18n.py` — `ContextVar`, vì mỗi request là
một task asyncio riêng nên không rò sang request khác, và không phải luồn tham số
qua hàng chục chữ ký hàm.
"""

from contextvars import ContextVar
from typing import Any, Dict, List, Optional

# Các khoá attachment mà agent con đặt vào context. Danh sách CỐ ĐỊNH: thêm loại
# attachment mới mà quên thêm vào đây thì nó lặng lẽ không bao giờ hiện ra.
KEYS = ("ui_images", "ui_charts", "ui_files", "ui_cards",
        "ui_kpis", "ui_tables", "ui_options", "ui_suggestions")

# Tool call THẬT của các agent con, kèm kết quả. Phải gom riêng vì chúng nằm trong
# state của agent con, không nằm trong state của orchestrator — sau khi chuyển sang
# gọi agent-như-tool, orchestrator chỉ thấy `ask_production_data(...)` còn
# `get_production_summary(...)` thì mất hẳn. Mất chúng làm hỏng hai thứ: khung debug
# trên /test không còn hiện tool nào chạy, và `core/reroute` tưởng "không tool nào
# chạy" nên bày đủ bốn nút hỏi-lại ngay dưới một câu trả lời đúng.
_INNER = "inner_tool_calls"
_RESULTS = "inner_tool_results"

_bucket: ContextVar[Optional[Dict[str, List[Any]]]] = ContextVar("ui_bucket", default=None)


def start() -> None:
    """Mở chỗ hứng cho request hiện tại."""
    box = {k: [] for k in KEYS}
    box[_INNER] = []
    box[_RESULTS] = []
    _bucket.set(box)


def absorb(context: Optional[Dict[str, Any]]) -> None:
    """
    Hút attachment từ context của một agent con vào chỗ hứng.

    Gọi sau MỖI lần chạy agent con, nên thứ tự trong danh sách là thứ tự agent
    được gọi — biểu đồ hiện theo đúng trình tự câu trả lời được dựng.
    """
    box = _bucket.get()
    if box is None or not context:
        return
    for k in KEYS:
        val = context.get(k)
        if not val:
            continue
        if isinstance(val, list):
            box[k].extend(val)
        else:
            box[k].append(val)


def add_tool_calls(calls: List[Dict[str, Any]],
                   results: Optional[Dict[str, Any]] = None) -> None:
    """Ghi lại tool call thật của một agent con, kèm kết quả nếu có."""
    box = _bucket.get()
    if box is None:
        return
    box[_INNER].extend(calls or [])
    if results:
        box[_RESULTS].append(results)


def inner_tool_calls() -> List[Dict[str, Any]]:
    box = _bucket.get()
    return list((box or {}).get(_INNER) or [])


def inner_tool_results() -> Dict[str, Any]:
    """Kết quả tool của mọi agent con, gộp theo tên tool."""
    box = _bucket.get()
    merged: Dict[str, Any] = {}
    for chunk in ((box or {}).get(_RESULTS) or []):
        merged.update(chunk or {})
    return merged


def collected() -> Dict[str, List[Any]]:
    """Toàn bộ attachment đã hứng, theo thứ tự agent được gọi."""
    box = _bucket.get()
    return {k: list(v) for k, v in (box or {}).items()}


def get(key: str) -> List[Any]:
    box = _bucket.get()
    return list((box or {}).get(key) or [])
