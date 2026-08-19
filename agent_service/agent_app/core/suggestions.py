"""
Gợi ý câu hỏi tiếp theo → chip bấm được ở FE.

Cùng hợp đồng với `options` trong ui_options.py: chip chứa NGUYÊN VĂN câu sẽ
gửi lại vào /api/agent/chat. Khác nhau ở ý nghĩa:

- `options`     — BẮT BUỘC chọn một (tên recipe mơ hồ, không chọn thì không đi tiếp được)
- `suggestions` — gợi ý tuỳ chọn, user thích thì bấm, không thì gõ câu khác

Lấy gợi ý ở đâu:

1. Chính LLM sinh ra, bọc trong khối [SUGGESTIONS]...[/SUGGESTIONS]. Ưu tiên
   cách này vì gợi ý bám sát ngữ cảnh cuộc hội thoại.
2. Nếu LLM quên khối đó → suy ra từ tool vừa chạy. Tất định, luôn hợp lệ.

Khối phân cách được chọn thay vì tự parse "1. 2. 3." trong văn xuôi: danh sách
đánh số xuất hiện đầy trong nội dung thật (bước khắc phục sự cố, dòng log,
bảng thống kê) nên parse kiểu đó sẽ cắt nhầm.
"""

import re
from typing import Any, Dict, List, Optional

_MAX_SUGGESTIONS = 4

# Bắt cả trường hợp LLM quên thẻ đóng (khớp tới hết chuỗi).
_BLOCK = re.compile(
    r"\[SUGGESTIONS\](?P<body>.*?)(?:\[/SUGGESTIONS\]|\Z)",
    re.DOTALL | re.IGNORECASE,
)

# Dòng gợi ý: "- ...", "* ...", "1. ...", "1) ..."
_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*(?P<text>.+?)\s*$")

# Gợi ý dự phòng theo tool vừa gọi. Viết dưới dạng câu user sẽ gõ.
_FALLBACK: Dict[str, List[str]] = {
    "check_service_status": [
        "Xem log gần đây của service",
        "Service có kết nối WebSocket không?",
    ],
    "get_service_logs": [
        "Trong log có lỗi gì không?",
        "Xem 100 dòng log cuối",
    ],
    "get_pass_fail_stats": [
        "So sánh với hôm qua",
        "Xu hướng 7 ngày qua",
        "Camera nào fail nhiều nhất?",
    ],
    "get_production_summary": [
        "Phân tích theo camera",
        "Phân tích theo từng giờ",
        "So sánh với hôm qua",
    ],
    "get_recipe_load_history": [
        "Ai load recipe nhiều nhất?",
        "Recipe nào đang chạy?",
    ],
    # list_recipes đã sinh `options` rồi, thêm gợi ý nữa chỉ gây nhiễu.
    "list_recipes": [],
}


# Gợi ý là chip bấm-một-phát, không có bước xác nhận nào. Nên KHÔNG bao giờ
# được gợi ý hành động phá huỷ — LLM từng tự đề xuất "Dừng Camera service",
# bấm vào là dừng dây chuyền đang chạy. Lọc ở server thay vì tin vào prompt.
_DESTRUCTIVE = re.compile(
    r"\b(dừng|dung|stop|tắt|tat|kill|restart|khởi động lại|khoi dong lai|"
    r"reboot|xoá|xoa|delete|reset)\b",
    re.IGNORECASE,
)


def _is_safe(text: str) -> bool:
    return not _DESTRUCTIVE.search(text)


def extract_suggestions(text: str) -> tuple[str, List[str]]:
    """
    Tách khối [SUGGESTIONS] khỏi câu trả lời.

    Returns:
        (text đã gỡ khối — thứ hiển thị cho user, danh sách gợi ý)
    """
    if not text:
        return text, []

    match = _BLOCK.search(text)
    if not match:
        return text, []

    items: List[str] = []
    for line in match.group("body").splitlines():
        item = _ITEM.match(line)
        if not item:
            continue
        # Bỏ ** đậm nếu LLM tự thêm vào
        value = item.group("text").strip().strip("*").strip()
        if value and value not in items and _is_safe(value):
            items.append(value)

    clean = (text[: match.start()] + text[match.end() :]).strip()
    return clean, items[:_MAX_SUGGESTIONS]


def fallback_suggestions(tool_calls: Optional[List[Dict[str, Any]]]) -> List[str]:
    """Suy gợi ý từ các tool vừa chạy, dùng khi LLM quên khối [SUGGESTIONS]."""
    if not tool_calls:
        return []

    items: List[str] = []
    # Duyệt ngược: tool chạy sau cùng sát ngữ cảnh nhất.
    for call in reversed(tool_calls):
        for value in _FALLBACK.get(call.get("tool") or "", []):
            if value not in items:
                items.append(value)

    return items[:_MAX_SUGGESTIONS]
