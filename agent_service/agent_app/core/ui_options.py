"""
Biến kết quả tool thành các lựa chọn bấm được cho FE.

Hợp đồng với FE cố tình để đơn giản nhất có thể: mỗi option có `value` là
NGUYÊN VĂN câu sẽ gửi lại vào /api/agent/chat khi user bấm. FE không cần biết
gì về recipe hay agent — bấm nút = gửi `value` như một tin nhắn bình thường,
cùng session_id. User vẫn có thể gõ tay thay vì bấm.
"""

from typing import Any, Dict, List, Optional

# Chặn số nút để không đổ một danh sách dài dằng dặc lên UI.
_MAX_OPTIONS = 8


def options_from_tool_result(tool_name: str, result: Any) -> Optional[List[Dict[str, str]]]:
    """
    Trả danh sách option nếu tool này có gì đó cần user chọn, ngược lại None.

    Ba nguồn:
    - tool nào trả `needs_disambiguation` (tên recipe khớp nhiều recipe)
    - tool nào trả `needs_format_choice` (chưa biết xuất báo cáo định dạng nào)
    - `list_recipes` (user chưa nêu recipe nào)
    """
    if not isinstance(result, dict):
        return None

    # Định dạng file có sẵn `value` do tool dựng — nó đã nhét ngày cụ thể vào
    # câu, nên không tái tạo lại ở đây.
    if result.get("needs_format_choice"):
        formats = [
            {"label": f["label"], "value": f["value"], "hint": f.get("hint", "")}
            for f in (result.get("formats") or [])[:_MAX_OPTIONS]
            if f.get("label") and f.get("value")
        ]
        return formats or None

    if result.get("needs_disambiguation"):
        rows = result.get("matches") or []
    elif tool_name == "list_recipes":
        rows = result.get("recipes") or []
    else:
        return None

    options = [
        {
            "label": row["recipe_name"],
            # Kèm ID để lượt sau khớp chính xác một recipe: _id_or_name() thấy
            # chuỗi 24 ký tự hex sẽ lọc theo ID thay vì khớp tên mờ.
            "value": f'Recipe {row["recipe_id"]} ({row["recipe_name"]})',
            "hint": f'{row["total"]:,} sản phẩm',
        }
        for row in rows[:_MAX_OPTIONS]
        if row.get("recipe_id") and row.get("recipe_name")
    ]

    return options or None
