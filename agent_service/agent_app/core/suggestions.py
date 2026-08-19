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

from agent_app.core.i18n import t

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
    # explain_failures trước đây không có mục nào, nên câu trả lời về nguyên nhân
    # fail — đúng chỗ người vận hành cần đi tiếp nhất — lại là chỗ ít chip nhất.
    "explain_failures": [
        "Camera nào fail nhiều nhất?",
        "So sánh với hôm qua",
        "Lúc đó log báo gì?",
    ],
    "get_recipe_load_history": [
        "Ai load recipe nhiều nhất?",
        "Recipe nào đang chạy?",
    ],
    # list_recipes đã sinh `options` rồi, thêm gợi ý nữa chỉ gây nhiễu.
    "list_recipes": [],
    "get_shift_handover": [
        "Vì sao dừng máy lúc đó?",
        "So sánh với ca trước",
        "Kiểm tra thiết bị",
    ],
    "check_reject_timing": [
        "Trigger có ổn không?",
        "Có module nào lỗi không?",
        "Hôm qua thế nào?",
    ],
    "check_trigger_health": [
        "Có sản phẩm nào bị bỏ sót?",
        "Cảm biến có ổn không?",
        "Vì sao service restart?",
    ],
    "check_sensor_pulse": [
        "Xung reject có đúng không?",
        "Nhịp dây chuyền hôm qua thế nào?",
    ],
    "check_subsystem_health": [
        "Log của module đó báo gì?",
        "Kiểm tra trigger và cảm biến",
    ],
    "get_target_progress": [
        "Còn thiếu bao nhiêu?",
        "So sánh với hôm qua",
        "Ca nào đóng góp nhiều nhất?",
    ],
    "get_downtime": [
        "Lúc đó log báo gì?",
        "Ca nào dừng nhiều nhất?",
        "So sánh với hôm qua",
    ],
    "compare_periods": [
        "So sánh 7 ngày qua",
        "Recipe nào tệ hơn kỳ trước?",
        "Xuất báo cáo kỳ này",
    ],
    "generate_report": [
        "Xuất bản Excel luôn",
        "Báo cáo 7 ngày qua",
        "Xuất PDF để in",
    ],
    "summarize_log_errors": [
        "Xem log gốc của lỗi này",
        "Hôm qua có lỗi tương tự không?",
        "Ai thao tác gì lúc đó?",
    ],
    "read_log_tail": [
        "Chỉ hiện dòng ERROR",
        "Gom nhóm lỗi trong ngày",
    ],
    "search_logs": [
        "Tìm trong 7 ngày qua",
        "Gom nhóm lỗi trong ngày",
    ],
    "get_audit_logs": [
        "Ai load recipe hôm nay?",
        "Lúc đó hệ thống báo lỗi gì?",
    ],
    "list_log_sources": [
        "Hôm nay có lỗi gì không?",
        "Xem log backend mới nhất",
    ],
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
        # Bỏ cả dấu ngoặc kép bao quanh: LLM có lượt trả về `- "Xem chi tiết..."`
        # và chip hiện ra kèm dấu " nhìn thấy được, lại còn được gửi nguyên văn
        # (kể cả dấu ngoặc) làm câu hỏi tiếp theo.
        value = item.group("text").strip().strip("*").strip().strip('"“”').strip()
        if value and value not in items and _is_safe(value):
            items.append(value)

    clean = (text[: match.start()] + text[match.end() :]).strip()
    return clean, items[:_MAX_SUGGESTIONS]


# Gợi ý nào trở thành vô nghĩa sau khi đã làm đúng việc đó. Khoá là (tool, tên
# tham số, giá trị tham số); giá trị là các câu phải bỏ đi.
#
# Có bảng này vì `_FALLBACK` tra theo TÊN TOOL nên mù hoàn toàn với tham số: hỏi
# "từ 16h đến 18h camera nào fail nhiều nhất" thì tool chạy với
# `group_by='camera'`, rồi chip vẫn mời "Phân tích theo camera" — đề nghị người
# dùng làm lại đúng việc họ vừa làm.
_REDUNDANT = {
    ("get_production_summary", "group_by", "camera"): {"Phân tích theo camera"},
    ("get_production_summary", "group_by", "hour"):   {"Phân tích theo từng giờ"},
    ("get_production_summary", "group_by", "shift"):  {"Ca nào đóng góp nhiều nhất?"},
    ("explain_failures", "group_by", "camera"):       {"Camera nào fail nhiều nhất?"},
}


def _followups(tool: str, args: Dict[str, Any], result: Any) -> List[str]:
    """
    Gợi ý suy TỪ KẾT QUẢ vừa có, nên cụ thể hơn bảng tra tĩnh.

    Đây là loại gợi ý đáng bấm: "Xem 5 sản phẩm lỗi đó" nói đúng con số vừa hiện
    trên màn hình, còn "Phân tích theo camera" thì đúng với mọi câu trả lời nào và
    vì thế chẳng dẫn đi đâu.

    Chỉ dựng câu khi dữ liệu thật sự có, để không bao giờ mời người dùng đi xem
    một thứ rỗng.
    """
    if not isinstance(result, dict) or not result.get("success"):
        return []
    out: List[str] = []

    if tool in ("get_production_summary", "get_pass_fail_stats"):
        # Ba tool ba shape khác nhau, phải đọc cả ba chứ không đoán một cái:
        #   get_production_summary → summary.fail_count
        #   get_pass_fail_stats    → fail.current  (có so sánh kỳ trước)
        #   dạng phẳng             → fail
        # Đọc sai chỗ thì `fail` luôn ra 0 và gợi ý quan trọng nhất — "xem đúng
        # mấy sản phẩm lỗi đó" — không bao giờ hiện, mà cũng không báo lỗi gì.
        fail = (result.get("summary") or {}).get("fail_count")
        if fail is None:
            fail = result.get("fail")
        if isinstance(fail, dict):
            fail = fail.get("current")
        try:
            fail = int(fail or 0)
        except (TypeError, ValueError):
            fail = 0
        if fail > 0:
            # Giữ nguyên khung thời gian user vừa hỏi — bỏ đi thì câu hỏi tiếp
            # nhảy về mặc định "hôm nay" và ra một con số khác con số đang hiện.
            # Dịch MẪU CÂU rồi mới điền số: tiếng Anh đảo trật tự
            # ("Show those 5 failed units"), ghép chuỗi rồi dịch sẽ không khớp.
            out.append(t("Xem {n} sản phẩm lỗi đó").format(
                n=f"{fail:,}".replace(",", ".")))

        rows = result.get("breakdown") or []
        worst = None
        for r in rows:
            if not isinstance(r, dict):
                continue
            if worst is None or (r.get("fail") or 0) > (worst.get("fail") or 0):
                worst = r
        # Chỉ mời đi sâu vào một camera khi có TỪ HAI camera trở lên; một camera
        # duy nhất thì "camera đó" chính là toàn bộ dữ liệu vừa xem.
        if worst and len(rows) > 1 and (worst.get("fail") or 0) > 0 and worst.get("camera"):
            out.append(t("Nguyên nhân fail của camera {cam}").format(
                cam=worst["camera"]))

    if tool == "explain_failures":
        causes = result.get("causes") or []
        if causes and isinstance(causes[0], dict) and causes[0].get("label"):
            out.append(t('Vì sao nhiều lỗi "{cause}"?').format(
                cause=causes[0]["label"]))
        # Khoá thật là `total_failed_products`, không phải `total_fail` — tên
        # dài hơn vì nó cố ý nói rõ đơn vị là SẢN PHẨM, phân biệt với số frame.
        if result.get("total_failed_products"):
            out.append("Hôm qua có lỗi tương tự không?")

    if tool == "get_downtime" and (result.get("stops") or []):
        first = (result.get("stops") or [])[0]
        if isinstance(first, dict) and first.get("from"):
            out.append(t("Ai thao tác gì lúc {time}?").format(
                time=str(first["from"])[11:16]))

    return out


def grounded_suggestions(
    tool_calls: Optional[List[Dict[str, Any]]],
    results: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Chỉ những gợi ý suy TỪ SỐ LIỆU vừa trả về. Rỗng nếu không suy được gì.

    Tách riêng khỏi `fallback_suggestions` để chỗ gọi có thể ƯU TIÊN nhóm này hơn
    khối [SUGGESTIONS] do LLM tự viết. Đó là đảo lại thứ tự cũ, và có lý do:
    LLM viết gợi ý mà không nhìn con số, nên sau câu "từ 16h đến 18h camera nào
    fail nhiều nhất" nó mời "Xem lịch sử load recipe gần đây" — một câu chẳng dính
    gì tới thứ đang xem. Còn "Xem 5 sản phẩm lỗi đó" thì lấy đúng con số vừa hiện
    trên màn hình.

    Cùng nguyên tắc đã dùng cho `charts`/`kpis`: thứ gì dựng được bằng code từ kết
    quả tool thì đừng để mô hình viết, vì mô hình viết sẽ lệch khỏi dữ liệu.
    """
    if not tool_calls:
        return []
    results = results or {}
    out: List[str] = []
    for call in reversed(tool_calls):
        for value in _followups(call.get("tool") or "", call.get("args") or {},
                                results.get(call.get("tool") or "")):
            if value not in out:
                out.append(value)
    return out


def fallback_suggestions(
    tool_calls: Optional[List[Dict[str, Any]]],
    results: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Suy gợi ý từ các tool vừa chạy, dùng khi LLM quên khối [SUGGESTIONS].

    Ba bước, theo đúng thứ tự ưu tiên: câu suy từ kết quả trước (cụ thể nhất), rồi
    mới tới bảng tra tĩnh, và cuối cùng loại những câu đã thành vô nghĩa vì user
    vừa làm đúng việc đó.
    """
    if not tool_calls:
        return []

    results = results or {}
    items: List[str] = []
    drop: set = set()

    # Duyệt ngược: tool chạy sau cùng sát ngữ cảnh nhất.
    for call in reversed(tool_calls):
        tool = call.get("tool") or ""
        args = call.get("args") or {}
        for key, val in args.items():
            drop |= _REDUNDANT.get((tool, key, val), set())
        for value in _followups(tool, args, results.get(tool)):
            if value not in items:
                items.append(value)

    for call in reversed(tool_calls):
        for value in _FALLBACK.get(call.get("tool") or "", []):
            if value not in items:
                items.append(value)

    items = [x for x in items if x not in drop]

    # Dịch ở cuối, sau khi đã loại trùng: `_FALLBACK` là bảng tra theo tên tool
    # nên phải giữ nguyên khoá tiếng Việt, còn chip hiện ra thì theo ngôn ngữ
    # đang chọn. Bấm vào chip là gửi chính chuỗi đó làm câu hỏi, và mô hình hiểu
    # cả hai thứ tiếng nên chip tiếng Anh vẫn chạy đúng tool.
    return [t(x) for x in items[:_MAX_SUGGESTIONS]]
