"""
Nút "hỏi lại bằng agent khác" khi định tuyến có thể đã sai.

Vì sao cần, bên cạnh việc siết từ khoá định tuyến: siết từ khoá chỉ giảm tỷ lệ
sai, không triệt được. Orchestrator phán đoán ý định bằng LLM, và tiếng Việt dùng
chung một chữ cho nhiều khái niệm — "lỗi" là sản phẩm fail hay là dòng ERROR
trong log, "ca" là ca làm việc hay là ba chữ đầu của "camera". Sẽ luôn có câu bị
đoán sai.

Cái tệ không phải là đoán sai, mà là người dùng KHÔNG CÓ ĐƯỜNG RA. Họ nhận một
câu trả lời nghe hợp lý ("không có sự cố nào được ghi lại"), không biết là nó
được trả lời bởi agent sai, và phải tự đoán cách diễn đạt lại. Nên khi có dấu
hiệu route sai, ta đưa luôn nút bấm hỏi lại CHÍNH câu đó bằng agent đúng.

Khác `ui_options`: `options` là lựa chọn BẮT BUỘC, hiện ra là chặn luồng cho tới
khi user bấm. Reroute thì không được chặn — câu trả lời vừa rồi có thể đúng, nút
chỉ là đường ra khi nó sai.
"""

from typing import Any, Dict, List, Optional

from agent_app.core.i18n import t

# Tool nào thuộc agent nào. Dùng để biết câu vừa rồi đã đi đường nào, và để đề
# nghị đường còn lại.
_TOOL_AGENT = {
    # historical_analytics
    "get_pass_fail_stats": "historical_analytics",
    "get_production_summary": "historical_analytics",
    "explain_failures": "historical_analytics",
    "list_recipes": "historical_analytics",
    "get_recipe_load_history": "historical_analytics",
    "compare_periods": "historical_analytics",
    "get_downtime": "historical_analytics",
    "get_target_progress": "historical_analytics",
    "get_shift_handover": "historical_analytics",
    "generate_report": "historical_analytics",
    # log_analysis
    "list_log_sources": "log_analysis",
    "read_log_tail": "log_analysis",
    "search_logs": "log_analysis",
    "summarize_log_errors": "log_analysis",
    "get_audit_logs": "log_analysis",
    "get_log_storage_report": "log_analysis",
    # equipment_health
    "check_reject_timing": "equipment_health",
    "check_trigger_health": "equipment_health",
    "check_sensor_pulse": "equipment_health",
    "check_subsystem_health": "equipment_health",
    # service_management
    "check_service_status": "service_management",
    "start_service": "service_management",
    "stop_service": "service_management",
}

_AGENT_LABEL = {
    "historical_analytics": "Số liệu sản xuất",
    "log_analysis": "Log & audit",
    "equipment_health": "Thiết bị",
    "service_management": "Camera service",
}

_AGENT_HINT = {
    "historical_analytics": "sản lượng, pass/fail, ảnh sản phẩm lỗi, báo cáo",
    "log_analysis": "dòng ERROR trong log, ai đã thao tác gì",
    "equipment_health": "reject, trigger, cảm biến, module",
    "service_management": "trạng thái start/stop của service",
}

# Tool trả về rỗng thì gợi ý sang đâu. Đây là các cặp đã nhầm thật, không phải
# suy đoán: `search_logs` rỗng bị hiểu thành "không có sản phẩm lỗi"; các tool
# sản xuất rỗng bị hiểu thành "không có sự cố".
_EMPTY_REDIRECT = {
    "search_logs": "historical_analytics",
    "summarize_log_errors": "historical_analytics",
    "read_log_tail": "historical_analytics",
    "get_production_summary": "log_analysis",
    "get_pass_fail_stats": "log_analysis",
    "explain_failures": "log_analysis",
}


# Cụm từ chỉ RÕ ý định, kèm agent đáng lẽ phải xử lý. Chỉ đưa vào đây những cụm
# gần như không thể hiểu theo nghĩa khác — bảng này càng rộng thì càng dễ hiện nút
# sai chỗ, và một nút "không đúng ý?" sau một câu trả lời đúng thì gieo nghi ngờ
# vào chính thứ đang đúng.
_INTENT = [
    # Sản phẩm fail: nằm trong database, KHÔNG nằm trong file log. Đây là chỗ đã
    # nhầm thật — hỏi "show 5 sản phẩm lỗi" mà bị đưa sang tìm trong log.
    ("historical_analytics", (
        "sản phẩm lỗi", "sản phẩm fail", "hàng lỗi", "sản phẩm không đạt",
        "ảnh sản phẩm", "ảnh fail", "frame fail", "sản phẩm bị lỗi",
        "failed unit", "failed product", "faulty unit", "reject image",
    )),
    ("log_analysis", (
        "dòng log", "trong log", "traceback", "log báo", "file log",
        "ai đăng nhập", "ai thao tác", "audit",
        "log line", "in the log", "who logged in",
    )),
]


def _intent_agent(message: str) -> Optional[str]:
    """
    Agent mà câu hỏi RÕ RÀNG thuộc về, hoặc None nếu không chắc.

    Cần thêm lớp này vì phép dò "tool trả về rỗng" không đủ: hỏi "show 5 sản phẩm
    lỗi" mà rơi vào `log_analysis`, tool `summarize_log_errors` vẫn tìm được vài
    cảnh báo trong log và trả về CÓ dữ liệu. Kết quả rỗng thì mới đáng ngờ, còn ở
    đây câu trả lời trông hoàn chỉnh — chỉ là trả lời sai câu hỏi. Không có lớp
    này thì đúng tình huống tệ nhất lại là tình huống không có nút hỏi lại.
    """
    low = (message or "").lower()
    for agent, phrases in _INTENT:
        if any(ph in low for ph in phrases):
            return agent
    return None


def _is_empty(name: str, result: Any) -> bool:
    """
    Tool này có trả về 'chẳng có gì' không?

    Chỉ xét đúng các trường đếm mà tool thật sự dùng, không đoán chung chung: một
    tool trả về `success: True` với `match_count: 0` là rỗng, còn một tool trả về
    lỗi thì đã có đường xử lý khác.
    """
    if not isinstance(result, dict) or not result.get("success"):
        return False
    for key in ("match_count", "total", "total_fail", "error_count", "stop_count"):
        if key in result:
            try:
                if int(result[key] or 0) == 0:
                    return True
            except (TypeError, ValueError):
                pass
    for key in ("matches", "samples", "breakdown", "groups", "stops", "entries"):
        if key in result and not result[key]:
            return True
    return False


def _button(agent: str, user_message: str = "") -> Dict[str, Any]:
    """Một nút hỏi-lại. `value` giữ NGUYÊN VĂN câu hỏi của user."""
    return {
        "label": t(_AGENT_LABEL[agent]),
        # Không diễn đạt lại câu hỏi: user đã nói rõ ý mình, vấn đề nằm ở chỗ chọn
        # agent. Đổi lời sẽ đổi luôn cả ý, và lúc đó nút thành một câu hỏi khác.
        "value": user_message,
        "hint": t(_AGENT_HINT[agent]),
        "agent_id": agent,
    }


# Dấu hiệu agent đang HỎI LẠI chứ không trả lời. Dùng để phân biệt hai trường hợp
# đều "không gọi tool nào": bí thật, và trả lời đúng từ ngữ cảnh câu trước.
_ASKING_BACK = (
    "bạn muốn", "bạn cần", "vui lòng", "cho tôi biết", "hãy nói rõ",
    "chưa rõ", "ý bạn là",
    "do you want", "could you", "please specify", "which ", "not sure",
)


def _asking_back(text: str) -> bool:
    low = (text or "").strip().lower()
    if not low:
        return True
    return low.endswith("?") or any(ph in low for ph in _ASKING_BACK)


def build_reroute(
    user_message: str,
    tool_calls: Optional[List[Dict[str, Any]]],
    results: Optional[Dict[str, Any]] = None,
    response_text: str = "",
    has_history: bool = False,
) -> List[Dict[str, Any]]:
    """
    Danh sách nút "hỏi lại bằng agent khác", có thể rỗng.

    Chỉ hiện trong hai tình huống, để nút không thành tiếng ồn sau mỗi câu trả
    lời đúng:

    1. **Không tool nào chạy.** Orchestrator không hiểu câu hỏi, hoặc hiểu thành
       một câu nó tự trả lời được. Đây là lúc người dùng cần đường ra nhất.
    2. **Tool chạy nhưng rỗng.** Nguy hiểm hơn trường hợp 1, vì câu trả lời trông
       như một kết luận. Ưu tiên đề nghị đúng agent hay bị nhầm với agent vừa rồi.

    Không hiện khi tool trả về có dữ liệu — câu trả lời khi đó gần như chắc đúng
    đường, và thêm nút chỉ làm người dùng nghi ngờ một kết quả tốt.
    """
    calls = tool_calls or []
    used = [_TOOL_AGENT.get(c.get("tool") or "") for c in calls]
    used_agents = [a for a in used if a]

    suggest: List[str] = []

    # Ý định trong câu hỏi lệch với agent đã trả lời ⇒ bày nút, KỂ CẢ khi tool trả
    # về đầy dữ liệu. Đây là ca tệ nhất: câu trả lời trông hoàn chỉnh nhưng trả
    # lời sai câu hỏi, nên người dùng không có lý do gì để nghi ngờ nó.
    intent = _intent_agent(user_message)
    if intent and used_agents and intent not in used_agents:
        return [_button(intent)]

    if not calls:
        # "Không gọi tool nào" có HAI nghĩa rất khác nhau, và bản đầu gộp chúng
        # làm một nên bày bốn nút cả sau những câu trả lời đúng:
        #
        #   a) Agent bí, hỏi lại người dùng     ⇒ đây mới là lúc cần nút.
        #   b) Agent trả lời đúng từ ngữ cảnh   ⇒ bày nút là gieo nghi ngờ vào
        #      câu trước, không cần chạy tool      một câu trả lời đang đúng.
        #
        # Phân biệt bằng: có hội thoại trước đó không, và câu trả lời có phải là
        # một câu hỏi lại không.
        if has_history and not _asking_back(response_text):
            return []
        # Không biết câu hỏi thuộc đâu ⇒ bày cả bốn, để user tự chỉ đường.
        suggest = ["historical_analytics", "log_analysis",
                   "equipment_health", "service_management"]
    else:
        empty = [c for c in calls
                 if _is_empty(c.get("tool") or "", (results or {}).get(c.get("tool") or ""))]
        if not empty:
            return []
        for c in empty:
            target = _EMPTY_REDIRECT.get(c.get("tool") or "")
            if target and target not in suggest:
                suggest.append(target)
        # Luôn kèm agent thiết bị: "không thấy gì trong log" và "không có sản
        # phẩm nào" đều có thể là do máy đang có vấn đề.
        if "equipment_health" not in suggest:
            suggest.append("equipment_health")

    # Không đề nghị lại chính agent vừa chạy — bấm vào sẽ ra đúng câu trả lời cũ.
    suggest = [a for a in suggest if a not in used_agents]
    if not suggest:
        return []

    return [_button(a, user_message) for a in suggest[:4]]
