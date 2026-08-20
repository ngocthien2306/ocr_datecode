"""
Bốn agent chuyên biệt, gói lại thành tool để orchestrator gọi được.

Đây là thay đổi kiến trúc, không phải tiện ích. Trước đây orchestrator chọn ĐÚNG MỘT
agent rồi giao đứt câu hỏi, nên ba hạn chế đi liền nhau:

1. Không tổng hợp được nhiều nguồn. "So sánh sản phẩm fail với cảnh báo thiết bị
   cùng giờ" là câu hợp lý mà hệ thống không trả lời được, vì nó cần cả agent sản
   xuất lẫn agent thiết bị.
2. Không chẻ được câu nhiều bước — mỗi câu hỏi là một chặng, người dùng phải tự chia.
3. Tốn một lượt LLM chỉ để định tuyến, rồi mới tới lượt của agent con.

Khi agent con là tool, cả ba tan cùng lúc: orchestrator gọi bao nhiêu agent cũng
được, gọi xong còn thiếu thì gọi thêm, và việc "chọn agent" trở thành chính việc
chọn tool nên không cần lượt LLM riêng nữa.

Mỗi tool nhận `question` — câu hỏi diễn đạt lại cho agent con — và trả về CHUỖI trả
lời của agent đó. Attachment (biểu đồ, ô KPI, ảnh) không đi qua giá trị trả về mà
được hút vào `core/collector.py`, vì chúng phải đến UI nguyên vẹn chứ không qua tay
mô hình.
"""

import logging
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent_app.base.base_agent import AgentState
from agent_app.core import collector, progress
from agent_app.core.registry import AgentRegistry
from agent_app.tools.base_tool import BaseTool, ToolMetadata

logger = logging.getLogger(__name__)


class AskAgentArgs(BaseModel):
    question: str = Field(
        description=(
            "Câu hỏi gửi cho agent chuyên biệt, viết đầy đủ và tự đứng được. "
            "PHẢI nêu rõ khoảng thời gian, tên recipe, số camera nếu câu gốc của "
            "user có — agent con không thấy được câu hỏi gốc nên đừng dùng 'đó', "
            "'cái này', 'như trên'."
        )
    )


# Ngữ cảnh hội thoại truyền cho agent con: đủ để hiểu câu hỏi tiếp nối, không đủ để
# nhắc lại toàn bộ phiên. Con số nhỏ vì `question` đã được orchestrator viết lại cho
# tự đứng được, nên agent con chủ yếu cần biết vừa nói về recipe/camera nào.
_HISTORY_FOR_SUB = 6


def _run(agent_id: str, question: str) -> str:
    """
    Chạy một agent con và trả về phần trả lời dạng chữ.

    Ngữ cảnh hội thoại lấy từ state của orchestrator qua ContextVar chứ không qua
    tham số: LangChain gọi tool chỉ với các field trong args schema, không có đường
    nào nhét state vào. Thiếu ngữ cảnh thì câu tiếp nối kiểu "còn hôm qua thì sao"
    không giải được, dù `question` đã được viết lại.
    """
    history = _sub_history.get() or []
    progress.agent_started(agent_id)
    try:
        agent = AgentRegistry.get_agent(agent_id)
    except ValueError as e:
        return f"(không gọi được agent {agent_id}: {e})"

    state = AgentState(
        messages=history + [HumanMessage(content=question)],
        user_id=_sub_user.get() or "",
        session_id=_sub_session.get() or "",
        context=dict(_sub_context.get() or {}),
    )

    try:
        result = agent.compile().invoke(state)
    except Exception as e:
        logger.error("Agent con %s lỗi: %s", agent_id, e, exc_info=True)
        return f"(agent {agent_id} gặp lỗi: {e})"

    ctx = result.get("context") if isinstance(result, dict) else getattr(result, "context", None)
    # Hút attachment TRƯỚC khi trả về: nếu để tầng trên tự gộp context thì agent gọi
    # sau sẽ ghi đè attachment của agent gọi trước.
    collector.absorb(ctx)

    msgs = result.get("messages") if isinstance(result, dict) else getattr(result, "messages", [])

    # Ghi lại tool THẬT mà agent con đã chạy. Orchestrator không thấy chúng vì agent
    # con chạy trong state riêng, và thiếu chúng thì /test không hiện được tool nào
    # đã chạy, còn `core/reroute` tưởng không có tool nào và bày nút hỏi-lại ngay
    # dưới một câu trả lời đúng.
    # CHỈ quét phần message sinh ra trong lượt này. Quét cả list là đọc luôn phần
    # history đã replay, nên tool của các lượt TRƯỚC bị báo lại như thể vừa chạy —
    # đã gặp thật: một câu hỏi về đăng nhập hiện kèm `explain_failures` của lượt
    # trước đó trong cùng session, và /test thì trông như log agent gọi được tool
    # của agent sản xuất.
    fresh = (msgs or [])[len(history) + 1:]

    inner, results = [], {}
    for m in fresh:
        for tc in (getattr(m, "tool_calls", None) or []):
            inner.append({"tool": tc.get("name"), "args": tc.get("args"),
                          "id": tc.get("id"), "via": agent_id})
        if m.__class__.__name__ == "ToolMessage" and getattr(m, "name", None):
            try:
                import ast
                parsed = ast.literal_eval(str(m.content))
                if isinstance(parsed, dict):
                    results[m.name] = parsed
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                pass
    collector.add_tool_calls(inner, results)
    for m in reversed(msgs or []):
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None) and m.content:
            text = str(m.content).strip()
            # Ghi lại nguyên văn câu trả lời của agent con để orchestrator có thể
            # trả thẳng ra cho user khi chỉ có một agent chạy (xem orchestrator).
            _last_answer.set((agent_id, text))
            return text
    return "(agent không trả lời được)"


# ── Ngữ cảnh dùng chung cho các tool, đặt bởi orchestrator mỗi lượt ─────────────
from contextvars import ContextVar  # noqa: E402

_sub_history: ContextVar[Optional[list]] = ContextVar("sub_history", default=None)
_sub_context: ContextVar[Optional[dict]] = ContextVar("sub_context", default=None)
_sub_user: ContextVar[Optional[str]] = ContextVar("sub_user", default=None)
_sub_session: ContextVar[Optional[str]] = ContextVar("sub_session", default=None)
# (agent_id, câu trả lời) của agent con chạy gần nhất.
_last_answer: ContextVar[Optional[tuple]] = ContextVar("sub_last", default=None)
# Các agent đã chạy trong lượt này, theo thứ tự.
_called: ContextVar[Optional[list]] = ContextVar("sub_called", default=None)


def bind(messages: list, context: dict, user_id: str, session_id: str) -> None:
    """Nối ngữ cảnh của lượt hiện tại vào các tool. Gọi ở đầu mỗi lượt."""
    _sub_history.set([m for m in messages if not isinstance(m, SystemMessage)][-_HISTORY_FOR_SUB:])
    _sub_context.set(context or {})
    _sub_user.set(user_id)
    _sub_session.set(session_id)
    _last_answer.set(None)
    _called.set([])


def called_agents() -> list:
    return list(_called.get() or [])


def last_answer() -> Optional[tuple]:
    return _last_answer.get()


def _mark(agent_id: str) -> None:
    lst = _called.get()
    if lst is not None:
        lst.append(agent_id)


_BY_ID = {}


def dispatch(agent_id: str, question: str) -> str:
    """
    Gọi một agent theo id. Dùng cho đường tắt của orchestrator.

    Đi qua đúng `_run` như khi mô hình gọi tool, nên attachment vẫn được hút vào
    collector và `last_answer` vẫn được ghi — đường tắt không phải một nhánh code
    song song mà chỉ là bỏ bước chọn.
    """
    _mark(agent_id)
    return _run(agent_id, question)


def ask_production_data(question: str) -> str:
    _mark("historical_analytics")
    return _run("historical_analytics", question)


def ask_logs(question: str) -> str:
    _mark("log_analysis")
    return _run("log_analysis", question)


def ask_equipment(question: str) -> str:
    _mark("equipment_health")
    return _run("equipment_health", question)


def ask_camera_service(question: str) -> str:
    _mark("service_management")
    return _run("service_management", question)


ask_production_data_tool = BaseTool.create_tool(
    func=ask_production_data,
    metadata=ToolMetadata(
        name="ask_production_data",
        description=(
            "Số liệu sản xuất và lịch sử: sản lượng, pass/fail, pass rate, xu hướng, "
            "nguyên nhân fail kèm ảnh sản phẩm lỗi, so sánh hai kỳ, dừng máy/uptime, "
            "chỉ tiêu, bản giao ca, lịch sử load recipe, xuất báo cáo. "
            "Dùng cho MỌI câu về SẢN PHẨM fail — sản phẩm lỗi nằm trong database, "
            "không nằm trong file log."
        ),
        category="agent",
    ),
    args_schema=AskAgentArgs,
)

ask_logs_tool = BaseTool.create_tool(
    func=ask_logs,
    metadata=ToolMetadata(
        name="ask_logs",
        description=(
            "File log và audit log: dòng ERROR/WARNING, traceback, vì sao service "
            "restart, dung lượng log, và MỌI câu về NGƯỜI DÙNG — ai đăng nhập, ai "
            "load/sửa recipe, ai tạo user, bao nhiêu người đăng nhập. "
            "Câu 'hôm nay bao nhiêu người đăng nhập' thuộc đây, không thuộc số liệu "
            "sản xuất."
        ),
        category="agent",
    ),
    args_schema=AskAgentArgs,
)

ask_equipment_tool = BaseTool.create_tool(
    func=ask_equipment,
    metadata=ToolMetadata(
        name="ask_equipment",
        description=(
            "Sức khoẻ thiết bị đo từ log có cấu trúc: xung reject và cơ cấu đẩy phôi, "
            "trigger có bỏ sót sản phẩm không, nhịp cảm biến/băng tải, module nào lỗi "
            "hoặc không khởi tạo được."
        ),
        category="agent",
    ),
    args_schema=AskAgentArgs,
)

ask_camera_service_tool = BaseTool.create_tool(
    func=ask_camera_service,
    metadata=ToolMetadata(
        name="ask_camera_service",
        description=(
            "Trạng thái tiến trình camera service: đang chạy hay không, WebSocket có "
            "kết nối, khởi động/dừng service."
        ),
        category="agent",
    ),
    args_schema=AskAgentArgs,
)

AGENT_TOOLS = [
    ask_production_data_tool,
    ask_logs_tool,
    ask_equipment_tool,
    ask_camera_service_tool,
]
