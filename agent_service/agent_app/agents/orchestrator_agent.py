"""
Orchestrator: gọi các agent chuyên biệt NHƯ TOOL.

Bản trước định tuyến một chặng: một lượt LLM riêng trả về JSON
`{agent_id, confidence}`, rồi giao đứt câu hỏi cho đúng một agent. Kiểu đó kéo
theo ba hạn chế đi liền nhau — không tổng hợp được nhiều nguồn ("so sánh sản phẩm
fail với cảnh báo thiết bị cùng giờ" cần hai agent), không chẻ được câu nhiều bước,
và tốn một lượt LLM chỉ để chọn agent.

Nay orchestrator là một vòng lặp tool-calling bình thường, tool của nó là bốn agent
con. Chọn agent trở thành chọn tool, nên lượt LLM định tuyến biến mất; gọi mấy agent
cũng được, nên tổng hợp được.

## Vì sao có nhánh "trả thẳng"

Vòng lặp tool tiêu chuẩn luôn quay lại LLM sau khi tool chạy, để nó viết câu trả
lời cuối. Với MỘT agent thì lượt đó vừa tốn tiền vừa có hại: agent con đã viết xong
một câu trả lời đầy đủ, kèm số liệu; để orchestrator viết lại là mở đúng cửa cho
lớp bug đã mất cả ngày để dẹp — mô hình chép số sai, làm tròn khác, hoặc kể lại
bảng số theo cách khác con số trên biểu đồ.

Nên: gọi một agent thì câu trả lời của agent đó được trả NGUYÊN VĂN, không thêm lượt
LLM nào. Gọi từ hai agent trở lên mới có một lượt tổng hợp.

Đánh đổi phải biết: mô hình được dặn gọi HẾT các agent cần thiết trong CÙNG một
lượt. Nếu nó gọi một agent, đọc kết quả, rồi mới nhận ra cần agent thứ hai thì nhánh
trả thẳng đã kết thúc lượt. Chấp nhận được vì nút `reroute` đã cho người dùng đường
hỏi lại, và trường hợp một-agent là đa số áp đảo.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from agent_app.base.base_agent import BaseAgent, AgentState
from agent_app.core.i18n import apply_language, t
from agent_app.core.intent import match as intent_match
from agent_app.core.llm import make_llm
from agent_app.core.registry import AgentRegistry
from agent_app.prompts.orchestrator_prompts import ORCHESTRATOR_SYSTEM_PROMPT
from agent_app.tools import agent_tools

logger = logging.getLogger(__name__)

# Ví dụ câu hỏi hiện lên khi orchestrator không hiểu ý user. Mỗi câu phải rơi
# gọn vào một agent chuyên biệt, và đều là việc CHỈ ĐỌC.
ENTRY_QUESTIONS = [
    "Hôm nay sản xuất bao nhiêu sản phẩm?",
    "Camera nào fail nhiều nhất hôm nay?",
    "Xu hướng pass rate 7 ngày qua",
    "Camera service có đang chạy không?",
]

# Trần số vòng gọi tool. Một câu hỏi thật cần nhiều nhất là gọi vài agent một lượt;
# quá con số này gần như chắc chắn là mô hình lặp vô ích, và mỗi vòng là một lần chạy
# agent con nên đắt.
_MAX_ROUNDS = 3


@AgentRegistry.register("orchestrator")
class OrchestratorAgent(BaseAgent):
    """
    Điều phối: nhận câu hỏi, gọi một hoặc nhiều agent chuyên biệt, gộp kết quả.

    Không tự trả lời câu hỏi nghiệp vụ và không có tool dữ liệu nào của riêng nó —
    mọi số liệu phải đi qua một agent chuyên biệt để chỉ có một chỗ duy nhất chịu
    trách nhiệm về mỗi loại dữ liệu.
    """

    def __init__(self, agent_id: str, model_name: Optional[str] = None,
                 temperature: float = 0.2, **kwargs):
        super().__init__(agent_id=agent_id, model_name=model_name,
                         temperature=temperature, **kwargs)
        self.system_prompt = self.get_system_prompt()
        workflow = self.build_graph()
        self.compiled_graph = workflow
        self.graph = None

    def get_tools(self) -> List[Any]:
        return list(agent_tools.AGENT_TOOLS)

    def get_system_prompt(self) -> str:
        return ORCHESTRATOR_SYSTEM_PROMPT

    def build_graph(self) -> StateGraph:
        llm = make_llm(self.model_name, self.temperature)
        llm_with_tools = llm.bind_tools(self.get_tools())
        by_name = {tl.name: tl for tl in self.get_tools()}

        def fast_route(state: AgentState):
            """
            Đường tắt: câu hỏi khớp một cụm RÕ RÀNG thì gọi thẳng agent đó, bỏ qua
            lượt LLM chọn tool.

            Lượt LLM đó đo được ~1,2s và với đa số câu hỏi lặp lại hằng ngày
            ("hôm nay bao nhiêu sản phẩm", "ai đăng nhập") nó chỉ xác nhận lại điều
            một phép so chuỗi đã biết. Bảng cụm cố tình hẹp (xem `core/intent.py`):
            khớp nhiều agent hoặc không khớp gì thì trả về None và đi đường LLM, vì
            đường tắt không có mô hình để tự sửa nếu đoán sai.
            """
            last_user = None
            for m in reversed(state.messages or []):
                if m.__class__.__name__ == "HumanMessage":
                    last_user = str(m.content)
                    break

            agent_id = intent_match(last_user or "")
            ctx = dict(state.context or {})
            if not agent_id:
                ctx["fast_route"] = None
                return {"context": ctx}

            agent_tools.bind(
                messages=state.messages,
                context=state.context or {},
                user_id=state.user_id,
                session_id=state.session_id,
            )
            logger.info("Đường tắt → %s (không gọi LLM điều phối)", agent_id)
            ctx["fast_route"] = agent_id
            # `question` là chính câu user gõ: đường tắt chỉ dùng khi câu hỏi đã rõ,
            # nên không cần viết lại. Ngữ cảnh các lượt trước vẫn được truyền qua
            # `bind`, nên câu tiếp nối vẫn giải được.
            text = agent_tools.dispatch(agent_id, last_user or "")
            return {"messages": state.messages + [AIMessage(content=text)],
                    "context": ctx}

        def took_fast_route(state: AgentState) -> str:
            return "end" if (state.context or {}).get("fast_route") else "call_model"

        def call_model(state: AgentState):
            messages = state.messages

            # Nối ngữ cảnh lượt hiện tại vào các tool agent. Phải làm trước lượt
            # LLM đầu tiên: tool được gọi ngay sau đó và nó cần history để giải
            # những câu tiếp nối kiểu "còn hôm qua thì sao".
            agent_tools.bind(
                messages=messages,
                context=state.context or {},
                user_id=state.user_id,
                session_id=state.session_id,
            )

            if not any(isinstance(m, SystemMessage) for m in messages):
                # Dòng ngôn ngữ nối theo LƯỢT, không nối vào self.system_prompt:
                # instance được AgentRegistry cache vĩnh viễn, nối một lần thì
                # request sau vẫn dùng ngôn ngữ của request đầu tiên.
                messages = [SystemMessage(
                    content=apply_language(self.system_prompt)
                )] + messages

            response = llm_with_tools.invoke(messages)
            return {"messages": state.messages + [response]}

        def run_tools(state: AgentState):
            """
            Chạy các tool agent mà mô hình vừa yêu cầu.

            KHÔNG dùng `langgraph.prebuilt.ToolNode` ở đây, dù nó có sẵn:
            `AgentState.messages` là field Pydantic thường, không gắn reducer
            `add_messages`, nên giá trị trả về THAY THẾ cả danh sách. ToolNode chỉ
            trả các ToolMessage mới, và thế là toàn bộ hội thoại bị xoá, còn lại
            mỗi mấy ToolMessage — OpenAI từ chối ngay với "messages with role
            'tool' must be a response to a preceeding message with 'tool_calls'".
            Nối vào `state.messages` đúng như các agent con vẫn làm.
            """
            last = state.messages[-1] if state.messages else None
            calls = getattr(last, "tool_calls", None) or []
            out = []
            for call in calls:
                name = call.get("name")
                args = call.get("args") or {}
                tool = by_name.get(name)
                if tool is None:
                    logger.warning("Mô hình gọi tool không tồn tại: %s", name)
                    text = f"(không có tool tên {name})"
                else:
                    logger.info("Orchestrator gọi %s: %s", name,
                                str(args.get("question"))[:120])
                    try:
                        text = tool.func(**args)
                    except Exception as e:
                        logger.error("Tool %s lỗi: %s", name, e, exc_info=True)
                        text = f"(tool {name} lỗi: {e})"
                out.append(ToolMessage(content=str(text),
                                       tool_call_id=call.get("id") or name,
                                       name=name))
            return {"messages": state.messages + out}

        def passthrough(state: AgentState):
            """
            Trả NGUYÊN VĂN câu trả lời của agent con duy nhất đã chạy.

            Không gọi LLM ở đây, và đó là toàn bộ mục đích: câu trả lời của agent
            con đã hoàn chỉnh, thêm một lượt để orchestrator kể lại chỉ tạo cơ hội
            chép sai số.
            """
            last = agent_tools.last_answer()
            text = (last or (None, ""))[1] or t("Chưa lấy được dữ liệu.")
            return {"messages": state.messages + [AIMessage(content=text)]}

        def after_tools(state: AgentState) -> str:
            """
            Sau khi tool chạy: trả thẳng, tổng hợp, hay chạy tiếp?

            Đếm số agent KHÁC NHAU đã chạy, không đếm số lần gọi tool: mô hình đôi
            khi gọi cùng một agent hai lần cho hai khoảng thời gian, và đó vẫn là
            một nguồn duy nhất nên vẫn trả thẳng được câu trả lời cuối.
            """
            rounds = sum(1 for m in state.messages
                         if isinstance(m, AIMessage) and getattr(m, "tool_calls", None))
            if rounds >= _MAX_ROUNDS:
                logger.warning("Orchestrator đạt trần %d vòng gọi tool", _MAX_ROUNDS)
                return "passthrough"

            distinct = set(agent_tools.called_agents())
            if len(distinct) <= 1 and agent_tools.last_answer():
                return "passthrough"
            # Nhiều nguồn ⇒ cần một lượt LLM để gộp. Đây là lượt duy nhất
            # orchestrator tự viết chữ.
            return "synthesize"

        def should_continue(state: AgentState) -> str:
            last = state.messages[-1] if state.messages else None
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                return "tools"
            # Không gọi tool nào: mô hình đang hỏi lại hoặc chào. Gợi ý câu vào bài
            # để user không bị cụt đường.
            return "end"

        def finish_clarification(state: AgentState):
            return {
                "messages": state.messages,
                "context": {**(state.context or {}),
                            "ui_suggestions": [t(q) for q in ENTRY_QUESTIONS]},
            }

        workflow = StateGraph(AgentState)
        workflow.add_node("fast_route", fast_route)
        workflow.add_node("call_model", call_model)
        workflow.add_node("tools", run_tools)
        workflow.add_node("synthesize", call_model)
        workflow.add_node("passthrough", passthrough)
        workflow.add_node("clarify", finish_clarification)

        workflow.set_entry_point("fast_route")
        workflow.add_conditional_edges(
            "fast_route", took_fast_route,
            {"end": END, "call_model": "call_model"},
        )
        workflow.add_conditional_edges(
            "call_model", should_continue,
            {"tools": "tools", "end": "clarify"},
        )
        workflow.add_conditional_edges(
            "tools", after_tools,
            {"passthrough": "passthrough", "synthesize": "synthesize"},
        )
        # Sau khi tổng hợp, mô hình có thể muốn gọi thêm agent — cho phép, nhưng
        # `after_tools` sẽ chặn ở trần _MAX_ROUNDS.
        workflow.add_conditional_edges(
            "synthesize", should_continue,
            {"tools": "tools", "end": END},
        )
        workflow.add_edge("passthrough", END)
        workflow.add_edge("clarify", END)

        return workflow.compile()
