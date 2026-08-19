"""
Log Analysis Agent
Đọc log file và audit log, giải thích nguyên nhân sự cố.
"""

from typing import List, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage
from agent_app.core.llm import make_llm

from agent_app.base.base_agent import BaseAgent, AgentState
from agent_app.core.registry import AgentRegistry
from agent_app.core.attachments import strip_for_llm
from agent_app.tools.log_tools import (
    list_log_sources_tool,
    read_log_tail_tool,
    search_logs_tool,
    summarize_log_errors_tool,
    get_audit_logs_tool,
)
from agent_app.prompts.log_prompts import LOG_ANALYSIS_SYSTEM_PROMPT
from agent_app.prompts.historical_prompts import build_date_context
import logging

logger = logging.getLogger(__name__)


@AgentRegistry.register("log_analysis")
class LogAnalysisAgent(BaseAgent):
    """
    Agent đọc log và chẩn đoán sự cố

    Capabilities:
    - Liệt kê các nhóm log và dung lượng
    - Đọc phần cuối log, lọc theo mức và từ khoá
    - Tìm kiếm xuyên nhiều ngày và nhiều nhóm log
    - Gom ERROR/WARNING thành nhóm vấn đề và giải thích nguyên nhân
    - Tra audit log: ai đã thao tác gì trên hệ thống
    """

    def __init__(self, agent_id: str, model_name: Optional[str] = None, temperature: float = 0.2, **kwargs):
        # temperature thấp hơn analytics: chẩn đoán log mà bịa thêm chi tiết thì
        # người vận hành sẽ đi sửa nhầm chỗ.
        super().__init__(agent_id=agent_id, model_name=model_name, temperature=temperature, **kwargs)

    def get_tools(self) -> List[Any]:
        return [
            list_log_sources_tool,
            read_log_tail_tool,
            search_logs_tool,
            summarize_log_errors_tool,
            get_audit_logs_tool,
        ]

    def get_system_prompt(self) -> str:
        return LOG_ANALYSIS_SYSTEM_PROMPT

    def build_graph(self) -> StateGraph:
        """Đồ thị giống historical_agent: gọi model → chạy tool → quay lại model."""

        llm = make_llm(self.model_name, self.temperature)
        llm_with_tools = llm.bind_tools(self.tools)

        def call_model(state: AgentState):
            messages = state.messages

            # System prompt dựng riêng cho lời gọi, KHÔNG ghi vào state — xem
            # ghi chú dài trong historical_agent.py về lý do.
            llm_input = [
                SystemMessage(content=self.system_prompt + build_date_context())
            ] + [m for m in messages if not isinstance(m, SystemMessage)]

            response = llm_with_tools.invoke(llm_input)

            if hasattr(response, "tool_calls") and response.tool_calls:
                logger.info(f"LLM requested {len(response.tool_calls)} tool calls")
                for tc in response.tool_calls:
                    logger.info(f"Tool: {tc.get('name')} with args: {tc.get('args')}")
            else:
                logger.info("LLM generated final response")

            return {"messages": messages + [response]}

        def execute_tools(state: AgentState):
            last_message = state.messages[-1]
            tool_calls = last_message.tool_calls if hasattr(last_message, "tool_calls") else []
            if not tool_calls:
                return {"messages": state.messages}

            from langchain_core.messages import ToolMessage
            tool_messages = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

                tool_func = None
                for tool in self.tools:
                    if hasattr(tool, "name") and tool.name == tool_name:
                        tool_func = tool.func
                        break

                if tool_func:
                    result = tool_func(**tool_args)
                    # Log kết quả log-tool thì rất dài (hàng trăm dòng); chỉ ghi
                    # phần đầu để chính file log của agent không phình ra.
                    logger.info(f"Tool {tool_name} result: {str(result)[:600]}")
                    tool_messages.append(
                        ToolMessage(
                            content=str(strip_for_llm(result)),
                            tool_call_id=tool_call.get("id", ""),
                            name=tool_name,
                        )
                    )
                else:
                    logger.error(f"Tool {tool_name} not found")

            return {"messages": state.messages + tool_messages, "context": state.context}

        def should_continue(state: AgentState):
            last_message = state.messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return "end"

        workflow = StateGraph(AgentState)
        workflow.add_node("call_model", call_model)
        workflow.add_node("execute_tools", execute_tools)
        workflow.set_entry_point("call_model")
        workflow.add_conditional_edges(
            "call_model", should_continue, {"tools": "execute_tools", "end": END}
        )
        workflow.add_edge("execute_tools", "call_model")

        return workflow.compile()
