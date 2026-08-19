"""
Equipment Health Agent
Đọc các log kỹ thuật có cấu trúc và so với cấu hình trong DB.
"""

from typing import List, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage
from agent_app.core.llm import make_llm

from agent_app.base.base_agent import BaseAgent, AgentState
from agent_app.core.registry import AgentRegistry
from agent_app.core.attachments import strip_for_llm
from agent_app.tools.equipment_tools import (
    check_reject_timing_tool,
    check_sensor_pulse_tool,
    check_subsystem_health_tool,
    check_trigger_health_tool,
)
from agent_app.prompts.equipment_prompts import EQUIPMENT_SYSTEM_PROMPT
from agent_app.prompts.historical_prompts import build_date_context
import logging

logger = logging.getLogger(__name__)


@AgentRegistry.register("equipment_health")
class EquipmentHealthAgent(BaseAgent):
    """
    Agent theo dõi sức khoẻ thiết bị

    Capabilities:
    - Xung reject: cấu hình vs thực tế đo được
    - Độ tin cậy trigger: timeout, lỗi chụp ảnh, service restart
    - Cảm biến đầu vào: độ rộng xung và độ trôi
    - Hệ thống con báo lỗi khởi tạo
    """

    def __init__(self, agent_id: str, model_name: Optional[str] = None, temperature: float = 0.15, **kwargs):
        # temperature thấp nhất trong các agent: một kết luận sai ở đây kéo kỹ sư
        # bảo trì ra máy để tìm cái không tồn tại, và lần sau họ bỏ qua cả cảnh
        # báo thật.
        super().__init__(agent_id=agent_id, model_name=model_name, temperature=temperature, **kwargs)

    def get_tools(self) -> List[Any]:
        return [
            check_reject_timing_tool,
            check_trigger_health_tool,
            check_sensor_pulse_tool,
            check_subsystem_health_tool,
        ]

    def get_system_prompt(self) -> str:
        return EQUIPMENT_SYSTEM_PROMPT

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
