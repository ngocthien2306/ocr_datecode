"""
Historical Analytics Agent
Specializes in analyzing production data, trends, and recipe history
"""

from typing import Any, Dict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent_app.core.llm import make_llm

from agent_app.base.base_agent import BaseAgent, AgentState
from agent_app.core.registry import AgentRegistry
from agent_app.core.attachments import (
    charts_from_tool_result,
    files_from_tool_result,
    images_from_tool_result,
    kpis_from_tool_result,
    strip_for_llm,
    tables_from_tool_result,
)
from agent_app.core.ui_options import options_from_tool_result
from agent_app.tools.report_tools import generate_report_tool
from agent_app.tools.analytics_tools import (
    compare_periods_tool,
    get_downtime_tool,
    get_target_progress_tool,
    get_pass_fail_stats_tool,
    get_production_summary_tool,
    get_recipe_load_history_tool,
    list_recipes_tool,
    explain_failures_tool,
)
from agent_app.prompts.historical_prompts import (
    HISTORICAL_ANALYTICS_SYSTEM_PROMPT,
    build_date_context,
)
import logging

logger = logging.getLogger(__name__)


@AgentRegistry.register("historical_analytics")
class HistoricalAnalyticsAgent(BaseAgent):
    """
    Agent for historical data analysis and production insights

    Capabilities:
    - Pass/fail statistics and trends
    - Production summaries by recipe/camera/time
    - Recipe load history tracking
    - User activity analysis
    - Xuất báo cáo ra file HTML/PDF/Excel/CSV/JSON
    - So sánh hai kỳ sản xuất, tính sẵn chênh lệch
    - Sản lượng theo ca làm việc
    - Thời gian dừng dây chuyền
    - Đối chiếu sản lượng với chỉ tiêu
    """

    def __init__(self, agent_id: str, model_name: Optional[str] = None, temperature: float = 0.3, **kwargs):
        super().__init__(agent_id=agent_id, model_name=model_name, temperature=temperature, **kwargs)

    def get_tools(self) -> List[Any]:
        """Return analytics tools"""
        return [
            list_recipes_tool,
            get_pass_fail_stats_tool,
            get_production_summary_tool,
            get_recipe_load_history_tool,
            explain_failures_tool,
            generate_report_tool,
            compare_periods_tool,
            get_downtime_tool,
            get_target_progress_tool,
        ]

    def get_system_prompt(self) -> str:
        """Return system prompt for historical analytics"""
        return HISTORICAL_ANALYTICS_SYSTEM_PROMPT

    def build_graph(self) -> StateGraph:
        """Build LangGraph workflow for analytics agent"""

        # Create LLM with tools
        llm = make_llm(self.model_name, self.temperature)
        llm_with_tools = llm.bind_tools(self.tools)

        def call_model(state: AgentState):
            """Call LLM to decide next action"""
            messages = state.messages

            # System prompt dựng RIÊNG cho lời gọi LLM, KHÔNG đưa vào state.
            #
            # Bản cũ nối SystemMessage vào chính danh sách rồi trả về state,
            # kéo theo ba hậu quả: (1) 6,2 KB prompt bị ghi vào MongoDB mỗi
            # session và replay lại mỗi lượt; (2) lượt sau thấy đã có
            # SystemMessage nên bỏ qua nhánh này ⇒ build_date_context() không
            # bao giờ chạy lại, ngày tháng đóng băng ở lượt đầu tiên; (3) chiếm
            # một suất trong MAX_HISTORY_MESSAGES.
            #
            # Ngày tháng phải tính tại đây chứ không phải trong __init__ —
            # agent instance bị AgentRegistry cache vĩnh viễn.
            llm_input = [
                SystemMessage(content=self.system_prompt + build_date_context())
            ] + [m for m in messages if not isinstance(m, SystemMessage)]

            response = llm_with_tools.invoke(llm_input)

            # Log tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                logger.info(f"LLM requested {len(response.tool_calls)} tool calls")
                for tool_call in response.tool_calls:
                    logger.info(f"Tool: {tool_call.get('name')} with args: {tool_call.get('args')}")
            else:
                logger.info("LLM generated final response")

            return {
                "messages": messages + [response]
            }

        def execute_tools(state: AgentState):
            """Execute requested tools"""
            last_message = state.messages[-1]
            tool_calls = last_message.tool_calls if hasattr(last_message, 'tool_calls') else []

            if not tool_calls:
                return {"messages": state.messages}

            # Execute tools
            tool_messages = []
            ui_options = None
            ui_images: List[Any] = []
            ui_charts: List[Any] = []
            ui_files: List[Any] = []
            ui_kpis: List[Any] = []
            ui_tables: List[Any] = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})

                logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

                # Find and execute tool
                tool_func = None
                for tool in self.tools:
                    if hasattr(tool, 'name') and tool.name == tool_name:
                        tool_func = tool.func
                        break

                if tool_func:
                    # Execute tool (all tools are now sync)
                    result = tool_func(**tool_args)
                    logger.info(f"Tool {tool_name} result: {result}")

                    # Kết quả có thứ cần user chọn (recipe mơ hồ / chưa nêu
                    # recipe nào) thì đẩy ra thành nút bấm cho FE.
                    ui_options = options_from_tool_result(tool_name, result) or ui_options

                    # Ảnh + biểu đồ suy tất định từ kết quả tool (không qua LLM,
                    # nên hình vẽ luôn khớp con số trong văn bản).
                    ui_images += images_from_tool_result(tool_name, result)
                    ui_charts += charts_from_tool_result(tool_name, tool_args, result)
                    ui_files += files_from_tool_result(tool_name, result)
                    # KPI và bảng phải cùng một tool, cùng một phạm vi.
                    #
                    # Giữ "bộ KPI đầu tiên" rồi cộng dồn bảng là sai: nếu LLM gọi
                    # get_pass_fail_stats trước compare_periods trong cùng lượt,
                    # người xem thấy dãy ô một kỳ (không có chênh lệch) nằm ngay
                    # trên một bảng so sánh hai kỳ. Lấy KPI của CHÍNH tool đã
                    # dựng bảng, và nếu chưa có bảng thì lấy tool đầu tiên có KPI.
                    tbl = tables_from_tool_result(tool_name, result)
                    kp = kpis_from_tool_result(tool_name, result)
                    if tbl:
                        ui_tables += tbl
                        if kp:
                            ui_kpis = kp
                    elif kp and not ui_kpis and not ui_tables:
                        ui_kpis = kp

                    # Create tool message
                    from langchain_core.messages import ToolMessage
                    tool_messages.append(
                        ToolMessage(
                            content=str(strip_for_llm(result)),
                            tool_call_id=tool_call.get("id", ""),
                            name=tool_name
                        )
                    )
                else:
                    logger.error(f"Tool {tool_name} not found")

            # Gộp với phần đã có trong context, không ghi đè: đồ thị chạy nhiều
            # vòng call_model → execute_tools, và ghi đè làm mất bảng của vòng
            # trước khi vòng sau gọi thêm tool.
            extra: Dict[str, Any] = {}
            prev = state.context or {}
            if ui_options:
                extra["ui_options"] = ui_options
            if ui_images:
                extra["ui_images"] = ui_images[:8]
            if ui_charts:
                extra["ui_charts"] = ui_charts[:4]
            if ui_files:
                extra["ui_files"] = ui_files[:4]
            if ui_kpis:
                extra["ui_kpis"] = ui_kpis[:6]
            elif prev.get("ui_kpis"):
                extra["ui_kpis"] = prev["ui_kpis"]
            merged_tables = (prev.get("ui_tables") or []) + ui_tables
            if merged_tables:
                extra["ui_tables"] = merged_tables[:2]

            return {
                "messages": state.messages + tool_messages,
                "context": {**state.context, **extra} if extra else state.context,
            }

        def should_continue(state: AgentState):
            """Decide whether to continue or end"""
            last_message = state.messages[-1]

            # If LLM called tools, execute them
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"

            # Otherwise, end
            return "end"

        # Build graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("call_model", call_model)
        workflow.add_node("execute_tools", execute_tools)

        # Set entry point
        workflow.set_entry_point("call_model")

        # Add conditional edges
        workflow.add_conditional_edges(
            "call_model",
            should_continue,
            {
                "tools": "execute_tools",
                "end": END
            }
        )

        # After executing tools, go back to model
        workflow.add_edge("execute_tools", "call_model")

        return workflow.compile()
