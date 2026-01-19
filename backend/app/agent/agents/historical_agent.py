"""
Historical Analytics Agent
Specializes in analyzing production data, trends, and recipe history
"""

from typing import List, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.agent.base.base_agent import BaseAgent, AgentState
from app.agent.core.registry import AgentRegistry
from app.agent.tools.analytics_tools import (
    get_pass_fail_stats_tool,
    get_production_summary_tool,
    get_recipe_load_history_tool
)
from app.agent.prompts.historical_prompts import HISTORICAL_ANALYTICS_SYSTEM_PROMPT
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
    """

    def __init__(self, agent_id: str, model_name: str = "gpt-4o-mini", temperature: float = 0.3, **kwargs):
        super().__init__(agent_id=agent_id, model_name=model_name, temperature=temperature, **kwargs)

    def get_tools(self) -> List[Any]:
        """Return analytics tools"""
        return [
            get_pass_fail_stats_tool,
            get_production_summary_tool,
            get_recipe_load_history_tool
        ]

    def get_system_prompt(self) -> str:
        """Return system prompt for historical analytics"""
        return HISTORICAL_ANALYTICS_SYSTEM_PROMPT

    def build_graph(self) -> StateGraph:
        """Build LangGraph workflow for analytics agent"""

        # Create LLM with tools
        llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)
        llm_with_tools = llm.bind_tools(self.tools)

        def call_model(state: AgentState):
            """Call LLM to decide next action"""
            messages = state.messages

            # Add system prompt if not present
            if not any(isinstance(msg, SystemMessage) for msg in messages):
                messages = [SystemMessage(content=self.system_prompt)] + messages

            response = llm_with_tools.invoke(messages)

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

                    # Create tool message
                    from langchain_core.messages import ToolMessage
                    tool_messages.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call.get("id", ""),
                            name=tool_name
                        )
                    )
                else:
                    logger.error(f"Tool {tool_name} not found")

            return {
                "messages": state.messages + tool_messages
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
