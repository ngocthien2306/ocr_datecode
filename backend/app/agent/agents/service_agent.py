"""
Service Management Agent
AI Agent chuyên quản lý services (camera_management, inference, etc.)
"""

from typing import List, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage
from app.agent.base.base_agent import BaseAgent, AgentState
from app.agent.core.registry import AgentRegistry
from app.agent.tools.service_tools import (
    check_service_status_tool,
    start_service_tool,
    stop_service_tool,
    get_service_logs_tool
)
import logging
import os

from app.agent.prompts.camera_service_prompts import CAMERA_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@AgentRegistry.register("service_management")
class ServiceManagementAgent(BaseAgent):
    """
    Agent chuyên quản lý services (camera_management, inference, etc.)

    Capabilities:
    - Check service status (process + WebSocket)
    - Start/Stop services with confirmation
    - Diagnose service issues
    - View and analyze logs
    - Provide troubleshooting guidance
    """

    def get_system_prompt(self) -> str:
        return CAMERA_SYSTEM_PROMPT
    def get_tools(self) -> List[Any]:
        """Return tools for service management"""
        return [
            check_service_status_tool,
            start_service_tool,
            stop_service_tool,
            get_service_logs_tool
        ]

    def build_graph(self) -> StateGraph:
        """Build LangGraph state machine"""

        # Get OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Initialize LLM with tools
        llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            streaming=True,
            api_key=api_key
        )
        llm_with_tools = llm.bind_tools(self.tools)

        # Define graph
        workflow = StateGraph(AgentState)

        # ====================================================================
        # Node: Call Model
        # ====================================================================
        def call_model(state: AgentState) -> AgentState:
            """
            Call LLM with tools

            The LLM decides:
            - If it needs to use a tool
            - If it has enough information to respond
            """
            messages = state.messages.copy()

            # Add system prompt if not present
            if not any(isinstance(m, SystemMessage) for m in messages):
                messages = [SystemMessage(content=self.get_system_prompt())] + messages

            # Call LLM
            response = llm_with_tools.invoke(messages)

            # Add response to messages
            state.messages.append(response)

            # Determine next step
            if hasattr(response, 'tool_calls') and response.tool_calls:
                state.next_step = "execute_tools"
                logger.info(f"LLM requested {len(response.tool_calls)} tool calls")
            else:
                state.next_step = "end"
                logger.info("LLM generated final response")

            return state

        # ====================================================================
        # Node: Execute Tools
        # ====================================================================
        def execute_tools(state: AgentState) -> AgentState:
            """
            Execute tool calls requested by LLM

            Each tool returns a result that is added to messages
            """
            last_message = state.messages[-1]

            if not hasattr(last_message, 'tool_calls'):
                logger.warning("No tool calls found in last message")
                state.next_step = "call_model"
                return state

            for tool_call in last_message.tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})

                logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

                # Find matching tool
                tool = next((t for t in self.tools if t.name == tool_name), None)

                if tool:
                    try:
                        # Execute tool
                        result = tool.invoke(tool_args)
                        logger.info(f"Tool {tool_name} result: {result}")

                        # Add tool result to messages
                        state.messages.append(
                            ToolMessage(
                                content=str(result),
                                tool_call_id=tool_call.get("id", "unknown")
                            )
                        )
                    except Exception as e:
                        logger.error(f"Error executing tool {tool_name}: {e}")
                        state.messages.append(
                            ToolMessage(
                                content=f"Error: {str(e)}",
                                tool_call_id=tool_call.get("id", "unknown")
                            )
                        )
                else:
                    logger.warning(f"Tool not found: {tool_name}")
                    state.messages.append(
                        ToolMessage(
                            content=f"Tool '{tool_name}' not found",
                            tool_call_id=tool_call.get("id", "unknown")
                        )
                    )

            # After executing tools, go back to model to process results
            state.next_step = "call_model"
            return state

        # ====================================================================
        # Node: Route
        # ====================================================================
        def route(state: AgentState) -> str:
            """Route to next step based on state"""
            next_step = state.next_step or "end"
            logger.debug(f"Routing to: {next_step}")
            return next_step

        # ====================================================================
        # Build Graph
        # ====================================================================

        # Add nodes
        workflow.add_node("call_model", call_model)
        workflow.add_node("execute_tools", execute_tools)

        # Set entry point
        workflow.set_entry_point("call_model")

        # Add conditional edges from call_model
        workflow.add_conditional_edges(
            "call_model",
            route,
            {
                "execute_tools": "execute_tools",
                "end": END
            }
        )

        # Add edge from execute_tools back to call_model
        workflow.add_edge("execute_tools", "call_model")

        logger.info("✅ ServiceAgent graph built successfully")

        return workflow
