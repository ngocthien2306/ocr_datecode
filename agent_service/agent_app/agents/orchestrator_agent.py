"""
Orchestrator Agent
Master agent that intelligently routes user requests to specialized agents
"""

from typing import List, Any, Dict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent_app.core.llm import make_llm
import json
import logging

from agent_app.base.base_agent import BaseAgent, AgentState
from agent_app.core.registry import AgentRegistry
from agent_app.prompts.orchestrator_prompts import ORCHESTRATOR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Ví dụ câu hỏi hiện lên khi orchestrator không hiểu ý user. Mỗi câu phải rơi
# gọn vào một agent chuyên biệt, và đều là việc CHỈ ĐỌC.
ENTRY_QUESTIONS = [
    "Hôm nay sản xuất bao nhiêu sản phẩm?",
    "Camera nào fail nhiều nhất hôm nay?",
    "Xu hướng pass rate 7 ngày qua",
    "Camera service có đang chạy không?",
]


@AgentRegistry.register("orchestrator")
class OrchestratorAgent(BaseAgent):
    """
    Master orchestrator agent that routes requests to specialized agents

    This agent does NOT answer questions directly.
    Instead, it analyzes user intent and routes to the appropriate specialized agent.

    Routing targets:
    - service_management: Service status, start/stop
    - historical_analytics: Statistics, trends, history
    - log_analysis: System logs, error diagnosis, audit trail
    - (future agents...)
    """

    def __init__(self, agent_id: str, model_name: Optional[str] = None, temperature: float = 0.2, **kwargs):
        super().__init__(agent_id=agent_id, model_name=model_name, temperature=temperature, **kwargs)
        # Lower temperature for more consistent routing decisions
        # Set system prompt as attribute
        self.system_prompt = self.get_system_prompt()
        # Build and compile graph immediately for orchestrator
        workflow = self.build_graph()
        self.compiled_graph = workflow  # Already compiled
        self.graph = None  # Don't store uncompiled graph

    def get_tools(self) -> List[Any]:
        """Orchestrator doesn't use tools - it routes to other agents"""
        return []

    def get_system_prompt(self) -> str:
        """Return orchestrator system prompt"""
        return ORCHESTRATOR_SYSTEM_PROMPT

    def build_graph(self) -> StateGraph:
        """Build LangGraph workflow for orchestrator"""

        llm = make_llm(self.model_name, self.temperature)

        def analyze_intent(state: AgentState):
            """
            Analyze user intent and decide which agent to route to

            Returns routing decision as JSON:
            {
                "agent_id": "service_management" | "historical_analytics" | "log_analysis" | null,
                "confidence": 0.95,
                "reason": "Explanation",
                "clarification": null | "Question to ask user"
            }
            """
            messages = state.messages

            # Add system prompt
            if not any(isinstance(msg, SystemMessage) for msg in messages):
                messages = [SystemMessage(content=self.system_prompt)] + messages

            # Add instruction to return JSON
            last_user_msg = None
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    last_user_msg = msg.content
                    break

            if last_user_msg:
                # Inject JSON format instruction
                json_instruction = HumanMessage(content=f"""
User query: "{last_user_msg}"

Analyze this query and return routing decision in JSON format:
{{
  "agent_id": "service_management" | "historical_analytics" | "log_analysis" | null,
  "confidence": 0.0-1.0,
  "reason": "explanation",
  "clarification": null | "question if needed"
}}
""")
                messages = messages[:-1] + [json_instruction]

            response = llm.invoke(messages)

            logger.info(f"Orchestrator analysis: {response.content}")

            # Store decision in context only, DO NOT add to messages
            # (we don't want user to see raw JSON)
            return {
                "messages": state.messages,  # Keep original messages only
                "context": {
                    **state.context,
                    "orchestrator_decision": response.content
                }
            }

        def parse_routing_decision(state: AgentState):
            """
            Parse routing decision and determine next step

            Updates state.context and returns routing path
            """
            decision_str = state.context.get("orchestrator_decision", "{}")

            try:
                # Extract JSON from response (may be wrapped in markdown)
                if "```json" in decision_str:
                    decision_str = decision_str.split("```json")[1].split("```")[0].strip()
                elif "```" in decision_str:
                    decision_str = decision_str.split("```")[1].split("```")[0].strip()

                decision = json.loads(decision_str)

                agent_id = decision.get("agent_id")
                confidence = decision.get("confidence", 0)
                clarification = decision.get("clarification")

                logger.info(f"Routing decision: agent={agent_id}, confidence={confidence}")

                # Update context with routing info
                updated_context = {
                    **state.context,
                    "routing_agent_id": agent_id,
                    "routing_confidence": confidence,
                    "routing_reason": decision.get("reason")
                }

                # Need clarification
                if clarification or agent_id is None or confidence < 0.7:
                    updated_context["clarification_question"] = clarification or "Xin lỗi, tôi chưa hiểu rõ câu hỏi. Bạn có thể nói rõ hơn không?"
                    updated_context["next_action"] = "ask_clarification"
                    return {"context": updated_context}

                # Valid routing
                # Danh sách này phải khớp với các agent đã đăng ký trong
                # AgentRegistry; thiếu một cái là câu hỏi đúng vẫn bị trả lời
                # "không xử lý được" dù agent tồn tại và prompt đã route đúng.
                if agent_id in ["service_management", "historical_analytics", "log_analysis"]:
                    updated_context["next_action"] = "route_to_agent"
                    return {"context": updated_context}

                # Unknown agent
                logger.warning(f"Unknown agent_id: {agent_id}")
                updated_context["clarification_question"] = "Xin lỗi, tôi không thể xử lý yêu cầu này. Vui lòng thử lại."
                updated_context["next_action"] = "ask_clarification"
                return {"context": updated_context}

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse routing decision: {e}")
                return {
                    "context": {
                        **state.context,
                        "clarification_question": "Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại.",
                        "next_action": "ask_clarification"
                    }
                }
            except Exception as e:
                logger.error(f"Error in routing: {e}")
                return {
                    "context": {
                        **state.context,
                        "clarification_question": "Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại.",
                        "next_action": "ask_clarification"
                    }
                }

        def route_to_specialized_agent(state: AgentState):
            """
            Route to specialized agent and get response
            """
            agent_id = state.context.get("routing_agent_id")
            logger.info(f"🎯 route_to_specialized_agent called with agent_id: {agent_id}")

            try:
                # Get specialized agent
                specialized_agent = AgentRegistry.get_agent(agent_id)

                logger.info(f"✅ Got specialized agent: {agent_id}, type: {type(specialized_agent)}")

                # Get compiled graph (trigger compilation if needed)
                graph = specialized_agent.compile()
                logger.info(f"📊 Invoking specialized agent graph")

                # Execute specialized agent with same state
                result = graph.invoke(state)
                logger.info(f"✅ Specialized agent returned result with {len(result.get('messages', []))} messages")

                # Extract final response from specialized agent.
                # PHẢI chuyển tiếp cả context: agent con đặt ui_options trong
                # đó, chỉ trả messages là mất nút bấm khi đi qua orchestrator.
                if isinstance(result, dict) and "messages" in result:
                    return {
                        "messages": result["messages"],
                        "context": {**state.context, **(result.get("context") or {})},
                    }

                return {"messages": state.messages}

            except Exception as e:
                logger.error(f"Error routing to {agent_id}: {e}")
                error_msg = AIMessage(content=f"Xin lỗi, không thể kết nối với {agent_id}. Lỗi: {str(e)}")
                return {"messages": state.messages + [error_msg]}

        def ask_for_clarification(state: AgentState):
            """
            Ask user for clarification.

            Kèm sẵn ví dụ câu hỏi trả lời được. Không có thì user bị cụt đường:
            agent nói "chưa hiểu" mà không gợi ý gì, người dùng phải tự đoán
            xem hệ thống làm được những gì.
            """
            clarification = state.context.get("clarification_question", "Bạn có thể nói rõ hơn không?")

            clarification_msg = AIMessage(content=clarification)

            return {
                "messages": state.messages + [clarification_msg],
                "context": {**state.context, "ui_suggestions": ENTRY_QUESTIONS},
            }

        def decide_next_step(state: AgentState) -> str:
            """
            Conditional edge function to decide next step based on routing decision
            """
            next_action = state.context.get("next_action", "end")
            logger.info(f"🔀 Deciding next step: {next_action}")
            return next_action

        # Build graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze_intent", analyze_intent)
        workflow.add_node("parse_decision", parse_routing_decision)
        workflow.add_node("route_to_agent", route_to_specialized_agent)
        workflow.add_node("ask_clarification", ask_for_clarification)

        # Set entry point
        workflow.set_entry_point("analyze_intent")

        # analyze_intent → parse_decision
        workflow.add_edge("analyze_intent", "parse_decision")

        # Conditional routing from parse_decision
        workflow.add_conditional_edges(
            "parse_decision",
            decide_next_step,
            {
                "route_to_agent": "route_to_agent",
                "ask_clarification": "ask_clarification",
                "end": END
            }
        )

        # Both end after their work
        workflow.add_edge("route_to_agent", END)
        workflow.add_edge("ask_clarification", END)

        return workflow.compile()
