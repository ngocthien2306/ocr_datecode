"""
Base Agent Class
Provides template for all AI agents
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class AgentState(BaseModel):
    """
    Base state for all agents

    This is passed through the LangGraph nodes
    """
    messages: List[BaseMessage] = Field(default_factory=list)
    user_id: str
    session_id: str
    context: Dict[str, Any] = Field(default_factory=dict)
    next_step: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class BaseAgent(ABC):
    """
    Abstract base class for all agents

    Template Method Pattern:
    - Defines the skeleton of agent execution
    - Subclasses implement specific behaviors

    Every agent must implement:
    - get_tools(): Return list of available tools
    - get_system_prompt(): Return system prompt
    - build_graph(): Construct LangGraph state machine
    """

    def __init__(
        self,
        agent_id: str,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        **kwargs
    ):
        self.agent_id = agent_id
        self.model_name = model_name
        self.temperature = temperature
        self.graph = None
        self.compiled_graph = None

        # Get tools from subclass
        self.tools = self.get_tools()

        logger.info(f"✅ Initialized {self.__class__.__name__} with {len(self.tools)} tools")

    @abstractmethod
    def get_tools(self) -> List[Any]:
        """
        Return list of tools this agent can use

        Example:
            return [check_status_tool, start_service_tool]
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Return the system prompt for this agent

        This defines the agent's personality and capabilities
        """
        pass

    @abstractmethod
    def build_graph(self) -> StateGraph:
        """
        Build the agent's state graph (workflow)

        This defines how the agent processes requests
        """
        pass

    def compile(self):
        """Compile the graph (cache for performance)"""
        if not self.compiled_graph:
            self.graph = self.build_graph()
            self.compiled_graph = self.graph.compile()
            logger.debug(f"Compiled graph for {self.agent_id}")
        return self.compiled_graph

    async def ainvoke(self, state: AgentState) -> AgentState:
        """
        Async invoke the agent

        Args:
            state: Initial state with user message

        Returns:
            Updated state with agent response
        """
        graph = self.compile()
        return await graph.ainvoke(state)

    async def astream(self, state: AgentState):
        """
        Stream agent execution (for real-time UI updates)

        Yields:
            State updates as they happen
        """
        graph = self.compile()
        async for chunk in graph.astream(state):
            yield chunk

    def invoke(self, state: AgentState) -> AgentState:
        """Synchronous invoke (for testing)"""
        graph = self.compile()
        return graph.invoke(state)
