"""
Agent Registry
Central registry for all agents
"""

from typing import Dict, Type, Optional, List, Any
from app.agent.base.base_agent import BaseAgent
import logging

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Singleton registry for all agents

    Benefits:
    - Central place to manage agents
    - Lazy initialization
    - Easy to add new agents
    """

    _agents: Dict[str, Type[BaseAgent]] = {}
    _instances: Dict[str, BaseAgent] = {}

    @classmethod
    def register(cls, agent_id: str):
        """
        Decorator to register an agent

        Usage:
            @AgentRegistry.register("my_agent")
            class MyAgent(BaseAgent):
                ...
        """
        def decorator(agent_class: Type[BaseAgent]):
            if agent_id in cls._agents:
                logger.warning(f"Agent '{agent_id}' already registered, overwriting")

            cls._agents[agent_id] = agent_class
            logger.info(f"✅ Registered agent: {agent_id} ({agent_class.__name__})")
            return agent_class

        return decorator

    @classmethod
    def get_agent(cls, agent_id: str, **kwargs) -> BaseAgent:
        """
        Get or create agent instance

        Uses caching to avoid recreating agents

        Args:
            agent_id: ID of agent to get
            **kwargs: Additional arguments for agent constructor

        Returns:
            Agent instance
        """
        # Create cache key
        model_name = kwargs.get('model_name', 'default')
        cache_key = f"{agent_id}:{model_name}"

        # Return cached instance if exists
        if cache_key in cls._instances:
            logger.debug(f"Using cached agent: {cache_key}")
            return cls._instances[cache_key]

        # Check if agent is registered
        if agent_id not in cls._agents:
            available = ', '.join(cls._agents.keys())
            raise ValueError(
                f"Agent '{agent_id}' not registered. "
                f"Available agents: {available}"
            )

        # Create new instance
        agent_class = cls._agents[agent_id]
        instance = agent_class(agent_id=agent_id, **kwargs)

        # Cache it
        cls._instances[cache_key] = instance
        logger.info(f"✅ Created new agent instance: {cache_key}")

        return instance

    @classmethod
    def list_agents(cls) -> List[Dict[str, Any]]:
        """
        List all registered agents

        Returns:
            List of agent metadata
        """
        return [
            {
                "agent_id": agent_id,
                "class_name": agent_class.__name__,
                "description": agent_class.__doc__.strip() if agent_class.__doc__ else "No description"
            }
            for agent_id, agent_class in cls._agents.items()
        ]

    @classmethod
    def clear_cache(cls):
        """Clear agent instance cache (useful for testing)"""
        cls._instances.clear()
        logger.info("Agent cache cleared")
