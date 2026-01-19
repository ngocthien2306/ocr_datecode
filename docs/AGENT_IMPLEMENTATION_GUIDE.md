# 🚀 AI Agent Implementation Guide

## 📋 Mục đích

Document này hướng dẫn chi tiết cách implement AI Agent system từng bước, từ setup môi trường đến deployment.

---

## 🎯 Implementation Phases

```
Phase 1: Foundation & Infrastructure (Day 1-2)
    ├── Setup dependencies
    ├── Create base classes
    ├── Build tool registry
    └── Test basic LangGraph workflow

Phase 2: Service Management Agent (Day 3-4)
    ├── Implement service tools
    ├── Create ServiceAgent
    ├── Build API endpoints
    └── Integration testing

Phase 3: Frontend UI (Day 5-6)
    ├── Create chat components
    ├── Implement streaming
    ├── Add service status indicator
    └── Polish UX

Phase 4: Enhancement & Testing (Day 7)
    ├── Add conversation memory
    ├── Error handling
    ├── Performance optimization
    └── End-to-end testing
```

---

## 📦 Step 1: Setup Dependencies

### 1.1 Install Required Packages

```bash
cd /Users/ngocthien.ai/Source/Projects/ocr_datecode/backend

# Activate virtual environment
source venv/bin/activate  # or your venv path

# Install LangGraph and dependencies
pip install langgraph==0.2.16
pip install langchain==0.3.0
pip install langchain-openai==0.2.0
pip install langchain-community==0.3.0
pip install langchain-mongodb==0.2.0

# Additional utilities
pip install tiktoken==0.7.0
pip install psutil==5.9.8

# Save to requirements
pip freeze > requirements.txt
```

### 1.2 Verify Installation

```python
# test_langgraph.py
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage
import os

class TestState(BaseModel):
    messages: List[BaseMessage]

# Test OpenAI connection
api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key loaded: {api_key[:10]}...")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
response = llm.invoke([HumanMessage(content="Say hello")])
print(f"OpenAI Response: {response.content}")

print("✅ LangGraph setup successful!")
```

---

## 🏗️ Step 2: Create Base Infrastructure

### 2.1 Create Directory Structure

```bash
cd backend/app
mkdir -p agent/{base,core,agents,tools,memory,prompts,utils}

# Create __init__.py files
touch agent/__init__.py
touch agent/base/__init__.py
touch agent/core/__init__.py
touch agent/agents/__init__.py
touch agent/tools/__init__.py
touch agent/memory/__init__.py
touch agent/prompts/__init__.py
touch agent/utils/__init__.py
```

### 2.2 Implement Base Agent Class

File: `backend/app/agent/base/base_agent.py`

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

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

        print(f"✅ Initialized {self.__class__.__name__} with {len(self.tools)} tools")

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
```

### 2.3 Implement Agent Registry

File: `backend/app/agent/core/registry.py`

```python
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
```

---

## 🔧 Step 3: Create Tool System

### 3.1 Base Tool Class

File: `backend/app/agent/tools/base_tool.py`

```python
from typing import Any, Callable, Optional, Dict
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
import logging

logger = logging.getLogger(__name__)

class ToolMetadata(BaseModel):
    """Metadata for tool registration"""
    name: str = Field(description="Tool name (unique identifier)")
    description: str = Field(description="What this tool does")
    category: str = Field(description="Tool category (service, camera, recipe, etc.)")
    requires_approval: bool = Field(
        default=False,
        description="Whether tool requires human approval before execution"
    )

class BaseTool:
    """
    Utility class for creating LangChain tools

    Simplifies tool creation with consistent interface
    """

    @staticmethod
    def create_tool(
        func: Callable,
        metadata: ToolMetadata,
        args_schema: Optional[BaseModel] = None
    ) -> StructuredTool:
        """
        Create a LangChain StructuredTool from a function

        Args:
            func: Python function to wrap
            metadata: Tool metadata
            args_schema: Pydantic model for function arguments

        Returns:
            StructuredTool ready to use with LangChain

        Example:
            def my_func(param: str) -> str:
                return f"Result: {param}"

            class MyFuncArgs(BaseModel):
                param: str = Field(description="Input parameter")

            tool = BaseTool.create_tool(
                func=my_func,
                metadata=ToolMetadata(
                    name="my_tool",
                    description="Does something useful",
                    category="utility"
                ),
                args_schema=MyFuncArgs
            )
        """
        logger.debug(f"Creating tool: {metadata.name}")

        return StructuredTool(
            name=metadata.name,
            description=metadata.description,
            func=func,
            args_schema=args_schema,
            metadata={
                "category": metadata.category,
                "requires_approval": metadata.requires_approval
            }
        )

class ToolRegistry:
    """
    Global registry for all tools

    Allows tools to be shared across multiple agents
    """

    _tools: Dict[str, StructuredTool] = {}

    @classmethod
    def register(cls, tool: StructuredTool):
        """Register a tool globally"""
        if tool.name in cls._tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting")

        cls._tools[tool.name] = tool
        logger.info(f"✅ Registered tool: {tool.name}")

    @classmethod
    def get_tool(cls, name: str) -> Optional[StructuredTool]:
        """Get a tool by name"""
        return cls._tools.get(name)

    @classmethod
    def get_tools_by_category(cls, category: str) -> List[StructuredTool]:
        """Get all tools in a category"""
        return [
            tool for tool in cls._tools.values()
            if tool.metadata.get("category") == category
        ]

    @classmethod
    def get_all_tools(cls) -> List[StructuredTool]:
        """Get all registered tools"""
        return list(cls._tools.values())

    @classmethod
    def list_tools(cls) -> List[Dict[str, Any]]:
        """List all tools with metadata"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "category": tool.metadata.get("category"),
                "requires_approval": tool.metadata.get("requires_approval", False)
            }
            for tool in cls._tools.values()
        ]
```

---

## 📝 Step 4: Implement Service Tools

File: `backend/app/agent/tools/service_tools.py`

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from app.agent.tools.base_tool import BaseTool, ToolMetadata, ToolRegistry
from app.api.websocket.camera_ws import camera_ws_manager
import subprocess
import psutil
import signal
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)

# ============================================================================
# Argument Schemas
# ============================================================================

class CheckServiceStatusArgs(BaseModel):
    """Arguments for check_service_status tool"""
    service_name: str = Field(
        default="camera_management",
        description="Name of the service to check (camera_management, inference, etc.)"
    )

class ServiceActionArgs(BaseModel):
    """Arguments for start/stop service tools"""
    service_name: str = Field(
        description="Name of the service (camera_management, inference, etc.)"
    )

class GetLogsArgs(BaseModel):
    """Arguments for get_service_logs tool"""
    service_name: str = Field(
        description="Name of the service"
    )
    lines: int = Field(
        default=50,
        description="Number of log lines to return (max 500)",
        ge=1,
        le=500
    )

# ============================================================================
# Tool Implementation Functions
# ============================================================================

def check_service_status(service_name: str = "camera_management") -> Dict[str, Any]:
    """
    Check if a service is running and its connection status

    Args:
        service_name: Name of the service to check

    Returns:
        dict with status information
    """
    logger.info(f"Checking status for service: {service_name}")

    if service_name == "camera_management":
        # Check WebSocket connection
        ws_connected = camera_ws_manager.is_connected()

        # Check if process is running
        is_running = False
        pid = None
        cpu_percent = None
        memory_mb = None

        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and 'camera_management_service.py' in ' '.join(cmdline):
                        is_running = True
                        pid = proc.info['pid']

                        # Get resource usage
                        process = psutil.Process(pid)
                        cpu_percent = process.cpu_percent(interval=0.1)
                        memory_mb = process.memory_info().rss / 1024 / 1024

                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.error(f"Error checking process: {e}")

        # Determine overall status
        if is_running and ws_connected:
            status = "healthy"
            message = "Service is running and connected"
        elif is_running and not ws_connected:
            status = "degraded"
            message = "Service is running but WebSocket not connected (may be starting up)"
        else:
            status = "stopped"
            message = "Service is not running"

        return {
            "service_name": service_name,
            "is_running": is_running,
            "pid": pid,
            "websocket_connected": ws_connected,
            "status": status,
            "message": message,
            "cpu_percent": cpu_percent,
            "memory_mb": round(memory_mb, 2) if memory_mb else None
        }

    return {
        "error": f"Unknown service: {service_name}",
        "available_services": ["camera_management"]
    }

def start_service(service_name: str) -> Dict[str, Any]:
    """
    Start a service

    Args:
        service_name: Name of the service to start

    Returns:
        dict with result
    """
    logger.info(f"Starting service: {service_name}")

    if service_name == "camera_management":
        # Check if already running
        status = check_service_status(service_name)
        if status.get("is_running"):
            return {
                "success": False,
                "message": f"Service is already running (PID: {status['pid']})",
                "pid": status['pid']
            }

        # Get script path
        project_root = Path(__file__).parent.parent.parent.parent.parent
        script_path = project_root / "ai_services" / "camera_management_service.py"

        if not script_path.exists():
            return {
                "success": False,
                "message": f"Service script not found: {script_path}"
            }

        try:
            # Start service as background process
            log_dir = project_root / "ai_services" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "camera_management.log"

            process = subprocess.Popen(
                ["python3", str(script_path)],
                cwd=str(script_path.parent),
                stdout=open(log_file, 'a'),
                stderr=subprocess.STDOUT,
                start_new_session=True,  # Detach from parent
                env=os.environ.copy()
            )

            logger.info(f"Started service with PID: {process.pid}")

            return {
                "success": True,
                "message": f"Service started successfully",
                "pid": process.pid,
                "log_file": str(log_file)
            }

        except Exception as e:
            logger.error(f"Failed to start service: {e}")
            return {
                "success": False,
                "message": f"Failed to start service: {str(e)}"
            }

    return {
        "success": False,
        "error": f"Unknown service: {service_name}",
        "available_services": ["camera_management"]
    }

def stop_service(service_name: str) -> Dict[str, Any]:
    """
    Stop a running service

    Args:
        service_name: Name of the service to stop

    Returns:
        dict with result
    """
    logger.info(f"Stopping service: {service_name}")

    if service_name == "camera_management":
        # Find the process
        found = False

        try:
            for proc in psutil.process_iter(['pid', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and 'camera_management_service.py' in ' '.join(cmdline):
                        pid = proc.info['pid']
                        found = True

                        # Graceful shutdown
                        process = psutil.Process(pid)
                        process.terminate()  # SIGTERM

                        # Wait for shutdown (max 5 seconds)
                        try:
                            process.wait(timeout=5)
                            logger.info(f"Service stopped gracefully (PID: {pid})")
                            return {
                                "success": True,
                                "message": f"Service stopped gracefully",
                                "pid": pid
                            }
                        except psutil.TimeoutExpired:
                            # Force kill if not responding
                            process.kill()  # SIGKILL
                            process.wait()
                            logger.warning(f"Service force killed (PID: {pid})")
                            return {
                                "success": True,
                                "message": f"Service force killed (was not responding to shutdown)",
                                "pid": pid
                            }

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.error(f"Error stopping service: {e}")
            return {
                "success": False,
                "message": f"Error stopping service: {str(e)}"
            }

        if not found:
            return {
                "success": False,
                "message": "Service is not running"
            }

    return {
        "success": False,
        "error": f"Unknown service: {service_name}",
        "available_services": ["camera_management"]
    }

def get_service_logs(service_name: str, lines: int = 50) -> Dict[str, Any]:
    """
    Get recent logs from a service

    Args:
        service_name: Name of the service
        lines: Number of lines to return

    Returns:
        dict with logs
    """
    logger.info(f"Getting logs for service: {service_name} (last {lines} lines)")

    if service_name == "camera_management":
        project_root = Path(__file__).parent.parent.parent.parent.parent
        log_file = project_root / "ai_services" / "logs" / "camera_management.log"

        if not log_file.exists():
            return {
                "logs": [],
                "message": "Log file not found (service may not have been started yet)",
                "log_file": str(log_file)
            }

        try:
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
                last_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

            return {
                "logs": [line.strip() for line in last_lines],
                "total_lines": len(all_lines),
                "returned_lines": len(last_lines),
                "log_file": str(log_file)
            }

        except Exception as e:
            logger.error(f"Error reading logs: {e}")
            return {
                "error": f"Failed to read logs: {str(e)}"
            }

    return {
        "error": f"Unknown service: {service_name}",
        "available_services": ["camera_management"]
    }

# ============================================================================
# Create and Register Tools
# ============================================================================

# Check service status tool
check_service_status_tool = BaseTool.create_tool(
    func=check_service_status,
    metadata=ToolMetadata(
        name="check_service_status",
        description=(
            "Check if a service is running and its connection status. "
            "Returns process info, WebSocket status, and resource usage. "
            "Use this FIRST when user asks about service status."
        ),
        category="service",
        requires_approval=False
    ),
    args_schema=CheckServiceStatusArgs
)

# Start service tool
start_service_tool = BaseTool.create_tool(
    func=start_service,
    metadata=ToolMetadata(
        name="start_service",
        description=(
            "Start a service. "
            "IMPORTANT: ALWAYS check service status first before starting. "
            "Ask user for confirmation before executing this."
        ),
        category="service",
        requires_approval=True
    ),
    args_schema=ServiceActionArgs
)

# Stop service tool
stop_service_tool = BaseTool.create_tool(
    func=stop_service,
    metadata=ToolMetadata(
        name="stop_service",
        description=(
            "Stop a running service gracefully. "
            "IMPORTANT: Ask user for confirmation before executing this. "
            "This will disconnect all cameras and stop inference."
        ),
        category="service",
        requires_approval=True
    ),
    args_schema=ServiceActionArgs
)

# Get logs tool
get_service_logs_tool = BaseTool.create_tool(
    func=get_service_logs,
    metadata=ToolMetadata(
        name="get_service_logs",
        description=(
            "Get recent log entries from a service. "
            "Useful for debugging issues or checking what the service is doing. "
            "Returns the last N lines of the log file."
        ),
        category="service",
        requires_approval=False
    ),
    args_schema=GetLogsArgs
)

# Register all tools
ToolRegistry.register(check_service_status_tool)
ToolRegistry.register(start_service_tool)
ToolRegistry.register(stop_service_tool)
ToolRegistry.register(get_service_logs_tool)

logger.info("✅ Service tools registered")
```

---

## 🎯 Next Steps

Với foundation này, bạn có thể:

1. **Implement Service Agent** (see AGENT_ARCHITECTURE.md)
2. **Create API endpoints**
3. **Build Frontend UI**
4. **Add more agents** (Recipe, Analytics, etc.)

Mỗi agent mới chỉ cần:
- Extend `BaseAgent`
- Register với `@AgentRegistry.register()`
- Implement 3 methods
- Tạo tools nếu cần

---

**Ready to start coding?**

Tôi suggest bắt đầu với:
1. Copy các file base classes vào project
2. Test với simple example
3. Implement ServiceAgent
4. Build API
5. Create UI

Bạn muốn tôi implement phần nào trước?
