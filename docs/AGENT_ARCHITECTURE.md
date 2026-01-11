# 🤖 AI Agent Architecture Design Document

## 📋 Tổng quan

Document này mô tả kiến trúc AI Agent system cho OCR Datecode project, sử dụng **LangGraph** framework với khả năng mở rộng cho nhiều specialized agents.

---

## 🎯 Mục tiêu

1. **Extensibility**: Dễ dàng thêm agents mới
2. **Modularity**: Mỗi agent độc lập, có thể tái sử dụng tools
3. **Scalability**: Hỗ trợ multi-agent collaboration
4. **Maintainability**: Code rõ ràng, dễ debug và test

---

## 🏗️ Kiến trúc tổng thể

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer                            │
│  - AgentChat.tsx (UI component)                             │
│  - agentService.ts (API client)                             │
└─────────────────────────────────────────────────────────────┘
                          ↓ HTTP/WebSocket
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                       │
│  - /api/agent/chat                                          │
│  - /api/agent/{agent_id}/chat                               │
│  - /ws/agent/stream                                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                Agent Orchestrator Layer                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         AgentRegistry                                 │  │
│  │  - Register agents                                    │  │
│  │  - Route requests to appropriate agent                │  │
│  │  - Manage agent lifecycle                             │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                Individual Agents (LangGraph)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Service    │  │   Recipe     │  │  Analytics   │      │
│  │  Management  │  │ Optimization │  │    Agent     │      │
│  │    Agent     │  │    Agent     │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         ↓                 ↓                   ↓              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Shared Tools Registry                    │  │
│  │  - Service tools                                     │  │
│  │  - Camera tools                                      │  │
│  │  - Recipe tools                                      │  │
│  │  - Analytics tools                                   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                   External Services                          │
│  - OpenAI API (GPT-4o-mini)                                 │
│  - MongoDB (Memory & State)                                 │
│  - Internal APIs (Cameras, Recipes, etc.)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Cấu trúc thư mục

```
backend/app/
├── agent/
│   ├── __init__.py
│   ├── base/
│   │   ├── __init__.py
│   │   ├── base_agent.py           # Abstract base agent class
│   │   ├── agent_state.py          # State definitions
│   │   └── agent_config.py         # Configuration schemas
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── registry.py             # Agent registry
│   │   ├── orchestrator.py         # Multi-agent orchestration
│   │   └── router.py               # Intent routing
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── service_agent.py        # Service management agent
│   │   ├── recipe_agent.py         # Recipe optimization agent
│   │   ├── analytics_agent.py      # Analytics agent
│   │   └── general_agent.py        # General purpose agent
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base_tool.py            # Base tool class
│   │   ├── service_tools.py        # Service management tools
│   │   ├── camera_tools.py         # Camera tools
│   │   ├── recipe_tools.py         # Recipe tools
│   │   ├── analytics_tools.py      # Analytics tools
│   │   └── tool_registry.py        # Tool registration
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── conversation_memory.py  # Conversation history
│   │   └── memory_store.py         # MongoDB-backed storage
│   │
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── system_prompts.py       # System prompts
│   │   └── few_shot_examples.py    # Example conversations
│   │
│   └── utils/
│       ├── __init__.py
│       ├── streaming.py            # Streaming utilities
│       └── validators.py           # Input validation
│
├── api/endpoints/
│   └── agent.py                    # Agent API endpoints
│
└── schemas/
    └── agent.py                    # Pydantic schemas
```

---

## 🧩 Core Components

### 1. Base Agent Class (Template)

```python
# backend/app/agent/base/base_agent.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

class AgentState(BaseModel):
    """Base state for all agents"""
    messages: List[BaseMessage]
    user_id: str
    session_id: str
    context: Dict[str, Any] = {}
    next_step: Optional[str] = None

class BaseAgent(ABC):
    """
    Abstract base class for all agents

    Every agent must implement:
    - build_graph(): Construct LangGraph state machine
    - get_tools(): Return list of available tools
    - get_system_prompt(): Return system prompt
    """

    def __init__(
        self,
        agent_id: str,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7
    ):
        self.agent_id = agent_id
        self.model_name = model_name
        self.temperature = temperature
        self.graph = None
        self.tools = self.get_tools()

    @abstractmethod
    def build_graph(self) -> StateGraph:
        """Build the agent's state graph"""
        pass

    @abstractmethod
    def get_tools(self) -> List[Any]:
        """Return list of tools this agent can use"""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent"""
        pass

    def invoke(self, state: AgentState) -> AgentState:
        """Execute the agent"""
        if not self.graph:
            self.graph = self.build_graph()
        return self.graph.invoke(state)

    async def ainvoke(self, state: AgentState) -> AgentState:
        """Async execute the agent"""
        if not self.graph:
            self.graph = self.build_graph()
        return await self.graph.ainvoke(state)

    async def astream(self, state: AgentState):
        """Stream agent execution"""
        if not self.graph:
            self.graph = self.build_graph()
        async for chunk in self.graph.astream(state):
            yield chunk
```

### 2. Agent Registry (Multi-Agent Support)

```python
# backend/app/agent/core/registry.py

from typing import Dict, Type, Optional
from app.agent.base.base_agent import BaseAgent

class AgentRegistry:
    """
    Central registry for all agents

    Allows:
    - Registering new agents
    - Retrieving agents by ID
    - Listing available agents
    """

    _agents: Dict[str, Type[BaseAgent]] = {}
    _instances: Dict[str, BaseAgent] = {}

    @classmethod
    def register(cls, agent_id: str):
        """Decorator to register an agent"""
        def decorator(agent_class: Type[BaseAgent]):
            cls._agents[agent_id] = agent_class
            return agent_class
        return decorator

    @classmethod
    def get_agent(cls, agent_id: str, **kwargs) -> BaseAgent:
        """Get or create agent instance"""
        # Return cached instance if exists
        cache_key = f"{agent_id}:{kwargs.get('model_name', 'default')}"
        if cache_key in cls._instances:
            return cls._instances[cache_key]

        # Create new instance
        if agent_id not in cls._agents:
            raise ValueError(f"Agent '{agent_id}' not registered")

        agent_class = cls._agents[agent_id]
        instance = agent_class(agent_id=agent_id, **kwargs)
        cls._instances[cache_key] = instance

        return instance

    @classmethod
    def list_agents(cls) -> List[Dict[str, Any]]:
        """List all registered agents"""
        return [
            {
                "agent_id": agent_id,
                "class": agent_class.__name__,
                "description": agent_class.__doc__
            }
            for agent_id, agent_class in cls._agents.items()
        ]
```

### 3. Tool Base Class

```python
# backend/app/agent/tools/base_tool.py

from typing import Any, Dict, Callable, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

class ToolMetadata(BaseModel):
    """Metadata for tool registration"""
    name: str
    description: str
    category: str  # "service", "camera", "recipe", "analytics"
    requires_approval: bool = False  # Human-in-the-loop

class BaseTool:
    """
    Base class for creating tools

    Makes it easy to create new tools with consistent interface
    """

    @staticmethod
    def create_tool(
        func: Callable,
        metadata: ToolMetadata,
        args_schema: Optional[BaseModel] = None
    ) -> StructuredTool:
        """
        Create a LangChain tool from a function

        Example:
            def my_func(param1: str) -> str:
                return f"Result: {param1}"

            tool = BaseTool.create_tool(
                func=my_func,
                metadata=ToolMetadata(
                    name="my_tool",
                    description="Does something",
                    category="service"
                )
            )
        """
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
    """Registry for all tools"""

    _tools: Dict[str, StructuredTool] = {}

    @classmethod
    def register(cls, tool: StructuredTool):
        """Register a tool"""
        cls._tools[tool.name] = tool

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
```

---

## 🎨 Agent Implementation Example

### Service Management Agent

```python
# backend/app/agent/agents/service_agent.py

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.agent.base.base_agent import BaseAgent, AgentState
from app.agent.core.registry import AgentRegistry
from app.agent.tools.service_tools import (
    check_service_status_tool,
    start_service_tool,
    stop_service_tool,
    get_service_logs_tool
)

@AgentRegistry.register("service_management")
class ServiceManagementAgent(BaseAgent):
    """
    Agent chuyên quản lý services (camera_management, inference, etc.)

    Capabilities:
    - Check service status
    - Start/Stop services
    - Diagnose service issues
    - View and analyze logs
    """

    def get_system_prompt(self) -> str:
        return """Bạn là Service Management Assistant cho hệ thống OCR Datecode.

Nhiệm vụ của bạn:
- Giúp người dùng quản lý các services (Camera Management Service, Inference Service, etc.)
- Kiểm tra trạng thái services
- Start/Stop services khi được yêu cầu
- Chẩn đoán và giải quyết vấn đề
- Phân tích logs để tìm lỗi

Quy tắc:
1. LUÔN check status trước khi thực hiện hành động
2. Hỏi xác nhận trước khi start/stop service
3. Nếu có lỗi, đọc logs và đưa ra giải pháp cụ thể
4. Trả lời bằng tiếng Việt, dễ hiểu, có emoji phù hợp
5. Nếu không chắc chắn, hỏi người dùng thay vì đoán

Bạn có quyền truy cập các tools:
- check_service_status: Kiểm tra trạng thái service
- start_service: Khởi động service
- stop_service: Dừng service
- get_service_logs: Xem logs của service
"""

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

        # Initialize LLM with tools
        llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            streaming=True
        )
        llm_with_tools = llm.bind_tools(self.tools)

        # Define graph
        workflow = StateGraph(AgentState)

        # Nodes
        def call_model(state: AgentState) -> AgentState:
            """Call LLM with tools"""
            messages = state.messages

            # Add system prompt
            if not any(isinstance(m, SystemMessage) for m in messages):
                messages = [SystemMessage(content=self.get_system_prompt())] + messages

            response = llm_with_tools.invoke(messages)
            state.messages.append(response)

            # Check if tool calls exist
            if response.tool_calls:
                state.next_step = "execute_tools"
            else:
                state.next_step = "end"

            return state

        def execute_tools(state: AgentState) -> AgentState:
            """Execute tool calls"""
            last_message = state.messages[-1]

            for tool_call in last_message.tool_calls:
                # Find matching tool
                tool = next((t for t in self.tools if t.name == tool_call["name"]), None)
                if tool:
                    result = tool.invoke(tool_call["args"])

                    # Add tool result to messages
                    from langchain_core.messages import ToolMessage
                    state.messages.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call["id"]
                        )
                    )

            state.next_step = "call_model"
            return state

        def route(state: AgentState) -> str:
            """Route to next step"""
            return state.next_step or "end"

        # Add nodes
        workflow.add_node("call_model", call_model)
        workflow.add_node("execute_tools", execute_tools)

        # Add edges
        workflow.set_entry_point("call_model")
        workflow.add_conditional_edges(
            "call_model",
            route,
            {
                "execute_tools": "execute_tools",
                "end": END
            }
        )
        workflow.add_edge("execute_tools", "call_model")

        return workflow.compile()
```

---

## 🔧 Tools Implementation Example

```python
# backend/app/agent/tools/service_tools.py

from pydantic import BaseModel, Field
from app.agent.tools.base_tool import BaseTool, ToolMetadata
from app.api.websocket.camera_ws import camera_ws_manager
import subprocess
import psutil
from pathlib import Path

# Tool argument schemas
class CheckServiceStatusArgs(BaseModel):
    """Arguments for check_service_status"""
    service_name: str = Field(
        default="camera_management",
        description="Service name to check (camera_management, inference, etc.)"
    )

class StartServiceArgs(BaseModel):
    """Arguments for start_service"""
    service_name: str = Field(description="Service name to start")

class GetLogsArgs(BaseModel):
    """Arguments for get_service_logs"""
    service_name: str = Field(description="Service name")
    lines: int = Field(default=50, description="Number of lines to fetch")

# Tool implementations
def check_service_status(service_name: str = "camera_management") -> dict:
    """
    Check if a service is running and connected

    Returns:
        dict with status information
    """
    if service_name == "camera_management":
        # Check WebSocket connection
        ws_connected = camera_ws_manager.is_connected()

        # Check process
        is_running = False
        pid = None

        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'camera_management_service.py' in cmdline:
                    is_running = True
                    pid = proc.info['pid']
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return {
            "service_name": service_name,
            "is_running": is_running,
            "pid": pid,
            "websocket_connected": ws_connected,
            "status": "healthy" if (is_running and ws_connected) else "unhealthy"
        }

    return {"error": f"Unknown service: {service_name}"}

def start_service(service_name: str) -> dict:
    """Start a service"""
    if service_name == "camera_management":
        script_path = Path(__file__).parent.parent.parent.parent.parent / "ai_services" / "camera_management_service.py"

        try:
            process = subprocess.Popen(
                ["python3", str(script_path)],
                cwd=str(script_path.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )

            return {
                "success": True,
                "message": f"Service {service_name} started",
                "pid": process.pid
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to start service: {str(e)}"
            }

    return {"error": f"Unknown service: {service_name}"}

def stop_service(service_name: str) -> dict:
    """Stop a service"""
    if service_name == "camera_management":
        for proc in psutil.process_iter(['pid', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'camera_management_service.py' in cmdline:
                    proc.terminate()
                    return {
                        "success": True,
                        "message": f"Service {service_name} stopped"
                    }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return {
            "success": False,
            "message": "Service is not running"
        }

    return {"error": f"Unknown service: {service_name}"}

def get_service_logs(service_name: str, lines: int = 50) -> dict:
    """Get service logs"""
    if service_name == "camera_management":
        log_path = Path(__file__).parent.parent.parent.parent.parent / "ai_services" / "logs" / "camera_management.log"

        if not log_path.exists():
            return {"logs": [], "message": "Log file not found"}

        with open(log_path, 'r') as f:
            all_lines = f.readlines()
            last_lines = all_lines[-lines:]

        return {
            "logs": [line.strip() for line in last_lines],
            "total_lines": len(all_lines)
        }

    return {"error": f"Unknown service: {service_name}"}

# Create tools
check_service_status_tool = BaseTool.create_tool(
    func=check_service_status,
    metadata=ToolMetadata(
        name="check_service_status",
        description="Check if a service is running and connected. Use this to diagnose service issues.",
        category="service",
        requires_approval=False
    ),
    args_schema=CheckServiceStatusArgs
)

start_service_tool = BaseTool.create_tool(
    func=start_service,
    metadata=ToolMetadata(
        name="start_service",
        description="Start a service. ALWAYS ask user for confirmation before using this.",
        category="service",
        requires_approval=True
    ),
    args_schema=StartServiceArgs
)

stop_service_tool = BaseTool.create_tool(
    func=stop_service,
    metadata=ToolMetadata(
        name="stop_service",
        description="Stop a running service. ALWAYS ask user for confirmation before using this.",
        category="service",
        requires_approval=True
    ),
    args_schema=StartServiceArgs
)

get_service_logs_tool = BaseTool.create_tool(
    func=get_service_logs,
    metadata=ToolMetadata(
        name="get_service_logs",
        description="Get recent logs from a service. Useful for debugging issues.",
        category="service",
        requires_approval=False
    ),
    args_schema=GetLogsArgs
)
```

---

## 🌐 API Endpoints

```python
# backend/app/api/endpoints/agent.py

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import json

from app.agent.core.registry import AgentRegistry
from app.agent.base.base_agent import AgentState
from app.api.dependencies.auth import get_current_user
from app.models.user import UserInDB
from langchain_core.messages import HumanMessage

router = APIRouter(prefix="/agent", tags=["AI Agent"])

class ChatRequest(BaseModel):
    """Chat request"""
    message: str
    agent_id: str = "service_management"  # Default agent
    session_id: Optional[str] = None
    stream: bool = True

class ChatResponse(BaseModel):
    """Chat response"""
    response: str
    agent_id: str
    session_id: str
    tool_calls: Optional[List[dict]] = None

@router.post("/chat", summary="Chat with AI agent")
async def chat(
    request: ChatRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Chat with AI agent

    Supports:
    - Multiple specialized agents
    - Streaming responses
    - Tool execution
    """
    try:
        # Get agent
        agent = AgentRegistry.get_agent(request.agent_id)

        # Create state
        state = AgentState(
            messages=[HumanMessage(content=request.message)],
            user_id=current_user.id,
            session_id=request.session_id or f"session_{current_user.id}",
            context={}
        )

        if request.stream:
            # Streaming response
            async def generate():
                async for chunk in agent.astream(state):
                    yield f"data: {json.dumps(chunk)}\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming response
            result = await agent.ainvoke(state)

            return ChatResponse(
                response=result.messages[-1].content,
                agent_id=request.agent_id,
                session_id=state.session_id,
                tool_calls=None  # TODO: extract tool calls
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents", summary="List available agents")
async def list_agents(current_user: UserInDB = Depends(get_current_user)):
    """List all available agents"""
    return AgentRegistry.list_agents()
```

---

## 📝 Usage Examples

### Example 1: Using Service Agent

```python
# User request
"Camera service có đang chạy không?"

# Agent execution flow:
1. Call LLM with system prompt
2. LLM decides to use check_service_status tool
3. Execute tool -> returns status
4. LLM formats response for user
5. Return: "Service đang chạy ✅ (PID: 12345, WebSocket: Connected)"
```

### Example 2: Adding New Agent

```python
# backend/app/agent/agents/recipe_agent.py

from app.agent.core.registry import AgentRegistry
from app.agent.base.base_agent import BaseAgent

@AgentRegistry.register("recipe_optimization")
class RecipeOptimizationAgent(BaseAgent):
    """Agent for recipe optimization"""

    def get_system_prompt(self) -> str:
        return """Bạn là Recipe Optimization Assistant..."""

    def get_tools(self) -> List[Any]:
        return [
            analyze_recipe_performance_tool,
            suggest_threshold_tool,
            test_recipe_tool
        ]

    def build_graph(self) -> StateGraph:
        # Similar to ServiceAgent
        pass
```

### Example 3: Multi-Agent Collaboration

```python
# Future: Agent Orchestrator
# User: "Tối ưu recipe cho camera 1"

# Orchestrator routes to:
1. ServiceAgent: Check if camera is connected
2. RecipeAgent: Analyze current recipe
3. AnalyticsAgent: Get historical data
4. RecipeAgent: Suggest optimizations
```

---

## 🚀 Deployment Considerations

### Environment Variables

```bash
# .env
OPENAI_API_KEY=sk-...
AGENT_MODEL=gpt-4o-mini
AGENT_TEMPERATURE=0.7
AGENT_MAX_TOKENS=2000
AGENT_STREAMING=true
```

### Configuration

```python
# backend/app/agent/base/agent_config.py

from pydantic import BaseModel

class AgentConfig(BaseModel):
    """Global agent configuration"""
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2000
    streaming: bool = True
    enable_memory: bool = True
    memory_backend: str = "mongodb"  # or "redis", "postgres"
```

---

## 📊 Testing Strategy

```python
# tests/agent/test_service_agent.py

import pytest
from app.agent.agents.service_agent import ServiceManagementAgent
from app.agent.base.base_agent import AgentState
from langchain_core.messages import HumanMessage

@pytest.mark.asyncio
async def test_service_agent_check_status():
    """Test service status check"""
    agent = ServiceManagementAgent(agent_id="service_management")

    state = AgentState(
        messages=[HumanMessage(content="Check camera service status")],
        user_id="test_user",
        session_id="test_session"
    )

    result = await agent.ainvoke(state)

    assert len(result.messages) > 1
    assert "status" in result.messages[-1].content.lower()
```

---

## 🎯 Roadmap

### Phase 1: MVP (Week 1-2)
- ✅ Base infrastructure
- ✅ Service Management Agent
- ✅ Basic chat API
- ✅ Simple frontend UI

### Phase 2: Enhanced (Week 3-4)
- ⬜ Recipe Optimization Agent
- ⬜ Analytics Agent
- ⬜ Memory & conversation history
- ⬜ WebSocket streaming

### Phase 3: Advanced (Week 5-6)
- ⬜ Multi-agent orchestration
- ⬜ Human-in-the-loop approvals
- ⬜ Advanced analytics
- ⬜ Voice interface

---

## 📚 References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Tools](https://python.langchain.com/docs/modules/tools/)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [FastAPI Streaming](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)

---

## 🤝 Contributing

Khi thêm agent mới:

1. Tạo file trong `app/agent/agents/`
2. Extend `BaseAgent`
3. Register với `@AgentRegistry.register()`
4. Implement 3 methods: `get_system_prompt()`, `get_tools()`, `build_graph()`
5. Thêm tools vào `app/agent/tools/`
6. Test thoroughly
7. Update docs

---

**Last Updated**: 2026-01-10
**Version**: 1.0.0
**Author**: AI Architecture Team
