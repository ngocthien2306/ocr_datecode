"""
AI Agent API Endpoints
Provides chat interface for AI agents
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import logging
from datetime import datetime

from app.agent.core.registry import AgentRegistry
from app.agent.base.base_agent import AgentState
from app.api.dependencies.auth import get_current_user
from app.models.user import UserInDB
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["AI Agent"])


# ============================================================================
# Pydantic Models
# ============================================================================

class ChatRequest(BaseModel):
    """Chat request from user"""
    message: str
    agent_id: str = "service_management"  # Default to service agent
    session_id: Optional[str] = None
    stream: bool = False  # Enable streaming for real-time updates


class ChatResponse(BaseModel):
    """Chat response to user"""
    response: str
    agent_id: str
    session_id: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    timestamp: str


class AgentInfo(BaseModel):
    """Agent information"""
    agent_id: str
    class_name: str
    description: str


# ============================================================================
# Helper Functions
# ============================================================================

def extract_assistant_message(state: AgentState) -> str:
    """
    Extract the final assistant message from state

    Args:
        state: Agent state after execution

    Returns:
        Assistant's response text
    """
    # Find last AI message
    for msg in reversed(state.messages):
        if isinstance(msg, AIMessage):
            return msg.content

    return "No response generated"


def extract_tool_calls(state: AgentState) -> List[Dict[str, Any]]:
    """
    Extract tool calls from state for debugging/transparency

    Args:
        state: Agent state after execution

    Returns:
        List of tool call information
    """
    tool_calls = []

    for msg in state.messages:
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_calls.append({
                    "tool": tool_call.get("name"),
                    "args": tool_call.get("args"),
                    "id": tool_call.get("id")
                })

    return tool_calls if tool_calls else None


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/chat", response_model=ChatResponse, summary="Chat with AI agent")
async def chat(
    request: ChatRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Chat with AI agent

    **Features:**
    - Multiple specialized agents
    - Tool execution (service management, analytics, etc.)
    - Session-based conversation
    - Streaming support (optional)

    **Available Agents:**
    - service_management: Manage AI services (camera, inference)
    - (more agents coming soon)

    **Example Request:**
    ```json
    {
        "message": "Camera service có đang chạy không?",
        "agent_id": "service_management",
        "stream": false
    }
    ```
    """
    try:
        # Get agent
        logger.info(f"User {current_user.username} chatting with agent: {request.agent_id}")

        try:
            agent = AgentRegistry.get_agent(request.agent_id)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )

        # Create session ID if not provided
        session_id = request.session_id or f"session_{current_user.id}_{datetime.now().timestamp()}"

        # Create initial state
        state = AgentState(
            messages=[HumanMessage(content=request.message)],
            user_id=current_user.id,
            session_id=session_id,
            context={
                "username": current_user.username,
                "role": current_user.role
            }
        )

        # Execute agent (non-streaming for now)
        logger.info(f"Executing agent with message: {request.message[:100]}...")
        result_state = await agent.ainvoke(state)

        # Extract response
        response_text = extract_assistant_message(result_state)
        tool_calls = extract_tool_calls(result_state)

        logger.info(f"Agent response generated ({len(response_text)} chars)")

        return ChatResponse(
            response=response_text,
            agent_id=request.agent_id,
            session_id=session_id,
            tool_calls=tool_calls,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}"
        )


@router.get("/agents", response_model=List[AgentInfo], summary="List available agents")
async def list_agents(current_user: UserInDB = Depends(get_current_user)):
    """
    List all available AI agents

    Returns information about each registered agent including:
    - Agent ID
    - Class name
    - Description/capabilities
    """
    try:
        agents = AgentRegistry.list_agents()
        return agents
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/health", summary="Agent system health check")
async def agent_health():
    """
    Check if agent system is healthy

    Verifies:
    - Agent registry is initialized
    - At least one agent is registered
    - OpenAI API key is configured
    """
    import os

    try:
        agents = AgentRegistry.list_agents()
        openai_key_configured = bool(os.getenv("OPENAI_API_KEY"))

        return {
            "status": "healthy" if (agents and openai_key_configured) else "unhealthy",
            "agents_registered": len(agents),
            "openai_configured": openai_key_configured,
            "message": "Agent system is operational" if openai_key_configured else "OpenAI API key not configured"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# ============================================================================
# Optional: Streaming endpoint (for future enhancement)
# ============================================================================

@router.post("/chat/stream", summary="Chat with streaming response")
async def chat_stream(
    request: ChatRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Chat with AI agent (streaming response)

    Returns Server-Sent Events (SSE) for real-time updates

    **Note:** This is for future enhancement with WebSocket/SSE
    """
    try:
        agent = AgentRegistry.get_agent(request.agent_id)
        session_id = request.session_id or f"session_{current_user.id}_{datetime.now().timestamp()}"

        state = AgentState(
            messages=[HumanMessage(content=request.message)],
            user_id=current_user.id,
            session_id=session_id,
            context={"username": current_user.username}
        )

        async def generate():
            """Generator for SSE"""
            async for chunk in agent.astream(state):
                # Extract useful information from chunk
                chunk_data = {
                    "type": "chunk",
                    "data": str(chunk),
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"

            # Send completion event
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )

    except Exception as e:
        logger.error(f"Error in streaming chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
