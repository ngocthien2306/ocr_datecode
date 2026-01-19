"""
Agent Schemas
Pydantic models for agent API
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class ChatRequest(BaseModel):
    """Request to chat with an agent"""
    message: str = Field(description="User's message")
    agent_id: str = Field(default="service_management", description="Agent to use")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation continuity")
    stream: bool = Field(default=False, description="Enable streaming response")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Camera service có đang chạy không?",
                "agent_id": "service_management",
                "stream": False
            }
        }


class ChatResponse(BaseModel):
    """Response from agent"""
    response: str = Field(description="Agent's response")
    agent_id: str = Field(description="Agent that generated response")
    session_id: str = Field(description="Session ID")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(default=None, description="Tools used by agent")
    timestamp: str = Field(description="Response timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "response": "Service đang chạy và đã kết nối WebSocket ✅",
                "agent_id": "service_management",
                "session_id": "session_123",
                "tool_calls": [
                    {
                        "tool": "check_service_status",
                        "args": {"service_name": "camera_management"}
                    }
                ],
                "timestamp": "2026-01-11T10:30:00"
            }
        }


class AgentInfo(BaseModel):
    """Information about an agent"""
    agent_id: str = Field(description="Agent identifier")
    class_name: str = Field(description="Python class name")
    description: str = Field(description="Agent description")

    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "service_management",
                "class_name": "ServiceManagementAgent",
                "description": "Agent chuyên quản lý services"
            }
        }


class AgentHealthResponse(BaseModel):
    """Agent system health status"""
    status: str = Field(description="Health status")
    agents_registered: int = Field(description="Number of registered agents")
    openai_configured: bool = Field(description="Whether OpenAI API is configured")
    message: str = Field(description="Status message")
