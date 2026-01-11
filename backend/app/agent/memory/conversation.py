"""
Conversation Memory Models
Stores AI agent conversation history in MongoDB
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
from datetime import datetime


class ConversationMessage(BaseModel):
    """Single message in a conversation"""
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None  # For tool response messages
    name: Optional[str] = None  # Tool name for tool messages


class ConversationHistory(BaseModel):
    """Conversation history stored in MongoDB"""
    model_config = ConfigDict(populate_by_name=True)

    id: Optional[str] = Field(None, alias="_id")
    session_id: str = Field(..., index=True)  # Unique session identifier
    user_id: str  # User who owns this conversation
    agent_id: str  # Which agent is being used
    messages: List[ConversationMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Extra context


class ConversationCreate(BaseModel):
    """Create new conversation"""
    session_id: str
    user_id: str
    agent_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationUpdate(BaseModel):
    """Update conversation with new messages"""
    messages: List[ConversationMessage]
    updated_at: datetime = Field(default_factory=datetime.now)
