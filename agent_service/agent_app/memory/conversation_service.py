"""
Conversation Memory Service
Manages AI agent conversation history in MongoDB
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage

from agent_app.db.mongodb import get_database
from agent_app.memory.conversation import (
    ConversationHistory,
    ConversationMessage,
    ConversationCreate,
    ConversationUpdate
)


class ConversationService:
    """Service for managing conversation history"""

    COLLECTION_NAME = "agent_conversations"

    @staticmethod
    async def create_conversation(
        session_id: str,
        user_id: str,
        agent_id: str,
        metadata: Dict[str, Any] = None
    ) -> ConversationHistory:
        """
        Create a new conversation session

        Args:
            session_id: Unique session identifier
            user_id: User ID who owns this conversation
            agent_id: Agent being used
            metadata: Additional context

        Returns:
            Created conversation
        """
        db = get_database()
        collection = db[ConversationService.COLLECTION_NAME]

        conversation = ConversationHistory(
            session_id=session_id,
            user_id=user_id,
            agent_id=agent_id,
            messages=[],
            metadata=metadata or {}
        )

        result = await collection.insert_one(conversation.model_dump(by_alias=True, exclude={"id"}))
        conversation.id = str(result.inserted_id)

        return conversation

    @staticmethod
    async def get_conversation(
        session_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[ConversationHistory]:
        """
        Get conversation by session ID.

        Args:
            session_id: Session identifier
            user_id: Nếu truyền vào, chỉ trả conversation thuộc user này.
                session_id do client gửi tự do và format đoán được
                (`session_{user_id}_{timestamp}`), nên MỌI đường đi từ request
                phải truyền user_id — bỏ qua là user này đọc được chat của
                user khác.

        Returns:
            Conversation or None if not found
        """
        db = get_database()
        collection = db[ConversationService.COLLECTION_NAME]

        query: Dict[str, Any] = {"session_id": session_id}
        if user_id is not None:
            query["user_id"] = user_id

        doc = await collection.find_one(query)
        if doc:
            doc["_id"] = str(doc["_id"])
            return ConversationHistory(**doc)

        return None

    @staticmethod
    async def add_messages(
        session_id: str,
        messages: List[ConversationMessage]
    ) -> bool:
        """
        Add messages to existing conversation

        Args:
            session_id: Session identifier
            messages: Messages to add

        Returns:
            True if successful
        """
        db = get_database()
        collection = db[ConversationService.COLLECTION_NAME]

        result = await collection.update_one(
            {"session_id": session_id},
            {
                "$push": {
                    "messages": {
                        "$each": [msg.model_dump() for msg in messages]
                    }
                },
                "$set": {"updated_at": datetime.now()}
            }
        )

        return result.modified_count > 0

    @staticmethod
    async def get_or_create_conversation(
        session_id: str,
        user_id: str,
        agent_id: str,
        metadata: Dict[str, Any] = None
    ) -> ConversationHistory:
        """
        Get existing conversation or create new one

        Args:
            session_id: Session identifier
            user_id: User ID
            agent_id: Agent ID
            metadata: Additional context

        Returns:
            Conversation (existing or new)

        Raises:
            PermissionError: session_id đã tồn tại nhưng thuộc user khác.
        """
        conversation = await ConversationService.get_conversation(session_id, user_id=user_id)
        if conversation:
            return conversation

        # Chưa thấy với user này — nhưng session_id là unique toàn collection,
        # nên phải phân biệt "chưa tồn tại" với "tồn tại của người khác".
        if await ConversationService.get_conversation(session_id) is not None:
            raise PermissionError(f"session_id '{session_id}' belongs to another user")

        return await ConversationService.create_conversation(
            session_id=session_id,
            user_id=user_id,
            agent_id=agent_id,
            metadata=metadata
        )

    @staticmethod
    def langchain_messages_to_conversation_messages(
        messages: List[BaseMessage]
    ) -> List[ConversationMessage]:
        """
        Convert LangChain messages to ConversationMessage format

        Args:
            messages: LangChain messages

        Returns:
            List of ConversationMessage
        """
        conversation_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                conversation_messages.append(ConversationMessage(
                    role="user",
                    content=msg.content,
                    timestamp=datetime.now()
                ))
            elif isinstance(msg, AIMessage):
                tool_calls = None
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_calls = [
                        {
                            "name": tc.get("name"),
                            "args": tc.get("args"),
                            "id": tc.get("id")
                        }
                        for tc in msg.tool_calls
                    ]

                conversation_messages.append(ConversationMessage(
                    role="assistant",
                    content=msg.content,
                    tool_calls=tool_calls,
                    timestamp=datetime.now()
                ))
            elif isinstance(msg, ToolMessage):
                conversation_messages.append(ConversationMessage(
                    role="tool",
                    content=msg.content,
                    tool_call_id=msg.tool_call_id if hasattr(msg, 'tool_call_id') else None,
                    name=msg.name if hasattr(msg, 'name') else None,
                    timestamp=datetime.now()
                ))
            elif isinstance(msg, SystemMessage):
                conversation_messages.append(ConversationMessage(
                    role="system",
                    content=msg.content,
                    timestamp=datetime.now()
                ))

        return conversation_messages

    @staticmethod
    def conversation_messages_to_langchain_messages(
        messages: List[ConversationMessage]
    ) -> List[BaseMessage]:
        """
        Convert ConversationMessage to LangChain messages

        Args:
            messages: Conversation messages

        Returns:
            List of LangChain BaseMessage
        """
        langchain_messages = []

        for msg in messages:
            if msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                ai_msg = AIMessage(content=msg.content)
                if msg.tool_calls:
                    ai_msg.tool_calls = msg.tool_calls
                langchain_messages.append(ai_msg)
            elif msg.role == "tool":
                langchain_messages.append(ToolMessage(
                    content=msg.content,
                    tool_call_id=msg.tool_call_id or "",
                    name=msg.name or ""
                ))
            elif msg.role == "system":
                langchain_messages.append(SystemMessage(content=msg.content))

        return langchain_messages

    @staticmethod
    async def delete_conversation(session_id: str, user_id: Optional[str] = None) -> bool:
        """
        Delete conversation by session ID

        Args:
            session_id: Session identifier
            user_id: Bắt buộc truyền khi gọi từ request — nếu không, bất kỳ ai
                cũng xoá được hội thoại của người khác chỉ bằng session_id.

        Returns:
            True if deleted
        """
        db = get_database()
        collection = db[ConversationService.COLLECTION_NAME]

        query: Dict[str, Any] = {"session_id": session_id}
        if user_id is not None:
            query["user_id"] = user_id

        result = await collection.delete_one(query)
        return result.deleted_count > 0

    @staticmethod
    async def get_user_conversations(
        user_id: str,
        limit: int = 50
    ) -> List[ConversationHistory]:
        """
        Get all conversations for a user

        Args:
            user_id: User ID
            limit: Maximum number of conversations to return

        Returns:
            List of conversations
        """
        db = get_database()
        collection = db[ConversationService.COLLECTION_NAME]

        cursor = collection.find({"user_id": user_id}).sort("updated_at", -1).limit(limit)
        conversations = []

        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            conversations.append(ConversationHistory(**doc))

        return conversations
