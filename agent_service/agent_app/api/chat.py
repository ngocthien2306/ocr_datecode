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

from agent_app.core.config import settings
from agent_app.core.registry import AgentRegistry
from agent_app.base.base_agent import AgentState
from agent_app.api.deps import get_current_user
from agent_app.core.suggestions import extract_suggestions, fallback_suggestions
from agent_app.memory.conversation_service import ConversationService
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["AI Agent"])


# ============================================================================
# Pydantic Models
# ============================================================================

class ChatRequest(BaseModel):
    """Chat request from user"""
    message: str
    agent_id: str = "orchestrator"
    session_id: Optional[str] = None
    stream: bool = False


class ChatOption(BaseModel):
    """
    Một lựa chọn bấm được ở FE.

    Bấm nút = gửi `value` vào /api/agent/chat như tin nhắn thường (cùng
    session_id). FE không cần xử lý gì đặc biệt, và user vẫn gõ tay được.
    """
    label: str          # hiển thị trên nút
    value: str          # câu gửi đi khi bấm
    hint: Optional[str] = None   # chú thích phụ, vd "49.503 sản phẩm"


class ChatResponse(BaseModel):
    """Chat response to user"""
    response: str
    agent_id: str
    session_id: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    options: Optional[List[ChatOption]] = None
    suggestions: Optional[List[str]] = None
    images: Optional[List[Dict[str, Any]]] = None
    charts: Optional[List[Dict[str, Any]]] = None
    # File tải về do tool sinh ra (báo cáo). Tách khỏi `images` vì đây là thứ
    # user bấm để tải, không phải ảnh hiển thị trong luồng chat.
    files: Optional[List[Dict[str, Any]]] = None
    # Thẻ thông tin (hiện tại: người thao tác trong audit log).
    cards: Optional[List[Dict[str, Any]]] = None
    # Ô KPI và bảng dữ liệu, dựng tất định từ kết quả tool.
    kpis: Optional[List[Dict[str, Any]]] = None
    tables: Optional[List[Dict[str, Any]]] = None
    timestamp: str


class AgentInfo(BaseModel):
    """Agent information"""
    agent_id: str
    class_name: str
    description: str


# ============================================================================
# Helper Functions
# ============================================================================

def extract_assistant_message(state) -> str:
    """Extract the final assistant message from state (dict or AgentState)."""
    messages = state.get("messages") if isinstance(state, dict) else state.messages

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content

    return "No response generated"


def extract_tool_calls(state, skip: int = 0) -> Optional[List[Dict[str, Any]]]:
    """
    Tool đã gọi TRONG LƯỢT NÀY, phục vụ debug/minh bạch.

    `skip` = số message lịch sử được replay vào. Không bỏ qua chúng thì mỗi lượt
    trả về toàn bộ tool call từ đầu hội thoại, danh sách phình mãi và người xem
    tưởng agent vừa gọi lại hết những thứ đó.
    """
    tool_calls = []

    messages = state.get("messages") if isinstance(state, dict) else state.messages
    messages = messages[skip:]

    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tool_call in msg.tool_calls:
                tool_calls.append({
                    "tool": tool_call.get("name"),
                    "args": tool_call.get("args"),
                    "id": tool_call.get("id")
                })

    return tool_calls or None


def _trim_history(messages: List[Any], limit: int) -> List[Any]:
    """
    Cắt bớt lịch sử NHƯNG chỉ tại ranh giới lượt hội thoại.

    Không được cắt thẳng bằng `messages[-limit:]`: nhát cắt rất dễ rơi vào giữa
    một chuỗi tool call, để lại ToolMessage mồ côi ở đầu danh sách. OpenAI từ
    chối thẳng:

        messages with role 'tool' must be a response to a preceeding
        message with 'tool_calls'

    Và vì lịch sử chỉ dài thêm sau mỗi lượt, session sẽ hỏng VĨNH VIỄN kể từ
    lần đầu dính lỗi — mọi câu hỏi sau đó đều 500.

    Cắt ở message 'user' đảm bảo mỗi lượt được giữ trọn vẹn: user → assistant
    (tool_calls) → tool → assistant.
    """
    # Vị trí các message 'user' — ứng viên duy nhất cho điểm bắt đầu an toàn.
    user_idx = [i for i, m in enumerate(messages) if m.role == "user"]
    if not user_idx:
        # Không có lượt nào hoàn chỉnh — thà bỏ hết còn hơn gửi mảnh vỡ.
        return []

    # Điểm bắt đầu xa nhất về trước mà vẫn nằm trong hạn mức (giữ được nhiều
    # ngữ cảnh nhất). Nếu ngay cả lượt cuối cũng vượt hạn mức thì vẫn giữ nó —
    # thiếu ngữ cảnh còn hơn gửi lịch sử hỏng.
    within = [i for i in user_idx if len(messages) - i <= limit]
    start = within[0] if within else user_idx[-1]

    return messages[start:]


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/chat", response_model=ChatResponse, summary="Chat with AI agent")
async def chat(
    request: ChatRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Chat with AI agent

    **Available agents:**
    - `orchestrator`: tự phân tích intent rồi route sang agent chuyên biệt
    - `service_management`: quản lý AI services (status / start / stop / logs)
    - `historical_analytics`: thống kê pass-fail, sản lượng, lịch sử recipe
    """
    try:
        logger.info("User %s chatting with agent: %s", current_user["username"], request.agent_id)

        try:
            agent = AgentRegistry.get_agent(request.agent_id)
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

        session_id = request.session_id or f"session_{current_user['id']}_{datetime.now().timestamp()}"

        # ------------------------------------------------------------------
        # Load conversation history (scoped tới chính user gọi — session_id do
        # client gửi nên không được tin, xem ConversationService)
        # ------------------------------------------------------------------
        try:
            conversation = await ConversationService.get_or_create_conversation(
                session_id=session_id,
                user_id=current_user["id"],
                agent_id=request.agent_id,
                metadata={
                    "username": current_user["username"],
                    "role": current_user["role"],
                }
            )
        except PermissionError:
            logger.warning(
                "User %s tried to use session_id owned by someone else: %s",
                current_user["username"], session_id
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Session does not belong to the current user",
            )

        # Bỏ system message trong lịch sử: agent tự dựng system prompt mới mỗi
        # lượt (kèm ngày tháng hiện tại). Các session cũ có thể còn lưu nguyên
        # cả prompt 6 KB do một bug đã sửa — lọc ở đây để không replay lại.
        usable = [m for m in conversation.messages if m.role != "system"]

        stored_messages = _trim_history(usable, settings.MAX_HISTORY_MESSAGES)
        historical_messages = ConversationService.conversation_messages_to_langchain_messages(
            stored_messages
        )

        logger.info(
            "Session %s: %d stored / %d replayed messages",
            session_id, len(conversation.messages), len(historical_messages)
        )

        current_messages = historical_messages + [HumanMessage(content=request.message)]

        state = AgentState(
            messages=current_messages,
            user_id=current_user["id"],
            session_id=session_id,
            context={
                "username": current_user["username"],
                "role": current_user["role"],
            }
        )

        result_state = await agent.ainvoke(state)

        response_text = extract_assistant_message(result_state)
        tool_calls = extract_tool_calls(result_state, skip=len(historical_messages))

        # Agent đặt ui_options vào context khi có thứ cần user chọn
        # (tên recipe mơ hồ, hoặc user chưa nêu recipe nào).
        result_context = result_state.get("context") if isinstance(result_state, dict) else result_state.context
        options = (result_context or {}).get("ui_options")
        images = (result_context or {}).get("ui_images")
        charts = (result_context or {}).get("ui_charts")
        files = (result_context or {}).get("ui_files")
        cards = (result_context or {}).get("ui_cards")
        kpis = (result_context or {}).get("ui_kpis")
        tables = (result_context or {}).get("ui_tables")

        # Gỡ khối [SUGGESTIONS] khỏi text hiển thị, tách thành chip riêng.
        # Ba nguồn theo thứ tự ưu tiên: LLM tự sinh → agent đặt sẵn trong
        # context (vd orchestrator không hiểu ý, gợi ý câu vào bài) → suy từ
        # tool vừa chạy.
        response_text, suggestions = extract_suggestions(response_text)
        if not suggestions:
            suggestions = (result_context or {}).get("ui_suggestions") or []
        if not suggestions:
            suggestions = fallback_suggestions(tool_calls)
        # Đang bắt user chọn recipe thì đừng bày thêm gợi ý gây phân tán.
        if options:
            suggestions = None

        # ------------------------------------------------------------------
        # Persist only the messages produced this turn
        # ------------------------------------------------------------------
        result_messages = result_state.get("messages") if isinstance(result_state, dict) else result_state.messages
        new_messages = result_messages[len(historical_messages):]

        conversation_messages = ConversationService.langchain_messages_to_conversation_messages(new_messages)

        # Không lưu system message — nó được dựng lại mỗi lượt, lưu chỉ tổ phình
        # DB và khiến lịch sử replay sai.
        conversation_messages = [m for m in conversation_messages if m.role != "system"]

        # Lưu bản đã gỡ thẻ. Nếu lưu nguyên, FE tải lại lịch sử qua
        # /agent/conversations sẽ hiện markup [SUGGESTIONS] thô cho user.
        for msg in conversation_messages:
            if msg.role == "assistant" and msg.content:
                msg.content, _ = extract_suggestions(msg.content)

        await ConversationService.add_messages(session_id, conversation_messages)

        logger.info(
            "Agent replied (%d chars), saved %d new message(s)",
            len(response_text), len(conversation_messages)
        )

        return ChatResponse(
            response=response_text,
            agent_id=request.agent_id,
            session_id=session_id,
            tool_calls=tool_calls,
            options=options,
            suggestions=suggestions,
            images=images,
            charts=charts,
            files=files,
            cards=cards,
            kpis=kpis,
            tables=tables,
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in chat endpoint: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}"
        )


@router.get("/agents", response_model=List[AgentInfo], summary="List available agents")
async def list_agents(current_user: Dict[str, Any] = Depends(get_current_user)):
    """List all registered AI agents."""
    return AgentRegistry.list_agents()


@router.get("/health", summary="Agent system health check")
async def agent_health():
    """
    Health check — không yêu cầu auth để dùng cho monitoring / systemd.

    Kiểm tra: agent đã register, OpenAI key đã cấu hình, backend có với tới được.
    """
    from agent_app.core.backend_client import is_backend_reachable

    agents = AgentRegistry.list_agents()
    openai_configured = bool(settings.OPENAI_API_KEY)

    return {
        "status": "healthy" if (agents and openai_configured) else "unhealthy",
        "agents_registered": len(agents),
        "openai_configured": openai_configured,
        "backend_url": settings.BACKEND_URL,
        "backend_reachable": is_backend_reachable(),
        "message": (
            "Agent system is operational" if openai_configured
            else "OPENAI_API_KEY not configured"
        ),
    }


@router.get("/service/status", summary="Get service status (no LLM)")
async def get_service_status(
    service_name: str = "camera_management",
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Trạng thái service, gọi thẳng tool — không đi qua LLM nên không tốn token.
    Dùng cho ServiceStatusBar poll liên tục ở FE.
    """
    from agent_app.tools.service_tools import check_service_status

    try:
        return check_service_status(service_name)
    except Exception as e:
        logger.error("Error checking service status: %s", e)
        return {
            "service_name": service_name,
            "is_running": False,
            "pid": None,
            "websocket_connected": None,
            "status": "unknown",
            "message": f"Error checking status: {str(e)}",
            "cpu_percent": None,
            "memory_mb": None,
        }


@router.delete("/conversation/{session_id}", summary="Clear conversation history")
async def clear_conversation(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Xoá lịch sử hội thoại của chính user đang gọi."""
    deleted = await ConversationService.delete_conversation(session_id, user_id=current_user["id"])

    if deleted:
        logger.info("User %s cleared conversation: %s", current_user["username"], session_id)
        return {"success": True, "message": "Conversation history cleared successfully"}

    return {"success": False, "message": "Conversation not found or already cleared"}


@router.get("/conversations", summary="Get user's conversation history")
async def get_conversations(
    current_user: Dict[str, Any] = Depends(get_current_user),
    limit: int = 50
):
    """Danh sách hội thoại của user hiện tại."""
    return await ConversationService.get_user_conversations(
        user_id=current_user["id"],
        limit=limit
    )


# ============================================================================
# Streaming (experimental — FE chưa dùng, KHÔNG lưu history)
# ============================================================================

@router.post("/chat/stream", summary="Chat with streaming response (experimental)")
async def chat_stream(
    request: ChatRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Server-Sent Events stream.

    LƯU Ý: endpoint này KHÔNG nạp và KHÔNG lưu conversation history — mỗi lần
    gọi là một lượt độc lập. Giữ nguyên trạng từ bản backend cũ.
    """
    try:
        agent = AgentRegistry.get_agent(request.agent_id)
        session_id = request.session_id or f"session_{current_user['id']}_{datetime.now().timestamp()}"

        state = AgentState(
            messages=[HumanMessage(content=request.message)],
            user_id=current_user["id"],
            session_id=session_id,
            context={"username": current_user["username"]}
        )

        async def generate():
            async for chunk in agent.astream(state):
                yield f"data: {json.dumps({'type': 'chunk', 'data': str(chunk), 'timestamp': datetime.now().isoformat()})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        logger.error("Error in streaming chat: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
