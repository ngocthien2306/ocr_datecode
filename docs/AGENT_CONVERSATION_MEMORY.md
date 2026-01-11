# 🧠 AI Agent Conversation Memory

## Overview

Đã implement **Backend-side conversation memory** cho AI Agent system, cho phép agent nhớ toàn bộ lịch sử trò chuyện của người dùng.

---

## 🎯 Vấn đề đã giải quyết

### Before (Không có memory):
```
User: "Camera service có đang chạy không?"
Agent: ✅ "Service đang chạy"

User: "Hãy kết nối lại"
Agent: ❌ "Bạn muốn kết nối service nào?" (Không nhớ đang nói về camera service)
```

### After (Có memory):
```
User: "Camera service có đang chạy không?"
Agent: ✅ "Service đang chạy"

User: "Hãy kết nối lại"
Agent: ✅ "Tôi sẽ kết nối lại Camera Management Service..." (Nhớ context)
```

---

## 🏗️ Architecture

### 1. **MongoDB Storage**

```python
# Collection: agent_conversations
{
    "_id": ObjectId,
    "session_id": "session_676e123...",  # Unique per user session
    "user_id": "user123",
    "agent_id": "service_management",
    "messages": [
        {
            "role": "user",
            "content": "Camera service có đang chạy không?",
            "timestamp": "2026-01-11T22:00:00",
            "tool_calls": null
        },
        {
            "role": "assistant",
            "content": "Service đang chạy ✅",
            "timestamp": "2026-01-11T22:00:02",
            "tool_calls": [
                {"tool": "check_service_status", "args": {...}}
            ]
        },
        {
            "role": "tool",
            "content": "{'status': 'running', ...}",
            "tool_call_id": "call_abc123",
            "name": "check_service_status"
        }
    ],
    "created_at": "2026-01-11T22:00:00",
    "updated_at": "2026-01-11T22:05:00",
    "metadata": {
        "username": "admin",
        "role": "admin"
    }
}
```

### 2. **Conversation Flow**

```
┌─────────────┐
│   User      │
│  Message    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│  1. Load conversation history       │
│     from MongoDB (session_id)       │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  2. Convert to LangChain messages   │
│     [HumanMessage, AIMessage, ...]  │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  3. Append new user message         │
│     history + [new HumanMessage]    │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  4. Execute agent with full context │
│     Agent can see entire history    │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  5. Save NEW messages to MongoDB    │
│     (user msg + agent msg + tools)  │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  6. Return response to user         │
└─────────────────────────────────────┘
```

---

## 📁 Files Created/Modified

### New Files:

**`backend/app/models/conversation.py`**
- `ConversationMessage`: Single message model
- `ConversationHistory`: Complete conversation document
- `ConversationCreate`, `ConversationUpdate`: Request schemas

**`backend/app/services/conversation_service.py`**
- `create_conversation()`: Create new session
- `get_conversation()`: Load by session_id
- `add_messages()`: Append messages
- `get_or_create_conversation()`: Helper
- `langchain_messages_to_conversation_messages()`: Convert to storage format
- `conversation_messages_to_langchain_messages()`: Convert to LangChain format
- `delete_conversation()`: Clear history
- `get_user_conversations()`: List user sessions

### Modified Files:

**`backend/app/api/endpoints/agent.py`**
- Load conversation history before executing agent
- Save new messages after agent execution
- Added `DELETE /agent/conversation/{session_id}` endpoint
- Added `GET /agent/conversations` endpoint

**`frontend-ts/src/services/agentService.ts`**
- Added `clearConversation()` method

**`frontend-ts/src/components/agent/AgentChatWidget.tsx`**
- Added 🗑️ button in header
- Implemented `handleClearChat()` function
- Confirmation dialog before clearing

---

## 🔧 API Endpoints

### 1. Chat (with memory)
```http
POST /api/agent/chat
{
    "message": "Hãy kết nối lại",
    "agent_id": "service_management",
    "session_id": "session_676e123..."  // Same session = remembered context
}
```

**Response:**
```json
{
    "response": "Tôi sẽ kết nối lại Camera Management Service...",
    "agent_id": "service_management",
    "session_id": "session_676e123...",
    "tool_calls": [...],
    "timestamp": "2026-01-11T22:00:00"
}
```

### 2. Clear Conversation
```http
DELETE /api/agent/conversation/{session_id}
```

**Response:**
```json
{
    "success": true,
    "message": "Conversation history cleared successfully"
}
```

### 3. Get User Conversations
```http
GET /api/agent/conversations?limit=50
```

**Response:**
```json
[
    {
        "id": "abc123",
        "session_id": "session_676e123...",
        "user_id": "user123",
        "agent_id": "service_management",
        "messages": [...],
        "created_at": "2026-01-11T22:00:00",
        "updated_at": "2026-01-11T22:05:00",
        "metadata": {...}
    }
]
```

---

## 💡 Key Features

### ✅ Persistent Memory
- Conversation history lưu trong MongoDB
- Survive server restarts
- Cross-device (same user, same session_id)

### ✅ Context Awareness
- Agent nhớ:
  - Previous questions
  - Previous answers
  - Tool calls executed
  - User preferences

### ✅ Session Management
- Unique `session_id` per conversation
- Frontend tự động tạo session ID khi mở chat widget
- Clear conversation = Generate new session ID

### ✅ Efficient Storage
- Chỉ lưu NEW messages mỗi lần (không duplicate)
- Convert giữa LangChain format ↔️ MongoDB format
- Index trên `session_id` cho fast lookup

---

## 🧪 Testing

### Test Multi-turn Conversation:

```bash
# Turn 1
POST /api/agent/chat
{
    "message": "Camera service có đang chạy không?",
    "session_id": "test_session_1"
}

# Turn 2 (same session)
POST /api/agent/chat
{
    "message": "Hãy stop nó lại",  # Agent knows "nó" = camera service
    "session_id": "test_session_1"
}

# Turn 3
POST /api/agent/chat
{
    "message": "Tại sao bạn stop?",  # Agent can reference its own previous action
    "session_id": "test_session_1"
}
```

### Test Clear Conversation:

```bash
# Clear history
DELETE /api/agent/conversation/test_session_1

# Try to reference previous context (should fail gracefully)
POST /api/agent/chat
{
    "message": "Service đó có chạy chưa?",  # "đó" has no context anymore
    "session_id": "test_session_2"  # New session
}
```

---

## 🎨 UI Features

### Clear Chat Button (🗑️)
- Located in chat header (next to minimize/close)
- Confirmation dialog before clearing
- Resets to welcome message
- Generates new session ID

### Session Persistence
- Frontend maintains `sessionId.current` using `useRef`
- Same session ID across multiple messages
- Only changes when:
  - User clears chat
  - Widget first opens (if no session provided)

---

## 📊 Performance Considerations

### Efficient Message Loading
```python
# Only fetch messages for current session (not all user conversations)
conversation = await ConversationService.get_conversation(session_id)

# Convert to LangChain format (in memory, fast)
messages = ConversationService.conversation_messages_to_langchain_messages(
    conversation.messages
)
```

### Incremental Storage
```python
# Only save NEW messages (after agent execution)
new_messages = result_messages[len(historical_messages):]
await ConversationService.add_messages(session_id, new_messages)
```

### MongoDB Indexing
```python
# Create index on session_id for fast lookup
await collection.create_index("session_id", unique=True)
```

---

## 🚀 Benefits

1. **Better UX**: Agent hiểu context, không hỏi lại
2. **Natural Conversations**: Users có thể nói "nó", "đó", "service đang nói"
3. **Debugging**: Complete conversation history for troubleshooting
4. **Analytics**: Track user interactions, common questions
5. **Persistence**: Conversations survive restarts, page refreshes

---

## 🔮 Future Enhancements

### Phase 1: Conversation Management UI
- [ ] List all previous conversations
- [ ] Switch between conversations
- [ ] Search within conversation
- [ ] Export conversation to JSON/PDF

### Phase 2: Smart Context Window
- [ ] Limit context to last N messages (prevent token overflow)
- [ ] Summarize old messages for long conversations
- [ ] Sliding window approach

### Phase 3: Advanced Features
- [ ] Conversation tagging/categorization
- [ ] Share conversations with team members
- [ ] Conversation templates
- [ ] Analytics dashboard

---

## 📝 Example Conversation with Memory

### Complete Multi-turn Dialogue:

```
👤 User: Camera service có đang chạy không?

🤖 Agent: 🔍 Checking service status...
         ✅ Camera Management Service đang chạy
         - PID: 12345
         - WebSocket: Connected
         🔧 Tools: check_service_status

👤 User: CPU usage bao nhiêu?

🤖 Agent: 📊 CPU usage của Camera Management Service là 2.5%
         Memory: 150 MB
         Service đang hoạt động bình thường
         [Agent nhớ đang nói về camera service, không hỏi lại]

👤 User: Hãy stop nó lại

🤖 Agent: ⚠️ Bạn có chắc muốn stop Camera Management Service không?
         Service hiện đang hoạt động tốt (CPU 2.5%, Memory 150MB)
         [Agent nhớ metrics vừa báo cáo]

👤 User: Có

🤖 Agent: 🛑 Đang stop Camera Management Service...
         ✅ Service đã stop thành công (PID 12345)
         🔧 Tools: stop_service

👤 User: Tại sao bạn stop?

🤖 Agent: 📝 Tôi đã stop Camera Management Service vì bạn yêu cầu.
         Trước đó service đang chạy tốt với:
         - CPU: 2.5%
         - Memory: 150MB
         - WebSocket: Connected

         Bạn có muốn start lại không?
         [Agent nhớ toàn bộ conversation history]
```

---

## ✅ Summary

- ✅ **Backend-side persistent memory** using MongoDB
- ✅ **Full conversation history** maintained per session
- ✅ **Context-aware responses** from agent
- ✅ **Clear conversation** feature with UI
- ✅ **Efficient storage** and retrieval
- ✅ **Production-ready** implementation

**Project Status**: 🎉 **Conversation Memory Complete!**

---

**Built with**: MongoDB, LangChain, FastAPI, React
**Version**: 2.0.0 (with Memory)
**Date**: January 2026
