# 🤖 AI Agent System - Quick Start Guide

## 📋 Tổng quan

AI Agent system cho phép bạn tương tác với hệ thống OCR Datecode qua natural language interface. Agent có thể:

- ✅ Check service status
- ✅ Start/Stop services
- ✅ Analyze logs
- ✅ Troubleshoot issues
- ✅ Provide proactive suggestions

---

## 🚀 Setup

### 1. Cài đặt dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Cấu hình OpenAI API Key

Thêm vào file `backend/.env`:

```bash
OPENAI_API_KEY=sk-your-key-here
```

### 3. Test Agent System

```bash
cd backend
python test_agent.py
```

Bạn sẽ thấy output như:

```
================================================================================
AI Agent System Test
================================================================================
🔍 Testing imports...
  ✅ BaseAgent imported
  ✅ AgentRegistry imported
  ✅ Tool system imported
  ✅ Service tools imported
  ✅ ServiceAgent imported

🔍 Testing OpenAI configuration...
  ✅ OPENAI_API_KEY found: sk-proj-...

🔍 Testing agent registry...
  ✅ Found 1 registered agent(s):
     - service_management: ServiceManagementAgent

🔍 Testing tool registry...
  ✅ Found 4 registered tool(s):
     - check_service_status (service)
     - start_service (service)
     - stop_service (service)
     - get_service_logs (service)

🔍 Testing agent creation...
  ✅ Created agent: ServiceManagementAgent
     Model: gpt-4o-mini
     Tools: 4

🔍 Testing tool execution...
  ✅ Tool executed successfully
     Status: stopped
     Running: False
     WebSocket: False

================================================================================
🎉 All tests passed! Agent system is ready to use.
================================================================================
```

### 4. Start Backend Server

```bash
cd backend
uvicorn app.main:app --reload
```

---

## 🧪 Test API với cURL

### 1. Check Agent Health

```bash
curl http://localhost:8000/api/agent/health
```

Response:
```json
{
  "status": "healthy",
  "agents_registered": 1,
  "openai_configured": true,
  "message": "Agent system is operational"
}
```

### 2. List Available Agents

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/agent/agents
```

Response:
```json
[
  {
    "agent_id": "service_management",
    "class_name": "ServiceManagementAgent",
    "description": "Agent chuyên quản lý services..."
  }
]
```

### 3. Chat with Agent

```bash
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "message": "Camera service có đang chạy không?",
    "agent_id": "service_management"
  }'
```

Response:
```json
{
  "response": "🔍 Tôi đã kiểm tra camera service:\n\nService không chạy ❌\n\nBạn có muốn tôi khởi động service không?",
  "agent_id": "service_management",
  "session_id": "session_user123_1704960000.0",
  "tool_calls": [
    {
      "tool": "check_service_status",
      "args": {"service_name": "camera_management"},
      "id": "call_abc123"
    }
  ],
  "timestamp": "2026-01-11T10:30:00"
}
```

---

## 💬 Example Conversations

### Conversation 1: Check Status

**User:** Camera service có đang chạy không?

**Agent:** 🔍 Tôi đã kiểm tra camera service:

Service đang chạy và đã kết nối WebSocket ✅
- PID: 12345
- CPU: 2.5%
- Memory: 150 MB

Hệ thống đang hoạt động bình thường!

---

### Conversation 2: Start Service

**User:** Hãy start camera service

**Agent:** 🔍 Để tôi kiểm tra trước...

Service chưa chạy. Bạn có chắc muốn khởi động Camera Management Service không?

**User:** Có

**Agent:** 🚀 Đang khởi động service...

✅ Service đã khởi động thành công!
- PID: 12346
- Logs: /path/to/logs/camera_management.log

⏳ Đang chờ WebSocket kết nối...

✅ Service đã kết nối! Hệ thống sẵn sàng hoạt động.

---

### Conversation 3: Troubleshooting

**User:** Service không connect được

**Agent:** 🔍 Để tôi chẩn đoán vấn đề...

Tình trạng hiện tại:
1. ✅ Service process đang chạy (PID: 12345)
2. ❌ WebSocket connection: Failed

📋 Logs gần đây:
```
[ERROR] Connection refused to ws://localhost:8000
[ERROR] Retrying in 5 seconds...
```

💡 Nguyên nhân có thể:
- Backend server không chạy trên port 8000
- Firewall đang block WebSocket connection
- Backend chưa start xong

🔧 Giải pháp:
1. Kiểm tra backend server: `curl http://localhost:8000/health`
2. Nếu backend chưa chạy, start backend trước
3. Restart camera service: tôi có thể giúp bạn restart

Bạn muốn tôi restart service không?

---

## 📝 API Documentation

Sau khi start server, truy cập:

```
http://localhost:8000/docs
```

Tìm section **"AI Agent"** để xem full API documentation.

---

## 🔧 Extending Agent System

### Thêm Agent mới

1. Tạo file `backend/app/agent/agents/my_agent.py`:

```python
from app.agent.base.base_agent import BaseAgent
from app.agent.core.registry import AgentRegistry

@AgentRegistry.register("my_agent")
class MyAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return "Your system prompt..."

    def get_tools(self) -> List[Any]:
        return [tool1, tool2]

    def build_graph(self) -> StateGraph:
        # Build your graph...
        pass
```

2. Import trong `main.py`:

```python
from app.agent.agents.my_agent import MyAgent
```

3. Agent tự động available tại `/api/agent/chat` với `agent_id="my_agent"`

---

### Thêm Tools mới

1. Tạo file `backend/app/agent/tools/my_tools.py`:

```python
from app.agent.tools.base_tool import BaseTool, ToolMetadata

def my_function(param: str) -> dict:
    return {"result": f"Processed: {param}"}

my_tool = BaseTool.create_tool(
    func=my_function,
    metadata=ToolMetadata(
        name="my_tool",
        description="What this tool does",
        category="my_category"
    )
)

ToolRegistry.register(my_tool)
```

2. Dùng tool trong agent:

```python
from app.agent.tools.my_tools import my_tool

def get_tools(self):
    return [my_tool, ...]
```

---

## 🐛 Troubleshooting

### Agent không khởi động

**Lỗi:** `OPENAI_API_KEY not found`

**Giải pháp:**
```bash
# Kiểm tra .env file
cat backend/.env | grep OPENAI_API_KEY

# Hoặc set trực tiếp
export OPENAI_API_KEY=sk-your-key-here
```

---

### Import errors

**Lỗi:** `ModuleNotFoundError: No module named 'langgraph'`

**Giải pháp:**
```bash
pip install langgraph langchain langchain-openai
```

---

### Tools không hoạt động

**Debug steps:**
1. Check tool registry:
   ```python
   from app.agent.tools.base_tool import ToolRegistry
   print(ToolRegistry.list_tools())
   ```

2. Test tool trực tiếp:
   ```python
   from app.agent.tools.service_tools import check_service_status
   result = check_service_status("camera_management")
   print(result)
   ```

---

## 📚 Resources

- **Architecture Doc:** `docs/AGENT_ARCHITECTURE.md`
- **Implementation Guide:** `docs/AGENT_IMPLEMENTATION_GUIDE.md`
- **LangGraph Docs:** https://langchain-ai.github.io/langgraph/
- **OpenAI Function Calling:** https://platform.openai.com/docs/guides/function-calling

---

## 🎉 Next Steps

1. ✅ Backend agent hoàn thành
2. ⬜ Frontend chat UI (React component)
3. ⬜ WebSocket streaming
4. ⬜ Conversation memory
5. ⬜ More agents (Recipe, Analytics, etc.)

---

**Happy Building! 🚀**
