# 🎨 AI Agent Chat UI Design Document

## 📋 Tổng quan

Document này mô tả chi tiết thiết kế UI cho AI Agent Chat system, tích hợp vào OCR Datecode frontend hiện tại.

---

## 🎯 Phân tích Frontend hiện tại

### Tech Stack
- **Framework**: React 19.2.0 + TypeScript
- **Styling**: TailwindCSS 3.x + Custom CSS
- **Build**: Vite 7.2.4
- **HTTP Client**: Axios 1.13.2
- **Real-time**: Socket.io-client 4.7.2

### Cấu trúc hiện tại
```
frontend-ts/src/
├── components/
│   ├── dashboard/     # Dashboard, Settings, Logs, Historical...
│   ├── camera/        # Camera management
│   ├── recipe/        # Recipe management
│   ├── shared/        # Toast, ConfirmDialog, etc.
│   └── ui/            # UI components
├── contexts/          # React contexts (Toast, User)
├── services/          # API services
└── styles/            # CSS files
```

### Style Pattern
- Dark/Light mode support
- Industrial/Technical aesthetic
- Sidebar navigation
- Modal-based workflows

---

## 🎨 UI Design Proposal

### **Option 1: Floating Chat Widget (Recommended ⭐⭐⭐)**

```
┌─────────────────────────────────────────────────────────────┐
│  Dashboard Header                                    [User] │
├───────────┬─────────────────────────────────────────────────┤
│           │                                                 │
│ Sidebar   │      Main Content Area                          │
│           │                                                 │
│ • Home    │                                                 │
│ • Camera  │                                                 │
│ • Recipe  │                                          ┌──────┤
│ • Logs    │                                          │ 🤖   │
│ • Config  │                                          │      │
│           │                                          │ Chat │
│           │                                          │      │
│           │                                          │ Panel│
│           │                                          │      │
│           │                                          │  ↕   │
│           │                                          │ 400px│
│           │                                          │      │
│           │                                          │ [✕]  │
└───────────┴──────────────────────────────────────────┴──────┘
                                                        300px
```

**Ưu điểm:**
- ✅ Không chiếm nhiều không gian
- ✅ Luôn accessible từ mọi trang
- ✅ Có thể minimize/maximize
- ✅ Familiar UX (như chat support)

**Vị trí:** Fixed bottom-right corner, overlay trên content

---

### **Option 2: Sidebar Tab**

```
┌─────────────────────────────────────────────────────────────┐
│  Dashboard Header                    [🤖 Agent] [User]      │
├───────────┬─────────────────────────────────────────────────┤
│           │                                                 │
│ Sidebar   │      Main Content Area                          │
│           │                                                 │
│ • Home    │                                                 │
│ • Camera  │                                                 │
│ • Recipe  │                                                 │
│ • Logs    │                                                 │
│ ──────────│                                                 │
│ 🤖 Agent  │  ← Click này mở Agent panel full height         │
│           │                                                 │
└───────────┴─────────────────────────────────────────────────┘
```

**Ưu điểm:**
- ✅ Integrated vào navigation
- ✅ Full screen height khi cần
- ✅ Clear separation

**Nhược điểm:**
- ❌ Phải switch tab để dùng
- ❌ Không multitask được

---

### **Option 3: Modal Dialog**

Tương tự RecipeFormModal hiện tại, mở full-screen overlay.

**Nhược điểm:**
- ❌ Block toàn bộ UI
- ❌ Không tiện cho quick questions

---

## ✅ Quyết định: **Option 1 - Floating Chat Widget**

Vì phù hợp nhất với use case:
- Quick support khi làm việc
- Không interrupt workflow
- Modern UX

---

## 🎨 Detailed Component Design

### 1. **AgentChatWidget Component** (Main Container)

**File:** `frontend-ts/src/components/agent/AgentChatWidget.tsx`

```typescript
interface AgentChatWidgetProps {
  // Optional: Control visibility from parent
  isOpen?: boolean;
  onToggle?: () => void;
}

States:
- isOpen: boolean        // Widget expanded/collapsed
- isMinimized: boolean   // Minimized to icon only
- messages: Message[]    // Chat history
- isTyping: boolean      // Agent typing indicator
- serviceStatus: Status  // Camera service status
```

**Layout:**
```
┌────────────────────────────┐
│ 🤖 Service Assistant  [−][✕]│  ← Header (draggable)
├────────────────────────────┤
│ 🟢 Camera Service: Running │  ← Status Bar (auto-update)
├────────────────────────────┤
│                            │
│  Messages Area             │  ← Scrollable chat
│  (Auto-scroll to bottom)   │
│                            │
│  [User message]            │
│                            │
│  [Agent response           │
│   with tool execution]     │
│                            │
│  [User message]            │
│                            │
│  [Agent typing...]         │  ← Typing indicator
│                            │
├────────────────────────────┤
│ Quick Actions:             │  ← Suggested actions
│ [Check Status] [View Logs] │
├────────────────────────────┤
│ Type your message...  [↑]  │  ← Input area
└────────────────────────────┘
```

**Dimensions:**
- Width: 380px (fixed)
- Height: 500px (resizable, min: 300px, max: 700px)
- Position: Fixed bottom-right, 20px margin

---

### 2. **MessageBubble Component**

**File:** `frontend-ts/src/components/agent/MessageBubble.tsx`

```typescript
interface MessageBubbleProps {
  message: Message;
  isUser: boolean;
}

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  toolCalls?: ToolCall[];  // Show which tools were used
  status?: 'pending' | 'success' | 'error';
}
```

**User Message:**
```
                    ┌─────────────────────┐
                    │ Start camera service│ ← Blue bg
                    │ 10:30 AM            │
                    └─────────────────────┘
```

**Agent Message:**
```
┌──────────────────────────────┐
│ 🔍 Checking service status...│ ← Gray bg
│                              │
│ ✅ Service started!          │
│ PID: 12345                   │
│                              │
│ 🔧 Tools used:               │
│ • check_service_status       │
│ • start_service              │
│                              │
│ 10:30 AM                     │
└──────────────────────────────┘
```

**Tool Execution Indicator:**
```
┌──────────────────────────────┐
│ ⚙️ Executing: start_service  │ ← Yellow bg, animated
│ Please wait...               │
└──────────────────────────────┘
```

---

### 3. **ServiceStatusBar Component**

**File:** `frontend-ts/src/components/agent/ServiceStatusBar.tsx`

```typescript
interface ServiceStatusBarProps {
  onRefresh?: () => void;
}

States:
- status: 'running' | 'stopped' | 'error'
- isConnected: boolean  // WebSocket connection
- lastUpdate: Date
```

**Layout:**
```
┌────────────────────────────────────────┐
│ 🟢 Camera Service     [Refresh] [Info] │ ← Running
├────────────────────────────────────────┤

┌────────────────────────────────────────┐
│ ⚠️ Service Degraded   [Restart] [Logs] │ ← Degraded
├────────────────────────────────────────┤

┌────────────────────────────────────────┐
│ ❌ Service Stopped    [Start]   [Help] │ ← Stopped
├────────────────────────────────────────┤
```

**Auto-refresh:** Poll every 5 seconds khi widget open

---

### 4. **QuickActions Component**

**File:** `frontend-ts/src/components/agent/QuickActions.tsx`

Suggested quick actions dựa trên context:

```typescript
interface QuickAction {
  label: string;
  icon: string;
  message: string;  // Pre-filled message to send
  condition?: () => boolean;  // When to show
}

const QUICK_ACTIONS: QuickAction[] = [
  {
    label: "Check Status",
    icon: "🔍",
    message: "Camera service có đang chạy không?"
  },
  {
    label: "View Logs",
    icon: "📝",
    message: "Cho tôi xem logs của camera service"
  },
  {
    label: "Restart Service",
    icon: "🔄",
    message: "Hãy restart camera service",
    condition: () => serviceStatus === 'running'
  }
]
```

**Layout:**
```
┌──────────────────────────────────────┐
│ Suggestions:                         │
│ [🔍 Check Status] [📝 View Logs]     │
│ [🔄 Restart] [❓ Help]                │
└──────────────────────────────────────┘
```

---

### 5. **ChatInput Component**

**File:** `frontend-ts/src/components/agent/ChatInput.tsx`

```typescript
interface ChatInputProps {
  onSend: (message: string) => void;
  isDisabled?: boolean;
  placeholder?: string;
}
```

**Layout:**
```
┌──────────────────────────────────────┐
│ Type your message...            [↑] │
│ ↳ Press Enter to send, Shift+Enter  │
│   for new line                       │
└──────────────────────────────────────┘
```

**Features:**
- Auto-resize textarea (max 4 lines)
- Enter to send, Shift+Enter for newline
- Disable khi đang gửi
- Character counter (optional)

---

## 🎨 Visual Design Specs

### Color Palette (Dark Mode)

```css
--agent-widget-bg: #1a1d24;
--agent-header-bg: #242830;
--agent-status-bg: #2d3139;

--message-user-bg: #3b82f6;      /* Blue */
--message-agent-bg: #374151;     /* Gray */
--message-system-bg: #fbbf24;    /* Yellow */

--status-running: #10b981;       /* Green */
--status-degraded: #f59e0b;      /* Orange */
--status-stopped: #ef4444;       /* Red */

--text-primary: #f3f4f6;
--text-secondary: #9ca3af;
```

### Typography

```css
--font-message: 'Inter', -apple-system, sans-serif;
--font-mono: 'Fira Code', 'Courier New', monospace;

--text-sm: 0.875rem;    /* 14px */
--text-base: 1rem;      /* 16px */
--text-lg: 1.125rem;    /* 18px */
```

### Spacing

```css
--spacing-xs: 4px;
--spacing-sm: 8px;
--spacing-md: 16px;
--spacing-lg: 24px;
```

### Animations

```css
/* Slide in from bottom-right */
@keyframes slideInFromBottomRight {
  from {
    transform: translate(100%, 100%);
    opacity: 0;
  }
  to {
    transform: translate(0, 0);
    opacity: 1;
  }
}

/* Typing indicator */
@keyframes typingDot {
  0%, 60%, 100% { opacity: 0.3; }
  30% { opacity: 1; }
}

/* Tool execution pulse */
@keyframes executingPulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}
```

---

## 🔧 Implementation Plan

### Phase 1: Basic Chat UI (Day 1)

**Files to create:**
```
frontend-ts/src/components/agent/
├── AgentChatWidget.tsx          ← Main widget
├── MessageList.tsx              ← Message display
├── MessageBubble.tsx            ← Individual message
├── ChatInput.tsx                ← Input component
└── index.ts                     ← Exports
```

**Features:**
- ✅ Basic chat UI
- ✅ Message display (user/agent)
- ✅ Send message
- ✅ Minimize/maximize
- ✅ Draggable position

---

### Phase 2: Agent Integration (Day 2)

**Files to create:**
```
frontend-ts/src/services/
└── agentService.ts              ← API calls to backend

frontend-ts/src/types/
└── agent.ts                     ← TypeScript types
```

**Features:**
- ✅ API integration with backend
- ✅ Tool execution visualization
- ✅ Error handling
- ✅ Loading states

---

### Phase 3: Service Status (Day 3)

**Files to create:**
```
frontend-ts/src/components/agent/
├── ServiceStatusBar.tsx         ← Status indicator
└── QuickActions.tsx             ← Quick action buttons
```

**Features:**
- ✅ Real-time service status
- ✅ Quick actions
- ✅ Auto-refresh status

---

### Phase 4: Polish & UX (Day 4)

**Features:**
- ✅ Typing indicator
- ✅ Message timestamps
- ✅ Smooth animations
- ✅ Keyboard shortcuts
- ✅ Responsive design

---

## 📝 Integration Points

### 1. Add to Dashboard

**File:** `frontend-ts/src/components/dashboard/Dashboard.tsx`

```tsx
import AgentChatWidget from '../agent/AgentChatWidget';

function Dashboard() {
  return (
    <div className="dashboard">
      {/* Existing dashboard content */}

      {/* Add Agent Chat Widget */}
      <AgentChatWidget />
    </div>
  );
}
```

### 2. Create Agent Service

**File:** `frontend-ts/src/services/agentService.ts`

```typescript
import axios from 'axios';
import { API_BASE_URL } from '../config/api';

export const agentService = {
  async chat(message: string, agentId = 'service_management') {
    const response = await axios.post(
      `${API_BASE_URL}/agent/chat`,
      { message, agent_id: agentId },
      { headers: { Authorization: `Bearer ${getToken()}` } }
    );
    return response.data;
  },

  async getAgents() {
    const response = await axios.get(`${API_BASE_URL}/agent/agents`, {
      headers: { Authorization: `Bearer ${getToken()}` }
    });
    return response.data;
  },

  async getHealth() {
    const response = await axios.get(`${API_BASE_URL}/agent/health`);
    return response.data;
  }
};
```

### 3. TypeScript Types

**File:** `frontend-ts/src/types/agent.ts`

```typescript
export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  toolCalls?: ToolCall[];
  status?: 'pending' | 'success' | 'error';
}

export interface ToolCall {
  tool: string;
  args: Record<string, any>;
  result?: any;
}

export interface ChatResponse {
  response: string;
  agent_id: string;
  session_id: string;
  tool_calls?: ToolCall[];
  timestamp: string;
}
```

---

## 🎯 User Flows

### Flow 1: Check Service Status

```
1. User clicks widget (minimized) → Widget expands
2. Quick action: "Check Status" → Auto-fill input
3. User presses Enter
4. Show "Agent typing..." indicator
5. Agent calls check_service_status tool
6. Show tool execution: "🔍 Checking service status..."
7. Agent responds with formatted status
8. Update status bar if needed
```

### Flow 2: Start Service

```
1. User types "Start camera service"
2. Agent checks status first (automatic)
3. Agent asks confirmation: "Bạn có chắc muốn start service không?"
4. User confirms (can type "yes" or click suggested action)
5. Agent executes start_service tool
6. Show progress: "🚀 Starting service..."
7. Agent confirms success
8. Status bar updates to "Running"
```

### Flow 3: Troubleshooting

```
1. User: "Service không connect được"
2. Agent:
   - Checks service status
   - Reads logs automatically
   - Analyzes errors
3. Agent provides:
   - Current status
   - Error logs (formatted)
   - Possible causes
   - Suggested solutions
4. User can click quick actions to apply fixes
```

---

## 🚀 Next Steps

**Ready to implement?**

1. ✅ Design document complete
2. ⬜ Create basic components
3. ⬜ Integrate with backend API
4. ⬜ Add service status tracking
5. ⬜ Polish UX & animations

Bạn muốn tôi bắt đầu code ngay không? 🚀
