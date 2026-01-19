/**
 * AI Agent Types
 */

export interface SuggestedAction {
  label: string;
  message: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  toolCalls?: ToolCall[];
  status?: 'pending' | 'success' | 'error';
  suggestedActions?: SuggestedAction[];
}

export interface ToolCall {
  tool: string;
  args: Record<string, any>;
  id?: string;
}

export interface ChatRequest {
  message: string;
  agent_id?: string;
  session_id?: string;
  stream?: boolean;
}

export interface ChatResponse {
  response: string;
  agent_id: string;
  session_id: string;
  tool_calls?: ToolCall[];
  timestamp: string;
}

export interface AgentInfo {
  agent_id: string;
  class_name: string;
  description: string;
}

export interface ServiceStatus {
  service_name: string;
  is_running: boolean;
  websocket_connected: boolean;
  status: 'healthy' | 'degraded' | 'stopped';
  message: string;
  pid?: number;
  cpu_percent?: number;
  memory_mb?: number;
}

export interface QuickAction {
  label: string;
  icon: string;
  message: string;
  condition?: () => boolean;
}
