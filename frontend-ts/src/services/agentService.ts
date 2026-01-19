/**
 * AI Agent Service
 * API calls to backend agent endpoints
 */

import api from './http'; // Use existing axios instance with auth interceptors
import { ChatRequest, ChatResponse, AgentInfo, ServiceStatus } from '../types/agent';

export const agentService = {
  /**
   * Send chat message to agent
   */
  async chat(request: ChatRequest): Promise<ChatResponse> {
    try {
      const response = await api.post<ChatResponse>(
        '/agent/chat',
        {
          message: request.message,
          agent_id: request.agent_id || 'service_management',
          session_id: request.session_id,
          stream: request.stream || false
        }
      );

      return response.data;
    } catch (error: any) {
      console.error('Agent chat error:', error);
      throw new Error(error.response?.data?.detail || 'Failed to send message to agent');
    }
  },

  /**
   * Get list of available agents
   */
  async getAgents(): Promise<AgentInfo[]> {
    try {
      const response = await api.get<AgentInfo[]>('/agent/agents');

      return response.data;
    } catch (error: any) {
      console.error('Get agents error:', error);
      throw new Error(error.response?.data?.detail || 'Failed to fetch agents');
    }
  },

  /**
   * Check agent system health
   */
  async getHealth(): Promise<{ status: string; agents_registered: number; openai_configured: boolean }> {
    try {
      const response = await api.get('/agent/health');

      return response.data;
    } catch (error: any) {
      console.error('Agent health check error:', error);
      return {
        status: 'unhealthy',
        agents_registered: 0,
        openai_configured: false
      };
    }
  },

  /**
   * Check service status (dedicated endpoint)
   * This calls a dedicated backend endpoint that directly checks service status
   * without using the LLM/chat functionality
   */
  async checkServiceStatus(serviceName: string = 'camera_management'): Promise<ServiceStatus> {
    try {
      const response = await api.get<ServiceStatus>(
        '/agent/service/status',
        {
          params: { service_name: serviceName }
        }
      );

      return response.data;
    } catch (error: any) {
      console.error('Service status check error:', error);
      return {
        service_name: serviceName,
        is_running: false,
        websocket_connected: false,
        status: 'stopped',
        message: 'Failed to check status'
      };
    }
  },

  /**
   * Clear conversation history
   * Deletes all messages for a given session
   */
  async clearConversation(sessionId: string): Promise<{ success: boolean; message: string }> {
    try {
      const response = await api.delete(`/agent/conversation/${sessionId}`);
      return response.data;
    } catch (error: any) {
      console.error('Clear conversation error:', error);
      throw new Error(error.response?.data?.detail || 'Failed to clear conversation');
    }
  }
};
