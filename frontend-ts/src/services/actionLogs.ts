import api from './http';
import type { ActionLog, ActionLogFilters } from '@/types';

export const actionLogsAPI = {
  getActionLogs: async (
    skip = 0,
    limit = 100,
    filters?: ActionLogFilters
  ): Promise<ActionLog[]> => {
    const params: any = { skip, limit };

    if (filters) {
      if (filters.user_id) params.user_id = filters.user_id;
      if ((filters as any).username) params.username = (filters as any).username;
      if (filters.action_type) params.action_type = filters.action_type;
      if (filters.resource_type) params.resource_type = filters.resource_type;
      if (filters.start_date) params.start_date = filters.start_date;
      if (filters.end_date) params.end_date = filters.end_date;
    }

    const response = await api.get<ActionLog[]>('/action-logs/', { params });
    return response.data;
  },

  getRecentActionLogs: async (limit = 20): Promise<ActionLog[]> => {
    const response = await api.get<ActionLog[]>('/action-logs/recent', {
      params: { limit }
    });
    return response.data;
  },

  getUserActionLogs: async (
    userId: string,
    skip = 0,
    limit = 50
  ): Promise<ActionLog[]> => {
    const response = await api.get<ActionLog[]>(`/action-logs/user/${userId}`, {
      params: { skip, limit }
    });
    return response.data;
  },

  getActionLogById: async (actionLogId: string): Promise<ActionLog> => {
    const response = await api.get<ActionLog>(`/action-logs/${actionLogId}`);
    return response.data;
  }
};