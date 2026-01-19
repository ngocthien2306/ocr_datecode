import api from './http';
import type { LoginResponse } from '@/types';

export const authAPI = {
  login: async (username: string, password: string): Promise<LoginResponse> => {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    const response = await api.post<LoginResponse>('/auth/login', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Bypass-Tunnel-Reminder': 'true',
      },
    });
    return response.data;
  },
};

export default authAPI;
