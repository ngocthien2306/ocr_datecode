import api from './http';
import type { User, UserCreate, UserUpdate, MessageResponse } from '@/types';

export const usersAPI = {
  getAllUsers: async (skip = 0, limit = 100): Promise<User[]> => {
    const response = await api.get<User[]>('/users/', { params: { skip, limit } });
    return response.data;
  },

  getCurrentUser: async (): Promise<User> => {
    const response = await api.get<User>('/users/me');
    return response.data;
  },

  updateCurrentUserProfile: async (userData: UserUpdate): Promise<User> => {
    const response = await api.put<User>('/users/me', userData);
    return response.data;
  },

  getUserById: async (userId: string): Promise<User> => {
    const response = await api.get<User>(`/users/${userId}`);
    return response.data;
  },

  getSimpleUsers: async (skip = 0, limit = 200): Promise<{id: string; username: string; full_name?: string; avatar_url?: string | null;}[]> => {
    const response = await api.get<{id: string; username: string; full_name?: string; avatar_url?: string | null;}[]>('/users/simple', { params: { skip, limit } });
    return response.data;
  },

  createUser: async (userData: UserCreate): Promise<User> => {
    const response = await api.post<User>('/users/', userData);
    return response.data;
  },

  updateUser: async (userId: string, userData: UserUpdate): Promise<User> => {
    const response = await api.put<User>(`/users/${userId}`, userData);
    return response.data;
  },

  deleteUser: async (userId: string): Promise<MessageResponse> => {
    const response = await api.delete<MessageResponse>(`/users/${userId}`);
    return response.data;
  },

  changePassword: async (oldPassword: string, newPassword: string): Promise<MessageResponse> => {
    const response = await api.post<MessageResponse>('/users/change-password', {
      old_password: oldPassword,
      new_password: newPassword,
    });
    return response.data;
  },

  resetPassword: async (userId: string, newPassword: string): Promise<MessageResponse> => {
    const response = await api.post<MessageResponse>(`/users/${userId}/reset-password`, null, {
      params: { new_password: newPassword },
    });
    return response.data;
  },
};

export default usersAPI;
