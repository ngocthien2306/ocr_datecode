import api from './http';
import type { Camera, CameraCreate, CameraUpdate } from '@/types';

export const camerasAPI = {
  getAllCameras: async (skip = 0, limit = 100): Promise<Camera[]> => {
    const response = await api.get<Camera[]>(`/cameras?skip=${skip}&limit=${limit}`);
    return response.data;
  },

  getConnectedCameras: async (): Promise<Camera[]> => {
    const response = await api.get<Camera[]>('/cameras/connected');
    return response.data;
  },

  getCameraById: async (cameraId: string): Promise<Camera> => {
    const response = await api.get<Camera>(`/cameras/${cameraId}`);
    return response.data;
  },

  getCamerasCount: async (): Promise<{ count: number }> => {
    const response = await api.get<{ count: number }>('/cameras/count/total');
    return response.data;
  },

  createCamera: async (cameraData: CameraCreate): Promise<Camera> => {
    const response = await api.post<Camera>('/cameras', cameraData);
    return response.data;
  },

  updateCamera: async (cameraId: string, cameraData: CameraUpdate): Promise<Camera> => {
    const response = await api.put<Camera>(`/cameras/${cameraId}`, cameraData);
    return response.data;
  },

  updateCameraConnection: async (cameraId: string, isConnected: boolean): Promise<Camera> => {
    const response = await api.patch<Camera>(`/cameras/${cameraId}/connection`, {
      is_connected: isConnected,
    });
    return response.data;
  },

  deleteCamera: async (cameraId: string): Promise<{ message?: string }> => {
    const response = await api.delete(`/cameras/${cameraId}`);
    return response.data;
  },
};

export default camerasAPI;
