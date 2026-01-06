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

  connectCamera: async (serialNumber: string, pixelFormat: string = 'Mono8'): Promise<any> => {
    const response = await api.post(`/cameras/${serialNumber}/connect?pixel_format=${pixelFormat}`);
    return response.data;
  },

  disconnectCamera: async (serialNumber: string): Promise<any> => {
    const response = await api.post(`/cameras/${serialNumber}/disconnect`);
    return response.data;
  },

  getCameraFrame: (serialNumber: string, quality: number = 85): string => {
    const token = localStorage.getItem('access_token');
    return `${api.defaults.baseURL}/cameras/${serialNumber}/frame?quality=${quality}&token=${token}`;
  },

  getCameraStatus: async (serialNumber: string): Promise<any> => {
    const response = await api.get(`/cameras/${serialNumber}/status`);
    return response.data;
  },

  updateCameraSettings: async (serialNumber: string, settings: { exposure_time?: number; gain?: number; delay_trigger?: number }): Promise<any> => {
    const response = await api.post(`/cameras/${serialNumber}/settings`, settings);
    return response.data;
  },

  updateDelayTrigger: async (serialNumber: string, delayMs: number): Promise<any> => {
    const response = await api.post(`/cameras/${serialNumber}/settings`, {
      delay_trigger: delayMs
    });
    return response.data;
  },

  getCameraSettings: async (serialNumber: string): Promise<any> => {
    const response = await api.get(`/cameras/${serialNumber}/settings`);
    return response.data;
  },
};

export default camerasAPI;
