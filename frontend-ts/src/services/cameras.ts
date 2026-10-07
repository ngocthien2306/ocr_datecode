import api from './http';
import type { Camera, CameraCreate, CameraUpdate, DiscoveredCamera } from '@/types';

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

  getCameraFrame: (serialNumber: string, quality: number = 85, saveToDisk: boolean = false): string => {
    const token = localStorage.getItem('access_token');
    return `${api.defaults.baseURL}/cameras/${serialNumber}/frame?quality=${quality}&save_to_disk=${saveToDisk}&token=${token}`;
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

  discoverCameras: async (): Promise<DiscoveredCamera[]> => {
    const response = await api.get<DiscoveredCamera[]>('/cameras/discover');
    return response.data;
  },

  getLatestFrames: async (serialNumber: string, count: number = 5, quality: number = 85): Promise<any> => {
    const response = await api.get(`/cameras/${serialNumber}/frames/latest`, {
      params: { count, quality }
    });
    return response.data;
  },

  rotateFrames: async (frames: { frame_base64: string; metadata: any }[], quality: number = 90): Promise<any> => {
    const response = await api.post('/cameras/frames/rotate', { frames, quality });
    return response.data;
  },

  /**
   * Re-derive a template's regions on a freshly captured frame, server-side
   * (backend, CPU): cap rotation → cap crop → text window → SuperPoint, the
   * same steps the AI service runs. Returns the rotated frame(s) with
   * `detected_regions` computed on them, so image and regions always belong
   * to the same frame. See backend/app/api/endpoints/live_template.py.
   */
  realignLiveFrame: async (body: {
    recipe_id: string;
    serial_number: string;
    frame_base64: string;
    template_name?: string;
    template_index?: number;
    quality?: number;
  }): Promise<LiveRealignResult> => {
    const response = await api.post<LiveRealignResult>('/live-template/realign', body, {
      // ~1-2 s of CPU per call; leave headroom for a queued second request.
      timeout: 30000,
    });
    return response.data;
  },
};

export interface LiveRealignCandidate {
  label: 'no_flip' | 'flip180';
  is_alt: boolean;
  chosen: boolean;
  success: boolean;
  error: string | null;
  confidence: number;
  inliers: number;
  total_matches: number;
  /** Rotation of the template→frame transform; the upright candidate is ≈ 0°. */
  rotation_deg: number | null;
  /** Regions copied from the other candidate through the cap centre (it did not match on its own). */
  mirrored: boolean;
  /** Same shape as an inference frame's detected_regions, in THIS candidate's frame. */
  detected_regions: any[];
  frame_base64: string;
  width: number;
  height: number;
}

export interface LiveRealignResult {
  mode: 'cap_rotation' | 'plain' | 'not_applicable';
  reason?: string;
  rotation_mode?: string;
  transform_model?: string;
  candidates: LiveRealignCandidate[];
  /** Index into candidates, or null when nothing matched. */
  chosen: number | null;
  notes: string[];
  elapsed_ms: number;
}

export default camerasAPI;
