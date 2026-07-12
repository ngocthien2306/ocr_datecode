import axios, { AxiosInstance, InternalAxiosRequestConfig } from 'axios';

// anomaly_service runs as its own FastAPI process (own port) on the same
// GPU workstation as backend — separate service, separate base URL, but
// reuses the same JWT (see anomaly_service/app/api/dependencies/auth.py).
export const ANOMALY_API_BASE_URL = 'http://localhost:8001/api/anomaly';

const anomalyApi: AxiosInstance = axios.create({
  baseURL: ANOMALY_API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
});

anomalyApi.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error),
);

anomalyApi.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token');
      localStorage.removeItem('user');
      window.location.href = '/';
    }
    return Promise.reject(error);
  },
);

// ──────── Types ─────────────────────────────────────────────────────────

export interface AnomalyProject {
  id: string;
  name: string;
  description?: string;
  created_at: string;
  updated_at: string;
  created_by?: string;
  normal_count: number;
  abnormal_count: number;
  status: 'active' | 'training' | 'trained';
}

export interface DatasetStats {
  normal_count: number;
  abnormal_count: number;
  defect_types: string[];
}

export interface AnomalyCandidate {
  inspection_id: string;
  recipe_id: string;
  recipe_name: string;
  camera_serial: string;
  frame_idx: number;
  product_pass_fail?: string;
  timestamp: string | null;
  crop_b64: string;
  imported_split: string | null; // "train" | "test" | null
}

export interface AnomalyImportSelection {
  inspection_id: string;
  camera_serial: string;
  frame_idx: number;
  label: 'normal' | 'abnormal';
  defect_type?: string;
}

export interface AnomalyImportResult {
  imported: number;
  skipped: number;
  errors: Array<{ inspection_id: string; reason: string }>;
  normal_count: number;
  abnormal_count: number;
}

// ──────── API ───────────────────────────────────────────────────────────

export const anomalyTrainingAPI = {
  // Projects
  listProjects: async (): Promise<AnomalyProject[]> => {
    const res = await anomalyApi.get('/projects');
    return res.data;
  },
  createProject: async (name: string, description?: string): Promise<AnomalyProject> => {
    const res = await anomalyApi.post('/projects', { name, description });
    return res.data;
  },
  getProject: async (id: string): Promise<AnomalyProject> => {
    const res = await anomalyApi.get(`/projects/${id}`);
    return res.data;
  },
  updateProject: async (id: string, data: { name?: string; description?: string }): Promise<AnomalyProject> => {
    const res = await anomalyApi.patch(`/projects/${id}`, data);
    return res.data;
  },
  deleteProject: async (id: string): Promise<void> => {
    await anomalyApi.delete(`/projects/${id}`);
  },
  datasetStats: async (id: string): Promise<DatasetStats> => {
    const res = await anomalyApi.get(`/projects/${id}/dataset-stats`);
    return res.data;
  },

  // Candidates / import
  getCandidates: async (params: {
    project_id: string;
    recipe_id?: string;
    date_from?: string;
    date_to?: string;
    limit?: number;
  }): Promise<{ candidates: AnomalyCandidate[]; count: number }> => {
    const res = await anomalyApi.get('/candidates', { params });
    return res.data;
  },
  importCandidates: async (
    projectId: string,
    selections: AnomalyImportSelection[],
  ): Promise<AnomalyImportResult> => {
    const res = await anomalyApi.post(`/projects/${projectId}/import`, { selections });
    return res.data;
  },
};

export default anomalyApi;
