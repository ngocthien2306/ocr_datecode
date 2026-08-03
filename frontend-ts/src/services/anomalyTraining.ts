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

// ──────── Dataset gallery types ────────────────────────────────────────

export interface DatasetImage {
  id: string;
  inspection_id: string;
  camera_serial: string;
  frame_idx: number;
  recipe_name?: string;
  label: 'normal' | 'abnormal';
  defect_type: string | null;
  split: 'train' | 'test';
  created_at: string;
  thumb_b64: string;
}

export interface DatasetImagesPage {
  images: DatasetImage[];
  count: number;
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface BulkActionResult {
  errors: Array<{ id: string; reason: string }>;
  normal: number;
  abnormal: number;
}

// ──────── Train / Eval / Export types ────────────────────────────────────

export type AnomalyAlgorithm = 'patchcore' | 'padim';

export interface AnomalyTrainRequest {
  algorithm: AnomalyAlgorithm;
  backbone: string;
  layers: string[];
  coreset_sampling_ratio: number;
  image_size: number;
  test_split: number;
}

export interface AnomalyModelMetrics {
  // null when the test set has no abnormal images: AUROC/F1 are undefined
  // there, not zero. Render with fmtMetric(), never `?? 0`.
  image_auroc: number | null;
  image_f1: number | null;
  metrics_available: boolean;
  threshold: number;
  n_normal_train: number;
  n_normal_test: number;
  n_abnormal_test: number;
}

/** Format an AUROC/F1 value, distinguishing "not measurable" from 0.000.
 *  A single-class test set (no abnormal images imported yet) makes both
 *  metrics undefined; showing 0.000 there reads as a broken model. */
export function fmtMetric(v: number | null | undefined): string {
  return v == null ? 'N/A' : v.toFixed(3);
}

export interface AnomalyModel {
  id: string;
  project_id: string;
  algorithm: AnomalyAlgorithm;
  params: Record<string, any>;
  metrics: AnomalyModelMetrics;
  checkpoint_path: string;
  onnx_path: string | null;
  engine_path: string | null;
  status: 'pending' | 'training' | 'completed' | 'failed' | 'cancelled';
  error: string | null;
  created_at: string;
  phase: string | null;
  progress: number;
}

export interface TrainLogEntry {
  idx: number;
  ts: number;
  level: string;
  msg: string;
}

export interface TrainLogsResponse {
  logs: TrainLogEntry[];
  next_since: number;
  phase: string | null;
  progress: number;
  status: string;
  error: string | null;
}

export interface TestResultItem {
  image_path: string;
  crop_b64: string;
  // Absent on models evaluated before heatmap export was added — re-train
  // to get one.
  heatmap_b64?: string;
  pred_score: number;
  gt_label: 'normal' | 'abnormal';
  pred_label: 'normal' | 'abnormal';
  correct: boolean;
}

export interface TestResultsResponse {
  image_auroc: number | null;
  image_f1: number | null;
  metrics_available: boolean;
  threshold: number;
  confusion_matrix: number[][]; // [[TN, FP], [FN, TP]]
  n_normal_test: number;
  n_abnormal_test: number;
  items: TestResultItem[];
}

export type InferenceEngine = 'onnx' | 'tensorrt';

export interface PredictResult {
  pred_score: number;
  crop_b64: string;
  heatmap_b64: string;
  image_size: number;
  engine: InferenceEngine;
  active_provider: string;
  inference_ms: number;
}

export interface VerifyTensorRTResult {
  active_provider: string;
  engine_cache_hit: boolean;
  build_or_load_ms: number;
  // null: build+deserialize only, no forward pass run (no pycuda in this
  // service's env) -- still a real build+load on this machine's GPU.
  inference_ms: number | null;
  output_shapes: number[][];
  cache_dir: string;
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

  // Dataset gallery
  listDatasetImages: async (
    projectId: string,
    opts: { label?: 'normal' | 'abnormal'; page?: number; pageSize?: number } = {},
  ): Promise<DatasetImagesPage> => {
    const res = await anomalyApi.get(`/projects/${projectId}/dataset/images`, {
      params: { label: opts.label, page: opts.page, page_size: opts.pageSize },
    });
    return res.data;
  },
  getDatasetImageFull: async (projectId: string, imageId: string): Promise<{ id: string; full_b64: string }> => {
    const res = await anomalyApi.get(`/projects/${projectId}/dataset/images/${imageId}/full`);
    return res.data;
  },
  relabelDatasetImage: async (
    projectId: string,
    imageId: string,
    label: 'normal' | 'abnormal',
    defectType?: string,
  ): Promise<{ ok: boolean; normal: number; abnormal: number }> => {
    const res = await anomalyApi.patch(`/projects/${projectId}/dataset/images/${imageId}`, {
      label, defect_type: defectType,
    });
    return res.data;
  },
  deleteDatasetImage: async (projectId: string, imageId: string): Promise<{ ok: boolean; normal: number; abnormal: number }> => {
    const res = await anomalyApi.delete(`/projects/${projectId}/dataset/images/${imageId}`);
    return res.data;
  },
  bulkRelabelDatasetImages: async (
    projectId: string,
    ids: string[],
    label: 'normal' | 'abnormal',
    defectType?: string,
  ): Promise<{ updated: number } & BulkActionResult> => {
    const res = await anomalyApi.post(`/projects/${projectId}/dataset/images/bulk-relabel`, {
      ids, label, defect_type: defectType,
    });
    return res.data;
  },
  bulkDeleteDatasetImages: async (
    projectId: string,
    ids: string[],
  ): Promise<{ deleted: number } & BulkActionResult> => {
    const res = await anomalyApi.post(`/projects/${projectId}/dataset/images/bulk-delete`, { ids });
    return res.data;
  },

  // Training
  startTraining: async (projectId: string, request: AnomalyTrainRequest): Promise<{ model_id: string; status: string }> => {
    const res = await anomalyApi.post(`/projects/${projectId}/train`, request);
    return res.data;
  },
  listModels: async (projectId: string): Promise<AnomalyModel[]> => {
    const res = await anomalyApi.get(`/projects/${projectId}/models`);
    return res.data;
  },
  getModelStatus: async (projectId: string, modelId: string): Promise<AnomalyModel> => {
    const res = await anomalyApi.get(`/projects/${projectId}/models/${modelId}/status`);
    return res.data;
  },
  getTrainLogs: async (projectId: string, modelId: string, since = 0): Promise<TrainLogsResponse> => {
    const res = await anomalyApi.get(`/projects/${projectId}/models/${modelId}/logs`, { params: { since } });
    return res.data;
  },
  cancelTraining: async (projectId: string, modelId: string): Promise<{ ok: boolean; mode: string }> => {
    const res = await anomalyApi.post(`/projects/${projectId}/models/${modelId}/cancel`);
    return res.data;
  },
  deleteModel: async (projectId: string, modelId: string): Promise<{ ok: boolean }> => {
    const res = await anomalyApi.delete(`/projects/${projectId}/models/${modelId}`);
    return res.data;
  },

  // Eval
  getTestResults: async (projectId: string, modelId: string, threshold = 0.5): Promise<TestResultsResponse> => {
    const res = await anomalyApi.get(`/projects/${projectId}/models/${modelId}/test-results`, { params: { threshold } });
    return res.data;
  },

  // Test — live inference on a freshly uploaded image via the exported
  // ONNX or standalone TensorRT engine (distinct from getTestResults
  // above, which only re-reads the stored training-time test-set
  // predictions). Sessions/engines are cached server-side per model, so
  // inference_ms reflects steady-state speed, not load time.
  predictImage: async (
    projectId: string, modelId: string, file: File, engine: InferenceEngine = 'onnx',
  ): Promise<PredictResult> => {
    const form = new FormData();
    form.append('file', file);
    const res = await anomalyApi.post(`/projects/${projectId}/models/${modelId}/predict`, form, {
      headers: { 'Content-Type': 'multipart/form-data' },
      params: { engine },
    });
    return res.data;
  },

  // Export
  exportOnnx: async (projectId: string, modelId: string): Promise<{ onnx_path: string; image_size: number }> => {
    const res = await anomalyApi.post(`/projects/${projectId}/models/${modelId}/export-onnx`);
    return res.data;
  },
  // Builds (or reuses) the same standalone .engine as exportTensorRT below
  // and confirms it deserializes correctly on this machine's GPU.
  verifyTensorRT: async (projectId: string, modelId: string): Promise<VerifyTensorRTResult> => {
    const res = await anomalyApi.post(`/projects/${projectId}/models/${modelId}/verify-tensorrt`);
    return res.data;
  },
  // Build a standalone, downloadable .engine (dynamic batch 1..maxBatch,
  // fixed HxW = training image_size).
  exportTensorRT: async (
    projectId: string,
    modelId: string,
    maxBatch = 8,
  ): Promise<{ engine_path: string; input_name: string; min_shape: number[]; opt_shape: number[]; max_shape: number[] }> => {
    const res = await anomalyApi.post(`/projects/${projectId}/models/${modelId}/export-tensorrt`, null, {
      params: { max_batch: maxBatch },
    });
    return res.data;
  },
  // Blob download (not a plain <a href> URL) — the export endpoint requires
  // the same Bearer auth as every other call here, and anchor-tag
  // navigation can't set custom headers, so we fetch + save client-side.
  downloadOnnx: async (projectId: string, modelId: string): Promise<void> => {
    const res = await anomalyApi.get(`/projects/${projectId}/models/${modelId}/export/onnx`, {
      responseType: 'blob',
    });
    const url = window.URL.createObjectURL(new Blob([res.data]));
    const a = document.createElement('a');
    a.href = url;
    a.download = `anomaly_${modelId}.onnx`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  },
  downloadEngine: async (projectId: string, modelId: string): Promise<void> => {
    const res = await anomalyApi.get(`/projects/${projectId}/models/${modelId}/export/tensorrt`, {
      responseType: 'blob',
    });
    const url = window.URL.createObjectURL(new Blob([res.data]));
    const a = document.createElement('a');
    a.href = url;
    a.download = `anomaly_${modelId}.engine`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  },
};

export default anomalyApi;
