import api from './http';

// ──────── Types ────────────────────────────────────────────────────────────

export interface MLProject {
  id: string;
  name: string;
  description?: string;
  created_at: string;
  updated_at: string;
  created_by?: string;
  image_count: number;
  labeled_count: number;
  status: 'active' | 'training' | 'trained';
}

export interface AvailableImage {
  filename: string;
  thumbnail_b64: string;
}

export interface ProjectImage {
  filename: string;
  thumbnail_b64: string;
  has_annotation: boolean;
}

export interface CharSegment {
  id: string;
  x: number; y: number; w: number; h: number;
  label?: 'OK' | 'NG' | null;
}

export interface AnnotationRegion {
  id: string;
  x: number; y: number; w: number; h: number;
  segments: CharSegment[];
}

export interface AnnotationData {
  project_id: string;
  filename: string;
  regions: AnnotationRegion[];
}

export interface LabeledCrop {
  segment_id: string;
  region_id: string;
  filename: string;
  label: 'OK' | 'NG';
  crop_b64: string;
}

export interface TrainRequest {
  algorithm: 'rf' | 'svm' | 'mlp';
  augment_factor: number;
  test_split?: number;
  n_estimators?: number;
  max_iter?: number;
  C?: number;
  hidden_layer_sizes?: number[];
}

export interface MLModelMetrics {
  accuracy_train: number;
  accuracy_test: number;
  n_ok: number;
  n_ng: number;
  n_total: number;
  confusion_matrix: number[][];
  report: string;
}

export interface MLModel {
  id: string;
  project_id: string;
  algorithm: string;
  params: Record<string, unknown>;
  augment_factor: number;
  metrics: MLModelMetrics;
  model_path: string;
  status: 'pending' | 'training' | 'completed' | 'failed';
  error?: string;
  created_at: string;
}

export interface PredictResult {
  id: string;
  x: number; y: number; w: number; h: number;
  prob_ok: number;
  label: 'OK' | 'NG';
  crop_b64: string;
}

// ──────── API client ───────────────────────────────────────────────────────

export const mlTrainingAPI = {
  // Projects
  createProject: (name: string, description?: string) =>
    api.post<MLProject>('/ml/projects', { name, description }).then(r => r.data),

  listProjects: () =>
    api.get<MLProject[]>('/ml/projects').then(r => r.data),

  getProject: (id: string) =>
    api.get<MLProject>(`/ml/projects/${id}`).then(r => r.data),

  updateProject: (id: string, name: string, description?: string) =>
    api.patch<MLProject>(`/ml/projects/${id}`, { name, description }).then(r => r.data),

  deleteProject: (id: string) =>
    api.delete(`/ml/projects/${id}`).then(r => r.data),

  // Available images (from /public/images buffer)
  listAvailableImages: () =>
    api.get<AvailableImage[]>('/ml/available-images').then(r => r.data),

  // Project images
  listProjectImages: (projectId: string) =>
    api.get<ProjectImage[]>(`/ml/projects/${projectId}/images`).then(r => r.data),

  copyImages: (projectId: string, filenames: string[]) =>
    api.post(`/ml/projects/${projectId}/images/copy`, { filenames }).then(r => r.data),

  uploadImages: (projectId: string, files: File[]) => {
    const form = new FormData();
    files.forEach(f => form.append('files', f));
    return api.post(`/ml/projects/${projectId}/images/upload`, form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then(r => r.data);
  },

  deleteImage: (projectId: string, filename: string) =>
    api.delete(`/ml/projects/${projectId}/images/${encodeURIComponent(filename)}`).then(r => r.data),

  getImageB64: (projectId: string, filename: string) =>
    api.get<{ filename: string; image_b64: string; width: number; height: number }>(
      `/ml/projects/${projectId}/images/${encodeURIComponent(filename)}`
    ).then(r => r.data),

  // Segmentation
  segmentRegion: (projectId: string, filename: string, region: { x: number; y: number; w: number; h: number }) =>
    api.post<{ segments: CharSegment[]; count: number }>(
      `/ml/projects/${projectId}/segment`,
      { filename, region }
    ).then(r => r.data),

  // Annotations
  getAnnotation: (projectId: string, filename: string) =>
    api.get<AnnotationData>(
      `/ml/projects/${projectId}/annotations/${encodeURIComponent(filename)}`
    ).then(r => r.data),

  saveAnnotation: (projectId: string, filename: string, regions: AnnotationRegion[]) =>
    api.put<AnnotationData>(
      `/ml/projects/${projectId}/annotations/${encodeURIComponent(filename)}`,
      { regions }
    ).then(r => r.data),

  // Labeled crops (Train tab)
  getLabeledCrops: (projectId: string) =>
    api.get<{ crops: LabeledCrop[]; count: number }>(
      `/ml/projects/${projectId}/labeled-crops`
    ).then(r => r.data),

  // Training
  startTraining: (projectId: string, req: TrainRequest) =>
    api.post<{ model_id: string; status: string }>(
      `/ml/projects/${projectId}/train`, req
    ).then(r => r.data),

  listModels: (projectId: string) =>
    api.get<MLModel[]>(`/ml/projects/${projectId}/models`).then(r => r.data),

  getModelStatus: (projectId: string, modelId: string) =>
    api.get<MLModel>(`/ml/projects/${projectId}/models/${modelId}/status`).then(r => r.data),

  // Prediction
  predict: (projectId: string, file: File) => {
    const form = new FormData();
    form.append('file', file);
    return api.post<{ model_id: string; algorithm: string; results: PredictResult[] }>(
      `/ml/projects/${projectId}/predict`, form,
      { headers: { 'Content-Type': 'multipart/form-data' } }
    ).then(r => r.data);
  },
};
