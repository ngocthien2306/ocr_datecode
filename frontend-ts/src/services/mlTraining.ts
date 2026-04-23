import api, { API_BASE_URL } from './http';

// Server origin (e.g. "http://localhost:8000") — used to build static image URLs
const SERVER_ORIGIN = API_BASE_URL.replace(/\/api\/?$/, '');

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
  url: string;
}

export interface ProjectImage {
  filename: string;
  url: string;
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

export interface SyntheticCrop {
  source_segment_id: string;
  filename: string;
  label: 'NG';
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

export interface TestSetImageResult {
  filename: string;
  predictions: PredictResult[];
  ok_count: number;
  ng_count: number;
}

// ──────── Static URL builders ──────────────────────────────────────────────

/** Full URL to a camera snapshot image (served from /public/images_temp) */
export const cameraImageUrl = (filename: string) =>
  `${SERVER_ORIGIN}/api/camera-images/${encodeURIComponent(filename)}`;

/** Full URL to a project image (served from /public/ml_projects/{id}/images) */
export const projectImageUrl = (projectId: string, filename: string) =>
  `${SERVER_ORIGIN}/api/ml-files/${projectId}/images/${encodeURIComponent(filename)}`;

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

  // Snapshot camera buffer → stable temp folder for current session
  snapshotImages: () =>
    api.post<{ copied: number; filenames: string[] }>('/ml/snapshot-images').then(r => r.data),

  // Available images (from /public/images_temp snapshot)
  listAvailableImages: () =>
    api.get<{ filename: string }[]>('/ml/available-images').then(r =>
      r.data.map(img => ({ filename: img.filename, url: cameraImageUrl(img.filename) }))
    ),

  // Project images
  listProjectImages: (projectId: string) =>
    api.get<{ filename: string; has_annotation: boolean }[]>(`/ml/projects/${projectId}/images`).then(r =>
      r.data.map(img => ({ ...img, url: projectImageUrl(projectId, img.filename) }))
    ),

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

  getImageMeta: (projectId: string, filename: string) =>
    api.get<{ filename: string; width: number; height: number }>(
      `/ml/projects/${projectId}/images/${encodeURIComponent(filename)}/meta`
    ).then(r => ({ ...r.data, url: projectImageUrl(projectId, filename) })),

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

  previewSynthetic: (projectId: string, augmentFactor: number) =>
    api.post<{ crops: SyntheticCrop[]; count: number }>(
      `/ml/projects/${projectId}/preview-synthetic`,
      { augment_factor: augmentFactor }
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
  predict: (projectId: string, file: File, modelId?: string) => {
    const form = new FormData();
    form.append('file', file);
    if (modelId) form.append('model_id', modelId);
    return api.post<{ model_id: string; algorithm: string; results: PredictResult[] }>(
      `/ml/projects/${projectId}/predict`, form,
      { headers: { 'Content-Type': 'multipart/form-data' } }
    ).then(r => r.data);
  },

  testSet: (projectId: string, modelId: string) =>
    api.post<{ results: TestSetImageResult[]; model_id: string; image_count: number }>(
      `/ml/projects/${projectId}/test-set`,
      { model_id: modelId }
    ).then(r => r.data),
};
