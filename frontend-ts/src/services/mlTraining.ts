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
  char_id?: string | null;   // auto-populated from recipe expected_text or manually set
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
  char_id?: string | null;
}

export interface SyntheticCrop {
  source_segment_id: string;
  filename: string;
  label: 'OK' | 'NG';
  crop_b64: string;
  char_id?: string | null;
}

export type MLAlgorithm = 'rf' | 'svm' | 'mlp' | 'golden_dist' | 'anomaly';

export interface TrainRequest {
  algorithm: MLAlgorithm;
  augment_factor: number;
  threshold?: number;         // prob_ok >= threshold → OK (rf/svm/mlp)
  test_split?: number;
  n_estimators?: number;      // rf, anomaly (IsolationForest)
  max_iter?: number;
  C?: number;
  hidden_layer_sizes?: number[];
  threshold_k?: number;       // golden_dist: mean + k*std
  contamination?: number;     // anomaly: IsolationForest contamination
}

export interface MLModelMetrics {
  accuracy_train: number;
  accuracy_test: number;
  n_ok: number;
  n_ng: number;
  n_total: number;
  confusion_matrix: number[][];
  report: string;
  golden_chars?: string[];   // chars with golden templates (v2 models only)
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
  char_id?: string | null;
  aligned_b64?: string | null;   // input after alignment to golden (v2 + golden)
  golden_b64?: string | null;    // reference golden for this char
  diff_b64?: string | null;      // JET-colormap diff heatmap
}

export interface ImportFromRecipeRequest {
  recipe_id: string;
  camera_serial: string;
  filenames: string[];
}

export interface ImportFromRecipeResponse {
  imported: number;
  skipped: number;
  errors: Array<{ filename: string; reason: string }>;
  char_ids: string[];
}

export interface CharCoverageResponse {
  covered: string[];
  missing: string[];
  coverage_pct: number;
  model_chars: string[];
}

export interface TestSetCropResult {
  crop_b64: string;
  true_label: 'OK' | 'NG';
  pred_label: 'OK' | 'NG';
  prob_ok: number;
  correct: boolean;
  char_id?: string | null;
}

export interface MLGoldenItem {
  char_id: string;
  golden_b64: string;   // base64 PNG of golden (upscaled)
  n_ok_train: number;   // total OK used (real + augmented)
  n_ng_train: number;
  n_ok_real: number;    // real samples (before augmentation)
  n_ng_real: number;
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

  previewSynthetic: (projectId: string, augmentFactor: number, label: 'NG' | 'OK' | 'BOTH' = 'NG') =>
    api.post<{ crops: SyntheticCrop[]; count: number }>(
      `/ml/projects/${projectId}/preview-synthetic`,
      { augment_factor: augmentFactor, label }
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

  getTestSetCrops: (projectId: string, modelId: string) =>
    api.get<{ crops: TestSetCropResult[]; count: number }>(
      `/ml/projects/${projectId}/models/${modelId}/test-set-crops`
    ).then(r => r.data),

  // Import recipe template bboxes into project (auto-populate char_id)
  importFromRecipe: (projectId: string, body: ImportFromRecipeRequest) =>
    api.post<ImportFromRecipeResponse>(
      `/ml/projects/${projectId}/import-from-recipe`, body,
    ).then(r => r.data),

  // Check which required chars are covered by the model's goldens
  charCoverage: (projectId: string, modelId: string, chars: string[]) =>
    api.get<CharCoverageResponse>(
      `/ml/projects/${projectId}/models/${modelId}/char-coverage`,
      { params: { chars: chars.join(',') } },
    ).then(r => r.data),

  // Retrieve per-char golden images + training sample counts
  getModelGoldens: (projectId: string, modelId: string) =>
    api.get<{ goldens: MLGoldenItem[]; count: number }>(
      `/ml/projects/${projectId}/models/${modelId}/goldens`,
    ).then(r => r.data),

  // Full training report (metrics + goldens + test-set with base64 images)
  getModelReport: (projectId: string, modelId: string, opts?: {
    include_goldens?: boolean; include_testset?: boolean;
  }) =>
    api.get(
      `/ml/projects/${projectId}/models/${modelId}/report`,
      { params: opts },
    ).then(r => r.data),
};
