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
  aug_type?: string;            // realistic NG types or legacy tags
}

export interface SyntheticOkCrop {
  crop_b64: string;
  char_id: string;
  font_name: string;
  rotation_deg: number;
  source: 'synthetic_ok';
}

export type MLAlgorithm = 'rf' | 'svm' | 'mlp' | 'centroid';

export interface SeverityDist {
  subtle: number;
  light: number;
  medium: number;
  heavy: number;
}

export interface TrainRequest {
  algorithm: MLAlgorithm;
  augment_factor: number;
  threshold?: number;         // prob_ok >= threshold → OK
  test_split?: number;
  n_estimators?: number;      // RF only
  max_iter?: number;
  C?: number;
  hidden_layer_sizes?: number[];
  severity_dist?: SeverityDist;     // NG augmentation severity weights
  ok_synth_target?: number;         // Top-up each char to N OK via font-render synth (0=off)
  centroid_temperature?: number;    // Sigmoid temperature for centroid algo (default 5.0)
  include_imported_chars?: boolean; // Merge labeled crops from Imported Chars pool (default true)
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
  phase?: string | null;
  progress?: number;
}

export interface TrainingLogEntry {
  idx: number;
  ts: number;          // unix epoch seconds (server clock)
  level: string;       // "INFO" | "WARNING" | "ERROR" ...
  msg: string;
}

export interface TrainingLogResponse {
  logs: TrainingLogEntry[];
  next_since: number;
  phase: string | null;
  progress: number;
  status: 'pending' | 'training' | 'completed' | 'failed';
  error: string | null;
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

  cloneProject: (id: string, name?: string, description?: string) =>
    api.post<MLProject>(`/ml/projects/${id}/clone`, { name, description }).then(r => r.data),

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

  // Preview synthetic NG (generated from OK samples). OK augmentation removed.
  previewSynthetic: (projectId: string, augmentFactor: number,
                     label: 'NG' = 'NG', severityDist?: SeverityDist) =>
    api.post<{ crops: SyntheticCrop[]; count: number }>(
      `/ml/projects/${projectId}/preview-synthetic`,
      { augment_factor: augmentFactor, label, severity_dist: severityDist }
    ).then(r => r.data),

  // Preview synthetic OK (font-render → composite on real BG → camera noise)
  previewSyntheticOk: (projectId: string, opts: {
    target_n_per_char?: number;
    only_below_threshold?: boolean;
    char_filter?: string[] | null;
    rotation_max_deg?: number;
    size_jitter?: number;
  }) =>
    api.post<{ crops: SyntheticOkCrop[]; count: number }>(
      `/ml/projects/${projectId}/preview-synthetic-ok`, opts
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

  getTrainingLogs: (projectId: string, modelId: string, since: number = 0) =>
    api.get<TrainingLogResponse>(
      `/ml/projects/${projectId}/models/${modelId}/logs`,
      { params: { since } },
    ).then(r => r.data),

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

  // Check which required chars are covered by the model
  charCoverage: (projectId: string, modelId: string, chars: string[]) =>
    api.get<CharCoverageResponse>(
      `/ml/projects/${projectId}/models/${modelId}/char-coverage`,
      { params: { chars: chars.join(',') } },
    ).then(r => r.data),

  // Full training report (metrics + test-set with base64 images)
  getModelReport: (projectId: string, modelId: string, opts?: {
    include_testset?: boolean;
  }) =>
    api.get(
      `/ml/projects/${projectId}/models/${modelId}/report`,
      { params: opts },
    ).then(r => r.data),

  // ── Active learning: search candidate chars from inspection history ──
  inspectionCandidates: (projectId: string, opts: {
    recipe_id?: string;
    date_from?: string;
    date_to?: string;
    include_pred_ok?: boolean;
    include_pred_ng?: boolean;
    limit?: number;
  }) =>
    api.get<{ candidates: InspectionCandidate[]; count: number }>(
      `/ml/projects/${projectId}/inspection-candidates`,
      { params: opts },
    ).then(r => r.data),

  // ── Imported Chars pool ──
  listCharImportBatches: (projectId: string) =>
    api.get<CharImportBatch[]>(`/ml/projects/${projectId}/char-imports/batches`).then(r => r.data),

  createCharImportBatch: (
    projectId: string,
    selections: Array<{ inspection_id: string; annotation_idx: number }>,
    batch_name?: string,
  ) =>
    api.post<{
      batch_id: string | null;
      batch_name?: string;
      imported: number;
      skipped: number;
      errors: Array<{ inspection_id: string; reason: string }>;
    }>(`/ml/projects/${projectId}/char-imports/batches`, { selections, batch_name }).then(r => r.data),

  renameCharImportBatch: (projectId: string, batchId: string, name: string) =>
    api.patch<{ id: string; name: string }>(
      `/ml/projects/${projectId}/char-imports/batches/${batchId}`, { name }
    ).then(r => r.data),

  deleteCharImportBatch: (projectId: string, batchId: string) =>
    api.delete(`/ml/projects/${projectId}/char-imports/batches/${batchId}`).then(r => r.data),

  listCharImports: (projectId: string, opts?: { batch_id?: string; label?: 'OK' | 'NG' }) =>
    api.get<CharImportItem[]>(`/ml/projects/${projectId}/char-imports/chars`, {
      params: opts,
    }).then(r => r.data),

  updateCharImport: (projectId: string, charId: string, update: { char_id?: string | null; label?: 'OK' | 'NG' }) =>
    api.patch<CharImportItem>(
      `/ml/projects/${projectId}/char-imports/chars/${charId}`, update,
    ).then(r => r.data),

  deleteCharImport: (projectId: string, charId: string) =>
    api.delete(`/ml/projects/${projectId}/char-imports/chars/${charId}`).then(r => r.data),

  bulkCharImports: (
    projectId: string,
    body: { char_ids: string[]; label?: 'OK' | 'NG'; delete?: boolean },
  ) =>
    api.patch<{ updated?: number; deleted?: number }>(
      `/ml/projects/${projectId}/char-imports/chars/bulk`, body,
    ).then(r => r.data),
};

export interface InspectionCandidate {
  inspection_id: string;
  recipe_id: string;
  recipe_name: string;
  camera_serial: string;
  frame_idx: number;
  annotation_idx: number;
  expected: string;
  ml_label: 'OK' | 'NG';
  ml_p_ok: number;
  timestamp: string | null;
  image_path: string;
  crop_b64: string;
  // Set when this (inspection_id, annotation_idx) pair is already in the
  // project's char-import pool — FE shows a "Already imported in <batch>"
  // badge and disables the checkbox.
  imported_batch_id: string | null;
  imported_batch_name: string | null;
}

export interface CharImportBatch {
  id: string;
  name: string;
  created_at: string;
  total: number;
  ok_count: number;
  ng_count: number;
}

export interface CharImportItem {
  id: string;
  batch_id: string;
  char_id: string | null;
  label: 'OK' | 'NG';
  crop_url: string;
  ml_label: 'OK' | 'NG';
  ml_p_ok: number;
  inspection_id: string;
  annotation_idx: number;
  recipe_name: string | null;
  camera_serial: string;
  frame_idx: number;
  source_timestamp: string | null;
  created_at: string;
}
