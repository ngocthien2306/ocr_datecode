import axios, { AxiosInstance, InternalAxiosRequestConfig } from 'axios';

// ocr_service runs as its own FastAPI process (own port) on the same GPU
// workstation as backend and anomaly_service — separate service, separate base
// URL, but the same JWT (see ocr_service/app/api/dependencies/auth.py).
// Ports: 8000 backend, 8001 anomaly_service, 8002 here.
export const OCR_API_BASE_URL = 'http://localhost:8002/api/ocr';

const ocrApi: AxiosInstance = axios.create({
  baseURL: OCR_API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
});

ocrApi.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = localStorage.getItem('access_token');
    if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
  },
  (error) => Promise.reject(error),
);

ocrApi.interceptors.response.use(
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

export interface OCRProject {
  id: string;
  name: string;
  description?: string;
  created_at: string;
  updated_at: string;
  created_by?: string;
  total_count: number;
  verified_count: number;
  need_review_count: number;
  status: 'active' | 'training' | 'trained';
}

export interface OCRDatasetStats {
  total_count: number;
  verified_count: number;
  need_review_count: number;
  rejected_count: number;
  /** = verified_count. Only verified items are trainable, so this is the number
   *  the Train tab gates on — a crop whose label is still need_review sits in
   *  the dataset directory but no run will read it. */
  trainable_count: number;
}

export type LabelStatus = 'need_review' | 'verified' | 'rejected';
export type SplitName = 'train' | 'test';
export type RegionType = 'text' | 'datecode';
export type MatchFilter = 'all' | 'pass' | 'fail';

export interface OCRCandidate {
  inspection_id: string;
  recipe_id: string;
  recipe_name: string;
  camera_serial: string;
  frame_idx: number;
  annotation_index: number;
  region_type: RegionType;
  timestamp: string | null;
  product_pass_fail?: string;
  expected_text: string | null;
  recognized_text: string | null;
  ocr_confidence: number | null;
  /** null when the frame carries no verification result for this annotation. */
  verify_match: boolean | null;
  /** Best guess at the ground truth: `expected` when the region matched (that
   *  IS the truth), `recognized` when it failed — on a real misprint what OCR
   *  saw is closer to what is printed than what the recipe wanted. */
  prefill_text: string;
  crop_b64: string;
  /** Non-null when this exact region is already in the dataset. */
  imported_status: LabelStatus | null;
}

export interface OCRCandidatesResponse {
  candidates: OCRCandidate[];
  count: number;
  docs_scanned: number;
  /** Regions whose annotation quad template alignment projected nonsensically
   *  (coordinates far outside the frame). Reported so "few candidates" stays
   *  distinguishable from "this recipe's alignment is failing". */
  skipped_degenerate: number;
}

export interface OCRRecipeOption {
  recipe_id: string;
  recipe_name: string;
  inspection_count: number;
  latest: string | null;
}

export interface OCRImportSelection {
  inspection_id: string;
  camera_serial: string;
  frame_idx: number;
  annotation_index: number;
  gt_text?: string;
}

export interface OCRImportResult {
  imported: number;
  skipped: number;
  errors: Array<{ inspection_id: string; reason: string }>;
  total: number;
  need_review: number;
  verified: number;
  rejected: number;
}

export interface OCRImportFolderResult extends OCRImportResult {
  per_split: Record<string, number>;
  error_count: number;
  source: string;
}

export interface OCRDatasetItem {
  id: string;
  project_id: string;
  inspection_id?: string | null;
  camera_serial?: string | null;
  frame_idx?: number | null;
  annotation_index?: number | null;
  recipe_id?: string | null;
  recipe_name?: string | null;
  region_type?: RegionType | null;
  gt_text: string;
  prefill_text: string;
  expected_text?: string | null;
  recognized_text?: string | null;
  ocr_confidence?: number | null;
  verify_match?: boolean | null;
  status: LabelStatus;
  split: SplitName;
  image_path: string;
  source: 'import' | 'upload' | 'seed';
  exclude_from_training: boolean;
  created_at: string;
  updated_at?: string | null;
  verified_by?: string | null;
  verified_at?: string | null;
  thumb_b64?: string;
}

export interface OCRDatasetItemsPage {
  items: OCRDatasetItem[];
  count: number;
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface OCRPrepareReport {
  n_candidates: number;
  n_train: number;
  n_test: number;
  /** 'pinned' when items already carry split=test; 'test_split' when the slice
   *  was carved by hashing item ids. */
  split_source: 'pinned' | 'test_split';
  dropped_count: number;
  dropped: Array<{ id: string; gt_text: string; length: number; reason: string }>;
  unknown_char_count: number;
  unknown_chars: string[];
  unknown_char_samples: Array<{ id: string; gt_text: string; chars: string[] }>;
  by_recipe_train: Record<string, number>;
  by_recipe_test: Record<string, number>;
  label_files: { train: string; test: string } | null;
  blocking_reason: string | null;
  dry_run: boolean;
}

// ──────── API ───────────────────────────────────────────────────────────

export const ocrTrainingAPI = {
  // Projects
  listProjects: async (): Promise<OCRProject[]> => (await ocrApi.get('/projects')).data,
  createProject: async (name: string, description?: string): Promise<OCRProject> =>
    (await ocrApi.post('/projects', { name, description })).data,
  getProject: async (id: string): Promise<OCRProject> => (await ocrApi.get(`/projects/${id}`)).data,
  updateProject: async (id: string, data: { name?: string; description?: string }): Promise<OCRProject> =>
    (await ocrApi.patch(`/projects/${id}`, data)).data,
  deleteProject: async (id: string): Promise<void> => { await ocrApi.delete(`/projects/${id}`); },
  datasetStats: async (id: string): Promise<OCRDatasetStats> =>
    (await ocrApi.get(`/projects/${id}/dataset-stats`)).data,

  // Candidates / import
  listRecipesWithOcrData: async (limit = 50): Promise<{ recipes: OCRRecipeOption[] }> =>
    (await ocrApi.get('/candidates/recipes', { params: { limit } })).data,
  getCandidates: async (params: {
    project_id: string;
    recipe_id?: string;
    date_from?: string;
    date_to?: string;
    region_type?: RegionType;
    match_filter?: MatchFilter;
    limit?: number;
    max_per_frame?: number;
  }): Promise<OCRCandidatesResponse> => (await ocrApi.get('/candidates', { params })).data,
  importCandidates: async (
    projectId: string, selections: OCRImportSelection[], split: SplitName = 'train',
  ): Promise<OCRImportResult> =>
    (await ocrApi.post(`/projects/${projectId}/import`, { selections, split })).data,
  /** Seed from an OpenOCR-format folder (rec_gt_train.txt / rec_gt_test.txt).
   *  These arrive `verified` — a rec_gt file is already reviewed ground truth. */
  importFolder: async (
    projectId: string, folder = 'data_ocr_merged', markVerified = true,
  ): Promise<OCRImportFolderResult> =>
    (await ocrApi.post(`/projects/${projectId}/import-folder`,
      { folder, mark_verified: markVerified })).data,

  // Dataset / label
  listItems: async (
    projectId: string,
    opts: { status?: LabelStatus; split?: SplitName; page?: number; pageSize?: number } = {},
  ): Promise<OCRDatasetItemsPage> =>
    (await ocrApi.get(`/projects/${projectId}/dataset/items`, {
      params: { status: opts.status, split: opts.split, page: opts.page, page_size: opts.pageSize },
    })).data,
  /** Ids only — lets "select all" span pages without pulling every page's
   *  base64 thumbnails. */
  listItemIds: async (
    projectId: string, status?: LabelStatus, split?: SplitName,
  ): Promise<{ ids: string[]; total: number }> =>
    (await ocrApi.get(`/projects/${projectId}/dataset/item-ids`, { params: { status, split } })).data,
  getItemFull: async (
    projectId: string, itemId: string,
  ): Promise<{ id: string; full_b64: string; width: number; height: number }> =>
    (await ocrApi.get(`/projects/${projectId}/dataset/items/${itemId}/full`)).data,
  relabelItem: async (
    projectId: string, itemId: string,
    data: { gt_text?: string; status?: LabelStatus; split?: SplitName },
  ): Promise<{ item: OCRDatasetItem } & OCRDatasetStats> =>
    (await ocrApi.patch(`/projects/${projectId}/dataset/items/${itemId}`, data)).data,
  bulkStatus: async (
    projectId: string, ids: string[], status: LabelStatus,
  ): Promise<{ modified: number; skipped_empty_text: number } & OCRDatasetStats> =>
    (await ocrApi.post(`/projects/${projectId}/dataset/items/bulk-status`, { ids, status })).data,
  bulkSplit: async (
    projectId: string, ids: string[], split: SplitName,
  ): Promise<{ modified: number }> =>
    (await ocrApi.post(`/projects/${projectId}/dataset/items/bulk-split`, { ids, split })).data,
  bulkExclude: async (
    projectId: string, ids: string[], excluded: boolean,
  ): Promise<{ modified: number; excluded: boolean }> =>
    (await ocrApi.post(`/projects/${projectId}/dataset/items/bulk-exclude`, { ids, excluded })).data,
  bulkDelete: async (
    projectId: string, ids: string[],
  ): Promise<{ deleted: number } & OCRDatasetStats> =>
    (await ocrApi.post(`/projects/${projectId}/dataset/items/bulk-delete`, { ids })).data,

  /** Validate labels and (unless dryRun) write the rec_gt files a run reads.
   *  Always call with dryRun to preview — it reports what training would
   *  silently do to bad labels. */
  prepare: async (
    projectId: string,
    opts: { testSplit?: number; useSpaceChar?: boolean; maxTextLength?: number; dryRun?: boolean } = {},
  ): Promise<OCRPrepareReport> =>
    (await ocrApi.post(`/projects/${projectId}/dataset/prepare`, null, {
      params: {
        test_split: opts.testSplit, use_space_char: opts.useSpaceChar,
        max_text_length: opts.maxTextLength, dry_run: opts.dryRun ?? true,
      },
    })).data,
};

export default ocrApi;

// ──────── Train / Eval / Export types ───────────────────────────────────

/** Which checkpoint a run fine-tunes from. There is no from-scratch option:
 *  a few thousand factory crops cannot train SVTRv2 from random init. */
export interface OCRBaseRef {
  kind: 'builtin' | 'model';
  builtin?: string;
  /** For kind='model'. May point at a DIFFERENT project — deliberately, so a
   *  broad project's model can seed a narrower one. */
  project_id?: string;
  model_id?: string;
}

export interface OCRTrainRequest {
  base: OCRBaseRef;
  use_space_char: boolean;
  epoch_num: number;
  batch_size: number;
  lr: number;
  test_split: number;
  image_h: number;
  image_w: number;
  max_text_length: number;
}

export interface OCRBuiltinBase {
  id: string;
  filename: string;
  available: boolean;
  recommended: boolean;
  use_space_char: boolean;
  vocab_size: number;
}

export interface OCRModelBaseOption {
  model_id: string;
  label: string;
  use_space_char: boolean;
  vocab_size: number;
  created_at: string;
  min_acc: number | null;
}

export interface OCRBaseCheckpoints {
  builtin: OCRBuiltinBase[];
  projects: Array<{ project_id: string; project_name: string; models: OCRModelBaseOption[] }>;
}

export interface OCRModelMetrics {
  /** CTC head. */
  acc: number | null;
  /** SMTR/GTC head. */
  gtc_acc: number | null;
  /** min(acc, gtc_acc) — what best-checkpoint selection tracks. Selecting on
   *  `acc` alone is blind to the SMTR head and can save a run whose GTC head is
   *  unusable while every log line looks healthy. */
  min_acc: number | null;
  norm_edit_dis: number | null;
  best_epoch: number | null;
  n_train: number;
  n_test: number;
  /** Accuracy of the exported artifacts, measured at batch=1. Batched inference
   *  pads crops with -1 and costs real accuracy, so a batched number cannot gate
   *  a model. */
  acc_onnx: number | null;
  acc_trt: number | null;
  acc_exact_trt: number | null;
}

export type OCRModelStatus = 'pending' | 'training' | 'completed' | 'failed' | 'cancelled';

export interface OCRModel {
  id: string;
  project_id: string;
  params: Record<string, any>;
  base_label: string;
  use_space_char: boolean;
  /** 99 without the space class, 100 with it. Two models with different vocab
   *  sizes cannot share an engine or a post-processor. */
  vocab_size: number;
  metrics: OCRModelMetrics;
  checkpoint_path: string;
  config_path: string | null;
  onnx_path: string | null;
  onnx_fp16_path: string | null;
  engine_path: string | null;
  dict_path: string | null;
  status: OCRModelStatus;
  error: string | null;
  phase: string | null;
  progress: number;
  created_at: string;
  completed_at: string | null;
}

export interface OCRStartTrainResult {
  model_id: string;
  status: string;
  n_train: number;
  n_test: number;
  dropped_count: number;
  /** Non-null when another job holds the GPU — this run will queue behind it. */
  gpu_holder: string | null;
}

export interface OCRTrainLogEntry { idx: number; ts: number; level: string; msg: string }

export interface OCRTrainLogsResponse {
  logs: OCRTrainLogEntry[];
  next_since: number;
  phase: string | null;
  progress: number;
  status: OCRModelStatus;
  error: string | null;
}

export type InferenceEngine = 'tensorrt' | 'onnx';

export interface OCREvalScores {
  norm_gtc: number; norm_ctc: number; norm_either: number;
  exact_gtc: number; exact_ctc: number; exact_either: number;
  n: number;
}

export interface OCREvalItem {
  id: string;
  gt_text: string;
  gtc_text: string;
  gtc_conf: number;
  ctc_text: string;
  ctc_conf: number;
  correct_norm: boolean;
  correct_exact: boolean;
  image_path: string;
  thumb_b64?: string;
}

export interface OCREvalResult {
  engine: InferenceEngine;
  scores: OCREvalScores;
  ms_per_image: number;
  items: OCREvalItem[];
  train_metrics: { min_acc: number | null; acc: number | null; gtc_acc: number | null };
}

export interface OCRPredictResult {
  engine: InferenceEngine;
  gtc_text: string;
  gtc_conf: number;
  ctc_text: string;
  ctc_conf: number;
  inference_ms: number;
  image_b64: string;
  size: [number, number];
}

export interface OCREngineInfo {
  inputs: Array<{ name: string; shape: number[]; profile?: number[][] }>;
  outputs: Array<{ name: string; shape: number[] }>;
  size_mb: number;
  /** False when the engine has anything other than two outputs — the assert
   *  ai_services' TextRecognizerSMTRTRT would hit at load time instead. */
  runtime_compatible: boolean;
}

export interface OCROnnxExportResult {
  onnx_path: string;
  onnx_fp16_path: string | null;
  gtc_shape: number[];
  ctc_shape: number[];
}

export type OCRArtifact = 'onnx' | 'onnx_fp16' | 'engine' | 'dict' | 'checkpoint';

// ──────── Train / Eval / Export API ─────────────────────────────────────

export const ocrTrainingModelAPI = {
  listBaseCheckpoints: async (): Promise<OCRBaseCheckpoints> =>
    (await ocrApi.get('/base-checkpoints')).data,

  startTraining: async (
    projectId: string, request: Partial<OCRTrainRequest>,
  ): Promise<OCRStartTrainResult> =>
    (await ocrApi.post(`/projects/${projectId}/train`, request)).data,

  listModels: async (projectId: string): Promise<OCRModel[]> =>
    (await ocrApi.get(`/projects/${projectId}/models`)).data,

  getModelStatus: async (projectId: string, modelId: string): Promise<OCRModel> =>
    (await ocrApi.get(`/projects/${projectId}/models/${modelId}/status`)).data,

  getTrainLogs: async (
    projectId: string, modelId: string, since = 0,
  ): Promise<OCRTrainLogsResponse> =>
    (await ocrApi.get(`/projects/${projectId}/models/${modelId}/logs`, { params: { since } })).data,

  cancelTraining: async (
    projectId: string, modelId: string,
  ): Promise<{ ok: boolean; mode: string }> =>
    (await ocrApi.post(`/projects/${projectId}/models/${modelId}/cancel`)).data,

  deleteModel: async (projectId: string, modelId: string): Promise<{ ok: boolean }> =>
    (await ocrApi.delete(`/projects/${projectId}/models/${modelId}`)).data,

  exportOnnx: async (projectId: string, modelId: string): Promise<OCROnnxExportResult> =>
    (await ocrApi.post(`/projects/${projectId}/models/${modelId}/export-onnx`)).data,

  exportTensorRT: async (
    projectId: string, modelId: string, fp16 = true,
  ): Promise<OCREngineInfo & { engine_path: string; dict_path: string }> =>
    (await ocrApi.post(`/projects/${projectId}/models/${modelId}/export-tensorrt`, null,
      { params: { fp16 } })).data,

  inspectExport: async (projectId: string, modelId: string): Promise<OCREngineInfo> =>
    (await ocrApi.get(`/projects/${projectId}/models/${modelId}/export/inspect`)).data,

  evaluate: async (
    projectId: string, modelId: string, engine: InferenceEngine = 'tensorrt', withThumbs = true,
  ): Promise<OCREvalResult> =>
    (await ocrApi.post(`/projects/${projectId}/models/${modelId}/evaluate`, null,
      { params: { engine, with_thumbs: withThumbs } })).data,

  predict: async (
    projectId: string, modelId: string, file: File, engine: InferenceEngine = 'tensorrt',
  ): Promise<OCRPredictResult> => {
    const form = new FormData();
    form.append('file', file);
    return (await ocrApi.post(`/projects/${projectId}/models/${modelId}/predict`, form, {
      headers: { 'Content-Type': 'multipart/form-data' },
      params: { engine },
    })).data;
  },

  /** Blob download, not an <a href>: every endpoint here needs the Bearer
   *  header and anchor navigation cannot set one. */
  download: async (projectId: string, modelId: string, artifact: OCRArtifact): Promise<void> => {
    const res = await ocrApi.get(
      `/projects/${projectId}/models/${modelId}/export/${artifact}`, { responseType: 'blob' });
    const ext: Record<OCRArtifact, string> = {
      onnx: 'onnx', onnx_fp16: 'fp16.onnx', engine: 'engine', dict: 'txt', checkpoint: 'pth',
    };
    const url = window.URL.createObjectURL(new Blob([res.data]));
    const a = document.createElement('a');
    a.href = url;
    a.download = `ocr_${modelId}.${ext[artifact]}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  },
};
