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
