import axios, { AxiosInstance, InternalAxiosRequestConfig } from 'axios';

// ocr_paddle_service: PP-OCRv5/v6 fine-tuning in its own conda env (Paddle GPU
// must not share a site-packages with torch/OpenOCR). Same JWT and the same
// /api/ocr path shape as ocr_service; projects + labelled items are SHARED
// with ocr_service and read-only here — label once in the studio, pick the
// architecture on the Train tab.
// Ports: 8000 backend, 8001 anomaly_service, 8002 ocr_service, 8003 here.
export const PADDLE_OCR_API_BASE_URL = 'http://localhost:8003/api/ocr';

const paddleApi: AxiosInstance = axios.create({
  baseURL: PADDLE_OCR_API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
});

paddleApi.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = localStorage.getItem('access_token');
    if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
  },
  (error) => Promise.reject(error),
);

// ──────── Types ─────────────────────────────────────────────────────────

export interface PaddleArch {
  id: string;
  label: string;
  recommended: boolean;
  /** Upstream *_pretrained.pdparams present on disk. Without it the arch can
   *  only continue from an already-trained model of the same arch. */
  pretrained_available: boolean;
  trained_bases: Array<{ project_id: string; project_name: string; model_id: string; label: string }>;
}

export interface PaddleTrainRequest {
  arch: string;
  base: { kind: 'pretrained' | 'model'; model_id?: string };
  epoch_num: number;
  batch_size: number;
  /** Omitted by the UI — the service default (5e-4) applies. */
  lr?: number;
  test_split: number;
  max_train_width: number;
  max_text_length: number;
  /** Add the project's synthetic defect crops (train items only). */
  use_synthetic?: boolean;
}

export type SyntheticDefectType = 'stroke_break' | 'dot_dropout' | 'overprint' | 'blot' | 'char_missing';

export interface SyntheticSample {
  type: SyntheticDefectType;
  char: string;
  severity: number;
  split: 'train' | 'test';
  source_label: string;
  label: string;
  /** PNG: original | damaged, side by side. */
  image_b64: string;
}

export interface SyntheticInfo {
  exists: boolean;
  reject_char: string;
  per_image?: number;
  max_chars?: number;
  types?: SyntheticDefectType[];
  localiser?: string;
  n_items?: number;
  n_localised?: number;
  n_train?: number;
  n_test?: number;
  by_type?: Record<string, number>;
  samples?: SyntheticSample[];
}

export interface SyntheticProgress {
  running: boolean;
  phase: 'loading' | 'generating' | 'previews' | null;
  done: number;
  total: number;
  percent: number;
}

export interface PaddleModelMetrics {
  /** PaddleOCR's own eval during training — picks the best epoch. */
  acc: number | null;
  norm_edit_dis: number | null;
  best_epoch: number | null;
  n_train: number;
  n_test: number;
  /** Exported artifacts scored with ai_services' preprocessing, batch 1. */
  acc_onnx: number | null;
  acc_exact_onnx: number | null;
  /** Engine scored by ai_services' own TextRecognizerTRT — the line's number. */
  acc_trt: number | null;
  acc_exact_trt: number | null;
  /** Share of synthetic defect crops of TEST items the line would FAIL. */
  syn_reject_onnx?: number | null;
  syn_reject_trt?: number | null;
  syn_n?: number;
  n_synth_train?: number;
}

export type PaddleModelStatus = 'pending' | 'training' | 'completed' | 'failed' | 'cancelled';

export interface PaddleModel {
  id: string;
  project_id: string;
  family: string;
  arch: string;
  params: Record<string, any>;
  base_label: string;
  num_classes: number | null;
  metrics: PaddleModelMetrics;
  checkpoint_path: string;
  onnx_path: string | null;
  engine_path: string | null;
  dict_path: string | null;
  status: PaddleModelStatus;
  error: string | null;
  phase: string | null;
  progress: number;
  created_at: string;
  completed_at: string | null;
}

export interface PaddleTrainedModels {
  projects: Array<{
    project_id: string;
    project_name: string;
    models: Array<{
      model_id: string; arch: string; label: string; runtime: 'tensorrt' | 'onnx';
      acc_trt: number | null; acc_onnx: number | null; created_at: string;
    }>;
  }>;
}

export interface PaddleLogEntry { idx: number; ts: number; level: string; msg: string }

export type PaddleArtifact = 'onnx' | 'engine' | 'dict' | 'checkpoint';

export type PaddleEngine = 'tensorrt' | 'onnx';

export interface PaddleEvalItem {
  id: string;
  gt_text: string;
  pred_text: string;
  confidence: number;
  correct_norm: boolean;
  correct_exact: boolean;
  image_path: string;
  thumb_b64?: string;
}

export interface PaddleEvalResult {
  engine: PaddleEngine;
  scores: { norm: number; exact: number; n: number };
  ms_per_image: number;
  items: PaddleEvalItem[];
  train_metrics: { acc: number | null; best_epoch: number | null };
}

export interface PaddlePredictResult {
  engine: 'onnx';
  text: string;
  confidence: number;
  inference_ms: number;
  image_b64: string;
  size: [number, number];
}

// ──────── API ───────────────────────────────────────────────────────────

export const paddleOcrAPI = {
  listArchs: async (): Promise<PaddleArch[]> => (await paddleApi.get('/archs')).data,

  /** Completed models with a deployable engine/ONNX — feeds the recipe picker. */
  listTrainedModels: async (): Promise<PaddleTrainedModels> =>
    (await paddleApi.get('/trained-models')).data,

  startTraining: async (projectId: string, req: PaddleTrainRequest): Promise<{
    model_id: string; n_train: number; n_test: number; n_synth_train: number;
    dropped_count: number; gpu_holder: string | null;
  }> => (await paddleApi.post(`/projects/${projectId}/train`, req)).data,

  getSynthetic: async (projectId: string, samples = 12): Promise<SyntheticInfo> =>
    (await paddleApi.get(`/projects/${projectId}/synthetic`, { params: { samples } })).data,

  generateSynthetic: async (projectId: string, req: {
    per_image: number; types: SyntheticDefectType[]; test_split: number; seed?: number;
    /** Each variant damages a random 1..max_chars characters. */
    max_chars?: number;
  }): Promise<SyntheticInfo> =>
    (await paddleApi.post(`/projects/${projectId}/synthetic`, req)).data,

  getSyntheticProgress: async (projectId: string): Promise<SyntheticProgress> =>
    (await paddleApi.get(`/projects/${projectId}/synthetic/progress`)).data,

  deleteSynthetic: async (projectId: string) =>
    (await paddleApi.delete(`/projects/${projectId}/synthetic`)).data,

  listModels: async (projectId: string): Promise<PaddleModel[]> =>
    (await paddleApi.get(`/projects/${projectId}/models`)).data,

  getTrainLogs: async (projectId: string, modelId: string, since = 0): Promise<{
    logs: PaddleLogEntry[]; next_since: number; phase: string | null; progress: number;
    status: PaddleModelStatus; error: string | null;
  }> => (await paddleApi.get(`/projects/${projectId}/models/${modelId}/logs`, { params: { since } })).data,

  cancelTraining: async (projectId: string, modelId: string) =>
    (await paddleApi.post(`/projects/${projectId}/models/${modelId}/cancel`)).data,

  deleteModel: async (projectId: string, modelId: string) =>
    (await paddleApi.delete(`/projects/${projectId}/models/${modelId}`)).data,

  exportOnnx: async (projectId: string, modelId: string) =>
    (await paddleApi.post(`/projects/${projectId}/models/${modelId}/export-onnx`)).data,

  exportTensorRT: async (projectId: string, modelId: string) =>
    (await paddleApi.post(`/projects/${projectId}/models/${modelId}/export-tensorrt`)).data,

  evaluate: async (
    projectId: string, modelId: string, engine: PaddleEngine, withThumbs = false,
  ): Promise<PaddleEvalResult> =>
    (await paddleApi.post(`/projects/${projectId}/models/${modelId}/evaluate`, null,
      { params: { engine, with_thumbs: withThumbs } })).data,

  /** ONNX on the service's CPU with ai_services' preprocessing — the engine
   *  itself only runs in the vision env, and measured identical to this ONNX. */
  predict: async (projectId: string, modelId: string, file: File): Promise<PaddlePredictResult> => {
    const form = new FormData();
    form.append('file', file);
    return (await paddleApi.post(`/projects/${projectId}/models/${modelId}/predict`, form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })).data;
  },

  /** Blob download: the endpoint needs the Bearer header. */
  download: async (projectId: string, modelId: string, artifact: PaddleArtifact): Promise<void> => {
    const res = await paddleApi.get(
      `/projects/${projectId}/models/${modelId}/export/${artifact}`, { responseType: 'blob' });
    const ext: Record<PaddleArtifact, string> = {
      onnx: 'onnx', engine: 'engine', dict: 'txt', checkpoint: 'pdparams',
    };
    const url = window.URL.createObjectURL(new Blob([res.data]));
    const a = document.createElement('a');
    a.href = url;
    a.download = `paddle_ocr_${modelId}.${ext[artifact]}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  },
};

export default paddleApi;
