import api from './api';

export interface TextVerificationResult {
  region_idx?: number;
  annotation_idx?: number;
  expected: string;
  recognized: string;
  match: boolean;
  confidence: number;
  threshold?: number;
}

export interface TextVerification {
  all_match: boolean;
  results: TextVerificationResult[];
}

export interface TemplateVerification {
  match: boolean;
  similarity: number;
  threshold: number;
  method?: string;
  template_bbox?: any;
}

export interface FrameResult {
  frame_idx: number;
  pass_fail: string;
  confidence: number;
  image_path?: string;
  template_name: string;
  template_id?: string | null;
  timings?: any;
  detected_regions?: any[];
  text_verification?: TextVerification;
  template_verification?: TemplateVerification;
}

export interface CameraResult {
  camera_id: string;
  serial_number: string;
  frames: FrameResult[];
}

export interface PerCameraStats {
  serial_number: string;
  confidence: number;
  inliers: number;
  total_matches: number;
  timings: any;
  frame_stats: {
    total_frames: number;
    pass_count: number;
    fail_count: number;
    error_count: number;
    avg_confidence: number;
  };
}

export interface InferenceResultResponse {
  id: string;
  recipe_id: string;
  recipe_name: string;
  product_pass_fail: string;
  camera_results: CameraResult[];
  metadata: {
    total_cameras: number;
    total_frames: number;
    inference_stats: {
      avg_confidence: number;
      total_inliers: number;
      total_matches: number;
      per_camera_stats: PerCameraStats[];
    };
  };
  timestamp: string;
  created_at: string;
}

export interface InferenceResultFilters {
  skip?: number;
  limit?: number;
  recipe_id?: string;
  pass_fail?: 'PASS' | 'FAIL';
  start_date?: string;
  end_date?: string;
}

export const inferenceResultsAPI = {
  /**
   * Get inference results with filters and pagination
   */
  getResults: async (filters: InferenceResultFilters = {}): Promise<InferenceResultResponse[]> => {
    const response = await api.get<any[]>('/inference-results/', {
      params: filters
    });
    // Map _id to id for frontend consistency
    return response.data.map((item: any) => ({
      ...item,
      id: item._id || item.id
    }));
  },

  /**
   * Get count of inference results with filters
   */
  getCount: async (filters: Omit<InferenceResultFilters, 'skip' | 'limit'>): Promise<{
    count: number;
    filters: any;
  }> => {
    const response = await api.get('/inference-results/count', {
      params: filters
    });
    return response.data;
  },

  /**
   * Get inference result by ID
   */
  getById: async (id: string): Promise<InferenceResultResponse> => {
    const response = await api.get<any>(`/inference-results/${id}`);
    // Map _id to id for frontend consistency
    return {
      ...response.data,
      id: response.data._id || response.data.id
    };
  },

  /**
   * Delete inference result by ID
   */
  delete: async (id: string): Promise<{ success: boolean; message: string }> => {
    const response = await api.delete(`/inference-results/${id}`);
    return response.data;
  }
};
