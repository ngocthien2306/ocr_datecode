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

type ThresholdValue = number | { source?: string; parsedValue?: number };

export interface CenterAlignmentCheck {
  ok: boolean;
  offset_x: number;
  abs_offset_x?: number;
  direction?: string;
  threshold_left?: ThresholdValue;
  threshold_right?: ThresholdValue;
  threshold_used?: ThresholdValue;
  template_center?: [number, number];
  product_center?: any[];
}

export interface ProductVerification {
  match: boolean;
  skipped: boolean;
  reason?: string;
  center_alignment_check?: CenterAlignmentCheck;
  wrinkled_check?: any;
  rotation_check?: any;
  misalignment_check?: any;
  label_region_check?: any;
  detected_boxes?: any;
  timing?: any;
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
  product_verification?: ProductVerification;
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

// Statistics interfaces
export interface CameraStats {
  camera_id: string;
  serial_number: string;
  total: number;
  pass: number;
  fail: number;
  pass_rate: number;
}

export interface RecipeStats {
  recipe_id: string;
  recipe_name: string;
  total: number;
  pass: number;
  fail: number;
  pass_rate: number;
}

export interface PeriodInfo {
  start_date: string;
  end_date: string;
}

export interface SummaryStatistics {
  total: number;
  pass: number;
  fail: number;
  pass_rate: number;
  period: PeriodInfo;
  by_camera: CameraStats[];
  by_recipe: RecipeStats[];
}

export interface RecipeBreakdown {
  recipe_id: string;
  recipe_name: string;
  total: number;
  pass: number;
  fail: number;
}

export interface CameraBreakdown {
  camera_id: string;
  serial_number: string;
  total: number;
  pass: number;
  fail: number;
}

export interface TimeseriesCameraData {
  camera_id: string;
  serial_number: string;
  total: number;
  pass: number;
  fail: number;
  pass_rate: number;
  by_recipe: RecipeBreakdown[];
}

export interface TimeseriesRecipeData {
  recipe_id: string;
  recipe_name: string;
  total: number;
  pass: number;
  fail: number;
  pass_rate: number;
}

export interface TimeseriesDataPoint {
  timestamp: string;
  total: number;
  pass: number;
  fail: number;
  pass_rate: number;
  by_recipe: TimeseriesRecipeData[];
}

export interface TimeseriesStatistics {
  granularity: 'minute' | 'hour' | 'day';
  period: PeriodInfo;
  data: TimeseriesDataPoint[];
}

export interface StatisticsFilters {
  start_date?: string;
  end_date?: string;
  recipe_id?: string;
}

export interface TimeseriesFilters extends StatisticsFilters {
  granularity?: 'minute' | 'hour' | 'day';
  recipe_ids?: string;  // Comma-separated
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
  },

  /**
   * Get summary statistics
   */
  getSummaryStatistics: async (filters: StatisticsFilters = {}): Promise<SummaryStatistics> => {
    const response = await api.get<SummaryStatistics>('/inference-results/statistics/summary', {
      params: filters
    });
    return response.data;
  },

  /**
   * Get timeseries statistics
   */
  getTimeseriesStatistics: async (filters: TimeseriesFilters = {}): Promise<TimeseriesStatistics> => {
    const response = await api.get<TimeseriesStatistics>('/inference-results/statistics/timeseries', {
      params: filters
    });
    return response.data;
  }
};
