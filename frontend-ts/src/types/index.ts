// ==================== User Types ====================
export interface User {
  id: string;
  username: string;
  email: string;
  full_name: string;
  phone_number?: string;
  avatar_url?: string;
  role: 'admin' | 'operator' | 'supervisor' | 'viewer';
  is_active: boolean;
  last_login?: string;
  created_at: string;
  updated_at: string;
}

export interface UserCreate {
  username: string;
  email?: string;
  password: string;
  full_name: string;
  phone_number?: string;
  avatar_url?: string;
  role: 'admin' | 'operator' | 'supervisor' | 'viewer';
  is_active?: boolean;
}

export interface UserUpdate {
  email?: string;
  full_name?: string;
  phone_number?: string;
  avatar_url?: string;
  role?: 'admin' | 'operator' | 'supervisor' | 'viewer';
  is_active?: boolean;
}

// ==================== Camera Types ====================
export interface CameraSettings {
  exposure_time?: number;
  gain?: number;
  width?: number;
  height?: number;
  delay_trigger?: number;
  delay_interval?: number;
}

export interface Camera {
  id: string;
  name: string;
  camera_id: string;
  // backend sometimes uses `model_name` instead of `model`
  model?: string;
  model_name?: string;
  serial_number?: string;
  ip_address?: string;
  port?: number;
  // boolean connection flag often used in UI
  is_connected?: boolean;
  // textual status retained for compatibility
  status?: 'connected' | 'disconnected' | 'error';
  is_active: boolean;
  location?: string;
  // optional resolution fields used when creating/updating cameras
  resolution_width?: number;
  resolution_height?: number;
  settings?: CameraSettings;
  created_at: string;
  updated_at: string;
}

export interface CameraCreate {
  name: string;
  camera_id: string;
  ip_address?: string;
  port?: number;
  is_active?: boolean;
  settings?: CameraSettings;
  // allow resolution fields in creation payloads
  resolution_width?: number;
  resolution_height?: number;
  model_name?: string;
  serial_number?: string;
  location?: string;
  // additional optional camera metadata
  max_frame_rate?: number;
  pixel_format?: string;
  description?: string;
}

export interface CameraUpdate {
  name?: string;
  ip_address?: string;
  port?: number;
  is_active?: boolean;
  is_connected?: boolean;
  settings?: CameraSettings;
  resolution_width?: number;
  resolution_height?: number;
  model_name?: string;
  serial_number?: string;
  location?: string;
  max_frame_rate?: number;
  pixel_format?: string;
  description?: string;
}

export interface CameraStats {
  total: number;
  connected: number;
  active: number;
}

export interface DiscoveredCamera {
  serial_number: string;
  model_name: string;
  ip_address?: string;
  resolution_width: number;
  resolution_height: number;
  device_class?: string;
}

// ==================== Recipe/Receipt Types ====================
export interface ModelThresholds {
  detection_threshold?: number;
  classification_threshold?: number;
  ocr_threshold?: number;
  recognition_threshold?: number;
}

export interface TemplateConfig {
  template_path?: string;
  match_threshold?: number;
  roi_coordinates?: number[];
}

export interface ROIConfig {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface RecipeCamera {
  camera_id: string;
  model_name: string;
  serial_number: string;
  location: string;
  exposure_time: number;
  delay_trigger: number;
  delay_interval: number;
  reject_pulse: number;
  gain: number;
  pixel_format: string;
  trigger_mode: string;  // 'continuous', 'software_trigger', 'hardware_trigger'
  trigger_config: {
    trigger_selector: string;  // 'FrameStart', 'ExposureStart', 'FrameBurstStart'
    trigger_activation: string;  // 'RisingEdge', 'FallingEdge', 'AnyEdge'
    di_number: number;  // Digital Input number (0-3) for software trigger
    trigger_source: string;  // 'Line0', 'Line1', 'Line2', 'Line3' for hardware trigger
  };
}

export interface Recipe {
  id: string;
  name: string;
  product_code: string;
  description?: string;
  cameras: RecipeCamera[]; // Camera objects with full configuration
  camera_templates: string[]; // Template file paths
  delay_reject?: number;
  reject_pulse?: number;
  reject_method?: string;
  do_reject_number?: number;
  do_alarm_number?: number;
  allow_late_reject?: boolean;
  normal_pulse_ms?: number;
  camera_settings?: CameraSettings;
  model_thresholds?: ModelThresholds;
  template_config?: TemplateConfig;
  roi_config?: ROIConfig;
  is_active: boolean;
  ocr_model_type?: string;
  ml_project_id?: string;
  ml_model_id?: string;
  defect_model?: string;
  classifier_backend?: string;
  cv_method?: string;
  template_bank_enabled?: boolean;
  template_bank_size?: number;
  char_denoise_enabled?: boolean;
  product_detection_method?: string;
  wrinkle_conf?: number;
  wrinkle_show_when_pass?: boolean;
  matching_conf?: number;
  mask_overlap_threshold?: number;
  match_erosion_enabled?: boolean;
  match_erosion_kernel_w?: number;
  match_erosion_kernel_h?: number;
  match_erosion_iterations?: number;
  created_by?: string;
  updated_by?: string;
  created_by_name?: string;  // Full name of creator
  updated_by_name?: string;  // Full name of updater
  created_at?: string;
  updated_at?: string;
}


export interface RecipeCreate {
  name: string;
  product_code: string;
  description?: string;
  cameras?: string[];
  camera_templates?: string[];
  delay_reject?: number;
  reject_pulse?: number;
  reject_method?: string;
  do_reject_number?: number;
  do_alarm_number?: number;
  allow_late_reject?: boolean;
  normal_pulse_ms?: number;
  camera_settings?: CameraSettings;
  model_thresholds?: ModelThresholds;
  template_config?: TemplateConfig;
  roi_config?: ROIConfig;
  is_active?: boolean;
}

export interface RecipeUpdate {
  name?: string;
  product_code?: string;
  description?: string;
  cameras?: string[];
  camera_templates?: string[];
  delay_reject?: number;
  reject_pulse?: number;
  reject_method?: string;
  do_reject_number?: number;
  do_alarm_number?: number;
  allow_late_reject?: boolean;
  normal_pulse_ms?: number;
  camera_settings?: CameraSettings;
  model_thresholds?: ModelThresholds;
  template_config?: TemplateConfig;
  roi_config?: ROIConfig;
  is_active?: boolean;
}

// UI-specific Receipt type (transformed from Recipe)
export interface Receipt {
  id: string;
  name: string;
  productCode: string;
  date?: string;
  camera: string;
  products: number;
  passed: number;
  failed: number;
  operator: string;
  status: 'Active' | 'Inactive';
  description?: string;
  cameras: RecipeCamera[];
  camera_templates: string[];
  delay_reject?: number;
  reject_pulse?: number;
  reject_method?: string;
  do_reject_number?: number;
  do_alarm_number?: number;
  allow_late_reject?: boolean;
  normal_pulse_ms?: number;
  cameraSettings?: CameraSettings;
  modelThresholds?: ModelThresholds;
  template_config?: TemplateConfig;
  roi_config?: ROIConfig;
  is_active: boolean;
  ocr_model_type?: string;
  ml_project_id?: string;
  ml_model_id?: string;
  defect_model?: string;
  classifier_backend?: string;
  cv_method?: string;
  template_bank_enabled?: boolean;
  template_bank_size?: number;
  char_denoise_enabled?: boolean;
  product_detection_method?: string;
  wrinkle_conf?: number;
  wrinkle_show_when_pass?: boolean;
  matching_conf?: number;
  mask_overlap_threshold?: number;
  match_erosion_enabled?: boolean;
  match_erosion_kernel_w?: number;
  match_erosion_kernel_h?: number;
  match_erosion_iterations?: number;
  createdAt: string;
  updatedAt: string;
}

// ==================== Annotation Types ====================
export type AnnotationShape = 'rectangle' | 'polygon';

export interface Annotation {
  id?: string;
  // original API fields
  type?: string; // e.g. 'template' | 'text' | 'ocr'
  shape: AnnotationShape;

  // rectangle fields
  x?: number;
  y?: number;
  width?: number;
  height?: number;

  // polygon fields (array of [x, y] pairs)
  points?: number[][];

  // optional text content for text annotations
  text?: string;

  // confidence threshold (0-1), default 0.85
  conf: number;

  metadata?: Record<string, unknown>;
}

export interface AnnotationCreate {
  shape: AnnotationShape;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  points?: number[][];
  text?: string;
  confidence?: number;
  metadata?: Record<string, unknown>;
}

// ==================== API Response Types ====================
export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  skip: number;
  limit: number;
}

export interface Statistics {
  totalReceipts: number;
  totalProducts: number;
  successRate: number;
}

export interface CountResponse {
  count: number;
}

export interface MessageResponse {
  message: string;
}

// ==================== Component Props Types ====================
export interface ConfirmDialogProps {
  isOpen: boolean;
  title: string;
  message: string;
  type?: 'warning' | 'danger' | 'info' | 'success';
  onConfirm: () => void;
  onCancel: () => void;
}

export interface ToastMessage {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  message: string;
  duration?: number;
}

// ==================== Loading Template Types ====================
export type LoadingTemplate = 
  | 'camera-vision' 
  | 'users' 
  | 'receipts' 
  | 'cameras' 
  | 'historical' 
  | 'settings';

export interface LoadingTemplates {
  dashboard: LoadingTemplate;
  users: LoadingTemplate;
  receipts: LoadingTemplate;
  cameras: LoadingTemplate;
  historical: LoadingTemplate;
  settings: LoadingTemplate;
}

export interface LoadingBackground {
  image: string;
  opacity: number;
}

// ==================== Settings Types ====================
export interface AppSettings {
  // System Settings
  systemName: string;
  version: string;
  language: string;
  timezone: string;

  // Loading Templates
  dashboardLoading: LoadingTemplate;
  usersLoading: LoadingTemplate;
  receiptsLoading: LoadingTemplate;
  camerasLoading: LoadingTemplate;
  historicalLoading: LoadingTemplate;
  settingsLoading: LoadingTemplate;

  // Loading Background
  loadingBackground: string;
  loadingBackgroundOpacity: number;

  // Camera Settings
  camera1Enabled: boolean;
  camera1Fps: number;
  camera1Resolution: string;
  camera2Enabled: boolean;
  camera2Fps: number;
  camera2Resolution: string;
  camera3Enabled: boolean;
  camera3Fps: number;
  camera3Resolution: string;

  // Processing Settings
  detectionThreshold: number;
  maxProcessingTime: number;
  autoReject: boolean;
  saveFailedImages: boolean;

  // Notification Settings
  emailNotifications: boolean;
  smsNotifications: boolean;
  alertThreshold: number;
  dailyReport: boolean;
}

// ==================== Receipt Load Types ====================
export interface ReceiptLoad {
  id: string;
  recipe_id: string;
  loaded_by: string;
  loaded_by_full_name?: string;
  loaded_at: string;
  loaded_at_local?: string;
  metadata?: {
    name: string;
    product_code: string;
    [key: string]: any;
  };
}
export interface FabricObject {
  type: string;
  left: number;
  top: number;
  width: number;
  height: number;
  scaleX: number;
  scaleY: number;
  angle: number;
  fill?: string;
  stroke?: string;
  strokeWidth?: number;
}

// ==================== Action Log Types ====================
export type ActionType =
  | 'login'
  | 'logout'
  | 'create_user'
  | 'update_user'
  | 'delete_user'
  | 'reset_user_password'
  | 'create_recipe'
  | 'update_recipe'
  | 'delete_recipe'
  | 'load_recipe'
  | 'stop_recipe'
  | 'create_camera'
  | 'update_camera'
  | 'delete_camera';

export type ResourceType = 'user' | 'recipe' | 'camera' | 'auth';

export interface ActionLog {
  id: string;
  user_id: string;
  username: string;
  action_type: ActionType;
  resource_type: ResourceType;
  resource_id?: string;
  description: string;
  old_value?: Record<string, any>;
  new_value?: Record<string, any>;
  ip_address?: string;
  user_agent?: string;
  timestamp: string;
}

export interface ActionLogFilters {
  user_id?: string;
  username?: string;
  action_type?: ActionType;
  resource_type?: ResourceType;
  start_date?: string;
  end_date?: string;
}

export interface CanvasState {
  objects: FabricObject[];
  background?: string;
}
