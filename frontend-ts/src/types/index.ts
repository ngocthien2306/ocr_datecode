// ==================== User Types ====================
export interface User {
  id: string;
  username: string;
  email: string;
  full_name: string;
  phone_number?: string;
  avatar_url?: string;
  role: 'admin' | 'operator' | 'viewer';
  is_active: boolean;
  last_login?: string;
  created_at: string;
  updated_at: string;
}

export interface UserCreate {
  username: string;
  email: string;
  password: string;
  full_name: string;
  phone_number?: string;
  avatar_url?: string;
  role: 'admin' | 'operator' | 'viewer';
  is_active?: boolean;
}

export interface UserUpdate {
  email?: string;
  full_name?: string;
  phone_number?: string;
  avatar_url?: string;
  role?: 'admin' | 'operator' | 'viewer';
  is_active?: boolean;
}

// ==================== Camera Types ====================
export interface CameraSettings {
  exposure_time?: number;
  gain?: number;
  width?: number;
  height?: number;
  delay_trigger?: number;
}

export interface Camera {
  id: string;
  name: string;
  camera_id: string;
  model?: string;
  ip_address?: string;
  port?: number;
  status: 'connected' | 'disconnected' | 'error';
  is_active: boolean;
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
}

export interface CameraUpdate {
  name?: string;
  ip_address?: string;
  port?: number;
  is_active?: boolean;
  settings?: CameraSettings;
}

export interface CameraStats {
  total: number;
  connected: number;
  active: number;
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

export interface Recipe {
  id: string;
  name: string;
  product_code: string;
  description?: string;
  cameras: string[]; // Camera IDs
  camera_templates: string[]; // Template file paths
  delay_reject?: number;
  camera_settings?: CameraSettings;
  model_thresholds?: ModelThresholds;
  template_config?: TemplateConfig;
  roi_config?: ROIConfig;
  is_active: boolean;
  created_by?: string;
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
  cameras: string[];
  camera_templates: string[];
  delay_reject?: number;
  cameraSettings?: CameraSettings;
  modelThresholds?: ModelThresholds;
  template_config?: TemplateConfig;
  roi_config?: ROIConfig;
  is_active: boolean;
  createdAt: string;
  updatedAt: string;
}

// ==================== Annotation Types ====================
export interface Annotation {
  id: string;
  type: 'rect' | 'polygon' | 'text' | 'ocr';
  coordinates: number[];
  label: string;
  confidence?: number;
  metadata?: Record<string, unknown>;
}

export interface AnnotationCreate {
  type: 'rect' | 'polygon' | 'text' | 'ocr';
  coordinates: number[];
  label: string;
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

// ==================== Fabric.js Types ====================
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

export interface CanvasState {
  objects: FabricObject[];
  background?: string;
}
