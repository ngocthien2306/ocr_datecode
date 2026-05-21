import { useState, useEffect, useRef, useMemo } from 'react';
import erosionBeforeImg from '@/assets/demo/erosion_before.jpg';
import erosionAfterImg from '@/assets/demo/erosion_after.jpg';
import TemplateEditor from './TemplateEditorRefactored';
import AnnotationsPanel from '@/components/shared/AnnotationsPanel';
import ConfirmDialog from '@/components/shared/ConfirmDialog';
import ColorSetupModal, { ColorConfig } from './ColorSetupModal';
import { camerasAPI } from '@/services/api';
import recipesAPI from '@/services/recipes';
import { mlTrainingAPI, MLProject, MLModel } from '@/services/mlTraining';
import { useToast } from '@/contexts/ToastContext';
import { useUser } from '@/contexts/UserContext';
import { API_BASE_URL } from '@/config/api';
import '@/styles/RecipeFormModal.css';
import { Camera, Recipe, Annotation, RecipeCamera } from '@/types';
import {
  validateRecipeForm, hasValidationErrors, parseApiErrors,
  type ValidationMode, type RecipeFormInput,
} from '@/utils/recipeValidation';

interface RecipeFormModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: any) => void;
  recipe?: Recipe | null;
  mode?: 'create' | 'edit';
}

interface FormDataType {
  name: string;
  product_code: string;
  description: string;
  delay_reject: number;
  reject_pulse: number;
  reject_method: string;
  do_reject_number: number;
  do_alarm_number: number;
  allow_late_reject: boolean;
  normal_pulse_ms: number;
  is_active: boolean;
  cameras: RecipeCamera[];
  camera_settings: {
    exposure_time: number;
    delay_trigger: number;
    delay_interval: number;
    gain: number;
    brightness: number;
    contrast: number;
  };
  model_thresholds: {
    detection_threshold: number;
    recognition_threshold: number;
    matching_threshold: number;
    min_text_size: number;
    max_text_size: number;
  };
  template_config: any;
  roi_config: any;
  ocr_model_type: string;
  ml_project_id: string;
  ml_model_id: string;
  defect_model: string;
  classifier_backend: string;
  cv_method: string;
  product_detection_method: string;
  product_box_wall_type: string;
  cap_rotation_method: string;
  cap_crop_method: string;
  crop_match_method: string;
  dual_rotation_check: boolean;
  template_bank_enabled: boolean;
  template_bank_size: number;
  char_denoise_enabled: boolean;
  wrinkle_conf: number;
  wrinkle_show_when_pass: boolean;
  matching_conf: number;
  mask_overlap_threshold: number;
  match_erosion_enabled: boolean;
  match_erosion_kernel_w: number;
  match_erosion_kernel_h: number;
  match_erosion_iterations: number;
}

interface Template {
  id?: string;
  name: string;
  image: string;
  image_url?: string;
  image_width?: number;
  image_height?: number;
  annotations: Annotation[];
  center_offset_threshold_left?: number;   // Center alignment threshold left (value in px or %, depending on center_offset_unit)
  center_offset_threshold_right?: number;  // Center alignment threshold right (value in px or %, depending on center_offset_unit)
  center_offset_unit?: 'px' | 'pct';       // Unit for center_offset_threshold_left/right
  wrinkle_area?: number;                   // Total wrinkle area threshold (sum of valid regions ≥ → FAIL)
  wrinkle_min_area?: number;               // Per-region: ignore regions smaller than this (0 = no filter)
  wrinkle_max_area?: number;               // Per-region: any region ≥ this triggers FAIL immediately (0 = disabled)
  color_config?: ColorConfig | null;       // HSV color check config (only used when function_type='Check_Color' + template has 'product')
}

interface CameraTemplates {
  [cameraId: string]: Template[];
}

function applyErosionToImageData(src: ImageData, kw: number, kh: number): ImageData {
  const { width, height, data } = src;
  const out = new ImageData(width, height);
  const halfW = Math.floor(kw / 2);
  const halfH = Math.floor(kh / 2);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let minR = 255, minG = 255, minB = 255;
      for (let dy = -halfH; dy <= halfH; dy++) {
        const ny = Math.max(0, Math.min(height - 1, y + dy));
        for (let dx = -halfW; dx <= halfW; dx++) {
          const nx = Math.max(0, Math.min(width - 1, x + dx));
          const i = (ny * width + nx) * 4;
          if (data[i]   < minR) minR = data[i];
          if (data[i+1] < minG) minG = data[i+1];
          if (data[i+2] < minB) minB = data[i+2];
        }
      }
      const o = (y * width + x) * 4;
      out.data[o] = minR; out.data[o+1] = minG; out.data[o+2] = minB; out.data[o+3] = 255;
    }
  }
  return out;
}

export default function RecipeFormModal({ isOpen, onClose, onSubmit, recipe = null, mode = 'create' }: RecipeFormModalProps) {
  const [activeTab, setActiveTab] = useState<string>('basic');
  const toast = useToast();
  const { canPerformAction, user } = useUser();
  const isOperator = user?.role === 'operator';
  const [erosionPreviewUrl, setErosionPreviewUrl] = useState<string>(erosionAfterImg);
  const [isComputingPreview, setIsComputingPreview] = useState(false);

  const computeErosionPreview = async () => {
    setIsComputingPreview(true);
    try {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = erosionBeforeImg;
      await new Promise<void>((res, rej) => { img.onload = () => res(); img.onerror = rej; });
      const canvas = document.createElement('canvas');
      canvas.width = img.width; canvas.height = img.height;
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(img, 0, 0);
      const kw = formData.match_erosion_kernel_w ?? 80;
      const kh = formData.match_erosion_kernel_h ?? 1;
      const iters = Math.max(1, formData.match_erosion_iterations ?? 1);
      let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      for (let i = 0; i < iters; i++) imageData = applyErosionToImageData(imageData, kw, kh);
      ctx.putImageData(imageData, 0, 0);
      setErosionPreviewUrl(canvas.toDataURL('image/jpeg', 0.9));
    } catch {
      toast.error('Preview failed');
    } finally {
      setIsComputingPreview(false);
    }
  };
  const [formData, setFormData] = useState<FormDataType>({
    name: '',
    product_code: '',
    description: '',
    delay_reject: 100.0,
    reject_pulse: 50.0,
    reject_method: 'DIO_OUT',
    do_reject_number: 2,
    do_alarm_number: 0,
    allow_late_reject: false,
    normal_pulse_ms: 0,
    is_active: true,
    cameras: [],
    camera_settings: {
      exposure_time: 50.0,
      delay_trigger: 100.0,
      delay_interval: 500.0,
      gain: 1.0,
      brightness: 0.5,
      contrast: 1.0
    },
    model_thresholds: {
      detection_threshold: 0.5,
      recognition_threshold: 0.5,
      matching_threshold: 0.85,
      min_text_size: 10,
      max_text_size: 200
    },
    template_config: null,
    roi_config: null,
    ocr_model_type: '',
    ml_project_id: '',
    ml_model_id: '',
    defect_model: 'arcface',
    classifier_backend: 'embedding',
    cv_method: 'legacy',
    product_detection_method: 'yolo_obb',
    cap_rotation_method: 'yolo_obb',
    cap_crop_method: 'none',
    crop_match_method: 'superpoint',
    dual_rotation_check: false,
    product_box_wall_type: 'outer',
    template_bank_enabled: false,
    template_bank_size: 10,
    char_denoise_enabled: false,
    wrinkle_conf: 0.25,
    wrinkle_show_when_pass: true,
    matching_conf: 0.20,
    mask_overlap_threshold: 0.6,
    match_erosion_enabled: false,
    match_erosion_kernel_w: 80,
    match_erosion_kernel_h: 1,
    match_erosion_iterations: 1,
  });

  const [templateImage, setTemplateImage] = useState<string | null>(null);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [segmenting, setSegmenting] = useState(false);

  // ── CV-method preview state ──
  type CvPair = {
    char_idx: number;
    folder?: string;
    logged_label?: string;
    logged_p?: number;
    tmpl_b64: string;
    tgt_b64: string;
    result_b64?: string | null;
    conf: number;
    label: 'OK' | 'NG';
    defect_type: string | null;
    extra?: Record<string, number>;
  };
  type CvPairKey = { folder: string; char_idx: number };
  const [cvPreviewPairs, setCvPreviewPairs] = useState<CvPair[]>([]);
  const [cvPreviewLoading, setCvPreviewLoading] = useState(false);
  const [cvPreviewError, setCvPreviewError] = useState<string | null>(null);
  const [cvPreviewFolder, setCvPreviewFolder] = useState<string | null>(null);
  // Locked pair selection — keeps the same 5 cards across method changes so
  // user can compare scores fairly. Refresh button clears this to re-shuffle.
  const [cvPreviewKeys, setCvPreviewKeys] = useState<CvPairKey[] | null>(null);
  
  // Camera management states
  const [availableCameras, setAvailableCameras] = useState<Camera[]>([]);
  const [loadingCameras, setLoadingCameras] = useState(false);
  const [selectedCameraForAdd, setSelectedCameraForAdd] = useState<string>('');

  // ML project / model states
  const [mlProjects, setMlProjects] = useState<MLProject[]>([]);
  const [mlModels, setMlModels] = useState<MLModel[]>([]);
  const [loadingMlProjects, setLoadingMlProjects] = useState(false);
  const [loadingMlModels, setLoadingMlModels] = useState(false);
  // Char coverage check (warns when selected ML model is missing chars
  // required by this recipe's template text/datecode bboxes)
  const [mlCoverage, setMlCoverage] = useState<{ covered: string[]; missing: string[]; pct: number } | null>(null);
  const [loadingMlCoverage, setLoadingMlCoverage] = useState(false);
  
  // Template editor states - now supports multiple templates per camera
  const [selectedCameraForTemplate, setSelectedCameraForTemplate] = useState<string>('');
  const [cameraTemplates, setCameraTemplates] = useState<CameraTemplates>({}); // { camera_id: [{ id, name, image, annotations }] }
  const [cameraFunctionTypes, setCameraFunctionTypes] = useState<{ [cameraId: string]: string }>({}); // { camera_id: function_type }
  const [selectedTemplateIndex, setSelectedTemplateIndex] = useState<number>(0); // Index of selected template for current camera
  const [selectedAnnotation, setSelectedAnnotation] = useState<number | null>(null);
  const [isGettingFrame, setIsGettingFrame] = useState(false);
  const [frameCount, setFrameCount] = useState<number>(2);
  const [autoRotate, setAutoRotate] = useState(false);
  const [rotatingTemplateIdx, setRotatingTemplateIdx] = useState<number | null>(null);
  const [filmstripExpanded, setFilmstripExpanded] = useState(true);
  const fabricCanvasRef = useRef<any>(null);

  // Color setup modal state — open for a specific (cameraId, templateIdx) pair.
  const [colorSetupTarget, setColorSetupTarget] = useState<{ cameraId: string; templateIdx: number } | null>(null);

  // Confirm dialog state
  const [confirmDialog, setConfirmDialog] = useState<{
    isOpen: boolean;
    title: string;
    message: string;
    type: 'warning' | 'danger' | 'info';
    onConfirm: (() => void) | null;
  }>({
    isOpen: false,
    title: '',
    message: '',
    type: 'warning',
    onConfirm: null
  });

  useEffect(() => {
    if (recipe && mode === 'edit' && isOpen) {
      const recipeAny = recipe as any; // Cast to bypass type checking for backend data structure
      
      // Ensure cameras array has proper structure with defaults
      const normalizedCameras = (recipeAny.cameras || []).map((cam: any) => {
        return {
          camera_id: cam.camera_id || '',
          model_name: cam.model_name || '',
          serial_number: cam.serial_number || '',
          location: cam.location || '',
          reject_pulse: cam.reject_pulse || 50.0,
          exposure_time: cam.exposure_time || 50.0,
          delay_trigger: cam.delay_trigger || 100.0,
          delay_interval: cam.delay_interval || 500.0,
          gain: cam.gain || 1.0,
          pixel_format: cam.pixel_format || 'Mono8',
          trigger_mode: cam.trigger_mode || 'continuous',
          trigger_config: {
            trigger_selector: cam.trigger_config?.trigger_selector || 'FrameStart',
            trigger_activation: cam.trigger_config?.trigger_activation || 'RisingEdge',
            di_number: cam.trigger_config?.di_number !== undefined ? cam.trigger_config.di_number : 0,
            trigger_source: cam.trigger_config?.trigger_source || 'Line0'
          }
        };
      });
      // Load camera templates
      const loadedCameraTemplates: CameraTemplates = {};
      const loadedFunctionTypes: { [cameraId: string]: string } = {};
      console.log('Recipe object:', recipeAny);
      console.log('camera_templates field:', recipeAny.camera_templates);
      console.log('Is array?', Array.isArray(recipeAny.camera_templates));
      console.log('Length:', recipeAny.camera_templates?.length);

      let firstCameraWithTemplates: string | null = null;

      if (recipeAny.camera_templates && Array.isArray(recipeAny.camera_templates) && recipeAny.camera_templates.length > 0) {
        console.log('Loading camera_templates from recipe:', recipeAny.camera_templates);
        recipeAny.camera_templates.forEach((camTemplate: any) => {
          if (camTemplate.templates && camTemplate.templates.length > 0) {
            loadedCameraTemplates[camTemplate.camera_id] = camTemplate.templates.map((template: any) => ({
              id: `template-${Date.now()}-${Math.random()}`,
              name: template.name,
              image: `${API_BASE_URL}${template.image_url}`, // Convert to full URL for preview
              image_url: template.image_url,
              image_width: template.image_width,
              image_height: template.image_height,
              annotations: template.annotations,
              center_offset_threshold_left: template.center_offset_threshold_left ?? 50.0,
              center_offset_threshold_right: template.center_offset_threshold_right ?? 50.0,
              center_offset_unit: (template.center_offset_unit as 'px' | 'pct') ?? 'px',
              wrinkle_area: template.wrinkle_area ?? 2000.0,
              wrinkle_min_area: template.wrinkle_min_area ?? 0.0,
              wrinkle_max_area: template.wrinkle_max_area ?? 0.0,
              color_config: template.color_config ?? null
            }));

            // Load function_type for this camera (default to 'OCR' if not set)
            loadedFunctionTypes[camTemplate.camera_id] = camTemplate.function_type || 'Check_Type_Product';

            // Remember first camera with templates for auto-selection
            if (!firstCameraWithTemplates) {
              firstCameraWithTemplates = camTemplate.camera_id;
            }
          }
        });
        console.log('Loaded camera templates:', loadedCameraTemplates);
        console.log('Loaded function types:', loadedFunctionTypes);
      } else {
        console.log('No camera_templates found in recipe - empty or undefined');
      }
      setCameraTemplates(loadedCameraTemplates);
      setCameraFunctionTypes(loadedFunctionTypes);
      
      // Auto-select first camera with templates
      if (firstCameraWithTemplates) {
        setSelectedCameraForTemplate(firstCameraWithTemplates);
        setSelectedTemplateIndex(0);
      }

      setFormData({
        name: recipeAny.name || '',
        product_code: recipeAny.productCode || recipeAny.product_code || '',
        description: recipeAny.description || '',
        delay_reject: recipeAny.delay_reject || 100.0,
        reject_pulse: recipeAny.reject_pulse || 50.0,
        reject_method: recipeAny.reject_method || 'DIO_OUT',
        do_reject_number: recipeAny.do_reject_number !== undefined ? recipeAny.do_reject_number : 0,
        do_alarm_number: recipeAny.do_alarm_number !== undefined ? recipeAny.do_alarm_number : 0,
        allow_late_reject: recipeAny.allow_late_reject === true,
        normal_pulse_ms: recipeAny.normal_pulse_ms !== undefined ? recipeAny.normal_pulse_ms : 0,
        is_active: recipeAny.is_active !== undefined ? recipeAny.is_active : (recipeAny.status === 'Active'),
        cameras: normalizedCameras,
        camera_settings: recipeAny.cameraSettings || recipeAny.camera_settings || {
          exposure_time: 50.0,
          delay_trigger: 100.0,
          gain: 1.0,
          brightness: 0.5,
          contrast: 1.0
        },
        model_thresholds: recipeAny.modelThresholds || recipeAny.model_thresholds || {
          detection_threshold: 0.5,
          recognition_threshold: 0.5,
          min_text_size: 10,
          max_text_size: 200
        },
        template_config: recipeAny.template_config || null,
        roi_config: recipeAny.roi_config || null,
        ocr_model_type: recipeAny.ocr_model_type || '',
        ml_project_id: recipeAny.ml_project_id || '',
        ml_model_id: recipeAny.ml_model_id || '',
        defect_model: recipeAny.defect_model || 'arcface',
        classifier_backend: recipeAny.classifier_backend || 'embedding',
        cv_method: recipeAny.cv_method || 'legacy',
        product_detection_method: recipeAny.product_detection_method || 'yolo_obb',
        cap_rotation_method: recipeAny.cap_rotation_method || 'yolo_obb',
        cap_crop_method: recipeAny.cap_crop_method || 'none',
        crop_match_method: recipeAny.crop_match_method || 'superpoint',
        dual_rotation_check: !!recipeAny.dual_rotation_check,
        product_box_wall_type: recipeAny.product_box_wall_type || 'outer',
        template_bank_enabled: recipeAny.template_bank_enabled ?? false,
        template_bank_size: recipeAny.template_bank_size ?? 10,
        char_denoise_enabled: recipeAny.char_denoise_enabled ?? false,
        wrinkle_conf: recipeAny.wrinkle_conf ?? 0.25,
        wrinkle_show_when_pass: recipeAny.wrinkle_show_when_pass ?? true,
        matching_conf: recipeAny.matching_conf ?? 0.20,
        mask_overlap_threshold: recipeAny.mask_overlap_threshold ?? 0.6,
        match_erosion_enabled: recipeAny.match_erosion_enabled ?? false,
        match_erosion_kernel_w: recipeAny.match_erosion_kernel_w ?? 80,
        match_erosion_kernel_h: recipeAny.match_erosion_kernel_h ?? 1,
        match_erosion_iterations: recipeAny.match_erosion_iterations ?? 1,
      });

      if (recipeAny.template_config?.template_image) {
        setTemplateImage(recipeAny.template_config.template_image);
      } else {
        setTemplateImage(null);
      }
      
      if (recipeAny.template_config?.annotations) {
        setAnnotations(recipeAny.template_config.annotations);
      } else {
        setAnnotations([]);
      }

      // For operator role, default to template tab
      if (user?.role === 'operator') {
        setActiveTab('template');
      }
    } else if (mode === 'create' && isOpen) {
      // Reset form for create mode
      setFormData({
        name: '',
        product_code: '',
        description: '',
        delay_reject: 100.0,
        reject_pulse: 50.0,
        reject_method: 'DIO_OUT',
        do_reject_number: 2,
        do_alarm_number: 0,
        allow_late_reject: false,
        normal_pulse_ms: 0,
        is_active: true,
        cameras: [],
        camera_settings: {
          exposure_time: 50.0,
          delay_trigger: 100.0,
          gain: 1.0,
          brightness: 0.5,
          contrast: 1.0,
          delay_interval: 500.0,
        },
        model_thresholds: {
          detection_threshold: 0.5,
          recognition_threshold: 0.5,
          matching_threshold: 0.85,
          min_text_size: 10,
          max_text_size: 200
        },
        template_config: null,
        roi_config: null,
        ocr_model_type: '',
        ml_project_id: '',
        ml_model_id: '',
        defect_model: 'arcface',
        classifier_backend: 'embedding',
        cv_method: 'legacy',
        template_bank_enabled: false,
        template_bank_size: 10,
        char_denoise_enabled: false,
        product_detection_method: 'yolo_obb',
        product_box_wall_type: 'outer',
        cap_rotation_method: 'yolo_obb',
        cap_crop_method: 'none',
        crop_match_method: 'superpoint',
        dual_rotation_check: false,
        wrinkle_conf: 0.25,
        wrinkle_show_when_pass: true,
        matching_conf: 0.20,
        mask_overlap_threshold: 0.6,
        match_erosion_enabled: false,
        match_erosion_kernel_w: 80,
        match_erosion_kernel_h: 1,
        match_erosion_iterations: 1,
      });
      setTemplateImage(null);
      setAnnotations([]);
      setCameraTemplates({});
      setCameraFunctionTypes({});
      setSelectedTemplateIndex(0);
    }
  }, [recipe, mode, isOpen]);

  // Load available cameras
  useEffect(() => {
    if (isOpen) {
      loadAvailableCameras();
      loadMlProjects();
    }
  }, [isOpen]);

  // CV-method preview: fetch 5 sample pair scores.
  // When `keys` is provided → reuse same pairs (compare across methods).
  // When `keys` is null → BE picks random pairs from latest folder.
  const fetchCvPreview = async (method: string, keys: CvPairKey[] | null = null) => {
    setCvPreviewLoading(true);
    setCvPreviewError(null);
    try {
      const body: Record<string, unknown> = { cv_method: method, count: 5, threshold: 0.80 };
      if (keys && keys.length) body.pair_keys = keys;
      const resp = await fetch(`${API_BASE_URL}/api/recipes/cv-preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      if (data.error) {
        setCvPreviewError(data.error);
        setCvPreviewPairs([]);
      } else {
        const pairs: CvPair[] = data.pairs || [];
        setCvPreviewPairs(pairs);
        setCvPreviewFolder(data.folder || null);
        // Cache keys so subsequent method changes reuse the same pairs
        const nextKeys: CvPairKey[] = pairs
          .filter((p) => p.folder)
          .map((p) => ({ folder: p.folder as string, char_idx: p.char_idx }));
        if (nextKeys.length) setCvPreviewKeys(nextKeys);
      }
    } catch (e) {
      setCvPreviewError((e as Error).message);
      setCvPreviewPairs([]);
    } finally {
      setCvPreviewLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen && activeTab === 'model' && formData.classifier_backend === 'embedding') {
      // Method change → reuse cached pairs to enable fair score comparison
      fetchCvPreview(formData.cv_method || 'legacy', cvPreviewKeys);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, activeTab, formData.cv_method, formData.classifier_backend]);

  // Load ML models whenever ml_project_id changes
  useEffect(() => {
    if (formData.ml_project_id) {
      loadMlModels(formData.ml_project_id);
    } else {
      setMlModels([]);
    }
  }, [formData.ml_project_id]);

  // Collect chars the ML model must be able to classify at runtime.
  // ML inspection is char-level only: `text` / `datecode` annotations go
  // through OCR (they don't need a per-char golden). Only `char` annotations
  // require ML coverage — each holds exactly one character.
  const recipeChars = useMemo(() => {
    const chars = new Set<string>();
    Object.values(cameraTemplates).forEach(tpls => {
      tpls.forEach(tpl => {
        (tpl.annotations || []).forEach((ann: any) => {
          if (ann.type !== 'char') return;
          const txt = typeof ann.text === 'string' ? ann.text.trim() : '';
          if (txt) chars.add(txt[0]);   // defensive: char annotation = 1 char
        });
      });
    });
    return Array.from(chars).sort();
  }, [cameraTemplates]);

  // Check coverage whenever ml_model_id or recipe chars change
  useEffect(() => {
    if (!formData.ml_project_id || !formData.ml_model_id || recipeChars.length === 0) {
      setMlCoverage(null);
      return;
    }
    let cancelled = false;
    (async () => {
      setLoadingMlCoverage(true);
      try {
        const cov = await mlTrainingAPI.charCoverage(
          formData.ml_project_id,
          formData.ml_model_id,
          recipeChars,
        );
        if (!cancelled) {
          setMlCoverage({
            covered: cov.covered,
            missing: cov.missing,
            pct: cov.coverage_pct,
          });
        }
      } catch (e) {
        if (!cancelled) setMlCoverage(null);
        console.warn('[RecipeForm] char-coverage check failed:', e);
      } finally {
        if (!cancelled) setLoadingMlCoverage(false);
      }
    })();
    return () => { cancelled = true; };
  }, [formData.ml_project_id, formData.ml_model_id, recipeChars]);

  const loadAvailableCameras = async () => {
    setLoadingCameras(true);
    try {
      const data = await camerasAPI.getAllCameras(0, 100);
      setAvailableCameras(data);
    } catch (error) {
      console.error('Failed to load cameras:', error);
    } finally {
      setLoadingCameras(false);
    }
  };

  const loadMlProjects = async () => {
    setLoadingMlProjects(true);
    try {
      const data = await mlTrainingAPI.listProjects();
      setMlProjects(data);
    } catch (error) {
      console.error('Failed to load ML projects:', error);
    } finally {
      setLoadingMlProjects(false);
    }
  };

  const loadMlModels = async (projectId: string) => {
    setLoadingMlModels(true);
    try {
      const data = await mlTrainingAPI.listModels(projectId);
      // Only show completed models
      setMlModels(data.filter(m => m.status === 'completed'));
    } catch (error) {
      console.error('Failed to load ML models:', error);
    } finally {
      setLoadingMlModels(false);
    }
  };

  const handleInputChange = (e: any) => {
    const { name, value, type, checked } = e.target;
    let finalValue: any = value;

    if (type === 'checkbox') {
      finalValue = checked;
    } else if (name === 'delay_reject' || name === 'reject_pulse' || name === 'normal_pulse_ms') {
      finalValue = parseFloat(value) || 0;
    } else if (name === 'do_reject_number' || name === 'do_alarm_number') {
      finalValue = parseInt(value) || 0;
    }

    setFormData(prev => ({
      ...prev,
      [name]: finalValue
    }));
  };

  const handleCameraSettingChange = (field: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      camera_settings: {
        ...prev.camera_settings,
        [field]: parseFloat(value) || 0
      }
    }));
  };

  const handleModelThresholdChange = (field: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      model_thresholds: {
        ...prev.model_thresholds,
        [field]: field.includes('threshold') ? parseFloat(value) || 0 : parseInt(value) || 0
      }
    }));
  };

  // Camera management functions
  const handleAddCamera = () => {
    if (!selectedCameraForAdd) return;
    
    const camera: any = availableCameras.find(c => c.camera_id === selectedCameraForAdd);
    if (!camera) return;

    // Check if camera already added
    if (formData.cameras.some(c => c.camera_id === camera.camera_id)) {
      toast.warning('This camera is already added to the recipe');
      return;
    }

    const newCamera: RecipeCamera = {
      camera_id: camera.camera_id,
      model_name: camera.model_name,
      serial_number: camera.serial_number,
      location: camera.location || '',
      reject_pulse: 50.0,
      exposure_time: 50.0,
      delay_trigger: 100.0,
      delay_interval: 500.0,
      gain: 1.0,
      pixel_format: 'Mono8',
      trigger_mode: 'continuous',
      trigger_config: {
        trigger_selector: 'FrameStart',
        trigger_activation: 'RisingEdge',
        di_number: 0,
        trigger_source: 'Line0'
      }
    };

    setFormData(prev => ({
      ...prev,
      cameras: [...prev.cameras, newCamera]
    }));
    setSelectedCameraForAdd('');
  };

  const handleRemoveCamera = (cameraId: string) => {
    setFormData(prev => ({
      ...prev,
      cameras: prev.cameras.filter(c => c.camera_id !== cameraId)
    }));
  };

  const handleCameraConfigChange = (cameraId: string, field: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      cameras: prev.cameras.map(cam =>
        cam.camera_id === cameraId
          ? { ...cam, [field]: parseFloat(value) || value }
          : cam
      )
    }));
  };

  const handleCameraTriggerConfigChange = (cameraId: string, field: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      cameras: prev.cameras.map(cam =>
        cam.camera_id === cameraId
          ? {
              ...cam,
              trigger_config: {
                ...cam.trigger_config,
                [field]: field === 'di_number' ? parseInt(value) : value
              }
            }
          : cam
      )
    }));
  };

  const handleSubmit = async (e: any) => {
    e.preventDefault();

    const validationInput: RecipeFormInput = {
      basicInfo: {
        name: formData.name,
        product_code: formData.product_code,
        description: formData.description,
        do_reject_number: formData.do_reject_number,
        normal_pulse_ms: formData.normal_pulse_ms,
      },
      cameras: formData.cameras,
      modelThresholds: formData.model_thresholds,
    };

    const validationErrors = validateRecipeForm(validationInput, mode as ValidationMode);

    // Block save when backend='ml' + char bboxes but no ML model. Show popup
    // explaining the issue and jump to Model tab.
    if (formData.classifier_backend === 'ml'
        && recipeChars.length > 0
        && (!formData.ml_project_id || !formData.ml_model_id)) {
      setConfirmDialog({
        isOpen: true,
        title: 'ML Model not selected',
        type: 'warning',
        message:
          `Active Method = "ML Trained Model" but no AI Project / Trained Model is selected. ` +
          `This recipe has ${recipeChars.length} char bbox${recipeChars.length > 1 ? 'es' : ''} that need verification — without a model they will be skipped at runtime.\n\n` +
          `Pick a model in the Model tab, or switch Active Method to "Embedding".`,
        onConfirm: () => {
          setConfirmDialog(prev => ({ ...prev, isOpen: false }));
          setActiveTab('model');
        },
      });
      return;
    }

    if (hasValidationErrors(validationErrors)) {
      setErrors(validationErrors);
      const keys = Object.keys(validationErrors);
      if (keys.some(k => ['name', 'product_code', 'description', 'do_reject_number', 'normal_pulse_ms'].includes(k))) {
        setActiveTab('basic');
      } else if (keys.some(k => k.startsWith('cameras'))) {
        setActiveTab('camera');
      } else if (keys.some(k => k.startsWith('model_thresholds'))) {
        setActiveTab('model');
      }
      return;
    }

    // Validate all camera templates before submission
    const templateErrors = validateTemplates();
    if (templateErrors.length > 0) {
      setConfirmDialog({
        isOpen: true,
        title: 'Template Validation Failed',
        message: `${templateErrors.join('\n')}\n\nRequirements:\n• Check_Color: no "template" region needed. Must have a 'product' annotation (color check) OR 'text/datecode' (OCR sub-mode).\n• Otherwise: 1 "template" region + at least 1 of text/barcode/datecode is required.\n• crop_area is always optional.`,
        type: 'warning',
        onConfirm: null
      });
      return;
    }

    setLoading(true);
    try {
      // Prepare camera templates data for submission
      const cameraTemplatesArray: any[] = [];
      Object.entries(cameraTemplates).forEach(([cameraId, templates]) => {
        if (templates && templates.length > 0) {
          cameraTemplatesArray.push({
            camera_id: cameraId,
            function_type: cameraFunctionTypes[cameraId] || 'Check_Type_Product', // Include function type
            templates: templates.map(template => ({
              name: template.name,
              image_url: template.image_url,
              image_width: template.image_width,
              image_height: template.image_height,
              annotations: template.annotations,
              center_offset_threshold_left: template.center_offset_threshold_left ?? 50.0,
              center_offset_threshold_right: template.center_offset_threshold_right ?? 50.0,
              center_offset_unit: template.center_offset_unit ?? 'px',
              wrinkle_area: template.wrinkle_area ?? 2000.0,
              wrinkle_min_area: template.wrinkle_min_area ?? 0.0,
              wrinkle_max_area: template.wrinkle_max_area ?? 0.0,
              color_config: template.color_config ?? null
            }))
          });
        }
      });

      const submitData = {
        ...formData,
        camera_templates: cameraTemplatesArray,
        // Legacy single template support (fallback)
        template_config: templateImage ? {
          template_image: templateImage,
          annotations: annotations
        } : formData.template_config,
        // Normalize empty-string sentinels to null so BE clears the field.
        // Empty strings come from "-- None --" <option value="">.
        ml_project_id: formData.ml_project_id || null,
        ml_model_id:   formData.ml_model_id   || null,
        defect_model:  formData.defect_model  || 'arcface',
        classifier_backend: formData.classifier_backend || 'embedding',
        cv_method: formData.cv_method || 'legacy',
        product_detection_method: formData.product_detection_method || 'yolo_obb',
        cap_rotation_method: formData.cap_rotation_method || 'yolo_obb',
        cap_crop_method: formData.cap_crop_method || 'none',
        crop_match_method: formData.crop_match_method || 'superpoint',
        dual_rotation_check: !!formData.dual_rotation_check,
        product_box_wall_type: formData.product_box_wall_type || 'outer',
        template_bank_enabled: formData.template_bank_enabled ?? false,
        template_bank_size: formData.template_bank_size ?? 10,
        char_denoise_enabled: formData.char_denoise_enabled ?? false,
        wrinkle_conf: formData.wrinkle_conf ?? 0.25,
        wrinkle_show_when_pass: formData.wrinkle_show_when_pass ?? true,
        matching_conf: formData.matching_conf ?? 0.20,
        mask_overlap_threshold: formData.mask_overlap_threshold ?? 0.6,
        match_erosion_enabled: formData.match_erosion_enabled ?? false,
        match_erosion_kernel_w: formData.match_erosion_kernel_w ?? 80,
        match_erosion_kernel_h: formData.match_erosion_kernel_h ?? 1,
        match_erosion_iterations: formData.match_erosion_iterations ?? 1,
      };

      await onSubmit(submitData);
      handleClose();
    } catch (error: any) {
      console.error('Submit error:', error);
      const status = error.response?.status;
      const detail = error.response?.data?.detail;

      if (status === 422 && detail) {
        // Parse Pydantic validation errors → show inline on fields
        const { fieldErrors, unhandled } = parseApiErrors(detail);
        if (hasValidationErrors(fieldErrors)) {
          setErrors(fieldErrors);
          // Auto-navigate to the tab that has the first error
          const keys = Object.keys(fieldErrors);
          if (keys.some(k => ['name', 'product_code', 'description', 'do_reject_number', 'normal_pulse_ms'].includes(k))) {
            setActiveTab('basic');
          } else if (keys.some(k => k.startsWith('cameras'))) {
            setActiveTab('camera');
          } else if (keys.some(k => k.startsWith('model_thresholds'))) {
            setActiveTab('model');
          }
        }
        unhandled.forEach(msg => toast.error(msg));
      } else {
        // Generic server/network error → toast
        const msg = typeof detail === 'string' ? detail : 'Failed to save recipe';
        toast.error(msg);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleGetMultipleFrames = async () => {
    if (!selectedCameraForTemplate) {
      toast.warning('Please select a camera first');
      return;
    }

    // Find the selected camera details
    const selectedCamera = formData.cameras.find(c => c.camera_id === selectedCameraForTemplate);
    if (!selectedCamera) {
      toast.error('Camera not found in recipe configuration');
      return;
    }

    const serialNumber = selectedCamera.serial_number;

    setIsGettingFrame(true);
    try {
      toast.info(`Fetching ${frameCount} frame${frameCount > 1 ? 's' : ''} from camera...`);

      // First check if camera is connected
      const statusResponse = await camerasAPI.getCameraStatus(serialNumber);

      if (!statusResponse.is_connected) {
        toast.error('Camera is not connected. Please go to Camera Management to connect the camera first.');
        return;
      }

      // Fetch N frames from ring buffer
      const framesResponse = await camerasAPI.getLatestFrames(serialNumber, frameCount, 95);

      if (!framesResponse || !framesResponse.frames || framesResponse.frames.length === 0) {
        toast.error('No frames available. Camera may not be streaming.');
        return;
      }

      let { frames } = framesResponse;

      // Auto-rotate: gửi frames lên BE để detect OBB và xoay
      if (autoRotate) {
        toast.info(`Auto-rotating ${frames.length} frame${frames.length > 1 ? 's' : ''}...`);
        try {
          const rotatedResponse = await camerasAPI.rotateFrames(frames, 95);
          if (rotatedResponse?.frames?.length > 0) {
            frames = rotatedResponse.frames;
            toast.success(`Rotated ${frames.length} frame${frames.length > 1 ? 's' : ''} successfully`);
          } else {
            toast.warning('Rotation returned no frames, using original');
          }
        } catch (rotateErr) {
          toast.warning('Auto-rotate failed, using original frames');
          console.error('Rotate error:', rotateErr);
        }
      }

      toast.info(`Processing ${frames.length} frame${frames.length > 1 ? 's' : ''}...`);

      // Get current templates for this camera
      const currentTemplates = cameraTemplates[selectedCameraForTemplate] || [];
      const initialTemplateCount = currentTemplates.length;

      console.log('===== GET MULTIPLE FRAMES =====');
      console.log(`Camera: ${selectedCameraForTemplate}`);
      console.log(`Existing templates: ${initialTemplateCount}`);
      currentTemplates.forEach((tmpl, idx) => {
        console.log(`  Template ${idx}: ${tmpl.name}, Annotations: ${tmpl.annotations.length}`);
      });

      // Determine if we should use cyclic mapping or single template cloning
      const numberOfTemplates = currentTemplates.length;
      const useCyclicMapping = numberOfTemplates > 1;

      console.log(`Cyclic mapping: ${useCyclicMapping} (${numberOfTemplates} templates)`);
      console.log(`Frames to capture: ${frameCount}`);
      console.log('===============================');

      if (useCyclicMapping) {
        toast.info(`Cloning annotations from ${numberOfTemplates} templates in cyclic pattern...`);
      } else if (numberOfTemplates === 1) {
        const annotationsCount = currentTemplates[0]?.annotations?.length || 0;
        if (annotationsCount > 0) {
          toast.info(`Cloning ${annotationsCount} annotation${annotationsCount > 1 ? 's' : ''} from template...`);
        }
      }

      // Process each frame sequentially
      const newTemplates: any[] = [];

      for (let i = 0; i < frames.length; i++) {
        const frameData = frames[i];
        const frameBase64 = frameData.frame_base64;

        // Convert base64 to blob
        const byteCharacters = atob(frameBase64);
        const byteNumbers = new Array(byteCharacters.length);
        for (let j = 0; j < byteCharacters.length; j++) {
          byteNumbers[j] = byteCharacters.charCodeAt(j);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: 'image/jpeg' });

        // Upload frame to server
        const uploadFormData = new FormData();
        uploadFormData.append('file', blob, `camera_${serialNumber}_frame${i + 1}_${Date.now()}.jpg`);

        const uploadToken = localStorage.getItem('access_token');
        const uploadResponse = await fetch(`${API_BASE_URL}/api/recipes/templates/upload`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${uploadToken}`,
            'Bypass-Tunnel-Reminder': 'true'
          },
          body: uploadFormData
        });

        if (!uploadResponse.ok) {
          console.error(`Failed to upload frame ${i + 1}`);
          continue;
        }

        const { url, width, height } = await uploadResponse.json();

        // Convert blob to base64 for canvas preview (await Promise)
        const imageDataUrl = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = (event) => {
            resolve(event.target?.result as string);
          };
          reader.onerror = reject;
          reader.readAsDataURL(blob);
        });

        // Determine which template to clone from (cyclic mapping)
        let sourceTemplate;
        let sourceTemplateIndex = -1;
        if (useCyclicMapping) {
          // Cyclic mapping: frame[i] gets annotations from template[i % numberOfTemplates]
          sourceTemplateIndex = i % numberOfTemplates;
          sourceTemplate = currentTemplates[sourceTemplateIndex];
          console.log(`Frame ${i} → Template ${sourceTemplateIndex} (cyclic mapping)`);
        } else if (numberOfTemplates === 1) {
          // Single template: clone from the only available template
          sourceTemplateIndex = 0;
          sourceTemplate = currentTemplates[0];
          console.log(`Frame ${i} → Template 0 (single template)`);
        } else {
          // No templates: use empty annotations
          sourceTemplate = null;
          console.log(`Frame ${i} → No template (empty)`);
        }

        const annotationsToClone = sourceTemplate?.annotations || [];
        console.log(`Frame ${i}: Cloning ${annotationsToClone.length} annotations from template ${sourceTemplateIndex}`);

        // Clone annotations from source template (deep copy with unique IDs)
        const clonedAnnotations = annotationsToClone.map((annotation: any, annIdx: number) => ({
          ...annotation,
          id: `annotation-${Date.now()}-frame${i}-ann${annIdx}-${Math.random().toString(36).substring(2, 11)}`  // Generate unique ID
        }));

        // Create template with cloned annotations
        const templateName = `Frame ${initialTemplateCount + i + 1}`;
        const newTemplate = {
          id: `template-${Date.now()}-frame${i}-${Math.random().toString(36).substring(2, 11)}`,
          name: templateName,
          image: imageDataUrl, // For preview in canvas
          image_url: url, // Server URL for submission
          image_width: width,
          image_height: height,
          annotations: clonedAnnotations,  // Use cloned annotations
          center_offset_threshold_left: sourceTemplate?.center_offset_threshold_left || 50.0,
          center_offset_threshold_right: sourceTemplate?.center_offset_threshold_right || 50.0,
          center_offset_unit: sourceTemplate?.center_offset_unit ?? 'px',
          wrinkle_area: sourceTemplate?.wrinkle_area ?? 2000.0,
          wrinkle_min_area: sourceTemplate?.wrinkle_min_area ?? 0.0,
          wrinkle_max_area: sourceTemplate?.wrinkle_max_area ?? 0.0,
          color_config: sourceTemplate?.color_config ?? null
        };

        newTemplates.push(newTemplate);
      }

      // Add all templates at once (batch update)
      if (newTemplates.length > 0) {
        console.log('===== SUMMARY =====');
        console.log(`Total frames processed: ${newTemplates.length}`);
        console.log(`Cyclic mapping used: ${useCyclicMapping}`);
        newTemplates.forEach((tmpl, idx) => {
          console.log(`  New Template ${idx}: ${tmpl.name}, Annotations: ${tmpl.annotations.length}`);
        });
        console.log('===================');

        setCameraTemplates(prev => ({
          ...prev,
          [selectedCameraForTemplate]: [...(prev[selectedCameraForTemplate] || []), ...newTemplates]
        }));

        // Auto-select last added template
        setSelectedTemplateIndex(initialTemplateCount + newTemplates.length - 1);

        // Success message with annotation info
        if (useCyclicMapping) {
          toast.success(`Successfully captured ${newTemplates.length} frame${newTemplates.length > 1 ? 's' : ''} with cyclic annotation mapping from ${numberOfTemplates} templates!`);
        } else if (numberOfTemplates === 1 && currentTemplates[0]?.annotations?.length) {
          const annotationsCount = currentTemplates[0]!.annotations.length;
          toast.success(`Successfully captured ${newTemplates.length} frame${newTemplates.length > 1 ? 's' : ''} with ${annotationsCount} annotation${annotationsCount > 1 ? 's' : ''} cloned!`);
        } else {
          toast.success(`Successfully captured ${newTemplates.length} frame${newTemplates.length > 1 ? 's' : ''}!`);
        }
      } else {
        toast.error('No frames were successfully processed');
      }

    } catch (error: any) {
      console.error('Get frames error:', error);
      toast.error(error.message || 'Failed to get frames from camera');
    } finally {
      setIsGettingFrame(false);
    }
  };

  const handleImageUpload = async (e: any) => {
    const file = e.target.files[0];
    if (file) {
      try {
        // Upload image to server
        const formData = new FormData();
        formData.append('file', file);

        const token = localStorage.getItem('access_token');
        const response = await fetch(`${API_BASE_URL}/api/recipes/templates/upload`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Bypass-Tunnel-Reminder': 'true'
          },
          body: formData
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || 'Failed to upload image');
        }

        const { url, width, height } = await response.json();

        // Also keep base64 for preview
        const reader = new FileReader();
        reader.onload = (event) => {
          const imageDataUrl = event.target?.result as string;

          if (selectedCameraForTemplate) {
            // Add new template to the selected camera's templates array
            const currentTemplates = cameraTemplates[selectedCameraForTemplate] || [];
            const templateName = `Template ${currentTemplates.length + 1}`;
            const newTemplate = {
              id: `template-${Date.now()}`,
              name: templateName,
              image: imageDataUrl, // For preview in canvas
              image_url: url, // Server URL for submission
              image_width: width,
              image_height: height,
              annotations: [],
              center_offset_threshold_left: 50.0,
              center_offset_threshold_right: 50.0,
              center_offset_unit: 'px' as const,    // Default unit (BC with old recipes)
              wrinkle_area: 2000.0,
              wrinkle_min_area: 0.0,
              wrinkle_max_area: 0.0,
              color_config: null as ColorConfig | null
            };

            setCameraTemplates(prev => ({
              ...prev,
              [selectedCameraForTemplate]: [...(prev[selectedCameraForTemplate] || []), newTemplate]
            }));

            // Auto-select the new template
            setSelectedTemplateIndex(currentTemplates.length);
          } else {
            setTemplateImage(imageDataUrl);
            setAnnotations([]);
          }
        };
        reader.readAsDataURL(file);

      } catch (error) {
        console.error('Upload error:', error);
        setConfirmDialog({
          isOpen: true,
          title: 'Upload Failed',
          message: 'Failed to upload template image. Please try again.',
          type: 'danger',
          onConfirm: null
        });
      }
    }

    // Reset file input
    e.target.value = '';
  };

  const handleAnnotationsChange = (newAnnotations: Annotation[]) => {
    console.log('handleAnnotationsChange called:', {
      selectedCamera: selectedCameraForTemplate,
      selectedTemplateIndex,
      newAnnotations,
      annotationsCount: newAnnotations.length
    });
    
    if (selectedCameraForTemplate) {
      setCameraTemplates(prev => {
        const templates = prev[selectedCameraForTemplate] || [];
        const updatedTemplates = [...templates];
        if (updatedTemplates[selectedTemplateIndex]) {
          updatedTemplates[selectedTemplateIndex] = {
            ...updatedTemplates[selectedTemplateIndex],
            annotations: newAnnotations
          };
        }
        
        console.log('Updated templates:', updatedTemplates);
        
        return {
          ...prev,
          [selectedCameraForTemplate]: updatedTemplates
        };
      });
    } else {
      setAnnotations(newAnnotations);
    }
  };

  const getCurrentTemplateImage = () => {
    if (selectedCameraForTemplate && cameraTemplates[selectedCameraForTemplate]) {
      const templates = cameraTemplates[selectedCameraForTemplate];
      return templates[selectedTemplateIndex]?.image || null;
    }
    return templateImage;
  };

  const getCurrentAnnotations = () => {
    if (selectedCameraForTemplate && cameraTemplates[selectedCameraForTemplate]) {
      const templates = cameraTemplates[selectedCameraForTemplate];
      const annotations = templates[selectedTemplateIndex]?.annotations || [];
      console.log('getCurrentAnnotations:', {
        camera: selectedCameraForTemplate,
        templateIndex: selectedTemplateIndex,
        annotationsCount: annotations.length,
        annotations
      });
      return annotations;
    }
    return annotations;
  };
  
  const getCurrentTemplate = () => {
    if (selectedCameraForTemplate && cameraTemplates[selectedCameraForTemplate]) {
      const templates = cameraTemplates[selectedCameraForTemplate];
      return templates[selectedTemplateIndex] || null;
    }
    return null;
  };
  
  const handleDeleteTemplate = (templateIndex: number) => {
    if (!selectedCameraForTemplate) return;
    
    setCameraTemplates(prev => {
      const templates = prev[selectedCameraForTemplate] || [];
      const updated = templates.filter((_, idx) => idx !== templateIndex);
      return {
        ...prev,
        [selectedCameraForTemplate]: updated
      };
    });
    
    // Adjust selected index if needed
    if (selectedTemplateIndex >= templateIndex && selectedTemplateIndex > 0) {
      setSelectedTemplateIndex(selectedTemplateIndex - 1);
    }
  };
  
  const handleRenameTemplate = (templateIndex: number, newName: string) => {
    if (!selectedCameraForTemplate) return;

    setCameraTemplates(prev => {
      const templates = prev[selectedCameraForTemplate] || [];
      const updated = [...templates];
      if (updated[templateIndex]) {
        updated[templateIndex] = {
          ...updated[templateIndex],
          name: newName
        };
      }
      return {
        ...prev,
        [selectedCameraForTemplate]: updated
      };
    });
  };

  const handleMoveTemplate = (templateIndex: number, direction: 'up' | 'down') => {
    if (!selectedCameraForTemplate) return;

    setCameraTemplates(prev => {
      const templates = prev[selectedCameraForTemplate] || [];
      const newIndex = direction === 'up' ? templateIndex - 1 : templateIndex + 1;

      // Check bounds
      if (newIndex < 0 || newIndex >= templates.length) return prev;

      // Swap templates
      const updated = [...templates];
      const temp = updated[templateIndex];
      updated[templateIndex] = updated[newIndex]!;
      updated[newIndex] = temp!;

      // Update selected index to follow the moved template
      if (selectedTemplateIndex === templateIndex) {
        setSelectedTemplateIndex(newIndex);
      } else if (selectedTemplateIndex === newIndex) {
        setSelectedTemplateIndex(templateIndex);
      }

      return {
        ...prev,
        [selectedCameraForTemplate]: updated
      };
    });
  };
  
  const handleRotateTemplate = async (templateIndex: number) => {
    if (!selectedCameraForTemplate) return;
    const templates = cameraTemplates[selectedCameraForTemplate] || [];
    const template = templates[templateIndex];
    if (!template?.image) return;

    setRotatingTemplateIdx(templateIndex);
    try {
      // 1. Resolve image to base64 — template.image may be a data URL or an HTTP URL
      let base64: string;
      if (template.image.startsWith('data:')) {
        base64 = template.image.replace(/^data:image\/\w+;base64,/, '');
      } else {
        // HTTP URL (e.g. loaded from server in edit mode) — fetch and convert
        const token = localStorage.getItem('access_token');
        const imgRes = await fetch(template.image, {
          headers: { Authorization: `Bearer ${token}`, 'Bypass-Tunnel-Reminder': 'true' },
        });
        if (!imgRes.ok) throw new Error('Failed to fetch template image');
        const imgBlob = await imgRes.blob();
        base64 = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve((reader.result as string).replace(/^data:image\/\w+;base64,/, ''));
          reader.onerror = reject;
          reader.readAsDataURL(imgBlob);
        });
      }

      // 2. Rotate via OBB detection
      const rotatedResponse = await camerasAPI.rotateFrames(
        [{ frame_base64: base64, metadata: {} }],
        95
      );

      if (!rotatedResponse?.frames?.length) {
        toast.warning('Rotation returned no result');
        return;
      }

      const rotatedBase64: string = rotatedResponse.frames[0].frame_base64;
      const rotatedDataUrl = `data:image/jpeg;base64,${rotatedBase64}`;

      // 2. Upload rotated image to server to get a persistent image_url
      const byteCharacters = atob(rotatedBase64);
      const byteArray = new Uint8Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteArray[i] = byteCharacters.charCodeAt(i);
      }
      const blob = new Blob([byteArray], { type: 'image/jpeg' });
      const uploadForm = new FormData();
      uploadForm.append('file', blob, `${template.name}_rotated.jpg`);

      const token = localStorage.getItem('access_token');
      const uploadRes = await fetch(`${API_BASE_URL}/api/recipes/templates/upload`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}`, 'Bypass-Tunnel-Reminder': 'true' },
        body: uploadForm,
      });

      if (!uploadRes.ok) {
        throw new Error('Failed to upload rotated image');
      }

      const { url, width, height } = await uploadRes.json();

      // 3. Update template with rotated image + new server URL
      setCameraTemplates(prev => {
        const updated = [...(prev[selectedCameraForTemplate] || [])];
        if (updated[templateIndex]) {
          updated[templateIndex] = {
            ...updated[templateIndex],
            image: rotatedDataUrl,
            image_url: url,
            image_width: width,
            image_height: height,
          };
        }
        return { ...prev, [selectedCameraForTemplate]: updated };
      });

      toast.success(`Template "${template.name}" rotated`);
    } catch (err) {
      toast.error('Failed to rotate template image');
      console.error('Rotate template error:', err);
    } finally {
      setRotatingTemplateIdx(null);
    }
  };

  // Use certain helper functions in no-op effects to prevent 'declared but never read' TS warnings
  useEffect(() => {
    // keep reference to handleCameraSettingChange to avoid unused variable warning
    // no-op
  }, [handleCameraSettingChange]);

  useEffect(() => {
    // keep reference to getCurrentTemplate to avoid unused variable warning
    void getCurrentTemplate();
  }, [selectedCameraForTemplate, selectedTemplateIndex]);
  
  const validateTemplates = () => {
    const errors: string[] = [];

    Object.entries(cameraTemplates).forEach(([cameraId, templates]) => {
      const functionType = cameraFunctionTypes[cameraId] || 'Check_Type_Product';
      templates.forEach((template) => {
        const hasTemplateRegion = template.annotations.some((ann: any) => ann.type === 'template');
        const hasProduct = template.annotations.some((ann: any) => ann.type === 'product');
        const hasRequiredAnnotation = template.annotations.some((ann: any) =>
          ['text', 'barcode', 'datecode'].includes(ann.type)
        );

        // Check_Color templates NEVER require the SuperPoint "template" region —
        // color-check uses image-proc (no SuperPoint), OCR-on-cap sub-mode within
        // Check_Color still relies on transformed_bboxes from the matcher but the
        // anchor template region isn't strictly required at the validation layer.
        if (functionType === 'Check_Color') {
          // Need at least one of: product (color check) or text/datecode (OCR sub-mode).
          if (!hasProduct && !hasRequiredAnnotation) {
            errors.push(
              `Camera ${cameraId} - ${template.name}: Check_Color template must have either a 'product' annotation (color check) or 'text/datecode' (OCR sub-mode)`
            );
          }
        } else {
          if (!hasTemplateRegion) {
            errors.push(`Camera ${cameraId} - ${template.name}: Missing required "template" region`);
          }
          if (!hasRequiredAnnotation) {
            errors.push(`Camera ${cameraId} - ${template.name}: Must have at least one annotation (text, barcode, or datecode)`);
          }
        }

        // Validate text field per annotation type
        template.annotations.forEach((ann: any, annIdx: number) => {
          const txt = (ann.text || '').trim();
          if ((ann.type === 'text' || ann.type === 'datecode') && !txt) {
            errors.push(`Camera ${cameraId} - ${template.name} - BBox #${annIdx + 1}: "${ann.type}" annotation requires text content`);
          }
          if (ann.type === 'char') {
            if (!txt) {
              errors.push(`Camera ${cameraId} - ${template.name} - BBox #${annIdx + 1}: "char" annotation requires expected character`);
            } else if (txt.length > 1) {
              errors.push(`Camera ${cameraId} - ${template.name} - BBox #${annIdx + 1}: "char" must be exactly one character (got "${txt}")`);
            }
          }
        });
      });
    });
    
    return errors;
  };

  const handleAnnotationTypeChange = (index: number, newType: string) => {
    const currentAnnotations = getCurrentAnnotations();
    const updated: any[] = [...currentAnnotations];
    if (updated[index]) {
      updated[index].type = newType;
    }
    handleAnnotationsChange(updated);
  };

  const handleAnnotationTextChange = (index: number, newText: string) => {
    const currentAnnotations = getCurrentAnnotations();
    const updated: any[] = [...currentAnnotations];
    if (updated[index]) {
      updated[index].text = newText;
    }
    handleAnnotationsChange(updated);
  };

  const handleAnnotationConfChange = (index: number, newConf: number) => {
    const currentAnnotations = getCurrentAnnotations();
    const updated: any[] = [...currentAnnotations];
    if (updated[index]) {
      updated[index].conf = newConf;
    }
    handleAnnotationsChange(updated);
  };

  const handleDeleteAnnotation = (index: number) => {
    const currentAnnotations = getCurrentAnnotations();
    const updated = currentAnnotations.filter((_, i) => i !== index);
    handleAnnotationsChange(updated);
    
    // Reset selection if deleted
    if (selectedAnnotation === index) {
      setSelectedAnnotation(null);
    } else if (selectedAnnotation !== null && selectedAnnotation > index) {
      setSelectedAnnotation(selectedAnnotation - 1);
    }
  };

  const handleAutoSegment = async (index: number) => {
    const currentAnnotations = getCurrentAnnotations();
    const ann = currentAnnotations[index];
    if (!ann || ann.shape !== 'rectangle') return;

    // Segment-with-OCR is only meaningful for text/datecode regions —
    // each segment becomes a `char` annotation pre-filled with the OCR'd text.
    if (ann.type !== 'text' && ann.type !== 'datecode') {
      toast.warning('Segment is only supported on Text OCR or Date Code regions');
      return;
    }

    // Get the template image_url (the server-relative path, e.g. /api/recipes/templates/images/abc.jpg)
    const template = getCurrentTemplate();
    const imageUrl = template?.image_url;
    if (!imageUrl) {
      toast.warning('No template image found for segmentation');
      return;
    }

    setSegmenting(true);
    try {
      const result = await recipesAPI.segmentTemplateRegion(
        imageUrl,
        { x: ann.x ?? 0, y: ann.y ?? 0, w: ann.width ?? 0, h: ann.height ?? 0 },
        { withOcr: true },
      );

      if (result.count === 0) {
        toast.warning('No characters found in this region');
        return;
      }

      // Each segment → new annotation type='char' with OCR'd text pre-filled.
      // Original text/datecode annotation is KEPT (still used for word-level OCR).
      const newAnnotations = result.segments.map((seg: any) => ({
        id: `annotation-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        type: 'char' as const,
        shape: 'rectangle' as const,
        x: seg.x,
        y: seg.y,
        width: seg.w,
        height: seg.h,
        text: seg.expected_text || '',
        conf: ann.conf ?? 0.5,
      }));

      const updated = [...currentAnnotations];
      updated.splice(index + 1, 0, ...newAnnotations);
      handleAnnotationsChange(updated);

      const ocrPreview = result.full_text ? ` ("${result.full_text}")` : '';
      toast.success(`Created ${result.count} char annotation(s)${ocrPreview}`);
    } catch (e: any) {
      console.error('Auto segment failed:', e);
      toast.error(e?.response?.data?.detail || 'Segmentation failed');
    } finally {
      setSegmenting(false);
    }
  };

  const handleClose = () => {
    // Don't reset formData here - let useEffect handle it when modal reopens
    setErrors({});
    setActiveTab('basic');
    setSelectedCameraForAdd('');
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="recipe-form-page">
      <div className="page-header">
        <button className="back-btn" onClick={handleClose}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M19 12H5M5 12L12 19M5 12L12 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Back to List
        </button>
        <h2>{mode === 'create' ? 'Create New Recipe' : 'Edit Recipe'}</h2>
        <div className="header-actions">
          <button type="button" className="btn btn-secondary" onClick={handleClose} disabled={loading}>
            Cancel
          </button>
          <button type="button" className="btn btn-primary" onClick={(e) => { e.preventDefault(); handleSubmit(e); }} disabled={loading}>
            {loading ? 'Saving...' : (mode === 'create' ? 'Create Recipe' : 'Update Recipe')}
          </button>
        </div>
      </div>

      <div className="recipe-form-container">
        <div className="recipe-form-layout">
          <div className="vertical-tabs">
            <button className={`tab-btn ${activeTab === 'basic' ? 'active' : ''}`} onClick={() => setActiveTab('basic')} disabled={isOperator} title="Basic Info">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M9 5H7C6.46957 5 5.96086 5.21071 5.58579 5.58579C5.21071 5.96086 5 6.46957 5 7V19C5 19.5304 5.21071 20.0391 5.58579 20.4142C5.96086 20.7893 6.46957 21 7 21H17C17.5304 21 18.0391 20.7893 18.4142 20.4142C18.7893 20.0391 19 19.5304 19 19V7C19 6.46957 18.7893 5.96086 18.4142 5.58579C18.0391 5.21071 17.5304 5 17 5H15M9 5C9 5.53043 9.21071 6.03914 9.58579 6.41421C9.96086 6.78929 10.4696 7 11 7H13C13.5304 7 14.0391 6.78929 14.4142 6.41421C14.7893 6.03914 15 5.53043 15 5M9 5C9 4.46957 9.21071 3.96086 9.58579 3.58579C9.96086 3.21071 10.4696 3 11 3H13C13.5304 3 14.0391 3.21071 14.4142 3.58579C14.7893 3.96086 15 4.46957 15 5" stroke="currentColor" strokeWidth="2"/>
              </svg>
            </button>
            <button className={`tab-btn ${activeTab === 'camera' ? 'active' : ''}`} onClick={() => setActiveTab('camera')} disabled={isOperator} title="Camera">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M23 19C23 19.5304 22.7893 20.0391 22.4142 20.4142C22.0391 20.7893 21.5304 21 21 21H3C2.46957 21 1.96086 20.7893 1.58579 20.4142C1.21071 20.0391 1 19.5304 1 19V8C1 7.46957 1.21071 6.96086 1.58579 6.58579C1.96086 6.21071 2.46957 6 3 6H7L9 3H15L17 6H21C21.5304 6 22.0391 6.21071 22.4142 6.58579C22.7893 6.96086 23 7.46957 23 8V19Z" stroke="currentColor" strokeWidth="2"/>
                <circle cx="12" cy="13" r="4" stroke="currentColor" strokeWidth="2"/>
              </svg>
            </button>
            <button className={`tab-btn ${activeTab === 'model' ? 'active' : ''}`} onClick={() => setActiveTab('model')} disabled={isOperator} title="Model">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
                <circle cx="12" cy="12" r="6" stroke="currentColor" strokeWidth="2"/>
                <circle cx="12" cy="12" r="2" fill="currentColor"/>
              </svg>
            </button>
            <button className={`tab-btn ${activeTab === 'template' ? 'active' : ''}`} onClick={() => setActiveTab('template')} title="Template">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="2"/>
                <circle cx="8.5" cy="8.5" r="1.5" fill="currentColor"/>
                <path d="M21 15L16 10L5 21" stroke="currentColor" strokeWidth="2"/>
              </svg>
              {annotations.length > 0 && <span className="badge">{annotations.length}</span>}
            </button>
          </div>

          <form onSubmit={handleSubmit} className="recipe-form">
          {errors.submit && <div className="error-message global-error">{errors.submit}</div>}

          <div className="tab-content">
            {activeTab === 'basic' && (
              <div className="form-section">
                <h3>Basic Information</h3>
                <div className="form-row">
                  <div className="form-group">
                    <label>Recipe Name <span className="required">*</span></label>
                    <input type="text" name="name" value={formData.name} onChange={handleInputChange} 
                           placeholder="Enter recipe name" className={errors.name ? 'error' : ''} />
                    {errors.name && <span className="error-message">{errors.name}</span>}
                  </div>
                  <div className="form-group">
                    <label>Product Code <span className="required">*</span></label>
                    <input type="text" name="product_code" value={formData.product_code} onChange={handleInputChange}
                           placeholder="Enter product code" className={errors.product_code ? 'error' : ''} />
                    {errors.product_code && <span className="error-message">{errors.product_code}</span>}
                  </div>
                </div>
                <div className="form-row">
                  <div className="form-group">
                    <label>Description</label>
                    <textarea name="description" value={formData.description} onChange={handleInputChange}
                             placeholder="Enter recipe description" rows={3}
                             className={errors.description ? 'error' : ''} />
                    {errors.description && <span className="error-message">{errors.description}</span>}
                  </div>
                  <div className="form-group">
                    <label>Delay Reject (ms)</label>
                    <input type="number" name="delay_reject" value={formData.delay_reject}
                           onChange={handleInputChange} step="0.1" min="0"
                           placeholder="Delay reject time in milliseconds" />
                  </div>
                  <div className="form-group">
                    <label>Reject Pulse (ms)</label>
                    <input type="number" name="reject_pulse" value={formData.reject_pulse}
                           onChange={handleInputChange} step="0.1" min="0"
                           placeholder="Reject pulse duration in milliseconds" />
                  </div>
                  <div className="form-group">
                    <label>Reject Output Method</label>
                    <select
                      name="reject_method"
                      value={formData.reject_method}
                      onChange={handleInputChange}
                    >
                      <option value="DIO">DIO_OUT</option>
                      <option value="PLC">PLC</option>
                    </select>
                    <small style={{display: 'block', marginTop: 4, color: '#666'}}>
                      Output method for reject signal
                    </small>
                  </div>
                  <div className="form-group">
                    <label>Reject DO Number</label>
                    <select
                      name="do_reject_number"
                      value={formData.do_reject_number.toString()}
                      onChange={handleInputChange}
                    >
                      <option value="0">DO 0</option>
                      <option value="1">DO 1</option>
                      <option value="2">DO 2</option>
                      <option value="3">DO 3</option>
                    </select>
                    <small style={{display: 'block', marginTop: 4, color: '#666'}}>
                      Digital Output port for reject control (0-3)
                    </small>
                  </div>
                  <div className="form-group">
                    <label>Alarm DO Number</label>
                    <select
                      name="do_alarm_number"
                      value={formData.do_alarm_number.toString()}
                      onChange={handleInputChange}
                    >
                      <option value="-1">None</option>
                      <option value="0">DO 0</option>
                      <option value="1">DO 1</option>
                      <option value="2">DO 2</option>
                      <option value="3">DO 3</option>
                      <option value="4">DO 4</option>
                    </select>
                    <small style={{display: 'block', marginTop: 4, color: '#666'}}>
                      Digital Output port for alarm output (0-4)
                    </small>
                  </div>
                  <div className="form-group">
                    <label>Allow Late Reject</label>
                    <input
                      type="checkbox"
                      name="allow_late_reject"
                      checked={formData.allow_late_reject}
                      onChange={handleInputChange}
                      style={{width: 'auto', alignSelf: 'flex-start', marginRight: 'auto'}}
                    />
                  </div>
                  <div className="form-group">
                    <label>Normal Pulse Width (ms)</label>
                    <input
                      type="number"
                      name="normal_pulse_ms"
                      value={formData.normal_pulse_ms}
                      onChange={handleInputChange}
                      step="10"
                      min="0"
                      max="999999"
                      placeholder="Expected pulse width per bottle (ms)"
                      className={errors.normal_pulse_ms ? 'error' : ''}
                    />
                    {errors.normal_pulse_ms && <span className="error-message">{errors.normal_pulse_ms}</span>}
                    <small style={{display: 'block', marginTop: 4, color: '#666'}}>
                      Expected DI pulse width per bottle (ms). Stuck detected when pulse &gt; 2.0×
                    </small>
                  </div>
                </div>
                <div className="form-group checkbox-group">
                  <label>
                    <input type="checkbox" name="is_active" checked={formData.is_active} onChange={handleInputChange} />
                    <span>Active Recipe</span>
                  </label>
                </div>
              </div>
            )}

            {activeTab === 'camera' && (
              <div className="form-section">
                <div className="camera-header">
                  <h3>Camera Configuration</h3>
                  <div className="camera-add-section">
                    <select 
                      value={selectedCameraForAdd} 
                      onChange={(e) => setSelectedCameraForAdd(e.target.value)}
                      disabled={loadingCameras}
                    >
                      <option value="">Select a camera...</option>
                      {availableCameras
                        .filter(cam => !formData.cameras.some(c => c.camera_id === cam.camera_id))
                        .map((cam: any) => (
                          <option key={cam.camera_id} value={cam.camera_id}>
                            {cam.camera_id} - {cam.model_name} ({cam.serial_number})
                          </option>
                        ))
                      }
                    </select>
                    <button 
                      type="button" 
                      className="btn btn-secondary" 
                      onClick={handleAddCamera}
                      disabled={!selectedCameraForAdd || loadingCameras}
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" style={{ marginRight: '6px' }}>
                        <path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                      </svg>
                      Add Camera
                    </button>
                  </div>
                </div>

                {errors.cameras && <span className="error-message">{errors.cameras}</span>}

                {formData.cameras.length === 0 ? (
                  <div className="empty-state">
                    <p>No cameras configured for this recipe</p>
                    <p className="hint">Add cameras using the dropdown above</p>
                  </div>
                ) : (
                  <div className="cameras-list">
                    {formData.cameras.map((camera, index) => (
                      <div key={camera.camera_id} className="camera-card">
                        <div className="camera-card-header">
                          <h4>Camera {index + 1}: {camera.camera_id}</h4>
                          <button
                            type="button"
                            className="btn-remove"
                            onClick={() => handleRemoveCamera(camera.camera_id)}
                            title="Remove camera"
                          >
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                              <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                            </svg>
                          </button>
                        </div>
                        
                        <div className="camera-info">
                          <div className="info-item">
                            <strong>Model:</strong> {camera.model_name}
                          </div>
                          <div className="info-item">
                            <strong>Serial:</strong> {camera.serial_number}
                          </div>
                          <div className="info-item">
                            <strong>Location:</strong> {camera.location || 'N/A'}
                          </div>
                        </div>

                        <div className="camera-settings">
                          <h5>Camera Settings</h5>
                          <div className="form-row">
                            <div className="form-group">
                              <label>Exposure Time (μs) <span className="required">*</span></label>
                              <input
                                type="number"
                                value={camera.exposure_time}
                                onChange={(e) => handleCameraConfigChange(camera.camera_id, 'exposure_time', e.target.value)}
                                step="0.1"
                                min="0"
                                className={errors[`cameras.${index}.exposure_time`] ? 'error' : ''}
                              />
                              {errors[`cameras.${index}.exposure_time`] && (
                                <span className="error-message">{errors[`cameras.${index}.exposure_time`]}</span>
                              )}
                            </div>
                            <div className="form-group">
                              <label>Delay Trigger (ms) <span className="required">*</span></label>
                              <input
                                type="number"
                                value={camera.delay_trigger}
                                onChange={(e) => handleCameraConfigChange(camera.camera_id, 'delay_trigger', e.target.value)}
                                step="0.1"
                                min="0"
                                className={errors[`cameras.${index}.delay_trigger`] ? 'error' : ''}
                              />
                              {errors[`cameras.${index}.delay_trigger`] && (
                                <span className="error-message">{errors[`cameras.${index}.delay_trigger`]}</span>
                              )}
                            </div>
                            <div className="form-group">
                              <label>Delay Interval (ms)</label>
                              <input
                                type="number"
                                value={camera.delay_interval}
                                onChange={(e) => handleCameraConfigChange(camera.camera_id, 'delay_interval', e.target.value)}
                                step="10"
                                min="0"
                                placeholder="500"
                                className={errors[`cameras.${index}.delay_interval`] ? 'error' : ''}
                              />
                              {errors[`cameras.${index}.delay_interval`] && (
                                <span className="error-message">{errors[`cameras.${index}.delay_interval`]}</span>
                              )}
                              <small className="form-help-text">Delay between frames (for multi-template)</small>
                            </div>
                            <div className="form-group">
                              <label>Gain <span className="required">*</span></label>
                              <input
                                type="number"
                                value={camera.gain}
                                onChange={(e) => handleCameraConfigChange(camera.camera_id, 'gain', e.target.value)}
                                step="0.1"
                                min="0"
                                className={errors[`cameras.${index}.gain`] ? 'error' : ''}
                              />
                              {errors[`cameras.${index}.gain`] && (
                                <span className="error-message">{errors[`cameras.${index}.gain`]}</span>
                              )}
                            </div>
                          </div>

                          <div className="form-row">
                            <div className="form-group">
                              <label>Pixel Format</label>
                              <select 
                                value={camera.pixel_format}
                                onChange={(e) => handleCameraConfigChange(camera.camera_id, 'pixel_format', e.target.value)}
                              >
                                <option value="Mono8">Mono8</option>
                                <option value="Mono12">Mono12</option>
                                <option value="RGB8">RGB8</option>
                                <option value="BGR8">BGR8</option>
                                <option value="YUV422">YUV422</option>
                                <option value="BayerRG8">BayerRG8</option>
                              </select> 
                            </div>
                          </div>

                          <h5>Trigger Configuration</h5>
                          <div className="form-row">
                            <div className="form-group">
                              <label>Trigger Mode <span className="required">*</span></label>
                              <select
                                value={camera.trigger_mode}
                                onChange={(e) => handleCameraConfigChange(camera.camera_id, 'trigger_mode', e.target.value)}
                              >
                                <option value="continuous">Continuous (Free-running) - TODO</option>
                                <option value="software_trigger">Software Trigger</option>
                                <option value="hardware_trigger" disabled>Hardware Trigger (I/O) - TODO</option>
                              </select>
                              <small style={{display: 'block', marginTop: 4, color: '#666'}}>
                                {camera.trigger_mode === 'continuous' && 'Camera captures continuously (not implemented yet)'}
                                {camera.trigger_mode === 'software_trigger' && 'Camera triggers on Digital Input (DI) signal'}
                                {camera.trigger_mode === 'hardware_trigger' && 'Camera triggers via hardware Line input (not implemented yet)'}
                              </small>
                            </div>
                          </div>

                          {camera.trigger_mode === 'software_trigger' && (
                            <>
                              <div className="form-row">
                                <div className="form-group">
                                  <label>Trigger Selector <span className="required">*</span></label>
                                  <select
                                    value={camera.trigger_config.trigger_selector}
                                    onChange={(e) => handleCameraTriggerConfigChange(camera.camera_id, 'trigger_selector', e.target.value)}
                                  >
                                    <option value="FrameStart">Frame Start</option>
                                    <option value="ExposureStart">Exposure Start</option>
                                    <option value="FrameBurstStart">Frame Burst Start</option>
                                  </select>
                                  <small style={{display: 'block', marginTop: 4, color: '#666'}}>
                                    Type of trigger event
                                  </small>
                                </div>
                              </div>

                              <div className="form-row">
                                <div className="form-group">
                                  <label>Digital Input (DI) Number <span className="required">*</span></label>
                                  <select
                                    value={camera.trigger_config.di_number.toString()}
                                    onChange={(e) => handleCameraTriggerConfigChange(camera.camera_id, 'di_number', parseInt(e.target.value))}
                                  >
                                    <option value="0">DI 0</option>
                                    <option value="1">DI 1</option>
                                    <option value="2">DI 2</option>
                                    <option value="3">DI 3</option>
                                  </select>
                                  <small style={{display: 'block', marginTop: 4, color: '#666'}}>
                                    Digital Input port to monitor (0-3)
                                  </small>
                                </div>
                                <div className="form-group">
                                  <label>Trigger Activation <span className="required">*</span></label>
                                  <select
                                    value={camera.trigger_config.trigger_activation}
                                    onChange={(e) => handleCameraTriggerConfigChange(camera.camera_id, 'trigger_activation', e.target.value)}
                                  >
                                    <option value="RisingEdge">Rising Edge (0 → 1)</option>
                                    <option value="FallingEdge">Falling Edge (1 → 0)</option>
                                    <option value="AnyEdge">Any Edge (Both)</option>
                                  </select>
                                  <small style={{display: 'block', marginTop: 4, color: '#666'}}>
                                    When to trigger capture
                                  </small>
                                </div>
                              </div>
                            </>
                          )}

                          {/* TODO: Hardware Trigger mode - Not implemented yet */}
                          {camera.trigger_mode === 'hardware_trigger' && (
                            <div style={{padding: '15px', background: '#fff3cd', border: '1px solid #ffc107', borderRadius: '4px', marginTop: '10px'}}>
                              <strong>⚠️ Hardware Trigger Mode (TODO)</strong>
                              <p style={{margin: '8px 0 0 0', fontSize: '14px'}}>
                                This mode is not implemented yet. It will use camera Line inputs (Line0-Line3) for hardware triggering.
                              </p>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {activeTab === 'model' && (
              <div className="form-section">
                {/* ── Top: OCR (left) + ML Quality (right) ── */}
                <div className="model-columns">
                  <div className="model-column">
                    <h3>OCR Model</h3>
                    <div className="form-group">
                      <label>OCR Model Type</label>
                      <select
                        name="ocr_model_type"
                        value={formData.ocr_model_type}
                        onChange={(e) => setFormData(prev => ({ ...prev, ocr_model_type: e.target.value }))}
                      >
                        <option value="">-- Default --</option>
                        <option value="SMTR">SMTR (large-x)</option>
                        <option value="SVTRV2_CTC">SVTRV2_CTC (large)</option>
                        <option value="OPENOCR_REPSVTR">OPENOCR_REPSVTR (medium)</option>
                        <option value="PADDLEV5">PADDLEV5 (small)</option>
                      </select>
                      {/* <small className="field-description">OCR recognition backbone used for text reading</small> */}
                    </div>
                    <div className="form-group">
                      <label>Defect Model</label>
                      <select
                        name="defect_model"
                        value={formData.defect_model}
                        onChange={(e) => setFormData(prev => ({ ...prev, defect_model: e.target.value }))}
                      >
                        <option value="arcface">ArcChar (class-anchored)</option>
                        <option value="supcon">SupCon (contrastive embedding)</option>
                      </select>
                      <small className="field-description">Template model used for per-character OK/NG classification</small>
                    </div>

                    {/* CV Method — chỉ áp dụng khi classifier_backend='embedding' */}
                    <div
                      className="form-group"
                      style={{ opacity: formData.classifier_backend === 'embedding' ? 1 : 0.55 }}
                    >
                      <label>CV Method</label>
                      <select
                        name="cv_method"
                        value={formData.cv_method}
                        disabled={formData.classifier_backend !== 'embedding'}
                        onChange={(e) => setFormData(prev => ({ ...prev, cv_method: e.target.value }))}
                      >
                        <option value="legacy">Pattern Match (Classic)</option>
                        <option value="v3">Ink Defect Detector</option>
                        <option value="v4">Ink Defect Detector (Scale-Tolerant)</option>
                        <option value="v5">Local Defect Detector (Tile-wise + Scale)</option>
                        <option value="shape_v7">Shape Outline Match</option>
                      </select>
                      {/* <small className="field-description">
                        <b>Pattern Match</b>: template similarity (blur TM + ECC-aligned IoU + pixel coverage).&nbsp;
                        <b>Ink Defect Detector</b>: detects over-ink (smudge) and under-ink (broken stroke) via directional pixel diff.&nbsp;
                        <b>Scale-Tolerant</b> variant: same but handles size variation via AFFINE alignment.&nbsp;
                        <b>Local Defect Detector</b>: splits char into 3×3 tiles + Scale-Tolerant — surfaces localized defects (small ink smudge, broken stroke piece) that get diluted in global metric.&nbsp;
                        <b>Shape Outline Match</b>: compares gradient orientation (LineMOD/Halcon style) — lighting-robust, shape-focused.
                      </small> */}

                      {/* Preview: 5 cặp char gần nhất với conf tính bằng method đang chọn */}
                      {formData.classifier_backend === 'embedding' && (
                        <div className="cv-preview">
                          <div className="cv-preview__header">
                            <span className="cv-preview__title">
                              Preview latest 5 pairs
                              {cvPreviewFolder && (
                                <span className="cv-preview__folder">— {cvPreviewFolder}</span>
                              )}
                            </span>
                            <button
                              type="button"
                              className="cv-preview__refresh"
                              onClick={() => {
                                setCvPreviewKeys(null);   // drop locked pairs → BE re-shuffles
                                fetchCvPreview(formData.cv_method || 'legacy', null);
                              }}
                              disabled={cvPreviewLoading}
                              title="Pick a new random set of pairs"
                            >
                              {cvPreviewLoading ? '...' : 'New pairs'}
                            </button>
                          </div>
                          {cvPreviewError && (
                            <div className="cv-preview__error">{cvPreviewError}</div>
                          )}
                          <div className="cv-preview__cards">
                            {cvPreviewPairs.length === 0 && !cvPreviewLoading && !cvPreviewError && (
                              <div className="cv-preview__empty">(no data)</div>
                            )}
                            {cvPreviewPairs.map((p) => {
                              const isOK = p.label === 'OK';
                              const extraStr = p.extra
                                ? Object.entries(p.extra).map(([k, v]) => `${k}=${v}`).join(' ')
                                : '';
                              return (
                                <div
                                  key={p.char_idx}
                                  className={`cv-preview__card ${isOK ? 'is-ok' : 'is-ng'}`}
                                  title={extraStr}
                                >
                                  <div className="cv-preview__thumbs">
                                    <img
                                      className="cv-preview__thumb"
                                      src={`data:image/png;base64,${p.tmpl_b64}`}
                                      alt="tmpl"
                                      title="Template"
                                    />
                                    <img
                                      className="cv-preview__thumb"
                                      src={`data:image/png;base64,${p.tgt_b64}`}
                                      alt="tgt"
                                      title="Target"
                                    />
                                    {p.result_b64 && (
                                      <img
                                        className="cv-preview__thumb"
                                        src={`data:image/png;base64,${p.result_b64}`}
                                        alt="diff"
                                        title="Diff / Result"
                                      />
                                    )}
                                  </div>
                                  <div className="cv-preview__meta">
                                    char{p.char_idx.toString().padStart(2, '0')}
                                  </div>
                                  <div className={`cv-preview__score ${isOK ? 'is-ok' : 'is-ng'}`}>
                                    {p.label} {p.conf.toFixed(2)}
                                  </div>
                                  {p.defect_type && (
                                    <div className="cv-preview__defect">{p.defect_type}</div>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* ── Char Denoise — largest-CC filter (embedding mode only) ── */}
                    {/* <div className={`template-bank-card ${formData.classifier_backend !== 'embedding' ? 'disabled' : ''}`}>
                      <label className="template-bank-card__title">
                        <input
                          type="checkbox"
                          checked={formData.char_denoise_enabled}
                          disabled={formData.classifier_backend !== 'embedding'}
                          onChange={(e) => setFormData(prev => ({
                            ...prev,
                            char_denoise_enabled: e.target.checked,
                          }))}
                        />
                        Char Denoise (largest-CC filter)
                      </label>
                    </div> */}

                    {/* ── Template Bank — adaptive multi-template (embedding mode only) ── */}
                    {/* <div className={`template-bank-card ${formData.classifier_backend !== 'embedding' ? 'disabled' : ''}`}>
                      <label className="template-bank-card__title">
                        <input
                          type="checkbox"
                          checked={formData.template_bank_enabled}
                          disabled={formData.classifier_backend !== 'embedding'}
                          onChange={(e) => setFormData(prev => ({
                            ...prev,
                            template_bank_enabled: e.target.checked,
                          }))}
                        />
                        Adaptive Template Bank
                      </label>

                      <div className="template-bank-card__row">
                        <label>Bank size:</label>
                        <input
                          type="number"
                          min={1}
                          max={50}
                          step={1}
                          value={formData.template_bank_size}
                          disabled={
                            !formData.template_bank_enabled ||
                            formData.classifier_backend !== 'embedding'
                          }
                          onChange={(e) => setFormData(prev => ({
                            ...prev,
                            template_bank_size: Math.max(1, Math.min(50, parseInt(e.target.value) || 10)),
                          }))}
                        />
                        <span className="unit">dynamic templates (1-50)</span>
                      </div>
                    </div> */}
                  </div>
                  <div className="model-column">
                    <h3>ML Quality Inspection</h3>
                    {/* Active method toggle — embedding vs ML trained model */}
                    <div className="form-group">
                      <label>Active Method</label>
                      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginTop: 4 }}>
                        {([
                          { value: 'embedding', label: 'Embedding (defect_model cosine)', hint: 'Pure template matching, no training' },
                          { value: 'ml',        label: 'ML Trained Model',                 hint: 'Use the trained classifier below' },
                        ] as const).map(opt => (
                          <label key={opt.value}
                            style={{ display: 'flex', alignItems: 'flex-start', gap: 6, cursor: 'pointer', flex: '1 1 220px' }}>
                            <input type="radio" name="classifier_backend"
                              value={opt.value}
                              checked={formData.classifier_backend === opt.value}
                              onChange={() => setFormData(prev => ({ ...prev, classifier_backend: opt.value }))} />
                            <div style={{ display: 'flex', flexDirection: 'column' }}>
                              <span style={{ fontSize: 13, fontWeight: formData.classifier_backend === opt.value ? 600 : 400 }}>
                                {opt.label}
                              </span>
                              <span style={{ fontSize: 10, opacity: 0.65 }}>{opt.hint}</span>
                            </div>
                          </label>
                        ))}
                      </div>
                    </div>
                    <div className="form-group" style={{ opacity: formData.classifier_backend === 'ml' ? 1 : 0.55 }}>
                      <label>AI Training Project {formData.classifier_backend === 'ml' && <span className="required">*</span>}</label>
                      <select
                        value={formData.ml_project_id}
                        onChange={(e) => setFormData(prev => ({ ...prev, ml_project_id: e.target.value, ml_model_id: '' }))}
                        disabled={loadingMlProjects || formData.classifier_backend !== 'ml'}
                      >
                        <option value="">-- None --</option>
                        {mlProjects.map(p => (
                          <option key={p.id} value={p.id}>{p.name}</option>
                        ))}
                      </select>
                      {loadingMlProjects && <small className="field-description">Loading projects…</small>}
                    </div>
                    <div className="form-group" style={{ opacity: formData.classifier_backend === 'ml' ? 1 : 0.55 }}>
                      <label>Trained Model {formData.classifier_backend === 'ml' && <span className="required">*</span>}</label>
                      <select
                        value={formData.ml_model_id}
                        onChange={(e) => setFormData(prev => ({ ...prev, ml_model_id: e.target.value }))}
                        disabled={!formData.ml_project_id || loadingMlModels || formData.classifier_backend !== 'ml'}
                      >
                        <option value="">-- None --</option>
                        {mlModels.map(m => (
                          <option key={m.id} value={m.id}>
                            {m.algorithm.toUpperCase()} · acc {(m.metrics.accuracy_test * 100).toFixed(1)}% · {new Date(m.created_at).toLocaleDateString()}
                          </option>
                        ))}
                      </select>
                      {loadingMlModels && <small className="field-description">Loading models…</small>}
                      {formData.ml_project_id && !loadingMlModels && mlModels.length === 0 && (
                        <small className="field-description" style={{ color: 'var(--color-warning, #f59e0b)' }}>No completed models in this project yet</small>
                      )}
                      {/* ── Char coverage warning ─────────────────────────── */}
                      {formData.ml_model_id && recipeChars.length > 0 && (
                        <div style={{ marginTop: 8, padding: 8, borderRadius: 4,
                          background: mlCoverage && mlCoverage.missing.length === 0
                            ? 'rgba(34,197,94,.12)'
                            : 'rgba(245,158,11,.12)',
                          border: `1px solid ${
                            mlCoverage && mlCoverage.missing.length === 0
                              ? 'rgba(34,197,94,.35)' : 'rgba(245,158,11,.35)'
                          }`,
                          fontSize: 12,
                        }}>
                          {loadingMlCoverage ? (
                            <span>Checking char coverage…</span>
                          ) : mlCoverage ? (
                            <>
                              <div>
                                <strong>Coverage: {mlCoverage.pct.toFixed(0)}%</strong>
                                <span style={{ marginLeft: 8, color: '#9ca3af' }}>
                                  ({mlCoverage.covered.length}/{recipeChars.length} chars)
                                </span>
                              </div>
                              {mlCoverage.missing.length > 0 && (
                                <div style={{ marginTop: 4 }}>
                                  <span style={{ color: 'var(--color-warning, #f59e0b)' }}>
                                    ⚠️ Missing: {mlCoverage.missing.map(c => `"${c}"`).join(', ')}
                                  </span>
                                  <br />
                                  {/* <small style={{ color: '#9ca3af' }}>
                                    ML check sẽ bị SKIP cho các bbox có chars này. Label & retrain project nếu cần full coverage.
                                  </small> */}
                                </div>
                              )}
                              {mlCoverage.missing.length === 0 && (
                                <div style={{ color: 'var(--color-success, #22c55e)', marginTop: 2 }}>
                                  ✓ All recipe chars covered by this model
                                </div>
                              )}
                            </>
                          ) : (
                            <small>Coverage check unavailable</small>
                          )}
                        </div>
                      )}
                    </div>

                    {/* ─── Bottle Edge Detection group ────────────────────── */}
                    <div className="bottle-edge-card">
                      <div className="bottle-edge-card__title">
                        Bottle Edge Detection
                      </div>

                      {/* Row 1: Method + Product Box Wall Type */}
                      <div className="bottle-edge-row bottle-edge-row--2col">
                        <div className="form-group">
                          <label>Method</label>
                          <select
                            name="product_detection_method"
                            value={formData.product_detection_method}
                            onChange={(e) =>
                              setFormData(prev => ({ ...prev, product_detection_method: e.target.value }))
                            }
                          >
                            <option value="yolo_obb">YOLO OBB (trained model)</option>
                            <option value="yolo_segment">YOLO Segment (trained model)</option>
                          </select>
                        </div>

                        <div
                          className={
                            'form-group' +
                            (formData.product_detection_method === 'yolo_segment'
                              ? ''
                              : ' bottle-edge-disabled')
                          }
                        >
                          <label>Product Box Wall Type</label>
                          <select
                            name="product_box_wall_type"
                            value={formData.product_box_wall_type}
                            onChange={(e) =>
                              setFormData(prev => ({ ...prev, product_box_wall_type: e.target.value }))
                            }
                          >
                            <option value="outer">Outer wall</option>
                            <option value="inner">Inner wall</option>
                          </select>
                        </div>
                      </div>

                      {/* Row 2: Cap Rotation Method + Cap Crop Method */}
                      <div className="bottle-edge-row bottle-edge-row--2col">
                        <div className="form-group">
                          <label>Cap Rotation Method</label>
                          <select
                            name="cap_rotation_method"
                            value={formData.cap_rotation_method}
                            onChange={(e) =>
                              setFormData(prev => ({ ...prev, cap_rotation_method: e.target.value }))
                            }
                          >
                            <option value="yolo_obb">YOLO OBB</option>
                            <option value="yolo_segment">YOLO Segment</option>
                          </select>
                        </div>

                        <div className="form-group">
                          <label>Cap Crop Method</label>
                          <select
                            name="cap_crop_method"
                            value={formData.cap_crop_method}
                            onChange={(e) =>
                              setFormData(prev => ({ ...prev, cap_crop_method: e.target.value }))
                            }
                          >
                            <option value="none">None (use crop_area)</option>
                            <option value="yolo_obb">YOLO OBB</option>
                            <option value="yolo_segment">YOLO Segment</option>
                          </select>
                        </div>
                      </div>

                      {/* Row 3: Crop Match Method (full row) */}
                      <div className="bottle-edge-row bottle-edge-row--1col">
                        <div className="form-group">
                          <label>Crop Match Method</label>
                          <select
                            name="crop_match_method"
                            value={formData.crop_match_method}
                            onChange={(e) =>
                              setFormData(prev => ({ ...prev, crop_match_method: e.target.value }))
                            }
                          >
                            <option value="superpoint">SuperPoint</option>
                            <option value="shape_outline">Shape Outline</option>
                          </select>
                        </div>
                      </div>

                      {/* Row 4: Dual Rotation Check (checkbox, full row) */}
                      <div className="bottle-edge-checkbox">
                        <label className="bottle-edge-checkbox__label">
                          <input
                            type="checkbox"
                            checked={formData.dual_rotation_check}
                            onChange={(e) =>
                              setFormData(prev => ({ ...prev, dual_rotation_check: e.target.checked }))
                            }
                          />
                          <span>Dual Rotation Check (Check_Color only — try both rotations, pick higher match)</span>
                        </label>
                      </div>
                    </div>
                  </div>
                </div>

                {/* ── Model Thresholds ── HIDDEN: not wired to AI service yet (dead config) ── */}
                {false && (
                  <>
                    <h3>Model Thresholds</h3>
                    <div className="thresholds-grid">
                      <div className="form-group">
                        <label>Detection <span className="required">*</span></label>
                        <input type="number" value={formData.model_thresholds.detection_threshold}
                               onChange={(e) => handleModelThresholdChange('detection_threshold', e.target.value)}
                               step="0.01" min="0" max="1" placeholder="0.0 - 1.0"
                               className={errors['model_thresholds.detection_threshold'] ? 'error' : ''} />
                        {errors['model_thresholds.detection_threshold'] && (
                          <span className="error-message">{errors['model_thresholds.detection_threshold']}</span>
                        )}
                      </div>
                      <div className="form-group">
                        <label>Recognition <span className="required">*</span></label>
                        <input type="number" value={formData.model_thresholds.recognition_threshold}
                               onChange={(e) => handleModelThresholdChange('recognition_threshold', e.target.value)}
                               step="0.01" min="0" max="1" placeholder="0.0 - 1.0"
                               className={errors['model_thresholds.recognition_threshold'] ? 'error' : ''} />
                        {errors['model_thresholds.recognition_threshold'] && (
                          <span className="error-message">{errors['model_thresholds.recognition_threshold']}</span>
                        )}
                      </div>
                      <div className="form-group">
                        <label>Matching <span className="required">*</span></label>
                        <input type="number" value={formData.model_thresholds.matching_threshold}
                               onChange={(e) => handleModelThresholdChange('matching_threshold', e.target.value)}
                               step="0.01" min="0" max="1" placeholder="0.0 - 1.0"
                               className={errors['model_thresholds.matching_threshold'] ? 'error' : ''} />
                        {errors['model_thresholds.matching_threshold'] && (
                          <span className="error-message">{errors['model_thresholds.matching_threshold']}</span>
                        )}
                      </div>
                      <div className="form-group">
                        <label>Min Text (px)</label>
                        <input type="number" value={formData.model_thresholds.min_text_size || ''}
                               onChange={(e) => handleModelThresholdChange('min_text_size', e.target.value)} min="1"
                               className={errors['model_thresholds.min_text_size'] ? 'error' : ''} />
                        {errors['model_thresholds.min_text_size'] && (
                          <span className="error-message">{errors['model_thresholds.min_text_size']}</span>
                        )}
                      </div>
                      <div className="form-group">
                        <label>Max Text (px)</label>
                        <input type="number" value={formData.model_thresholds.max_text_size || ''}
                               onChange={(e) => handleModelThresholdChange('max_text_size', e.target.value)} min="1"
                               className={errors['model_thresholds.max_text_size'] ? 'error' : ''} />
                        {errors['model_thresholds.max_text_size'] && (
                          <span className="error-message">{errors['model_thresholds.max_text_size']}</span>
                        )}
                      </div>
                    </div>
                  </>
                )}

                {/* ── Matching Confidence + Wrinkle Detection (50/50 row) ── */}
                <div className="model-columns">
                  <div className="model-column">
                    <h3>Matching Confidence</h3>
                    <div className="form-group">
                      <label>
                        Threshold
                        <span style={{ marginLeft: 12, fontWeight: 600, fontFamily: 'monospace' }}>
                          {((formData.matching_conf ?? 0.20) * 100).toFixed(0)}%
                        </span>
                      </label>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.01}
                          value={formData.matching_conf ?? 0.20}
                          onChange={(e) => setFormData(prev => ({ ...prev, matching_conf: parseFloat(e.target.value) }))}
                          style={{ flex: 1 }}
                        />
                        <input
                          type="number"
                          min={0}
                          max={1}
                          step={0.01}
                          value={formData.matching_conf ?? 0.20}
                          onChange={(e) => {
                            const v = Math.max(0, Math.min(1, parseFloat(e.target.value) || 0));
                            setFormData(prev => ({ ...prev, matching_conf: v }));
                          }}
                          style={{ width: 80 }}
                        />
                      </div>
                      <small className="field-description">
                        Skip OCR/verify when SuperPoint inlier ratio falls below this threshold. Higher = stricter (fewer false PASS, more FAIL).
                      </small>
                    </div>
                    <div className="form-group" style={{ marginTop: 8 }}>
                      <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                        <input
                          type="checkbox"
                          checked={formData.match_erosion_enabled ?? false}
                          onChange={(e) => setFormData(prev => ({ ...prev, match_erosion_enabled: e.target.checked }))}
                        />
                        <span>Horizontal Erosion Pre-processing</span>
                      </label>
                      <small className="field-description">
                        Apply morphological erosion horizontally before SuperPoint matching to suppress variable date code text.
                      </small>
                    </div>
                    {(formData.match_erosion_enabled ?? false) && (
                      <>
                        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                          <div className="form-group" style={{ flex: 1, minWidth: 100 }}>
                            <label>Kernel Width (px)</label>
                            <input
                              type="number"
                              min={1}
                              max={300}
                              value={formData.match_erosion_kernel_w ?? 80}
                              onChange={(e) => setFormData(prev => ({ ...prev, match_erosion_kernel_w: Math.max(1, Math.min(300, parseInt(e.target.value) || 1)) }))}
                            />
                          </div>
                          <div className="form-group" style={{ flex: 1, minWidth: 100 }}>
                            <label>Kernel Height (px)</label>
                            <input
                              type="number"
                              min={1}
                              max={50}
                              value={formData.match_erosion_kernel_h ?? 1}
                              onChange={(e) => setFormData(prev => ({ ...prev, match_erosion_kernel_h: Math.max(1, Math.min(50, parseInt(e.target.value) || 1)) }))}
                            />
                          </div>
                          <div className="form-group" style={{ flex: 1, minWidth: 100 }}>
                            <label>Iterations</label>
                            <input
                              type="number"
                              min={1}
                              max={5}
                              value={formData.match_erosion_iterations ?? 1}
                              onChange={(e) => setFormData(prev => ({ ...prev, match_erosion_iterations: Math.max(1, Math.min(5, parseInt(e.target.value) || 1)) }))}
                            />
                          </div>
                        </div>
                        <div style={{ marginTop: 10, borderRadius: 8, overflow: 'hidden', border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(0,0,0,0.25)' }}>
                          <div style={{ padding: '6px 10px 6px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                            <span style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#475569' }}>Effect Preview</span>
                            <button
                              type="button"
                              onClick={computeErosionPreview}
                              disabled={isComputingPreview}
                              style={{ fontSize: 11, padding: '3px 10px', borderRadius: 5, border: '1px solid rgba(59,130,246,0.5)', background: 'rgba(59,130,246,0.15)', color: '#60a5fa', cursor: 'pointer', fontWeight: 600, opacity: isComputingPreview ? 0.6 : 1 }}
                            >
                              {isComputingPreview ? 'Computing…' : 'Preview'}
                            </button>
                          </div>
                          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1, background: 'rgba(255,255,255,0.05)' }}>
                            <div style={{ padding: '8px 10px', background: 'rgba(0,0,0,0.2)' }}>
                              <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6, fontWeight: 600 }}>Original</div>
                              <img src={erosionBeforeImg} alt="Before erosion" style={{ width: '100%', borderRadius: 4, display: 'block' }} />
                            </div>
                            <div style={{ padding: '8px 10px', background: 'rgba(0,0,0,0.2)' }}>
                              <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6, fontWeight: 600 }}>After Erosion</div>
                              <img src={erosionPreviewUrl} alt="After erosion" style={{ width: '100%', borderRadius: 4, display: 'block', opacity: isComputingPreview ? 0.4 : 1, transition: 'opacity 0.2s' }} />
                            </div>
                          </div>
                          <div style={{ padding: '6px 10px', fontSize: 11, color: '#475569', fontStyle: 'italic' }}>
                            Date code numbers merge into uniform dark bands — structural layout preserved for SuperPoint matching.
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                  <div className="model-column">
                    <h3>Wrinkle Detection</h3>
                    <div className="form-group">
                      <label>
                        Confidence Threshold
                        <span style={{ marginLeft: 12, fontWeight: 600, fontFamily: 'monospace' }}>
                          {(formData.wrinkle_conf ?? 0.25).toFixed(2)}
                        </span>
                      </label>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.01}
                          value={formData.wrinkle_conf ?? 0.25}
                          onChange={(e) => setFormData(prev => ({ ...prev, wrinkle_conf: parseFloat(e.target.value) }))}
                          style={{ flex: 1 }}
                        />
                        <input
                          type="number"
                          min={0}
                          max={1}
                          step={0.01}
                          value={formData.wrinkle_conf ?? 0.25}
                          onChange={(e) => {
                            const v = Math.max(0, Math.min(1, parseFloat(e.target.value) || 0));
                            setFormData(prev => ({ ...prev, wrinkle_conf: v }));
                          }}
                          style={{ width: 80 }}
                        />
                      </div>
                      <small className="field-description">
                        Detection score threshold for the wrinkle segmentation model. Lower = more sensitive (more regions kept), higher = stricter.
                      </small>
                    </div>
                    <div className="form-group" style={{ marginTop: 8 }}>
                      <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                        <input
                          type="checkbox"
                          checked={formData.wrinkle_show_when_pass ?? true}
                          onChange={(e) => setFormData(prev => ({ ...prev, wrinkle_show_when_pass: e.target.checked }))}
                        />
                        <span>Show wrinkle regions even when frame passes</span>
                      </label>
                      <small className="field-description">
                        On → draws wrinkle contour on the result image even when the bottle PASSes (debug/monitoring). Off → draws only on FAIL.
                      </small>
                    </div>
                    <div className="form-group" style={{ marginTop: 8 }}>
                      <label>
                        Mask Overlap Threshold
                        <span style={{ marginLeft: 12, fontWeight: 600, fontFamily: 'monospace' }}>
                          {((formData.mask_overlap_threshold ?? 0.6) * 100).toFixed(0)}%
                        </span>
                      </label>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.05}
                          value={formData.mask_overlap_threshold ?? 0.6}
                          onChange={(e) => setFormData(prev => ({ ...prev, mask_overlap_threshold: parseFloat(e.target.value) }))}
                          style={{ flex: 1 }}
                        />
                        <input
                          type="number"
                          min={0}
                          max={1}
                          step={0.05}
                          value={formData.mask_overlap_threshold ?? 0.6}
                          onChange={(e) => {
                            const v = Math.max(0, Math.min(1, parseFloat(e.target.value) || 0));
                            setFormData(prev => ({ ...prev, mask_overlap_threshold: v }));
                          }}
                          style={{ width: 80 }}
                        />
                      </div>
                      <small className="field-description">
                        Wrinkle regions with this fraction (or more) of pixels inside a "mask" annotation are excluded from the wrinkle check.
                      </small>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'template' && (
              <div className="form-section template-section">
                <div className="template-header">
                  <h3>Template Configuration</h3>
                  <div className="template-camera-selector">
                    <label>Select Camera:</label>
                    <select
                      value={selectedCameraForTemplate}
                      onChange={(e) => setSelectedCameraForTemplate(e.target.value)}
                      disabled={formData.cameras.length === 0}
                    >
                      <option value="">-- Select Camera --</option>
                      {formData.cameras.map((cam: any) => (
                        <option key={cam.camera_id} value={cam.camera_id}>
                          {cam.camera_id} - {cam.model_name}
                        </option>
                      ))}
                    </select>
                  </div>
                  {selectedCameraForTemplate && (
                    <div className="template-function-type-selector">
                      <label>Function Type:</label>
                      <select
                        value={cameraFunctionTypes[selectedCameraForTemplate] || 'Check_Type_Product'}
                        onChange={(e) => {
                          setCameraFunctionTypes(prev => ({
                            ...prev,
                            [selectedCameraForTemplate]: e.target.value
                          }));
                        }}
                      >
                        {/* <option value="OCR">OCR (Text Recognition)</option> */}
                        <option value="Check_Type_Product">Check Wrinkle/OCR</option>
                        <option value="Check_Color">Check Color/OCR</option>
                        {/* <option value="Check_Defect">Check Defect</option>
                        <option value="Check_Position">Check Position</option>
                        <option value="Barcode_Detection">Barcode Detection</option>
                        <option value="DateCode_Detection">Date Code Detection</option>  */}
                      </select>
                      {/* <small style={{display: 'block', marginTop: 4, color: '#666'}}>
                        All templates for this camera will use this function type
                      </small> */}
                    </div>
                  )}
                </div>

                {formData.cameras.length === 0 ? (
                  <div className="template-placeholder">
                    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                    <p>Please add cameras in the Camera tab first</p>
                    <p className="hint">Template configuration requires at least one camera to be selected</p>
                  </div>
                ) : !selectedCameraForTemplate ? (
                  <div className="template-placeholder">
                    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <rect x="3" y="3" width="18" height="18" rx="2" strokeWidth="2"/>
                      <circle cx="8.5" cy="8.5" r="1.5" fill="currentColor"/>
                      <polyline points="21,15 16,10 5,21" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                    <p>Select a camera to configure templates</p>
                    <p className="hint">Choose a camera from the dropdown above</p>
                  </div>
                ) : (
                  <>
                    <div className="template-actions">
                      <div className="template-add-buttons">
                        <input
                          type="file"
                          id={`template-upload-${selectedCameraForTemplate}`}
                          accept="image/*"
                          onChange={handleImageUpload}
                          style={{ display: 'none' }}
                        />
                        <label
                          htmlFor={`template-upload-${selectedCameraForTemplate}`}
                          className="btn btn-secondary"
                        >
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" style={{ marginRight: '6px' }}>
                            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                          </svg>
                          Upload Image
                        </label>

                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <label style={{ fontSize: '14px', fontWeight: 500 }}>Frames:</label>
                          <input
                            type="number"
                            min="1"
                            max="5"
                            value={frameCount}
                            onChange={(e) => setFrameCount(Math.min(5, Math.max(1, parseInt(e.target.value) || 1)))}
                            style={{
                              width: '60px',
                              padding: '6px 8px',
                              border: '1px solid #d1d5db',
                              borderRadius: '4px',
                              fontSize: '14px'
                            }}
                            disabled={isGettingFrame}
                          />
                        </div>

                        <button
                          type="button"
                          className="btn btn-secondary"
                          onClick={handleGetMultipleFrames}
                          disabled={isGettingFrame}
                          title={`Get ${frameCount} frame${frameCount > 1 ? 's' : ''} from camera ring buffer`}
                        >
                          {isGettingFrame ? (
                            <>
                              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="spin">
                                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" opacity="0.25"/>
                                <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="2"/>
                              </svg>
                              Getting Frames...
                            </>
                          ) : (
                            <>
                              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" style={{ marginRight: '6px' }}>
                                <rect x="2" y="4" width="20" height="16" rx="2" stroke="currentColor" strokeWidth="2"/>
                                <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2"/>
                                <path d="M7 4V2M17 4V2" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                              </svg>
                              Get {frameCount} Frame{frameCount > 1 ? 's' : ''}
                            </>
                          )}
                        </button>

                        <label
                          style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', fontSize: '13px', userSelect: 'none' }}
                          title="Detect bottle orientation via OBB model and rotate frames upright before creating templates"
                        >
                          <input
                            type="checkbox"
                            checked={autoRotate}
                            onChange={(e) => setAutoRotate(e.target.checked)}
                            disabled={isGettingFrame}
                            style={{ width: '14px', height: '14px', cursor: 'pointer' }}
                          />
                          Auto-rotate
                        </label>
                      </div>
                      <span className="template-info">
                        Camera: {selectedCameraForTemplate} |
                        Templates: {cameraTemplates[selectedCameraForTemplate]?.length || 0}
                      </span>
                    </div>

                    {/* 3-column layout: Filmstrip | Canvas | Annotations */}
                    <div className="template-workspace">
                      {/* Filmstrip sidebar */}
                      {(cameraTemplates[selectedCameraForTemplate]?.length || 0) > 0 && (
                        <div className={`template-filmstrip ${filmstripExpanded ? 'expanded' : 'collapsed'}`}>
                          <button
                            type="button"
                            className="filmstrip-toggle"
                            onClick={() => setFilmstripExpanded(prev => !prev)}
                            title={filmstripExpanded ? 'Collapse' : 'Expand'}
                          >
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                              <path d={filmstripExpanded ? 'M15 18l-6-6 6-6' : 'M9 6l6 6-6 6'} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                            </svg>
                          </button>
                          <div className="filmstrip-list">
                            {cameraTemplates[selectedCameraForTemplate]?.map((template, idx) => {
                              const hasTemplateRegion = template.annotations.some((ann: any) => ann.type === 'template');
                              const hasRequiredAnnotation = template.annotations.some((ann: any) =>
                                ['text', 'barcode', 'datecode'].includes(ann.type)
                              );
                              const hasProductAnn = template.annotations.some((ann: any) => ann.type === 'product');
                              const hasCropArea = template.annotations.some((ann: any) => ann.type === 'crop_area');
                              // Check_Color templates: skip the "template" region requirement
                              // entirely. Need either a 'product' (color check) or text/datecode
                              // (rotate-OCR sub-mode) to be considered valid.
                              const isCheckColorCam = cameraFunctionTypes[selectedCameraForTemplate] === 'Check_Color';
                              const isValid = isCheckColorCam
                                ? (hasProductAnn || hasRequiredAnnotation)
                                : (hasTemplateRegion && hasRequiredAnnotation);

                              return (
                                <div
                                  key={template.id}
                                  className={`filmstrip-item ${idx === selectedTemplateIndex ? 'active' : ''} ${!isValid ? 'invalid' : ''}`}
                                  onClick={() => setSelectedTemplateIndex(idx)}
                                >
                                  <div className="filmstrip-thumbnail">
                                    <img src={template.image} alt={template.name} />
                                    <span className="filmstrip-index">{idx + 1}</span>
                                    {!isValid && <span className="filmstrip-warn" title="Missing required regions">!</span>}
                                  </div>

                                  {filmstripExpanded && (
                                    <div className="filmstrip-details">
                                      <input
                                        type="text"
                                        value={template.name}
                                        onChange={(e) => { e.stopPropagation(); handleRenameTemplate(idx, e.target.value); }}
                                        onClick={(e) => e.stopPropagation()}
                                        className="filmstrip-name-input"
                                      />
                                      <div className="filmstrip-settings">
                                        {(() => {
                                          const hasLabel = template.annotations?.some(a => a.type === 'label');
                                          const hasProduct = template.annotations?.some(a => a.type === 'product');
                                          const ft = cameraFunctionTypes[selectedCameraForTemplate] || 'Check_Type_Product';

                                          // Check_Color: only "Setup Color" matters, and only when product polygon is drawn.
                                          if (ft === 'Check_Color') {
                                            if (!hasProduct) {
                                              return (
                                                <div className="filmstrip-setting-hint" style={{ fontSize: 11, opacity: 0.7, padding: '4px 6px', fontStyle: 'italic' }}>
                                                  ⓘ Add a 'product' annotation to set up HSV color check.
                                                </div>
                                              );
                                            }
                                            const configured = !!template.color_config;
                                            return (
                                              <div className="filmstrip-setting-row" style={{ gap: 6 }}>
                                                <button
                                                  type="button"
                                                  className="filmstrip-action-btn"
                                                  onClick={(e) => {
                                                    e.stopPropagation();
                                                    setColorSetupTarget({ cameraId: selectedCameraForTemplate, templateIdx: idx });
                                                  }}
                                                  title="Setup HSV color check"
                                                  style={{ flex: 1, padding: '4px 8px', fontSize: 11, fontWeight: 500, justifyContent: 'center' }}
                                                >
                                                  Setup Color
                                                </button>
                                                <span
                                                  style={{
                                                    fontSize: 10,
                                                    fontWeight: 600,
                                                    padding: '2px 6px',
                                                    borderRadius: 4,
                                                    background: configured ? '#dcfce7' : '#fef3c7',
                                                    color: configured ? '#166534' : '#92400e',
                                                  }}
                                                >
                                                  {configured ? 'Configured' : 'Not set'}
                                                </span>
                                              </div>
                                            );
                                          }

                                          // Check_Type_Product (and others): existing offset/wrinkle UI requires both product+label.
                                          if (!hasLabel || !hasProduct) {
                                            return (
                                              <div className="filmstrip-setting-hint" style={{ fontSize: 11, opacity: 0.7, padding: '4px 6px', fontStyle: 'italic' }}>
                                                ⓘ Add 'product' and 'label' annotations to configure offset/wrinkle thresholds.
                                              </div>
                                            );
                                          }
                                          return (<>
                                        <div className="filmstrip-setting-row">
                                          <span className="filmstrip-setting-label">Offset L/R:</span>
                                          {(() => {
                                            const unit = template.center_offset_unit ?? 'px';
                                            const isPct = unit === 'pct';
                                            const maxVal = isPct ? 100 : 500;
                                            const stepVal = isPct ? 0.5 : 1;
                                            return (
                                              <>
                                                <input type="number" min="0" max={maxVal} step={stepVal}
                                                  value={template.center_offset_threshold_left ?? (isPct ? 25 : 50)}
                                                  onChange={(e) => { e.stopPropagation(); const v = Math.max(0, Math.min(maxVal, parseFloat(e.target.value) || 0)); setCameraTemplates(prev => ({ ...prev, [selectedCameraForTemplate]: prev[selectedCameraForTemplate]?.map((t, i) => i === idx ? { ...t, center_offset_threshold_left: v } : t) || [] })); }}
                                                  onClick={(e) => e.stopPropagation()}
                                                  className="filmstrip-setting-input"
                                                  title={`Left offset threshold (${isPct ? '% of label width' : 'pixels'})`}
                                                />
                                                <input type="number" min="0" max={maxVal} step={stepVal}
                                                  value={template.center_offset_threshold_right ?? (isPct ? 25 : 50)}
                                                  onChange={(e) => { e.stopPropagation(); const v = Math.max(0, Math.min(maxVal, parseFloat(e.target.value) || 0)); setCameraTemplates(prev => ({ ...prev, [selectedCameraForTemplate]: prev[selectedCameraForTemplate]?.map((t, i) => i === idx ? { ...t, center_offset_threshold_right: v } : t) || [] })); }}
                                                  onClick={(e) => e.stopPropagation()}
                                                  className="filmstrip-setting-input"
                                                  title={`Right offset threshold (${isPct ? '% of label width' : 'pixels'})`}
                                                />
                                                <select
                                                  value={unit}
                                                  onChange={(e) => {
                                                    e.stopPropagation();
                                                    const newUnit = e.target.value as 'px' | 'pct';
                                                    setCameraTemplates(prev => ({
                                                      ...prev,
                                                      [selectedCameraForTemplate]: prev[selectedCameraForTemplate]?.map((t, i) => {
                                                        if (i !== idx) return t;
                                                        // Reset thresholds to sensible defaults for the new unit
                                                        const defaultVal = newUnit === 'pct' ? 25 : 50;
                                                        return {
                                                          ...t,
                                                          center_offset_unit: newUnit,
                                                          center_offset_threshold_left: defaultVal,
                                                          center_offset_threshold_right: defaultVal,
                                                        };
                                                      }) || [],
                                                    }));
                                                  }}
                                                  onClick={(e) => e.stopPropagation()}
                                                  className="filmstrip-setting-input"
                                                  style={{ minWidth: 56 }}
                                                  title="Unit: pixels or percent of label reference width"
                                                >
                                                  <option value="px">px</option>
                                                  <option value="pct">%</option>
                                                </select>
                                              </>
                                            );
                                          })()}
                                        </div>
                                        <div className="filmstrip-setting-row">
                                          <span className="filmstrip-setting-label">Wrinkle Total:</span>
                                          <input type="number" min="0" max="100000" step="100"
                                            value={template.wrinkle_area ?? 2000}
                                            onChange={(e) => { e.stopPropagation(); const v = Math.max(0, parseFloat(e.target.value) || 0); setCameraTemplates(prev => ({ ...prev, [selectedCameraForTemplate]: prev[selectedCameraForTemplate]?.map((t, i) => i === idx ? { ...t, wrinkle_area: v } : t) || [] })); }}
                                            onClick={(e) => e.stopPropagation()}
                                            className="filmstrip-setting-input wide"
                                            title="Total wrinkle area threshold — sum of valid regions ≥ this value → FAIL (px²)"
                                          />
                                        </div>
                                        <div className="filmstrip-setting-row">
                                          <span className="filmstrip-setting-label">Wrinkle Min:</span>
                                          <input type="number" min="0" max="100000" step="100"
                                            value={template.wrinkle_min_area ?? 0}
                                            onChange={(e) => { e.stopPropagation(); const v = Math.max(0, parseFloat(e.target.value) || 0); setCameraTemplates(prev => ({ ...prev, [selectedCameraForTemplate]: prev[selectedCameraForTemplate]?.map((t, i) => i === idx ? { ...t, wrinkle_min_area: v } : t) || [] })); }}
                                            onClick={(e) => e.stopPropagation()}
                                            className="filmstrip-setting-input wide"
                                            title="Per-region min area — regions smaller than this are ignored (0 = no filter, px²)"
                                          />
                                        </div>
                                        <div className="filmstrip-setting-row">
                                          <span className="filmstrip-setting-label">Wrinkle Max:</span>
                                          <input type="number" min="0" max="100000" step="100"
                                            value={template.wrinkle_max_area ?? 0}
                                            onChange={(e) => { e.stopPropagation(); const v = Math.max(0, parseFloat(e.target.value) || 0); setCameraTemplates(prev => ({ ...prev, [selectedCameraForTemplate]: prev[selectedCameraForTemplate]?.map((t, i) => i === idx ? { ...t, wrinkle_max_area: v } : t) || [] })); }}
                                            onClick={(e) => e.stopPropagation()}
                                            className="filmstrip-setting-input wide"
                                            title="Per-region critical — any region ≥ this triggers FAIL immediately (0 = disabled, px²)"
                                          />
                                        </div>
                                        </>);
                                        })()}
                                        <div className="filmstrip-stats">
                                          {hasTemplateRegion && <span className="stat">T</span>}
                                          <span className="stat">{template.annotations.filter(a => ['text', 'barcode', 'datecode'].includes(a.type!)).length}R</span>
                                          {hasCropArea && <span className="stat crop">C</span>}
                                        </div>
                                      </div>
                                      <div className="filmstrip-actions">
                                        <button type="button" className="filmstrip-action-btn" onClick={(e) => { e.stopPropagation(); handleMoveTemplate(idx, 'up'); }} disabled={idx === 0} title="Move up">
                                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none"><path d="M12 19V5M5 12l7-7 7 7" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
                                        </button>
                                        <button type="button" className="filmstrip-action-btn" onClick={(e) => { e.stopPropagation(); handleMoveTemplate(idx, 'down'); }} disabled={idx === (cameraTemplates[selectedCameraForTemplate]?.length || 0) - 1} title="Move down">
                                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none"><path d="M12 5v14M19 12l-7 7-7-7" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
                                        </button>
                                        <button type="button" className="filmstrip-action-btn" onClick={(e) => { e.stopPropagation(); handleRotateTemplate(idx); }} disabled={rotatingTemplateIdx !== null} title="Rotate">
                                          {rotatingTemplateIdx === idx ? (
                                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" className="spin"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" opacity="0.25"/><path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="2"/></svg>
                                          ) : (
                                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none"><path d="M4 12a8 8 0 018-8V2l4 4-4 4V8a6 6 0 100 6" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
                                          )}
                                        </button>
                                        <button type="button" className="filmstrip-action-btn danger" onClick={(e) => { e.stopPropagation(); setConfirmDialog({ isOpen: true, title: 'Delete Template', message: `Delete ${template.name}?`, type: 'danger', onConfirm: () => handleDeleteTemplate(idx) }); }} title="Delete" style={{ display: canPerformAction('deleteTemplate', 'template') ? 'flex' : 'none' }}>
                                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none"><path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"/></svg>
                                        </button>
                                      </div>
                                    </div>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}

                      {/* Canvas + Annotations */}
                      {getCurrentTemplateImage() ? (
                        <div className="template-editor-layout">
                          <div className="template-editor-canvas">
                            <TemplateEditor
                              templateImage={getCurrentTemplateImage()}
                              annotations={getCurrentAnnotations() as any}
                              onAnnotationsChange={handleAnnotationsChange}
                              selectedAnnotation={selectedAnnotation}
                              onSelectAnnotation={setSelectedAnnotation}
                              fabricCanvasRef={fabricCanvasRef}
                            />
                          </div>
                          <div className="template-editor-sidebar">
                            <AnnotationsPanel
                              annotations={getCurrentAnnotations() as any}
                              selectedAnnotation={selectedAnnotation}
                              onSelectAnnotation={setSelectedAnnotation}
                              onAnnotationTypeChange={handleAnnotationTypeChange}
                              onAnnotationTextChange={handleAnnotationTextChange}
                              onAnnotationConfChange={handleAnnotationConfChange}
                              onDeleteAnnotation={handleDeleteAnnotation}
                              onAutoSegment={handleAutoSegment}
                              segmenting={segmenting}
                              fabricCanvasRef={fabricCanvasRef}
                              imageWidth={getCurrentTemplate()?.image_width}
                              imageHeight={getCurrentTemplate()?.image_height}
                            />
                          </div>
                        </div>
                      ) : (
                        <div className="template-placeholder">
                          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                            <rect x="3" y="3" width="18" height="18" rx="2" strokeWidth="2"/>
                            <circle cx="8.5" cy="8.5" r="1.5" fill="currentColor"/>
                            <polyline points="21,15 16,10 5,21" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                          </svg>
                          <p>No templates added yet</p>
                        </div>
                      )}
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        </form>
        </div>
      </div>

      {/* Confirmation Dialog */}
      <ConfirmDialog
        isOpen={confirmDialog.isOpen}
        onClose={() => setConfirmDialog({ ...confirmDialog, isOpen: false })}
        onConfirm={confirmDialog.onConfirm || (() => setConfirmDialog({ ...confirmDialog, isOpen: false }))}
        title={confirmDialog.title}
        message={confirmDialog.message}
        type={confirmDialog.type}
        confirmText={confirmDialog.onConfirm ? 'Confirm' : 'OK'}
        cancelText={confirmDialog.onConfirm ? 'Cancel' : ''}
      />

      {/* Color Setup Modal — opened from filmstrip "Setup Color" button */}
      {colorSetupTarget && (() => {
        const tmpl = cameraTemplates[colorSetupTarget.cameraId]?.[colorSetupTarget.templateIdx];
        if (!tmpl) {
          // Stale target — close.
          setColorSetupTarget(null);
          return null;
        }
        // Each 'product' annotation → [x,y][] polygon in TEMPLATE PIXEL coords.
        //
        // Annotations are persisted in NORMALIZED [0, 1] coords (relative to
        // imageBounds — see TemplateEditorRefactored.normalize). To overlay
        // them on the actual template image inside ColorSetupModal we must
        // denormalize by the template's image_width / image_height.
        const imgW = tmpl.image_width ?? 0;
        const imgH = tmpl.image_height ?? 0;
        const productPolys: Array<Array<[number, number]>> = (tmpl.annotations || [])
          .filter((a: any) => a.type === 'product')
          .map((a: any): Array<[number, number]> | null => {
            if (a.points && Array.isArray(a.points) && a.points.length >= 3) {
              return a.points.map((p: any) => {
                const px = p.x ?? p[0];
                const py = p.y ?? p[1];
                return [px * imgW, py * imgH] as [number, number];
              });
            }
            const w = a.width;
            const h = a.height;
            if (a.x != null && a.y != null && w && h) {
              const x1 = a.x * imgW;
              const y1 = a.y * imgH;
              const x2 = (a.x + w) * imgW;
              const y2 = (a.y + h) * imgH;
              return [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
              ];
            }
            return null;
          })
          .filter((p): p is Array<[number, number]> => p !== null);

        // crop_area annotation → pixel bbox (same normalization story as polygons)
        let cropAreaPx: { x1: number; y1: number; x2: number; y2: number } | null = null;
        const ca = (tmpl.annotations || []).find((a: any) => a.type === 'crop_area');
        if (ca) {
          if (ca.points && Array.isArray(ca.points) && ca.points.length >= 3) {
            const xs = ca.points.map((p: any) => (p.x ?? p[0]) * imgW);
            const ys = ca.points.map((p: any) => (p.y ?? p[1]) * imgH);
            cropAreaPx = {
              x1: Math.round(Math.min(...xs)), y1: Math.round(Math.min(...ys)),
              x2: Math.round(Math.max(...xs)), y2: Math.round(Math.max(...ys)),
            };
          } else if (ca.x != null && ca.y != null && ca.width && ca.height) {
            cropAreaPx = {
              x1: Math.round(ca.x * imgW), y1: Math.round(ca.y * imgH),
              x2: Math.round((ca.x + ca.width) * imgW), y2: Math.round((ca.y + ca.height) * imgH),
            };
          }
        }

        return (
          <ColorSetupModal
            isOpen={true}
            templateImage={tmpl.image}
            templateImageUrl={tmpl.image_url}
            imageWidth={tmpl.image_width ?? 0}
            imageHeight={tmpl.image_height ?? 0}
            productPolygons={productPolys}
            cropArea={cropAreaPx}
            initialConfig={tmpl.color_config ?? null}
            templateName={tmpl.name}
            onClose={() => setColorSetupTarget(null)}
            onSave={(cfg) => {
              setCameraTemplates(prev => ({
                ...prev,
                [colorSetupTarget.cameraId]: (prev[colorSetupTarget.cameraId] || []).map((t, i) =>
                  i === colorSetupTarget.templateIdx ? { ...t, color_config: cfg } : t
                ),
              }));
              setColorSetupTarget(null);
            }}
          />
        );
      })()}
    </div>
  );
}
