import { useState, useEffect, useRef } from 'react';
import '@/styles/InferenceRealtime.css';
import { socketService } from '@/services/socketio';
import api from '@/services/api';
import { receiptsAPI } from '@/services/recipes';
import { camerasAPI } from '@/services/cameras';
import ConfirmDialog from '@/components/shared/ConfirmDialog';
import InferenceRealtimeSettingsModal from './InferenceRealtimeSettingsModal';
import TemplateTextEditDialog, { type TextEditItem } from './TemplateTextEditDialog';
import { API_BASE_URL } from '@/config/api';
import { useToast } from '@/contexts/ToastContext';

interface CharConf {
  char: string;
  conf: number;
}

interface CharResult {
  idx: number;
  confidence: number;
  tm_conf: number;
  iou: number;
  pixel_conf: number;
  defects: string[];
  tmpl_char_b64: string;
  tgt_char_b64: string;
}

interface TextVerificationResult {
  region_idx: number;
  expected: string;
  recognized: string;
  match: boolean;
  confidence: number;
  threshold: number;
  match_sim: boolean;
  similarity: number;
  ml_pass?: boolean | null;
  ml_p_ok?: number | null;
  ml_label?: string | null;
  char_confs?: CharConf[] | null;
  char_results?: CharResult[];
}

interface TextVerification {
  all_match: boolean;
  results: TextVerificationResult[];
}

interface CharVerificationResult {
  annotation_idx: number;
  expected: string;
  match: boolean;
  ml_label: string;          // 'OK' | 'NG'
  ml_p_ok: number;
  threshold: number;
  error?: string | null;
  mask_diff_b64?: string | null;  // PNG base64 — XOR diff of aligned template/target masks
}

interface CharVerification {
  all_match: boolean;
  results: CharVerificationResult[];
}

interface InferenceLog {
  id: string;
  timestamp: string;
  result: 'PASS' | 'FAIL';
  recipe_name: string;
  product_code: string;
  message: string;
  inferenceResult?: InferenceResult;
}

interface TemplateVerification {
  match: boolean;
  similarity: number;
  threshold: number;
  method: string;
  template_bbox?: any;
}

interface WrinkleBox {
  score: number;
  class?: string;
  area?: number;
  area_pct?: number;
  contour?: number[][];
  corners?: number[][];
}

interface WrinkledCheck {
  ok?: boolean;
  skipped?: boolean;
  reason?: string;
  has_wrinkled?: boolean;
  wrinkled_count?: number;
  total_area?: number;
  min_area?: number | { source?: string; parsedValue?: number };
  wrinkled_boxes?: WrinkleBox[];
}

interface ProductVerification {
  match: boolean;
  skipped: boolean;
  error?: string;
  center_alignment_check?: {
    ok: boolean;
    offset_x: number;
    abs_offset_x?: number;
    direction?: string;
    threshold: number;
    threshold_left?: number;
    threshold_right?: number;
    template_center: [number, number];
    product_center: [number, number];
  };
  wrinkled_check?: WrinkledCheck;
  timing?: {
    total: number;
    yolo_inference?: number;
    frames_checked?: number;
    frames_total?: number;
  };
}

interface FrameResult {
  frame_idx: number;
  pass_fail: string;
  confidence: number;
  image_base64?: string;
  image_path?: string;
  template_name: string;
  timings?: any;
  detected_regions?: any[];
  text_verification?: TextVerification;
  char_verification?: CharVerification;
  template_verification?: TemplateVerification;
  product_verification?: ProductVerification;
}

interface CameraResult {
  camera_id: string;
  serial_number: string;
  delay_trigger?: number;
  pass_fail?: string;
  frames: FrameResult[];
}

interface PerCameraStats {
  serial_number: string;
  confidence: number;
  inliers: number;
  total_matches: number;
  timings: {
    total?: number;
    trt_inference?: number;
    preprocess?: number;
    postprocess?: number;
    [key: string]: number | undefined;
  };
  frame_stats: {
    total_frames: number;
    pass_count: number;
    fail_count: number;
    error_count: number;
    avg_confidence: number;
  };
}

interface PerCameraCumulativeStats {
  serial_number: string;
  pass_count: number;
  fail_count: number;
  error_count: number;
  total_pass_count: number;
  total_fail_count: number;
  total_error_count: number;
  camera_pass_fail: string;
}

interface InferenceResult {
  id: string;
  recipe_id: string;
  recipe_name: string;
  product_pass_fail: string;
  camera_results: CameraResult[];
  statistics?: {
    total_triggers: number;
    total_inferences: number;
    total_pass: number;
    total_fail: number;
    total_reject: number;
    per_camera: PerCameraCumulativeStats[];
  };
  metadata: {
    total_cameras: number;
    total_frames: number;
    inference_stats: {
      avg_confidence: number;
      total_inliers: number;
      total_matches: number;
      per_camera_stats: PerCameraStats[];
      overall_timings: any;
    };
  };
}

interface InferenceRealtimeProps {
  runningRecipeId: string | null;
  onClose?: () => void;
  embedded?: boolean; // If true, render without overlay wrapper
}

interface ConfirmDialogState {
  isOpen: boolean;
  title: string;
  message: string;
  type: 'warning' | 'danger' | 'info';
  onConfirm: (() => void) | null;
}

export default function InferenceRealtime({ runningRecipeId, onClose, embedded = false }: InferenceRealtimeProps) {
  const toast = useToast();
  const [logs, setLogs] = useState<InferenceLog[]>([]);
  const [latestResults, setLatestResults] = useState<InferenceResult | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [runningRecipe, setRunningRecipe] = useState<any>(null);
  const [isHeaderCollapsed, setIsHeaderCollapsed] = useState(() => {
    const saved = localStorage.getItem('inference_header_collapsed');
    return saved === 'true';
  });

  // Initialize isOnline from localStorage (default to false/OFFLINE for safety)
  const [isOnline, setIsOnline] = useState(() => {
    const saved = localStorage.getItem('inference_mode_online');
    return saved === 'true'; // Default to false (OFFLINE) if not set
  });

  const [isStopping, setIsStopping] = useState(false);
  const [expandedLogs, setExpandedLogs] = useState<Set<string>>(new Set());
  const [expandedCharConfs, setExpandedCharConfs] = useState<Set<string>>(new Set());
  const [expandedCharImages, setExpandedCharImages] = useState<Set<string>>(new Set());
  const [recipeStartTime, setRecipeStartTime] = useState<Date | null>(null);
  const [currentDuration, setCurrentDuration] = useState<string>('00:00:00');
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Delay trigger management - track each camera's delay
  const [_cameraDelays, setCameraDelays] = useState<Record<string, number>>({});
  const [updatingDelay, setUpdatingDelay] = useState<string | null>(null);

  // Settings modal state
  const [showSettingsModal, setShowSettingsModal] = useState(false);

  // Confirmation dialog state
  const [confirmDialog, setConfirmDialog] = useState<ConfirmDialogState>({
    isOpen: false,
    title: '',
    message: '',
    type: 'warning',
    onConfirm: null
  });

  // Template visibility toggle
  const [showTemplates, setShowTemplates] = useState(() => {
    const saved = localStorage.getItem('inference_show_templates');
    return saved === 'true'; // Default to false (hidden)
  });

  // Live template snapshot - keyed by serial_number
  // liveTemplateFrames: annotated (overlay drawn) for display
  // liveRawFrames: raw base64 for upload
  const [liveTemplateFrames, setLiveTemplateFrames] = useState<{[sn: string]: string[]}>({});
  const [liveRawFrames, setLiveRawFrames] = useState<{[sn: string]: string[]}>({});
  const [isGettingLiveTemplate, setIsGettingLiveTemplate] = useState(false);
  const [settingTemplateKey, setSettingTemplateKey] = useState<string | null>(null); // "sn_frameIdx"

  interface PendingTemplateData {
    sn: string;
    frameIdx: number;
    newImageUrl: string;
    uploadWidth: number;
    uploadHeight: number;
    recipeCam: any;
    templateName: string | undefined;
    recipeAnnotations: any[];
    detectedRegions: any[];
    textVerification: any;
    textEditItems: TextEditItem[];
  }
  const [pendingTemplateData, setPendingTemplateData] = useState<PendingTemplateData | null>(null);
  const [isConfirmingTemplate, setIsConfirmingTemplate] = useState(false);

  // Auto scroll to bottom
  const scrollToBottom = () => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [logs]);

  // Load persisted state from localStorage on mount
  useEffect(() => {
    try {
      const savedLatestLog = localStorage.getItem('inference_latest_log');
      const savedResults = localStorage.getItem('inference_latest_results');
      const savedExpandedLogs = localStorage.getItem('inference_expanded_logs');
      const savedStartTime = localStorage.getItem('inference_start_time');

      if (savedLatestLog) {
        const parsedLog = JSON.parse(savedLatestLog);
        // console.log('[InferenceRealtime] Restored latest log from localStorage');
        setLogs([parsedLog]); // Only restore the latest log
      }
      if (savedResults) {
        const parsedResults = JSON.parse(savedResults);
        console.log('[InferenceRealtime] Restored latest results from localStorage');
        setLatestResults(parsedResults);
      }
      if (savedExpandedLogs) {
        const parsedExpanded = new Set<string>(JSON.parse(savedExpandedLogs));
        // console.log(`[InferenceRealtime] Restored ${parsedExpanded.size} expanded logs`);
        setExpandedLogs(parsedExpanded);
      }
      if (savedStartTime) {
        // Parse ISO string (already in correct format with timezone)
        const startTime = new Date(savedStartTime);
        // console.log(`[InferenceRealtime] Restored start time: ${startTime.toLocaleString()}`);
        setRecipeStartTime(startTime);
      }
    } catch (error) {
      console.error('Error loading persisted state:', error);
    }
  }, []);

  // Persist state to localStorage when it changes
  useEffect(() => {
    try {
      // Only save the latest log (with image_base64 intact)
      if (logs.length > 0) {
        const latestLog = logs[logs.length - 1];
        localStorage.setItem('inference_latest_log', JSON.stringify(latestLog));
        console.log('[InferenceRealtime] Saved latest log to localStorage');
      }
    } catch (error) {
      console.error('Error saving latest log:', error);
      // If still fails, try clearing old data
      if (error instanceof DOMException && error.name === 'QuotaExceededError') {
        console.warn('localStorage quota exceeded, clearing old inference data');
        localStorage.removeItem('inference_latest_log');
        localStorage.removeItem('inference_latest_results');
      }
    }
  }, [logs]);

  useEffect(() => {
    try {
      if (latestResults) {
        // Save with image_base64 (only latest result, so size is manageable)
        localStorage.setItem('inference_latest_results', JSON.stringify(latestResults));
        console.log('[InferenceRealtime] Saved latest results to localStorage');
      }
    } catch (error) {
      console.error('Error saving latest results:', error);
      if (error instanceof DOMException && error.name === 'QuotaExceededError') {
        console.warn('localStorage quota exceeded for latest results');
        localStorage.removeItem('inference_latest_results');
      }
    }
  }, [latestResults]);

  useEffect(() => {
    try {
      localStorage.setItem('inference_expanded_logs', JSON.stringify(Array.from(expandedLogs)));
    } catch (error) {
      console.error('Error saving expanded logs:', error);
    }
  }, [expandedLogs]);

  // Save header collapsed state to localStorage
  useEffect(() => {
    try {
      localStorage.setItem('inference_header_collapsed', isHeaderCollapsed.toString());
    } catch (error) {
      console.error('Error saving header collapsed state:', error);
    }
  }, [isHeaderCollapsed]);

  // Save template visibility to localStorage
  useEffect(() => {
    try {
      localStorage.setItem('inference_show_templates', showTemplates.toString());
    } catch (error) {
      console.error('Error saving template visibility:', error);
    }
  }, [showTemplates]);

  // Get crop_area annotation (relative coords) for a specific camera+template from recipe metadata
  const getCropAreaForFrame = (sn: string, templateName?: string): { x: number; y: number; width: number; height: number } | null => {
    const camera = runningRecipe?.metadata?.cameras?.find((c: any) => c.serial_number === sn);
    if (!camera) return null;
    const cameraTemplate = runningRecipe?.metadata?.camera_templates?.find(
      (ct: any) => ct.camera_id === camera.camera_id
    );
    if (!cameraTemplate) return null;
    const template = templateName
      ? cameraTemplate.templates?.find((t: any) => t.name === templateName)
      : cameraTemplate.templates?.[0];
    if (!template) return null;
    const cropAnn = template.annotations?.find((a: any) => a.type === 'crop_area');
    if (!cropAnn) return null;
    return { x: cropAnn.x, y: cropAnn.y, width: cropAnn.width, height: cropAnn.height };
  };

  // Draw crop_area (relative coords) and detected_regions (absolute pixel coords) onto a live frame
  const drawDetectedRegionsOnImage = (
    imageSrc: string,
    detectedRegions: any[],
    cropArea?: { x: number; y: number; width: number; height: number } | null
  ): Promise<string> => {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(img, 0, 0);

        const lw = Math.max(2, img.width / 400);

        // Draw crop_area as dashed yellow rectangle (relative → absolute)
        if (cropArea) {
          const cx = cropArea.x * img.width;
          const cy = cropArea.y * img.height;
          const cw = cropArea.width * img.width;
          const ch = cropArea.height * img.height;
          ctx.save();
          ctx.strokeStyle = '#ffeb3b';
          ctx.lineWidth = lw;
          ctx.setLineDash([Math.max(6, img.width / 200), Math.max(3, img.width / 400)]);
          ctx.strokeRect(cx, cy, cw, ch);
          ctx.restore();
        }

        // Draw detected_regions polygons (absolute pixel coords)
        detectedRegions.forEach((region) => {
          const points: [number, number][] = region.points;
          if (!points || points.length < 2) return;

          let color = '#ff9800'; // orange default
          if (region.type === 'template') color = '#00bcd4'; // cyan
          else if (region.type === 'text') color = '#4caf50'; // green
          else if (region.type === 'datecode') color = '#5096ff'; // blue
          else if (region.type === 'char') color = '#fbbf24'; // amber (matches FE annotation type)
          // Override with red if this region failed verification
          if (region.verification_status === 'fail') color = '#ef4444';

          ctx.beginPath();
          const [x0, y0] = points[0] as [number, number];
          ctx.moveTo(x0, y0);
          for (let i = 1; i < points.length; i++) {
            const [xi, yi] = points[i] as [number, number];
            ctx.lineTo(xi, yi);
          }
          ctx.closePath();
          ctx.strokeStyle = color;
          ctx.lineWidth = lw;
          ctx.stroke();

          // Label for non-template regions
          if (region.type !== 'template') {
            const minX = Math.min(...points.map((p) => p[0]));
            const minY = Math.min(...points.map((p) => p[1]));
            const label = region.text
              ? `[${region.annotation_index ?? ''}] ${region.text}`
              : region.type;
            const fontSize = Math.max(12, img.width / 80);
            ctx.font = `bold ${fontSize}px monospace`;
            ctx.fillStyle = color;
            const labelY = minY > fontSize + 4 ? minY - 4 : minY + fontSize + 4;
            ctx.fillText(label, minX + 2, labelY);
          }
        });

        resolve(canvas.toDataURL('image/jpeg', 0.92));
      };
      img.onerror = () => resolve(imageSrc); // fallback: return original
      img.src = imageSrc;
    });
  };

  // Build full recipe payload for DB save (PUT /recipes/{id})
  // Merges updatedData over current runningRecipe.metadata, preserving all required fields
  // (name, image_url, wrinkle_*, etc.) that realtime payloads omit.
  const buildFullRecipePayloadForDB = (metadata: any, updatedData: any) => {
    const cameras = (metadata?.cameras ?? []).map((cam: any) => {
      const patch = (updatedData.cameras ?? []).find((c: any) => c.camera_id === cam.camera_id);
      return patch ? { ...cam, ...patch } : cam;
    });

    const camera_templates = (metadata?.camera_templates ?? []).map((ct: any) => {
      const patchCT = (updatedData.camera_templates ?? []).find((c: any) => c.camera_id === ct.camera_id);
      return {
        camera_id: ct.camera_id,
        templates: (ct.templates ?? []).map((tmpl: any, idx: number) => {
          const patchTmpl = patchCT?.templates?.[idx];
          if (!patchTmpl) return tmpl;
          return {
            ...tmpl,
            ...patchTmpl,
            name: tmpl.name,
            image_url: patchTmpl.image_url ?? tmpl.image_url,
            image_width: patchTmpl.image_width ?? tmpl.image_width,
            image_height: patchTmpl.image_height ?? tmpl.image_height,
          };
        }),
      };
    });

    return {
      delay_reject: updatedData.delay_reject ?? metadata?.delay_reject,
      reject_pulse: updatedData.reject_pulse ?? metadata?.reject_pulse,
      cameras,
      camera_templates,
    };
  };

  // Apply realtime patch to camera service AND persist full payload to DB.
  // Returns the merged metadata patch so the caller can update local state in one shot.
  // Realtime errors propagate; DB errors only toast a warning (no throw).
  const applyRealtimeAndPersist = async (
    recipeId: string,
    metadataSnapshot: any,
    patch: any,
    successMsg: string,
    warningPrefix: string,
  ): Promise<any> => {
    await receiptsAPI.updateRecipeRealtime(recipeId, patch);

    const dbPayload = buildFullRecipePayloadForDB(metadataSnapshot, patch);
    try {
      await receiptsAPI.updateReceipt(recipeId, dbPayload as any);
      toast.success(successMsg);
    } catch (dbErr: any) {
      if (dbErr?.response?.status === 403) {
        toast.warning(`${warningPrefix} (no permission to save to database)`);
      } else {
        toast.warning(`${warningPrefix} but failed to save to database`);
      }
    }

    return dbPayload;
  };

  const handleGetLiveTemplates = async () => {
    if (!latestResults?.camera_results?.length) return;
    setIsGettingLiveTemplate(true);
    try {
      const updates: {[sn: string]: string[]} = {};
      const rawUpdates: {[sn: string]: string[]} = {};
      for (const cameraResult of latestResults.camera_results) {
        const sn = cameraResult.serial_number;
        const count = cameraResult.frames.length || 1;
        try {
          const statusResponse = await camerasAPI.getCameraStatus(sn);
          if (!statusResponse.is_connected) {
            toast.warning(`Camera ${sn} is not connected`);
            continue;
          }
          const framesResponse = await camerasAPI.getLatestFrames(sn, count, 95);
          if (framesResponse?.frames?.length) {
            const annotatedFrames: string[] = [];
            const rawFrames: string[] = [];
            for (let i = 0; i < framesResponse.frames.length; i++) {
              const rawSrc = `data:image/jpeg;base64,${framesResponse.frames[i].frame_base64}`;
              rawFrames.push(rawSrc);
              const correspondingFrame = cameraResult.frames[i];
              const detectedRegions = correspondingFrame?.detected_regions ?? [];
              const cropArea = getCropAreaForFrame(sn, correspondingFrame?.template_name);
              const annotated = await drawDetectedRegionsOnImage(rawSrc, detectedRegions, cropArea);
              annotatedFrames.push(annotated);
            }
            updates[sn] = annotatedFrames;
            rawUpdates[sn] = rawFrames;
          }
        } catch (err) {
          console.error(`Failed to get frames for camera ${sn}:`, err);
        }
      }
      if (Object.keys(updates).length > 0) {
        setLiveTemplateFrames(updates);
        setLiveRawFrames(rawUpdates);
        toast.success('Captured live frames as template');
      } else {
        toast.error('Failed to get frames from any camera');
      }
    } catch (error: any) {
      toast.error(error.message || 'Failed to get live template frames');
    } finally {
      setIsGettingLiveTemplate(false);
    }
  };

  const handleReturnOriginalTemplates = () => {
    setLiveTemplateFrames({});
    setLiveRawFrames({});
  };

  // Build new annotations by merging recipe shape info + detected_regions coords + text_verification text
  const buildNewAnnotations = (
    recipeAnnotations: any[],
    detectedRegions: any[],
    textVerification: any,
    uploadWidth: number,
    uploadHeight: number
  ): any[] => {
    return recipeAnnotations.map((ann: any, idx: number) => {
      // crop_area: keep as-is (fixed zone, no coordinate update)
      if (ann.type === 'crop_area') return ann;

      // Find matching detected region
      let detected: any = null;
      if (ann.type === 'template') {
        detected = detectedRegions.find((d: any) => d.type === 'template');
      } else {
        detected = detectedRegions.find((d: any) => d.annotation_index === idx);
      }

      // Not detected → keep original annotation as safe fallback
      if (!detected) return ann;

      // Convert absolute polygon → axis-aligned bounding rect → relative coords
      const pts: [number, number][] = detected.points;
      const minX = Math.min(...pts.map((p) => p[0]));
      const maxX = Math.max(...pts.map((p) => p[0]));
      const minY = Math.min(...pts.map((p) => p[1]));
      const maxY = Math.max(...pts.map((p) => p[1]));

      const newAnn: any = {
        ...ann, // keep type, shape, conf and all other fields
        x: minX / uploadWidth,
        y: minY / uploadHeight,
        width: (maxX - minX) / uploadWidth,
        height: (maxY - minY) / uploadHeight,
        points: null,
      };

      // Update text from text_verification.recognized for text/datecode types.
      // Only override when OCR actually read something — avoid wiping recipe text on empty/failed OCR.
      // (Dialog usually overrides this via confirmedItems; this is the fallback when dialog is skipped.)
      if (ann.type === 'text' || ann.type === 'datecode') {
        const tvResult = textVerification?.results?.find(
          (r: any) => r.annotation_idx === idx
        );
        if (tvResult?.recognized) {
          newAnn.text = tvResult.recognized;
        }
      }

      return newAnn;
    });
  };

  // Upload a raw base64 frame to /templates/upload and return {url, width, height}
  const uploadRawFrame = async (rawBase64: string): Promise<{ url: string; width: number; height: number }> => {
    const res = await fetch(rawBase64);
    const blob = await res.blob();
    const file = new File([blob], `live_template_${Date.now()}.jpg`, { type: 'image/jpeg' });
    const formData = new FormData();
    formData.append('file', file);
    const token = localStorage.getItem('access_token');
    const uploadRes = await fetch(`${API_BASE_URL}/api/recipes/templates/upload`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}`, 'Bypass-Tunnel-Reminder': 'true' },
      body: formData,
    });
    if (!uploadRes.ok) {
      const err = await uploadRes.json().catch(() => ({}));
      throw new Error(err.detail || 'Failed to upload template image');
    }
    return uploadRes.json();
  };

  // Steps 4-8: build annotations (with user-confirmed texts), apply realtime + save DB
  const finishSetAsTemplate = async (data: PendingTemplateData, confirmedItems: TextEditItem[]) => {
    const { frameIdx, newImageUrl, uploadWidth, uploadHeight, recipeCam, templateName, recipeAnnotations, detectedRegions, textVerification } = data;

    let newAnnotations = buildNewAnnotations(
      recipeAnnotations, detectedRegions, textVerification, uploadWidth, uploadHeight
    );

    // Apply text overrides from dialog (by original annotation_index)
    confirmedItems.forEach(item => {
      if (newAnnotations[item.idx]) {
        newAnnotations[item.idx] = { ...newAnnotations[item.idx], text: item.newText };
      }
    });

    // Reorder display items (text/datecode/char) based on dialog's new order.
    // Non-display items (template, crop_area, product, label, mask) keep their array slots.
    // Display slots are filled in confirmedItems' new order — char drag-drop reorder feeds in here.
    if (confirmedItems.length > 0) {
      const displaySlots: number[] = [];
      newAnnotations.forEach((ann: any, i: number) => {
        if (ann.type === 'text' || ann.type === 'datecode' || ann.type === 'char') {
          displaySlots.push(i);
        }
      });
      if (displaySlots.length === confirmedItems.length) {
        const annByOriginalIdx: Record<number, any> = {};
        newAnnotations.forEach((ann: any, i: number) => { annByOriginalIdx[i] = ann; });
        const reordered = [...newAnnotations];
        displaySlots.forEach((slot, i) => {
          const originalIdx = confirmedItems[i]?.idx;
          if (originalIdx !== undefined && annByOriginalIdx[originalIdx]) {
            reordered[slot] = annByOriginalIdx[originalIdx];
          }
        });
        newAnnotations = reordered;
      }
    }

    const metadataSnapshot = runningRecipe?.metadata;

    const cameras = (metadataSnapshot?.cameras ?? []).map((cam: any) => ({
      camera_id: cam.camera_id,
      exposure_time: cam.exposure_time,
      delay_trigger: cam.delay_trigger,
      delay_interval: cam.delay_interval,
      gain: cam.gain,
    }));

    const cameraTemplates = (metadataSnapshot?.camera_templates ?? []).map((ct: any) => ({
      camera_id: ct.camera_id,
      templates: (ct.templates ?? []).map((t: any, tIdx: number) => {
        const isTarget = ct.camera_id === recipeCam?.camera_id &&
          (templateName ? t.name === templateName : tIdx === frameIdx);
        return {
          annotations: isTarget ? newAnnotations : (t.annotations ?? []),
          image_url: isTarget ? newImageUrl : t.image_url,
          image_width: isTarget ? uploadWidth : t.image_width,
          image_height: isTarget ? uploadHeight : t.image_height,
          center_offset_threshold_left: t.center_offset_threshold_left,
          center_offset_threshold_right: t.center_offset_threshold_right,
        };
      }),
    }));

    const patch = {
      delay_reject: metadataSnapshot?.delay_reject,
      reject_pulse: metadataSnapshot?.reject_pulse,
      cameras,
      camera_templates: cameraTemplates,
    };

    const merged = await applyRealtimeAndPersist(
      runningRecipeId!,
      metadataSnapshot,
      patch,
      `Template updated and saved: ${templateName ?? `Frame ${frameIdx + 1}`}`,
      'Template applied to camera service only',
    );

    setRunningRecipe((prev: any) =>
      prev ? { ...prev, metadata: { ...prev.metadata, ...merged } } : prev
    );
  };

  // Set live captured frame as new template.
  // If the frame has text/datecode annotations, shows a review dialog before applying.
  const handleSetAsTemplate = async (sn: string, frameIdx: number) => {
    const key = `${sn}_${frameIdx}`;
    if (settingTemplateKey === key) return;

    const rawSrc = liveRawFrames[sn]?.[frameIdx];
    if (!rawSrc) { toast.error('No raw frame captured'); return; }
    if (!latestResults || !runningRecipeId) { toast.error('No inference result available'); return; }

    setSettingTemplateKey(key);
    try {
      // 1. Upload raw frame
      const { url: newImageUrl, width: uploadWidth, height: uploadHeight } = await uploadRawFrame(rawSrc);

      // 2. Find camera and its result frame
      const cameraResult = latestResults.camera_results.find((c) => c.serial_number === sn);
      const inferenceFrame = cameraResult?.frames[frameIdx];
      const detectedRegions: any[] = inferenceFrame?.detected_regions ?? [];
      const textVerification = inferenceFrame?.text_verification;

      // 3. Get recipe annotations for this frame
      const recipeCam = runningRecipe?.metadata?.cameras?.find((c: any) => c.serial_number === sn);
      const recipeCT = runningRecipe?.metadata?.camera_templates?.find(
        (ct: any) => ct.camera_id === recipeCam?.camera_id
      );
      const templateName = inferenceFrame?.template_name;
      const recipeTemplate = templateName
        ? recipeCT?.templates?.find((t: any) => t.name === templateName)
        : recipeCT?.templates?.[frameIdx];
      const recipeAnnotations: any[] = recipeTemplate?.annotations ?? [];

      const pendingData: PendingTemplateData = {
        sn, frameIdx, newImageUrl, uploadWidth, uploadHeight,
        recipeCam, templateName, recipeAnnotations, detectedRegions, textVerification,
        textEditItems: [],
      };

      // 4. Collect text/datecode/char annotations for user review.
      //    - text/datecode prefill from OCR `recognized` (what AI actually read)
      //    - char prefill from recipe text (ML classifier verifies OK/NG, doesn't recognize chars)
      const textEditItems: TextEditItem[] = recipeAnnotations
        .map((ann: any, idx: number): TextEditItem | null => {
          if (ann.type !== 'text' && ann.type !== 'datecode' && ann.type !== 'char') return null;
          const tvResult = textVerification?.results?.find((r: any) => r.annotation_idx === idx);
          const recognized = ann.type === 'char' ? '' : (tvResult?.recognized ?? '');
          return {
            idx,
            type: ann.type,
            label: `Region ${idx + 1}`,
            oldText: ann.text ?? '',
            newText: recognized || (ann.text ?? ''),
          };
        })
        .filter((item): item is TextEditItem => item !== null);

      if (textEditItems.length > 0) {
        // Show review dialog — release button loading, dialog takes over
        setPendingTemplateData({ ...pendingData, textEditItems });
        setSettingTemplateKey(null);
        return;
      }

      // No text/datecode annotations — apply directly
      await finishSetAsTemplate(pendingData, []);
    } catch (err: any) {
      toast.error(err.message || 'Failed to set template');
    } finally {
      setSettingTemplateKey(null);
    }
  };

  const handleDialogConfirm = async (confirmedItems: TextEditItem[]) => {
    if (!pendingTemplateData) return;
    setIsConfirmingTemplate(true);
    try {
      await finishSetAsTemplate(pendingTemplateData, confirmedItems);
      setPendingTemplateData(null);
    } catch (err: any) {
      toast.error(err.message || 'Failed to set template');
    } finally {
      setIsConfirmingTemplate(false);
    }
  };

  const handleDialogCancel = () => {
    setPendingTemplateData(null);
    toast.info('Template update cancelled');
  };

  // Update duration every second
  useEffect(() => {
    if (!recipeStartTime) return;

    const interval = setInterval(() => {
      const now = new Date();
      const diff = now.getTime() - recipeStartTime.getTime();
      const hours = Math.floor(diff / (1000 * 60 * 60));
      const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
      const seconds = Math.floor((diff % (1000 * 60)) / 1000);

      setCurrentDuration(
        `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
      );
    }, 1000);

    return () => clearInterval(interval);
  }, [recipeStartTime]);

  // Load running recipe info
  useEffect(() => {
    const loadRunningRecipe = async () => {
      if (!runningRecipeId) {
        setRunningRecipe(null);
        setRecipeStartTime(null);
        localStorage.removeItem('inference_start_time');
        return;
      }

      try {
        const latest = await receiptsAPI.getLatestLoadedRecipe();
        if (latest && latest.recipe_id === runningRecipeId) {
          setRunningRecipe(latest);

          // Set start time from loaded_at_local (Vietnam time) or loaded_at (UTC)
          const loadedAtStr = latest.loaded_at_local || latest.loaded_at;
          if (loadedAtStr) {
            // If using loaded_at (UTC without 'Z'), add 'Z' to parse correctly
            let startTime: Date;
            if (latest.loaded_at_local) {
              // Has timezone info, parse directly
              startTime = new Date(latest.loaded_at_local);
            } else if (typeof latest.loaded_at === 'string' && !latest.loaded_at.endsWith('Z')) {
              // UTC time without 'Z', add it to parse as UTC
              startTime = new Date(latest.loaded_at + 'Z');
            } else {
              startTime = new Date(latest.loaded_at);
            }

            setRecipeStartTime(startTime);
            localStorage.setItem('inference_start_time', startTime.toISOString());
          }
        }
      } catch (error) {
        console.error('Error loading running recipe:', error);
      }
    };

    loadRunningRecipe();
  }, [runningRecipeId]);

  
  // Subscribe to realtime inference results
  useEffect(() => {
    if (!runningRecipeId) return;

    socketService.connect();

    function stripBase64FromResult(data: any) {
    return {
      ...data,
      camera_results: data.camera_results?.map((camera: any) => ({
        ...camera,
        frames: camera.frames?.map((frame: any) => {
          const { image_base64, ...frameWithoutBase64 } = frame;
          return frameWithoutBase64;
        })
      }))
    };
  }


    const handleNewResult = (data: any) => {
      console.log('New inference result:', stripBase64FromResult(data));


      const fullResult = data as InferenceResult;

      // Add to logs
      const newLog: InferenceLog = {
        id: data.id || `log-${Date.now()}`,
        timestamp: new Date().toLocaleTimeString(),
        result: data.product_pass_fail || 'FAIL',
        recipe_name: data.recipe_name || 'Unknown',
        product_code: data.product_code || 'Unknown',
        message: `${data.recipe_name} - ${data.product_pass_fail}`,
        inferenceResult: fullResult
      };

      setLogs(prev => [...prev, newLog].slice(-100)); // Keep last 100 logs

      // Auto-expand if FAIL and has any verification details
      if (data.product_pass_fail === 'FAIL') {
        const hasAnyVerification = data.camera_results?.some((cam: CameraResult) =>
          cam.frames.some((frame: FrameResult) =>
            frame.text_verification || frame.char_verification
          )
        );
        if (hasAnyVerification) {
          setExpandedLogs(prev => new Set(prev).add(newLog.id));
        }
      }

      // Store full inference result for multi-camera display
      setLatestResults(fullResult);

      // Sync delay_trigger from result to local state
      if (data.camera_results) {
        const newDelays: Record<string, number> = {};
        data.camera_results.forEach((camResult: CameraResult) => {
          if (camResult.delay_trigger !== undefined) {
            newDelays[camResult.serial_number] = camResult.delay_trigger;
          }
        });
        setCameraDelays(prev => ({ ...prev, ...newDelays }));
      }
    };

    socketService.subscribeToInferenceResults(handleNewResult);

    return () => {
      socketService.unsubscribeFromInferenceResults(handleNewResult);
    };
  }, [runningRecipeId]);

  const handleSimulateTrigger = async () => {
    if (isSimulating) return;

    setIsSimulating(true);

    try {
      await api.post('/trigger-simulator/simulate', {
        trigger_type: 'rising_edge'
      });

      // Don't add log here - wait for real result from WebSocket
      // The result will be automatically added by handleNewResult callback
    } catch (error: any) {
      console.error('Error simulating trigger:', error);
      const errorLog: InferenceLog = {
        id: `error-${Date.now()}`,
        timestamp: new Date().toLocaleTimeString(),
        result: 'FAIL',
        recipe_name: '',
        product_code: '',
        message: `❌ Error: ${error.response?.data?.detail || 'Failed to simulate trigger'}`
      };

      setLogs(prev => [...prev, errorLog]);
    } finally {
      setTimeout(() => setIsSimulating(false), 1000);
    }
  };

  const handleStopRecipe = () => {
    if (!runningRecipeId || isStopping) return;

    setConfirmDialog({
      isOpen: true,
      title: 'Stop Recipe',
      message: `Stop recipe "${runningRecipe?.metadata?.name || 'Unknown'}"?\n\nThis will stop inference and set cameras to idle mode.`,
      type: 'danger',
      onConfirm: async () => {
        setIsStopping(true);

        try {
          await receiptsAPI.stopReceipt(runningRecipeId);

          // Notify other views (e.g. Receipts page) so their cached
          // runningRecipeId can be cleared without waiting for a remount.
          window.dispatchEvent(new CustomEvent('recipeStopped', {
            detail: { recipeId: runningRecipeId }
          }));

          // Add log
          const stopLog: InferenceLog = {
            id: `stop-${Date.now()}`,
            timestamp: new Date().toLocaleTimeString(),
            result: 'PASS',
            recipe_name: runningRecipe?.metadata?.name || 'Unknown',
            product_code: runningRecipe?.metadata?.product_code || '',
            message: '🛑 Recipe stopped'
          };
          setLogs(prev => [...prev, stopLog]);

          // Clear persisted state when recipe stops
          localStorage.removeItem('inference_latest_log');
          localStorage.removeItem('inference_latest_results');
          localStorage.removeItem('inference_expanded_logs');
          localStorage.removeItem('inference_start_time');
          localStorage.removeItem('inference_mode_online'); // Clear inference mode state

          // Redirect to Recipes page after 1 second
          setTimeout(() => {
            // If onClose prop exists, use it to return to previous view
            if (onClose) {
              onClose();
            } else {
              // Fallback: try to find and click recipes link
              const recipesLink = document.querySelector('a[href="#receipts"]') as HTMLAnchorElement;
              if (recipesLink) {
                recipesLink.click();
              }
            }
          }, 1000);
        } catch (error: any) {
          console.error('Error stopping recipe:', error);
          alert(`Failed to stop recipe: ${error.response?.data?.detail || error.message}`);
        } finally {
          setIsStopping(false);
        }
      }
    });
  };

  const handleToggleInferenceMode = async () => {
    if (!runningRecipeId) return;

    const newMode = !isOnline;

    try {
      await receiptsAPI.setInferenceMode(runningRecipeId, newMode);
      setIsOnline(newMode);

      // Save to localStorage for persistence
      localStorage.setItem('inference_mode_online', newMode.toString());
      console.log(`[InferenceMode] Toggled to ${newMode ? 'ONLINE' : 'OFFLINE'}, saved to localStorage`);

      // Add log
      const modeLog: InferenceLog = {
        id: `mode-${Date.now()}`,
        timestamp: new Date().toLocaleTimeString(),
        result: 'PASS',
        recipe_name: runningRecipe?.metadata?.name || 'Unknown',
        product_code: runningRecipe?.metadata?.product_code || '',
        message: newMode ? '🟢 Inference ONLINE' : '⏸️ Inference OFFLINE'
      };
      setLogs(prev => [...prev, modeLog]);
    } catch (error: any) {
      console.error('Error toggling inference mode:', error);
      alert(`Failed to change mode: ${error.response?.data?.detail || error.message}`);
    }
  };

  const _handleUpdateDelay = async (serialNumber: string, newDelay: number) => {
    if (updatingDelay || newDelay < 0 || newDelay > 10000) return;

    setUpdatingDelay(serialNumber);

    try {
      await camerasAPI.updateDelayTrigger(serialNumber, newDelay);

      // Update local state
      setCameraDelays(prev => ({ ...prev, [serialNumber]: newDelay }));

      // Add success log
      const delayLog: InferenceLog = {
        id: `delay-${Date.now()}`,
        timestamp: new Date().toLocaleTimeString(),
        result: 'PASS',
        recipe_name: runningRecipe?.metadata?.name || 'Unknown',
        product_code: '',
        message: `⚙️ Camera ${serialNumber}: Delay updated to ${newDelay}ms`
      };
      setLogs(prev => [...prev, delayLog]);
    } catch (error: any) {
      console.error('Error updating delay:', error);
      alert(`Failed to update delay: ${error.response?.data?.detail || error.message}`);
    } finally {
      setUpdatingDelay(null);
    }
  };

  // Helper: Check if log entry has text verification data
  const hasTextVerification = (log: InferenceLog): boolean => {
    return log.inferenceResult?.camera_results?.some((cam) =>
      cam.frames.some((frame) => frame.text_verification)
    ) || false;
  };

  // Helper: Check if log entry has char (ML) verification data
  const hasCharVerification = (log: InferenceLog): boolean => {
    return log.inferenceResult?.camera_results?.some((cam) =>
      cam.frames.some((frame) =>
        frame.char_verification && (frame.char_verification.results?.length ?? 0) > 0
      )
    ) || false;
  };

  // Helper: Check if log entry has wrinkle data (any frame)
  const hasWrinkleVerification = (log: InferenceLog): boolean => {
    return log.inferenceResult?.camera_results?.some((cam) =>
      cam.frames.some((frame) => !!frame.product_verification?.wrinkled_check)
    ) || false;
  };

  // Helper: parse min_area which may be number or { parsedValue }
  const parseMinArea = (val: WrinkledCheck['min_area']): string => {
    if (val == null) return '-';
    if (typeof val === 'number') return val.toFixed(0);
    if (typeof val === 'object' && val.parsedValue != null) return Number(val.parsedValue).toFixed(0);
    return '-';
  };

  // Wrinkle details expand state (per frame)
  const [expandedWrinkles, setExpandedWrinkles] = useState<Set<string>>(new Set());

  const toggleWrinkles = (key: string) => {
    setExpandedWrinkles(prev => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key); else next.add(key);
      return next;
    });
  };

  // Helper: Toggle log expansion
  const toggleLogExpansion = (logId: string) => {
    setExpandedLogs(prev => {
      const newSet = new Set(prev);
      if (newSet.has(logId)) {
        newSet.delete(logId);
      } else {
        newSet.add(logId);
      }
      return newSet;
    });
  };

  // Helper: Get confidence color class
  const getConfidenceClass = (confidence: number): string => {
    if (confidence > 0.8) return 'high';
    if (confidence > 0.5) return 'medium';
    return 'low';
  };

  // Helper: Get camera info from serial_number
  const getCameraInfo = (serialNumber: string): { camera_id: string; location: string } | null => {
    if (!runningRecipe?.metadata?.cameras) return null;

    const camera = runningRecipe.metadata.cameras.find(
      (cam: any) => cam.serial_number === serialNumber
    );

    if (!camera) return null;

    return {
      camera_id: camera.camera_id,
      location: camera.location || ''
    };
  };

  // Helper: Get template image URL from recipe metadata
  const getTemplateImageUrl = (serialNumber: string, templateName: string): string | null => {
    if (!runningRecipe?.metadata) return null;

    // Find camera by serial_number
    const camera = runningRecipe.metadata.cameras?.find(
      (cam: any) => cam.serial_number === serialNumber
    );
    if (!camera) return null;

    // Find camera_templates by camera_id
    const cameraTemplate = runningRecipe.metadata.camera_templates?.find(
      (ct: any) => ct.camera_id === camera.camera_id
    );
    if (!cameraTemplate) return null;

    // Find template by name
    const template = cameraTemplate.templates?.find(
      (t: any) => t.name === templateName
    );
    if (!template) return null;

    // Return full image URL with API_BASE_URL
    const imageUrl = template.image_url;
    // If URL already includes http, return as is, otherwise prepend API_BASE_URL
    if (imageUrl.startsWith('http')) {
      return imageUrl;
    }
    return `${API_BASE_URL}${imageUrl}`;
  };

  // Render text verification table for a camera
  const renderTextVerificationTable = (cameraResult: CameraResult, logId: string) => {
    // Get ALL frames with text verification (multi-template support)
    const framesWithVerification = cameraResult.frames.filter(f => f.text_verification);
    const cameraInfo = getCameraInfo(cameraResult.serial_number);
    const displayName = cameraInfo ? cameraInfo.camera_id : cameraResult.serial_number;

    if (framesWithVerification.length === 0) {
      return (
        <div className="camera-verification-card">
          <div className="camera-verification-header">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
              <rect x="2" y="7" width="20" height="14" rx="2" stroke="currentColor" strokeWidth="2"/>
              <circle cx="12" cy="14" r="3" stroke="currentColor" strokeWidth="2"/>
            </svg>
            <span className="camera-serial" title={cameraInfo ? `${cameraInfo.location} (${cameraResult.serial_number})` : cameraResult.serial_number}>
              {displayName}
            </span>
          </div>
          <div className="no-verification-data">No text verification data</div>
        </div>
      );
    }

    // Render a table for each frame with verification
    return (
      <div className="camera-verification-card">
        <div className="camera-verification-header">
          <div className="camera-info">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
              <rect x="2" y="7" width="20" height="14" rx="2" stroke="currentColor" strokeWidth="2"/>
              <circle cx="12" cy="14" r="3" stroke="currentColor" strokeWidth="2"/>
            </svg>
            <span className="camera-serial" title={cameraInfo ? `${cameraInfo.location} (${cameraResult.serial_number})` : cameraResult.serial_number}>
              {displayName}
            </span>
          </div>
        </div>

        {framesWithVerification.map((frame) => {
          const verification = frame.text_verification!;
          const matchCount = verification.results.filter(r => r.match).length;
          const totalCount = verification.results.length;
          const allMatch = verification.all_match;

          return (
            <div key={frame.frame_idx} className="frame-verification-section">
              <div className="frame-verification-header">
                <span className="frame-label">
                  {frame.template_name || `Frame ${frame.frame_idx}`}
                </span>
                <span className={`match-summary ${allMatch ? 'all-match' : 'has-mismatch'}`}>
                  {matchCount}/{totalCount} Match {allMatch ? '✓' : '✗'}
                </span>
              </div>
              <div className="verification-table-wrapper">
                <table className="verification-table">
                  <thead>
                    <tr>
                      <th className="col-region">#</th>
                      <th className="col-expected">Expected</th>
                      <th className="col-recognized">Recognized</th>
                      <th className="col-match">Match</th>
                      <th className="col-confidence">Conf</th>
                      <th className="col-confidence">ML</th>
                      <th className="col-threshold-ocr">Conf</th>

                    </tr>
                  </thead>
                  <tbody>
                    {verification.results.map((result) => {
                      const rowKey = `${logId}-${cameraResult.serial_number}-${frame.frame_idx}-${result.region_idx}`;
                      const charConfsKey = rowKey;
                      const charImagesKey = `${rowKey}-imgs`;
                      const isCharConfsExpanded = expandedCharConfs.has(charConfsKey);
                      const isCharImagesExpanded = expandedCharImages.has(charImagesKey);
                      const hasCharConfs = result.char_confs && result.char_confs.length > 0;
                      const hasCharImages = result.char_results && result.char_results.length > 0;
                      return (
                        <>
                          <tr key={result.region_idx} className={result.match ? '' : 'mismatch-row'}>
                            <td className="col-region">{result.region_idx}</td>
                            <td className="col-expected">{result.expected || '—'}</td>
                            <td className="col-recognized">
                              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                <span>{result.recognized || '—'}</span>
                                {hasCharConfs && (
                                  <button
                                    className="char-conf-toggle-btn"
                                    onClick={() => setExpandedCharConfs(prev => {
                                      const next = new Set(prev);
                                      if (next.has(charConfsKey)) next.delete(charConfsKey);
                                      else next.add(charConfsKey);
                                      return next;
                                    })}
                                    title="Per-character OCR confidence"
                                  >
                                    {isCharConfsExpanded ? '▲' : '▼'}
                                  </button>
                                )}
                              </div>
                            </td>
                            <td className="col-match">
                              <span className={`match-icon ${result.match ? 'match' : 'no-match'}`}>
                                {result.match ? '✓' : '✗'}
                              </span>
                            </td>
                            <td className="col-confidence">
                              <span className={`confidence-value ${getConfidenceClass(result.confidence)}`}>
                                {(result.confidence * 100).toFixed(1)}%
                              </span>
                            </td>
                            <td className="col-confidence">
                              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                <span className={`confidence-value ${result.ml_p_ok != null ? getConfidenceClass(result.ml_p_ok) : ''}`}>
                                  {result.ml_p_ok != null ? (result.ml_p_ok * 100).toFixed(1) + '%' : '—'}
                                </span>
                                {hasCharImages && (
                                  <button
                                    className="char-conf-toggle-btn"
                                    onClick={() => setExpandedCharImages(prev => {
                                      const next = new Set(prev);
                                      if (next.has(charImagesKey)) next.delete(charImagesKey);
                                      else next.add(charImagesKey);
                                      return next;
                                    })}
                                    title="Visual character comparison"
                                  >
                                    {isCharImagesExpanded ? '▲' : '▼'}
                                  </button>
                                )}
                              </div>
                            </td>
                            <td className="col-threshold-ocr">
                              <span className={`confidence-value ${getConfidenceClass(result.threshold)}`}>
                                {(result.threshold * 100).toFixed(1)}%
                              </span>
                            </td>
                          </tr>

                          {/* Per-character OCR confidence row */}
                          {hasCharConfs && isCharConfsExpanded && (
                            <tr key={`${result.region_idx}-char-confs`} className={result.match ? 'char-conf-row' : 'char-conf-row mismatch-row'}>
                              <td colSpan={7} className="char-conf-cell">
                                <div className="char-conf-list">
                                  {result.char_confs!.map((cc, i) => (
                                    <div key={i} className={`char-conf-badge ${getConfidenceClass(cc.conf)}`}>
                                      <span className="char-conf-char">{cc.char}</span>
                                      <span className="char-conf-pct">{(cc.conf * 100).toFixed(0)}%</span>
                                    </div>
                                  ))}
                                </div>
                              </td>
                            </tr>
                          )}

                          {/* Visual character comparison row */}
                          {hasCharImages && isCharImagesExpanded && (
                            <tr key={`${result.region_idx}-char-images`} className="char-images-row">
                              <td colSpan={7} className="char-images-cell">
                                <div className="char-comparison-strip">
                                  {result.char_results!.map((cr) => {
                                    const isPass = cr.confidence >= (result.threshold ?? 0.75);
                                    return (
                                      <div key={cr.idx} className={`char-comparison-cell ${isPass ? 'char-cell-pass' : 'char-cell-fail'}`}>
                                        <img
                                          src={`data:image/png;base64,${cr.tmpl_char_b64}`}
                                          className="char-img char-img-tmpl"
                                          title="Template"
                                        />
                                        <img
                                          src={`data:image/png;base64,${cr.tgt_char_b64}`}
                                          className={`char-img char-img-tgt ${isPass ? 'char-img-pass' : 'char-img-fail'}`}
                                          title="Target"
                                        />
                                        <div className={`char-cell-score ${isPass ? 'score-pass' : 'score-fail'}`}>
                                          <span>{(cr.confidence * 100).toFixed(0)}%</span>
                                          <span>{isPass ? 'PASS' : 'FAIL'}</span>
                                        </div>
                                        {cr.defects.length > 0 && (
                                          <div className="char-cell-defects" title={cr.defects.join(', ')}>
                                            {cr.defects[0]}
                                          </div>
                                        )}
                                      </div>
                                    );
                                  })}
                                </div>
                              </td>
                            </tr>
                          )}
                        </>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  // Render template verification info for a camera
  const renderTemplateVerificationInfo = (cameraResult: CameraResult) => {
    // Get frames with template verification
    const framesWithTemplateVerification = cameraResult.frames.filter(f => f.template_verification);
    const cameraInfo = getCameraInfo(cameraResult.serial_number);
    const displayName = cameraInfo ? cameraInfo.camera_id : cameraResult.serial_number;

    if (framesWithTemplateVerification.length === 0) {
      return null;
    }

    return (
      <div className="camera-verification-card template-verification">
        <div className="camera-verification-header">
          <div className="camera-info">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
              <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="2"/>
              <path d="M9 12l2 2 4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
            <span className="camera-serial" title={cameraInfo ? `${cameraInfo.location} (${cameraResult.serial_number})` : cameraResult.serial_number}>
              {displayName}
            </span>
            <span className="verification-type">Template Matching</span>
          </div>
        </div>

        {framesWithTemplateVerification.map((frame) => {
          const verification = frame.template_verification!;
          const similarityPercent = (verification.similarity * 100).toFixed(2);
          const thresholdPercent = (verification.threshold * 100).toFixed(0);
          const isMatch = verification.match;

          return (
            <div key={frame.frame_idx} className="frame-verification-section template-frame">
              <div className="frame-verification-header">
                <span className="frame-label">
                  {frame.template_name || `Frame ${frame.frame_idx}`}
                </span>
                <span className={`match-summary ${isMatch ? 'all-match' : 'has-mismatch'}`}>
                  {isMatch ? 'MATCH ✓' : 'NO MATCH ✗'}
                </span>
              </div>
              <div className="template-verification-details">
                <div className="verification-row">
                  <span className="label">Similarity:</span>
                  <span className={`value similarity ${isMatch ? 'high' : 'low'}`}>
                    {similarityPercent}%
                  </span>
                </div>
                <div className="verification-row">
                  <span className="label">Threshold:</span>
                  <span className="value threshold">{thresholdPercent}%</span>
                </div>
                {!isMatch && (
                  <div className="verification-warning">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
                      <path d="M12 8v4m0 4h.01" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                    </svg>
                    <span>Template similarity below threshold ({similarityPercent}% &lt; {thresholdPercent}%)</span>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  // ── Wrinkle verification — per-frame wrinkle detection ──
  const renderWrinkleVerificationInfo = (cameraResult: CameraResult, logId: string) => {
    const framesWithWrinkle = cameraResult.frames.filter(f => f.product_verification?.wrinkled_check);
    const cameraInfo = getCameraInfo(cameraResult.serial_number);
    const displayName = cameraInfo ? cameraInfo.camera_id : cameraResult.serial_number;

    if (framesWithWrinkle.length === 0) return null;

    return (
      <div className="camera-verification-card wrinkle-verification">
        <div className="camera-verification-header">
          <div className="camera-info">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
              <path d="M3 12c3-4 6-4 9 0s6 4 9 0" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              <path d="M3 18c3-4 6-4 9 0s6 4 9 0" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
            <span className="camera-serial" title={cameraInfo ? `${cameraInfo.location} (${cameraResult.serial_number})` : cameraResult.serial_number}>
              {displayName}
            </span>
            <span className="verification-type">Wrinkle Detection</span>
          </div>
        </div>

        {framesWithWrinkle.map((frame) => {
          const wc = frame.product_verification!.wrinkled_check!;
          const boxes = wc.wrinkled_boxes ?? [];
          const isSkipped = !!wc.skipped;
          const isOk = !!wc.ok;
          const wrinkleKey = `${logId}-${cameraResult.serial_number}-${frame.frame_idx}`;
          const isExpanded = expandedWrinkles.has(wrinkleKey);

          return (
            <div key={frame.frame_idx} className="frame-verification-section wrinkle-frame">
              <div className="frame-verification-header">
                <span className="frame-label">
                  {frame.template_name || `Frame ${frame.frame_idx}`}
                </span>
                <span className={`match-summary ${isSkipped ? '' : isOk ? 'all-match' : 'has-mismatch'}`}>
                  {isSkipped ? 'SKIPPED' : isOk ? 'OK ✓' : 'WRINKLED ✗'}
                </span>
              </div>

              {isSkipped && wc.reason ? (
                <div className="template-verification-details">
                  <div className="verification-row">
                    <span className="label">Reason:</span>
                    <span className="value">{wc.reason}</span>
                  </div>
                </div>
              ) : (
                <div className="template-verification-details">
                  <div className="wrinkle-summary-grid">
                    <div className="wrinkle-chip">
                      <span className="wrinkle-chip-label">Count</span>
                      <span className={`wrinkle-chip-value ${(wc.wrinkled_count ?? 0) > 0 ? 'bad' : 'good'}`}>
                        {wc.wrinkled_count ?? 0}
                      </span>
                    </div>
                    <div className="wrinkle-chip">
                      <span className="wrinkle-chip-label">Total Area</span>
                      <span className="wrinkle-chip-value">
                        {wc.total_area != null ? `${wc.total_area}px²` : '-'}
                      </span>
                    </div>
                    {/* <div className="wrinkle-chip">
                      <span className="wrinkle-chip-label">Min Area</span>
                      <span className="wrinkle-chip-value">
                        {`${parseMinArea(wc.min_area)}px²`}
                      </span>
                    </div> */}
                    <div className="wrinkle-chip">
                      <span className="wrinkle-chip-label">Has Wrinkle</span>
                      <span className={`wrinkle-chip-value ${wc.has_wrinkled ? 'bad' : 'good'}`}>
                        {wc.has_wrinkled ? 'Yes' : 'No'}
                      </span>
                    </div>
                  </div>

                  {boxes.length > 0 && (
                    <button
                      type="button"
                      className="wrinkle-toggle-btn"
                      onClick={() => toggleWrinkles(wrinkleKey)}
                    >
                      {isExpanded ? '▲ Hide details' : `▼ Details (${boxes.length})`}
                    </button>
                  )}

                  {boxes.length > 0 && isExpanded && (
                    <div className="wrinkle-details-wrap">
                      <table className="wrinkle-details-table">
                        <thead>
                          <tr>
                            <th>#</th>
                            <th>Class</th>
                            <th>Score</th>
                            <th>Area (px²)</th>
                            <th>Area %</th>
                            {/* <th>BBox</th> */}
                          </tr>
                        </thead>
                        <tbody>
                          {boxes.map((b, i) => {
                            const corners = Array.isArray(b.corners) ? b.corners : [];
                            let bbox = '-';
                            if (corners.length >= 4) {
                              const xs = corners.map((c: any) => Number(c?.[0])).filter(n => Number.isFinite(n));
                              const ys = corners.map((c: any) => Number(c?.[1])).filter(n => Number.isFinite(n));
                              if (xs.length && ys.length) {
                                const x = Math.min(...xs);
                                const y = Math.min(...ys);
                                const w = Math.max(...xs) - x;
                                const h = Math.max(...ys) - y;
                                bbox = `${x},${y} ${w}×${h}`;
                              }
                            }
                            const score = typeof b.score === 'number' ? `${(b.score * 100).toFixed(1)}%` : '-';
                            const areaPct = typeof b.area_pct === 'number' ? `${b.area_pct.toFixed(2)}%` : '-';
                            return (
                              <tr key={i}>
                                <td>{i + 1}</td>
                                <td>{b.class || '-'}</td>
                                <td>{score}</td>
                                <td>{b.area ?? '-'}</td>
                                <td>{areaPct}</td>
                                {/* <td className="wrinkle-bbox-cell">{bbox}</td> */}
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  // ── Char (ML) verification — per-character ML predictions ──
  const renderCharVerificationTable = (cameraResult: CameraResult) => {
    const framesWithChar = cameraResult.frames.filter(f => f.char_verification);
    const cameraInfo = getCameraInfo(cameraResult.serial_number);
    const displayName = cameraInfo ? cameraInfo.camera_id : cameraResult.serial_number;

    if (framesWithChar.length === 0) return null;

    return (
      <div className="camera-verification-card">
        <div className="camera-verification-header">
          <div className="camera-info">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
              <rect x="2" y="7" width="20" height="14" rx="2" stroke="currentColor" strokeWidth="2"/>
              <circle cx="12" cy="14" r="3" stroke="currentColor" strokeWidth="2"/>
            </svg>
            <span className="camera-serial"
                  title={cameraInfo ? `${cameraInfo.location} (${cameraResult.serial_number})` : cameraResult.serial_number}>
              {displayName}
            </span>
          </div>
        </div>

        {framesWithChar.map((frame) => {
          const verification = frame.char_verification!;
          const okCount = verification.results.filter(r => r.match).length;
          const totalCount = verification.results.length;
          const allMatch = verification.all_match;

          return (
            <div key={frame.frame_idx} className="frame-verification-section">
              <div className="frame-verification-header">
                <span className="frame-label">{frame.template_name || `Frame ${frame.frame_idx}`}</span>
                <span className={`match-summary ${allMatch ? 'all-match' : 'has-mismatch'}`}>
                  {okCount}/{totalCount} OK {allMatch ? '✓' : '✗'}
                </span>
              </div>

              {/* Visual badge row — char + p_ok% inline for at-a-glance review */}
              <div className="char-badge-row">
                {verification.results.map((r) => (
                  <span
                    key={r.annotation_idx}
                    className={`char-badge ${r.match ? 'ok' : 'ng'}`}
                    title={`#${r.annotation_idx} · expected ${r.expected} · ${r.ml_label}${r.error ? ` · ${r.error}` : ''}`}
                  >
                    <span className="char-badge-char">{r.expected || '?'}</span>
                    <span className="char-badge-conf">{(r.ml_p_ok * 100).toFixed(0)}%</span>
                  </span>
                ))}
              </div>

              {/* Detail table — always shown so QC can read exact p_ok per char */}
              <div className="verification-table-wrapper">
                <table className="verification-table">
                  <thead>
                    <tr>
                      <th className="col-region">#</th>
                      <th className="col-expected">Char</th>
                      <th className="col-match">AI</th>
                      <th className="col-confidence">p_ok</th>
                      <th className="col-confidence">Match</th>
                      <th className="col-confidence">Threshold</th>
                      <th className="col-confidence">Diff Mask</th>
                    </tr>
                  </thead>
                  <tbody>
                    {verification.results.map((r) => (
                      <tr key={r.annotation_idx} className={r.match ? '' : 'mismatch-row'}>
                        <td className="col-region">{r.annotation_idx}</td>
                        <td className="col-expected" style={{ fontFamily: 'monospace', fontWeight: 600 }}>
                          {r.expected || '—'}
                        </td>
                        <td className="col-match">
                          <span className={`match-icon ${r.ml_label === 'OK' ? 'match' : 'no-match'}`}>
                            {r.ml_label}
                          </span>
                        </td>
                        <td className="col-confidence">
                          <span className={`confidence-value ${getConfidenceClass(r.ml_p_ok)}`}>
                            {(r.ml_p_ok * 100).toFixed(1)}%
                          </span>
                        </td>
                        <td className="col-confidence">
                          <span className={`match-icon ${r.match ? 'match' : 'no-match'}`}>
                            {r.match ? '✓' : '✗'}
                          </span>
                        </td>
                        <td className="col-confidence">
                          <span className="confidence-value">
                            {(r.threshold * 100).toFixed(1)}%
                          </span>
                        </td>
                        <td className="col-confidence">
                          {r.mask_diff_b64 ? (
                            <img
                              src={`data:image/png;base64,${r.mask_diff_b64}`}
                              alt={`diff mask #${r.annotation_idx}`}
                              title={`#${r.annotation_idx} · diff XOR (white = differing pixels)`}
                              style={{
                                width: 32,
                                height: 32,
                                imageRendering: 'pixelated',
                                border: '1px solid var(--border-color, #ccc)',
                                background: '#000',
                                cursor: 'zoom-in',
                              }}
                              onClick={(e) => {
                                // Toggle 32 ↔ 128 on click for quick visual inspection
                                const img = e.currentTarget;
                                const big = img.style.width === '128px';
                                img.style.width = big ? '32px' : '128px';
                                img.style.height = big ? '32px' : '128px';
                              }}
                            />
                          ) : (
                            <span style={{ color: 'var(--text-muted, #888)' }}>—</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  // Render empty state
  const renderEmptyState = () => (
    <div className="inference-realtime-empty">
      <div className="empty-state">
        <svg width="64" height="64" viewBox="0 0 24 24" fill="none">
          <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
          <path d="M12 6v6l4 2" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
        </svg>
        <h3>No Recipe Running</h3>
        <p>Load a recipe from the Recipes page to start inference</p>
      </div>
    </div>
  );

  // Render main content
  const renderMainContent = () => (
    <div className="inference-realtime">
      <div className={`realtime-header ${isHeaderCollapsed ? 'collapsed' : ''}`}>
        <div className="header-info">
          <h2>Inference Realtime Monitor</h2>
          {runningRecipe && (
            <div className="running-recipe-info">
              <span className="recipe-badge">
                <span className="pulse-dot"></span>
                RUNNING
              </span>
              <span className="recipe-name">{runningRecipe.metadata?.name || 'Unknown Recipe'}</span>
              <span className="product-code">({runningRecipe.metadata?.product_code || 'N/A'})</span>
              <div className="recipe-time-info">
                {recipeStartTime && (
                  <>
                    <div className="time-item">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                        <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
                        <path d="M12 6v6l4 2" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                      </svg>
                      <span className="time-label">Started:</span>
                      <span className="time-value">{recipeStartTime.toLocaleTimeString()}</span>
                    </div>
                    <div className="time-separator">•</div>
                    <div className="time-item">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                        <path d="M12 2v20M17 12H7" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                      </svg>
                      <span className="time-label">Duration:</span>
                      <span className="time-value duration">{currentDuration}</span>
                    </div>
                  </>
                )}
              </div>
            </div>
          )}
        </div>
        <div className="header-actions">
          {/* Online/Offline Toggle Switch */}
          <div className="mode-toggle-wrapper">
            <span className="mode-label">Inference Mode:</span>
            <button
              className={`mode-toggle-switch ${isOnline ? 'online' : 'offline'}`}
              onClick={handleToggleInferenceMode}
              title={isOnline ? 'Click to switch to OFFLINE mode' : 'Click to switch to ONLINE mode'}
            >
              <span className="toggle-track">
                <span className="toggle-thumb"></span>
              </span>
              <span className="toggle-label">
                {isOnline ? (
                  <>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
                      <path d="M12 6v6l4 2" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                    </svg>
                    ONLINE
                  </>
                ) : (
                  <>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                      <rect x="6" y="6" width="12" height="12" stroke="currentColor" strokeWidth="2"/>
                    </svg>
                    OFFLINE
                  </>
                )}
              </span>
            </button>
          </div>

          {/* Settings Button */}
          <button
            className="btn-action btn-settings"
            onClick={() => setShowSettingsModal(true)}
            title="Realtime Settings (not saved to DB)"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2"/>
              <path d="M12 1v6m0 6v10M1 12h6m6 0h10" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              <path d="M4.22 4.22l4.24 4.24m5.66 5.66l4.24 4.24M4.22 19.78l4.24-4.24m5.66-5.66l4.24-4.24" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
            Settings
          </button>

          {/* Simulate Trigger */}
          <button
            className={`btn-action btn-trigger ${isSimulating ? 'simulating' : ''}`}
            onClick={handleSimulateTrigger}
            disabled={isSimulating}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M13 2L3 14h8l-1 8 10-12h-8l1-8z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
            </svg>
            {isSimulating ? 'Triggering...' : 'Simulate Trigger'}
          </button>

          {/* Stop Recipe */}
          <button
            className={`btn-action btn-stop ${isStopping ? 'stopping' : ''}`}
            onClick={handleStopRecipe}
            disabled={isStopping}
            title="Stop recipe and return to idle"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <rect x="6" y="6" width="12" height="12" fill="currentColor"/>
            </svg>
            {isStopping ? 'Stopping...' : 'Stop Recipe'}
          </button>

          {/* Toggle Header Collapse */}
          <button
            className="btn-toggle-header"
            onClick={() => setIsHeaderCollapsed(!isHeaderCollapsed)}
            title={isHeaderCollapsed ? 'Expand header' : 'Collapse header'}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              {isHeaderCollapsed ? (
                <path d="M19 9l-7 7-7-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              ) : (
                <path d="M5 15l7-7 7 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              )}
            </svg>
          </button>
        </div>
      </div>

      <div className="realtime-content">
        <div className="cameras-grid">
          <div className="panel-header">
            <div className="panel-header-left">
              <h3>Latest Result</h3>
              {/* Template Toggle Arrow Button */}
              <button
                className={`btn-toggle-templates-arrow ${showTemplates ? 'expanded' : ''}`}
                onClick={() => setShowTemplates(!showTemplates)}
                title={showTemplates ? 'Hide templates' : 'Show templates'}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                  <path d="M19 9l-7 7-7-7" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
              {/* Get Live Template Button */}
              <button
                className="btn-get-live-template"
                onClick={handleGetLiveTemplates}
                disabled={isGettingLiveTemplate || !latestResults}
                title="Capture current frames as template"
              >
                {isGettingLiveTemplate ? (
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" style={{animation: 'spin 1s linear infinite'}}>
                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" strokeDasharray="30 70" strokeLinecap="round"/>
                  </svg>
                ) : (
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                    <rect x="2" y="7" width="20" height="14" rx="2" stroke="currentColor" strokeWidth="2"/>
                    <circle cx="12" cy="14" r="3" stroke="currentColor" strokeWidth="2"/>
                    <path d="M8 7V5a2 2 0 012-2h4a2 2 0 012 2v2" stroke="currentColor" strokeWidth="2"/>
                  </svg>
                )}
              </button>
              {/* Return to Original Templates Button */}
              {Object.keys(liveTemplateFrames).length > 0 && (
                <button
                  className="btn-return-template"
                  onClick={handleReturnOriginalTemplates}
                  title="Return to original templates"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                    <path d="M9 14l-4-4 4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M5 10h8a6 6 0 010 12H7" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                </button>
              )}
            </div>
            {latestResults?.statistics && (
              <div className="statistics-summary">
                <div className="stat-badge stat-triggers">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                    <path d="M13 2L3 14h8l-1 8 10-12h-8l1-8z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
                  </svg>
                  <span className="stat-label">Triggers:</span>
                  <span className="stat-value">{latestResults.statistics.total_triggers}</span>
                </div>
                <div className="stat-badge stat-inferences">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
                    <path d="M12 6v6l4 2" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                  <span className="stat-label">Inferences:</span>
                  <span className="stat-value">{latestResults.statistics.total_inferences}</span>
                </div>
                <div className="stat-badge stat-pass">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
                    <path d="M9 12l2 2 4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                  <span className="stat-label">Pass:</span>
                  <span className="stat-value">{latestResults.statistics.total_pass}</span>
                </div>
                <div className="stat-badge stat-fail">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
                    <path d="M15 9l-6 6m0-6l6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                  <span className="stat-label">Fail:</span>
                  <span className="stat-value">{latestResults.statistics.total_fail}</span>
                </div>
                <div className="stat-badge stat-reject">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                    <path d="M12 9v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                  <span className="stat-label">Reject:</span>
                  <span className="stat-value">{latestResults.statistics.total_reject}</span>
                </div>
              </div>
            )}
          </div>

          {!latestResults ? (
            <div className="no-results">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
                <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="2"/>
                <circle cx="8.5" cy="8.5" r="1.5" fill="currentColor"/>
                <path d="M21 15l-5-5L5 21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              <p>Waiting for first result...</p>
            </div>
          ) : (
            <div className={`cameras-container cameras-${Math.min(latestResults.camera_results.length, 6)}`}>
              {latestResults.camera_results.map((cameraResult) => {
                // Find per-camera stats
                const cameraStats = latestResults.metadata?.inference_stats?.per_camera_stats?.find(
                  (s) => s.serial_number === cameraResult.serial_number
                );

                // Determine overall pass/fail status based on frames
                const overallPassFail = cameraResult.frames.some(f => f.pass_fail.toLowerCase() === 'fail') ? 'fail' : 'pass';

                return (
                  <div key={cameraResult.serial_number} className={`camera-card-infer ${overallPassFail}`}>
                    {/* <div className="camera-card-header">
                      <div className="camera-info">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                          <rect x="2" y="7" width="20" height="14" rx="2" stroke="currentColor" strokeWidth="2"/>
                          <circle cx="12" cy="14" r="3" stroke="currentColor" strokeWidth="2"/>
                        </svg>
                        <span className="camera-serial">{cameraResult.serial_number}</span>
                      </div>
                      {cameraStats && (
                        <span className={`camera-status ${cameraStats.frame_stats.fail_count > 0 ? 'fail' : 'pass'}`}>
                          {cameraStats.frame_stats.fail_count > 0 ? 'FAIL' : 'PASS'}
                        </span>
                      )}
                    </div> */}

                    <div className={`camera-frames ${overallPassFail}`}>
                      {cameraResult.frames.map((frame) => {
                        const imageUrl = frame.image_base64
                          ? `data:image/jpeg;base64,${frame.image_base64}`
                          : frame.image_path
                            ? `/api/uploads/${frame.image_path}`
                            : null;

                        // Find cumulative stats for this camera
                        const cumulativeStats = latestResults.statistics?.per_camera?.find(
                          (s) => s.serial_number === cameraResult.serial_number
                        );

                        // Get camera info for display
                        const cameraInfo = getCameraInfo(cameraResult.serial_number);

                        return (
                          <div key={frame.frame_idx} className="frame-container">
                            <div className="frame-aspect-wrapper">
                              {imageUrl ? (
                                <img src={imageUrl} alt={`Frame ${frame.frame_idx}`} className="frame-image" />
                              ) : (
                                <div className="frame-placeholder">
                                  <span>Frame {frame.frame_idx}</span>
                                </div>
                              )}
                              <div className="frame-overlay">
                                {/* Top Row: Camera ID + Offset (center) + Pass/Fail Badge */}
                                <div className="frame-top-row">
                                  <span
                                    className="camera-serial-badge"
                                    title={cameraInfo ? `${cameraInfo.location} (${cameraResult.serial_number})` : cameraResult.serial_number}
                                  >
                                    {cameraInfo ? cameraInfo.camera_id : cameraResult.serial_number}
                                  </span>
                                  {frame.product_verification?.center_alignment_check && (
                                    <span className="frame-offset-x">
                                      Offset: {(frame.product_verification.center_alignment_check.abs_offset_x ?? Math.abs(frame.product_verification.center_alignment_check.offset_x)).toFixed(1)}px
                                      {frame.product_verification.center_alignment_check.direction === 'left' ? ' L' : ' R'}
                                      (L:{frame.product_verification.center_alignment_check.threshold_left?.toFixed(0) ?? '-'} / R:{frame.product_verification.center_alignment_check.threshold_right?.toFixed(0) ?? '-'})
                                    </span>
                                  )}
                                  <span className={`frame-badge ${frame.pass_fail.toLowerCase()}`}>
                                    {frame.pass_fail}
                                  </span>
                                </div>

                                {/* Bottom Row: Counters */}
                                {cumulativeStats && (
                                  <div className="frame-bottom-row">
                                    <div className="counter-group">
                                      <span className="counter-label">Session:</span>
                                      <span className="counter-values">
                                        <span className="counter-pass">✓{cumulativeStats.pass_count}</span>
                                        <span className="counter-separator">/</span>
                                        <span className="counter-fail">✗{cumulativeStats.fail_count}</span>
                                      </span>
                                    </div>
                                    <div className="counter-group">
                                      <span className="counter-label">Total:</span>
                                      <span className="counter-values">
                                        <span className="counter-pass">✓{cumulativeStats.total_pass_count}</span>
                                        <span className="counter-separator">/</span>
                                        <span className="counter-fail">✗{cumulativeStats.total_fail_count}</span>
                                      </span>
                                    </div>
                                  </div>
                                )}

                                {/* Bottom Right: Score and Time Stats */}
                                {cameraStats && (
                                  <div className="frame-stats-overlay">
                                    <div className="stat-item">
                                      <span className="stat-label">Score:</span>
                                      <span className="stat-value">{(cameraStats.confidence * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="stat-separator">|</div>
                                    <div className="stat-item">
                                      <span className="stat-label">Time:</span>
                                      <span className="stat-value">
                                        {(
                                          (cameraStats.timings.trt_inference || 0) +
                                          (cameraStats.timings.postprocess || 0) +
                                          (cameraStats.timings.product_verification_ms || 0) +
                                          (cameraStats.timings.text_verification_ms || 0)
                                        ).toFixed(1)} ms
                                      </span>
                                    </div>
                                  </div>
                                )}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    {/* Template Preview Section - Show all templates */}
                    {showTemplates && cameraResult.frames.length > 0 && (
                      <div className="template-preview-section">
                        {liveTemplateFrames[cameraResult.serial_number] ? (
                          // Live frames (annotated overlay) + "Set as Template" button per frame
                          liveTemplateFrames[cameraResult.serial_number]!.map((imgSrc, frameIdx) => {
                            const key = `${cameraResult.serial_number}_${frameIdx}`;
                            const isSetting = settingTemplateKey === key;
                            const templateName = cameraResult.frames[frameIdx]?.template_name;
                            return (
                              <div key={frameIdx} className="template-preview-item">
                                <div className="template-image-wrapper">
                                  <img
                                    src={imgSrc}
                                    alt={`Live Frame ${frameIdx + 1}`}
                                    className="template-image"
                                  />
                                  <div className="template-preview-header">
                                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                                      <rect x="2" y="7" width="20" height="14" rx="2" stroke="currentColor" strokeWidth="2"/>
                                      <circle cx="12" cy="14" r="3" stroke="currentColor" strokeWidth="2"/>
                                    </svg>
                                    <span className="template-label">Live {templateName ?? frameIdx + 1}</span>
                                  </div>
                                  {/* Set as Template button */}
                                  <button
                                    className={`btn-set-as-template ${isSetting ? 'loading' : ''}`}
                                    onClick={() => handleSetAsTemplate(cameraResult.serial_number, frameIdx)}
                                    disabled={isSetting || !!settingTemplateKey}
                                    title="Upload this frame as new template with updated annotations"
                                  >
                                    {isSetting ? (
                                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" style={{animation: 'spin 1s linear infinite'}}>
                                        <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" strokeDasharray="30 70" strokeLinecap="round"/>
                                      </svg>
                                    ) : (
                                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                                        <path d="M9 12l2 2 4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                                        <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="2"/>
                                      </svg>
                                    )}
                                    {isSetting ? 'Setting...' : 'Set as Template'}
                                  </button>
                                </div>
                              </div>
                            );
                          })
                        ) : (
                          // Original template images from recipe
                          cameraResult.frames.map((frame, frameIdx) => {
                            const templateImageUrl = getTemplateImageUrl(cameraResult.serial_number, frame.template_name);

                            if (!templateImageUrl) return null;

                            return (
                              <div key={frameIdx} className="template-preview-item">
                                <div className="template-image-wrapper">
                                  <img
                                    src={templateImageUrl}
                                    alt={`Template ${frame.template_name}`}
                                    className="template-image"
                                  />
                                  <div className="template-preview-header">
                                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                                      <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="2"/>
                                      <path d="M3 9h18M9 3v18" stroke="currentColor" strokeWidth="2"/>
                                    </svg>
                                    <span className="template-label">
                                      {frame.template_name}
                                    </span>
                                  </div>
                                </div>
                              </div>
                            );
                          })
                        )}
                      </div>
                    )}

                  </div>
                );
              })}
            </div>
          )}
        </div>

        <div className="log-panel">
          <div className="panel-header">
            <h3>Results Log</h3>
            <span className="log-count">{logs.length} entries</span>
          </div>
          <div className="log-container">
            {logs.length === 0 ? (
              <div className="no-logs">
                <p>No results yet. Click "Simulate Trigger" to test.</p>
              </div>
            ) : (
              <div className="log-list">
                {logs.map((log) => {
                  const showWrinkle = hasWrinkleVerification(log);
                  const hasVerification = hasTextVerification(log) || hasCharVerification(log) || showWrinkle;
                  const showChar = hasCharVerification(log);
                  const isExpanded = expandedLogs.has(log.id);

                  return (
                    <div
                      key={log.id}
                      className={`inference-log-entry ${log.result.toLowerCase()} ${hasVerification ? 'expandable' : ''} ${isExpanded ? 'expanded' : ''}`}
                    >
                      <div
                        className="log-entry-header"
                        onClick={() => hasVerification && toggleLogExpansion(log.id)}
                      >
                        <span className="log-timestamp">{log.timestamp}</span>
                        <span className={`log-badge ${log.result.toLowerCase()}`}>
                          {log.result}
                        </span>
                        <span className="log-message">{log.message}</span>
                        {hasVerification && (
                          <span className="log-expand-icon">
                            {isExpanded ? '▲' : '▼'}
                          </span>
                        )}
                      </div>

                      {hasVerification && isExpanded && log.inferenceResult && (
                        <div className="verification-sections">
                          <div className="text-verification-section">
                            <div className="verification-label">Text Verification Details:</div>
                            {log.inferenceResult.camera_results.map((cameraResult) => (
                              <div key={cameraResult.serial_number}>
                                {renderTextVerificationTable(cameraResult, log.id)}
                              </div>
                            ))}
                          </div>
                          {showChar && (
                            <div className="char-verification-section">
                              <div className="verification-label">Character ML Inspection:</div>
                              {log.inferenceResult.camera_results.map((cameraResult) => (
                                <div key={cameraResult.serial_number}>
                                  {renderCharVerificationTable(cameraResult)}
                                </div>
                              ))}
                            </div>
                          )}
                          <div className="template-verification-section">
                            <div className="verification-label">Template Matching Details:</div>
                            {log.inferenceResult.camera_results.map((cameraResult) => (
                              <div key={cameraResult.serial_number}>
                                {renderTemplateVerificationInfo(cameraResult)}
                              </div>
                            ))}
                          </div>
                          {showWrinkle && (
                            <div className="wrinkle-verification-section">
                              <div className="verification-label">Wrinkle Detection Details:</div>
                              {log.inferenceResult.camera_results.map((cameraResult) => (
                                <div key={cameraResult.serial_number}>
                                  {renderWrinkleVerificationInfo(cameraResult, log.id)}
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
                <div ref={logsEndRef} />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Settings Modal */}
      <InferenceRealtimeSettingsModal
        isOpen={showSettingsModal}
        onClose={() => setShowSettingsModal(false)}
        runningRecipe={runningRecipe}
        onUpdate={async (updatedData) => {
          if (!runningRecipeId) return;

          // Snapshot metadata BEFORE setState — async state mustn't leak into DB payload build
          const metadataSnapshot = runningRecipe?.metadata;

          try {
            const merged = await applyRealtimeAndPersist(
              runningRecipeId,
              metadataSnapshot,
              updatedData,
              'Recipe updated and saved to database',
              'Applied to camera service only',
            );
            setRunningRecipe((prev: any) =>
              prev ? { ...prev, metadata: { ...prev.metadata, ...merged } } : prev
            );
          } catch (error: any) {
            console.error('Error updating recipe realtime:', error);
            toast.error(error.response?.data?.detail || 'Failed to update recipe');
            throw error;
          }
        }}
      />

      {/* Template Text Edit Dialog */}
      <TemplateTextEditDialog
        isOpen={!!pendingTemplateData}
        templateName={pendingTemplateData?.templateName}
        items={pendingTemplateData?.textEditItems ?? []}
        isLoading={isConfirmingTemplate}
        onConfirm={handleDialogConfirm}
        onCancel={handleDialogCancel}
      />

      {/* Confirm Dialog */}
      <ConfirmDialog
        isOpen={confirmDialog.isOpen}
        onClose={() => setConfirmDialog({ ...confirmDialog, isOpen: false })}
        onConfirm={() => {
          if (confirmDialog.onConfirm) {
            confirmDialog.onConfirm();
          }
          setConfirmDialog({ ...confirmDialog, isOpen: false });
        }}
        title={confirmDialog.title}
        message={confirmDialog.message}
        type={confirmDialog.type}
        confirmText="Confirm"
        cancelText="Cancel"
      />
    </div>
  );

  // Return with or without overlay wrapper based on embedded prop
  if (!runningRecipeId) {
    if (embedded) {
      return renderEmptyState();
    }
    return (
      <div className="inference-realtime-overlay">
        <div className="inference-realtime-container">
          {renderEmptyState()}
          {onClose && (
            <div className="inference-realtime-footer">
              <button className="btn-close-inference" onClick={onClose}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <path d="M19 12H5M12 19l-7-7 7-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                Back to Dashboard
              </button>
            </div>
          )}
        </div>
      </div>
    );
  }

  // Main content - embedded mode (no overlay)
  if (embedded) {
    return renderMainContent();
  }

  // Main content - overlay mode (with backdrop and footer)
  return (
    <div className="inference-realtime-overlay">
      <div className="inference-realtime-container">
        {renderMainContent()}
        {onClose && (
          <div className="inference-realtime-footer">
            <button className="btn-close-inference" onClick={onClose}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M19 12H5M12 19l-7-7 7-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              Back to Dashboard
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
