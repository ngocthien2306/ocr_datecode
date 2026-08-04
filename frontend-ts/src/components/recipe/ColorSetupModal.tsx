import React, { useEffect, useRef, useState, useCallback } from 'react';
import '@/styles/ColorSetupModal.css';
import { API_BASE_URL } from '@/config/api';
import { drawProfileChart, isDarkMode, useDarkMode, type Pt, type SideProfile } from './edgeProfileChart';

// EdgeParams subset that matters for cap-edge detection. Field names must match
// EdgeParams in ai_services/.../image_proc_detector.py — it rebuilds them via
// EdgeParams.from_config() and silently ignores anything it doesn't know.
export interface CapEdgeConfig {
  detect_mode: 'gradient' | 'brightness';
  edge_polarity: 'light_to_dark' | 'dark_to_light';
  find_by: 'farthest' | 'nearest' | 'strongest';
  outer_min_hratio: number;
  specular_thr: number;
  edge_width: number;
  strong_thr: number;
  peak_height: number;
  peak_prom: number;
  peak_dist: number;
}

export const DEFAULT_CAP_EDGE_CONFIG: CapEdgeConfig = {
  detect_mode: 'gradient',
  edge_polarity: 'light_to_dark',
  find_by: 'farthest',
  outer_min_hratio: 0.55,
  specular_thr: 230,
  edge_width: 3,
  strong_thr: 0.15,
  peak_height: 0.05,
  peak_prom: 0.02,
  peak_dist: 4,
};

interface CapEdgeResult {
  detected: boolean;
  frac_L: number | null;
  frac_R: number | null;
  col_L: number | null;
  col_R: number | null;
  region_w_L: number | null;
  region_w_R: number | null;
  left_line: Pt[] | null;
  right_line: Pt[] | null;
  profiles?: { left: SideProfile | null; right: SideProfile | null } | null;
  reason: string | null;
}

export interface ColorConfig {
  h_min: number;
  h_max: number;
  s_min: number;
  s_max: number;
  v_min: number;
  v_max: number;
  pixel_threshold: number;
  // Upper bound, absolute px. Catches a MISSING label when the product itself
  // shares the label's colour (yellow label over yellow turmeric): a bare bottle
  // matches even MORE of the range than a printed label does, so a lower bound
  // alone can't see it. 0 = disabled.
  pixel_max: number;
  localization_method?: 'image_proc' | 'superpoint' | 'edge_regions';
  roi_circle: { center: [number, number]; radius: number };
  // edge_regions only: EdgeParams-shaped tuning for the cap-edge detector, and a
  // guard rejecting corrections that would drag the ROI more than N px.
  edge_config?: CapEdgeConfig | null;
  cap_max_shift_px?: number;
  // Bottle detection (image-proc) tuning — passed to color_verifier._detect_bottle
  bottle_sharp_threshold?: number;
  bottle_min_height_ratio?: number;
  bottle_min_aspect?: number;
}

interface ColorSetupModalProps {
  isOpen: boolean;
  templateImage: string;
  templateImageUrl?: string;          // server URL — needed for /detect-bottle-preview
  imageWidth: number;
  imageHeight: number;
  productPolygons: Array<Array<[number, number]>>;
  edgeLeftPolygon?: Pt[] | null;      // 'edge_left' annotation, TEMPLATE pixel coords
  edgeRightPolygon?: Pt[] | null;     // 'edge_right' annotation, TEMPLATE pixel coords
  cropArea?: { x1: number; y1: number; x2: number; y2: number } | null;
  initialConfig?: ColorConfig | null;
  templateName?: string;
  onSave: (config: ColorConfig) => void;
  onClose: () => void;
}

// OpenCV BGR2HSV convention: H in [0,180), S/V in [0,255]
function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  const v = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const delta = v - min;
  let h = 0;
  const s = v === 0 ? 0 : Math.round((delta / v) * 255);
  if (delta !== 0) {
    if (v === r) h = 30 * ((g - b) / delta);
    else if (v === g) h = 60 + 30 * ((b - r) / delta);
    else h = 120 + 30 * ((r - g) / delta);
    if (h < 0) h += 180;
  }
  return [Math.round(h) % 180, s, v];
}

// 5%-of-peak histogram range
function rangeFromHistogram(hist: number[]): [number, number] {
  let peak = 0;
  for (let i = 0; i < hist.length; i++) {
    const v = hist[i] ?? 0;
    if (v > peak) peak = v;
  }
  if (peak === 0) return [0, hist.length - 1];
  const thr = peak * 0.05;
  let lo = -1;
  let hi = -1;
  for (let i = 0; i < hist.length; i++) {
    if ((hist[i] ?? 0) > thr) {
      if (lo < 0) lo = i;
      hi = i;
    }
  }
  return lo < 0 ? [0, hist.length - 1] : [lo, hi];
}

const DEFAULT_CONFIG: ColorConfig = {
  h_min: 0, h_max: 180,
  s_min: 0, s_max: 255,
  v_min: 0, v_max: 255,
  pixel_threshold: 1000,
  pixel_max: 0,
  localization_method: 'image_proc',
  roi_circle: { center: [0, 0], radius: 50 },
  bottle_sharp_threshold: 0.30,
  bottle_min_height_ratio: 0.20,
  bottle_min_aspect: 1.2,
};

const ColorSetupModal: React.FC<ColorSetupModalProps> = ({
  isOpen,
  templateImage,
  templateImageUrl,
  imageWidth,
  imageHeight,
  productPolygons,
  edgeLeftPolygon = null,
  edgeRightPolygon = null,
  cropArea,
  initialConfig,
  templateName,
  onSave,
  onClose,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const histCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const capLeftProfRef = useRef<HTMLCanvasElement | null>(null);
  const capRightProfRef = useRef<HTMLCanvasElement | null>(null);
  // Last histograms, kept so the canvas can be repainted on a theme toggle
  // (it is otherwise only drawn as a side effect of auto-detect).
  const lastHistRef = useRef<{ h: number[]; s: number[]; v: number[] } | null>(null);

  // Caches built once per image load:
  const imageRef = useRef<HTMLImageElement | null>(null);
  const hsvRef = useRef<Uint8Array | null>(null);   // [h0,s0,v0,_,h1,s1,v1,_,...]
  const polyMaskRef = useRef<Uint8Array | null>(null); // 1 byte per pixel
  const polyPixelCountRef = useRef<number>(0);
  const fullImageDataRef = useRef<ImageData | null>(null);

  const [displayScale, setDisplayScale] = useState(1);
  const [config, setConfig] = useState<ColorConfig>({
    ...DEFAULT_CONFIG,
    roi_circle: {
      center: [imageWidth / 2, imageHeight / 2],
      radius: Math.max(20, Math.min(imageWidth, imageHeight) / 12),
    },
  });
  const [matchCount, setMatchCount] = useState(0);
  const [isReady, setIsReady] = useState(false);
  const [draggingCircle, setDraggingCircle] = useState(false);
  // Detected bottle bbox (TEMPLATE pixel coords) from /detect-bottle-preview
  const [detectedBottle, setDetectedBottle] = useState<{ x: number; y: number; w: number; h: number; score: number } | null>(null);
  const [bottleDetecting, setBottleDetecting] = useState(false);
  const [bottleDetectError, setBottleDetectError] = useState<string | null>(null);
  // Cap-edge tuning (localization_method='edge_regions')
  const [capResult, setCapResult] = useState<CapEdgeResult | null>(null);
  const [capDetecting, setCapDetecting] = useState(false);
  const [capError, setCapError] = useState<string | null>(null);
  const [showCapAdvanced, setShowCapAdvanced] = useState(false);
  // Canvas is painted imperatively → must repaint when the theme toggles.
  const darkMode = useDarkMode();

  const method = config.localization_method || 'image_proc';
  const isRegionMode = method === 'edge_regions';
  // The image-proc bottle detector only runs on the legacy path; both SuperPoint
  // modes localise from the match instead.
  const isSuperPointLocalization = method === 'superpoint' || isRegionMode;
  const capRegionsMissing = isRegionMode && (!edgeLeftPolygon || !edgeRightPolygon);
  const capCfg: CapEdgeConfig = { ...DEFAULT_CAP_EDGE_CONFIG, ...(config.edge_config || {}) };
  const setCap = (k: keyof CapEdgeConfig) => (v: number | string) =>
    setConfig((p) => ({
      ...p,
      edge_config: { ...DEFAULT_CAP_EDGE_CONFIG, ...(p.edge_config || {}), [k]: v } as CapEdgeConfig,
    }));

  // Init config from props (edit mode) when modal opens.
  useEffect(() => {
    if (!isOpen) return;
    if (initialConfig) {
      // Hydrate from saved config.
      setConfig({
        ...initialConfig,
        roi_circle: initialConfig.roi_circle || {
          center: [imageWidth / 2, imageHeight / 2],
          radius: Math.max(20, Math.min(imageWidth, imageHeight) / 12),
        },
      });
    } else {
      setConfig({
        ...DEFAULT_CONFIG,
        roi_circle: {
          center: [imageWidth / 2, imageHeight / 2],
          radius: Math.max(20, Math.min(imageWidth, imageHeight) / 12),
        },
      });
    }
  }, [isOpen, initialConfig, imageWidth, imageHeight]);

  // Load image + precompute HSV array + polygon mask.
  useEffect(() => {
    if (!isOpen || !templateImage) return;
    setIsReady(false);
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const w = img.naturalWidth;
      const h = img.naturalHeight;
      const off = document.createElement('canvas');
      off.width = w;
      off.height = h;
      const octx = off.getContext('2d');
      if (!octx) return;
      octx.drawImage(img, 0, 0);
      const imageData = octx.getImageData(0, 0, w, h);
      fullImageDataRef.current = imageData;

      // HSV (reuse alpha channel as padding)
      const hsv = new Uint8Array(imageData.data.length);
      const src = imageData.data;
      for (let i = 0; i < src.length; i += 4) {
        const [hh, ss, vv] = rgbToHsv(src[i]!, src[i + 1]!, src[i + 2]!);
        hsv[i] = hh;
        hsv[i + 1] = ss;
        hsv[i + 2] = vv;
      }
      hsvRef.current = hsv;

      // Polygon mask via canvas fill — STRICT: only pixels inside a product
      // polygon are marked. If no product polygons are provided the mask stays
      // empty so the yellow overlay won't appear anywhere outside the user's
      // drawn product region.
      const maskCanvas = document.createElement('canvas');
      maskCanvas.width = w;
      maskCanvas.height = h;
      const mctx = maskCanvas.getContext('2d');
      if (!mctx) return;
      mctx.fillStyle = 'white';
      for (const poly of productPolygons) {
        if (!poly || poly.length < 3) continue;
        const p0 = poly[0]!;
        mctx.beginPath();
        mctx.moveTo(p0[0], p0[1]);
        for (let i = 1; i < poly.length; i++) {
          const p = poly[i]!;
          mctx.lineTo(p[0], p[1]);
        }
        mctx.closePath();
        mctx.fill();
      }
      const maskData = mctx.getImageData(0, 0, w, h).data;
      const polyMask = new Uint8Array(w * h);
      let polyCount = 0;
      for (let i = 0, j = 0; i < maskData.length; i += 4, j++) {
        if (maskData[i]! > 0) {
          polyMask[j] = 1;
          polyCount++;
        }
      }
      polyMaskRef.current = polyMask;
      polyPixelCountRef.current = polyCount;
      imageRef.current = img;
      setIsReady(true);
    };
    img.onerror = () => {
      console.error('[ColorSetupModal] Failed to load template image');
    };
    img.src = templateImage;
  }, [isOpen, templateImage, productPolygons]);

  // Compute display scale to fit canvas in viewport.
  useEffect(() => {
    if (!isReady) return;
    const MAX_W = 900;
    const MAX_H = 600;
    const scale = Math.min(MAX_W / imageWidth, MAX_H / imageHeight, 1);
    setDisplayScale(scale);
  }, [isReady, imageWidth, imageHeight]);

  // Compute matching count (whole image, no debounce — fast enough at ~2MP).
  const recomputeMatch = useCallback(() => {
    const hsv = hsvRef.current;
    const polyMask = polyMaskRef.current;
    if (!hsv || !polyMask) return;
    const { h_min, h_max, s_min, s_max, v_min, v_max } = config;
    let count = 0;
    for (let i = 0, j = 0; i < hsv.length; i += 4, j++) {
      if (!polyMask[j]) continue;
      const hh = hsv[i]!;
      const ss = hsv[i + 1]!;
      const vv = hsv[i + 2]!;
      if (hh >= h_min && hh <= h_max && ss >= s_min && ss <= s_max && vv >= v_min && vv <= v_max) {
        count++;
      }
    }
    setMatchCount(count);
  }, [config]);

  useEffect(() => {
    if (!isReady) return;
    recomputeMatch();
  }, [isReady, config.h_min, config.h_max, config.s_min, config.s_max, config.v_min, config.v_max, recomputeMatch]);

  // Auto-detect HSV range from circular ROI pixels (5%-of-peak per channel).
  const autoDetectFromRoi = useCallback(() => {
    const hsv = hsvRef.current;
    if (!hsv) return;
    const { center, radius } = config.roi_circle;
    const cx = center[0]!;
    const cy = center[1]!;
    const r2 = radius * radius;
    const hHist = new Array(180).fill(0);
    const sHist = new Array(256).fill(0);
    const vHist = new Array(256).fill(0);
    let inside = 0;
    for (let y = Math.max(0, Math.floor(cy - radius)); y < Math.min(imageHeight, Math.ceil(cy + radius)); y++) {
      for (let x = Math.max(0, Math.floor(cx - radius)); x < Math.min(imageWidth, Math.ceil(cx + radius)); x++) {
        const dx = x - cx;
        const dy = y - cy;
        if (dx * dx + dy * dy > r2) continue;
        const idx = (y * imageWidth + x) * 4;
        const hh = hsv[idx]!;
        const ss = hsv[idx + 1]!;
        const vv = hsv[idx + 2]!;
        hHist[hh] = (hHist[hh] ?? 0) + 1;
        sHist[ss] = (sHist[ss] ?? 0) + 1;
        vHist[vv] = (vHist[vv] ?? 0) + 1;
        inside++;
      }
    }
    if (inside === 0) return;
    const [hmin, hmax] = rangeFromHistogram(hHist);
    const [smin, smax] = rangeFromHistogram(sHist);
    const [vmin, vmax] = rangeFromHistogram(vHist);
    setConfig((prev) => ({
      ...prev,
      h_min: hmin, h_max: hmax,
      s_min: smin, s_max: smax,
      v_min: vmin, v_max: vmax,
    }));
    drawHistogram(hHist, sHist, vHist);
  }, [config.roi_circle, imageWidth, imageHeight]);

  // Re-run auto-detect once ROI/image is ready.
  useEffect(() => {
    if (!isReady) return;
    // Only auto-detect on initial open when no saved config existed.
    if (!initialConfig) autoDetectFromRoi();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isReady]);

  // Render the visible canvas (image + polygons + match overlay + ROI circle).
  const drawDisplay = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    const polyMask = polyMaskRef.current;
    const hsv = hsvRef.current;
    if (!canvas || !img || !polyMask || !hsv) return;
    const dw = Math.round(imageWidth * displayScale);
    const dh = Math.round(imageHeight * displayScale);
    canvas.width = dw;
    canvas.height = dh;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(img, 0, 0, dw, dh);

    // Build a downsampled match overlay (yellow inside product polygons where pixel matches HSV)
    const { h_min, h_max, s_min, s_max, v_min, v_max } = config;
    const overlay = ctx.createImageData(dw, dh);
    const od = overlay.data;
    // Sample original-res mask + hsv at each display pixel
    for (let dy = 0; dy < dh; dy++) {
      const sy = Math.floor(dy / displayScale);
      for (let dx = 0; dx < dw; dx++) {
        const sx = Math.floor(dx / displayScale);
        const srcIdx = sy * imageWidth + sx;
        if (!polyMask[srcIdx]) continue;
        const hsvIdx = srcIdx * 4;
        const hh = hsv[hsvIdx]!;
        const ss = hsv[hsvIdx + 1]!;
        const vv = hsv[hsvIdx + 2]!;
        if (hh >= h_min && hh <= h_max && ss >= s_min && ss <= s_max && vv >= v_min && vv <= v_max) {
          const oi = (dy * dw + dx) * 4;
          od[oi] = 255;        // R
          od[oi + 1] = 255;    // G
          od[oi + 2] = 0;      // B (yellow)
          od[oi + 3] = 160;    // alpha
        }
      }
    }
    ctx.putImageData(_compositeOverlay(ctx, overlay, dw, dh), 0, 0);

    // Draw product polygons (outline only)
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#7513dd';
    for (const poly of productPolygons) {
      if (!poly || poly.length < 3) continue;
      const p0 = poly[0]!;
      ctx.beginPath();
      ctx.moveTo(p0[0] * displayScale, p0[1] * displayScale);
      for (let i = 1; i < poly.length; i++) {
        const p = poly[i]!;
        ctx.lineTo(p[0] * displayScale, p[1] * displayScale);
      }
      ctx.closePath();
      ctx.stroke();
    }

    // Draw detected bottle bbox (orange) from /detect-bottle-preview
    if (detectedBottle) {
      ctx.lineWidth = 3;
      ctx.strokeStyle = '#f59e0b';      // amber
      const bx = detectedBottle.x * displayScale;
      const by = detectedBottle.y * displayScale;
      const bw = detectedBottle.w * displayScale;
      const bh = detectedBottle.h * displayScale;
      ctx.strokeRect(bx, by, bw, bh);
      ctx.font = '14px sans-serif';
      ctx.fillStyle = '#f59e0b';
      ctx.fillText(`bottle ${(detectedBottle.score * 100).toFixed(0)}%`, bx + 4, Math.max(14, by - 4));
    }

    // Cap-edge regions (dashed) + the edge detected inside each (solid).
    // The dashed box is what you drew; the solid line is what the detector found.
    // If the solid line isn't on the cap edge, tune below — that is exactly the
    // measurement inference will regress the colour ROI against.
    if (isRegionMode) {
      const dashPoly = (pts: Pt[] | null | undefined, color: string) => {
        if (!pts || pts.length < 3) return;
        ctx.save();
        ctx.setLineDash([6, 4]);
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = color;
        ctx.fillStyle = color + '22';
        ctx.beginPath();
        ctx.moveTo(pts[0]![0] * displayScale, pts[0]![1] * displayScale);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i]![0] * displayScale, pts[i]![1] * displayScale);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        ctx.restore();
      };
      dashPoly(edgeLeftPolygon, '#fb923c');
      dashPoly(edgeRightPolygon, '#2dd4bf');

      const edgeLine = (ln: Pt[] | null | undefined, color: string) => {
        if (!ln || ln.length < 2) return;
        ctx.save();
        ctx.lineWidth = 3;
        ctx.strokeStyle = color;
        ctx.beginPath();
        ctx.moveTo(ln[0]![0] * displayScale, ln[0]![1] * displayScale);
        ctx.lineTo(ln[1]![0] * displayScale, ln[1]![1] * displayScale);
        ctx.stroke();
        ctx.restore();
      };
      edgeLine(capResult?.left_line, '#facc15');
      edgeLine(capResult?.right_line, '#facc15');
    }

    // Draw ROI circle (cyan)
    const { center, radius } = config.roi_circle;
    const cx0 = center[0]!;
    const cy0 = center[1]!;
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#22d3ee';
    ctx.beginPath();
    ctx.arc(cx0 * displayScale, cy0 * displayScale, radius * displayScale, 0, Math.PI * 2);
    ctx.stroke();
    // Center crosshair
    const ccx = cx0 * displayScale;
    const ccy = cy0 * displayScale;
    ctx.beginPath();
    ctx.moveTo(ccx - 5, ccy);
    ctx.lineTo(ccx + 5, ccy);
    ctx.moveTo(ccx, ccy - 5);
    ctx.lineTo(ccx, ccy + 5);
    ctx.stroke();
  }, [config, displayScale, imageWidth, imageHeight, productPolygons, detectedBottle,
      isRegionMode, edgeLeftPolygon, edgeRightPolygon, capResult]);

  // Composite an overlay ImageData on top of current canvas content
  // (we already drew the image, then need to alpha-blend overlay on top).
  function _compositeOverlay(ctx: CanvasRenderingContext2D, overlay: ImageData, w: number, h: number): ImageData {
    const base = ctx.getImageData(0, 0, w, h);
    const bd = base.data;
    const od = overlay.data;
    for (let i = 0; i < bd.length; i += 4) {
      const a = od[i + 3]!;
      if (a === 0) continue;
      const af = a / 255;
      bd[i] = bd[i]! * (1 - af) + od[i]! * af;
      bd[i + 1] = bd[i + 1]! * (1 - af) + od[i + 1]! * af;
      bd[i + 2] = bd[i + 2]! * (1 - af) + od[i + 2]! * af;
    }
    return base;
  }

  useEffect(() => {
    if (!isReady) return;
    drawDisplay();
  }, [isReady, drawDisplay]);

  function drawHistogram(hHist: number[], sHist: number[], vHist: number[]) {
    lastHistRef.current = { h: hHist, s: sHist, v: vHist };
    const canvas = histCanvasRef.current;
    if (!canvas) return;
    const w = canvas.width;
    const h = canvas.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = isDarkMode() ? '#111827' : '#f3f4f6';
    ctx.fillRect(0, 0, w, h);

    const drawLine = (hist: number[], color: string, yOff: number, hSlot: number) => {
      let peak = 1;
      for (const v of hist) if (v > peak) peak = v;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < hist.length; i++) {
        const x = (i / (hist.length - 1)) * w;
        const y = yOff + hSlot - ((hist[i] ?? 0) / peak) * (hSlot - 2);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };
    const slot = h / 3;
    const dark = isDarkMode();
    drawLine(hHist, dark ? '#f87171' : '#ef4444', 0, slot);
    drawLine(sHist, dark ? '#4ade80' : '#22c55e', slot, slot);
    drawLine(vHist, dark ? '#60a5fa' : '#3b82f6', slot * 2, slot);
  }

  // Repaint the histogram when the theme flips (canvas can't restyle itself).
  useEffect(() => {
    const last = lastHistRef.current;
    if (last) drawHistogram(last.h, last.s, last.v);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [darkMode]);

  // Mouse handling for ROI center drag.
  const getCanvasCoords = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) / displayScale,
      y: (e.clientY - rect.top) / displayScale,
    };
  };

  const onMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const pt = getCanvasCoords(e);
    if (!pt) return;
    setDraggingCircle(true);
    setConfig((prev) => ({
      ...prev,
      roi_circle: { ...prev.roi_circle, center: [pt.x, pt.y] },
    }));
  };
  const onMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!draggingCircle) return;
    const pt = getCanvasCoords(e);
    if (!pt) return;
    setConfig((prev) => ({
      ...prev,
      roi_circle: { ...prev.roi_circle, center: [pt.x, pt.y] },
    }));
  };
  const onMouseUp = () => {
    if (draggingCircle) {
      // Re-sample HSV from the new ROI position so the yellow overlay updates
      // as the user expects ("kéo ROI tới đâu thì pixel match update theo").
      autoDetectFromRoi();
    }
    setDraggingCircle(false);
  };

  // Run BE /templates/detect-bottle-preview to test bottle detection with
  // current tuning params. Overlays the resulting bbox on the canvas.
  const runDetectBottle = useCallback(async () => {
    if (isSuperPointLocalization) {
      setBottleDetectError('Bottle preview is only used for image-proc localization');
      setDetectedBottle(null);
      return;
    }
    if (!templateImageUrl || productPolygons.length === 0) {
      setBottleDetectError('Need server-side image_url and at least one product polygon');
      return;
    }
    setBottleDetecting(true);
    setBottleDetectError(null);
    try {
      const poly = productPolygons[0]!; // use first product polygon as hint
      const body: any = {
        image_url: templateImageUrl,
        product_polygon: poly,
        sharp_threshold: config.bottle_sharp_threshold ?? 0.30,
        min_height_ratio: config.bottle_min_height_ratio ?? 0.20,
        min_aspect: config.bottle_min_aspect ?? 1.2,
      };
      if (cropArea) body.crop_area = cropArea;
      const token = localStorage.getItem('access_token');
      const res = await fetch(
        `${API_BASE_URL}/api/recipes/templates/detect-bottle-preview`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Bypass-Tunnel-Reminder': 'true',
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
          },
          body: JSON.stringify(body),
        }
      );
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`HTTP ${res.status}: ${txt}`);
      }
      const data = await res.json();
      if (data.detected && data.bbox) {
        setDetectedBottle({
          x: data.bbox.x, y: data.bbox.y, w: data.bbox.w, h: data.bbox.h,
          score: data.score ?? 0,
        });
        setBottleDetectError(null);
      } else {
        setDetectedBottle(null);
        setBottleDetectError(data.reason || 'No bottle detected');
      }
    } catch (e: any) {
      setBottleDetectError(String(e?.message || e));
      setDetectedBottle(null);
    } finally {
      setBottleDetecting(false);
    }
  }, [templateImageUrl, productPolygons, cropArea, config.bottle_sharp_threshold,
      config.bottle_min_height_ratio, config.bottle_min_aspect, isSuperPointLocalization]);

  // Run BE /templates/detect-cap-edges-preview with the current tuning. This is
  // the SAME measurement MatcherFactory runs at recipe-build time, so a success
  // here is a `template_cap_edges.json` that will exist at inference.
  const runDetectCapEdges = useCallback(async () => {
    if (!templateImageUrl) {
      setCapError('Save the template first to enable preview');
      return;
    }
    if (!edgeLeftPolygon || !edgeRightPolygon) {
      setCapError("This template has no 'edge_left' / 'edge_right' annotation");
      setCapResult(null);
      return;
    }
    setCapDetecting(true);
    setCapError(null);
    try {
      const token = localStorage.getItem('access_token');
      const res = await fetch(`${API_BASE_URL}/api/recipes/templates/detect-cap-edges-preview`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Bypass-Tunnel-Reminder': 'true',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          image_url: templateImageUrl,
          edge_left_polygon: edgeLeftPolygon,
          edge_right_polygon: edgeRightPolygon,
          params: capCfg,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
      const data: CapEdgeResult = await res.json();
      setCapResult(data);
      if (!data.detected) setCapError(data.reason || 'Cap edge detection failed');
    } catch (e: any) {
      setCapError(String(e?.message || e));
      setCapResult(null);
    } finally {
      setCapDetecting(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [templateImageUrl, edgeLeftPolygon, edgeRightPolygon, JSON.stringify(capCfg)]);

  // Auto-detect when entering region mode, and whenever a discrete choice
  // changes (the sliders stay manual — re-running on every drag would thrash).
  useEffect(() => {
    if (!isReady || !isRegionMode || capRegionsMissing) return;
    void runDetectCapEdges();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isReady, isRegionMode, capCfg.detect_mode, capCfg.edge_polarity, capCfg.find_by]);

  // Render the cap-edge profile charts whenever a result arrives.
  useEffect(() => {
    if (!isRegionMode) return;
    const lbl = { chosen: 'EDGE', reference: 'ref' };
    drawProfileChart(capLeftProfRef.current, capResult?.profiles?.left, 'LEFT cap edge', lbl);
    drawProfileChart(capRightProfRef.current, capResult?.profiles?.right, 'RIGHT cap edge', lbl);
  }, [capResult, isRegionMode, darkMode]);

  // Generic slider handler factory
  const setField = (k: keyof ColorConfig) => (val: number) => {
    setConfig((prev) => ({ ...prev, [k]: val } as ColorConfig));
  };

  // Min/Max validators (keep min <= max)
  const setHMin = (v: number) => setConfig((p) => ({ ...p, h_min: Math.min(v, p.h_max) }));
  const setHMax = (v: number) => setConfig((p) => ({ ...p, h_max: Math.max(v, p.h_min) }));
  const setSMin = (v: number) => setConfig((p) => ({ ...p, s_min: Math.min(v, p.s_max) }));
  const setSMax = (v: number) => setConfig((p) => ({ ...p, s_max: Math.max(v, p.s_min) }));
  const setVMin = (v: number) => setConfig((p) => ({ ...p, v_min: Math.min(v, p.v_max) }));
  const setVMax = (v: number) => setConfig((p) => ({ ...p, v_max: Math.max(v, p.v_min) }));

  if (!isOpen) return null;

  const matchPct =
    polyPixelCountRef.current > 0 ? (matchCount / polyPixelCountRef.current) * 100 : 0;
  const maxOn = (config.pixel_max ?? 0) > 0;
  const overMax = maxOn && matchCount > config.pixel_max;
  const willPass = matchCount >= config.pixel_threshold && !overMax;

  return (
    <div className="color-setup-backdrop" onClick={onClose}>
      <div className="color-setup-modal" onClick={(e) => e.stopPropagation()}>
        <div className="color-setup-header">
          <h3>Setup Color Check {templateName ? `— ${templateName}` : ''}</h3>
          <div className="color-setup-header-actions">
            <button type="button" className="btn btn-secondary" onClick={onClose}>
              Cancel
            </button>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => onSave(config)}
              disabled={!isReady}
            >
              Save
            </button>
          </div>
        </div>

        <div className="color-setup-body">
          {/* LEFT: canvas + histogram below */}
          <div className="color-setup-canvas-col">
            <div className="color-setup-canvas-wrap">
              {!isReady && <div className="color-setup-loading">Loading image…</div>}
              <canvas
                ref={canvasRef}
                onMouseDown={onMouseDown}
                onMouseMove={onMouseMove}
                onMouseUp={onMouseUp}
                onMouseLeave={onMouseUp}
                style={{ cursor: draggingCircle ? 'grabbing' : 'crosshair' }}
              />
              <div className="color-setup-hint">
                <span style={{ color: '#22d3ee' }}>●</span> ROI &nbsp;
                <span style={{ color: '#7513dd' }}>●</span> Product polygon &nbsp;
                {!isRegionMode && <><span style={{ color: '#f59e0b' }}>●</span> Detected bottle &nbsp;</>}
                <span style={{ color: '#facc15' }}>●</span> {isRegionMode ? 'HSV match / detected cap edge' : 'HSV match pixels'}
                {isRegionMode && (
                  <>
                    &nbsp; <span style={{ color: '#fb923c' }}>▢</span> Edge left &nbsp;
                    <span style={{ color: '#2dd4bf' }}>▢</span> Edge right
                  </>
                )}
              </div>
            </div>

            {/* Histogram BELOW canvas */}
            <div className="cs-section cs-histogram-section">
              <div className="cs-section-title">Histogram (R=H, G=S, B=V)</div>
              <canvas ref={histCanvasRef} width={720} height={120} className="cs-histogram" />
            </div>

            {/* Cap-edge profiles — amber peaks pass the coverage threshold */}
            {isRegionMode && (
              <div className="cs-section cs-histogram-section">
                <div className="cs-section-title">Cap edge profile (columns = px outward)</div>
                <div className="cs-cap-profiles">
                  <canvas ref={capLeftProfRef} width={352} height={140} />
                  <canvas ref={capRightProfRef} width={352} height={140} />
                </div>
              </div>
            )}
          </div>

          {/* RIGHT sidebar */}
          <div className="color-setup-controls">
            <div className="cs-section">
              <div className="cs-section-title">Localization</div>
              <div className="cs-row">
                <label>ROI source:</label>
                <select
                  className="cs-num-input"
                  value={config.localization_method || 'image_proc'}
                  onChange={(e) => {
                    const method = e.target.value as ColorConfig['localization_method'];
                    setConfig((prev) => ({ ...prev, localization_method: method }));
                    setDetectedBottle(null);
                    setBottleDetectError(null);
                  }}
                >
                  <option value="image_proc">Image processing</option>
                  <option value="superpoint">SuperPoint product region</option>
                  <option value="edge_regions">SuperPoint + cap edges (round bottle)</option>
                </select>
              </div>
              {isRegionMode && (
                <div className={`cs-row cs-stats ${capRegionsMissing ? 'cs-roi-warn' : ''}`}>
                  <span>
                    {capRegionsMissing ? (
                      <>
                        This template has no <b>edge_left</b> / <b>edge_right</b> annotation.
                        Add both over the cap edges in the template editor, save, then reopen.
                      </>
                    ) : (
                      <>
                        Match on the cap; the detected cap edges slide and scale this product
                        polygon horizontally at inference. Draw the regions <b>at least twice
                        as wide as the cap can drift</b> — if an edge leaves its region the
                        plain SuperPoint polygon is used instead (never worse, just uncorrected).
                      </>
                    )}
                  </span>
                </div>
              )}
              {/* <div className="cs-row cs-stats">
                <span>
                  {isSuperPointLocalization
                    ? 'Inference will use the SuperPoint-transformed product polygon as the color ROI.'
                    : 'Inference will detect the bottle inside crop_area/full frame, then count HSV pixels in that ROI.'}
                </span>
              </div> */}
            </div>

            <div className="cs-section">
              <div className="cs-section-title">ROI</div>
              <SliderRow
                label="Radius (px)"
                min={5}
                max={Math.min(imageWidth, imageHeight) / 2}
                value={config.roi_circle.radius}
                onChange={(v) =>
                  setConfig((prev) => ({
                    ...prev,
                    roi_circle: { ...prev.roi_circle, radius: v },
                  }))
                }
              />
              <button
                type="button"
                className="btn btn-secondary cs-auto-btn"
                onClick={autoDetectFromRoi}
                disabled={!isReady}
              >
                Auto-detect HSV from ROI
              </button>
            </div>

            {/* Cap-edge detection tuning (localization_method='edge_regions') */}
            {isRegionMode && (
              <div className="cs-section">
                <div className="cs-section-title">Cap Edge Detection</div>
                <select
                  className="cs-num-input cs-full"
                  value={capCfg.detect_mode}
                  onChange={(e) => setCap('detect_mode')(e.target.value)}
                  title="How the edge signal is built"
                >
                  <option value="gradient">Gradient — edge |Scharr| (default)</option>
                  <option value="brightness">Brightness — signed edge by polarity</option>
                </select>
                {capCfg.detect_mode === 'brightness' && (
                  <select
                    className="cs-num-input cs-full"
                    value={capCfg.edge_polarity}
                    onChange={(e) => setCap('edge_polarity')(e.target.value)}
                    title="Edge direction when scanning outward from the cap"
                  >
                    <option value="light_to_dark">Polarity — light → dark (bright cap, default)</option>
                    <option value="dark_to_light">Polarity — dark → light (inverted contrast)</option>
                  </select>
                )}
                <select
                  className="cs-num-input cs-full"
                  value={capCfg.find_by}
                  onChange={(e) => setCap('find_by')(e.target.value)}
                  title="Which qualifying peak becomes the cap edge"
                >
                  <option value="farthest">Find by — farthest peak (outermost, default)</option>
                  <option value="nearest">Find by — nearest peak (innermost)</option>
                  <option value="strongest">Find by — strongest peak</option>
                </select>

                <SliderRow label="Coverage" min={0} max={1}
                  value={capCfg.outer_min_hratio} onChange={setCap('outer_min_hratio')} />
                <SliderRow label="Specular" min={0} max={255}
                  value={capCfg.specular_thr} onChange={setCap('specular_thr')} />
                <SliderRow label="Edge width" min={1} max={15}
                  value={capCfg.edge_width} onChange={setCap('edge_width')} />

                <button
                  type="button" className="cs-advanced-toggle"
                  onClick={() => setShowCapAdvanced((v) => !v)}
                  aria-expanded={showCapAdvanced}
                >
                  {showCapAdvanced ? '▾' : '▸'} Advanced
                </button>
                {showCapAdvanced && (
                  <>
                    <SliderRow label="Strong thr" min={0} max={1}
                      value={capCfg.strong_thr} onChange={setCap('strong_thr')} />
                    <SliderRow label="Peak height" min={0} max={1}
                      value={capCfg.peak_height} onChange={setCap('peak_height')} />
                    <SliderRow label="Peak prom" min={0} max={1}
                      value={capCfg.peak_prom} onChange={setCap('peak_prom')} />
                    <SliderRow label="Peak dist" min={1} max={50}
                      value={capCfg.peak_dist} onChange={setCap('peak_dist')} />
                    <SliderRow label="Max shift px" min={0} max={400}
                      value={config.cap_max_shift_px ?? 0}
                      onChange={(v) => setConfig((p) => ({ ...p, cap_max_shift_px: v }))} />
                  </>
                )}

                <button
                  type="button"
                  className="btn btn-secondary cs-auto-btn"
                  onClick={runDetectCapEdges}
                  disabled={!isReady || capDetecting || !templateImageUrl || capRegionsMissing}
                  title={!templateImageUrl ? 'Save template first to enable preview' : ''}
                >
                  {capDetecting ? 'Detecting…' : 'Detect cap edges'}
                </button>

                {capResult?.detected && (
                  <div className="cs-row cs-stats">
                    <span>
                      ✓ Edge at <b>{capResult.col_L}</b>/<b>{capResult.col_R}</b> px of{' '}
                      {capResult.region_w_L}/{capResult.region_w_R} px region<br />
                      Saved reference: frac <b>{capResult.frac_L?.toFixed(3)}</b> /{' '}
                      <b>{capResult.frac_R?.toFixed(3)}</b>
                      {(capResult.frac_L != null && capResult.frac_R != null &&
                        (Math.min(capResult.frac_L, capResult.frac_R) < 0.2 ||
                         Math.max(capResult.frac_L, capResult.frac_R) > 0.8)) && (
                        <><br /><span className="cs-roi-warn">
                          An edge sits near a region border — little room for the cap to
                          drift before it falls outside. Widen or recentre that region.
                        </span></>
                      )}
                    </span>
                  </div>
                )}
                {capError && <div className="cs-roi-warn" style={{ marginTop: 4 }}>{capError}</div>}
              </div>
            )}

            {/* Bottle Detection tuning (image-proc) */}
            <div className="cs-section" style={{ opacity: isSuperPointLocalization ? 0.5 : 1 }}>
              <div className="cs-section-title">Bottle Detection</div>
              <SliderRow
                label="Sharp"
                min={0.05} max={0.95}
                value={config.bottle_sharp_threshold ?? 0.30}
                onChange={(v) => setConfig((p) => ({ ...p, bottle_sharp_threshold: v }))}
              />
              <SliderRow
                label="MinH"
                min={0.05} max={0.95}
                value={config.bottle_min_height_ratio ?? 0.20}
                onChange={(v) => setConfig((p) => ({ ...p, bottle_min_height_ratio: v }))}
              />
              <SliderRow
                label="Aspect"
                min={0.5} max={5.0}
                value={config.bottle_min_aspect ?? 1.2}
                onChange={(v) => setConfig((p) => ({ ...p, bottle_min_aspect: v }))}
              />
              <button
                type="button"
                className="btn btn-secondary cs-auto-btn"
                onClick={runDetectBottle}
                disabled={!isReady || bottleDetecting || !templateImageUrl || isSuperPointLocalization}
                title={
                  isSuperPointLocalization
                    ? 'SuperPoint mode does not use image-proc bottle preview'
                    : (!templateImageUrl ? 'Save template first to enable preview' : '')
                }
              >
                {bottleDetecting ? 'Detecting…' : 'Detect Bottle'}
              </button>
              {isSuperPointLocalization && (
                <div className="cs-roi-warn" style={{ marginTop: 4 }}>
                  Disabled in SuperPoint mode.
                </div>
              )}
              {detectedBottle && (
                <div className="cs-row cs-stats">
                  <span>Detected: <b>{detectedBottle.w}×{detectedBottle.h}</b> px at ({detectedBottle.x}, {detectedBottle.y}), score={detectedBottle.score.toFixed(2)}</span>
                </div>
              )}
              {bottleDetectError && (
                <div className="cs-roi-warn" style={{ marginTop: 4 }}>{bottleDetectError}</div>
              )}
            </div>

            <div className="cs-section">
              <div className="cs-section-title">Hue (0-180)</div>
              <SliderRow label="Min" min={0} max={180} value={config.h_min} onChange={setHMin} />
              <SliderRow label="Max" min={0} max={180} value={config.h_max} onChange={setHMax} />
            </div>
            <div className="cs-section">
              <div className="cs-section-title">Saturation (0-255)</div>
              <SliderRow label="Min" min={0} max={255} value={config.s_min} onChange={setSMin} />
              <SliderRow label="Max" min={0} max={255} value={config.s_max} onChange={setSMax} />
            </div>
            <div className="cs-section">
              <div className="cs-section-title">Value (0-255)</div>
              <SliderRow label="Min" min={0} max={255} value={config.v_min} onChange={setVMin} />
              <SliderRow label="Max" min={0} max={255} value={config.v_max} onChange={setVMax} />
            </div>

            <div className="cs-section">
              <div className="cs-section-title">Pass criterion</div>
              <div className="cs-row">
                <label>Min pixels:</label>
                <input
                  type="number"
                  min={0}
                  step={100}
                  value={config.pixel_threshold}
                  onChange={(e) =>
                    setField('pixel_threshold')(Math.max(0, parseInt(e.target.value) || 0))
                  }
                  className="cs-num-input"
                  title="Below this = wrong label (not enough of the expected colour)"
                />
              </div>
              <div className="cs-row">
                <label>Max pixels:</label>
                <input
                  type="number"
                  min={0}
                  step={100}
                  value={config.pixel_max ?? 0}
                  onChange={(e) =>
                    setField('pixel_max')(Math.max(0, parseInt(e.target.value) || 0))
                  }
                  className="cs-num-input"
                  title="Above this = label missing (bare product showing). 0 = disabled"
                />
              </div>
              <div className="cs-row cs-stats">
                <span>
                  {maxOn
                    ? <>Above <b>{config.pixel_max.toLocaleString()}</b> px ⇒ label missing (bare product shows more of the colour than a printed label does). Set 0 to disable.</>
                    : <>Max = 0 ⇒ upper bound off. Set it when the product itself shares the label's colour, otherwise a bottle with no label still passes.</>}
                </span>
              </div>
              {maxOn && config.pixel_max <= config.pixel_threshold && (
                <div className="cs-roi-warn">
                  Max ({config.pixel_max.toLocaleString()}) is not above Min
                  ({config.pixel_threshold.toLocaleString()}) — nothing can ever pass.
                </div>
              )}
              <div className="cs-row cs-stats">
                <span>Matching: <b>{matchCount.toLocaleString()}</b> px ({matchPct.toFixed(1)}% of product area)</span>
              </div>
              <div className="cs-row cs-stats">
                <span>Product area: {polyPixelCountRef.current.toLocaleString()} px</span>
              </div>
              <div className={`cs-pass-badge ${willPass ? 'pass' : 'fail'}`}>
                Will: {willPass ? 'PASS' : overMax ? 'FAIL — label missing' : 'FAIL'}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

interface SliderRowProps {
  label: string;
  min: number;
  max: number;
  value: number;
  onChange: (v: number) => void;
  step?: number;
}

const SliderRow: React.FC<SliderRowProps> = ({ label, min, max, value, onChange, step }) => {
  // Auto step: integer slider if range > 10, else fine-grained.
  const stepActual = step ?? (max - min > 10 ? 1 : 0.01);
  const decimals = stepActual < 1 ? 2 : 0;
  // Display value: keep float precision when step < 1.
  const displayValue = stepActual < 1
    ? Number(value.toFixed(decimals))
    : Math.round(value);
  return (
    <div className="cs-slider-row">
      <span className="cs-slider-label">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={stepActual}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="cs-slider"
      />
      <input
        type="number"
        min={min}
        max={max}
        step={stepActual}
        value={displayValue}
        onChange={(e) => {
          const v = parseFloat(e.target.value);
          if (!isNaN(v)) onChange(Math.max(min, Math.min(max, v)));
        }}
        className="cs-slider-num"
      />
    </div>
  );
};

export default ColorSetupModal;
