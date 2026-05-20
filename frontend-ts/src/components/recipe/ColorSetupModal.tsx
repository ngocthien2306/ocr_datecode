import React, { useEffect, useRef, useState, useCallback } from 'react';
import '@/styles/ColorSetupModal.css';

export interface ColorConfig {
  h_min: number;
  h_max: number;
  s_min: number;
  s_max: number;
  v_min: number;
  v_max: number;
  pixel_threshold: number;
  roi_circle: { center: [number, number]; radius: number };
}

interface ColorSetupModalProps {
  isOpen: boolean;
  templateImage: string;
  imageWidth: number;
  imageHeight: number;
  productPolygons: Array<Array<[number, number]>>;
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
  roi_circle: { center: [0, 0], radius: 50 },
};

const ColorSetupModal: React.FC<ColorSetupModalProps> = ({
  isOpen,
  templateImage,
  imageWidth,
  imageHeight,
  productPolygons,
  initialConfig,
  templateName,
  onSave,
  onClose,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const histCanvasRef = useRef<HTMLCanvasElement | null>(null);

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
  }, [config, displayScale, imageWidth, imageHeight, productPolygons]);

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
    const canvas = histCanvasRef.current;
    if (!canvas) return;
    const w = canvas.width;
    const h = canvas.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = '#f3f4f6';
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
    drawLine(hHist, '#ef4444', 0, slot);
    drawLine(sHist, '#22c55e', slot, slot);
    drawLine(vHist, '#3b82f6', slot * 2, slot);
  }

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
  const willPass = matchCount >= config.pixel_threshold;

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
              Click/drag on image to move ROI circle. Yellow = pixels matching current HSV range inside product polygon.
            </div>
          </div>

          <div className="color-setup-controls">
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
              <div className="cs-section-title">Histogram (R=H, G=S, B=V)</div>
              <canvas ref={histCanvasRef} width={360} height={120} className="cs-histogram" />
            </div>

            <div className="cs-section">
              <div className="cs-section-title">Pass criterion</div>
              <div className="cs-row">
                <label>Pixel threshold:</label>
                <input
                  type="number"
                  min={0}
                  step={100}
                  value={config.pixel_threshold}
                  onChange={(e) =>
                    setField('pixel_threshold')(Math.max(0, parseInt(e.target.value) || 0))
                  }
                  className="cs-num-input"
                />
              </div>
              <div className="cs-row cs-stats">
                <span>Matching: <b>{matchCount.toLocaleString()}</b> px ({matchPct.toFixed(1)}% of product area)</span>
              </div>
              <div className="cs-row cs-stats">
                <span>Product area: {polyPixelCountRef.current.toLocaleString()} px</span>
              </div>
              <div className={`cs-pass-badge ${willPass ? 'pass' : 'fail'}`}>
                Will: {willPass ? 'PASS' : 'FAIL'}
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
}

const SliderRow: React.FC<SliderRowProps> = ({ label, min, max, value, onChange }) => (
  <div className="cs-slider-row">
    <span className="cs-slider-label">{label}</span>
    <input
      type="range"
      min={min}
      max={max}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className="cs-slider"
    />
    <input
      type="number"
      min={min}
      max={max}
      value={Math.round(value)}
      onChange={(e) => {
        const v = parseFloat(e.target.value);
        if (!isNaN(v)) onChange(Math.max(min, Math.min(max, v)));
      }}
      className="cs-slider-num"
    />
  </div>
);

export default ColorSetupModal;
