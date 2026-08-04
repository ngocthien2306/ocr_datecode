import React, { useEffect, useRef, useState, useCallback } from 'react';
import '@/styles/EdgeSetupModal.css';
import { API_BASE_URL } from '@/config/api';
import { drawProfileChart, useDarkMode, type Pt, type SideProfile } from './edgeProfileChart';

export interface EdgeWalls {
  inner_L: number; inner_R: number;
  outer_L: number; outer_R: number;
  plastic_L: number; plastic_R: number;
}

// Mirrors EdgeConfig / EdgeParams in the backend (image_proc_detector.py).
export interface EdgeConfig {
  outer_search_max: number;
  inner_search_max: number;
  edge_margin: number;
  y_extension: number;
  fill_keep_ratio: number;
  peak_height: number;
  peak_prom: number;
  peak_dist: number;
  strong_thr: number;
  outer_min_hratio: number;
  inner_min_hratio: number;
  inner_tol_px: number;
  specular_thr: number;
  detect_mode: 'gradient' | 'brightness';
  edge_polarity: 'light_to_dark' | 'dark_to_light';
  find_by: 'farthest' | 'nearest' | 'strongest';
  edge_width: number;
  // 'label_strip'  = search strip derived from the label quad (original behaviour)
  // 'edge_regions' = scan the template's user-drawn edge_left/edge_right regions
  anchor_mode: 'label_strip' | 'edge_regions';
  template_walls?: EdgeWalls | null;
}

export const DEFAULT_EDGE_CONFIG: EdgeConfig = {
  outer_search_max: 150,
  inner_search_max: 80,
  edge_margin: 2,
  y_extension: 0.2,
  fill_keep_ratio: 0.6,
  peak_height: 0.05,
  peak_prom: 0.02,
  peak_dist: 4,
  strong_thr: 0.15,
  outer_min_hratio: 0.55,
  inner_min_hratio: 0.20,
  inner_tol_px: 12,
  specular_thr: 230,
  detect_mode: 'gradient',
  edge_polarity: 'light_to_dark',
  find_by: 'farthest',
  edge_width: 3,
  anchor_mode: 'label_strip',
  template_walls: null,
};

interface FrameImage {
  image_path: string;
  org_path: string;
  timestamp: string | null;
  frame_idx: number | null;
  template_name: string | null;
  pass_fail: string;
  label_polygon: Pt[] | null;
}

interface EdgeSetupModalProps {
  isOpen: boolean;
  recipeId?: string;
  serialNumber?: string;
  templateImage: string;              // dataURL/base64 or URL for canvas
  templateImageUrl?: string;          // server URL → preview on the template
  imageWidth: number;
  imageHeight: number;
  labelPolygon: Pt[] | null;          // template label quad, TEMPLATE pixel coords
  productPolygons?: Pt[][];           // template product polygons (display only)
  edgeLeftPolygon?: Pt[] | null;      // 'edge_left' annotation, TEMPLATE pixel coords
  edgeRightPolygon?: Pt[] | null;     // 'edge_right' annotation, TEMPLATE pixel coords
  wallType?: 'outer' | 'inner';
  initialConfig?: EdgeConfig | null;
  templateName?: string;
  onSave: (config: EdgeConfig) => void;
  onClose: () => void;
}

interface ActiveImage {
  canvasSrc: string;       // what <img> loads for the canvas
  previewUrl: string;      // image_url sent to the BE preview
  label: Pt[] | null;      // label polygon in this image's pixel coords
  product: Pt[][];         // product polygons in this image's pixel coords
  isTemplate: boolean;
  key: string;             // unique key (for badges)
}

interface PreviewResult {
  detected: boolean;
  inner_corners: Pt[] | null;
  outer_corners: Pt[] | null;
  corners: Pt[] | null;
  template_walls: EdgeWalls | null;
  detection_info: any | null;
  profiles?: { left: SideProfile | null; right: SideProfile | null } | null;
  edge_regions?: { left: Pt[]; right: Pt[] } | null;  // regions actually scanned, this image's coords
  reason: string | null;
}

function authHeaders(): Record<string, string> {
  const token = localStorage.getItem('access_token');
  return {
    'Content-Type': 'application/json',
    'Bypass-Tunnel-Reminder': 'true',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

const EdgeSetupModal: React.FC<EdgeSetupModalProps> = ({
  isOpen,
  recipeId,
  serialNumber,
  templateImage,
  templateImageUrl,
  imageWidth,
  imageHeight,
  labelPolygon,
  productPolygons = [],
  edgeLeftPolygon = null,
  edgeRightPolygon = null,
  wallType = 'outer',
  initialConfig,
  templateName,
  onSave,
  onClose,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imgElRef = useRef<HTMLImageElement | null>(null);
  const leftProfRef = useRef<HTMLCanvasElement | null>(null);
  const rightProfRef = useRef<HTMLCanvasElement | null>(null);

  const [config, setConfig] = useState<EdgeConfig>({ ...DEFAULT_EDGE_CONFIG, ...(initialConfig || {}) });
  const [active, setActive] = useState<ActiveImage | null>(null);
  const [natW, setNatW] = useState(imageWidth || 0);
  const [natH, setNatH] = useState(imageHeight || 0);
  const [displayScale, setDisplayScale] = useState(1);
  const [imgReady, setImgReady] = useState(false);

  const [preview, setPreview] = useState<PreviewResult | null>(null);
  const [detecting, setDetecting] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [showSearch, setShowSearch] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Recorded-frame loading + per-image test badges
  const [frames, setFrames] = useState<FrameImage[]>([]);
  const [frameResult, setFrameResult] = useState<'PASS' | 'FAIL' | ''>('PASS');
  const [frameLimit, setFrameLimit] = useState(40);
  const [loadingFrames, setLoadingFrames] = useState(false);
  const [framesError, setFramesError] = useState<string | null>(null);
  const [badges, setBadges] = useState<Record<string, 'detected' | 'failed' | 'testing'>>({});
  const [testingAll, setTestingAll] = useState(false);
  // Canvas is painted imperatively → must repaint when the theme toggles.
  const darkMode = useDarkMode();

  // ── Init: config + the template as the active image ───────────────────────
  useEffect(() => {
    if (!isOpen) return;
    setConfig({ ...DEFAULT_EDGE_CONFIG, ...(initialConfig || {}) });
    setActive({
      canvasSrc: templateImage,
      previewUrl: templateImageUrl || '',
      label: labelPolygon,
      product: productPolygons,
      isTemplate: true,
      key: 'template',
    });
    setPreview(null);
    setPreviewError(null);
    setBadges({});
  }, [isOpen]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Load active image, capture natural size ───────────────────────────────
  useEffect(() => {
    if (!isOpen || !active) return;
    setImgReady(false);
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      imgElRef.current = img;
      setNatW(img.naturalWidth);
      setNatH(img.naturalHeight);
      setImgReady(true);
    };
    img.onerror = () => {
      console.error('[EdgeSetupModal] image load failed:', active.canvasSrc);
      setPreviewError(`Failed to load image: ${active.canvasSrc}`);
    };
    img.src = active.canvasSrc;
  }, [isOpen, active]);

  // ── Display scale to fit ──────────────────────────────────────────────────
  useEffect(() => {
    if (!imgReady || !natW || !natH) return;
    const MAX_W = 820, MAX_H = 560;
    setDisplayScale(Math.min(MAX_W / natW, MAX_H / natH, 1));
  }, [imgReady, natW, natH]);

  // ── Draw image + label/product polygons + detected walls ──────────────────
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgElRef.current;
    if (!canvas || !img || !active) return;
    const dw = Math.round(natW * displayScale);
    const dh = Math.round(natH * displayScale);
    canvas.width = dw;
    canvas.height = dh;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(img, 0, 0, dw, dh);

    const poly = (pts: Pt[] | null | undefined, color: string, width = 2) => {
      if (!pts || pts.length < 2) return;
      ctx.lineWidth = width;
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(pts[0]![0] * displayScale, pts[0]![1] * displayScale);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i]![0] * displayScale, pts[i]![1] * displayScale);
      ctx.closePath();
      ctx.stroke();
    };

    // Edge-region mode: draw the regions actually scanned. Prefer the ones the BE
    // echoed back (already mapped onto a recorded frame); fall back to the template
    // annotations so the regions are visible before the first detect.
    if (showSearch && config.anchor_mode === 'edge_regions') {
      const shownL = preview?.edge_regions?.left || (active.isTemplate ? edgeLeftPolygon : null);
      const shownR = preview?.edge_regions?.right || (active.isTemplate ? edgeRightPolygon : null);
      const regionPoly = (pts: Pt[] | null | undefined, color: string) => {
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
      regionPoly(shownL, '#fb923c');
      regionPoly(shownR, '#2dd4bf');
    }

    // Search-range bands (dashed): outer search (blue) outward from label edge,
    // inner search (pink) inward into the label. Drawn first so walls sit on top.
    if (showSearch && config.anchor_mode !== 'edge_regions' && active.label && active.label.length >= 4) {
      const dashPoly = (pts: Pt[], color: string) => {
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
      for (const side of ['left', 'right'] as const) {
        dashPoly(bandQuad(active.label, side, 0, config.outer_search_max, config.y_extension), '#3b82f6');
        dashPoly(bandQuad(active.label, side, -config.inner_search_max, 0, config.y_extension), '#ec4899');
      }
    }

    // Specular-suppressed regions (translucent red) — what `specular_thr` removes.
    // Lower the threshold until glare blobs are covered but the bottle rim is NOT.
    const specFill = (pts: Pt[] | null | undefined) => {
      if (!pts || pts.length < 3) return;
      ctx.save();
      ctx.fillStyle = 'rgba(239,68,68,0.35)';
      ctx.strokeStyle = 'rgba(239,68,68,0.85)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pts[0]![0] * displayScale, pts[0]![1] * displayScale);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i]![0] * displayScale, pts[i]![1] * displayScale);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    };
    for (const reg of preview?.profiles?.left?.specular_regions || []) specFill(reg);
    for (const reg of preview?.profiles?.right?.specular_regions || []) specFill(reg);

    // Product polygons (purple), label (cyan)
    for (const p of active.product) poly(p, '#7513dd', 2);
    poly(active.label, '#22d3ee', 2);

    // Detected walls: outer (yellow), inner (orange)
    if (preview?.outer_corners) poly(preview.outer_corners, '#facc15', 3);
    if (preview?.inner_corners) poly(preview.inner_corners, '#f59e0b', 2);
  }, [active, natW, natH, displayScale, preview, showSearch, edgeLeftPolygon, edgeRightPolygon,
      config.anchor_mode, config.outer_search_max, config.inner_search_max, config.y_extension]);

  useEffect(() => {
    if (!imgReady) return;
    draw();
  }, [imgReady, draw]);

  // Render the L/R edge-profile charts whenever a detection result arrives.
  useEffect(() => {
    const lbl = { chosen: 'OUT', secondary: 'IN', reference: 'pred' };
    drawProfileChart(leftProfRef.current, preview?.profiles?.left, 'LEFT profile', lbl);
    drawProfileChart(rightProfRef.current, preview?.profiles?.right, 'RIGHT profile', lbl);
  }, [preview, darkMode]);

  // ── Build the preview request body ────────────────────────────────────────
  const buildBody = (img: ActiveImage, walls?: EdgeWalls | null) => {
    const { template_walls, ...params } = config;
    const body: any = {
      image_url: img.previewUrl,
      label_polygon: img.label,
      wall_type: wallType,
      params,
    };
    if (config.anchor_mode === 'edge_regions') {
      // Regions live in TEMPLATE coords; the BE maps them onto recorded frames
      // through the label quad, so it needs the template label too.
      body.edge_left_polygon = edgeLeftPolygon;
      body.edge_right_polygon = edgeRightPolygon;
      body.template_label_polygon = labelPolygon;
      return body;   // walls are unused in this mode
    }
    // For non-template frames we MUST supply walls (computed on the template).
    // Only reuse stored walls off-template: on the template itself, omitting them
    // makes the BE recompute from the current params (otherwise the very first
    // detect freezes plastic_L/R and later tuning silently has no effect).
    const w = walls ?? (img.isTemplate ? null : config.template_walls);
    if (w) body.template_walls = w;
    return body;
  };

  const callPreview = useCallback(async (img: ActiveImage, walls?: EdgeWalls | null): Promise<PreviewResult> => {
    const res = await fetch(`${API_BASE_URL}/api/recipes/templates/detect-walls-preview`, {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify(buildBody(img, walls)),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    return res.json();
  }, [config, wallType, edgeLeftPolygon, edgeRightPolygon, labelPolygon]);

  const regionMode = config.anchor_mode === 'edge_regions';
  const regionsMissing = regionMode && (!edgeLeftPolygon || !edgeRightPolygon);
  // Walls are only a prerequisite for saving/among-frames testing in label_strip mode.
  const canSave = regionMode ? !regionsMissing : !!config.template_walls;

  // ── Detect on the CURRENT active image ────────────────────────────────────
  const runDetect = useCallback(async () => {
    if (!active || !active.previewUrl) {
      setPreviewError('Missing server-side image URL (save template first).');
      return;
    }
    if (!active.label || active.label.length < 4) {
      setPreviewError('This image has no label polygon to anchor edge detection.');
      return;
    }
    if (config.anchor_mode === 'edge_regions' && (!edgeLeftPolygon || !edgeRightPolygon)) {
      setPreviewError("Draw an 'edge_left' and an 'edge_right' region on the template first.");
      setPreview(null);
      return;
    }
    setDetecting(true);
    setPreviewError(null);
    try {
      const data = await callPreview(active);
      setPreview(data);
      if (!data.detected) {
        setPreviewError(data.reason || 'Detection failed for this image');
      } else if (active.isTemplate && data.template_walls) {
        // Persist the freshly-computed walls into the config for inference + frame tests.
        setConfig((p) => ({ ...p, template_walls: data.template_walls }));
      }
    } catch (e: any) {
      setPreviewError(String(e?.message || e));
      setPreview(null);
    } finally {
      setDetecting(false);
    }
  }, [active, callPreview, config.anchor_mode, edgeLeftPolygon, edgeRightPolygon]);

  // Auto-detect on the template when it first becomes ready.
  useEffect(() => {
    if (!imgReady || !active?.isTemplate) return;
    void runDetect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imgReady, active?.key]);

  // Re-run detection when the user switches anchoring, detection mode or polarity.
  useEffect(() => {
    if (!imgReady || !active?.previewUrl) return;
    void runDetect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config.anchor_mode, config.detect_mode, config.edge_polarity]);

  // ── Load recorded frames for this camera ──────────────────────────────────
  const loadFrames = useCallback(async () => {
    if (!recipeId || !serialNumber) {
      setFramesError('Recipe/camera not available for loading recorded images.');
      return;
    }
    setLoadingFrames(true);
    setFramesError(null);
    try {
      const qs = new URLSearchParams({
        recipe_id: String(recipeId),
        serial_number: String(serialNumber),
        limit: String(frameLimit),
      });
      if (frameResult) qs.set('result', frameResult);
      const res = await fetch(`${API_BASE_URL}/api/inference-results/frame-images?${qs.toString()}`, {
        headers: authHeaders(),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
      const data: FrameImage[] = await res.json();
      setFrames(data);
      setBadges({});
      if (data.length === 0) setFramesError('No recorded images found for this filter.');
    } catch (e: any) {
      setFramesError(String(e?.message || e));
      setFrames([]);
    } finally {
      setLoadingFrames(false);
    }
  }, [recipeId, serialNumber, frameLimit, frameResult]);

  // ── Pick a recorded frame → make it the active image + detect ─────────────
  const pickFrame = useCallback((f: FrameImage) => {
    setActive({
      canvasSrc: `${API_BASE_URL}/api/uploads/${f.org_path}`,
      previewUrl: f.org_path,
      label: f.label_polygon,
      product: [],
      isTemplate: false,
      key: f.image_path,
    });
    setPreview(null);
    setPreviewError(null);
  }, []);

  // When a frame becomes active + image ready, auto-run detection on it.
  useEffect(() => {
    if (!imgReady || !active || active.isTemplate) return;
    void runDetect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imgReady, active?.key]);

  // ── Test ALL loaded frames → badges (uses saved template_walls) ───────────
  const testAll = useCallback(async () => {
    if (frames.length === 0) return;
    if (!regionMode && !config.template_walls) {
      setPreviewError('Detect on the template first to compute walls, then Test all.');
      return;
    }
    setTestingAll(true);
    const next: Record<string, 'detected' | 'failed' | 'testing'> = {};
    frames.forEach((f) => (next[f.image_path] = 'testing'));
    setBadges({ ...next });
    for (const f of frames) {
      const imgRef: ActiveImage = {
        canvasSrc: '', previewUrl: f.org_path, label: f.label_polygon,
        product: [], isTemplate: false, key: f.image_path,
      };
      try {
        if (!f.label_polygon || f.label_polygon.length < 4) {
          next[f.image_path] = 'failed';
        } else {
          const data = await callPreview(imgRef, config.template_walls);
          next[f.image_path] = data.detected ? 'detected' : 'failed';
        }
      } catch {
        next[f.image_path] = 'failed';
      }
      setBadges({ ...next });
    }
    setTestingAll(false);
  }, [frames, config.template_walls, callPreview, regionMode]);

  if (!isOpen) return null;

  const di = preview?.detection_info;
  const passCount = Object.values(badges).filter((b) => b === 'detected').length;
  const doneCount = Object.values(badges).filter((b) => b !== 'testing').length;

  const num = (k: keyof EdgeConfig) => config[k] as number;
  const setNum = (k: keyof EdgeConfig) => (v: number) =>
    setConfig((p) => ({ ...p, [k]: v } as EdgeConfig));

  return (
    <div className="edge-setup-backdrop" onClick={onClose}>
      <div className="edge-setup-modal" onClick={(e) => e.stopPropagation()}>
        <div className="edge-setup-header">
          <h3>Setup Edge Detection {templateName ? `— ${templateName}` : ''}</h3>
          <div className="edge-setup-header-actions">
            <button type="button" className="btn btn-secondary" onClick={onClose}>Cancel</button>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => onSave(config)}
              disabled={!canSave}
              title={canSave ? '' : (regionMode
                ? "Draw 'edge_left' and 'edge_right' regions on the template first"
                : 'Detect on the template first to compute walls')}
            >
              Save
            </button>
          </div>
        </div>

        <div className="edge-setup-body">
          {/* LEFT: recorded-image list (2-col grid, rows scroll) */}
          <div className="edge-setup-frames-col">
            <div className="es-frames-toolbar">
              <span className="es-frames-title">Recorded images</span>
              <select value={frameResult} onChange={(e) => setFrameResult(e.target.value as any)}>
                <option value="PASS">PASS</option>
                <option value="FAIL">FAIL</option>
                <option value="">All</option>
              </select>
              <label className="es-max">
                Max
                <input
                  type="number" min={1} max={300} value={frameLimit}
                  onChange={(e) => setFrameLimit(Math.max(1, Math.min(300, parseInt(e.target.value) || 1)))}
                />
              </label>
              <button type="button" className="btn btn-secondary" onClick={loadFrames} disabled={loadingFrames}>
                {loadingFrames ? 'Loading…' : 'Load'}
              </button>
              <button
                type="button" className="btn btn-secondary"
                onClick={testAll}
                disabled={frames.length === 0 || testingAll || !canSave}
                title={canSave ? '' : (regionMode
                  ? "Draw the edge regions on the template first"
                  : 'Detect on the template first')}
              >
                {testingAll ? `Testing ${doneCount}/${frames.length}…` : 'Test all'}
              </button>
              {frames.length > 0 && (
                <span className="es-frames-count">{passCount}/{frames.length} detected</span>
              )}
            </div>
            {framesError && <div className="es-frames-error">{framesError}</div>}
            <div className="es-thumbs">
              {frames.map((f) => {
                const b = badges[f.image_path];
                return (
                  <div
                    key={f.image_path}
                    className={`es-thumb ${active?.key === f.image_path ? 'active' : ''}`}
                    onClick={() => pickFrame(f)}
                    title={`${f.pass_fail} · frame ${f.frame_idx ?? '?'} · ${f.timestamp ?? ''}`}
                  >
                    <img src={`${API_BASE_URL}/api/uploads/${f.image_path}`} alt="frame" loading="lazy" />
                    {b && (
                      <span className={`es-thumb-badge ${b}`}>
                        {b === 'detected' ? '✓' : b === 'failed' ? '✗' : '…'}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* MIDDLE: canvas + detection info + profiles */}
          <div className="edge-setup-canvas-col">
            <div className="edge-setup-canvas-wrap">
              {!imgReady && <div className="edge-setup-loading">Loading image…</div>}
              <canvas ref={canvasRef} />
              <div className="edge-setup-hint">
                <span style={{ color: '#22d3ee' }}>●</span> Label &nbsp;
                <span style={{ color: '#7513dd' }}>●</span> Product &nbsp;
                <span style={{ color: '#facc15' }}>●</span> {regionMode ? 'Detected wall' : 'Outer wall'} &nbsp;
                {!regionMode && <><span style={{ color: '#f59e0b' }}>●</span> Inner wall &nbsp;</>}
                {regionMode ? (
                  <>
                    <span style={{ color: '#fb923c' }}>▢</span> Edge left &nbsp;
                    <span style={{ color: '#2dd4bf' }}>▢</span> Edge right &nbsp;
                  </>
                ) : (
                  <>
                    <span style={{ color: '#3b82f6' }}>▢</span> Outer search &nbsp;
                    <span style={{ color: '#ec4899' }}>▢</span> Inner search &nbsp;
                  </>
                )}
                <span style={{ color: '#ef4444' }}>▩</span> Specular cut &nbsp;
                <label className="es-show-search">
                  <input type="checkbox" checked={showSearch} onChange={(e) => setShowSearch(e.target.checked)} />
                  {regionMode ? 'show regions' : 'show search range'}
                </label>
                <b style={{ marginLeft: 8 }}>
                  {active?.isTemplate ? 'Template' : 'Recorded frame'}
                </b>
              </div>
            </div>

            <div className="edge-setup-statusline">
              {detecting && <span>Detecting…</span>}
              {!detecting && preview?.detected && di && regionMode && (
                <span>
                  <b className="es-ok">✓ Detected</b> · edge at L/R={fmt(di.col_L)}/{fmt(di.col_R)}px
                  from the inner side of each region (width {fmt(di.region_w_L)}/{fmt(di.region_w_R)}px)
                </span>
              )}
              {!detecting && preview?.detected && di && !regionMode && (
                <span>
                  <b className="es-ok">✓ Detected</b> · outer L/R={fmt(di.outer_L)}/{fmt(di.outer_R)} ·
                  inner L/R={fmt(di.eff_L)}/{fmt(di.eff_R)} ·
                  hidden L/R={String(di.hidden_L)}/{String(di.hidden_R)}
                </span>
              )}
              {!detecting && previewError && <span className="es-fail">✗ {previewError}</span>}
            </div>

            {/* Edge profile charts (L/R) — peaks + chosen outer(amber)/inner(green)/pred */}
            <div className="edge-setup-profiles">
              <canvas ref={leftProfRef} width={360} height={150} />
              <canvas ref={rightProfRef} width={360} height={150} />
            </div>

          </div>

          {/* RIGHT: param sliders */}
          <div className="edge-setup-controls">
            <div className="es-section">
              <div className="es-section-title">Search area</div>
              <select
                className="es-mode-select"
                value={config.anchor_mode}
                onChange={(e) => setConfig((p) => ({ ...p, anchor_mode: e.target.value as EdgeConfig['anchor_mode'] }))}
                title="Where the edge is searched for"
              >
                <option value="label_strip">Anchor — strip beside the label (default)</option>
                <option value="edge_regions">Anchor — drawn edge_left / edge_right regions</option>
              </select>
              {regionMode && (
                <div className={`es-anchor-hint ${regionsMissing ? 'es-anchor-hint-warn' : ''}`}>
                  {regionsMissing
                    ? "This template has no 'edge_left' / 'edge_right' annotation. Add both in the template editor, save, then reopen."
                    : 'Regions follow the LABEL, so draw them wide enough to cover how far the bottle can shift. "Outer coverage" is measured against the region height — keep the region within the straight part of the wall.'}
                </div>
              )}
            </div>

            <div className="es-section">
              <div className="es-section-title">Detection mode</div>
              <select
                className="es-mode-select"
                value={config.detect_mode}
                onChange={(e) => setConfig((p) => ({ ...p, detect_mode: e.target.value as 'gradient' | 'brightness' }))}
              >
                <option value="gradient">Gradient — edge |Scharr| (default)</option>
                <option value="brightness">Brightness — signed edge by polarity</option>
              </select>
              {config.detect_mode === 'brightness' && (
                <select
                  className="es-mode-select"
                  value={config.edge_polarity}
                  onChange={(e) => setConfig((p) => ({ ...p, edge_polarity: e.target.value as 'light_to_dark' | 'dark_to_light' }))}
                  title="Edge direction when scanning outward from the label"
                >
                  <option value="light_to_dark">Polarity — light → dark (bright rim, default)</option>
                  <option value="dark_to_light">Polarity — dark → light (inverted contrast)</option>
                </select>
              )}
            </div>

            <button
              type="button" className="btn btn-primary es-detect-btn"
              onClick={runDetect} disabled={detecting || !active?.previewUrl}
            >
              {detecting ? 'Detecting…' : (active?.isTemplate ? 'Detect on template' : 'Detect on this frame')}
            </button>

            <div className="es-section">
              <div className="es-section-title">Tuning</div>
              <select
                className="es-mode-select"
                value={config.find_by}
                onChange={(e) => setConfig((p) => ({ ...p, find_by: e.target.value as 'farthest' | 'nearest' | 'strongest' }))}
                title="Which qualifying peak becomes the OUTER wall"
              >
                <option value="farthest">Find by — farthest peak (outermost, default)</option>
                <option value="nearest">Find by — nearest peak (closest to label)</option>
                <option value="strongest">Find by — strongest peak (highest profile)</option>
              </select>
              {!regionMode && (
                <Slider label="Outer max" min={10} max={400} value={num('outer_search_max')} onChange={setNum('outer_search_max')} />
              )}
              <Slider label="Outer coverage" min={0} max={1} value={num('outer_min_hratio')} onChange={setNum('outer_min_hratio')} />
              <Slider label="Specular" min={0} max={255} value={num('specular_thr')} onChange={setNum('specular_thr')} />
            </div>

            <div className="es-section">
              <button
                type="button" className="es-advanced-toggle"
                onClick={() => setShowAdvanced((v) => !v)}
                aria-expanded={showAdvanced}
              >
                {showAdvanced ? '▾' : '▸'} Advanced
              </button>

              {showAdvanced && (
                <>
                  {/* Strip geometry — in edge_regions mode the drawn region IS the geometry. */}
                  {!regionMode && (
                    <>
                      <Slider label="Inner max" min={0} max={300} value={num('inner_search_max')} onChange={setNum('inner_search_max')} />
                      <Slider label="Edge margin" min={0} max={30} value={num('edge_margin')} onChange={setNum('edge_margin')} />
                      <Slider label="Y extension" min={0} max={1} value={num('y_extension')} onChange={setNum('y_extension')} />
                    </>
                  )}
                  <Slider label="Strong thr" min={0} max={1} value={num('strong_thr')} onChange={setNum('strong_thr')} />
                  <Slider label="Peak height" min={0} max={1} value={num('peak_height')} onChange={setNum('peak_height')} />
                  <Slider label="Peak prom" min={0} max={1} value={num('peak_prom')} onChange={setNum('peak_prom')} />
                  <Slider label="Peak dist" min={1} max={50} value={num('peak_dist')} onChange={setNum('peak_dist')} />
                  {!regionMode && (
                    <Slider label="Fill keep" min={0} max={1} value={num('fill_keep_ratio')} onChange={setNum('fill_keep_ratio')} />
                  )}
                  <Slider label="Edge width" min={1} max={15} value={num('edge_width')} onChange={setNum('edge_width')} />
                  {!regionMode && wallType === 'inner' && (
                    <>
                      <Slider label="Inner coverage" min={0} max={1} value={num('inner_min_hratio')} onChange={setNum('inner_min_hratio')} />
                      <Slider label="Inner tol px" min={0} max={60} value={num('inner_tol_px')} onChange={setNum('inner_tol_px')} />
                    </>
                  )}
                </>
              )}
            </div>

            <div className="es-section">
              <div className="es-section-title">Template walls</div>
              {regionMode ? (
                <div className="es-walls es-walls-empty">
                  Not used in edge-region mode — the drawn regions locate the walls directly.
                </div>
              ) : config.template_walls ? (
                <div className="es-walls">
                  inner L/R: <b>{fmt(config.template_walls.inner_L)}/{fmt(config.template_walls.inner_R)}</b><br />
                  outer L/R: <b>{fmt(config.template_walls.outer_L)}/{fmt(config.template_walls.outer_R)}</b><br />
                  plastic L/R: <b>{fmt(config.template_walls.plastic_L)}/{fmt(config.template_walls.plastic_R)}</b>
                </div>
              ) : (
                <div className="es-walls es-walls-empty">Not computed — click "Detect on template".</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Compute a search-band quad on one side of the label, mirroring the geometry
// in image_proc_detector (_compute_strip_profile): point = p_top + gap * perp,
// with the strip extended along Y by y_extension. Returns [innerTop, outerTop,
// outerBot, innerBot] in image pixel coords.
function bandQuad(
  label: Pt[], side: 'left' | 'right', innerGap: number, outerGap: number, yExt: number,
): Pt[] {
  const pTop = side === 'left' ? label[0]! : label[1]!;
  const pBot = side === 'left' ? label[3]! : label[2]!;
  const ex = pBot[0] - pTop[0];
  const ey = pBot[1] - pTop[1];
  const elen = Math.hypot(ex, ey) || 1;
  const edx = ex / elen, edy = ey / elen;
  const perpx = side === 'left' ? -edy : edy;
  const perpy = side === 'left' ? edx : -edx;
  const ye = yExt * elen;
  const topX = pTop[0] - ye * edx, topY = pTop[1] - ye * edy;
  const botX = pBot[0] + ye * edx, botY = pBot[1] + ye * edy;
  const at = (bx: number, by: number, gap: number): Pt => [bx + gap * perpx, by + gap * perpy];
  return [
    at(topX, topY, innerGap),
    at(topX, topY, outerGap),
    at(botX, botY, outerGap),
    at(botX, botY, innerGap),
  ];
}


function fmt(v: any): string {
  if (v === null || v === undefined) return '–';
  const n = Number(v);
  return Number.isFinite(n) ? (Math.round(n * 10) / 10).toString() : String(v);
}

interface SliderProps {
  label: string; min: number; max: number; value: number; onChange: (v: number) => void;
}
const Slider: React.FC<SliderProps> = ({ label, min, max, value, onChange }) => {
  const step = max - min > 10 ? 1 : 0.01;
  const decimals = step < 1 ? 2 : 0;
  return (
    <div className="es-slider-row">
      <span className="es-slider-label">{label}</span>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="es-slider"
      />
      <input
        type="number" min={min} max={max} step={step}
        value={step < 1 ? Number(value.toFixed(decimals)) : Math.round(value)}
        onChange={(e) => {
          const v = parseFloat(e.target.value);
          if (!isNaN(v)) onChange(Math.max(min, Math.min(max, v)));
        }}
        className="es-slider-num"
      />
    </div>
  );
};

export default EdgeSetupModal;
