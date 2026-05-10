import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, FabricImage, Rect } from 'fabric';
import { mlTrainingAPI, AnnotationRegion, CharSegment, MLProject, ProjectImage } from '@/services/mlTraining';
import ConfirmDialog from '@/components/shared/ConfirmDialog';
import { useToast } from '@/contexts/ToastContext';

interface ConfirmState {
  title: string;
  message: string;
  type?: 'warning' | 'danger' | 'info';
  confirmText?: string;
  onConfirm: () => void;
}

const uuidv4 = () => crypto.randomUUID();

interface Props {
  project: MLProject;
  onRefresh: () => void;
  // Deep-link from Train-tab "Open in Label" button — auto-loads the image
  // and selects the matching segment so the user can edit in context.
  deepLink?: { tab: 'label'; filename: string; segmentId: string } | null;
  onDeepLinkConsumed?: () => void;
}

type DrawMode = 'select' | 'draw-region' | 'draw-char';
type Label = 'OK' | 'NG' | null;

const REGION_COLOR  = '#f59e0b';
const SEG_UNLABELED = '#6b7280';
const SEG_OK_COLOR  = '#22c55e';
const SEG_NG_COLOR  = '#ef4444';

interface AnnotatedRect extends Rect {
  _regionId?: string;
  _segmentId?: string;
  _label?: Label;
  _isRegion?: boolean;
}

function segColor(label: Label) {
  return label === 'OK' ? SEG_OK_COLOR : label === 'NG' ? SEG_NG_COLOR : SEG_UNLABELED;
}

// Inverse-scale stroke + corner so zoomed-in bboxes stay visually thin
function applyZoomVisuals(canvas: Canvas, zoom: number) {
  const sw = Math.max(0.5, 2 / zoom);
  const cs = Math.max(4, 8 / zoom);
  canvas.getObjects().forEach(o => {
    const r = o as AnnotatedRect;
    if (!r._isRegion && !r._segmentId) return;
    r.set('strokeWidth', sw);
    r.set('cornerSize', cs);
  });
}

export default function LabelTab({ project, onRefresh, deepLink, onDeepLinkConsumed }: Props) {
  const toast = useToast();
  const [images, setImages]         = useState<ProjectImage[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);

  const canvasRef    = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const fabricRef    = useRef<Canvas | null>(null);
  const imageBoundsRef = useRef<{ left: number; top: number; width: number; height: number } | null>(null);

  const [regions, setRegions]               = useState<AnnotationRegion[]>([]);
  const [drawMode, setDrawMode]             = useState<DrawMode>('select');
  const [segmenting, setSegmenting]         = useState(false);
  const [saving, setSaving]                 = useState(false);
  const [autoSaveStatus, setAutoSaveStatus] = useState<'idle' | 'saving' | 'saved'>('idle');
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);
  const [selectedSegId, setSelectedSegId]   = useState<string | null>(null);
  const [expandedRegions, setExpandedRegions] = useState<Set<string>>(new Set());

  const [copyPrevRegions, setCopyPrevRegions] = useState(false);
  const [confirmDialog, setConfirmDialog] = useState<ConfirmState | null>(null);
  const [resizeDelta, setResizeDelta]     = useState<number>(2);

  const autoSaveTimer      = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isInitLoadRef      = useRef(true);
  const regionsRef         = useRef<AnnotationRegion[]>([]);   // always-current snapshot
  const prevRegionsRef     = useRef<AnnotationRegion[]>([]);   // regions of last image
  const copyPrevRegionsRef = useRef(false);

  // keep refs in sync
  useEffect(() => { regionsRef.current = regions; }, [regions]);
  useEffect(() => { copyPrevRegionsRef.current = copyPrevRegions; }, [copyPrevRegions]);

  // ── Refs so canvas handlers always see current values (no stale closures) ──
  const drawModeRef         = useRef<DrawMode>('select');
  const selectedRegionIdRef = useRef<string | null>(null);
  const isModifyingRef      = useRef(false);

  useEffect(() => { drawModeRef.current = drawMode; }, [drawMode]);
  useEffect(() => { selectedRegionIdRef.current = selectedRegionId; }, [selectedRegionId]);

  // In draw modes, skipTargetFind prevents Fabric from picking existing rects on click —
  // otherwise a click on a region/segment box starts a fabric-drag AND our draw-rect
  // simultaneously, so each drag commits a phantom char/region.
  useEffect(() => {
    const canvas = fabricRef.current;
    if (!canvas) return;
    canvas.skipTargetFind = drawMode !== 'select';
    if (drawMode !== 'select') {
      canvas.discardActiveObject();
      canvas.requestRenderAll();
    }
  }, [drawMode]);

  // Drawing state
  const drawStartRef  = useRef<{ x: number; y: number } | null>(null);
  const drawRectRef   = useRef<AnnotatedRect | null>(null);
  const isDrawingRef  = useRef(false);
  const mouseDnPosRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  // ── Load image list ───────────────────────────────────────────────────────
  const loadImages = useCallback(async () => {
    try { setImages(await mlTrainingAPI.listProjectImages(project.id)); } catch { /* ignore */ }
  }, [project.id]);

  useEffect(() => {
    loadImages();
    setSelectedFile(null);
    setRegions([]);
    setExpandedRegions(new Set());
  }, [project.id]);

  // Consume incoming deep-link from Train tab. Runs once per non-null link
  // value, then signals parent to clear so re-mounts don't re-trigger.
  useEffect(() => {
    if (!deepLink) return;
    selectImage(deepLink.filename).then(() => {
      // selectImage finishes after annotation load; segment is rendered then.
      setSelectedSegId(deepLink.segmentId);
    }).catch(() => {
      // Surface to user — silent failure here looked like the link did nothing.
      toast.error(`Image "${deepLink.filename}" not found — it may have been deleted`);
    });
    onDeepLinkConsumed?.();
    // selectImage depends on project state (closure); no need to thread it
    // into deps because deepLink is the only trigger we want.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deepLink]);

  // ── Select image → load into canvas ──────────────────────────────────────
  const selectImage = useCallback(async (filename: string) => {
    // Save current regions (strip segments) before switching
    prevRegionsRef.current = regionsRef.current.map(r => ({ ...r, segments: [] }));

    isInitLoadRef.current = true;
    setSelectedFile(filename);
    setRegions([]);
    setSelectedRegionId(null);
    setSelectedSegId(null);
    setDrawMode('select');
    setExpandedRegions(new Set());

    try {
      const [imgMeta, annData] = await Promise.all([
        mlTrainingAPI.getImageMeta(project.id, filename),
        mlTrainingAPI.getAnnotation(project.id, filename),
      ]);

      const canvas = fabricRef.current;
      if (!canvas || !containerRef.current) return;

      canvas.clear();
      canvas.setViewportTransform([1, 0, 0, 1, 0, 0]);

      const img = await FabricImage.fromURL(imgMeta.url, { crossOrigin: 'anonymous' });
      const cw = containerRef.current.clientWidth;
      const ch = containerRef.current.clientHeight;
      const scale = Math.min((cw - 40) / imgMeta.width, (ch - 40) / imgMeta.height);
      img.scale(scale);
      const left = (cw - imgMeta.width * scale) / 2;
      const top  = (ch - imgMeta.height * scale) / 2;
      img.set({ left, top, selectable: false, evented: false });

      canvas.backgroundImage = img;
      imageBoundsRef.current = { left, top, width: imgMeta.width * scale, height: imgMeta.height * scale };
      canvas.renderAll();

      if (annData.regions.length > 0) {
        // Image already has annotations — use them
        setRegions(annData.regions);
        setExpandedRegions(new Set(annData.regions.map(r => r.id)));
        _renderAllRegions(canvas, annData.regions, imageBoundsRef.current);
      } else if (copyPrevRegionsRef.current && prevRegionsRef.current.length > 0) {
        // No annotations yet + copy-prev enabled → paste region boxes (no segments)
        const copied = prevRegionsRef.current.map(r => ({ ...r, id: uuidv4(), segments: [] }));
        setRegions(copied);
        setExpandedRegions(new Set(copied.map(r => r.id)));
        _renderAllRegions(canvas, copied, imageBoundsRef.current);
      }

      // Allow auto-save after initial data is settled
      setTimeout(() => { isInitLoadRef.current = false; }, 100);
    } catch (e) {
      console.error('Failed to load image', e);
      isInitLoadRef.current = false;
    }
  }, [project.id]);

  // ── Init Fabric canvas ────────────────────────────────────────────────────
  useEffect(() => {
    if (!canvasRef.current || !containerRef.current) return;

    const canvas = new Canvas(canvasRef.current, {
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight,
      backgroundColor: '#1a1d27',
      selection: false,
    });
    fabricRef.current = canvas;

    // ── Mouse wheel zoom ────────────────────────────────────────────────────
    canvas.on('mouse:wheel', (opt) => {
      const delta = opt.e.deltaY;
      let zoom = canvas.getZoom();
      zoom *= 0.999 ** delta;
      zoom = Math.min(Math.max(zoom, 0.1), 10);
      canvas.zoomToPoint({ x: opt.e.offsetX, y: opt.e.offsetY }, zoom);
      applyZoomVisuals(canvas, zoom);
      opt.e.preventDefault();
      opt.e.stopPropagation();
    });

    // ── Panning (Space+drag or middle mouse) ────────────────────────────────
    let isPanning = false;
    let panLastX = 0, panLastY = 0;

    canvas.on('mouse:down', (opt) => {
      const e = opt.e as MouseEvent;
      const ptr = canvas.getScenePoint(e);
      mouseDnPosRef.current = { x: ptr.x, y: ptr.y };

      // Pan trigger
      if ((canvas as any)._spacePressed || e.button === 1) {
        isPanning = true;
        panLastX = e.clientX;
        panLastY = e.clientY;
        canvas.setCursor('grabbing');
        return;
      }

      // Draw region or char
      const mode = drawModeRef.current;
      if (mode !== 'draw-region' && mode !== 'draw-char') return;
      const bounds = imageBoundsRef.current;
      if (!bounds) return;

      const clampX = Math.max(bounds.left, Math.min(ptr.x, bounds.left + bounds.width));
      const clampY = Math.max(bounds.top,  Math.min(ptr.y, bounds.top  + bounds.height));
      drawStartRef.current = { x: clampX, y: clampY };
      isDrawingRef.current = true;

      const color = mode === 'draw-region' ? REGION_COLOR : SEG_UNLABELED;
      const curZoom = canvas.getZoom();
      const rect = new Rect({
        left: clampX, top: clampY, width: 0, height: 0,
        stroke: color, strokeWidth: Math.max(0.5, 2 / curZoom),
        fill: `${color}22`,
        selectable: false, evented: false,
      }) as AnnotatedRect;
      rect._isRegion = mode === 'draw-region';
      drawRectRef.current = rect;
      canvas.add(rect);
    });

    canvas.on('mouse:move', (opt) => {
      const e = opt.e as MouseEvent;
      if (isPanning) {
        const vpt = canvas.viewportTransform;
        vpt[4] += e.clientX - panLastX;
        vpt[5] += e.clientY - panLastY;
        panLastX = e.clientX;
        panLastY = e.clientY;
        canvas.requestRenderAll();
        return;
      }
      if (!isDrawingRef.current || !drawStartRef.current || !drawRectRef.current) return;
      const bounds = imageBoundsRef.current;
      if (!bounds) return;
      const ptr = canvas.getScenePoint(e);
      const ex = Math.max(bounds.left, Math.min(ptr.x, bounds.left + bounds.width));
      const ey = Math.max(bounds.top,  Math.min(ptr.y, bounds.top  + bounds.height));
      const { x, y } = drawStartRef.current;
      drawRectRef.current.set({
        left: Math.min(x, ex), top: Math.min(y, ey),
        width: Math.abs(ex - x), height: Math.abs(ey - y),
      });
      canvas.renderAll();
    });

    canvas.on('mouse:up', (opt) => {
      const e = opt.e as MouseEvent;

      if (isPanning) {
        isPanning = false;
        canvas.setViewportTransform(canvas.viewportTransform);
        canvas.setCursor('default');
        return;
      }

      // Click on canvas object → focus matching row in right panel
      if (!isDrawingRef.current && opt.target) {
        const obj = opt.target as AnnotatedRect;
        const ptr = canvas.getScenePoint(e);
        const dist = Math.hypot(ptr.x - mouseDnPosRef.current.x, ptr.y - mouseDnPosRef.current.y);
        if (dist < 5) {
          if (obj._segmentId && obj._regionId) {
            setSelectedSegId(obj._segmentId);
            setSelectedRegionId(obj._regionId);
            setTimeout(() => {
              document.getElementById(`seg-${obj._segmentId}`)
                ?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
            }, 0);
          } else if (obj._isRegion && obj._regionId) {
            setSelectedRegionId(obj._regionId);
            setSelectedSegId(null);
            setTimeout(() => {
              document.getElementById(`region-${obj._regionId}`)
                ?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
            }, 0);
          }
        }
      }

      if (!isDrawingRef.current || !drawStartRef.current || !drawRectRef.current) return;
      isDrawingRef.current = false;

      const rect   = drawRectRef.current;
      const bounds = imageBoundsRef.current;
      const mode   = drawModeRef.current;

      if (!bounds || (rect.width ?? 0) < 5 || (rect.height ?? 0) < 5) {
        canvas.remove(rect);
        drawStartRef.current = null;
        drawRectRef.current  = null;
        return;
      }

      const normX = ((rect.left ?? 0) - bounds.left) / bounds.width;
      const normY = ((rect.top  ?? 0) - bounds.top)  / bounds.height;
      const normW = (rect.width  ?? 0) / bounds.width;
      const normH = (rect.height ?? 0) / bounds.height;

      if (mode === 'draw-region') {
        const regionId = uuidv4();
        rect._regionId = regionId;
        rect._isRegion = true;
        rect.set({ selectable: false, evented: false, hasControls: false });
        canvas.renderAll();
        setRegions(prev => [...prev, { id: regionId, x: normX, y: normY, w: normW, h: normH, segments: [] }]);
        setExpandedRegions(prev => new Set([...prev, regionId]));
        setSelectedRegionId(regionId);
        setDrawMode('select');

      } else if (mode === 'draw-char') {
        canvas.remove(rect);
        const segId    = uuidv4();
        const regionId = selectedRegionIdRef.current;
        const seg: CharSegment = { id: segId, x: normX, y: normY, w: normW, h: normH, label: null };

        if (regionId) {
          // Add to existing selected region
          setRegions(prev => prev.map(r => r.id !== regionId ? r : { ...r, segments: [...r.segments, seg] }));
        } else {
          // No region selected → create standalone region for this char
          const newRegionId = uuidv4();
          const newRegion: AnnotationRegion = { id: newRegionId, x: normX, y: normY, w: normW, h: normH, segments: [seg] };
          setRegions(prev => [...prev, newRegion]);
          setSelectedRegionId(newRegionId);
        }
      }

      drawStartRef.current = null;
      drawRectRef.current  = null;
    });

    // ── Segment move / resize sync ──────────────────────────────────────────
    canvas.on('object:moving',  () => { isModifyingRef.current = true; });
    canvas.on('object:scaling', () => { isModifyingRef.current = true; });

    canvas.on('object:modified', (e) => {
      isModifyingRef.current = false;
      const obj    = e.target as AnnotatedRect;
      if (!obj._regionId) return;
      const bounds = imageBoundsRef.current;
      if (!bounds) return;

      // Bake scale into width/height
      const w = (obj.width  ?? 0) * (obj.scaleX ?? 1);
      const h = (obj.height ?? 0) * (obj.scaleY ?? 1);
      obj.set({ width: w, height: h, scaleX: 1, scaleY: 1 });
      obj.setCoords();

      const nx = ((obj.left ?? 0) - bounds.left) / bounds.width;
      const ny = ((obj.top  ?? 0) - bounds.top)  / bounds.height;
      const nw = w / bounds.width;
      const nh = h / bounds.height;

      if (obj._segmentId) {
        // Segment moved/resized
        setRegions(prev => prev.map(r => r.id !== obj._regionId ? r : {
          ...r,
          segments: r.segments.map(s => s.id === obj._segmentId ? { ...s, x: nx, y: ny, w: nw, h: nh } : s),
        }));
      } else if (obj._isRegion) {
        // Region box itself moved/resized
        setRegions(prev => prev.map(r => r.id !== obj._regionId ? r : {
          ...r, x: nx, y: ny, w: nw, h: nh,
        }));
      }
    });

    // ── Space key pan + ESC to exit draw mode ──────────────────────────────
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      if (e.code === 'Escape') {
        setDrawMode('select');
        return;
      }
      if (e.code === 'Space' && !(canvas as any)._spacePressed) {
        e.preventDefault();
        (canvas as any)._spacePressed = true;
        canvas.setCursor('grab');
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        (canvas as any)._spacePressed = false;
        canvas.setCursor('default');
      }
    };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup',   onKeyUp);

    return () => {
      canvas.dispose();
      fabricRef.current = null;
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup',   onKeyUp);
    };
  }, []);

  // ── Re-render regions when state changes (skip during drag) ───────────────
  useEffect(() => {
    const canvas = fabricRef.current;
    const bounds = imageBoundsRef.current;
    if (!canvas || !bounds || !selectedFile || isModifyingRef.current) return;
    _renderAllRegions(canvas, regions, bounds);

    // Restore canvas active object from React selection state.
    // Only meaningful in select mode (draw modes hide handles via discardActiveObject).
    if (drawModeRef.current === 'select') {
      let target: AnnotatedRect | undefined;
      if (selectedSegId) {
        target = canvas.getObjects().find(o => (o as AnnotatedRect)._segmentId === selectedSegId) as AnnotatedRect | undefined;
      } else if (selectedRegionId) {
        target = canvas.getObjects().find(o => {
          const r = o as AnnotatedRect;
          return r._isRegion && r._regionId === selectedRegionId;
        }) as AnnotatedRect | undefined;
      }
      if (target) {
        canvas.setActiveObject(target);
        canvas.requestRenderAll();
      } else if (canvas.getActiveObject()) {
        canvas.discardActiveObject();
        canvas.requestRenderAll();
      }
    }
  }, [regions, selectedFile, selectedRegionId, selectedSegId]);

  // ── Focus dimming: fade non-focused regions/chars ─────────────────────────
  useEffect(() => {
    const canvas = fabricRef.current;
    if (!canvas) return;
    // Skip dim during draw modes — user needs full visibility to draw
    const inDrawMode = drawMode !== 'select';
    const hasFocus   = !inDrawMode && (selectedSegId !== null || selectedRegionId !== null);

    canvas.getObjects().forEach(o => {
      const r = o as AnnotatedRect;
      if (!r._isRegion && !r._segmentId) return;

      let focused = true;
      if (hasFocus) {
        if (selectedSegId) {
          // 1 char focus → that char + its parent region stay bright
          focused = r._segmentId === selectedSegId
                 || (r._isRegion === true && r._regionId === selectedRegionId);
        } else if (selectedRegionId) {
          // region focus → that region + all chars inside stay bright
          focused = r._regionId === selectedRegionId;
        }
      }
      r.set('opacity', focused ? 1 : 0.25);
    });
    canvas.requestRenderAll();
  }, [selectedRegionId, selectedSegId, regions, drawMode]);

  // ── Render helper ─────────────────────────────────────────────────────────
  function _renderAllRegions(
    canvas: Canvas,
    regionList: AnnotationRegion[],
    bounds: { left: number; top: number; width: number; height: number },
  ) {
    const toRemove = canvas.getObjects().filter(o => (o as AnnotatedRect)._isRegion || (o as AnnotatedRect)._segmentId);
    toRemove.forEach(o => canvas.remove(o));

    regionList.forEach(region => {
      // Region outline — selectable so user can move/resize before segmenting
      const regionRect = new Rect({
        left:   bounds.left + region.x * bounds.width,
        top:    bounds.top  + region.y * bounds.height,
        width:  region.w * bounds.width,
        height: region.h * bounds.height,
        stroke: REGION_COLOR, strokeWidth: 2,
        fill: `${REGION_COLOR}12`,
        selectable: true, evented: true,
        hasControls: true,
        cornerSize: 8,
        cornerColor: REGION_COLOR,
        borderColor: REGION_COLOR,
        transparentCorners: false,
        lockRotation: true,
      }) as AnnotatedRect;
      regionRect._isRegion = true;
      regionRect._regionId = region.id;
      canvas.add(regionRect);

      // Segment rects — selectable, movable, resizable
      region.segments.forEach(seg => {
        const color = segColor(seg.label ?? null);
        const segRect = new Rect({
          left:   bounds.left + seg.x * bounds.width,
          top:    bounds.top  + seg.y * bounds.height,
          width:  seg.w * bounds.width,
          height: seg.h * bounds.height,
          stroke: color, strokeWidth: 2,
          fill: `${color}33`,
          selectable: true, evented: true,
          hasControls: true,
          cornerSize: 8,
          cornerColor: color,
          borderColor: color,
          transparentCorners: false,
          lockRotation: true,
        }) as AnnotatedRect;
        segRect._segmentId = seg.id;
        segRect._regionId  = region.id;
        segRect._label     = seg.label ?? null;
        canvas.add(segRect);
      });
    });

    applyZoomVisuals(canvas, canvas.getZoom());
    canvas.renderAll();
  }

  // ── Auto segment ──────────────────────────────────────────────────────────
  const handleSegment = useCallback(async (regionId: string) => {
    if (!selectedFile) return;
    const region = regions.find(r => r.id === regionId);
    if (!region) return;

    const runSegment = async () => {
      setSegmenting(true);
      try {
        const result = await mlTrainingAPI.segmentRegion(project.id, selectedFile, {
          x: region.x, y: region.y, w: region.w, h: region.h,
        });
        const segments: CharSegment[] = result.segments.map(s => ({ ...s, label: null }));
        setRegions(prev => prev.map(r => r.id === regionId ? { ...r, segments } : r));
      } catch (e) {
        console.error('Segmentation failed', e);
      } finally {
        setSegmenting(false);
      }
    };

    if (region.segments.length > 0) {
      setConfirmDialog({
        title: 'Replace existing characters?',
        message: `This region already has ${region.segments.length} character${region.segments.length > 1 ? 's' : ''}. Auto-segmenting will discard them and re-detect from scratch. Continue?`,
        type: 'warning',
        confirmText: 'Replace',
        onConfirm: () => { void runSegment(); },
      });
      return;
    }
    await runSegment();
  }, [project.id, selectedFile, regions]);

  // ── char_id edit from panel input ────────────────────────────────────────
  const handleCharIdChange = useCallback((regionId: string, segId: string, raw: string) => {
    // Model outputs 1 char — enforce maxLength=1 after trim.
    // If user paste multi-char string, keep first non-whitespace char.
    const trimmed = (raw || '').replace(/\s+/g, '');
    const char_id = trimmed ? trimmed.charAt(0) : '';
    setRegions(prev => prev.map(r => r.id !== regionId ? r : {
      ...r,
      segments: r.segments.map(s => s.id === segId ? { ...s, char_id: char_id || null } : s),
    }));
  }, []);

  // ── Label change from panel select ───────────────────────────────────────
  const handleLabelChange = useCallback((regionId: string, segId: string, label: Label) => {
    // Update state
    setRegions(prev => prev.map(r => r.id !== regionId ? r : {
      ...r,
      segments: r.segments.map(s => s.id === segId ? { ...s, label } : s),
    }));
    // Sync canvas rect immediately (no full re-render needed)
    const canvas = fabricRef.current;
    if (!canvas) return;
    const obj = canvas.getObjects().find(o => (o as AnnotatedRect)._segmentId === segId) as AnnotatedRect | undefined;
    if (!obj) return;
    const color = segColor(label);
    obj._label = label;
    obj.set({ stroke: color, fill: `${color}33`, cornerColor: color, borderColor: color });
    canvas.renderAll();
  }, []);

  // ── Zoom controls ─────────────────────────────────────────────────────────
  const handleZoom = useCallback((factor: number) => {
    const canvas = fabricRef.current;
    if (!canvas) return;
    const zoom = Math.min(Math.max(canvas.getZoom() * factor, 0.1), 10);
    canvas.zoomToPoint({ x: (canvas.width ?? 800) / 2, y: (canvas.height ?? 600) / 2 }, zoom);
    applyZoomVisuals(canvas, zoom);
    canvas.requestRenderAll();
  }, []);

  const handleResetZoom = useCallback(() => {
    const canvas = fabricRef.current;
    if (!canvas) return;
    canvas.setViewportTransform([1, 0, 0, 1, 0, 0]);
    applyZoomVisuals(canvas, 1);
    canvas.renderAll();
  }, []);

  // ── Save / delete ─────────────────────────────────────────────────────────
  const handleSave = useCallback(async () => {
    if (!selectedFile) return;
    setSaving(true);
    try {
      await mlTrainingAPI.saveAnnotation(project.id, selectedFile, regions);
      await loadImages();
      onRefresh();
    } catch (e) {
      console.error('Save failed', e);
    } finally {
      setSaving(false);
    }
  }, [project.id, selectedFile, regions, loadImages, onRefresh]);

  // ── Auto-save (debounced 1.5 s after any regions change) ─────────────────
  useEffect(() => {
    if (!selectedFile) return;
    if (isInitLoadRef.current) return; // skip the first load from server

    if (autoSaveTimer.current) clearTimeout(autoSaveTimer.current);
    setAutoSaveStatus('idle');

    autoSaveTimer.current = setTimeout(async () => {
      setAutoSaveStatus('saving');
      try {
        await mlTrainingAPI.saveAnnotation(project.id, selectedFile, regions);
        await loadImages();
        onRefresh();
        setAutoSaveStatus('saved');
        setTimeout(() => setAutoSaveStatus('idle'), 1500);
      } catch {
        setAutoSaveStatus('idle');
      }
    }, 1500);

    return () => { if (autoSaveTimer.current) clearTimeout(autoSaveTimer.current); };
  }, [regions]);

  // ── Bulk label all segments in a region ───────────────────────────────────
  const handleBulkLabel = useCallback((regionId: string, label: Label) => {
    setRegions(prev => prev.map(r => r.id !== regionId ? r : {
      ...r, segments: r.segments.map(s => ({ ...s, label })),
    }));
    const canvas = fabricRef.current;
    if (!canvas) return;
    canvas.getObjects().forEach(o => {
      const obj = o as AnnotatedRect;
      if (obj._regionId !== regionId || !obj._segmentId) return;
      const color = segColor(label);
      obj._label = label;
      obj.set({ stroke: color, fill: `${color}33`, cornerColor: color, borderColor: color });
    });
    canvas.renderAll();
  }, []);

  const deleteRegion = useCallback((regionId: string, segCount: number) => {
    const performDelete = () => {
      setRegions(prev => prev.filter(r => r.id !== regionId));
      setExpandedRegions(prev => { const n = new Set(prev); n.delete(regionId); return n; });
      if (selectedRegionId === regionId) setSelectedRegionId(null);
    };
    if (segCount === 0) {
      performDelete();
      return;
    }
    setConfirmDialog({
      title: 'Delete region',
      message: `This region contains ${segCount} character${segCount > 1 ? 's' : ''}. Deleting it will remove the region and all chars inside. Continue?`,
      type: 'danger',
      confirmText: 'Delete',
      onConfirm: performDelete,
    });
  }, [selectedRegionId]);

  const deleteSegment = useCallback((regionId: string, segId: string) => {
    setRegions(prev => prev.map(r => r.id !== regionId ? r : {
      ...r, segments: r.segments.filter(s => s.id !== segId),
    }));
  }, []);

  // ── Selection bridge: panel click → state + canvas activeObject ───────────
  // Always force select mode so resize handles are usable.
  const selectFromPanel = useCallback((regionId: string | null, segId: string | null) => {
    setSelectedRegionId(regionId);
    setSelectedSegId(segId);
    if (drawModeRef.current !== 'select') setDrawMode('select');

    const canvas = fabricRef.current;
    if (!canvas) return;
    let target: AnnotatedRect | undefined;
    if (segId) {
      target = canvas.getObjects().find(o => (o as AnnotatedRect)._segmentId === segId) as AnnotatedRect | undefined;
    } else if (regionId) {
      target = canvas.getObjects().find(o => {
        const r = o as AnnotatedRect;
        return r._isRegion && r._regionId === regionId;
      }) as AnnotatedRect | undefined;
    }
    if (target) {
      canvas.setActiveObject(target);
      canvas.requestRenderAll();
    } else if (canvas.getActiveObject()) {
      canvas.discardActiveObject();
      canvas.requestRenderAll();
    }
  }, []);

  // ── Resize selected char(s): symmetric, center-fixed, clamped to image ────
  const applyResize = useCallback((axis: 'w' | 'h', sign: 1 | -1) => {
    const bounds = imageBoundsRef.current;
    if (!bounds || !selectedRegionId) return;
    const dPx = Math.max(1, resizeDelta) * sign;
    const dn  = axis === 'w' ? dPx / bounds.width : dPx / bounds.height;

    setRegions(prev => prev.map(r => {
      if (r.id !== selectedRegionId) return r;
      return {
        ...r,
        segments: r.segments.map(s => {
          if (selectedSegId && s.id !== selectedSegId) return s;
          if (axis === 'w') {
            const newW = Math.min(1, Math.max(0.001, s.w + dn));
            const newX = Math.min(1 - newW, Math.max(0, s.x - (newW - s.w) / 2));
            return { ...s, x: newX, w: newW };
          } else {
            const newH = Math.min(1, Math.max(0.001, s.h + dn));
            const newY = Math.min(1 - newH, Math.max(0, s.y - (newH - s.h) / 2));
            return { ...s, y: newY, h: newH };
          }
        }),
      };
    }));
  }, [selectedRegionId, selectedSegId, resizeDelta]);

  // ── Stats ─────────────────────────────────────────────────────────────────
  const allSegs       = regions.flatMap(r => r.segments);
  const okCount       = allSegs.filter(s => s.label === 'OK').length;
  const ngCount       = allSegs.filter(s => s.label === 'NG').length;
  const unlabeledCount = allSegs.filter(s => !s.label).length;
  // Segments that have an OK/NG label but no char_id — required before training.
  const missingCharIdCount = useMemo(
    () => allSegs.filter(s => (s.label === 'OK' || s.label === 'NG') && !s.char_id).length,
    [allSegs],
  );

  // ── Cursor for canvas wrapper ─────────────────────────────────────────────
  const wrapperCursor = drawMode !== 'select' ? 'crosshair' : 'default';

  return (
    <div className="ml-label-tab">

      {/* ── Left: image list ── */}
      <div className="ml-label-list">
        <div className="ml-panel-header" style={{ padding: '10px 12px', flexDirection: 'column', alignItems: 'flex-start', gap: '6px' }}>
          <span>Images</span>
          <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', fontWeight: 400, color: '#6b7280', cursor: 'pointer', userSelect: 'none' }}>
            <input
              type="checkbox"
              checked={copyPrevRegions}
              onChange={e => setCopyPrevRegions(e.target.checked)}
              style={{ cursor: 'pointer', accentColor: REGION_COLOR }}
            />
            Copy prev regions
          </label>
        </div>
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {images.length === 0 && (
            <div className="ml-empty-state" style={{ padding: '20px' }}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" style={{opacity:.4}}>
                <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="2"/>
                <circle cx="8.5" cy="8.5" r="1.5" fill="currentColor"/>
                <path d="M21 15l-5-5L5 21" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
              </svg>
              No images
            </div>
          )}
          {images.map(img => (
            <div
              key={img.filename}
              className={`ml-label-image-item ${selectedFile === img.filename ? 'active' : ''} ${img.has_annotation ? 'has-label' : ''}`}
              onClick={() => selectImage(img.filename)}
              style={{ position: 'relative' }}
            >
              <img className="ml-label-image-thumb" src={img.url} alt={img.filename} loading="lazy" crossOrigin="anonymous" />
              <span className="ml-label-image-name">{img.filename}</span>
              {/* Annotation done badge */}
              {img.has_annotation && (
                <span title="Annotated" style={{
                  position: 'absolute', top: 4, right: 4,
                  width: 16, height: 16, borderRadius: '50%',
                  background: SEG_OK_COLOR,
                  border: '2px solid #fff',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '9px', color: '#fff', fontWeight: 700, lineHeight: 1,
                  flexShrink: 0,
                }}>✓</span>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* ── Center: canvas + toolbar ── */}
      <div className="ml-canvas-area">

        {/* Toolbar */}
        <div className="ml-canvas-toolbar">

          {/* Draw modes */}
          <button
            className={`ml-btn ml-btn-sm ${drawMode === 'select' ? 'ml-btn-primary' : 'ml-btn-secondary'}`}
            onClick={() => setDrawMode('select')}
            title="Select / pan (Space+drag to pan)"
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
              <path d="M4 4l7 18 3-7 7-3L4 4z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
            </svg>
            Select
          </button>

          <button
            className={`ml-btn ml-btn-sm ${drawMode === 'draw-region' ? 'ml-btn-primary' : 'ml-btn-secondary'}`}
            onClick={() => setDrawMode('draw-region')}
            disabled={!selectedFile}
            title="Draw a region containing characters"
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
              <rect x="3" y="6" width="18" height="12" rx="1" stroke="currentColor" strokeWidth="2"/>
            </svg>
            Draw Region
          </button>

          <button
            className={`ml-btn ml-btn-sm ${drawMode === 'draw-char' ? 'ml-btn-primary' : 'ml-btn-secondary'}`}
            onClick={() => setDrawMode('draw-char')}
            disabled={!selectedFile}
            title="Manually draw a single character box"
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
              <rect x="7" y="7" width="10" height="10" rx="1" stroke="currentColor" strokeWidth="2"/>
              <path d="M12 4v2M12 18v2M4 12h2M18 12h2" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
            Draw Char
          </button>

          {/* Auto segment for selected region */}
          {selectedRegionId && (
            <>
              <div className="ml-canvas-toolbar-sep" />
              <button
                className="ml-btn ml-btn-success ml-btn-sm"
                onClick={() => handleSegment(selectedRegionId)}
                disabled={segmenting || !selectedFile}
              >
                {segmenting
                  ? <><span className="ml-loading-spinner" style={{width:12,height:12,borderWidth:2}}/> Segmenting...</>
                  : <><svg width="13" height="13" viewBox="0 0 24 24" fill="none"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/></svg> Auto Segment</>
                }
              </button>
            </>
          )}

          {/* Zoom */}
          <div className="ml-canvas-toolbar-sep" />
          <button className="ml-btn ml-btn-secondary ml-btn-sm" onClick={() => handleZoom(1.25)} title="Zoom in">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
              <circle cx="11" cy="11" r="8" stroke="currentColor" strokeWidth="2"/>
              <path d="M11 8v6M8 11h6M21 21l-4.35-4.35" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>
          <button className="ml-btn ml-btn-secondary ml-btn-sm" onClick={() => handleZoom(0.8)} title="Zoom out">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
              <circle cx="11" cy="11" r="8" stroke="currentColor" strokeWidth="2"/>
              <path d="M8 11h6M21 21l-4.35-4.35" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>
          <button className="ml-btn ml-btn-secondary ml-btn-sm" onClick={handleResetZoom} title="Reset zoom">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
              <path d="M1 4v6h6M23 20v-6h-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>

          {/* Stats + Auto-save indicator + Save */}
          {selectedFile && (
            <div style={{ marginLeft: 'auto', display: 'flex', gap: '10px', alignItems: 'center' }}>
              <span style={{ fontSize: '11px', display: 'flex', gap: '8px' }}>
                <span style={{ color: SEG_OK_COLOR }}>{okCount} OK</span>
                <span style={{ color: SEG_NG_COLOR }}>{ngCount} NG</span>
                {unlabeledCount > 0 && <span style={{ color: '#6b7280' }}>{unlabeledCount} ?</span>}
                {missingCharIdCount > 0 && (
                  <span
                    style={{ color: '#ef4444', fontWeight: 600 }}
                    title="Labeled segments missing char_id — required for per-char golden template"
                  >
                    ⚠ {missingCharIdCount} no char
                  </span>
                )}
              </span>
              {autoSaveStatus === 'saving' && (
                <span style={{ fontSize: '11px', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <span className="ml-loading-spinner" style={{ width: 10, height: 10, borderWidth: 2 }} />
                  Saving…
                </span>
              )}
              {autoSaveStatus === 'saved' && (
                <span style={{ fontSize: '11px', color: SEG_OK_COLOR }}>✓ Saved</span>
              )}
              <button className="ml-btn ml-btn-primary ml-btn-sm" onClick={handleSave} disabled={saving}>
                {saving ? 'Saving...' : <>
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
                    <path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
                    <polyline points="17 21 17 13 7 13 7 21" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
                    <polyline points="7 3 7 8 15 8" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
                  </svg>
                  Save
                </>}
              </button>
            </div>
          )}
        </div>

        {/* Canvas */}
        <div className="ml-canvas-wrapper" ref={containerRef} style={{ cursor: wrapperCursor }}>
          <canvas ref={canvasRef} />
          {!selectedFile && (
            <div className="ml-empty-state" style={{ position: 'absolute', inset: 0 }}>
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" style={{opacity:.4}}>
                <rect x="9" y="4" width="12" height="16" rx="2" stroke="currentColor" strokeWidth="2"/>
                <path d="M9 12H3m0 0l3-3m-3 3l3 3" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              </svg>
              Select an image to start labeling
            </div>
          )}
        </div>

        {/* Status bar */}
        {selectedFile && (
          <div style={{ padding: '5px 12px', background: '#1a1d27', borderTop: '1px solid #2d3148', fontSize: '11px', color: '#6b7280', display: 'flex', gap: '16px' }}>
            {drawMode === 'draw-region' && <span style={{ color: REGION_COLOR }}>Draw a rectangle around a character region → Auto Segment · <b>Esc</b> to exit</span>}
            {drawMode === 'draw-char'   && <span style={{ color: SEG_UNLABELED }}>Draw a single character box manually{selectedRegionId ? '' : ' (will create new region)'} · <b>Esc</b> to exit</span>}
            {drawMode === 'select'      && <span>Scroll to zoom · Space+drag to pan · Click char to focus in panel · Drag to move/resize</span>}
          </div>
        )}
      </div>

      {/* ── Right: annotation panel ── */}
      <div className="ml-annotation-panel">
        <div className="ml-panel-header" style={{ padding: '10px 12px' }}>
          Regions {regions.length > 0 && <span className="ml-tab-count">{regions.length}</span>}
        </div>

        {/* ── Resize selection panel ── */}
        {(() => {
          const selRegion = regions.find(r => r.id === selectedRegionId);
          const targetCount = selectedSegId ? 1 : (selRegion?.segments.length ?? 0);
          const enabled = !!selectedRegionId && targetCount > 0;
          const labelText = !selectedRegionId
            ? 'Resize: (no selection)'
            : selectedSegId
              ? 'Resize: 1 char'
              : `Resize: ${targetCount} char${targetCount > 1 ? 's' : ''} in region`;
          return (
            <div style={{
              padding: '8px 12px',
              borderBottom: '1px solid #e2e8f0',
              background: '#f8fafc',
              opacity: enabled ? 1 : 0.5,
            }}>
              <div style={{ fontSize: '10px', fontWeight: 600, color: '#64748b', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '.5px' }}>
                {labelText}
              </div>
              <div style={{ display: 'flex', gap: '6px', alignItems: 'center', flexWrap: 'wrap' }}>
                <label style={{ fontSize: '11px', color: '#64748b' }}>Δ</label>
                <input
                  type="number"
                  min={1}
                  max={50}
                  value={resizeDelta}
                  onChange={e => setResizeDelta(Math.max(1, Number(e.target.value) || 1))}
                  disabled={!enabled}
                  style={{ width: 44, fontSize: '11px', padding: '2px 4px', borderRadius: 4, border: '1px solid #cbd5e1', textAlign: 'center' }}
                />
                <span style={{ fontSize: '10px', color: '#94a3b8' }}>px</span>
                <button className="ml-btn ml-btn-secondary ml-btn-sm" style={{ padding: '2px 6px', fontSize: '11px' }}
                        disabled={!enabled} onClick={() => applyResize('w', -1)} title="Shrink width (symmetric)">W−</button>
                <button className="ml-btn ml-btn-secondary ml-btn-sm" style={{ padding: '2px 6px', fontSize: '11px' }}
                        disabled={!enabled} onClick={() => applyResize('w', 1)} title="Expand width (symmetric)">W+</button>
                <button className="ml-btn ml-btn-secondary ml-btn-sm" style={{ padding: '2px 6px', fontSize: '11px' }}
                        disabled={!enabled} onClick={() => applyResize('h', -1)} title="Shrink height (symmetric)">H−</button>
                <button className="ml-btn ml-btn-secondary ml-btn-sm" style={{ padding: '2px 6px', fontSize: '11px' }}
                        disabled={!enabled} onClick={() => applyResize('h', 1)} title="Expand height (symmetric)">H+</button>
              </div>
            </div>
          );
        })()}

        <div style={{ flex: 1, overflowY: 'auto' }}>
          {regions.length === 0 && (
            <div className="ml-empty-state" style={{ padding: '16px', fontSize: '11px' }}>
              Draw a region on the image
            </div>
          )}

          {regions.map((region, ri) => {
            const isExpanded = expandedRegions.has(region.id);
            const toggleExpand = (e: React.MouseEvent) => {
              e.stopPropagation();
              setExpandedRegions(prev => {
                const n = new Set(prev);
                n.has(region.id) ? n.delete(region.id) : n.add(region.id);
                return n;
              });
            };
            return (
            <div key={region.id} id={`region-${region.id}`} style={{ borderBottom: '1px solid #e2e8f0' }}>

              {/* Region header */}
              <div
                className={`ml-segment-item ${selectedRegionId === region.id && !selectedSegId ? 'selected' : ''}`}
                onClick={() => {
                  const isCurrent = selectedRegionId === region.id && !selectedSegId;
                  selectFromPanel(isCurrent ? null : region.id, null);
                }}
                style={{
                  background: selectedRegionId === region.id && !selectedSegId ? '#fef3c7' : '#f8fafc',
                  borderLeft: selectedRegionId === region.id && !selectedSegId ? `3px solid ${REGION_COLOR}` : '3px solid transparent',
                }}
              >
                {/* Expand/collapse chevron */}
                <button
                  onClick={toggleExpand}
                  style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '0 2px', color: '#94a3b8', flexShrink: 0, lineHeight: 1 }}
                  title={isExpanded ? 'Collapse' : 'Expand'}
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" style={{ transition: 'transform .15s', transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)' }}>
                    <path d="M9 6l6 6-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </button>

                <span style={{ fontSize: '12px', fontWeight: 600, color: REGION_COLOR }}>
                  Region {ri + 1}
                  <span style={{ fontWeight: 400, color: '#94a3b8', marginLeft: 4 }}>
                    ({region.segments.length})
                  </span>
                </span>
                <div style={{ display: 'flex', gap: '4px', alignItems: 'center', marginLeft: 'auto' }}>
                  {/* Bulk label select — shown when region has segments */}
                  {region.segments.length > 0 && (
                    <select
                      value=""
                      onClick={e => e.stopPropagation()}
                      onChange={e => {
                        const v = e.target.value;
                        if (v === 'OK' || v === 'NG') handleBulkLabel(region.id, v);
                        else if (v === 'reset') handleBulkLabel(region.id, null);
                        e.target.value = '';
                      }}
                      style={{ fontSize: '10px', padding: '2px 4px', borderRadius: '4px', border: '1px solid #94a3b8', background: '#f1f5f9', color: '#374151', cursor: 'pointer', outline: 'none' }}
                      title="Set all chars label"
                    >
                      <option value="" disabled>All…</option>
                      <option value="OK">All OK</option>
                      <option value="NG">All NG</option>
                      <option value="reset">Reset</option>
                    </select>
                  )}
                  {region.segments.length === 0 && (
                    <button
                      className="ml-btn ml-btn-success ml-btn-sm"
                      style={{ padding: '2px 6px', fontSize: '10px' }}
                      onClick={e => { e.stopPropagation(); selectFromPanel(region.id, null); handleSegment(region.id); }}
                      disabled={segmenting}
                    >
                      <svg width="10" height="10" viewBox="0 0 24 24" fill="none"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/></svg>
                      Segment
                    </button>
                  )}
                  <button
                    className="ml-btn ml-btn-secondary ml-btn-sm"
                    style={{ padding: '2px 6px', fontSize: '10px' }}
                    title="Add char manually to this region"
                    onClick={e => { e.stopPropagation(); setSelectedRegionId(region.id); setSelectedSegId(null); setDrawMode('draw-char'); }}
                  >
                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none"><path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>
                  </button>
                  <button
                    style={{ background: 'none', border: 'none', color: '#ef4444', cursor: 'pointer', padding: '0 2px', fontSize: '13px' }}
                    onClick={e => { e.stopPropagation(); deleteRegion(region.id, region.segments.length); }}
                    title="Delete region (and all chars inside)"
                  >✕</button>
                </div>
              </div>

              {/* Segments — collapsible */}
              {isExpanded && region.segments.map((seg, si) => (
                <div
                  key={seg.id}
                  id={`seg-${seg.id}`}
                  className={`ml-segment-item${selectedSegId === seg.id ? ' selected' : ''}`}
                  style={{
                    paddingLeft: '18px',
                    fontSize: '11px',
                    gap: '6px',
                    background: selectedSegId === seg.id ? '#eff6ff' : undefined,
                    borderLeft: selectedSegId === seg.id ? `3px solid ${segColor(seg.label ?? null)}` : '3px solid transparent',
                    boxShadow: selectedSegId === seg.id ? 'inset 0 0 0 1px #bfdbfe' : undefined,
                  }}
                  onClick={() => selectFromPanel(region.id, seg.id)}
                >
                  {/* Color dot */}
                  <span style={{ width: 8, height: 8, borderRadius: '50%', background: segColor(seg.label ?? null), flexShrink: 0, display: 'inline-block' }} />
                  <span style={{ color: '#6b7280', flexShrink: 0 }}>Char {si + 1}</span>

                  {/* char_id input — required when label set; red border if empty */}
                  {(() => {
                    const needsChar = (seg.label === 'OK' || seg.label === 'NG') && !seg.char_id;
                    return (
                      <input
                        type="text"
                        className={`ml-char-input${needsChar ? ' empty' : ''}`}
                        value={seg.char_id ?? ''}
                        onChange={e => handleCharIdChange(region.id, seg.id, e.target.value)}
                        onClick={e => e.stopPropagation()}
                        maxLength={1}
                        placeholder="?"
                        title={needsChar ? 'char_id required' : 'Character identity'}
                      />
                    );
                  })()}

                  {/* Label select — syncs with canvas */}
                  <select
                    value={seg.label ?? ''}
                    onChange={e => handleLabelChange(region.id, seg.id, (e.target.value || null) as Label)}
                    onClick={e => e.stopPropagation()}
                    style={{
                      flex: 1,
                      fontSize: '11px',
                      padding: '2px 4px',
                      borderRadius: '4px',
                      border: `1px solid ${segColor(seg.label ?? null)}`,
                      background: 'transparent',
                      color: segColor(seg.label ?? null),
                      cursor: 'pointer',
                      outline: 'none',
                    }}
                  >
                    <option value="">Unlabeled</option>
                    <option value="OK">OK</option>
                    <option value="NG">NG</option>
                  </select>

                  <button
                    style={{ background: 'none', border: 'none', color: '#ef4444', cursor: 'pointer', padding: '0 2px', fontSize: '12px', flexShrink: 0 }}
                    onClick={() => deleteSegment(region.id, seg.id)}
                    title="Delete char"
                  >✕</button>
                </div>
              ))}
            </div>
          );})}

        </div>
      </div>

      {/* ── Confirm dialog ── */}
      <ConfirmDialog
        isOpen={confirmDialog !== null}
        onClose={() => setConfirmDialog(null)}
        onConfirm={() => confirmDialog?.onConfirm()}
        title={confirmDialog?.title}
        message={confirmDialog?.message ?? ''}
        type={confirmDialog?.type}
        confirmText={confirmDialog?.confirmText}
      />
    </div>
  );
}
