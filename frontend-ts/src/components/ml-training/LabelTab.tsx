import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Canvas, FabricImage, Rect } from 'fabric';
import { mlTrainingAPI, AnnotationRegion, CharSegment, MLProject, ProjectImage } from '@/services/mlTraining';

const uuidv4 = () => crypto.randomUUID();

interface Props {
  project: MLProject;
  onRefresh: () => void;
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

export default function LabelTab({ project, onRefresh }: Props) {
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
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);
  const [selectedSegId, setSelectedSegId]   = useState<string | null>(null);

  // Ref to scroll panel to focused segment
  const panelRef = useRef<HTMLDivElement>(null);

  // ── Refs so canvas handlers always see current values (no stale closures) ──
  const drawModeRef         = useRef<DrawMode>('select');
  const selectedRegionIdRef = useRef<string | null>(null);
  const isModifyingRef      = useRef(false);

  useEffect(() => { drawModeRef.current = drawMode; }, [drawMode]);
  useEffect(() => { selectedRegionIdRef.current = selectedRegionId; }, [selectedRegionId]);

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
  }, [project.id]);

  // ── Select image → load into canvas ──────────────────────────────────────
  const selectImage = useCallback(async (filename: string) => {
    setSelectedFile(filename);
    setRegions([]);
    setSelectedRegionId(null);
    setSelectedSegId(null);
    setDrawMode('select');

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
        setRegions(annData.regions);
        _renderAllRegions(canvas, annData.regions, imageBoundsRef.current);
      }
    } catch (e) {
      console.error('Failed to load image', e);
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
      const rect = new Rect({
        left: clampX, top: clampY, width: 0, height: 0,
        stroke: color, strokeWidth: 2,
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

      // Click on segment → focus it in panel (no label cycling)
      if (!isDrawingRef.current && opt.target) {
        const obj = opt.target as AnnotatedRect;
        if (obj._segmentId && obj._regionId) {
          const ptr = canvas.getScenePoint(e);
          const dist = Math.hypot(ptr.x - mouseDnPosRef.current.x, ptr.y - mouseDnPosRef.current.y);
          if (dist < 5) {
            setSelectedSegId(obj._segmentId!);
            setSelectedRegionId(obj._regionId!);
            // Scroll panel to this segment
            setTimeout(() => {
              const el = document.getElementById(`seg-${obj._segmentId}`);
              el?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
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
      if (!obj._segmentId || !obj._regionId) return;
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

      setRegions(prev => prev.map(r => r.id !== obj._regionId ? r : {
        ...r,
        segments: r.segments.map(s => s.id === obj._segmentId ? { ...s, x: nx, y: ny, w: nw, h: nh } : s),
      }));
    });

    // ── Space key pan ───────────────────────────────────────────────────────
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
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
  }, [regions, selectedFile]);

  // ── Render helper ─────────────────────────────────────────────────────────
  function _renderAllRegions(
    canvas: Canvas,
    regionList: AnnotationRegion[],
    bounds: { left: number; top: number; width: number; height: number },
  ) {
    const toRemove = canvas.getObjects().filter(o => (o as AnnotatedRect)._isRegion || (o as AnnotatedRect)._segmentId);
    toRemove.forEach(o => canvas.remove(o));

    regionList.forEach(region => {
      // Region outline (not selectable — just visual)
      const regionRect = new Rect({
        left:   bounds.left + region.x * bounds.width,
        top:    bounds.top  + region.y * bounds.height,
        width:  region.w * bounds.width,
        height: region.h * bounds.height,
        stroke: REGION_COLOR, strokeWidth: 2,
        fill: `${REGION_COLOR}12`,
        selectable: false, evented: false, hasControls: false,
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

    canvas.renderAll();
  }

  // ── Auto segment ──────────────────────────────────────────────────────────
  const handleSegment = useCallback(async (regionId: string) => {
    if (!selectedFile) return;
    const region = regions.find(r => r.id === regionId);
    if (!region) return;
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
  }, [project.id, selectedFile, regions]);

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
  }, []);

  const handleResetZoom = useCallback(() => {
    const canvas = fabricRef.current;
    if (!canvas) return;
    canvas.setViewportTransform([1, 0, 0, 1, 0, 0]);
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

  const deleteRegion = useCallback((regionId: string) => {
    setRegions(prev => prev.filter(r => r.id !== regionId));
    if (selectedRegionId === regionId) setSelectedRegionId(null);
  }, [selectedRegionId]);

  const deleteSegment = useCallback((regionId: string, segId: string) => {
    setRegions(prev => prev.map(r => r.id !== regionId ? r : {
      ...r, segments: r.segments.filter(s => s.id !== segId),
    }));
  }, []);

  // ── Stats ─────────────────────────────────────────────────────────────────
  const allSegs       = regions.flatMap(r => r.segments);
  const okCount       = allSegs.filter(s => s.label === 'OK').length;
  const ngCount       = allSegs.filter(s => s.label === 'NG').length;
  const unlabeledCount = allSegs.filter(s => !s.label).length;

  // ── Cursor for canvas wrapper ─────────────────────────────────────────────
  const wrapperCursor = drawMode !== 'select' ? 'crosshair' : 'default';

  return (
    <div className="ml-label-tab">

      {/* ── Left: image list ── */}
      <div className="ml-label-list">
        <div className="ml-panel-header" style={{ padding: '10px 12px' }}>Images</div>
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
            >
              <img className="ml-label-image-thumb" src={img.url} alt={img.filename} loading="lazy" />
              <span className="ml-label-image-name">{img.filename}</span>
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

          {/* Stats + Save */}
          {selectedFile && (
            <div style={{ marginLeft: 'auto', display: 'flex', gap: '10px', alignItems: 'center' }}>
              <span style={{ fontSize: '11px', display: 'flex', gap: '8px' }}>
                <span style={{ color: SEG_OK_COLOR }}>{okCount} OK</span>
                <span style={{ color: SEG_NG_COLOR }}>{ngCount} NG</span>
                {unlabeledCount > 0 && <span style={{ color: '#6b7280' }}>{unlabeledCount} ?</span>}
              </span>
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
            {drawMode === 'draw-region' && <span style={{ color: REGION_COLOR }}>Draw a rectangle around a character region → Auto Segment</span>}
            {drawMode === 'draw-char'   && <span style={{ color: SEG_UNLABELED }}>Draw a single character box manually{selectedRegionId ? '' : ' (will create new region)'}</span>}
            {drawMode === 'select'      && <span>Scroll to zoom · Space+drag to pan · Click char to cycle label · Drag to move/resize</span>}
          </div>
        )}
      </div>

      {/* ── Right: annotation panel ── */}
      <div className="ml-annotation-panel">
        <div className="ml-panel-header" style={{ padding: '10px 12px' }}>
          Regions {regions.length > 0 && <span className="ml-tab-count">{regions.length}</span>}
        </div>

        <div style={{ flex: 1, overflowY: 'auto' }}>
          {regions.length === 0 && (
            <div className="ml-empty-state" style={{ padding: '16px', fontSize: '11px' }}>
              Draw a region on the image
            </div>
          )}

          {regions.map((region, ri) => (
            <div key={region.id} style={{ borderBottom: '1px solid #e2e8f0' }}>

              {/* Region header */}
              <div
                className={`ml-segment-item ${selectedRegionId === region.id ? 'selected' : ''}`}
                onClick={() => setSelectedRegionId(prev => prev === region.id ? null : region.id)}
                style={{ background: selectedRegionId === region.id ? '#fef9ec' : '#f8fafc' }}
              >
                <span style={{ fontSize: '12px', fontWeight: 600, color: REGION_COLOR }}>
                  Region {ri + 1}
                  {region.segments.length > 0 && (
                    <span style={{ fontWeight: 400, color: '#94a3b8', marginLeft: 4 }}>
                      ({region.segments.length})
                    </span>
                  )}
                </span>
                <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
                  {region.segments.length === 0 && (
                    <button
                      className="ml-btn ml-btn-success ml-btn-sm"
                      style={{ padding: '2px 6px', fontSize: '10px' }}
                      onClick={e => { e.stopPropagation(); setSelectedRegionId(region.id); handleSegment(region.id); }}
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
                    onClick={e => { e.stopPropagation(); setSelectedRegionId(region.id); setDrawMode('draw-char'); }}
                  >
                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none"><path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>
                  </button>
                  <button
                    style={{ background: 'none', border: 'none', color: '#ef4444', cursor: 'pointer', padding: '0 2px', fontSize: '13px' }}
                    onClick={e => { e.stopPropagation(); deleteRegion(region.id); }}
                    title="Delete region"
                  >✕</button>
                </div>
              </div>

              {/* Segments */}
              {region.segments.map((seg, si) => (
                <div
                  key={seg.id}
                  id={`seg-${seg.id}`}
                  className={`ml-segment-item${selectedSegId === seg.id ? ' selected' : ''}`}
                  style={{ paddingLeft: '18px', fontSize: '11px', gap: '6px' }}
                  onClick={() => { setSelectedSegId(seg.id); setSelectedRegionId(region.id); }}
                >
                  {/* Color dot */}
                  <span style={{ width: 8, height: 8, borderRadius: '50%', background: segColor(seg.label ?? null), flexShrink: 0, display: 'inline-block' }} />
                  <span style={{ color: '#6b7280', flexShrink: 0 }}>Char {si + 1}</span>

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
          ))}
        </div>
      </div>
    </div>
  );
}
