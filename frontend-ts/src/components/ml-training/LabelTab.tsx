import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Canvas, FabricImage, Rect } from 'fabric';
import { mlTrainingAPI, AnnotationRegion, CharSegment, MLProject, ProjectImage } from '@/services/mlTraining';

const uuidv4 = () => crypto.randomUUID();

interface Props {
  project: MLProject;
  onRefresh: () => void;
}

type DrawMode = 'select' | 'draw-region';

const REGION_COLOR   = '#f59e0b';  // amber — user-drawn region
const SEG_UNLABELED  = '#6b7280';  // gray
const SEG_OK_COLOR   = '#22c55e';  // green
const SEG_NG_COLOR   = '#ef4444';  // red

// ── Canvas object augmented with metadata ─────────────────────────────────
interface AnnotatedRect extends Rect {
  _regionId?: string;
  _segmentId?: string;
  _label?: 'OK' | 'NG' | null;
  _isRegion?: boolean;
}

export default function LabelTab({ project, onRefresh }: Props) {
  // Image list
  const [images, setImages] = useState<ProjectImage[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);

  // Canvas
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const fabricRef = useRef<Canvas | null>(null);
  const imageBoundsRef = useRef<{ left: number; top: number; width: number; height: number } | null>(null);

  // Annotation state (source of truth)
  const [regions, setRegions] = useState<AnnotationRegion[]>([]);
  const [drawMode, setDrawMode] = useState<DrawMode>('select');
  const [segmenting, setSegmenting] = useState(false);
  const [saving, setSaving] = useState(false);
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState<{ w: number; h: number }>({ w: 1, h: 1 });

  // Drawing state
  const drawStartRef = useRef<{ x: number; y: number } | null>(null);
  const drawRectRef = useRef<AnnotatedRect | null>(null);
  const isDrawingRef = useRef(false);

  // ── Load image list ───────────────────────────────────────────────────
  const loadImages = useCallback(async () => {
    try {
      const data = await mlTrainingAPI.listProjectImages(project.id);
      setImages(data);
    } catch { /* ignore */ }
  }, [project.id]);

  useEffect(() => {
    loadImages();
    setSelectedFile(null);
    setRegions([]);
  }, [project.id]);

  // ── Select image → load into canvas ──────────────────────────────────
  const selectImage = useCallback(async (filename: string) => {
    setSelectedFile(filename);
    setRegions([]);
    setSelectedRegionId(null);
    setDrawMode('select');

    try {
      const [imgData, annData] = await Promise.all([
        mlTrainingAPI.getImageB64(project.id, filename),
        mlTrainingAPI.getAnnotation(project.id, filename),
      ]);

      setImageSize({ w: imgData.width, h: imgData.height });

      const canvas = fabricRef.current;
      if (!canvas || !containerRef.current) return;

      // Clear canvas
      canvas.clear();

      // Load image
      const dataUrl = `data:image/jpeg;base64,${imgData.image_b64}`;
      const img = await FabricImage.fromURL(dataUrl, { crossOrigin: 'anonymous' });

      const cw = containerRef.current.clientWidth;
      const ch = containerRef.current.clientHeight;
      const scale = Math.min((cw - 40) / imgData.width, (ch - 40) / imgData.height);
      img.scale(scale);
      const left = (cw - imgData.width * scale) / 2;
      const top = (ch - imgData.height * scale) / 2;
      img.set({ left, top, selectable: false, evented: false });

      canvas.backgroundImage = img;
      imageBoundsRef.current = { left, top, width: imgData.width * scale, height: imgData.height * scale };
      canvas.renderAll();

      // Restore saved annotations
      if (annData.regions.length > 0) {
        setRegions(annData.regions);
        _renderAllRegions(canvas, annData.regions, imageBoundsRef.current);
      }
    } catch (e) {
      console.error('Failed to load image', e);
    }
  }, [project.id]);

  // ── Init Fabric canvas ────────────────────────────────────────────────
  useEffect(() => {
    if (!canvasRef.current || !containerRef.current) return;

    const canvas = new Canvas(canvasRef.current, {
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight,
      backgroundColor: '#0f1117',
      selection: false,
    }) as Canvas;

    fabricRef.current = canvas;

    const handleMouseDown = (e: any) => {
      if (isDrawingRef.current || drawMode !== 'draw-region') return;
      const bounds = imageBoundsRef.current;
      if (!bounds) return;
      const pointer = canvas.getScenePoint(e.e);
      // Clamp to image bounds
      const x = Math.max(bounds.left, Math.min(pointer.x, bounds.left + bounds.width));
      const y = Math.max(bounds.top, Math.min(pointer.y, bounds.top + bounds.height));
      drawStartRef.current = { x, y };
      isDrawingRef.current = true;

      const rect = new Rect({
        left: x, top: y, width: 0, height: 0,
        stroke: REGION_COLOR, strokeWidth: 2,
        fill: `${REGION_COLOR}22`,
        selectable: false, evented: false,
      }) as AnnotatedRect;
      rect._isRegion = true;
      drawRectRef.current = rect;
      canvas.add(rect);
    };

    const handleMouseMove = (e: any) => {
      if (!isDrawingRef.current || !drawStartRef.current || !drawRectRef.current) return;
      const bounds = imageBoundsRef.current;
      if (!bounds) return;
      const pointer = canvas.getScenePoint(e.e);
      const ex = Math.max(bounds.left, Math.min(pointer.x, bounds.left + bounds.width));
      const ey = Math.max(bounds.top, Math.min(pointer.y, bounds.top + bounds.height));
      const { x, y } = drawStartRef.current;
      drawRectRef.current.set({
        left: Math.min(x, ex),
        top: Math.min(y, ey),
        width: Math.abs(ex - x),
        height: Math.abs(ey - y),
      });
      canvas.renderAll();
    };

    const handleMouseUp = () => {
      if (!isDrawingRef.current || !drawStartRef.current || !drawRectRef.current) return;
      isDrawingRef.current = false;
      const rect = drawRectRef.current;
      const bounds = imageBoundsRef.current;

      if (!bounds || (rect.width ?? 0) < 10 || (rect.height ?? 0) < 10) {
        canvas.remove(rect);
        drawStartRef.current = null;
        drawRectRef.current = null;
        return;
      }

      // Normalize to 0-1 relative to image
      const regionId = uuidv4();
      const normRegion: AnnotationRegion = {
        id: regionId,
        x: ((rect.left ?? 0) - bounds.left) / bounds.width,
        y: ((rect.top ?? 0) - bounds.top) / bounds.height,
        w: (rect.width ?? 0) / bounds.width,
        h: (rect.height ?? 0) / bounds.height,
        segments: [],
      };

      rect._regionId = regionId;
      rect.set({ selectable: true });
      canvas.renderAll();

      setRegions(prev => [...prev, normRegion]);
      setSelectedRegionId(regionId);
      drawStartRef.current = null;
      drawRectRef.current = null;
      setDrawMode('select');
    };

    canvas.on('mouse:down', handleMouseDown);
    canvas.on('mouse:move', handleMouseMove);
    canvas.on('mouse:up', handleMouseUp);

    // Click segment rects to cycle label
    canvas.on('mouse:down', (e: any) => {
      const target = e.target as AnnotatedRect | null;
      if (!target || target._isRegion) return;
      if (!target._segmentId || !target._regionId) return;

      // Cycle label: null → OK → NG → null
      const current = target._label ?? null;
      const next: 'OK' | 'NG' | null = current === null ? 'OK' : current === 'OK' ? 'NG' : null;
      target._label = next;
      target.set({
        stroke: next === 'OK' ? SEG_OK_COLOR : next === 'NG' ? SEG_NG_COLOR : SEG_UNLABELED,
        fill: next === 'OK' ? `${SEG_OK_COLOR}33` : next === 'NG' ? `${SEG_NG_COLOR}33` : `${SEG_UNLABELED}22`,
      });
      canvas.renderAll();

      // Sync to state
      setRegions(prev => prev.map(r => {
        if (r.id !== target._regionId) return r;
        return {
          ...r,
          segments: r.segments.map(s =>
            s.id === target._segmentId ? { ...s, label: next } : s
          ),
        };
      }));
    });

    return () => { canvas.dispose(); fabricRef.current = null; };
  }, []);

  // Keep drawMode accessible inside canvas handlers via ref
  const drawModeRef = useRef(drawMode);
  useEffect(() => { drawModeRef.current = drawMode; }, [drawMode]);

  // ── Render all regions on canvas ──────────────────────────────────────
  function _renderAllRegions(
    canvas: Canvas,
    regionList: AnnotationRegion[],
    bounds: { left: number; top: number; width: number; height: number },
  ) {
    // Remove old annotation objects (keep background)
    const toRemove = canvas.getObjects().filter((o) => {
      const r = o as AnnotatedRect;
      return r._isRegion || r._segmentId;
    });
    toRemove.forEach(o => canvas.remove(o));

    regionList.forEach(region => {
      // Region rect
      const rx = bounds.left + region.x * bounds.width;
      const ry = bounds.top + region.y * bounds.height;
      const rw = region.w * bounds.width;
      const rh = region.h * bounds.height;

      const regionRect = new Rect({
        left: rx, top: ry, width: rw, height: rh,
        stroke: REGION_COLOR, strokeWidth: 2,
        fill: `${REGION_COLOR}15`,
        selectable: true, evented: true,
      }) as AnnotatedRect;
      regionRect._isRegion = true;
      regionRect._regionId = region.id;
      canvas.add(regionRect);

      // Segment rects
      region.segments.forEach(seg => {
        const color = seg.label === 'OK' ? SEG_OK_COLOR
          : seg.label === 'NG' ? SEG_NG_COLOR
          : SEG_UNLABELED;
        const sx = bounds.left + seg.x * bounds.width;
        const sy = bounds.top + seg.y * bounds.height;
        const sw = seg.w * bounds.width;
        const sh = seg.h * bounds.height;

        const segRect = new Rect({
          left: sx, top: sy, width: sw, height: sh,
          stroke: color, strokeWidth: 1.5,
          fill: `${color}33`,
          selectable: false, evented: true,
        }) as AnnotatedRect;
        segRect._segmentId = seg.id;
        segRect._regionId = region.id;
        segRect._label = seg.label as 'OK' | 'NG' | null;
        canvas.add(segRect);
      });
    });

    canvas.renderAll();
  }

  // Re-render when regions change
  useEffect(() => {
    const canvas = fabricRef.current;
    const bounds = imageBoundsRef.current;
    if (!canvas || !bounds || !selectedFile) return;
    _renderAllRegions(canvas, regions, bounds);
  }, [regions, selectedFile]);

  // ── Auto-segment selected region ──────────────────────────────────────
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
      setRegions(prev => prev.map(r =>
        r.id === regionId ? { ...r, segments } : r
      ));
    } catch (e) {
      console.error('Segmentation failed', e);
    } finally {
      setSegmenting(false);
    }
  }, [project.id, selectedFile, regions]);

  // ── Save annotations ──────────────────────────────────────────────────
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

  // ── Delete region ─────────────────────────────────────────────────────
  const deleteRegion = useCallback((regionId: string) => {
    setRegions(prev => prev.filter(r => r.id !== regionId));
    if (selectedRegionId === regionId) setSelectedRegionId(null);
  }, [selectedRegionId]);

  // ── Stats ─────────────────────────────────────────────────────────────
  const allSegs = regions.flatMap(r => r.segments);
  const okCount = allSegs.filter(s => s.label === 'OK').length;
  const ngCount = allSegs.filter(s => s.label === 'NG').length;
  const unlabeledCount = allSegs.filter(s => !s.label).length;

  return (
    <div className="ml-label-tab">
      {/* Image list */}
      <div className="ml-label-list">
        <div className="ml-panel-header" style={{ padding: '10px 12px' }}>Images</div>
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {images.length === 0 && (
            <div className="ml-empty-state" style={{ padding: '20px' }}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" style={{opacity:.4}}><rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="2"/><circle cx="8.5" cy="8.5" r="1.5" fill="currentColor"/><path d="M21 15l-5-5L5 21" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/></svg>
              No images
            </div>
          )}
          {images.map(img => (
            <div
              key={img.filename}
              className={`ml-label-image-item ${selectedFile === img.filename ? 'active' : ''} ${img.has_annotation ? 'has-label' : ''}`}
              onClick={() => selectImage(img.filename)}
            >
              <img
                className="ml-label-image-thumb"
                src={`data:image/jpeg;base64,${img.thumbnail_b64}`}
                alt={img.filename}
              />
              <span className="ml-label-image-name">{img.filename}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Canvas area */}
      <div className="ml-canvas-area">
        <div className="ml-canvas-toolbar">
          <button
            className={`ml-btn ml-btn-sm ${drawMode === 'select' ? 'ml-btn-primary' : 'ml-btn-secondary'}`}
            onClick={() => setDrawMode('select')}
            title="Select / pan (V)"
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none"><path d="M4 4l7 18 3-7 7-3L4 4z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/></svg>
            Select
          </button>
          <button
            className={`ml-btn ml-btn-sm ${drawMode === 'draw-region' ? 'ml-btn-primary' : 'ml-btn-secondary'}`}
            onClick={() => setDrawMode('draw-region')}
            disabled={!selectedFile}
            title="Draw region (R)"
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none"><rect x="3" y="6" width="18" height="12" rx="1" stroke="currentColor" strokeWidth="2"/></svg>
            Draw Region
          </button>
          <div className="ml-canvas-toolbar-sep" />
          {selectedRegionId && (
            <button
              className="ml-btn ml-btn-success ml-btn-sm"
              onClick={() => handleSegment(selectedRegionId)}
              disabled={segmenting}
            >
              {segmenting ? <><span className="ml-loading-spinner" style={{width:12,height:12,borderWidth:2}}/> Segmenting...</> : <><svg width="13" height="13" viewBox="0 0 24 24" fill="none"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/></svg> Auto Segment</>}
            </button>
          )}
          <div style={{ marginLeft: 'auto', display: 'flex', gap: '8px', alignItems: 'center' }}>
            {selectedFile && (
              <>
                <span style={{ fontSize: '11px', color: '#6b7280' }}>
                  <span style={{ color: '#22c55e', display:'flex', alignItems:'center', gap:3 }}>
                    <svg width="11" height="11" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
                    {okCount} OK
                  </span>
                  {' · '}
                  <span style={{ color: '#ef4444', display:'flex', alignItems:'center', gap:3 }}>
                    <svg width="11" height="11" viewBox="0 0 24 24" fill="none"><path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"/></svg>
                    {ngCount} NG
                  </span>
                  {unlabeledCount > 0 && ` · ${unlabeledCount} unlabeled`}
                </span>
                <button
                  className="ml-btn ml-btn-primary ml-btn-sm"
                  onClick={handleSave}
                  disabled={saving}
                >
                  {saving ? 'Saving...' : <><svg width="13" height="13" viewBox="0 0 24 24" fill="none"><path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/><polyline points="17 21 17 13 7 13 7 21" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/><polyline points="7 3 7 8 15 8" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/></svg> Save</>}
                </button>
              </>
            )}
          </div>
        </div>

        {!selectedFile ? (
          <div className="ml-empty-state" style={{ flex: 1 }}>
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" style={{opacity:.4}}><rect x="9" y="4" width="12" height="16" rx="2" stroke="currentColor" strokeWidth="2"/><path d="M9 12H3m0 0l3-3m-3 3l3 3" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>
            Select an image to start labeling
          </div>
        ) : (
          <div className="ml-canvas-wrapper" ref={containerRef} style={{ cursor: drawMode === 'draw-region' ? 'crosshair' : 'default' }}>
            <canvas ref={canvasRef} />
          </div>
        )}

        {selectedFile && drawMode === 'draw-region' && (
          <div style={{ padding: '6px 12px', background: '#1a1d27', borderTop: '1px solid #2d3148', fontSize: '11px', color: '#f59e0b' }}>
            ✏️ Draw a rectangle around the character region, then click "Auto Segment"
          </div>
        )}
        {selectedFile && drawMode === 'select' && allSegs.length > 0 && (
          <div style={{ padding: '6px 12px', background: '#1a1d27', borderTop: '1px solid #2d3148', fontSize: '11px', color: '#9ca3af' }}>
            💡 Click each character box to cycle label: gray → <span style={{ color: '#22c55e' }}>OK</span> → <span style={{ color: '#ef4444' }}>NG</span>
          </div>
        )}
      </div>

      {/* Right annotation panel */}
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
            <div key={region.id} style={{ borderBottom: '1px solid #2d3148' }}>
              {/* Region header */}
              <div
                className={`ml-segment-item ${selectedRegionId === region.id ? 'selected' : ''}`}
                onClick={() => setSelectedRegionId(prev => prev === region.id ? null : region.id)}
                style={{ background: '#1a1d27' }}
              >
                <span style={{ fontSize: '12px', fontWeight: 600, color: '#f59e0b' }}>
                  Region {ri + 1}
                </span>
                <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
                  {region.segments.length === 0 ? (
                    <button
                      className="ml-btn ml-btn-success ml-btn-sm"
                      style={{ padding: '2px 6px', fontSize: '10px' }}
                      onClick={e => { e.stopPropagation(); setSelectedRegionId(region.id); handleSegment(region.id); }}
                      disabled={segmenting}
                    >
                      <svg width="11" height="11" viewBox="0 0 24 24" fill="none"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/></svg> Segment
                    </button>
                  ) : (
                    <span style={{ fontSize: '10px', color: '#6b7280' }}>{region.segments.length} chars</span>
                  )}
                  <button
                    style={{ background: 'none', border: 'none', color: '#ef4444', cursor: 'pointer', fontSize: '12px' }}
                    onClick={e => { e.stopPropagation(); deleteRegion(region.id); }}
                    title="Delete region"
                  >✕</button>
                </div>
              </div>

              {/* Segments */}
              {region.segments.map((seg, si) => (
                <div
                  key={seg.id}
                  className="ml-segment-item"
                  style={{ paddingLeft: '20px', fontSize: '11px' }}
                >
                  <span style={{ color: '#6b7280' }}>Char {si + 1}</span>
                  <span className={`ml-label-badge ${seg.label === 'OK' ? 'ok' : seg.label === 'NG' ? 'ng' : 'none'}`}>
                    {seg.label ?? 'unlabeled'}
                  </span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
