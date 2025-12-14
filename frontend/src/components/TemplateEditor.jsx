import React, { useState, useRef, useEffect } from 'react';
import * as fabric from 'fabric';
import '../styles/TemplateEditor.css';

export default function TemplateEditor({ 
  templateImage, 
  annotations, 
  onAnnotationsChange, 
  selectedAnnotation, 
  onSelectAnnotation,
  fabricCanvasRef: externalCanvasRef 
}) {
  const canvasRef = useRef(null);
  const internalCanvasRef = useRef(null);
  const fabricCanvasRef = externalCanvasRef || internalCanvasRef;
  const containerRef = useRef(null);
  
  const [drawMode, setDrawMode] = useState('select');
  const [polygonPoints, setPolygonPoints] = useState([]);
  const [tempLines, setTempLines] = useState([]);
  const [showHints, setShowHints] = useState(true);
  const [objectBeforeTransform, setObjectBeforeTransform] = useState(null);
  const [isSpacePressed, setIsSpacePressed] = useState(false);
  const [polygonEditPoints, setPolygonEditPoints] = useState([]); // Circles để edit polygon points

  const ANNOTATION_TYPES = [
    { value: 'text', label: 'Text OCR', color: '#50fa7b', needsText: true },
    { value: 'barcode', label: 'Barcode', color: '#ffdc5c', needsText: false },
    { value: 'template', label: 'Template Match', color: '#ff5555', needsText: false },
    { value: 'crop_area', label: 'Crop Area', color: '#ff64ff', needsText: false },
    { value: 'datecode', label: 'Date Code', color: '#5096ff', needsText: true }
  ];

  useEffect(() => {
    if (!canvasRef.current || !containerRef.current) return;

    const canvas = new fabric.Canvas(canvasRef.current, {
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight,
      backgroundColor: '#1e1e1e',
      selection: drawMode === 'select'
    });

    fabricCanvasRef.current = canvas;

    // Enable panning with Space + Drag or Middle Mouse Button
    canvas.on('mouse:down', function(opt) {
      const evt = opt.e;
      if (evt.button === 1 || (evt.button === 0 && (evt.shiftKey || isSpacePressed))) { // Middle click or Shift/Space + Left click
        this.isDragging = true;
        this.selection = false;
        this.lastPosX = evt.clientX;
        this.lastPosY = evt.clientY;
        evt.preventDefault();
      }
    });

    canvas.on('mouse:move', function(opt) {
      if (this.isDragging) {
        const e = opt.e;
        const vpt = this.viewportTransform;
        vpt[4] += e.clientX - this.lastPosX;
        vpt[5] += e.clientY - this.lastPosY;
        this.requestRenderAll();
        this.lastPosX = e.clientX;
        this.lastPosY = e.clientY;
      }
    });

    canvas.on('mouse:up', function() {
      this.setViewportTransform(this.viewportTransform);
      this.isDragging = false;
      this.selection = drawMode === 'select';
    });

    // Load background image
    if (templateImage) {
      fabric.FabricImage.fromURL(templateImage, 
        { crossOrigin: 'anonymous' }
      ).then((img) => {
        const scale = Math.min(
          (canvas.width - 100) / img.width,
          (canvas.height - 100) / img.height
        );
        
        img.scale(scale);
        img.set({
          left: (canvas.width - img.width * scale) / 2,
          top: (canvas.height - img.height * scale) / 2,
          selectable: false,
          evented: false,
          hasControls: false,
          lockMovementX: true,
          lockMovementY: true
        });
        
        canvas.backgroundImage = img;
        canvas.requestRenderAll();
        loadAnnotations(canvas);
      }).catch(err => {
        console.error('Error loading image:', err);
      });
    }

    // Handle object selection
    canvas.on('selection:created', (e) => {
      const obj = e.selected[0];
      if (obj && obj.annotationIndex !== undefined) {
        onSelectAnnotation?.(obj.annotationIndex);
        // Nếu là polygon, tạo edit points
        if (obj.type === 'polygon') {
          createPolygonEditPoints(canvas, obj);
        }
      }
    });

    canvas.on('selection:updated', (e) => {
      const obj = e.selected[0];
      if (obj && obj.annotationIndex !== undefined) {
        onSelectAnnotation?.(obj.annotationIndex);
        // Xóa edit points cũ
        removePolygonEditPoints(canvas);
        // Nếu là polygon, tạo edit points mới
        if (obj.type === 'polygon') {
          createPolygonEditPoints(canvas, obj);
        }
      }
    });

    canvas.on('selection:cleared', () => {
      onSelectAnnotation?.(null);
      // Xóa edit points
      removePolygonEditPoints(canvas);
    });

    // Save object state before transformation
    canvas.on('object:moving', (e) => {
      if (!objectBeforeTransform && e.target.annotationIndex !== undefined) {
        setObjectBeforeTransform({
          target: e.target,
          left: e.target.left,
          top: e.target.top,
          scaleX: e.target.scaleX,
          scaleY: e.target.scaleY,
          angle: e.target.angle
        });
      }
    });

    canvas.on('object:scaling', (e) => {
      if (!objectBeforeTransform && e.target.annotationIndex !== undefined) {
        setObjectBeforeTransform({
          target: e.target,
          left: e.target.left,
          top: e.target.top,
          scaleX: e.target.scaleX,
          scaleY: e.target.scaleY,
          angle: e.target.angle
        });
      }
    });

    canvas.on('object:rotating', (e) => {
      if (!objectBeforeTransform && e.target.annotationIndex !== undefined) {
        setObjectBeforeTransform({
          target: e.target,
          left: e.target.left,
          top: e.target.top,
          scaleX: e.target.scaleX,
          scaleY: e.target.scaleY,
          angle: e.target.angle
        });
      }
    });

    // Handle object modification
    canvas.on('object:modified', (e) => {
      updateAnnotationFromObject(e.target);

      // Update edit points if polygon was moved
      if (e.target.type === 'polygon' && e.target.annotationIndex !== undefined) {
        removePolygonEditPoints(canvas);
        createPolygonEditPoints(canvas, e.target);
      }

      setObjectBeforeTransform(null); // Clear saved state after successful modification
    });

    // Clear saved state when mouse up without modification
    canvas.on('mouse:up', () => {
      // Only clear if object wasn't actually modified
      setTimeout(() => {
        if (objectBeforeTransform) {
          setObjectBeforeTransform(null);
        }
      }, 100);
    });

    return () => {
      canvas.dispose();
    };
  }, [templateImage, objectBeforeTransform]);

  // Handle drawing mouse events - separate useEffect for drawMode dependency
  useEffect(() => {
    const canvas = fabricCanvasRef.current;
    if (!canvas) return;

    // Remove old handlers
    canvas.off('mouse:down');

    // Add new handlers based on drawMode
    const handleMouseDown = (e) => {
      if (drawMode === 'polygon') {
        addPolygonPoint(canvas, e);
      } else if (drawMode === 'select') {
        // Click on empty canvas to deselect
        if (!e.target) {
          canvas.discardActiveObject();
          canvas.requestRenderAll();
          onSelectAnnotation?.(null);
        }
      }
    };

    canvas.on('mouse:down', handleMouseDown);

    return () => {
      canvas.off('mouse:down', handleMouseDown);
    };
  }, [drawMode, polygonPoints, annotations]);

  // Keyboard shortcuts - CVAT style
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Ignore if typing in input
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      
      // Handle Space key for panning
      if (e.code === 'Space' && !isSpacePressed) {
        e.preventDefault();
        setIsSpacePressed(true);
        if (fabricCanvasRef.current) {
          fabricCanvasRef.current.defaultCursor = 'grab';
          fabricCanvasRef.current.renderAll();
        }
        return;
      }
      
      switch(e.key.toLowerCase()) {
        case 'v':
          setDrawMode('select');
          break;
        case 'p':
          setDrawMode('polygon');
          setPolygonPoints([]);
          break;
        case 'escape':
          handleCancelDraw();
          // Also deselect any selected object and restore if transforming
          if (fabricCanvasRef.current) {
            // If object is being transformed, restore to previous state
            if (objectBeforeTransform) {
              const obj = objectBeforeTransform.target;
              obj.set({
                left: objectBeforeTransform.left,
                top: objectBeforeTransform.top,
                scaleX: objectBeforeTransform.scaleX,
                scaleY: objectBeforeTransform.scaleY,
                angle: objectBeforeTransform.angle
              });
              obj.setCoords();
              fabricCanvasRef.current.requestRenderAll();
              setObjectBeforeTransform(null);
            }
            fabricCanvasRef.current.discardActiveObject();
            fabricCanvasRef.current.requestRenderAll();
          }
          onSelectAnnotation?.(null);
          break;
        case 'delete':
        case 'backspace':
          // Parent handles deletion via selectedAnnotation
          e.preventDefault();
          break;
        case 'h':
          setShowHints(prev => !prev);
          break;
        default:
          break;
      }
    };

    const handleKeyUp = (e) => {
      // Release Space key
      if (e.code === 'Space' && isSpacePressed) {
        e.preventDefault();
        setIsSpacePressed(false);
        if (fabricCanvasRef.current) {
          fabricCanvasRef.current.defaultCursor = 'default';
          fabricCanvasRef.current.renderAll();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [selectedAnnotation, drawMode, isSpacePressed, objectBeforeTransform]);

  useEffect(() => {
    if (fabricCanvasRef.current) {
      fabricCanvasRef.current.selection = drawMode === 'select';
      fabricCanvasRef.current.forEachObject((obj) => {
        if (obj.annotationIndex !== undefined) {
          obj.selectable = drawMode === 'select';
          obj.evented = drawMode === 'select';
        }
      });
      fabricCanvasRef.current.requestRenderAll();
    }
  }, [drawMode]);

  // Reload annotations when they change
  useEffect(() => {
    if (fabricCanvasRef.current && annotations) {
      loadAnnotations(fabricCanvasRef.current);
    }
  }, [annotations]);

  const createPolygonEditPoints = (canvas, polygon) => {
    const ann = annotations[polygon.annotationIndex];
    if (!ann || !ann.points) return;

    const color = ANNOTATION_TYPES.find(t => t.value === ann.type)?.color || '#ffffff';
    const editPoints = [];

    ann.points.forEach((point, idx) => {
      const circle = new fabric.Circle({
        left: point[0],
        top: point[1],
        radius: 10, // Tăng size để dễ click
        fill: color,
        stroke: '#ffffff',
        strokeWidth: 2,
        originX: 'center',
        originY: 'center',
        selectable: true,
        hasControls: false,
        hasBorders: false,
        hoverCursor: 'pointer',
        isPolygonEditPoint: true,
        polygonRef: polygon,
        pointIndex: idx
      });

      // Handle dragging edit point (visual update only)
      circle.on('moving', function() {
        const poly = this.polygonRef;
        const idx = this.pointIndex;

        // Update polygon point visually
        poly.points[idx].x = this.left;
        poly.points[idx].y = this.top;

        // Recalculate polygon path
        poly.set({ dirty: true });
        poly.setCoords();
        canvas.requestRenderAll();
      });

      // Handle when edit point drag finished (save to annotations)
      circle.on('modified', function() {
        const poly = this.polygonRef;
        const idx = this.pointIndex;

        // Update annotation data
        const updated = [...annotations];
        const ann = updated[poly.annotationIndex];
        if (ann && ann.points) {
          ann.points[idx] = [this.left, this.top];
          onAnnotationsChange(updated);
        }
      });

      canvas.add(circle);
      editPoints.push(circle);
    });

    setPolygonEditPoints(editPoints);
  };

  const removePolygonEditPoints = (canvas) => {
    // Remove all edit point circles
    const pointsToRemove = canvas.getObjects().filter(obj => obj.isPolygonEditPoint);
    pointsToRemove.forEach(obj => canvas.remove(obj));
    setPolygonEditPoints([]);
  };

  const loadAnnotations = (canvas) => {
    // Remove existing annotation objects
    const objectsToRemove = canvas.getObjects().filter(obj =>
      obj.annotationIndex !== undefined || obj.isLabel || obj.isPolygonEditPoint
    );
    objectsToRemove.forEach(obj => canvas.remove(obj));

    annotations.forEach((ann, index) => {
      const color = ANNOTATION_TYPES.find(t => t.value === ann.type)?.color || '#ffffff';
      
      if (ann.shape === 'rectangle') {
        const rect = new fabric.Rect({
          left: ann.x,
          top: ann.y,
          width: ann.width,
          height: ann.height,
          fill: color + '20',
          stroke: color,
          strokeWidth: 2,
          selectable: true,
          evented: true,
          hasControls: true,
          hasBorders: true,
          lockRotation: true, // Không cho xoay
          lockScalingFlip: true, // Không cho flip khi resize
          lockScalingX: false, // Cho phép resize theo X
          lockScalingY: false, // Cho phép resize theo Y
          lockMovementX: false, // Cho phép di chuyển X
          lockMovementY: false, // Cho phép di chuyển Y
          lockSkewingX: true,
          lockSkewingY: true,
          cornerSize: 10,
          transparentCorners: false,
          cornerColor: color,
          cornerStrokeColor: '#ffffff',
          borderColor: color,
          objectCaching: false, // Tắt cache để update real-time
          annotationIndex: index,
          annotationType: ann.type
        });

        // Đảm bảo tất cả controls hiển thị (trừ rotation)
        rect.setControlsVisibility({
          mt: true,  // middle top
          mb: true,  // middle bottom
          ml: true,  // middle left
          mr: true,  // middle right
          tl: true,  // top left
          tr: true,  // top right
          bl: true,  // bottom left
          br: true,  // bottom right
          mtr: false // rotation (ẩn vì đã lock)
        });

        canvas.add(rect);
        addLabel(canvas, ann.type, ann.x, ann.y, color);
      } else if (ann.shape === 'polygon' && ann.points) {
        const points = ann.points.map(p => ({ x: p[0], y: p[1] }));
        const polygon = new fabric.Polygon(points, {
          fill: color + '20',
          stroke: color,
          strokeWidth: 2,
          selectable: true,
          hasControls: false, // Tắt hết controls mặc định
          hasBorders: true,
          lockRotation: true,
          lockScalingX: true,
          lockScalingY: true,
          cornerSize: 10,
          transparentCorners: false,
          cornerColor: color,
          cornerStrokeColor: '#ffffff',
          borderColor: color,
          objectCaching: false,
          perPixelTargetFind: true,
          annotationIndex: index,
          annotationType: ann.type
        });

        canvas.add(polygon);
        addLabel(canvas, ann.type, ann.points[0][0], ann.points[0][1], color);
      }
    });
    
    canvas.requestRenderAll();
  };

  const addLabel = (canvas, type, x, y, color) => {
    const label = new fabric.FabricText(type.toUpperCase(), {
      left: x + 5,
      top: y - 25,
      fontSize: 14,
      fill: color,
      selectable: false,
      evented: false,
      fontFamily: 'Arial',
      isLabel: true
    });
    canvas.add(label);
  };

  const startDrawingRect = (canvas, e) => {
    const pointer = canvas.getPointer(e.e);
    const color = '#00ffff'; // Bright cyan for visibility while drawing
    
    const rect = new fabric.Rect({
      left: pointer.x,
      top: pointer.y,
      width: 0,
      height: 0,
      fill: color + '30',
      stroke: color,
      strokeWidth: 3,
      selectable: false
    });
    
    canvas.add(rect);
    
    let isDrawing = true;
    
    canvas.on('mouse:move', function(e) {
      if (!isDrawing) return;
      const pointer = canvas.getPointer(e.e);
      rect.set({
        width: Math.abs(pointer.x - rect.left),
        height: Math.abs(pointer.y - rect.top)
      });
      canvas.requestRenderAll();
    });
    
    canvas.on('mouse:up', function() {
      if (!isDrawing) return;
      isDrawing = false;
      canvas.off('mouse:move');
      canvas.off('mouse:up');
      
      if (rect.width > 10 && rect.height > 10) {
        // Auto-add annotation with default type
        rect.set({
          selectable: false,
          evented: false,
          isPendingShape: true
        });
        addAnnotation({
          shape: 'rectangle',
          x: rect.left,
          y: rect.top,
          width: rect.width,
          height: rect.height,
          fabricObject: rect
        });
        setDrawMode('select');
      } else {
        canvas.remove(rect);
        setDrawMode('select');
      }
    });
  };

  const addPolygonPoint = (canvas, e) => {
    const pointer = canvas.getPointer(e.e);
    const color = '#00ffff'; // Bright cyan for visibility while drawing
    
    const newPoint = { x: pointer.x, y: pointer.y };
    const updatedPoints = [...polygonPoints, newPoint];
    setPolygonPoints(updatedPoints);
    
    // Draw point
    const circle = new fabric.Circle({
      left: pointer.x - 5,
      top: pointer.y - 5,
      radius: 5,
      fill: color,
      stroke: '#ffffff',
      strokeWidth: 2,
      selectable: false,
      evented: false,
      isLabel: true
    });
    canvas.add(circle);
    setTempLines(prev => [...prev, circle]);
    
    // Draw line from previous point
    if (updatedPoints.length > 1) {
      const prevPoint = updatedPoints[updatedPoints.length - 2];
      const line = new fabric.Line(
        [prevPoint.x, prevPoint.y, newPoint.x, newPoint.y],
        {
          stroke: color,
          strokeWidth: 3,
          selectable: false,
          evented: false
        }
      );
      canvas.add(line);
      setTempLines(prev => [...prev, line]);
    }
    
    // Complete polygon after 4 points
    if (updatedPoints.length === 4) {
      // Remove temp objects (points and lines)
      tempLines.forEach(obj => canvas.remove(obj));
      setTempLines([]);
      
      // Draw final polygon
      const polygon = new fabric.Polygon(updatedPoints, {
        fill: color + '30',
        stroke: color,
        strokeWidth: 3,
        selectable: false,
        evented: false,
        isPendingShape: true
      });
      canvas.add(polygon);
      canvas.requestRenderAll();
      
      // Auto-add annotation with default type
      addAnnotation({
        shape: 'polygon',
        points: updatedPoints.map(p => [p.x, p.y]),
        fabricObject: polygon
      });
      setPolygonPoints([]);
      setDrawMode('select');
    }
    
    canvas.requestRenderAll();
  };

  const updateAnnotationFromObject = (obj) => {
    if (obj.annotationIndex === undefined) return;
    
    const updated = [...annotations];
    const ann = updated[obj.annotationIndex];
    
    if (ann.shape === 'rectangle') {
      ann.x = obj.left;
      ann.y = obj.top;
      ann.width = obj.width * obj.scaleX;
      ann.height = obj.height * obj.scaleY;
    } else if (ann.shape === 'polygon') {
      const matrix = obj.calcTransformMatrix();
      const transformedPoints = [];
      
      obj.points.forEach(p => {
        const point = fabric.util.transformPoint(
          { x: p.x - (obj.pathOffset?.x || 0), y: p.y - (obj.pathOffset?.y || 0) },
          matrix
        );
        transformedPoints.push([point.x, point.y]);
      });
      
      ann.points = transformedPoints;
    }
    
    onAnnotationsChange(updated);
  };



  const handleZoomIn = () => {
    const canvas = fabricCanvasRef.current;
    if (canvas) {
      const zoom = canvas.getZoom();
      canvas.setZoom(Math.min(zoom * 1.2, 3));
      canvas.requestRenderAll();
    }
  };

  const handleZoomOut = () => {
    const canvas = fabricCanvasRef.current;
    if (canvas) {
      const zoom = canvas.getZoom();
      canvas.setZoom(Math.max(zoom / 1.2, 0.5));
      canvas.requestRenderAll();
    }
  };

  const handleResetZoom = () => {
    const canvas = fabricCanvasRef.current;
    if (canvas) {
      canvas.setZoom(1);
      canvas.viewportTransform[4] = 0;
      canvas.viewportTransform[5] = 0;
      canvas.requestRenderAll();
    }
  };

  const handleCancelDraw = () => {
    if (fabricCanvasRef.current) {
      // Remove temp objects
      tempLines.forEach(obj => fabricCanvasRef.current.remove(obj));
      setTempLines([]);
      setPolygonPoints([]);
      fabricCanvasRef.current.requestRenderAll();
    }
    setDrawMode('select');
  };

  const addAnnotation = (shapeData, type = 'text') => {
    // Remove pending shape from canvas if it exists
    if (shapeData.fabricObject && fabricCanvasRef.current) {
      fabricCanvasRef.current.remove(shapeData.fabricObject);
    }

    // Create annotation without fabricObject reference
    const { fabricObject, ...cleanShapeData } = shapeData;
    const newAnnotation = {
      type,
      text: '', // Default empty text, can be edited in panel
      ...cleanShapeData
    };

    const updatedAnnotations = [...annotations, newAnnotation];
    onAnnotationsChange(updatedAnnotations);
    
    // Auto-select the newly created annotation
    onSelectAnnotation(updatedAnnotations.length - 1);
  };

  return (
    <div className="template-editor">
      <div className="editor-toolbar">
        <div className="toolbar-group">
          <button
            type="button"
            className={`tool-btn ${drawMode === 'select' ? 'active' : ''}`}
            onClick={() => setDrawMode('select')}
            title="Select & Edit"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M3 3L10.07 19.97L12.58 12.58L19.97 10.07L3 3Z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
            </svg>
          </button>
          <button
            type="button"
            className={`tool-btn ${drawMode === 'polygon' ? 'active' : ''}`}
            onClick={() => {
              setDrawMode('polygon');
              setPolygonPoints([]);
            }}
            title="Draw Polygon (4 points)"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M12 2L22 8.5L18 20.5H6L2 8.5L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
            </svg>
          </button>
          
          {drawMode === 'polygon' && (
            <button type="button" className="tool-btn cancel-btn" onClick={handleCancelDraw} title="Cancel Drawing">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <line x1="18" y1="6" x2="6" y2="18" stroke="currentColor" strokeWidth="2"/>
                <line x1="6" y1="6" x2="18" y2="18" stroke="currentColor" strokeWidth="2"/>
              </svg>
            </button>
          )}
        </div>

        <div className="toolbar-group">
          <button type="button" className="tool-btn" onClick={handleZoomOut} title="Zoom Out">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <circle cx="11" cy="11" r="8" stroke="currentColor" strokeWidth="2"/>
              <line x1="8" y1="11" x2="14" y2="11" stroke="currentColor" strokeWidth="2"/>
              <line x1="17" y1="17" x2="21" y2="21" stroke="currentColor" strokeWidth="2"/>
            </svg>
          </button>
          <span className="zoom-label">
            {fabricCanvasRef.current ? Math.round(fabricCanvasRef.current.getZoom() * 100) : 100}%
          </span>
          <button type="button" className="tool-btn" onClick={handleZoomIn} title="Zoom In">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <circle cx="11" cy="11" r="8" stroke="currentColor" strokeWidth="2"/>
              <line x1="11" y1="8" x2="11" y2="14" stroke="currentColor" strokeWidth="2"/>
              <line x1="8" y1="11" x2="14" y2="11" stroke="currentColor" strokeWidth="2"/>
              <line x1="17" y1="17" x2="21" y2="21" stroke="currentColor" strokeWidth="2"/>
            </svg>
          </button>
          <button type="button" className="tool-btn" onClick={handleResetZoom} title="Reset Zoom">
            100%
          </button>
        </div>
      </div>

      <div className="editor-content">
        <div className="canvas-container" ref={containerRef}>
          <canvas ref={canvasRef} />
          
          {/* Canvas Hints - CVAT Style */}
          {showHints && (
            <div className="cvat-canvas-hints-container">
              <button 
                type="button"
                className="cvat-canvas-hints-hide-button"
                onClick={() => setShowHints(false)}
                title="Hide hints"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                  <line x1="18" y1="6" x2="6" y2="18" stroke="white" strokeWidth="2"/>
                  <line x1="6" y1="6" x2="18" y2="18" stroke="white" strokeWidth="2"/>
                </svg>
              </button>
              
              <div className="cvat-canvas-hints-block">
                <strong>🖱️ Navigation</strong>
                <ul>
                  <li><kbd>Space</kbd> + Drag - Pan canvas</li>
                  <li><kbd>Shift</kbd> + Drag - Pan canvas</li>
                  <li><kbd>Middle Click</kbd> + Drag - Pan canvas</li>
                  <li><kbd>Scroll</kbd> - Zoom in/out</li>
                </ul>
              </div>
              
              <div className="cvat-canvas-hints-block">
                <strong>⌨️ Shortcuts</strong>
                <ul>
                  <li><kbd>V</kbd> - Select mode</li>
                  <li><kbd>P</kbd> - Polygon tool (4 points)</li>
                  <li><kbd>Esc</kbd> - Cancel drawing</li>
                  <li><kbd>Del</kbd> - Delete selected</li>
                </ul>
              </div>

              <div className="cvat-canvas-hints-block">
                <strong>✏️ Drawing</strong>
                <ul>
                  <li>Polygon: Click 4 points</li>
                  <li>Edit: Drag corners/points to resize</li>
                  <li>Move: Drag shape to reposition</li>
                </ul>
              </div>
            </div>
          )}
          
          {!showHints && (
            <button 
              type="button"
              className="cvat-canvas-hints-show-button"
              onClick={() => setShowHints(true)}
              title="Show hints"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
                <path d="M12 16v-4M12 8h.01" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              </svg>
            </button>
          )}
          
          {polygonPoints.length > 0 && (
            <div className="polygon-hint">
              Click {4 - polygonPoints.length} more point{4 - polygonPoints.length !== 1 ? 's' : ''} to complete polygon
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
