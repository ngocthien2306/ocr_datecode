import React, { useState, useRef, useEffect } from 'react';
import * as fabric from 'fabric';
import '../styles/TemplateEditor.css';

export default function TemplateEditor({ templateImage, annotations, onAnnotationsChange }) {
  const canvasRef = useRef(null);
  const fabricCanvasRef = useRef(null);
  const containerRef = useRef(null);
  
  const [drawMode, setDrawMode] = useState('select');
  const [selectedAnnotation, setSelectedAnnotation] = useState(null);
  const [polygonPoints, setPolygonPoints] = useState([]);
  const [tempLines, setTempLines] = useState([]);
  
  // Type selection dialog states
  const [showTypeDialog, setShowTypeDialog] = useState(false);
  const [pendingShape, setPendingShape] = useState(null);
  const [selectedType, setSelectedType] = useState('text');
  const [textContent, setTextContent] = useState('');

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

    // Enable panning with Alt + Drag
    canvas.on('mouse:down', function(opt) {
      const evt = opt.e;
      if (evt.altKey === true) {
        this.isDragging = true;
        this.selection = false;
        this.lastPosX = evt.clientX;
        this.lastPosY = evt.clientY;
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
        setSelectedAnnotation(obj.annotationIndex);
      }
    });

    canvas.on('selection:updated', (e) => {
      const obj = e.selected[0];
      if (obj && obj.annotationIndex !== undefined) {
        setSelectedAnnotation(obj.annotationIndex);
      }
    });

    canvas.on('selection:cleared', () => {
      setSelectedAnnotation(null);
    });

    // Handle object modification
    canvas.on('object:modified', (e) => {
      updateAnnotationFromObject(e.target);
    });

    // Handle mouse events for drawing
    canvas.on('mouse:down', (e) => {
      if (drawMode === 'rectangle') {
        startDrawingRect(canvas, e);
      } else if (drawMode === 'polygon') {
        addPolygonPoint(canvas, e);
      }
    });

    return () => {
      canvas.dispose();
    };
  }, [templateImage]);

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

  const loadAnnotations = (canvas) => {
    // Remove existing annotation objects
    const objectsToRemove = canvas.getObjects().filter(obj => 
      obj.annotationIndex !== undefined || obj.isLabel
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
          hasControls: true,
          hasBorders: true,
          cornerSize: 8,
          transparentCorners: false,
          cornerColor: color,
          cornerStrokeColor: '#ffffff',
          borderColor: color,
          annotationIndex: index,
          annotationType: ann.type
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
          hasControls: true,
          hasBorders: true,
          cornerSize: 8,
          transparentCorners: false,
          cornerColor: color,
          cornerStrokeColor: '#ffffff',
          borderColor: color,
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
    const color = '#666666'; // Neutral color while drawing
    
    const rect = new fabric.Rect({
      left: pointer.x,
      top: pointer.y,
      width: 0,
      height: 0,
      fill: color + '20',
      stroke: color,
      strokeWidth: 2,
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
        // Store the shape data and show type dialog
        canvas.remove(rect); // Remove temporary shape
        setPendingShape({
          shape: 'rectangle',
          x: rect.left,
          y: rect.top,
          width: rect.width,
          height: rect.height
        });
        setShowTypeDialog(true);
        setDrawMode('select');
      } else {
        canvas.remove(rect);
        setDrawMode('select');
      }
    });
  };

  const addPolygonPoint = (canvas, e) => {
    const pointer = canvas.getPointer(e.e);
    const color = '#666666'; // Neutral color while drawing
    
    const newPoint = { x: pointer.x, y: pointer.y };
    const updatedPoints = [...polygonPoints, newPoint];
    setPolygonPoints(updatedPoints);
    
    // Draw point
    const circle = new fabric.Circle({
      left: pointer.x - 4,
      top: pointer.y - 4,
      radius: 4,
      fill: color,
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
          strokeWidth: 2,
          selectable: false,
          evented: false
        }
      );
      canvas.add(line);
      setTempLines(prev => [...prev, line]);
    }
    
    // Complete polygon after 4 points
    if (updatedPoints.length === 4) {
      // Remove temp objects
      tempLines.forEach(obj => canvas.remove(obj));
      setTempLines([]);
      
      // Store the shape data and show type dialog
      setPendingShape({
        shape: 'polygon',
        points: updatedPoints.map(p => [p.x, p.y])
      });
      setShowTypeDialog(true);
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

  const handleAnnotationTypeChange = (index, newType) => {
    const updated = [...annotations];
    updated[index].type = newType;
    onAnnotationsChange(updated);
    
    // Reload annotations to update colors
    if (fabricCanvasRef.current) {
      loadAnnotations(fabricCanvasRef.current);
    }
  };

  const handleDeleteAnnotation = (index) => {
    const updated = annotations.filter((_, i) => i !== index);
    onAnnotationsChange(updated);
    
    if (fabricCanvasRef.current) {
      loadAnnotations(fabricCanvasRef.current);
    }
    
    if (selectedAnnotation === index) {
      setSelectedAnnotation(null);
    }
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

  const handleTypeDialogConfirm = () => {
    if (!pendingShape) return;

    const newAnnotation = {
      type: selectedType,
      ...pendingShape
    };

    // Add text content if needed
    const typeInfo = ANNOTATION_TYPES.find(t => t.value === selectedType);
    if (typeInfo?.needsText && textContent.trim()) {
      newAnnotation.text = textContent.trim();
    }

    onAnnotationsChange([...annotations, newAnnotation]);
    
    // Reset dialog state
    setShowTypeDialog(false);
    setPendingShape(null);
    setSelectedType('text');
    setTextContent('');
  };

  const handleTypeDialogCancel = () => {
    setShowTypeDialog(false);
    setPendingShape(null);
    setSelectedType('text');
    setTextContent('');
  };

  return (
    <div className="template-editor">
      {/* Type Selection Dialog */}
      {showTypeDialog && (
        <div className="type-dialog-overlay" onClick={handleTypeDialogCancel}>
          <div className="type-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="type-dialog-header">
              <h3>Select Annotation Type</h3>
              <button className="close-btn" onClick={handleTypeDialogCancel}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <line x1="18" y1="6" x2="6" y2="18" stroke="currentColor" strokeWidth="2"/>
                  <line x1="6" y1="6" x2="18" y2="18" stroke="currentColor" strokeWidth="2"/>
                </svg>
              </button>
            </div>
            
            <div className="type-dialog-body">
              <div className="type-options">
                {ANNOTATION_TYPES.map(type => (
                  <label key={type.value} className="type-option">
                    <input
                      type="radio"
                      name="annotationType"
                      value={type.value}
                      checked={selectedType === type.value}
                      onChange={(e) => setSelectedType(e.target.value)}
                    />
                    <span className="type-color" style={{ backgroundColor: type.color }} />
                    <span className="type-label">{type.label}</span>
                  </label>
                ))}
              </div>

              {ANNOTATION_TYPES.find(t => t.value === selectedType)?.needsText && (
                <div className="text-input-section">
                  <label htmlFor="textContent">Text Content:</label>
                  <input
                    id="textContent"
                    type="text"
                    value={textContent}
                    onChange={(e) => setTextContent(e.target.value)}
                    placeholder="Enter text content..."
                    autoFocus
                  />
                </div>
              )}
            </div>
            
            <div className="type-dialog-footer">
              <button className="cancel-btn" onClick={handleTypeDialogCancel}>
                Cancel
              </button>
              <button 
                className="confirm-btn" 
                onClick={handleTypeDialogConfirm}
                disabled={ANNOTATION_TYPES.find(t => t.value === selectedType)?.needsText && !textContent.trim()}
              >
                Add Annotation
              </button>
            </div>
          </div>
        </div>
      )}
      <div className="editor-toolbar">
        <div className="toolbar-group">
          <button
            className={`tool-btn ${drawMode === 'select' ? 'active' : ''}`}
            onClick={() => setDrawMode('select')}
            title="Select & Edit"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M3 3L10.07 19.97L12.58 12.58L19.97 10.07L3 3Z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
            </svg>
          </button>
          <button
            className={`tool-btn ${drawMode === 'rectangle' ? 'active' : ''}`}
            onClick={() => setDrawMode('rectangle')}
            title="Draw Rectangle"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <rect x="3" y="3" width="18" height="18" stroke="currentColor" strokeWidth="2"/>
            </svg>
          </button>
          <button
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
          
          {(drawMode === 'rectangle' || drawMode === 'polygon') && (
            <button className="tool-btn cancel-btn" onClick={handleCancelDraw} title="Cancel Drawing">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <line x1="18" y1="6" x2="6" y2="18" stroke="currentColor" strokeWidth="2"/>
                <line x1="6" y1="6" x2="18" y2="18" stroke="currentColor" strokeWidth="2"/>
              </svg>
            </button>
          )}
        </div>

        <div className="toolbar-group">
          <button className="tool-btn" onClick={handleZoomOut} title="Zoom Out">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <circle cx="11" cy="11" r="8" stroke="currentColor" strokeWidth="2"/>
              <line x1="8" y1="11" x2="14" y2="11" stroke="currentColor" strokeWidth="2"/>
              <line x1="17" y1="17" x2="21" y2="21" stroke="currentColor" strokeWidth="2"/>
            </svg>
          </button>
          <span className="zoom-label">
            {fabricCanvasRef.current ? Math.round(fabricCanvasRef.current.getZoom() * 100) : 100}%
          </span>
          <button className="tool-btn" onClick={handleZoomIn} title="Zoom In">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <circle cx="11" cy="11" r="8" stroke="currentColor" strokeWidth="2"/>
              <line x1="11" y1="8" x2="11" y2="14" stroke="currentColor" strokeWidth="2"/>
              <line x1="8" y1="11" x2="14" y2="11" stroke="currentColor" strokeWidth="2"/>
              <line x1="17" y1="17" x2="21" y2="21" stroke="currentColor" strokeWidth="2"/>
            </svg>
          </button>
          <button className="tool-btn" onClick={handleResetZoom} title="Reset Zoom">
            100%
          </button>
        </div>
      </div>

      <div className="editor-content">
        <div className="canvas-container" ref={containerRef}>
          <canvas ref={canvasRef} />
          {polygonPoints.length > 0 && (
            <div className="polygon-hint">
              Click {4 - polygonPoints.length} more point{4 - polygonPoints.length !== 1 ? 's' : ''} to complete polygon
            </div>
          )}
          <div className="canvas-help">
            <span>💡 Alt + Drag to pan | Scroll to zoom | Drag corners to resize</span>
          </div>
        </div>

        <div className="annotations-panel">
          <h3>Bounding Boxes ({annotations.length})</h3>
          <div className="annotations-list">
            {annotations.map((ann, index) => (
              <div
                key={index}
                className={`annotation-item ${selectedAnnotation === index ? 'selected' : ''}`}
                onClick={() => {
                  setSelectedAnnotation(index);
                  if (fabricCanvasRef.current) {
                    const obj = fabricCanvasRef.current.getObjects().find(
                      o => o.annotationIndex === index
                    );
                    if (obj) {
                      fabricCanvasRef.current.setActiveObject(obj);
                      fabricCanvasRef.current.requestRenderAll();
                    }
                  }
                }}
              >
                <div className="annotation-header">
                  <span
                    className="annotation-color"
                    style={{
                      backgroundColor: ANNOTATION_TYPES.find(t => t.value === ann.type)?.color || '#ffffff'
                    }}
                  />
                  <span className="annotation-index">BBox #{index + 1}</span>
                </div>
                
                <select
                  value={ann.type}
                  onChange={(e) => handleAnnotationTypeChange(index, e.target.value)}
                  className="annotation-type-select"
                  onClick={(e) => e.stopPropagation()}
                >
                  {ANNOTATION_TYPES.map(type => (
                    <option key={type.value} value={type.value}>
                      {type.label}
                    </option>
                  ))}
                </select>

                <div className="annotation-info">
                  <span className="info-label">Shape:</span>
                  <span className="info-value">{ann.shape}</span>
                </div>

                {ann.shape === 'rectangle' && (
                  <div className="annotation-info">
                    <span className="info-label">Size:</span>
                    <span className="info-value">
                      {Math.round(ann.width)} × {Math.round(ann.height)}
                    </span>
                  </div>
                )}

                {ann.shape === 'polygon' && ann.points && (
                  <div className="annotation-info">
                    <span className="info-label">Points:</span>
                    <span className="info-value">{ann.points.length}</span>
                  </div>
                )}

                {ann.text && (
                  <div className="annotation-info">
                    <span className="info-label">Text:</span>
                    <span className="info-value" style={{ fontStyle: 'italic' }}>{ann.text}</span>
                  </div>
                )}

                <button
                  className="delete-annotation-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDeleteAnnotation(index);
                  }}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                    <path d="M3 6H5H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                    <path d="M8 6V4C8 3.46957 8.21071 2.96086 8.58579 2.58579C8.96086 2.21071 9.46957 2 10 2H14C14.5304 2 15.0391 2.21071 15.4142 2.58579C15.7893 2.96086 16 3.46957 16 4V6M19 6V20C19 20.5304 18.7893 21.0391 18.4142 21.4142C18.0391 21.7893 17.5304 22 17 22H7C6.46957 22 5.96086 21.7893 5.58579 21.4142C5.21071 21.0391 5 20.5304 5 20V6H19Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </button>
              </div>
            ))}

            {annotations.length === 0 && (
              <div className="empty-message">
                No annotations yet. Use the tools above to draw bounding boxes.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
