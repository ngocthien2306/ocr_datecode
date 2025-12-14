import React, { useState, useRef, useEffect } from 'react';
import { Canvas, FabricImage, Control, Point, util } from 'fabric';
import { TYPE_CONFIGS } from '../fabric/types';
import * as objectUtils from '../fabric/utils/objectUtils';
import * as canvasActions from '../fabric/actions/canvasActions';
import { PolygonDrawer, startDrawingRectangle as startRectDraw } from '../fabric/utils/drawingUtils';
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
  const isTransformingRef = useRef(false);
  
  const [drawMode, setDrawMode] = useState('select');
  const polygonDrawerRef = useRef(null);
  const rectangleCleanupRef = useRef(null);
  const [showHints, setShowHints] = useState(true);
  const [isSpacePressed, setIsSpacePressed] = useState(false);

  // Initialize canvas
  useEffect(() => {
    if (!canvasRef.current || !containerRef.current) return;

    const canvas = new Canvas(canvasRef.current, {
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight,
      backgroundColor: '#1e1e1e',
      selection: drawMode === 'select'
    });

    fabricCanvasRef.current = canvas;

    // Setup panning
    objectUtils.setupCanvasPanning(canvas);

    // Load background image
    if (templateImage) {
      FabricImage.fromURL(templateImage, 
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

    // Handle selection
    canvas.on('selection:created', (e) => {
      const obj = e.selected[0];
      if (obj && !obj.isTemp && obj.annotationIndex !== undefined) {
        onSelectAnnotation?.(obj.annotationIndex);
      }
    });

    canvas.on('selection:updated', (e) => {
      const obj = e.selected[0];
      if (obj && !obj.isTemp && obj.annotationIndex !== undefined) {
        onSelectAnnotation?.(obj.annotationIndex);
      }
    });

    canvas.on('selection:cleared', () => {
      onSelectAnnotation?.(null);
    });

    // Handle object modification
    canvas.on('object:modified', (e) => {
      // Skip temp objects and objects without annotation index
      if (e.target.isTemp || e.target.annotationIndex === undefined) {
        return;
      }
      
      console.log('object:modified - updating and will set isTransforming to false');
      updateAnnotationFromObject(e.target);
      updateLabelPosition(canvas, e.target);
      
      // Set to false AFTER update to allow useEffect to process the final state
      setTimeout(() => {
        console.log('Setting isTransforming to false (delayed)');
        isTransformingRef.current = false;
      }, 0);
    });
    
    // Update annotation data and label while moving (real-time sync)
    canvas.on('object:moving', (e) => {
      if (e.target.annotationIndex !== undefined) {
        console.log('object:moving - setting isTransforming to true');
        isTransformingRef.current = true;
        updateAnnotationFromObject(e.target);
        updateLabelPosition(canvas, e.target);
      }
    });
    
    // Update annotation data and label while scaling (for rectangles)
    canvas.on('object:scaling', (e) => {
      if (e.target.annotationIndex !== undefined) {
        console.log('object:scaling - setting isTransforming to true');
        isTransformingRef.current = true;
        updateAnnotationFromObject(e.target);
        updateLabelPosition(canvas, e.target);
      }
    });

    return () => {
      canvas.dispose();
    };
  }, [templateImage]);

  // Update selection mode when drawMode changes
  useEffect(() => {
    const canvas = fabricCanvasRef.current;
    if (!canvas) return;

    canvas.selection = drawMode === 'select';
    canvas.forEachObject((obj) => {
      // Annotation objects only selectable in select mode
      if (obj.annotationIndex !== undefined) {
        obj.selectable = drawMode === 'select';
        obj.evented = drawMode === 'select';
      }
    });
    canvas.requestRenderAll();
  }, [drawMode]);

  // Reload annotations when they change
  useEffect(() => {
    if (fabricCanvasRef.current && annotations) {
      const canvas = fabricCanvasRef.current;
      
      console.log('useEffect triggered:', {
        isTransforming: isTransformingRef.current,
        annotationsCount: annotations.length
      });
      
      // Skip reload during active transform (moving/scaling)
      if (isTransformingRef.current) {
        console.log('Skipping reload - transform in progress');
        return;
      }
      
      // Smart update: check if we can just update properties instead of full reload
      // Filter out labels and temp objects - only count actual shapes
      const existingObjects = canvas.getObjects().filter(obj => 
        obj.annotationIndex !== undefined && !obj.isLabel && !obj.isTemp
      );
      
      console.log('Existing objects:', existingObjects.length, 'Annotations:', annotations.length);
      
      // Handle count mismatch - add new or remove deleted annotations
      if (existingObjects.length !== annotations.length) {
        if (existingObjects.length < annotations.length) {
          // New annotations added - only add the new ones
          console.log('Adding new annotations');
          for (let i = existingObjects.length; i < annotations.length; i++) {
            const ann = annotations[i];
            const typeConfig = TYPE_CONFIGS.find(t => t.value === ann.type) || TYPE_CONFIGS[0];
            const color = typeConfig.color;
            
            console.log('Creating new object:', { index: i, type: ann.type, color, shape: ann.shape });
            
            let obj;
            if (ann.shape === 'rectangle') {
              obj = objectUtils.createRectangleObject(ann, i, color);
            } else if (ann.shape === 'polygon') {
              obj = objectUtils.createPolygonObject(ann, i, color);
              setupPolygonControls(obj, color);
            }
            
            if (obj) {
              canvas.add(obj);
              const label = objectUtils.createLabel(ann.type, ann.x || ann.points[0][0], ann.y || ann.points[0][1], color, i);
              canvas.add(label);
            }
          }
          canvas.requestRenderAll();
        } else {
          // Annotations removed - remove objects with higher indices
          console.log('Removing deleted annotations');
          const objectsToRemove = existingObjects.filter(obj => obj.annotationIndex >= annotations.length);
          objectsToRemove.forEach(obj => {
            // Remove associated label
            const label = canvas.getObjects().find(o => 
              o.isLabel && o.annotationIndex === obj.annotationIndex
            );
            if (label) canvas.remove(label);
            canvas.remove(obj);
          });
          canvas.requestRenderAll();
        }
        return;
      }
      
      // Update existing objects
      let needsFullReload = false;
      annotations.forEach((ann, index) => {
        const obj = existingObjects.find(o => o.annotationIndex === index);
        
        if (!obj) {
          needsFullReload = true;
          return;
        }
        
        // Check if shape type changed
        const objShape = obj.type === 'rect' ? 'rectangle' : 'polygon';
        if (objShape !== ann.shape) {
          needsFullReload = true;
          return;
        }
        
        // Update type and color if changed
        const typeConfig = TYPE_CONFIGS.find(t => t.value === ann.type) || TYPE_CONFIGS[0];
        const newColor = typeConfig.color;
        
        if (obj.annotationType !== ann.type) {
          obj.annotationType = ann.type;
          obj.set({
            stroke: newColor,
            fill: newColor + '20',
            cornerColor: newColor,
            borderColor: newColor
          });
          
          // Update label
          const label = canvas.getObjects().find(o => 
            o.isLabel && o.annotationIndex === index
          );
          if (label) {
            label.set({
              text: ann.type.toUpperCase(),
              fill: newColor
            });
          }
        }
      });
      
      if (needsFullReload) {
        loadAnnotations(canvas);
      } else {
        canvas.requestRenderAll();
      }
    }
  }, [annotations]);


  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      
      if (e.code === 'Space' && !isSpacePressed) {
        e.preventDefault();
        setIsSpacePressed(true);
        if (fabricCanvasRef.current) {
          fabricCanvasRef.current.isSpacePressed = true;
          fabricCanvasRef.current.defaultCursor = 'grab';
          fabricCanvasRef.current.renderAll();
        }
        return;
      }
      
      switch(e.key.toLowerCase()) {
        case 'v':
          handleStopDrawing();
          break;
        case 'r':
          startDrawingRectangle();
          break;
        case 'p':
          startDrawingPolygon();
          break;
        case 'escape':
          handleCancelDraw();
          break;
        case 'delete':
        case 'backspace':
          e.preventDefault();
          if (selectedAnnotation !== null) {
            handleDeleteAnnotation(selectedAnnotation);
          }
          break;
        case 'h':
          setShowHints(prev => !prev);
          break;
        default:
          break;
      }
    };

    const handleKeyUp = (e) => {
      if (e.code === 'Space' && isSpacePressed) {
        e.preventDefault();
        setIsSpacePressed(false);
        if (fabricCanvasRef.current) {
          fabricCanvasRef.current.isSpacePressed = false;
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
  }, [selectedAnnotation, drawMode, isSpacePressed]);

  const loadAnnotations = (canvas) => {
    // Remove existing annotation objects
    objectUtils.removeObjectsByPredicate(canvas, obj =>
      obj.annotationIndex !== undefined || obj.isLabel
    );

    annotations.forEach((ann, index) => {
      const typeConfig = TYPE_CONFIGS.find(t => t.value === ann.type) || TYPE_CONFIGS[0];
      const color = typeConfig.color;
      
      let obj;
      if (ann.shape === 'rectangle') {
        obj = objectUtils.createRectangleObject(ann, index, color);
      } else if (ann.shape === 'polygon' && ann.points) {
        obj = objectUtils.createPolygonObject(ann, index, color);
        // Add custom controls for polygon point editing
        setupPolygonControls(obj, color);
      }
      
      if (obj) {
        canvas.add(obj);
        const label = objectUtils.createLabel(ann.type, ann.x || ann.points[0][0], ann.y || ann.points[0][1], color, index);
        canvas.add(label);
      }
    });
    
    canvas.requestRenderAll();
  };

  const setupPolygonControls = (polygon, color) => {
    // Clear existing controls except default ones
    polygon.controls = {}; 
    
    // Add control for each point
    polygon.points.forEach((point, idx) => {
      polygon.controls['p' + idx] = new Control({
        positionHandler: function(dim, finalMatrix, fabricObject) {
          const x = fabricObject.points[idx].x - fabricObject.pathOffset.x;
          const y = fabricObject.points[idx].y - fabricObject.pathOffset.y;
          return util.transformPoint(
            new Point(x, y),
            util.multiplyTransformMatrices(
              fabricObject.canvas.viewportTransform,
              fabricObject.calcTransformMatrix()
            )
          );
        },
        actionHandler: function(eventData, transform, x, y) {
          const poly = transform.target;
          
          // Calculate mouse position relative to polygon center
          const center = poly.getCenterPoint();
          const mouseX = x - center.x;
          const mouseY = y - center.y;
          
          // Apply inverse transformation
          const angle = -poly.angle * Math.PI / 180;
          const cos = Math.cos(angle);
          const sin = Math.sin(angle);
          const localX = (mouseX * cos - mouseY * sin) / (poly.scaleX || 1);
          const localY = (mouseX * sin + mouseY * cos) / (poly.scaleY || 1);
          
          // Calculate final point position
          const polygonBaseSize = new Point(
            poly.width + (poly.strokeWidth || 0),
            poly.height + (poly.strokeWidth || 0)
          );
          const size = poly._getTransformedDimensions(0, 0);
          const finalPointPosition = {
            x: (localX * polygonBaseSize.x) / size.x + poly.pathOffset.x,
            y: (localY * polygonBaseSize.y) / size.y + poly.pathOffset.y
          };
          
          poly.points[idx] = new Point(finalPointPosition.x, finalPointPosition.y);
          return true;
        },
        actionName: 'modifyPolygon',
        cursorStyle: 'pointer',
        render: function(ctx, left, top, styleOverride, fabricObject) {
          ctx.save();
          ctx.fillStyle = color;
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.arc(left, top, 8, 0, 2 * Math.PI);
          ctx.fill();
          ctx.stroke();
          ctx.restore();
        }
      });
    });
  };

  const updateLabelPosition = (canvas, obj) => {
    const label = canvas.getObjects().find(o => 
      o.isLabel && o.annotationIndex === obj.annotationIndex
    );
    
    if (label) {
      let x, y;
      if (obj.type === 'rect') {
        x = obj.left;
        y = obj.top;
      } else if (obj.type === 'polygon' && obj.points && obj.points.length > 0) {
        // Get transformed first point
        const matrix = obj.calcTransformMatrix();
        const point = util.transformPoint(
          new Point(
            obj.points[0].x - obj.pathOffset.x,
            obj.points[0].y - obj.pathOffset.y
          ),
          matrix
        );
        x = point.x;
        y = point.y;
      }
      
      if (x !== undefined && y !== undefined) {
        label.set({
          left: x,
          top: y - 25
        });
        label.setCoords();
      }
    }
  };

  const updateAnnotationFromObject = (obj) => {
    if (obj.annotationIndex === undefined) return;
    
    console.log('updateAnnotationFromObject called for index:', obj.annotationIndex, 'isTransforming:', isTransformingRef.current);
    
    const updated = [...annotations];
    const ann = updated[obj.annotationIndex];
    
    if (!ann) return;
    
    if (ann.shape === 'rectangle') {
      const data = objectUtils.getRectangleData(obj);
      console.log('Updating rectangle:', data);
      Object.assign(ann, data);
    } else if (ann.shape === 'polygon') {
      const data = objectUtils.getPolygonData(obj);
      console.log('Updating polygon:', data);
      Object.assign(ann, data);
    }
    
    onAnnotationsChange(updated);
  };

  const startDrawingRectangle = () => {
    setDrawMode('rectangle');
    const canvas = fabricCanvasRef.current;
    if (!canvas) return;

    const typeConfig = TYPE_CONFIGS.find(t => t.value === 'text') || TYPE_CONFIGS[0];
    
    // Wait for next mouse down to start drawing
    const handleMouseDown = (e) => {
      if (!e.target || e.target === canvas.backgroundImage) {
        rectangleCleanupRef.current = startRectDraw(
          canvas,
          e.e,
          typeConfig.color,
          (rectData) => {
            addAnnotation(rectData);
            setDrawMode('select');
            rectangleCleanupRef.current = null;
          }
        );
        canvas.off('mouse:down', handleMouseDown);
      }
    };
    
    canvas.on('mouse:down', handleMouseDown);
  };

  const startDrawingPolygon = () => {
    setDrawMode('polygon');
    const canvas = fabricCanvasRef.current;
    if (!canvas) return;

    const typeConfig = TYPE_CONFIGS.find(t => t.value === 'text') || TYPE_CONFIGS[0];
    
    polygonDrawerRef.current = new PolygonDrawer(
      canvas,
      typeConfig.color,
      (polygonData) => {
        addAnnotation(polygonData);
        setDrawMode('select');
        polygonDrawerRef.current = null;
      }
    );
  };

  const handleStopDrawing = () => {
    if (polygonDrawerRef.current) {
      polygonDrawerRef.current.cancel();
      polygonDrawerRef.current = null;
    }
    if (rectangleCleanupRef.current) {
      rectangleCleanupRef.current();
      rectangleCleanupRef.current = null;
    }
    setDrawMode('select');
  };

  const handleCancelDraw = () => {
    handleStopDrawing();
    if (fabricCanvasRef.current) {
      fabricCanvasRef.current.discardActiveObject();
      fabricCanvasRef.current.requestRenderAll();
    }
    onSelectAnnotation?.(null);
  };

  const addAnnotation = (shapeData, type = 'text') => {
    const newAnnotation = {
      id: `annotation-${Date.now()}`,
      type,
      text: '',
      ...shapeData
    };

    const updatedAnnotations = [...annotations, newAnnotation];
    onAnnotationsChange(updatedAnnotations);
    onSelectAnnotation(updatedAnnotations.length - 1);
  };

  const handleDeleteAnnotation = (index) => {
    const canvas = fabricCanvasRef.current;
    if (!canvas) return;
    
    canvasActions.deleteByIndex(canvas, index);
    
    const updated = annotations.filter((_, i) => i !== index);
    onAnnotationsChange(updated);
    onSelectAnnotation(null);
  };

  const handleZoomIn = () => {
    const canvas = fabricCanvasRef.current;
    if (canvas) {
      const zoom = canvas.getZoom();
      objectUtils.setCanvasZoom(canvas, zoom * 1.2);
    }
  };

  const handleZoomOut = () => {
    const canvas = fabricCanvasRef.current;
    if (canvas) {
      const zoom = canvas.getZoom();
      objectUtils.setCanvasZoom(canvas, zoom / 1.2);
    }
  };

  const handleResetZoom = () => {
    objectUtils.resetCanvasZoom(fabricCanvasRef.current);
  };

  // Handle mouse down for drawing
  useEffect(() => {
    const canvas = fabricCanvasRef.current;
    if (!canvas) return;

    const handleMouseDown = (e) => {
      // Only add polygon point when clicking on canvas background (not on existing objects)
      if (drawMode === 'polygon' && polygonDrawerRef.current) {
        if (!e.target || e.target === canvas.backgroundImage) {
          polygonDrawerRef.current.addPoint(e.e);
        }
      } else if (drawMode === 'select' && !e.target) {
        canvas.discardActiveObject();
        canvas.requestRenderAll();
        onSelectAnnotation?.(null);
      }
    };

    canvas.on('mouse:down', handleMouseDown);
    return () => {
      canvas.off('mouse:down', handleMouseDown);
    };
  }, [drawMode, polygonDrawerRef.current]);

  return (
    <div className="template-editor">
      <div className="editor-toolbar">
        <div className="toolbar-group">
          <button
            type="button"
            className={`tool-btn ${drawMode === 'select' ? 'active' : ''}`}
            onClick={handleStopDrawing}
            title="Select & Edit"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M3 3L10.07 19.97L12.58 12.58L19.97 10.07L3 3Z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
            </svg>
          </button>
          <button
            type="button"
            className={`tool-btn ${drawMode === 'rectangle' ? 'active' : ''}`}
            onClick={startDrawingRectangle}
            title="Draw Rectangle (R)"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <rect x="3" y="3" width="18" height="18" stroke="currentColor" strokeWidth="2" fill="none"/>
            </svg>
          </button>

          <button
            type="button"
            className={`tool-btn ${drawMode === 'polygon' ? 'active' : ''}`}
            onClick={startDrawingPolygon}
            title="Draw Polygon (4 points)"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M12 2L22 8.5L18 20.5H6L2 8.5L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
            </svg>
          </button>
          
          {(drawMode === 'polygon' || drawMode === 'rectangle') && (
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
                  <li><kbd>Mouse Wheel</kbd> - Zoom in/out</li>
                  <li><kbd>Trackpad Pinch</kbd> - Zoom in/out</li>
                </ul>
              </div>
              
              <div className="cvat-canvas-hints-block">
                <strong>⌨️ Shortcuts</strong>
                <ul>
                  <li><kbd>V</kbd> - Select mode</li>
                  <li><kbd>R</kbd> - Rectangle tool</li>
                  <li><kbd>P</kbd> - Polygon tool (4 points)</li>
                  <li><kbd>Esc</kbd> - Cancel drawing</li>
                  <li><kbd>Del</kbd> - Delete selected</li>
                </ul>
              </div>

              <div className="cvat-canvas-hints-block">
                <strong>✏️ Drawing</strong>
                <ul>
                  <li>Rectangle: Click and drag</li>
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
          
          {drawMode === 'rectangle' && (
            <div className="polygon-hint">
              Click and drag to draw rectangle
            </div>
          )}
          
          {drawMode === 'polygon' && polygonDrawerRef.current && (
            <div className="polygon-hint">
              Click {polygonDrawerRef.current.getRemainingPoints()} more point{polygonDrawerRef.current.getRemainingPoints() !== 1 ? 's' : ''} to complete polygon
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
