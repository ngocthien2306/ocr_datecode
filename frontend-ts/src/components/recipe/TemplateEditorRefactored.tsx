// Duplicate type imports removed

import React, { useState, useRef, useEffect, useMemo } from 'react';
import type { Canvas as FabricCanvas, Object as FabricObject } from 'fabric';
import { Canvas, FabricImage, Control, Point, util } from 'fabric';
import { TYPE_CONFIGS } from '@/fabric/types';
import * as objectUtils from '@/fabric/utils/objectUtils';
import * as canvasActions from '@/fabric/actions/canvasActions';
import { PolygonDrawer, startDrawingRectangle as startRectDraw } from '@/fabric/utils/drawingUtils';
import '@/styles/TemplateEditor.css';
import type { Annotation } from '@/types';

// Extend fabric object types for annotation properties
type FabricAnnotationObject = FabricObject & {
  annotationIndex?: number;
  annotationType?: string;
  isLabel?: boolean;
  isTemp?: boolean;
  isPolygonEditPoint?: boolean;
  name?: string;
  points?: any[];
  _currentTransform?: any;
  pathOffset?: { x: number; y: number };
  [key: string]: any;
};

type CanvasWithImageBounds = FabricCanvas & {
  imageBounds?: {
    left: number;
    top: number;
    right: number;
    bottom: number;
    width: number;
    height: number;
  };
};

interface TemplateEditorProps {
  templateImage: string | null;
  annotations: Annotation[];
  onAnnotationsChange: (annotations: Annotation[]) => void;
  selectedAnnotation: number | null;
  onSelectAnnotation: (index: number | null) => void;
  fabricCanvasRef?: React.MutableRefObject<FabricCanvas | null>;
}

export default function TemplateEditor({ 
  templateImage, 
  annotations, 
  onAnnotationsChange, 
  selectedAnnotation, 
  onSelectAnnotation,
  fabricCanvasRef: externalCanvasRef 
}: TemplateEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const internalCanvasRef = useRef<CanvasWithImageBounds | null>(null);
  const fabricCanvasRef = (externalCanvasRef as React.MutableRefObject<CanvasWithImageBounds | null>) || internalCanvasRef;
  const containerRef = useRef<HTMLDivElement | null>(null);
  const isTransformingRef = useRef(false);

  // Refs to always access latest props in event handlers (avoid stale closures)
  const annotationsRef = useRef(annotations);
  annotationsRef.current = annotations;
  const onAnnotationsChangeRef = useRef(onAnnotationsChange);
  onAnnotationsChangeRef.current = onAnnotationsChange;
  const onSelectAnnotationRef = useRef(onSelectAnnotation);
  onSelectAnnotationRef.current = onSelectAnnotation;

  const [drawMode, setDrawMode] = useState('select');
  const polygonDrawerRef = useRef<any>(null);
  const polygonCleanupRef = useRef<(() => void) | null>(null);
  const rectangleCleanupRef = useRef<(() => void) | null>(null);
  const [showHints, setShowHints] = useState(false);
  const [showLabels, setShowLabels] = useState(false);
  const [isSpacePressed, setIsSpacePressed] = useState(false);

  // Resize-chars panel state
  const [resizeDelta, setResizeDelta] = useState<number>(2);
  const [resizeScope, setResizeScope] = useState<string>('all');  // 'all' | `region:${idx}` | `selected:${idx}` | 'unassigned'
  const [resizeAxis,  setResizeAxis]  = useState<'both' | 'w' | 'h'>('both');

  // Initialize canvas
  useEffect(() => {
    if (!canvasRef.current || !containerRef.current) return;


    const canvas = new Canvas(canvasRef.current as HTMLCanvasElement, {
      width: containerRef.current!.clientWidth,
      height: containerRef.current!.clientHeight,
      backgroundColor: '#1e1e1e',
      selection: drawMode === 'select',
    }) as CanvasWithImageBounds;

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
        const imgLeft = (canvas.width - img.width * scale) / 2;
        const imgTop = (canvas.height - img.height * scale) / 2;
        
        img.set({
          left: imgLeft,
          top: imgTop,
          selectable: false,
          evented: false,
          hasControls: false,
          lockMovementX: true,
          lockMovementY: true
        });
        
        // Store image boundaries for constraining annotations
        (canvas as CanvasWithImageBounds).imageBounds = {
          left: imgLeft,
          top: imgTop,
          right: imgLeft + img.width * scale,
          bottom: imgTop + img.height * scale,
          width: img.width * scale,
          height: img.height * scale
        };
        
        canvas.backgroundImage = img;
        canvas.requestRenderAll();
        loadAnnotations(canvas);
      }).catch(err => {
        console.error('Error loading image:', err);
      });
    }

    // Handle selection - use refs to avoid stale closures
    canvas.on('selection:created', (e) => {
      const obj = e.selected[0] as FabricAnnotationObject;
      if (obj && !obj.isTemp && obj.annotationIndex !== undefined) {
        onSelectAnnotationRef.current?.(obj.annotationIndex);
      }
    });

    canvas.on('selection:updated', (e) => {
      const obj = e.selected[0] as FabricAnnotationObject;
      if (obj && !obj.isTemp && obj.annotationIndex !== undefined) {
        onSelectAnnotationRef.current?.(obj.annotationIndex);
      }
    });

    canvas.on('selection:cleared', () => {
      onSelectAnnotationRef.current?.(null);
    });

    // Handle object modification
    canvas.on('object:modified', (e) => {
      // Skip temp objects and objects without annotation index
      const target = e.target as FabricAnnotationObject;
      if (target.isTemp || target.annotationIndex === undefined) {
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
      const target = e.target as FabricAnnotationObject;
      if (target.annotationIndex !== undefined) {
        console.log('object:moving - setting isTransforming to true');
        isTransformingRef.current = true;
        // Constrain object within image bounds
        const c = canvas as CanvasWithImageBounds;
        if (c.imageBounds) {
          const obj = target;
          const bounds = c.imageBounds;
          // Get current bounding rect of the object
          obj.setCoords();
          const objBounds = obj.getBoundingRect();
          // Calculate constrained position
          let newLeft = obj.left;
          let newTop = obj.top;
          // Constrain horizontally
          if (objBounds.left < bounds.left) {
            newLeft = obj.left + (bounds.left - objBounds.left);
          } else if (objBounds.left + objBounds.width > bounds.right) {
            newLeft = obj.left - ((objBounds.left + objBounds.width) - bounds.right);
          }
          // Constrain vertically
          if (objBounds.top < bounds.top) {
            newTop = obj.top + (bounds.top - objBounds.top);
          } else if (objBounds.top + objBounds.height > bounds.bottom) {
            newTop = obj.top - ((objBounds.top + objBounds.height) - bounds.bottom);
          }
          // Apply constrained position
          obj.set({
            left: newLeft,
            top: newTop
          });
          obj.setCoords();
        }
        updateAnnotationFromObject(target);
        updateLabelPosition(canvas, target);
      }
    });
    
    // Update annotation data and label while scaling (for rectangles)
    canvas.on('object:scaling', (e) => {
      const target = e.target as FabricAnnotationObject;
      if (target.annotationIndex !== undefined) {
        console.log('object:scaling - setting isTransforming to true');
        isTransformingRef.current = true;
        
        // Constrain scaled object within image bounds
        if (canvas.imageBounds) {
          const obj = e.target;
          const bounds = obj.getBoundingRect();
          const imgBounds = canvas.imageBounds;
          
          // Prevent scaling beyond image boundaries
          if (bounds.left < imgBounds.left || 
              bounds.top < imgBounds.top ||
              bounds.left + bounds.width > imgBounds.right ||
              bounds.top + bounds.height > imgBounds.bottom) {
            
            // Revert scale if it exceeds bounds
            const transform = (obj as FabricAnnotationObject)._currentTransform;
            if (transform && transform.original) {
              obj.scaleX = transform.original.scaleX;
              obj.scaleY = transform.original.scaleY;
              obj.left = transform.original.left;
              obj.top = transform.original.top;
            }
            obj.setCoords();
          }
        }
        
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

    // In select mode, disable selection and enable panning
    canvas.selection = false;
    canvas.forEachObject((obj) => {
      const objAnn = obj as FabricAnnotationObject;
      if (objAnn.annotationIndex !== undefined) {
        // Objects are not selectable in select mode (it's a pan mode)
        obj.selectable = false;
        obj.evented = false;
      }
    });
    canvas.requestRenderAll();
  }, [drawMode]);

  // Setup panning for select mode
  useEffect(() => {
    const canvas = fabricCanvasRef.current;
    if (!canvas) return;

    if (drawMode === 'select') {
      let isDragging = false;
      let lastPosX = 0;
      let lastPosY = 0;

      // Mouse events
      const handleMouseDown = function(this: any, opt: any) {
        const evt = opt.e;
        // Only left button (button 0) without clicking on an object
        if (evt.button === 0 && !opt.target) {
          isDragging = true;
          this.selection = false;
          lastPosX = evt.clientX;
          lastPosY = evt.clientY;
          evt.preventDefault();
        }
      };

      const handleMouseMove = function(this: any, opt: any) {
        if (isDragging) {
          const e = opt.e;
          const vpt = this.viewportTransform;
          vpt[4] += e.clientX - lastPosX;
          vpt[5] += e.clientY - lastPosY;
          this.requestRenderAll();
          lastPosX = e.clientX;
          lastPosY = e.clientY;
        }
      };

      const handleMouseUp = function(this: any) {
        if (isDragging) {
          this.setViewportTransform(this.viewportTransform);
          isDragging = false;
        }
      };

      const handleMouseLeave = function(this: any) {
        if (isDragging) {
          this.setViewportTransform(this.viewportTransform);
          isDragging = false;
        }
      };

      // Touch events
      const handleTouchStart = (e: TouchEvent) => {
        if (e.touches.length === 1) {
          const touch = e.touches[0]!;
          isDragging = true;
          lastPosX = touch.clientX;
          lastPosY = touch.clientY;
          e.preventDefault();
        }
      };

      const handleTouchMove = (e: TouchEvent) => {
        if (isDragging && e.touches.length === 1) {
          const touch = e.touches[0]!;
          const vpt = canvas.viewportTransform;
          vpt[4] += touch.clientX - lastPosX;
          vpt[5] += touch.clientY - lastPosY;
          canvas.requestRenderAll();
          lastPosX = touch.clientX;
          lastPosY = touch.clientY;
          e.preventDefault();
        }
      };

      const handleTouchEnd = () => {
        if (isDragging) {
          canvas.setViewportTransform(canvas.viewportTransform);
          isDragging = false;
        }
      };

      // Register mouse events
      canvas.on('mouse:down', handleMouseDown);
      canvas.on('mouse:move', handleMouseMove);
      canvas.on('mouse:up', handleMouseUp);

      // Register touch events on canvas element
      const canvasElement = canvas.getElement();
      if (canvasElement) {
        canvasElement.addEventListener('touchstart', handleTouchStart, { passive: false });
        canvasElement.addEventListener('touchmove', handleTouchMove, { passive: false });
        canvasElement.addEventListener('touchend', handleTouchEnd);
        canvasElement.addEventListener('mouseleave', handleMouseLeave);
      }

      return () => {
        canvas.off('mouse:down', handleMouseDown);
        canvas.off('mouse:move', handleMouseMove);
        canvas.off('mouse:up', handleMouseUp);

        if (canvasElement) {
          canvasElement.removeEventListener('touchstart', handleTouchStart);
          canvasElement.removeEventListener('touchmove', handleTouchMove);
          canvasElement.removeEventListener('touchend', handleTouchEnd);
          canvasElement.removeEventListener('mouseleave', handleMouseLeave);
        }
      };
    }
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
        (obj as FabricAnnotationObject).annotationIndex !== undefined && !(obj as FabricAnnotationObject).isLabel && !(obj as FabricAnnotationObject).isTemp
      );
      
      console.log('Existing objects:', existingObjects.length, 'Annotations:', annotations.length);
      
      // Debug: Log all objects on canvas
      const allObjects = canvas.getObjects();
      console.log('All canvas objects:', allObjects.map(o => ({
        type: o.type,
        annotationIndex: (o as FabricAnnotationObject).annotationIndex,
        isLabel: (o as FabricAnnotationObject).isLabel,
        isTemp: (o as FabricAnnotationObject).isTemp,
        name: (o as FabricAnnotationObject).name
      })));
      
      // Validate index integrity: check for gaps or orphaned objects
      const objectIndices = existingObjects.map(o => (o as FabricAnnotationObject).annotationIndex).sort((a, b) => a! - b!);
      const expectedIndices = annotations.map((_, i) => i);
      const hasIndexMismatch = JSON.stringify(objectIndices) !== JSON.stringify(expectedIndices);

      if (hasIndexMismatch) {
        console.warn('Index mismatch detected. Expected:', expectedIndices, 'Got:', objectIndices);
        console.log('Forcing full reload to fix index sync');
        loadAnnotations(canvas);
        return;
      }

      // Handle count mismatch - always full reload to keep indices in sync
      // This handles both insert (splice) and delete (filter) correctly
      if (existingObjects.length !== annotations.length) {
        console.log('Count mismatch detected, full reload:', existingObjects.length, '→', annotations.length);
        loadAnnotations(canvas);
        return;
      }
      
      // Update existing objects
      let needsFullReload = false;
      annotations.forEach((ann, index) => {
        // Find ALL objects with this index (in case of duplicates)
        const objectsWithIndex = existingObjects.filter(o => (o as FabricAnnotationObject).annotationIndex === index);
        
        if (objectsWithIndex.length === 0) {
          needsFullReload = true;
          return;
        }
        
        // If there are duplicates, remove all and trigger full reload
        if (objectsWithIndex.length > 1) {
          console.warn(`Found ${objectsWithIndex.length} duplicate objects for index ${index}, removing all`);
          objectsWithIndex.forEach(obj => {
            const label = canvas.getObjects().find(o => 
              (o as FabricAnnotationObject).isLabel && (o as FabricAnnotationObject).annotationIndex === index
            );
            if (label) canvas.remove(label);
            canvas.remove(obj);
          });
          needsFullReload = true;
          return;
        }
        
        const obj = objectsWithIndex[0];
        
        // Check if shape type changed
        const objShape = obj && obj.type === 'rect' ? 'rectangle' : 'polygon';
        if (obj && objShape !== (ann as any).shape) {
          needsFullReload = true;
          return;
        }
        
        // Update type and color if changed
        let typeConfig = TYPE_CONFIGS.find(t => t.value === ann.type);
        if (!typeConfig && TYPE_CONFIGS.length > 0) typeConfig = TYPE_CONFIGS[0];
        const newColor = typeConfig ? typeConfig.color : '#fff';
        
        if (obj && (obj as FabricAnnotationObject).annotationType !== ann.type) {
          (obj as FabricAnnotationObject).annotationType = ann.type;
          obj.set({
            stroke: newColor,
            fill: newColor + '20',
            cornerColor: newColor,
            borderColor: newColor
          });
          
          // Update label with new type and text
          updateLabelText(canvas, index, ann, typeConfig || { color: '#fff' });
        } else {
          // Update label text even if type didn't change (text might have changed)
          updateLabelText(canvas, index, ann, typeConfig || { color: '#fff' });
        }
      });
      
      if (needsFullReload) {
        loadAnnotations(canvas);
      } else {
        canvas.requestRenderAll();
      }
    }
  }, [annotations]);

  // Toggle label visibility
  useEffect(() => {
    const canvas = fabricCanvasRef.current;
    if (!canvas) return;
    canvas.getObjects().forEach(obj => {
      if ((obj as FabricAnnotationObject).isLabel) {
        obj.visible = showLabels;
      }
    });
    canvas.requestRenderAll();
  }, [showLabels, annotations]);

  // Dim non-selected annotations when one is focused
  useEffect(() => {
    const canvas = fabricCanvasRef.current;
    if (!canvas) return;
    canvas.getObjects().forEach(obj => {
      const fObj = obj as FabricAnnotationObject;
      if (fObj.annotationIndex === undefined) return;
      if (selectedAnnotation !== null && fObj.annotationIndex !== selectedAnnotation) {
        obj.set({ opacity: 0.25 });
      } else {
        obj.set({ opacity: 1 });
      }
      // Also dim/show associated labels
      if (fObj.isLabel) {
        const labelIdx = fObj.annotationIndex;
        if (selectedAnnotation !== null && labelIdx !== selectedAnnotation) {
          obj.set({ opacity: 0.15 });
        } else {
          obj.set({ opacity: 1 });
        }
      }
    });
    canvas.requestRenderAll();
  }, [selectedAnnotation, annotations]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // @ts-ignore - target typing from KeyboardEvent
      if ((e as any).target && ((e as any).target.tagName === 'INPUT' || (e as any).target.tagName === 'TEXTAREA')) return;
      if (e.code === 'Space' && !isSpacePressed) {
        e.preventDefault();
        setIsSpacePressed(true);
        if (fabricCanvasRef.current) {
          (fabricCanvasRef.current as any).isSpacePressed = true;
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
        case 'l':
          setShowLabels(prev => !prev);
          break;
        default:
          break;
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space' && isSpacePressed) {
        e.preventDefault();
        setIsSpacePressed(false);
        if (fabricCanvasRef.current) {
          (fabricCanvasRef.current as any).isSpacePressed = false;
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

  const loadAnnotations = (canvas: FabricCanvas) => {
    // Remove existing annotation objects
    objectUtils.removeObjectsByPredicate(canvas, (obj: any) =>
      (obj.annotationIndex !== undefined) || !!obj.isLabel
    );

    // Always use ref to get latest annotations (avoid stale closure in async/event callbacks)
    const currentAnnotations = annotationsRef.current;
    currentAnnotations.forEach((ann, index) => {
      let typeConfig = TYPE_CONFIGS.find(t => t.value === ann.type);
      if (!typeConfig && TYPE_CONFIGS.length > 0) typeConfig = TYPE_CONFIGS[0];
      const color = typeConfig ? typeConfig.color : '#fff';
      let obj: FabricObject | undefined;
      const annForCanvas = { ...(ann as any), _canvas: canvas } as any;
      if ((ann as any).shape === 'rectangle') {
        obj = objectUtils.createRectangleObject(annForCanvas, index, color);
      } else if ((ann as any).shape === 'polygon' && (ann as any).points) {
        obj = objectUtils.createPolygonObject(annForCanvas, index, color);
        // Add custom controls for polygon point editing
        setupPolygonControls(obj as any, color);
      }
      if (obj) {
        canvas.add(obj);
        const annAny = annForCanvas as any;
        const label = objectUtils.createLabel(annForCanvas, annAny.x || (annAny.points && annAny.points[0][0]), annAny.y || (annAny.points && annAny.points[0][1]), color, index);
        label.visible = showLabels;
        canvas.add(label);
      }
    });
    canvas.requestRenderAll();
  };

  const setupPolygonControls = (polygon: FabricAnnotationObject, color: string) => {
    // Clear existing controls except default ones
    polygon.controls = {};
    // Add control for each point
    (polygon.points || []).forEach((_point: any, idx: number) => {
      polygon.controls['p' + idx] = new Control({
        positionHandler: function(_dim: any, _finalMatrix: any, fabricObject: FabricAnnotationObject) {
          if (!fabricObject.points || !fabricObject.pathOffset) return new Point(0, 0);
          const x = fabricObject.points[idx].x - (fabricObject.pathOffset?.x || 0);
          const y = fabricObject.points[idx].y - (fabricObject.pathOffset?.y || 0);
          return util.transformPoint(
            new Point(x, y),
            util.multiplyTransformMatrices(
              (fabricObject.canvas?.viewportTransform as any) || [1, 0, 0, 1, 0, 0],
              fabricObject.calcTransformMatrix()
            )
          );
        },
        actionHandler: function(_eventData: any, transform: any, x: number, y: number) {
          const poly = transform.target as FabricAnnotationObject;
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
            (poly.width as number) + ((poly.strokeWidth as number) || 0),
            (poly.height as number) + ((poly.strokeWidth as number) || 0)
          );
          const size = (poly._getTransformedDimensions as any) ? (poly._getTransformedDimensions() as any) : { x: 1, y: 1 };
          const finalPointPosition = {
            x: (localX * polygonBaseSize.x) / size.x + (poly.pathOffset?.x || 0),
            y: (localY * polygonBaseSize.y) / size.y + (poly.pathOffset?.y || 0)
          };
          if (poly.points) {
            poly.points[idx] = new Point(finalPointPosition.x, finalPointPosition.y);
          }
          return true;
        },
        actionName: 'modifyPolygon',
        cursorStyle: 'pointer',
        render: function(ctx: CanvasRenderingContext2D, left: number, top: number) {
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

  const updateLabelPosition = (canvas: FabricCanvas, obj: FabricAnnotationObject) => {
    const label = canvas.getObjects().find(
      (o) => (o as FabricAnnotationObject).isLabel && (o as FabricAnnotationObject).annotationIndex === obj.annotationIndex
    ) as FabricObject | undefined;
    if (label) {
      let x: number | undefined, y: number | undefined;
      if (obj.type === 'rect') {
        x = obj.left;
        y = obj.top;
      } else if (obj.type === 'polygon' && obj.points && obj.points.length > 0) {
        // Get transformed first point
        const matrix = obj.calcTransformMatrix();
        const point = util.transformPoint(
          new Point(
            obj.points[0].x - (obj.pathOffset?.x || 0),
            obj.points[0].y - (obj.pathOffset?.y || 0)
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

  const updateLabelText = (
    canvas: FabricCanvas,
    annotationIndex: number,
    annotation: any,
    typeConfig?: { color: string }
  ) => {
    const label = canvas.getObjects().find(
      (o) => (o as FabricAnnotationObject).isLabel && (o as FabricAnnotationObject).annotationIndex === annotationIndex
    ) as FabricObject | undefined;
    if (label) {
      let labelText = annotation.type.toUpperCase();
      if (annotation.text && annotation.text.trim() !== '') {
        labelText = `${labelText}: ${annotation.text}`;
      }
      label.set('text', labelText);
      label.set('fill', (typeConfig && typeConfig.color) || '#fff');
      canvas.requestRenderAll();
    }
  };

  const updateAnnotationFromObject = (obj: FabricAnnotationObject) => {
    if (obj.annotationIndex === undefined) return;
    const currentAnnotations = annotationsRef.current;
    console.log('updateAnnotationFromObject called for index:', obj.annotationIndex, 'isTransforming:', isTransformingRef.current, 'total:', currentAnnotations.length);
    const updated = [...currentAnnotations];
    const ann = updated[obj.annotationIndex];
    if (!ann) return;
    if ((ann as any).shape === 'rectangle') {
      const data = objectUtils.getRectangleData(obj);
      console.log('Updating rectangle:', data);
      Object.assign(ann, data);
    } else if ((ann as any).shape === 'polygon') {
      const data = objectUtils.getPolygonData(obj);
      console.log('Updating polygon:', data);
      Object.assign(ann, data);
    }
    onAnnotationsChangeRef.current(updated);
  };

  const startDrawingRectangle = () => {
    setDrawMode('rectangle');
    const canvas = fabricCanvasRef.current;
    if (!canvas) return;

    let typeConfig = TYPE_CONFIGS.find(t => t.value === 'text');
    if (!typeConfig && TYPE_CONFIGS.length > 0) typeConfig = TYPE_CONFIGS[0];
    
    // Wait for next mouse down to start drawing
    const handleMouseDown = (e: any) => {
      if (!e.target || e.target === canvas.backgroundImage) {
        rectangleCleanupRef.current = startRectDraw(
          canvas,
          e.e,
          (typeConfig && typeConfig.color) || '#fff',
          (rectData: any) => {
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

    let typeConfig = TYPE_CONFIGS.find(t => t.value === 'text');
    if (!typeConfig && TYPE_CONFIGS.length > 0) typeConfig = TYPE_CONFIGS[0];

    // Add mouse down handler for polygon point clicks
    const handlePolygonMouseDown = (e: any) => {
      if (polygonDrawerRef.current && (!e.target || e.target === canvas.backgroundImage)) {
        polygonDrawerRef.current.addPoint(e.e);
      }
    };

    polygonDrawerRef.current = new PolygonDrawer(
      canvas,
      (typeConfig && typeConfig.color) || '#fff',
      (polygonData: any) => {
        addAnnotation(polygonData);
        setDrawMode('select');
        polygonDrawerRef.current = null;
        canvas.off('mouse:down', handlePolygonMouseDown);
        polygonCleanupRef.current = null;
      }
    );

    canvas.on('mouse:down', handlePolygonMouseDown);

    // Store cleanup function
    polygonCleanupRef.current = () => {
      canvas.off('mouse:down', handlePolygonMouseDown);
    };
  };

  const handleStopDrawing = () => {
    if (polygonDrawerRef.current) {
      polygonDrawerRef.current.cancel();
      polygonDrawerRef.current = null;
    }
    if (polygonCleanupRef.current) {
      polygonCleanupRef.current();
      polygonCleanupRef.current = null;
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

  const addAnnotation = (shapeData: any, type: string = 'text') => {
    const canvas = fabricCanvasRef.current as CanvasWithImageBounds | null;

    const normalize = (data: any) => {
      if (!canvas || !canvas.imageBounds) return data;
      const b = canvas.imageBounds;
      const isRect = data.width !== undefined && data.height !== undefined;
      if (isRect) {
        const pxX = data.x ?? data.left ?? 0;
        const pxY = data.y ?? data.top ?? 0;
        const pxW = data.width ?? data.w ?? 0;
        const pxH = data.height ?? data.h ?? 0;
        return {
          ...data,
          shape: data.shape || 'rectangle',
          x: (pxX - b.left) / b.width,
          y: (pxY - b.top) / b.height,
          width: pxW / b.width,
          height: pxH / b.height
        };
      }
      // polygon
      if (Array.isArray(data.points)) {
        const pts = data.points.map((p: any) => {
          const px = p[0];
          const py = p[1];
          return [(px - b.left) / b.width, (py - b.top) / b.height];
        });
        return { ...data, shape: data.shape || 'polygon', points: pts };
      }
      return data;
    };

    const normalized = normalize(shapeData);
    const newAnnotation = {
      id: `annotation-${Date.now()}`,
      type,
      text: '',
      conf: 0.5,
      ...normalized
    };

    const updatedAnnotations = [...annotationsRef.current, newAnnotation];
    onAnnotationsChangeRef.current(updatedAnnotations);
    onSelectAnnotationRef.current(updatedAnnotations.length - 1);
  };

  const handleDeleteAnnotation = (index: number) => {
    const canvas = fabricCanvasRef.current;
    if (!canvas) return;

    canvasActions.deleteByIndex(canvas, index);

    const updated = annotationsRef.current.filter((_, i) => i !== index);
    onAnnotationsChangeRef.current(updated);
    onSelectAnnotationRef.current(null);
  };

  const handleZoomIn = () => {
    const canvas = fabricCanvasRef.current;
    if (canvas) {
      const zoom = (canvas as any).getZoom();
      objectUtils.setCanvasZoom(canvas, zoom * 1.2);
    }
  };

  const handleZoomOut = () => {
    const canvas = fabricCanvasRef.current;
    if (canvas) {
      const zoom = (canvas as any).getZoom();
      objectUtils.setCanvasZoom(canvas, zoom / 1.2);
    }
  };

  const handleResetZoom = () => {
    objectUtils.resetCanvasZoom(fabricCanvasRef.current as any);
  };

  // ── Resize chars: group chars by spatial parent (text/datecode region) ───
  // Char belongs to a region if its center point falls inside the region's
  // bounding box (or polygon). Multiple matches → smallest-area region wins.
  // If a single char is selected, a "Selected: char #N" group is added on top.
  const truncate = (s: string, n = 30) => (s.length > n ? s.slice(0, n - 1) + '…' : s);

  const selectedCharIdx = useMemo(() => {
    if (selectedAnnotation === null) return null;
    const a = annotations[selectedAnnotation];
    if (!a || a.type !== 'char' || a.shape !== 'rectangle') return null;
    return selectedAnnotation;
  }, [selectedAnnotation, annotations]);

  const charGroups = useMemo(() => {
    const charItems = annotations
      .map((a, i) => ({ a, i }))
      .filter(({ a }) => a.type === 'char' && a.shape === 'rectangle');

    const regionItems = annotations
      .map((a, i) => ({ a, i }))
      .filter(({ a }) => a.type === 'text' || a.type === 'datecode');

    const regionBBox = (a: Annotation): { x: number; y: number; w: number; h: number } | null => {
      if (a.shape === 'rectangle') {
        return { x: a.x ?? 0, y: a.y ?? 0, w: a.width ?? 0, h: a.height ?? 0 };
      }
      if (a.shape === 'polygon' && a.points && a.points.length > 0) {
        const xs = a.points.map(p => Number(p[0] ?? 0));
        const ys = a.points.map(p => Number(p[1] ?? 0));
        const minX = Math.min(...xs), maxX = Math.max(...xs);
        const minY = Math.min(...ys), maxY = Math.max(...ys);
        return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
      }
      return null;
    };

    const pointInRect = (px: number, py: number, b: { x: number; y: number; w: number; h: number }) =>
      px >= b.x && px <= b.x + b.w && py >= b.y && py <= b.y + b.h;

    const charToRegion = new Map<number, number>();   // char idx → region idx (smallest matching)
    for (const { a: char, i: cIdx } of charItems) {
      const cx = (char.x ?? 0) + (char.width ?? 0) / 2;
      const cy = (char.y ?? 0) + (char.height ?? 0) / 2;
      let bestRegionIdx: number | null = null;
      let bestArea = Infinity;
      for (const { a: region, i: rIdx } of regionItems) {
        const bb = regionBBox(region);
        if (!bb) continue;
        if (!pointInRect(cx, cy, bb)) continue;
        const area = bb.w * bb.h;
        if (area < bestArea) { bestArea = area; bestRegionIdx = rIdx; }
      }
      if (bestRegionIdx !== null) charToRegion.set(cIdx, bestRegionIdx);
    }

    type Group = { id: string; label: string; indices: number[] };
    const groups: Group[] = [];

    // Single-char selection takes top spot (auto-selected via useEffect below)
    if (selectedCharIdx !== null) {
      groups.push({
        id: `selected:${selectedCharIdx}`,
        label: `Selected: char #${selectedCharIdx}`,
        indices: [selectedCharIdx],
      });
    }

    groups.push({
      id: 'all',
      label: `All chars (${charItems.length})`,
      indices: charItems.map(({ i }) => i),
    });
    for (const { a: region, i: rIdx } of regionItems) {
      const indices = charItems.filter(({ i }) => charToRegion.get(i) === rIdx).map(({ i }) => i);
      if (indices.length === 0) continue;
      const rawLabel = (region.text || '').toString().trim()
        || `${region.type === 'datecode' ? 'Date' : 'Text'} #${rIdx}`;
      groups.push({
        id: `region:${rIdx}`,
        label: `${truncate(rawLabel, 24)} (${indices.length} chars)`,
        indices,
      });
    }
    const unassignedIndices = charItems
      .filter(({ i }) => !charToRegion.has(i))
      .map(({ i }) => i);
    if (unassignedIndices.length > 0) {
      groups.push({ id: 'unassigned', label: `Unassigned (${unassignedIndices.length} chars)`, indices: unassignedIndices });
    }
    return groups;
  }, [annotations, selectedCharIdx]);

  // Auto-select "Selected: char #N" scope whenever a char gets selected on canvas
  useEffect(() => {
    if (selectedCharIdx !== null) {
      setResizeScope(`selected:${selectedCharIdx}`);
    }
  }, [selectedCharIdx]);

  // If selected scope vanishes (region deleted) → fall back to 'all'
  useEffect(() => {
    if (!charGroups.find(g => g.id === resizeScope)) setResizeScope('all');
  }, [charGroups, resizeScope]);

  const activeGroup = charGroups.find(g => g.id === resizeScope) ?? charGroups[0];
  const resizeEnabled = !!activeGroup && activeGroup.indices.length > 0;

  // Apply symmetric Δ to chars in active group, on selected axis.
  // resizeAxis: 'both' | 'w' | 'h'. sign=+1 expands, -1 shrinks. Polygon chars skipped.
  // Annotations store NORMALIZED coords [0..1] (see objectUtils.getRectangleData),
  // so we convert px → normalized for storage, then back to px for fabric objects.
  const applyResize = (sign: 1 | -1) => {
    if (!resizeEnabled || !activeGroup) return;
    const canvas = fabricCanvasRef.current as CanvasWithImageBounds | null;
    const bounds = canvas?.imageBounds;
    if (!canvas || !bounds) return;
    const dPx = Math.max(1, resizeDelta) * sign;
    const dxN = (resizeAxis === 'h') ? 0 : dPx / bounds.width;
    const dyN = (resizeAxis === 'w') ? 0 : dPx / bounds.height;
    const indexSet = new Set(activeGroup.indices);

    const next = annotations.map((ann, i) => {
      if (!indexSet.has(i)) return ann;
      if (ann.shape !== 'rectangle') return ann;     // skip polygon chars
      const cur = { ...ann };
      const newW = Math.min(1, Math.max(0.001, (cur.width  ?? 0) + dxN));
      const newH = Math.min(1, Math.max(0.001, (cur.height ?? 0) + dyN));
      const newX = (cur.x ?? 0) - dxN / 2;
      const newY = (cur.y ?? 0) - dyN / 2;
      cur.width  = newW;
      cur.height = newH;
      cur.x = Math.max(0, Math.min(1 - newW, newX));
      cur.y = Math.max(0, Math.min(1 - newH, newY));
      return cur;
    });

    // Sync fabric objects directly — the smart-update useEffect on `annotations`
    // only refreshes color/labels, not geometry, so we update size+pos here.
    // Convert normalized [0..1] → canvas pixels using imageBounds.
    canvas.getObjects().forEach(obj => {
      const fObj = obj as FabricAnnotationObject;
      const idx = fObj.annotationIndex;
      if (idx === undefined || fObj.isLabel || fObj.isTemp) return;
      if (!indexSet.has(idx)) return;
      const updated = next[idx];
      if (!updated || updated.shape !== 'rectangle') return;
      obj.set({
        left:   bounds.left + (updated.x      ?? 0) * bounds.width,
        top:    bounds.top  + (updated.y      ?? 0) * bounds.height,
        width:                (updated.width  ?? 0) * bounds.width,
        height:               (updated.height ?? 0) * bounds.height,
        scaleX: 1,
        scaleY: 1,
      });
      obj.setCoords();
    });
    canvas.requestRenderAll();

    onAnnotationsChange(next);
  };

  // Full editor UI (toolbar, zoom controls, hints, canvas)
  return (
    <div className="template-editor">
      <div className="editor-toolbar">
        <div className="toolbar-row">
        <div className="toolbar-group">
          <button
            type="button"
            className={`tool-btn ${drawMode === 'select' ? 'active' : ''}`}
            onClick={handleStopDrawing}
            title="Pan Canvas (Click & Drag)"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M9 11V6C9 4.89543 9.89543 4 11 4C12.1046 4 13 4.89543 13 6V11M13 11V6C13 4.89543 13.8954 4 15 4C16.1046 4 17 4.89543 17 6V11M17 11V7C17 5.89543 17.8954 5 19 5C20.1046 5 21 5.89543 21 7V12C21 12 21 20 12 20C3 20 3 12 3 12V11C3 9.89543 3.89543 9 5 9C6.10457 9 7 9.89543 7 11V12M13 11H9M13 11H17" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
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

        <div className="toolbar-group toolbar-group-end">
          <button type="button" className="tool-btn" onClick={handleZoomOut} title="Zoom Out">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <circle cx="11" cy="11" r="8" stroke="currentColor" strokeWidth="2"/>
              <line x1="8" y1="11" x2="14" y2="11" stroke="currentColor" strokeWidth="2"/>
              <line x1="17" y1="17" x2="21" y2="21" stroke="currentColor" strokeWidth="2"/>
            </svg>
          </button>
          <span className="zoom-label">
            {fabricCanvasRef.current ? Math.round(((fabricCanvasRef.current as any).getZoom() || 1) * 100) : 100}%
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

        <div className="toolbar-group">
          <button
            type="button"
            className={`tool-btn ${showLabels ? 'active' : ''}`}
            onClick={() => setShowLabels(prev => !prev)}
            title={showLabels ? 'Hide Labels (L)' : 'Show Labels (L)'}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M4 7V4H20V7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M12 4V20" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              <path d="M8 20H16" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>
        </div>
        </div>

        {/* ── Row 2: resize chars panel ──────────────────────────────── */}
        <div className="toolbar-row toolbar-row-secondary">
        {/* ── Resize chars: scope + axis + Δ + −/+ ───────────────────── */}
        <div className={`toolbar-group resize-chars-group${resizeEnabled ? '' : ' disabled'}`}>
          <select
            className="resize-chars-select"
            value={resizeScope}
            onChange={e => setResizeScope(e.target.value)}
            disabled={charGroups.length === 0}
            title={charGroups.find(g => g.id === resizeScope)?.label || 'Choose which chars to resize'}
          >
            {charGroups.map(g => (
              <option key={g.id} value={g.id} title={g.label}>{g.label}</option>
            ))}
          </select>
          <input
            type="number"
            className="resize-chars-input"
            min={1}
            max={50}
            value={resizeDelta}
            onChange={e => setResizeDelta(Math.max(1, Number(e.target.value) || 1))}
            disabled={!resizeEnabled}
            title="Step in pixels"
          />
          <div className="resize-chars-axis-seg" role="group" aria-label="Resize axis">
            {(['both', 'w', 'h'] as const).map(ax => (
              <button
                key={ax}
                type="button"
                className={`resize-chars-axis-btn${resizeAxis === ax ? ' active' : ''}`}
                onClick={() => setResizeAxis(ax)}
                disabled={!resizeEnabled}
                title={ax === 'both' ? 'Resize both width & height' : ax === 'w' ? 'Resize width only' : 'Resize height only'}
              >
                {ax === 'both' ? 'W·H' : ax.toUpperCase()}
              </button>
            ))}
          </div>
          <button type="button" className="tool-btn tool-btn-sm" disabled={!resizeEnabled}
                  onClick={() => applyResize(-1)}
                  title={`Shrink ${resizeDelta}px (${resizeAxis === 'both' ? 'W & H' : resizeAxis.toUpperCase()})`}>−</button>
          <button type="button" className="tool-btn tool-btn-sm" disabled={!resizeEnabled}
                  onClick={() => applyResize( 1)}
                  title={`Expand ${resizeDelta}px (${resizeAxis === 'both' ? 'W & H' : resizeAxis.toUpperCase()})`}>+</button>
        </div>
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
                  <li><strong>Pan Mode (V):</strong> Click & Drag to pan</li>
                  <li><kbd>Space</kbd> + Drag - Pan (any mode)</li>
                  <li><kbd>Shift</kbd> + Drag - Pan (any mode)</li>
                  <li><kbd>Middle Click</kbd> + Drag - Pan (any mode)</li>
                  <li><kbd>Mouse Wheel</kbd> - Zoom in/out</li>
                  <li><kbd>Trackpad Pinch</kbd> - Zoom in/out</li>
                </ul>
              </div>
              
              <div className="cvat-canvas-hints-block">
                <strong>⌨️ Shortcuts</strong>
                <ul>
                  <li><kbd>V</kbd> - Pan mode (Click & Drag to move canvas)</li>
                  <li><kbd>R</kbd> - Rectangle tool</li>
                  <li><kbd>P</kbd> - Polygon tool (4 points)</li>
                  <li><kbd>Esc</kbd> - Cancel drawing</li>
                  <li><kbd>Del</kbd> - Delete selected</li>
                </ul>
              </div>

              <div className="cvat-canvas-hints-block">
                <strong>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" style={{ marginRight: '6px', verticalAlign: 'middle' }}>
                    <path d="M12 20h9M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  Drawing & Navigation
                </strong>
                <ul>
                  <li>Pan: Click and drag to move canvas</li>
                  <li>Rectangle: Click and drag to draw</li>
                  <li>Polygon: Click 4 points to draw</li>
                  <li>Zoom: Mouse wheel or pinch gesture</li>
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
