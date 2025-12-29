// Duplicate type imports removed

import React, { useState, useRef, useEffect } from 'react';
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
  
  const [drawMode, setDrawMode] = useState('select');
  const polygonDrawerRef = useRef<any>(null);
  const polygonCleanupRef = useRef<(() => void) | null>(null);
  const rectangleCleanupRef = useRef<(() => void) | null>(null);
  const [showHints, setShowHints] = useState(false);
  const [isSpacePressed, setIsSpacePressed] = useState(false);

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

    // Handle selection
    canvas.on('selection:created', (e) => {
      const obj = e.selected[0] as FabricAnnotationObject;
      if (obj && !obj.isTemp && obj.annotationIndex !== undefined) {
        onSelectAnnotation?.(obj.annotationIndex);
      }
    });

    canvas.on('selection:updated', (e) => {
      const obj = e.selected[0] as FabricAnnotationObject;
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

    canvas.selection = drawMode === 'select';
    canvas.forEachObject((obj) => {
      // Annotation objects only selectable in select mode
      const objAnn = obj as FabricAnnotationObject;
      if (objAnn.annotationIndex !== undefined) {
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
      
      // Handle count mismatch - add new or remove deleted annotations
      if (existingObjects.length !== annotations.length) {
        if (existingObjects.length < annotations.length) {
          // New annotations added - only add the new ones
          console.log('Adding new annotations');
          for (let i = existingObjects.length; i < annotations.length; i++) {
            const ann = annotations[i];
            if (!ann) return;
            let typeConfig = TYPE_CONFIGS.find(t => t.value === ann.type);
            if (!typeConfig && TYPE_CONFIGS.length > 0) typeConfig = TYPE_CONFIGS[0];
            const color = typeConfig ? typeConfig.color : '#fff';
            console.log('Creating new object:', { index: i, type: ann.type, color, shape: (ann as any).shape });
            let obj;
            // pass canvas reference via a shallow copy so objectUtils can convert relative coords
            const annForCanvas = { ...(ann as any), _canvas: canvas } as any;
            if ((ann as any).shape === 'rectangle') {
              obj = objectUtils.createRectangleObject(annForCanvas, i, color);
            } else if ((ann as any).shape === 'polygon') {
              obj = objectUtils.createPolygonObject(annForCanvas, i, color);
              setupPolygonControls(obj, color);
            }
            if (obj) {
              canvas.add(obj);
              const annAny = annForCanvas as any;
              const label = objectUtils.createLabel(annForCanvas, annAny.x || (annAny.points && annAny.points[0][0]), annAny.y || (annAny.points && annAny.points[0][1]), color, i);
              canvas.add(label);
            }
          }
          canvas.requestRenderAll();
        } else {
          // Annotations removed - remove objects with higher indices
          console.log('Removing deleted annotations');
          const safeExisting = existingObjects.filter(Boolean) as FabricObject[];
          const objectsToRemove = safeExisting.filter((obj: any) => obj && obj.annotationIndex !== undefined && obj.annotationIndex >= annotations.length);
          objectsToRemove.forEach(obj => {
            // Remove associated label
            const label = canvas.getObjects().find(o => 
              (o as FabricAnnotationObject).isLabel && (o as FabricAnnotationObject).annotationIndex === (obj as FabricAnnotationObject).annotationIndex
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

    annotations.forEach((ann, index) => {
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
    console.log('updateAnnotationFromObject called for index:', obj.annotationIndex, 'isTransforming:', isTransformingRef.current);
    const updated = [...annotations];
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
    onAnnotationsChange(updated);
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
      ...normalized
    };

    const updatedAnnotations = [...annotations, newAnnotation];
    onAnnotationsChange(updatedAnnotations);
    onSelectAnnotation(updatedAnnotations.length - 1);
  };

  const handleDeleteAnnotation = (index: number) => {
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

  // Full editor UI (toolbar, zoom controls, hints, canvas)
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
