/**
 * Fabric.js utilities for canvas manipulation
 */
import { Rect, Polygon, Circle, FabricText, util } from 'fabric';

/**
 * Find object by custom property (annotationIndex)
 * @param {fabric.Canvas} canvas 
 * @param {number} index 
 * @returns {fabric.Object | undefined}
 */
export const findObjectByIndex = (canvas: any, index: number): any => {
  if (!canvas) return undefined;
  return canvas.getObjects().find((obj: any) => obj.annotationIndex === index);
};

/**
 * Get all annotation objects (exclude background, labels, etc.)
 * @param {fabric.Canvas} canvas 
 * @returns {Array<fabric.Object>}
 */
export const getAnnotationObjects = (canvas: any): any[] => {
  if (!canvas) return [];
  return canvas.getObjects().filter((obj: any) => 
    obj.annotationIndex !== undefined && 
    !obj.isLabel && 
    !obj.isPolygonEditPoint
  );
};

/**
 * Remove all objects of a specific type
 * @param {fabric.Canvas} canvas 
 * @param {Function} predicate - Filter function
 */
export const removeObjectsByPredicate = (canvas: any, predicate: (obj: any) => boolean): void => {
  if (!canvas) return;
  const objectsToRemove = canvas.getObjects().filter(predicate);
  objectsToRemove.forEach((obj: any) => canvas.remove(obj));
};

/**
 * Create a fabric Rectangle with annotation properties
 * @param {Object} annotation 
 * @param {number} index 
 * @param {string} color 
 * @returns {fabric.Rect}
 */
export const createRectangleObject = (annotation: any, index: number, color: string): any => {
  // Accept annotation coordinates either as absolute pixels or relative (0..1)
  let left = annotation.x;
  let top = annotation.y;
  let width = annotation.width;
  let height = annotation.height;
  // If a canvas with imageBounds was passed inside annotation._canvas, use it to convert relative coords
  const canvas = (annotation && annotation._canvas) || null;
  if (canvas && canvas.imageBounds) {
    const b = canvas.imageBounds;
    const isRel = (v: any) => typeof v === 'number' && v >= 0 && v <= 1;
    if (isRel(left) && isRel(width)) {
      left = b.left + left * b.width;
      top = b.top + top * b.height;
      width = width * b.width;
      height = height * b.height;
    }
  }

  return new Rect({
    left,
    top,
    width,
    height,
    fill: color + '20', // Add transparency
    stroke: color,
    strokeWidth: 2,
    selectable: true,
    evented: true,
    hasControls: true,
    hasBorders: true,
    lockRotation: true,
    lockScalingFlip: true,
    lockSkewingX: true,
    lockSkewingY: true,
    cornerSize: 10,
    transparentCorners: false,
    cornerColor: color,
    cornerStrokeColor: '#ffffff',
    borderColor: color,
    objectCaching: false,
    annotationIndex: index,
    annotationType: annotation.type
  } as any);
};

/**
 * Create a fabric Polygon with annotation properties
 * @param {Object} annotation 
 * @param {number} index 
 * @param {string} color 
 * @returns {fabric.Polygon}
 */
export const createPolygonObject = (annotation: any, index: number, color: string): any => {
  // Points may be relative (0..1) or absolute pixels. Support annotation._canvas as conversion hint.
  const canvas = (annotation && annotation._canvas) || null;
  const points = (annotation.points || []).map((p: any) => {
    let px = p[0];
    let py = p[1];
    if (canvas && canvas.imageBounds) {
      const b = canvas.imageBounds;
      const isRel = (v: any) => typeof v === 'number' && v >= 0 && v <= 1;
      if (isRel(px) && isRel(py)) {
        px = b.left + px * b.width;
        py = b.top + py * b.height;
      }
    }
    return { x: px, y: py };
  });

  return new Polygon(points, {
    fill: color + '20',
    stroke: color,
    strokeWidth: 2,
    selectable: true,
    hasControls: true,
    hasBorders: true,
    lockRotation: true,
    lockScalingX: true,
    lockScalingY: true,
    lockMovementX: false,
    lockMovementY: false,
    cornerSize: 10,
    transparentCorners: false,
    cornerColor: color,
    cornerStrokeColor: '#ffffff',
    borderColor: color,
    objectCaching: false,
    perPixelTargetFind: true,
    annotationIndex: index,
    annotationType: annotation.type
  } as any);
};

/**
 * Create a text label for annotation
 * @param {string} text 
 * @param {number} x 
 * @param {number} y 
 * @param {string} color 
 * @param {number} index - Annotation index to link label
 * @returns {fabric.Text}
 */
export const createLabel = (annotation: any, x: number, y: number, color: string, index: number): any => {
  // Show type and text (if available)
  let labelText = annotation.type.toUpperCase();
  if (annotation.text && annotation.text.trim() !== '') {
    labelText = `${labelText}: ${annotation.text}`;
  }
  
  // If annotation provides _canvas and x/y are relative, convert to pixels
  const canvas = (annotation && annotation._canvas) || null;
  let left = x;
  let top = y;
  if (canvas && canvas.imageBounds && typeof x === 'number' && typeof y === 'number') {
    const isRel = (v: any) => typeof v === 'number' && v >= 0 && v <= 1;
    if (isRel(x) && isRel(y)) {
      left = canvas.imageBounds.left + x * canvas.imageBounds.width;
      top = canvas.imageBounds.top + y * canvas.imageBounds.height;
    }
  }

  return new FabricText(labelText, {
    left: left + 5,
    top: top - 25,
    fontSize: 14,
    fill: color,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    padding: 4,
    selectable: false,
    evented: false,
    fontFamily: 'Arial',
    isLabel: true,
    annotationIndex: index
  } as any);
};

/**
 * Create edit point circle for polygon
 * @param {Object} point - { x, y }
 * @param {number} index - Point index
 * @param {string} color 
 * @param {fabric.Polygon} polygon - Reference to parent polygon
 * @returns {fabric.Circle}
 */
export const createPolygonEditPoint = (point: any, index: number, color: string, polygon: any): any => {
  return new Circle({
    left: point[0],
    top: point[1],
    radius: 10,
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
    pointIndex: index,
    parentPolygon: polygon
  } as any);
};

/**
 * Update rectangle annotation from fabric object
 * @param {fabric.Rect} obj 
 * @returns {Object} Updated annotation data
 */
export const getRectangleData = (obj: any): any => {
  const px = {
    x: obj.left,
    y: obj.top,
    width: obj.width * obj.scaleX,
    height: obj.height * obj.scaleY
  };
  const canvas = obj.canvas || (obj && obj._canvas) || null;
  if (canvas && canvas.imageBounds) {
    const b = canvas.imageBounds;
    return {
      x: (px.x - b.left) / b.width,
      y: (px.y - b.top) / b.height,
      width: px.width / b.width,
      height: px.height / b.height
    };
  }
  return px;
};

/**
 * Update polygon annotation from fabric object
 * @param {fabric.Polygon} obj 
 * @returns {Object} Updated annotation data
 */
export const getPolygonData = (obj: any): any => {
  const matrix = obj.calcTransformMatrix();
  const transformedPoints: any[] = [];

  obj.points.forEach((p: any) => {
    const point = util.transformPoint(
      { x: p.x - (obj.pathOffset?.x || 0), y: p.y - (obj.pathOffset?.y || 0) },
      matrix
    );
    transformedPoints.push([point.x, point.y]);
  });
  // If canvas has imageBounds, convert to relative coords
  const canvas = obj.canvas || (obj && obj._canvas) || null;
  if (canvas && canvas.imageBounds) {
    const b = canvas.imageBounds;
    return {
      points: transformedPoints.map((p: any) => [ (p[0] - b.left) / b.width, (p[1] - b.top) / b.height ])
    };
  }
  return {
    points: transformedPoints
  };
};
/**
 * Setup canvas pan and zoom
 * @param {fabric.Canvas} canvas 
 */
export const setupCanvasPanning = (canvas: any): void => {
  canvas.on('mouse:down', function(this: any, opt: any) {
    const evt = opt.e;
    if (evt.button === 1 || (evt.button === 0 && (evt.shiftKey || this.isSpacePressed))) {
      this.isDragging = true;
      this.selection = false;
      this.lastPosX = evt.clientX;
      this.lastPosY = evt.clientY;
      evt.preventDefault();
    }
  });

  canvas.on('mouse:move', function(this: any, opt: any) {
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

  canvas.on('mouse:up', function(this: any) {
    this.setViewportTransform(this.viewportTransform);
    this.isDragging = false;
    this.selection = true;
  });
  
  // Mouse wheel zoom
  canvas.on('mouse:wheel', function(this: any, opt: any) {
    const delta = opt.e.deltaY;
    let zoom = canvas.getZoom();
    zoom *= 0.999 ** delta;
    
    if (zoom > 3) zoom = 3;
    if (zoom < 0.5) zoom = 0.5;
    
    canvas.zoomToPoint({ x: opt.e.offsetX, y: opt.e.offsetY }, zoom);
    opt.e.preventDefault();
    opt.e.stopPropagation();
  });
};

/**
 * Set canvas zoom
 * @param {fabric.Canvas} canvas 
 * @param {number} zoom - Zoom level (0.5 to 3)
 */
export const setCanvasZoom = (canvas: any, zoom: number): void => {
  if (!canvas) return;
  canvas.setZoom(Math.min(Math.max(zoom, 0.5), 3));
  canvas.requestRenderAll();
};

/**
 * Reset canvas zoom and position
 * @param {fabric.Canvas} canvas 
 */
export const resetCanvasZoom = (canvas: any): void => {
  if (!canvas) return;
  canvas.setZoom(1);
  canvas.viewportTransform[4] = 0;
  canvas.viewportTransform[5] = 0;
  canvas.requestRenderAll();
};
