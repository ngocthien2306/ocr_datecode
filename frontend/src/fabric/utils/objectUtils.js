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
export const findObjectByIndex = (canvas, index) => {
  if (!canvas) return undefined;
  return canvas.getObjects().find(obj => obj.annotationIndex === index);
};

/**
 * Get all annotation objects (exclude background, labels, etc.)
 * @param {fabric.Canvas} canvas 
 * @returns {Array<fabric.Object>}
 */
export const getAnnotationObjects = (canvas) => {
  if (!canvas) return [];
  return canvas.getObjects().filter(obj => 
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
export const removeObjectsByPredicate = (canvas, predicate) => {
  if (!canvas) return;
  const objectsToRemove = canvas.getObjects().filter(predicate);
  objectsToRemove.forEach(obj => canvas.remove(obj));
};

/**
 * Create a fabric Rectangle with annotation properties
 * @param {Object} annotation 
 * @param {number} index 
 * @param {string} color 
 * @returns {fabric.Rect}
 */
export const createRectangleObject = (annotation, index, color) => {
  return new Rect({
    left: annotation.x,
    top: annotation.y,
    width: annotation.width,
    height: annotation.height,
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
  });
};

/**
 * Create a fabric Polygon with annotation properties
 * @param {Object} annotation 
 * @param {number} index 
 * @param {string} color 
 * @returns {fabric.Polygon}
 */
export const createPolygonObject = (annotation, index, color) => {
  const points = annotation.points.map(p => ({ x: p[0], y: p[1] }));
  
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
  });
};

/**
 * Create a text label for annotation
 * @param {string} text 
 * @param {number} x 
 * @param {number} y 
 * @param {string} color 
 * @returns {fabric.Text}
 */
export const createLabel = (text, x, y, color) => {
  return new FabricText(text.toUpperCase(), {
    left: x + 5,
    top: y - 25,
    fontSize: 14,
    fill: color,
    selectable: false,
    evented: false,
    fontFamily: 'Arial',
    isLabel: true
  });
};

/**
 * Create edit point circle for polygon
 * @param {Object} point - { x, y }
 * @param {number} index - Point index
 * @param {string} color 
 * @param {fabric.Polygon} polygon - Reference to parent polygon
 * @returns {fabric.Circle}
 */
export const createPolygonEditPoint = (point, index, color, polygon) => {
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
  });
};

/**
 * Update rectangle annotation from fabric object
 * @param {fabric.Rect} obj 
 * @returns {Object} Updated annotation data
 */
export const getRectangleData = (obj) => {
  return {
    x: obj.left,
    y: obj.top,
    width: obj.width * obj.scaleX,
    height: obj.height * obj.scaleY
  };
};

/**
 * Update polygon annotation from fabric object
 * @param {fabric.Polygon} obj 
 * @returns {Object} Updated annotation data
 */
export const getPolygonData = (obj) => {
  const matrix = obj.calcTransformMatrix();
  const transformedPoints = [];
  
  obj.points.forEach(p => {
    const point = util.transformPoint(
      { x: p.x - (obj.pathOffset?.x || 0), y: p.y - (obj.pathOffset?.y || 0) },
      matrix
    );
    transformedPoints.push([point.x, point.y]);
  });
  
  return {
    points: transformedPoints
  };
};

/**
 * Setup canvas pan and zoom
 * @param {fabric.Canvas} canvas 
 */
export const setupCanvasPanning = (canvas) => {
  canvas.on('mouse:down', function(opt) {
    const evt = opt.e;
    if (evt.button === 1 || (evt.button === 0 && (evt.shiftKey || this.isSpacePressed))) {
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
    this.selection = true;
  });
  
  // Mouse wheel zoom
  canvas.on('mouse:wheel', function(opt) {
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
export const setCanvasZoom = (canvas, zoom) => {
  if (!canvas) return;
  canvas.setZoom(Math.min(Math.max(zoom, 0.5), 3));
  canvas.requestRenderAll();
};

/**
 * Reset canvas zoom and position
 * @param {fabric.Canvas} canvas 
 */
export const resetCanvasZoom = (canvas) => {
  if (!canvas) return;
  canvas.setZoom(1);
  canvas.viewportTransform[4] = 0;
  canvas.viewportTransform[5] = 0;
  canvas.requestRenderAll();
};
