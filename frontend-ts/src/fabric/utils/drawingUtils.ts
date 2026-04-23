/**
 * Drawing utilities for annotations
 */
import { Rect, Circle, Line } from 'fabric';

/**
 * Start drawing a rectangle
 * @param {fabric.Canvas} canvas 
 * @param {MouseEvent} e 
 * @param {string} color 
 * @param {Function} onComplete - Callback when rectangle is completed
 * @returns {Function} cleanup function
 */
export const startDrawingRectangle = (canvas: any, e: any, color: string, onComplete: (data: any) => void) => {
  const pointer = canvas.getPointer(e);
  
  // Constrain initial point within image bounds
  let startX = pointer.x;
  let startY = pointer.y;
  
  if (canvas.imageBounds) {
    const bounds = canvas.imageBounds;
    startX = Math.max(bounds.left, Math.min(bounds.right, pointer.x));
    startY = Math.max(bounds.top, Math.min(bounds.bottom, pointer.y));
  }
  
  const zoom = canvas.getZoom?.() || 1;
  const rect = new Rect({
    left: startX,
    top: startY,
    width: 0,
    height: 0,
    fill: color + '30',
    stroke: color,
    strokeWidth: Math.max(0.5, 3 / zoom),
    selectable: false,
    evented: false
  });
  
  canvas.add(rect);
  let isDrawing = true;
  
  const handleMouseMove = (opt: any) => {
    if (!isDrawing) return;
    let pointer = canvas.getPointer(opt.e);
    
    // Constrain pointer within image bounds
    if (canvas.imageBounds) {
      const bounds = canvas.imageBounds;
      pointer.x = Math.max(bounds.left, Math.min(bounds.right, pointer.x));
      pointer.y = Math.max(bounds.top, Math.min(bounds.bottom, pointer.y));
    }
    
    const newWidth = Math.abs(pointer.x - startX);
    const newHeight = Math.abs(pointer.y - startY);
    
    let left = startX;
    let top = startY;
    
    if (pointer.x < startX) {
      left = pointer.x;
    }
    if (pointer.y < startY) {
      top = pointer.y;
    }
    
    rect.set({
      left: left,
      top: top,
      width: newWidth,
      height: newHeight
    });
    canvas.requestRenderAll();
  };
  
  const handleMouseUp = () => {
    if (!isDrawing) return;
    isDrawing = false;
    canvas.off('mouse:move', handleMouseMove);
    canvas.off('mouse:up', handleMouseUp);
    
    if (rect.width > 10 && rect.height > 10) {
      onComplete({
        shape: 'rectangle',
        x: rect.left,
        y: rect.top,
        width: rect.width,
        height: rect.height
      });
    }
    
    canvas.remove(rect);
    canvas.requestRenderAll();
  };
  
  canvas.on('mouse:move', handleMouseMove);
  canvas.on('mouse:up', handleMouseUp);
  
  return () => {
    isDrawing = false;
    canvas.off('mouse:move', handleMouseMove);
    canvas.off('mouse:up', handleMouseUp);
    canvas.remove(rect);
    canvas.requestRenderAll();
  };
};

/**
 * Manage polygon drawing state
 */
export class PolygonDrawer {
  canvas: any;
  color: string;
  onComplete: (data: any) => void;
  points: any[];
  tempObjects: any[];
  maxPoints: number;

  constructor(canvas: any, color: string, onComplete: (data: any) => void) {
    this.canvas = canvas;
    this.color = color;
    this.onComplete = onComplete;
    this.points = [];
    this.tempObjects = [];
    this.maxPoints = 4; // For quadrilateral
  }
  
  /**
   * Add a point to the polygon
   * @param {MouseEvent} e 
   */
  addPoint(e: any) {
    let pointer = this.canvas.getPointer(e);
    
    // Constrain point within image bounds
    if (this.canvas.imageBounds) {
      const bounds = this.canvas.imageBounds;
      pointer.x = Math.max(bounds.left, Math.min(bounds.right, pointer.x));
      pointer.y = Math.max(bounds.top, Math.min(bounds.bottom, pointer.y));
    }
    
    const newPoint = { x: pointer.x, y: pointer.y };
    
    this.points.push(newPoint);
    
    // Draw point circle
    const pZoom = this.canvas.getZoom?.() || 1;
    const pRadius = Math.max(2, 5 / pZoom);
    const circle = new Circle({
      left: pointer.x - pRadius,
      top: pointer.y - pRadius,
      radius: pRadius,
      fill: this.color,
      stroke: '#ffffff',
      strokeWidth: Math.max(0.5, 2 / pZoom),
      selectable: false,
      evented: false,
      isTemp: true
    });
    this.canvas.add(circle);
    this.tempObjects.push(circle);

    // Draw line from previous point
    if (this.points.length > 1) {
      const prevPoint = this.points[this.points.length - 2];
      const line = new Line(
        [prevPoint.x, prevPoint.y, newPoint.x, newPoint.y],
        {
          stroke: this.color,
          strokeWidth: Math.max(0.5, 3 / pZoom),
          selectable: false,
          evented: false,
          isTemp: true
        }
      );
      this.canvas.add(line);
      this.tempObjects.push(line);
    }
    
    this.canvas.requestRenderAll();
    
    // Complete polygon after max points
    if (this.points.length === this.maxPoints) {
      this.complete();
    }
  }
  
  /**
   * Complete the polygon
   */
  complete() {
    if (this.points.length < 3) {
      this.cancel();
      return;
    }
    
    // Remove temp objects
    this.tempObjects.forEach((obj: any) => this.canvas.remove(obj));
    this.tempObjects = [];
    
    // Return polygon data
    this.onComplete({
      shape: 'polygon',
      points: this.points.map((p: any) => [p.x, p.y])
    });
    
    this.canvas.requestRenderAll();
  }
  
  /**
   * Cancel polygon drawing
   */
  cancel() {
    this.tempObjects.forEach((obj: any) => this.canvas.remove(obj));
    this.tempObjects = [];
    this.points = [];
    this.canvas.requestRenderAll();
  }
  
  /**
   * Get remaining points needed
   */
  getRemainingPoints() {
    return this.maxPoints - this.points.length;
  }
}
