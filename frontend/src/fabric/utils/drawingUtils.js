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
export const startDrawingRectangle = (canvas, e, color, onComplete) => {
  const pointer = canvas.getPointer(e);
  
  const rect = new Rect({
    left: pointer.x,
    top: pointer.y,
    width: 0,
    height: 0,
    fill: color + '30',
    stroke: color,
    strokeWidth: 3,
    selectable: false,
    evented: false
  });
  
  canvas.add(rect);
  let isDrawing = true;
  
  const handleMouseMove = (opt) => {
    if (!isDrawing) return;
    const pointer = canvas.getPointer(opt.e);
    
    const newWidth = Math.abs(pointer.x - rect.left);
    const newHeight = Math.abs(pointer.y - rect.top);
    
    if (pointer.x < rect.left) {
      rect.set({ left: pointer.x });
    }
    if (pointer.y < rect.top) {
      rect.set({ top: pointer.y });
    }
    
    rect.set({
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
  constructor(canvas, color, onComplete) {
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
  addPoint(e) {
    const pointer = this.canvas.getPointer(e);
    const newPoint = { x: pointer.x, y: pointer.y };
    
    this.points.push(newPoint);
    
    // Draw point circle
    const circle = new Circle({
      left: pointer.x - 5,
      top: pointer.y - 5,
      radius: 5,
      fill: this.color,
      stroke: '#ffffff',
      strokeWidth: 2,
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
          strokeWidth: 3,
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
    this.tempObjects.forEach(obj => this.canvas.remove(obj));
    this.tempObjects = [];
    
    // Return polygon data
    this.onComplete({
      shape: 'polygon',
      points: this.points.map(p => [p.x, p.y])
    });
    
    this.canvas.requestRenderAll();
  }
  
  /**
   * Cancel polygon drawing
   */
  cancel() {
    this.tempObjects.forEach(obj => this.canvas.remove(obj));
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
