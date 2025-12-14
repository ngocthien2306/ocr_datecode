/**
 * Fabric.js actions for canvas operations
 */

/**
 * Delete selected objects from canvas
 * @param {fabric.Canvas} canvas 
 */
export const deleteSelected = (canvas) => {
  if (!canvas) return;
  
  const activeObjects = canvas.getActiveObjects();
  if (activeObjects.length === 0) return;
  
  activeObjects.forEach(obj => {
    if (obj.annotationIndex !== undefined) {
      canvas.remove(obj);
      // Also remove associated labels
      const labels = canvas.getObjects().filter(o => 
        o.isLabel && 
        Math.abs(o.left - obj.left) < 50 && 
        Math.abs(o.top - obj.top + 25) < 50
      );
      labels.forEach(label => canvas.remove(label));
    }
  });
  
  canvas.discardActiveObject();
  canvas.requestRenderAll();
};

/**
 * Delete object by annotation index
 * @param {fabric.Canvas} canvas 
 * @param {number} index 
 */
export const deleteByIndex = (canvas, index) => {
  if (!canvas) return;
  
  const objects = canvas.getObjects().filter(obj => obj.annotationIndex === index);
  objects.forEach(obj => canvas.remove(obj));
  
  // Remove associated labels and edit points
  const related = canvas.getObjects().filter(obj => 
    (obj.isLabel || obj.isPolygonEditPoint) && 
    obj.annotationIndex === index
  );
  related.forEach(obj => canvas.remove(obj));
  
  canvas.requestRenderAll();
};

/**
 * Clear all drawing helpers (temp lines, points, etc.)
 * @param {fabric.Canvas} canvas 
 * @param {Array} tempObjects - Array of temp objects to remove
 */
export const clearDrawingHelpers = (canvas, tempObjects) => {
  if (!canvas) return;
  
  tempObjects.forEach(obj => canvas.remove(obj));
  canvas.requestRenderAll();
};

/**
 * Highlight object by index
 * @param {fabric.Canvas} canvas 
 * @param {number} index 
 */
export const highlightObject = (canvas, index) => {
  if (!canvas) return;
  
  const obj = canvas.getObjects().find(o => o.annotationIndex === index);
  if (obj) {
    canvas.setActiveObject(obj);
    canvas.requestRenderAll();
  }
};

/**
 * Deselect all objects
 * @param {fabric.Canvas} canvas 
 */
export const deselectAll = (canvas) => {
  if (!canvas) return;
  canvas.discardActiveObject();
  canvas.requestRenderAll();
};

/**
 * Bring object to front
 * @param {fabric.Canvas} canvas 
 * @param {fabric.Object} obj 
 */
export const bringToFront = (canvas, obj) => {
  if (!canvas || !obj) return;
  canvas.bringToFront(obj);
  canvas.requestRenderAll();
};

/**
 * Send object to back
 * @param {fabric.Canvas} canvas 
 * @param {fabric.Object} obj 
 */
export const sendToBack = (canvas, obj) => {
  if (!canvas || !obj) return;
  canvas.sendToBack(obj);
  canvas.requestRenderAll();
};
