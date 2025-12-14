import React, { useRef, useState, useEffect } from 'react';
import Board from '@dansreis/react-canvas-annotator';

/**
 * Template Editor using react-canvas-annotator library
 * This is an alternative implementation to TemplateEditorRefactored
 */
export default function TemplateEditorWithLibrary({ 
  templateImage, 
  annotations, 
  onAnnotationsChange,
  selectedAnnotation,
  onSelectAnnotation 
}) {
  const boardRef = useRef(null);
  const [items, setItems] = useState([]);

  // Convert annotations to react-canvas-annotator format
  useEffect(() => {
    if (!annotations) return;

    const convertedItems = annotations.map((ann, index) => {
      const coords = ann.shape === 'rectangle' 
        ? [
            { x: ann.x, y: ann.y },
            { x: ann.x + ann.width, y: ann.y },
            { x: ann.x + ann.width, y: ann.y + ann.height },
            { x: ann.x, y: ann.y + ann.height }
          ]
        : ann.points.map(p => ({ x: p[0], y: p[1] }));

      return {
        id: ann.id || `annotation-${index}`,
        category: ann.type || 'text',
        color: getColorForType(ann.type),
        value: ann.text || '',
        coords: coords
      };
    });

    setItems(convertedItems);
  }, [annotations]);

  // Get color based on annotation type
  const getColorForType = (type) => {
    const colors = {
      'text': '#50fa7b',
      'barcode': '#8be9fd',
      'template': '#ffb86c',
      'crop_area': '#ff79c6',
      'datecode': '#bd93f9'
    };
    return colors[type] || '#50fa7b';
  };

  // Handle changes from Board component
  const handleItemsChange = (newItems) => {
    // Convert back to our annotation format
    const convertedAnnotations = newItems.map((item) => {
      const isRectangle = item.coords.length === 4 && 
        item.coords[0].x === item.coords[3].x &&
        item.coords[1].x === item.coords[2].x;

      if (isRectangle) {
        const x = Math.min(item.coords[0].x, item.coords[2].x);
        const y = Math.min(item.coords[0].y, item.coords[2].y);
        const width = Math.abs(item.coords[2].x - item.coords[0].x);
        const height = Math.abs(item.coords[2].y - item.coords[0].y);

        return {
          id: item.id,
          type: item.category,
          text: item.value || '',
          shape: 'rectangle',
          x,
          y,
          width,
          height
        };
      } else {
        return {
          id: item.id,
          type: item.category,
          text: item.value || '',
          shape: 'polygon',
          points: item.coords.map(c => [c.x, c.y])
        };
      }
    });

    onAnnotationsChange?.(convertedAnnotations);
  };

  return (
    <div style={{ 
      width: '100%', 
      height: '600px',
      border: '1px solid #ccc',
      borderRadius: '4px',
      overflow: 'hidden'
    }}>
      <Board
        ref={boardRef}
        image={{ 
          name: 'template', 
          src: templateImage 
        }}
        items={items}
        onChange={handleItemsChange}
      />
    </div>
  );
}
