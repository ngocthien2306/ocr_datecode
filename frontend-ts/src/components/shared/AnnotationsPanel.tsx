import React from 'react';
import '@/styles/AnnotationsPanel.css';

interface AnnotationType {
  value: string;
  label: string;
  color: string;
  needsText: boolean;
}

const ANNOTATION_TYPES: AnnotationType[] = [
  { value: 'text', label: 'Text OCR', color: '#50fa7b', needsText: true },
  { value: 'barcode', label: 'Barcode', color: '#ffdc5c', needsText: false },
  { value: 'template', label: 'Template Match', color: '#ff5555', needsText: false },
  { value: 'crop_area', label: 'Crop Area', color: '#ff64ff', needsText: false },
  { value: 'datecode', label: 'Date Code', color: '#5096ff', needsText: true }
];

interface Point {
  x: number;
  y: number;
}

interface Annotation {
  type: string;
  shape: 'rectangle' | 'polygon';
  width?: number;
  height?: number;
  points?: Point[];
  text?: string;
}

interface FabricCanvas {
  getObjects: () => any[];
  setActiveObject: (obj: any) => void;
  requestRenderAll: () => void;
}

interface AnnotationsPanelProps {
  annotations: Annotation[];
  selectedAnnotation: number | null;
  onSelectAnnotation?: (index: number) => void;
  onAnnotationTypeChange?: (index: number, type: string) => void;
  onAnnotationTextChange?: (index: number, text: string) => void;
  onDeleteAnnotation?: (index: number) => void;
  fabricCanvasRef?: React.RefObject<FabricCanvas>;
}

const AnnotationsPanel: React.FC<AnnotationsPanelProps> = ({ 
  annotations, 
  selectedAnnotation, 
  onSelectAnnotation,
  onAnnotationTypeChange, 
  onAnnotationTextChange,
  onDeleteAnnotation,
  fabricCanvasRef 
}) => {
  
  const handleAnnotationClick = (index: number) => {
    onSelectAnnotation?.(index);
    
    // Also select on canvas if fabricCanvasRef is provided
    if (fabricCanvasRef?.current) {
      const obj = fabricCanvasRef.current.getObjects().find(
        (o: any) => o.annotationIndex === index
      );
      if (obj) {
        fabricCanvasRef.current.setActiveObject(obj);
        fabricCanvasRef.current.requestRenderAll();
      }
    }
  };

  return (
    <div className="annotations-panel">
      <h3>Bounding Boxes ({annotations.length})</h3>
      <div className="annotations-list">
        {annotations.map((ann, index) => (
          <div
            key={index}
            className={`annotation-item ${selectedAnnotation === index ? 'selected' : ''}`}
            onClick={() => handleAnnotationClick(index)}
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
              onChange={(e) => onAnnotationTypeChange?.(index, e.target.value)}
              className="annotation-type-select"
              onClick={(e) => e.stopPropagation()}
            >
              {ANNOTATION_TYPES.map(type => (
                <option key={type.value} value={type.value}>
                  {type.label}
                </option>
              ))}
            </select>

            {/* Text input for types that need text */}
            {ANNOTATION_TYPES.find(t => t.value === ann.type)?.needsText && (
              <div className="annotation-text-field">
                <input
                  type="text"
                  value={ann.text || ''}
                  onChange={(e) => onAnnotationTextChange?.(index, e.target.value)}
                  placeholder="Enter text content... *"
                  className={`annotation-text-input ${(!ann.text || ann.text.trim() === '') ? 'required-empty' : ''}`}
                  onClick={(e) => e.stopPropagation()}
                  required
                />
                {(!ann.text || ann.text.trim() === '') && (
                  <span className="required-indicator">* Required</span>
                )}
              </div>
            )}

            <div className="annotation-info">
              <span className="info-label">Shape:</span>
              <span className="info-value">{ann.shape}</span>
            </div>

            {ann.shape === 'rectangle' && ann.width !== undefined && ann.height !== undefined && (
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

            <button
              type="button"
              className="delete-annotation-btn"
              onClick={(e) => {
                e.stopPropagation();
                onDeleteAnnotation?.(index);
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
  );
};

export default AnnotationsPanel;
