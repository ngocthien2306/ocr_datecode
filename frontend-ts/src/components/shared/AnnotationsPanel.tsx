import React, { useState } from 'react';
import '@/styles/AnnotationsPanel.css';

interface AnnotationType {
  value: string;
  label: string;
  color: string;
  needsText: boolean;
}

const ANNOTATION_TYPES: AnnotationType[] = [
  { value: 'text', label: 'Text OCR', color: '#50fa7b', needsText: true },
  { value: 'template', label: 'Template Match', color: '#ff5555', needsText: false },
  { value: 'crop_area', label: 'Crop Area', color: '#ff64ff', needsText: false },
  { value: 'datecode', label: 'Date Code', color: '#5096ff', needsText: true },
  { value: 'product', label: 'Product', color: '#7513dd', needsText: false },
  { value: 'label', label: 'Label', color: '#ad6df1', needsText: false },
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
  conf: number;
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
  onAnnotationConfChange?: (index: number, conf: number) => void;
  onDeleteAnnotation?: (index: number) => void;
  onAutoSegment?: (index: number) => void;
  segmenting?: boolean;
  fabricCanvasRef?: React.RefObject<FabricCanvas>;
  imageWidth?: number;
  imageHeight?: number;
  readOnlyType?: boolean;
  hideMetadata?: boolean;
}

const AnnotationsPanel: React.FC<AnnotationsPanelProps> = ({
  annotations,
  selectedAnnotation,
  onSelectAnnotation,
  onAnnotationTypeChange,
  onAnnotationTextChange,
  onAnnotationConfChange,
  onDeleteAnnotation,
  onAutoSegment,
  segmenting = false,
  fabricCanvasRef,
  imageWidth,
  imageHeight,
  readOnlyType = false,
  hideMetadata = false
}) => {
  const [showMetadata, setShowMetadata] = useState(false);

  const handleAnnotationClick = (index: number) => {
    onSelectAnnotation?.(index);

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
      <div className="annotations-panel-header">
        <h3>Bounding Boxes ({annotations.length})</h3>
      </div>
      <div className="annotations-list">
        {annotations.map((ann, index) => {
          const typeConfig = ANNOTATION_TYPES.find(t => t.value === ann.type);
          const color = typeConfig?.color || '#ffffff';
          const expanded = selectedAnnotation === index;

          return (
            <div
              key={index}
              className={`annotation-item ${selectedAnnotation === index ? 'selected' : ''} ${expanded ? 'expanded' : 'collapsed'}`}
              onClick={() => handleAnnotationClick(index)}
            >
              {/* Compact header row - always visible */}
              <div className="annotation-header">
                <span className="annotation-color" style={{ backgroundColor: color }} />
                <span className="annotation-index">#{index + 1}</span>
                <span className="annotation-type-badge" style={{ color, borderColor: color + '60' }}>
                  {typeConfig?.label || ann.type}
                </span>

                {/* Text preview in collapsed mode */}
                {!expanded && typeConfig?.needsText && ann.text && (
                  <span className="annotation-text-preview" title={ann.text}>
                    {ann.text}
                  </span>
                )}

                <div className="annotation-header-actions">
                  {/* Auto Segment icon button */}
                  {onAutoSegment && ann.shape === 'rectangle' && (ann.type === 'text' || ann.type === 'datecode') && (
                    <button
                      type="button"
                      className="auto-segment-icon-btn"
                      disabled={segmenting}
                      title={segmenting ? 'Segmenting...' : 'Auto Segment'}
                      onClick={(e) => {
                        e.stopPropagation();
                        onAutoSegment(index);
                      }}
                    >
                      {segmenting ? (
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="spin">
                          <path d="M12 2V6M12 18V22M4.93 4.93L7.76 7.76M16.24 16.24L19.07 19.07M2 12H6M18 12H22M4.93 19.07L7.76 16.24M16.24 7.76L19.07 4.93" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                        </svg>
                      ) : (
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                          <rect x="3" y="3" width="7" height="7" stroke="currentColor" strokeWidth="2" rx="1"/>
                          <rect x="14" y="3" width="7" height="7" stroke="currentColor" strokeWidth="2" rx="1"/>
                          <rect x="3" y="14" width="7" height="7" stroke="currentColor" strokeWidth="2" rx="1"/>
                          <rect x="14" y="14" width="7" height="7" stroke="currentColor" strokeWidth="2" rx="1"/>
                        </svg>
                      )}
                    </button>
                  )}

                  {/* Delete button */}
                  <button
                    type="button"
                    className="delete-annotation-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteAnnotation?.(index);
                    }}
                    title="Delete"
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                      <line x1="18" y1="6" x2="6" y2="18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                      <line x1="6" y1="6" x2="18" y2="18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                    </svg>
                  </button>
                </div>
              </div>

              {/* Expanded details */}
              {expanded && (
                <div className="annotation-details">
                  <select
                    value={ann.type}
                    onChange={(e) => onAnnotationTypeChange?.(index, e.target.value)}
                    className="annotation-type-select"
                    onClick={(e) => e.stopPropagation()}
                    disabled={readOnlyType}
                  >
                    {ANNOTATION_TYPES.map(type => (
                      <option key={type.value} value={type.value}>
                        {type.label}
                      </option>
                    ))}
                  </select>

                  {typeConfig?.needsText && (
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
                        <span className="required-indicator">*</span>
                      )}
                    </div>
                  )}

                  <div className="annotation-conf-field">
                    <label className="conf-label">Conf:</label>
                    <input
                      type="number"
                      min="0"
                      max="1"
                      step="0.01"
                      value={ann.conf ?? 0.5}
                      onChange={(e) => {
                        const value = parseFloat(e.target.value);
                        if (!isNaN(value) && value >= 0 && value <= 1) {
                          onAnnotationConfChange?.(index, value);
                        }
                      }}
                      onBlur={(e) => {
                        const value = parseFloat(e.target.value);
                        if (isNaN(value) || value < 0) {
                          onAnnotationConfChange?.(index, 0);
                        } else if (value > 1) {
                          onAnnotationConfChange?.(index, 1);
                        }
                      }}
                      className="annotation-conf-input"
                      onClick={(e) => e.stopPropagation()}
                    />
                  </div>

                  {/* Collapsible metadata */}
                  {!hideMetadata && (
                    <div className="annotation-metadata">
                      <button
                        type="button"
                        className="metadata-toggle"
                        onClick={(e) => {
                          e.stopPropagation();
                          setShowMetadata(prev => !prev);
                        }}
                      >
                        <svg
                          width="10" height="10" viewBox="0 0 24 24" fill="none"
                          className={`chevron ${showMetadata ? 'open' : ''}`}
                        >
                          <path d="M9 18L15 12L9 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                        <span>Details</span>
                      </button>
                      {showMetadata && (
                        <div className="metadata-content">
                          <div className="annotation-info">
                            <span className="info-label">Shape:</span>
                            <span className="info-value">{ann.shape}</span>
                          </div>
                          {ann.shape === 'rectangle' && ann.width !== undefined && ann.height !== undefined && (
                            <div className="annotation-info">
                              <span className="info-label">Size:</span>
                              <span className="info-value">
                                {imageWidth && imageHeight
                                  ? `${Math.round(ann.width * imageWidth)} x ${Math.round(ann.height * imageHeight)}`
                                  : `${ann.width.toFixed(3)} x ${ann.height.toFixed(3)}`
                                }
                              </span>
                            </div>
                          )}
                          {ann.shape === 'polygon' && ann.points && (
                            <div className="annotation-info">
                              <span className="info-label">Points:</span>
                              <span className="info-value">{ann.points.length}</span>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}

        {annotations.length === 0 && (
          <div className="empty-message">
            No annotations yet. Draw bounding boxes on the canvas.
          </div>
        )}
      </div>
    </div>
  );
};

export default AnnotationsPanel;
