import { useState, useEffect } from 'react';
import AnnotationsPanel from '@/components/shared/AnnotationsPanel';
import InferenceTemplateViewer from './InferenceTemplateViewer';
import { API_BASE_URL } from '@/config/api';
import '@/styles/InferenceRealtimeSettingsModal.css';
import type { Annotation } from '@/types';
import { validateRealtimeSettings, hasValidationErrors } from '@/utils/recipeValidation';

interface InferenceRealtimeSettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  runningRecipe: any; // Current running recipe data from load event
  onUpdate: (updatedData: any) => Promise<void>;
}

interface TemplateData {
  name: string;
  image_url: string;
  image_width?: number;
  image_height?: number;
  annotations: Annotation[];
  center_offset_threshold_left?: number;
  center_offset_threshold_right?: number;
}

export default function InferenceRealtimeSettingsModal({
  isOpen,
  onClose,
  runningRecipe,
  onUpdate
}: InferenceRealtimeSettingsModalProps) {
  const [activeTab, setActiveTab] = useState<'basic' | 'camera' | 'template'>('basic');
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});

  // Form data state
  const [formData, setFormData] = useState({
    delay_reject: 100,
    reject_pulse: 50,
    cameras: [] as any[],
    camera_templates: [] as any[]
  });

  // Template tab states
  const [selectedCameraForTemplate, setSelectedCameraForTemplate] = useState<string>('');
  const [selectedTemplateIndex, setSelectedTemplateIndex] = useState<number>(0);
  const [selectedAnnotation, setSelectedAnnotation] = useState<number | null>(null);

  // Load data from runningRecipe when modal opens
  useEffect(() => {
    if (isOpen && runningRecipe?.metadata) {
      const metadata = runningRecipe.metadata;

      setFormData({
        delay_reject: metadata.delay_reject || 100,
        reject_pulse: metadata.reject_pulse || 50,
        cameras: JSON.parse(JSON.stringify(metadata.cameras || [])), // Deep clone
        camera_templates: JSON.parse(JSON.stringify(metadata.camera_templates || []))
      });

      // Auto-select first camera with templates
      const firstCameraWithTemplates = (metadata.camera_templates || []).find(
        (ct: any) => ct.templates && ct.templates.length > 0
      );
      if (firstCameraWithTemplates) {
        setSelectedCameraForTemplate(firstCameraWithTemplates.camera_id);
        setSelectedTemplateIndex(0);
      }
    }
  }, [isOpen, runningRecipe]);

  const handleInputChange = (field: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleCameraChange = (cameraId: string, field: string, value: any) => {
    const parsed = parseFloat(value);
    const finalValue = isNaN(parsed) ? undefined : parsed;
    setFormData(prev => ({
      ...prev,
      cameras: prev.cameras.map(cam =>
        cam.camera_id === cameraId
          ? { ...cam, [field]: finalValue }
          : cam
      )
    }));
  };

  const handleTemplateThresholdChange = (field: 'center_offset_threshold_left' | 'center_offset_threshold_right', value: number) => {
    if (!selectedCameraForTemplate) return;

    setFormData(prev => {
      const updatedCameraTemplates = prev.camera_templates.map(ct => {
        if (ct.camera_id === selectedCameraForTemplate) {
          const updatedTemplates = [...ct.templates];
          if (updatedTemplates[selectedTemplateIndex]) {
            updatedTemplates[selectedTemplateIndex] = {
              ...updatedTemplates[selectedTemplateIndex],
              [field]: value
            };
          }
          return { ...ct, templates: updatedTemplates };
        }
        return ct;
      });
      return { ...prev, camera_templates: updatedCameraTemplates };
    });
  };

  const handleAnnotationsChange = (newAnnotations: Annotation[]) => {
    if (!selectedCameraForTemplate) return;

    setFormData(prev => {
      const updatedCameraTemplates = prev.camera_templates.map(ct => {
        if (ct.camera_id === selectedCameraForTemplate) {
          const updatedTemplates = [...ct.templates];
          if (updatedTemplates[selectedTemplateIndex]) {
            updatedTemplates[selectedTemplateIndex] = {
              ...updatedTemplates[selectedTemplateIndex],
              annotations: newAnnotations
            };
          }
          return { ...ct, templates: updatedTemplates };
        }
        return ct;
      });

      return {
        ...prev,
        camera_templates: updatedCameraTemplates
      };
    });
  };

  const getCurrentTemplateAnnotations = (): Annotation[] => {
    const cameraTemplate = formData.camera_templates.find(
      ct => ct.camera_id === selectedCameraForTemplate
    );
    if (!cameraTemplate) return [];

    const template = cameraTemplate.templates[selectedTemplateIndex];
    const annotations = template?.annotations || [];

    // Ensure all annotations have required fields with proper types
    return annotations.map((ann: any): Annotation => ({
      ...ann,
      type: ann.type || 'text', // Required: Default to 'text' if not specified
      shape: ann.shape || 'rectangle', // Required: Default to 'rectangle'
      conf: ann.conf ?? 0.5 // Required: Default to 0.85
    }));
  };

  const getCurrentTemplate = (): TemplateData | null => {
    const cameraTemplate = formData.camera_templates.find(
      ct => ct.camera_id === selectedCameraForTemplate
    );

    if (!cameraTemplate) return null;

    return cameraTemplate.templates[selectedTemplateIndex] || null;
  };

  const handleAnnotationTypeChange = (index: number, newType: string) => {
    const currentAnnotations = getCurrentTemplateAnnotations();
    const updated = [...currentAnnotations];
    if (updated[index]) {
      updated[index].type = newType as any;
    }
    handleAnnotationsChange(updated);
  };

  const handleAnnotationTextChange = (index: number, newText: string) => {
    const currentAnnotations = getCurrentTemplateAnnotations();
    const updated = [...currentAnnotations];
    if (updated[index]) {
      updated[index].text = newText;
    }
    handleAnnotationsChange(updated);
  };

  const handleAnnotationConfChange = (index: number, newConf: number) => {
    const currentAnnotations = getCurrentTemplateAnnotations();
    const updated = [...currentAnnotations];
    if (updated[index]) {
      updated[index].conf = newConf;
    }
    handleAnnotationsChange(updated);
  };

  const handleDeleteAnnotation = (index: number) => {
    const currentAnnotations = getCurrentTemplateAnnotations();
    const updated = currentAnnotations.filter((_, i) => i !== index);
    handleAnnotationsChange(updated);

    if (selectedAnnotation === index) {
      setSelectedAnnotation(null);
    } else if (selectedAnnotation !== null && selectedAnnotation > index) {
      setSelectedAnnotation(selectedAnnotation - 1);
    }
  };

  const handleSubmit = async () => {
    const validationErrors = validateRealtimeSettings({
      delay_reject: formData.delay_reject,
      reject_pulse: formData.reject_pulse,
      cameras: formData.cameras,
    });
    if (hasValidationErrors(validationErrors)) {
      setErrors(validationErrors);
      const keys = Object.keys(validationErrors);
      if (keys.some(k => ['delay_reject', 'reject_pulse'].includes(k))) {
        setActiveTab('basic');
      } else if (keys.some(k => k.startsWith('cameras'))) {
        setActiveTab('camera');
      }
      return;
    }
    setErrors({});

    setLoading(true);
    try {
      // Build update data (only changed fields)
      const updateData: any = {
        delay_reject: formData.delay_reject,
        reject_pulse: formData.reject_pulse,
        cameras: formData.cameras.map(cam => ({
          camera_id: cam.camera_id,
          exposure_time: cam.exposure_time,
          delay_trigger: cam.delay_trigger,
          delay_interval: cam.delay_interval,
          gain: cam.gain
        })),
        camera_templates: formData.camera_templates.map((ct: any) => ({
          camera_id: ct.camera_id,
          // Preserve function_type so the downstream DB save (PUT /recipes/{id})
          // doesn't fall back to the Pydantic default 'OCR'.
          function_type: ct.function_type ?? 'OCR',
          templates: ct.templates.map((tmpl: any) => ({
            annotations: tmpl.annotations,
            center_offset_threshold_left: tmpl.center_offset_threshold_left,
            center_offset_threshold_right: tmpl.center_offset_threshold_right,
            // Preserve color_config so Check_Color templates keep their HSV setup
            // when realtime settings get saved back to DB.
            color_config: tmpl.color_config ?? null,
          }))
        }))
      };

      await onUpdate(updateData);
      onClose();
    } catch (error: any) {
      console.error('Error updating recipe realtime:', error);
      // Error toast is handled in parent component (InferenceRealtime)
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="inference-settings-overlay">
      <div className="inference-settings-modal">
        <div className="modal-header">
          <h2>Realtime Settings</h2>
          <button className="close-btn" onClick={onClose} disabled={loading}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>
        </div>

        {/* <div className="modal-warning">
          ⚠️ Changes are temporary and NOT saved to database. Stop/Load recipe to restore original settings.
        </div> */}

        <div className="modal-tabs">
          <button
            className={`tab-btn ${activeTab === 'basic' ? 'active' : ''}`}
            onClick={() => setActiveTab('basic')}
          >
            Basic Info
          </button>
          <button
            className={`tab-btn ${activeTab === 'camera' ? 'active' : ''}`}
            onClick={() => setActiveTab('camera')}
          >
            Camera
          </button>
          <button
            className={`tab-btn ${activeTab === 'template' ? 'active' : ''}`}
            onClick={() => setActiveTab('template')}
          >
            Template
          </button>
        </div>

        <div className="modal-content modal-content-wide">
          {/* Basic Info Tab */}
          {activeTab === 'basic' && (
            <div className="tab-panel tab-panel-wide">
              <h3>Basic Information</h3>
              <div className="form-row">
                <div className="form-group">
                  <label>Delay Reject (ms)</label>
                  <input
                    type="number"
                    value={formData.delay_reject}
                    onChange={(e) => handleInputChange('delay_reject', parseFloat(e.target.value) || 0)}
                    step="0.1"
                    min="0"
                    className={errors.delay_reject ? 'error' : ''}
                  />
                  {errors.delay_reject && <span className="error-message">{errors.delay_reject}</span>}
                </div>
                <div className="form-group">
                  <label>Reject Pulse (ms)</label>
                  <input
                    type="number"
                    value={formData.reject_pulse}
                    onChange={(e) => handleInputChange('reject_pulse', parseFloat(e.target.value) || 0)}
                    step="0.1"
                    min="0"
                    className={errors.reject_pulse ? 'error' : ''}
                  />
                  {errors.reject_pulse && <span className="error-message">{errors.reject_pulse}</span>}
                </div>
              </div>
            </div>
          )}

          {/* Camera Tab */}
          {activeTab === 'camera' && (
            <div className="tab-panel tab-panel-wide">
              <h3>Camera Settings</h3>
              {formData.cameras.length === 0 ? (
                <div className="empty-state">No cameras configured</div>
              ) : (
                <div className="cameras-list">
                  {formData.cameras.map((camera, index) => (
                    <div key={camera.camera_id} className="camera-card">
                      <h4>Camera {index + 1}: {camera.camera_id}</h4>
                      <div className="camera-info">
                        <div><strong>Serial:</strong> {camera.serial_number}</div>
                        <div><strong>Location:</strong> {camera.location || 'N/A'}</div>
                      </div>
                      <div className="form-row">
                        <div className="form-group">
                          <label>Exposure Time (μs) *</label>
                          <input
                            type="number"
                            value={camera.exposure_time || 0}
                            onChange={(e) => handleCameraChange(camera.camera_id, 'exposure_time', e.target.value)}
                            step="0.1"
                            min="0"
                            className={errors[`cameras.${index}.exposure_time`] ? 'error' : ''}
                          />
                          {errors[`cameras.${index}.exposure_time`] && (
                            <span className="error-message">{errors[`cameras.${index}.exposure_time`]}</span>
                          )}
                        </div>
                        <div className="form-group">
                          <label>Delay Trigger (ms) *</label>
                          <input
                            type="number"
                            value={camera.delay_trigger || 0}
                            onChange={(e) => handleCameraChange(camera.camera_id, 'delay_trigger', e.target.value)}
                            step="0.1"
                            min="0"
                            className={errors[`cameras.${index}.delay_trigger`] ? 'error' : ''}
                          />
                          {errors[`cameras.${index}.delay_trigger`] && (
                            <span className="error-message">{errors[`cameras.${index}.delay_trigger`]}</span>
                          )}
                        </div>
                      </div>
                      <div className="form-row">
                        <div className="form-group">
                          <label>Delay Interval (ms)</label>
                          <input
                            type="number"
                            value={camera.delay_interval || 0}
                            onChange={(e) => handleCameraChange(camera.camera_id, 'delay_interval', e.target.value)}
                            step="10"
                            min="0"
                            className={errors[`cameras.${index}.delay_interval`] ? 'error' : ''}
                          />
                          {errors[`cameras.${index}.delay_interval`] && (
                            <span className="error-message">{errors[`cameras.${index}.delay_interval`]}</span>
                          )}
                        </div>
                        <div className="form-group">
                          <label>Gain *</label>
                          <input
                            type="number"
                            value={camera.gain || 0}
                            onChange={(e) => handleCameraChange(camera.camera_id, 'gain', e.target.value)}
                            step="0.1"
                            min="0"
                            className={errors[`cameras.${index}.gain`] ? 'error' : ''}
                          />
                          {errors[`cameras.${index}.gain`] && (
                            <span className="error-message">{errors[`cameras.${index}.gain`]}</span>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Template Tab */}
          {activeTab === 'template' && (
            <div className="tab-panel tab-panel-wide template-panel">
              <div className="template-header">
                <h3>Template Annotations</h3>
                <select
                  value={selectedCameraForTemplate}
                  onChange={(e) => {
                    setSelectedCameraForTemplate(e.target.value);
                    setSelectedTemplateIndex(0);
                  }}
                >
                  <option value="">-- Select Camera --</option>
                  {formData.camera_templates.map((ct: any) => (
                    <option key={ct.camera_id} value={ct.camera_id}>
                      {ct.camera_id}
                    </option>
                  ))}
                </select>
              </div>

              {!selectedCameraForTemplate ? (
                <div className="empty-state">Select a camera to view templates</div>
              ) : (
                <>
                  {/* Template selector */}
                  {formData.camera_templates.find(ct => ct.camera_id === selectedCameraForTemplate)?.templates?.length > 0 && (
                    <div className="template-selector">
                      {formData.camera_templates
                        .find((ct: any) => ct.camera_id === selectedCameraForTemplate)
                        ?.templates.map((tmpl: any, idx: number) => (
                          <button
                            key={idx}
                            className={`template-btn ${idx === selectedTemplateIndex ? 'active' : ''}`}
                            onClick={() => setSelectedTemplateIndex(idx)}
                          >
                            {tmpl.name || `Template ${idx + 1}`}
                          </button>
                        ))}
                    </div>
                  )}

                  {/* Center Offset Threshold */}
                  {getCurrentTemplate() && (
                    <div className="inference-template-threshold">
                      <span className="inference-threshold-label">Center Offset (px):</span>
                      <div className="inference-threshold-inputs-row">
                        <div className="inference-threshold-input-group">
                          <label>L</label>
                          <input
                            type="number"
                            min="0"
                            max="500"
                            value={getCurrentTemplate()?.center_offset_threshold_left ?? 50}
                            onChange={(e) => {
                              const value = parseFloat(e.target.value) || 50;
                              handleTemplateThresholdChange('center_offset_threshold_left', Math.max(0, Math.min(500, value)));
                            }}
                            className="inference-threshold-input"
                          />
                        </div>
                        <div className="inference-threshold-input-group">
                          <label>R</label>
                          <input
                            type="number"
                            min="0"
                            max="500"
                            value={getCurrentTemplate()?.center_offset_threshold_right ?? 50}
                            onChange={(e) => {
                              const value = parseFloat(e.target.value) || 50;
                              handleTemplateThresholdChange('center_offset_threshold_right', Math.max(0, Math.min(500, value)));
                            }}
                            className="inference-threshold-input"
                          />
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Template preview + Annotations panel */}
                  <div className="template-editor-layout">
                    <div className="template-preview">
                      {getCurrentTemplate()?.image_url ? (
                        <div className="canvas-container">
                          <InferenceTemplateViewer
                            imageUrl={`${API_BASE_URL}${getCurrentTemplate()!.image_url}`}
                            annotations={getCurrentTemplateAnnotations()}
                            selectedAnnotation={selectedAnnotation}
                            onSelectAnnotation={setSelectedAnnotation}
                          />
                          <div className="canvas-hint">
                            💡 Click on bounding box to select (read-only, cannot edit positions)
                          </div>
                        </div>
                      ) : (
                        <div className="no-template">
                          No template image
                          <div style={{ fontSize: '0.8rem', marginTop: '10px', color: '#9ca3af' }}>
                            {selectedCameraForTemplate ? (
                              <>
                                Camera: {selectedCameraForTemplate}<br/>
                                Template #{selectedTemplateIndex}<br/>
                                Total templates: {formData.camera_templates.find(ct => ct.camera_id === selectedCameraForTemplate)?.templates?.length || 0}
                              </>
                            ) : (
                              'Select a camera to view templates'
                            )}
                          </div>
                        </div>
                      )}
                    </div>

                    <div className="annotations-sidebar">
                      <AnnotationsPanel
                        annotations={getCurrentTemplateAnnotations() as any}
                        selectedAnnotation={selectedAnnotation}
                        onSelectAnnotation={setSelectedAnnotation}
                        onAnnotationTypeChange={handleAnnotationTypeChange}
                        onAnnotationTextChange={handleAnnotationTextChange}
                        onAnnotationConfChange={handleAnnotationConfChange}
                        onDeleteAnnotation={handleDeleteAnnotation}
                        imageWidth={getCurrentTemplate()?.image_width}
                        imageHeight={getCurrentTemplate()?.image_height}
                        readOnlyType={true}
                        hideMetadata={true}
                      />
                    </div>
                  </div>
                </>
              )}
            </div>
          )}
        </div>

        <div className="modal-footer">
          <button className="btn-cancel" onClick={onClose} disabled={loading}>
            Cancel
          </button>
          <button className="btn-submit" onClick={handleSubmit} disabled={loading}>
            {loading ? 'Updating...' : 'Update (Realtime)'}
          </button>
        </div>
      </div>
    </div>
  );
}
