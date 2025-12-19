import React, { useState, useEffect, ChangeEvent, FormEvent } from 'react';
import { camerasAPI } from '@/services/api';
import { useToast } from '@/contexts/ToastContext';
import type { Receipt, RecipeCreate, RecipeUpdate, Camera, CameraSettings, ModelThresholds } from '@/types';
import '@/styles/RecipeFormModal.css';

interface RecipeFormModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: RecipeCreate | RecipeUpdate) => Promise<void>;
  recipe?: Receipt | null;
  mode?: 'create' | 'edit';
}

interface FormData {
  name: string;
  product_code: string;
  description: string;
  delay_reject: number;
  is_active: boolean;
  cameras: string[];
  camera_settings: CameraSettings;
  model_thresholds: ModelThresholds;
}

const RecipeFormModal: React.FC<RecipeFormModalProps> = ({ 
  isOpen, 
  onClose, 
  onSubmit, 
  recipe = null, 
  mode = 'create' 
}) => {
  const [activeTab, setActiveTab] = useState('basic');
  const toast = useToast();
  const [loading, setLoading] = useState(false);
  const [availableCameras, setAvailableCameras] = useState<Camera[]>([]);
  const [formData, setFormData] = useState<FormData>({
    name: '',
    product_code: '',
    description: '',
    delay_reject: 100.0,
    is_active: true,
    cameras: [],
    camera_settings: {
      exposure_time: 50.0,
      gain: 1.0,
    },
    model_thresholds: {
      detection_threshold: 0.5,
      ocr_threshold: 0.5,
    }
  });

  useEffect(() => {
    if (recipe && mode === 'edit' && isOpen) {
      setFormData({
        name: recipe.name || '',
        product_code: recipe.productCode || '',
        description: recipe.description || '',
        delay_reject: recipe.delay_reject || 100.0,
        is_active: recipe.is_active,
        cameras: recipe.cameras || [],
        camera_settings: recipe.cameraSettings || {
          exposure_time: 50.0,
          gain: 1.0,
        },
        model_thresholds: recipe.modelThresholds || {
          detection_threshold: 0.5,
          ocr_threshold: 0.5,
        }
      });
    } else if (mode === 'create' && isOpen) {
      setFormData({
        name: '',
        product_code: '',
        description: '',
        delay_reject: 100.0,
        is_active: true,
        cameras: [],
        camera_settings: {
          exposure_time: 50.0,
          gain: 1.0,
        },
        model_thresholds: {
          detection_threshold: 0.5,
          ocr_threshold: 0.5,
        }
      });
    }
  }, [recipe, mode, isOpen]);

  useEffect(() => {
    if (isOpen) {
      loadAvailableCameras();
    }
  }, [isOpen]);

  const loadAvailableCameras = async () => {
    try {
      const data = await camerasAPI.getAllCameras(0, 100);
      setAvailableCameras(data);
    } catch (error) {
      console.error('Failed to load cameras:', error);
      toast.error('Failed to load cameras');
    }
  };

  const handleInputChange = (e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const target = e.target as HTMLInputElement;
    const { name, value, type } = target;
    const checked = target.checked;
    
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleCameraSettingChange = (field: keyof CameraSettings, value: string) => {
    setFormData(prev => ({
      ...prev,
      camera_settings: {
        ...prev.camera_settings,
        [field]: parseFloat(value) || 0
      }
    }));
  };

  const handleModelThresholdChange = (field: keyof ModelThresholds, value: string) => {
    setFormData(prev => ({
      ...prev,
      model_thresholds: {
        ...prev.model_thresholds,
        [field]: parseFloat(value) || 0
      }
    }));
  };

  const handleCameraToggle = (cameraId: string) => {
    setFormData(prev => ({
      ...prev,
      cameras: prev.cameras.includes(cameraId)
        ? prev.cameras.filter(id => id !== cameraId)
        : [...prev.cameras, cameraId]
    }));
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      await onSubmit(formData);
      handleClose();
    } catch (error: any) {
      console.error('Submit error:', error);
      toast.error(error.response?.data?.detail || 'Failed to save recipe');
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    setActiveTab('basic');
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={handleClose}>
      <div className="modal-content recipe-form-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>{mode === 'create' ? 'Create New Recipe' : 'Edit Recipe'}</h2>
          <button className="close-btn" onClick={handleClose}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2"/>
            </svg>
          </button>
        </div>

        {/* Tabs */}
        <div className="tabs">
          <button 
            className={`tab ${activeTab === 'basic' ? 'active' : ''}`}
            onClick={() => setActiveTab('basic')}
          >
            Basic Info
          </button>
          <button 
            className={`tab ${activeTab === 'cameras' ? 'active' : ''}`}
            onClick={() => setActiveTab('cameras')}
          >
            Cameras
          </button>
          <button 
            className={`tab ${activeTab === 'settings' ? 'active' : ''}`}
            onClick={() => setActiveTab('settings')}
          >
            Settings
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="modal-body">
            {/* Basic Info Tab */}
            {activeTab === 'basic' && (
              <div className="tab-content">
                <div className="form-group">
                  <label>Recipe Name *</label>
                  <input
                    type="text"
                    name="name"
                    value={formData.name}
                    onChange={handleInputChange}
                    placeholder="Enter recipe name"
                    required
                  />
                </div>

                <div className="form-group">
                  <label>Product Code *</label>
                  <input
                    type="text"
                    name="product_code"
                    value={formData.product_code}
                    onChange={handleInputChange}
                    placeholder="Enter product code"
                    required
                  />
                </div>

                <div className="form-group">
                  <label>Description</label>
                  <textarea
                    name="description"
                    value={formData.description}
                    onChange={handleInputChange}
                    placeholder="Enter description"
                    rows={4}
                  />
                </div>

                <div className="form-group">
                  <label>Delay Reject (ms)</label>
                  <input
                    type="number"
                    name="delay_reject"
                    value={formData.delay_reject}
                    onChange={handleInputChange}
                    step="0.1"
                  />
                </div>

                <div className="form-group checkbox-group">
                  <label>
                    <input
                      type="checkbox"
                      name="is_active"
                      checked={formData.is_active}
                      onChange={handleInputChange}
                    />
                    <span>Active</span>
                  </label>
                </div>
              </div>
            )}

            {/* Cameras Tab */}
            {activeTab === 'cameras' && (
              <div className="tab-content">
                <h3>Select Cameras</h3>
                <div className="cameras-list">
                  {availableCameras.map(camera => (
                    <div key={camera.id} className="camera-item">
                      <label>
                        <input
                          type="checkbox"
                          checked={formData.cameras.includes(camera.id)}
                          onChange={() => handleCameraToggle(camera.id)}
                        />
                        <span className="camera-info">
                          <strong>{camera.name}</strong>
                          <small>{camera.camera_id}</small>
                        </span>
                      </label>
                    </div>
                  ))}
                </div>

                <div className="selected-count">
                  Selected: {formData.cameras.length} camera(s)
                </div>
              </div>
            )}

            {/* Settings Tab */}
            {activeTab === 'settings' && (
              <div className="tab-content">
                <h3>Camera Settings</h3>
                <div className="form-grid">
                  <div className="form-group">
                    <label>Exposure Time (ms)</label>
                    <input
                      type="number"
                      value={formData.camera_settings.exposure_time}
                      onChange={(e) => handleCameraSettingChange('exposure_time', e.target.value)}
                      step="0.1"
                      min="0"
                    />
                  </div>

                  <div className="form-group">
                    <label>Gain</label>
                    <input
                      type="number"
                      value={formData.camera_settings.gain}
                      onChange={(e) => handleCameraSettingChange('gain', e.target.value)}
                      step="0.1"
                      min="0"
                    />
                  </div>
                </div>

                <h3>Model Thresholds</h3>
                <div className="form-grid">
                  <div className="form-group">
                    <label>Detection Threshold</label>
                    <input
                      type="number"
                      value={formData.model_thresholds.detection_threshold}
                      onChange={(e) => handleModelThresholdChange('detection_threshold', e.target.value)}
                      step="0.01"
                      min="0"
                      max="1"
                    />
                    <small>{(formData.model_thresholds.detection_threshold! * 100).toFixed(1)}%</small>
                  </div>

                  <div className="form-group">
                    <label>OCR Threshold</label>
                    <input
                      type="number"
                      value={formData.model_thresholds.ocr_threshold}
                      onChange={(e) => handleModelThresholdChange('ocr_threshold', e.target.value)}
                      step="0.01"
                      min="0"
                      max="1"
                    />
                    <small>{(formData.model_thresholds.ocr_threshold! * 100).toFixed(1)}%</small>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="modal-footer">
            <button type="button" className="btn btn-secondary" onClick={handleClose}>
              Cancel
            </button>
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? 'Saving...' : mode === 'create' ? 'Create Recipe' : 'Update Recipe'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default RecipeFormModal;
