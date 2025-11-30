import React, { useState, useEffect } from 'react';
import '../styles/RecipeFormModal.css';

export default function RecipeFormModal({ isOpen, onClose, onSubmit, recipe = null, mode = 'create' }) {
  const [formData, setFormData] = useState({
    name: '',
    product_code: '',
    description: '',
    is_active: true,
    camera_settings: {
      exposure_time: 50.0,
      delay_trigger: 100.0,
      gain: 1.0,
      brightness: 0.5,
      contrast: 1.0
    },
    model_thresholds: {
      detection_threshold: 0.5,
      recognition_threshold: 0.5,
      min_text_size: 10,
      max_text_size: 200
    },
    template_config: null,
    roi_config: null
  });

  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);

  // Load recipe data when editing
  useEffect(() => {
    if (recipe && mode === 'edit') {
      setFormData({
        name: recipe.name || '',
        product_code: recipe.productCode || recipe.product_code || '',
        description: recipe.description || '',
        is_active: recipe.is_active !== undefined ? recipe.is_active : (recipe.status === 'Active'),
        camera_settings: recipe.cameraSettings || recipe.camera_settings || {
          exposure_time: 50.0,
          delay_trigger: 100.0,
          gain: 1.0,
          brightness: 0.5,
          contrast: 1.0
        },
        model_thresholds: recipe.modelThresholds || recipe.model_thresholds || {
          detection_threshold: 0.5,
          recognition_threshold: 0.5,
          min_text_size: 10,
          max_text_size: 200
        },
        template_config: recipe.template_config || null,
        roi_config: recipe.roi_config || null
      });
    }
  }, [recipe, mode]);

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleCameraSettingChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      camera_settings: {
        ...prev.camera_settings,
        [field]: parseFloat(value) || 0
      }
    }));
  };

  const handleModelThresholdChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      model_thresholds: {
        ...prev.model_thresholds,
        [field]: field.includes('threshold') ? parseFloat(value) || 0 : parseInt(value) || 0
      }
    }));
  };

  const validateForm = () => {
    const newErrors = {};

    if (!formData.name.trim()) {
      newErrors.name = 'Recipe name is required';
    }

    if (!formData.product_code.trim()) {
      newErrors.product_code = 'Product code is required';
    }

    if (formData.camera_settings.exposure_time <= 0) {
      newErrors.exposure_time = 'Exposure time must be greater than 0';
    }

    if (formData.camera_settings.delay_trigger < 0) {
      newErrors.delay_trigger = 'Delay trigger cannot be negative';
    }

    if (formData.model_thresholds.detection_threshold < 0 || formData.model_thresholds.detection_threshold > 1) {
      newErrors.detection_threshold = 'Detection threshold must be between 0 and 1';
    }

    if (formData.model_thresholds.recognition_threshold < 0 || formData.model_thresholds.recognition_threshold > 1) {
      newErrors.recognition_threshold = 'Recognition threshold must be between 0 and 1';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    setLoading(true);
    try {
      await onSubmit(formData);
      handleClose();
    } catch (error) {
      console.error('Error submitting recipe:', error);
      setErrors({ submit: error.response?.data?.detail || 'Failed to save recipe' });
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    setFormData({
      name: '',
      product_code: '',
      description: '',
      is_active: true,
      camera_settings: {
        exposure_time: 50.0,
        delay_trigger: 100.0,
        gain: 1.0,
        brightness: 0.5,
        contrast: 1.0
      },
      model_thresholds: {
        detection_threshold: 0.5,
        recognition_threshold: 0.5,
        min_text_size: 10,
        max_text_size: 200
      },
      template_config: null,
      roi_config: null
    });
    setErrors({});
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
              <line x1="18" y1="6" x2="6" y2="18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              <line x1="6" y1="6" x2="18" y2="18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>
        </div>

        <form onSubmit={handleSubmit} className="recipe-form">
          {errors.submit && (
            <div className="error-message global-error">{errors.submit}</div>
          )}

          {/* Basic Information */}
          <div className="form-section">
            <h3>Basic Information</h3>
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="name">Recipe Name <span className="required">*</span></label>
                <input
                  type="text"
                  id="name"
                  name="name"
                  value={formData.name}
                  onChange={handleInputChange}
                  placeholder="Enter recipe name"
                  className={errors.name ? 'error' : ''}
                />
                {errors.name && <span className="error-message">{errors.name}</span>}
              </div>

              <div className="form-group">
                <label htmlFor="product_code">Product Code <span className="required">*</span></label>
                <input
                  type="text"
                  id="product_code"
                  name="product_code"
                  value={formData.product_code}
                  onChange={handleInputChange}
                  placeholder="Enter product code"
                  className={errors.product_code ? 'error' : ''}
                />
                {errors.product_code && <span className="error-message">{errors.product_code}</span>}
              </div>
            </div>

            <div className="form-group">
              <label htmlFor="description">Description</label>
              <textarea
                id="description"
                name="description"
                value={formData.description}
                onChange={handleInputChange}
                placeholder="Enter recipe description"
                rows="3"
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
                <span>Active Recipe</span>
              </label>
            </div>
          </div>

          {/* Camera Settings */}
          <div className="form-section">
            <h3>Camera Settings</h3>
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="exposure_time">Exposure Time (ms) <span className="required">*</span></label>
                <input
                  type="number"
                  id="exposure_time"
                  value={formData.camera_settings.exposure_time}
                  onChange={(e) => handleCameraSettingChange('exposure_time', e.target.value)}
                  step="0.1"
                  min="0"
                  className={errors.exposure_time ? 'error' : ''}
                />
                {errors.exposure_time && <span className="error-message">{errors.exposure_time}</span>}
              </div>

              <div className="form-group">
                <label htmlFor="delay_trigger">Delay Trigger (ms) <span className="required">*</span></label>
                <input
                  type="number"
                  id="delay_trigger"
                  value={formData.camera_settings.delay_trigger}
                  onChange={(e) => handleCameraSettingChange('delay_trigger', e.target.value)}
                  step="0.1"
                  min="0"
                  className={errors.delay_trigger ? 'error' : ''}
                />
                {errors.delay_trigger && <span className="error-message">{errors.delay_trigger}</span>}
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label htmlFor="gain">Gain</label>
                <input
                  type="number"
                  id="gain"
                  value={formData.camera_settings.gain}
                  onChange={(e) => handleCameraSettingChange('gain', e.target.value)}
                  step="0.1"
                  min="0"
                />
              </div>

              <div className="form-group">
                <label htmlFor="brightness">Brightness</label>
                <input
                  type="number"
                  id="brightness"
                  value={formData.camera_settings.brightness}
                  onChange={(e) => handleCameraSettingChange('brightness', e.target.value)}
                  step="0.1"
                  min="0"
                  max="1"
                />
              </div>

              <div className="form-group">
                <label htmlFor="contrast">Contrast</label>
                <input
                  type="number"
                  id="contrast"
                  value={formData.camera_settings.contrast}
                  onChange={(e) => handleCameraSettingChange('contrast', e.target.value)}
                  step="0.1"
                  min="0"
                />
              </div>
            </div>
          </div>

          {/* Model Thresholds */}
          <div className="form-section">
            <h3>Model Thresholds</h3>
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="detection_threshold">
                  Detection Threshold <span className="required">*</span>
                  <span className="hint"> (0.0 - 1.0)</span>
                </label>
                <input
                  type="number"
                  id="detection_threshold"
                  value={formData.model_thresholds.detection_threshold}
                  onChange={(e) => handleModelThresholdChange('detection_threshold', e.target.value)}
                  step="0.01"
                  min="0"
                  max="1"
                  className={errors.detection_threshold ? 'error' : ''}
                />
                {errors.detection_threshold && <span className="error-message">{errors.detection_threshold}</span>}
              </div>

              <div className="form-group">
                <label htmlFor="recognition_threshold">
                  Recognition Threshold <span className="required">*</span>
                  <span className="hint"> (0.0 - 1.0)</span>
                </label>
                <input
                  type="number"
                  id="recognition_threshold"
                  value={formData.model_thresholds.recognition_threshold}
                  onChange={(e) => handleModelThresholdChange('recognition_threshold', e.target.value)}
                  step="0.01"
                  min="0"
                  max="1"
                  className={errors.recognition_threshold ? 'error' : ''}
                />
                {errors.recognition_threshold && <span className="error-message">{errors.recognition_threshold}</span>}
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label htmlFor="min_text_size">Min Text Size (px)</label>
                <input
                  type="number"
                  id="min_text_size"
                  value={formData.model_thresholds.min_text_size || ''}
                  onChange={(e) => handleModelThresholdChange('min_text_size', e.target.value)}
                  min="1"
                />
              </div>

              <div className="form-group">
                <label htmlFor="max_text_size">Max Text Size (px)</label>
                <input
                  type="number"
                  id="max_text_size"
                  value={formData.model_thresholds.max_text_size || ''}
                  onChange={(e) => handleModelThresholdChange('max_text_size', e.target.value)}
                  min="1"
                />
              </div>
            </div>
          </div>

          {/* Form Actions */}
          <div className="form-actions">
            <button type="button" className="btn btn-secondary" onClick={handleClose} disabled={loading}>
              Cancel
            </button>
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? 'Saving...' : (mode === 'create' ? 'Create Recipe' : 'Update Recipe')}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
