import React, { useState, useEffect, useRef } from 'react';
import TemplateEditor from './TemplateEditorRefactored';
import AnnotationsPanel from './AnnotationsPanel';
import { camerasAPI } from '../services/api';
import { useToast } from '../contexts/ToastContext';
import { API_BASE_URL } from '../config/api';
import '../styles/RecipeFormModal.css';

export default function RecipeFormModal({ isOpen, onClose, onSubmit, recipe = null, mode = 'create' }) {
  const [activeTab, setActiveTab] = useState('basic');
  const toast = useToast();
  const [formData, setFormData] = useState({
    name: '',
    product_code: '',
    description: '',
    delay_reject: 100.0,
    is_active: true,
    cameras: [],
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

  const [templateImage, setTemplateImage] = useState(null);
  const [annotations, setAnnotations] = useState([]);
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  
  // Camera management states
  const [availableCameras, setAvailableCameras] = useState([]);
  const [loadingCameras, setLoadingCameras] = useState(false);
  const [selectedCameraForAdd, setSelectedCameraForAdd] = useState('');
  
  // Template editor states - now supports multiple templates per camera
  const [selectedCameraForTemplate, setSelectedCameraForTemplate] = useState('');
  const [cameraTemplates, setCameraTemplates] = useState({}); // { camera_id: [{ id, name, image, annotations }] }
  const [selectedTemplateIndex, setSelectedTemplateIndex] = useState(0); // Index of selected template for current camera
  const [selectedAnnotation, setSelectedAnnotation] = useState(null);
  const fabricCanvasRef = useRef(null);

  useEffect(() => {
    if (recipe && mode === 'edit' && isOpen) {
      // Ensure cameras array has proper structure with defaults
      const normalizedCameras = (recipe.cameras || []).map(cam => {
        return {
          camera_id: cam.camera_id || '',
          model_name: cam.model_name || '',
          serial_number: cam.serial_number || '',
          location: cam.location || '',
          exposure_time: cam.exposure_time || 50.0,
          delay_trigger: cam.delay_trigger || 100.0,
          gain: cam.gain || 1.0,
          pixel_format: cam.pixel_format || 'Mono8',
          trigger_config: {
            trigger_mode: cam.trigger_config?.trigger_mode !== undefined ? cam.trigger_config.trigger_mode : true,
            trigger_source: cam.trigger_config?.trigger_source || 'Software',
            trigger_selector: cam.trigger_config?.trigger_selector || 'FrameStart',
            trigger_activation: cam.trigger_config?.trigger_activation || 'RisingEdge'
          }
        };
      });
      // Load camera templates
      const loadedCameraTemplates = {};
      console.log('Recipe object:', recipe);
      console.log('camera_templates field:', recipe.camera_templates);
      console.log('Is array?', Array.isArray(recipe.camera_templates));
      console.log('Length:', recipe.camera_templates?.length);
      
      let firstCameraWithTemplates = null;
      
      if (recipe.camera_templates && Array.isArray(recipe.camera_templates) && recipe.camera_templates.length > 0) {
        console.log('Loading camera_templates from recipe:', recipe.camera_templates);
        recipe.camera_templates.forEach(camTemplate => {
          if (camTemplate.templates && camTemplate.templates.length > 0) {
            loadedCameraTemplates[camTemplate.camera_id] = camTemplate.templates.map(template => ({
              id: `template-${Date.now()}-${Math.random()}`,
              name: template.name,
              image: `${API_BASE_URL}${template.image_url}`, // Convert to full URL for preview
              image_url: template.image_url,
              image_width: template.image_width,
              image_height: template.image_height,
              annotations: template.annotations
            }));
            
            // Remember first camera with templates for auto-selection
            if (!firstCameraWithTemplates) {
              firstCameraWithTemplates = camTemplate.camera_id;
            }
          }
        });
        console.log('Loaded camera templates:', loadedCameraTemplates);
      } else {
        console.log('No camera_templates found in recipe - empty or undefined');
      }
      setCameraTemplates(loadedCameraTemplates);
      
      // Auto-select first camera with templates
      if (firstCameraWithTemplates) {
        setSelectedCameraForTemplate(firstCameraWithTemplates);
        setSelectedTemplateIndex(0);
      }

      setFormData({
        name: recipe.name || '',
        product_code: recipe.productCode || recipe.product_code || '',
        description: recipe.description || '',
        delay_reject: recipe.delay_reject || 100.0,
        is_active: recipe.is_active !== undefined ? recipe.is_active : (recipe.status === 'Active'),
        cameras: normalizedCameras,
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

      if (recipe.template_config?.template_image) {
        setTemplateImage(recipe.template_config.template_image);
      } else {
        setTemplateImage(null);
      }
      
      if (recipe.template_config?.annotations) {
        setAnnotations(recipe.template_config.annotations);
      } else {
        setAnnotations([]);
      }
    } else if (mode === 'create' && isOpen) {
      // Reset form for create mode
      setFormData({
        name: '',
        product_code: '',
        description: '',
        delay_reject: 100.0,
        is_active: true,
        cameras: [],
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
      setTemplateImage(null);
      setAnnotations([]);
      setCameraTemplates({});
      setSelectedTemplateIndex(0);
    }
  }, [recipe, mode, isOpen]);

  // Load available cameras
  useEffect(() => {
    if (isOpen) {
      loadAvailableCameras();
    }
  }, [isOpen]);

  const loadAvailableCameras = async () => {
    setLoadingCameras(true);
    try {
      const data = await camerasAPI.getAllCameras(0, 100);
      setAvailableCameras(data);
    } catch (error) {
      console.error('Failed to load cameras:', error);
    } finally {
      setLoadingCameras(false);
    }
  };

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

  // Camera management functions
  const handleAddCamera = () => {
    if (!selectedCameraForAdd) return;
    
    const camera = availableCameras.find(c => c.camera_id === selectedCameraForAdd);
    if (!camera) return;

    // Check if camera already added
    if (formData.cameras.some(c => c.camera_id === camera.camera_id)) {
      toast.warning('This camera is already added to the recipe');
      return;
    }

    const newCamera = {
      camera_id: camera.camera_id,
      model_name: camera.model_name,
      serial_number: camera.serial_number,
      location: camera.location || '',
      exposure_time: 50.0,
      delay_trigger: 100.0,
      gain: 1.0,
      pixel_format: 'Mono8',
      trigger_config: {
        trigger_mode: true,
        trigger_source: 'Software',
        trigger_selector: 'FrameStart',
        trigger_activation: 'RisingEdge'
      }
    };

    setFormData(prev => ({
      ...prev,
      cameras: [...prev.cameras, newCamera]
    }));
    setSelectedCameraForAdd('');
  };

  const handleRemoveCamera = (cameraId) => {
    setFormData(prev => ({
      ...prev,
      cameras: prev.cameras.filter(c => c.camera_id !== cameraId)
    }));
  };

  const handleCameraConfigChange = (cameraId, field, value) => {
    setFormData(prev => ({
      ...prev,
      cameras: prev.cameras.map(cam => 
        cam.camera_id === cameraId 
          ? { ...cam, [field]: parseFloat(value) || value }
          : cam
      )
    }));
  };

  const handleCameraTriggerConfigChange = (cameraId, field, value) => {
    setFormData(prev => ({
      ...prev,
      cameras: prev.cameras.map(cam => 
        cam.camera_id === cameraId 
          ? { 
              ...cam, 
              trigger_config: {
                ...cam.trigger_config,
                [field]: field === 'trigger_mode' ? value : value
              }
            }
          : cam
      )
    }));
  };

  const validateForm = () => {
    const newErrors = {};
    if (!formData.name.trim()) newErrors.name = 'Recipe name is required';
    if (!formData.product_code.trim()) newErrors.product_code = 'Product code is required';
    if (formData.camera_settings.exposure_time <= 0) newErrors.exposure_time = 'Must be greater than 0';
    if (formData.camera_settings.delay_trigger < 0) newErrors.delay_trigger = 'Cannot be negative';
    if (formData.model_thresholds.detection_threshold < 0 || formData.model_thresholds.detection_threshold > 1) {
      newErrors.detection_threshold = 'Must be between 0 and 1';
    }
    if (formData.model_thresholds.recognition_threshold < 0 || formData.model_thresholds.recognition_threshold > 1) {
      newErrors.recognition_threshold = 'Must be between 0 and 1';
    }
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateForm()) return;

    // Validate all camera templates before submission
    const validationErrors = validateTemplates();
    if (validationErrors.length > 0) {
      alert(`Template validation failed:\n\n${validationErrors.join('\n')}\n\nEach template must have:\n• 1 "template" region (required)\n• At least 1 annotation: text, barcode, or datecode (required)\n• crop_area (optional)`);
      return;
    }

    setLoading(true);
    try {
      // Prepare camera templates data for submission
      const cameraTemplatesArray = [];
      Object.entries(cameraTemplates).forEach(([cameraId, templates]) => {
        if (templates && templates.length > 0) {
          cameraTemplatesArray.push({
            camera_id: cameraId,
            templates: templates.map(template => ({
              name: template.name,
              image_url: template.image_url,
              image_width: template.image_width,
              image_height: template.image_height,
              annotations: template.annotations
            }))
          });
        }
      });

      const submitData = {
        ...formData,
        camera_templates: cameraTemplatesArray,
        // Legacy single template support (fallback)
        template_config: templateImage ? {
          template_image: templateImage,
          annotations: annotations
        } : formData.template_config
      };
      
      await onSubmit(submitData);
      handleClose();
    } catch (error) {
      console.error('Submit error:', error);
      setErrors({ submit: error.response?.data?.detail || 'Failed to save recipe' });
    } finally {
      setLoading(false);
    }
  };

  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (file) {
      try {
        // Upload image to server
        const formData = new FormData();
        formData.append('file', file);
        
        const token = localStorage.getItem('access_token');
        const response = await fetch(`${API_BASE_URL}/api/recipes/templates/upload`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Bypass-Tunnel-Reminder': 'true'
          },
          body: formData
        });
        
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || 'Failed to upload image');
        }
        
        const { url, width, height } = await response.json();
        
        // Also keep base64 for preview
        const reader = new FileReader();
        reader.onload = (event) => {
          const imageDataUrl = event.target.result;
          
          if (selectedCameraForTemplate) {
            // Add new template to the selected camera's templates array
            const currentTemplates = cameraTemplates[selectedCameraForTemplate] || [];
            const templateName = `Template ${currentTemplates.length + 1}`;
            const newTemplate = {
              id: `template-${Date.now()}`,
              name: templateName,
              image: imageDataUrl, // For preview in canvas
              image_url: url, // Server URL for submission
              image_width: width,
              image_height: height,
              annotations: []
            };
            
            setCameraTemplates(prev => ({
              ...prev,
              [selectedCameraForTemplate]: [...(prev[selectedCameraForTemplate] || []), newTemplate]
            }));
            
            // Auto-select the new template
            setSelectedTemplateIndex(currentTemplates.length);
          } else {
            setTemplateImage(imageDataUrl);
            setAnnotations([]);
          }
        };
        reader.readAsDataURL(file);
        
      } catch (error) {
        console.error('Upload error:', error);
        alert('Failed to upload template image. Please try again.');
      }
    }
    
    // Reset file input
    e.target.value = '';
  };

  const handleAnnotationsChange = (newAnnotations) => {
    if (selectedCameraForTemplate) {
      setCameraTemplates(prev => {
        const templates = prev[selectedCameraForTemplate] || [];
        const updatedTemplates = [...templates];
        if (updatedTemplates[selectedTemplateIndex]) {
          updatedTemplates[selectedTemplateIndex] = {
            ...updatedTemplates[selectedTemplateIndex],
            annotations: newAnnotations
          };
        }
        return {
          ...prev,
          [selectedCameraForTemplate]: updatedTemplates
        };
      });
    } else {
      setAnnotations(newAnnotations);
    }
  };

  const getCurrentTemplateImage = () => {
    if (selectedCameraForTemplate && cameraTemplates[selectedCameraForTemplate]) {
      const templates = cameraTemplates[selectedCameraForTemplate];
      return templates[selectedTemplateIndex]?.image || null;
    }
    return templateImage;
  };

  const getCurrentAnnotations = () => {
    if (selectedCameraForTemplate && cameraTemplates[selectedCameraForTemplate]) {
      const templates = cameraTemplates[selectedCameraForTemplate];
      return templates[selectedTemplateIndex]?.annotations || [];
    }
    return annotations;
  };
  
  const getCurrentTemplate = () => {
    if (selectedCameraForTemplate && cameraTemplates[selectedCameraForTemplate]) {
      const templates = cameraTemplates[selectedCameraForTemplate];
      return templates[selectedTemplateIndex] || null;
    }
    return null;
  };
  
  const handleDeleteTemplate = (templateIndex) => {
    if (!selectedCameraForTemplate) return;
    
    setCameraTemplates(prev => {
      const templates = prev[selectedCameraForTemplate] || [];
      const updated = templates.filter((_, idx) => idx !== templateIndex);
      return {
        ...prev,
        [selectedCameraForTemplate]: updated
      };
    });
    
    // Adjust selected index if needed
    if (selectedTemplateIndex >= templateIndex && selectedTemplateIndex > 0) {
      setSelectedTemplateIndex(selectedTemplateIndex - 1);
    }
  };
  
  const handleRenameTemplate = (templateIndex, newName) => {
    if (!selectedCameraForTemplate) return;
    
    setCameraTemplates(prev => {
      const templates = prev[selectedCameraForTemplate] || [];
      const updated = [...templates];
      if (updated[templateIndex]) {
        updated[templateIndex] = {
          ...updated[templateIndex],
          name: newName
        };
      }
      return {
        ...prev,
        [selectedCameraForTemplate]: updated
      };
    });
  };
  
  const validateTemplates = () => {
    const errors = [];
    
    Object.entries(cameraTemplates).forEach(([cameraId, templates]) => {
      templates.forEach((template, idx) => {
        const hasTemplateRegion = template.annotations.some(ann => ann.type === 'template');
        const hasRequiredAnnotation = template.annotations.some(ann => 
          ['text', 'barcode', 'datecode'].includes(ann.type)
        );
        
        if (!hasTemplateRegion) {
          errors.push(`Camera ${cameraId} - ${template.name}: Missing required "template" region`);
        }
        
        if (!hasRequiredAnnotation) {
          errors.push(`Camera ${cameraId} - ${template.name}: Must have at least one annotation (text, barcode, or datecode)`);
        }
      });
    });
    
    return errors;
  };

  const handleAnnotationTypeChange = (index, newType) => {
    const currentAnnotations = getCurrentAnnotations();
    const updated = [...currentAnnotations];
    updated[index].type = newType;
    handleAnnotationsChange(updated);
  };

  const handleAnnotationTextChange = (index, newText) => {
    const currentAnnotations = getCurrentAnnotations();
    const updated = [...currentAnnotations];
    updated[index].text = newText;
    handleAnnotationsChange(updated);
  };

  const handleDeleteAnnotation = (index) => {
    const currentAnnotations = getCurrentAnnotations();
    const updated = currentAnnotations.filter((_, i) => i !== index);
    handleAnnotationsChange(updated);
    
    // Reset selection if deleted
    if (selectedAnnotation === index) {
      setSelectedAnnotation(null);
    } else if (selectedAnnotation > index) {
      setSelectedAnnotation(selectedAnnotation - 1);
    }
  };

  const handleClose = () => {
    // Don't reset formData here - let useEffect handle it when modal reopens
    setErrors({});
    setActiveTab('basic');
    setSelectedCameraForAdd('');
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={handleClose}>
      <div className="modal-content recipe-form-modal large" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>{mode === 'create' ? 'Create New Recipe' : 'Edit Recipe'}</h2>
          <button className="close-btn" onClick={handleClose}>×</button>
        </div>

        <div className="modal-tabs">
          <button className={`tab-btn ${activeTab === 'basic' ? 'active' : ''}`} onClick={() => setActiveTab('basic')}>
            📋 Basic Info
          </button>
          <button className={`tab-btn ${activeTab === 'camera' ? 'active' : ''}`} onClick={() => setActiveTab('camera')}>
            📷 Camera
          </button>
          <button className={`tab-btn ${activeTab === 'model' ? 'active' : ''}`} onClick={() => setActiveTab('model')}>
            🎯 Model
          </button>
          <button className={`tab-btn ${activeTab === 'template' ? 'active' : ''}`} onClick={() => setActiveTab('template')}>
            🖼️ Template {annotations.length > 0 && <span className="badge">{annotations.length}</span>}
          </button>
        </div>

        <form onSubmit={handleSubmit} className="recipe-form">
          {errors.submit && <div className="error-message global-error">{errors.submit}</div>}

          <div className="tab-content">
            {activeTab === 'basic' && (
              <div className="form-section">
                <h3>Basic Information</h3>
                <div className="form-row">
                  <div className="form-group">
                    <label>Recipe Name <span className="required">*</span></label>
                    <input type="text" name="name" value={formData.name} onChange={handleInputChange} 
                           placeholder="Enter recipe name" className={errors.name ? 'error' : ''} />
                    {errors.name && <span className="error-message">{errors.name}</span>}
                  </div>
                  <div className="form-group">
                    <label>Product Code <span className="required">*</span></label>
                    <input type="text" name="product_code" value={formData.product_code} onChange={handleInputChange}
                           placeholder="Enter product code" className={errors.product_code ? 'error' : ''} />
                    {errors.product_code && <span className="error-message">{errors.product_code}</span>}
                  </div>
                </div>
                <div className="form-row">
                  <div className="form-group">
                    <label>Description</label>
                    <textarea name="description" value={formData.description} onChange={handleInputChange}
                             placeholder="Enter recipe description" rows="3" />
                  </div>
                  <div className="form-group">
                    <label>Delay Reject (ms)</label>
                    <input type="number" name="delay_reject" value={formData.delay_reject} 
                           onChange={handleInputChange} step="0.1" min="0" 
                           placeholder="Delay reject time in milliseconds" />
                  </div>
                </div>
                <div className="form-group checkbox-group">
                  <label>
                    <input type="checkbox" name="is_active" checked={formData.is_active} onChange={handleInputChange} />
                    <span>Active Recipe</span>
                  </label>
                </div>
              </div>
            )}

            {activeTab === 'camera' && (
              <div className="form-section">
                <div className="camera-header">
                  <h3>Camera Configuration</h3>
                  <div className="camera-add-section">
                    <select 
                      value={selectedCameraForAdd} 
                      onChange={(e) => setSelectedCameraForAdd(e.target.value)}
                      disabled={loadingCameras}
                    >
                      <option value="">Select a camera...</option>
                      {availableCameras
                        .filter(cam => !formData.cameras.some(c => c.camera_id === cam.camera_id))
                        .map(cam => (
                          <option key={cam.camera_id} value={cam.camera_id}>
                            {cam.camera_id} - {cam.model_name} ({cam.serial_number})
                          </option>
                        ))
                      }
                    </select>
                    <button 
                      type="button" 
                      className="btn btn-secondary" 
                      onClick={handleAddCamera}
                      disabled={!selectedCameraForAdd || loadingCameras}
                    >
                      ➕ Add Camera
                    </button>
                  </div>
                </div>

                {formData.cameras.length === 0 ? (
                  <div className="empty-state">
                    <p>No cameras configured for this recipe</p>
                    <p className="hint">Add cameras using the dropdown above</p>
                  </div>
                ) : (
                  <div className="cameras-list">
                    {formData.cameras.map((camera, index) => (
                      <div key={camera.camera_id} className="camera-card">
                        <div className="camera-card-header">
                          <h4>Camera {index + 1}: {camera.camera_id}</h4>
                          <button 
                            type="button" 
                            className="btn-remove" 
                            onClick={() => handleRemoveCamera(camera.camera_id)}
                            title="Remove camera"
                          >
                            🗑️
                          </button>
                        </div>
                        
                        <div className="camera-info">
                          <div className="info-item">
                            <strong>Model:</strong> {camera.model_name}
                          </div>
                          <div className="info-item">
                            <strong>Serial:</strong> {camera.serial_number}
                          </div>
                          <div className="info-item">
                            <strong>Location:</strong> {camera.location || 'N/A'}
                          </div>
                        </div>

                        <div className="camera-settings">
                          <h5>Camera Settings</h5>
                          <div className="form-row">
                            <div className="form-group">
                              <label>Exposure Time (ms) <span className="required">*</span></label>
                              <input 
                                type="number" 
                                value={camera.exposure_time}
                                onChange={(e) => handleCameraConfigChange(camera.camera_id, 'exposure_time', e.target.value)}
                                step="0.1" 
                                min="0" 
                              />
                            </div>
                            <div className="form-group">
                              <label>Delay Trigger (ms) <span className="required">*</span></label>
                              <input 
                                type="number" 
                                value={camera.delay_trigger}
                                onChange={(e) => handleCameraConfigChange(camera.camera_id, 'delay_trigger', e.target.value)}
                                step="0.1" 
                                min="0" 
                              />
                            </div>
                            <div className="form-group">
                              <label>Gain <span className="required">*</span></label>
                              <input 
                                type="number" 
                                value={camera.gain}
                                onChange={(e) => handleCameraConfigChange(camera.camera_id, 'gain', e.target.value)}
                                step="0.1" 
                                min="0" 
                              />
                            </div>
                          </div>

                          <div className="form-row">
                            <div className="form-group">
                              <label>Pixel Format</label>
                              <select 
                                value={camera.pixel_format}
                                onChange={(e) => handleCameraConfigChange(camera.camera_id, 'pixel_format', e.target.value)}
                              >
                                <option value="Mono8">Mono8</option>
                                <option value="Mono12">Mono12</option>
                                <option value="RGB8">RGB8</option>
                                <option value="YUV422">YUV422</option>
                                <option value="BayerRG8">BayerRG8</option>
                              </select>
                            </div>
                          </div>

                          <h5>Trigger Configuration</h5>
                          <div className="form-row">
                            <div className="form-group checkbox-group">
                              <label>
                                <input 
                                  type="checkbox" 
                                  checked={camera.trigger_config.trigger_mode}
                                  onChange={(e) => handleCameraTriggerConfigChange(camera.camera_id, 'trigger_mode', e.target.checked)}
                                />
                                <span>Trigger Mode Enabled</span>
                              </label>
                            </div>
                          </div>
                          
                          <div className="form-row">
                            <div className="form-group">
                              <label>Trigger Source</label>
                              <select 
                                value={camera.trigger_config.trigger_source}
                                onChange={(e) => handleCameraTriggerConfigChange(camera.camera_id, 'trigger_source', e.target.value)}
                              >
                                <option value="Software">Software</option>
                                <option value="Line1">Line1</option>
                                <option value="Line2">Line2</option>
                                <option value="Line3">Line3</option>
                              </select>
                            </div>
                            <div className="form-group">
                              <label>Trigger Selector</label>
                              <select 
                                value={camera.trigger_config.trigger_selector}
                                onChange={(e) => handleCameraTriggerConfigChange(camera.camera_id, 'trigger_selector', e.target.value)}
                              >
                                <option value="FrameStart">FrameStart</option>
                                <option value="ExposureStart">ExposureStart</option>
                                <option value="FrameBurstStart">FrameBurstStart</option>
                              </select>
                            </div>
                            <div className="form-group">
                              <label>Trigger Activation</label>
                              <select 
                                value={camera.trigger_config.trigger_activation}
                                onChange={(e) => handleCameraTriggerConfigChange(camera.camera_id, 'trigger_activation', e.target.value)}
                              >
                                <option value="RisingEdge">RisingEdge</option>
                                <option value="FallingEdge">FallingEdge</option>
                                <option value="AnyEdge">AnyEdge</option>
                              </select>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {activeTab === 'model' && (
              <div className="form-section">
                <h3>Model Thresholds</h3>
                <div className="form-row">
                  <div className="form-group">
                    <label>Detection Threshold <span className="required">*</span> <span className="hint">(0.0 - 1.0)</span></label>
                    <input type="number" value={formData.model_thresholds.detection_threshold}
                           onChange={(e) => handleModelThresholdChange('detection_threshold', e.target.value)}
                           step="0.01" min="0" max="1" className={errors.detection_threshold ? 'error' : ''} />
                    {errors.detection_threshold && <span className="error-message">{errors.detection_threshold}</span>}
                  </div>
                  <div className="form-group">
                    <label>Recognition Threshold <span className="required">*</span> <span className="hint">(0.0 - 1.0)</span></label>
                    <input type="number" value={formData.model_thresholds.recognition_threshold}
                           onChange={(e) => handleModelThresholdChange('recognition_threshold', e.target.value)}
                           step="0.01" min="0" max="1" className={errors.recognition_threshold ? 'error' : ''} />
                    {errors.recognition_threshold && <span className="error-message">{errors.recognition_threshold}</span>}
                  </div>
                </div>
                <div className="form-row">
                  <div className="form-group">
                    <label>Min Text Size (px)</label>
                    <input type="number" value={formData.model_thresholds.min_text_size || ''}
                           onChange={(e) => handleModelThresholdChange('min_text_size', e.target.value)} min="1" />
                  </div>
                  <div className="form-group">
                    <label>Max Text Size (px)</label>
                    <input type="number" value={formData.model_thresholds.max_text_size || ''}
                           onChange={(e) => handleModelThresholdChange('max_text_size', e.target.value)} min="1" />
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'template' && (
              <div className="form-section template-section">
                <div className="template-header">
                  <h3>Template Configuration</h3>
                  <div className="template-camera-selector">
                    <label>Select Camera:</label>
                    <select 
                      value={selectedCameraForTemplate} 
                      onChange={(e) => setSelectedCameraForTemplate(e.target.value)}
                      disabled={formData.cameras.length === 0}
                    >
                      <option value="">-- Select Camera --</option>
                      {formData.cameras.map(cam => (
                        <option key={cam.camera_id} value={cam.camera_id}>
                          {cam.camera_id} - {cam.model_name}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                {formData.cameras.length === 0 ? (
                  <div className="template-placeholder">
                    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                    <p>Please add cameras in the Camera tab first</p>
                    <p className="hint">Template configuration requires at least one camera to be selected</p>
                  </div>
                ) : !selectedCameraForTemplate ? (
                  <div className="template-placeholder">
                    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <rect x="3" y="3" width="18" height="18" rx="2" strokeWidth="2"/>
                      <circle cx="8.5" cy="8.5" r="1.5" fill="currentColor"/>
                      <polyline points="21,15 16,10 5,21" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                    <p>Select a camera to configure templates</p>
                    <p className="hint">Choose a camera from the dropdown above</p>
                  </div>
                ) : (
                  <>
                    <div className="template-actions">
                      <input 
                        type="file" 
                        id={`template-upload-${selectedCameraForTemplate}`}
                        accept="image/*" 
                        onChange={handleImageUpload} 
                        style={{ display: 'none' }} 
                      />
                      <label 
                        htmlFor={`template-upload-${selectedCameraForTemplate}`}
                        className="btn btn-secondary"
                      >
                        ➕ Add Template Image
                      </label>
                      <span className="template-info">
                        Camera: {selectedCameraForTemplate} | 
                        Templates: {cameraTemplates[selectedCameraForTemplate]?.length || 0}
                      </span>
                    </div>

                    {/* Templates List */}
                    {cameraTemplates[selectedCameraForTemplate]?.length > 0 && (
                      <div className="templates-list">
                        {cameraTemplates[selectedCameraForTemplate].map((template, idx) => {
                          const hasTemplateRegion = template.annotations.some(ann => ann.type === 'template');
                          const hasRequiredAnnotation = template.annotations.some(ann => 
                            ['text', 'barcode', 'datecode'].includes(ann.type)
                          );
                          const hasCropArea = template.annotations.some(ann => ann.type === 'crop_area');
                          const isValid = hasTemplateRegion && hasRequiredAnnotation;
                          
                          return (
                            <div 
                              key={template.id} 
                              className={`template-item ${idx === selectedTemplateIndex ? 'active' : ''} ${!isValid ? 'invalid' : ''}`}
                              onClick={() => setSelectedTemplateIndex(idx)}
                            >
                              <div className="template-thumbnail">
                                <img src={template.image} alt={template.name} />
                                {!isValid && (
                                  <div className="warning-badge" title="Missing required regions">⚠️</div>
                                )}
                              </div>
                              <div className="template-details">
                                <input
                                  type="text"
                                  value={template.name}
                                  onChange={(e) => {
                                    e.stopPropagation();
                                    handleRenameTemplate(idx, e.target.value);
                                  }}
                                  onClick={(e) => e.stopPropagation()}
                                  className="template-name-input"
                                />
                                <div className="template-stats">
                                  {hasTemplateRegion && <span className="stat">📐 Template</span>}
                                  <span className="stat">
                                    {template.annotations.filter(a => ['text', 'barcode', 'datecode'].includes(a.type)).length} 
                                    {' '}regions
                                  </span>
                                  {hasCropArea && <span className="stat crop">✂️ Crop</span>}
                                </div>
                                {!isValid && (
                                  <div className="validation-error">
                                    {!hasTemplateRegion && '⚠️ Missing "template" region'}
                                    {!hasTemplateRegion && !hasRequiredAnnotation && ' • '}
                                    {!hasRequiredAnnotation && '⚠️ Missing text/barcode/datecode'}
                                  </div>
                                )}
                              </div>
                              <button
                                type="button"
                                className="btn-delete-template"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  if (confirm(`Delete ${template.name}?`)) {
                                    handleDeleteTemplate(idx);
                                  }
                                }}
                                title="Delete template"
                              >
                                🗑️
                              </button>
                            </div>
                          );
                        })}
                      </div>
                    )}

                    {getCurrentTemplateImage() ? (
                      <div className="template-editor-layout">
                        <div className="template-editor-canvas">
                          <TemplateEditor
                            templateImage={getCurrentTemplateImage()}
                            annotations={getCurrentAnnotations()}
                            onAnnotationsChange={handleAnnotationsChange}
                            selectedAnnotation={selectedAnnotation}
                            onSelectAnnotation={setSelectedAnnotation}
                            fabricCanvasRef={fabricCanvasRef}
                          />
                        </div>
                        <div className="template-editor-sidebar">
                          <AnnotationsPanel
                            annotations={getCurrentAnnotations()}
                            selectedAnnotation={selectedAnnotation}
                            onSelectAnnotation={setSelectedAnnotation}
                            onAnnotationTypeChange={handleAnnotationTypeChange}
                            onAnnotationTextChange={handleAnnotationTextChange}
                            onDeleteAnnotation={handleDeleteAnnotation}
                            fabricCanvasRef={fabricCanvasRef}
                          />
                        </div>
                      </div>
                    ) : (
                      <div className="template-placeholder">
                        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                          <rect x="3" y="3" width="18" height="18" rx="2" strokeWidth="2"/>
                          <circle cx="8.5" cy="8.5" r="1.5" fill="currentColor"/>
                          <polyline points="21,15 16,10 5,21" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                        <p>No templates added yet</p>
                        <p className="hint">Click "Add Template Image" to get started</p>
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </div>

          <div className="form-actions">
            <button type="button" className="btn btn-secondary" onClick={handleClose} disabled={loading}>Cancel</button>
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? 'Saving...' : (mode === 'create' ? 'Create Recipe' : 'Update Recipe')}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
