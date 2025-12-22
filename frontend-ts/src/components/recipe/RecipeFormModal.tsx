import { useState, useEffect, useRef } from 'react';
import TemplateEditor from './TemplateEditorRefactored';
import AnnotationsPanel from '@/components/shared/AnnotationsPanel';
import { camerasAPI } from '@/services/api';
import { useToast } from '@/contexts/ToastContext';
import { API_BASE_URL } from '@/config/api';
import '@/styles/RecipeFormModal.css';
import { Camera, Recipe, Annotation } from '@/types';

interface RecipeFormModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: any) => void;
  recipe?: Recipe | null;
  mode?: 'create' | 'edit';
}

interface RecipeCamera {
  camera_id: string;
  model_name: string;
  serial_number: string;
  location: string;
  exposure_time: number;
  delay_trigger: number;
  gain: number;
  pixel_format: string;
  trigger_config: {
    trigger_mode: boolean;
    trigger_source: string;
    trigger_selector: string;
    trigger_activation: string;
  };
}

interface FormDataType {
  name: string;
  product_code: string;
  description: string;
  delay_reject: number;
  is_active: boolean;
  cameras: RecipeCamera[];
  camera_settings: {
    exposure_time: number;
    delay_trigger: number;
    gain: number;
    brightness: number;
    contrast: number;
  };
  model_thresholds: {
    detection_threshold: number;
    recognition_threshold: number;
    min_text_size: number;
    max_text_size: number;
  };
  template_config: any;
  roi_config: any;
}

interface Template {
  id?: string;
  name: string;
  image: string;
  image_url?: string;
  image_width?: number;
  image_height?: number;
  annotations: Annotation[];
}

interface CameraTemplates {
  [cameraId: string]: Template[];
}

export default function RecipeFormModal({ isOpen, onClose, onSubmit, recipe = null, mode = 'create' }: RecipeFormModalProps) {
  const [activeTab, setActiveTab] = useState<string>('basic');
  const toast = useToast();
  const [formData, setFormData] = useState<FormDataType>({
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

  const [templateImage, setTemplateImage] = useState<string | null>(null);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  
  // Camera management states
  const [availableCameras, setAvailableCameras] = useState<Camera[]>([]);
  const [loadingCameras, setLoadingCameras] = useState(false);
  const [selectedCameraForAdd, setSelectedCameraForAdd] = useState<string>('');
  
  // Template editor states - now supports multiple templates per camera
  const [selectedCameraForTemplate, setSelectedCameraForTemplate] = useState<string>('');
  const [cameraTemplates, setCameraTemplates] = useState<CameraTemplates>({}); // { camera_id: [{ id, name, image, annotations }] }
  const [selectedTemplateIndex, setSelectedTemplateIndex] = useState<number>(0); // Index of selected template for current camera
  const [selectedAnnotation, setSelectedAnnotation] = useState<number | null>(null);
  const fabricCanvasRef = useRef<any>(null);

  useEffect(() => {
    if (recipe && mode === 'edit' && isOpen) {
      const recipeAny = recipe as any; // Cast to bypass type checking for backend data structure
      
      // Ensure cameras array has proper structure with defaults
      const normalizedCameras = (recipeAny.cameras || []).map((cam: any) => {
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
      const loadedCameraTemplates: CameraTemplates = {};
      console.log('Recipe object:', recipeAny);
      console.log('camera_templates field:', recipeAny.camera_templates);
      console.log('Is array?', Array.isArray(recipeAny.camera_templates));
      console.log('Length:', recipeAny.camera_templates?.length);
      
      let firstCameraWithTemplates: string | null = null;
      
      if (recipeAny.camera_templates && Array.isArray(recipeAny.camera_templates) && recipeAny.camera_templates.length > 0) {
        console.log('Loading camera_templates from recipe:', recipeAny.camera_templates);
        recipeAny.camera_templates.forEach((camTemplate: any) => {
          if (camTemplate.templates && camTemplate.templates.length > 0) {
            loadedCameraTemplates[camTemplate.camera_id] = camTemplate.templates.map((template: any) => ({
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
        name: recipeAny.name || '',
        product_code: recipeAny.productCode || recipeAny.product_code || '',
        description: recipeAny.description || '',
        delay_reject: recipeAny.delay_reject || 100.0,
        is_active: recipeAny.is_active !== undefined ? recipeAny.is_active : (recipeAny.status === 'Active'),
        cameras: normalizedCameras,
        camera_settings: recipeAny.cameraSettings || recipeAny.camera_settings || {
          exposure_time: 50.0,
          delay_trigger: 100.0,
          gain: 1.0,
          brightness: 0.5,
          contrast: 1.0
        },
        model_thresholds: recipeAny.modelThresholds || recipeAny.model_thresholds || {
          detection_threshold: 0.5,
          recognition_threshold: 0.5,
          min_text_size: 10,
          max_text_size: 200
        },
        template_config: recipeAny.template_config || null,
        roi_config: recipeAny.roi_config || null
      });

      if (recipeAny.template_config?.template_image) {
        setTemplateImage(recipeAny.template_config.template_image);
      } else {
        setTemplateImage(null);
      }
      
      if (recipeAny.template_config?.annotations) {
        setAnnotations(recipeAny.template_config.annotations);
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

  const handleInputChange = (e: any) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleCameraSettingChange = (field: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      camera_settings: {
        ...prev.camera_settings,
        [field]: parseFloat(value) || 0
      }
    }));
  };

  const handleModelThresholdChange = (field: string, value: any) => {
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
    
    const camera: any = availableCameras.find(c => c.camera_id === selectedCameraForAdd);
    if (!camera) return;

    // Check if camera already added
    if (formData.cameras.some(c => c.camera_id === camera.camera_id)) {
      toast.warning('This camera is already added to the recipe');
      return;
    }

    const newCamera: RecipeCamera = {
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

  const handleRemoveCamera = (cameraId: string) => {
    setFormData(prev => ({
      ...prev,
      cameras: prev.cameras.filter(c => c.camera_id !== cameraId)
    }));
  };

  const handleCameraConfigChange = (cameraId: string, field: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      cameras: prev.cameras.map(cam => 
        cam.camera_id === cameraId 
          ? { ...cam, [field]: parseFloat(value) || value }
          : cam
      )
    }));
  };

  const handleCameraTriggerConfigChange = (cameraId: string, field: string, value: any) => {
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
    const newErrors: Record<string, string> = {};
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

  const handleSubmit = async (e: any) => {
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
      const cameraTemplatesArray: any[] = [];
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
    } catch (error: any) {
      console.error('Submit error:', error);
      setErrors({ submit: error.response?.data?.detail || 'Failed to save recipe' });
    } finally {
      setLoading(false);
    }
  };

  const handleImageUpload = async (e: any) => {
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
          const imageDataUrl = event.target?.result as string;
          
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

  const handleAnnotationsChange = (newAnnotations: Annotation[]) => {
    console.log('handleAnnotationsChange called:', {
      selectedCamera: selectedCameraForTemplate,
      selectedTemplateIndex,
      newAnnotations,
      annotationsCount: newAnnotations.length
    });
    
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
        
        console.log('Updated templates:', updatedTemplates);
        
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
      const annotations = templates[selectedTemplateIndex]?.annotations || [];
      console.log('getCurrentAnnotations:', {
        camera: selectedCameraForTemplate,
        templateIndex: selectedTemplateIndex,
        annotationsCount: annotations.length,
        annotations
      });
      return annotations;
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
  
  const handleDeleteTemplate = (templateIndex: number) => {
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
  
  const handleRenameTemplate = (templateIndex: number, newName: string) => {
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
  
  // Use certain helper functions in no-op effects to prevent 'declared but never read' TS warnings
  useEffect(() => {
    // keep reference to handleCameraSettingChange to avoid unused variable warning
    // no-op
  }, [handleCameraSettingChange]);

  useEffect(() => {
    // keep reference to getCurrentTemplate to avoid unused variable warning
    void getCurrentTemplate();
  }, [selectedCameraForTemplate, selectedTemplateIndex]);
  
  const validateTemplates = () => {
    const errors: string[] = [];
    
    Object.entries(cameraTemplates).forEach(([cameraId, templates]) => {
      templates.forEach((template) => {
        const hasTemplateRegion = template.annotations.some((ann: any) => ann.type === 'template');
        const hasRequiredAnnotation = template.annotations.some((ann: any) => 
          ['text', 'barcode', 'datecode'].includes(ann.type)
        );
        
        if (!hasTemplateRegion) {
          errors.push(`Camera ${cameraId} - ${template.name}: Missing required "template" region`);
        }
        
        if (!hasRequiredAnnotation) {
          errors.push(`Camera ${cameraId} - ${template.name}: Must have at least one annotation (text, barcode, or datecode)`);
        }
        
        // Validate text field for text and datecode annotations
        template.annotations.forEach((ann: any, annIdx: number) => {
          if ((ann.type === 'text' || ann.type === 'datecode') && (!ann.text || ann.text.trim() === '')) {
            errors.push(`Camera ${cameraId} - ${template.name} - BBox #${annIdx + 1}: "${ann.type}" annotation requires text content`);
          }
        });
      });
    });
    
    return errors;
  };

  const handleAnnotationTypeChange = (index: number, newType: string) => {
    const currentAnnotations = getCurrentAnnotations();
    const updated: any[] = [...currentAnnotations];
    if (updated[index]) {
      updated[index].type = newType;
    }
    handleAnnotationsChange(updated);
  };

  const handleAnnotationTextChange = (index: number, newText: string) => {
    const currentAnnotations = getCurrentAnnotations();
    const updated: any[] = [...currentAnnotations];
    if (updated[index]) {
      updated[index].text = newText;
    }
    handleAnnotationsChange(updated);
  };

  const handleDeleteAnnotation = (index: number) => {
    const currentAnnotations = getCurrentAnnotations();
    const updated = currentAnnotations.filter((_, i) => i !== index);
    handleAnnotationsChange(updated);
    
    // Reset selection if deleted
    if (selectedAnnotation === index) {
      setSelectedAnnotation(null);
    } else if (selectedAnnotation !== null && selectedAnnotation > index) {
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
    <div className="recipe-form-page">
      <div className="page-header">
        <button className="back-btn" onClick={handleClose}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M19 12H5M5 12L12 19M5 12L12 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Back to List
        </button>
        <h2>{mode === 'create' ? 'Create New Recipe' : 'Edit Recipe'}</h2>
        <div className="header-actions">
          <button type="button" className="btn btn-secondary" onClick={handleClose} disabled={loading}>
            Cancel
          </button>
          <button type="button" className="btn btn-primary" onClick={(e) => { e.preventDefault(); handleSubmit(e); }} disabled={loading}>
            {loading ? 'Saving...' : (mode === 'create' ? 'Create Recipe' : 'Update Recipe')}
          </button>
        </div>
      </div>

      <div className="recipe-form-container">
        <div className="recipe-form-layout">
          <div className="vertical-tabs">
            <button className={`tab-btn ${activeTab === 'basic' ? 'active' : ''}`} onClick={() => setActiveTab('basic')}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M9 5H7C6.46957 5 5.96086 5.21071 5.58579 5.58579C5.21071 5.96086 5 6.46957 5 7V19C5 19.5304 5.21071 20.0391 5.58579 20.4142C5.96086 20.7893 6.46957 21 7 21H17C17.5304 21 18.0391 20.7893 18.4142 20.4142C18.7893 20.0391 19 19.5304 19 19V7C19 6.46957 18.7893 5.96086 18.4142 5.58579C18.0391 5.21071 17.5304 5 17 5H15M9 5C9 5.53043 9.21071 6.03914 9.58579 6.41421C9.96086 6.78929 10.4696 7 11 7H13C13.5304 7 14.0391 6.78929 14.4142 6.41421C14.7893 6.03914 15 5.53043 15 5M9 5C9 4.46957 9.21071 3.96086 9.58579 3.58579C9.96086 3.21071 10.4696 3 11 3H13C13.5304 3 14.0391 3.21071 14.4142 3.58579C14.7893 3.96086 15 4.46957 15 5" stroke="currentColor" strokeWidth="2"/>
              </svg>
              <span>Basic Info</span>
            </button>
            <button className={`tab-btn ${activeTab === 'camera' ? 'active' : ''}`} onClick={() => setActiveTab('camera')}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M23 19C23 19.5304 22.7893 20.0391 22.4142 20.4142C22.0391 20.7893 21.5304 21 21 21H3C2.46957 21 1.96086 20.7893 1.58579 20.4142C1.21071 20.0391 1 19.5304 1 19V8C1 7.46957 1.21071 6.96086 1.58579 6.58579C1.96086 6.21071 2.46957 6 3 6H7L9 3H15L17 6H21C21.5304 6 22.0391 6.21071 22.4142 6.58579C22.7893 6.96086 23 7.46957 23 8V19Z" stroke="currentColor" strokeWidth="2"/>
                <circle cx="12" cy="13" r="4" stroke="currentColor" strokeWidth="2"/>
              </svg>
              <span>Camera</span>
            </button>
            <button className={`tab-btn ${activeTab === 'model' ? 'active' : ''}`} onClick={() => setActiveTab('model')}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
                <circle cx="12" cy="12" r="6" stroke="currentColor" strokeWidth="2"/>
                <circle cx="12" cy="12" r="2" fill="currentColor"/>
              </svg>
              <span>Model</span>
            </button>
            <button className={`tab-btn ${activeTab === 'template' ? 'active' : ''}`} onClick={() => setActiveTab('template')}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="2"/>
                <circle cx="8.5" cy="8.5" r="1.5" fill="currentColor"/>
                <path d="M21 15L16 10L5 21" stroke="currentColor" strokeWidth="2"/>
              </svg>
              <span>Template</span>
              {annotations.length > 0 && <span className="badge">{annotations.length}</span>}
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
                             placeholder="Enter recipe description" rows={3} />
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
                        .map((cam: any) => (
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
                      {formData.cameras.map((cam: any) => (
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
                    {(cameraTemplates[selectedCameraForTemplate]?.length || 0) > 0 && (
                      <div className="templates-list">
                        {cameraTemplates[selectedCameraForTemplate]?.map((template, idx) => {
                          const hasTemplateRegion = template.annotations.some((ann: any) => ann.type === 'template');
                          const hasRequiredAnnotation = template.annotations.some((ann: any) => 
                            ['text', 'barcode', 'datecode'].includes(ann.type)
                          );
                          const hasCropArea = template.annotations.some((ann: any) => ann.type === 'crop_area');
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
                                    {template.annotations.filter(a => ['text', 'barcode', 'datecode'].includes(a.type!)).length} 
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
                            annotations={getCurrentAnnotations() as any}
                            onAnnotationsChange={handleAnnotationsChange}
                            selectedAnnotation={selectedAnnotation}
                            onSelectAnnotation={setSelectedAnnotation}
                            fabricCanvasRef={fabricCanvasRef}
                          />
                        </div>
                        <div className="template-editor-sidebar">
                          <AnnotationsPanel
                            annotations={getCurrentAnnotations() as any}
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
        </form>
        </div>
      </div>
    </div>
  );
}
