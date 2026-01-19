import React from 'react';
import type { Receipt } from '@/types';
import '@/styles/RecipeViewModal.css';

interface RecipeViewModalProps {
  isOpen: boolean;
  onClose: () => void;
  recipe: Receipt | null;
  onEdit: () => void;
}

const RecipeViewModal: React.FC<RecipeViewModalProps> = ({ isOpen, onClose, recipe, onEdit }) => {
  if (!isOpen || !recipe) return null;

  return (
    <div className="recipe-view-page">
      <div className="page-header">
        <button className="back-btn" onClick={onClose}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M19 12H5M5 12L12 19M5 12L12 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Back to List
        </button>
        <h2>Recipe Details</h2>
        <button className="btn btn-primary" onClick={onEdit}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
            <path d="M11 4H4C3.46957 4 2.96086 4.21071 2.58579 4.58579C2.21071 4.96086 2 5.46957 2 6V20C2 20.5304 2.21071 21.0391 2.58579 21.4142C2.96086 21.7893 3.46957 22 4 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M18.5 2.50001C18.8978 2.10219 19.4374 1.87869 20 1.87869C20.5626 1.87869 21.1022 2.10219 21.5 2.50001C21.8978 2.89784 22.1213 3.4374 22.1213 4.00001C22.1213 4.56262 21.8978 5.10219 21.5 5.50001L12 15L8 16L9 12L18.5 2.50001Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Edit Recipe
        </button>
      </div>

      <div className="recipe-view-container">

        <div className="recipe-view-content">
          {/* Basic Information */}
          <div className="info-section">
            <h3>Basic Information</h3>
            <div className="info-grid">
              <div className="info-item">
                <label>Recipe Name</label>
                <div className="info-value">{recipe.name}</div>
              </div>
              <div className="info-item">
                <label>Product Code</label>
                <div className="info-value">{(recipe as any).product_code || recipe.productCode}</div>
              </div>
              <div className="info-item">
                <label>Status</label>
                <div className="info-value">
                  <span className={`status-badge ${recipe.is_active ? 'active' : 'inactive'}`}>
                    {recipe.is_active ? 'Active' : 'Inactive'}
                  </span>
                </div>
              </div>
              <div className="info-item">
                <label>Delay Reject</label>
                <div className="info-value">{recipe.delay_reject || 'N/A'} ms</div>
              </div>
            </div>
            {recipe.description && (
              <div className="info-item full-width">
                <label>Description</label>
                <div className="info-value">{recipe.description}</div>
              </div>
            )}
          </div>

          {/* Individual Camera Configurations */}
          {recipe.cameras && recipe.cameras.length > 0 && (
            <div className="info-section">
              <h3>Camera Configurations</h3>
              {recipe.cameras.map((camera: any, index: number) => (
                <div key={index} className="camera-config">
                  <h4>Camera {index + 1}: {camera.model_name} ({camera.serial_number})</h4>
                  <div className="info-grid">
                    <div className="info-item">
                      <label>Location</label>
                      <div className="info-value">{camera.location || 'N/A'}</div>
                    </div>
                    <div className="info-item">
                      <label>Exposure Time</label>
                      <div className="info-value">{camera.exposure_time} ms</div>
                    </div>
                    <div className="info-item">
                      <label>Delay Trigger</label>
                      <div className="info-value">{camera.delay_trigger} ms</div>
                    </div>
                    <div className="info-item">
                      <label>Gain</label>
                      <div className="info-value">{camera.gain}</div>
                    </div>
                    <div className="info-item">
                      <label>Pixel Format</label>
                      <div className="info-value">{camera.pixel_format}</div>
                    </div>
                  </div>
                  {camera.trigger_config && (
                    <div className="trigger-config">
                      <h5>Trigger Configuration</h5>
                      <div className="info-grid">
                        <div className="info-item">
                          <label>Trigger Mode</label>
                          <div className="info-value">{camera.trigger_config.trigger_mode ? 'Enabled' : 'Disabled'}</div>
                        </div>
                        <div className="info-item">
                          <label>Trigger Source</label>
                          <div className="info-value">{camera.trigger_config.trigger_source}</div>
                        </div>
                        <div className="info-item">
                          <label>Trigger Selector</label>
                          <div className="info-value">{camera.trigger_config.trigger_selector}</div>
                        </div>
                        <div className="info-item">
                          <label>Trigger Activation</label>
                          <div className="info-value">{camera.trigger_config.trigger_activation}</div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Model Thresholds */}
          <div className="info-section">
            <h3>Model Thresholds</h3>
            <div className="info-grid">
              <div className="info-item">
                <label>Detection Threshold</label>
                <div className="info-value">
                  {((recipe as any).modelThresholds?.detection_threshold || (recipe as any).model_thresholds?.detection_threshold) ? 
                    `${(((recipe as any).modelThresholds?.detection_threshold || (recipe as any).model_thresholds?.detection_threshold) * 100).toFixed(1)}%` : 'N/A'}
                </div>
              </div>
              <div className="info-item">
                <label>Recognition Threshold</label>
                <div className="info-value">
                  {((recipe as any).modelThresholds?.recognition_threshold || (recipe as any).model_thresholds?.recognition_threshold) ? 
                    `${(((recipe as any).modelThresholds?.recognition_threshold || (recipe as any).model_thresholds?.recognition_threshold) * 100).toFixed(1)}%` : 'N/A'}
                </div>
              </div>
              <div className="info-item">
                <label>Min Text Size</label>
                <div className="info-value">
                  {(recipe as any).modelThresholds?.min_text_size || (recipe as any).model_thresholds?.min_text_size || 'N/A'}
                </div>
              </div>
              <div className="info-item">
                <label>Max Text Size</label>
                <div className="info-value">
                  {(recipe as any).modelThresholds?.max_text_size || (recipe as any).model_thresholds?.max_text_size || 'N/A'}
                </div>
              </div>
            </div>
          </div>

          {/* Camera Templates */}
          {(recipe as any).camera_templates && (recipe as any).camera_templates.length > 0 && (
            <div className="info-section">
              <h3>Camera Templates</h3>
              {(recipe as any).camera_templates.map((camTemplate: any, camIndex: number) => (
                <div key={camIndex} className="camera-template">
                  <h4>Camera: {camTemplate.camera_id}</h4>
                  {camTemplate.templates && camTemplate.templates.map((template: any, tempIndex: number) => (
                    <div key={tempIndex} className="template-info">
                      <h5>Template: {template.name}</h5>
                      <div className="info-grid">
                        <div className="info-item">
                          <label>Image Size</label>
                          <div className="info-value">{template.image_width} × {template.image_height}</div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RecipeViewModal;
