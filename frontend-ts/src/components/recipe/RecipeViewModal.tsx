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
                <label>Recipe ID</label>
                <div className="info-value">{recipe.id}</div>
              </div>
              <div className="info-item">
                <label>Recipe Name</label>
                <div className="info-value">{recipe.name}</div>
              </div>
              <div className="info-item">
                <label>Product Code</label>
                <div className="info-value">{recipe.productCode}</div>
              </div>
              <div className="info-item">
                <label>Status</label>
                <div className="info-value">
                  <span className={`status-badge ${recipe.status === 'Active' ? 'active' : 'inactive'}`}>
                    {recipe.status}
                  </span>
                </div>
              </div>
            </div>
            {recipe.description && (
              <div className="info-item full-width">
                <label>Description</label>
                <div className="info-value">{recipe.description}</div>
              </div>
            )}
          </div>

          {/* Camera Settings */}
          <div className="info-section">
            <h3>Camera Settings</h3>
            <div className="info-grid">
              <div className="info-item">
                <label>Exposure Time</label>
                <div className="info-value">{recipe.cameraSettings?.exposure_time} ms</div>
              </div>
              <div className="info-item">
                <label>Gain</label>
                <div className="info-value">{recipe.cameraSettings?.gain || 'N/A'}</div>
              </div>
            </div>
          </div>

          {/* Model Thresholds */}
          <div className="info-section">
            <h3>Model Thresholds</h3>
            <div className="info-grid">
              <div className="info-item">
                <label>Detection Threshold</label>
                <div className="info-value">
                  {recipe.modelThresholds?.detection_threshold ? 
                    `${(recipe.modelThresholds.detection_threshold * 100).toFixed(1)}%` : 'N/A'}
                </div>
              </div>
              <div className="info-item">
                <label>OCR Threshold</label>
                <div className="info-value">
                  {recipe.modelThresholds?.ocr_threshold ? 
                    `${(recipe.modelThresholds.ocr_threshold * 100).toFixed(1)}%` : 'N/A'}
                </div>
              </div>
            </div>
          </div>

          {/* Metadata */}
          <div className="info-section">
            <h3>Metadata</h3>
            <div className="info-grid">
              <div className="info-item">
                <label>Created By</label>
                <div className="info-value">{recipe.operator}</div>
              </div>
              <div className="info-item">
                <label>Created At</label>
                <div className="info-value">
                  {recipe.createdAt ? new Date(recipe.createdAt).toLocaleString() : recipe.date}
                </div>
              </div>
              <div className="info-item">
                <label>Updated At</label>
                <div className="info-value">
                  {recipe.updatedAt ? new Date(recipe.updatedAt).toLocaleString() : 'N/A'}
                </div>
              </div>
            </div>
          </div>

          {/* Configuration JSON */}
          {(recipe.template_config || recipe.roi_config) && (
            <div className="info-section">
              <h3>Advanced Configuration</h3>
              {recipe.template_config && (
                <div className="config-block">
                  <label>Template Configuration</label>
                  <pre className="config-json">
                    {JSON.stringify(recipe.template_config, null, 2)}
                  </pre>
                </div>
              )}
              {recipe.roi_config && (
                <div className="config-block">
                  <label>ROI Configuration</label>
                  <pre className="config-json">
                    {JSON.stringify(recipe.roi_config, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RecipeViewModal;
