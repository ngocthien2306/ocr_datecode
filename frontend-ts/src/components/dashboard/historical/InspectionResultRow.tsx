import React, { useState, useEffect } from 'react';
import { type InferenceResultResponse } from '@/services/inferenceResults';
import { formatTimestamp } from '@/utils/timezone';
import { API_BASE_URL } from '@/config/api';
import ImageModal from '@/components/shared/ImageModal';
import { camerasAPI } from '@/services/api';
import { receiptsAPI } from '@/services/recipes';
import type { Camera } from '@/types';

// Helper: extract numeric value from threshold (number or {parsedValue} object)
const getThresholdNum = (val: any): string => {
  if (val == null) return '-';
  if (typeof val === 'number') return val.toFixed(1);
  if (typeof val === 'object' && val.parsedValue != null) return Number(val.parsedValue).toFixed(1);
  return '-';
};

interface InspectionResultRowProps {
  result: InferenceResultResponse;
  isExpanded: boolean;
  isSelected: boolean;
  onToggleExpand: () => void;
  onToggleSelect: () => void;
  onDelete: () => void;
  canDelete: boolean;
}

const InspectionResultRow: React.FC<InspectionResultRowProps> = ({
  result,
  isExpanded,
  isSelected,
  onToggleExpand,
  onToggleSelect,
  onDelete,
  canDelete
}) => {
  // Image modal state
  const [modalImage, setModalImage] = useState<{ src: string; alt: string } | null>(null);

  // Camera lookup map
  const [cameraMap, setCameraMap] = useState<Map<string, Camera>>(new Map());

  // Template images from recipe load
  const [templateImages, setTemplateImages] = useState<Map<string, string>>(new Map());

  // Fetch cameras and template images when expanded
  useEffect(() => {
    const fetchCameras = async () => {
      try {
        const cameras = await camerasAPI.getAllCameras(0, 20);
        const map = new Map<string, Camera>();
        cameras.forEach(camera => {
          if (camera.serial_number) {
            map.set(camera.serial_number, camera);
          }
        });
        setCameraMap(map);
      } catch (error) {
        console.error('Error fetching cameras:', error);
      }
    };

    const fetchTemplateImages = async () => {
      try {
        // Fetch template images for each camera
        const templateMap = new Map<string, string>();

        for (const cameraResult of result.camera_results) {
          try {
            const response = await receiptsAPI.getTemplateImagesAtTimestamp(
              result.recipe_id,
              result.timestamp,
              cameraResult.serial_number
            );

            // Map template name to visualization URL
            for (const camTemplate of response.templates) {
              for (const template of camTemplate.templates) {
                const key = `${cameraResult.serial_number}_${template.name}`;
                templateMap.set(key, template.visualization_url);
              }
            }
          } catch (error) {
            console.error(`Error fetching template images for camera ${cameraResult.serial_number}:`, error);
          }
        }

        setTemplateImages(templateMap);
      } catch (error) {
        console.error('Error fetching template images:', error);
      }
    };

    if (isExpanded) {
      fetchCameras();
      fetchTemplateImages();
    }
  }, [isExpanded, result.recipe_id, result.timestamp, result.camera_results]);

  // Format timestamp using timezone utility
  const formatTime = (timestamp: string) => {
    return formatTimestamp(timestamp, 'time');
  };

  const formatDate = (timestamp: string) => {
    return formatTimestamp(timestamp, 'date');
  };

  // Open image modal
  const handleImageClick = (imagePath: string, frameIdx: number, templateName: string) => {
    const imageUrl = `${API_BASE_URL}/api/uploads/${imagePath}`;
    const alt = `Frame ${frameIdx + 1}: ${templateName} - Inspection Image`;
    setModalImage({ src: imageUrl, alt });
  };

  // Close image modal
  const closeModal = () => {
    setModalImage(null);
  };

  // Get camera display name
  const getCameraDisplayName = (serialNumber: string): string => {
    const camera = cameraMap.get(serialNumber);
    if (!camera) {
      return serialNumber; // Fallback to serial number
    }

    // Show name and location if available
    const parts: string[] = [];
    if (camera.name) {
      parts.push(camera.name);
    }
    if (camera.location) {
      parts.push(`${camera.location}`);
    }

    return parts.length > 0 ? parts.join(' ') : serialNumber;
  };

  // Calculate camera stats
  const totalCameras = result.camera_results.length;
  const camerasWithPass = result.camera_results.filter(cr =>
    cr.frames.some(f => f.pass_fail === 'PASS')
  ).length;

  // Calculate text verification stats
  const getAllTextVerificationResults = () => {
    const allResults: any[] = [];
    result.camera_results.forEach(cr => {
      cr.frames.forEach(frame => {
        if (frame.text_verification && frame.text_verification.results) {
          allResults.push(...frame.text_verification.results);
        }
      });
    });
    return allResults;
  };

  const textVerResults = getAllTextVerificationResults();
  const hasTextVerification = textVerResults.length > 0;
  const textMatches = textVerResults.filter(r => r.match).length;
  const textTotal = textVerResults.length;

  // Get average confidence
  const avgConfidence = result.metadata?.inference_stats?.avg_confidence || 0;

  return (
    <>
      <tr className={`result-row ${isExpanded ? 'expanded' : ''} ${isSelected ? 'selected' : ''}`}>
        {canDelete && (
          <td onClick={(e) => e.stopPropagation()}>
            <input
              type="checkbox"
              checked={isSelected}
              onChange={(e) => {
                e.stopPropagation();
                onToggleSelect();
              }}
              onClick={(e) => e.stopPropagation()}
              className="row-checkbox"
            />
          </td>
        )}
        <td>
          <button
            className="expand-btn"
            onClick={(e) => {
              e.stopPropagation();
              onToggleExpand();
            }}
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              style={{
                transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)',
                transition: 'transform 0.2s'
              }}
            >
              <polyline points="9,18 15,12 9,6" stroke="currentColor" strokeWidth="2"/>
            </svg>
          </button>
        </td>
        <td>
          <div className="time-cell">
            <div className="time">{formatTime(result.timestamp)}</div>
            <div className="date">{formatDate(result.timestamp)}</div>
          </div>
        </td>
        <td>
          <div className="recipe-cell">
            <div className="recipe-name">{result.recipe_name}</div>
          </div>
        </td>
        <td>
          <span className={`status-badge ${result.product_pass_fail.toLowerCase()}`}>
            {result.product_pass_fail}
          </span>
        </td>
        <td>
          <span className="camera-count">
            {camerasWithPass}/{totalCameras}
          </span>
        </td>
        <td>
          {hasTextVerification ? (
            <span className={`text-verify-badge ${textMatches === textTotal ? 'success' : 'error'}`}>
              {textMatches}/{textTotal} {textMatches === textTotal ? '✓' : '✗'}
            </span>
          ) : (
            <span className="text-verify-badge na">N/A</span>
          )}
        </td>
        <td>
          <span className="confidence-value">
            {(avgConfidence * 100).toFixed(1)}%
          </span>
        </td>
        <td>
          <div className="action-buttons">
            <button
              className="action-btn view"
              onClick={onToggleExpand}
              title="View details"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                <path d="M1 12S5 4 12 4s11 8 11 8-4 8-11 8S1 12 1 12z" stroke="currentColor" strokeWidth="2"/>
                <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2"/>
              </svg>
            </button>
            {canDelete && (
              <button
                className="action-btn delete"
                onClick={onDelete}
                title="Delete"
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                  <polyline points="3,6 5,6 21,6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6m3 0V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
            )}
          </div>
        </td>
      </tr>

      {/* Expanded Detail Row */}
      {isExpanded && (
        <tr className="expanded-detail-row">
          <td colSpan={canDelete ? 9 : 8}>
            <div className="expanded-detail">
              <h4>Inspection Details - ID: {result.id}</h4>

              {/* Camera Results */}
              <div className="camera-results-section">
                <h5 style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                    <rect x="2" y="4" width="20" height="16" rx="2" stroke="currentColor" strokeWidth="2"/>
                    <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2"/>
                    <path d="M7 4V2M17 4V2" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                  <span>Camera Results:</span>
                </h5>
                {result.camera_results.map((cameraResult, idx) => (
                  <div key={idx} className="camera-result-card">
                    <div className="camera-header">
                      <span className="camera-name">
                        {getCameraDisplayName(cameraResult.serial_number)}
                      </span>
                      <span className="camera-serial" style={{ fontSize: '12px', color: '#9ca3af', marginLeft: '8px' }}>
                        SN: {cameraResult.serial_number}
                      </span>
                      {/* <span className={`camera-status ${cameraResult.frames[0]?.pass_fail.toLowerCase()}`}>
                        {cameraResult.frames[0]?.pass_fail}
                      </span> */}
                      {cameraResult.frames[0]?.confidence !== undefined && (
                        <span className="camera-confidence">
                          Conf: {(cameraResult.frames[0].confidence * 100).toFixed(1)}%
                        </span>
                      )}
                    </div>

                    {/* Frames */}
                    {cameraResult.frames.map((frame, frameIdx) => (
                      <div key={frameIdx} className="frame-card">
                        {/* Frame Header */}
                        <div className={`frame-header ${frame.pass_fail.toLowerCase()}`}>
                          <span className="frame-title">
                            📸 Frame {frameIdx}: {frame.template_name}
                          </span>
                          <span className={`frame-status ${frame.pass_fail.toLowerCase()}`}>
                            {frame.pass_fail === 'PASS' ? '✓' : '✗'} {frame.pass_fail}
                          </span>
                        </div>

                        <div className="frame-body">
                          {/* Detailed Timings */}
                          {frame.timings && (
                            <div className="timing-section">
                              <span className="timing-icon">⏱️</span>
                              <span className="timing-details">
                                Processing: <strong>{frame.timings.total?.toFixed(0) || 0}ms</strong>
                                {frame.timings.trt_inference && (
                                  <> (TRT: {frame.timings.trt_inference.toFixed(0)}ms</>
                                )}
                                {frame.timings.preprocess && (
                                  <> | Pre: {frame.timings.preprocess.toFixed(0)}ms</>
                                )}
                                {frame.timings.postprocess && (
                                  <> | Post: {frame.timings.postprocess.toFixed(0)}ms</>
                                )}
                                {frame.timings.trt_inference && <>)</>}
                              </span>
                            </div>
                          )}

                          {/* Template Verification */}
                          {frame.template_verification && (
                            <div className="template-verification-section">
                              <div className="section-title">🎯 Template Verification:</div>
                              <div className="template-info">
                                <span className={`template-match ${frame.template_verification.match ? 'success' : 'error'}`}>
                                  {frame.template_verification.match ? '✓' : '✗'} Match: {(frame.template_verification.similarity * 100).toFixed(1)}%
                                </span>
                                <span className="template-threshold">
                                  Threshold: {(frame.template_verification.threshold * 100).toFixed(1)}%
                                </span>
                              </div>
                            </div>
                          )}

                          {/* Text Verification */}
                          {frame.text_verification && frame.text_verification.results && frame.text_verification.results.length > 0 && (
                            <div className="text-verification-section-v2">
                              <div className="section-title">
                                📝 Text Verification:
                                <span className={`verification-summary ${frame.text_verification.all_match ? 'success' : 'error'}`}>
                                  {frame.text_verification.results.filter(r => r.match).length}/{frame.text_verification.results.length}
                                  {frame.text_verification.all_match ? ' ✓' : ' ✗'}
                                </span>
                              </div>
                              <div
                                className="text-results-grid"
                                style={{
                                  display: 'grid',
                                  gridTemplateColumns: 'repeat(2, 1fr)',
                                  gap: '12px',
                                  marginTop: '12px'
                                }}
                              >
                                {frame.text_verification.results.map((textResult, textIdx) => (
                                  <div
                                    key={textIdx}
                                    className={`text-result-card ${textResult.match ? 'match' : 'mismatch'}`}
                                  >
                                    <div className="text-result-header-v2">
                                      <span className="region-label">Region {textResult.annotation_idx}:</span>
                                      <span className={`match-badge ${textResult.match ? 'success' : 'error'}`}>
                                        {textResult.match ? '✓' : '✗'} {(textResult.confidence * 100).toFixed(1)}%
                                      </span>
                                    </div>
                                    <div className="text-comparison-v2">
                                      <div className="text-line">
                                        <span className="text-label">Expected:</span>
                                        <span className="text-value">"{textResult.expected}"</span>
                                      </div>
                                      <div className="text-line">
                                        <span className="text-label">Recognized:</span>
                                        <span className="text-value">"{textResult.recognized || '(empty)'}"</span>
                                      </div>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Product Verification */}
                          {frame.product_verification && (
                            <div className="product-verification-section">
                              <div className="section-title">
                                📦 Product Verification:
                                <span className={`verification-summary ${
                                  frame.product_verification.skipped
                                    ? 'na'
                                    : frame.product_verification.match
                                    ? 'success'
                                    : 'error'
                                }`}>
                                  {frame.product_verification.skipped
                                    ? 'Skipped'
                                    : frame.product_verification.match
                                    ? '✓ PASS'
                                    : '✗ FAIL'}
                                </span>
                              </div>
                              <div className="center-alignment-label">Center Alignment:</div>
                              <div className="center-alignment-grid">
                                <div className="alignment-item">
                                  <span className="alignment-label">Status</span>
                                  <span className={`alignment-value ${
                                    frame.product_verification.center_alignment_check
                                      ? frame.product_verification.center_alignment_check.ok ? 'success' : 'error'
                                      : ''
                                  }`}>
                                    {frame.product_verification.center_alignment_check
                                      ? frame.product_verification.center_alignment_check.ok ? '✓ OK' : '✗ FAIL'
                                      : '-'}
                                  </span>
                                </div>
                                <div className="alignment-item">
                                  <span className="alignment-label">Offset</span>
                                  <span className="alignment-value">
                                    {frame.product_verification.center_alignment_check
                                      ? `${(
                                          frame.product_verification.center_alignment_check.abs_offset_x ??
                                          Math.abs(frame.product_verification.center_alignment_check.offset_x)
                                        ).toFixed(1)}px`
                                      : '-'}
                                  </span>
                                </div>
                                <div className="alignment-item">
                                  <span className="alignment-label">Direction</span>
                                  <span className="alignment-value">
                                    {frame.product_verification.center_alignment_check?.direction || '-'}
                                  </span>
                                </div>
                                <div className="alignment-item">
                                  <span className="alignment-label">Threshold L</span>
                                  <span className="alignment-value">
                                    {frame.product_verification.center_alignment_check
                                      ? `${getThresholdNum(frame.product_verification.center_alignment_check.threshold_left)}px`
                                      : '-'}
                                  </span>
                                </div>
                                <div className="alignment-item">
                                  <span className="alignment-label">Threshold R</span>
                                  <span className="alignment-value">
                                    {frame.product_verification.center_alignment_check
                                      ? `${getThresholdNum(frame.product_verification.center_alignment_check.threshold_right)}px`
                                      : '-'}
                                  </span>
                                </div>
                              </div>
                              {frame.product_verification.skipped && frame.product_verification.reason && (
                                <div className="skip-reason">{frame.product_verification.reason}</div>
                              )}
                            </div>
                          )}

                          {/* Image Comparison */}
                          <div className="inspection-image-section">
                            <div className="section-title">📷 Image Comparison:</div>
                            <div className="image-comparison-grid">
                              {/* Template Image */}
                              <div className="comparison-item">
                                <div className="comparison-label">Original Template</div>
                                {(() => {
                                  const templateKey = `${cameraResult.serial_number}_${frame.template_name}`;
                                  const templateUrl = templateImages.get(templateKey);

                                  return templateUrl ? (
                                    <div className="image-container">
                                      <img
                                        src={`${API_BASE_URL}${templateUrl}`}
                                        alt={`Template: ${frame.template_name}`}
                                        className="inspection-thumbnail"
                                        onClick={() => {
                                          setModalImage({
                                            src: `${API_BASE_URL}${templateUrl}`,
                                            alt: `Template: ${frame.template_name}`
                                          });
                                        }}
                                        onError={(e) => {
                                          e.currentTarget.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300"><rect fill="%23ddd" width="400" height="300"/><text x="50%" y="50%" text-anchor="middle" fill="%23999">Template not found</text></svg>';
                                        }}
                                      />
                                      <div className="image-overlay">
                                        <span>Click to view full size</span>
                                      </div>
                                    </div>
                                  ) : (
                                    <div className="no-image">
                                      Loading template...
                                    </div>
                                  );
                                })()}
                              </div>

                              {/* Inspection Result Image */}
                              <div className="comparison-item">
                                <div className="comparison-label">Inspection Result</div>
                                {frame.image_path ? (
                                  <div className="image-container">
                                    <img
                                      src={`${API_BASE_URL}/api/uploads/${frame.image_path}`}
                                      alt={`Frame ${frameIdx}`}
                                      className="inspection-thumbnail"
                                      onClick={() => handleImageClick(frame.image_path!, frameIdx, frame.template_name)}
                                      onError={(e) => {
                                        e.currentTarget.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300"><rect fill="%23ddd" width="400" height="300"/><text x="50%" y="50%" text-anchor="middle" fill="%23999">Image not found</text></svg>';
                                      }}
                                    />
                                    <div className="image-overlay">
                                      <span>Click to view full size</span>
                                    </div>
                                  </div>
                                ) : (
                                  <div className="no-image">
                                    N/A ({frame.pass_fail} - no image saved)
                                  </div>
                                )}
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ))}
              </div>

              {/* Metadata Stats */}
              {result.metadata?.inference_stats && (
                <div className="metadata-section hidden">
                  <h5>📊 Inference Statistics:</h5>
                  <div className="stats-grid">
                    <div className="stat-item">
                      <span className="stat-label">Total Cameras:</span>
                      <span className="stat-value">{result.metadata.total_cameras}</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Total Frames:</span>
                      <span className="stat-value">{result.metadata.total_frames}</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Avg Confidence:</span>
                      <span className="stat-value">
                        {(result.metadata.inference_stats.avg_confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Total Inliers:</span>
                      <span className="stat-value">{result.metadata.inference_stats.total_inliers}</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Total Matches:</span>
                      <span className="stat-value">{result.metadata.inference_stats.total_matches}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </td>
        </tr>
      )}

      {/* Image Modal */}
      {modalImage && (
        <ImageModal
          isOpen={!!modalImage}
          imageSrc={modalImage.src}
          imageAlt={modalImage.alt}
          onClose={closeModal}
        />
      )}
    </>
  );
};

export default InspectionResultRow;
