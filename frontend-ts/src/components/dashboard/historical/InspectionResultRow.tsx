import React from 'react';
import { type InferenceResultResponse } from '@/services/inferenceResults';

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
  // Format timestamp
  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  const formatDate = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
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
                <h5>📷 Camera Results:</h5>
                {result.camera_results.map((cameraResult, idx) => (
                  <div key={idx} className="camera-result-card">
                    <div className="camera-header">
                      <span className="camera-name">
                        {cameraResult.serial_number}
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
                      <div key={frameIdx} className="frame-details">
                        {/* Timings */}
                        {frame.timings && (
                          <div className="timing-info">
                            <span>⏱️ Processing: {frame.timings.total?.toFixed(0) || 0}ms</span>
                            {frame.timings.trt_inference && (
                              <span>(TRT: {frame.timings.trt_inference.toFixed(0)}ms)</span>
                            )}
                          </div>
                        )}

                        {/* Text Verification */}
                        {frame.text_verification && frame.text_verification.results && frame.text_verification.results.length > 0 && (
                          <div className="text-verification-section">
                            <h6>Text Verification:</h6>
                            {frame.text_verification.results.map((textResult, textIdx) => (
                              <div
                                key={textIdx}
                                className={`text-result ${textResult.match ? 'match' : 'mismatch'}`}
                              >
                                <div className="text-result-header">
                                  <span className="region-label">Region {textResult.region_idx}:</span>
                                  <span className={`match-badge ${textResult.match ? 'success' : 'error'}`}>
                                    {textResult.match ? '✓' : '✗'}
                                  </span>
                                  <span className="confidence-badge">
                                    {(textResult.confidence * 100).toFixed(1)}%
                                  </span>
                                </div>
                                <div className="text-comparison">
                                  <div className="expected">
                                    <strong>Expected:</strong> "{textResult.expected}"
                                  </div>
                                  <div className="recognized">
                                    <strong>Got:</strong> "{textResult.recognized || '(empty)'}"
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Image */}
                        {frame.image_path && (
                          <div className="frame-image">
                            <img
                              src={`/api/uploads/${frame.image_path}`}
                              alt={`Frame ${frameIdx}`}
                              onError={(e) => {
                                e.currentTarget.style.display = 'none';
                              }}
                            />
                          </div>
                        )}
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
    </>
  );
};

export default InspectionResultRow;
