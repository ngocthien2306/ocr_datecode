import { useState, useEffect } from 'react';
import { useToast } from '@/contexts/ToastContext';
import { API_BASE_URL } from '@/config/api';
import socketService from '@/services/socketio';
import '@/styles/InferenceResults.css';

interface InferenceFrame {
  template_name: string;
  frame_idx: number;
  pass_fail: 'PASS' | 'FAIL';
  confidence: number;
  image_path: string;
  detected_regions?: any[];
}

interface CameraResult {
  camera_id: string;
  serial_number: string;
  frames: InferenceFrame[];
}

interface InferenceResult {
  id: string;
  recipe_id: string;
  recipe_name: string;
  product_pass_fail: 'PASS' | 'FAIL';
  camera_results: CameraResult[];
  metadata?: any;
  timestamp: string;
  created_at: string;
}

export default function InferenceResults() {
  const toast = useToast();
  const [results, setResults] = useState<InferenceResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedResult, setSelectedResult] = useState<InferenceResult | null>(null);

  // Filters
  const [recipeFilter, setRecipeFilter] = useState('');
  const [passFailFilter, setPassFailFilter] = useState<'' | 'PASS' | 'FAIL'>('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

  // Pagination
  const [skip, setSkip] = useState(0);
  const [limit] = useState(50);
  const [totalCount, setTotalCount] = useState(0);

  useEffect(() => {
    loadResults();
  }, [skip, recipeFilter, passFailFilter, startDate, endDate]);

  // SocketIO real-time updates
  useEffect(() => {
    // Connect to SocketIO
    socketService.connect();

    // Handle new inference results
    const handleNewResult = (data: any) => {
      console.log('New inference result received:', data);
      toast.success(`New result: ${data.recipe_name} - ${data.product_pass_fail}`);

      // Reload results if on first page
      if (skip === 0) {
        loadResults();
      }
    };

    // Subscribe to inference results
    socketService.subscribeToInferenceResults(handleNewResult);

    // Cleanup on unmount
    return () => {
      socketService.unsubscribeFromInferenceResults(handleNewResult);
    };
  }, [skip]);

  const loadResults = async () => {
    setLoading(true);
    try {
      const token = localStorage.getItem('access_token');
      const params = new URLSearchParams({
        skip: skip.toString(),
        limit: limit.toString(),
      });

      if (recipeFilter) params.append('recipe_id', recipeFilter);
      if (passFailFilter) params.append('pass_fail', passFailFilter);
      if (startDate) params.append('start_date', startDate);
      if (endDate) params.append('end_date', endDate);

      const [resultsResponse, countResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/api/inference-results/?${params}`, {
          headers: { 'Authorization': `Bearer ${token}` }
        }),
        fetch(`${API_BASE_URL}/api/inference-results/count?${params}`, {
          headers: { 'Authorization': `Bearer ${token}` }
        })
      ]);

      if (resultsResponse.ok && countResponse.ok) {
        const resultsData = await resultsResponse.json();
        const countData = await countResponse.json();
        setResults(resultsData);
        setTotalCount(countData.count);
      } else {
        throw new Error('Failed to load results');
      }
    } catch (error: any) {
      console.error('Error loading inference results:', error);
      toast.error('Failed to load inference results');
    } finally {
      setLoading(false);
    }
  };

  const handleViewDetails = (result: InferenceResult) => {
    setSelectedResult(result);
  };

  const handleCloseDetails = () => {
    setSelectedResult(null);
  };

  const handleDelete = async (resultId: string) => {
    if (!confirm('Are you sure you want to delete this result?')) return;

    try {
      const token = localStorage.getItem('access_token');
      const response = await fetch(`${API_BASE_URL}/api/inference-results/${resultId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (response.ok) {
        toast.success('Result deleted successfully');
        loadResults();
        if (selectedResult?.id === resultId) {
          setSelectedResult(null);
        }
      } else {
        throw new Error('Failed to delete result');
      }
    } catch (error) {
      console.error('Error deleting result:', error);
      toast.error('Failed to delete result');
    }
  };

  const handleApplyFilters = () => {
    setSkip(0);
    loadResults();
  };

  const handleClearFilters = () => {
    setRecipeFilter('');
    setPassFailFilter('');
    setStartDate('');
    setEndDate('');
    setSkip(0);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  return (
    <div className="inference-results-container">
      <div className="inference-header">
        <h2>Inference Results</h2>
        <div className="inference-stats">
          <div className="stat-item">
            <span className="stat-label">Total Results:</span>
            <span className="stat-value">{totalCount}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Current Page:</span>
            <span className="stat-value">{Math.floor(skip / limit) + 1} / {Math.ceil(totalCount / limit)}</span>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="inference-filters">
        <div className="filter-group">
          <label>Recipe ID:</label>
          <input
            type="text"
            value={recipeFilter}
            onChange={(e) => setRecipeFilter(e.target.value)}
            placeholder="Filter by recipe ID"
          />
        </div>
        <div className="filter-group">
          <label>Result:</label>
          <select value={passFailFilter} onChange={(e) => setPassFailFilter(e.target.value as any)}>
            <option value="">All</option>
            <option value="PASS">PASS</option>
            <option value="FAIL">FAIL</option>
          </select>
        </div>
        <div className="filter-group">
          <label>Start Date:</label>
          <input
            type="datetime-local"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
          />
        </div>
        <div className="filter-group">
          <label>End Date:</label>
          <input
            type="datetime-local"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
          />
        </div>
        <div className="filter-actions">
          <button className="btn-apply" onClick={handleApplyFilters}>Apply</button>
          <button className="btn-clear" onClick={handleClearFilters}>Clear</button>
        </div>
      </div>

      {/* Results Table */}
      <div className="inference-table-container">
        {loading ? (
          <div className="loading-state">Loading results...</div>
        ) : results.length === 0 ? (
          <div className="empty-state">
            <p>No inference results found</p>
            <p className="hint">Results will appear here as products are inspected</p>
          </div>
        ) : (
          <table className="inference-table">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Recipe</th>
                <th>Product Code</th>
                <th>Result</th>
                <th>Cameras</th>
                <th>Total Frames</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {results.map((result) => {
                const totalFrames = result.camera_results.reduce((sum, cam) => sum + cam.frames.length, 0);
                return (
                  <tr key={result.id}>
                    <td>{formatDate(result.timestamp)}</td>
                    <td>{result.recipe_name}</td>
                    <td>{result.metadata?.product_code || 'N/A'}</td>
                    <td>
                      <span className={`badge badge-${result.product_pass_fail.toLowerCase()}`}>
                        {result.product_pass_fail}
                      </span>
                    </td>
                    <td>{result.camera_results.length}</td>
                    <td>{totalFrames}</td>
                    <td>
                      <div className="action-buttons">
                        <button
                          className="btn-view"
                          onClick={() => handleViewDetails(result)}
                          title="View Details"
                        >
                          👁️ View
                        </button>
                        <button
                          className="btn-delete"
                          onClick={() => handleDelete(result.id)}
                          title="Delete Result"
                        >
                          🗑️
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Pagination */}
      {totalCount > limit && (
        <div className="inference-pagination">
          <button
            className="btn-page"
            onClick={() => setSkip(Math.max(0, skip - limit))}
            disabled={skip === 0}
          >
            ← Previous
          </button>
          <span className="page-info">
            Showing {skip + 1}-{Math.min(skip + limit, totalCount)} of {totalCount}
          </span>
          <button
            className="btn-page"
            onClick={() => setSkip(skip + limit)}
            disabled={skip + limit >= totalCount}
          >
            Next →
          </button>
        </div>
      )}

      {/* Details Modal */}
      {selectedResult && (
        <div className="inference-modal-overlay" onClick={handleCloseDetails}>
          <div className="inference-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Inference Result Details</h3>
              <button className="btn-close" onClick={handleCloseDetails}>✕</button>
            </div>
            <div className="modal-body">
              <div className="result-summary">
                <div className="summary-item">
                  <strong>Recipe:</strong> {selectedResult.recipe_name}
                </div>
                <div className="summary-item">
                  <strong>Product Code:</strong> {selectedResult.metadata?.product_code || 'N/A'}
                </div>
                <div className="summary-item">
                  <strong>Result:</strong>
                  <span className={`badge badge-${selectedResult.product_pass_fail.toLowerCase()}`}>
                    {selectedResult.product_pass_fail}
                  </span>
                </div>
                <div className="summary-item">
                  <strong>Timestamp:</strong> {formatDate(selectedResult.timestamp)}
                </div>
              </div>

              {/* Camera Results */}
              {selectedResult.camera_results.map((cameraResult, camIdx) => (
                <div key={camIdx} className="camera-result-section">
                  <h4>Camera: {cameraResult.camera_id} ({cameraResult.serial_number})</h4>
                  <div className="frames-grid">
                    {cameraResult.frames.map((frame, frameIdx) => (
                      <div key={frameIdx} className="frame-card">
                        <div className="frame-header">
                          <span className="frame-title">{frame.template_name}</span>
                          <span className={`frame-badge badge-${frame.pass_fail.toLowerCase()}`}>
                            {frame.pass_fail}
                          </span>
                        </div>
                        <div className="frame-image">
                          <img
                            src={`${API_BASE_URL}${frame.image_path}`}
                            alt={`Frame ${frame.frame_idx}`}
                            onError={(e) => {
                              (e.target as HTMLImageElement).src = '/placeholder-image.png';
                            }}
                          />
                        </div>
                        <div className="frame-info">
                          <div className="info-row">
                            <span>Frame Index:</span>
                            <span>{frame.frame_idx}</span>
                          </div>
                          <div className="info-row">
                            <span>Confidence:</span>
                            <span>{(frame.confidence * 100).toFixed(2)}%</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
