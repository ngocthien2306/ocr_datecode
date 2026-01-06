import { useState, useEffect, useRef } from 'react';
import '@/styles/InferenceRealtime.css';
import { socketService } from '@/services/socketio';
import api from '@/services/api';
import { receiptsAPI } from '@/services/recipes';
import ConfirmDialog from '@/components/shared/ConfirmDialog';

interface InferenceLog {
  id: string;
  timestamp: string;
  result: 'PASS' | 'FAIL';
  recipe_name: string;
  product_code: string;
  message: string;
}

interface FrameResult {
  frame_idx: number;
  pass_fail: string;
  confidence: number;
  image_base64?: string;
  image_path?: string;
  template_name: string;
  timings?: any;
  detected_regions?: any[];
}

interface CameraResult {
  camera_id: string;
  serial_number: string;
  frames: FrameResult[];
}

interface PerCameraStats {
  serial_number: string;
  confidence: number;
  inliers: number;
  total_matches: number;
  timings: {
    total?: number;
    trt_inference?: number;
    preprocess?: number;
    postprocess?: number;
    [key: string]: number | undefined;
  };
  frame_stats: {
    total_frames: number;
    pass_count: number;
    fail_count: number;
    error_count: number;
    avg_confidence: number;
  };
}

interface InferenceResult {
  id: string;
  recipe_id: string;
  recipe_name: string;
  product_pass_fail: string;
  camera_results: CameraResult[];
  metadata: {
    total_cameras: number;
    total_frames: number;
    inference_stats: {
      avg_confidence: number;
      total_inliers: number;
      total_matches: number;
      per_camera_stats: PerCameraStats[];
      overall_timings: any;
    };
  };
}

interface InferenceRealtimeProps {
  runningRecipeId: string | null;
}

interface ConfirmDialogState {
  isOpen: boolean;
  title: string;
  message: string;
  type: 'warning' | 'danger' | 'info';
  onConfirm: (() => void) | null;
}

export default function InferenceRealtime({ runningRecipeId }: InferenceRealtimeProps) {
  const [logs, setLogs] = useState<InferenceLog[]>([]);
  const [latestResults, setLatestResults] = useState<InferenceResult | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [runningRecipe, setRunningRecipe] = useState<any>(null);
  const [isOnline, setIsOnline] = useState(true); // Inference mode: ONLINE/OFFLINE
  const [isStopping, setIsStopping] = useState(false);
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Confirmation dialog state
  const [confirmDialog, setConfirmDialog] = useState<ConfirmDialogState>({
    isOpen: false,
    title: '',
    message: '',
    type: 'warning',
    onConfirm: null
  });

  // Auto scroll to bottom
  const scrollToBottom = () => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [logs]);

  // Load running recipe info
  useEffect(() => {
    const loadRunningRecipe = async () => {
      if (!runningRecipeId) {
        setRunningRecipe(null);
        return;
      }

      try {
        const latest = await receiptsAPI.getLatestLoadedRecipe();
        if (latest && latest.recipe_id === runningRecipeId) {
          setRunningRecipe(latest);
        }
      } catch (error) {
        console.error('Error loading running recipe:', error);
      }
    };

    loadRunningRecipe();
  }, [runningRecipeId]);

  // Subscribe to realtime inference results
  useEffect(() => {
    if (!runningRecipeId) return;

    socketService.connect();

    const handleNewResult = (data: any) => {
      console.log('New inference result:', data);

      // Add to logs
      const newLog: InferenceLog = {
        id: data.id || `log-${Date.now()}`,
        timestamp: new Date().toLocaleTimeString(),
        result: data.product_pass_fail || 'FAIL',
        recipe_name: data.recipe_name || 'Unknown',
        product_code: data.product_code || 'Unknown',
        message: `${data.recipe_name} - ${data.product_pass_fail}`
      };

      setLogs(prev => [...prev, newLog].slice(-100)); // Keep last 100 logs

      // Store full inference result for multi-camera display
      setLatestResults(data as InferenceResult);
    };

    socketService.subscribeToInferenceResults(handleNewResult);

    return () => {
      socketService.unsubscribeFromInferenceResults(handleNewResult);
    };
  }, [runningRecipeId]);

  const handleSimulateTrigger = async () => {
    if (isSimulating) return;

    setIsSimulating(true);

    try {
      await api.post('/trigger-simulator/simulate', {
        trigger_type: 'rising_edge'
      });

      // Don't add log here - wait for real result from WebSocket
      // The result will be automatically added by handleNewResult callback
    } catch (error: any) {
      console.error('Error simulating trigger:', error);
      const errorLog: InferenceLog = {
        id: `error-${Date.now()}`,
        timestamp: new Date().toLocaleTimeString(),
        result: 'FAIL',
        recipe_name: '',
        product_code: '',
        message: `❌ Error: ${error.response?.data?.detail || 'Failed to simulate trigger'}`
      };

      setLogs(prev => [...prev, errorLog]);
    } finally {
      setTimeout(() => setIsSimulating(false), 1000);
    }
  };

  const handleStopRecipe = () => {
    if (!runningRecipeId || isStopping) return;

    setConfirmDialog({
      isOpen: true,
      title: 'Stop Recipe',
      message: `Stop recipe "${runningRecipe?.metadata?.name || 'Unknown'}"?\n\nThis will stop inference and set cameras to idle mode.`,
      type: 'danger',
      onConfirm: async () => {
        setIsStopping(true);

        try {
          await receiptsAPI.stopReceipt(runningRecipeId);

          // Add log
          const stopLog: InferenceLog = {
            id: `stop-${Date.now()}`,
            timestamp: new Date().toLocaleTimeString(),
            result: 'PASS',
            recipe_name: runningRecipe?.metadata?.name || 'Unknown',
            product_code: runningRecipe?.metadata?.product_code || '',
            message: '🛑 Recipe stopped'
          };
          setLogs(prev => [...prev, stopLog]);

          // Redirect to Recipes page after 1 second
          setTimeout(() => {
            const recipesLink = document.querySelector('a[href="#receipts"]') as HTMLAnchorElement;
            if (recipesLink) {
              recipesLink.click();
            }
          }, 1000);
        } catch (error: any) {
          console.error('Error stopping recipe:', error);
          alert(`Failed to stop recipe: ${error.response?.data?.detail || error.message}`);
        } finally {
          setIsStopping(false);
        }
      }
    });
  };

  const handleToggleInferenceMode = async () => {
    if (!runningRecipeId) return;

    const newMode = !isOnline;

    try {
      await receiptsAPI.setInferenceMode(runningRecipeId, newMode);
      setIsOnline(newMode);

      // Add log
      const modeLog: InferenceLog = {
        id: `mode-${Date.now()}`,
        timestamp: new Date().toLocaleTimeString(),
        result: 'PASS',
        recipe_name: runningRecipe?.metadata?.name || 'Unknown',
        product_code: runningRecipe?.metadata?.product_code || '',
        message: newMode ? '🟢 Inference ONLINE' : '⏸️ Inference OFFLINE'
      };
      setLogs(prev => [...prev, modeLog]);
    } catch (error: any) {
      console.error('Error toggling inference mode:', error);
      alert(`Failed to change mode: ${error.response?.data?.detail || error.message}`);
    }
  };

  if (!runningRecipeId) {
    return (
      <div className="inference-realtime-empty">
        <div className="empty-state">
          <svg width="64" height="64" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
            <path d="M12 6v6l4 2" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
          <h3>No Recipe Running</h3>
          <p>Load a recipe from the Recipes page to start inference</p>
        </div>
      </div>
    );
  }

  return (
    <div className="inference-realtime">
      <div className="realtime-header">
        <div className="header-info">
          <h2>Inference Realtime Monitor</h2>
          {runningRecipe && (
            <div className="running-recipe-info">
              <span className="recipe-badge">
                <span className="pulse-dot"></span>
                RUNNING
              </span>
              <span className="recipe-name">{runningRecipe.metadata?.name || 'Unknown Recipe'}</span>
              <span className="product-code">({runningRecipe.metadata?.product_code || 'N/A'})</span>
            </div>
          )}
        </div>
        <div className="header-actions">
          {/* Online/Offline Toggle Switch */}
          <div className="mode-toggle-wrapper">
            <span className="mode-label">Inference Mode:</span>
            <button
              className={`mode-toggle-switch ${isOnline ? 'online' : 'offline'}`}
              onClick={handleToggleInferenceMode}
              title={isOnline ? 'Click to switch to OFFLINE mode' : 'Click to switch to ONLINE mode'}
            >
              <span className="toggle-track">
                <span className="toggle-thumb"></span>
              </span>
              <span className="toggle-label">
                {isOnline ? (
                  <>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
                      <path d="M12 6v6l4 2" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                    </svg>
                    ONLINE
                  </>
                ) : (
                  <>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                      <rect x="6" y="6" width="12" height="12" stroke="currentColor" strokeWidth="2"/>
                    </svg>
                    OFFLINE
                  </>
                )}
              </span>
            </button>
          </div>

          {/* Simulate Trigger */}
          <button
            className={`btn-action btn-trigger ${isSimulating ? 'simulating' : ''}`}
            onClick={handleSimulateTrigger}
            disabled={isSimulating}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M13 2L3 14h8l-1 8 10-12h-8l1-8z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
            </svg>
            {isSimulating ? 'Triggering...' : 'Simulate Trigger'}
          </button>

          {/* Stop Recipe */}
          <button
            className={`btn-action btn-stop ${isStopping ? 'stopping' : ''}`}
            onClick={handleStopRecipe}
            disabled={isStopping}
            title="Stop recipe and return to idle"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <rect x="6" y="6" width="12" height="12" fill="currentColor"/>
            </svg>
            {isStopping ? 'Stopping...' : 'Stop Recipe'}
          </button>
        </div>
      </div>

      <div className="realtime-content">
        <div className="cameras-grid">
          <div className="panel-header">
            <h3>Latest Result</h3>
            {/* <span className="camera-count">
              {latestResults ? `${latestResults.camera_results.length} camera(s)` : 'No results'}
            </span> */}
          </div>

          {!latestResults ? (
            <div className="no-results">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
                <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="2"/>
                <circle cx="8.5" cy="8.5" r="1.5" fill="currentColor"/>
                <path d="M21 15l-5-5L5 21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              <p>Waiting for first result...</p>
            </div>
          ) : (
            <div className={`cameras-container cameras-${Math.min(latestResults.camera_results.length, 4)}`}>
              {latestResults.camera_results.map((cameraResult) => {
                // Find per-camera stats
                const cameraStats = latestResults.metadata?.inference_stats?.per_camera_stats?.find(
                  (s) => s.serial_number === cameraResult.serial_number
                );

                return (
                  <div key={cameraResult.serial_number} className="camera-card-infer">
                    {/* <div className="camera-card-header">
                      <div className="camera-info">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                          <rect x="2" y="7" width="20" height="14" rx="2" stroke="currentColor" strokeWidth="2"/>
                          <circle cx="12" cy="14" r="3" stroke="currentColor" strokeWidth="2"/>
                        </svg>
                        <span className="camera-serial">{cameraResult.serial_number}</span>
                      </div>
                      {cameraStats && (
                        <span className={`camera-status ${cameraStats.frame_stats.fail_count > 0 ? 'fail' : 'pass'}`}>
                          {cameraStats.frame_stats.fail_count > 0 ? 'FAIL' : 'PASS'}
                        </span>
                      )}
                    </div> */}

                    <div className="camera-frames">
                      {cameraResult.frames.map((frame) => {
                        const imageUrl = frame.image_base64
                          ? `data:image/jpeg;base64,${frame.image_base64}`
                          : frame.image_path
                            ? `/api/uploads/${frame.image_path}`
                            : null;

                        return (
                          <div key={frame.frame_idx} className="frame-container">
                            <div className="frame-aspect-wrapper">
                              {imageUrl ? (
                                <img src={imageUrl} alt={`Frame ${frame.frame_idx}`} className="frame-image" />
                              ) : (
                                <div className="frame-placeholder">
                                  <span>Frame {frame.frame_idx}</span>
                                </div>
                              )}
                              {/* <div className="frame-overlay">
                                <span className={`frame-badge ${frame.pass_fail.toLowerCase()}`}>
                                  {frame.pass_fail}
                                </span>
                                <span className="frame-confidence">
                                  {(frame.confidence * 100).toFixed(1)}%
                                </span>
                              </div> */}
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    {cameraStats && (
                      <>
                        <div className="camera-stats-compact">
                          <div className="stat-item">
                            <span className="stat-label">Confidence:</span>
                            <span className="stat-value">{(cameraStats.confidence * 100).toFixed(1)}%</span>
                          </div>
                          <div className="stat-separator">|</div>
                          <div className="stat-item">
                            <span className="stat-label">Time:</span>
                            <span className="stat-value">
                              {(
                                (cameraStats.timings.trt_inference || 0) + 
                                (cameraStats.timings.postprocess || 0)
                              ).toFixed(1)} ms
                            </span>
                          </div>
                          {/* <button
                            className="btn-details"
                            onClick={(e) => {
                              const details = e.currentTarget.parentElement?.nextElementSibling as HTMLElement;
                              if (details) {
                                const isHidden = details.style.display === 'none' || !details.style.display;
                                details.style.display = isHidden ? 'block' : 'none';
                                e.currentTarget.textContent = isHidden ? '▲' : '▼';
                              }
                            }}
                          >
                            ▼
                          </button> */}
                        </div>
                        <div className="stats-details" style={{ display: 'none' }}>
                          <div className="detail-row">
                            <span className="stat-label">├ TRT:</span>
                            <span className="stat-value">{cameraStats.timings.trt_inference?.toFixed(1) || 0}ms</span>
                          </div>
                          <div className="detail-row">
                            <span className="stat-label">├ Pre:</span>
                            <span className="stat-value">{cameraStats.timings.preprocess?.toFixed(1) || 0}ms</span>
                          </div>
                          <div className="detail-row">
                            <span className="stat-label">├ Post:</span>
                            <span className="stat-value">{cameraStats.timings.postprocess?.toFixed(1) || 0}ms</span>
                          </div>
                          <div className="detail-row">
                            <span className="stat-label">└ Inliers:</span>
                            <span className="stat-value">{cameraStats.inliers}/{cameraStats.total_matches}</span>
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <div className="log-panel">
          <div className="panel-header">
            <h3>Results Log</h3>
            <span className="log-count">{logs.length} entries</span>
          </div>
          <div className="log-container">
            {logs.length === 0 ? (
              <div className="no-logs">
                <p>No results yet. Click "Simulate Trigger" to test.</p>
              </div>
            ) : (
              <div className="log-list">
                {logs.map((log) => (
                  <div key={log.id} className={`inference-log-entry ${log.result.toLowerCase()}`}>
                    <span className="log-timestamp">{log.timestamp}</span>
                    <span className={`log-badge ${log.result.toLowerCase()}`}>
                      {log.result}
                    </span>
                    {/* <span className="log-message">{log.message}</span> */}
                  </div>
                ))}
                <div ref={logsEndRef} />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Confirm Dialog */}
      <ConfirmDialog
        isOpen={confirmDialog.isOpen}
        onClose={() => setConfirmDialog({ ...confirmDialog, isOpen: false })}
        onConfirm={() => {
          if (confirmDialog.onConfirm) {
            confirmDialog.onConfirm();
          }
          setConfirmDialog({ ...confirmDialog, isOpen: false });
        }}
        title={confirmDialog.title}
        message={confirmDialog.message}
        type={confirmDialog.type}
        confirmText="Confirm"
        cancelText="Cancel"
      />
    </div>
  );
}
