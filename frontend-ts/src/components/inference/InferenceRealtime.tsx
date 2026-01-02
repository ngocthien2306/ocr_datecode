import { useState, useEffect, useRef } from 'react';
import '@/styles/InferenceRealtime.css';
import { socketService } from '@/services/socketio';
import api from '@/services/api';
import { receiptsAPI } from '@/services/recipes';

interface InferenceLog {
  id: string;
  timestamp: string;
  result: 'PASS' | 'FAIL';
  recipe_name: string;
  product_code: string;
  message: string;
}

interface InferenceRealtimeProps {
  runningRecipeId: string | null;
}

export default function InferenceRealtime({ runningRecipeId }: InferenceRealtimeProps) {
  const [logs, setLogs] = useState<InferenceLog[]>([]);
  const [latestImage, setLatestImage] = useState<string | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [runningRecipe, setRunningRecipe] = useState<any>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);

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

      // Update latest image if available (base64 from WebSocket)
      if (data.camera_results && data.camera_results.length > 0) {
        const firstCamera = data.camera_results[0];
        if (firstCamera.frames && firstCamera.frames.length > 0) {
          const frame = firstCamera.frames[0];

          // Use base64 image if available (realtime)
          if (frame.image_base64) {
            setLatestImage(`data:image/jpeg;base64,${frame.image_base64}`);
          }
          // Fallback to URL path (historical)
          else if (frame.image_path) {
            setLatestImage(`/api/uploads/${frame.image_path}`);
          }
        }
      }
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
        <button
          className={`btn-trigger ${isSimulating ? 'simulating' : ''}`}
          onClick={handleSimulateTrigger}
          disabled={isSimulating}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M13 2L3 14h8l-1 8 10-12h-8l1-8z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
          </svg>
          {isSimulating ? 'Triggering...' : 'Simulate Trigger'}
        </button>
      </div>

      <div className="realtime-content">
        <div className="image-panel">
          <div className="panel-header">
            <h3>Latest Result</h3>
            <span className="image-count">{latestImage ? '1 image' : 'No image'}</span>
          </div>
          <div className="image-container">
            {latestImage ? (
              <img src={latestImage} alt="Latest inference result" />
            ) : (
              <div className="no-image">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
                  <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="2"/>
                  <circle cx="8.5" cy="8.5" r="1.5" fill="currentColor"/>
                  <path d="M21 15l-5-5L5 21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                <p>Waiting for first result...</p>
              </div>
            )}
          </div>
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
                    <span className="log-message">{log.message}</span>
                  </div>
                ))}
                <div ref={logsEndRef} />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
