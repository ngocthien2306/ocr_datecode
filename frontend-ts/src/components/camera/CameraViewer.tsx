import React, { useState, useEffect, useRef } from 'react';
import { camerasAPI } from '@/services/api';
import { useToast } from '@/contexts/ToastContext';
import { socketService } from '@/services/socketio';
import type { Camera } from '@/types';
import '@/styles/CameraViewer.css';

interface CameraViewerProps {
  camera: Camera;
  onBack: () => void;
}

interface CameraSettings {
  exposure: number;
  gain: number;
  quality: number;
  autoRefresh: boolean;
  refreshInterval: number;
}

const CameraViewer: React.FC<CameraViewerProps> = ({ camera, onBack }) => {
  const toast = useToast();
  const [isLiveMode, setIsLiveMode] = useState(false);
  const [frameUrl, setFrameUrl] = useState<string>('');
  const [lastCapture, setLastCapture] = useState<Date | null>(null);
  const [settings, setSettings] = useState<CameraSettings>({
    exposure: 500,
    gain: 1.0,
    quality: 85,
    autoRefresh: false,
    refreshInterval: 100
  });
  const [isApplyingSettings, setIsApplyingSettings] = useState(false);
  const [hasUnsavedSettings, setHasUnsavedSettings] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  const serialNumber = (camera as any).serial_number || camera.camera_id;

  useEffect(() => {
    // Load camera settings on mount
    const loadSettings = async () => {
      try {
        const response = await camerasAPI.getCameraSettings(serialNumber);
        if (response.settings) {
          setSettings(prev => ({
            ...prev,
            exposure: response.settings.exposure_time || 10000,
            gain: response.settings.gain || 1.0
          }));
        }
      } catch (err) {
        console.error('Error loading camera settings:', err);
      }
    };

    loadSettings();

    return () => {
      // Cleanup on unmount
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [serialNumber]);

  useEffect(() => {
    if (isLiveMode) {
      // Use SocketIO streaming (Backend reads from shared memory)
      socketService.connect();

      // Subscribe to camera frames
      const handleCameraFrame = (data: any) => {
        if (data.serial_number === serialNumber) {
          const frameDataUrl = `data:image/jpeg;base64,${data.frame_base64}`;
          setFrameUrl(frameDataUrl);
          setLastCapture(new Date());
        }
      };

      socketService.subscribeToCameraFrames(handleCameraFrame);
      socketService.startCameraStream(serialNumber, 10); // 10 FPS

      return () => {
        socketService.stopCameraStream(serialNumber);
        socketService.unsubscribeFromCameraFrames(handleCameraFrame);
      };
    } else if (settings.autoRefresh) {
      // Use API polling for auto refresh mode
      const interval = setInterval(() => {
        captureFrame(true);
      }, settings.refreshInterval);
      intervalRef.current = interval;

      return () => {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
      };
    }
  }, [isLiveMode, settings.autoRefresh, settings.refreshInterval, settings.quality, serialNumber]);

  const captureFrame = async (silent: boolean = false) => {
    try {
      const url = camerasAPI.getCameraFrame(serialNumber, settings.quality);
      // Add timestamp to prevent caching
      setFrameUrl(`${url}&t=${Date.now()}`);
      setLastCapture(new Date());

      if (!silent) {
        toast.success('Frame captured successfully!');
      }
    } catch (err) {
      console.error('Error capturing frame:', err);
      if (!silent) {
        toast.error('Failed to capture frame');
      }
    }
  };

  const toggleLiveMode = () => {
    setIsLiveMode(prev => !prev);
    if (!isLiveMode) {
      toast.info('Live streaming started');
    } else {
      toast.info('Live streaming stopped');
    }
  };

  const handleSettingChange = (key: keyof CameraSettings, value: number | boolean) => {
    setSettings(prev => ({
      ...prev,
      [key]: value
    }));

    // Mark as unsaved if exposure or gain changed
    if (key === 'exposure' || key === 'gain') {
      setHasUnsavedSettings(true);
    }
  };

  const applySettings = async () => {
    try {
      setIsApplyingSettings(true);
      await camerasAPI.updateCameraSettings(serialNumber, {
        exposure_time: settings.exposure,
        gain: settings.gain
      });
      setHasUnsavedSettings(false);
      toast.success('Camera settings applied successfully!');
    } catch (err) {
      console.error('Error applying camera settings:', err);
      toast.error('Failed to apply camera settings');
    } finally {
      setIsApplyingSettings(false);
    }
  };

  const downloadFrame = () => {
    if (!frameUrl) {
      toast.error('No frame to download');
      return;
    }

    const link = document.createElement('a');
    link.href = frameUrl;
    link.download = `camera_${serialNumber}_${Date.now()}.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    toast.success('Frame downloaded!');
  };

  return (
    <div className="camera-viewer-page">
      {/* Header */}
      <div className="section-header">
        <button className="back-button" onClick={onBack}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M19 12H5M12 19l-7-7 7-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Back to List
        </button>
        <div className="camera-header-info">
          <h2><span className="camera-details">{(camera as any).model_name || camera.name} • Serial: {(camera as any).serial_number || camera.camera_id}</span> </h2>
          
        </div>
      </div>

      <div className="camera-viewer-content">
          {/* Left side - Camera feed */}
          <div className="camera-feed-section">
            <div className="camera-feed">
              {frameUrl ? (
                <img
                  ref={imgRef}
                  src={frameUrl}
                  alt="Camera feed"
                  className="camera-feed-image"
                />
              ) : (
                <div className="camera-feed-placeholder">
                  <svg width="64" height="64" viewBox="0 0 24 24" fill="none">
                    <rect x="2" y="4" width="20" height="16" rx="2" stroke="currentColor" strokeWidth="2"/>
                    <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2"/>
                  </svg>
                  <p>Click "Capture" or "Start Live" to view camera feed</p>
                </div>
              )}
            </div>

            {/* Control buttons */}
            <div className="camera-controls">
              <button
                className={`control-btn ${isLiveMode ? 'active' : ''}`}
                onClick={toggleLiveMode}
              >
                {isLiveMode ? (
                  <>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                      <rect x="6" y="4" width="4" height="16" fill="currentColor"/>
                      <rect x="14" y="4" width="4" height="16" fill="currentColor"/>
                    </svg>
                    Stop Live
                  </>
                ) : (
                  <>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                      <polygon points="5 3 19 12 5 21 5 3" fill="currentColor"/>
                    </svg>
                    Start Live
                  </>
                )}
              </button>

              <button
                className="control-btn"
                onClick={() => captureFrame(false)}
                disabled={isLiveMode}
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
                  <circle cx="12" cy="12" r="4" fill="currentColor"/>
                </svg>
                Capture Frame
              </button>

              <button
                className="control-btn"
                onClick={downloadFrame}
                disabled={!frameUrl}
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                </svg>
                Download
              </button>
            </div>

            {lastCapture && (
              <div className="camera-info">
                <span>Last capture: {lastCapture.toLocaleTimeString()}</span>
                <span className={`status-indicator ${isLiveMode ? 'live' : ''}`}>
                  {isLiveMode ? 'LIVE' : 'IDLE'}
                </span>
              </div>
            )}
          </div>

          {/* Right side - Settings */}
          <div className="camera-settings-section">
            <h4>Camera Settings</h4>

            <div className="setting-group">
              <label>
                <span>Exposure Time (μs)</span>
                <input
                  type="number"
                  min="0"
                  max="100000"
                  step="100"
                  value={settings.exposure}
                  onChange={(e) => handleSettingChange('exposure', Number(e.target.value))}
                  className="setting-value-input"
                />
              </label>
              <input
                type="range"
                min="0"
                max="100000"
                step="100"
                value={settings.exposure}
                onChange={(e) => handleSettingChange('exposure', Number(e.target.value))}
              />
            </div>

            <div className="setting-group">
              <label>
                <span>Gain</span>
                <div className="setting-value">{settings.gain.toFixed(1)}</div>
              </label>
              <input
                type="range"
                min="1"
                max="10"
                step="0.1"
                value={settings.gain}
                onChange={(e) => handleSettingChange('gain', Number(e.target.value))}
              />
            </div>

            {hasUnsavedSettings && (
              <div className="setting-group">
                <button
                  className="apply-settings-btn"
                  onClick={applySettings}
                  disabled={isApplyingSettings}
                >
                  {isApplyingSettings ? (
                    <>
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="spin">
                        <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" opacity="0.25"/>
                        <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="2"/>
                      </svg>
                      Applying...
                    </>
                  ) : (
                    <>
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                        <path d="M20 6L9 17l-5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                      Apply Settings
                    </>
                  )}
                </button>
              </div>
            )}

            <div className="setting-group">
              <label>
                <span>JPEG Quality (%)</span>
                <div className="setting-value">{settings.quality}</div>
              </label>
              <input
                type="range"
                min="50"
                max="100"
                step="5"
                value={settings.quality}
                onChange={(e) => handleSettingChange('quality', Number(e.target.value))}
              />
            </div>

            <div className="setting-group">
              <label>
                <span>Refresh Interval (ms)</span>
                <div className="setting-value">{settings.refreshInterval}</div>
              </label>
              <input
                type="range"
                min="33"
                max="1000"
                step="33"
                value={settings.refreshInterval}
                onChange={(e) => handleSettingChange('refreshInterval', Number(e.target.value))}
              />
            </div>

            <div className="setting-group checkbox">
              <label>
                <input
                  type="checkbox"
                  checked={settings.autoRefresh}
                  onChange={(e) => handleSettingChange('autoRefresh', e.target.checked)}
                />
                <span>Auto Refresh</span>
              </label>
            </div>

            <div className="camera-metadata">
              <h5>Camera Info</h5>
              <div className="metadata-item">
                <span>Serial Number:</span>
                <code>{serialNumber}</code>
              </div>
              <div className="metadata-item">
                <span>Resolution:</span>
                <span>{(camera as any).resolution_width || 1920} × {(camera as any).resolution_height || 1200}</span>
              </div>
              <div className="metadata-item">
                <span>Max FPS:</span>
                <span>{(camera as any).max_frame_rate || 30}</span>
              </div>
              <div className="metadata-item">
                <span>IP Address:</span>
                <code>{camera.ip_address || 'N/A'}</code>
              </div>
            </div>
          </div>
        </div>
      </div>
  );
};

export default CameraViewer;
