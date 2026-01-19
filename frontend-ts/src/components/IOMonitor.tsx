import React, { useEffect, useState } from 'react';
import socketService from '@/services/socketio';
import { useToast } from '@/contexts/ToastContext';

interface IOStatus {
  di: number[];
  do: number[];
  timestamp: number;
}

const IOMonitor: React.FC = () => {
  const toast = useToast();
  const [ioStatus, setIoStatus] = useState<IOStatus>({
    di: [0, 0, 0, 0],
    do: [0, 0, 0, 0],
    timestamp: 0
  });
  const [isTracking, setIsTracking] = useState(false);
  const [isUpdating, setIsUpdating] = useState<number | null>(null);

  useEffect(() => {
    const handleIOStatus = (data: IOStatus) => {
      setIoStatus(data);
    };

    socketService.onIOStatusUpdate(handleIOStatus);

    return () => {
      socketService.offIOStatusUpdate(handleIOStatus);
      if (isTracking) {
        socketService.stopIOTracking();
      }
    };
  }, [isTracking]);

  const handleStartTracking = () => {
    socketService.requestIOStatus();
    setIsTracking(true);
    toast.success('I/O tracking started');
  };

  const handleStopTracking = () => {
    socketService.stopIOTracking();
    setIsTracking(false);
    toast.info('I/O tracking stopped');
  };

  const handleDOToggle = async (pinIndex: number, newValue: boolean) => {
    setIsUpdating(pinIndex);

    try {
      socketService.setDOPin(pinIndex, newValue ? 1 : 0);

      setIoStatus(prev => ({
        ...prev,
        do: prev.do.map((val, idx) => idx === pinIndex ? (newValue ? 1 : 0) : val)
      }));

      toast.success(`DO${pinIndex} set to ${newValue ? 'HIGH' : 'LOW'}`);
    } catch (error: any) {
      toast.error(`Failed to set DO${pinIndex}: ${error.message}`);
    } finally {
      setIsUpdating(null);
    }
  };

  return (
    <div className="io-monitor">
      {/* Header */}
      <div className="io-header">
        <div className="io-title-section">
          <h2>I/O Monitor</h2>
          <p>Real-time monitoring of Digital Input/Output pins</p>
        </div>
        <div>
          {!isTracking ? (
            <button onClick={handleStartTracking} className="tracking-btn start">
              Start Tracking
            </button>
          ) : (
            <button onClick={handleStopTracking} className="tracking-btn stop">
              Stop Tracking
            </button>
          )}
        </div>
      </div>

      <div className="tracking-status">
        <div className={`status-dot ${isTracking ? 'active' : ''}`} />
        <span className="status-text">
          {isTracking ? 'Tracking Active' : 'Tracking Stopped'}
        </span>
        {ioStatus.timestamp > 0 && (
          <span className="last-update">
            Last update: {new Date(ioStatus.timestamp * 1000).toLocaleTimeString()}
          </span>
        )}
      </div>

      {/* I/O Cards */}
      <div className="io-cards-grid">
        {/* Digital Inputs Card */}
        <div className="io-card">
          <div className="io-card-header">
            <h2>Digital Inputs (DI)</h2>
            <span className="io-badge readonly">Read Only</span>
          </div>

          <div className="pin-list">
            {[0, 1, 2, 3].map((pinIndex) => (
              <div key={`di-${pinIndex}`} className="pin-item">
                <div className="pin-info">
                  <div className={`pin-indicator ${ioStatus.di[pinIndex] === 1 ? 'active-di' : ''}`} />
                  <div>
                    <div className="pin-label">DI{pinIndex}</div>
                    <div className="pin-description">Digital Input {pinIndex}</div>
                  </div>
                </div>
                <div className="pin-value">
                  <div className={`pin-value-number ${ioStatus.di[pinIndex] === 1 ? 'active-di' : ''}`}>
                    {ioStatus.di[pinIndex]}
                  </div>
                  <div className="pin-value-state">
                    {ioStatus.di[pinIndex] === 1 ? 'HIGH' : 'LOW'}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Digital Outputs Card */}
        <div className="io-card">
          <div className="io-card-header">
            <h2>Digital Outputs (DO)</h2>
            <span className="io-badge controllable">Controllable</span>
          </div>

          <div className="pin-list">
            {[0, 1, 2, 3].map((pinIndex) => (
              <div key={`do-${pinIndex}`} className="pin-item">
                <div className="pin-info">
                  <div className={`pin-indicator ${ioStatus.do[pinIndex] === 1 ? 'active-do' : ''}`} />
                  <div>
                    <div className="pin-label">DO{pinIndex}</div>
                    <div className="pin-description">Digital Output {pinIndex}</div>
                  </div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                  <div className="pin-value">
                    <div className={`pin-value-number ${ioStatus.do[pinIndex] === 1 ? 'active-do' : ''}`}>
                      {ioStatus.do[pinIndex]}
                    </div>
                    <div className="pin-value-state">
                      {ioStatus.do[pinIndex] === 1 ? 'HIGH' : 'LOW'}
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    {isUpdating === pinIndex && (
                      <div className="spinner" />
                    )}
                    <label className="toggle-switch">
                      <input
                        type="checkbox"
                        checked={ioStatus.do[pinIndex] === 1}
                        onChange={(e) => handleDOToggle(pinIndex, e.target.checked)}
                        disabled={isUpdating === pinIndex}
                      />
                      <span className="toggle-slider"></span>
                    </label>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Info Card */}
      <div className="info-card">
        <h2>Information</h2>
        <div className="info-grid">
          <div className="info-section">
            <h3>Digital Inputs (DI)</h3>
            <ul>
              <li>DI0-DI3: Hardware trigger pins for cameras</li>
              <li>Read-only status monitoring</li>
              <li>Updates in real-time (10Hz polling)</li>
            </ul>
          </div>
          <div className="info-section">
            <h3>Digital Outputs (DO)</h3>
            <ul>
              <li>DO0-DO3: Controllable output pins</li>
              <li>Toggle HIGH (1) / LOW (0) via switch</li>
              <li>Can be used for external device control</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default IOMonitor;
