import React, { useEffect, useState } from 'react';
import socketService from '@/services/socketio';
import { useToast } from '@/contexts/ToastContext';

interface IOStatus {
  di: number[];  // Digital Input status (0-3)
  do: number[];  // Digital Output status (0-3)
  timestamp: number;
}

const System: React.FC = () => {
  const toast = useToast();
  const [ioStatus, setIoStatus] = useState<IOStatus>({
    di: [0, 0, 0, 0],
    do: [0, 0, 0, 0],
    timestamp: 0
  });
  const [isTracking, setIsTracking] = useState(false);
  const [isUpdating, setIsUpdating] = useState<number | null>(null);

  useEffect(() => {
    // Subscribe to I/O status updates
    const handleIOStatus = (data: IOStatus) => {
      setIoStatus(data);
    };

    socketService.onIOStatusUpdate(handleIOStatus);

    // Cleanup
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
      // Send DO control command via SocketIO
      socketService.setDOPin(pinIndex, newValue ? 1 : 0);

      // Optimistically update UI
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
    <div style={{ padding: '24px', maxWidth: '1200px', margin: '0 auto' }}>
      {/* Header */}
      <div style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
          <div>
            <h1 style={{ fontSize: '30px', fontWeight: 'bold', margin: '0 0 8px 0' }}>System I/O Monitor</h1>
            <p style={{ color: '#6b7280', margin: 0 }}>Real-time monitoring of Digital Input/Output pins</p>
          </div>
          <div>
            {!isTracking ? (
              <button
                onClick={handleStartTracking}
                style={{
                  padding: '10px 20px',
                  backgroundColor: '#16a34a',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontWeight: '500',
                  fontSize: '14px'
                }}
                onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#15803d'}
                onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#16a34a'}
              >
                Start Tracking
              </button>
            ) : (
              <button
                onClick={handleStopTracking}
                style={{
                  padding: '10px 20px',
                  backgroundColor: '#dc2626',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontWeight: '500',
                  fontSize: '14px'
                }}
                onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#b91c1c'}
                onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#dc2626'}
              >
                Stop Tracking
              </button>
            )}
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div
            style={{
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              backgroundColor: isTracking ? '#22c55e' : '#9ca3af',
              animation: isTracking ? 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite' : 'none'
            }}
          />
          <span style={{ fontSize: '14px', color: '#6b7280' }}>
            {isTracking ? 'Tracking Active' : 'Tracking Stopped'}
          </span>
          {ioStatus.timestamp > 0 && (
            <span style={{ fontSize: '12px', color: '#9ca3af', marginLeft: '16px' }}>
              Last update: {new Date(ioStatus.timestamp * 1000).toLocaleTimeString()}
            </span>
          )}
        </div>
      </div>

      {/* I/O Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '24px' }}>
        {/* Digital Inputs Card */}
        <div style={{
          border: '1px solid #e5e7eb',
          borderRadius: '8px',
          padding: '24px',
          backgroundColor: 'white'
        }}>
          <div style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <h2 style={{ fontSize: '18px', fontWeight: '600', margin: 0 }}>Digital Inputs (DI)</h2>
            <span style={{
              fontSize: '12px',
              padding: '2px 8px',
              backgroundColor: '#f3f4f6',
              color: '#6b7280',
              borderRadius: '4px'
            }}>
              Read Only
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {[0, 1, 2, 3].map((pinIndex) => (
              <div
                key={`di-${pinIndex}`}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '16px',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <div
                    style={{
                      width: '16px',
                      height: '16px',
                      borderRadius: '50%',
                      backgroundColor: ioStatus.di[pinIndex] === 1 ? '#22c55e' : '#d1d5db'
                    }}
                  />
                  <div>
                    <div style={{ fontWeight: '500' }}>DI{pinIndex}</div>
                    <div style={{ fontSize: '14px', color: '#6b7280' }}>Digital Input {pinIndex}</div>
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{
                    fontSize: '28px',
                    fontWeight: 'bold',
                    color: ioStatus.di[pinIndex] === 1 ? '#16a34a' : '#9ca3af'
                  }}>
                    {ioStatus.di[pinIndex]}
                  </div>
                  <div style={{ fontSize: '12px', color: '#6b7280' }}>
                    {ioStatus.di[pinIndex] === 1 ? 'HIGH' : 'LOW'}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Digital Outputs Card */}
        <div style={{
          border: '1px solid #e5e7eb',
          borderRadius: '8px',
          padding: '24px',
          backgroundColor: 'white'
        }}>
          <div style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <h2 style={{ fontSize: '18px', fontWeight: '600', margin: 0 }}>Digital Outputs (DO)</h2>
            <span style={{
              fontSize: '12px',
              padding: '2px 8px',
              backgroundColor: '#3b82f6',
              color: 'white',
              borderRadius: '4px'
            }}>
              Controllable
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {[0, 1, 2, 3].map((pinIndex) => (
              <div
                key={`do-${pinIndex}`}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '16px',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <div
                    style={{
                      width: '16px',
                      height: '16px',
                      borderRadius: '50%',
                      backgroundColor: ioStatus.do[pinIndex] === 1 ? '#3b82f6' : '#d1d5db'
                    }}
                  />
                  <div>
                    <div style={{ fontWeight: '500' }}>DO{pinIndex}</div>
                    <div style={{ fontSize: '14px', color: '#6b7280' }}>Digital Output {pinIndex}</div>
                  </div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{
                      fontSize: '28px',
                      fontWeight: 'bold',
                      color: ioStatus.do[pinIndex] === 1 ? '#2563eb' : '#9ca3af'
                    }}>
                      {ioStatus.do[pinIndex]}
                    </div>
                    <div style={{ fontSize: '12px', color: '#6b7280' }}>
                      {ioStatus.do[pinIndex] === 1 ? 'HIGH' : 'LOW'}
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    {isUpdating === pinIndex && (
                      <div style={{
                        width: '16px',
                        height: '16px',
                        border: '2px solid #9ca3af',
                        borderTopColor: 'transparent',
                        borderRadius: '50%',
                        animation: 'spin 1s linear infinite'
                      }} />
                    )}
                    <label style={{ position: 'relative', display: 'inline-block', width: '48px', height: '24px' }}>
                      <input
                        type="checkbox"
                        checked={ioStatus.do[pinIndex] === 1}
                        onChange={(e) => handleDOToggle(pinIndex, e.target.checked)}
                        disabled={isUpdating === pinIndex}
                        style={{ opacity: 0, width: 0, height: 0 }}
                      />
                      <span
                        style={{
                          position: 'absolute',
                          cursor: isUpdating === pinIndex ? 'not-allowed' : 'pointer',
                          top: 0,
                          left: 0,
                          right: 0,
                          bottom: 0,
                          backgroundColor: ioStatus.do[pinIndex] === 1 ? '#3b82f6' : '#cbd5e1',
                          transition: '0.4s',
                          borderRadius: '24px'
                        }}
                      >
                        <span
                          style={{
                            position: 'absolute',
                            content: '',
                            height: '18px',
                            width: '18px',
                            left: ioStatus.do[pinIndex] === 1 ? '26px' : '3px',
                            bottom: '3px',
                            backgroundColor: 'white',
                            transition: '0.4s',
                            borderRadius: '50%'
                          }}
                        />
                      </span>
                    </label>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Info Card */}
      <div style={{
        border: '1px solid #e5e7eb',
        borderRadius: '8px',
        padding: '24px',
        backgroundColor: 'white'
      }}>
        <h2 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '16px' }}>Information</h2>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', fontSize: '14px' }}>
          <div>
            <h3 style={{ fontWeight: '600', marginBottom: '8px' }}>Digital Inputs (DI)</h3>
            <ul style={{ margin: 0, paddingLeft: '20px', color: '#6b7280', lineHeight: '1.8' }}>
              <li>DI0-DI3: Hardware trigger pins for cameras</li>
              <li>Read-only status monitoring</li>
              <li>Updates in real-time (10Hz polling)</li>
            </ul>
          </div>
          <div>
            <h3 style={{ fontWeight: '600', marginBottom: '8px' }}>Digital Outputs (DO)</h3>
            <ul style={{ margin: 0, paddingLeft: '20px', color: '#6b7280', lineHeight: '1.8' }}>
              <li>DO0-DO3: Controllable output pins</li>
              <li>Toggle HIGH (1) / LOW (0) via switch</li>
              <li>Can be used for external device control</li>
            </ul>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default System;
