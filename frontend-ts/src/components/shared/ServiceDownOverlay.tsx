import React, { useEffect, useState, useCallback } from 'react';
import { socketService } from '../../services/socketio';

type ServiceState = 'ok' | 'backend_down' | 'camera_down';

export const ServiceDownOverlay: React.FC = () => {
  const [state, setState] = useState<ServiceState>('ok');

  const handleConnect = useCallback(() => {
    // Backend socket reconnected — camera service state will be re-announced via event
    setState(prev => prev === 'backend_down' ? 'ok' : prev);
  }, []);

  const handleDisconnect = useCallback((_reason: string) => {
    setState('backend_down');
  }, []);

  const handleCameraServiceStatus = useCallback((data: any) => {
    if (data.connected === false) {
      setState('camera_down');
    } else if (data.connected === true) {
      setState('ok');
    }
  }, []);

  useEffect(() => {
    socketService.connect();
    socketService.onSocketConnect(handleConnect);
    socketService.onSocketDisconnect(handleDisconnect);
    socketService.onCameraServiceStatus(handleCameraServiceStatus);

    // Sync initial socket state
    if (!socketService.isConnected()) {
      setState('backend_down');
    }

    return () => {
      socketService.offSocketConnect(handleConnect);
      socketService.offSocketDisconnect(handleDisconnect);
      socketService.offCameraServiceStatus(handleCameraServiceStatus);
    };
  }, [handleConnect, handleDisconnect, handleCameraServiceStatus]);

  if (state === 'ok') return null;

  const isBackendDown = state === 'backend_down';

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 9999,
      background: 'rgba(10, 14, 26, 0.88)',
      backdropFilter: 'blur(6px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
    }}>
      <div style={{
        background: 'linear-gradient(135deg, #0f1629 0%, #1a2040 100%)',
        border: '1px solid rgba(99,179,237,0.25)',
        borderRadius: 16,
        padding: '40px 52px',
        textAlign: 'center',
        maxWidth: 420,
        boxShadow: '0 0 60px rgba(66,153,225,0.15)',
      }}>
        {/* Spinner */}
        <div style={{ marginBottom: 24, display: 'flex', justifyContent: 'center' }}>
          <div style={{
            width: 56, height: 56,
            border: '3px solid rgba(99,179,237,0.2)',
            borderTopColor: '#63b3ed',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite',
          }} />
        </div>

        {/* Icon + Title */}
        <div style={{ fontSize: 13, color: '#63b3ed', letterSpacing: 2, textTransform: 'uppercase', marginBottom: 8 }}>
          {isBackendDown ? '⚠ Backend Service' : '⚠ Camera Service'}
        </div>
        <div style={{ fontSize: 22, fontWeight: 700, color: '#e2e8f0', marginBottom: 12 }}>
          {isBackendDown ? 'Mất kết nối' : 'Đang khởi động lại'}
        </div>
        <div style={{ fontSize: 14, color: '#94a3b8', lineHeight: 1.6 }}>
          {isBackendDown
            ? 'Backend không phản hồi. Đang thử kết nối lại...'
            : 'Camera management service bị gián đoạn. Hệ thống đang tự khởi động lại, vui lòng chờ...'
          }
        </div>

        {/* Pulse dots */}
        <div style={{ marginTop: 28, display: 'flex', justifyContent: 'center', gap: 8 }}>
          {[0, 1, 2].map(i => (
            <div key={i} style={{
              width: 8, height: 8, borderRadius: '50%',
              background: '#63b3ed',
              animation: `pulse 1.4s ease-in-out ${i * 0.2}s infinite`,
              opacity: 0.4,
            }} />
          ))}
        </div>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse {
          0%, 80%, 100% { opacity: 0.4; transform: scale(1); }
          40% { opacity: 1; transform: scale(1.3); }
        }
      `}</style>
    </div>
  );
};
