import React, { useEffect, useState, useCallback } from 'react';
import { socketService } from '../../services/socketio';
import './ServiceDownOverlay.css';

type ServiceState = 'ok' | 'backend_down' | 'camera_down';

export const ServiceDownOverlay: React.FC = () => {
  const [state, setState] = useState<ServiceState>('ok');

  const handleConnect = useCallback(() => {
    // Backend socket reconnected — camera service state will be re-announced via event
    setState(prev => prev === 'backend_down' ? 'ok' : prev);
  }, []);

  const handleDisconnect = useCallback((reason: string) => {
    // 'io client disconnect' = we disconnected on purpose (logout / teardown).
    if (reason === 'io client disconnect') return;
    setState('backend_down');
  }, []);

  const handleCameraServiceStatus = useCallback((data: any) => {
    if (data?.connected === false) {
      setState('camera_down');
    } else if (data?.connected === true) {
      setState('ok');
    }
  }, []);

  useEffect(() => {
    socketService.connect();
    socketService.onSocketConnect(handleConnect);
    socketService.onSocketDisconnect(handleDisconnect);
    socketService.onCameraServiceStatus(handleCameraServiceStatus);

    // Don't flash "backend down" on every page load — the socket connects
    // asynchronously and is NOT connected for the first ~1s. Give it a grace
    // window; only declare the backend down if it still hasn't connected.
    let graceTimer: ReturnType<typeof setTimeout> | null = null;
    if (!socketService.isConnected()) {
      graceTimer = setTimeout(() => {
        if (!socketService.isConnected()) {
          setState(prev => prev === 'ok' ? 'backend_down' : prev);
        }
      }, 4000);
    }

    return () => {
      if (graceTimer) clearTimeout(graceTimer);
      socketService.offSocketConnect(handleConnect);
      socketService.offSocketDisconnect(handleDisconnect);
      socketService.offCameraServiceStatus(handleCameraServiceStatus);
    };
  }, [handleConnect, handleDisconnect, handleCameraServiceStatus]);

  if (state === 'ok') return null;

  const isBackendDown = state === 'backend_down';

  return (
    <div className="sdo-overlay">
      <div className="sdo-card">
        <div className="sdo-spinner-wrap">
          <div className="sdo-spinner" />
        </div>

        <div className="sdo-label">
          {isBackendDown ? '⚠ Backend Service' : '⚠ Camera Service'}
        </div>
        <div className="sdo-title">
          {isBackendDown ? 'Mất kết nối' : 'Đang khởi động lại'}
        </div>
        <div className="sdo-message">
          {isBackendDown
            ? 'Backend không phản hồi. Đang thử kết nối lại...'
            : 'Camera management service bị gián đoạn. Hệ thống đang tự khởi động lại, vui lòng chờ...'
          }
        </div>

        <div className="sdo-dots">
          {[0, 1, 2].map(i => (
            <div key={i} className="sdo-dot" style={{ animationDelay: `${i * 0.2}s` }} />
          ))}
        </div>
      </div>
    </div>
  );
};
