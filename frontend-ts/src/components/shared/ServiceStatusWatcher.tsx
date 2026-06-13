import React, { useEffect } from 'react';
import { socketService } from '../../services/socketio';
import { useToast } from '../../contexts/ToastContext';

/**
 * Headless watcher: listens to backend socket + camera-service status and shows
 * a TOAST (not a blocking overlay) on state transitions.
 *
 * Edge-triggered via refs so a burst of identical events (the BE re-emits
 * camera_service_status{false} on every failed API call) only produces ONE toast
 * per outage, and exactly one "recovered" toast when it comes back.
 *
 * Renders nothing.
 */
export const ServiceStatusWatcher: React.FC = () => {
  const { success, error, warning } = useToast();

  useEffect(() => {
    // Plain module-scoped flags via closure refs — kept across renders because
    // the effect only runs once (stable toast fns).
    let backendDown = false;
    let cameraDown = false;

    socketService.connect();

    const onConnect = () => {
      if (backendDown) {
        backendDown = false;
        success('Đã kết nối lại với backend');
      }
    };

    const onDisconnect = (reason: string) => {
      // 'io client disconnect' = we disconnected on purpose (logout / teardown).
      // Real load (F5) never fires 'disconnect' before connecting, so no false toast.
      if (reason === 'io client disconnect') return;
      if (!backendDown) {
        backendDown = true;
        error('Mất kết nối backend — đang thử kết nối lại...', 5000);
      }
    };

    const onCameraStatus = (data: any) => {
      if (data?.connected === false) {
        if (!cameraDown) {
          cameraDown = true;
          warning('Camera management mất kết nối — đang khởi động lại...', 6000);
        }
      } else if (data?.connected === true) {
        if (cameraDown) {
          cameraDown = false;
          success('Camera management đã kết nối lại');
        }
      }
    };

    socketService.onSocketConnect(onConnect);
    socketService.onSocketDisconnect(onDisconnect);
    socketService.onCameraServiceStatus(onCameraStatus);

    return () => {
      socketService.offSocketConnect(onConnect);
      socketService.offSocketDisconnect(onDisconnect);
      socketService.offCameraServiceStatus(onCameraStatus);
    };
  }, [success, error, warning]);

  return null;
};
