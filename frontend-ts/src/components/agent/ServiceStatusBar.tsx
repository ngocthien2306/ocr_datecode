/**
 * Service Status Bar Component
 * Shows real-time status of camera service
 */

import React, { useEffect, useState } from 'react';
import { ServiceStatus } from '../../types/agent';
import { agentService } from '../../services/agentService';
import { RefreshIcon, LoadingIcon } from './AgentIcons';

interface ServiceStatusBarProps {
  onRefresh?: () => void;
}

export const ServiceStatusBar: React.FC<ServiceStatusBarProps> = ({ onRefresh }) => {
  const [status, setStatus] = useState<ServiceStatus>({
    service_name: 'camera_management',
    is_running: false,
    websocket_connected: false,
    status: 'stopped',
    message: 'Checking...'
  });
  const [isRefreshing, setIsRefreshing] = useState(false);

  const checkStatus = async () => {
    setIsRefreshing(true);
    try {
      const result = await agentService.checkServiceStatus('camera_management');
      setStatus(result);
    } catch (error) {
      console.error('Failed to check service status:', error);
      setStatus({
        service_name: 'camera_management',
        is_running: false,
        websocket_connected: false,
        status: 'stopped',
        message: 'Failed to check status'
      });
    } finally {
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    // Initial check
    checkStatus();

    // Auto-refresh every 10 seconds
    const interval = setInterval(checkStatus, 10000);

    return () => clearInterval(interval);
  }, []);

  const handleRefresh = () => {
    checkStatus();
    onRefresh?.();
  };

  const getStatusIcon = () => {
    switch (status.status) {
      case 'healthy':
        return '🟢';
      case 'degraded':
        return '⚠️';
      case 'stopped':
        return '❌';
      default:
        return '⚪';
    }
  };

  const getStatusText = () => {
    if (status.is_running && status.websocket_connected) {
      return 'Camera Service: Running';
    } else if (status.is_running && !status.websocket_connected) {
      return 'Service: Degraded';
    } else {
      return 'Service: Stopped';
    }
  };

  return (
    <div className="agent-status-bar">
      <div className="status-info">
        <div className={`status-dot ${status.status}`}></div>
        <span className="status-text">
          {getStatusIcon()} {getStatusText()}
        </span>
        {status.pid && (
          <span style={{ fontSize: '11px', color: '#6b7280' }}>
            (PID: {status.pid})
          </span>
        )}
      </div>

      <div className="status-actions">
        <button
          className="status-btn"
          onClick={handleRefresh}
          disabled={isRefreshing}
          title="Refresh status"
        >
          {isRefreshing ? <LoadingIcon size={16} /> : <RefreshIcon size={16} />}
        </button>
      </div>
    </div>
  );
};
