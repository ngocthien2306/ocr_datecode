import React, { useEffect } from 'react';
import ReactDOM from 'react-dom';
import '@/styles/Toast.css';

interface ToastProps {
  message: string;
  type?: 'success' | 'error' | 'warning' | 'info';
  duration?: number;
  onClose: () => void;
}

const Toast: React.FC<ToastProps> = ({ message, type = 'info', duration = 3000, onClose }) => {
  useEffect(() => {
    const timer = setTimeout(() => {
      onClose();
    }, duration);

    return () => clearTimeout(timer);
  }, [duration, onClose]);

  const icons: Record<string, string> = {
    success: '✓',
    error: '✕',
    warning: '⚠',
    info: 'ℹ'
  };

  // Use portal to render at document body level
  return ReactDOM.createPortal(
    <div className="toast-container">
      <div className={`toast toast-${type}`} onClick={onClose}>
        <div className="toast-icon">{icons[type]}</div>
        <div className="toast-message">{message}</div>
        <button className="toast-close" onClick={onClose}>×</button>
      </div>
    </div>,
    document.body
  );
};

export default Toast;
