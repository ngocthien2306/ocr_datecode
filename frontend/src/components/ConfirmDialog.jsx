import React, { useEffect, useState } from 'react';
import '../styles/ConfirmDialog.css';

export default function ConfirmDialog({ 
  isOpen, 
  onClose, 
  onConfirm, 
  title = 'Confirm Action',
  message,
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  type = 'warning' // 'warning', 'danger', 'info'
}) {
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    // Check initial dark mode state
    setIsDarkMode(document.body.classList.contains('dark-mode'));

    // Listen for changes in dark mode
    const observer = new MutationObserver(() => {
      setIsDarkMode(document.body.classList.contains('dark-mode'));
    });

    observer.observe(document.body, {
      attributes: true,
      attributeFilter: ['class']
    });

    return () => observer.disconnect();
  }, []);

  if (!isOpen) return null;

  const handleConfirm = () => {
    onConfirm();
    onClose();
  };

  const handleCancel = () => {
    onClose();
  };

  const getIcon = () => {
    switch (type) {
      case 'danger':
        return (
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
            <path d="M12 8V12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            <circle cx="12" cy="16" r="0.5" fill="currentColor" stroke="currentColor"/>
          </svg>
        );
      case 'info':
        return (
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
            <path d="M12 16V12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            <circle cx="12" cy="8" r="0.5" fill="currentColor" stroke="currentColor"/>
          </svg>
        );
      default: // warning
        return (
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <line x1="12" y1="9" x2="12" y2="13" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            <circle cx="12" cy="17" r="0.5" fill="currentColor" stroke="currentColor"/>
          </svg>
        );
    }
  };

  return (
    <div className="confirm-dialog-overlay" onClick={handleCancel}>
      <div className={`confirm-dialog-container ${isDarkMode ? 'dark' : 'light'}`} onClick={(e) => e.stopPropagation()}>
        <div className={`confirm-dialog-icon ${type}`}>
          {getIcon()}
        </div>
        
        <div className="confirm-dialog-content">
          <h2 className="confirm-dialog-title">{title}</h2>
          <p className="confirm-dialog-message">{message}</p>
        </div>

        <div className="confirm-dialog-actions">
          <button 
            className="confirm-dialog-btn cancel" 
            onClick={handleCancel}
          >
            {cancelText}
          </button>
          <button 
            className={`confirm-dialog-btn confirm ${type}`}
            onClick={handleConfirm}
          >
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  );
}
