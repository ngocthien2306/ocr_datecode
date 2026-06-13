import React, { useState } from 'react';
import Dashboard from './components/dashboard/Dashboard';
import Login from './components/login/Login';
import { ToastProvider } from './contexts/ToastContext';
import { UserProvider } from './contexts/UserContext';
import { ServiceStatusWatcher } from './components/shared/ServiceStatusWatcher';
import './styles/App.css';

export default function SuntechAutomation() {
  const [isLightMode, setIsLightMode] = useState(() => {
    const savedMode = localStorage.getItem('appThemeMode');
    return savedMode ? savedMode === 'light' : true;
  });
  const [isLoggedIn, setIsLoggedIn] = useState(() => {
    // Check if user is already logged in
    return !!localStorage.getItem('access_token');
  });
  const [showLoginLoading, setShowLoginLoading] = useState(false);

  const handleLoginSuccess = () => {
    // Show login loading screen for 2 seconds
    setShowLoginLoading(true);

    setTimeout(() => {
      setShowLoginLoading(false);
      setIsLoggedIn(true);
    }, 2000);
  };

  const handleLogout = () => {
    console.log('Logging out...');

    // Clear auth data
    localStorage.removeItem('access_token');
    localStorage.removeItem('user');

    // Reload theme from localStorage when logging out
    const savedMode = localStorage.getItem('appThemeMode');
    setIsLightMode(savedMode ? savedMode === 'light' : true);

    setIsLoggedIn(false);
  };

  // Show Dashboard if logged in
  if (isLoggedIn) {
    return (
      <ToastProvider>
        <UserProvider>
          <Dashboard onLogout={handleLogout} />
          <ServiceStatusWatcher />
        </UserProvider>
      </ToastProvider>
    );
  }

  // Show login loading screen
  if (showLoginLoading) {
    return (
      <div className={`login-loading-overlay ${isLightMode ? 'light-mode' : ''}`}>
        <div className="login-loading-content">
          <div className="system-boot-container">
            <div className="boot-circle-container">
              <div className="boot-outer-ring"></div>
              <div className="boot-middle-ring"></div>
              <div className="boot-inner-ring"></div>
              <div className="boot-center-icon"></div>
            </div>

            <div className="boot-system-info">
              <div className="boot-info-line">
                <span>Initializing Camera Vision System</span>
                <span className="boot-info-check">✓</span>
              </div>
              <div className="boot-info-line">
                <span>Loading OCR Engine Modules</span>
                <span className="boot-info-check">✓</span>
              </div>
              <div className="boot-info-line">
                <span>Connecting to Database</span>
                <span className="boot-info-check">✓</span>
              </div>
              <div className="boot-info-line">
                <span>Authenticating User Session</span>
                <span className="boot-info-check">✓</span>
              </div>
              <div className="boot-info-line">
                <span>Preparing Dashboard Interface</span>
                <span className="boot-info-check">✓</span>
              </div>
            </div>

            <div className="boot-progress-container">
              <div className="boot-progress-bar"></div>
            </div>

            <div className="boot-welcome-text">
              Welcome to System
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Render login page
  return (
    <ToastProvider>
      <Login onLoginSuccess={handleLoginSuccess} isLightMode={isLightMode} />
    </ToastProvider>
  );
}
