import React, { useState, useEffect } from 'react';
import { useToast } from '../contexts/ToastContext';
import ConfirmDialog from './ConfirmDialog';

export default function Settings() {
  const toast = useToast();
  const [confirmDialog, setConfirmDialog] = useState({
    isOpen: false,
    title: '',
    message: '',
    type: 'warning',
    onConfirm: null
  });
  
  const [settings, setSettings] = useState({
    // System Settings
    systemName: 'Suntech Automation',
    version: 'v1.0.0',
    language: 'en',
    timezone: 'UTC+7',

    // Loading Templates for Each Tab
    dashboardLoading: 'camera-vision',
    usersLoading: 'users',
    receiptsLoading: 'receipts',
    camerasLoading: 'cameras',
    historicalLoading: 'historical',
    settingsLoading: 'settings',

    // Loading Background Settings
    loadingBackground: 'none',
    loadingBackgroundOpacity: 0.2,

    // Camera Settings
    camera1Enabled: true,
    camera1Fps: 30,
    camera1Resolution: '1920x1080',
    camera2Enabled: true,
    camera2Fps: 30,
    camera2Resolution: '1920x1080',
    camera3Enabled: true,
    camera3Fps: 30,
    camera3Resolution: '1920x1080',

    // Processing Settings
    detectionThreshold: 0.85,
    maxProcessingTime: 5,
    autoReject: true,
    saveFailedImages: true,

    // Notification Settings
    emailNotifications: true,
    smsNotifications: false,
    alertThreshold: 10,
    dailyReport: true,
  });

  // Load saved loading templates from localStorage
  useEffect(() => {
    const tabKeys = ['dashboard', 'users', 'receipts', 'cameras', 'historical', 'settings'];
    const loadedSettings = {};
    
    tabKeys.forEach(tab => {
      const saved = localStorage.getItem(`${tab}Loading`);
      if (saved) {
        loadedSettings[`${tab}Loading`] = saved;
      }
    });

    // Load background settings
    const savedBackground = localStorage.getItem('loadingBackground');
    const savedOpacity = localStorage.getItem('loadingBackgroundOpacity');
    if (savedBackground) {
      loadedSettings.loadingBackground = savedBackground;
    }
    if (savedOpacity) {
      loadedSettings.loadingBackgroundOpacity = parseFloat(savedOpacity);
    }
    
    if (Object.keys(loadedSettings).length > 0) {
      setSettings(prev => ({ ...prev, ...loadedSettings }));
    }
  }, []);

  const handleChange = (key, value) => {
    setSettings(prev => ({ ...prev, [key]: value }));
    
    // Save tab loading templates to localStorage when changed
    if (key.endsWith('Loading') && key !== 'loadingBackground') {
      localStorage.setItem(key, value);
      // Dispatch custom event to notify Dashboard component
      window.dispatchEvent(new CustomEvent('tabLoadingChanged', { 
        detail: { tab: key, template: value } 
      }));
    }

    // Save background settings
    if (key === 'loadingBackground' || key === 'loadingBackgroundOpacity') {
      localStorage.setItem(key, value);
      // Dispatch custom event to notify Dashboard component
      window.dispatchEvent(new CustomEvent('loadingBackgroundChanged', { 
        detail: { 
          background: key === 'loadingBackground' ? value : settings.loadingBackground, 
          opacity: key === 'loadingBackgroundOpacity' ? value : settings.loadingBackgroundOpacity 
        } 
      }));
    }
  };

  const handleSave = () => {
    console.log('Saving settings:', settings);
    toast.success('Settings saved successfully!');
  };

  const handleReset = () => {
    setConfirmDialog({
      isOpen: true,
      title: 'Reset Settings',
      message: 'Are you sure you want to reset all settings to default?\n\nThis action cannot be undone.',
      type: 'warning',
      onConfirm: () => {
        console.log('Resetting settings to default');
        toast.success('Settings reset to default values!');
      }
    });
  };

  return (
    <div className="settings-page">
      <div className="section-header">
        <h1>System Settings</h1>
        <div className="header-actions">
          <button className="dashboard-btn secondary" onClick={handleReset}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M23 4V10H17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M1 20V14H7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M3.51 9C4.01717 7.56679 4.87913 6.2854 6.01547 5.27542C7.1518 4.26543 8.52547 3.55976 10.0083 3.22426C11.4911 2.88875 13.0348 2.93434 14.4952 3.35677C15.9556 3.77921 17.2853 4.56471 18.36 5.64L23 10M1 14L5.64 18.36C6.71475 19.4353 8.04437 20.2208 9.50481 20.6432C10.9652 21.0657 12.5089 21.1112 13.9917 20.7757C15.4745 20.4402 16.8482 19.7346 17.9845 18.7246C19.1209 17.7146 19.9828 16.4332 20.49 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Reset to Default
          </button>
          <button className="dashboard-btn" onClick={handleSave}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M19 21H5C4.46957 21 3.96086 20.7893 3.58579 20.4142C3.21071 20.0391 3 19.5304 3 19V5C3 4.46957 3.21071 3.96086 3.58579 2.58579C3.96086 2.21071 4.46957 2 5 2H16L21 7V19C21 19.5304 20.7893 20.0391 20.4142 20.4142C20.0391 20.7893 19.5304 21 19 21Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <polyline points="17,21 17,13 7,13 7,21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <polyline points="7,3 7,8 15,8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Save Changes
          </button>
        </div>
      </div>

      {/* Settings Sections */}
      <div className="settings-sections">
        {/* System Settings */}
        <div className="settings-section">
          <h2>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <rect x="4" y="4" width="16" height="16" rx="2" ry="2" stroke="currentColor" strokeWidth="2"/>
              <rect x="9" y="9" width="6" height="6" stroke="currentColor" strokeWidth="2"/>
              <line x1="9" y1="1" x2="9" y2="4" stroke="currentColor" strokeWidth="2"/>
              <line x1="15" y1="1" x2="15" y2="4" stroke="currentColor" strokeWidth="2"/>
              <line x1="9" y1="20" x2="9" y2="23" stroke="currentColor" strokeWidth="2"/>
              <line x1="15" y1="20" x2="15" y2="23" stroke="currentColor" strokeWidth="2"/>
              <line x1="20" y1="9" x2="23" y2="9" stroke="currentColor" strokeWidth="2"/>
              <line x1="20" y1="15" x2="23" y2="15" stroke="currentColor" strokeWidth="2"/>
              <line x1="1" y1="9" x2="4" y2="9" stroke="currentColor" strokeWidth="2"/>
              <line x1="1" y1="15" x2="4" y2="15" stroke="currentColor" strokeWidth="2"/>
            </svg>
            System Configuration
          </h2>
          <div className="settings-grid">
            <div className="setting-item">
              <label>System Name</label>
              <input
                type="text"
                value={settings.systemName}
                onChange={(e) => handleChange('systemName', e.target.value)}
              />
            </div>
            <div className="setting-item">
              <label>Version</label>
              <input type="text" value={settings.version} disabled />
            </div>
            <div className="setting-item">
              <label>Language</label>
              <select value={settings.language} onChange={(e) => handleChange('language', e.target.value)}>
                <option value="en">English</option>
                <option value="vi">Tiếng Việt</option>
                <option value="zh">中文</option>
              </select>
            </div>
            <div className="setting-item">
              <label>Timezone</label>
              <select value={settings.timezone} onChange={(e) => handleChange('timezone', e.target.value)}>
                <option value="UTC+7">UTC+7 (Bangkok, Hanoi)</option>
                <option value="UTC+8">UTC+8 (Singapore, Beijing)</option>
                <option value="UTC+9">UTC+9 (Tokyo, Seoul)</option>
              </select>
            </div>
          </div>
        </div>

        {/* Loading Templates Settings */}
        <div className="settings-section">
          <h2>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2"/>
              <path d="M12 1v6m0 6v6M1 12h6m6 0h6" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
            Loading Animation Templates
          </h2>
          <p style={{ fontSize: '14px', color: '#6b7280', marginBottom: '20px' }}>
            Customize loading animations for each section of the application
          </p>
          <div className="settings-grid">
            <div className="setting-item">
              <label>Dashboard Loading</label>
              <select 
                value={settings.dashboardLoading} 
                onChange={(e) => handleChange('dashboardLoading', e.target.value)}
              >
                <option value="camera-vision">Camera Vision System</option>
                <option value="spinner">Classic Spinner</option>
                <option value="pulse">Pulse Wave</option>
                <option value="radar">Radar Scan</option>
                <option value="grid">Grid Matrix</option>
                <option value="circuit">Circuit Board</option>
                <option value="barcode">Barcode Scanner</option>
                <option value="ocr">OCR Text Recognition</option>
                <option value="users">Users Network</option>
                <option value="receipts">Receipt Scanner</option>
                <option value="cameras">Multi-Lens System</option>
                <option value="historical">Timeline History</option>
                <option value="settings">Gear System</option>
              </select>
            </div>

            <div className="setting-item">
              <label>User Management Loading</label>
              <select 
                value={settings.usersLoading} 
                onChange={(e) => handleChange('usersLoading', e.target.value)}
              >
                <option value="camera-vision">Camera Vision System</option>
                <option value="spinner">Classic Spinner</option>
                <option value="pulse">Pulse Wave</option>
                <option value="radar">Radar Scan</option>
                <option value="grid">Grid Matrix</option>
                <option value="circuit">Circuit Board</option>
                <option value="barcode">Barcode Scanner</option>
                <option value="ocr">OCR Text Recognition</option>
                <option value="users">Users Network</option>
                <option value="receipts">Receipt Scanner</option>
                <option value="cameras">Multi-Lens System</option>
                <option value="historical">Timeline History</option>
                <option value="settings">Gear System</option>
              </select>
            </div>

            <div className="setting-item">
              <label>Receipts Loading</label>
              <select 
                value={settings.receiptsLoading} 
                onChange={(e) => handleChange('receiptsLoading', e.target.value)}
              >
                <option value="camera-vision">Camera Vision System</option>
                <option value="spinner">Classic Spinner</option>
                <option value="pulse">Pulse Wave</option>
                <option value="radar">Radar Scan</option>
                <option value="grid">Grid Matrix</option>
                <option value="circuit">Circuit Board</option>
                <option value="barcode">Barcode Scanner</option>
                <option value="ocr">OCR Text Recognition</option>
                <option value="users">Users Network</option>
                <option value="receipts">Receipt Scanner</option>
                <option value="cameras">Multi-Lens System</option>
                <option value="historical">Timeline History</option>
                <option value="settings">Gear System</option>
              </select>
            </div>

            <div className="setting-item">
              <label>Cameras Loading</label>
              <select 
                value={settings.camerasLoading} 
                onChange={(e) => handleChange('camerasLoading', e.target.value)}
              >
                <option value="camera-vision">Camera Vision System</option>
                <option value="spinner">Classic Spinner</option>
                <option value="pulse">Pulse Wave</option>
                <option value="radar">Radar Scan</option>
                <option value="grid">Grid Matrix</option>
                <option value="circuit">Circuit Board</option>
                <option value="barcode">Barcode Scanner</option>
                <option value="ocr">OCR Text Recognition</option>
                <option value="users">Users Network</option>
                <option value="receipts">Receipt Scanner</option>
                <option value="cameras">Multi-Lens System</option>
                <option value="historical">Timeline History</option>
                <option value="settings">Gear System</option>
              </select>
            </div>

            <div className="setting-item">
              <label>Historical Loading</label>
              <select 
                value={settings.historicalLoading} 
                onChange={(e) => handleChange('historicalLoading', e.target.value)}
              >
                <option value="camera-vision">Camera Vision System</option>
                <option value="spinner">Classic Spinner</option>
                <option value="pulse">Pulse Wave</option>
                <option value="radar">Radar Scan</option>
                <option value="grid">Grid Matrix</option>
                <option value="circuit">Circuit Board</option>
                <option value="barcode">Barcode Scanner</option>
                <option value="ocr">OCR Text Recognition</option>
                <option value="users">Users Network</option>
                <option value="receipts">Receipt Scanner</option>
                <option value="cameras">Multi-Lens System</option>
                <option value="historical">Timeline History</option>
                <option value="settings">Gear System</option>
              </select>
            </div>

            <div className="setting-item">
              <label>Settings Loading</label>
              <select 
                value={settings.settingsLoading} 
                onChange={(e) => handleChange('settingsLoading', e.target.value)}
              >
                <option value="camera-vision">Camera Vision System</option>
                <option value="spinner">Classic Spinner</option>
                <option value="pulse">Pulse Wave</option>
                <option value="radar">Radar Scan</option>
                <option value="grid">Grid Matrix</option>
                <option value="circuit">Circuit Board</option>
                <option value="barcode">Barcode Scanner</option>
                <option value="ocr">OCR Text Recognition</option>
                <option value="users">Users Network</option>
                <option value="receipts">Receipt Scanner</option>
                <option value="cameras">Multi-Lens System</option>
                <option value="historical">Timeline History</option>
                <option value="settings">Gear System</option>
              </select>
            </div>
          </div>
        </div>

        {/* Loading Background Settings */}
        <div className="settings-section">
          <h2>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="2"/>
              <path d="M3 9h18M9 21V9" stroke="currentColor" strokeWidth="2"/>
            </svg>
            Loading Background
          </h2>
          <p style={{ fontSize: '14px', color: '#6b7280', marginBottom: '20px' }}>
            Customize background image and opacity for loading screens
          </p>
          <div className="settings-grid">
            <div className="setting-item">
              <label>Background Image</label>
              <select 
                value={settings.loadingBackground} 
                onChange={(e) => handleChange('loadingBackground', e.target.value)}
              >
                <option value="none">None</option>
                <option value="background1">Background 1</option>
                <option value="background2">Background 2</option>
                <option value="background3">Background 3</option>
                <option value="background4">Background 4</option>
              </select>
            </div>

            <div className="setting-item">
              <label>Background Opacity: {(settings.loadingBackgroundOpacity * 100).toFixed(0)}%</label>
              <input 
                type="range" 
                min="0" 
                max="1" 
                step="0.05"
                value={settings.loadingBackgroundOpacity}
                onChange={(e) => handleChange('loadingBackgroundOpacity', parseFloat(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>
          </div>
        </div>

        {/* Camera Settings */}
        <div className="settings-section">
          <h2>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <rect x="2" y="6" width="20" height="12" rx="2" stroke="currentColor" strokeWidth="2"/>
              <circle cx="12" cy="12" r="2.5" stroke="currentColor" strokeWidth="2"/>
            </svg>
            Camera Configuration
          </h2>

          {/* Camera 1 */}
          <div className="camera-settings-group">
            <div className="camera-header">
              <h3>Camera 1 - Production Line A</h3>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={settings.camera1Enabled}
                  onChange={(e) => handleChange('camera1Enabled', e.target.checked)}
                />
                <span className="slider"></span>
              </label>
            </div>
            <div className="settings-grid">
              <div className="setting-item">
                <label>Frame Rate (FPS)</label>
                <input
                  type="number"
                  value={settings.camera1Fps}
                  onChange={(e) => handleChange('camera1Fps', parseInt(e.target.value))}
                  disabled={!settings.camera1Enabled}
                />
              </div>
              <div className="setting-item">
                <label>Resolution</label>
                <select
                  value={settings.camera1Resolution}
                  onChange={(e) => handleChange('camera1Resolution', e.target.value)}
                  disabled={!settings.camera1Enabled}
                >
                  <option value="1920x1080">1920x1080 (Full HD)</option>
                  <option value="1280x720">1280x720 (HD)</option>
                  <option value="3840x2160">3840x2160 (4K)</option>
                </select>
              </div>
            </div>
          </div>

          {/* Camera 2 */}
          <div className="camera-settings-group">
            <div className="camera-header">
              <h3>Camera 2 - Production Line B</h3>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={settings.camera2Enabled}
                  onChange={(e) => handleChange('camera2Enabled', e.target.checked)}
                />
                <span className="slider"></span>
              </label>
            </div>
            <div className="settings-grid">
              <div className="setting-item">
                <label>Frame Rate (FPS)</label>
                <input
                  type="number"
                  value={settings.camera2Fps}
                  onChange={(e) => handleChange('camera2Fps', parseInt(e.target.value))}
                  disabled={!settings.camera2Enabled}
                />
              </div>
              <div className="setting-item">
                <label>Resolution</label>
                <select
                  value={settings.camera2Resolution}
                  onChange={(e) => handleChange('camera2Resolution', e.target.value)}
                  disabled={!settings.camera2Enabled}
                >
                  <option value="1920x1080">1920x1080 (Full HD)</option>
                  <option value="1280x720">1280x720 (HD)</option>
                  <option value="3840x2160">3840x2160 (4K)</option>
                </select>
              </div>
            </div>
          </div>

          {/* Camera 3 */}
          <div className="camera-settings-group">
            <div className="camera-header">
              <h3>Camera 3 - Quality Control</h3>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={settings.camera3Enabled}
                  onChange={(e) => handleChange('camera3Enabled', e.target.checked)}
                />
                <span className="slider"></span>
              </label>
            </div>
            <div className="settings-grid">
              <div className="setting-item">
                <label>Frame Rate (FPS)</label>
                <input
                  type="number"
                  value={settings.camera3Fps}
                  onChange={(e) => handleChange('camera3Fps', parseInt(e.target.value))}
                  disabled={!settings.camera3Enabled}
                />
              </div>
              <div className="setting-item">
                <label>Resolution</label>
                <select
                  value={settings.camera3Resolution}
                  onChange={(e) => handleChange('camera3Resolution', e.target.value)}
                  disabled={!settings.camera3Enabled}
                >
                  <option value="1920x1080">1920x1080 (Full HD)</option>
                  <option value="1280x720">1280x720 (HD)</option>
                  <option value="3840x2160">3840x2160 (4K)</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Processing Settings */}
        <div className="settings-section">
          <h2>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2"/>
              <path d="M12 1V3" stroke="currentColor" strokeWidth="2"/>
              <path d="M12 21V23" stroke="currentColor" strokeWidth="2"/>
              <path d="M4.22 4.22L5.64 5.64" stroke="currentColor" strokeWidth="2"/>
              <path d="M18.36 18.36L19.78 19.78" stroke="currentColor" strokeWidth="2"/>
              <path d="M1 12H3" stroke="currentColor" strokeWidth="2"/>
              <path d="M21 12H23" stroke="currentColor" strokeWidth="2"/>
              <path d="M4.22 19.78L5.64 18.36" stroke="currentColor" strokeWidth="2"/>
              <path d="M18.36 5.64L19.78 4.22" stroke="currentColor" strokeWidth="2"/>
            </svg>
            Processing Configuration
          </h2>
          <div className="settings-grid">
            <div className="setting-item">
              <label>Detection Threshold</label>
              <div className="range-input">
                <input
                  type="range"
                  min="0.5"
                  max="1.0"
                  step="0.05"
                  value={settings.detectionThreshold}
                  onChange={(e) => handleChange('detectionThreshold', parseFloat(e.target.value))}
                />
                <span>{(settings.detectionThreshold * 100).toFixed(0)}%</span>
              </div>
            </div>
            <div className="setting-item">
              <label>Max Processing Time (seconds)</label>
              <input
                type="number"
                value={settings.maxProcessingTime}
                onChange={(e) => handleChange('maxProcessingTime', parseInt(e.target.value))}
              />
            </div>
            <div className="setting-item">
              <label>Auto Reject Failed Products</label>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={settings.autoReject}
                  onChange={(e) => handleChange('autoReject', e.target.checked)}
                />
                <span className="slider"></span>
              </label>
            </div>
            <div className="setting-item">
              <label>Save Failed Product Images</label>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={settings.saveFailedImages}
                  onChange={(e) => handleChange('saveFailedImages', e.target.checked)}
                />
                <span className="slider"></span>
              </label>
            </div>
          </div>
        </div>

        {/* Notification Settings */}
        <div className="settings-section">
          <h2>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M18 8C18 6.4087 17.3679 4.88258 16.2426 3.75736C15.1174 2.63214 13.5913 2 12 2C10.4087 2 8.88258 2.63214 7.75736 3.75736C6.63214 4.88258 6 6.4087 6 8C6 15 3 17 3 17H21C21 17 18 15 18 8Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M13.73 21C13.5542 21.3031 13.3019 21.5547 12.9982 21.7295C12.6946 21.9044 12.3504 21.9965 12 21.9965C11.6496 21.9965 11.3054 21.9044 11.0018 21.7295C10.6982 21.5547 10.4458 21.3031 10.27 21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Notifications
          </h2>
          <div className="settings-grid">
            <div className="setting-item">
              <label>Email Notifications</label>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={settings.emailNotifications}
                  onChange={(e) => handleChange('emailNotifications', e.target.checked)}
                />
                <span className="slider"></span>
              </label>
            </div>
            <div className="setting-item">
              <label>SMS Notifications</label>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={settings.smsNotifications}
                  onChange={(e) => handleChange('smsNotifications', e.target.checked)}
                />
                <span className="slider"></span>
              </label>
            </div>
            <div className="setting-item">
              <label>Alert Threshold (failures/hour)</label>
              <input
                type="number"
                value={settings.alertThreshold}
                onChange={(e) => handleChange('alertThreshold', parseInt(e.target.value))}
              />
            </div>
            <div className="setting-item">
              <label>Daily Summary Report</label>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={settings.dailyReport}
                  onChange={(e) => handleChange('dailyReport', e.target.checked)}
                />
                <span className="slider"></span>
              </label>
            </div>
          </div>
        </div>
      </div>

      {/* Confirmation Dialog */}
      <ConfirmDialog
        isOpen={confirmDialog.isOpen}
        onClose={() => setConfirmDialog({ ...confirmDialog, isOpen: false })}
        onConfirm={confirmDialog.onConfirm}
        title={confirmDialog.title}
        message={confirmDialog.message}
        type={confirmDialog.type}
      />
    </div>
  );
}
