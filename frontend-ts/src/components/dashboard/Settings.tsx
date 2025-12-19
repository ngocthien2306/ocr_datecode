import React, { useState, useEffect } from 'react';
import { useToast } from '@/contexts/ToastContext';
import ConfirmDialog from '@/components/shared/ConfirmDialog';

interface SettingsState {
  // System Settings
  systemName: string;
  version: string;
  language: string;
  timezone: string;

  // Loading Templates
  dashboardLoading: string;
  usersLoading: string;
  receiptsLoading: string;
  camerasLoading: string;
  historicalLoading: string;
  settingsLoading: string;

  // Loading Background
  loadingBackground: string;
  loadingBackgroundOpacity: number;

  // Camera Settings
  camera1Enabled: boolean;
  camera1Fps: number;
  camera1Resolution: string;
  camera2Enabled: boolean;
  camera2Fps: number;
  camera2Resolution: string;
  camera3Enabled: boolean;
  camera3Fps: number;
  camera3Resolution: string;

  // Processing Settings
  detectionThreshold: number;
  maxProcessingTime: number;
  autoReject: boolean;
  saveFailedImages: boolean;

  // Notification Settings
  emailNotifications: boolean;
  smsNotifications: boolean;
  alertThreshold: number;
  dailyReport: boolean;
}

const loadingTemplates = [
  'camera-vision',
  'spinner',
  'pulse',
  'radar',
  'grid',
  'circuit',
  'barcode',
  'ocr',
  'users',
  'receipts',
  'cameras',
  'historical',
  'settings',
];

const Settings: React.FC = () => {
  const toast = useToast();
  const [confirmDialog, setConfirmDialog] = useState({
    isOpen: false,
    title: '',
    message: '',
    type: 'warning' as const,
    onConfirm: (() => {}) as () => void
  });

  const [settings, setSettings] = useState<SettingsState>({
    systemName: 'Suntech Automation',
    version: 'v1.0.0',
    language: 'en',
    timezone: 'UTC+7',
    dashboardLoading: 'camera-vision',
    usersLoading: 'users',
    receiptsLoading: 'receipts',
    camerasLoading: 'cameras',
    historicalLoading: 'historical',
    settingsLoading: 'settings',
    loadingBackground: 'none',
    loadingBackgroundOpacity: 0.2,
    camera1Enabled: true,
    camera1Fps: 30,
    camera1Resolution: '1920x1080',
    camera2Enabled: true,
    camera2Fps: 30,
    camera2Resolution: '1920x1080',
    camera3Enabled: true,
    camera3Fps: 30,
    camera3Resolution: '1920x1080',
    detectionThreshold: 0.85,
    maxProcessingTime: 5,
    autoReject: true,
    saveFailedImages: true,
    emailNotifications: true,
    smsNotifications: false,
    alertThreshold: 10,
    dailyReport: true,
  });

  useEffect(() => {
    const tabKeys = ['dashboard', 'users', 'receipts', 'cameras', 'historical', 'settings'];
    const loadedSettings: Partial<SettingsState> = {};

    tabKeys.forEach(tab => {
      const saved = localStorage.getItem(`${tab}Loading`);
      if (saved) {
        loadedSettings[`${tab}Loading` as keyof SettingsState] = saved as any;
      }
    });

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

  const handleChange = <K extends keyof SettingsState>(key: K, value: SettingsState[K]) => {
    setSettings(prev => ({ ...prev, [key]: value }));

    if (typeof key === 'string' && key.endsWith('Loading') && key !== 'loadingBackground') {
      localStorage.setItem(key, value as string);
      window.dispatchEvent(new CustomEvent('tabLoadingChanged', {
        detail: { tab: key, template: value }
      }));
    }

    if (key === 'loadingBackground' || key === 'loadingBackgroundOpacity') {
      localStorage.setItem(key, String(value));
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
        <div>
          <h2>System Settings</h2>
          <p>Configure application preferences</p>
        </div>
        <div className="header-actions">
          <button className="btn btn-secondary" onClick={handleReset}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M23 4V10H17M1 20V14H7" stroke="currentColor" strokeWidth="2"/>
              <path d="M3.51 9C4.88 5.28 8.18 3 12 3c5.52 0 10 4.48 10 10M1 14l4.64 4.36C6.71 19.43 8.04 20.22 9.5 20.64c1.46.42 3.01.47 4.49.14 1.48-.33 2.85-1.04 3.99-2.05 1.13-1.01 1.99-2.29 2.5-3.73" stroke="currentColor" strokeWidth="2"/>
            </svg>
            Reset to Default
          </button>
          <button className="btn btn-primary" onClick={handleSave}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v13a2 2 0 01-2 2z" stroke="currentColor" strokeWidth="2"/>
              <polyline points="17,21 17,13 7,13 7,21" stroke="currentColor" strokeWidth="2"/>
              <polyline points="7,3 7,8 15,8" stroke="currentColor" strokeWidth="2"/>
            </svg>
            Save Changes
          </button>
        </div>
      </div>

      <div className="settings-sections">
        {/* System Configuration */}
        <div className="settings-section">
          <h3>System Configuration</h3>
          <div className="settings-grid">
            <div className="form-group">
              <label>System Name</label>
              <input
                type="text"
                value={settings.systemName}
                onChange={(e) => handleChange('systemName', e.target.value)}
              />
            </div>
            <div className="form-group">
              <label>Version</label>
              <input type="text" value={settings.version} disabled />
            </div>
            <div className="form-group">
              <label>Language</label>
              <select value={settings.language} onChange={(e) => handleChange('language', e.target.value)}>
                <option value="en">English</option>
                <option value="vi">Tiếng Việt</option>
                <option value="zh">中文</option>
              </select>
            </div>
            <div className="form-group">
              <label>Timezone</label>
              <select value={settings.timezone} onChange={(e) => handleChange('timezone', e.target.value)}>
                <option value="UTC+7">UTC+7 (Bangkok, Hanoi)</option>
                <option value="UTC+8">UTC+8 (Singapore, Beijing)</option>
                <option value="UTC+9">UTC+9 (Tokyo, Seoul)</option>
              </select>
            </div>
          </div>
        </div>

        {/* Loading Templates */}
        <div className="settings-section">
          <h3>Loading Animation Templates</h3>
          <p className="section-description">Customize loading animations for each section</p>
          <div className="settings-grid">
            {['dashboard', 'users', 'receipts', 'cameras', 'historical', 'settings'].map(tab => (
              <div key={tab} className="form-group">
                <label>{tab.charAt(0).toUpperCase() + tab.slice(1)} Loading</label>
                <select
                  value={settings[`${tab}Loading` as keyof SettingsState] as string}
                  onChange={(e) => handleChange(`${tab}Loading` as keyof SettingsState, e.target.value as any)}
                >
                  {loadingTemplates.map(template => (
                    <option key={template} value={template}>
                      {template.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
                    </option>
                  ))}
                </select>
              </div>
            ))}
          </div>
        </div>

        {/* Loading Background */}
        <div className="settings-section">
          <h3>Loading Background</h3>
          <p className="section-description">Customize background for loading screens</p>
          <div className="settings-grid">
            <div className="form-group">
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
            <div className="form-group">
              <label>Background Opacity: {(settings.loadingBackgroundOpacity * 100).toFixed(0)}%</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={settings.loadingBackgroundOpacity}
                onChange={(e) => handleChange('loadingBackgroundOpacity', parseFloat(e.target.value))}
              />
            </div>
          </div>
        </div>

        {/* Camera Configuration */}
        <div className="settings-section">
          <h3>Camera Configuration</h3>
          <div className="settings-grid">
            {[1, 2, 3].map(num => (
              <React.Fragment key={num}>
                <div className="form-group checkbox-group">
                  <label>
                    <input
                      type="checkbox"
                      checked={settings[`camera${num}Enabled` as keyof SettingsState] as boolean}
                      onChange={(e) => handleChange(`camera${num}Enabled` as keyof SettingsState, e.target.checked as any)}
                    />
                    <span>Camera {num} Enabled</span>
                  </label>
                </div>
                <div className="form-group">
                  <label>Camera {num} FPS</label>
                  <input
                    type="number"
                    value={settings[`camera${num}Fps` as keyof SettingsState] as number}
                    onChange={(e) => handleChange(`camera${num}Fps` as keyof SettingsState, parseInt(e.target.value) as any)}
                    min="1"
                    max="60"
                  />
                </div>
                <div className="form-group">
                  <label>Camera {num} Resolution</label>
                  <select
                    value={settings[`camera${num}Resolution` as keyof SettingsState] as string}
                    onChange={(e) => handleChange(`camera${num}Resolution` as keyof SettingsState, e.target.value as any)}
                  >
                    <option value="1920x1080">1920x1080 (Full HD)</option>
                    <option value="1280x720">1280x720 (HD)</option>
                    <option value="640x480">640x480 (VGA)</option>
                  </select>
                </div>
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Processing Settings */}
        <div className="settings-section">
          <h3>Processing Configuration</h3>
          <div className="settings-grid">
            <div className="form-group">
              <label>Detection Threshold: {(settings.detectionThreshold * 100).toFixed(0)}%</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={settings.detectionThreshold}
                onChange={(e) => handleChange('detectionThreshold', parseFloat(e.target.value))}
              />
            </div>
            <div className="form-group">
              <label>Max Processing Time (seconds)</label>
              <input
                type="number"
                value={settings.maxProcessingTime}
                onChange={(e) => handleChange('maxProcessingTime', parseInt(e.target.value))}
                min="1"
                max="30"
              />
            </div>
            <div className="form-group checkbox-group">
              <label>
                <input
                  type="checkbox"
                  checked={settings.autoReject}
                  onChange={(e) => handleChange('autoReject', e.target.checked)}
                />
                <span>Auto Reject Failed Images</span>
              </label>
            </div>
            <div className="form-group checkbox-group">
              <label>
                <input
                  type="checkbox"
                  checked={settings.saveFailedImages}
                  onChange={(e) => handleChange('saveFailedImages', e.target.checked)}
                />
                <span>Save Failed Images for Review</span>
              </label>
            </div>
          </div>
        </div>

        {/* Notification Settings */}
        <div className="settings-section">
          <h3>Notification Preferences</h3>
          <div className="settings-grid">
            <div className="form-group checkbox-group">
              <label>
                <input
                  type="checkbox"
                  checked={settings.emailNotifications}
                  onChange={(e) => handleChange('emailNotifications', e.target.checked)}
                />
                <span>Email Notifications</span>
              </label>
            </div>
            <div className="form-group checkbox-group">
              <label>
                <input
                  type="checkbox"
                  checked={settings.smsNotifications}
                  onChange={(e) => handleChange('smsNotifications', e.target.checked)}
                />
                <span>SMS Notifications</span>
              </label>
            </div>
            <div className="form-group">
              <label>Alert Threshold (errors per hour)</label>
              <input
                type="number"
                value={settings.alertThreshold}
                onChange={(e) => handleChange('alertThreshold', parseInt(e.target.value))}
                min="1"
                max="100"
              />
            </div>
            <div className="form-group checkbox-group">
              <label>
                <input
                  type="checkbox"
                  checked={settings.dailyReport}
                  onChange={(e) => handleChange('dailyReport', e.target.checked)}
                />
                <span>Daily Summary Report</span>
              </label>
            </div>
          </div>
        </div>
      </div>

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
};

export default Settings;
