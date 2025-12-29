import { useState, useEffect, useRef } from 'react';
import '@/styles/Dashboard.css';
import UserManagement from './UserManagement';
import Receipts from '../recipe/Receipts';
import Historical from './Historical';
import Settings from './Settings';
import CameraManagement from '../camera/CameraManagement';
import ConfirmDialog from '../shared/ConfirmDialog';
import { camerasAPI } from '@/services/api';
import { API_BASE_URL } from '@/config/api';
import type { Camera as BaseCamera } from '@/types';

interface DashboardCamera extends Omit<BaseCamera, 'status'> {
  is_connected: boolean;
  location?: string;
  model_name: string;
  serial_number?: string;
  max_frame_rate?: number;
  resolution_width: number;
  resolution_height: number;
  is_active: boolean;
}

interface DashboardProps {
  onLogout: () => void;
}

type Section = 'dashboard' | 'users' | 'receipts' | 'cameras' | 'historical' | 'settings';

type LoadingTemplate = 'camera-vision' | 'users' | 'receipts' | 'cameras' | 'historical' | 'settings' | 'spinner' | 'pulse' | 'radar' | 'grid' | 'circuit' | 'barcode' | 'ocr' | 'dots' | 'waves';

interface LoadingTemplates {
  dashboard: LoadingTemplate;
  users: LoadingTemplate;
  receipts: LoadingTemplate;
  cameras: LoadingTemplate;
  historical: LoadingTemplate;
  settings: LoadingTemplate;
}

interface LoadingBackground {
  image: string;
  opacity: number;
}

interface ConfirmDialogState {
  isOpen: boolean;
  title: string;
  message: string;
  type: 'warning' | 'danger' | 'info';
  onConfirm: (() => void) | null;
}

interface CameraStats {
  total: number;
  connected: number;
  active: number;
}

export default function Dashboard({ onLogout }: DashboardProps) {
  const [currentSection, setCurrentSection] = useState<Section>('dashboard');
  const [isLoading, setIsLoading] = useState(true);
  const [loadingTemplates, setLoadingTemplates] = useState<LoadingTemplates>(() => {
    return {
      dashboard: (localStorage.getItem('dashboardLoading') as LoadingTemplate) || 'camera-vision',
      users: (localStorage.getItem('usersLoading') as LoadingTemplate) || 'users',
      receipts: (localStorage.getItem('receiptsLoading') as LoadingTemplate) || 'receipts',
      cameras: (localStorage.getItem('camerasLoading') as LoadingTemplate) || 'cameras',
      historical: (localStorage.getItem('historicalLoading') as LoadingTemplate) || 'historical',
      settings: (localStorage.getItem('settingsLoading') as LoadingTemplate) || 'settings'
    };
  });
  const [loadingBackground, setLoadingBackground] = useState<LoadingBackground>(() => {
    return {
      image: localStorage.getItem('loadingBackground') || 'background1',
      opacity: parseFloat(localStorage.getItem('loadingBackgroundOpacity') || '0.2')
    };
  });
  const [darkMode, setDarkMode] = useState(() => {
    const savedMode = localStorage.getItem('appThemeMode');
    return savedMode === 'dark';
  });
  const [currentTime, setCurrentTime] = useState(new Date());
  const [totalProductsToday, setTotalProductsToday] = useState(1247);
  const [cameras, setCameras] = useState<DashboardCamera[]>([]);
  const [confirmDialog, setConfirmDialog] = useState<ConfirmDialogState>({
    isOpen: false,
    title: '',
    message: '',
    type: 'warning',
    onConfirm: null
  });
  const [cameraStats, setCameraStats] = useState<CameraStats>({
    total: 0,
    connected: 0,
    active: 0
  });
  const [currentUser, setCurrentUser] = useState<any>(null);

  const camera1ChartRef = useRef<HTMLCanvasElement>(null);
  const camera2ChartRef = useRef<HTMLCanvasElement>(null);
  const camera3ChartRef = useRef<HTMLCanvasElement>(null);

  const [camera1Data, setCamera1Data] = useState([45, 52, 48, 65, 58, 72, 68, 75, 82, 78, 85, 92, 88, 95, 91, 98, 102, 96, 105, 110, 108, 115, 112, 120]);
  const [camera2Data, setCamera2Data] = useState([38, 42, 45, 52, 48, 58, 62, 68, 65, 72, 75, 80, 76, 83, 79, 86, 90, 85, 92, 95, 98, 102, 99, 105]);
  const [camera3Data, setCamera3Data] = useState([32, 35, 38, 42, 45, 48, 52, 55, 58, 62, 65, 68, 64, 70, 67, 73, 76, 72, 78, 82, 79, 85, 88, 90]);

  const hours = ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'];

  // Helper function to render loading template
  const renderLoadingTemplate = (template: LoadingTemplate) => {
    switch(template) {
      case 'camera-vision':
        return (
          <>
            <div className="camera-focus-ring">
              <div className="scan-lines">
                <div className="scan-line"></div>
                <div className="scan-line"></div>
              </div>
              <div className="focus-ring"></div>
              <div className="focus-ring"></div>
              <div className="focus-ring"></div>
              <div className="focus-ring"></div>
              <div className="camera-lens"></div>
              <div className="corner-brackets">
                <div className="bracket top-left"></div>
                <div className="bracket top-right"></div>
                <div className="bracket bottom-left"></div>
                <div className="bracket bottom-right"></div>
              </div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">VISION SYSTEM</div>
              <div className="loading-subtitle">INITIALIZING...</div>
            </div>
            <div className="progress-indicators">
              <div className="progress-bar-container">
                <div className="progress-bar-fill"></div>
              </div>
              <div className="status-dots">
                <div className="status-dot"></div>
                <div className="status-dot"></div>
                <div className="status-dot"></div>
                <div className="status-dot"></div>
              </div>
            </div>
            <div className="system-info">
              <div className="system-info-item">
                <span className="status-indicator"></span>
                <span>CAMERA MODULE</span>
              </div>
              <div className="system-info-item">
                <span className="status-indicator"></span>
                <span>OCR ENGINE</span>
              </div>
              <div className="system-info-item">
                <span className="status-indicator"></span>
                <span>DATABASE</span>
              </div>
            </div>
          </>
        );

      case 'spinner':
        return (
          <>
            <div className="spinner-loader">
              <div className="spinner-circle"></div>
              <div className="spinner-circle"></div>
              <div className="spinner-circle"></div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">LOADING</div>
              <div className="loading-subtitle">PLEASE WAIT...</div>
            </div>
          </>
        );

      case 'pulse':
        return (
          <>
            <div className="pulse-loader">
              <div className="pulse-ring"></div>
              <div className="pulse-ring"></div>
              <div className="pulse-ring"></div>
              <div className="pulse-core"></div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">PROCESSING</div>
              <div className="loading-subtitle">ANALYZING DATA...</div>
            </div>
          </>
        );

      case 'radar':
        return (
          <>
            <div className="radar-loader">
              <div className="radar-grid">
                <div className="radar-circle"></div>
                <div className="radar-circle"></div>
                <div className="radar-circle"></div>
                <div className="radar-circle"></div>
              </div>
              <div className="radar-sweep"></div>
              <div className="radar-dot"></div>
              <div className="radar-dot"></div>
              <div className="radar-dot"></div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">SCANNING</div>
              <div className="loading-subtitle">DETECTING OBJECTS...</div>
            </div>
          </>
        );

      case 'grid':
        return (
          <>
            <div className="grid-loader">
              <div className="grid-container">
                {[...Array(25)].map((_, i) => (
                  <div key={i} className="grid-cell"></div>
                ))}
              </div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">MATRIX LOADING</div>
              <div className="loading-subtitle">SYNCING SYSTEMS...</div>
            </div>
          </>
        );

      case 'circuit':
        return (
          <>
            <div className="circuit-loader">
              <div className="circuit-board">
                <div className="circuit-line horizontal line-1"></div>
                <div className="circuit-line horizontal line-2"></div>
                <div className="circuit-line vertical line-3"></div>
                <div className="circuit-line vertical line-4"></div>
                <div className="circuit-node node-1"></div>
                <div className="circuit-node node-2"></div>
                <div className="circuit-node node-3"></div>
                <div className="circuit-node node-4"></div>
                <div className="circuit-chip"></div>
              </div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">SYSTEM BOOT</div>
              <div className="loading-subtitle">INITIALIZING HARDWARE...</div>
            </div>
          </>
        );

      case 'barcode':
        return (
          <>
            <div className="barcode-loader">
              <div className="barcode-container">
                {[...Array(15)].map((_, i) => (
                  <div key={i} className="barcode-line"></div>
                ))}
                <div className="scanner-beam"></div>
              </div>
              <div className="scanner-frame">
                <div className="scanner-corner top-left"></div>
                <div className="scanner-corner top-right"></div>
                <div className="scanner-corner bottom-left"></div>
                <div className="scanner-corner bottom-right"></div>
              </div>
              <div className="barcode-digits">8 901234 567890</div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">BARCODE SCANNING</div>
              <div className="loading-subtitle">READING CODE...</div>
            </div>
          </>
        );

      case 'ocr':
        return (
          <>
            <div className="ocr-loader">
              <div className="ocr-document">
                <div className="ocr-text-line"></div>
                <div className="ocr-text-line"></div>
                <div className="ocr-text-line"></div>
                <div className="ocr-text-line"></div>
                <div className="ocr-text-line"></div>
                <div className="ocr-scan-overlay"></div>
                <div className="ocr-highlight-box"></div>
                <div className="ocr-highlight-box"></div>
                <div className="ocr-highlight-box"></div>
              </div>
              <div className="ocr-analysis-indicators">
                <div className="ocr-indicator">
                  <div className="ocr-indicator-icon">A</div>
                  <div className="ocr-indicator-label">TEXT</div>
                </div>
                <div className="ocr-indicator">
                  <div className="ocr-indicator-icon">1</div>
                  <div className="ocr-indicator-label">DIGIT</div>
                </div>
                <div className="ocr-indicator">
                  <div className="ocr-indicator-icon">✓</div>
                  <div className="ocr-indicator-label">VERIFY</div>
                </div>
              </div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">OCR PROCESSING</div>
              <div className="loading-subtitle">RECOGNIZING TEXT...</div>
            </div>
          </>
        );

      case 'users':
        return (
          <>
            <div className="users-loader">
              <div className="users-circle-container">
                <div className="user-avatar"></div>
                <div className="user-avatar"></div>
                <div className="user-avatar"></div>
              </div>
              <div className="users-connection"></div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">USER MANAGEMENT</div>
              <div className="loading-subtitle">LOADING USERS...</div>
            </div>
          </>
        );

      case 'receipts':
        return (
          <>
            <div className="receipts-loader">
              <div className="receipt-paper">
                <div className="receipt-header">
                  <div className="receipt-icon"></div>
                  <div className="receipt-title"></div>
                </div>
                <div className="receipt-line"></div>
                <div className="receipt-line"></div>
                <div className="receipt-line"></div>
                <div className="receipt-line"></div>
                <div className="receipt-scanner-line"></div>
              </div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">RECEIPTS</div>
              <div className="loading-subtitle">SCANNING DOCUMENTS...</div>
            </div>
          </>
        );

      case 'cameras':
        return (
          <>
            <div className="cameras-loader">
              <div className="camera-grid">
                <div className="camera-lens"></div>
                <div className="camera-lens"></div>
                <div className="camera-lens"></div>
                <div className="camera-lens"></div>
              </div>
              <div className="camera-status-indicator">
                <div className="camera-dot"></div>
                <div className="camera-dot"></div>
                <div className="camera-dot"></div>
                <div className="camera-dot"></div>
              </div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">CAMERAS</div>
              <div className="loading-subtitle">CONNECTING DEVICES...</div>
            </div>
          </>
        );

      case 'historical':
        return (
          <>
            <div className="historical-loader">
              <div className="timeline-container">
                <div className="timeline-line"></div>
                <div className="timeline-point"></div>
                <div className="timeline-point"></div>
                <div className="timeline-point"></div>
                <div className="timeline-card"></div>
                <div className="timeline-card"></div>
              </div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">HISTORICAL DATA</div>
              <div className="loading-subtitle">LOADING TIMELINE...</div>
            </div>
          </>
        );

      case 'settings':
        return (
          <>
            <div className="settings-loader">
              <div className="gears-container">
                <div className="gear gear-large"></div>
                <div className="gear gear-small-1"></div>
                <div className="gear gear-small-2"></div>
                <div className="settings-particles">
                  <div className="settings-particle"></div>
                  <div className="settings-particle"></div>
                  <div className="settings-particle"></div>
                </div>
              </div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">SETTINGS</div>
              <div className="loading-subtitle">CONFIGURING...</div>
            </div>
          </>
        );

      default:
        return (
          <>
            <div className="spinner-loader">
              <div className="spinner-circle"></div>
              <div className="spinner-circle"></div>
              <div className="spinner-circle"></div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">LOADING</div>
              <div className="loading-subtitle">PLEASE WAIT...</div>
            </div>
          </>
        );
    }
  };

  // Fetch cameras on component mount
  useEffect(() => {
    fetchCameras();
  }, []);

  // Load current user info
  useEffect(() => {
    const userData = localStorage.getItem('user');
    if (userData) {
      try {
        const user = JSON.parse(userData);
        setCurrentUser(user);
      } catch (error) {
        console.error('Error parsing user data:', error);
      }
    }
  }, []);

  // Listen for loading template changes from Settings
  useEffect(() => {
    const handleTemplateChange = (event: any) => {
      const { tab, template } = event.detail;
      const tabName = tab.replace('Loading', '') as keyof LoadingTemplates;
      setLoadingTemplates(prev => ({
        ...prev,
        [tabName]: template
      }));
    };

    const handleBackgroundChange = (event: any) => {
      const { background, opacity } = event.detail;
      setLoadingBackground({
        image: background,
        opacity: opacity
      });
    };

    window.addEventListener('tabLoadingChanged', handleTemplateChange);
    window.addEventListener('loadingBackgroundChanged', handleBackgroundChange);

    return () => {
      window.removeEventListener('tabLoadingChanged', handleTemplateChange);
      window.removeEventListener('loadingBackgroundChanged', handleBackgroundChange);
    };
  }, []);

  const fetchCameras = async () => {
    const startTime = Date.now();
    
    try {
      const [allCameras, countData] = await Promise.all([
        camerasAPI.getAllCameras(0, 100),
        camerasAPI.getCamerasCount()
      ]);
      
      setCameras(allCameras as unknown as DashboardCamera[]);
      
      const connectedCount = (allCameras as unknown as DashboardCamera[]).filter(c => c.is_connected).length;
      const activeCount = (allCameras as unknown as DashboardCamera[]).filter(c => c.is_active).length;
      
      setCameraStats({
        total: countData.count || allCameras.length,
        connected: connectedCount,
        active: activeCount
      });
    } catch (err) {
      console.error('Error fetching cameras:', err);
    } finally {
      // Ensure minimum loading time of 1000ms
      const elapsedTime = Date.now() - startTime;
      const remainingTime = Math.max(0, 1000 - elapsedTime);
      
      setTimeout(() => {
        setIsLoading(false);
      }, remainingTime);
    }
  };

  // Handle section change with loading
  const handleSectionChange = (section: Section) => {
    setIsLoading(true);
    setCurrentSection(section);
    
    // Simulate loading time of 1000ms
    setTimeout(() => {
      setIsLoading(false);
    }, 1000);
  };

  // Dark mode effect
  useEffect(() => {
    if (darkMode) {
      document.body.classList.add('dark-mode');
      localStorage.setItem('appThemeMode', 'dark');
    } else {
      document.body.classList.remove('dark-mode');
      localStorage.setItem('appThemeMode', 'light');
    }
  }, [darkMode]);

  // Update time every minute
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 60000);
    return () => clearInterval(timer);
  }, []);

  // Draw charts
  useEffect(() => {
    drawLineChart(camera1ChartRef.current, camera1Data, '#6366f1');
    drawLineChart(camera2ChartRef.current, camera2Data, '#3b82f6');
    drawLineChart(camera3ChartRef.current, camera3Data, '#8b5cf6');
  }, [camera1Data, camera2Data, camera3Data, darkMode]);

  // Simulate data updates
  useEffect(() => {
    const interval = setInterval(() => {
      setTotalProductsToday(prev => prev + Math.floor(Math.random() * 5));

      setCamera1Data(prev => {
        const newData = [...prev];
        newData.shift();
        newData.push(Math.floor(Math.random() * 40) + 60);
        return newData;
      });

      setCamera2Data(prev => {
        const newData = [...prev];
        newData.shift();
        newData.push(Math.floor(Math.random() * 35) + 55);
        return newData;
      });

      setCamera3Data(prev => {
        const newData = [...prev];
        newData.shift();
        newData.push(Math.floor(Math.random() * 30) + 45);
        return newData;
      });
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  const drawLineChart = (canvas: HTMLCanvasElement | null, data: number[], color: string) => {
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    const padding = 40;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    const maxValue = Math.max(...data);
    const minValue = Math.min(...data);
    const valueRange = maxValue - minValue || 1;

    // Draw grid lines
    ctx.strokeStyle = darkMode ? '#374151' : '#e5e7eb';
    ctx.lineWidth = 1;

    for (let i = 0; i <= 4; i++) {
      const y = padding + (chartHeight / 4) * i;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    // Draw line
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();

    data.forEach((value, index) => {
      const x = padding + (index / (data.length - 1)) * chartWidth;
      const y = padding + chartHeight - ((value - minValue) / valueRange) * chartHeight;

      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();

    // Draw area under line
    ctx.lineTo(padding + chartWidth, padding + chartHeight);
    ctx.lineTo(padding, padding + chartHeight);
    ctx.closePath();

    const gradient = ctx.createLinearGradient(0, padding, 0, padding + chartHeight);
    gradient.addColorStop(0, color + '40');
    gradient.addColorStop(1, color + '00');
    ctx.fillStyle = gradient;
    ctx.fill();

    // Draw points
    ctx.fillStyle = color;
    data.forEach((value, index) => {
      const x = padding + (index / (data.length - 1)) * chartWidth;
      const y = padding + chartHeight - ((value - minValue) / valueRange) * chartHeight;

      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();
    });

    // Draw labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
    ctx.textAlign = 'center';

    const labelStep = Math.ceil(hours.length / 6);
    hours.forEach((hour, index) => {
      if (index % labelStep === 0 || index === hours.length - 1) {
        const x = padding + (index / (data.length - 1)) * chartWidth;
        ctx.fillText(hour, x, height - 15);
      }
    });

    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const value = minValue + (valueRange / 4) * (4 - i);
      const y = padding + (chartHeight / 4) * i + 5;
      ctx.fillText(Math.round(value).toString(), padding - 10, y);
    }
  };

  const handleLogout = () => {
    setConfirmDialog({
      isOpen: true,
      title: 'Logout',
      message: 'Are you sure you want to logout?',
      type: 'warning',
      onConfirm: () => {
        document.body.classList.remove('dark-mode');
        onLogout();
      }
    });
  };

  const timeString = currentTime.toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    hour12: true
  });

  const dateString = currentTime.toLocaleDateString('en-US', {
    weekday: 'long',
    day: 'numeric',
    month: 'long',
    year: 'numeric'
  });

  return (
    <div className="dashboard-container">
      {/* Background for entire dashboard-container */}
      {loadingBackground.image !== 'none' && (
        <div 
          className="dashboard-background"
          style={{
            backgroundImage: `url(/background/${loadingBackground.image}.png)`,
            opacity: loadingBackground.opacity
          }}
        />
      )}
      
      {/* Header */}
      <header className="dashboard-header-top">
        <div className="header-left">
          <div className="dashboard-logo-section">
            <div className="company-logo">
              <img src="logo.png" alt="Suntech Automation" className="h-16 w-auto" />
            </div>
            <div className="dashboard-logo-divider"></div>
            <div className="dashboard-company-info">
              <h1 className="dashboard-company-name">SUNTECH AUTOMATION</h1>
              <p className="dashboard-version">v1.0.0 PRODUCTION</p>
            </div>
          </div>
        </div>
        <div className="header-center">
          <div className="search-bar">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <circle cx="11" cy="11" r="8" stroke="currentColor" strokeWidth="2"/>
              <path d="m21 21-4.35-4.35" stroke="currentColor" strokeWidth="2"/>
            </svg>
            <input type="text" placeholder="Search cameras, receipts..." />
          </div>
        </div>
        <div className="header-right">
          <div className="connection-status">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="2" fill={cameraStats.connected === cameraStats.total && cameraStats.total > 0 ? "#10b981" : "#f59e0b"}/>
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z" stroke="currentColor" strokeWidth="2"/>
            </svg>
            <span>{cameraStats.connected === cameraStats.total && cameraStats.total > 0 ? 'All cameras online' : `${cameraStats.connected}/${cameraStats.total} cameras online`}</span>
          </div>
          <div className="profile">
            {currentUser?.avatar_url ? (
              <img 
                src={`${API_BASE_URL}${currentUser.avatar_url}`} 
                alt="User Avatar" 
                className="profile-avatar" 
              />
            ) : (
              <div className="profile-avatar-placeholder">
                {currentUser?.full_name?.charAt(0) || currentUser?.username?.charAt(0) || 'U'}
              </div>
            )}
            <span>{currentUser?.full_name || currentUser?.username || 'User'}</span>
          </div>
        </div>
      </header>

      <div className="main-content-wrapper">
        {/* Sidebar */}
        <aside className="sidebar">
          <h3>Menu</h3>
          <nav className="nav-menu">
            <a
              href="#dashboard"
              className={`nav-item ${currentSection === 'dashboard' ? 'active' : ''}`}
              onClick={(e) => { e.preventDefault(); handleSectionChange('dashboard'); }}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <rect x="3" y="3" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
                <rect x="14" y="3" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
                <rect x="14" y="14" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
                <rect x="3" y="14" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
              </svg>
              Dashboard
            </a>
            <a
              href="#users"
              className={`nav-item ${currentSection === 'users' ? 'active' : ''}`}
              onClick={(e) => { e.preventDefault(); handleSectionChange('users'); }}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M17 21V19C17 17.9391 16.5786 16.9217 15.8284 16.1716C15.0783 15.4214 14.0609 15 13 15H5C3.93913 15 2.92172 15.4214 2.17157 16.1716C1.42143 16.9217 1 17.9391 1 19V21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <circle cx="9" cy="7" r="4" stroke="currentColor" strokeWidth="2"/>
                <path d="M23 21V19C22.9993 18.1137 22.7044 17.2528 22.1614 16.5523C21.6184 15.8519 20.8581 15.3516 20 15.13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M16 3.13C16.8604 3.35031 17.623 3.85071 18.1676 4.55232C18.7122 5.25392 19.0078 6.11683 19.0078 7.005C19.0078 7.89318 18.7122 8.75608 18.1676 9.45769C17.623 10.1593 16.8604 10.6597 16 10.88" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              User Management
            </a>
            <a
              href="#receipts"
              className={`nav-item ${currentSection === 'receipts' ? 'active' : ''}`}
              onClick={(e) => { e.preventDefault(); handleSectionChange('receipts'); }}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <polyline points="14,2 14,8 20,8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <line x1="16" y1="13" x2="8" y2="13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <line x1="16" y1="17" x2="8" y2="17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <polyline points="10,9 9,9 8,9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              Receipts
            </a>
            <a
              href="#cameras"
              className={`nav-item ${currentSection === 'cameras' ? 'active' : ''}`}
              onClick={(e) => { e.preventDefault(); handleSectionChange('cameras'); }}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <rect x="2" y="4" width="20" height="16" rx="2" stroke="currentColor" strokeWidth="2"/>
                <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2"/>
                <path d="M7 4v2M17 4v2" stroke="currentColor" strokeWidth="2"/>
              </svg>
              Cameras
            </a>
            <a
              href="#historical"
              className={`nav-item ${currentSection === 'historical' ? 'active' : ''}`}
              onClick={(e) => { e.preventDefault(); handleSectionChange('historical'); }}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
                <polyline points="12,6 12,12 16,14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              Historical
            </a>
          </nav>

          <h3>Appearance</h3>
          <div className="appearance-controls">
            <div className="dark-mode-toggle">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M21 12.79A9 9 0 1 1 11.21 3A7 7 0 0 0 21 12.79Z" stroke="currentColor" strokeWidth="2"/>
              </svg>
              <span>Dark mode</span>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={darkMode}
                  onChange={(e) => setDarkMode(e.target.checked)}
                />
                <span className="slider"></span>
              </label>
            </div>
          </div>

          <h3>Action</h3>
          <nav className="action-menu">
            <a
              href="#settings"
              className={`nav-item ${currentSection === 'settings' ? 'active' : ''}`}
              onClick={(e) => { e.preventDefault(); handleSectionChange('settings'); }}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2"/>
                <path d="M19.4 15A1.65 1.65 0 0 0 21 13.35A1.65 1.65 0 0 0 19.4 11.65L18.7 12.35A7.81 7.81 0 0 0 12 5.29V4A1 1 0 0 0 11 3A1 1 0 0 0 10 4V5.29A7.81 7.81 0 0 0 3.3 12.35L2.6 11.65A1.65 1.65 0 0 0 1 13.35A1.65 1.65 0 0 0 2.6 15L3.3 14.3A7.81 7.81 0 0 0 10 20.71V22A1 1 0 0 0 11 23A1 1 0 0 0 12 22V20.71A7.81 7.81 0 0 0 18.7 14.3L19.4 15Z" stroke="currentColor" strokeWidth="2"/>
              </svg>
              Settings
            </a>
            <a href="#logout" className="nav-item" onClick={(e) => { e.preventDefault(); handleLogout(); }}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M9 21H5A2 2 0 0 1 3 19V5A2 2 0 0 1 5 3H9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <polyline points="16,17 21,12 16,7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <line x1="21" y1="12" x2="9" y2="12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              Logout
            </a>
          </nav>
        </aside>

        {/* Main Dashboard */}
        <main className="dashboard-main">
          {isLoading ? (
            <div className="loading-overlay">
              <div className="loading-container">
                {renderLoadingTemplate(loadingTemplates[currentSection] || 'camera-vision')}
              </div>
            </div>
          ) : (
            <>
              {currentSection === 'dashboard' && (
                <section className="content-section active">
              <div className="dashboard-header-section">
                <div className="welcome-section">
                  <h1>Camera Production Dashboard</h1>
                  <p>Monitor camera performance and product processing in real-time</p>
                </div>
                <div className="time-section">
                  <div className="time">{timeString}</div>
                  <div className="date">{dateString}</div>
                </div>
              </div>

              {/* Summary Cards Row */}
              <div className="summary-cards">
                <div className="summary-card">
                  <div className="card-icon camera">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                      <rect x="2" y="6" width="20" height="12" rx="2" stroke="currentColor" strokeWidth="2"/>
                      <circle cx="12" cy="12" r="2.5" stroke="currentColor" strokeWidth="2"/>
                    </svg>
                  </div>
                  <div className="card-info">
                    <h3>Total Cameras</h3>
                    <div className="card-value">{cameraStats.total}</div>
                    <span className={`card-status ${cameraStats.connected === cameraStats.total && cameraStats.total > 0 ? 'online' : 'offline'}`}>
                      {cameraStats.connected === cameraStats.total && cameraStats.total > 0 ? 'All online' : `${cameraStats.connected} online`}
                    </span>
                  </div>
                </div>

                <div className="summary-card">
                  <div className="card-icon products">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                      <path d="M21 16V8C20.9996 7.64927 20.9071 7.30481 20.7315 7.00116C20.556 6.69751 20.3037 6.44536 20 6.27L13 2.27C12.696 2.09446 12.3511 2.00205 12 2.00205C11.6489 2.00205 11.304 2.09446 11 2.27L4 6.27C3.69626 6.44536 3.44398 6.69751 3.26846 7.00116C3.09294 7.30481 3.00036 7.64927 3 8V16C3.00036 16.3507 3.09294 16.6952 3.26846 16.9988C3.44398 17.3025 3.69626 17.5546 4 17.73L11 21.73C11.304 21.9055 11.6489 21.998 12 21.998C12.3511 21.998 12.696 21.9055 13 21.73L20 17.73C20.3037 17.5546 20.556 17.3025 20.7315 16.9988C20.9071 16.6952 20.9996 16.3507 21 16Z" stroke="currentColor" strokeWidth="2"/>
                      <polyline points="3.27,6.96 12,12.01 20.73,6.96" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      <line x1="12" y1="22.08" x2="12" y2="12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                  <div className="card-info">
                    <h3>Products Today</h3>
                    <div className="card-value">{totalProductsToday.toLocaleString()}</div>
                    <span className="card-status increase">+12% vs yesterday</span>
                  </div>
                </div>

                <div className="summary-card">
                  <div className="card-icon processing">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
                      <polyline points="12,6 12,12 16,14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                  <div className="card-info">
                    <h3>Avg Processing Time</h3>
                    <div className="card-value">2.3s</div>
                    <span className="card-status normal">Per product</span>
                  </div>
                </div>

                <div className="summary-card">
                  <div className="card-icon success">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                      <polyline points="20,6 9,17 4,12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                  <div className="card-info">
                    <h3>Success Rate</h3>
                    <div className="card-value">98.5%</div>
                    <span className="card-status success">Excellent</span>
                  </div>
                </div>
              </div>

              <div className="dashboard-main-content">
                {/* Left Section - Camera Previews */}
                <div className="dashboard-left">
                  <div className="camera-previews">
                    {cameras.slice(0, 3).map((camera, index) => (
                      <div key={camera.camera_id} className="camera-preview-card">
                        <div className="camera-preview-header">
                          <div className="camera-preview-title">
                            <div className={`camera-status ${camera.is_connected ? 'active' : 'inactive'}`}></div>
                            <span>{camera.camera_id} - {camera.location || camera.model_name}</span>
                          </div>
                          {camera.is_connected && <span className="live-badge">LIVE</span>}
                        </div>
                        <div className="camera-preview-frame">
                          <img 
                            src={
                              index === 0 ? 'https://images.pexels.com/photos/1267338/pexels-photo-1267338.jpeg?auto=compress&cs=tinysrgb&w=800' :
                              index === 1 ? 'https://images.pexels.com/photos/3862130/pexels-photo-3862130.jpeg?auto=compress&cs=tinysrgb&w=800' :
                              'https://images.pexels.com/photos/5022849/pexels-photo-5022849.jpeg?auto=compress&cs=tinysrgb&w=800'
                            }
                            alt={`${camera.camera_id} Feed`} 
                          />
                          <div className="camera-overlay-info">
                            <div className="overlay-stat">
                              <span className="overlay-label">FPS:</span>
                              <span className="overlay-value">{camera.max_frame_rate || 30}</span>
                            </div>
                            <div className="overlay-stat">
                              <span className="overlay-label">Resolution:</span>
                              <span className="overlay-value">{camera.resolution_width}x{camera.resolution_height}</span>
                            </div>
                            <div className="overlay-stat">
                              <span className="overlay-label">IP:</span>
                              <span className="overlay-value">{camera.ip_address || 'N/A'}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                    {cameras.length === 0 && (
                      <div style={{ padding: '2rem', textAlign: 'center', color: '#6b7280' }}>
                        No cameras available. Please add cameras in Camera Management.
                      </div>
                    )}
                  </div>
                </div>

                {/* Center Section - Camera Statistics Charts */}
                <div className="dashboard-center">
                  <div className="camera-stats">
                    {cameras.slice(0, 3).map((camera, index) => {
                      const chartRef = index === 0 ? camera1ChartRef : index === 1 ? camera2ChartRef : camera3ChartRef;
                      return (
                        <div key={camera.camera_id} className="camera-card">
                          <div className="camera-card-header">
                            <div className="camera-title">
                              <div className={`camera-status ${camera.is_connected ? 'active' : 'inactive'}`}></div>
                              <h3>{camera.camera_id} - {camera.location || camera.model_name}</h3>
                            </div>
                            <select className="time-filter">
                              <option>Today</option>
                              <option>Last 7 days</option>
                              <option>Last 30 days</option>
                            </select>
                          </div>
                          <div className="camera-card-body">
                            <div className="camera-metrics">
                              <div className="metric">
                                <span className="metric-label">Model</span>
                                <span className="metric-value" style={{ fontSize: '14px' }}>{camera.model_name}</span>
                              </div>
                              <div className="metric">
                                <span className="metric-label">Serial</span>
                                <span className="metric-value" style={{ fontSize: '14px' }}>{camera.serial_number}</span>
                              </div>
                              <div className="metric success">
                                <span className="metric-label">Status</span>
                                <span className="metric-value" style={{ fontSize: '14px' }}>
                                  {camera.is_connected ? 'Connected' : 'Disconnected'}
                                </span>
                              </div>
                              <div className="metric">
                                <span className="metric-label">FPS</span>
                                <span className="metric-value">{camera.max_frame_rate || 'N/A'}</span>
                              </div>
                            </div>
                            <div className="camera-chart">
                              <canvas ref={chartRef} width="1000" height="250"></canvas>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                    {cameras.length === 0 && (
                      <div style={{ padding: '3rem', textAlign: 'center', color: '#6b7280', background: 'white', borderRadius: '12px' }}>
                        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" style={{ margin: '0 auto 1rem', opacity: 0.3 }}>
                          <rect x="2" y="4" width="20" height="16" rx="2" stroke="currentColor" strokeWidth="2"/>
                          <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2"/>
                        </svg>
                        <p>No cameras available</p>
                        <p style={{ fontSize: '14px', marginTop: '0.5rem' }}>Add cameras in Camera Management to see statistics</p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Right Section */}
                <div className="dashboard-right">
                  {/* Recent Members */}
                  <div className="right-card recent-members">
                    <h3>Recent Visited Members</h3>
                    <div className="member-list">
                      <div className="member-item">
                        <img src="https://images.pexels.com/photos/30004320/pexels-photo-30004320.jpeg?auto=compress&cs=tinysrgb&h=350" alt="User" className="member-avatar" />
                        <div className="member-info">
                          <h4>Corey Rhiel Madsen</h4>
                          <span>Today, 4:23 AM</span>
                        </div>
                      </div>
                      <div className="member-item">
                        <img src="https://images.pexels.com/photos/30004322/pexels-photo-30004322.jpeg?auto=compress&cs=tinysrgb&h=350" alt="User" className="member-avatar" />
                        <div className="member-info">
                          <h4>Gretchen Calzoni</h4>
                          <span>Today, 8:30 AM</span>
                        </div>
                      </div>
                      <div className="member-item">
                        <img src="https://images.pexels.com/photos/30004493/pexels-photo-30004493.jpeg?auto=compress&cs=tinysrgb&h=350" alt="User" className="member-avatar" />
                        <div className="member-info">
                          <h4>Charlie George</h4>
                          <span>Today, 2:56 PM</span>
                        </div>
                      </div>
                    </div>
                    <a href="#" className="view-all">View all Members</a>
                  </div>

                  {/* Recent Products */}
                  <div className="right-card recent-products">
                    <h3>Recent Products</h3>
                    <div className="product-list">
                      {[
                        { id: 'A-1247', line: 'Production Line A', start: '09:15:23', end: '09:15:25', duration: '2.1s', success: true },
                        { id: 'B-0423', line: 'Production Line B', start: '09:14:18', end: '09:14:20', duration: '2.3s', success: true },
                        { id: 'Q-0368', line: 'Quality Control', start: '09:13:45', end: '09:13:47', duration: '2.6s', success: false },
                        { id: 'A-1246', line: 'Production Line A', start: '09:13:10', end: '09:13:12', duration: '2.0s', success: true },
                        { id: 'B-0422', line: 'Production Line B', start: '09:12:35', end: '09:12:37', duration: '2.2s', success: true },
                        { id: 'A-1245', line: 'Production Line A', start: '09:11:58', end: '09:12:00', duration: '2.0s', success: true },
                        { id: 'Q-0367', line: 'Quality Control', start: '09:11:20', end: '09:11:22', duration: '2.5s', success: false },
                        { id: 'B-0421', line: 'Production Line B', start: '09:10:45', end: '09:10:47', duration: '2.3s', success: true }
                      ].map((product, index) => (
                        <div key={index} className="product-item">
                          <div className="product-header">
                            <div className={`product-icon ${product.success ? 'success' : 'fail'}`}>
                              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                                {product.success ? (
                                  <polyline points="20,6 9,17 4,12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                                ) : (
                                  <>
                                    <line x1="18" y1="6" x2="6" y2="18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                                    <line x1="6" y1="6" x2="18" y2="18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                                  </>
                                )}
                              </svg>
                            </div>
                            <div className="product-info">
                              <h4>Product #{product.id}</h4>
                              <span className="product-line">{product.line}</span>
                            </div>
                          </div>
                          <div className="product-details">
                            <div className="detail-row">
                              <span className="detail-label">Start:</span>
                              <span className="detail-value">{product.start}</span>
                            </div>
                            <div className="detail-row">
                              <span className="detail-label">End:</span>
                              <span className="detail-value">{product.end}</span>
                            </div>
                            <div className="detail-row">
                              <span className="detail-label">Duration:</span>
                              <span className="detail-value">{product.duration}</span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </section>
          )}

          {currentSection === 'users' && <UserManagement />}
          {currentSection === 'receipts' && <Receipts />}
          {currentSection === 'cameras' && <CameraManagement />}
          {currentSection === 'historical' && <Historical />}
          {currentSection === 'settings' && <Settings />}
            </>
          )}
        </main>
      </div>

      {/* Confirmation Dialog */}
      <ConfirmDialog
        isOpen={confirmDialog.isOpen}
        onClose={() => setConfirmDialog({ ...confirmDialog, isOpen: false })}
        onConfirm={confirmDialog.onConfirm || (() => {})}
        title={confirmDialog.title}
        message={confirmDialog.message}
        type={confirmDialog.type}
      />
    </div>
  );
}
