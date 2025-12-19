import React, { useState, useEffect } from 'react';
import UserManagement from './UserManagement';
import Receipts from '../recipe/Receipts';
import Historical from './Historical';
import Settings from './Settings';
import CameraManagement from '../camera/CameraManagement';
import { camerasAPI } from '@/services/api';
import type { Camera } from '@/types';
import '@/styles/Dashboard.css';

interface DashboardProps {
  onLogout: () => void;
}

type Section = 'dashboard' | 'users' | 'receipts' | 'cameras' | 'historical' | 'settings';

interface CameraStats {
  total: number;
  connected: number;
  active: number;
}

const Dashboard: React.FC<DashboardProps> = ({ onLogout }) => {
  const [currentSection, setCurrentSection] = useState<Section>('dashboard');
  const [isLoading, setIsLoading] = useState(true);
  const [currentTime, setCurrentTime] = useState(new Date());
  const [totalProductsToday] = useState(1247);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [cameraStats, setCameraStats] = useState<CameraStats>({
    total: 0,
    connected: 0,
    active: 0
  });

  useEffect(() => {
    // Simulate loading
    const timer = setTimeout(() => setIsLoading(false), 1500);
    return () => clearTimeout(timer);
  }, [currentSection]);

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    fetchCameras();
  }, []);

  const fetchCameras = async () => {
    try {
      const data = await camerasAPI.getAllCameras();
      setCameras(data);
      setCameraStats({
        total: data.length,
        connected: data.filter(c => c.status === 'connected').length,
        active: data.filter(c => c.is_active).length
      });
    } catch (error) {
      console.error('Error fetching cameras:', error);
    }
  };

  const handleSectionChange = (section: Section) => {
    setIsLoading(true);
    setCurrentSection(section);
  };

  const renderContent = () => {
    if (isLoading) {
      return (
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading...</p>
        </div>
      );
    }

    switch (currentSection) {
      case 'users':
        return <UserManagement />;
      case 'receipts':
        return <Receipts />;
      case 'cameras':
        return <CameraManagement />;
      case 'historical':
        return <Historical />;
      case 'settings':
        return <Settings />;
      default:
        return (
          <div className="dashboard-main">
            <div className="dashboard-header">
              <div>
                <h1>Production Dashboard</h1>
                <p className="dashboard-time">{currentTime.toLocaleString()}</p>
              </div>
            </div>

            {/* Stats Cards */}
            <div className="stats-grid">
              <div className="stat-card">
                <div className="stat-icon">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                    <rect x="3" y="3" width="18" height="18" stroke="currentColor" strokeWidth="2"/>
                    <path d="M3 9h18M9 21V9" stroke="currentColor" strokeWidth="2"/>
                  </svg>
                </div>
                <div className="stat-content">
                  <div className="stat-label">Total Products Today</div>
                  <div className="stat-value">{totalProductsToday}</div>
                  <span className="stat-trend increase">+12% vs yesterday</span>
                </div>
              </div>

              <div className="stat-card">
                <div className="stat-icon success">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                    <rect x="2" y="6" width="20" height="12" rx="2" stroke="currentColor" strokeWidth="2"/>
                    <circle cx="12" cy="12" r="2.5" stroke="currentColor" strokeWidth="2"/>
                  </svg>
                </div>
                <div className="stat-content">
                  <div className="stat-label">Active Cameras</div>
                  <div className="stat-value">{cameraStats.active}/{cameraStats.total}</div>
                  <span className="stat-trend">{cameraStats.connected} connected</span>
                </div>
              </div>

              <div className="stat-card">
                <div className="stat-icon warning">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                    <path d="M22 12h-4l-3 9L9 3l-3 9H2" stroke="currentColor" strokeWidth="2"/>
                  </svg>
                </div>
                <div className="stat-content">
                  <div className="stat-label">Success Rate</div>
                  <div className="stat-value">98.5%</div>
                  <span className="stat-trend increase">+0.3% improvement</span>
                </div>
              </div>

              <div className="stat-card">
                <div className="stat-icon danger">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                    <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" stroke="currentColor" strokeWidth="2"/>
                    <line x1="12" y1="9" x2="12" y2="13" stroke="currentColor" strokeWidth="2"/>
                    <line x1="12" y1="17" x2="12.01" y2="17" stroke="currentColor" strokeWidth="2"/>
                  </svg>
                </div>
                <div className="stat-content">
                  <div className="stat-label">Rejected Items</div>
                  <div className="stat-value">19</div>
                  <span className="stat-trend">1.5% rejection rate</span>
                </div>
              </div>
            </div>

            {/* Recent Activity */}
            <div className="activity-section">
              <h3>Camera Status</h3>
              <div className="camera-list">
                {cameras.slice(0, 3).map((camera) => (
                  <div key={camera.id} className="camera-card">
                    <div className="camera-info">
                      <h4>{camera.name}</h4>
                      <p>{camera.model || 'Unknown Model'}</p>
                    </div>
                    <div className="camera-status">
                      <span className={`status-badge ${camera.status}`}>
                        {camera.status}
                      </span>
                      <span className={`status-badge ${camera.is_active ? 'success' : 'inactive'}`}>
                        {camera.is_active ? 'Active' : 'Inactive'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        );
    }
  };

  return (
    <div className="dashboard-container">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <h2>Suntech Vision</h2>
        </div>

        <nav className="sidebar-nav">
          <button
            className={`nav-item ${currentSection === 'dashboard' ? 'active' : ''}`}
            onClick={() => handleSectionChange('dashboard')}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <rect x="3" y="3" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
              <rect x="14" y="3" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
              <rect x="14" y="14" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
              <rect x="3" y="14" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
            </svg>
            Dashboard
          </button>

          <button
            className={`nav-item ${currentSection === 'cameras' ? 'active' : ''}`}
            onClick={() => handleSectionChange('cameras')}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <rect x="2" y="6" width="20" height="12" rx="2" stroke="currentColor" strokeWidth="2"/>
              <circle cx="12" cy="12" r="2.5" stroke="currentColor" strokeWidth="2"/>
            </svg>
            Cameras
          </button>

          <button
            className={`nav-item ${currentSection === 'receipts' ? 'active' : ''}`}
            onClick={() => handleSectionChange('receipts')}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" stroke="currentColor" strokeWidth="2"/>
              <polyline points="14,2 14,8 20,8" stroke="currentColor" strokeWidth="2"/>
              <line x1="16" y1="13" x2="8" y2="13" stroke="currentColor" strokeWidth="2"/>
              <line x1="16" y1="17" x2="8" y2="17" stroke="currentColor" strokeWidth="2"/>
              <polyline points="10,9 9,9 8,9" stroke="currentColor" strokeWidth="2"/>
            </svg>
            Recipes
          </button>

          <button
            className={`nav-item ${currentSection === 'historical' ? 'active' : ''}`}
            onClick={() => handleSectionChange('historical')}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <polyline points="22,12 18,12 15,21 9,3 6,12 2,12" stroke="currentColor" strokeWidth="2"/>
            </svg>
            Historical
          </button>

          <button
            className={`nav-item ${currentSection === 'users' ? 'active' : ''}`}
            onClick={() => handleSectionChange('users')}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2" stroke="currentColor" strokeWidth="2"/>
              <circle cx="12" cy="7" r="4" stroke="currentColor" strokeWidth="2"/>
            </svg>
            Users
          </button>

          <button
            className={`nav-item ${currentSection === 'settings' ? 'active' : ''}`}
            onClick={() => handleSectionChange('settings')}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2"/>
              <path d="M12 1v6m0 6v6M1 12h6m6 0h6" stroke="currentColor" strokeWidth="2"/>
            </svg>
            Settings
          </button>
        </nav>

        <div className="sidebar-footer">
          <button className="logout-btn" onClick={onLogout}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4" stroke="currentColor" strokeWidth="2"/>
              <polyline points="16,17 21,12 16,7" stroke="currentColor" strokeWidth="2"/>
              <line x1="21" y1="12" x2="9" y2="12" stroke="currentColor" strokeWidth="2"/>
            </svg>
            Logout
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        {renderContent()}
      </main>
    </div>
  );
};

export default Dashboard;
