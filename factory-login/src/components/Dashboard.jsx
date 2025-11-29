import React, { useState, useEffect, useRef } from 'react';
import './Dashboard.css';

export default function Dashboard({ onLogout }) {
  const [currentSection, setCurrentSection] = useState('dashboard');
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem('darkMode') === 'enabled';
  });
  const [currentTime, setCurrentTime] = useState(new Date());
  const [totalProductsToday, setTotalProductsToday] = useState(1247);

  const camera1ChartRef = useRef(null);
  const camera2ChartRef = useRef(null);
  const camera3ChartRef = useRef(null);

  const [camera1Data, setCamera1Data] = useState([45, 52, 48, 65, 58, 72, 68, 75, 82, 78, 85, 92, 88, 95, 91, 98, 102, 96, 105, 110, 108, 115, 112, 120]);
  const [camera2Data, setCamera2Data] = useState([38, 42, 45, 52, 48, 58, 62, 68, 65, 72, 75, 80, 76, 83, 79, 86, 90, 85, 92, 95, 98, 102, 99, 105]);
  const [camera3Data, setCamera3Data] = useState([32, 35, 38, 42, 45, 48, 52, 55, 58, 62, 65, 68, 64, 70, 67, 73, 76, 72, 78, 82, 79, 85, 88, 90]);

  const hours = ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'];

  // Dark mode effect
  useEffect(() => {
    if (darkMode) {
      document.body.classList.add('dark-mode');
      localStorage.setItem('darkMode', 'enabled');
    } else {
      document.body.classList.remove('dark-mode');
      localStorage.setItem('darkMode', 'disabled');
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

  const drawLineChart = (canvas, data, color) => {
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
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
      ctx.fillText(Math.round(value), padding - 10, y);
    }
  };

  const handleLogout = () => {
    if (window.confirm('Are you sure you want to logout?')) {
      document.body.classList.remove('dark-mode');
      onLogout();
    }
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
              <circle cx="12" cy="12" r="2" fill="#10b981"/>
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z" stroke="currentColor" strokeWidth="2"/>
            </svg>
            <span>All cameras online</span>
          </div>
          <div className="profile">
            <img src="https://images.pexels.com/photos/30004493/pexels-photo-30004493.jpeg?auto=compress&cs=tinysrgb&h=350" alt="User" className="profile-avatar" />
            <span>Admin User</span>
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
              onClick={(e) => { e.preventDefault(); setCurrentSection('dashboard'); }}
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
              onClick={(e) => { e.preventDefault(); setCurrentSection('users'); }}
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
              onClick={(e) => { e.preventDefault(); setCurrentSection('receipts'); }}
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
              href="#historical"
              className={`nav-item ${currentSection === 'historical' ? 'active' : ''}`}
              onClick={(e) => { e.preventDefault(); setCurrentSection('historical'); }}
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
              onClick={(e) => { e.preventDefault(); setCurrentSection('settings'); }}
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
                    <div className="card-value">3</div>
                    <span className="card-status online">All online</span>
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
                    {[
                      { id: 1, name: 'Camera 1 - Line A', fps: 30, products: 456, image: 'https://images.pexels.com/photos/1267338/pexels-photo-1267338.jpeg?auto=compress&cs=tinysrgb&w=800' },
                      { id: 2, name: 'Camera 2 - Line B', fps: 30, products: 423, image: 'https://images.pexels.com/photos/3862130/pexels-photo-3862130.jpeg?auto=compress&cs=tinysrgb&w=800' },
                      { id: 3, name: 'Camera 3 - QC', fps: 30, products: 368, image: 'https://images.pexels.com/photos/5022849/pexels-photo-5022849.jpeg?auto=compress&cs=tinysrgb&w=800' }
                    ].map(camera => (
                      <div key={camera.id} className="camera-preview-card">
                        <div className="camera-preview-header">
                          <div className="camera-preview-title">
                            <div className="camera-status active"></div>
                            <span>{camera.name}</span>
                          </div>
                          <span className="live-badge">LIVE</span>
                        </div>
                        <div className="camera-preview-frame">
                          <img src={camera.image} alt={`${camera.name} Feed`} />
                          <div className="camera-overlay-info">
                            <div className="overlay-stat">
                              <span className="overlay-label">FPS:</span>
                              <span className="overlay-value">{camera.fps}</span>
                            </div>
                            <div className="overlay-stat">
                              <span className="overlay-label">Products:</span>
                              <span className="overlay-value">{camera.products}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Center Section - Camera Statistics Charts */}
                <div className="dashboard-center">
                  <div className="camera-stats">
                    {[
                      { id: 1, name: 'Camera 1 - Production Line A', chartRef: camera1ChartRef, processed: 456, success: 452, failed: 4, avgTime: '2.1s' },
                      { id: 2, name: 'Camera 2 - Production Line B', chartRef: camera2ChartRef, processed: 423, success: 417, failed: 6, avgTime: '2.3s' },
                      { id: 3, name: 'Camera 3 - Quality Control', chartRef: camera3ChartRef, processed: 368, success: 360, failed: 8, avgTime: '2.6s' }
                    ].map(camera => (
                      <div key={camera.id} className="camera-card">
                        <div className="camera-card-header">
                          <div className="camera-title">
                            <div className="camera-status active"></div>
                            <h3>{camera.name}</h3>
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
                              <span className="metric-label">Products Processed</span>
                              <span className="metric-value">{camera.processed}</span>
                            </div>
                            <div className="metric success">
                              <span className="metric-label">Success</span>
                              <span className="metric-value">{camera.success}</span>
                            </div>
                            <div className="metric fail">
                              <span className="metric-label">Failed</span>
                              <span className="metric-value">{camera.failed}</span>
                            </div>
                            <div className="metric">
                              <span className="metric-label">Avg Time</span>
                              <span className="metric-value">{camera.avgTime}</span>
                            </div>
                          </div>
                          <div className="camera-chart">
                            <canvas ref={camera.chartRef} width="1000" height="250"></canvas>
                          </div>
                        </div>
                      </div>
                    ))}
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

          {currentSection !== 'dashboard' && (
            <section className="content-section active">
              <div className="section-header">
                <h1>
                  {currentSection === 'users' && 'User Management'}
                  {currentSection === 'receipts' && 'Receipts'}
                  {currentSection === 'historical' && 'Historical Data'}
                  {currentSection === 'settings' && 'Settings'}
                </h1>
                {currentSection !== 'settings' && (
                  <button className="dashboard-btn">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                      <line x1="12" y1="5" x2="12" y2="19" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                      <line x1="5" y1="12" x2="19" y2="12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                    </svg>
                    {currentSection === 'users' && 'Add User'}
                    {currentSection === 'receipts' && 'Create Receipt'}
                    {currentSection === 'historical' && 'Export Data'}
                  </button>
                )}
              </div>
              <div className="placeholder-content">
                <svg width="64" height="64" viewBox="0 0 24 24" fill="none">
                  {currentSection === 'users' && (
                    <path d="M17 21V19C17 17.9391 16.5786 16.9217 15.8284 16.1716C15.0783 15.4214 14.0609 15 13 15H5C3.93913 15 2.92172 15.4214 2.17157 16.1716C1.42143 16.9217 1 17.9391 1 19V21" stroke="currentColor" strokeWidth="2"/>
                  )}
                  {currentSection === 'receipts' && (
                    <path d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z" stroke="currentColor" strokeWidth="2"/>
                  )}
                  {currentSection === 'historical' && (
                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
                  )}
                  {currentSection === 'settings' && (
                    <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2"/>
                  )}
                </svg>
                <h2>
                  {currentSection === 'users' && 'User Management'}
                  {currentSection === 'receipts' && 'Receipts Management'}
                  {currentSection === 'historical' && 'Historical Records'}
                  {currentSection === 'settings' && 'System Settings'}
                </h2>
                <p>
                  {currentSection === 'users' && 'Manage system users, roles, and permissions'}
                  {currentSection === 'receipts' && 'Create, view, edit, and delete production receipts'}
                  {currentSection === 'historical' && 'View and analyze historical production data and trends'}
                  {currentSection === 'settings' && 'Configure system preferences and camera settings'}
                </p>
              </div>
            </section>
          )}
        </main>
      </div>
    </div>
  );
}
