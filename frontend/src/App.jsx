import React, { useState, useEffect } from 'react';
import Dashboard from './components/Dashboard';
import { authAPI } from './services/api';
import './App.css';

export default function SuntechAutomation() {
  const [currentSession, setCurrentSession] = useState(1);
  const [isLightMode, setIsLightMode] = useState(() => {
    const savedMode = localStorage.getItem('appThemeMode');
    return savedMode ? savedMode === 'light' : true;
  });
  const [isLoggedIn, setIsLoggedIn] = useState(() => {
    // Check if user is already logged in
    return !!localStorage.getItem('access_token');
  });
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loginError, setLoginError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showLoginLoading, setShowLoginLoading] = useState(false);
  const totalSessions = 3;

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentSession(prev => (prev % totalSessions) + 1);
    }, 15000); // 15 seconds - matches animation duration

    return () => clearInterval(interval);
  }, []);

  const getProductImage = (productNumber) => {
    return `product${currentSession}/${productNumber}.png`;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoginError('');
    setIsLoading(true);

    try {
      const response = await authAPI.login(username, password);

      // Save token and user info
      localStorage.setItem('access_token', response.access_token);
      localStorage.setItem('user', JSON.stringify(response.user));

      console.log('Login successful:', response.user);
      
      // Show login loading screen for 2 seconds
      setIsLoading(false);
      setShowLoginLoading(true);
      
      setTimeout(() => {
        setShowLoginLoading(false);
        setIsLoggedIn(true);
      }, 2000);
    } catch (error) {
      console.error('Login error:', error);
      const errorMessage = error.response?.data?.detail || 'Login failed. Please check your credentials.';
      setLoginError(errorMessage);
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    console.log('Logging out...');

    // Clear auth data
    localStorage.removeItem('access_token');
    localStorage.removeItem('user');

    // Reload theme from localStorage when logging out
    const savedMode = localStorage.getItem('appThemeMode');
    setIsLightMode(savedMode ? savedMode === 'light' : true);

    // Reset form
    setUsername('');
    setPassword('');
    setLoginError('');

    setIsLoggedIn(false);
  };

  // Show Dashboard if logged in
  if (isLoggedIn) {
    return <Dashboard onLogout={handleLogout} />;
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

  return (
    <>
      <div className={isLightMode ? 'bg-gradient-to-br from-gray-100 via-blue-50 to-gray-200' : 'grid-bg'}>
        <div className="relative w-full flex items-center justify-center overflow-hidden" style={{ height: '1090px' }}>
          
          {/* Header */}
          <div className="absolute top-0 left-0 right-0 z-50 px-8 py-6 flex justify-between items-center">
            <div className="flex items-center gap-6">
              <div className="company-logo">
                <img src="logo.png" alt="Suntech Automation" className="h-16 w-auto" />
              </div>

              <div className={`border-l-2 pl-6 ${isLightMode ? 'border-gray-400' : 'border-slate-600'}`}>
                <h1 className={`orbitron text-2xl font-black tracking-wider ${isLightMode ? 'text-gray-800' : 'text-white'}`}>SUNTECH AUTOMATION</h1>
                <p className={`text-sm tracking-widest font-medium ${isLightMode ? 'text-blue-600' : 'text-cyan-400'}`}>v1.0.0 PRODUCTION</p>
              </div>
            </div>

            <div className="flex items-center gap-6">
              <div className={`flex items-center gap-3 px-5 py-2.5 rounded-xl border backdrop-blur-sm ${
                isLightMode
                  ? 'bg-white/80 border-gray-300'
                  : 'bg-slate-800/60 border-slate-600/50'
              }`}>
                <div className="w-2.5 h-2.5 bg-green-400 rounded-full pulse-green"></div>
                <span className="text-green-400 font-semibold text-sm tracking-wide">SYSTEM ONLINE</span>
              </div>
              <div className="text-right">
                <p className={`text-xs font-medium tracking-wider ${isLightMode ? 'text-gray-500' : 'text-gray-400'}`}>CURRENT SHIFT</p>
                <p className={`orbitron text-lg font-bold ${isLightMode ? 'text-blue-600' : 'text-cyan-400'}`}>A-MORNING</p>
              </div>
            </div>
          </div>
          
          {/* Left Machine Vision Camera */}
          <div className="absolute left-20 top-1/2 -translate-y-1/2 z-30 camera-left" style={{ transformOrigin: 'center right' }}>
            <div className="relative">
              <div className="relative flex items-center">
                <div className={`w-20 h-24 rounded-lg border-2 shadow-2xl relative ${
                  isLightMode
                    ? 'bg-gradient-to-b from-gray-300 via-gray-400 to-gray-500 border-gray-400'
                    : 'bg-gradient-to-b from-slate-700 via-slate-800 to-slate-900 border-slate-600'
                }`}>
                  <div className={`absolute inset-1.5 rounded border ${
                    isLightMode ? 'bg-gray-200 border-gray-300' : 'bg-slate-900 border-slate-700'
                  }`}></div>
                  <div className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-14 h-14 rounded border-3 flex items-center justify-center glow-cyan ${
                    isLightMode
                      ? 'bg-gradient-to-br from-gray-100 to-gray-200 border-blue-500'
                      : 'bg-gradient-to-br from-slate-900 to-black border-cyan-500'
                  }`}>
                    <div className={`w-10 h-10 bg-gradient-to-br rounded-sm lens-rotate opacity-70 flex items-center justify-center ${
                      isLightMode ? 'from-blue-500 to-blue-600' : 'from-cyan-600 to-blue-500'
                    }`}>
                      <div className={`w-5 h-5 rounded-sm opacity-60 ${
                        isLightMode ? 'bg-blue-400' : 'bg-cyan-400'
                      }`}></div>
                    </div>
                  </div>
                  <div className="absolute bottom-2 left-1/2 -translate-x-1/2 flex gap-1">
                    <div className={`w-1 h-1 rounded-full pulse-green ${
                      isLightMode ? 'bg-blue-500' : 'bg-cyan-500'
                    }`}></div>
                    <div className={`w-1 h-1 rounded-full pulse-green ${
                      isLightMode ? 'bg-blue-500' : 'bg-cyan-500'
                    }`} style={{ animationDelay: '0.2s' }}></div>
                    <div className={`w-1 h-1 rounded-full pulse-green ${
                      isLightMode ? 'bg-blue-500' : 'bg-cyan-500'
                    }`} style={{ animationDelay: '0.4s' }}></div>
                  </div>
                  <div className="absolute right-0.5 top-3 flex flex-col gap-0.5">
                    <div className={`w-0.5 h-0.5 ${isLightMode ? 'bg-gray-400' : 'bg-slate-600'}`}></div>
                    <div className={`w-0.5 h-0.5 ${isLightMode ? 'bg-gray-400' : 'bg-slate-600'}`}></div>
                    <div className={`w-0.5 h-0.5 ${isLightMode ? 'bg-gray-400' : 'bg-slate-600'}`}></div>
                    <div className={`w-0.5 h-0.5 ${isLightMode ? 'bg-gray-400' : 'bg-slate-600'}`}></div>
                  </div>
                  <div className="absolute left-0.5 top-3 flex flex-col gap-0.5">
                    <div className={`w-0.5 h-0.5 ${isLightMode ? 'bg-gray-400' : 'bg-slate-600'}`}></div>
                    <div className={`w-0.5 h-0.5 ${isLightMode ? 'bg-gray-400' : 'bg-slate-600'}`}></div>
                    <div className={`w-0.5 h-0.5 ${isLightMode ? 'bg-gray-400' : 'bg-slate-600'}`}></div>
                    <div className={`w-0.5 h-0.5 ${isLightMode ? 'bg-gray-400' : 'bg-slate-600'}`}></div>
                  </div>
                </div>
                <div className={`absolute -left-4 top-1/2 -translate-y-1/2 w-6 h-28 rounded-l-lg border-2 flex flex-col justify-center items-center gap-2 ${
                  isLightMode
                    ? 'bg-gray-400 border-gray-500'
                    : 'bg-slate-700 border-slate-600'
                }`}>
                  <div className={`w-2 h-2 rounded-full ${isLightMode ? 'bg-gray-500' : 'bg-slate-800'}`}></div>
                  <div className={`w-2 h-2 rounded-full ${isLightMode ? 'bg-gray-500' : 'bg-slate-800'}`}></div>
                </div>
                <div className={`absolute -left-2 top-2 w-1 h-3 rounded ${
                  isLightMode ? 'bg-gray-500' : 'bg-slate-600'
                }`}></div>
              </div>
              <div className="absolute left-full top-1/2 -translate-y-1/2 w-[200px] pointer-events-none scanner-beam-left" style={{ height: '224px' }}>
                <div className="w-full h-full bg-gradient-to-r from-cyan-500/40 via-blue-500/30 to-transparent" style={{ transform: 'skewY(-15deg)', transformOrigin: 'left center' }}></div>
                <div className="absolute top-0 left-0 w-1 h-full bg-cyan-400/60 shadow-lg shadow-cyan-400/50" style={{ transform: 'skewY(-15deg)', transformOrigin: 'left top' }}></div>
                <div className="absolute bottom-0 left-0 w-1 h-full bg-cyan-400/60 shadow-lg shadow-cyan-400/50" style={{ transform: 'skewY(-15deg)', transformOrigin: 'left bottom' }}></div>
                <div className="absolute top-1/4 left-0 w-full h-px bg-cyan-400/40" style={{ transform: 'skewY(-15deg)' }}></div>
                <div className="absolute top-2/4 left-0 w-full h-px bg-cyan-400/30" style={{ transform: 'skewY(-15deg)' }}></div>
                <div className="absolute top-3/4 left-0 w-full h-px bg-cyan-400/20" style={{ transform: 'skewY(-15deg)' }}></div>
              </div>
              <div className={`absolute -bottom-16 left-0 whitespace-nowrap backdrop-blur-sm px-3 py-1.5 rounded-lg border ${
                isLightMode
                  ? 'bg-white/80 border-blue-400/50'
                  : 'bg-slate-800/80 border-cyan-400/50'
              }`}>
                <p className={`orbitron text-xs font-bold tracking-wider ${
                  isLightMode ? 'text-blue-600' : 'text-cyan-400'
                }`}>CAM-1 VISION</p>
              </div>
            </div>
          </div>
          
          {/* Right Rejection System (Pusher) */}
          <div className="absolute right-20 top-1/2 -translate-y-1/2 z-30 camera-right" style={{ transformOrigin: 'center left' }}>
            <div className="relative">
              <div className="relative flex items-center">
                <div className={`w-24 h-32 rounded-lg border-2 shadow-2xl relative ${
                  isLightMode
                    ? 'bg-gradient-to-b from-gray-300 via-gray-400 to-gray-500 border-gray-400'
                    : 'bg-gradient-to-b from-slate-700 via-slate-800 to-slate-900 border-slate-600'
                }`}>
                  <div className={`absolute inset-2 rounded border ${
                    isLightMode ? 'bg-gray-200 border-gray-300' : 'bg-slate-900 border-slate-700'
                  }`}></div>
                  <div className={`absolute top-3 left-2 right-2 h-1 bg-gradient-to-r from-transparent to-transparent opacity-70 ${
                    isLightMode ? 'via-blue-500' : 'via-yellow-500'
                  }`}></div>
                  <div className={`absolute bottom-3 left-2 right-2 h-1 bg-gradient-to-r from-transparent to-transparent opacity-70 ${
                    isLightMode ? 'via-blue-500' : 'via-yellow-500'
                  }`}></div>
                  <div className={`absolute top-1/2 -translate-y-1/2 -left-2 w-3 h-16 rounded-l-lg border-2 ${
                    isLightMode ? 'bg-gray-300 border-gray-400' : 'bg-slate-950 border-slate-700'
                  }`}></div>
                  <div className="absolute top-2 right-2 flex flex-col gap-1.5">
                    <div className="w-2 h-2 bg-red-500 rounded-full pulse-green reject-led-alert"></div>
                    <div className="w-2 h-2 bg-yellow-500 rounded-full pulse-green" style={{ animationDelay: '0.3s' }}></div>
                    <div className="w-2 h-2 bg-green-500 rounded-full pulse-green" style={{ animationDelay: '0.6s' }}></div>
                  </div>
                  <div className={`absolute bottom-2 left-1/2 -translate-x-1/2 px-2 py-0.5 rounded ${
                    isLightMode ? 'bg-blue-500/20' : 'bg-yellow-500/20'
                  }`}>
                    <span className={`text-[8px] font-bold ${
                      isLightMode ? 'text-blue-600' : 'text-yellow-400'
                    }`}>REJECT</span>
                  </div>
                </div>
                <div className="pusher-arm absolute left-0 top-1/2 -translate-y-1/2 w-[120px] h-8 pointer-events-none" style={{ transformOrigin: 'right center' }}>
                  <div className={`absolute right-0 top-0 bottom-0 w-full border-2 rounded-l-lg shadow-lg ${
                    isLightMode
                      ? 'bg-gradient-to-l from-gray-500 via-gray-400 to-gray-500 border-gray-600'
                      : 'bg-gradient-to-l from-orange-600 via-orange-500 to-orange-600 border-orange-700'
                  }`}>
                    <div className={`absolute left-1 top-1/2 -translate-y-1/2 w-2 h-2 rounded-full ${
                      isLightMode ? 'bg-gray-600' : 'bg-orange-800'
                    }`}></div>
                    <div className={`absolute left-1/3 top-1/2 -translate-y-1/2 w-1 h-1 rounded-full ${
                      isLightMode ? 'bg-gray-600' : 'bg-orange-800'
                    }`}></div>
                    <div className={`absolute left-2/3 top-1/2 -translate-y-1/2 w-1 h-1 rounded-full ${
                      isLightMode ? 'bg-gray-600' : 'bg-orange-800'
                    }`}></div>
                    <div className={`absolute -left-1 top-1/2 -translate-y-1/2 w-3 h-10 border-2 rounded-l ${
                      isLightMode ? 'bg-gray-500 border-gray-600' : 'bg-orange-700 border-orange-800'
                    }`}></div>
                  </div>
                </div>
                <div className={`absolute -right-4 top-1/2 -translate-y-1/2 w-6 h-36 rounded-r-lg border-2 flex flex-col justify-center items-center gap-2 ${
                  isLightMode
                    ? 'bg-gray-400 border-gray-500'
                    : 'bg-slate-700 border-slate-600'
                }`}>
                  <div className={`w-2 h-2 rounded-full ${isLightMode ? 'bg-gray-500' : 'bg-slate-800'}`}></div>
                  <div className={`w-2 h-2 rounded-full ${isLightMode ? 'bg-gray-500' : 'bg-slate-800'}`}></div>
                  <div className={`w-2 h-2 rounded-full ${isLightMode ? 'bg-gray-500' : 'bg-slate-800'}`}></div>
                </div>
                <div className={`absolute -right-2 top-2 w-1 h-4 rounded ${
                  isLightMode ? 'bg-gray-500' : 'bg-slate-600'
                }`}></div>
              </div>
              <div className="absolute right-full top-1/2 -translate-y-1/2 w-[150px] pointer-events-none detection-zone-alert" style={{ height: '224px' }}>
                <div className="w-full h-full bg-gradient-to-l from-orange-500/20 via-red-500/15 to-transparent"></div>
                <div className="absolute top-0 right-0 w-1 h-full bg-red-400/40 warning-line-pulse"></div>
                <div className="absolute bottom-0 right-0 w-1 h-full bg-red-400/40 warning-line-pulse"></div>
                <div className="absolute top-1/4 left-0 w-full h-px bg-red-400/30 warning-line-pulse"></div>
                <div className="absolute top-2/4 left-0 w-full h-px bg-red-400/40 warning-line-pulse"></div>
                <div className="absolute top-3/4 left-0 w-full h-px bg-red-400/30 warning-line-pulse"></div>
              </div>
              <div className={`absolute -bottom-20 right-0 whitespace-nowrap backdrop-blur-sm px-3 py-1.5 rounded-lg border ${
                isLightMode
                  ? 'bg-white/80 border-gray-400/50'
                  : 'bg-slate-800/80 border-orange-400/50'
              }`}>
                <p className={`orbitron text-xs font-bold tracking-wider ${
                  isLightMode ? 'text-gray-700' : 'text-orange-400'
                }`}>REJECTION SYSTEM</p>
              </div>
            </div>
          </div>
          
          {/* Center Machine Vision Camera */}
          <div className="absolute left-1/2 top-2 z-40 camera-center">
            <div className="relative">
              <div className="relative">
                <div className={`w-28 h-32 rounded-lg border-2 shadow-2xl relative ${
                  isLightMode
                    ? 'bg-gradient-to-b from-gray-300 via-gray-400 to-gray-500 border-gray-400'
                    : 'bg-gradient-to-b from-slate-700 via-slate-800 to-slate-900 border-slate-600'
                }`}>
                  <div className={`absolute inset-2 rounded border ${
                    isLightMode ? 'bg-gray-200 border-gray-300' : 'bg-slate-900 border-slate-700'
                  }`}></div>
                  <div className={`absolute top-6 left-1/2 -translate-x-1/2 w-16 h-16 rounded-lg border-3 flex items-center justify-center glow-cyan ${
                    isLightMode
                      ? 'bg-gradient-to-br from-gray-100 to-gray-200 border-blue-500'
                      : 'bg-gradient-to-br from-slate-900 to-black border-emerald-500'
                  }`}>
                    <div className={`w-12 h-12 bg-gradient-to-br rounded-full lens-rotate opacity-70 flex items-center justify-center ${
                      isLightMode ? 'from-blue-500 to-blue-600' : 'from-emerald-600 to-emerald-400'
                    }`} style={{ animationDelay: '-2s' }}>
                      <div className={`w-6 h-6 rounded-full opacity-60 ${
                        isLightMode ? 'bg-blue-400' : 'bg-emerald-300'
                      }`}></div>
                    </div>
                  </div>
                  <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-1.5">
                    <div className={`w-1.5 h-1.5 rounded-full pulse-green ${
                      isLightMode ? 'bg-blue-500' : 'bg-emerald-500'
                    }`}></div>
                    <div className={`w-1.5 h-1.5 rounded-full pulse-green ${
                      isLightMode ? 'bg-blue-500' : 'bg-emerald-500'
                    }`} style={{ animationDelay: '0.2s' }}></div>
                    <div className={`w-1.5 h-1.5 rounded-full pulse-green ${
                      isLightMode ? 'bg-blue-500' : 'bg-emerald-500'
                    }`} style={{ animationDelay: '0.4s' }}></div>
                  </div>
                  <div className="absolute right-1 top-1/2 -translate-y-1/2 flex flex-col gap-1">
                    <div className={`w-1 h-0.5 ${isLightMode ? 'bg-gray-400' : 'bg-slate-600'}`}></div>
                    <div className={`w-1 h-0.5 ${isLightMode ? 'bg-gray-400' : 'bg-slate-600'}`}></div>
                    <div className={`w-1 h-0.5 ${isLightMode ? 'bg-gray-400' : 'bg-slate-600'}`}></div>
                  </div>
                  <div className="absolute left-1 top-1/2 -translate-y-1/2 flex flex-col gap-1">
                    <div className={`w-1 h-0.5 ${isLightMode ? 'bg-gray-400' : 'bg-slate-600'}`}></div>
                    <div className={`w-1 h-0.5 ${isLightMode ? 'bg-gray-400' : 'bg-slate-600'}`}></div>
                    <div className={`w-1 h-0.5 ${isLightMode ? 'bg-gray-400' : 'bg-slate-600'}`}></div>
                  </div>
                </div>
                <div className="absolute -top-12 left-1/2 -translate-x-1/2 flex flex-col items-center">
                  <div className={`w-3 h-10 rounded-t-lg border-2 ${
                    isLightMode ? 'bg-gray-400 border-gray-500' : 'bg-slate-700 border-slate-600'
                  }`}></div>
                  <div className={`w-12 h-4 rounded-lg border-2 ${
                    isLightMode ? 'bg-gray-500 border-gray-600' : 'bg-slate-800 border-slate-600'
                  }`}></div>
                </div>
                <div className={`absolute -top-10 left-1/2 -translate-x-1/2 translate-x-4 w-1.5 h-8 rounded-full ${
                  isLightMode ? 'bg-gray-500' : 'bg-slate-600'
                }`}></div>
              </div>
              <div className="absolute top-full left-1/2 -translate-x-1/2 w-[350px] pointer-events-none scanner-beam-center">
                <div className="w-full h-full bg-gradient-to-b from-emerald-500/40 via-emerald-400/20 to-transparent"></div>
                <div className="absolute top-0 left-0 right-0 h-1 bg-emerald-400/80 shadow-lg shadow-emerald-400/50 blur-sm"></div>
                <div className="absolute top-4 left-1/4 right-1/4 h-px bg-emerald-400/60"></div>
                <div className="absolute top-8 left-1/3 right-1/3 h-px bg-emerald-400/40"></div>
                <div className="absolute top-12 left-0 right-0 h-px bg-emerald-400/30"></div>
                <div className="absolute top-0 left-0 w-px h-20 bg-emerald-400/50"></div>
                <div className="absolute top-0 right-0 w-px h-20 bg-emerald-400/50"></div>
              </div>
              <div className={`absolute -right-36 top-1/2 -translate-y-1/2 whitespace-nowrap backdrop-blur-sm px-4 py-2 rounded-lg border ${
                isLightMode
                  ? 'bg-white/80 border-blue-400/50'
                  : 'bg-slate-800/80 border-emerald-400/50'
              }`}>
                <p className={`orbitron text-xs font-bold tracking-wider ${
                  isLightMode ? 'text-blue-600' : 'text-emerald-400'
                }`}>CAM-3 VISION</p>
              </div>
            </div>
          </div>
          
          {/* Rejection Conveyor Belt */}
          <div className="absolute bottom-24 right-24 z-5">
            <div className="relative w-64 h-24">
              <div className={`absolute inset-0 rounded-lg border-2 shadow-2xl ${
                isLightMode
                  ? 'bg-gradient-to-b from-gray-400 to-gray-500 border-gray-500'
                  : 'bg-gradient-to-b from-slate-700 to-slate-900 border-orange-500'
              }`}>
                <div className={`absolute inset-2 rounded overflow-hidden ${
                  isLightMode
                    ? 'bg-gradient-to-b from-gray-500 via-gray-600 to-gray-500'
                    : 'bg-gradient-to-b from-slate-800 via-slate-900 to-slate-800'
                }`}>
                  <div className={`absolute top-1/2 -translate-y-1/2 w-full h-1 ${
                    isLightMode ? 'bg-gray-400' : 'bg-slate-600'
                  }`}></div>
                  <div className={`absolute top-1/3 w-full h-px ${
                    isLightMode ? 'bg-gray-400/50' : 'bg-slate-500/50'
                  }`}></div>
                  <div className={`absolute top-2/3 w-full h-px ${
                    isLightMode ? 'bg-gray-400/50' : 'bg-slate-500/50'
                  }`}></div>
                  <div className="absolute inset-0 flex items-center justify-around" style={{ animation: 'rejectBeltMove 2s linear infinite' }}>
                    <div className={`w-1 h-full ${isLightMode ? 'bg-gray-400/50' : 'bg-slate-600/50'}`}></div>
                    <div className={`w-1 h-full ${isLightMode ? 'bg-gray-400/50' : 'bg-slate-600/50'}`}></div>
                    <div className={`w-1 h-full ${isLightMode ? 'bg-gray-400/50' : 'bg-slate-600/50'}`}></div>
                    <div className={`w-1 h-full ${isLightMode ? 'bg-gray-400/50' : 'bg-slate-600/50'}`}></div>
                    <div className={`w-1 h-full ${isLightMode ? 'bg-gray-400/50' : 'bg-slate-600/50'}`}></div>
                  </div>
                </div>
                <div className={`absolute -top-1 left-0 right-0 h-1 bg-gradient-to-r ${
                  isLightMode
                    ? 'from-blue-400 via-blue-500 to-blue-400'
                    : 'from-yellow-500 via-orange-500 to-yellow-500'
                }`}></div>
                <div className={`absolute -bottom-1 left-0 right-0 h-1 bg-gradient-to-r ${
                  isLightMode
                    ? 'from-blue-400 via-blue-500 to-blue-400'
                    : 'from-yellow-500 via-orange-500 to-yellow-500'
                }`}></div>
                <div className="absolute top-1/2 -translate-y-1/2 right-2">
                  <svg className={`w-6 h-6 ${isLightMode ? 'text-gray-600' : 'text-orange-400'}`} fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd"/>
                  </svg>
                </div>
                <div className={`absolute -left-2 top-1/2 -translate-y-1/2 w-4 h-20 rounded-l border-2 ${
                  isLightMode ? 'bg-gray-400 border-gray-500' : 'bg-slate-700 border-slate-600'
                }`}></div>
                <div className={`absolute -right-2 top-1/2 -translate-y-1/2 w-4 h-20 rounded-r border-2 ${
                  isLightMode ? 'bg-gray-400 border-gray-500' : 'bg-slate-700 border-slate-600'
                }`}></div>
              </div>
              <div className={`absolute -top-8 left-0 backdrop-blur-sm px-3 py-1 rounded-lg border ${
                isLightMode
                  ? 'bg-white/80 border-gray-400/50'
                  : 'bg-slate-800/80 border-orange-400/50'
              }`}>
                <p className={`text-xs font-bold ${
                  isLightMode ? 'text-gray-700' : 'text-orange-400'
                }`}>REJECT CONVEYOR</p>
              </div>
            </div>
          </div>

          {/* Conveyor Belt */}
          <div
            className="absolute top-1/2 -translate-y-1/2 left-0 w-full h-56 conveyor z-10"
            style={isLightMode ? {
              background: 'linear-gradient(180deg, #9ca3af 0%, #6b7280 50%, #9ca3af 100%)',
              borderTop: '3px solid #4b5563',
              borderBottom: '3px solid #374151',
              boxShadow: 'inset 0 10px 20px rgba(0, 0, 0, 0.2), inset 0 -10px 20px rgba(0, 0, 0, 0.2)'
            } : {}}>
            <div className="relative w-full h-full overflow-visible">
              <div className={`absolute top-1/2 -translate-y-1/2 w-full h-1 ${
                isLightMode ? 'bg-gray-400' : 'bg-slate-600'
              }`}></div>
              <div className={`absolute top-1/3 w-full h-px ${
                isLightMode ? 'bg-gray-400/50' : 'bg-slate-500/50'
              }`}></div>
              <div className={`absolute top-2/3 w-full h-px ${
                isLightMode ? 'bg-gray-400/50' : 'bg-slate-500/50'
              }`}></div>
              
              {/* Products */}
              {[1, 2, 3, 4, 5].map((num) => (
                <div key={num} className={`product product-${num} ${num === 1 ? 'ejecting' : ''} absolute top-1/2 -translate-y-1/2`} 
                     style={{ 
                       width: num === 1 ? '6rem' : num === 2 ? '5rem' : num === 3 ? '8rem' : num === 4 ? '7rem' : '6rem',
                       height: num === 1 ? '6rem' : num === 2 ? '8rem' : num === 3 ? '6rem' : num === 4 ? '7rem' : '5rem'
                     }}>
                  <div className={num === 1 ? 'product-1-inner w-full h-full' : 'w-full h-full'}>
                    <img src={getProductImage(num)} alt={`Product ${num}`} className="w-full h-full object-cover rounded-lg" />
                  </div>
                  
                  {/* OCR Popup */}
                  <div className={`ocr-popup ocr-popup-${num}`}>
                    <div className={`relative ${num === 1 ? 'bg-gradient-to-br from-red-500 via-rose-500 to-red-600' : 'bg-gradient-to-br from-emerald-500 via-green-500 to-emerald-600'} text-white px-5 py-3 rounded-xl border-2 ${num === 1 ? 'border-red-300' : 'border-emerald-300'} backdrop-blur-sm overflow-hidden`} 
                         style={{ animation: num === 1 ? 'popupGlowFailed 2s ease-in-out infinite' : 'popupGlow 2s ease-in-out infinite' }}>
                      <div className="scan-line-effect"></div>
                      <div className="absolute top-0 left-0 w-3 h-3 bg-white rounded-full opacity-60"></div>
                      <div className="absolute bottom-0 right-0 w-2 h-2 bg-white rounded-full opacity-40"></div>
                      <div className="relative z-10">
                        <div className="flex items-center gap-2 mb-1.5">
                          <div className="bg-white/20 p-1 rounded-lg backdrop-blur-sm">
                            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                              {num === 1 ? (
                                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd"/>
                              ) : (
                                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"/>
                              )}
                            </svg>
                          </div>
                          <span className="font-black text-sm tracking-wide">
                            {num === 1 ? 'QUALITY FAILED' : 'QUALITY PASSED'}
                          </span>
                        </div>
                        <div className="flex items-center justify-between gap-3 text-xs">
                          <div className="flex items-center gap-2">
                            <div className="bg-white/20 px-2 py-0.5 rounded-full backdrop-blur-sm">
                              <span className="font-semibold">OCR</span>
                            </div>
                            <span className="opacity-90">{num === 1 ? '✗ Rejected' : '✓ Verified'}</span>
                          </div>
                          <div className="flex items-center gap-1 bg-white/20 px-2 py-0.5 rounded-full backdrop-blur-sm">
                            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M13.477 14.89A6 6 0 015.11 6.524l8.367 8.368zm1.414-1.414L6.524 5.11a6 6 0 018.367 8.367zM18 10a8 8 0 11-16 0 8 8 0 0116 0z" clipRule="evenodd"/>
                            </svg>
                            <span className="font-bold">
                              {num === 1 ? '287ms' : num === 2 ? '156ms' : num === 3 ? '243ms' : num === 4 ? '198ms' : '224ms'}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            
            {/* Belt Rollers */}
            <div className={`absolute -left-10 top-1/2 -translate-y-1/2 w-20 h-64 rounded-l-2xl border-2 shadow-2xl ${
              isLightMode
                ? 'bg-gradient-to-r from-gray-400 to-gray-500 border-gray-600'
                : 'bg-gradient-to-r from-slate-600 to-slate-700 border-slate-800'
            }`}></div>
            <div className={`absolute -right-10 top-1/2 -translate-y-1/2 w-20 h-64 rounded-r-2xl border-2 shadow-2xl ${
              isLightMode
                ? 'bg-gradient-to-l from-gray-400 to-gray-500 border-gray-600'
                : 'bg-gradient-to-l from-slate-600 to-slate-700 border-slate-800'
            }`}></div>
          </div>
          
          {/* Login Panel */}
          <div className="relative z-40 w-[520px]">
            <div className={`rounded-3xl p-10 backdrop-blur-xl shadow-2xl ${
              isLightMode
                ? 'bg-white/90 border-2 border-gray-200'
                : 'metal-panel'
            }`}>
              <div className="text-center mb-8">
                <h2 className={`orbitron text-xl font-black mb-3 tracking-wider ${
                  isLightMode ? 'text-gray-800' : 'text-white'
                }`}>VISION SYSTEM ACCESS</h2>
                <p className={`text-sm tracking-[0.3em] font-medium ${
                  isLightMode ? 'text-blue-600' : 'text-cyan-400'
                }`}>SECURE AUTHENTICATION REQUIRED</p>
              </div>
              
              <form className="space-y-6" onSubmit={handleSubmit}>
                <div>
                  <label className={`flex items-center gap-2 text-sm font-semibold mb-3 tracking-wide ${
                    isLightMode ? 'text-gray-700' : 'text-gray-300'
                  }`}>
                    <svg className={`w-5 h-5 ${isLightMode ? 'text-blue-600' : 'text-cyan-400'}`} fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd"/>
                    </svg>
                    EMPLOYEE ID
                  </label>
                  <div className="relative">
                    <input
                      type="text"
                      placeholder="Enter your employee ID"
                      value={username}
                      onChange={(e) => setUsername(e.target.value)}
                      required
                      className={`input-field w-full border-2 rounded-xl px-5 py-4 focus:outline-none ${
                        isLightMode
                          ? 'bg-gray-50 border-gray-300 text-gray-800 placeholder-gray-400 focus:border-blue-500 focus:ring-2 focus:ring-blue-200'
                          : 'bg-slate-900/80 border-slate-600 text-white placeholder-slate-500'
                      }`}
                    />
                    <div className="absolute right-4 top-1/2 -translate-y-1/2">
                      <svg className={`w-6 h-6 ${isLightMode ? 'text-blue-600' : 'text-cyan-400'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 11c0 3.517-1.009 6.799-2.753 9.571m-3.44-2.04l.054-.09A13.916 13.916 0 008 11a4 4 0 118 0c0 1.017-.07 2.019-.203 3m-2.118 6.844A21.88 21.88 0 0015.171 17m3.839 1.132c.645-2.266.99-4.659.99-7.132A8 8 0 008 4.07M3 15.364c.64-1.319 1-2.8 1-4.364 0-1.457.39-2.823 1.07-4"/>
                      </svg>
                    </div>
                  </div>
                </div>
                
                <div>
                  <label className={`flex items-center gap-2 text-sm font-semibold mb-3 tracking-wide ${
                    isLightMode ? 'text-gray-700' : 'text-gray-300'
                  }`}>
                    <svg className={`w-5 h-5 ${isLightMode ? 'text-blue-600' : 'text-cyan-400'}`} fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"/>
                    </svg>
                    SECURITY CODE
                  </label>
                  <div className="relative">
                    <input
                      type="password"
                      placeholder="Enter your security code"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      required
                      className={`input-field w-full border-2 rounded-xl px-5 py-4 focus:outline-none ${
                        isLightMode
                          ? 'bg-gray-50 border-gray-300 text-gray-800 placeholder-gray-400 focus:border-blue-500 focus:ring-2 focus:ring-blue-200'
                          : 'bg-slate-900/80 border-slate-600 text-white placeholder-slate-500'
                      }`}
                    />
                    <div className="absolute right-4 top-1/2 -translate-y-1/2">
                      <svg className={`w-6 h-6 ${isLightMode ? 'text-blue-600' : 'text-cyan-400'}`} fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clipRule="evenodd"/>
                      </svg>
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center justify-between text-sm">
                  <label className="flex items-center gap-3 cursor-pointer group">
                    <input type="checkbox" className={`w-5 h-5 rounded border-2 cursor-pointer ${
                      isLightMode
                        ? 'border-gray-300 bg-gray-50 checked:bg-blue-500 checked:border-blue-500'
                        : 'border-slate-600 bg-slate-900 checked:bg-blue-500 checked:border-blue-500'
                    }`}/>
                    <span className={`transition-colors ${
                      isLightMode
                        ? 'text-gray-600 group-hover:text-gray-800'
                        : 'text-gray-400 group-hover:text-gray-300'
                    }`}>Remember Device</span>
                  </label>
                  <a href="#" className={`transition-colors font-medium ${
                    isLightMode
                      ? 'text-blue-600 hover:text-blue-700'
                      : 'text-cyan-400 hover:text-cyan-300'
                  }`}>Reset Code</a>
                </div>

                {loginError && (
                  <div className="bg-red-500/10 border border-red-500/50 rounded-xl px-4 py-3 flex items-center gap-3">
                    <svg className="w-5 h-5 text-red-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                    <span className="text-red-500 text-sm font-medium">{loginError}</span>
                  </div>
                )}

                <button
                  type="submit"
                  disabled={isLoading}
                  className="btn-primary relative w-full bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white font-bold py-5 rounded-xl transition-all duration-300 shadow-lg orbitron tracking-widest text-lg disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <span className="relative z-10 flex items-center justify-center gap-3">
                    {isLoading ? (
                      <>
                        <svg className="animate-spin h-6 w-6" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        AUTHENTICATING...
                      </>
                    ) : (
                      <>
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11 16l-4-4m0 0l4-4m-4 4h14m-5 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h7a3 3 0 013 3v1"/>
                        </svg>
                        ACCESS SYSTEM
                      </>
                    )}
                  </span>
                </button>
              </form>
              
              <div className={`mt-8 pt-6 border-t ${isLightMode ? 'border-gray-300' : 'border-slate-700'}`}>
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div className="flex flex-col items-center gap-2">
                    <svg className="w-6 h-6 text-green-400" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"/>
                    </svg>
                    <span className={`text-xs font-medium ${isLightMode ? 'text-gray-600' : 'text-gray-500'}`}>ENCRYPTED</span>
                  </div>
                  <div className="flex flex-col items-center gap-2">
                    <svg className={`w-6 h-6 ${isLightMode ? 'text-blue-600' : 'text-cyan-400'}`} fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M2 5a2 2 0 012-2h12a2 2 0 012 2v10a2 2 0 01-2 2H4a2 2 0 01-2-2V5zm3.293 1.293a1 1 0 011.414 0l3 3a1 1 0 010 1.414l-3 3a1 1 0 01-1.414-1.414L7.586 10 5.293 7.707a1 1 0 010-1.414zM11 12a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd"/>
                    </svg>
                    <span className={`text-xs font-medium ${isLightMode ? 'text-gray-600' : 'text-gray-500'}`}>SECURE</span>
                  </div>
                  <div className="flex flex-col items-center gap-2">
                    <svg className={`w-6 h-6 ${isLightMode ? 'text-blue-500' : 'text-blue-400'}`} fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd"/>
                    </svg>
                    <span className={`text-xs font-medium ${isLightMode ? 'text-gray-600' : 'text-gray-500'}`}>24/7</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          {/* Stats Footer */}
          <div className="absolute bottom-10 left-1/2 -translate-x-1/2 z-50">
            <div className="flex items-center gap-12">
              <div className="text-center">
                <p className="orbitron text-3xl font-bold text-green-400">2,847</p>
                <p className={`text-xs tracking-widest font-medium ${isLightMode ? 'text-gray-600' : 'text-gray-500'}`}>UNITS TODAY</p>
              </div>
              <div className={`w-px h-16 ${isLightMode ? 'bg-gray-400' : 'bg-slate-700'}`}></div>
              <div className="text-center">
                <p className={`orbitron text-3xl font-bold ${isLightMode ? 'text-blue-600' : 'text-cyan-400'}`}>94.2%</p>
                <p className={`text-xs tracking-widest font-medium ${isLightMode ? 'text-gray-600' : 'text-gray-500'}`}>EFFICIENCY</p>
              </div>
              <div className={`w-px h-16 ${isLightMode ? 'bg-gray-400' : 'bg-slate-700'}`}></div>
              <div className="text-center">
                <p className={`orbitron text-3xl font-bold ${isLightMode ? 'text-blue-500' : 'text-blue-400'}`}>127</p>
                <p className={`text-xs tracking-widest font-medium ${isLightMode ? 'text-gray-600' : 'text-gray-500'}`}>ACTIVE STAFF</p>
              </div>
            </div>
          </div>

          {/* Bottom Fade */}
          <div className={`absolute bottom-0 left-0 w-full h-40 z-20 ${
            isLightMode
              ? 'bg-gradient-to-t from-gray-200 to-transparent'
              : 'bg-gradient-to-t from-slate-950 to-transparent'
          }`}></div>

          {/* Version Info */}
          <div className={`absolute bottom-4 right-8 text-right text-xs z-30 ${
            isLightMode ? 'text-gray-400' : 'text-slate-700'
          }`}>
            <p className="font-medium">Suntech Automation Co., Ltd © 2025</p>
            <p className={isLightMode ? 'text-gray-500' : 'text-slate-800'}>Build 20241125-1.0.0</p>
          </div>
        </div>
      </div>
    </>
  );
}