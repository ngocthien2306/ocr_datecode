import React, { useState } from 'react';
import Dashboard from './components/dashboard/Dashboard';
import { authAPI } from './services/api';
import { ToastProvider } from './contexts/ToastContext';
import './styles/App.css';

const App: React.FC = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(() => {
    return !!localStorage.getItem('access_token');
  });
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loginError, setLoginError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showLoginLoading, setShowLoginLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoginError('');
    setIsLoading(true);

    try {
      const response = await authAPI.login(username, password);

      localStorage.setItem('access_token', response.access_token);
      localStorage.setItem('user', JSON.stringify(response.user));

      console.log('Login successful:', response.user);

      setIsLoading(false);
      setShowLoginLoading(true);

      setTimeout(() => {
        setShowLoginLoading(false);
        setIsLoggedIn(true);
      }, 1500);
    } catch (error: any) {
      console.error('Login error:', error);
      const errorMessage = error.response?.data?.detail || 'Login failed. Please check your credentials.';
      setLoginError(errorMessage);
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    console.log('Logging out...');
    localStorage.removeItem('access_token');
    localStorage.removeItem('user');
    setUsername('');
    setPassword('');
    setLoginError('');
    setIsLoggedIn(false);
  };

  if (isLoggedIn) {
    return (
      <ToastProvider>
        <Dashboard onLogout={handleLogout} />
      </ToastProvider>
    );
  }

  if (showLoginLoading) {
    return (
      <div className="login-loading-overlay">
        <div className="login-loading-content">
          <div className="loading-spinner"></div>
          <div className="loading-text">
            <h2>Initializing System</h2>
            <p>Please wait...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="login-page">
      <div className="login-container">
        <div className="login-header">
          <h1>Suntech Vision System</h1>
          <p>OCR & Quality Control Platform</p>
        </div>

        <form className="login-form" onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter your username"
              required
              disabled={isLoading}
            />
          </div>

          <div className="form-group">
            <label>Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
              required
              disabled={isLoading}
            />
          </div>

          {loginError && (
            <div className="error-message">
              {loginError}
            </div>
          )}

          <button type="submit" className="login-btn" disabled={isLoading}>
            {isLoading ? 'Logging in...' : 'Login'}
          </button>
        </form>

        <div className="login-footer">
          <p>© 2024 Suntech Automation. All rights reserved.</p>
        </div>
      </div>
    </div>
  );
};

export default App;
