import React, { useState, useEffect } from 'react';

interface FuturisticTemplateProps {
  username: string;
  password: string;
  rememberDevice: boolean;
  loginError: string;
  isLoading: boolean;
  isLightMode: boolean;
  onUsernameChange: (value: string) => void;
  onPasswordChange: (value: string) => void;
  onRememberDeviceChange: (checked: boolean) => void;
  onSubmit: (e: React.FormEvent<HTMLFormElement>) => void;
}

export default function FuturisticTemplate({
  username,
  password,
  rememberDevice,
  loginError,
  isLoading,
  isLightMode,
  onUsernameChange,
  onPasswordChange,
  onRememberDeviceChange,
  onSubmit,
}: FuturisticTemplateProps) {
  const [scanLine, setScanLine] = useState(0);
  const [particles, setParticles] = useState<Array<{x: number, y: number, speed: number}>>([]);

  useEffect(() => {
    // Animated scan line
    const interval = setInterval(() => {
      setScanLine(prev => (prev + 1) % 100);
    }, 50);

    // Generate particles
    const newParticles = Array.from({ length: 50 }, () => ({
      x: Math.random() * 100,
      y: Math.random() * 100,
      speed: Math.random() * 0.5 + 0.2
    }));
    setParticles(newParticles);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="futuristic-login-container">
      {/* Animated Grid Background */}
      <div className="futuristic-grid-bg">
        <div className="grid-lines horizontal"></div>
        <div className="grid-lines vertical"></div>
      </div>

      {/* Floating Particles */}
      <div className="particles-container">
        {particles.map((particle, i) => (
          <div
            key={i}
            className="particle"
            style={{
              left: `${particle.x}%`,
              top: `${particle.y}%`,
              animationDuration: `${20 / particle.speed}s`,
              animationDelay: `${-Math.random() * 20}s`
            }}
          />
        ))}
      </div>

      {/* Scan Line Effect */}
      <div className="scan-line-vertical" style={{ left: `${scanLine}%` }} />

      {/* Main Content */}
      <div className="futuristic-content">
        {/* Top HUD */}
        <div className="futuristic-hud-top">
          <div className="hud-section">
            <div className="hud-logo-container">
              <img src="logo.png" alt="Suntech" className="hud-logo" />
              <div className="hud-logo-glow"></div>
            </div>
            <div className="hud-title">
              <h1 className="orbitron neon-text-cyan">SUNTECH AUTOMATION</h1>
              <p className="neon-text-purple">QUANTUM VISION SYSTEM v2.0</p>
            </div>
          </div>

          <div className="hud-section-right">
            <div className="status-indicator online">
              <div className="status-dot"></div>
              <span>NEURAL LINK ACTIVE</span>
            </div>
            <div className="hud-time">
              <div className="time-label">SESSION</div>
              <div className="time-value orbitron">GAMMA-07</div>
            </div>
          </div>
        </div>

        {/* Central Login Hologram */}
        <div className="hologram-container">
          {/* Corner Brackets */}
          <div className="holo-bracket top-left"></div>
          <div className="holo-bracket top-right"></div>
          <div className="holo-bracket bottom-left"></div>
          <div className="holo-bracket bottom-right"></div>

          {/* Holographic Panel */}
          <div className="holographic-panel">
            <div className="holo-header">
              <div className="holo-icon-container">
                <svg className="holo-icon" viewBox="0 0 24 24" fill="none">
                  <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
              <div className="holo-title">
                <h2 className="orbitron">NEURAL AUTHENTICATION</h2>
                <p>BIOMETRIC VERIFICATION REQUIRED</p>
              </div>
            </div>

            {/* Login Form */}
            <form className="holo-form" onSubmit={onSubmit}>
              <div className="holo-input-container">
                <div className="input-label-container">
                  <svg className="input-icon" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/>
                    <circle cx="12" cy="7" r="4"/>
                  </svg>
                  <label>OPERATOR ID</label>
                </div>
                <div className="holo-input-wrapper">
                  <input
                    type="text"
                    value={username}
                    onChange={(e) => onUsernameChange(e.target.value)}
                    placeholder="Enter operator credentials"
                    className="holo-input"
                    required
                  />
                  <div className="input-scan-line"></div>
                </div>
              </div>

              <div className="holo-input-container">
                <div className="input-label-container">
                  <svg className="input-icon" viewBox="0 0 24 24" fill="currentColor">
                    <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
                    <path d="M7 11V7a5 5 0 0110 0v4"/>
                  </svg>
                  <label>NEURAL KEY</label>
                </div>
                <div className="holo-input-wrapper">
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => onPasswordChange(e.target.value)}
                    placeholder="Enter neural passkey"
                    className="holo-input"
                    required
                  />
                  <div className="input-scan-line"></div>
                </div>
              </div>

              <div className="holo-options">
                <label className="holo-checkbox">
                  <input
                    type="checkbox"
                    checked={rememberDevice}
                    onChange={(e) => onRememberDeviceChange(e.target.checked)}
                  />
                  <span className="checkbox-custom"></span>
                  <span className="checkbox-label">REMEMBER DEVICE</span>
                </label>
                <a href="#" className="holo-link">RESET CREDENTIALS</a>
              </div>

              {loginError && (
                <div className="holo-error">
                  <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2L2 7L12 12L22 7L12 2Z"/>
                    <path d="M2 17L12 22L22 17"/>
                    <path d="M2 12L12 17L22 12"/>
                  </svg>
                  <span>{loginError}</span>
                </div>
              )}

              <button
                type="submit"
                disabled={isLoading}
                className="holo-submit-btn"
              >
                <span className="btn-glow"></span>
                <span className="btn-content orbitron">
                  {isLoading ? (
                    <>
                      <svg className="spinner" viewBox="0 0 24 24">
                        <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" fill="none" strokeDasharray="60" strokeDashoffset="30"/>
                      </svg>
                      AUTHENTICATING
                    </>
                  ) : (
                    <>
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
                      </svg>
                      INITIATE ACCESS
                    </>
                  )}
                </span>
              </button>
            </form>

            {/* Security Badges */}
            <div className="security-badges">
              <div className="badge">
                <svg viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z"/>
                </svg>
                <span>QUANTUM ENCRYPTED</span>
              </div>
              <div className="badge">
                <svg viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                </svg>
                <span>NEURAL VERIFIED</span>
              </div>
              <div className="badge">
                <svg viewBox="0 0 24 24" fill="currentColor">
                  <circle cx="12" cy="12" r="10"/>
                  <polyline points="12 6 12 12 16 14" fill="none" stroke="white" strokeWidth="2"/>
                </svg>
                <span>24/7 UPTIME</span>
              </div>
            </div>
          </div>

          {/* Scanning Effect */}
          <div className="holo-scan-effect"></div>
        </div>

        {/* Side Panels - Vision Systems */}
        <div className="side-panel left-panel">
          <div className="panel-header">
            <div className="panel-indicator"></div>
            <span>CAM-ALPHA</span>
          </div>
          <div className="panel-content">
            <div className="camera-feed">
              <div className="feed-overlay">
                <div className="crosshair"></div>
                <div className="scan-area"></div>
              </div>
            </div>
            <div className="panel-stats">
              <div className="stat">
                <span className="stat-label">FPS</span>
                <span className="stat-value neon-text-green">120</span>
              </div>
              <div className="stat">
                <span className="stat-label">ACCURACY</span>
                <span className="stat-value neon-text-cyan">99.7%</span>
              </div>
            </div>
          </div>
        </div>

        <div className="side-panel right-panel">
          <div className="panel-header">
            <div className="panel-indicator"></div>
            <span>NEURAL-CORE</span>
          </div>
          <div className="panel-content">
            <div className="camera-feed">
              <div className="feed-overlay">
                <div className="crosshair"></div>
                <div className="scan-area"></div>
              </div>
            </div>
            <div className="panel-stats">
              <div className="stat">
                <span className="stat-label">PROCESSED</span>
                <span className="stat-value neon-text-purple">8,432</span>
              </div>
              <div className="stat">
                <span className="stat-label">EFFICIENCY</span>
                <span className="stat-value neon-text-cyan">94.2%</span>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Stats HUD */}
        <div className="futuristic-hud-bottom">
          <div className="stat-module">
            <div className="stat-icon">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z"/>
              </svg>
            </div>
            <div className="stat-data">
              <div className="stat-label">UNITS TODAY</div>
              <div className="stat-number orbitron neon-text-green">2,847</div>
            </div>
          </div>

          <div className="stat-module">
            <div className="stat-icon">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M16 11c1.66 0 2.99-1.34 2.99-3S17.66 5 16 5c-1.66 0-3 1.34-3 3s1.34 3 3 3zm-8 0c1.66 0 2.99-1.34 2.99-3S9.66 5 8 5C6.34 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5c0-2.33-4.67-3.5-7-3.5zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5z"/>
              </svg>
            </div>
            <div className="stat-data">
              <div className="stat-label">ACTIVE OPERATORS</div>
              <div className="stat-number orbitron neon-text-cyan">127</div>
            </div>
          </div>

          <div className="stat-module">
            <div className="stat-icon">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
              </svg>
            </div>
            <div className="stat-data">
              <div className="stat-label">SYSTEM STATUS</div>
              <div className="stat-number orbitron neon-text-purple">OPTIMAL</div>
            </div>
          </div>
        </div>

        {/* Corner Info */}
        <div className="corner-info bottom-right">
          <p className="neon-text-dim">SUNTECH AUTOMATION © 2025</p>
          <p className="neon-text-dim">QUANTUM BUILD v2.0.1</p>
        </div>
      </div>
    </div>
  );
}
