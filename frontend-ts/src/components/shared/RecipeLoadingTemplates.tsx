import React from 'react';
import '@/styles/RecipeLoadingAnimations.css';
import '@/styles/IndustrialLoading.css';
import '@/styles/AIVisionTemplates.css';
import '@/styles/NeuralProcessingTemplate.css';

interface RecipeLoadingProps {
  template: 'ocr-scanner' | 'camera-vision' | 'barcode-scanner' | 'neural-network' | 'qr-detector' | 'industrial-factory' | 'ai-vision-pipeline' | 'neural-processing';
  progress: number;
  isLightMode?: boolean;
  overlayMode?: 'fullscreen' | 'dashboard-main';
}

const RecipeLoadingTemplates: React.FC<RecipeLoadingProps> = ({
  template,
  progress,
  isLightMode = false,
  overlayMode = 'fullscreen'
}) => {
  const renderTemplate = () => {
    switch (template) {
      case 'ocr-scanner':
        return (
          <div className="ocr-scanner-animation">
            <div className="ocr-title">OCR SCANNING</div>
            <div className="ocr-scan-area">
              {/* Text lines to simulate document */}
              <div className="ocr-text-lines">
                <div className="ocr-text-line"></div>
                <div className="ocr-text-line"></div>
                <div className="ocr-text-line"></div>
                <div className="ocr-text-line"></div>
                <div className="ocr-text-line"></div>
              </div>
              {/* Scanning beam */}
              <div className="ocr-scan-beam"></div>
              {/* Corner brackets */}
              <div className="ocr-corner top-left"></div>
              <div className="ocr-corner top-right"></div>
              <div className="ocr-corner bottom-left"></div>
              <div className="ocr-corner bottom-right"></div>
            </div>
            <div className="ocr-progress-text">ANALYZING TEXT...</div>
          </div>
        );

      case 'camera-vision':
        return (
          <div className="camera-vision-animation">
            <div className="vision-title">VISION SYSTEM</div>
            <div className="vision-grid-container">
              <div className="vision-grid">
                {Array.from({ length: 48 }).map((_, i) => (
                  <div key={i} className="vision-cell"></div>
                ))}
              </div>
              <div className="vision-crosshair"></div>
            </div>
          </div>
        );

      case 'barcode-scanner':
        return (
          <div className="barcode-scanner-animation">
            <div className="barcode-title">BARCODE SCANNER</div>
            <div className="barcode-container">
              <div className="barcode-lines">
                {Array.from({ length: 12 }).map((_, i) => (
                  <div key={i} className="barcode-bar"></div>
                ))}
              </div>
              <div className="barcode-laser"></div>
              <div className="barcode-status">SCANNING...</div>
            </div>
          </div>
        );

      case 'neural-network':
        return (
          <div className="neural-network-animation">
            <div className="neural-title">AI PROCESSING</div>
            <div className="neural-network-container">
              <div className="neural-layers">
                {/* Input layer */}
                <div className="neural-layer">
                  <div className="neural-node"></div>
                  <div className="neural-node"></div>
                  <div className="neural-node"></div>
                </div>
                {/* Hidden layer */}
                <div className="neural-layer">
                  <div className="neural-node"></div>
                  <div className="neural-node"></div>
                  <div className="neural-node"></div>
                  <div className="neural-node"></div>
                </div>
                {/* Output layer */}
                <div className="neural-layer">
                  <div className="neural-node"></div>
                  <div className="neural-node"></div>
                  <div className="neural-node"></div>
                </div>
              </div>
            </div>
          </div>
        );

      case 'qr-detector':
        return (
          <div className="qr-detector-animation">
            <div className="qr-title">QR DETECTION</div>
            <div className="qr-detector-container">
              <div className="qr-frame"></div>
              <div className="qr-corner-marker tl"></div>
              <div className="qr-corner-marker tr"></div>
              <div className="qr-corner-marker bl"></div>
              <div className="qr-corner-marker br"></div>
              <div className="qr-grid">
                {Array.from({ length: 81 }).map((_, i) => (
                  <div key={i} className="qr-pixel"></div>
                ))}
              </div>
              <div className="qr-status-text">Detecting...</div>
            </div>
          </div>
        );

      case 'industrial-factory':
        return (
          <div className="industrial-container">
            {/* Header */}
            <div className="factory-header">
              <div className="factory-title">PROCESSING</div>
              <div className="factory-subtitle">Loading Recipe Configuration</div>
            </div>

            {/* Gear System */}
            <div className="gear-system">
              <div className="gear gear-1">
                <div className="gear-center"></div>
              </div>
              <div className="chain chain-1"></div>
              <div className="gear gear-2">
                <div className="gear-center"></div>
              </div>
              <div className="chain chain-2"></div>
              <div className="gear gear-3">
                <div className="gear-center"></div>
              </div>
            </div>

            {/* Progress Bar */}
            <div className="industrial-progress">
              <div className="progress-frame">
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${progress}%` }}></div>
                  <div className="progress-scanner"></div>
                  <div className="progress-percentage">{progress}%</div>
                </div>
              </div>

              <div className="status-info">
                <div className="status-text">
                  <div className="status-indicator"></div>
                  <span>LOADING RECIPE DATA</span>
                </div>
                <div className="time-remaining">~3s remaining</div>
              </div>

              {/* LED Indicators */}
              <div className="led-panel">
                <div className="led-item">
                  <div className={`led ${progress >= 20 ? 'active' : ''}`}></div>
                  <div className="led-label">Init</div>
                </div>
                <div className="led-item">
                  <div className={`led ${progress >= 40 ? 'active' : ''}`}></div>
                  <div className="led-label">Load</div>
                </div>
                <div className="led-item">
                  <div className={`led ${progress >= 60 ? 'active' : ''}`}></div>
                  <div className="led-label">Config</div>
                </div>
                <div className="led-item">
                  <div className={`led ${progress >= 80 ? 'active' : ''}`}></div>
                  <div className="led-label">Setup</div>
                </div>
                <div className="led-item">
                  <div className={`led ${progress >= 100 ? 'active' : ''}`}></div>
                  <div className="led-label">Ready</div>
                </div>
              </div>
            </div>

            {/* Hydraulic Pistons */}
            <div className="pistons">
              <div className="piston piston-1"></div>
              <div className="piston piston-2"></div>
              <div className="piston piston-3"></div>
            </div>
          </div>
        );

      case 'ai-vision-pipeline':
        const currentStep = progress < 20 ? 1 : progress < 40 ? 2 : progress < 60 ? 3 : progress < 80 ? 4 : 5;
        const steps = [
          { id: 1, label: 'Load Recipe', icon: 'M4 19.5A2.5 2.5 0 0 1 6.5 17H20M6.5 2H20V22H6.5A2.5 2.5 0 0 1 4 19.5V4.5A2.5 2.5 0 0 1 6.5 2Z' },
          { id: 2, label: 'Image Capture', icon: 'M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z M12 17a4 4 0 1 0 0-8 4 4 0 0 0 0 8z' },
          { id: 3, label: 'Load AI Model', icon: 'M12 2a10 10 0 1 0 10 10h-6m0 0V6m0 6l4-4m-4 4l-4-4' },
          { id: 4, label: 'OCR Process', icon: 'M4 7V4h16v3M9 20h6M12 4v16' },
          { id: 5, label: 'Complete', icon: 'M22 11.08V12a10 10 0 1 1-5.93-9.14M22 4L12 14.01l-3-3' }
        ];

        return (
          <div className="ai-vision-container">
            {/* Header */}
            <div className="ai-vision-header">
              <div className="ai-vision-logo">
                <div className="ai-vision-logo-icon">
                  <svg viewBox="0 0 24 24" fill="none">
                    <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </div>
                <h1 className="ai-vision-title">AI Recipe Vision System</h1>
              </div>
              <p className="ai-vision-subtitle">Smart Recipe Loading • Image Capture • OCR Processing • AI Analysis</p>
            </div>

            {/* Steps */}
            <div className="ai-vision-steps">
              {steps.map((step, index) => (
                <React.Fragment key={step.id}>
                  <div
                    className={`ai-vision-step ${currentStep >= step.id ? 'active' : 'inactive'}`}
                    data-step={step.id}
                  >
                    <div className="ai-vision-step-icon">
                      <svg viewBox="0 0 24 24" fill="none">
                        <path d={step.icon} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    </div>
                    <span className="ai-vision-step-label">{step.label}</span>
                    <div className="ai-vision-step-dot"></div>
                  </div>

                  {index < steps.length - 1 && (
                    <div className="ai-vision-progress-line">
                      <div
                        className="ai-vision-progress-fill"
                        style={{ width: currentStep > step.id ? '100%' : currentStep === step.id ? '50%' : '0%' }}
                      ></div>
                    </div>
                  )}
                </React.Fragment>
              ))}
            </div>

            {/* Status Panel */}
            <div className="ai-vision-status">
              <div className="ai-vision-status-header">
                <div className="ai-vision-status-text">
                  <h3>
                    {currentStep === 1 && 'Initializing Recipe Database...'}
                    {currentStep === 2 && 'Capturing High-Resolution Image...'}
                    {currentStep === 3 && 'Loading AI Neural Networks...'}
                    {currentStep === 4 && 'Starting OCR Text Recognition...'}
                    {currentStep === 5 && 'Recipe Processing Complete!'}
                  </h3>
                  <p>
                    {currentStep === 1 && 'Connecting to recipe library'}
                    {currentStep === 2 && 'Adjusting focus and lighting'}
                    {currentStep === 3 && 'Initializing deep learning models'}
                    {currentStep === 4 && 'Processing ingredients and steps'}
                    {currentStep === 5 && 'All systems ready'}
                  </p>
                </div>
                <div className="ai-vision-progress-number">
                  <div className="percentage">{Math.floor(progress)}%</div>
                  <div className="label">Processing Pipeline</div>
                </div>
              </div>

              <div className="ai-vision-main-progress">
                <div className="ai-vision-main-fill" style={{ width: `${progress}%` }}>
                  <div className="progress-dot"></div>
                </div>
              </div>

              <div className="ai-vision-stats">
                <div className="ai-vision-stat-card">
                  <svg viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2">
                    <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
                  </svg>
                  <div className="label">Recipes</div>
                  <div className="value">{Math.min(Math.floor(progress / 25), 4)}</div>
                </div>
                <div className="ai-vision-stat-card">
                  <svg viewBox="0 0 24 24" fill="none" stroke="#3b82f6" strokeWidth="2">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                    <circle cx="8.5" cy="8.5" r="1.5"/>
                    <polyline points="21 15 16 10 5 21"/>
                  </svg>
                  <div className="label">Images</div>
                  <div className="value">{Math.min(Math.floor(progress / 20), 5)}</div>
                </div>
                <div className="ai-vision-stat-card">
                  <svg viewBox="0 0 24 24" fill="none" stroke="#8b5cf6" strokeWidth="2">
                    <rect x="4" y="4" width="16" height="16" rx="2" ry="2"/>
                    <rect x="9" y="9" width="6" height="6"/>
                    <line x1="9" y1="1" x2="9" y2="4"/>
                    <line x1="15" y1="1" x2="15" y2="4"/>
                    <line x1="9" y1="20" x2="9" y2="23"/>
                    <line x1="15" y1="20" x2="15" y2="23"/>
                    <line x1="20" y1="9" x2="23" y2="9"/>
                    <line x1="20" y1="14" x2="23" y2="14"/>
                    <line x1="1" y1="9" x2="4" y2="9"/>
                    <line x1="1" y1="14" x2="4" y2="14"/>
                  </svg>
                  <div className="label">AI Models</div>
                  <div className="value">{Math.min(Math.floor(progress / 33), 3)}</div>
                </div>
                <div className="ai-vision-stat-card">
                  <svg viewBox="0 0 24 24" fill="none" stroke="#f59e0b" strokeWidth="2">
                    <polyline points="4 7 4 4 20 4 20 7"/>
                    <line x1="9" y1="20" x2="15" y2="20"/>
                    <line x1="12" y1="4" x2="12" y2="20"/>
                  </svg>
                  <div className="label">Text Blocks</div>
                  <div className="value">{Math.min(Math.floor(progress * 2.5), 250)}</div>
                </div>
                <div className="ai-vision-stat-card">
                  <svg viewBox="0 0 24 24" fill="none" stroke="#ec4899" strokeWidth="2">
                    <circle cx="12" cy="12" r="10"/>
                    <polyline points="12 6 12 12 16 14"/>
                  </svg>
                  <div className="label">Duration</div>
                  <div className="value">{Math.floor(progress / 20)}s</div>
                </div>
              </div>
            </div>

            {/* Footer */}
            <div className="ai-vision-footer">
              <div className="ai-vision-footer-dot"></div>
              <div className="ai-vision-footer-dot"></div>
              <div className="ai-vision-footer-dot"></div>
              <div className="ai-vision-footer-dot"></div>
              <div className="ai-vision-footer-dot"></div>
            </div>
          </div>
        );

      case 'neural-processing':
        const neuralStep = progress < 20 ? 1 : progress < 40 ? 2 : progress < 65 ? 3 : progress < 90 ? 4 : 5;
        const neuralSteps = [
          { id: 1, name: 'Load Recipe', subtitle: 'config.json', icon: 'M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z M14 2V8H20', logMessage: 'Loading recipe configuration...', logDetail: 'Reading parameters from config.json' },
          { id: 2, name: 'Capture Image', subtitle: 'camera.init', icon: 'M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z M12 17a4 4 0 1 0 0-8 4 4 0 0 0 0 8z', logMessage: 'Initializing capture device...', logDetail: 'Resolution: 1920x1080 | FPS: 60' },
          { id: 3, name: 'Load AI Model', subtitle: 'neural.net', icon: 'M12 2a10 10 0 1 0 10 10h-6m0 0V6m0 6l4-4m-4 4l-4-4', logMessage: 'Loading neural network model...', logDetail: 'Model: vision-transformer-v2.3' },
          { id: 4, name: 'Process OCR', subtitle: 'ocr.engine', icon: 'M4 7V4h16v3M9 20h6M12 4v16', logMessage: 'Running text recognition...', logDetail: 'Engine: Tesseract 5.0 | Lang: multi' },
          { id: 5, name: 'Complete', subtitle: 'success', icon: 'M22 11.08V12a10 10 0 1 1-5.93-9.14M22 4L12 14.01l-3-3', logMessage: 'Processing completed', logDetail: 'Time elapsed: 3.42s | Accuracy: 98.7%' }
        ];

        const getStatusMessage = (step: number) => {
          const messages = [
            'Initializing Recipe Loader',
            'Capturing High-Resolution Image',
            'Initializing AI Neural Network',
            'Processing OCR Analysis',
            'Processing Complete - System Ready'
          ];
          return messages[step - 1] || messages[0];
        };

        return (
          <div className="neural-processing-container">
            {/* Header */}
            <div className="neural-header">
              <div className="neural-logo">
                <div className="neural-logo-blur"></div>
                <div className="neural-logo-icon">
                  <svg viewBox="0 0 24 24" fill="none">
                    <rect x="4" y="4" width="16" height="16" rx="2" ry="2" stroke="currentColor" strokeWidth="2"/>
                    <rect x="9" y="9" width="6" height="6" stroke="currentColor" strokeWidth="2"/>
                    <line x1="9" y1="1" x2="9" y2="4" stroke="currentColor" strokeWidth="2"/>
                    <line x1="15" y1="1" x2="15" y2="4" stroke="currentColor" strokeWidth="2"/>
                    <line x1="9" y1="20" x2="9" y2="23" stroke="currentColor" strokeWidth="2"/>
                    <line x1="15" y1="20" x2="15" y2="23" stroke="currentColor" strokeWidth="2"/>
                    <line x1="20" y1="9" x2="23" y2="9" stroke="currentColor" strokeWidth="2"/>
                    <line x1="20" y1="14" x2="23" y2="14" stroke="currentColor" strokeWidth="2"/>
                    <line x1="1" y1="9" x2="4" y2="9" stroke="currentColor" strokeWidth="2"/>
                    <line x1="1" y1="14" x2="4" y2="14" stroke="currentColor" strokeWidth="2"/>
                  </svg>
                </div>
              </div>
              <h1 className="neural-main-title">AI Vision Processing System</h1>
              <p className="neural-main-subtitle">Neural Pipeline Initialization</p>
            </div>

            {/* Main Panel */}
            <div className="neural-panel">
              {/* Progress Section */}
              <div className="neural-progress-section">
                <div className="neural-progress-header">
                  <div className="neural-progress-status">
                    <div className="neural-status-dot"></div>
                    <span>{getStatusMessage(neuralStep)}</span>
                  </div>
                  <div className="neural-progress-count">
                    <span className="count">{Math.floor(progress)}/100</span>
                    <span className="percentage">{Math.floor(progress)}%</span>
                  </div>
                </div>

                <div className="neural-progress-bar-container">
                  <div className="neural-progress-bar" style={{ width: `${progress}%` }}></div>
                </div>

                <div className="neural-progress-labels">
                  <span>START</span>
                  <span>COMPLETE</span>
                </div>
              </div>

              {/* Steps Grid */}
              <div className="neural-steps-grid">
                {neuralSteps.map((step) => (
                  <div
                    key={step.id}
                    className={`neural-step-card ${neuralStep === step.id ? 'active' : neuralStep > step.id ? 'completed' : ''}`}
                  >
                    <div className="neural-step-spinner"></div>
                    <div className="neural-step-content">
                      <div className="neural-step-icon-wrapper">
                        <div className="scan-line"></div>
                        <svg viewBox="0 0 24 24" fill="none">
                          <path d={step.icon} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      </div>
                      <div className="neural-step-label">
                        <span className="title">{step.name}</span>
                        <span className="subtitle">{step.subtitle}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Bottom Section */}
              <div className="neural-bottom">
                {/* Visualization Panel */}
                <div className="neural-visual-panel">
                  <div className="neural-visual-header">
                    <h3>Pipeline Visualization</h3>
                    <div className="neural-visual-badge">
                      <div className="dot"></div>
                      <span>ACTIVE</span>
                    </div>
                  </div>
                  <div className="neural-visual-content">
                    <div className="neural-visual-spinner"></div>
                  </div>
                  <p className="neural-visual-text">Processing Neural Data</p>
                </div>

                {/* Logs Panel */}
                <div className="neural-logs-panel">
                  <div className="neural-logs-header">
                    <h3>System Logs</h3>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="4 17 10 11 4 5"/>
                      <line x1="12" y1="19" x2="20" y2="19"/>
                    </svg>
                  </div>
                  <div className="neural-logs-content">
                    {neuralSteps.slice(0, Math.min(neuralStep + 1, 5)).map((step, index) => (
                      <div key={index} className="neural-log-entry">
                        <svg viewBox="0 0 24 24" fill="none" stroke={index === neuralStep - 1 ? '#60a5fa' : '#64748b'} strokeWidth="2">
                          <polyline points="9 18 15 12 9 6"/>
                        </svg>
                        <div className="log-text">
                          <p className="message">[{['INIT', 'CAMERA', 'AI', 'OCR', 'SUCCESS'][index]}] {step.logMessage}</p>
                          <p className="detail">{step.logDetail}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Info Bar */}
              <div className="neural-info-bar">
                <div className="neural-info-items">
                  <div className="neural-info-item">
                    <div className="neural-info-item-icon blue">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <rect x="4" y="4" width="16" height="16" rx="2" ry="2"/>
                        <rect x="9" y="9" width="6" height="6"/>
                      </svg>
                    </div>
                    <div className="neural-info-item-text">
                      <span className="label">GPU Acceleration</span>
                      <span className="value">CUDA 12.1</span>
                    </div>
                  </div>

                  <div className="neural-info-item">
                    <div className="neural-info-item-icon purple">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                      </svg>
                    </div>
                    <div className="neural-info-item-text">
                      <span className="label">Security</span>
                      <span className="value">AES-256</span>
                    </div>
                  </div>

                  <div className="neural-info-item">
                    <div className="neural-info-item-icon pink">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
                      </svg>
                    </div>
                    <div className="neural-info-item-text">
                      <span className="label">Performance</span>
                      <span className="value">Real-time</span>
                    </div>
                  </div>
                </div>

                <div className="neural-version">
                  <span className="label">System Version</span>
                  <span className="value">v2.5.1</span>
                </div>
              </div>
            </div>

            {/* Footer */}
            <div className="neural-footer">
              <p>Powered by Advanced Machine Vision & Neural Processing Technology</p>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  const overlayClasses = [
    'recipe-loading-overlay',
    isLightMode ? 'light-mode' : '',
    overlayMode === 'dashboard-main' ? 'dashboard-main-only' : ''
  ].filter(Boolean).join(' ');

  return (
    <div className={overlayClasses}>
      <div className="recipe-loading-container">
        {renderTemplate()}

        {/* Progress display - hide for industrial template (has its own) */}
        {template !== 'industrial-factory' && (
          <div className="recipe-progress-display">
            <div className="recipe-progress-bar">
              <div
                className="recipe-progress-fill"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
            <div className="recipe-progress-percent">{progress}%</div>
            <div className="recipe-status-message">Loading Recipe Configuration...</div>
          </div>
        )}
      </div>
    </div>
  );
};

export default RecipeLoadingTemplates;
