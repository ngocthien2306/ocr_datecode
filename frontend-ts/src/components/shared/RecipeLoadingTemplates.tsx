import React from 'react';
import '@/styles/RecipeLoadingAnimations.css';
import '@/styles/IndustrialLoading.css';

interface RecipeLoadingProps {
  template: 'ocr-scanner' | 'camera-vision' | 'barcode-scanner' | 'neural-network' | 'qr-detector' | 'industrial-factory';
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
