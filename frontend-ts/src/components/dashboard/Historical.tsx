import React, { useState, useEffect, useRef } from 'react';

const Historical: React.FC = () => {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const [dateRange, setDateRange] = useState('7days');
  const [selectedCamera, setSelectedCamera] = useState('all');

  const historicalData = [
    { date: '2024-12-13', camera1: 445, camera2: 412, camera3: 355 },
    { date: '2024-12-14', camera1: 438, camera2: 420, camera3: 362 },
    { date: '2024-12-15', camera1: 452, camera2: 415, camera3: 358 },
    { date: '2024-12-16', camera1: 448, camera2: 425, camera3: 365 },
    { date: '2024-12-17', camera1: 455, camera2: 418, camera3: 360 },
    { date: '2024-12-18', camera1: 445, camera2: 412, camera3: 355 },
    { date: '2024-12-19', camera1: 456, camera2: 423, camera3: 368 },
  ];

  useEffect(() => {
    drawChart();
  }, [dateRange, selectedCamera]);

  const drawChart = () => {
    if (!chartRef.current) return;

    const canvas = chartRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    const padding = 60;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    // Draw grid
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;

    for (let i = 0; i <= 5; i++) {
      const y = padding + (chartHeight / 5) * i;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    // Draw data
    const cameras = selectedCamera === 'all'
      ? ['camera1', 'camera2', 'camera3']
      : [selectedCamera];

    const colors: Record<string, string> = {
      camera1: '#6366f1',
      camera2: '#3b82f6',
      camera3: '#8b5cf6'
    };

    cameras.forEach(camera => {
      const data = historicalData.map(d => d[camera as keyof typeof d] as number);
      const maxValue = Math.max(...data);
      const minValue = Math.min(...data);
      const valueRange = maxValue - minValue || 1;

      ctx.strokeStyle = colors[camera] || '#6366f1';
      ctx.lineWidth = 3;
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

      // Draw points
      ctx.fillStyle = colors[camera] || '#6366f1';
      data.forEach((value, index) => {
        const x = padding + (index / (data.length - 1)) * chartWidth;
        const y = padding + chartHeight - ((value - minValue) / valueRange) * chartHeight;

        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
      });
    });

    // Draw labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
    ctx.textAlign = 'center';

    historicalData.forEach((item, index) => {
      const x = padding + (index / (historicalData.length - 1)) * chartWidth;
      const dateLabel = new Date(item.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      ctx.fillText(dateLabel, x, height - 20);
    });
  };

  const totalProducts = historicalData.reduce((sum, day) => 
    sum + day.camera1 + day.camera2 + day.camera3, 0
  );
  
  const avgDaily = Math.round(totalProducts / historicalData.length);

  return (
    <div className="historical-page">
      <div className="section-header">
        <div>
          <h2>Historical Data</h2>
          <p>Production analytics and trends</p>
        </div>
        <button className="create-button">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M21 15V19C21 19.5304 20.7893 20.0391 20.4142 20.4142C20.0391 20.7893 19.5304 21 19 21H5C4.46957 21 3.96086 20.7893 3.58579 20.4142C3.21071 20.0391 3 19.5304 3 19V15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <polyline points="7,10 12,15 17,10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <line x1="12" y1="15" x2="12" y2="3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Export Data
        </button>
      </div>

      {/* Stats Overview */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <rect x="3" y="3" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
              <rect x="14" y="3" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
              <rect x="14" y="14" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
              <rect x="3" y="14" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
            </svg>
          </div>
          <div className="stat-content">
            <div className="stat-label">Total Products (7 days)</div>
            <div className="stat-value">{totalProducts.toLocaleString()}</div>
            <span className="stat-trend increase">+15% vs prev week</span>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon success">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <polyline points="22,12 18,12 15,21 9,3 6,12 2,12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <div className="stat-content">
            <div className="stat-label">Average Daily Output</div>
            <div className="stat-value">{avgDaily.toLocaleString()}</div>
            <span className="stat-trend">{Math.round(avgDaily / 24)} per hour</span>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon warning">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <div className="stat-content">
            <div className="stat-label">Peak Performance</div>
            <div className="stat-value">456</div>
            <span className="stat-trend">Dec 19, 2024</span>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="filter-controls">
        <div className="filter-group">
          <label>Date Range:</label>
          <select value={dateRange} onChange={(e) => setDateRange(e.target.value)}>
            <option value="7days">Last 7 Days</option>
            <option value="30days">Last 30 Days</option>
            <option value="90days">Last 90 Days</option>
          </select>
        </div>

        <div className="filter-group">
          <label>Camera:</label>
          <select value={selectedCamera} onChange={(e) => setSelectedCamera(e.target.value)}>
            <option value="all">All Cameras</option>
            <option value="camera1">Camera 1</option>
            <option value="camera2">Camera 2</option>
            <option value="camera3">Camera 3</option>
          </select>
        </div>
      </div>

      {/* Chart */}
      <div className="chart-container">
        <canvas
          ref={chartRef}
          width={800}
          height={400}
          style={{ width: '100%', height: 'auto' }}
        />
      </div>

      {/* Legend */}
      <div className="chart-legend">
        <div className="legend-item">
          <span className="legend-color" style={{ backgroundColor: '#6366f1' }}></span>
          <span>Camera 1</span>
        </div>
        <div className="legend-item">
          <span className="legend-color" style={{ backgroundColor: '#3b82f6' }}></span>
          <span>Camera 2</span>
        </div>
        <div className="legend-item">
          <span className="legend-color" style={{ backgroundColor: '#8b5cf6' }}></span>
          <span>Camera 3</span>
        </div>
      </div>

      {/* Data Table */}
      <div className="data-table-container">
        <h3>Daily Breakdown</h3>
        <table className="data-table">
          <thead>
            <tr>
              <th>Date</th>
              <th>Camera 1</th>
              <th>Camera 2</th>
              <th>Camera 3</th>
              <th>Total</th>
            </tr>
          </thead>
          <tbody>
            {historicalData.map((day) => (
              <tr key={day.date}>
                <td>{new Date(day.date).toLocaleDateString()}</td>
                <td>{day.camera1}</td>
                <td>{day.camera2}</td>
                <td>{day.camera3}</td>
                <td><strong>{day.camera1 + day.camera2 + day.camera3}</strong></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default Historical;
