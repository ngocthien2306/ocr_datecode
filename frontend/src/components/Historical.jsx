import React, { useState, useEffect, useRef } from 'react';

export default function Historical() {
  const chartRef = useRef(null);
  const [dateRange, setDateRange] = useState('7days');
  const [selectedCamera, setSelectedCamera] = useState('all');

  const historicalData = [
    { date: '2024-01-09', camera1: 445, camera2: 412, camera3: 355 },
    { date: '2024-01-10', camera1: 438, camera2: 420, camera3: 362 },
    { date: '2024-01-11', camera1: 452, camera2: 415, camera3: 358 },
    { date: '2024-01-12', camera1: 448, camera2: 425, camera3: 365 },
    { date: '2024-01-13', camera1: 455, camera2: 418, camera3: 360 },
    { date: '2024-01-14', camera1: 445, camera2: 412, camera3: 355 },
    { date: '2024-01-15', camera1: 456, camera2: 423, camera3: 368 },
  ];

  useEffect(() => {
    drawChart();
  }, [dateRange, selectedCamera]);

  const drawChart = () => {
    if (!chartRef.current) return;

    const canvas = chartRef.current;
    const ctx = canvas.getContext('2d');
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

    const colors = {
      camera1: '#6366f1',
      camera2: '#3b82f6',
      camera3: '#8b5cf6'
    };

    cameras.forEach(camera => {
      const data = historicalData.map(d => d[camera]);
      const maxValue = Math.max(...data);
      const minValue = Math.min(...data);
      const valueRange = maxValue - minValue || 1;

      ctx.strokeStyle = colors[camera];
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
      ctx.fillStyle = colors[camera];
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

  return (
    <div className="historical-page">
      <div className="section-header">
        <h1>Historical Data</h1>
        <button className="dashboard-btn">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M21 15V19C21 19.5304 20.7893 20.0391 20.4142 20.4142C20.0391 20.7893 19.5304 21 19 21H5C4.46957 21 3.96086 20.7893 3.58579 20.4142C3.21071 20.0391 3 19.5304 3 19V15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <polyline points="7,10 12,15 17,10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <line x1="12" y1="15" x2="12" y2="3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Export Data
        </button>
      </div>

      {/* Stats Overview */}
      <div className="summary-cards">
        <div className="summary-card">
          <div className="card-icon products">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <rect x="3" y="3" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
              <rect x="14" y="3" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
              <rect x="14" y="14" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
              <rect x="3" y="14" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
            </svg>
          </div>
          <div className="card-info">
            <h3>Total Products (7 days)</h3>
            <div className="card-value">8,687</div>
            <span className="card-status increase">+15% vs prev week</span>
          </div>
        </div>

        <div className="summary-card">
          <div className="card-icon success">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <polyline points="22,12 18,12 15,21 9,3 6,12 2,12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <div className="card-info">
            <h3>Average Daily Output</h3>
            <div className="card-value">1,241</div>
            <span className="card-status normal">Per day</span>
          </div>
        </div>

        <div className="summary-card">
          <div className="card-icon camera">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <path d="M23 19C23 19.5304 22.7893 20.0391 22.4142 20.4142C22.0391 20.7893 21.5304 21 21 21H3C2.46957 21 1.96086 20.7893 1.58579 20.4142C1.21071 20.0391 1 19.5304 1 19V8C1 7.46957 1.21071 6.96086 1.58579 6.58579C1.96086 6.21071 2.46957 6 3 6H7L9 3H15L17 6H21C21.5304 6 22.0391 6.21071 22.4142 6.58579C22.7893 6.96086 23 7.46957 23 8V19Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <circle cx="12" cy="13" r="4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <div className="card-info">
            <h3>Best Performing Camera</h3>
            <div className="card-value">Camera 1</div>
            <span className="card-status success">3,189 products</span>
          </div>
        </div>
      </div>

      {/* Chart Controls */}
      <div className="chart-controls">
        <div className="control-group">
          <label>Date Range:</label>
          <select value={dateRange} onChange={(e) => setDateRange(e.target.value)}>
            <option value="7days">Last 7 Days</option>
            <option value="30days">Last 30 Days</option>
            <option value="90days">Last 90 Days</option>
          </select>
        </div>
        <div className="control-group">
          <label>Camera:</label>
          <select value={selectedCamera} onChange={(e) => setSelectedCamera(e.target.value)}>
            <option value="all">All Cameras</option>
            <option value="camera1">Camera 1 - Line A</option>
            <option value="camera2">Camera 2 - Line B</option>
            <option value="camera3">Camera 3 - QC</option>
          </select>
        </div>
      </div>

      {/* Historical Chart */}
      <div className="historical-chart-container">
        <h3>Production Trends</h3>
        <div className="chart-legend">
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: '#6366f1' }}></span>
            Camera 1 - Line A
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: '#3b82f6' }}></span>
            Camera 2 - Line B
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: '#8b5cf6' }}></span>
            Camera 3 - QC
          </div>
        </div>
        <canvas ref={chartRef} width="1200" height="400"></canvas>
      </div>

      {/* Data Table */}
      <div className="data-table-container" style={{ marginTop: '32px' }}>
        <h3>Daily Production Records</h3>
        <table className="data-table">
          <thead>
            <tr>
              <th>Date</th>
              <th>Camera 1 - Line A</th>
              <th>Camera 2 - Line B</th>
              <th>Camera 3 - QC</th>
              <th>Total</th>
              <th>Trend</th>
            </tr>
          </thead>
          <tbody>
            {historicalData.reverse().map((record, index) => {
              const total = record.camera1 + record.camera2 + record.camera3;
              const prevTotal = index < historicalData.length - 1
                ? historicalData[index + 1].camera1 + historicalData[index + 1].camera2 + historicalData[index + 1].camera3
                : total;
              const trend = total - prevTotal;

              return (
                <tr key={record.date}>
                  <td><strong>{record.date}</strong></td>
                  <td>{record.camera1}</td>
                  <td>{record.camera2}</td>
                  <td>{record.camera3}</td>
                  <td><strong>{total}</strong></td>
                  <td>
                    {trend > 0 ? (
                      <span className="trend-badge up">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                          <polyline points="18,15 12,9 6,15" stroke="currentColor" strokeWidth="2"/>
                        </svg>
                        +{trend}
                      </span>
                    ) : trend < 0 ? (
                      <span className="trend-badge down">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                          <polyline points="6,9 12,15 18,9" stroke="currentColor" strokeWidth="2"/>
                        </svg>
                        {trend}
                      </span>
                    ) : (
                      <span className="trend-badge neutral">—</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
