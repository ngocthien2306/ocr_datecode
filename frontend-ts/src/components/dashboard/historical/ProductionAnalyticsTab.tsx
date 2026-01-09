import React, { useEffect, useRef, useState } from 'react';

type DayRecord = { date: string; camera1: number; camera2: number; camera3: number };

const ProductionAnalyticsTab: React.FC = () => {
  const chartRef = useRef<HTMLCanvasElement | null>(null);
  const [dateRange, setDateRange] = useState('7days');
  const [selectedCamera, setSelectedCamera] = useState('all');

  // Hardcoded data (will be replaced with real API data later)
  const historicalData: DayRecord[] = [
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

    cameras.forEach((camera) => {
      const data = historicalData.map(d => (d as any)[camera] as number);
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

  const reversedData = [...historicalData].reverse();

  return (
    <div className="production-analytics-tab">
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

      {/* Production Trends Chart */}
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
        <canvas ref={chartRef} width={1200} height={400}></canvas>
      </div>

      {/* Daily Production Records Table */}
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
            {reversedData.map((record, index) => {
              const total = record.camera1 + record.camera2 + record.camera3;
              const prev = index < reversedData.length - 1
                ? reversedData[index + 1]!.camera1 + reversedData[index + 1]!.camera2 + reversedData[index + 1]!.camera3
                : total;
              const trend = total - prev;

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
};

export default ProductionAnalyticsTab;
