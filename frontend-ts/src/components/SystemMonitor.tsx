import React, { useEffect, useState } from 'react';
import socketService from '@/services/socketio';
import { useToast } from '@/contexts/ToastContext';
import { SystemMetrics, Alert } from '@/types/jetson';
import { API_BASE_URL } from '@/config/api';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface HistoricalDataPoint {
  timestamp: string;
  cpu_total: number;
  cpu_per_core: { [key: string]: number };
  gpu: number;
}

const MAX_HISTORY_POINTS = 60; // Keep last 60 points (2 minutes at 2s interval)

const SystemMonitor: React.FC = () => {
  const toast = useToast();
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [historicalData, setHistoricalData] = useState<HistoricalDataPoint[]>([]);

  useEffect(() => {
    // Subscribe to real-time metrics
    const handleMetrics = (data: any) => {
      setMetrics(data);
      setIsLoading(false);

      // Store historical data
      setHistoricalData(prev => {
        const newPoint: HistoricalDataPoint = {
          timestamp: new Date(data.timestamp).toLocaleTimeString(),
          cpu_total: data.cpu.usage_percent,
          cpu_per_core: data.cpu.per_core_usage,
          gpu: data.gpu.usage_percent
        };

        const updated = [...prev, newPoint];
        // Keep only last MAX_HISTORY_POINTS
        if (updated.length > MAX_HISTORY_POINTS) {
          return updated.slice(updated.length - MAX_HISTORY_POINTS);
        }
        return updated;
      });
    };

    const handleAlerts = (data: { alerts: Alert[] }) => {
      setAlerts(prev => [...prev, ...data.alerts]);
      data.alerts.forEach(alert => {
        if (alert.severity === 'critical') {
          toast.error(alert.message);
        } else {
          toast.warning(alert.message);
        }
      });
    };

    socketService.subscribeToJetsonMonitoring();
    socketService.onJetsonMetrics(handleMetrics);
    socketService.onJetsonAlerts(handleAlerts);

    // Fetch initial metrics
    fetchMetrics();

    return () => {
      socketService.offJetsonMetrics(handleMetrics);
      socketService.offJetsonAlerts(handleAlerts);
      socketService.unsubscribeFromJetsonMonitoring();
    };
  }, []);

  const fetchMetrics = async () => {
    try {
      const token = localStorage.getItem('access_token');
      const response = await fetch(`${API_BASE_URL}/api/jetson-monitoring/metrics`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (response.ok) {
        const result = await response.json();
        setMetrics(result.data);
        setIsLoading(false);
      }
    } catch (error: any) {
      console.error('Failed to fetch metrics:', error);
      setIsLoading(false);
    }
  };

  const formatUptime = (seconds: number): string => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);

    if (days > 0) return `${days}d ${hours}h ${minutes}m`;
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  };

  const formatBytes = (mb: number): string => {
    if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
    return `${mb.toFixed(0)} MB`;
  };

  const getProgressColor = (percent: number): string => {
    if (percent >= 90) return '#ef4444';
    if (percent >= 75) return '#f59e0b';
    return '#10b981';
  };

  // Build CPU chart data (multi-line for each core)
  const getCPUChartData = () => {
    if (historicalData.length === 0) return null;

    const labels = historicalData.map(d => d.timestamp);

    // Get all core names
    const coreNames = Object.keys(historicalData[0]?.cpu_per_core || {}).sort();

    const colors = [
      '#6366f1', '#3b82f6', '#8b5cf6', '#ec4899',
      '#f59e0b', '#10b981', '#06b6d4', '#f97316'
    ];

    const datasets = coreNames.map((coreName, index) => ({
      label: coreName,
      data: historicalData.map(d => d.cpu_per_core[coreName] || 0),
      borderColor: colors[index % colors.length],
      backgroundColor: colors[index % colors.length] + '20',
      borderWidth: 2,
      pointRadius: 0,
      pointHoverRadius: 4,
      tension: 0.4,
      fill: false
    }));

    return { labels, datasets };
  };

  // Build GPU chart data
  const getGPUChartData = () => {
    if (historicalData.length === 0) return null;

    const labels = historicalData.map(d => d.timestamp);

    return {
      labels,
      datasets: [{
        label: 'GPU Usage',
        data: historicalData.map(d => d.gpu),
        borderColor: '#10b981',
        backgroundColor: '#10b98120',
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 4,
        tension: 0.4,
        fill: true
      }]
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index' as const,
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          usePointStyle: true,
          padding: 10,
          font: { size: 11 }
        }
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        padding: 10,
        titleFont: { size: 12, weight: 'bold' as const },
        bodyFont: { size: 11 }
      }
    },
    scales: {
      x: {
        grid: { color: '#e5e7eb', drawBorder: false },
        ticks: {
          font: { size: 10 },
          color: '#6b7280',
          maxRotation: 0,
          autoSkipPadding: 20
        }
      },
      y: {
        min: 0,
        max: 100,
        grid: { color: '#e5e7eb', drawBorder: false },
        ticks: {
          font: { size: 11 },
          color: '#6b7280',
          callback: (value: any) => `${value}%`
        }
      }
    }
  };

  if (isLoading) {
    return (
      <div className="system-monitor-loading">
        <div className="spinner" />
        <p>Loading system metrics...</p>
      </div>
    );
  }

  if (!metrics) {
    return (
      <div className="system-monitor-error">
        <p>Failed to load system metrics</p>
        <button onClick={fetchMetrics} className="retry-btn">Retry</button>
      </div>
    );
  }

  return (
    <div className="system-monitor">
      {/* Header */}
      <div className="monitor-header">
        <div>
          <h2>System Monitoring</h2>
          <p>Real-time Jetson system metrics</p>
        </div>
        <div className="monitor-info">
          {metrics.nvp_model && (
            <span className="nvp-badge">{metrics.nvp_model}</span>
          )}
          <span className="uptime">Uptime: {formatUptime(metrics.uptime_seconds)}</span>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="metrics-grid">
        {/* CPU Card */}
        <div className="metric-card">
          <div className="metric-header">
            <h3>CPU</h3>
            {metrics.cpu.temperature_celsius !== null && (
              <span className="metric-temp">{metrics.cpu.temperature_celsius.toFixed(1)}°C</span>
            )}
          </div>
          <div className="metric-main">
            <div className="metric-value">{metrics.cpu.usage_percent.toFixed(1)}%</div>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{
                  width: `${metrics.cpu.usage_percent}%`,
                  backgroundColor: getProgressColor(metrics.cpu.usage_percent)
                }}
              />
            </div>
          </div>
          <div className="metric-details">
            <div className="detail-item">
              <span className="detail-label">Frequency</span>
              <span className="detail-value">
                {metrics.cpu.frequency_mhz > 0 ? `${metrics.cpu.frequency_mhz.toFixed(0)} MHz` : 'N/A'}
              </span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Cores</span>
              <span className="detail-value">{Object.keys(metrics.cpu.per_core_usage).length}</span>
            </div>
          </div>
        </div>

        {/* GPU Card */}
        <div className="metric-card">
          <div className="metric-header">
            <h3>GPU</h3>
            {metrics.gpu.temperature_celsius !== null && (
              <span className="metric-temp">{metrics.gpu.temperature_celsius.toFixed(1)}°C</span>
            )}
          </div>
          <div className="metric-main">
            <div className="metric-value">{metrics.gpu.usage_percent.toFixed(1)}%</div>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{
                  width: `${metrics.gpu.usage_percent}%`,
                  backgroundColor: getProgressColor(metrics.gpu.usage_percent)
                }}
              />
            </div>
          </div>
          <div className="metric-details">
            <div className="detail-item">
              <span className="detail-label">Frequency</span>
              <span className="detail-value">
                {metrics.gpu.frequency_mhz > 0 ? `${metrics.gpu.frequency_mhz.toFixed(0)} MHz` : 'N/A'}
              </span>
            </div>
          </div>
        </div>

        {/* RAM Card */}
        <div className="metric-card">
          <div className="metric-header">
            <h3>RAM</h3>
            <span className="metric-subtitle">
              {formatBytes(metrics.ram.used_mb)} / {formatBytes(metrics.ram.total_mb)}
            </span>
          </div>
          <div className="metric-main">
            <div className="metric-value">{metrics.ram.usage_percent.toFixed(1)}%</div>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{
                  width: `${metrics.ram.usage_percent}%`,
                  backgroundColor: getProgressColor(metrics.ram.usage_percent)
                }}
              />
            </div>
          </div>
          <div className="metric-details">
            <div className="detail-item">
              <span className="detail-label">Available</span>
              <span className="detail-value">{formatBytes(metrics.ram.available_mb)}</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Swap</span>
              <span className="detail-value">{metrics.ram.swap_percent.toFixed(1)}%</span>
            </div>
          </div>
        </div>

        {/* Disk Card */}
        <div className="metric-card">
          <div className="metric-header">
            <h3>Disk</h3>
            <span className="metric-subtitle">
              {metrics.disk.used_gb.toFixed(1)} GB / {metrics.disk.total_gb.toFixed(1)} GB
            </span>
          </div>
          <div className="metric-main">
            <div className="metric-value">{metrics.disk.usage_percent.toFixed(1)}%</div>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{
                  width: `${metrics.disk.usage_percent}%`,
                  backgroundColor: getProgressColor(metrics.disk.usage_percent)
                }}
              />
            </div>
          </div>
          <div className="metric-details">
            <div className="detail-item">
              <span className="detail-label">Available</span>
              <span className="detail-value">{metrics.disk.available_gb.toFixed(1)} GB</span>
            </div>
            {metrics.disk.write_speed_mbps !== null && (
              <div className="detail-item">
                <span className="detail-label">Write</span>
                <span className="detail-value">{metrics.disk.write_speed_mbps.toFixed(2)} MB/s</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* CPU Chart */}
      {historicalData.length > 0 && getCPUChartData() && (
        <div className="chart-card">
          <h3>CPU Usage Over Time</h3>
          <div className="chart-container">
            <Line data={getCPUChartData()!} options={chartOptions} />
          </div>
        </div>
      )}

      {/* GPU Chart */}
      {historicalData.length > 0 && getGPUChartData() && (
        <div className="chart-card">
          <h3>GPU Usage Over Time</h3>
          <div className="chart-container">
            <Line data={getGPUChartData()!} options={chartOptions} />
          </div>
        </div>
      )}

      {/* Power Card */}
      {metrics.power && metrics.power.total_mw !== null && (
        <div className="power-card">
          <h3>Power Consumption</h3>
          <div className="power-stats">
            <div className="power-stat">
              <span className="power-label">Total</span>
              <span className="power-value">{(metrics.power.total_mw / 1000).toFixed(2)} W</span>
            </div>
            {metrics.power.cpu_gpu_cv_mw !== null && (
              <div className="power-stat">
                <span className="power-label">CPU+GPU+CV</span>
                <span className="power-value">{(metrics.power.cpu_gpu_cv_mw / 1000).toFixed(2)} W</span>
              </div>
            )}
            {metrics.power.soc_mw !== null && (
              <div className="power-stat">
                <span className="power-label">SOC</span>
                <span className="power-value">{(metrics.power.soc_mw / 1000).toFixed(2)} W</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Alerts */}
      {alerts.length > 0 && (
        <div className="alerts-panel">
          <div className="alerts-header">
            <h3>Recent Alerts</h3>
            <button onClick={() => setAlerts([])} className="clear-alerts-btn">
              Clear
            </button>
          </div>
          <div className="alerts-list">
            {alerts.slice(-5).reverse().map((alert, idx) => (
              <div key={idx} className={`alert-item ${alert.severity}`}>
                <div className="alert-icon">
                  {alert.severity === 'critical' ? '🔴' : '⚠️'}
                </div>
                <div className="alert-content">
                  <div className="alert-message">{alert.message}</div>
                  <div className="alert-meta">
                    {new Date(alert.timestamp).toLocaleTimeString()} •
                    Threshold: {alert.threshold_value.toFixed(1)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default SystemMonitor;
