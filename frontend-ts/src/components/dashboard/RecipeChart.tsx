import { useEffect, useRef } from 'react';

interface RecipeChartProps {
  recipeId: string;
  recipeName: string;
  timeRange: 'today' | 'week' | 'month' | 'year';
  labels: string[];
  data: number[];
}

export default function RecipeChart({ recipeName, timeRange, labels, data }: RecipeChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const getTimeRangeLabel = (range: string) => {
    switch (range) {
      case 'today': return 'Today (24h)';
      case 'week': return 'Last 7 Days';
      case 'month': return 'Last 30 Days';
      case 'year': return 'Last Year';
      default: return range;
    }
  };

  useEffect(() => {
    if (!canvasRef.current || labels.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const padding = { top: 40, right: 40, bottom: 60, left: 60 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Calculate max value for Y-axis
    const maxValue = Math.max(...data, 10);
    const ySteps = 5;

    // Draw Y-axis grid and labels
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    for (let i = 0; i <= ySteps; i++) {
      const y = padding.top + (chartHeight / ySteps) * i;
      const value = Math.round(maxValue * (1 - i / ySteps));

      // Grid line
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();

      // Y-axis label
      ctx.fillText(value.toString(), padding.left - 10, y);
    }

    // Draw line chart
    if (data.length > 0) {
      ctx.strokeStyle = '#6366f1';
      ctx.lineWidth = 3;
      ctx.beginPath();

      data.forEach((value, index) => {
        const x = padding.left + (index / Math.max(data.length - 1, 1)) * chartWidth;
        const y = padding.top + chartHeight - (value / maxValue) * chartHeight;

        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });

      ctx.stroke();

      // Draw points
      ctx.fillStyle = '#6366f1';
      data.forEach((value, index) => {
        const x = padding.left + (index / Math.max(data.length - 1, 1)) * chartWidth;
        const y = padding.top + chartHeight - (value / maxValue) * chartHeight;

        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();

        // White border
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
      });
    }

    // Draw X-axis labels (sample some labels to avoid crowding)
    ctx.fillStyle = '#6b7280';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    const maxLabels = 8;
    const labelStep = Math.ceil(labels.length / maxLabels);

    labels.forEach((label, index) => {
      if (index % labelStep === 0 || index === labels.length - 1) {
        const x = padding.left + (index / Math.max(labels.length - 1, 1)) * chartWidth;
        ctx.fillText(label, x, height - padding.bottom + 10);
      }
    });

  }, [labels, data]);

  const total = data.reduce((sum, val) => sum + val, 0);

  return (
    <div className="camera-card" style={{ marginBottom: '1.5rem' }}>
      <div className="camera-card-header">
        <div className="camera-title">
          <div className="camera-status active"></div>
          <h3>{recipeName} - {getTimeRangeLabel(timeRange)}</h3>
        </div>
        <div style={{ fontSize: '14px', color: '#6b7280' }}>
          Total: {total.toLocaleString()}
        </div>
      </div>
      <div className="camera-card-body">
        <div className="camera-chart">
          <canvas ref={canvasRef} width="1000" height="250"></canvas>
        </div>
      </div>
    </div>
  );
}
