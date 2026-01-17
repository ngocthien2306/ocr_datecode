import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface RecipeChartProps {
  recipeId: string;
  recipeName: string;
  timeRange: 'today' | 'week' | 'month' | 'year';
  labels: string[];
  totalData: number[];
  passData: number[];
  failData: number[];
}

export default function RecipeChart({ recipeName, timeRange, labels, totalData, passData, failData }: RecipeChartProps) {
  const getTimeRangeLabel = (range: string) => {
    switch (range) {
      case 'today': return 'Today (24h)';
      // case 'week': return 'Last 7 Days';
      case 'month': return 'Last 30 Days';
      // case 'year': return 'Last Year';
      default: return range;
    }
  };

  const total = totalData.reduce((sum, val) => sum + val, 0);

  // Bar chart data for Pass/Fail breakdown (for today and month)
  const barChartData = {
    labels,
    datasets: [
      {
        label: 'Pass',
        data: passData,
        backgroundColor: '#10b981CC', // Green with 80% opacity
        borderColor: '#10b981',
        borderWidth: 1,
        barPercentage: 0.8,
        categoryPercentage: 0.9
      },
      {
        label: 'Fail',
        data: failData,
        backgroundColor: '#ef444466', // Red with 40% opacity
        borderColor: '#ef4444',
        borderWidth: 1,
        barPercentage: 0.8,
        categoryPercentage: 0.9
      }
    ]
  };

  const barChartOptions = {
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
          padding: 15,
          font: {
            size: 12
          }
        }
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        padding: 12,
        titleFont: {
          size: 14,
          weight: 'bold' as const
        },
        bodyFont: {
          size: 13
        },
        callbacks: {
          footer: (tooltipItems: any[]) => {
            const total = tooltipItems.reduce((sum, item) => sum + item.parsed.y, 0);
            return `\nTotal: ${total}`;
          }
        }
      }
    },
    scales: {
      x: {
        stacked: true,
        grid: {
          color: '#e5e7eb',
          drawBorder: false
        },
        ticks: {
          font: {
            size: 11
          },
          color: '#6b7280'
        }
      },
      y: {
        stacked: true,
        grid: {
          color: '#e5e7eb',
          drawBorder: false
        },
        ticks: {
          font: {
            size: 12
          },
          color: '#6b7280'
        },
        title: {
          display: true,
          text: 'Inspections',
          font: {
            size: 12,
            weight: 'bold' as const
          }
        },
        beginAtZero: true
      }
    }
  };

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
        <div className="camera-chart" style={{ height: '300px' }}>
          <Bar data={barChartData} options={barChartOptions} />
        </div>
      </div>
    </div>
  );
}
