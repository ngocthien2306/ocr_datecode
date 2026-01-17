import React, { useEffect, useState } from 'react';
import { inferenceResultsAPI, type TimeseriesStatistics } from '@/services/inferenceResults';
import { recipesAPI } from '@/services/recipes';
import type { Recipe } from '@/types';
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
import { Line, Bar } from 'react-chartjs-2';

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

const ProductionAnalyticsTab: React.FC = () => {
  // State
  const [dateRange, setDateRange] = useState('7d');
  const [selectedRecipes, setSelectedRecipes] = useState<string[]>([]);
  const [showRecipeDropdown, setShowRecipeDropdown] = useState(false);

  const [timeseriesData, setTimeseriesData] = useState<TimeseriesStatistics | null>(null);
  const [recipes, setRecipes] = useState<Recipe[]>([]);
  const [loading, setLoading] = useState(false);

  // Fetch data on mount
  useEffect(() => {
    fetchRecipes();
  }, []);

  // Fetch timeseries when filters change
  useEffect(() => {
    fetchTimeseriesData();
  }, [dateRange, selectedRecipes]);

  const fetchRecipes = async () => {
    try {
      const recipeList = await recipesAPI.getAllRecipes(0, 100, true);
      setRecipes(recipeList);
    } catch (error) {
      console.error('Error fetching recipes:', error);
    }
  };

  const getDateRange = (range: string): { start_date: string; end_date: string } => {
    const now = new Date();
    const end = new Date(now);
    let start = new Date(now);

    switch (range) {
      case 'today':
        // Today: 00:00:00 to now
        start.setHours(0, 0, 0, 0);
        break;
      case '1h':
        start.setHours(now.getHours() - 1);
        break;
      case '8h':
        start.setHours(now.getHours() - 8);
        break;
      case '1d':
        start.setDate(now.getDate() - 1);
        break;
      case '7d':
        start.setDate(now.getDate() - 7);
        break;
      case '30d':
        start.setDate(now.getDate() - 30);
        break;
      default:
        start.setDate(now.getDate() - 7);
    }

    return {
      start_date: start.toISOString(),
      end_date: end.toISOString()
    };
  };

  const getGranularity = (range: string): 'minute' | 'hour' | 'day' => {
    switch (range) {
      case '1h':
        return 'minute'; // 1 hour: group by minute
      case 'today':
      case '8h':
      case '1d':
        return 'hour'; // Today/8h/1d: group by hour
      case '7d':
      case '30d':
      default:
        return 'day'; // Default: group by day
    }
  };

  const fetchTimeseriesData = async () => {
    try {
      setLoading(true);

      const { start_date, end_date } = getDateRange(dateRange);
      const granularity = getGranularity(dateRange);

      const recipe_ids = selectedRecipes.length > 0 ? selectedRecipes.join(',') : undefined;

      console.log('=== Fetch Timeseries ===');
      console.log('Date range:', dateRange);
      console.log('Granularity:', granularity);
      console.log('Selected recipe IDs:', selectedRecipes);
      console.log('Selected recipe names:', selectedRecipes.map(id => {
        const recipe = recipes.find(r => r.id === id);
        return recipe ? recipe.name : 'Unknown';
      }));
      console.log('API recipe_ids param:', recipe_ids);

      const stats = await inferenceResultsAPI.getTimeseriesStatistics({
        start_date,
        end_date,
        granularity,
        recipe_ids
      });

      console.log('=== API Response ===');
      console.log('Total data points:', stats.data.length);
      if (stats.data.length > 0) {
        console.log('First data point:', stats.data[0]);
        console.log('Number of recipes in response:', stats.data[0]!.by_recipe.length);
        console.log('Recipe IDs in response:', stats.data[0]!.by_recipe.map((r: any) => r.recipe_id));
        console.log('Recipe names in response:', stats.data[0]!.by_recipe.map((r: any) => r.recipe_name));
      }

      setTimeseriesData(stats);
    } catch (error) {
      console.error('Error fetching timeseries data:', error);
      setTimeseriesData(null);
    } finally {
      setLoading(false);
    }
  };

  const getChartData = () => {
    if (!timeseriesData || timeseriesData.data.length === 0) {
      return { labels: [], datasets: [] };
    }

    // Extract labels (timestamps) - format based on granularity
    // Note: Backend already returns VN timezone, no conversion needed
    const labels = timeseriesData.data.map(point => {
      const date = new Date(point.timestamp);

      if (timeseriesData.granularity === 'minute') {
        // For minute: "10:35"
        return date.toLocaleTimeString('en-US', {
          hour: '2-digit',
          minute: '2-digit'
        });
      } else if (timeseriesData.granularity === 'hour') {
        // For hourly: "10:00 AM" (today/1d) or "Jan 17, 10:00" (other days)
        if (dateRange === 'today' || dateRange === '1d') {
          return date.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
          });
        } else {
          return date.toLocaleString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
          });
        }
      } else {
        // For daily: "Jan 17"
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      }
    });

    // Get all unique recipes from ALL data points (not just first one)
    const recipeMap = new Map<string, { recipe_id: string; recipe_name: string }>();
    timeseriesData.data.forEach(point => {
      point.by_recipe.forEach(recipe => {
        if (!recipeMap.has(recipe.recipe_id)) {
          recipeMap.set(recipe.recipe_id, {
            recipe_id: recipe.recipe_id,
            recipe_name: recipe.recipe_name
          });
        }
      });
    });
    const uniqueRecipes = Array.from(recipeMap.values());

    console.log('=== Chart Data ===');
    console.log('Total unique recipes:', uniqueRecipes.length);
    console.log('Recipes:', uniqueRecipes.map(r => r.recipe_name));

    // Color palette
    const colors = [
      '#6366f1', '#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b',
      '#10b981', '#ef4444', '#06b6d4', '#f97316', '#84cc16'
    ];

    // Create datasets for each recipe
    const datasets = uniqueRecipes.map((recipe, index) => {
      const data = timeseriesData.data.map(point => {
        const found = point.by_recipe.find(r => r.recipe_id === recipe.recipe_id);
        return found ? found.total : 0;
      });

      const color = colors[index % colors.length] || '#6366f1';

      return {
        label: recipe.recipe_name,
        data: data,
        borderColor: color,
        backgroundColor: color + '20', // 20 = 12% opacity
        borderWidth: 3,
        pointRadius: 5,
        pointHoverRadius: 7,
        tension: 0.3, // Smooth curves
        fill: false
      };
    });

    return { labels, datasets };
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
        grid: {
          color: '#e5e7eb',
          drawBorder: false
        },
        ticks: {
          font: {
            size: 12
          },
          color: '#6b7280'
        }
      },
      y: {
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
          text: 'Total Inspections',
          font: {
            size: 13,
            weight: 'bold' as const
          },
          color: '#374151'
        }
      }
    }
  };

  const getBarChartData = () => {
    if (!timeseriesData || timeseriesData.data.length === 0) {
      return { labels: [], datasets: [] };
    }

    // Extract labels (timestamps) - format based on granularity
    // Note: Backend already returns VN timezone, no conversion needed
    const labels = timeseriesData.data.map(point => {
      const date = new Date(point.timestamp);

      if (timeseriesData.granularity === 'minute') {
        // For minute: "10:35"
        return date.toLocaleTimeString('en-US', {
          hour: '2-digit',
          minute: '2-digit'
        });
      } else if (timeseriesData.granularity === 'hour') {
        // For hourly: "10:00 AM" (today/1d) or "Jan 17, 10:00" (other days)
        if (dateRange === 'today' || dateRange === '1d') {
          return date.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
          });
        } else {
          return date.toLocaleString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
          });
        }
      } else {
        // For daily: "Jan 17"
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      }
    });

    // Get all unique recipes
    const recipeMap = new Map<string, { recipe_id: string; recipe_name: string }>();
    timeseriesData.data.forEach(point => {
      point.by_recipe.forEach(recipe => {
        if (!recipeMap.has(recipe.recipe_id)) {
          recipeMap.set(recipe.recipe_id, {
            recipe_id: recipe.recipe_id,
            recipe_name: recipe.recipe_name
          });
        }
      });
    });
    const uniqueRecipes = Array.from(recipeMap.values());

    // Color palette for recipes (same as line chart)
    const recipeColors = [
      '#6366f1', '#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b',
      '#10b981', '#ef4444', '#06b6d4', '#f97316', '#84cc16'
    ];

    // Create datasets: For each recipe, create Pass and Fail datasets
    const datasets: any[] = [];

    uniqueRecipes.forEach((recipe, index) => {
      const baseColor = recipeColors[index % recipeColors.length];

      // Pass dataset
      const passData = timeseriesData.data.map(point => {
        const found = point.by_recipe.find(r => r.recipe_id === recipe.recipe_id);
        return found ? found.pass : 0;
      });

      // Fail dataset
      const failData = timeseriesData.data.map(point => {
        const found = point.by_recipe.find(r => r.recipe_id === recipe.recipe_id);
        return found ? found.fail : 0;
      });

      // Convert hex to rgba with opacity for Pass (lighter)
      const passColor = baseColor + 'CC'; // 80% opacity
      const failColor = baseColor + '66'; // 40% opacity (darker/muted)

      // Add Pass dataset
      datasets.push({
        label: `${recipe.recipe_name} (Pass)`,
        data: passData,
        backgroundColor: passColor,
        borderColor: baseColor,
        borderWidth: 1,
        stack: `recipe-${index}`,
        barPercentage: 0.8,
        categoryPercentage: 0.9
      });

      // Add Fail dataset
      datasets.push({
        label: `${recipe.recipe_name} (Fail)`,
        data: failData,
        backgroundColor: failColor,
        borderColor: baseColor,
        borderWidth: 1,
        stack: `recipe-${index}`,
        barPercentage: 0.8,
        categoryPercentage: 0.9
      });
    });

    return { labels, datasets };
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
            // Group by stack to show total per recipe
            const stacks: Record<string, number> = {};
            tooltipItems.forEach((item: any) => {
              const stack = item.dataset.stack;
              if (!stacks[stack]) stacks[stack] = 0;
              stacks[stack] += item.parsed.y;
            });

            let footer = '\n';
            Object.entries(stacks).forEach(([, total]) => {
              footer += `Total: ${total}\n`;
            });
            return footer;
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
            size: 12
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
          text: 'Total Inspections',
          font: {
            size: 13,
            weight: 'bold' as const
          },
          color: '#374151'
        }
      }
    }
  };

  // Multi-select handlers
  const toggleRecipe = (id: string) => {
    setSelectedRecipes(prev => {
      if (prev.includes(id)) {
        return prev.filter(r => r !== id);
      } else if (prev.length >= 30) {
        alert('Maximum 30 recipes allowed');
        return prev;
      } else {
        return [...prev, id];
      }
    });
  };

  return (
    <div className="production-analytics-tab">
      {/* Controls */}
      <div className="chart-controls" style={{ marginBottom: '20px', display: 'flex', gap: '20px', flexWrap: 'wrap', alignItems: 'flex-start' }}>
        {/* Date Range */}
        <div className="control-group">
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 500 }}>Date Range:</label>
          <select value={dateRange} onChange={(e) => setDateRange(e.target.value)} style={{ width: '150px', padding: '8px', borderRadius: '4px', border: '1px solid #d1d5db' }}>
            <option value="today">Today (24h)</option>
            <option value="1h">Last 1 Hour</option>
            <option value="8h">Last 8 Hours</option>
            <option value="1d">Last 1 Day</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>
        </div>

        {/* Recipe Filter */}
        <div className="control-group" style={{ position: 'relative' }}>
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 500 }}>Recipes:</label>
          <button
            onClick={() => setShowRecipeDropdown(!showRecipeDropdown)}
            style={{ padding: '8px 12px', borderRadius: '4px', border: '1px solid #d1d5db', background: 'white', cursor: 'pointer', minWidth: '200px', textAlign: 'left' }}
          >
            {selectedRecipes.length === 0 ? 'All Recipes' : `${selectedRecipes.length} selected`}
          </button>
          {showRecipeDropdown && (
            <div style={{ position: 'absolute', top: '100%', left: 0, background: 'white', border: '1px solid #d1d5db', borderRadius: '4px', padding: '10px', zIndex: 1000, minWidth: '200px', maxHeight: '300px', overflow: 'auto', marginTop: '4px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
              <div style={{ marginBottom: '8px', paddingBottom: '8px', borderBottom: '1px solid #e5e7eb' }}>
                <button onClick={() => setSelectedRecipes([])} style={{ marginRight: '8px', padding: '4px 8px', fontSize: '12px', border: '1px solid #d1d5db', borderRadius: '3px', background: 'white', cursor: 'pointer' }}>Clear All</button>
                <button onClick={() => setSelectedRecipes(recipes.slice(0, 30).map(r => r.id))} style={{ padding: '4px 8px', fontSize: '12px', border: '1px solid #d1d5db', borderRadius: '3px', background: 'white', cursor: 'pointer' }}>Select All</button>
              </div>
              {recipes.map(recipe => (
                <label key={recipe.id} style={{ display: 'block', padding: '6px 0', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={selectedRecipes.includes(recipe.id)}
                    onChange={() => toggleRecipe(recipe.id)}
                    style={{ marginRight: '8px' }}
                  />
                  {recipe.name}
                </label>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Line Chart */}
      <div className="historical-chart-container" style={{ position: 'relative', marginBottom: '40px' }}>
        <h3>Production Trends by Recipe</h3>
        {loading && <p>Loading...</p>}
        {!loading && timeseriesData && timeseriesData.data.length > 0 && (
          <div style={{ height: '500px', position: 'relative' }}>
            <Line data={getChartData()} options={chartOptions} />
          </div>
        )}
        {!loading && (!timeseriesData || timeseriesData.data.length === 0) && (
          <p>No data available</p>
        )}
      </div>

      {/* Stacked Bar Chart for Pass/Fail */}
      <div className="historical-chart-container" style={{ position: 'relative' }}>
        <h3>Pass/Fail Breakdown by Recipe</h3>
        {loading && <p>Loading...</p>}
        {!loading && timeseriesData && timeseriesData.data.length > 0 && (
          <div style={{ height: '500px', position: 'relative' }}>
            <Bar data={getBarChartData()} options={barChartOptions} />
          </div>
        )}
        {!loading && (!timeseriesData || timeseriesData.data.length === 0) && (
          <p>No data available</p>
        )}
      </div>

    </div>
  );
};

export default ProductionAnalyticsTab;
