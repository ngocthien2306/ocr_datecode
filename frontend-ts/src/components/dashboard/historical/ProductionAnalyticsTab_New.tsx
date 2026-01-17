import React, { useEffect, useRef, useState } from 'react';
import { inferenceResultsAPI, type TimeseriesStatistics } from '@/services/inferenceResults';
import { camerasAPI } from '@/services/cameras';
import { recipesAPI } from '@/services/recipes';
import type { Camera, Recipe } from '@/types';

const ProductionAnalyticsTab: React.FC = () => {
  const chartRef = useRef<HTMLCanvasElement | null>(null);

  // State
  const [dateRange, setDateRange] = useState('7days');
  const [groupBy, setGroupBy] = useState<'camera' | 'recipe'>('camera');
  const [selectedCameras, setSelectedCameras] = useState<string[]>([]);
  const [selectedRecipes, setSelectedRecipes] = useState<string[]>([]);
  const [showCameraDropdown, setShowCameraDropdown] = useState(false);
  const [showRecipeDropdown, setShowRecipeDropdown] = useState(false);

  const [timeseriesData, setTimeseriesData] = useState<TimeseriesStatistics | null>(null);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [recipes, setRecipes] = useState<Recipe[]>([]);
  const [loading, setLoading] = useState(false);

  // Fetch data on mount
  useEffect(() => {
    fetchCameras();
    fetchRecipes();
  }, []);

  // Fetch timeseries when filters change
  useEffect(() => {
    fetchTimeseriesData();
  }, [dateRange, groupBy, selectedCameras, selectedRecipes]);

  // Draw chart when data changes
  useEffect(() => {
    if (timeseriesData && timeseriesData.data.length > 0) {
      drawChart();
    }
  }, [timeseriesData]);

  const fetchCameras = async () => {
    try {
      const cameraList = await camerasAPI.getAllCameras();
      setCameras(cameraList);
    } catch (error) {
      console.error('Error fetching cameras:', error);
    }
  };

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
      case '7days':
        start.setDate(now.getDate() - 7);
        break;
      case '30days':
        start.setDate(now.getDate() - 30);
        break;
      case '90days':
        start.setDate(now.getDate() - 90);
        break;
      default:
        start.setDate(now.getDate() - 7);
    }

    return {
      start_date: start.toISOString(),
      end_date: end.toISOString()
    };
  };

  const fetchTimeseriesData = async () => {
    try {
      setLoading(true);

      const { start_date, end_date } = getDateRange(dateRange);

      const camera_ids = selectedCameras.length > 0 ? selectedCameras.join(',') : undefined;
      const recipe_ids = selectedRecipes.length > 0 ? selectedRecipes.join(',') : undefined;

      const stats = await inferenceResultsAPI.getTimeseriesStatistics({
        start_date,
        end_date,
        granularity: 'day',
        camera_ids,
        recipe_ids,
        group_by: groupBy
      });

      setTimeseriesData(stats);
    } catch (error) {
      console.error('Error fetching timeseries data:', error);
      setTimeseriesData(null);
    } finally {
      setLoading(false);
    }
  };

  const drawChart = () => {
    if (!chartRef.current || !timeseriesData || timeseriesData.data.length === 0) return;

    const canvas = chartRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    const padding = 60;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    // Determine what to plot based on group_by
    const items = groupBy === 'camera'
      ? timeseriesData.data[0]?.by_camera || []
      : timeseriesData.data[0]?.by_recipe || [];

    if (items.length === 0) return;

    // Calculate max value for Y-axis
    const allValues: number[] = [];
    timeseriesData.data.forEach(point => {
      const dataItems = groupBy === 'camera' ? point.by_camera : point.by_recipe;
      dataItems.forEach(item => allValues.push(item.total));
    });
    const maxValue = Math.max(...allValues, 100);

    // Draw grid and Y-axis labels
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'right';

    for (let i = 0; i <= 5; i++) {
      const y = padding + (chartHeight / 5) * i;
      const value = Math.round(maxValue - (maxValue / 5) * i);

      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();

      ctx.fillText(value.toString(), padding - 10, y + 4);
    }

    // Draw data lines
    const colors = ['#6366f1', '#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b'];

    items.forEach((item, index) => {
      const itemKey = groupBy === 'camera' ? item.serial_number : item.recipe_id;
      const data = timeseriesData.data.map(point => {
        const dataItems = groupBy === 'camera' ? point.by_camera : point.by_recipe;
        const found = dataItems.find(d =>
          groupBy === 'camera'
            ? d.serial_number === itemKey
            : d.recipe_id === itemKey
        );
        return found ? found.total : 0;
      });

      ctx.strokeStyle = colors[index % colors.length];
      ctx.lineWidth = 3;
      ctx.beginPath();

      data.forEach((value, idx) => {
        const x = padding + (idx / Math.max(data.length - 1, 1)) * chartWidth;
        const y = padding + chartHeight - (value / maxValue) * chartHeight;

        if (idx === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });

      ctx.stroke();

      // Draw points
      ctx.fillStyle = colors[index % colors.length];
      data.forEach((value, idx) => {
        const x = padding + (idx / Math.max(data.length - 1, 1)) * chartWidth;
        const y = padding + chartHeight - (value / maxValue) * chartHeight;

        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
      });
    });

    // Draw X-axis labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';

    timeseriesData.data.forEach((point, idx) => {
      const x = padding + (idx / Math.max(timeseriesData.data.length - 1, 1)) * chartWidth;
      const date = new Date(point.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      ctx.fillText(date, x, height - 20);
    });
  };

  // Multi-select handlers
  const toggleCamera = (serial: string) => {
    setSelectedCameras(prev =>
      prev.includes(serial) ? prev.filter(s => s !== serial) : [...prev, serial]
    );
  };

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

  const getCameraLabel = (serial: string) => {
    const cam = cameras.find(c => c.serial_number === serial);
    return cam ? cam.location : serial;
  };

  const getRecipeName = (id: string) => {
    const recipe = recipes.find(r => r._id === id);
    return recipe ? recipe.name : id;
  };

  return (
    <div className="production-analytics-tab">
      {/* Controls */}
      <div className="chart-controls" style={{ marginBottom: '20px', display: 'flex', gap: '20px', flexWrap: 'wrap', alignItems: 'flex-start' }}>
        {/* Date Range */}
        <div className="control-group">
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 500 }}>Date Range:</label>
          <select value={dateRange} onChange={(e) => setDateRange(e.target.value)} style={{ padding: '8px', borderRadius: '4px', border: '1px solid #d1d5db' }}>
            <option value="7days">Last 7 Days</option>
            <option value="30days">Last 30 Days</option>
            <option value="90days">Last 90 Days</option>
          </select>
        </div>

        {/* Group By */}
        <div className="control-group">
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 500 }}>Group By:</label>
          <div style={{ display: 'flex', gap: '15px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '5px', cursor: 'pointer' }}>
              <input type="radio" name="groupBy" checked={groupBy === 'camera'} onChange={() => setGroupBy('camera')} />
              Camera
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: '5px', cursor: 'pointer' }}>
              <input type="radio" name="groupBy" checked={groupBy === 'recipe'} onChange={() => setGroupBy('recipe')} />
              Recipe
            </label>
          </div>
        </div>

        {/* Camera Filter */}
        <div className="control-group" style={{ position: 'relative' }}>
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 500 }}>Cameras:</label>
          <button
            onClick={() => setShowCameraDropdown(!showCameraDropdown)}
            style={{ padding: '8px 12px', borderRadius: '4px', border: '1px solid #d1d5db', background: 'white', cursor: 'pointer', minWidth: '200px', textAlign: 'left' }}
          >
            {selectedCameras.length === 0 ? 'All Cameras' : `${selectedCameras.length} selected`}
          </button>
          {showCameraDropdown && (
            <div style={{ position: 'absolute', top: '100%', left: 0, background: 'white', border: '1px solid #d1d5db', borderRadius: '4px', padding: '10px', zIndex: 1000, minWidth: '200px', maxHeight: '300px', overflow: 'auto', marginTop: '4px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
              <div style={{ marginBottom: '8px', paddingBottom: '8px', borderBottom: '1px solid #e5e7eb' }}>
                <button onClick={() => setSelectedCameras([])} style={{ marginRight: '8px', padding: '4px 8px', fontSize: '12px', border: '1px solid #d1d5db', borderRadius: '3px', background: 'white', cursor: 'pointer' }}>Clear All</button>
                <button onClick={() => setSelectedCameras(cameras.map(c => c.serial_number))} style={{ padding: '4px 8px', fontSize: '12px', border: '1px solid #d1d5db', borderRadius: '3px', background: 'white', cursor: 'pointer' }}>Select All</button>
              </div>
              {cameras.map(cam => (
                <label key={cam.serial_number} style={{ display: 'block', padding: '6px 0', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={selectedCameras.includes(cam.serial_number)}
                    onChange={() => toggleCamera(cam.serial_number)}
                    style={{ marginRight: '8px' }}
                  />
                  {cam.location}
                </label>
              ))}
            </div>
          )}
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
                <button onClick={() => setSelectedRecipes(recipes.slice(0, 30).map(r => r._id))} style={{ padding: '4px 8px', fontSize: '12px', border: '1px solid #d1d5db', borderRadius: '3px', background: 'white', cursor: 'pointer' }}>Select All</button>
              </div>
              {recipes.map(recipe => (
                <label key={recipe._id} style={{ display: 'block', padding: '6px 0', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={selectedRecipes.includes(recipe._id)}
                    onChange={() => toggleRecipe(recipe._id)}
                    style={{ marginRight: '8px' }}
                  />
                  {recipe.name}
                </label>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Chart */}
      <div className="historical-chart-container">
        <h3>Production Trends ({groupBy === 'camera' ? 'By Camera' : 'By Recipe'})</h3>
        {loading && <p>Loading...</p>}
        {!loading && timeseriesData && timeseriesData.data.length > 0 && (
          <canvas ref={chartRef} width={1200} height={400}></canvas>
        )}
        {!loading && (!timeseriesData || timeseriesData.data.length === 0) && (
          <p>No data available</p>
        )}
      </div>

      {/* Simple Table */}
      <div className="data-table-container" style={{ marginTop: '32px' }}>
        <h3>Daily Production Records</h3>
        {!loading && timeseriesData && timeseriesData.data.length > 0 && (
          <table className="data-table">
            <thead>
              <tr>
                <th>Date</th>
                {groupBy === 'camera'
                  ? timeseriesData.data[0].by_camera.map(cam => (
                      <th key={cam.serial_number}>{getCameraLabel(cam.serial_number)}</th>
                    ))
                  : timeseriesData.data[0].by_recipe.map(recipe => (
                      <th key={recipe.recipe_id}>{recipe.recipe_name}</th>
                    ))
                }
                <th>Total</th>
              </tr>
            </thead>
            <tbody>
              {[...timeseriesData.data].reverse().map(point => (
                <tr key={point.timestamp}>
                  <td><strong>{new Date(point.timestamp).toLocaleDateString()}</strong></td>
                  {groupBy === 'camera'
                    ? timeseriesData.data[0].by_camera.map(cam => {
                        const data = point.by_camera.find(c => c.serial_number === cam.serial_number);
                        return <td key={cam.serial_number}>{data?.total || 0}</td>;
                      })
                    : timeseriesData.data[0].by_recipe.map(recipe => {
                        const data = point.by_recipe.find(r => r.recipe_id === recipe.recipe_id);
                        return <td key={recipe.recipe_id}>{data?.total || 0}</td>;
                      })
                  }
                  <td><strong>{point.total}</strong></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};

export default ProductionAnalyticsTab;
