import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';
import { inferenceResultsAPI, type TimeseriesStatistics } from '@/services/inferenceResults';
import { recipesAPI } from '@/services/recipes';
import type { Recipe } from '@/types';

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

  const fetchTimeseriesData = async () => {
    try {
      setLoading(true);

      const { start_date, end_date } = getDateRange(dateRange);

      const recipe_ids = selectedRecipes.length > 0 ? selectedRecipes.join(',') : undefined;

      console.log('=== Fetch Timeseries ===');
      console.log('Selected recipes:', selectedRecipes);
      console.log('recipe_ids param:', recipe_ids);

      const stats = await inferenceResultsAPI.getTimeseriesStatistics({
        start_date,
        end_date,
        granularity: 'day',
        recipe_ids
      });

      console.log('=== API Response ===');
      console.log('Total data points:', stats.data.length);
      if (stats.data.length > 0) {
        console.log('First data point:', stats.data[0]);
        console.log('Number of recipes in response:', stats.data[0]!.by_recipe.length);
        console.log('Recipes:', stats.data[0]!.by_recipe.map((r: any) => r.recipe_id));
      }

      setTimeseriesData(stats);
    } catch (error) {
      console.error('Error fetching timeseries data:', error);
      setTimeseriesData(null);
    } finally {
      setLoading(false);
    }
  };

  // Prepare Plotly data
  const getPlotlyData = () => {
    if (!timeseriesData || timeseriesData.data.length === 0) return [];

    // Get all unique recipes
    const allRecipes = timeseriesData.data[0]?.by_recipe || [];

    const colors = ['#6366f1', '#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#f97316', '#8b5cf6'];

    return allRecipes.map((recipe, index) => {
      const recipeId = (recipe as any).recipe_id;
      const recipeName = (recipe as any).recipe_name;

      // Extract data for this recipe across all timepoints
      const xData = timeseriesData.data.map(point => new Date(point.timestamp));
      const yData = timeseriesData.data.map(point => {
        const found = point.by_recipe.find(r => (r as any).recipe_id === recipeId);
        return found ? found.total : 0;
      });

      return {
        x: xData,
        y: yData,
        type: 'scatter' as const,
        mode: 'lines+markers' as const,
        name: recipeName,
        line: {
          color: colors[index % colors.length],
          width: 3
        },
        marker: {
          size: 8,
          color: colors[index % colors.length]
        }
      };
    });
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
          <select value={dateRange} onChange={(e) => setDateRange(e.target.value)} style={{ padding: '8px', borderRadius: '4px', border: '1px solid #d1d5db' }}>
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

      {/* Chart */}
      <div className="historical-chart-container" style={{ position: 'relative' }}>
        <h3>Production Trends by Recipe</h3>
        {loading && <p>Loading...</p>}
        {!loading && timeseriesData && timeseriesData.data.length > 0 && (
          <Plot
            data={getPlotlyData()}
            layout={{
              autosize: true,
              height: 500,
              title: '',
              xaxis: {
                title: 'Time',
                showgrid: true,
                gridcolor: '#e5e7eb'
              },
              yaxis: {
                title: 'Total Inspections',
                showgrid: true,
                gridcolor: '#e5e7eb'
              },
              hovermode: 'x unified',
              showlegend: true,
              legend: {
                orientation: 'h',
                yanchor: 'bottom',
                y: 1.02,
                xanchor: 'right',
                x: 1
              },
              margin: {
                l: 60,
                r: 40,
                t: 40,
                b: 60
              }
            }}
            config={{
              responsive: true,
              displayModeBar: true,
              displaylogo: false
            }}
            style={{ width: '100%' }}
          />
        )}
        {!loading && (!timeseriesData || timeseriesData.data.length === 0) && (
          <p>No data available</p>
        )}
      </div>

    </div>
  );
};

export default ProductionAnalyticsTab;
